/**
 * POST /api/pipeline/retrain — Train the logistic-regression engagement model
 *
 * v6.1: Now actually trains a real model on MongoDB `ground_truth_posts` data
 * and persists weights to `model_registry`. Previously returned simulated
 * metrics. Falls back to a demo-mode response when MongoDB is not connected.
 */

import { NextResponse } from 'next/server';
import { healthCheck, ModelRegistryRepository, getDb } from '@/lib/db/client';
import { extractCaptionFeatures, buildFeatureVector } from '@/lib/ai/feature-extractor';
import { classifyCategory } from '@/lib/ai/category-rules';
import {
  crossValidate,
  computeAuc,
  computeAccuracy,
  predictProba,
  FEATURE_NAMES,
  TrainedModel,
} from '@/lib/ai/logistic-regression';

interface GroundTruthDoc {
  caption?: string;
  engagement_rate?: number;
  category?: string;
}

export async function POST() {
  try {
    const connected = await healthCheck();

    if (!connected) {
      return NextResponse.json({
        success: true,
        modelVersion: 'heuristic-demo',
        aucScore: 0.5,
        accuracy: 0.5,
        duration: 0,
        samples: 0,
        message: 'MongoDB not connected — no training performed. Set MONGO_URI to enable real training.',
      });
    }

    const t0 = Date.now();
    const db = await getDb();
    const docs = (await db.collection('ground_truth_posts').find({}).toArray()) as GroundTruthDoc[];

    if (docs.length < 20) {
      return NextResponse.json({
        success: false,
        modelVersion: '',
        aucScore: 0,
        accuracy: 0,
        duration: Date.now() - t0,
        samples: docs.length,
        message: `Need at least 20 ground-truth posts to train. Found ${docs.length}. Run scripts/seed-mongodb.ts to add data.`,
      });
    }

    // Build training rows — label = top-50% engagement rate per category
    const byCategory: Record<string, number[]> = {};
    for (const d of docs) {
      const cat = d.category || classifyCategory(d.caption || '');
      if (!byCategory[cat]) byCategory[cat] = [];
      byCategory[cat].push(d.engagement_rate || 0);
    }
    const medians: Record<string, number> = {};
    for (const [cat, rates] of Object.entries(byCategory)) {
      const sorted = [...rates].sort((a, b) => a - b);
      medians[cat] = sorted[Math.floor(sorted.length / 2)] || 0;
    }

    const rows = docs
      .filter(d => d.caption && d.caption.trim().length > 0)
      .map(d => {
        const cat = d.category || classifyCategory(d.caption || '');
        const cf = extractCaptionFeatures(d.caption || '', cat);
        const features = buildFeatureVector(cf, null);
        const label = (d.engagement_rate || 0) >= medians[cat] ? 1 : 0;
        return { features, label };
      });

    if (rows.length < 20) {
      return NextResponse.json({
        success: false,
        modelVersion: '',
        aucScore: 0,
        accuracy: 0,
        duration: Date.now() - t0,
        samples: rows.length,
        message: `Not enough valid training rows (need ≥20). Got ${rows.length}.`,
      });
    }

    // 5-fold cross-validation + final model trained on all data
    const cv = crossValidate(rows, 5, 200, 0.05, 0.001);
    const meanCvAuc = cv.foldAucs.reduce((s, a) => s + a, 0) / cv.foldAucs.length;

    // Compute training-set metrics of final model
    const finalModel = cv.finalModel;
    const trainPredictions = rows.map(r => ({
      p: predictProba(r.features, finalModel.weights, finalModel.bias),
      y: r.label,
    }));
    const trainAuc = computeAuc(trainPredictions);
    const trainAcc = computeAccuracy(trainPredictions);

    // Persist
    const trainedModel: TrainedModel = {
      weights: finalModel.weights,
      bias: finalModel.bias,
      featureNames: FEATURE_NAMES,
      baseline: finalModel.baseline,
      version: `v${Date.now()}`,
      trainedAt: new Date().toISOString(),
      samples: rows.length,
      auc: trainAuc,
      accuracy: trainAcc,
      loss: finalModel.history[finalModel.history.length - 1].loss,
      epochs: 200,
      learningRate: 0.05,
    };

    const repo = new ModelRegistryRepository();
    await repo.insertOne({
      model_type: 'logistic-regression',
      version: trainedModel.version,
      auc: trainedModel.auc,
      auc_score: trainedModel.auc,
      accuracy: trainedModel.accuracy,
      samples: trainedModel.samples,
      dataset_size: trainedModel.samples,
      features: trainedModel.featureNames,
      fold_aucs: cv.foldAucs,
      foldAucs: cv.foldAucs,
      trained_at: trainedModel.trainedAt,
      trainedAt: trainedModel.trainedAt,
      weights: trainedModel.weights,
      bias: trainedModel.bias,
      baseline: trainedModel.baseline,
      loss: trainedModel.loss,
      epochs: trainedModel.epochs,
      learning_rate: trainedModel.learningRate,
    });

    return NextResponse.json({
      success: true,
      modelVersion: trainedModel.version,
      aucScore: trainAuc,
      cvAuc: meanCvAuc,
      accuracy: trainAcc,
      duration: Date.now() - t0,
      samples: rows.length,
      foldAucs: cv.foldAucs,
      message: `Model trained and deployed. CV AUC=${meanCvAuc.toFixed(4)}, train AUC=${trainAuc.toFixed(4)}, accuracy=${trainAcc.toFixed(4)}`,
    });
  } catch (error) {
    return NextResponse.json(
      { success: false, modelVersion: '', aucScore: 0, accuracy: 0, duration: 0, message: String(error) },
      { status: 500 }
    );
  }
}
