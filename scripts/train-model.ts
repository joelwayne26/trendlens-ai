/**
 * Train the TrendLens engagement model on MongoDB ground_truth_posts.
 *
 * Usage:
 *   npx tsx scripts/train-model.ts                # train with defaults
 *   npx tsx scripts/train-model.ts --epochs 500   # custom epochs
 *   npx tsx scripts/train-model.ts --dry-run      # don't persist
 *
 * Reads all documents from `ground_truth_posts` collection, builds feature
 * vectors from each caption, trains a logistic regression with stratified
 * k-fold cross-validation, then persists the trained weights to the
 * `model_registry` collection.
 *
 * The label convention: a post is "high engagement" (label=1) if its
 * `engagement_rate` is in the top 50% of its category, else label=0.
 */

import 'dotenv/config';
import { extractCaptionFeatures, buildFeatureVector } from '../src/lib/ai/feature-extractor';
import { classifyCategory } from '../src/lib/ai/category-rules';
import {
  trainLogisticRegression,
  crossValidate,
  computeAuc,
  computeAccuracy,
  predictProba,
  FEATURE_NAMES,
  TrainingRow,
  TrainedModel,
} from '../src/lib/ai/logistic-regression';
import { getDb, healthCheck, ModelRegistryRepository } from '../src/lib/db/client';

interface GroundTruthDoc {
  caption?: string;
  engagement_rate?: number;
  category?: string;
  has_cta?: boolean;
  has_price?: boolean;
  hashtags?: string[];
}

async function fetchGroundTruth(): Promise<GroundTruthDoc[]> {
  const db = await getDb();
  const docs = await db.collection('ground_truth_posts').find({}).toArray();
  return docs.map(d => ({
    caption: (d.caption as string) || '',
    engagement_rate: (d.engagement_rate as number) || 0,
    category: (d.category as string) || 'general',
    has_cta: (d.has_cta as boolean) || false,
    has_price: (d.has_price as boolean) || false,
    hashtags: (d.hashtags as string[]) || [],
  }));
}

function buildRows(docs: GroundTruthDoc[]): TrainingRow[] {
  // Determine the median engagement rate per category for binary labelling
  const byCategory: Record<string, number[]> = {};
  for (const d of docs) {
    const cat = d.category || classifyCategory(d.caption);
    if (!byCategory[cat]) byCategory[cat] = [];
    byCategory[cat].push(d.engagement_rate || 0);
  }

  const medians: Record<string, number> = {};
  for (const [cat, rates] of Object.entries(byCategory)) {
    const sorted = [...rates].sort((a, b) => a - b);
    medians[cat] = sorted[Math.floor(sorted.length / 2)] || 0;
  }

  const rows: TrainingRow[] = [];
  for (const d of docs) {
    if (!d.caption || d.caption.trim().length === 0) continue;
    const cat = d.category || classifyCategory(d.caption);
    const cf = extractCaptionFeatures(d.caption, cat);
    // We don't have image quality in ground truth, so use null (defaults)
    const features = buildFeatureVector(cf, null);
    const label = (d.engagement_rate || 0) >= medians[cat] ? 1 : 0;
    rows.push({ features, label });
  }
  return rows;
}

async function main() {
  const args = process.argv.slice(2);
  const epochsArg = args.indexOf('--epochs');
  const epochs = epochsArg >= 0 ? parseInt(args[epochsArg + 1], 10) : 200;
  const dryRun = args.includes('--dry-run');

  console.log('=== TrendLens Model Training ===');
  console.log(`Epochs: ${epochs}, Dry-run: ${dryRun}`);

  const connected = await healthCheck();
  if (!connected) {
    console.error('❌ MongoDB is not connected. Set MONGO_URI in your environment.');
    process.exit(1);
  }

  console.log('✓ MongoDB connected');
  console.log('Fetching ground_truth_posts...');
  const docs = await fetchGroundTruth();
  console.log(`  → ${docs.length} ground-truth documents found`);

  if (docs.length < 20) {
    console.warn(`⚠️  Only ${docs.length} documents — need at least 20 for meaningful training.`);
    console.warn('   Add more data via scripts/seed-mongodb.ts and re-run.');
  }

  const rows = buildRows(docs);
  const positives = rows.filter(r => r.label === 1).length;
  console.log(`  → ${rows.length} training rows (${positives} positive, ${rows.length - positives} negative)`);

  if (rows.length === 0) {
    console.error('❌ No training rows could be built. Exiting.');
    process.exit(1);
  }

  console.log('\nRunning 5-fold cross-validation...');
  const cv = crossValidate(rows, 5, epochs, 0.05, 0.001);
  const meanAuc = cv.foldAucs.reduce((s, a) => s + a, 0) / cv.foldAucs.length;
  console.log(`  → Fold AUCs: ${cv.foldAucs.map(a => a.toFixed(4)).join(', ')}`);
  console.log(`  → Mean CV AUC: ${meanAuc.toFixed(4)}`);

  // Compute training-set accuracy of final model
  const finalModel = cv.finalModel;
  const trainPredictions = rows.map(r => ({
    p: predictProba(r.features, finalModel.weights, finalModel.bias),
    y: r.label,
  }));
  const trainAuc = computeAuc(trainPredictions);
  const trainAcc = computeAccuracy(trainPredictions);
  console.log(`  → Training AUC: ${trainAuc.toFixed(4)}, Accuracy: ${trainAcc.toFixed(4)}`);

  // Print top features by absolute weight magnitude
  console.log('\nTop features by weight magnitude:');
  const sortedWeights = finalModel.weights
    .map((w, i) => ({ name: FEATURE_NAMES[i] || `F${i}`, weight: w }))
    .sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight));
  for (const { name, weight } of sortedWeights.slice(0, 8)) {
    const sign = weight >= 0 ? '+' : ' ';
    console.log(`  ${sign}${weight.toFixed(4)}  ${name}`);
  }

  if (dryRun) {
    console.log('\n--dry-run specified; not persisting model.');
    process.exit(0);
  }

  console.log('\nPersisting trained model to model_registry...');
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
    epochs,
    learningRate: 0.05,
  };

  const repo = new ModelRegistryRepository();
  await repo.insertOne({
    model_type: 'logistic-regression',
    version: trainedModel.version,
    auc: trainedModel.auc,
    auc_score: trainedModel.auc, // alias for chart compatibility
    accuracy: trainedModel.accuracy,
    samples: trainedModel.samples,
    dataset_size: trainedModel.samples, // alias for chart compatibility
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

  console.log(`✓ Model ${trainedModel.version} persisted. AUC=${trainedModel.auc.toFixed(4)}, Acc=${trainedModel.accuracy.toFixed(4)}`);
  console.log('\nNext steps:');
  console.log('  - The /api/evaluate endpoint will now load this model automatically.');
  console.log('  - SHAP values in the UI are now derived from the trained weights.');
  console.log('  - Re-run this script whenever you add new ground-truth data.');
}

main().catch(err => {
  console.error('Training failed:', err);
  process.exit(1);
});
