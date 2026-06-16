/**
 * TrendLens AI v6.1 — Logistic Regression Model (pure TypeScript)
 *
 * Replaces the previous "fake XGBoost" — this is a real ML model trained
 * on MongoDB `ground_truth_posts` data using gradient descent. Weights are
 * persisted to the `model_registry` collection and loaded at inference
 * time. SHAP values for linear models have a closed-form solution:
 *
 *     contribution_i = weight_i * (value_i - baseline_i)
 *
 * so the SHAP explainer can now produce mathematically-grounded contributions
 * derived from the trained weights rather than hand-tuned coefficients.
 *
 * Why logistic regression and not XGBoost?
 *   - Pure TypeScript, no native dependencies, runs on Vercel serverless.
 *   - Fast to train (sub-second on 200 samples).
 *   - Interpretable weights → real SHAP values.
 *   - Sufficient for the binary "high engagement vs not" task at our data
 *     scale (200-1000 samples). Tree ensembles would overfit at this scale
 *     without careful regularization.
 */

import { CaptionFeatures, ImageQualityMetrics } from '../types';
import { extractCaptionFeatures, buildFeatureVector } from './feature-extractor';
import { getCategoryRule } from './category-rules';

// ─── Public Types ───────────────────────────────────────────────────────────

export interface TrainedModel {
  weights: number[];          // one per feature dimension
  bias: number;
  featureNames: string[];     // human-readable names parallel to weights
  baseline: number[];         // mean of training feature values (for SHAP)
  version: string;
  trainedAt: string;
  samples: number;
  auc: number;
  accuracy: number;
  loss: number;               // final training loss
  epochs: number;
  learningRate: number;
}

export interface TrainingRow {
  features: number[];         // feature vector
  label: number;              // 1 = high engagement, 0 = low
}

export interface TrainingResult {
  model: TrainedModel;
  foldAucs: number[];
  history: { epoch: number; loss: number; auc: number }[];
}

// ─── Feature Names (parallel to buildFeatureVector) ────────────────────────

export const FEATURE_NAMES: string[] = [
  // Caption features (10 dims) — must match buildFeatureVector order
  'Hashtags (normalized)',
  'Word Count (normalized)',
  'Emoji Count (normalized)',
  'Has Price',
  'Has CTA',
  'Sentiment Polarity',
  'Readability',
  'Trend Alignment',
  'Caption Score (0-1)',
  'Has Required Keywords',
  // Image quality features (6 dims)
  'Image Brightness',
  'Image Contrast',
  'Image Saturation',
  'Image Sharpness',
  'Image Aspect Ratio',
  'Image Quality Rating',
];

// ─── Core Logistic Regression ──────────────────────────────────────────────

function sigmoid(z: number): number {
  if (z >= 0) {
    return 1 / (1 + Math.exp(-z));
  }
  const ez = Math.exp(z);
  return ez / (1 + ez);
}

/**
 * Predict probability of the positive class given features, weights, and bias.
 * Exported so tests and other modules can re-use the same inference path.
 */
export function predictProba(features: number[], weights: number[], bias: number): number {
  let z = bias;
  for (let i = 0; i < features.length; i++) {
    z += weights[i] * features[i];
  }
  return sigmoid(z);
}

/**
 * Train logistic regression via gradient descent.
 *
 * @param rows       Training data (feature vectors + binary labels)
 * @param epochs     Number of full passes over the data
 * @param lr         Learning rate
 * @param l2         L2 regularization strength
 * @param verbose    If true, logs loss per epoch to console
 * @returns          Trained weights, bias, baseline (feature means), and
 *                   training history (loss + AUC per epoch)
 */
export function trainLogisticRegression(
  rows: TrainingRow[],
  epochs: number = 200,
  lr: number = 0.05,
  l2: number = 0.001,
  verbose: boolean = false,
): { weights: number[]; bias: number; baseline: number[]; history: { epoch: number; loss: number; auc: number }[] } {
  if (rows.length === 0) {
    throw new Error('Cannot train on 0 rows');
  }

  const dim = rows[0].features.length;
  const weights = new Array(dim).fill(0);
  let bias = 0;

  // Compute baseline = per-feature mean (used later for SHAP baselines)
  const baseline = new Array(dim).fill(0);
  for (const row of rows) {
    for (let i = 0; i < dim; i++) baseline[i] += row.features[i];
  }
  for (let i = 0; i < dim; i++) baseline[i] /= rows.length;

  const history: { epoch: number; loss: number; auc: number }[] = [];

  for (let epoch = 0; epoch < epochs; epoch++) {
    let totalLoss = 0;
    const gradW = new Array(dim).fill(0);
    let gradB = 0;

    for (const row of rows) {
      const p = predictProba(row.features, weights, bias);
      const y = row.label;
      // Binary cross-entropy loss
      totalLoss += -(y * Math.log(p + 1e-12) + (1 - y) * Math.log(1 - p + 1e-12));
      // Gradient
      const err = p - y;
      for (let i = 0; i < dim; i++) {
        gradW[i] += err * row.features[i];
      }
      gradB += err;
    }

    // Average + L2 regularization
    const n = rows.length;
    for (let i = 0; i < dim; i++) {
      gradW[i] = gradW[i] / n + l2 * weights[i];
    }
    gradB = gradB / n;

    // Update
    for (let i = 0; i < dim; i++) {
      weights[i] -= lr * gradW[i];
    }
    bias -= lr * gradB;

    const avgLoss = totalLoss / n;
    const auc = computeAuc(rows.map(r => ({ p: predictProba(r.features, weights, bias), y: r.label })));

    if (verbose && (epoch % 20 === 0 || epoch === epochs - 1)) {
      console.log(`  epoch ${epoch.toString().padStart(3)}  loss=${avgLoss.toFixed(4)}  auc=${auc.toFixed(4)}`);
    }
    history.push({ epoch, loss: avgLoss, auc });
  }

  return { weights, bias, baseline, history };
}

// ─── AUC (Area Under ROC Curve) ────────────────────────────────────────────

interface ScoredRow { p: number; y: number; }

export function computeAuc(rows: ScoredRow[]): number {
  const positives = rows.filter(r => r.y === 1);
  const negatives = rows.filter(r => r.y === 0);
  if (positives.length === 0 || negatives.length === 0) return 0.5;

  let concordant = 0;
  let ties = 0;
  for (const pos of positives) {
    for (const neg of negatives) {
      if (pos.p > neg.p) concordant++;
      else if (pos.p === neg.p) ties++;
    }
  }
  return (concordant + 0.5 * ties) / (positives.length * negatives.length);
}

// ─── Accuracy ──────────────────────────────────────────────────────────────

export function computeAccuracy(rows: ScoredRow[]): number {
  if (rows.length === 0) return 0;
  let correct = 0;
  for (const r of rows) {
    const predicted = r.p >= 0.5 ? 1 : 0;
    if (predicted === r.y) correct++;
  }
  return correct / rows.length;
}

// ─── k-Fold Cross-Validation ───────────────────────────────────────────────

/**
 * Stratified k-fold CV. Returns per-fold AUCs and a final model trained on
 * the full dataset.
 */
export function crossValidate(
  rows: TrainingRow[],
  k: number = 5,
  epochs: number = 200,
  lr: number = 0.05,
  l2: number = 0.001,
): { foldAucs: number[]; finalModel: ReturnType<typeof trainLogisticRegression> } {
  // Split positives/negatives for stratification
  const positives = rows.filter(r => r.label === 1);
  const negatives = rows.filter(r => r.label === 0);

  // Shuffle deterministically (seeded)
  const seededShuffle = <T,>(arr: T[]): T[] => {
    const a = [...arr];
    let s = 42;
    for (let i = a.length - 1; i > 0; i--) {
      s = (s * 9301 + 49297) % 233280;
      const j = Math.floor((s / 233280) * (i + 1));
      [a[i], a[j]] = [a[j], a[i]];
    }
    return a;
  };

  const posShuffled = seededShuffle(positives);
  const negShuffled = seededShuffle(negatives);

  const foldAucs: number[] = [];
  for (let fold = 0; fold < k; fold++) {
    const valStartPos = Math.floor((fold / k) * posShuffled.length);
    const valEndPos = Math.floor(((fold + 1) / k) * posShuffled.length);
    const valStartNeg = Math.floor((fold / k) * negShuffled.length);
    const valEndNeg = Math.floor(((fold + 1) / k) * negShuffled.length);

    const valRows = [
      ...posShuffled.slice(valStartPos, valEndPos),
      ...negShuffled.slice(valStartNeg, valEndNeg),
    ];
    const trainRows = [
      ...posShuffled.slice(0, valStartPos),
      ...posShuffled.slice(valEndPos),
      ...negShuffled.slice(0, valStartNeg),
      ...negShuffled.slice(valEndNeg),
    ];

    if (trainRows.length === 0 || valRows.length === 0) continue;

    const { weights, bias } = trainLogisticRegression(trainRows, epochs, lr, l2);
    const scored = valRows.map(r => ({ p: predictProba(r.features, weights, bias), y: r.label }));
    foldAucs.push(computeAuc(scored));
  }

  // Final model trained on ALL data
  const finalModel = trainLogisticRegression(rows, epochs, lr, l2);
  return { foldAucs, finalModel };
}

// ─── Inference ─────────────────────────────────────────────────────────────

/**
 * Predict probability of "high engagement" using a trained model.
 * Returns 0.5 (neutral) if no model is provided.
 */
export function predictEngagement(
  features: number[],
  model: { weights: number[]; bias: number } | null,
): number {
  if (!model) return 0.5;
  return predictProba(features, model.weights, model.bias);
}

// ─── SHAP for Linear Models ────────────────────────────────────────────────

export interface LinearShapValue {
  feature: string;
  value: number;         // normalized feature value (0-1)
  contribution: number;  // weight * (value - baseline) — true SHAP for linear models
  description: string;
}

/**
 * Compute SHAP values for a linear model.
 *
 * For linear models, SHAP has a closed form:
 *     phi_i = w_i * (x_i - E[x_i])
 *
 * where w_i is the learned weight, x_i is the current feature value, and
 * E[x_i] is the baseline (training-set mean). This is mathematically
 * equivalent to the Shapley value for an additive linear model.
 *
 * If no trained model is provided, falls back to hand-tuned heuristic
 * contributions (the previous behavior) so the UI still has something
 * to display.
 */
export function computeLinearShap(
  features: number[],
  model: { weights: number[]; baseline: number[]; featureNames: string[] } | null,
  featureDescriptions?: Record<string, string>,
): LinearShapValue[] {
  const descriptions: Record<string, string> = featureDescriptions || {
    'Hashtags (normalized)': 'Number of hashtags in the caption',
    'Word Count (normalized)': 'Word count in the caption',
    'Emoji Count (normalized)': 'Emoji count in the caption',
    'Has Price': 'Includes pricing information',
    'Has CTA': 'Presence of a call-to-action phrase',
    'Sentiment Polarity': 'Positive tone in the caption',
    'Readability': 'How easy the caption is to read',
    'Trend Alignment': 'Alignment with current trending topics',
    'Caption Score (0-1)': 'Composite caption quality score',
    'Has Required Keywords': 'Required keywords for the category',
    'Image Brightness': 'Poster image brightness level',
    'Image Contrast': 'Poster image contrast level',
    'Image Saturation': 'Color saturation of the poster',
    'Image Sharpness': 'Image clarity (blur detection)',
    'Image Aspect Ratio': 'Width/height ratio of the poster',
    'Image Quality Rating': 'Composite image quality rating',
  };

  if (!model) {
    // No model available — return zero contributions (caller may fall back
    // to the heuristic SHAP explainer)
    return [];
  }

  const values: LinearShapValue[] = [];
  for (let i = 0; i < model.weights.length; i++) {
    const name = model.featureNames[i] || `Feature ${i}`;
    const value = features[i] ?? 0;
    const baseline = model.baseline[i] ?? 0;
    const contribution = model.weights[i] * (value - baseline);
    values.push({
      feature: name,
      value: Math.round(value * 100) / 100,
      contribution: Math.round(contribution * 100) / 100,
      description: descriptions[name] || '',
    });
  }

  // Sort by absolute contribution (largest impact first)
  return values.sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));
}

// ─── Convenience: Build Features From Caption ──────────────────────────────

export function buildInferenceFeatures(
  caption: string,
  category: string,
  imageQuality: ImageQualityMetrics | null,
): { features: number[]; captionFeatures: CaptionFeatures } {
  const captionFeatures = extractCaptionFeatures(caption, category);
  // Apply category rule to populate categoryChecks
  const rules = getCategoryRule(category);
  void rules; // rules already applied inside extractCaptionFeatures
  const features = buildFeatureVector(captionFeatures, imageQuality);
  return { features, captionFeatures };
}
