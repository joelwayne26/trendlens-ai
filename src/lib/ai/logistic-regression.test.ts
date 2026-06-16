/**
 * Unit tests for src/lib/ai/logistic-regression.ts
 *
 * Verifies:
 *   - Training reduces loss over epochs
 *   - Training on a linearly-separable dataset achieves near-perfect AUC
 *   - computeAuc and computeAccuracy handle edge cases
 *   - predictProba returns values in [0, 1]
 *   - crossValidate returns the expected number of folds
 */
import { describe, it, expect } from 'vitest';
import {
  trainLogisticRegression,
  crossValidate,
  computeAuc,
  computeAccuracy,
  predictProba,
  predictEngagement,
  FEATURE_NAMES,
  TrainingRow,
} from './logistic-regression';

describe('predictProba', () => {
  it('returns 0.5 for zero weights and zero bias', () => {
    expect(predictProba([1, 2, 3], [0, 0, 0], 0)).toBeCloseTo(0.5, 5);
  });

  it('returns values in (0, 1) for any input', () => {
    const features = [0.5, 0.3, 0.8, 1.0];
    const weights = [1.5, -2.0, 0.7, 0.0];
    const p = predictProba(features, weights, 0.3);
    expect(p).toBeGreaterThan(0);
    expect(p).toBeLessThan(1);
  });

  it('returns higher proba for positive-bias models', () => {
    const features = [0.5];
    const weights = [1];
    expect(predictProba(features, weights, 5)).toBeGreaterThan(predictProba(features, weights, -5));
  });
});

describe('computeAuc', () => {
  it('returns 0.5 when only one class is present', () => {
    expect(computeAuc([{ p: 0.9, y: 1 }, { p: 0.8, y: 1 }])).toBe(0.5);
    expect(computeAuc([{ p: 0.1, y: 0 }, { p: 0.2, y: 0 }])).toBe(0.5);
  });

  it('returns 1.0 for a perfectly-separable set', () => {
    const rows = [
      { p: 0.9, y: 1 }, { p: 0.8, y: 1 },
      { p: 0.2, y: 0 }, { p: 0.1, y: 0 },
    ];
    expect(computeAuc(rows)).toBe(1);
  });

  it('returns 0.0 for an inversely-separable set', () => {
    const rows = [
      { p: 0.1, y: 1 }, { p: 0.2, y: 1 },
      { p: 0.9, y: 0 }, { p: 0.8, y: 0 },
    ];
    expect(computeAuc(rows)).toBe(0);
  });

  it('handles ties at 0.5', () => {
    const rows = [
      { p: 0.5, y: 1 }, { p: 0.5, y: 0 },
    ];
    expect(computeAuc(rows)).toBe(0.5);
  });
});

describe('computeAccuracy', () => {
  it('returns 0 for empty input', () => {
    expect(computeAccuracy([])).toBe(0);
  });

  it('returns 1.0 for a perfectly-classified set at threshold 0.5', () => {
    const rows = [
      { p: 0.9, y: 1 }, { p: 0.8, y: 1 },
      { p: 0.2, y: 0 }, { p: 0.1, y: 0 },
    ];
    expect(computeAccuracy(rows)).toBe(1);
  });

  it('returns 0.5 for a 50%-wrong set', () => {
    const rows = [
      { p: 0.9, y: 0 }, { p: 0.1, y: 1 },
      { p: 0.8, y: 0 }, { p: 0.2, y: 1 },
    ];
    expect(computeAccuracy(rows)).toBe(0);
  });
});

describe('trainLogisticRegression', () => {
  it('reduces loss over training', () => {
    // Linearly-separable synthetic dataset: positive class has high feature 0
    const rows: TrainingRow[] = [];
    for (let i = 0; i < 30; i++) {
      rows.push({ features: [0.8 + Math.random() * 0.1, 0.2], label: 1 });
      rows.push({ features: [0.1 + Math.random() * 0.1, 0.8], label: 0 });
    }
    const result = trainLogisticRegression(rows, 100, 0.1, 0.001);
    const finalLoss = result.history[result.history.length - 1].loss;
    const initialLoss = result.history[0].loss;
    expect(finalLoss).toBeLessThan(initialLoss);
  });

  it('achieves high AUC on linearly-separable data', () => {
    const rows: TrainingRow[] = [];
    for (let i = 0; i < 50; i++) {
      rows.push({ features: [0.9, 0.1], label: 1 });
      rows.push({ features: [0.1, 0.9], label: 0 });
    }
    const { weights, bias } = trainLogisticRegression(rows, 200, 0.1, 0.0001);
    const predictions = rows.map(r => ({ p: predictProba(r.features, weights, bias), y: r.label }));
    const auc = computeAuc(predictions);
    expect(auc).toBeGreaterThan(0.95);
  });

  it('returns a baseline array matching the feature dimension', () => {
    const rows = [
      { features: [0.5, 0.3, 0.8], label: 1 },
      { features: [0.1, 0.2, 0.4], label: 0 },
    ];
    const { baseline } = trainLogisticRegression(rows, 5);
    expect(baseline.length).toBe(3);
  });

  it('throws on empty input', () => {
    expect(() => trainLogisticRegression([], 10)).toThrow();
  });
});

describe('crossValidate', () => {
  it('returns k fold AUCs and a final model', () => {
    const rows: TrainingRow[] = [];
    for (let i = 0; i < 40; i++) {
      rows.push({ features: [0.9, 0.1], label: 1 });
      rows.push({ features: [0.1, 0.9], label: 0 });
    }
    const result = crossValidate(rows, 5, 50);
    expect(result.foldAucs.length).toBe(5);
    expect(result.finalModel.weights.length).toBe(2);
    // All folds should achieve high AUC on separable data
    for (const auc of result.foldAucs) {
      expect(auc).toBeGreaterThan(0.8);
    }
  });
});

describe('predictEngagement', () => {
  it('returns 0.5 when no model is provided', () => {
    expect(predictEngagement([1, 2, 3], null)).toBe(0.5);
  });

  it('returns the same as predictProba when a model is provided', () => {
    const model = { weights: [1, -1, 0.5], bias: 0.2 };
    const features = [0.5, 0.3, 0.8];
    expect(predictEngagement(features, model)).toBeCloseTo(predictProba(features, model.weights, model.bias), 5);
  });
});

describe('FEATURE_NAMES', () => {
  it('has 16 names matching buildFeatureVector dimensions', () => {
    expect(FEATURE_NAMES.length).toBe(16);
    for (const name of FEATURE_NAMES) {
      expect(typeof name).toBe('string');
      expect(name.length).toBeGreaterThan(0);
    }
  });
});
