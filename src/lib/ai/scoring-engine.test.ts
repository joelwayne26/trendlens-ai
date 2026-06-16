/**
 * Unit tests for src/lib/ai/scoring-engine.ts
 *
 * These tests pin down the mode-aware scoring behavior we just fixed:
 *   - heuristicScore properly rewards CTA / hashtag / price presence
 *   - scoreTo1to10 maps correctly to the 1-10 range
 *   - computePosterScore responds to image quality
 *   - computeCaptionScore responds to caption features
 */
import { describe, it, expect } from 'vitest';
import {
  scoreTo1to10,
  heuristicScore,
  adjustScoreWithBenchmarks,
  computeConfidenceInterval,
  computePosterScore,
  computeCaptionScore,
} from './scoring-engine';
import { extractCaptionFeatures } from './feature-extractor';
import type { CaptionFeatures, ImageQualityMetrics, BenchmarkData } from '../types';

function makeFeatures(caption: string, category: string = 'general'): CaptionFeatures {
  return extractCaptionFeatures(caption, category);
}

const excellentImage: ImageQualityMetrics = {
  brightness: 0.5,
  contrast: 0.45,
  saturation: 0.4,
  blurScore: 0.7,
  resolution: { width: 1920, height: 1080 },
  aspectRatio: 16 / 9,
  qualityRating: 'excellent',
};

const poorImage: ImageQualityMetrics = {
  brightness: 0.05,
  contrast: 0.04,
  saturation: 0.05,
  blurScore: 0.1,
  resolution: { width: 200, height: 200 },
  aspectRatio: 1,
  qualityRating: 'poor',
};

describe('scoreTo1to10', () => {
  it('maps 0 → 1', () => {
    expect(scoreTo1to10(0)).toBeCloseTo(1, 1);
  });

  it('maps 100 → 10', () => {
    expect(scoreTo1to10(100)).toBeCloseTo(10, 1);
  });

  it('maps 50 → 5.5', () => {
    expect(scoreTo1to10(50)).toBeCloseTo(5.5, 1);
  });

  it('clamps below 1', () => {
    expect(scoreTo1to10(-50)).toBeGreaterThanOrEqual(1);
  });

  it('clamps above 10', () => {
    expect(scoreTo1to10(200)).toBeLessThanOrEqual(10);
  });
});

describe('heuristicScore', () => {
  it('increases when a CTA is added (regression: SHAP/score divergence)', () => {
    const withoutCta = makeFeatures('best food in uganda #uganda #food');
    const withCta = makeFeatures('best food in uganda #uganda #food whatsapp us on 0700 123 456');
    const scoreWithout = heuristicScore(withoutCta, null);
    const scoreWith = heuristicScore(withCta, null);
    // Adding "whatsapp" CTA should bump the score by roughly +18 (from -8 → +10)
    expect(scoreWith - scoreWithout).toBeGreaterThan(10);
  });

  it('increases when more hashtags are added', () => {
    const few = makeFeatures('cakes');
    const many = makeFeatures('cakes #a #b #c #d #e #f #g #h');
    expect(heuristicScore(many, null)).toBeGreaterThan(heuristicScore(few, null));
  });

  it('increases when a price is added', () => {
    const without = makeFeatures('cakes DM to order');
    const withPrice = makeFeatures('cakes UGX 50,000 DM to order');
    expect(heuristicScore(withPrice, null)).toBeGreaterThan(heuristicScore(without, null));
  });

  it('increases when an excellent image is added', () => {
    const f = makeFeatures('cakes DM to order');
    expect(heuristicScore(f, excellentImage)).toBeGreaterThan(heuristicScore(f, null));
  });

  it('decreases when a poor image is added', () => {
    const f = makeFeatures('cakes DM to order');
    expect(heuristicScore(f, poorImage)).toBeLessThan(heuristicScore(f, null));
  });

  it('is bounded in [0, 100]', () => {
    const f = makeFeatures('cakes');
    const s = heuristicScore(f, excellentImage);
    expect(s).toBeGreaterThanOrEqual(0);
    expect(s).toBeLessThanOrEqual(100);
  });
});

describe('computePosterScore', () => {
  const noBench: BenchmarkData = {
    dbConnected: false,
    categorySamples: 0,
    industryAvgEngagement: 0,
    top10Engagement: 0,
    hashtagPerformance: {},
    ctaEngagementBoost: 0,
    priceEngagementBoost: 0,
    modelVersion: 'none',
    modelAuc: 0,
  };

  it('rewards excellent images over poor images', () => {
    const f = makeFeatures('test');
    const excellent = computePosterScore(excellentImage, f, noBench);
    const poor = computePosterScore(poorImage, f, noBench);
    expect(excellent).toBeGreaterThan(poor);
  });

  it('is bounded in [1, 10]', () => {
    const f = makeFeatures('test');
    const s = computePosterScore(excellentImage, f, noBench);
    expect(s).toBeGreaterThanOrEqual(1);
    expect(s).toBeLessThanOrEqual(10);
  });

  it('still returns a baseline score when no image is provided', () => {
    const f = makeFeatures('test');
    const s = computePosterScore(null, f, noBench);
    expect(s).toBeGreaterThanOrEqual(1);
    expect(s).toBeLessThanOrEqual(10);
  });
});

describe('computeCaptionScore', () => {
  const noBench: BenchmarkData = {
    dbConnected: false,
    categorySamples: 0,
    industryAvgEngagement: 0,
    top10Engagement: 0,
    hashtagPerformance: {},
    ctaEngagementBoost: 0,
    priceEngagementBoost: 0,
    modelVersion: 'none',
    modelAuc: 0,
  };

  it('is higher for a rich caption than a sparse one', () => {
    const rich = makeFeatures(
      'Fresh cakes available! UGX 50,000. DM to order. #CakeKampala #UgandanBakery #WeddingCake #BirthdayCake #Cupcakes #FreshBakes #KampalaFood #UGFoodie 🎂🍰',
      'cake',
    );
    const sparse = makeFeatures('cakes', 'cake');
    expect(computeCaptionScore(rich, 'cake', noBench)).toBeGreaterThan(
      computeCaptionScore(sparse, 'cake', noBench),
    );
  });

  it('is bounded in [1, 10]', () => {
    const f = makeFeatures('cakes');
    const s = computeCaptionScore(f, 'general', noBench);
    expect(s).toBeGreaterThanOrEqual(1);
    expect(s).toBeLessThanOrEqual(10);
  });
});

describe('adjustScoreWithBenchmarks', () => {
  it('is a no-op when DB is not connected', () => {
    const f = makeFeatures('cakes DM to order');
    const noBench: BenchmarkData = {
      dbConnected: false,
      categorySamples: 0,
      industryAvgEngagement: 0,
      top10Engagement: 0,
      hashtagPerformance: {},
      ctaEngagementBoost: 0,
      priceEngagementBoost: 0,
      modelVersion: 'none',
      modelAuc: 0,
    };
    const adjusted = adjustScoreWithBenchmarks(6.0, f, noBench);
    expect(adjusted).toBe(6.0);
  });

  it('is a no-op when categorySamples < 5', () => {
    const f = makeFeatures('cakes DM to order');
    const smallBench: BenchmarkData = {
      dbConnected: true,
      categorySamples: 3,
      industryAvgEngagement: 0.05,
      top10Engagement: 0.1,
      hashtagPerformance: {},
      ctaEngagementBoost: 0.02,
      priceEngagementBoost: 0.03,
      modelVersion: 'heuristic',
      modelAuc: 0,
    };
    expect(adjustScoreWithBenchmarks(6.0, f, smallBench)).toBe(6.0);
  });

  it('adjusts upward for CTA when benchmark shows positive CTA boost', () => {
    const withCta = makeFeatures('cakes DM to order');
    const bench: BenchmarkData = {
      dbConnected: true,
      categorySamples: 50,
      industryAvgEngagement: 0.05,
      top10Engagement: 0.1,
      hashtagPerformance: {},
      ctaEngagementBoost: 0.1,
      priceEngagementBoost: 0,
      modelVersion: 'heuristic',
      modelAuc: 0,
    };
    const adjusted = adjustScoreWithBenchmarks(6.0, withCta, bench);
    expect(adjusted).toBeGreaterThan(6.0);
  });
});

describe('computeConfidenceInterval', () => {
  it('returns an interval that brackets the score', () => {
    const { lower, upper } = computeConfidenceInterval(60);
    expect(lower).toBeLessThanOrEqual(60);
    expect(upper).toBeGreaterThanOrEqual(60);
  });

  it('respects the [0, 100] bounds', () => {
    const { lower, upper } = computeConfidenceInterval(2);
    expect(lower).toBeGreaterThanOrEqual(0);
    expect(upper).toBeLessThanOrEqual(100);
  });
});
