/**
 * Unit tests for src/lib/ai/category-rules.ts
 *
 * Verifies the improved v6.1 classifier:
 *   - Single strong keyword is enough (was 2+ keywords before)
 *   - Multi-word phrases score higher
 *   - Hashtags contribute (e.g. #CakeKampala → cake)
 *   - Generic words like "food" alone do NOT trigger restaurant
 *   - Falls back to "general" when no signal
 */
import { describe, it, expect } from 'vitest';
import { classifyCategory, getCategoryRule, CATEGORY_RULES } from './category-rules';

describe('classifyCategory', () => {
  it('classifies "best food in uganda" as general (only generic word)', () => {
    // "food" alone is weight 1 — below the threshold of 2
    expect(classifyCategory('best food in uganda')).toBe('general');
  });

  it('classifies "best food in uganda #food" as general (still weight 1)', () => {
    expect(classifyCategory('best food in uganda #food')).toBe('general');
  });

  it('classifies a cake caption correctly', () => {
    expect(classifyCategory('Birthday cake available! DM to order')).toBe('cake');
  });

  it('classifies a wedding cake caption correctly', () => {
    expect(classifyCategory('Wedding cake special — book now')).toBe('cake');
  });

  it('classifies a bakery caption correctly', () => {
    expect(classifyCategory('Fresh bread baked every morning')).toBe('bakery');
  });

  it('classifies a croissant caption as bakery', () => {
    expect(classifyCategory('Butter croissants just out of the oven')).toBe('bakery');
  });

  it('classifies a restaurant caption correctly', () => {
    expect(classifyCategory('Best restaurant in Kampala — chef special today')).toBe('restaurant');
  });

  it('classifies a Ugandan dish caption as restaurant', () => {
    expect(classifyCategory('Try our luwombo with matooke')).toBe('restaurant');
  });

  it('picks up category via hashtag (#CakeKampala → cake)', () => {
    expect(classifyCategory('Check out our latest creation #CakeKampala #UgandanBakery')).toBe('cake');
  });

  it('picks up category via hashtag (#KampalaRestaurant → restaurant)', () => {
    expect(classifyCategory('Special today #KampalaRestaurant')).toBe('restaurant');
  });

  it('returns general for completely unrelated text', () => {
    expect(classifyCategory('Hello world this is a test caption')).toBe('general');
  });

  it('returns general for empty caption', () => {
    expect(classifyCategory('')).toBe('general');
  });

  it('incorporates OCR text', () => {
    // Caption has no signal but OCR (poster text) does
    expect(classifyCategory('check this out', 'Wedding cake 50,000 UGX')).toBe('cake');
  });
});

describe('getCategoryRule', () => {
  it('returns the rule for known categories', () => {
    expect(getCategoryRule('cake')).toBe(CATEGORY_RULES.cake);
    expect(getCategoryRule('bakery')).toBe(CATEGORY_RULES.bakery);
    expect(getCategoryRule('restaurant')).toBe(CATEGORY_RULES.restaurant);
    expect(getCategoryRule('general')).toBe(CATEGORY_RULES.general);
  });

  it('falls back to general for unknown categories', () => {
    expect(getCategoryRule('unknown')).toBe(CATEGORY_RULES.general);
    expect(getCategoryRule('')).toBe(CATEGORY_RULES.general);
  });

  it('has well-formed rules for each category', () => {
    for (const [name, rule] of Object.entries(CATEGORY_RULES)) {
      expect(rule.idealHashtags).toBeGreaterThanOrEqual(rule.minHashtags);
      expect(rule.idealCaptionLength[0]).toBeLessThan(rule.idealCaptionLength[1]);
      expect(Array.isArray(rule.requiredKeywords)).toBe(true);
      expect(typeof rule.priceRequired).toBe('boolean');
      expect(typeof rule.ctaRequired).toBe('boolean');
      // Smoke-test: name should be a non-empty string
      expect(name.length).toBeGreaterThan(0);
    }
  });
});
