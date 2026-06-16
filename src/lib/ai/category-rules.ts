/**
 * TrendLens AI v6.1 — Category Rules & Validation
 * Domain-specific rules for Ugandan food business categories.
 *
 * v6.1 changes:
 *   - classifyCategory now uses a weighted, expanded keyword set with
 *     synonyms, hashtag awareness, and partial matching. Previously a
 *     caption like "best food in uganda #food" fell into "general" because
 *     only 1 keyword matched (restaurant needs 2+). The new classifier:
 *       * awards stronger weight to category-specific terms (e.g. "cake"=3,
 *         "birthday cake"=4) and weaker weight to generic ones ("food"=1)
 *       * considers hashtags as well as caption text
 *       * accepts the best-scoring category if score >= 2 (down from 2+
 *         matched keywords, which was a stricter bar)
 */

import { CategoryRule } from '../types';

export const CATEGORY_RULES: Record<string, CategoryRule> = {
  cake: {
    idealHashtags: 8,
    minHashtags: 5,
    idealCaptionLength: [80, 200],
    requiredKeywords: ['cake', 'order', 'delivery'],
    priceRequired: true,
    ctaRequired: true,
  },
  bakery: {
    idealHashtags: 7,
    minHashtags: 4,
    idealCaptionLength: [60, 180],
    requiredKeywords: ['bakery', 'fresh', 'bread'],
    priceRequired: true,
    ctaRequired: true,
  },
  restaurant: {
    idealHashtags: 8,
    minHashtags: 5,
    idealCaptionLength: [80, 220],
    requiredKeywords: ['food', 'restaurant', 'menu'],
    priceRequired: false,
    ctaRequired: true,
  },
  general: {
    idealHashtags: 6,
    minHashtags: 3,
    idealCaptionLength: [50, 200],
    requiredKeywords: [],
    priceRequired: false,
    ctaRequired: false,
  },
};

export function getCategoryRule(category: string): CategoryRule {
  return CATEGORY_RULES[category] || CATEGORY_RULES.general;
}

// ─── Category Classifier ───────────────────────────────────────────────────

interface CategoryKeyword {
  word: string;
  weight: number;
}

// Each category has a set of weighted keywords. Multi-word phrases get
// higher weight. Hashtag-style matches (e.g. "#cakekampala") also count,
// so we check both the literal word and a hashtag-prefixed variant.
const CATEGORY_KEYWORDS: Record<string, CategoryKeyword[]> = {
  cake: [
    { word: 'cake', weight: 3 },
    { word: 'cakes', weight: 3 },
    { word: 'birthday cake', weight: 4 },
    { word: 'wedding cake', weight: 4 },
    { word: 'cupcake', weight: 4 },
    { word: 'cupcakes', weight: 4 },
    { word: 'icing', weight: 3 },
    { word: 'fondant', weight: 3 },
    { word: 'cheesecake', weight: 4 },
    { word: 'red velvet', weight: 4 },
    { word: 'black forest', weight: 4 },
    { word: 'buttercream', weight: 3 },
    { word: 'cakekampala', weight: 3 }, // common hashtag stem
    { word: 'customcakes', weight: 3 },
    { word: 'bakery', weight: 1 },      // overlap with bakery category
  ],
  bakery: [
    { word: 'bakery', weight: 3 },
    { word: 'bread', weight: 3 },
    { word: 'breads', weight: 3 },
    { word: 'pastry', weight: 3 },
    { word: 'pastries', weight: 3 },
    { word: 'croissant', weight: 4 },
    { word: 'croissants', weight: 4 },
    { word: 'sourdough', weight: 4 },
    { word: 'loaf', weight: 3 },
    { word: 'dough', weight: 2 },
    { word: 'flour', weight: 2 },
    { word: 'muffin', weight: 3 },
    { word: 'donut', weight: 3 },
    { word: 'danish', weight: 3 },
    { word: 'baguette', weight: 4 },
    { word: 'cinnamon roll', weight: 4 },
    { word: 'freshbread', weight: 3 },
    { word: 'kampalabakery', weight: 3 },
  ],
  restaurant: [
    { word: 'restaurant', weight: 3 },
    { word: 'menu', weight: 2 },
    { word: 'dish', weight: 2 },
    { word: 'meal', weight: 2 },
    { word: 'food', weight: 1 },          // generic — low weight
    { word: 'kitchen', weight: 2 },
    { word: 'chef', weight: 3 },
    { word: 'dining', weight: 3 },
    { word: 'dinner', weight: 2 },
    { word: 'lunch', weight: 2 },
    { word: 'breakfast', weight: 2 },
    { word: 'rolex', weight: 4 },         // Ugandan dish
    { word: 'matooke', weight: 4 },
    { word: 'matoke', weight: 4 },
    { word: 'luwombo', weight: 4 },
    { word: 'pilau', weight: 4 },
    { word: 'kikomando', weight: 4 },
    { word: 'grilled chicken', weight: 3 },
    { word: 'beef stew', weight: 3 },
    { word: 'fish', weight: 1 },
    { word: 'rice', weight: 1 },
    { word: 'kachumbari', weight: 4 },
    { word: 'ugandan food', weight: 4 },
    { word: 'kampalarestaurant', weight: 3 },
    { word: 'klafoodie', weight: 2 },
    { word: 'ugfoodie', weight: 2 },
  ],
};

/**
 * Classify a caption into one of: cake | bakery | restaurant | general.
 *
 * Strategy:
 *   1. Normalize text (lowercase, strip non-alphanumeric for hashtag matching)
 *   2. Sum weighted keyword matches for each category
 *   3. Return the highest-scoring category if its score >= 2, else 'general'
 */
export function classifyCategory(caption: string, ocrText: string = ''): string {
  const combined = `${caption} ${ocrText}`.toLowerCase();

  // Hashtag-stripped view: "#CakeKampala" → "cakekampala" so single-word
  // hashtag variants can also match.
  const stripped = combined.replace(/#/g, '');

  const scores: Record<string, number> = { cake: 0, bakery: 0, restaurant: 0 };

  for (const [cat, keywords] of Object.entries(CATEGORY_KEYWORDS)) {
    let score = 0;
    for (const { word, weight } of keywords) {
      if (stripped.includes(word)) {
        score += weight;
      }
    }
    scores[cat] = score;
  }

  const sorted = Object.entries(scores).sort((a, b) => b[1] - a[1]);
  const [bestCat, bestScore] = sorted[0];

  // Require a minimum signal of 2 weighted points to commit to a category.
  // This is more lenient than the previous "needs 2+ matched keywords"
  // rule — a single strong indicator (e.g. "cake" = 3 points) is now
  // enough to classify.
  return bestScore >= 2 ? bestCat : 'general';
}
