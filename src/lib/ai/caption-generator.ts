/**
 * TrendLens AI v6.1 — Local Caption Improver
 *
 * IMPORTANT BEHAVIOUR CHANGE (v6.1):
 *   Previously `generateImprovedCaption` would discard the user's wording and
 *   construct an entirely new caption from templates. Users complained that
 *   their own voice was being replaced. The new behaviour PRESERVES the
 *   user's original caption verbatim and only APPENDS missing elements
 *   (price, CTA, emojis, additional hashtags). The from-scratch fallback is
 *   only used when the user provided an empty caption.
 *
 * Template-based NLG with intelligent composition — NO external LLM APIs.
 * Tailored for Ugandan food businesses.
 */

import { CaptionFeatures } from '../types';
import { getCategoryRule } from './category-rules';

// ─── Pattern Library ───────────────────────────────────────────────────────

interface CaptionPattern {
  ctas: string[];
  hashtagSets: Record<string, string[]>;
  emojiSets: string[];
  pricePhrases: string[];
}

const PATTERNS: Record<string, CaptionPattern> = {
  cake: {
    ctas: [
      'DM to order yours today!',
      'WhatsApp 0700 XXX XXX to place your order',
      'Link in bio to customize your cake',
      'Order now and make your celebration unforgettable',
      'Limited slots available — book yours now!',
    ],
    hashtagSets: {
      core: ['#CakeKampala', '#UgandanBakery', '#CustomCakesUG', '#CakeLover'],
      type: ['#WeddingCake', '#BirthdayCake', '#Cupcakes', '#FondantCake'],
      local: ['#KampalaFood', '#UgandaFood', '#KampalaEats', '#UGFoodie'],
    },
    emojiSets: ['🎂🍰🧁', '✨🎉💕', '🤤😋🔥', '💝🎂✨'],
    pricePhrases: ['Starting at UGX 50,000', 'Prices from UGX 30,000', 'Affordable luxury from UGX 25,000'],
  },
  bakery: {
    ctas: [
      'Visit us today or DM to order!',
      'WhatsApp 0700 XXX XXX for bulk orders',
      'Link in bio for our full menu',
      'Early bird gets the freshest bread!',
    ],
    hashtagSets: {
      core: ['#KampalaBakery', '#FreshBreadUG', '#ArtisanBaking', '#BakeryLife'],
      type: ['#Sourdough', '#Croissant', '#Pastries', '#FreshBread'],
      local: ['#KampalaFood', '#UgandaEats', '#UGBreakfast', '#KlaFoodie'],
    },
    emojiSets: ['🥖🍞🥐', '☀️☕🥐', '🔥😋🥖', '💛✨🍞'],
    pricePhrases: ['Starting at UGX 5,000', 'From UGX 3,000 a loaf', 'Prices from UGX 2,500'],
  },
  restaurant: {
    ctas: [
      'Reserve your table — DM or call!',
      'WhatsApp 0700 XXX XXX for delivery',
      'Tag someone who needs to try this',
      'Link in bio for our full menu',
    ],
    hashtagSets: {
      core: ['#KampalaRestaurant', '#UGFoodie', '#KlaDining', '#UgandaEats'],
      type: ['#LocalFood', '#FoodLover', '#UGFood', '#KampalaEats'],
      local: ['#UgandanFood', '#EastAfricanFood', '#KlaNightlife', '#UGDining'],
    },
    emojiSets: ['🍽️🥘🔥', '😋👨‍🍳✨', '🤤🍗🌶️', '❤️🍴🥂'],
    pricePhrases: ['Meals from UGX 15,000', 'Starting at UGX 10,000', 'Affordable plates from UGX 8,000'],
  },
  general: {
    ctas: [
      'DM to order!',
      'WhatsApp us at 0700 XXX XXX',
      'Link in bio for details',
      'Limited stock — order now!',
    ],
    hashtagSets: {
      core: ['#KampalaBusiness', '#SupportLocalUG', '#Uganda', '#MadeInUG'],
      type: ['#Quality', '#SmallBusiness', '#LocalFirst', '#ShopLocal'],
      local: ['#Kampala', '#UgandaLife', '#UGBusiness', '#KlaHustle'],
    },
    emojiSets: ['✨🔥💯', '💪🇺🇬❤️', '🎯⭐💫', '🚀💯❤️'],
    pricePhrases: ['Starting at UGX 20,000', 'Prices from UGX 10,000', 'Affordable options from UGX 5,000'],
  },
};

// ─── Helpers ────────────────────────────────────────────────────────────────

function pick<T>(arr: T[]): T {
  return arr[Math.floor(Math.random() * arr.length)];
}

const EMOJI_REGEX = /[\u{1F600}-\u{1F64F}\u{1F300}-\u{1F5FF}\u{1F680}-\u{1F6FF}\u{1F1E0}-\u{1F1FF}\u{1F900}-\u{1F9FF}\u{1FA70}-\u{1FAFF}\u{2600}-\u{26FF}\u{2700}-\u{27BF}]/u;

function hasEmoji(text: string): boolean {
  return EMOJI_REGEX.test(text);
}

/**
 * IMPROVE the user's caption — preserves the original wording verbatim and
 * only appends missing elements (price, CTA, emojis, additional hashtags).
 *
 * This is the v6.1 behaviour. The previous "generate an entirely new caption"
 * behaviour is retained only as the empty-caption fallback.
 */
export function generateImprovedCaption(
  originalCaption: string,
  features: CaptionFeatures,
  category: string,
  trendKeywords: string[] = [],
  topHashtags: string[] = [],
): string {
  const trimmed = originalCaption.trim();
  if (!trimmed) {
    return generateCaptionFromScratch(category, trendKeywords, topHashtags);
  }

  const rules = getCategoryRule(category);
  const pattern = PATTERNS[category] || PATTERNS.general;
  const additions: string[] = [];
  const existingHashtags = (trimmed.match(/#\w+/g) || []).map(h => h.toLowerCase());

  // 1. Append a price phrase if a price is missing AND the category requires one
  if (!features.hasPrice && rules.priceRequired) {
    additions.push(pick(pattern.pricePhrases));
  }

  // 2. Append a CTA if missing
  if (!features.hasCta) {
    additions.push(pick(pattern.ctas));
  }

  // 3. Inject a trend callout if the caption has low trend alignment
  if (trendKeywords.length > 0 && features.trendAlignment.score < 0.3) {
    additions.push(`Trending now: ${trendKeywords.slice(0, 2).join(' & ')}`);
  }

  // 4. Append emojis if the original has none
  if (!hasEmoji(trimmed)) {
    additions.push(pick(pattern.emojiSets));
  }

  // 5. Compute how many additional hashtags are needed to reach the ideal count
  const neededCount = Math.max(0, rules.idealHashtags - existingHashtags.length);
  const newHashtags: string[] = [];

  // Pull from category hashtag sets first
  outer: for (const tags of Object.values(pattern.hashtagSets)) {
    for (const tag of tags) {
      if (newHashtags.length >= neededCount) break outer;
      const lower = tag.toLowerCase();
      if (!existingHashtags.includes(lower) && !newHashtags.includes(tag)) {
        newHashtags.push(tag);
      }
    }
  }

  // Then from top-performing DB hashtags
  for (const tag of topHashtags.slice(0, 5)) {
    if (newHashtags.length >= neededCount) break;
    const formatted = tag.startsWith('#') ? tag : `#${tag}`;
    const lower = formatted.toLowerCase();
    if (!existingHashtags.includes(lower) && !newHashtags.includes(formatted)) {
      newHashtags.push(formatted);
    }
  }

  // Compose — always start with the user's original text. Additions go on a
  // new line so the user's caption stays visually distinct.
  const parts: string[] = [trimmed];
  if (additions.length > 0) parts.push(additions.join(' '));
  if (newHashtags.length > 0) parts.push(newHashtags.join(' '));

  return parts.join('\n\n').replace(/\n{3,}/g, '\n\n').trim();
}

/**
 * Fallback used when the user provided no caption text at all.
 * Composes a category-appropriate caption from scratch.
 */
function generateCaptionFromScratch(
  category: string,
  trendKeywords: string[],
  topHashtags: string[],
): string {
  const pattern = PATTERNS[category] || PATTERNS.general;
  const rules = getCategoryRule(category);
  const item = category === 'cake' ? 'cake' : category === 'bakery' ? 'fresh bread' : category === 'restaurant' ? 'dish' : 'product';

  const parts: string[] = [];

  // Compose a short opening
  parts.push(`Indulge in our ${item} — crafted with love and the finest ingredients.`);

  // Price
  if (rules.priceRequired) {
    parts.push(pick(pattern.pricePhrases) + '.');
  }

  // Trend
  if (trendKeywords.length > 0) {
    parts.push(`Join the ${trendKeywords[0]} trend!`);
  }

  // CTA
  parts.push(pick(pattern.ctas));

  // Emojis
  parts.push(pick(pattern.emojiSets));

  // Hashtags
  const allTags: string[] = [];
  for (const tags of Object.values(pattern.hashtagSets)) {
    allTags.push(...tags);
  }
  for (const tag of topHashtags.slice(0, 5)) {
    const formatted = tag.startsWith('#') ? tag : `#${tag}`;
    if (!allTags.includes(formatted)) allTags.push(formatted);
  }
  parts.push('\n\n' + allTags.slice(0, rules.idealHashtags).join(' '));

  return parts.join(' ').trim();
}

// ─── Platform-Specific Variants ────────────────────────────────────────────

export function generatePlatformVariants(
  baseCaption: string,
  features: CaptionFeatures,
  category: string,
): { platform: 'instagram' | 'twitter' | 'facebook'; caption: string; hashtags: string[]; scorePrediction: number; reasoning: string }[] {
  const rules = getCategoryRule(category);
  const hashtags = (baseCaption.match(/#\w+/g) || []);
  const textOnly = baseCaption.replace(/#\w+/g, '').replace(/\n{2,}/g, '\n').trim();

  return [
    {
      platform: 'instagram',
      caption: baseCaption, // Full caption with all hashtags
      hashtags,
      scorePrediction: Math.min(10, 6 + features.hashtagCount * 0.2 + (features.hasCta ? 1 : 0) + (features.hasPrice ? 0.5 : 0)),
      reasoning: 'Instagram favors longer captions with 8+ hashtags, CTA, and price mentions for maximum discoverability and conversion',
    },
    {
      platform: 'twitter',
      caption: `${textOnly.slice(0, 220)}... ${hashtags.slice(0, 3).join(' ')}`.trim().slice(0, 280),
      hashtags: hashtags.slice(0, 3),
      scorePrediction: Math.min(10, 5.5 + (features.hasCta ? 1.5 : 0) + (features.sentiment.polarity > 0 ? 0.5 : 0)),
      reasoning: 'Twitter rewards brevity — concise text with 2-3 strategic hashtags and strong CTA drives the most engagement',
    },
    {
      platform: 'facebook',
      caption: textOnly + (hashtags.length > 0 ? '\n\n' + hashtags.slice(0, 5).join(' ') : ''),
      hashtags: hashtags.slice(0, 5),
      scorePrediction: Math.min(10, 6 + (features.wordCount > 80 ? 1 : 0) + (features.hasCta ? 1 : 0) + (features.sentiment.polarity > 0 ? 0.5 : 0)),
      reasoning: 'Facebook values storytelling — longer, personal captions with moderate hashtags and emotional connection perform best',
    },
  ];
}

// Avoid unused import warnings
export const _PATTERNS = PATTERNS;
