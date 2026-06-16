/**
 * Unit tests for src/lib/ai/feature-extractor.ts
 *
 * These tests pin down the behaviors that previously caused production bugs:
 *   1. CTA detection (whatsapp/dm/link in bio/etc.)
 *   2. Price detection — must NOT match phone numbers (the bug we fixed)
 *   3. Hashtag counting
 *   4. Emoji counting
 *   5. Empty-caption safety
 */
import { describe, it, expect } from 'vitest';
import { extractCaptionFeatures } from './feature-extractor';

describe('extractCaptionFeatures', () => {
  describe('CTA detection', () => {
    it('detects "whatsapp" as a CTA', () => {
      const f = extractCaptionFeatures('Best food in Uganda. WhatsApp us on 0700 123 456');
      expect(f.hasCta).toBe(true);
      expect(f.ctaType).toBe('whatsapp');
    });

    it('detects "DM to order" as a CTA', () => {
      const f = extractCaptionFeatures('Fresh cakes! DM to order yours today.');
      expect(f.hasCta).toBe(true);
    });

    it('detects "link in bio" as a CTA', () => {
      const f = extractCaptionFeatures('Link in bio for our full menu');
      expect(f.hasCta).toBe(true);
    });

    it('detects "order now" as a CTA', () => {
      const f = extractCaptionFeatures('Order now while stock lasts');
      expect(f.hasCta).toBe(true);
    });

    it('returns false for captions without a CTA', () => {
      const f = extractCaptionFeatures('Best food in Uganda. We make tasty things.');
      expect(f.hasCta).toBe(false);
    });
  });

  describe('price detection (regression: phone-number false positive)', () => {
    it('detects UGX price', () => {
      const f = extractCaptionFeatures('Cakes starting at UGX 50,000. DM to order.');
      expect(f.hasPrice).toBe(true);
    });

    it('detects $ price', () => {
      const f = extractCaptionFeatures('Special offer: $5 only');
      expect(f.hasPrice).toBe(true);
    });

    it('detects standalone 4-7 digit price number', () => {
      const f = extractCaptionFeatures('Birthday cake 50000. DM to order.');
      expect(f.hasPrice).toBe(true);
    });

    it('does NOT flag 9-digit phone numbers as prices (regression test)', () => {
      // This was the bug: "070735363" (9 digits) matched the old /\d{3,}/ regex
      const f = extractCaptionFeatures('best food in uganda #uganda #food whatsapp us on 070735363');
      expect(f.hasPrice).toBe(false);
    });

    it('does NOT flag 10-digit phone numbers as prices', () => {
      const f = extractCaptionFeatures('Call 0700123456 to order');
      expect(f.hasPrice).toBe(false);
    });

    it('does NOT flag 3-digit numbers as prices (too ambiguous)', () => {
      const f = extractCaptionFeatures('We have 3 days left');
      expect(f.hasPrice).toBe(false);
    });
  });

  describe('hashtag counting', () => {
    it('counts hashtags correctly', () => {
      const f = extractCaptionFeatures('best food in uganda #uganda #food');
      expect(f.hashtagCount).toBe(2);
    });

    it('returns 0 when no hashtags', () => {
      const f = extractCaptionFeatures('best food in uganda');
      expect(f.hashtagCount).toBe(0);
    });

    it('handles hashtags with underscores and digits', () => {
      const f = extractCaptionFeatures('#Cake_Kampala #Fresh2024');
      expect(f.hashtagCount).toBe(2);
    });
  });

  describe('emoji counting', () => {
    it('counts food emojis correctly', () => {
      const f = extractCaptionFeatures('Fresh cakes 🎂🍰🧁');
      expect(f.emojiCount).toBe(3);
    });

    it('counts heart emojis correctly (regression-style)', () => {
      const f = extractCaptionFeatures('best food in uganda #food💕💕💕');
      expect(f.emojiCount).toBe(3);
    });

    it('returns 0 when no emojis', () => {
      const f = extractCaptionFeatures('Just text, no emojis here');
      expect(f.emojiCount).toBe(0);
    });
  });

  describe('word counting', () => {
    it('counts words correctly', () => {
      const f = extractCaptionFeatures('best food in uganda');
      expect(f.wordCount).toBe(4);
    });

    it('handles empty caption safely', () => {
      const f = extractCaptionFeatures('');
      expect(f.wordCount).toBe(0);
      expect(f.hashtagCount).toBe(0);
      expect(f.emojiCount).toBe(0);
      expect(f.hasCta).toBe(false);
      expect(f.hasPrice).toBe(false);
    });
  });

  describe('sentiment', () => {
    it('detects positive sentiment', () => {
      const f = extractCaptionFeatures('Amazing delicious fresh cakes!');
      expect(f.sentiment.polarity).toBeGreaterThan(0);
    });

    it('detects negative sentiment', () => {
      const f = extractCaptionFeatures('Sorry, terrible experience, worst ever');
      expect(f.sentiment.polarity).toBeLessThan(0);
    });

    it('is neutral for plain text', () => {
      const f = extractCaptionFeatures('We sell food in Kampala');
      expect(Math.abs(f.sentiment.polarity)).toBeLessThan(0.05);
    });
  });

  describe('captionScore', () => {
    it('is higher for a complete caption (CTA + hashtags + emoji + price)', () => {
      const complete = extractCaptionFeatures(
        'Fresh cakes available! UGX 50,000. DM to order. #CakeKampala #UgandanBakery #WeddingCake #BirthdayCake #Cupcakes #FreshBakes #KampalaFood #UGFoodie 🎂🍰🧁',
        'cake',
      );
      const sparse = extractCaptionFeatures('cakes', 'cake');
      expect(complete.captionScore).toBeGreaterThan(sparse.captionScore);
    });
  });
});
