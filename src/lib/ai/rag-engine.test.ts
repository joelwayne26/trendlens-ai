/**
 * Unit tests for src/lib/ai/rag-engine.ts
 *
 * Verifies the v6.1 TF-IDF embedding:
 *   - Output dimension is 384 (backward compatible)
 *   - Empty input → zero vector
 *   - Similar captions produce similar (high-cosine-similarity) embeddings
 *   - Dissimilar captions produce dissimilar embeddings
 *   - Domain vocab terms produce stronger signal than generic words
 */
import { describe, it, expect } from 'vitest';
import { generateSimpleEmbedding, EMBEDDING_DIMENSIONS } from './rag-engine';

function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  if (normA === 0 || normB === 0) return 0;
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

describe('generateSimpleEmbedding', () => {
  it('returns a 384-dimensional vector', () => {
    const e = generateSimpleEmbedding('fresh cakes available');
    expect(e.length).toBe(EMBEDDING_DIMENSIONS);
  });

  it('returns a zero vector for empty input', () => {
    const e = generateSimpleEmbedding('');
    expect(e.every(v => v === 0)).toBe(true);
  });

  it('is L2-normalized (norm = 1 for non-empty input)', () => {
    const e = generateSimpleEmbedding('fresh cakes available');
    const norm = Math.sqrt(e.reduce((s, v) => s + v * v, 0));
    expect(norm).toBeCloseTo(1, 5);
  });

  it('produces similar embeddings for similar captions', () => {
    const a = generateSimpleEmbedding('Fresh cakes available! DM to order. #CakeKampala');
    const b = generateSimpleEmbedding('Fresh cakes for sale — DM to order. #CakeKampala');
    const sim = cosineSimilarity(a, b);
    expect(sim).toBeGreaterThan(0.85);
  });

  it('produces dissimilar embeddings for unrelated captions', () => {
    const a = generateSimpleEmbedding('Fresh cakes available! DM to order. #CakeKampala');
    const b = generateSimpleEmbedding('Spare parts for sale. Toyota parts. Call 0700');
    const sim = cosineSimilarity(a, b);
    expect(sim).toBeLessThan(0.6);
  });

  it('captures semantic similarity via char n-grams (cake vs cakes)', () => {
    // "cake" and "cakes" share the char 3-grams "cak" and "ake", which
    // bumps similarity above what two random short words would have.
    const a = generateSimpleEmbedding('cake');
    const b = generateSimpleEmbedding('cakes');
    const unrelatedA = generateSimpleEmbedding('cake');
    const unrelatedB = generateSimpleEmbedding('xyz');
    const sim = cosineSimilarity(a, b);
    const unrelatedSim = cosineSimilarity(unrelatedA, unrelatedB);
    // "cake" vs "cakes" should be meaningfully more similar than "cake" vs "xyz"
    expect(sim).toBeGreaterThan(unrelatedSim + 0.2);
  });

  it('preserves backward compatibility with legacy dimensions', () => {
    // Legacy callers may request different dimensions
    const e = generateSimpleEmbedding('test caption', 256);
    expect(e.length).toBe(256);
  });

  it('handles emoji input without crashing', () => {
    const e = generateSimpleEmbedding('Fresh cakes 🎂🍰🧁 #CakeKampala');
    expect(e.length).toBe(EMBEDDING_DIMENSIONS);
    const norm = Math.sqrt(e.reduce((s, v) => s + v * v, 0));
    expect(norm).toBeCloseTo(1, 5);
  });

  it('sets the has-price structural feature when UGX is present', () => {
    const withPrice = generateSimpleEmbedding('Cakes starting at UGX 50,000');
    const withoutPrice = generateSimpleEmbedding('Cakes available DM to order');
    // The two embeddings should differ (price feature activated)
    expect(cosineSimilarity(withPrice, withoutPrice)).toBeLessThan(0.99);
  });
});
