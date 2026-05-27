/** GET /api/trends — Current trending terms from live sources */
import { NextRequest, NextResponse } from 'next/server';
import { fetchTrends } from '@/lib/ai/trend-engine';

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const category = searchParams.get('category') || 'general';
  const limit = parseInt(searchParams.get('limit') || '20');

  try {
    const signals = await fetchTrends(category, limit);

    // Count sources
    const sources = new Set(signals.map(s => s.source));

    return NextResponse.json({
      category,
      count: signals.length,
      sources: Array.from(sources),
      liveData: !sources.has('domain_knowledge') || sources.size > 1,
      trends: signals,
      fetchedAt: new Date().toISOString(),
    });
  } catch (error) {
    return NextResponse.json({
      category,
      count: 0,
      sources: [],
      liveData: false,
      trends: [],
      error: String(error),
      fetchedAt: new Date().toISOString(),
    });
  }
}
