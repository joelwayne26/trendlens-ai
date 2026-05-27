/**
 * GET /api/cron/refresh-trends
 * Vercel Cron Job endpoint — refreshes trend data every 4 hours.
 * Called automatically by Vercel's cron scheduler.
 */
import { NextRequest, NextResponse } from 'next/server';
import { forceRefreshTrends } from '@/lib/ai/trend-engine';

export async function GET(request: NextRequest) {
  // Verify this is called by Vercel Cron (or authorized)
  const authHeader = request.headers.get('authorization');
  if (authHeader !== `Bearer ${process.env.CRON_SECRET}` && process.env.CRON_SECRET) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  try {
    const results = await forceRefreshTrends(['cake', 'bakery', 'restaurant', 'general']);

    const totalTrends = Object.values(results).reduce((sum, n) => sum + n, 0);

    return NextResponse.json({
      success: true,
      refreshedAt: new Date().toISOString(),
      categories: results,
      totalTrends,
      message: `Refreshed ${totalTrends} trends across ${Object.keys(results).length} categories`,
    });
  } catch (error) {
    return NextResponse.json({
      success: false,
      error: String(error),
      refreshedAt: new Date().toISOString(),
    }, { status: 500 });
  }
}
