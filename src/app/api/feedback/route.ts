/** POST /api/feedback — User feedback on suggestions */
import { NextRequest, NextResponse } from 'next/server';
import { healthCheck, FeedbackRepository } from '@/lib/db/client';

export async function POST(request: NextRequest) {
  try {
    const { evaluationId, type, rating, comment } = await request.json();
    if (!evaluationId || !type || !rating) {
      return NextResponse.json({ error: 'Missing required fields' }, { status: 400 });
    }

    const dbConnected = await healthCheck();
    if (!dbConnected) {
      // Graceful degradation: accept feedback even without DB
      return NextResponse.json({
        success: true,
        stored: false,
        message: 'Feedback received (not stored — database not connected)',
      });
    }

    const repo = new FeedbackRepository();
    await repo.insertOne({ evaluation_id: evaluationId, type, rating, comment: comment || '' });
    return NextResponse.json({ success: true, stored: true });
  } catch (error) {
    return NextResponse.json({ error: String(error) }, { status: 500 });
  }
}
