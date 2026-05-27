// TrendLens AI v6.0 - Drift Chart Component

'use client';

import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, ReferenceLine } from 'recharts';

interface DriftChartProps {
  measurements: Array<Record<string, unknown>>;
  threshold?: number;
}

const FEATURE_LABELS: Record<string, string> = {
  hashtag_relevance: 'Hashtags',
  cta_strength: 'CTA',
  image_quality_score: 'Image',
  trend_alignment_score: 'Trends',
  caption_readability: 'Caption',
  engagement_prediction: 'Engagement',
  posting_time_score: 'Time',
  audience_fit_score: 'Audience',
  visual_appeal_score: 'Visual',
  sentiment_score: 'Sentiment',
  overall: 'Overall',
};

export function DriftChart({ measurements, threshold = 0.05 }: DriftChartProps) {
  if (!measurements || measurements.length === 0) {
    return (
      <div className="flex items-center justify-center h-48 text-muted-foreground text-sm">
        No drift measurements available
      </div>
    );
  }

  const chartData = measurements.map(m => {
    const mmdValue = Number(m.mmd_score || m.mmdValue || m.mmdValue || 0);
    const feature = String(m.feature || 'overall');
    return {
      name: FEATURE_LABELS[feature] || feature,
      mmdValue,
      pValue: Number(m.p_value || m.pValue || 0),
      isDrift: Boolean(m.is_drift ?? m.isDrift ?? mmdValue > threshold),
      fill: (m.is_drift ?? m.isDrift ?? mmdValue > threshold) ? '#ef4444' : '#0ea5e9',
    };
  });

  return (
    <div className="w-full">
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={chartData} margin={{ left: 10, right: 20, top: 10, bottom: 40 }}>
          <XAxis
            dataKey="name"
            fontSize={10}
            angle={-35}
            textAnchor="end"
            height={60}
          />
          <YAxis
            fontSize={11}
            tickFormatter={(v: number) => v.toFixed(2)}
            label={{ value: 'MMD', angle: -90, position: 'insideLeft', fontSize: 11 }}
          />
          <Tooltip
            formatter={(value, name) => {
              const numVal = Number(value ?? 0);
              const nameStr = String(name ?? '');
              if (nameStr === 'mmdValue') return [numVal.toFixed(4), 'MMD Value'];
              return [numVal, nameStr];
            }}
            contentStyle={{ fontSize: '12px' }}
          />
          <ReferenceLine
            y={threshold}
            stroke="#f59e0b"
            strokeDasharray="5 5"
            label={{ value: `Threshold (${threshold})`, fontSize: 10, fill: '#f59e0b' }}
          />
          <Bar dataKey="mmdValue" name="MMD Value" radius={[4, 4, 0, 0]} maxBarSize={40}>
            {chartData.map((entry, index) => (
              <Cell key={index} fill={entry.fill} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <div className="flex items-center justify-center gap-4 mt-2 text-xs text-muted-foreground">
        <span className="flex items-center gap-1">
          <span className="w-3 h-3 rounded-sm bg-sky-500 inline-block" /> No drift
        </span>
        <span className="flex items-center gap-1">
          <span className="w-3 h-3 rounded-sm bg-red-500 inline-block" /> Drift detected
        </span>
        <span className="flex items-center gap-1">
          <span className="w-8 border-t-2 border-dashed border-yellow-500 inline-block" /> Threshold
        </span>
      </div>
    </div>
  );
}
