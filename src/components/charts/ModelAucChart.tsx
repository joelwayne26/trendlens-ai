// TrendLens AI v6.0 - Model AUC Chart Component

'use client';

import { XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Area, AreaChart } from 'recharts';

interface ModelAucChartProps {
  models: Array<Record<string, unknown>>;
}

export function ModelAucChart({ models }: ModelAucChartProps) {
  if (!models || models.length === 0) {
    return (
      <div className="flex items-center justify-center h-48 text-muted-foreground text-sm">
        No model data available
      </div>
    );
  }

  const chartData = [...models]
    .sort((a, b) => new Date(String(a.trained_at || a.trainedAt || '')).getTime() - new Date(String(b.trained_at || b.trainedAt || '')).getTime())
    .map(m => ({
      version: String(m.version || ''),
      auc: Number(m.auc || m.aucScore || 0),
      accuracy: Number(m.accuracy || 0),
      trainedAt: new Date(String(m.trained_at || m.trainedAt || '')).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
      datasetSize: Number(m.samples || m.datasetSize || 0),
    }));

  return (
    <div className="w-full">
      <ResponsiveContainer width="100%" height={280}>
        <AreaChart data={chartData} margin={{ left: 10, right: 20, top: 10, bottom: 10 }}>
          <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
          <XAxis dataKey="version" fontSize={11} />
          <YAxis
            domain={[0.5, 1.0]}
            fontSize={11}
            tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
          />
          <Tooltip
            formatter={(value, name) => {
              const numVal = Number(value ?? 0);
              const nameStr = String(name ?? '');
              const label = nameStr === 'auc' ? 'AUC' : 'Accuracy';
              return [`${(numVal * 100).toFixed(1)}%`, label];
            }}
            contentStyle={{ fontSize: '12px' }}
          />
          <Area
            type="monotone"
            dataKey="auc"
            stroke="#0ea5e9"
            fill="#0ea5e9"
            fillOpacity={0.1}
            strokeWidth={2}
            name="auc"
          />
          <Area
            type="monotone"
            dataKey="accuracy"
            stroke="#f59e0b"
            fill="#f59e0b"
            fillOpacity={0.1}
            strokeWidth={2}
            name="accuracy"
          />
        </AreaChart>
      </ResponsiveContainer>
      <div className="flex items-center justify-center gap-4 mt-2 text-xs text-muted-foreground">
        <span className="flex items-center gap-1">
          <span className="w-3 h-1 bg-sky-500 inline-block rounded" /> AUC Score
        </span>
        <span className="flex items-center gap-1">
          <span className="w-3 h-1 bg-yellow-500 inline-block rounded" /> Accuracy
        </span>
      </div>
    </div>
  );
}
