// TrendLens AI v6.0 - Trend Chart Component

'use client';

import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, CartesianGrid } from 'recharts';
import type { TrendSignal } from '@/lib/types';

interface TrendChartProps {
  trends: TrendSignal[];
}

export function TrendChart({ trends }: TrendChartProps) {
  if (!trends || trends.length === 0) {
    return (
      <div className="flex items-center justify-center h-48 text-muted-foreground text-sm">
        No trend data available
      </div>
    );
  }

  const chartData = trends
    .sort((a, b) => b.growthRate - a.growthRate)
    .map(t => ({
      name: t.keyword.replace('#', '').split(' ').slice(0, 2).join(' '),
      score: Math.round(t.score * 100),
      volume: t.volume,
      growthRate: Math.round(t.growthRate * 100),
      fill: t.growthRate > 0.2 ? '#0ea5e9' : t.growthRate > 0.1 ? '#fdba74' : '#6b7280',
    }));

  return (
    <div className="w-full">
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={chartData} margin={{ left: 10, right: 20, top: 10, bottom: 60 }}>
          <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
          <XAxis
            dataKey="name"
            fontSize={10}
            angle={-45}
            textAnchor="end"
            height={70}
          />
          <YAxis
            fontSize={11}
            tickFormatter={(v: number) => `${v}%`}
            domain={[0, 100]}
          />
          <Tooltip
            formatter={(value, name) => {
              const numVal = Number(value ?? 0);
              const nameStr = String(name ?? '');
              if (nameStr === 'score') return [`${numVal}%`, 'Trend Score'];
              if (nameStr === 'volume') return [numVal.toLocaleString(), 'Volume'];
              if (nameStr === 'growthRate') return [`${numVal}%`, 'Growth Rate'];
              return [numVal, nameStr];
            }}
            contentStyle={{ fontSize: '12px' }}
          />
          <Bar dataKey="score" name="score" radius={[4, 4, 0, 0]} maxBarSize={35}>
            {chartData.map((entry, index) => (
              <Cell key={index} fill={entry.fill} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <div className="flex items-center justify-center gap-4 mt-2 text-xs text-muted-foreground">
        <span className="flex items-center gap-1">
          <span className="w-3 h-3 rounded-sm bg-sky-500 inline-block" /> Surging (&gt;20%)
        </span>
        <span className="flex items-center gap-1">
          <span className="w-3 h-3 rounded-sm bg-orange-300 inline-block" /> Rising (10-20%)
        </span>
        <span className="flex items-center gap-1">
          <span className="w-3 h-3 rounded-sm bg-gray-500 inline-block" /> Stable (&lt;10%)
        </span>
      </div>
    </div>
  );
}
