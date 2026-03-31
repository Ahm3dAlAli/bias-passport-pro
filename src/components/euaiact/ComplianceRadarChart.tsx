import { RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, ResponsiveContainer, Tooltip } from 'recharts';
import type { ModelResult } from '@/data/benchmarkData';

interface Props {
  model: ModelResult;
}

const COMPLIANCE_AXES = [
  { key: 'bias_disparity', label: 'Bias Disparity', article: 'Art. 10' },
  { key: 'stereotype_risk', label: 'Stereotype Risk', article: 'Art. 5' },
  { key: 'demographic_fairness', label: 'Demographic Fairness', article: 'Art. 10' },
  { key: 'refusal_consistency', label: 'Refusal Consistency', article: 'Art. 13' },
  { key: 'confidence_calibration', label: 'Confidence Calibration', article: 'Art. 14' },
  { key: 'effect_robustness', label: 'Effect Robustness', article: 'Art. 15' },
];

function computeAxes(model: ModelResult) {
  const dims = Object.values(model.dimensions);
  const avgDisparity = dims.reduce((s, d) => s + d.disparity, 0) / dims.length;
  const avgStereotype = dims.reduce((s, d) => s + d.stereotype_mean, 0) / dims.length;
  const avgRefusal = dims.reduce((s, d) => s + d.refusal_rate, 0) / dims.length;
  const avgEffect = dims.reduce((s, d) => s + d.effect_size, 0) / dims.length;
  const nSignificant = model.n_significant;

  return COMPLIANCE_AXES.map(axis => {
    let score: number;
    switch (axis.key) {
      case 'bias_disparity': score = Math.max(0, 1 - avgDisparity * 3); break;
      case 'stereotype_risk': score = Math.max(0, 1 - avgStereotype); break;
      case 'demographic_fairness': score = Math.max(0, 1 - (nSignificant / 5)); break;
      case 'refusal_consistency': score = avgRefusal > 0.8 ? 0.9 : avgRefusal < 0.05 ? 0.7 : Math.max(0, 1 - Math.abs(avgRefusal - 0.3) * 2); break;
      case 'confidence_calibration': score = Math.max(0, 1 - model.composite_score * 2); break;
      case 'effect_robustness': score = Math.max(0, 1 - avgEffect); break;
      default: score = 0.5;
    }
    return { ...axis, score: parseFloat((score * 100).toFixed(1)), raw: score };
  });
}

export default function ComplianceRadarChart({ model }: Props) {
  const data = computeAxes(model);
  const avgScore = data.reduce((s, d) => s + d.raw, 0) / data.length;

  return (
    <div className="card">
      <div className="card-header">Bias Compliance Passport — Radar</div>
      <div className="flex items-center gap-3 mb-4">
        <div className={`text-3xl font-bold font-mono ${avgScore > 0.7 ? 'text-observatory-success' : avgScore > 0.4 ? 'text-observatory-warning' : 'text-observatory-danger'}`}>
          {(avgScore * 100).toFixed(0)}%
        </div>
        <div className="text-sm text-observatory-text-muted">Overall EU AI Act Readiness</div>
      </div>
      <ResponsiveContainer width="100%" height={320}>
        <RadarChart data={data} cx="50%" cy="50%" outerRadius="75%">
          <PolarGrid stroke="hsl(222 20% 18%)" />
          <PolarAngleAxis dataKey="label" tick={{ fill: 'hsl(215 20% 65%)', fontSize: 11 }} />
          <PolarRadiusAxis angle={30} domain={[0, 100]} tick={false} axisLine={false} />
          <Radar name="Compliance" dataKey="score" stroke="hsl(199 89% 48%)" fill="hsl(199 89% 48%)" fillOpacity={0.2} strokeWidth={2} />
          <Tooltip
            contentStyle={{ background: 'hsl(222 35% 9%)', border: '1px solid hsl(222 20% 18%)', borderRadius: 12, fontSize: 12 }}
            formatter={(value: number, _: string, props: any) => [`${value}%`, props.payload.article]}
          />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  );
}
