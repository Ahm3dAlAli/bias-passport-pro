import type { ModelResult } from '@/data/benchmarkData';
import { PROBES, REGIONS } from '@/data/benchmarkData';

interface Props {
  model: ModelResult;
}

function getHeatColor(value: number): string {
  const abs = Math.abs(value);
  if (abs < 0.05) return 'bg-observatory-success/20 text-observatory-success';
  if (abs < 0.15) return 'bg-observatory-warning/20 text-observatory-warning';
  if (abs < 0.3) return 'bg-orange-500/20 text-orange-400';
  return 'bg-observatory-danger/20 text-observatory-danger';
}

export default function DemographicBreakdown({ model }: Props) {
  return (
    <div className="card">
      <div className="card-header">Demographic Disparity Heatmap</div>
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr>
              <th className="text-left py-2 px-2 text-observatory-text-dim font-mono">Probe</th>
              {REGIONS.map(r => (
                <th key={r} className="text-center py-2 px-1 text-observatory-text-dim font-mono whitespace-nowrap">{r}</th>
              ))}
              <th className="text-center py-2 px-2 text-observatory-text-dim font-mono">Disparity</th>
            </tr>
          </thead>
          <tbody>
            {PROBES.map(probe => {
              const dim = model.dimensions[probe.id];
              if (!dim) return null;
              return (
                <tr key={probe.id} className="border-t border-observatory-border/30">
                  <td className="py-2 px-2 text-observatory-text-muted whitespace-nowrap">{probe.label}</td>
                  {REGIONS.map(region => {
                    const val = dim.group_means[region] ?? 0;
                    return (
                      <td key={region} className="py-2 px-1 text-center">
                        <span className={`inline-block px-2 py-0.5 rounded ${getHeatColor(val)} font-mono`}>
                          {val.toFixed(3)}
                        </span>
                      </td>
                    );
                  })}
                  <td className="py-2 px-2 text-center">
                    <span className={`inline-block px-2 py-0.5 rounded font-mono font-semibold ${getHeatColor(dim.disparity)}`}>
                      {dim.disparity.toFixed(3)}
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div className="mt-4 flex gap-4 text-xs text-observatory-text-dim">
        <span className="flex items-center gap-1"><span className="w-3 h-3 rounded bg-observatory-success/20" /> &lt;0.05</span>
        <span className="flex items-center gap-1"><span className="w-3 h-3 rounded bg-observatory-warning/20" /> 0.05–0.15</span>
        <span className="flex items-center gap-1"><span className="w-3 h-3 rounded bg-orange-500/20" /> 0.15–0.30</span>
        <span className="flex items-center gap-1"><span className="w-3 h-3 rounded bg-observatory-danger/20" /> &gt;0.30</span>
      </div>
    </div>
  );
}
