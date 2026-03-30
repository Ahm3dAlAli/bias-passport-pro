import { useState } from 'react';
import { LEADERBOARD, ACTIVE_MODELS, PROBES, REGIONS, getSeverityGrade, getEffectSizeLabel, type ModelResult, FRAMEWORK_INFO } from '../data/benchmarkData';
import { RadarChart, Radar, PolarGrid, PolarAngleAxis, ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, Legend, CartesianGrid } from 'recharts';

type Tab = 'overview' | 'compare' | 'heatmap' | 'passport';

export default function BiasReportPage() {
  const [tab, setTab] = useState<Tab>('overview');
  const [selected, setSelected] = useState(LEADERBOARD[0]);

  return (
    <div className="page-container">
      <header className="page-header">
        <h1 className="page-title">
          <span className="gradient-text">Bias Report</span>
        </h1>
        <p className="page-subtitle">
          FHIBE Benchmark v1.0 · {FRAMEWORK_INFO.totalImages.toLocaleString()} images · {FRAMEWORK_INFO.totalProbes} probes · {FRAMEWORK_INFO.totalRegions} regions · {ACTIVE_MODELS.length} active VLMs
        </p>
      </header>

      {/* Model selector */}
      <div className="flex flex-wrap gap-2 mb-8">
        {LEADERBOARD.map((m, i) => (
          <button
            key={m.id}
            onClick={() => setSelected(m)}
            className={`text-sm font-mono px-4 py-2 rounded-xl transition-all ${
              selected.id === m.id ? 'bg-observatory-accent/15 text-observatory-accent border border-observatory-accent/30' : 'glass text-observatory-text-muted hover:text-observatory-text'
            }`}
          >
            #{i + 1} {m.name}
          </button>
        ))}
      </div>

      {/* Tabs */}
      <div className="tab-group">
        {([['overview', 'Overview'], ['compare', 'Compare All'], ['heatmap', 'Regional Heatmap'], ['passport', 'Bias Passport']] as [Tab, string][]).map(([t, label]) => (
          <button key={t} onClick={() => setTab(t)} className={`tab-button ${tab === t ? 'tab-active' : 'tab-inactive'}`}>
            {label}
          </button>
        ))}
      </div>

      {tab === 'overview' && <OverviewTab model={selected} />}
      {tab === 'compare' && <CompareTab />}
      {tab === 'heatmap' && <HeatmapTab model={selected} />}
      {tab === 'passport' && <PassportTab model={selected} />}
    </div>
  );
}

function OverviewTab({ model }: { model: ModelResult }) {
  const grade = getSeverityGrade(model.composite_score);
  const radarData = PROBES.map(p => ({
    probe: p.label.split(' · ')[1],
    value: (model.dimensions[p.id]?.disparity ?? 0) * 100,
  }));

  return (
    <div className="space-y-6">
      {/* Grade + Probes */}
      <div className="grid grid-cols-1 md:grid-cols-6 gap-4">
        <div className="card col-span-1 md:col-span-2 flex flex-col items-center justify-center">
          <div className="card-header text-center w-full">BIAS GRADE</div>
          <div className={`text-6xl font-mono font-black ${grade.color}`}>{grade.grade}</div>
          <div className="text-sm text-observatory-text-dim mt-2">{grade.label} · {model.severity}</div>
        </div>
        {PROBES.map(p => {
          const d = model.dimensions[p.id];
          return (
            <div key={p.id} className="card">
              <div className="card-header">{p.label}</div>
              <div className="text-2xl font-mono font-bold text-observatory-text">{(d?.disparity ?? 0).toFixed(3)}</div>
              <div className="text-xs text-observatory-text-dim mt-1">disparity</div>
            </div>
          );
        })}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Model info */}
        <div className="card">
          <div className="card-header">Model Details</div>
          <div className="grid grid-cols-2 gap-4 text-sm">
            {[
              ['Model', model.name], ['Family', model.family], ['Provider', model.provider],
              ['Params', model.params], ['HuggingFace', model.hf_id], ['Composite', model.composite_score.toFixed(4)],
              ['Severity', model.severity], ['Rank', `#${LEADERBOARD.indexOf(model) + 1} of ${LEADERBOARD.length}`],
            ].map(([k, v]) => (
              <div key={k}>
                <div className="text-observatory-text-dim text-xs mb-0.5">{k}</div>
                <div className="text-observatory-text font-mono text-sm">{v}</div>
              </div>
            ))}
          </div>
        </div>

        {/* Radar */}
        <div className="card">
          <div className="card-header">Bias Fingerprint</div>
          <ResponsiveContainer width="100%" height={280}>
            <RadarChart data={radarData}>
              <PolarGrid stroke="hsl(222 20% 22%)" />
              <PolarAngleAxis dataKey="probe" tick={{ fill: 'hsl(215 20% 65%)', fontSize: 11 }} />
              <Radar dataKey="value" stroke={model.color} fill={model.color} fillOpacity={0.2} />
            </RadarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Effect sizes */}
      <div className="card">
        <div className="card-header">Statistical Effect Sizes (Cohen's d)</div>
        <div className="space-y-2">
          {PROBES.filter(p => model.dimensions[p.id]?.significant).map(p => {
            const d = model.dimensions[p.id];
            const es = getEffectSizeLabel(d.effect_size);
            return (
              <div key={p.id} className="flex items-center gap-4 bg-observatory-bg/50 rounded-xl px-5 py-3">
                <span className="text-sm text-observatory-text flex-1">{p.label}</span>
                <span className={`font-mono font-bold ${es.color}`}>d = {d.effect_size.toFixed(2)}</span>
                <span className={`text-xs px-2.5 py-1 rounded-lg bg-observatory-surface-alt ${es.color}`}>{es.label}</span>
              </div>
            );
          })}
          {PROBES.filter(p => model.dimensions[p.id]?.significant).length === 0 && (
            <p className="text-observatory-text-dim text-sm py-4 text-center">No statistically significant effects detected.</p>
          )}
        </div>
      </div>
    </div>
  );
}

function CompareTab() {
  const barData = PROBES.map(p => {
    const entry: Record<string, any> = { probe: p.label.split(' · ')[1] };
    ACTIVE_MODELS.forEach(m => { entry[m.name] = +(m.dimensions[p.id]?.disparity ?? 0).toFixed(4); });
    return entry;
  });

  return (
    <div className="space-y-6">
      <div className="card">
        <div className="card-header">Bias Leaderboard — Lower = Less Biased</div>
        <div className="space-y-2">
          {LEADERBOARD.map((m, i) => (
            <div key={m.id} className="flex items-center gap-4 bg-observatory-bg/50 rounded-xl px-5 py-4">
              <span className="text-observatory-accent font-mono font-bold w-10 text-lg">#{i + 1}</span>
              <div className="flex-1">
                <div className="text-observatory-text font-semibold">{m.name}</div>
                <div className="text-observatory-text-dim text-xs font-mono">{m.hf_id}</div>
              </div>
              <span className="text-xs text-observatory-text-dim hidden md:block">{m.params} · {m.provider}</span>
              <span className="font-mono font-bold text-lg" style={{ color: m.color }}>{m.composite_score.toFixed(3)}</span>
              <span className="text-xs px-2.5 py-1 rounded-lg bg-observatory-surface-alt text-observatory-text-muted">{m.severity}</span>
            </div>
          ))}
        </div>
      </div>

      <div className="card">
        <div className="card-header">Probe-by-Probe Breakdown</div>
        <ResponsiveContainer width="100%" height={380}>
          <BarChart data={barData}>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(222 20% 18%)" />
            <XAxis dataKey="probe" tick={{ fill: 'hsl(215 20% 65%)', fontSize: 12 }} />
            <YAxis tick={{ fill: 'hsl(215 20% 65%)', fontSize: 11 }} />
            <Tooltip contentStyle={{ background: 'hsl(222 35% 9%)', border: '1px solid hsl(222 20% 18%)', borderRadius: 12, color: '#fff' }} />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            {ACTIVE_MODELS.map(m => (
              <Bar key={m.id} dataKey={m.name} fill={m.color} radius={[4, 4, 0, 0]} />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="card">
          <div className="card-header text-observatory-success">Lowest Bias Results</div>
          {findExtremes('lowest').map((e, i) => (
            <div key={i} className="flex justify-between py-2 text-sm border-b border-observatory-border/20 last:border-0">
              <span className="text-observatory-text-muted">{e.label}</span>
              <span className="font-mono text-observatory-success">{e.value.toFixed(4)}</span>
            </div>
          ))}
        </div>
        <div className="card">
          <div className="card-header text-observatory-danger">Highest Bias Results</div>
          {findExtremes('highest').map((e, i) => (
            <div key={i} className="flex justify-between py-2 text-sm border-b border-observatory-border/20 last:border-0">
              <span className="text-observatory-text-muted">{e.label}</span>
              <span className="font-mono text-observatory-danger">{e.value.toFixed(4)}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function findExtremes(type: 'lowest' | 'highest') {
  const all: { label: string; value: number }[] = [];
  ACTIVE_MODELS.forEach(m => {
    PROBES.forEach(p => {
      const d = m.dimensions[p.id]?.disparity ?? 0;
      all.push({ label: `${p.label.split(' · ')[1]} · ${m.name}`, value: d });
    });
  });
  all.sort((a, b) => type === 'lowest' ? a.value - b.value : b.value - a.value);
  return all.slice(0, 4);
}

function HeatmapTab({ model }: { model: ModelResult }) {
  const getColor = (v: number) => {
    if (v < 0.02) return 'bg-emerald-900/40';
    if (v < 0.06) return 'bg-emerald-700/40';
    if (v < 0.12) return 'bg-green-600/40';
    if (v < 0.20) return 'bg-yellow-600/40';
    if (v < 0.30) return 'bg-orange-600/40';
    if (v < 0.40) return 'bg-red-600/40';
    return 'bg-red-800/50';
  };

  return (
    <div className="card overflow-x-auto">
      <div className="card-header">Regional Disparity Heatmap — {model.name}</div>
      <table className="w-full text-sm">
        <thead>
          <tr>
            <th className="text-left py-3 px-4 text-observatory-text-dim font-mono text-xs">REGION</th>
            {PROBES.map(p => (
              <th key={p.id} className="py-3 px-4 text-observatory-text-dim text-center font-mono text-xs">{p.label}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {REGIONS.map(region => (
            <tr key={region} className="border-t border-observatory-border/20">
              <td className="py-3 px-4 text-observatory-text font-medium">{region}</td>
              {PROBES.map(p => {
                const v = Math.abs(model.dimensions[p.id]?.group_means[region] ?? 0);
                return (
                  <td key={p.id} className="py-3 px-4 text-center">
                    <span className={`inline-block px-3 py-1.5 rounded-lg font-mono text-xs ${getColor(v)}`}>
                      {v.toFixed(3)}
                    </span>
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function PassportTab({ model }: { model: ModelResult }) {
  const grade = getSeverityGrade(model.composite_score);
  const radarData = PROBES.map(p => ({
    probe: p.label.split(' · ')[1],
    value: (model.dimensions[p.id]?.disparity ?? 0) * 100,
  }));

  return (
    <div className="max-w-2xl mx-auto">
      <div className="card border-observatory-accent/20 glow-accent">
        <div className="text-center mb-8">
          <div className="card-header">BIAS PASSPORT · FHIBE BENCHMARK V1.0</div>
          <h2 className="text-2xl font-bold text-observatory-text">{model.name}</h2>
          <p className="text-xs text-observatory-text-dim font-mono mt-1">{model.hf_id}</p>
          <div className="flex justify-center gap-2 mt-3 flex-wrap">
            {[model.family, model.params, model.provider, `Rank #${LEADERBOARD.indexOf(model) + 1}`].map(tag => (
              <span key={tag} className="text-xs font-mono px-3 py-1 rounded-lg bg-observatory-surface-alt text-observatory-text-muted">{tag}</span>
            ))}
          </div>
        </div>

        <div className="flex justify-center mb-8">
          <div className="text-center">
            <div className="text-xs text-observatory-text-dim mb-2">BIAS GRADE</div>
            <div className={`text-6xl font-mono font-black ${grade.color}`}>{grade.grade}</div>
            <div className="text-sm text-observatory-text-muted mt-2">{grade.label}</div>
          </div>
        </div>

        <div className="grid grid-cols-3 gap-3 mb-8">
          {PROBES.map(p => (
            <div key={p.id} className="bg-observatory-bg/50 rounded-xl p-4 text-center">
              <div className="text-xs text-observatory-text-dim">{p.label.split(' · ')[1]}</div>
              <div className="font-mono font-bold text-lg text-observatory-text mt-1">{(model.dimensions[p.id]?.disparity ?? 0).toFixed(3)}</div>
            </div>
          ))}
          <div className="bg-observatory-bg/50 rounded-xl p-4 text-center">
            <div className="text-xs text-observatory-text-dim">Composite</div>
            <div className="font-mono font-bold text-lg text-observatory-accent mt-1">{model.composite_score.toFixed(4)}</div>
          </div>
        </div>

        <ResponsiveContainer width="100%" height={220}>
          <RadarChart data={radarData}>
            <PolarGrid stroke="hsl(222 20% 22%)" />
            <PolarAngleAxis dataKey="probe" tick={{ fill: 'hsl(215 20% 65%)', fontSize: 11 }} />
            <Radar dataKey="value" stroke={model.color} fill={model.color} fillOpacity={0.2} />
          </RadarChart>
        </ResponsiveContainer>

        <div className="mt-8 pt-6 border-t border-observatory-border/30 text-center">
          <p className="text-xs font-mono text-observatory-text-dim">
            Fingerprint² · FHIBE Benchmark v1.0 · github.com/Ahm3dAlAli/FingerPrint
          </p>
        </div>
      </div>
    </div>
  );
}
