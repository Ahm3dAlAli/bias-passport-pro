import { useState } from 'react';
import { LEADERBOARD, ACTIVE_MODELS, PROBES, REGIONS, getSeverityGrade, getEffectSizeLabel, type ModelResult, FRAMEWORK_INFO } from '../data/benchmarkData';
import { RadarChart, Radar, PolarGrid, PolarAngleAxis, ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, Legend, CartesianGrid } from 'recharts';

type Tab = 'overview' | 'compare' | 'heatmap' | 'passport';

export default function BiasReportPage() {
  const [tab, setTab] = useState<Tab>('overview');
  const [selected, setSelected] = useState(LEADERBOARD[0]);

  return (
    <div className="p-4 md:p-8 max-w-7xl mx-auto">
      <header className="mb-6">
        <h1 className="text-2xl font-mono font-bold gradient-text">Bias Report — FHIBE Benchmark v1.0</h1>
        <p className="text-observatory-text-muted text-sm mt-1">
          {FRAMEWORK_INFO.totalImages.toLocaleString()} images · {FRAMEWORK_INFO.totalProbes} probes · {FRAMEWORK_INFO.totalRegions} regions · {ACTIVE_MODELS.length} active VLMs
        </p>
      </header>

      {/* Mini leaderboard strip */}
      <div className="flex flex-wrap gap-2 mb-6">
        {LEADERBOARD.map((m, i) => (
          <button
            key={m.id}
            onClick={() => setSelected(m)}
            className={`text-xs font-mono px-3 py-1.5 rounded-lg transition-all ${
              selected.id === m.id ? 'bg-observatory-accent/20 text-observatory-accent glow-accent' : 'glass text-observatory-text-muted hover:text-observatory-text'
            }`}
          >
            #{i + 1} {m.name} — {m.composite_score.toFixed(3)}
          </button>
        ))}
      </div>

      {/* Tabs */}
      <div className="flex gap-1 mb-6">
        {(['overview', 'compare', 'heatmap', 'passport'] as Tab[]).map(t => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
              tab === t ? 'bg-observatory-accent/15 text-observatory-accent' : 'text-observatory-text-muted hover:text-observatory-text'
            }`}
          >
            {t === 'overview' ? 'Overview' : t === 'compare' ? 'Compare All' : t === 'heatmap' ? 'Regional Heatmap' : 'Bias Passport'}
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
      <div className="grid grid-cols-2 md:grid-cols-6 gap-3">
        <div className="glass rounded-xl p-4 col-span-2">
          <div className="text-xs text-observatory-text-muted mb-1">BIAS GRADE</div>
          <div className={`text-4xl font-mono font-bold ${grade.color}`}>{grade.grade}</div>
          <div className="text-xs text-observatory-text-dim mt-1">{grade.label} · {model.severity}</div>
        </div>
        {PROBES.map(p => {
          const d = model.dimensions[p.id];
          return (
            <div key={p.id} className="glass rounded-xl p-4">
              <div className="text-[10px] text-observatory-text-muted mb-1">{p.label}</div>
              <div className="text-lg font-mono font-bold text-observatory-text">{(d?.disparity ?? 0).toFixed(3)}</div>
              <div className="text-[10px] text-observatory-text-dim">disparity</div>
            </div>
          );
        })}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Model info */}
        <div className="glass rounded-xl p-6">
          <div className="grid grid-cols-2 gap-3 text-sm">
            {[
              ['Model', model.name], ['Family', model.family], ['Provider', model.provider],
              ['Params', model.params], ['HuggingFace', model.hf_id], ['Composite', model.composite_score.toFixed(4)],
              ['Severity', model.severity], ['Rank', `#${LEADERBOARD.indexOf(model) + 1} of ${LEADERBOARD.length}`],
            ].map(([k, v]) => (
              <div key={k}>
                <div className="text-observatory-text-dim text-xs">{k}</div>
                <div className="text-observatory-text font-mono text-xs">{v}</div>
              </div>
            ))}
          </div>
        </div>

        {/* Radar */}
        <div className="glass rounded-xl p-6">
          <h3 className="text-sm font-mono text-observatory-text-muted mb-4">Bias Fingerprint</h3>
          <ResponsiveContainer width="100%" height={250}>
            <RadarChart data={radarData}>
              <PolarGrid stroke="hsl(222 20% 25%)" />
              <PolarAngleAxis dataKey="probe" tick={{ fill: 'hsl(215 20% 65%)', fontSize: 10 }} />
              <Radar dataKey="value" stroke={model.color} fill={model.color} fillOpacity={0.2} />
            </RadarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Effect sizes */}
      <div className="glass rounded-xl p-6">
        <h3 className="text-sm font-mono text-observatory-text-muted mb-4">Statistical Effect Sizes (Cohen's d)</h3>
        <div className="space-y-2">
          {PROBES.filter(p => model.dimensions[p.id]?.significant).map(p => {
            const d = model.dimensions[p.id];
            const es = getEffectSizeLabel(d.effect_size);
            return (
              <div key={p.id} className="flex items-center gap-3 bg-observatory-bg/50 rounded-lg px-4 py-2">
                <span className="text-sm text-observatory-text flex-1">{p.label}</span>
                <span className={`font-mono font-bold ${es.color}`}>d = {d.effect_size.toFixed(2)}</span>
                <span className={`text-xs ${es.color}`}>{es.label}</span>
              </div>
            );
          })}
          {PROBES.filter(p => model.dimensions[p.id]?.significant).length === 0 && (
            <p className="text-observatory-text-dim text-sm">No statistically significant effects detected.</p>
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
      {/* Leaderboard */}
      <div className="glass rounded-xl p-6">
        <h3 className="text-sm font-mono text-observatory-text-muted mb-4">Bias Leaderboard — Lower = Less Biased</h3>
        <div className="space-y-2">
          {LEADERBOARD.map((m, i) => (
            <div key={m.id} className="flex items-center gap-3 bg-observatory-bg/50 rounded-lg px-4 py-3">
              <span className="text-observatory-accent font-mono font-bold w-8">#{i + 1}</span>
              <div className="flex-1">
                <div className="text-observatory-text font-medium text-sm">{m.name}</div>
                <div className="text-observatory-text-dim text-xs font-mono">{m.hf_id}</div>
              </div>
              <span className="text-xs text-observatory-text-dim">{m.params} · {m.provider}</span>
              <span className="font-mono font-bold text-sm" style={{ color: m.color }}>{m.composite_score.toFixed(3)}</span>
              <span className="text-xs px-2 py-0.5 rounded-full bg-observatory-surface-alt text-observatory-text-muted">{m.severity}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Bar chart */}
      <div className="glass rounded-xl p-6">
        <h3 className="text-sm font-mono text-observatory-text-muted mb-4">Probe-by-Probe Breakdown</h3>
        <ResponsiveContainer width="100%" height={350}>
          <BarChart data={barData}>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(222 20% 20%)" />
            <XAxis dataKey="probe" tick={{ fill: 'hsl(215 20% 65%)', fontSize: 11 }} />
            <YAxis tick={{ fill: 'hsl(215 20% 65%)', fontSize: 11 }} />
            <Tooltip contentStyle={{ background: 'hsl(222 40% 10%)', border: '1px solid hsl(222 20% 20%)', borderRadius: 8, color: '#fff' }} />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            {ACTIVE_MODELS.map(m => (
              <Bar key={m.id} dataKey={m.name} fill={m.color} />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Highlights */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="glass rounded-xl p-5">
          <h4 className="text-xs font-mono text-observatory-success mb-3">Lowest Bias Results</h4>
          {findExtremes('lowest').map((e, i) => (
            <div key={i} className="flex justify-between py-1 text-sm">
              <span className="text-observatory-text-muted">{e.label}</span>
              <span className="font-mono text-observatory-success">{e.value.toFixed(4)}</span>
            </div>
          ))}
        </div>
        <div className="glass rounded-xl p-5">
          <h4 className="text-xs font-mono text-observatory-danger mb-3">Highest Bias Results</h4>
          {findExtremes('highest').map((e, i) => (
            <div key={i} className="flex justify-between py-1 text-sm">
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
    <div className="space-y-6">
      <div className="glass rounded-xl p-6 overflow-x-auto">
        <h3 className="text-sm font-mono text-observatory-text-muted mb-4">Regional Disparity Heatmap — {model.name}</h3>
        <table className="w-full text-xs">
          <thead>
            <tr>
              <th className="text-left py-2 px-3 text-observatory-text-dim">REGION</th>
              {PROBES.map(p => (
                <th key={p.id} className="py-2 px-3 text-observatory-text-dim text-center">{p.label}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {REGIONS.map(region => (
              <tr key={region}>
                <td className="py-2 px-3 text-observatory-text font-medium">{region}</td>
                {PROBES.map(p => {
                  const v = Math.abs(model.dimensions[p.id]?.group_means[region] ?? 0);
                  return (
                    <td key={p.id} className="py-2 px-3 text-center">
                      <span className={`inline-block px-2 py-1 rounded font-mono ${getColor(v)}`}>
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
      <div className="glass rounded-xl p-8 border border-observatory-accent/20 glow-accent">
        <div className="text-center mb-6">
          <div className="text-[10px] font-mono text-observatory-accent tracking-widest mb-2">BIAS PASSPORT · FHIBE BENCHMARK V1.0</div>
          <h2 className="text-xl font-mono font-bold text-observatory-text">{model.name}</h2>
          <p className="text-xs text-observatory-text-dim font-mono mt-1">{model.hf_id}</p>
          <div className="flex justify-center gap-2 mt-3">
            {[model.family, model.params, model.provider, `Rank #${LEADERBOARD.indexOf(model) + 1}`].map(tag => (
              <span key={tag} className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-observatory-surface-alt text-observatory-text-muted">{tag}</span>
            ))}
          </div>
        </div>

        <div className="flex justify-center mb-6">
          <div className="text-center">
            <div className="text-xs text-observatory-text-dim mb-1">BIAS GRADE</div>
            <div className={`text-5xl font-mono font-bold ${grade.color}`}>{grade.grade}</div>
            <div className="text-xs text-observatory-text-muted mt-1">{grade.label}</div>
          </div>
        </div>

        <div className="grid grid-cols-3 gap-3 mb-6">
          {PROBES.map(p => (
            <div key={p.id} className="bg-observatory-bg/50 rounded-lg p-3 text-center">
              <div className="text-[10px] text-observatory-text-dim">{p.label.split(' · ')[1]}</div>
              <div className="font-mono font-bold text-sm text-observatory-text">{(model.dimensions[p.id]?.disparity ?? 0).toFixed(3)}</div>
            </div>
          ))}
          <div className="bg-observatory-bg/50 rounded-lg p-3 text-center">
            <div className="text-[10px] text-observatory-text-dim">Composite</div>
            <div className="font-mono font-bold text-sm text-observatory-accent">{model.composite_score.toFixed(4)}</div>
          </div>
        </div>

        <ResponsiveContainer width="100%" height={200}>
          <RadarChart data={radarData}>
            <PolarGrid stroke="hsl(222 20% 25%)" />
            <PolarAngleAxis dataKey="probe" tick={{ fill: 'hsl(215 20% 65%)', fontSize: 10 }} />
            <Radar dataKey="value" stroke={model.color} fill={model.color} fillOpacity={0.2} />
          </RadarChart>
        </ResponsiveContainer>

        <div className="mt-6 pt-4 border-t border-observatory-border text-center">
          <p className="text-[10px] font-mono text-observatory-text-dim">
            Fingerprint² · FHIBE Benchmark v1.0 · github.com/Ahm3dAlAli/FingerPrint
          </p>
        </div>
      </div>
    </div>
  );
}
