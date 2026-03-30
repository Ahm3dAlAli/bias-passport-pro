import { Link } from 'react-router-dom';
import { Fingerprint, BarChart3, FlaskConical, Plane, ScanLine, Shield, Wrench, ArrowRight, ExternalLink } from 'lucide-react';
import { LEADERBOARD, PROBES, FRAMEWORK_INFO, ACTIVE_MODELS, getSeverityGrade } from '../data/benchmarkData';
import { RadarChart, PolarGrid, PolarAngleAxis, Radar, ResponsiveContainer, Legend } from 'recharts';

const MODULES = [
  { to: '/report', icon: BarChart3, label: 'Bias Report', desc: 'Full bias fingerprint analysis', tag: 'Real Data' },
  { to: '/lab', icon: FlaskConical, label: 'Fingerprint Lab', desc: 'Explore probes, get Bias Passport', tag: 'Interactive' },
  { to: '/airport', icon: Plane, label: 'AI Airport', desc: 'Real-time screening simulation', tag: 'Live' },
  { to: '/eid', icon: ScanLine, label: 'E-ID Portal', desc: 'Swiss electronic ID bias testing', tag: 'Simulation' },
  { to: '/eu-ai-act', icon: Shield, label: 'EU AI Act', desc: 'High-risk AI compliance', tag: 'Legal' },
  { to: '/mitigation', icon: Wrench, label: 'Mitigation', desc: 'Strategies to reduce bias', tag: 'Solutions' },
];

const KEY_FINDINGS = [
  { highlight: '14/25', text: 'Africa = worst group', color: 'text-red-400' },
  { highlight: '16/25', text: 'Oceania = best group', color: 'text-emerald-400' },
  { highlight: '7×', text: 'Bias range: 0.045–0.316', color: 'text-yellow-400' },
  { highlight: 'd=1.44', text: 'moondream2 P5 (Very Large)', color: 'text-red-400' },
  { highlight: '3/5', text: 'Models all 5 probes significant', color: 'text-observatory-accent' },
];

// Build radar data from top 5 models
const radarProbeLabels = ['P1', 'P2', 'P3', 'P4', 'P5'];
const radarData = radarProbeLabels.map((label, i) => {
  const probeId = PROBES[i].id;
  const point: Record<string, any> = { probe: label };
  LEADERBOARD.slice(0, 5).forEach(m => {
    point[m.name] = m.dimensions[probeId]?.disparity ?? 0;
  });
  return point;
});

function BiasBar({ value, max = 0.6 }: { value: number; max?: number }) {
  const pct = Math.min((value / max) * 100, 100);
  const color = value < 0.1 ? 'bg-emerald-500' : value < 0.25 ? 'bg-yellow-500' : 'bg-red-500';
  return (
    <div className="w-full h-3 bg-observatory-bg/80 rounded-sm overflow-hidden">
      <div className={`h-full ${color} rounded-sm`} style={{ width: `${pct}%` }} />
    </div>
  );
}

export default function LandingPage() {
  const top5 = LEADERBOARD.slice(0, 5);

  return (
    <div className="min-h-screen bg-observatory-bg text-observatory-text">
      {/* Top bar */}
      <div className="border-b border-observatory-border/50 px-4 py-2 flex items-center justify-between text-[10px] font-mono tracking-wider">
        <span className="text-observatory-accent font-bold">ETHICAL & RESPONSIBLE GEN AI TRACK</span>
        <span className="text-observatory-text-dim hidden sm:block">
          SONY FHIBE · {FRAMEWORK_INFO.totalImages.toLocaleString()} IMAGES · {FRAMEWORK_INFO.totalRegions} REGIONS · MARCH 2026 HACKATHON
        </span>
      </div>

      <div className="max-w-[1400px] mx-auto px-4 py-6">
        {/* Two-column poster layout */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

          {/* ===== LEFT COLUMN ===== */}
          <div className="space-y-5">
            {/* Title */}
            <div>
              <h1 className="text-4xl md:text-5xl font-mono font-black tracking-tight">
                Fingerprint<sup className="text-observatory-accent text-2xl">²</sup> Bench
              </h1>
              <p className="text-observatory-text-muted text-sm mt-1">
                Multi-dimensional bias fingerprinting for vision-language models
              </p>
            </div>

            {/* Stats row */}
            <div className="flex gap-3">
              {[
                { value: FRAMEWORK_INFO.totalImages.toLocaleString(), label: 'IMAGES' },
                { value: FRAMEWORK_INFO.totalRegions.toString(), label: 'REGIONS' },
                { value: FRAMEWORK_INFO.totalProbes.toString(), label: 'PROBES' },
                { value: ACTIVE_MODELS.length.toString(), label: 'VLMs' },
              ].map(s => (
                <div key={s.label} className="border border-observatory-border/60 rounded px-4 py-2 text-center min-w-[70px]">
                  <div className="text-2xl font-mono font-black text-observatory-text">{s.value}</div>
                  <div className="text-[9px] font-mono text-observatory-text-dim tracking-wider">{s.label}</div>
                </div>
              ))}
            </div>

            {/* Team */}
            <div>
              <h2 className="text-xs font-mono font-bold text-observatory-text-dim mb-2 tracking-wider">TEAM</h2>
              <div className="grid grid-cols-2 gap-4">
                {[
                  { name: 'Razije', age: '30–38', from: 'Switzerland', role: 'Professional' },
                  { name: 'Ahmed', age: '20–28', from: 'UAE', role: 'PhD Researcher' },
                ].map(p => (
                  <div key={p.name} className="flex gap-3 items-start">
                    <div className="w-14 h-14 rounded bg-observatory-surface-alt flex items-center justify-center text-xl font-bold text-observatory-accent shrink-0">
                      {p.name[0]}
                    </div>
                    <div className="text-xs space-y-0.5">
                      <div className="font-bold text-observatory-text text-sm">{p.name}</div>
                      <div className="text-observatory-text-dim">Age: <span className="text-observatory-text-muted">{p.age}</span></div>
                      <div className="text-observatory-text-dim">From: <span className="text-observatory-text-muted">{p.from}</span></div>
                      <div className="text-observatory-text-dim">Role: <span className="text-observatory-text-muted">{p.role}</span></div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Key Findings */}
            <div>
              <h2 className="text-xs font-mono font-bold text-observatory-text-dim mb-2 tracking-wider">KEY FINDINGS</h2>
              <div className="space-y-1">
                {KEY_FINDINGS.map((f, i) => (
                  <div key={i} className="flex items-center gap-2 text-sm">
                    <span className={`font-mono font-bold ${f.color} min-w-[50px]`}>{f.highlight}</span>
                    <span className="text-observatory-text-muted">{f.text}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Six Probes */}
            <div>
              <h2 className="text-xs font-mono font-bold text-observatory-text-dim mb-2 tracking-wider">PROBES</h2>
              <div className="space-y-1">
                {[
                  { id: 'P1', name: 'OCCUPATION', desc: 'Economic stereotyping' },
                  { id: 'P2', name: 'EDUCATION', desc: 'Class inference' },
                  { id: 'P3', name: 'TRUSTWORTHINESS', desc: 'Direct prejudice' },
                  { id: 'P4', name: 'LIFESTYLE', desc: 'Compound stereotyping' },
                  { id: 'P5', name: 'NEIGHBOURHOOD', desc: 'Socioeconomic proxy' },
                ].map(p => (
                  <div key={p.id} className="flex items-center gap-3 text-sm">
                    <span className="font-mono font-bold text-observatory-accent w-6">{p.id}</span>
                    <span className="font-bold text-observatory-text">{p.name}</span>
                    <span className="text-observatory-text-dim">· {p.desc}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Modules */}
            <div>
              <h2 className="text-xs font-mono font-bold text-observatory-text-dim mb-2 tracking-wider">
                LIVE DEMO · CHOOSE A MODULE
              </h2>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
                {MODULES.map(m => (
                  <Link
                    key={m.to}
                    to={m.to}
                    className="border border-observatory-border/40 rounded-lg p-3 hover:border-observatory-accent/40 hover:bg-observatory-surface-alt/30 transition-all group"
                  >
                    <div className="flex items-center gap-2 mb-1">
                      <m.icon className="w-3.5 h-3.5 text-observatory-accent" />
                      <span className="font-bold text-xs text-observatory-text">{m.label}</span>
                    </div>
                    <p className="text-[10px] text-observatory-text-dim leading-tight">{m.desc}</p>
                    <div className="flex items-center justify-between mt-1.5">
                      <span className="text-[8px] font-mono px-1.5 py-0.5 rounded bg-observatory-accent/10 text-observatory-accent">{m.tag}</span>
                      <ArrowRight className="w-3 h-3 text-observatory-accent opacity-0 group-hover:opacity-100 transition-opacity" />
                    </div>
                  </Link>
                ))}
              </div>
            </div>
          </div>

          {/* ===== RIGHT COLUMN ===== */}
          <div className="space-y-5">
            {/* Bias Leaderboard */}
            <div>
              <h2 className="text-xs font-mono font-bold text-observatory-accent mb-1 tracking-wider">BIAS LEADERBOARD</h2>
              <p className="text-[10px] text-observatory-text-dim mb-3">Composite disparity · lower = less biased · ALL LOW severity</p>

              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="border-b border-observatory-border/50">
                      <th className="text-left py-1.5 px-1 font-mono text-observatory-text-dim w-8"></th>
                      <th className="text-left py-1.5 px-1 font-mono text-observatory-text-dim">MODEL</th>
                      {radarProbeLabels.map(p => (
                        <th key={p} className="text-center py-1.5 px-1 font-mono text-observatory-text-dim w-14">{p}</th>
                      ))}
                      <th className="text-right py-1.5 px-1 font-mono text-observatory-text-dim w-16">COMP</th>
                    </tr>
                  </thead>
                  <tbody>
                    {top5.map((m, i) => (
                      <tr key={m.id} className="border-b border-observatory-border/30 hover:bg-observatory-surface-alt/20 transition-colors">
                        <td className="py-2 px-1 font-mono font-bold text-observatory-accent">#{i + 1}</td>
                        <td className="py-2 px-1">
                          <div className="font-bold text-observatory-text">{m.name}</div>
                          <div className="text-[9px] text-observatory-text-dim">{m.params}</div>
                        </td>
                        {PROBES.map(probe => {
                          const val = m.dimensions[probe.id]?.disparity ?? 0;
                          return (
                            <td key={probe.id} className="py-2 px-1">
                              <BiasBar value={val} />
                              <div className="text-[9px] font-mono text-observatory-text-dim text-center mt-0.5">{val.toFixed(2)}</div>
                            </td>
                          );
                        })}
                        <td className="py-2 px-1 text-right">
                          <span className="font-mono font-black text-sm" style={{ color: m.color }}>{m.composite_score.toFixed(3)}</span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Radar Chart */}
            <div>
              <h2 className="text-xs font-mono font-bold text-observatory-text-dim mb-1 tracking-wider">
                BIAS FINGERPRINT RADAR · {FRAMEWORK_INFO.totalRegions}-REGION CUT
              </h2>
              <ResponsiveContainer width="100%" height={280}>
                <RadarChart data={radarData}>
                  <PolarGrid stroke="rgba(255,255,255,0.08)" />
                  <PolarAngleAxis dataKey="probe" tick={{ fontSize: 11, fill: '#94a3b8', fontFamily: 'monospace' }} />
                  {top5.map(m => (
                    <Radar
                      key={m.id}
                      name={m.name}
                      dataKey={m.name}
                      stroke={m.color}
                      fill={m.color}
                      fillOpacity={0.08}
                      strokeWidth={1.5}
                    />
                  ))}
                  <Legend
                    wrapperStyle={{ fontSize: '10px', fontFamily: 'monospace' }}
                    formatter={(value: string) => {
                      const model = top5.find(m => m.name === value);
                      return `${value} ${model?.composite_score.toFixed(3) ?? ''}`;
                    }}
                  />
                </RadarChart>
              </ResponsiveContainer>
            </div>

            {/* GitHub + Dataset links */}
            <div className="flex gap-3 text-xs">
              <a
                href={FRAMEWORK_INFO.repo}
                target="_blank"
                rel="noopener"
                className="flex items-center gap-1.5 px-3 py-2 rounded border border-observatory-border/40 text-observatory-accent hover:bg-observatory-accent/10 transition-all"
              >
                <ExternalLink className="w-3 h-3" /> GitHub Repository
              </a>
              <a
                href={FRAMEWORK_INFO.datasetUrl}
                target="_blank"
                rel="noopener"
                className="flex items-center gap-1.5 px-3 py-2 rounded border border-observatory-border/40 text-observatory-text-muted hover:bg-observatory-surface-alt transition-all"
              >
                <ExternalLink className="w-3 h-3" /> FHIBE Dataset
              </a>
            </div>
          </div>
        </div>
      </div>

      {/* Footer bar */}
      <div className="border-t border-observatory-border/50 mt-6 px-4 py-3 flex items-center justify-between text-[10px] font-mono">
        <span className="font-bold text-observatory-text">FINGERPRINT² BENCH</span>
        <span className="text-observatory-text-dim hidden sm:block">
          Composite 0.045→0.316 · 7× range · Africa worst group 14/25 · Cohen's d=1.44
        </span>
        <span className="text-observatory-text-dim">MARCH 2026 · ETHICAL AI TRACK</span>
      </div>
    </div>
  );
}
