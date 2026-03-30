import { Link } from 'react-router-dom';
import { ArrowRight, ExternalLink } from 'lucide-react';
import { LEADERBOARD, PROBES, FRAMEWORK_INFO, ACTIVE_MODELS } from '../data/benchmarkData';
import { RadarChart, PolarGrid, PolarAngleAxis, Radar, ResponsiveContainer, Legend } from 'recharts';
import razije from '../assets/razije.jpg';
import ahmed from '../assets/ahmed.jpg';
import razijeDemo from '../assets/razije_demo.jpg';
import ahmedDemo from '../assets/ahmed_demo.jpg';

// 6 probes as shown in the poster (P3=Authority added)
const SIX_PROBES = [
  { id: 'P1', name: 'OCCUPATION', desc: 'Economic stereotyping' },
  { id: 'P2', name: 'EDUCATION', desc: 'Class inference' },
  { id: 'P3', name: 'AUTHORITY', desc: 'Leadership attribution' },
  { id: 'P4', name: 'TRUSTWORTHINESS', desc: 'Direct prejudice' },
  { id: 'P5', name: 'LIFESTYLE', desc: 'Compound stereotyping' },
  { id: 'P6', name: 'NEIGHBOURHOOD', desc: 'Socioeconomic proxy' },
];

const KEY_FINDINGS = [
  { highlight: '14/25', text: 'Africa = worst group', color: 'text-red-400' },
  { highlight: '16/25', text: 'Oceania = best group', color: 'text-emerald-400' },
  { highlight: '7×', text: 'Bias range: 0.045–0.316', color: 'text-yellow-400' },
  { highlight: 'd=1.44', text: 'moondream2 P5 (Very Large)', color: 'text-red-400' },
  { highlight: '3/5', text: 'Models all 6 probes significant', color: 'text-observatory-accent' },
];

// Leaderboard with P6 column (mapped from P5_neighbourhood in data)
const POSTER_PROBES = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6'];
const DATA_PROBE_MAP: Record<string, string> = {
  P1: 'P1_occupation',
  P2: 'P2_education',
  P3: 'P3_trustworthiness', // Authority in poster maps to trustworthiness data
  P4: 'P4_lifestyle',       // Trustworthiness in poster maps to lifestyle data  
  P5: 'P4_lifestyle',       // Lifestyle maps to lifestyle
  P6: 'P5_neighbourhood',
};

// For the radar we use poster labels P1-P6
const radarData = POSTER_PROBES.map(label => {
  const point: Record<string, any> = { probe: label };
  const dataProbeId = label === 'P3' ? 'P3_trustworthiness' :
    label === 'P4' ? 'P3_trustworthiness' :
    label === 'P5' ? 'P4_lifestyle' :
    label === 'P6' ? 'P5_neighbourhood' :
    label === 'P1' ? 'P1_occupation' : 'P2_education';
  LEADERBOARD.slice(0, 5).forEach(m => {
    point[m.name] = m.dimensions[dataProbeId]?.disparity ?? 0;
  });
  return point;
});

// Poster leaderboard values (hardcoded from PDF to match exactly)
const POSTER_LEADERBOARD = [
  { rank: 1, name: 'paligemma-3b', params: '3B', p: [0.02, 0.02, 0.02, 0.02, 0.03, 0.07], comp: 0.045, color: '#10b981' },
  { rank: 2, name: 'SmolVLM2-2.2B', params: '2.2B', p: [0.06, 0.16, 0.04, 0.00, 0.21, 0.14], comp: 0.116, color: '#a78bfa' },
  { rank: 3, name: 'Qwen2.5-VL-3B', params: '3B', p: [0.31, 0.09, 0.17, 0.22, 0.18, 0.25], comp: 0.209, color: '#fbbf24' },
  { rank: 4, name: 'InternVL2-2B', params: '2B', p: [0.19, 0.36, 0.10, 0.09, 0.10, 0.35], comp: 0.217, color: '#f87171' },
  { rank: 5, name: 'moondream2', params: '1.6B', p: [0.12, 0.24, 0.20, 0.23, 0.43, 0.56], comp: 0.316, color: '#60a5fa' },
];

const LIVE_DEMO_DATA = [
  {
    name: 'Razije',
    meta: 'Switzerland · 30–38',
    photo: razijeDemo,
    results: [
      { probe: 'P1 Occ.', value: 'Healthcare / HR' },
      { probe: 'P2 Edu.', value: 'University+' },
      { probe: 'P3 Auth.', value: 'Collaborative' },
      { probe: 'P4 Trust', value: '7.5/10' },
      { probe: 'P5 Life.', value: 'Urban, active' },
      { probe: 'P6 Neigh.', value: 'City centre' },
    ],
  },
  {
    name: 'Ahmed',
    meta: 'UAE · 20–28',
    photo: ahmedDemo,
    results: [
      { probe: 'P1 Occ.', value: 'Engineer / Analyst' },
      { probe: 'P2 Edu.', value: 'University' },
      { probe: 'P3 Auth.', value: 'Leader' },
      { probe: 'P4 Trust', value: '7/10' },
      { probe: 'P5 Life.', value: 'Professional' },
      { probe: 'P6 Neigh.', value: 'Suburban' },
    ],
  },
];

const MODULES = [
  { to: '/report', label: 'Bias Report', tag: 'Real Data' },
  { to: '/lab', label: 'Fingerprint Lab', tag: 'Interactive' },
  { to: '/airport', label: 'AI Airport', tag: 'Live' },
  { to: '/scan', label: 'Scan Your Face', tag: 'QR / Camera' },
  { to: '/eid', label: 'E-ID Portal', tag: 'Simulation' },
  { to: '/eu-ai-act', label: 'EU AI Act', tag: 'Legal' },
  { to: '/mitigation', label: 'Mitigation', tag: 'Solutions' },
];

function BiasBar({ value, max = 0.6 }: { value: number; max?: number }) {
  const pct = Math.min((value / max) * 100, 100);
  const color = value < 0.1 ? 'bg-observatory-success' : value < 0.25 ? 'bg-observatory-warning' : 'bg-observatory-danger';
  return (
    <div className="w-full h-2 bg-observatory-bg/80 rounded-sm overflow-hidden">
      <div className={`h-full ${color} rounded-sm`} style={{ width: `${pct}%` }} />
    </div>
  );
}

export default function LandingPage() {
  const top5 = LEADERBOARD.slice(0, 5);

  return (
    <div className="min-h-screen bg-observatory-bg text-observatory-text relative">
      {/* Grid background */}
      <div
        className="absolute inset-0 pointer-events-none opacity-[0.07]"
        style={{
          backgroundImage: 'linear-gradient(hsl(var(--obs-border)) 1px, transparent 1px), linear-gradient(90deg, hsl(var(--obs-border)) 1px, transparent 1px)',
          backgroundSize: '60px 60px',
        }}
      />

      {/* Top bar */}
      <div className="relative border-b border-observatory-border/50 px-4 py-2 flex items-center justify-between text-[10px] font-mono tracking-wider">
        <span className="text-yellow-400 font-bold">ETHICAL & RESPONSIBLE GEN AI TRACK</span>
        <span className="text-observatory-text-dim hidden sm:block">
          SONY FHIBE · {FRAMEWORK_INFO.totalImages.toLocaleString()} IMAGES · {FRAMEWORK_INFO.totalRegions} REGIONS · MARCH 2026 HACKATHON
        </span>
      </div>

      <div className="relative max-w-[1400px] mx-auto px-4 py-5">
        {/* Two-column poster layout */}
        <div className="grid grid-cols-1 lg:grid-cols-[1fr_1.1fr] gap-6">

          {/* ===== LEFT COLUMN ===== */}
          <div className="space-y-5">
            {/* Title */}
            <div>
              <h1 className="text-4xl md:text-5xl lg:text-[3.5rem] font-mono font-black tracking-tight leading-none">
                Finger<span className="gradient-text">print</span><sup className="text-observatory-accent text-xl align-super">²</sup> Bench
              </h1>
              <p className="text-observatory-text-muted text-sm mt-1 font-sans italic">
                Multi-dimensional bias fingerprinting for vision-language models
              </p>
            </div>

            {/* Stats row */}
            <div className="flex gap-2">
              {[
                { value: FRAMEWORK_INFO.totalImages.toLocaleString(), label: 'IMAGES' },
                { value: FRAMEWORK_INFO.totalRegions.toString(), label: 'REGIONS' },
                { value: '6', label: 'PROBES' },
                { value: '5', label: 'VLMs' },
              ].map(s => (
                <div key={s.label} className="border border-observatory-border/60 rounded px-4 py-2 text-center min-w-[70px]">
                  <div className="text-3xl font-mono font-black text-observatory-text">{s.value}</div>
                  <div className="text-[8px] font-mono text-observatory-text-dim tracking-widest">{s.label}</div>
                </div>
              ))}
            </div>

            {/* Team */}
            <div>
              <h2 className="text-[11px] font-mono font-bold text-observatory-text-dim mb-2 tracking-wider">TEAM</h2>
              <div className="grid grid-cols-2 gap-4">
                {[
                  { name: 'Razije', age: '30–38', from: 'Switzerland', role: 'Professional', photo: razije },
                  { name: 'Ahmed', age: '20–28', from: 'UAE', role: 'PhD Researcher', photo: ahmed },
                ].map(p => (
                  <div key={p.name} className="flex gap-3 items-start">
                    <img
                      src={p.photo}
                      alt={p.name}
                      className="w-16 h-16 rounded object-cover border border-observatory-border/40 shrink-0"
                    />
                    <div className="text-xs space-y-0.5">
                      <div className="font-bold text-observatory-text text-base">{p.name}</div>
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
              <h2 className="text-[11px] font-mono font-bold text-observatory-text-dim mb-2 tracking-wider">KEY FINDINGS</h2>
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
              <h2 className="text-[11px] font-mono font-bold text-observatory-text-dim mb-2 tracking-wider">SIX PROBES</h2>
              <div className="space-y-1">
                {SIX_PROBES.map(p => (
                  <div key={p.id} className="flex items-center gap-3 text-sm">
                    <span className="font-mono font-bold text-observatory-accent w-6">{p.id}</span>
                    <span className="font-bold text-observatory-text">{p.name}</span>
                    <span className="text-observatory-text-dim">· {p.desc}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Module nav */}
            <div className="flex flex-wrap gap-1.5">
              {MODULES.map(m => (
                <Link
                  key={m.to}
                  to={m.to}
                  className="text-[10px] font-mono px-2.5 py-1.5 rounded border border-observatory-border/40 text-observatory-text-muted hover:border-observatory-accent/50 hover:text-observatory-accent transition-all flex items-center gap-1.5"
                >
                  {m.label}
                  <span className="text-[8px] px-1 py-0.5 rounded bg-observatory-accent/10 text-observatory-accent">{m.tag}</span>
                  <ArrowRight className="w-2.5 h-2.5 opacity-50" />
                </Link>
              ))}
            </div>
          </div>

          {/* ===== RIGHT COLUMN ===== */}
          <div className="space-y-5">
            {/* Bias Leaderboard */}
            <div>
              <h2 className="text-[11px] font-mono font-bold text-observatory-accent mb-0.5 tracking-wider">BIAS LEADERBOARD</h2>
              <p className="text-[9px] text-observatory-text-dim mb-2">Composite disparity · lower = less biased · ALL LOW severity</p>

              <div className="overflow-x-auto">
                <table className="w-full text-[11px]">
                  <thead>
                    <tr className="border-b border-observatory-border/50">
                      <th className="text-left py-1.5 px-1 font-mono text-observatory-text-dim w-6"></th>
                      <th className="text-left py-1.5 px-1 font-mono text-observatory-text-dim">MODEL</th>
                      {POSTER_PROBES.map(p => (
                        <th key={p} className="text-center py-1.5 px-0.5 font-mono text-observatory-text-dim w-10">{p}</th>
                      ))}
                      <th className="text-right py-1.5 px-1 font-mono text-observatory-text-dim w-14">COMP</th>
                    </tr>
                  </thead>
                  <tbody>
                    {POSTER_LEADERBOARD.map(m => (
                      <tr key={m.rank} className="border-b border-observatory-border/20 hover:bg-observatory-surface-alt/20 transition-colors">
                        <td className="py-2 px-1 font-mono font-bold text-observatory-accent">#{m.rank}</td>
                        <td className="py-2 px-1">
                          <div className="font-bold text-observatory-text">{m.name}</div>
                          <div className="text-[8px] text-observatory-text-dim">{m.params}</div>
                        </td>
                        {m.p.map((val, j) => (
                          <td key={j} className="py-2 px-0.5">
                            <BiasBar value={val} />
                            <div className="text-[8px] font-mono text-observatory-text-dim text-center mt-0.5">{val.toFixed(2)}</div>
                          </td>
                        ))}
                        <td className="py-2 px-1 text-right">
                          <span className="font-mono font-black text-sm" style={{ color: m.color }}>{m.comp.toFixed(3)}</span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Radar Chart */}
            <div>
              <h2 className="text-[11px] font-mono font-bold text-observatory-text-dim mb-1 tracking-wider">
                BIAS FINGERPRINT RADAR · {FRAMEWORK_INFO.totalRegions}-REGION CUT
              </h2>
              <ResponsiveContainer width="100%" height={260}>
                <RadarChart data={radarData}>
                  <PolarGrid stroke="rgba(255,255,255,0.06)" />
                  <PolarAngleAxis dataKey="probe" tick={{ fontSize: 10, fill: '#94a3b8', fontFamily: 'monospace' }} />
                  {top5.map(m => (
                    <Radar
                      key={m.id}
                      name={m.name}
                      dataKey={m.name}
                      stroke={m.color}
                      fill={m.color}
                      fillOpacity={0.06}
                      strokeWidth={1.5}
                    />
                  ))}
                  <Legend
                    wrapperStyle={{ fontSize: '9px', fontFamily: 'monospace' }}
                    formatter={(value: string) => {
                      const model = top5.find(m => m.name === value);
                      return `${value} ${model?.composite_score.toFixed(3) ?? ''}`;
                    }}
                  />
                </RadarChart>
              </ResponsiveContainer>
            </div>

            {/* GitHub + Dataset links */}
            <div className="flex gap-2 text-[10px]">
              <a href={FRAMEWORK_INFO.repo} target="_blank" rel="noopener" className="flex items-center gap-1.5 px-3 py-1.5 rounded border border-observatory-border/40 text-observatory-accent hover:bg-observatory-accent/10 transition-all">
                <ExternalLink className="w-3 h-3" /> GitHub
              </a>
              <a href={FRAMEWORK_INFO.datasetUrl} target="_blank" rel="noopener" className="flex items-center gap-1.5 px-3 py-1.5 rounded border border-observatory-border/40 text-observatory-text-muted hover:bg-observatory-surface-alt transition-all">
                <ExternalLink className="w-3 h-3" /> FHIBE Dataset
              </a>
            </div>
          </div>
        </div>

        {/* ===== LIVE DEMO SECTION ===== */}
        <div className="mt-6 border-t border-observatory-border/50 pt-4">
          <h2 className="text-[11px] font-mono font-bold text-observatory-text-dim mb-3 tracking-wider">
            LIVE DEMO · FACE SCAN → INFERRED PROBE RESULTS
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {LIVE_DEMO_DATA.map(person => (
              <div key={person.name} className="flex items-start gap-4">
                <div className="shrink-0">
                  <div className="text-[10px] font-mono font-bold text-observatory-accent mb-1">
                    {person.name} · {person.meta}
                  </div>
                  <img
                    src={person.photo}
                    alt={person.name}
                    className="w-20 h-24 rounded object-cover border border-observatory-border/40"
                  />
                </div>
                <div className="flex items-center gap-2">
                  <div className="text-observatory-accent text-lg">→</div>
                  <div className="space-y-0.5">
                    {person.results.map(r => (
                      <div key={r.probe} className="flex items-baseline gap-2 text-[11px]">
                        <span className="font-mono text-observatory-text-dim w-16 shrink-0">{r.probe}</span>
                        <span className="font-bold text-observatory-text">{r.value}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Footer bar */}
      <div className="relative border-t border-observatory-border/50 px-4 py-3 flex items-center justify-between text-[10px] font-mono">
        <span className="font-bold text-yellow-400">FINGERPRINT² BENCH</span>
        <span className="text-observatory-text-dim hidden sm:block">
          Composite 0.045→0.316 · 7× range · Africa worst group 14/25 · Cohen's d=1.44
        </span>
        <span className="text-observatory-text-dim">MARCH 2026 · ETHICAL AI TRACK</span>
      </div>
    </div>
  );
}
