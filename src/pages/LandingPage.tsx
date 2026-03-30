import { Link } from 'react-router-dom';
import { ArrowRight, BarChart3, FlaskConical, Plane, Camera, ScanLine, Shield, Wrench, Fingerprint, Github, ExternalLink } from 'lucide-react';
import { LEADERBOARD, FRAMEWORK_INFO } from '../data/benchmarkData';
import razije from '../assets/razije.jpg';
import ahmed from '../assets/ahmed.jpg';

const MODULES = [
  { to: '/report', icon: BarChart3, label: 'Bias Report', desc: 'FHIBE benchmark results across 6 VLMs with real data', tag: 'Data' },
  { to: '/lab', icon: FlaskConical, label: 'Fingerprint Lab', desc: 'Compare models side-by-side and search HuggingFace', tag: 'Interactive' },
  { to: '/airport', icon: Plane, label: 'AI Airport', desc: 'Live bias scanner with multi-model comparison & PDF export', tag: 'Live' },
  { to: '/scan', icon: Camera, label: 'Scan Your Face', desc: 'Camera capture → 5 VLMs → instant bias fingerprint', tag: 'Camera' },
  { to: '/eid', icon: ScanLine, label: 'E-ID Portal', desc: 'Swiss electronic identity verification bias analysis', tag: 'Simulation' },
  { to: '/eu-ai-act', icon: Shield, label: 'EU AI Act', desc: 'Compliance framework, risk categories & timeline', tag: 'Legal' },
  { to: '/mitigation', icon: Wrench, label: 'Fix Bias', desc: 'Evidence-based mitigation strategies and fairness metrics', tag: 'Solutions' },
];

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-observatory-bg text-observatory-text">
      {/* Hero */}
      <div className="relative overflow-hidden">
        <div className="absolute inset-0 pointer-events-none opacity-[0.04]" style={{
          backgroundImage: 'radial-gradient(circle at 1px 1px, hsl(var(--obs-accent)) 1px, transparent 0)',
          backgroundSize: '40px 40px',
        }} />
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[600px] h-[600px] bg-observatory-accent/5 rounded-full blur-[120px]" />

        <div className="relative max-w-6xl mx-auto px-6 pt-20 pb-16">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-12 h-12 rounded-2xl bg-observatory-accent/15 flex items-center justify-center">
              <Fingerprint className="w-7 h-7 text-observatory-accent" />
            </div>
            <span className="text-xs font-mono px-3 py-1 rounded-full bg-observatory-warning/10 text-observatory-warning border border-observatory-warning/20">
              Ethical & Responsible GenAI Track · March 2026
            </span>
          </div>

          <h1 className="text-5xl md:text-7xl font-black tracking-tight leading-[0.95] mb-6">
            Finger<span className="gradient-text">print</span><sup className="text-observatory-accent text-2xl align-super">²</sup>
          </h1>
          <p className="text-xl text-observatory-text-muted max-w-2xl mb-8 leading-relaxed">
            Multi-dimensional bias fingerprinting for vision-language models. 
            Benchmark, detect, and mitigate bias with real data from Sony FHIBE.
          </p>

          <div className="flex flex-wrap gap-3 mb-12">
            <Link to="/report" className="inline-flex items-center gap-2 px-6 py-3 rounded-xl bg-observatory-accent text-observatory-bg font-semibold text-sm hover:bg-observatory-accent-glow transition-all">
              View Bias Report <ArrowRight className="w-4 h-4" />
            </Link>
            <Link to="/scan" className="inline-flex items-center gap-2 px-6 py-3 rounded-xl border border-observatory-border text-observatory-text-muted font-medium text-sm hover:border-observatory-accent/50 hover:text-observatory-accent transition-all">
              <Camera className="w-4 h-4" /> Scan Your Face
            </Link>
            <a href={FRAMEWORK_INFO.repo} target="_blank" rel="noopener" className="inline-flex items-center gap-2 px-6 py-3 rounded-xl border border-observatory-border text-observatory-text-muted font-medium text-sm hover:border-observatory-text-dim transition-all">
              <Github className="w-4 h-4" /> GitHub
            </a>
          </div>

          {/* Stats */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {[
              { value: '3,000', label: 'Images' },
              { value: '6', label: 'Regions' },
              { value: '5', label: 'Probes' },
              { value: '6', label: 'VLMs' },
            ].map(s => (
              <div key={s.label} className="glass rounded-2xl px-5 py-4">
                <div className="text-3xl font-black font-mono text-observatory-text">{s.value}</div>
                <div className="text-xs text-observatory-text-dim uppercase tracking-wider mt-1">{s.label}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Leaderboard Preview */}
      <div className="max-w-6xl mx-auto px-6 py-12">
        <h2 className="text-2xl font-bold mb-2">Bias Leaderboard</h2>
        <p className="text-sm text-observatory-text-muted mb-6">Lower composite score = less biased · All models ranked on FHIBE benchmark</p>
        <div className="space-y-2">
          {LEADERBOARD.map((m, i) => (
            <div key={m.id} className="glass rounded-xl px-5 py-4 flex items-center gap-4 hover:border-observatory-accent/20 transition-all">
              <span className="text-lg font-mono font-bold text-observatory-accent w-10">#{i + 1}</span>
              <div className="flex-1 min-w-0">
                <div className="font-semibold">{m.name}</div>
                <div className="text-xs text-observatory-text-dim font-mono">{m.hf_id}</div>
              </div>
              <span className="text-xs text-observatory-text-dim hidden md:block">{m.params} · {m.provider}</span>
              <span className="font-mono font-bold text-lg" style={{ color: m.color }}>{m.composite_score.toFixed(3)}</span>
              <span className="text-xs px-2.5 py-1 rounded-lg bg-observatory-surface-alt text-observatory-text-muted">{m.severity}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Modules Grid */}
      <div className="max-w-6xl mx-auto px-6 py-12">
        <h2 className="text-2xl font-bold mb-2">Explore Modules</h2>
        <p className="text-sm text-observatory-text-muted mb-6">Each module demonstrates a different aspect of VLM bias analysis</p>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {MODULES.map(m => (
            <Link
              key={m.to}
              to={m.to}
              className="glass rounded-2xl p-6 hover:border-observatory-accent/30 transition-all group"
            >
              <div className="flex items-center gap-3 mb-3">
                <div className="w-10 h-10 rounded-xl bg-observatory-accent/10 flex items-center justify-center">
                  <m.icon className="w-5 h-5 text-observatory-accent" />
                </div>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded-md bg-observatory-surface-alt text-observatory-text-dim uppercase">{m.tag}</span>
              </div>
              <h3 className="font-semibold text-lg mb-1 group-hover:text-observatory-accent transition-colors">{m.label}</h3>
              <p className="text-sm text-observatory-text-muted">{m.desc}</p>
              <div className="mt-4 flex items-center gap-1 text-xs text-observatory-accent opacity-0 group-hover:opacity-100 transition-opacity">
                Open <ArrowRight className="w-3 h-3" />
              </div>
            </Link>
          ))}
        </div>
      </div>

      {/* Team */}
      <div className="max-w-6xl mx-auto px-6 py-12">
        <h2 className="text-2xl font-bold mb-6">Team</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {[
            { name: 'Razije', role: 'Professional', from: 'Switzerland', age: '30–38', photo: razije },
            { name: 'Ahmed', role: 'PhD Researcher', from: 'UAE', age: '20–28', photo: ahmed },
          ].map(p => (
            <div key={p.name} className="glass rounded-2xl p-6 flex items-center gap-5">
              <img src={p.photo} alt={p.name} className="w-20 h-20 rounded-xl object-cover border border-observatory-border/40" />
              <div>
                <h3 className="text-lg font-bold">{p.name}</h3>
                <p className="text-sm text-observatory-text-muted">{p.role}</p>
                <p className="text-xs text-observatory-text-dim mt-1">{p.from} · Age: {p.age}</p>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Footer */}
      <div className="border-t border-observatory-border/30 px-6 py-6">
        <div className="max-w-6xl mx-auto flex items-center justify-between text-xs text-observatory-text-dim">
          <span className="font-mono font-bold">Fingerprint² Bench</span>
          <div className="flex items-center gap-4">
            <a href={FRAMEWORK_INFO.repo} target="_blank" rel="noopener" className="hover:text-observatory-text-muted transition-colors flex items-center gap-1">
              <ExternalLink className="w-3 h-3" /> GitHub
            </a>
            <a href={FRAMEWORK_INFO.datasetUrl} target="_blank" rel="noopener" className="hover:text-observatory-text-muted transition-colors flex items-center gap-1">
              <ExternalLink className="w-3 h-3" /> FHIBE Dataset
            </a>
            <span>March 2026 · Ethical AI Track</span>
          </div>
        </div>
      </div>
    </div>
  );
}
