import { Link } from 'react-router-dom';
import { Fingerprint, BarChart3, FlaskConical, Plane, ScanLine, CreditCard, Shield, Wrench, ArrowRight, ExternalLink } from 'lucide-react';
import { LEADERBOARD, FRAMEWORK_INFO } from '../data/benchmarkData';

const MODULES = [
  { to: '/report', icon: BarChart3, label: 'Bias Report', desc: 'Full bias fingerprint analysis of VLMs', tag: 'Real Data' },
  { to: '/lab', icon: FlaskConical, label: 'Fingerprint Lab', desc: 'Choose a VLM, explore probes, get Bias Passport', tag: 'Interactive' },
  { to: '/airport', icon: Plane, label: 'AI Airport', desc: 'Real-time airport screening simulation with bias detection', tag: 'Live' },
  { to: '/eid', icon: ScanLine, label: 'E-ID Portal', desc: 'Electronic ID verification and AI bias impact', tag: 'Simulation' },
  { to: '/banking', icon: CreditCard, label: 'Credit Scoring', desc: 'AI bias in banking credit approvals', tag: 'New' },
  { to: '/eu-ai-act', icon: Shield, label: 'EU AI Act', desc: 'Compliance framework for high-risk AI systems', tag: 'Legal' },
  { to: '/mitigation', icon: Wrench, label: 'Mitigation', desc: 'Evidence-based strategies to reduce AI bias', tag: 'Solutions' },
];

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-observatory-bg">
      {/* Hero */}
      <header className="relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-br from-observatory-accent/5 via-transparent to-purple-500/5" />
        <div className="max-w-6xl mx-auto px-6 py-16 md:py-24 relative z-10">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-12 h-12 rounded-xl bg-observatory-accent/20 flex items-center justify-center glow-accent">
              <Fingerprint className="w-7 h-7 text-observatory-accent" />
            </div>
            <div>
              <h1 className="text-3xl md:text-4xl font-mono font-bold gradient-text">Fingerprint²</h1>
              <p className="text-observatory-text-muted text-sm">AI Bias Observatory</p>
            </div>
          </div>
          <p className="text-xl md:text-2xl text-observatory-text max-w-3xl leading-relaxed mb-4">
            Measuring systematic bias in Vision-Language Models for <span className="text-observatory-accent font-semibold">fair deployment</span> in airports, identity verification, and credit scoring.
          </p>
          <p className="text-observatory-text-muted mb-8 max-w-2xl">
            EU AI Act compliant evaluation framework. Real FHIBE benchmark results from{' '}
            <a href={FRAMEWORK_INFO.repo} target="_blank" rel="noopener" className="text-observatory-accent hover:underline inline-flex items-center gap-1">
              github.com/Ahm3dAlAli/FingerPrint <ExternalLink className="w-3 h-3" />
            </a>
          </p>

          {/* Stats */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-12">
            {[
              { value: '8', label: 'Models Evaluated' },
              { value: '3K', label: 'Images Tested' },
              { value: '5', label: 'Bias Probes' },
              { value: '6', label: 'Geographic Regions' },
            ].map(s => (
              <div key={s.label} className="glass rounded-xl p-4 text-center">
                <div className="text-2xl font-mono font-bold text-observatory-accent">{s.value}</div>
                <div className="text-xs text-observatory-text-muted mt-1">{s.label}</div>
              </div>
            ))}
          </div>

          {/* Modules */}
          <h2 className="text-lg font-mono font-semibold text-observatory-text mb-4">CHOOSE A MODULE</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-12">
            {MODULES.map(m => (
              <Link
                key={m.to}
                to={m.to}
                className="glass rounded-xl p-5 hover:bg-observatory-surface-alt/80 transition-all group"
              >
                <div className="flex items-start justify-between mb-3">
                  <m.icon className="w-5 h-5 text-observatory-accent" />
                  <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-observatory-accent/10 text-observatory-accent">{m.tag}</span>
                </div>
                <h3 className="font-semibold text-observatory-text mb-1">{m.label}</h3>
                <p className="text-sm text-observatory-text-muted">{m.desc}</p>
                <ArrowRight className="w-4 h-4 text-observatory-accent mt-3 opacity-0 group-hover:opacity-100 transition-opacity" />
              </Link>
            ))}
          </div>

          {/* Mini leaderboard */}
          <div className="glass rounded-xl p-6">
            <h3 className="font-mono font-semibold text-observatory-text mb-4">Bias Leaderboard — Top 5</h3>
            <div className="space-y-2">
              {LEADERBOARD.slice(0, 5).map((m, i) => (
                <div key={m.id} className="flex items-center gap-3 py-2 px-3 rounded-lg bg-observatory-bg/50">
                  <span className="text-observatory-accent font-mono font-bold w-6">#{i + 1}</span>
                  <span className="text-observatory-text font-medium flex-1 text-sm">{m.name}</span>
                  <span className="text-observatory-text-dim text-xs font-mono">{m.params}</span>
                  <span className="text-observatory-text-muted text-xs">{m.provider}</span>
                  <span className="font-mono font-bold text-sm" style={{ color: m.color }}>{m.composite_score.toFixed(3)}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </header>

      {/* Footer */}
      <footer className="border-t border-observatory-border py-6 text-center">
        <p className="text-xs text-observatory-text-dim font-mono">
          Fingerprint² — AI Bias Observatory · Built for EU AI Act compliance research
        </p>
        <p className="text-xs text-observatory-text-dim mt-1">
          FHIBE Dataset · {FRAMEWORK_INFO.totalImages.toLocaleString()} diverse human-centric images · Sony AI
        </p>
      </footer>
    </div>
  );
}
