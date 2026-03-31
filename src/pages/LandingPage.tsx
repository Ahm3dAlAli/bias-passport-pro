import { Link } from 'react-router-dom';
import { ArrowRight, BarChart3, FlaskConical, Plane, Camera, Shield, Wrench, Fingerprint, Github, ExternalLink, Zap } from 'lucide-react';
import { motion, useInView } from 'framer-motion';
import { useRef, useEffect, useState } from 'react';
import { LEADERBOARD, FRAMEWORK_INFO, getSeverityGrade } from '../data/benchmarkData';
import razije from '../assets/razije.jpg';
import ahmed from '../assets/ahmed.jpg';

const MODULES = [
  { to: '/report', icon: BarChart3, label: 'Bias Report', desc: 'FHIBE benchmark results across 6 VLMs with real data', tag: 'Data' },
  { to: '/lab', icon: FlaskConical, label: 'Fingerprint Lab', desc: 'Compare models side-by-side and search HuggingFace', tag: 'Interactive' },
  { to: '/airport', icon: Plane, label: 'AI Airport', desc: 'Live bias scanner with multi-model comparison & PDF export', tag: 'Live' },
  { to: '/scan', icon: Camera, label: 'Scan Your Face', desc: 'Camera capture → 5 VLMs → instant bias fingerprint', tag: 'Camera' },
  
  { to: '/eu-ai-act', icon: Shield, label: 'EU AI Act', desc: 'Compliance framework, risk categories & timeline', tag: 'Legal' },
  { to: '/mitigation', icon: Wrench, label: 'Fix Bias', desc: 'Evidence-based mitigation strategies and fairness metrics', tag: 'Solutions' },
];

function AnimatedCounter({ target, suffix = '' }: { target: string; suffix?: string }) {
  const [count, setCount] = useState(0);
  const ref = useRef<HTMLDivElement>(null);
  const isInView = useInView(ref, { once: true });
  const numericTarget = parseInt(target.replace(/,/g, ''));

  useEffect(() => {
    if (!isInView) return;
    const duration = 1500;
    const steps = 40;
    const increment = numericTarget / steps;
    let current = 0;
    const timer = setInterval(() => {
      current += increment;
      if (current >= numericTarget) {
        setCount(numericTarget);
        clearInterval(timer);
      } else {
        setCount(Math.floor(current));
      }
    }, duration / steps);
    return () => clearInterval(timer);
  }, [isInView, numericTarget]);

  return (
    <div ref={ref} className="text-2xl sm:text-3xl font-black font-mono text-observatory-text">
      {count.toLocaleString()}{suffix}
    </div>
  );
}

function SeverityBadge({ severity }: { severity: string }) {
  const colorMap: Record<string, string> = {
    LOW: 'bg-observatory-success/15 text-observatory-success border-observatory-success/30',
    MODERATE: 'bg-observatory-warning/15 text-observatory-warning border-observatory-warning/30',
    SEVERE: 'bg-observatory-danger/15 text-observatory-danger border-observatory-danger/30',
    ELEVATED: 'bg-observatory-warning/15 text-observatory-warning border-observatory-warning/30',
    REFUSED: 'bg-observatory-surface-alt text-observatory-text-dim border-observatory-border',
  };
  return (
    <span className={`text-xs px-2.5 py-1 rounded-lg border font-medium ${colorMap[severity] || colorMap.LOW}`}>
      {severity}
    </span>
  );
}

const fadeUp = {
  hidden: { opacity: 0, y: 30 },
  visible: (i: number) => ({
    opacity: 1, y: 0,
    transition: { delay: i * 0.08, duration: 0.5, ease: [0.25, 0.46, 0.45, 0.94] },
  }),
};

const stagger = {
  hidden: {},
  visible: { transition: { staggerChildren: 0.06 } },
};

export default function LandingPage() {
  const leaderboardRef = useRef(null);
  const leaderboardInView = useInView(leaderboardRef, { once: true, margin: '-50px' });

  return (
    <div className="min-h-screen bg-observatory-bg text-observatory-text">
      {/* Hero */}
      <div className="relative overflow-hidden">
        <div className="absolute inset-0 pointer-events-none opacity-[0.04]" style={{
          backgroundImage: 'radial-gradient(circle at 1px 1px, hsl(var(--obs-accent)) 1px, transparent 0)',
          backgroundSize: '40px 40px',
        }} />
        <motion.div
          className="absolute top-0 left-1/2 -translate-x-1/2 w-[600px] h-[600px] bg-observatory-accent/5 rounded-full blur-[120px]"
          animate={{ scale: [1, 1.15, 1], opacity: [0.5, 0.8, 0.5] }}
          transition={{ duration: 6, repeat: Infinity, ease: 'easeInOut' }}
        />

        <div className="relative max-w-6xl mx-auto px-6 pt-20 pb-16">
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5 }}
            className="flex items-center gap-3 mb-6"
          >
            <div className="w-12 h-12 rounded-2xl bg-observatory-accent/15 flex items-center justify-center">
              <Fingerprint className="w-7 h-7 text-observatory-accent" />
            </div>
            <motion.span
              className="text-xs font-mono px-3 py-1 rounded-full bg-observatory-warning/10 text-observatory-warning border border-observatory-warning/20"
              animate={{ opacity: [1, 0.6, 1] }}
              transition={{ duration: 2, repeat: Infinity }}
            >
              <Zap className="w-3 h-3 inline mr-1" />
              Ethical & Responsible GenAI Track · March 2026
            </motion.span>
          </motion.div>

          <motion.h1
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.1 }}
            className="text-4xl sm:text-5xl md:text-7xl font-black tracking-tight leading-[0.95] mb-6"
          >
            Finger<span className="gradient-text">print</span><sup className="text-observatory-accent text-2xl align-super">²</sup>
          </motion.h1>
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="text-base sm:text-xl text-observatory-text-muted max-w-2xl mb-8 leading-relaxed"
          >
            Multi-dimensional bias fingerprinting for vision-language models. 
            Benchmark, detect, and mitigate bias with real data from Sony FHIBE.
          </motion.p>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.3 }}
            className="flex flex-wrap gap-2 sm:gap-3 mb-12"
          >
            <Link to="/report" className="group inline-flex items-center gap-2 px-4 sm:px-6 py-2.5 sm:py-3 rounded-xl bg-observatory-accent text-observatory-bg font-semibold text-xs sm:text-sm hover:bg-observatory-accent-glow transition-all hover:shadow-[0_0_30px_hsl(var(--obs-accent)/0.3)]">
              View Bias Report <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
            </Link>
            <Link to="/scan" className="group inline-flex items-center gap-2 px-4 sm:px-6 py-2.5 sm:py-3 rounded-xl border border-observatory-border text-observatory-text-muted font-medium text-xs sm:text-sm hover:border-observatory-accent/50 hover:text-observatory-accent transition-all">
              <Camera className="w-4 h-4 group-hover:scale-110 transition-transform" /> Scan Your Face
            </Link>
            <a href={FRAMEWORK_INFO.repo} target="_blank" rel="noopener" className="inline-flex items-center gap-2 px-4 sm:px-6 py-2.5 sm:py-3 rounded-xl border border-observatory-border text-observatory-text-muted font-medium text-xs sm:text-sm hover:border-observatory-text-dim transition-all">
              <Github className="w-4 h-4" /> GitHub
            </a>
          </motion.div>

          {/* Animated Stats */}
          <motion.div
            variants={stagger}
            initial="hidden"
            animate="visible"
            className="grid grid-cols-2 md:grid-cols-4 gap-4"
          >
            {[
              { value: '3000', label: 'Images', suffix: '' },
              { value: '6', label: 'Regions', suffix: '' },
              { value: '5', label: 'Probes', suffix: '' },
              { value: '6', label: 'VLMs', suffix: '' },
            ].map((s, i) => (
              <motion.div key={s.label} variants={fadeUp} custom={i} className="glass rounded-2xl px-4 sm:px-5 py-3 sm:py-4 hover:border-observatory-accent/20 transition-all cursor-default group">
                <AnimatedCounter target={s.value} suffix={s.suffix} />
                <div className="text-xs text-observatory-text-dim uppercase tracking-wider mt-1 group-hover:text-observatory-text-muted transition-colors">{s.label}</div>
              </motion.div>
            ))}
          </motion.div>
        </div>
      </div>

      {/* Leaderboard */}
      <div ref={leaderboardRef} className="max-w-6xl mx-auto px-4 sm:px-6 py-8 sm:py-12">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={leaderboardInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.5 }}
        >
          <h2 className="text-2xl font-bold mb-2">Bias Leaderboard</h2>
          <p className="text-sm text-observatory-text-muted mb-2">Lower composite score = less biased · All models ranked on FHIBE benchmark</p>
          <div className="flex flex-wrap gap-3 text-xs text-observatory-text-dim mb-6">
            <span className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-full bg-observatory-success" /> 0.0–0.1 Low</span>
            <span className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-full bg-observatory-warning" /> 0.1–0.2 Moderate</span>
            <span className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-full bg-observatory-danger" /> 0.3–0.4 Severe</span>
          </div>
        </motion.div>
        <div className="space-y-2">
          {LEADERBOARD.map((m, i) => {
            const grade = getSeverityGrade(m.composite_score);
            return (
              <motion.div
                key={m.id}
                initial={{ opacity: 0, x: -30 }}
                animate={leaderboardInView ? { opacity: 1, x: 0 } : {}}
                transition={{ delay: i * 0.1, duration: 0.4 }}
                whileHover={{ scale: 1.01, x: 8 }}
                className="glass rounded-xl px-3 sm:px-5 py-3 sm:py-4 flex items-center gap-2 sm:gap-4 hover:border-observatory-accent/30 transition-all cursor-default"
              >
                <span className="text-base sm:text-lg font-mono font-bold text-observatory-accent w-8 sm:w-10">#{i + 1}</span>
                <div className="flex-1 min-w-0">
                  <div className="font-semibold text-sm sm:text-base truncate">{m.name}</div>
                  <div className="text-[10px] sm:text-xs text-observatory-text-dim font-mono truncate">{m.hf_id}</div>
                </div>
                <span className="text-xs text-observatory-text-dim hidden sm:block">{m.params} · {m.provider}</span>
                <div className="text-right">
                  <span className="font-mono font-bold text-base sm:text-lg" style={{ color: m.color }}>{m.composite_score.toFixed(3)}</span>
                  <div className={`text-[10px] font-medium ${grade.color}`}>{grade.label}</div>
                </div>
                <SeverityBadge severity={m.severity} />
              </motion.div>
            );
          })}
        </div>
      </div>

      {/* Modules Grid */}
      <div className="max-w-6xl mx-auto px-4 sm:px-6 py-8 sm:py-12">
        <h2 className="text-2xl font-bold mb-2">Explore Modules</h2>
        <p className="text-sm text-observatory-text-muted mb-6">Each module demonstrates a different aspect of VLM bias analysis</p>
        <motion.div
          variants={stagger}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: '-50px' }}
          className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4"
        >
          {MODULES.map((m, i) => (
            <motion.div key={m.to} variants={fadeUp} custom={i}>
              <Link
                to={m.to}
                className="glass rounded-2xl p-4 sm:p-6 hover:border-observatory-accent/30 transition-all group block hover:shadow-[0_0_40px_hsl(var(--obs-accent)/0.08)]"
              >
                <div className="flex items-center gap-3 mb-3">
                  <div className="w-10 h-10 rounded-xl bg-observatory-accent/10 flex items-center justify-center group-hover:bg-observatory-accent/20 transition-colors">
                    <m.icon className="w-5 h-5 text-observatory-accent group-hover:scale-110 transition-transform" />
                  </div>
                  <span className="text-[10px] font-mono px-2 py-0.5 rounded-md bg-observatory-surface-alt text-observatory-text-dim uppercase">{m.tag}</span>
                </div>
                <h3 className="font-semibold text-lg mb-1 group-hover:text-observatory-accent transition-colors">{m.label}</h3>
                <p className="text-sm text-observatory-text-muted">{m.desc}</p>
                <div className="mt-4 flex items-center gap-1 text-xs text-observatory-accent opacity-0 group-hover:opacity-100 translate-y-1 group-hover:translate-y-0 transition-all">
                  Open <ArrowRight className="w-3 h-3" />
                </div>
              </Link>
            </motion.div>
          ))}
        </motion.div>
      </div>

      {/* Team */}
      <div className="max-w-6xl mx-auto px-4 sm:px-6 py-8 sm:py-12">
        <h2 className="text-xl sm:text-2xl font-bold mb-6">Team</h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 sm:gap-6">
          {[
            { name: 'Razije', role: 'PhD Researcher', from: 'Switzerland', age: '30–38', photo: razije },
            { name: 'Ahmed', role: 'PhD Researcher', from: 'UAE', age: '20–28', photo: ahmed },
          ].map((p, i) => (
            <motion.div
              key={p.name}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: i * 0.15 }}
              whileHover={{ y: -4 }}
              className="glass rounded-2xl p-4 sm:p-6 flex items-center gap-4 sm:gap-5 hover:border-observatory-accent/20 transition-all"
            >
              <img src={p.photo} alt={p.name} className="w-16 h-16 sm:w-20 sm:h-20 rounded-xl object-cover border border-observatory-border/40" />
              <div>
                <h3 className="text-lg font-bold">{p.name}</h3>
                <p className="text-sm text-observatory-text-muted">{p.role}</p>
                <p className="text-xs text-observatory-text-dim mt-1">{p.from} · Age: {p.age}</p>
              </div>
            </motion.div>
          ))}
        </div>
      </div>

      {/* Footer */}
      <div className="border-t border-observatory-border/30 px-4 sm:px-6 py-4 sm:py-6">
        <div className="max-w-6xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-2 text-xs text-observatory-text-dim">
          <span className="font-mono font-bold">Fingerprint² Bench</span>
          <div className="flex items-center gap-3 sm:gap-4 flex-wrap justify-center">
            <a href={FRAMEWORK_INFO.repo} target="_blank" rel="noopener" className="hover:text-observatory-text-muted transition-colors flex items-center gap-1">
              <ExternalLink className="w-3 h-3" /> GitHub
            </a>
            <a href={FRAMEWORK_INFO.datasetUrl} target="_blank" rel="noopener" className="hover:text-observatory-text-muted transition-colors flex items-center gap-1">
              <ExternalLink className="w-3 h-3" /> FHIBE Dataset
            </a>
            <span>March 2026</span>
          </div>
        </div>
      </div>
    </div>
  );
}
