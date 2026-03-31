import { useState } from 'react';
import { Shield, AlertTriangle, CheckCircle, XCircle } from 'lucide-react';
import ModelComplianceReport from '@/components/euaiact/ModelComplianceReport';

type Tab = 'risks' | 'airport' | 'checker' | 'model' | 'timeline';

const RISK_CATEGORIES = [
  {
    level: 'UNACCEPTABLE RISK',
    color: 'border-red-500/30',
    textColor: 'text-red-400',
    bgColor: 'bg-red-500/5',
    airport: 2,
    items: [
      { name: 'Social scoring by public authorities', airport: false },
      { name: 'Real-time remote biometric identification in public spaces', airport: true, desc: 'Banned for law enforcement (Art. 5). Airport facial recognition for security falls here.' },
      { name: 'Emotion recognition in workplaces and schools', airport: false },
      { name: 'Biometric categorisation inferring sensitive attributes', airport: true, desc: 'Inferring race, political opinions, religion from biometrics.' },
    ],
  },
  {
    level: 'HIGH RISK',
    color: 'border-orange-500/30',
    textColor: 'text-orange-400',
    bgColor: 'bg-orange-500/5',
    airport: 5,
    items: [
      { name: 'Biometric identification and categorisation (Annex III.1)', airport: true, desc: 'E-ID verification, passport matching, face recognition at gates.' },
      { name: 'Management of critical infrastructure', airport: true, desc: 'Air traffic management, baggage handling AI.' },
      { name: 'Migration, asylum, border control (Annex III.7)', airport: true, desc: 'Automated border gates, visa risk assessment.' },
      { name: 'Access to essential services — credit scoring (Annex III.5)', airport: false },
      { name: 'Law enforcement profiling (Annex III.6)', airport: true, desc: 'Predictive policing, risk assessment at borders.' },
      { name: 'Employment & worker management', airport: true, desc: 'AI screening of airport staff applications.' },
    ],
  },
  {
    level: 'LIMITED RISK',
    color: 'border-yellow-500/30',
    textColor: 'text-yellow-400',
    bgColor: 'bg-yellow-500/5',
    airport: 2,
    items: [
      { name: 'Chatbots and virtual assistants', airport: true, desc: 'Airport info kiosks must disclose AI nature.' },
      { name: 'Deepfakes and AI-generated content', airport: true, desc: 'Must be labelled.' },
    ],
  },
  {
    level: 'MINIMAL RISK',
    color: 'border-green-500/30',
    textColor: 'text-green-400',
    bgColor: 'bg-green-500/5',
    airport: 0,
    items: [
      { name: 'AI-enabled video games', airport: false },
      { name: 'Spam filters', airport: false },
    ],
  },
];

export default function EUAIActPage() {
  const [tab, setTab] = useState<Tab>('risks');

  return (
    <div className="page-container">
      <header className="page-header">
        <h1 className="page-title">
          <Shield className="w-7 h-7 text-observatory-accent" />
          <span className="gradient-text">EU AI Act</span>
        </h1>
        <p className="page-subtitle">Regulation (EU) 2024/1689 · Entered into force 1 August 2024</p>
      </header>

      <div className="tab-group">
        {([['risks', 'Risk Categories'], ['model', 'Model Report'], ['airport', 'Airport AI Rules'], ['checker', 'Compliance Checker'], ['timeline', 'Timeline']] as [Tab, string][]).map(([t, label]) => (
          <button key={t} onClick={() => setTab(t)} className={`tab-button ${tab === t ? 'tab-active' : 'tab-inactive'}`}>
            {label}
          </button>
        ))}
      </div>

      {tab === 'risks' && (
        <div className="space-y-6">
          {RISK_CATEGORIES.map(cat => (
            <div key={cat.level} className={`card border ${cat.color} ${cat.bgColor}`}>
              <div className="flex items-center justify-between mb-4">
                <h3 className={`font-mono font-bold ${cat.textColor}`}>{cat.level}</h3>
                {cat.airport > 0 && (
                  <span className="text-xs font-mono px-3 py-1 rounded-lg bg-observatory-surface-alt text-observatory-text-muted">
                    ✈️ {cat.airport} airport-relevant
                  </span>
                )}
              </div>
              <div className="space-y-2">
                {cat.items.map(item => (
                  <div key={item.name} className="bg-observatory-bg/40 rounded-xl px-5 py-3">
                    <div className="flex items-start gap-2">
                      {item.airport ? <span className="text-xs mt-0.5">✈️</span> : <span className="text-xs mt-0.5 text-observatory-text-dim">·</span>}
                      <div>
                        <div className="text-sm text-observatory-text">{item.name}</div>
                        {item.desc && <div className="text-xs text-observatory-text-dim mt-1">{item.desc}</div>}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}

      {tab === 'airport' && (
        <div className="card">
          <div className="card-header">Airport AI — High-Risk Classification</div>
          <div className="space-y-4 text-sm text-observatory-text-muted">
            <p>Airport AI systems involving biometric identification, border control, and risk profiling are classified as <b className="text-observatory-danger">High-Risk AI</b> under Annex III.</p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {[
                { article: 'Art. 5', title: 'Prohibited Practices', items: ['Real-time biometric identification in public spaces (with limited exceptions)', 'Biometric categorisation inferring race, religion, sexual orientation'] },
                { article: 'Art. 6-7', title: 'High-Risk Requirements', items: ['Conformity assessment before deployment', 'Registration in EU database', 'Ongoing monitoring obligations'] },
                { article: 'Art. 10', title: 'Data Governance', items: ['Training data must be representative', 'Bias detection and mitigation required', 'This is where Fingerprint² helps'] },
                { article: 'Art. 14', title: 'Human Oversight', items: ['Human-in-the-loop for high-impact decisions', 'Override capability required', 'Audit trails mandatory'] },
              ].map(s => (
                <div key={s.article} className="bg-observatory-bg/50 rounded-xl p-5">
                  <div className="font-mono text-observatory-accent text-xs mb-1">{s.article}</div>
                  <div className="font-semibold text-observatory-text mb-2">{s.title}</div>
                  <ul className="space-y-1">
                    {s.items.map(item => <li key={item} className="text-xs text-observatory-text-muted">• {item}</li>)}
                  </ul>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {tab === 'model' && <ModelComplianceReport />}

      {tab === 'checker' && <ComplianceChecker />}

      {tab === 'timeline' && (
        <div className="card">
          <div className="card-header">EU AI Act Timeline</div>
          <div className="space-y-4">
            {[
              { date: '1 Aug 2024', event: 'Regulation enters into force', status: 'done' },
              { date: '2 Feb 2025', event: 'Prohibited AI practices banned', status: 'done' },
              { date: '2 Aug 2025', event: 'General-purpose AI rules apply', status: 'done' },
              { date: '2 Aug 2026', event: 'High-risk AI obligations apply', status: 'upcoming' },
              { date: '2 Aug 2027', event: 'Full enforcement for all AI systems', status: 'upcoming' },
            ].map(t => (
              <div key={t.date} className="flex items-center gap-4 bg-observatory-bg/50 rounded-xl px-5 py-4">
                <div className={`w-3 h-3 rounded-full shrink-0 ${t.status === 'done' ? 'bg-observatory-success' : 'bg-observatory-warning'}`} />
                <div className="flex-1">
                  <div className="text-sm text-observatory-text font-medium">{t.event}</div>
                  <div className="text-xs text-observatory-text-dim font-mono">{t.date}</div>
                </div>
                <span className={`text-xs px-3 py-1 rounded-lg ${t.status === 'done' ? 'bg-observatory-success/10 text-observatory-success' : 'bg-observatory-warning/10 text-observatory-warning'}`}>
                  {t.status === 'done' ? 'Active' : 'Upcoming'}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function ComplianceChecker() {
  const checks = [
    { q: 'Does the system use biometric identification?', risk: 'high' },
    { q: 'Is it deployed in a public space?', risk: 'unacceptable' },
    { q: 'Does it process personal data for profiling?', risk: 'high' },
    { q: 'Is there human oversight capability?', risk: 'required' },
    { q: 'Has bias testing been conducted (Art. 10)?', risk: 'required' },
    { q: 'Is the system registered in the EU AI database?', risk: 'required' },
  ];
  const [answers, setAnswers] = useState<Record<number, boolean>>({});

  return (
    <div className="card">
      <div className="card-header">Quick Compliance Check</div>
      <div className="space-y-3">
        {checks.map((c, i) => (
          <div key={i} className="flex items-center justify-between bg-observatory-bg/50 rounded-xl px-5 py-4">
            <span className="text-sm text-observatory-text-muted flex-1">{c.q}</span>
            <div className="flex gap-2">
              <button
                onClick={() => setAnswers({ ...answers, [i]: true })}
                className={`px-4 py-1.5 rounded-lg text-xs font-medium ${answers[i] === true ? 'bg-observatory-success/20 text-observatory-success' : 'text-observatory-text-dim hover:bg-observatory-surface-alt'}`}
              >
                Yes
              </button>
              <button
                onClick={() => setAnswers({ ...answers, [i]: false })}
                className={`px-4 py-1.5 rounded-lg text-xs font-medium ${answers[i] === false ? 'bg-observatory-danger/20 text-observatory-danger' : 'text-observatory-text-dim hover:bg-observatory-surface-alt'}`}
              >
                No
              </button>
            </div>
          </div>
        ))}
      </div>
      {Object.keys(answers).length >= 4 && (
        <div className="mt-6 p-5 rounded-xl bg-observatory-accent/10 border border-observatory-accent/20">
          <div className="text-sm text-observatory-accent font-semibold">Assessment Result</div>
          <div className="text-sm text-observatory-text-muted mt-1">
            Based on your answers, this system likely falls under <b className="text-observatory-warning">High-Risk</b> classification.
            Fingerprint² can help with Art. 10 bias testing requirements.
          </div>
        </div>
      )}
    </div>
  );
}
