import { useState } from 'react';
import { Shield, AlertTriangle, CheckCircle, XCircle } from 'lucide-react';

type Tab = 'risks' | 'airport' | 'checker' | 'timeline';

const RISK_CATEGORIES = [
  {
    level: 'UNACCEPTABLE RISK',
    color: 'bg-red-500/20 text-red-400 border-red-500/30',
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
    color: 'bg-orange-500/20 text-orange-400 border-orange-500/30',
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
    color: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30',
    airport: 2,
    items: [
      { name: 'Chatbots and virtual assistants', airport: true, desc: 'Airport info kiosks must disclose AI nature.' },
      { name: 'Deepfakes and AI-generated content', airport: true, desc: 'Must be labelled.' },
    ],
  },
  {
    level: 'MINIMAL RISK',
    color: 'bg-green-500/20 text-green-400 border-green-500/30',
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
    <div className="p-4 md:p-8 max-w-7xl mx-auto">
      <header className="mb-6">
        <h1 className="text-2xl font-mono font-bold gradient-text flex items-center gap-2">
          <Shield className="w-6 h-6" /> EU AI Act — Compliance Framework
        </h1>
        <p className="text-observatory-text-muted text-sm mt-1">Regulation (EU) 2024/1689 · Entered into force 1 August 2024</p>
      </header>

      <div className="flex gap-1 mb-6">
        {(['risks', 'airport', 'checker', 'timeline'] as Tab[]).map(t => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
              tab === t ? 'bg-observatory-accent/15 text-observatory-accent' : 'text-observatory-text-muted hover:text-observatory-text'
            }`}
          >
            {t === 'risks' ? 'Risk Categories' : t === 'airport' ? 'Airport AI Rules' : t === 'checker' ? 'Compliance Checker' : 'Timeline'}
          </button>
        ))}
      </div>

      {tab === 'risks' && (
        <div className="space-y-6">
          {RISK_CATEGORIES.map(cat => (
            <div key={cat.level} className={`glass rounded-xl p-6 border ${cat.color.split(' ')[2]}`}>
              <div className="flex items-center justify-between mb-4">
                <h3 className={`font-mono font-bold text-sm ${cat.color.split(' ')[1]}`}>{cat.level}</h3>
                {cat.airport > 0 && (
                  <span className="text-xs font-mono px-2 py-0.5 rounded-full bg-observatory-surface-alt text-observatory-text-muted">
                    ✈️ {cat.airport} airport-relevant
                  </span>
                )}
              </div>
              <div className="space-y-2">
                {cat.items.map(item => (
                  <div key={item.name} className="bg-observatory-bg/50 rounded-lg px-4 py-3">
                    <div className="flex items-start gap-2">
                      {item.airport ? (
                        <span className="text-xs mt-0.5">✈️</span>
                      ) : (
                        <span className="text-xs mt-0.5 text-observatory-text-dim">·</span>
                      )}
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
        <div className="glass rounded-xl p-6">
          <h3 className="font-mono font-bold text-observatory-text mb-4">Airport AI — High-Risk Classification</h3>
          <div className="space-y-4 text-sm text-observatory-text-muted">
            <p>Airport AI systems involving biometric identification, border control, and risk profiling are classified as <b className="text-observatory-danger">High-Risk AI</b> under Annex III.</p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {[
                { article: 'Art. 5', title: 'Prohibited Practices', items: ['Real-time biometric identification in public spaces (with limited exceptions)', 'Biometric categorisation inferring race, religion, sexual orientation'] },
                { article: 'Art. 6-7', title: 'High-Risk Requirements', items: ['Conformity assessment before deployment', 'Registration in EU database', 'Ongoing monitoring obligations'] },
                { article: 'Art. 10', title: 'Data Governance', items: ['Training data must be representative', 'Bias detection and mitigation required', 'This is where Fingerprint² helps'] },
                { article: 'Art. 14', title: 'Human Oversight', items: ['Human-in-the-loop for high-impact decisions', 'Override capability required', 'Audit trails mandatory'] },
              ].map(s => (
                <div key={s.article} className="bg-observatory-bg/50 rounded-lg p-4">
                  <div className="font-mono text-observatory-accent text-xs mb-1">{s.article}</div>
                  <div className="font-medium text-observatory-text text-sm mb-2">{s.title}</div>
                  <ul className="space-y-1">
                    {s.items.map(item => <li key={item} className="text-xs text-observatory-text-muted">• {item}</li>)}
                  </ul>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {tab === 'checker' && <ComplianceChecker />}

      {tab === 'timeline' && (
        <div className="glass rounded-xl p-6">
          <h3 className="font-mono font-bold text-observatory-text mb-4">EU AI Act Timeline</h3>
          <div className="space-y-4">
            {[
              { date: '1 Aug 2024', event: 'Regulation enters into force', status: 'done' },
              { date: '2 Feb 2025', event: 'Prohibited AI practices banned', status: 'done' },
              { date: '2 Aug 2025', event: 'General-purpose AI rules apply', status: 'done' },
              { date: '2 Aug 2026', event: 'High-risk AI obligations apply', status: 'upcoming' },
              { date: '2 Aug 2027', event: 'Full enforcement for all AI systems', status: 'upcoming' },
            ].map(t => (
              <div key={t.date} className="flex items-center gap-4">
                <div className={`w-3 h-3 rounded-full shrink-0 ${t.status === 'done' ? 'bg-observatory-success' : 'bg-observatory-warning'}`} />
                <div className="flex-1">
                  <div className="text-sm text-observatory-text">{t.event}</div>
                  <div className="text-xs text-observatory-text-dim font-mono">{t.date}</div>
                </div>
                <span className={`text-xs px-2 py-0.5 rounded-full ${t.status === 'done' ? 'bg-observatory-success/10 text-observatory-success' : 'bg-observatory-warning/10 text-observatory-warning'}`}>
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
    <div className="glass rounded-xl p-6">
      <h3 className="font-mono font-bold text-observatory-text mb-4">Quick Compliance Check</h3>
      <div className="space-y-3">
        {checks.map((c, i) => (
          <div key={i} className="flex items-center justify-between bg-observatory-bg/50 rounded-lg px-4 py-3">
            <span className="text-sm text-observatory-text-muted flex-1">{c.q}</span>
            <div className="flex gap-2">
              <button
                onClick={() => setAnswers({ ...answers, [i]: true })}
                className={`px-3 py-1 rounded text-xs ${answers[i] === true ? 'bg-observatory-success/20 text-observatory-success' : 'text-observatory-text-dim hover:bg-observatory-surface-alt'}`}
              >
                Yes
              </button>
              <button
                onClick={() => setAnswers({ ...answers, [i]: false })}
                className={`px-3 py-1 rounded text-xs ${answers[i] === false ? 'bg-observatory-danger/20 text-observatory-danger' : 'text-observatory-text-dim hover:bg-observatory-surface-alt'}`}
              >
                No
              </button>
            </div>
          </div>
        ))}
      </div>
      {Object.keys(answers).length >= 4 && (
        <div className="mt-4 p-4 rounded-lg bg-observatory-accent/10">
          <div className="text-sm text-observatory-accent font-mono">Assessment Result</div>
          <div className="text-xs text-observatory-text-muted mt-1">
            Based on your answers, this system likely falls under <b className="text-observatory-warning">High-Risk</b> classification.
            Fingerprint² can help with Art. 10 bias testing requirements.
          </div>
        </div>
      )}
    </div>
  );
}
