import { useState } from 'react';
import { CreditCard, AlertTriangle, CheckCircle, Shield, TrendingUp, Users } from 'lucide-react';

interface Applicant {
  name: string; nationality: string; flag: string; ethnicity: string; emoji: string;
  income: string; creditHistory: string; employment: string;
  biasedScore: number; fairScore: number;
  biasedRate: number; fairRate: number;
}

const APPLICANTS: Applicant[] = [
  { name: 'James Wilson', nationality: 'American', flag: '🇺🇸', ethnicity: 'White', emoji: '👨🏻', income: '€65,000', creditHistory: 'Good', employment: 'Employed', biasedScore: 92, fairScore: 88, biasedRate: 4.2, fairRate: 4.8 },
  { name: 'Fatima Hassan', nationality: 'Moroccan', flag: '🇲🇦', ethnicity: 'Arab', emoji: '👩🏽', income: '€68,000', creditHistory: 'Good', employment: 'Employed', biasedScore: 64, fairScore: 89, biasedRate: 7.8, fairRate: 4.7 },
  { name: 'Chen Wei', nationality: 'Chinese', flag: '🇨🇳', ethnicity: 'Asian', emoji: '👨🏻', income: '€72,000', creditHistory: 'Excellent', employment: 'Employed', biasedScore: 78, fairScore: 91, biasedRate: 5.9, fairRate: 4.5 },
  { name: 'Adama Diallo', nationality: 'Senegalese', flag: '🇸🇳', ethnicity: 'Black', emoji: '👨🏿', income: '€61,000', creditHistory: 'Good', employment: 'Employed', biasedScore: 58, fairScore: 86, biasedRate: 8.5, fairRate: 5.1 },
  { name: 'Sophie Laurent', nationality: 'Swiss', flag: '🇨🇭', ethnicity: 'White', emoji: '👩🏼', income: '€70,000', creditHistory: 'Good', employment: 'Employed', biasedScore: 94, fairScore: 89, biasedRate: 3.9, fairRate: 4.7 },
  { name: 'Priya Sharma', nationality: 'Indian', flag: '🇮🇳', ethnicity: 'South Asian', emoji: '👩🏽', income: '€67,000', creditHistory: 'Good', employment: 'Employed', biasedScore: 69, fairScore: 88, biasedRate: 6.8, fairRate: 4.8 },
];

export default function BankingPage() {
  const [modelType, setModelType] = useState<'biased' | 'fair'>('biased');
  const [selected, setSelected] = useState<Applicant | null>(null);

  const getScoreColor = (s: number) => s >= 80 ? 'text-observatory-success' : s >= 65 ? 'text-observatory-warning' : 'text-observatory-danger';

  return (
    <div className="p-4 md:p-8 max-w-7xl mx-auto">
      <header className="mb-6">
        <h1 className="text-2xl font-mono font-bold gradient-text flex items-center gap-2">
          <CreditCard className="w-6 h-6" /> AI Credit Scoring — Bias Simulator
        </h1>
        <p className="text-observatory-text-muted text-sm mt-1">
          How AI bias in passport/photo verification affects credit approvals and interest rates
        </p>
      </header>

      {/* Model toggle */}
      <div className="glass rounded-xl p-4 mb-6 flex flex-wrap items-center gap-4">
        <div className="text-xs text-observatory-text-dim">AI Scoring Model:</div>
        {(['biased', 'fair'] as const).map(t => (
          <button
            key={t}
            onClick={() => setModelType(t)}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
              modelType === t ? 'bg-observatory-accent/15 text-observatory-accent' : 'text-observatory-text-muted hover:bg-observatory-surface-alt'
            }`}
          >
            {t === 'biased' ? '⚠️ Standard AI (with bias)' : '✅ Debiased AI (Fingerprint²)'}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Applicants */}
        <div className="glass rounded-xl p-5">
          <h3 className="text-xs font-mono text-observatory-text-dim mb-3">Credit Applicants ({APPLICANTS.length})</h3>
          <div className="space-y-2">
            {APPLICANTS.map(a => {
              const score = modelType === 'biased' ? a.biasedScore : a.fairScore;
              return (
                <button
                  key={a.name}
                  onClick={() => setSelected(a)}
                  className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-left transition-all ${
                    selected?.name === a.name ? 'bg-observatory-accent/15' : 'bg-observatory-bg/50 hover:bg-observatory-surface-alt'
                  }`}
                >
                  <span className="text-lg">{a.emoji}</span>
                  <div className="flex-1 min-w-0">
                    <div className="text-sm text-observatory-text truncate">{a.name}</div>
                    <div className="text-xs text-observatory-text-dim">{a.nationality} {a.flag} · {a.income}</div>
                  </div>
                  <div className={`font-mono font-bold text-sm ${getScoreColor(score)}`}>{score}</div>
                </button>
              );
            })}
          </div>
        </div>

        {/* Application result */}
        <div className="glass rounded-xl p-5">
          {!selected ? (
            <div className="flex flex-col items-center justify-center h-full text-center py-12">
              <CreditCard className="w-12 h-12 text-observatory-text-dim mb-4" />
              <p className="text-observatory-text-muted text-sm">Select an applicant to review their credit application</p>
            </div>
          ) : (
            <div>
              <div className="text-center mb-6">
                <div className="text-4xl mb-2">{selected.emoji}</div>
                <h3 className="text-lg font-semibold text-observatory-text">{selected.name}</h3>
                <p className="text-sm text-observatory-text-muted">{selected.nationality} {selected.flag}</p>
              </div>

              <div className="space-y-3 mb-6">
                {[
                  ['Annual Income', selected.income],
                  ['Credit History', selected.creditHistory],
                  ['Employment', selected.employment],
                  ['Ethnicity', selected.ethnicity],
                ].map(([k, v]) => (
                  <div key={k} className="flex justify-between text-sm">
                    <span className="text-observatory-text-dim">{k}</span>
                    <span className="text-observatory-text font-mono">{v}</span>
                  </div>
                ))}
              </div>

              <div className="text-center mb-4">
                <div className="text-xs text-observatory-text-dim mb-1">CREDIT SCORE</div>
                <div className={`text-5xl font-mono font-bold ${getScoreColor(modelType === 'biased' ? selected.biasedScore : selected.fairScore)}`}>
                  {modelType === 'biased' ? selected.biasedScore : selected.fairScore}
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3 mb-4">
                <div className="bg-observatory-bg/50 rounded-lg p-3 text-center">
                  <div className="text-[10px] text-observatory-text-dim">Interest Rate</div>
                  <div className="font-mono font-bold text-sm text-observatory-text">
                    {modelType === 'biased' ? selected.biasedRate : selected.fairRate}%
                  </div>
                </div>
                <div className="bg-observatory-bg/50 rounded-lg p-3 text-center">
                  <div className="text-[10px] text-observatory-text-dim">Decision</div>
                  <div className={`font-bold text-sm ${(modelType === 'biased' ? selected.biasedScore : selected.fairScore) >= 70 ? 'text-observatory-success' : 'text-observatory-danger'}`}>
                    {(modelType === 'biased' ? selected.biasedScore : selected.fairScore) >= 70 ? 'APPROVED' : 'DENIED'}
                  </div>
                </div>
              </div>

              {modelType === 'biased' && selected.biasedScore < selected.fairScore - 10 && (
                <div className="p-3 rounded-lg bg-observatory-danger/10 text-xs text-observatory-danger">
                  ⚠️ Bias detected: This applicant's score is {selected.fairScore - selected.biasedScore} points lower due to demographic bias.
                  With Fingerprint² debiased scoring: {selected.fairScore} → {selected.fairRate}% rate.
                </div>
              )}
            </div>
          )}
        </div>

        {/* Impact analysis */}
        <div className="space-y-4">
          <div className="glass rounded-xl p-5">
            <h3 className="text-xs font-mono text-observatory-text-dim mb-3">
              <Shield className="w-4 h-4 inline mr-1" /> How Fingerprint² Removes Credit Bias
            </h3>
            <div className="space-y-3 text-xs text-observatory-text-muted">
              <div className="bg-observatory-bg/50 rounded-lg p-3">
                <div className="font-medium text-observatory-accent mb-1">1. Detect bias in identity verification</div>
                VLMs used for passport/photo matching show systematic bias by ethnicity, skin tone, and nationality.
              </div>
              <div className="bg-observatory-bg/50 rounded-lg p-3">
                <div className="font-medium text-observatory-accent mb-1">2. Measure disparity with FHIBE probes</div>
                Our 5-probe Social Inference Battery quantifies exactly how much bias each model introduces.
              </div>
              <div className="bg-observatory-bg/50 rounded-lg p-3">
                <div className="font-medium text-observatory-accent mb-1">3. Apply fairness constraints</div>
                Calibrate model outputs to equalize false positive/negative rates across demographic groups.
              </div>
              <div className="bg-observatory-bg/50 rounded-lg p-3">
                <div className="font-medium text-observatory-accent mb-1">4. Continuous monitoring</div>
                Deploy Fingerprint² bias dashboards to track drift and alert operators to emerging disparities.
              </div>
            </div>
          </div>

          <div className="glass rounded-xl p-5">
            <h3 className="text-xs font-mono text-observatory-text-dim mb-3">EU AI Act · Credit Scoring</h3>
            <div className="space-y-2 text-xs text-observatory-text-muted">
              <p><span className="text-observatory-accent font-mono">Art. 6 + Annex III(5)(b)</span> — AI systems for creditworthiness assessment are classified as <b>high-risk</b>.</p>
              <p><span className="text-observatory-accent font-mono">Art. 10(2)(f)</span> — Training data must be examined for biases that lead to discrimination.</p>
              <p><span className="text-observatory-accent font-mono">Art. 22 GDPR</span> — Right not to be subject to automated decision-making with legal effects.</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
