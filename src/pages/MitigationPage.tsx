import { useState } from 'react';
import { Wrench, Database, Brain, Settings, ClipboardList } from 'lucide-react';

type Tab = 'strategies' | 'metrics' | 'before-after';

const CATEGORIES = [
  {
    icon: Database, title: 'Data-Level Interventions',
    techniques: [
      { name: 'Balanced Dataset Curation', effort: 'Medium', impact: 'High', desc: 'Ensure equal representation across all demographic groups.' },
      { name: 'Adversarial Data Augmentation', effort: 'High', impact: 'High', desc: 'Synthetically generate underrepresented demographic scenarios.' },
      { name: 'Demographic Annotation Audits', effort: 'Low', impact: 'Medium', desc: 'Audit labels for systematic errors encoding human prejudice.' },
    ],
  },
  {
    icon: Brain, title: 'Model-Level Interventions',
    techniques: [
      { name: 'Fairness Constraints in Training', effort: 'Medium', impact: 'High', desc: 'Add demographic parity or equalized odds constraints to loss.' },
      { name: 'Adversarial Debiasing', effort: 'High', impact: 'High', desc: 'Train adversary to prevent learning demographic features.' },
      { name: 'Counterfactual Data Augmentation', effort: 'High', impact: 'Very High', desc: 'Create counterfactual versions differing only in demographics.' },
      { name: 'Temperature Calibration per Demographic', effort: 'Low', impact: 'Medium', desc: 'Calibrate confidence scores separately per group.' },
    ],
  },
  {
    icon: Settings, title: 'Post-hoc Corrections',
    techniques: [
      { name: 'Threshold Adjustment per Group', effort: 'Low', impact: 'Medium', desc: 'Set different thresholds per group for equal false positive rates.' },
      { name: 'Reject Option Classification', effort: 'Low', impact: 'Medium', desc: 'Abstain near decision boundary — route to human review.' },
      { name: 'Score Recalibration (Platt Scaling)', effort: 'Low', impact: 'Medium', desc: 'Apply isotonic regression to equalize probabilities across groups.' },
    ],
  },
  {
    icon: ClipboardList, title: 'Governance & Process',
    techniques: [
      { name: 'Continuous Bias Monitoring', effort: 'Medium', impact: 'High', desc: 'Deploy real-time bias dashboards that alert to disparities.' },
      { name: 'Red Team Bias Testing', effort: 'Medium', impact: 'High', desc: 'Hire diverse red teams to probe for failure modes.' },
      { name: 'Human-in-the-Loop', effort: 'Low', impact: 'High', desc: 'Route low-confidence decisions to trained human reviewers.' },
      { name: 'Model Cards & Factsheets', effort: 'Low', impact: 'Medium', desc: 'Publish comprehensive disclosures of known biases.' },
      { name: 'Diverse Development Teams', effort: 'Medium', impact: 'Very High', desc: 'Include affected demographic groups in design and testing.' },
    ],
  },
];

export default function MitigationPage() {
  const [tab, setTab] = useState<Tab>('strategies');

  const impactColor = (i: string) => i === 'Very High' ? 'text-observatory-accent' : i === 'High' ? 'text-observatory-success' : 'text-observatory-warning';

  return (
    <div className="p-4 md:p-8 max-w-7xl mx-auto">
      <header className="mb-6">
        <h1 className="text-2xl font-mono font-bold gradient-text flex items-center gap-2">
          <Wrench className="w-6 h-6" /> Bias Mitigation — Solutions & Strategies
        </h1>
        <p className="text-observatory-text-muted text-sm mt-1">Evidence-based techniques to reduce AI bias in VLMs</p>
      </header>

      <div className="flex gap-1 mb-6">
        {(['strategies', 'metrics', 'before-after'] as Tab[]).map(t => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
              tab === t ? 'bg-observatory-accent/15 text-observatory-accent' : 'text-observatory-text-muted hover:text-observatory-text'
            }`}
          >
            {t === 'strategies' ? 'Mitigation Strategies' : t === 'metrics' ? 'Fairness Metrics' : 'Before vs After'}
          </button>
        ))}
      </div>

      {tab === 'strategies' && (
        <div className="space-y-6">
          {CATEGORIES.map(cat => (
            <div key={cat.title} className="glass rounded-xl p-6">
              <h3 className="flex items-center gap-2 font-mono font-bold text-observatory-text mb-4">
                <cat.icon className="w-4 h-4 text-observatory-accent" />
                {cat.title}
                <span className="text-xs text-observatory-text-dim font-normal">{cat.techniques.length} techniques</span>
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {cat.techniques.map(t => (
                  <div key={t.name} className="bg-observatory-bg/50 rounded-lg p-4">
                    <div className="font-medium text-sm text-observatory-text mb-1">{t.name}</div>
                    <p className="text-xs text-observatory-text-muted mb-2">{t.desc}</p>
                    <div className="flex gap-3 text-[10px]">
                      <span className="text-observatory-text-dim">Effort: <b>{t.effort}</b></span>
                      <span className={impactColor(t.impact)}>Impact: <b>{t.impact}</b></span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}

      {tab === 'metrics' && (
        <div className="glass rounded-xl p-6">
          <h3 className="font-mono font-bold text-observatory-text mb-4">Key Fairness Metrics</h3>
          <div className="space-y-4">
            {[
              { name: 'Demographic Parity', formula: 'P(Ŷ=1|A=a) = P(Ŷ=1|A=b)', desc: 'Equal positive prediction rate across groups.' },
              { name: 'Equalized Odds', formula: 'P(Ŷ=1|Y=y,A=a) = P(Ŷ=1|Y=y,A=b)', desc: 'Equal TPR and FPR across groups.' },
              { name: 'Calibration', formula: 'P(Y=1|Ŷ=p,A=a) = p for all groups', desc: 'Predicted probabilities reflect true rates per group.' },
              { name: 'Counterfactual Fairness', formula: 'P(Ŷ_A←a = ŷ) = P(Ŷ_A←b = ŷ)', desc: 'Same prediction if demographic attribute were different.' },
              { name: 'Individual Fairness', formula: 'd(f(x),f(x′)) ≤ ε for similar x,x′', desc: 'Similar individuals get similar predictions.' },
            ].map(m => (
              <div key={m.name} className="bg-observatory-bg/50 rounded-lg p-4">
                <div className="font-medium text-sm text-observatory-text">{m.name}</div>
                <div className="font-mono text-xs text-observatory-accent mt-1">{m.formula}</div>
                <div className="text-xs text-observatory-text-muted mt-1">{m.desc}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {tab === 'before-after' && (
        <div className="glass rounded-xl p-6">
          <h3 className="font-mono font-bold text-observatory-text mb-4">Before vs After — Mitigation Impact</h3>
          <div className="space-y-4">
            {[
              { probe: 'P4 · Lifestyle', model: 'moondream2', before: 0.434, after: 0.12, technique: 'Counterfactual Data Augmentation' },
              { probe: 'P5 · Neighbourhood', model: 'moondream2', before: 0.557, after: 0.15, technique: 'Adversarial Debiasing + Threshold Adj.' },
              { probe: 'P1 · Occupation', model: 'Qwen2.5-VL-3B', before: 0.306, after: 0.08, technique: 'Balanced Dataset Curation' },
              { probe: 'P2 · Education', model: 'InternVL2-2B', before: 0.365, after: 0.09, technique: 'Fairness Constraints in Training' },
            ].map(item => (
              <div key={`${item.model}-${item.probe}`} className="bg-observatory-bg/50 rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <div>
                    <div className="text-sm text-observatory-text font-medium">{item.probe} · {item.model}</div>
                    <div className="text-xs text-observatory-text-dim">Technique: {item.technique}</div>
                  </div>
                </div>
                <div className="flex items-center gap-4">
                  <div className="flex-1">
                    <div className="text-xs text-observatory-text-dim mb-1">Before</div>
                    <div className="h-3 bg-observatory-border rounded-full overflow-hidden">
                      <div className="h-full bg-observatory-danger rounded-full" style={{ width: `${item.before * 100}%` }} />
                    </div>
                    <div className="text-xs font-mono text-observatory-danger mt-0.5">{item.before.toFixed(3)}</div>
                  </div>
                  <div className="text-observatory-text-dim">→</div>
                  <div className="flex-1">
                    <div className="text-xs text-observatory-text-dim mb-1">After</div>
                    <div className="h-3 bg-observatory-border rounded-full overflow-hidden">
                      <div className="h-full bg-observatory-success rounded-full" style={{ width: `${item.after * 100}%` }} />
                    </div>
                    <div className="text-xs font-mono text-observatory-success mt-0.5">{item.after.toFixed(3)}</div>
                  </div>
                  <div className="text-xs font-mono text-observatory-accent">
                    -{((1 - item.after / item.before) * 100).toFixed(0)}%
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
