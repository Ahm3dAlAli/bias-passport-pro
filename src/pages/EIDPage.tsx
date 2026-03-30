import { useState } from 'react';
import { ScanLine, Shield, AlertTriangle, CheckCircle } from 'lucide-react';

interface Passenger {
  name: string; nationality: string; flag: string; ethnicity: string; emoji: string;
  standardScore: number; fairScore: number;
}

const QUEUE: Passenger[] = [
  { name: 'Anna Müller', nationality: 'German', flag: '🇩🇪', ethnicity: 'White', emoji: '👩🏼', standardScore: 97, fairScore: 94 },
  { name: 'Emeka Okonkwo', nationality: 'Nigerian', flag: '🇳🇬', ethnicity: 'Black', emoji: '👨🏿', standardScore: 61, fairScore: 91 },
  { name: 'Aiko Yamamoto', nationality: 'Japanese', flag: '🇯🇵', ethnicity: 'Asian', emoji: '👩🏻', standardScore: 89, fairScore: 93 },
  { name: 'Mohammed Al-Rashid', nationality: 'Egyptian', flag: '🇪🇬', ethnicity: 'Arab', emoji: '👨🏽', standardScore: 52, fairScore: 90 },
  { name: 'María González', nationality: 'Mexican', flag: '🇲🇽', ethnicity: 'Hispanic', emoji: '👩🏽', standardScore: 74, fairScore: 92 },
  { name: 'Pierre Dubois', nationality: 'French', flag: '🇫🇷', ethnicity: 'White', emoji: '👨🏻', standardScore: 96, fairScore: 94 },
  { name: 'Li Wei', nationality: 'Chinese', flag: '🇨🇳', ethnicity: 'Asian', emoji: '👨🏻', standardScore: 79, fairScore: 92 },
  { name: 'Nomsa Dlamini', nationality: 'S. African', flag: '🇿🇦', ethnicity: 'Black', emoji: '👩🏿', standardScore: 58, fairScore: 91 },
];

const EU_ARTICLES = [
  { id: 'Art. 5(1)(b)', desc: 'Prohibition on real-time biometric categorisation in public spaces for law enforcement.' },
  { id: 'Art. 6', desc: 'E-ID verification systems fall under Annex III as high-risk AI — biometric ID & categorisation.' },
  { id: 'Art. 10(2)(f)', desc: 'Training data must be checked for biases that could lead to discriminatory results.' },
  { id: 'Art. 13', desc: 'Transparency and provision of information to deployers: capabilities and limitations.' },
  { id: 'Art. 14', desc: 'Human oversight measures — especially where automated decisions have legal effect.' },
];

export default function EIDPage() {
  const [modelType, setModelType] = useState<'standard' | 'fair'>('standard');
  const [selected, setSelected] = useState<Passenger | null>(null);

  const getScoreColor = (score: number) => {
    if (score >= 90) return 'text-observatory-success';
    if (score >= 70) return 'text-observatory-warning';
    return 'text-observatory-danger';
  };

  return (
    <div className="p-4 md:p-8 max-w-7xl mx-auto">
      <header className="mb-6">
        <h1 className="text-2xl font-mono font-bold gradient-text flex items-center gap-2">
          <ScanLine className="w-6 h-6" /> E-ID Verification Portal
        </h1>
        <p className="text-observatory-text-muted text-sm mt-1">AI-powered electronic identity verification — bias impact simulator</p>
      </header>

      {/* Model toggle */}
      <div className="glass rounded-xl p-4 mb-6">
        <div className="text-xs text-observatory-text-dim mb-2">AI Model:</div>
        <div className="flex gap-2">
          {(['standard', 'fair'] as const).map(t => (
            <button
              key={t}
              onClick={() => setModelType(t)}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                modelType === t ? 'bg-observatory-accent/15 text-observatory-accent' : 'text-observatory-text-muted hover:bg-observatory-surface-alt'
              }`}
            >
              {t === 'standard' ? 'Standard Biometric AI' : 'Fairness-Constrained AI'}
            </button>
          ))}
        </div>
        <p className="text-xs text-observatory-text-dim mt-2">
          {modelType === 'standard'
            ? 'Standard model shows demographic bias in clearance scores.'
            : 'Fairness-constrained model equalizes scores across demographics.'}
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Queue */}
        <div className="glass rounded-xl p-5">
          <h3 className="text-xs font-mono text-observatory-text-dim mb-3">Passenger Queue ({QUEUE.length})</h3>
          <div className="space-y-2">
            {QUEUE.map(p => {
              const score = modelType === 'standard' ? p.standardScore : p.fairScore;
              return (
                <button
                  key={p.name}
                  onClick={() => setSelected(p)}
                  className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-left transition-all ${
                    selected?.name === p.name ? 'bg-observatory-accent/15' : 'bg-observatory-bg/50 hover:bg-observatory-surface-alt'
                  }`}
                >
                  <span className="text-lg">{p.emoji}</span>
                  <div className="flex-1 min-w-0">
                    <div className="text-sm text-observatory-text truncate">{p.name}</div>
                    <div className="text-xs text-observatory-text-dim">{p.nationality} {p.flag}</div>
                  </div>
                  <div className="text-right">
                    <div className={`font-mono font-bold text-sm ${getScoreColor(score)}`}>{score}%</div>
                    <div className="text-[10px] text-observatory-text-dim">{p.ethnicity}</div>
                  </div>
                </button>
              );
            })}
          </div>
        </div>

        {/* Scan result */}
        <div className="glass rounded-xl p-5">
          {!selected ? (
            <div className="flex flex-col items-center justify-center h-full text-center py-12">
              <ScanLine className="w-12 h-12 text-observatory-text-dim mb-4" />
              <p className="text-observatory-text-muted text-sm">Select a passenger to scan their E-ID</p>
            </div>
          ) : (
            <div className="text-center">
              <div className="text-4xl mb-3">{selected.emoji}</div>
              <h3 className="text-lg font-semibold text-observatory-text">{selected.name}</h3>
              <p className="text-sm text-observatory-text-muted">{selected.nationality} {selected.flag} · {selected.ethnicity}</p>

              <div className="my-6">
                <div className="text-xs text-observatory-text-dim mb-1">CLEARANCE SCORE</div>
                <div className={`text-5xl font-mono font-bold ${getScoreColor(modelType === 'standard' ? selected.standardScore : selected.fairScore)}`}>
                  {modelType === 'standard' ? selected.standardScore : selected.fairScore}%
                </div>
              </div>

              <div className={`flex items-center justify-center gap-2 py-2 px-4 rounded-lg ${
                (modelType === 'standard' ? selected.standardScore : selected.fairScore) >= 70
                  ? 'bg-observatory-success/10 text-observatory-success'
                  : 'bg-observatory-danger/10 text-observatory-danger'
              }`}>
                {(modelType === 'standard' ? selected.standardScore : selected.fairScore) >= 70
                  ? <><CheckCircle className="w-4 h-4" /> CLEARED</>
                  : <><AlertTriangle className="w-4 h-4" /> FLAGGED FOR REVIEW</>}
              </div>

              {modelType === 'standard' && selected.standardScore < 70 && (
                <div className="mt-4 p-3 rounded-lg bg-observatory-danger/10 text-xs text-observatory-danger">
                  ⚠️ This score shows demographic bias. With fairness constraints, this passenger scores {selected.fairScore}%.
                </div>
              )}
            </div>
          )}
        </div>

        {/* EU Articles */}
        <div className="glass rounded-xl p-5">
          <h3 className="text-xs font-mono text-observatory-text-dim mb-3">Applicable EU AI Act Articles</h3>
          <div className="space-y-3">
            {EU_ARTICLES.map(a => (
              <div key={a.id} className="bg-observatory-bg/50 rounded-lg p-3">
                <div className="text-xs font-mono text-observatory-accent mb-1">{a.id}</div>
                <div className="text-xs text-observatory-text-muted">{a.desc}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
