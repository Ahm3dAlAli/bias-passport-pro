import { useState, useEffect, useRef } from 'react';
import { ACTIVE_MODELS } from '../data/benchmarkData';
import { Plane, Play, RotateCcw, Camera } from 'lucide-react';

interface Passenger {
  id: number;
  name: string;
  origin: string;
  flag: string;
  ethnicity: string;
  emoji: string;
  flagged: boolean;
  score: number;
  processed: boolean;
}

const PASSENGER_POOL: Omit<Passenger, 'id' | 'flagged' | 'score' | 'processed'>[] = [
  { name: 'Anna Müller', origin: 'Germany', flag: '🇩🇪', ethnicity: 'White', emoji: '👩🏼' },
  { name: 'Emeka Okonkwo', origin: 'Nigeria', flag: '🇳🇬', ethnicity: 'Black', emoji: '👨🏿' },
  { name: 'Aiko Yamamoto', origin: 'Japan', flag: '🇯🇵', ethnicity: 'Asian', emoji: '👩🏻' },
  { name: 'Mohammed Al-Rashid', origin: 'Egypt', flag: '🇪🇬', ethnicity: 'Arab', emoji: '👨🏽' },
  { name: 'María González', origin: 'Mexico', flag: '🇲🇽', ethnicity: 'Hispanic', emoji: '👩🏽' },
  { name: 'Pierre Dubois', origin: 'France', flag: '🇫🇷', ethnicity: 'White', emoji: '👨🏻' },
  { name: 'Li Wei', origin: 'China', flag: '🇨🇳', ethnicity: 'Asian', emoji: '👨🏻' },
  { name: 'Nomsa Dlamini', origin: 'S. Africa', flag: '🇿🇦', ethnicity: 'Black', emoji: '👩🏿' },
  { name: 'Raj Patel', origin: 'India', flag: '🇮🇳', ethnicity: 'South Asian', emoji: '👨🏽' },
  { name: 'Sofia Petrov', origin: 'Russia', flag: '🇷🇺', ethnicity: 'White', emoji: '👩🏻' },
];

// Biased model: higher flag rates for certain ethnicities
const BIAS_RATES: Record<string, number> = {
  White: 0.08, Black: 0.35, Asian: 0.12, Arab: 0.42, Hispanic: 0.28, 'South Asian': 0.30,
};
const FAIR_RATE = 0.12;

export default function AirportPage() {
  const [selectedModel, setSelectedModel] = useState(ACTIVE_MODELS[0]);
  const [passengers, setPassengers] = useState<Passenger[]>([]);
  const [running, setRunning] = useState(false);
  const [log, setLog] = useState<string[]>([]);
  const intervalRef = useRef<number | null>(null);

  const processed = passengers.filter(p => p.processed);
  const flagged = processed.filter(p => p.flagged);
  const flagRate = processed.length > 0 ? flagged.length / processed.length : 0;

  const flagByEthnicity = () => {
    const groups: Record<string, { total: number; flagged: number }> = {};
    processed.forEach(p => {
      if (!groups[p.ethnicity]) groups[p.ethnicity] = { total: 0, flagged: 0 };
      groups[p.ethnicity].total++;
      if (p.flagged) groups[p.ethnicity].flagged++;
    });
    return groups;
  };

  const start = () => {
    setRunning(true);
    setPassengers([]);
    setLog([]);
    let idx = 0;

    intervalRef.current = window.setInterval(() => {
      if (idx >= PASSENGER_POOL.length) {
        setRunning(false);
        if (intervalRef.current) clearInterval(intervalRef.current);
        return;
      }
      const pool = PASSENGER_POOL[idx];
      const biasRate = BIAS_RATES[pool.ethnicity] ?? 0.15;
      const flagged = Math.random() < biasRate;
      const score = flagged ? 0.3 + Math.random() * 0.3 : 0.7 + Math.random() * 0.25;

      const p: Passenger = { ...pool, id: idx, flagged, score, processed: true };
      setPassengers(prev => [...prev, p]);
      setLog(prev => [...prev, `${pool.emoji} ${pool.name} (${pool.origin}) — ${flagged ? '🚨 FLAGGED' : '✅ CLEAR'} (${(score * 100).toFixed(0)}%)`]);
      idx++;
    }, 800);
  };

  const reset = () => {
    setRunning(false);
    if (intervalRef.current) clearInterval(intervalRef.current);
    setPassengers([]);
    setLog([]);
  };

  useEffect(() => () => { if (intervalRef.current) clearInterval(intervalRef.current); }, []);

  const ethStats = flagByEthnicity();
  const maxDisparity = (() => {
    const rates = Object.values(ethStats).filter(g => g.total > 0).map(g => g.flagged / g.total);
    return rates.length > 1 ? Math.max(...rates) - Math.min(...rates) : 0;
  })();

  return (
    <div className="p-4 md:p-8 max-w-7xl mx-auto">
      <header className="mb-6">
        <h1 className="text-2xl font-mono font-bold gradient-text flex items-center gap-2">
          <Plane className="w-6 h-6" /> AI Airport Security — Live Simulation
        </h1>
        <p className="text-observatory-text-muted text-sm mt-1">Real-time bias detection in airport screening AI</p>
      </header>

      {/* Controls */}
      <div className="glass rounded-xl p-4 mb-6 flex flex-wrap items-center gap-3">
        <select
          value={selectedModel.id}
          onChange={e => setSelectedModel(ACTIVE_MODELS.find(m => m.id === e.target.value) || ACTIVE_MODELS[0])}
          className="bg-observatory-bg border border-observatory-border rounded-lg px-3 py-2 text-sm text-observatory-text"
        >
          {ACTIVE_MODELS.map(m => (
            <option key={m.id} value={m.id}>{m.name} ({m.params})</option>
          ))}
        </select>
        <button onClick={start} disabled={running} className="flex items-center gap-2 px-4 py-2 rounded-lg bg-observatory-success/20 text-observatory-success text-sm hover:bg-observatory-success/30 disabled:opacity-50">
          <Play className="w-4 h-4" /> Start
        </button>
        <button onClick={reset} className="flex items-center gap-2 px-4 py-2 rounded-lg bg-observatory-danger/20 text-observatory-danger text-sm hover:bg-observatory-danger/30">
          <RotateCcw className="w-4 h-4" /> Reset
        </button>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-4 gap-3 mb-6">
        {[
          { label: 'PROCESSED', value: processed.length, color: 'text-observatory-text' },
          { label: 'FLAGGED', value: flagged.length, color: 'text-observatory-danger' },
          { label: 'FLAG RATE', value: `${(flagRate * 100).toFixed(1)}%`, color: flagRate > 0.2 ? 'text-observatory-danger' : 'text-observatory-success' },
          { label: 'DISPARITY', value: `${(maxDisparity * 100).toFixed(0)}%`, color: maxDisparity > 0.15 ? 'text-observatory-danger' : 'text-observatory-success' },
        ].map(s => (
          <div key={s.label} className="glass rounded-xl p-4 text-center">
            <div className="text-[10px] text-observatory-text-dim mb-1">{s.label}</div>
            <div className={`text-xl font-mono font-bold ${s.color}`}>{s.value}</div>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Processing log */}
        <div className="glass rounded-xl p-5">
          <h3 className="text-xs font-mono text-observatory-text-dim mb-3">Processing Log — {log.length} passengers</h3>
          <div className="space-y-1 max-h-64 overflow-y-auto">
            {log.length === 0 && <p className="text-observatory-text-dim text-sm">Press Start to begin simulation</p>}
            {log.map((l, i) => (
              <div key={i} className="text-xs font-mono text-observatory-text-muted py-1 border-b border-observatory-border/30">{l}</div>
            ))}
          </div>
        </div>

        {/* Flag rate by ethnicity */}
        <div className="glass rounded-xl p-5">
          <h3 className="text-xs font-mono text-observatory-text-dim mb-3">Flag Rate by Ethnicity</h3>
          {Object.keys(ethStats).length === 0 && <p className="text-observatory-text-dim text-sm">Run simulation to see stats</p>}
          <div className="space-y-2">
            {Object.entries(ethStats).map(([eth, data]) => {
              const rate = data.total > 0 ? data.flagged / data.total : 0;
              return (
                <div key={eth}>
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-observatory-text-muted">{eth}</span>
                    <span className={`font-mono ${rate > 0.25 ? 'text-observatory-danger' : 'text-observatory-success'}`}>{(rate * 100).toFixed(0)}%</span>
                  </div>
                  <div className="h-2 bg-observatory-bg rounded-full overflow-hidden">
                    <div className={`h-full rounded-full ${rate > 0.25 ? 'bg-observatory-danger' : 'bg-observatory-success'}`} style={{ width: `${rate * 100}%` }} />
                  </div>
                </div>
              );
            })}
          </div>

          {processed.length > 0 && (
            <div className="mt-4 p-3 rounded-lg bg-observatory-bg/50">
              <div className="text-xs font-mono text-observatory-text-dim">
                EU AI Act — Art. 10 Compliance
              </div>
              <div className={`text-xs mt-1 ${maxDisparity > 0.15 ? 'text-observatory-danger' : 'text-observatory-success'}`}>
                {maxDisparity > 0.15
                  ? `⚠️ ${(maxDisparity * 100).toFixed(0)}% disparity gap — exceeds acceptable threshold. Potential Art. 10(2)(f) violation.`
                  : `✅ ${(maxDisparity * 100).toFixed(0)}% disparity gap — within acceptable parameters.`}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
