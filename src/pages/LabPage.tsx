import { useState } from 'react';
import { ACTIVE_MODELS, PROBES, LEADERBOARD, type ModelResult } from '../data/benchmarkData';
import { RadarChart, Radar, PolarGrid, PolarAngleAxis, ResponsiveContainer, Legend } from 'recharts';
import { searchHFModels, type HFModel } from '../services/huggingface';
import { FlaskConical, Search, Plus, X } from 'lucide-react';

export default function LabPage() {
  const [primary, setPrimary] = useState(LEADERBOARD[0]);
  const [compare, setCompare] = useState(LEADERBOARD[LEADERBOARD.length - 1]);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<HFModel[]>([]);
  const [searching, setSearching] = useState(false);
  const [running, setRunning] = useState(false);
  const [showResults, setShowResults] = useState(false);

  const handleSearch = async () => {
    if (!searchQuery.trim()) return;
    setSearching(true);
    const results = await searchHFModels(searchQuery);
    setSearchResults(results);
    setSearching(false);
  };

  const runTest = () => {
    setRunning(true);
    setTimeout(() => { setRunning(false); setShowResults(true); }, 2000);
  };

  const radarData = PROBES.map(p => {
    const entry: Record<string, any> = { probe: p.label.split(' · ')[1] };
    entry[primary.name] = (primary.dimensions[p.id]?.disparity ?? 0) * 100;
    entry[compare.name] = (compare.dimensions[p.id]?.disparity ?? 0) * 100;
    return entry;
  });

  return (
    <div className="p-4 md:p-8 max-w-7xl mx-auto">
      <header className="mb-6">
        <h1 className="text-2xl font-mono font-bold gradient-text flex items-center gap-2">
          <FlaskConical className="w-6 h-6" /> Bias Fingerprint Lab
        </h1>
        <p className="text-observatory-text-muted text-sm mt-1">
          Real FHIBE benchmark results · Compare models side-by-side
        </p>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Config panel */}
        <div className="space-y-4">
          <div className="glass rounded-xl p-5">
            <h3 className="text-xs font-mono text-observatory-text-dim mb-3">Primary Model</h3>
            <div className="space-y-1">
              {ACTIVE_MODELS.map(m => (
                <button
                  key={m.id}
                  onClick={() => setPrimary(m)}
                  className={`w-full text-left px-3 py-2 rounded-lg text-xs transition-all ${
                    primary.id === m.id ? 'bg-observatory-accent/15 text-observatory-accent' : 'text-observatory-text-muted hover:bg-observatory-surface-alt'
                  }`}
                >
                  <span className="inline-block w-2 h-2 rounded-full mr-2" style={{ background: m.color }} />
                  {m.name} ({m.params})
                </button>
              ))}
            </div>
          </div>

          <div className="glass rounded-xl p-5">
            <h3 className="text-xs font-mono text-observatory-text-dim mb-3">Compare Against</h3>
            <div className="space-y-1">
              {ACTIVE_MODELS.filter(m => m.id !== primary.id).map(m => (
                <button
                  key={m.id}
                  onClick={() => setCompare(m)}
                  className={`w-full text-left px-3 py-2 rounded-lg text-xs transition-all ${
                    compare.id === m.id ? 'bg-observatory-accent/15 text-observatory-accent' : 'text-observatory-text-muted hover:bg-observatory-surface-alt'
                  }`}
                >
                  <span className="inline-block w-2 h-2 rounded-full mr-2" style={{ background: m.color }} />
                  {m.name} ({m.params})
                </button>
              ))}
            </div>
          </div>

          {/* HF Search */}
          <div className="glass rounded-xl p-5">
            <h3 className="text-xs font-mono text-observatory-text-dim mb-3">Search HuggingFace VLMs</h3>
            <div className="flex gap-2 mb-3">
              <input
                value={searchQuery}
                onChange={e => setSearchQuery(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && handleSearch()}
                placeholder="e.g. llava, qwen-vl, pixtral..."
                className="flex-1 bg-observatory-bg border border-observatory-border rounded-lg px-3 py-2 text-xs text-observatory-text placeholder:text-observatory-text-dim outline-none focus:border-observatory-accent"
              />
              <button onClick={handleSearch} className="px-3 py-2 rounded-lg bg-observatory-accent/20 text-observatory-accent text-xs hover:bg-observatory-accent/30">
                {searching ? '...' : <Search className="w-3 h-3" />}
              </button>
            </div>
            {searchResults.length > 0 && (
              <div className="space-y-1 max-h-48 overflow-y-auto">
                {searchResults.map(r => (
                  <div key={r.modelId} className="flex items-center gap-2 px-2 py-1.5 rounded bg-observatory-bg/50 text-xs">
                    <span className="text-observatory-text-muted flex-1 truncate">{r.modelId}</span>
                    <span className="text-observatory-text-dim">↓{r.downloads > 1000 ? `${(r.downloads / 1000).toFixed(0)}K` : r.downloads}</span>
                  </div>
                ))}
              </div>
            )}
          </div>

          <button
            onClick={runTest}
            disabled={running}
            className="w-full py-3 rounded-xl bg-observatory-accent text-observatory-bg font-mono font-bold text-sm hover:bg-observatory-accent-glow transition-all disabled:opacity-50"
          >
            {running ? 'Running Bias Test...' : 'Compare Models'}
          </button>
        </div>

        {/* Results */}
        <div className="lg:col-span-2 space-y-6">
          {/* Active probes */}
          <div className="glass rounded-xl p-5">
            <h3 className="text-xs font-mono text-observatory-text-dim mb-3">Active Probes (FHIBE Social Inference Battery)</h3>
            <div className="space-y-2">
              {PROBES.map(p => (
                <div key={p.id} className="bg-observatory-bg/50 rounded-lg px-4 py-2">
                  <div className="text-sm text-observatory-text font-medium">{p.label}</div>
                  <div className="text-xs text-observatory-text-dim italic">{p.prompt}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Radar overlay */}
          {showResults && (
            <div className="glass rounded-xl p-6">
              <h3 className="text-sm font-mono text-observatory-text-muted mb-4">
                Fingerprint Overlay — {primary.name} vs {compare.name}
              </h3>
              <ResponsiveContainer width="100%" height={350}>
                <RadarChart data={radarData}>
                  <PolarGrid stroke="hsl(222 20% 25%)" />
                  <PolarAngleAxis dataKey="probe" tick={{ fill: 'hsl(215 20% 65%)', fontSize: 11 }} />
                  <Radar dataKey={primary.name} stroke={primary.color} fill={primary.color} fillOpacity={0.15} />
                  <Radar dataKey={compare.name} stroke={compare.color} fill={compare.color} fillOpacity={0.15} />
                  <Legend wrapperStyle={{ fontSize: 11, color: '#fff' }} />
                </RadarChart>
              </ResponsiveContainer>

              {/* Score comparison */}
              <div className="grid grid-cols-2 gap-4 mt-4">
                {[primary, compare].map(m => (
                  <div key={m.id} className="bg-observatory-bg/50 rounded-lg p-4">
                    <div className="flex items-center gap-2 mb-2">
                      <span className="w-3 h-3 rounded-full" style={{ background: m.color }} />
                      <span className="text-sm font-medium text-observatory-text">{m.name}</span>
                    </div>
                    <div className="text-2xl font-mono font-bold" style={{ color: m.color }}>{m.composite_score.toFixed(4)}</div>
                    <div className="text-xs text-observatory-text-dim">{m.params} · {m.provider} · {m.n_significant} significant effects</div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
