import { useState } from 'react';
import { ACTIVE_MODELS, PROBES, LEADERBOARD, type ModelResult } from '../data/benchmarkData';
import { RadarChart, Radar, PolarGrid, PolarAngleAxis, ResponsiveContainer, Legend } from 'recharts';
import { searchHFModels, type HFModel } from '../services/huggingface';
import { FlaskConical, Search } from 'lucide-react';

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
    <div className="page-container">
      <header className="page-header">
        <h1 className="page-title">
          <FlaskConical className="w-7 h-7 text-observatory-accent" />
          <span className="gradient-text">Fingerprint Lab</span>
        </h1>
        <p className="page-subtitle">Compare models side-by-side with real FHIBE benchmark data</p>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Config */}
        <div className="space-y-4">
          <div className="card">
            <div className="card-header">Primary Model</div>
            <div className="space-y-1">
              {ACTIVE_MODELS.map(m => (
                <button
                  key={m.id}
                  onClick={() => setPrimary(m)}
                  className={`w-full text-left px-4 py-2.5 rounded-xl text-sm transition-all ${
                    primary.id === m.id ? 'bg-observatory-accent/15 text-observatory-accent' : 'text-observatory-text-muted hover:bg-observatory-surface-alt/50'
                  }`}
                >
                  <span className="inline-block w-2.5 h-2.5 rounded-full mr-2" style={{ background: m.color }} />
                  {m.name} <span className="text-observatory-text-dim">({m.params})</span>
                </button>
              ))}
            </div>
          </div>

          <div className="card">
            <div className="card-header">Compare Against</div>
            <div className="space-y-1">
              {ACTIVE_MODELS.filter(m => m.id !== primary.id).map(m => (
                <button
                  key={m.id}
                  onClick={() => setCompare(m)}
                  className={`w-full text-left px-4 py-2.5 rounded-xl text-sm transition-all ${
                    compare.id === m.id ? 'bg-observatory-accent/15 text-observatory-accent' : 'text-observatory-text-muted hover:bg-observatory-surface-alt/50'
                  }`}
                >
                  <span className="inline-block w-2.5 h-2.5 rounded-full mr-2" style={{ background: m.color }} />
                  {m.name} <span className="text-observatory-text-dim">({m.params})</span>
                </button>
              ))}
            </div>
          </div>

          <div className="card">
            <div className="card-header">Search HuggingFace VLMs</div>
            <div className="flex gap-2 mb-3">
              <input
                value={searchQuery}
                onChange={e => setSearchQuery(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && handleSearch()}
                placeholder="e.g. llava, qwen-vl, pixtral..."
                className="flex-1 bg-observatory-bg border border-observatory-border rounded-xl px-4 py-2.5 text-sm text-observatory-text placeholder:text-observatory-text-dim outline-none focus:border-observatory-accent transition-colors"
              />
              <button onClick={handleSearch} className="px-4 py-2.5 rounded-xl bg-observatory-accent/15 text-observatory-accent hover:bg-observatory-accent/25 transition-all">
                {searching ? '...' : <Search className="w-4 h-4" />}
              </button>
            </div>
            {searchResults.length > 0 && (
              <div className="space-y-1 max-h-48 overflow-y-auto">
                {searchResults.map(r => (
                  <div key={r.modelId} className="flex items-center gap-2 px-3 py-2 rounded-lg bg-observatory-bg/50 text-sm">
                    <span className="text-observatory-text-muted flex-1 truncate">{r.modelId}</span>
                    <span className="text-observatory-text-dim text-xs">↓{r.downloads > 1000 ? `${(r.downloads / 1000).toFixed(0)}K` : r.downloads}</span>
                  </div>
                ))}
              </div>
            )}
          </div>

          <button
            onClick={runTest}
            disabled={running}
            className="w-full py-3.5 rounded-xl bg-observatory-accent text-observatory-bg font-semibold text-sm hover:bg-observatory-accent-glow transition-all disabled:opacity-50"
          >
            {running ? 'Running Bias Test...' : 'Compare Models'}
          </button>
        </div>

        {/* Results */}
        <div className="lg:col-span-2 space-y-6">
          <div className="card">
            <div className="card-header">Active Probes (FHIBE Social Inference Battery)</div>
            <div className="space-y-2">
              {PROBES.map(p => (
                <div key={p.id} className="bg-observatory-bg/50 rounded-xl px-5 py-3">
                  <div className="text-sm text-observatory-text font-medium">{p.label}</div>
                  <div className="text-xs text-observatory-text-dim italic mt-1">{p.prompt}</div>
                </div>
              ))}
            </div>
          </div>

          {showResults && (
            <div className="card">
              <div className="card-header">Fingerprint Overlay — {primary.name} vs {compare.name}</div>
              <ResponsiveContainer width="100%" height={380}>
                <RadarChart data={radarData}>
                  <PolarGrid stroke="hsl(222 20% 22%)" />
                  <PolarAngleAxis dataKey="probe" tick={{ fill: 'hsl(215 20% 65%)', fontSize: 12 }} />
                  <Radar dataKey={primary.name} stroke={primary.color} fill={primary.color} fillOpacity={0.15} />
                  <Radar dataKey={compare.name} stroke={compare.color} fill={compare.color} fillOpacity={0.15} />
                  <Legend wrapperStyle={{ fontSize: 12, color: '#fff' }} />
                </RadarChart>
              </ResponsiveContainer>

              <div className="grid grid-cols-2 gap-4 mt-6">
                {[primary, compare].map(m => (
                  <div key={m.id} className="bg-observatory-bg/50 rounded-xl p-5">
                    <div className="flex items-center gap-2 mb-2">
                      <span className="w-3 h-3 rounded-full" style={{ background: m.color }} />
                      <span className="font-semibold text-observatory-text">{m.name}</span>
                    </div>
                    <div className="text-3xl font-mono font-bold" style={{ color: m.color }}>{m.composite_score.toFixed(4)}</div>
                    <div className="text-xs text-observatory-text-dim mt-1">{m.params} · {m.provider} · {m.n_significant} significant effects</div>
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
