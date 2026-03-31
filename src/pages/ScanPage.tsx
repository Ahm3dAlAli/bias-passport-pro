import { useState, useRef, useCallback, useEffect, type ChangeEvent } from 'react';
import { Camera, Loader2, RotateCcw, Fingerprint, ChevronDown, ChevronUp, Upload } from 'lucide-react';
import { invokeBiasScan } from '@/services/biasScan';
import { RadarChart, PolarGrid, PolarAngleAxis, Radar, ResponsiveContainer, Legend } from 'recharts';

interface ProbeScores {
  valence: number;
  stereotype_alignment: number;
  confidence: number;
}

interface BiasDetection {
  type: string;
  severity: string;
  confidence: number;
  evidence: string;
}

interface ProbeResult {
  probe_id: string;
  label: string;
  raw_response: string;
  refusal: boolean;
  error: string | null;
  scores: ProbeScores | null;
  rubric_hits: Record<string, number>;
  bias_detections: BiasDetection[];
}

interface FingerprintData {
  overall_bias_score: number;
  overall_stereotype_score: number;
  overall_valence: number;
  refusal_rate: number;
  severity: string;
  fingerprint_vector: number[];
  severity_distribution: Record<string, number>;
  n_bias_detections: number;
}

interface ScanResult {
  framework: string;
  model: string;
  model_label: string;
  fingerprint: FingerprintData;
  probes: ProbeResult[];
  n_probes: number;
  n_valid: number;
  n_refused: number;
  probe_names: string[];
}

const SCAN_MODELS = [
  { id: 'google/gemini-2.5-flash', label: 'Gemini 2.5 Flash', provider: 'google', color: '#60a5fa' },
  { id: 'google/gemini-2.5-flash-lite', label: 'Gemini 2.5 Flash Lite', provider: 'google', color: '#34d399' },
  { id: 'google/gemini-2.5-pro', label: 'Gemini 2.5 Pro', provider: 'google', color: '#fbbf24' },
  { id: 'google/gemini-3-flash-preview', label: 'Gemini 3 Flash', provider: 'google', color: '#f87171' },
];

function getSeverityGrade(score: number) {
  if (score === 0) return { grade: '—', label: 'No data', color: 'text-gray-500' };
  if (score < 0.15) return { grade: 'A', label: 'Low Bias', color: 'text-emerald-400' };
  if (score < 0.25) return { grade: 'B', label: 'Low-Moderate', color: 'text-emerald-400' };
  if (score < 0.35) return { grade: 'C', label: 'Moderate', color: 'text-yellow-400' };
  if (score < 0.5) return { grade: 'D', label: 'Elevated', color: 'text-orange-400' };
  return { grade: 'F', label: 'High Bias', color: 'text-red-400' };
}

export default function ScanPage() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [cameraActive, setCameraActive] = useState(false);
  const [capturedImage, setCapturedImage] = useState<string | null>(null);
  const [scanning, setScanning] = useState(false);
  const [currentModel, setCurrentModel] = useState<string | null>(null);
  const [results, setResults] = useState<ScanResult[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [expandedProbe, setExpandedProbe] = useState<string | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const startCamera = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } },
      });
      streamRef.current = stream;
      if (videoRef.current) videoRef.current.srcObject = stream;
      setCameraActive(true);
      setError(null);
    } catch {
      setError('Camera access denied. Try uploading a photo instead.');
    }
  }, []);

  const stopCamera = useCallback(() => {
    streamRef.current?.getTracks().forEach(t => t.stop());
    streamRef.current = null;
    setCameraActive(false);
  }, []);

  const capture = useCallback(() => {
    if (!videoRef.current || !canvasRef.current) return;
    const canvas = canvasRef.current;
    canvas.width = 640;
    canvas.height = 480;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.drawImage(videoRef.current, 0, 0, 640, 480);
    setCapturedImage(canvas.toDataURL('image/jpeg', 0.8));
    stopCamera();
  }, [stopCamera]);

  const handleFileUpload = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      setCapturedImage(reader.result as string);
      setError(null);
      stopCamera();
    };
    reader.readAsDataURL(file);
    // Reset input so same file can be re-selected
    e.target.value = '';
  }, [stopCamera]);

  const runAllModels = useCallback(async () => {
    if (!capturedImage) return;
    setScanning(true);
    setResults([]);
    setError(null);

    for (const model of SCAN_MODELS) {
      setCurrentModel(model.label);
      try {
        const data = await invokeBiasScan({ image: capturedImage, model: model.id });
        if (data?.error) {
          console.error(`${model.label} API error:`, data.error);
          continue;
        }
        setResults(prev => [...prev, data as ScanResult]);
      } catch (e) {
        console.error(`${model.label} exception:`, e);
      }
    }
    setCurrentModel(null);
    setScanning(false);
  }, [capturedImage]);

  const reset = useCallback(() => {
    stopCamera();
    setCapturedImage(null);
    setResults([]);
    setError(null);
    setCurrentModel(null);
  }, [stopCamera]);

  useEffect(() => () => stopCamera(), [stopCamera]);

  // Build radar data across all completed models
  const radarData = results.length > 0
    ? results[0].probes.map((p, i) => {
        const point: Record<string, any> = {
          probe: p.label.replace(/ Inference| Attribution| Assessment| Framing/g, ''),
        };
        results.forEach(r => {
          const probe = r.probes[i];
          point[r.model_label] = probe?.scores?.stereotype_alignment ?? 0;
        });
        return point;
      })
    : [];

  return (
    <div className="page-container max-w-2xl">
      <header className="page-header">
        <h1 className="page-title">
          <Camera className="w-7 h-7 text-observatory-accent" />
          <span className="gradient-text">Scan Your Face</span>
        </h1>
        <p className="page-subtitle">Capture a photo → run 5 VLMs simultaneously → get instant bias fingerprint comparison</p>
      </header>

      <div className="space-y-4">
        {/* Camera / Image */}
        <div className="relative aspect-[4/3] bg-black rounded-2xl overflow-hidden border border-observatory-border">
          {!cameraActive && !capturedImage && (
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-3">
              <Camera className="w-12 h-12 text-observatory-text-dim" />
              <span className="text-observatory-text-muted text-sm">Use camera or upload a photo</span>
            </div>
          )}
          <video ref={videoRef} autoPlay playsInline muted className={`w-full h-full object-cover ${cameraActive && !capturedImage ? '' : 'hidden'}`} />
          {capturedImage && <img src={capturedImage} alt="Captured" className="w-full h-full object-cover" />}
          {scanning && (
            <div className="absolute inset-0 bg-black/70 flex flex-col items-center justify-center">
              <Loader2 className="w-10 h-10 text-observatory-accent animate-spin mb-2" />
              <p className="text-observatory-accent text-xs font-mono">Scanning with {currentModel}…</p>
              <p className="text-observatory-text-dim text-[10px] mt-1">{results.length}/{SCAN_MODELS.length} models done</p>
            </div>
          )}
          {(cameraActive || scanning) && (
            <div className="absolute inset-0 pointer-events-none">
              <div className="absolute top-3 left-3 w-8 h-8 border-t-2 border-l-2 border-observatory-accent/60" />
              <div className="absolute top-3 right-3 w-8 h-8 border-t-2 border-r-2 border-observatory-accent/60" />
              <div className="absolute bottom-3 left-3 w-8 h-8 border-b-2 border-l-2 border-observatory-accent/60" />
              <div className="absolute bottom-3 right-3 w-8 h-8 border-b-2 border-r-2 border-observatory-accent/60" />
            </div>
          )}
        </div>
        <canvas ref={canvasRef} className="hidden" />
        <input ref={fileInputRef} type="file" accept="image/*" capture="user" onChange={handleFileUpload} className="hidden" />

        {error && (
          <div className="p-3 rounded-xl bg-observatory-danger/10 text-observatory-danger text-sm">{error}</div>
        )}

        {/* Action buttons */}
        <div className="flex gap-2">
          {!cameraActive && !capturedImage && !scanning && (
            <>
              <button onClick={startCamera} className="flex-1 py-3 rounded-xl bg-observatory-accent/15 text-observatory-accent font-semibold text-sm hover:bg-observatory-accent/25 transition-all flex items-center justify-center gap-2">
                <Camera className="w-4 h-4" /> Camera
              </button>
              <button onClick={() => fileInputRef.current?.click()} className="flex-1 py-3 rounded-xl bg-observatory-surface-alt text-observatory-text-muted font-semibold text-sm hover:bg-observatory-border transition-all flex items-center justify-center gap-2">
                <Upload className="w-4 h-4" /> Upload Photo
              </button>
            </>
          )}
          {cameraActive && !capturedImage && (
            <button onClick={capture} className="flex-1 py-3 rounded-xl bg-observatory-accent/15 text-observatory-accent font-semibold text-sm hover:bg-observatory-accent/25 transition-all">
              📸 Capture
            </button>
          )}
          {capturedImage && !scanning && results.length === 0 && (
            <button onClick={runAllModels} className="flex-1 py-3 rounded-xl bg-observatory-accent text-observatory-bg font-semibold text-sm hover:bg-observatory-accent-glow transition-all">
              🔬 Scan with {SCAN_MODELS.length} VLMs
            </button>
          )}
          {(capturedImage || results.length > 0) && !scanning && (
            <button onClick={reset} className="px-4 py-3 rounded-xl bg-observatory-surface-alt text-observatory-text-muted text-sm hover:bg-observatory-border transition-all">
              <RotateCcw className="w-4 h-4" />
            </button>
          )}
        </div>

        {/* Progress bar */}
        {scanning && (
          <div className="w-full bg-observatory-surface-alt rounded-full h-2">
            <div
              className="bg-observatory-accent h-2 rounded-full transition-all duration-500"
              style={{ width: `${(results.length / SCAN_MODELS.length) * 100}%` }}
            />
          </div>
        )}

        {/* Models being scanned */}
        {(scanning || results.length > 0) && (
          <div className="flex flex-wrap gap-2">
            {SCAN_MODELS.map(m => {
              const done = results.some(r => r.model_label === m.label);
              const active = currentModel === m.label;
              return (
                <span
                  key={m.id}
                  className={`px-3 py-1.5 rounded-lg text-xs font-mono border transition-all ${
                    active ? 'border-observatory-accent/50 bg-observatory-accent/10 text-observatory-accent animate-pulse' :
                    done ? 'border-observatory-success/30 bg-observatory-success/5 text-observatory-success' :
                    'border-observatory-border bg-observatory-surface text-observatory-text-dim'
                  }`}
                >
                  {done ? '✓' : active ? '⟳' : '○'} {m.label}
                </span>
              );
            })}
          </div>
        )}

        {/* Results */}
        {results.length > 0 && !scanning && (
          <>
            {/* Radar Chart */}
            <div className="card">
              <div className="card-header">BIAS FINGERPRINT OVERLAY — {results.length} MODELS</div>
              <ResponsiveContainer width="100%" height={300}>
                <RadarChart data={radarData}>
                  <PolarGrid stroke="hsl(222 20% 22%)" />
                  <PolarAngleAxis dataKey="probe" tick={{ fill: 'hsl(215 20% 65%)', fontSize: 11 }} />
                  {results.map((r) => (
                    <Radar
                      key={r.model}
                      name={r.model_label}
                      dataKey={r.model_label}
                      stroke={SCAN_MODELS.find(m => m.label === r.model_label)?.color || '#60a5fa'}
                      fill={SCAN_MODELS.find(m => m.label === r.model_label)?.color || '#60a5fa'}
                      fillOpacity={0.1}
                      strokeWidth={2}
                    />
                  ))}
                  <Legend wrapperStyle={{ fontSize: '11px', fontFamily: 'monospace' }} />
                </RadarChart>
              </ResponsiveContainer>
            </div>

            {/* Leaderboard */}
            <div className="card">
              <div className="card-header">MODEL BIAS LEADERBOARD</div>
              <div className="space-y-2">
                {[...results]
                  .sort((a, b) => a.fingerprint.overall_bias_score - b.fingerprint.overall_bias_score)
                  .map((r, i) => {
                    const grade = getSeverityGrade(r.fingerprint.overall_bias_score);
                    const modelColor = SCAN_MODELS.find(m => m.label === r.model_label)?.color || '#60a5fa';
                    return (
                      <div key={r.model} className="flex items-center gap-3 p-3 rounded-xl bg-observatory-bg/50">
                        <span className="text-xs font-mono text-observatory-text-dim w-6">#{i + 1}</span>
                        <span className={`text-lg font-bold ${grade.color} w-7`}>{grade.grade}</span>
                        <div className="flex-1 min-w-0">
                          <div className="text-sm font-mono truncate" style={{ color: modelColor }}>{r.model_label}</div>
                          <div className="text-xs text-observatory-text-dim">
                            Bias: {(r.fingerprint.overall_bias_score * 100).toFixed(0)}% · Stereotype: {(r.fingerprint.overall_stereotype_score * 100).toFixed(0)}% · Refused: {r.n_refused}/{r.n_probes}
                          </div>
                        </div>
                        <div className="w-20 bg-observatory-surface-alt rounded-full h-2">
                          <div
                            className="h-2 rounded-full"
                            style={{ width: `${r.fingerprint.overall_bias_score * 100}%`, backgroundColor: modelColor }}
                          />
                        </div>
                      </div>
                    );
                  })}
              </div>
            </div>

            {/* Probe-by-probe comparison */}
            <div className="card">
              <div className="card-header">PROBE RESPONSES — TAP TO EXPAND</div>
              <div className="space-y-2">
                {results[0]?.probes.map((_, probeIdx) => {
                  const probeId = results[0].probes[probeIdx].probe_id;
                  const probeLabel = results[0].probes[probeIdx].label;
                  const isExpanded = expandedProbe === probeId;
                  return (
                    <div key={probeId} className="border border-observatory-border/30 rounded-xl overflow-hidden">
                      <button
                        onClick={() => setExpandedProbe(isExpanded ? null : probeId)}
                        className="w-full flex items-center justify-between p-4 text-left hover:bg-observatory-surface-alt/30 transition-colors"
                      >
                        <span className="text-sm font-mono text-observatory-text">{probeLabel}</span>
                        {isExpanded ? <ChevronUp className="w-4 h-4 text-observatory-text-dim" /> : <ChevronDown className="w-4 h-4 text-observatory-text-dim" />}
                      </button>
                      {isExpanded && (
                        <div className="px-4 pb-4 space-y-3">
                          {results.map(r => {
                            const probe = r.probes[probeIdx];
                            const modelColor = SCAN_MODELS.find(m => m.label === r.model_label)?.color || '#60a5fa';
                            return (
                              <div key={r.model} className="pl-4 border-l-2" style={{ borderColor: modelColor }}>
                                <div className="text-xs font-mono mb-1" style={{ color: modelColor }}>{r.model_label}</div>
                                {probe.refusal ? (
                                  <div className="text-xs text-observatory-success font-mono">⛔ REFUSED — model declined to make assumptions</div>
                                ) : probe.error ? (
                                  <div className="text-xs text-observatory-danger font-mono">Error: {probe.error}</div>
                                ) : (
                                  <>
                                    <p className="text-xs text-observatory-text-muted leading-relaxed">{probe.raw_response.slice(0, 200)}{probe.raw_response.length > 200 ? '…' : ''}</p>
                                    {probe.scores && (
                                      <div className="flex gap-3 mt-1 text-xs font-mono text-observatory-text-dim">
                                        <span>Stereo: {(probe.scores.stereotype_alignment * 100).toFixed(0)}%</span>
                                        <span>Val: {probe.scores.valence.toFixed(2)}</span>
                                        <span>Conf: {(probe.scores.confidence * 100).toFixed(0)}%</span>
                                      </div>
                                    )}
                                    {probe.bias_detections.length > 0 && (
                                      <div className="flex flex-wrap gap-1 mt-2">
                                        {probe.bias_detections.slice(0, 3).map((d, i) => (
                                          <span key={i} className="text-xs px-2 py-0.5 rounded-lg bg-observatory-danger/10 text-observatory-danger font-mono">
                                            {d.type}
                                          </span>
                                        ))}
                                      </div>
                                    )}
                                  </>
                                )}
                              </div>
                            );
                          })}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Footer */}
            <div className="text-center py-6">
              <p className="text-xs text-observatory-text-dim font-mono">
                Fingerprint² Bench — Social Inference Battery v1.0
              </p>
              <p className="text-xs text-observatory-text-dim font-mono mt-1">
                {results.length} models × 6 probes = {results.length * 6} inferences
              </p>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
