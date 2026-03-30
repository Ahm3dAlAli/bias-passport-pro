import { useState, useRef, useCallback, useEffect } from 'react';
import { Camera, Loader2, RotateCcw, Fingerprint, ChevronDown, ChevronUp } from 'lucide-react';
import { supabase } from '@/integrations/supabase/client';
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
  { id: 'google/gemini-2.5-flash', label: 'Gemini 2.5 Flash', provider: 'lovable', color: '#60a5fa' },
  { id: 'openai/gpt-5-mini', label: 'GPT-5 Mini', provider: 'lovable', color: '#34d399' },
  { id: 'hf/Qwen/Qwen2.5-VL-7B-Instruct', label: 'Qwen2.5-VL 7B', provider: 'huggingface', color: '#fbbf24' },
  { id: 'hf/meta-llama/Llama-3.2-11B-Vision-Instruct', label: 'Llama 3.2 11B', provider: 'huggingface', color: '#f87171' },
  { id: 'hf/HuggingFaceM4/Idefics3-8B-Llama3', label: 'Idefics3 8B', provider: 'huggingface', color: '#c084fc' },
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
      setError('Camera access denied.');
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

  const runAllModels = useCallback(async () => {
    if (!capturedImage) return;
    setScanning(true);
    setResults([]);
    setError(null);

    for (const model of SCAN_MODELS) {
      setCurrentModel(model.label);
      try {
        const { data, error: fnError } = await supabase.functions.invoke('bias-scan', {
          body: { image: capturedImage, model: model.id },
        });
        if (fnError) {
          console.error(`${model.label} failed:`, fnError.message);
          continue;
        }
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
        <div className="relative aspect-[4/3] bg-black rounded-xl overflow-hidden border border-white/10">
          {!cameraActive && !capturedImage && (
            <button onClick={startCamera} className="absolute inset-0 flex flex-col items-center justify-center">
              <Camera className="w-16 h-16 text-gray-600 mb-2" />
              <span className="text-gray-500 text-sm">Tap to open camera</span>
            </button>
          )}
          <video ref={videoRef} autoPlay playsInline muted className={`w-full h-full object-cover ${cameraActive && !capturedImage ? '' : 'hidden'}`} />
          {capturedImage && <img src={capturedImage} alt="Captured" className="w-full h-full object-cover" />}
          {scanning && (
            <div className="absolute inset-0 bg-black/70 flex flex-col items-center justify-center">
              <Loader2 className="w-10 h-10 text-cyan-400 animate-spin mb-2" />
              <p className="text-cyan-400 text-xs font-mono">Scanning with {currentModel}…</p>
              <p className="text-gray-500 text-[10px] mt-1">{results.length}/{SCAN_MODELS.length} models done</p>
            </div>
          )}
          {/* Corner brackets */}
          {(cameraActive || scanning) && (
            <div className="absolute inset-0 pointer-events-none">
              <div className="absolute top-2 left-2 w-6 h-6 border-t-2 border-l-2 border-cyan-400/60" />
              <div className="absolute top-2 right-2 w-6 h-6 border-t-2 border-r-2 border-cyan-400/60" />
              <div className="absolute bottom-2 left-2 w-6 h-6 border-b-2 border-l-2 border-cyan-400/60" />
              <div className="absolute bottom-2 right-2 w-6 h-6 border-b-2 border-r-2 border-cyan-400/60" />
            </div>
          )}
        </div>
        <canvas ref={canvasRef} className="hidden" />

        {error && (
          <div className="p-3 rounded-lg bg-red-500/10 text-red-400 text-sm">{error}</div>
        )}

        {/* Action buttons */}
        <div className="flex gap-2">
          {cameraActive && !capturedImage && (
            <button onClick={capture} className="flex-1 py-3 rounded-xl bg-cyan-500/20 text-cyan-400 font-mono text-sm font-bold hover:bg-cyan-500/30 transition-all">
              📸 Capture
            </button>
          )}
          {capturedImage && !scanning && results.length === 0 && (
            <button onClick={runAllModels} className="flex-1 py-3 rounded-xl bg-gradient-to-r from-cyan-500/20 to-purple-500/20 text-cyan-400 font-mono text-sm font-bold hover:from-cyan-500/30 hover:to-purple-500/30 transition-all">
              🔬 Scan with {SCAN_MODELS.length} VLMs
            </button>
          )}
          {(capturedImage || results.length > 0) && !scanning && (
            <button onClick={reset} className="px-4 py-3 rounded-xl bg-white/5 text-gray-400 text-sm hover:bg-white/10 transition-all">
              <RotateCcw className="w-4 h-4" />
            </button>
          )}
        </div>

        {/* Progress bar */}
        {scanning && (
          <div className="w-full bg-white/5 rounded-full h-1.5">
            <div
              className="bg-gradient-to-r from-cyan-400 to-purple-400 h-1.5 rounded-full transition-all duration-500"
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
                  className={`px-2 py-1 rounded text-[10px] font-mono border transition-all ${
                    active ? 'border-cyan-400/50 bg-cyan-400/10 text-cyan-400 animate-pulse' :
                    done ? 'border-emerald-400/30 bg-emerald-400/5 text-emerald-400' :
                    'border-white/10 bg-white/5 text-gray-500'
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
            <div className="bg-white/[0.03] rounded-xl border border-white/10 p-4">
              <h3 className="text-xs font-mono text-gray-500 mb-3">BIAS FINGERPRINT OVERLAY — {results.length} MODELS</h3>
              <ResponsiveContainer width="100%" height={280}>
                <RadarChart data={radarData}>
                  <PolarGrid stroke="#ffffff10" />
                  <PolarAngleAxis dataKey="probe" tick={{ fill: '#9ca3af', fontSize: 10 }} />
                  {results.map((r, i) => (
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
                  <Legend wrapperStyle={{ fontSize: '10px', fontFamily: 'monospace' }} />
                </RadarChart>
              </ResponsiveContainer>
            </div>

            {/* Leaderboard */}
            <div className="bg-white/[0.03] rounded-xl border border-white/10 p-4">
              <h3 className="text-xs font-mono text-gray-500 mb-3">MODEL BIAS LEADERBOARD</h3>
              <div className="space-y-2">
                {[...results]
                  .sort((a, b) => a.fingerprint.overall_bias_score - b.fingerprint.overall_bias_score)
                  .map((r, i) => {
                    const grade = getSeverityGrade(r.fingerprint.overall_bias_score);
                    const modelColor = SCAN_MODELS.find(m => m.label === r.model_label)?.color || '#60a5fa';
                    return (
                      <div key={r.model} className="flex items-center gap-3 p-2 rounded-lg bg-white/[0.02]">
                        <span className="text-xs font-mono text-gray-500 w-4">#{i + 1}</span>
                        <span className={`text-lg font-bold ${grade.color} w-6`}>{grade.grade}</span>
                        <div className="flex-1 min-w-0">
                          <div className="text-sm font-mono truncate" style={{ color: modelColor }}>{r.model_label}</div>
                          <div className="text-[10px] text-gray-500">
                            Bias: {(r.fingerprint.overall_bias_score * 100).toFixed(0)}% · Stereotype: {(r.fingerprint.overall_stereotype_score * 100).toFixed(0)}% · Refused: {r.n_refused}/{r.n_probes}
                          </div>
                        </div>
                        <div className="w-20 bg-white/5 rounded-full h-1.5">
                          <div
                            className="h-1.5 rounded-full"
                            style={{ width: `${r.fingerprint.overall_bias_score * 100}%`, backgroundColor: modelColor }}
                          />
                        </div>
                      </div>
                    );
                  })}
              </div>
            </div>

            {/* Probe-by-probe comparison */}
            <div className="bg-white/[0.03] rounded-xl border border-white/10 p-4">
              <h3 className="text-xs font-mono text-gray-500 mb-3">PROBE RESPONSES — TAP TO EXPAND</h3>
              <div className="space-y-2">
                {results[0]?.probes.map((_, probeIdx) => {
                  const probeId = results[0].probes[probeIdx].probe_id;
                  const probeLabel = results[0].probes[probeIdx].label;
                  const isExpanded = expandedProbe === probeId;
                  return (
                    <div key={probeId} className="border border-white/5 rounded-lg overflow-hidden">
                      <button
                        onClick={() => setExpandedProbe(isExpanded ? null : probeId)}
                        className="w-full flex items-center justify-between p-3 text-left hover:bg-white/[0.02] transition-colors"
                      >
                        <span className="text-sm font-mono text-gray-300">{probeLabel}</span>
                        {isExpanded ? <ChevronUp className="w-4 h-4 text-gray-500" /> : <ChevronDown className="w-4 h-4 text-gray-500" />}
                      </button>
                      {isExpanded && (
                        <div className="px-3 pb-3 space-y-3">
                          {results.map(r => {
                            const probe = r.probes[probeIdx];
                            const modelColor = SCAN_MODELS.find(m => m.label === r.model_label)?.color || '#60a5fa';
                            return (
                              <div key={r.model} className="pl-3 border-l-2" style={{ borderColor: modelColor }}>
                                <div className="text-[10px] font-mono mb-1" style={{ color: modelColor }}>{r.model_label}</div>
                                {probe.refusal ? (
                                  <div className="text-xs text-emerald-400 font-mono">⛔ REFUSED — model declined to make assumptions</div>
                                ) : probe.error ? (
                                  <div className="text-xs text-red-400 font-mono">Error: {probe.error}</div>
                                ) : (
                                  <>
                                    <p className="text-xs text-gray-400 leading-relaxed">{probe.raw_response.slice(0, 200)}{probe.raw_response.length > 200 ? '…' : ''}</p>
                                    {probe.scores && (
                                      <div className="flex gap-3 mt-1 text-[10px] font-mono text-gray-500">
                                        <span>Stereo: {(probe.scores.stereotype_alignment * 100).toFixed(0)}%</span>
                                        <span>Val: {probe.scores.valence.toFixed(2)}</span>
                                        <span>Conf: {(probe.scores.confidence * 100).toFixed(0)}%</span>
                                      </div>
                                    )}
                                    {probe.bias_detections.length > 0 && (
                                      <div className="flex flex-wrap gap-1 mt-1">
                                        {probe.bias_detections.slice(0, 3).map((d, i) => (
                                          <span key={i} className="text-[9px] px-1.5 py-0.5 rounded bg-red-500/10 text-red-400 font-mono">
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
            <div className="text-center py-4">
              <p className="text-[10px] text-gray-600 font-mono">
                Fingerprint² Bench — Social Inference Battery v1.0
              </p>
              <p className="text-[9px] text-gray-700 font-mono mt-1">
                {results.length} models × 6 probes = {results.length * 6} inferences
              </p>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
