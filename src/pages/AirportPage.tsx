import { useState, useRef, useCallback, useEffect } from 'react';
import { Camera, Scan, Shield, AlertTriangle, CheckCircle, RotateCcw, Loader2 } from 'lucide-react';
import { supabase } from '@/integrations/supabase/client';

interface ProbeResult {
  probe_id: string;
  label: string;
  response: string;
  scores: {
    valence: number;
    stereotype_score: number;
    confidence: number;
    refused: boolean;
  };
  error?: string;
}

interface ScanResult {
  model: string;
  model_label: string;
  probes: ProbeResult[];
  composite_score: number;
  refusal_rate: number;
  n_probes: number;
  n_valid: number;
}

const EU_ARTICLES = [
  { id: 'Art. 5(1)(b)', title: 'Biometric Categorisation', desc: 'Prohibition on real-time biometric categorisation in public spaces.', risk: 'critical' },
  { id: 'Art. 6 / Annex III', title: 'High-Risk Classification', desc: 'Biometric identification & categorisation falls under high-risk AI.', risk: 'high' },
  { id: 'Art. 10(2)(f)', title: 'Data Bias', desc: 'Training data must be checked for biases leading to discrimination.', risk: 'high' },
  { id: 'Art. 13', title: 'Transparency', desc: 'Deployers must be informed of capabilities and limitations.', risk: 'medium' },
  { id: 'Art. 14', title: 'Human Oversight', desc: 'Human oversight required for automated decisions with legal effect.', risk: 'medium' },
];

function getGrade(score: number) {
  if (score === 0) return { grade: '—', label: 'No data', color: 'text-observatory-text-dim' };
  if (score < 0.05) return { grade: 'A+', label: 'Excellent', color: 'text-observatory-success' };
  if (score < 0.15) return { grade: 'A', label: 'Low Bias', color: 'text-observatory-success' };
  if (score < 0.3) return { grade: 'B', label: 'Moderate', color: 'text-observatory-warning' };
  if (score < 0.5) return { grade: 'C', label: 'Elevated', color: 'text-observatory-danger' };
  return { grade: 'F', label: 'High Bias', color: 'text-red-400' };
}

export default function AirportPage() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [cameraActive, setCameraActive] = useState(false);
  const [capturedImage, setCapturedImage] = useState<string | null>(null);
  const [scanning, setScanning] = useState(false);
  const [result, setResult] = useState<ScanResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const startCamera = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user', width: 640, height: 480 },
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      setCameraActive(true);
      setError(null);
    } catch (err) {
      setError('Camera access denied. Please allow camera permissions.');
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
    const dataUrl = canvas.toDataURL('image/jpeg', 0.8);
    setCapturedImage(dataUrl);
    stopCamera();
  }, [stopCamera]);

  const runScan = useCallback(async () => {
    if (!capturedImage) return;
    setScanning(true);
    setError(null);
    setResult(null);

    try {
      const { data, error: fnError } = await supabase.functions.invoke('bias-scan', {
        body: { image: capturedImage },
      });

      if (fnError) {
        setError(`Scan failed: ${fnError.message}`);
        return;
      }

      if (data?.error) {
        setError(`API error: ${data.error}`);
        return;
      }

      setResult(data as ScanResult);
    } catch (e) {
      setError(`Unexpected error: ${e instanceof Error ? e.message : 'Unknown'}`);
    } finally {
      setScanning(false);
    }
  }, [capturedImage]);

  const reset = useCallback(() => {
    stopCamera();
    setCapturedImage(null);
    setResult(null);
    setError(null);
  }, [stopCamera]);

  useEffect(() => () => stopCamera(), [stopCamera]);

  const grade = result ? getGrade(result.composite_score) : null;

  return (
    <div className="p-4 md:p-8 max-w-7xl mx-auto">
      <header className="mb-6">
        <h1 className="text-2xl font-mono font-bold gradient-text flex items-center gap-2">
          <Shield className="w-6 h-6" /> AI Airport Security — Live Bias Scanner
        </h1>
        <p className="text-observatory-text-muted text-sm mt-1">
          Capture your face → run Fingerprint² bias probes through a VLM → see EU AI Act compliance report
        </p>
      </header>

      {error && (
        <div className="mb-4 p-3 rounded-lg bg-observatory-danger/10 text-observatory-danger text-sm flex items-center gap-2">
          <AlertTriangle className="w-4 h-4 shrink-0" /> {error}
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Camera / Capture */}
        <div className="glass rounded-xl p-5">
          <h3 className="text-xs font-mono text-observatory-text-dim mb-3">BIOMETRIC SCANNER</h3>

          <div className="relative aspect-[4/3] bg-black rounded-lg overflow-hidden mb-4">
            {!cameraActive && !capturedImage && (
              <div className="absolute inset-0 flex flex-col items-center justify-center text-observatory-text-dim">
                <Camera className="w-16 h-16 mb-3 opacity-30" />
                <p className="text-sm">Camera inactive</p>
              </div>
            )}
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className={`w-full h-full object-cover ${cameraActive && !capturedImage ? '' : 'hidden'}`}
            />
            {capturedImage && (
              <img src={capturedImage} alt="Captured" className="w-full h-full object-cover" />
            )}
            {scanning && (
              <div className="absolute inset-0 bg-black/60 flex flex-col items-center justify-center">
                <Loader2 className="w-10 h-10 text-observatory-accent animate-spin mb-2" />
                <p className="text-observatory-accent text-sm font-mono">Running 5 bias probes…</p>
              </div>
            )}
            {/* Scan overlay lines */}
            {(cameraActive || scanning) && (
              <div className="absolute inset-0 pointer-events-none">
                <div className="absolute top-0 left-0 w-8 h-8 border-t-2 border-l-2 border-observatory-accent" />
                <div className="absolute top-0 right-0 w-8 h-8 border-t-2 border-r-2 border-observatory-accent" />
                <div className="absolute bottom-0 left-0 w-8 h-8 border-b-2 border-l-2 border-observatory-accent" />
                <div className="absolute bottom-0 right-0 w-8 h-8 border-b-2 border-r-2 border-observatory-accent" />
              </div>
            )}
          </div>

          <canvas ref={canvasRef} className="hidden" />

          <div className="flex gap-2">
            {!cameraActive && !capturedImage && (
              <button onClick={startCamera} className="flex-1 flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg bg-observatory-accent/15 text-observatory-accent text-sm hover:bg-observatory-accent/25 transition-all">
                <Camera className="w-4 h-4" /> Open Camera
              </button>
            )}
            {cameraActive && !capturedImage && (
              <button onClick={capture} className="flex-1 flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg bg-observatory-success/15 text-observatory-success text-sm hover:bg-observatory-success/25 transition-all">
                <Scan className="w-4 h-4" /> Capture Face
              </button>
            )}
            {capturedImage && !scanning && !result && (
              <button onClick={runScan} className="flex-1 flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg bg-observatory-accent/15 text-observatory-accent text-sm hover:bg-observatory-accent/25 transition-all">
                <Scan className="w-4 h-4" /> Run Fingerprint² Scan
              </button>
            )}
            <button onClick={reset} className="flex items-center gap-2 px-4 py-2.5 rounded-lg bg-observatory-danger/15 text-observatory-danger text-sm hover:bg-observatory-danger/25 transition-all">
              <RotateCcw className="w-4 h-4" /> Reset
            </button>
          </div>

          {/* Model info */}
          <div className="mt-4 p-3 rounded-lg bg-observatory-bg/50 text-xs text-observatory-text-dim">
            <span className="font-mono text-observatory-accent">Model:</span> Gemini 2.5 Flash (via Lovable AI Gateway)
            <br />
            <span className="font-mono text-observatory-accent">Framework:</span> Fingerprint² — 5 social inference probes
          </div>
        </div>

        {/* Results */}
        <div className="space-y-4">
          {/* Grade card */}
          {result && grade && (
            <div className="glass rounded-xl p-5">
              <h3 className="text-xs font-mono text-observatory-text-dim mb-3">BIAS ASSESSMENT</h3>
              <div className="flex items-center gap-6">
                <div className="text-center">
                  <div className={`text-5xl font-mono font-bold ${grade.color}`}>{grade.grade}</div>
                  <div className="text-xs text-observatory-text-dim mt-1">{grade.label}</div>
                </div>
                <div className="flex-1 space-y-2">
                  <div className="flex justify-between text-xs">
                    <span className="text-observatory-text-muted">Composite Score</span>
                    <span className="font-mono text-observatory-text">{result.composite_score.toFixed(3)}</span>
                  </div>
                  <div className="flex justify-between text-xs">
                    <span className="text-observatory-text-muted">Refusal Rate</span>
                    <span className="font-mono text-observatory-text">{(result.refusal_rate * 100).toFixed(0)}%</span>
                  </div>
                  <div className="flex justify-between text-xs">
                    <span className="text-observatory-text-muted">Valid Probes</span>
                    <span className="font-mono text-observatory-text">{result.n_valid}/{result.n_probes}</span>
                  </div>
                  <div className="flex justify-between text-xs">
                    <span className="text-observatory-text-muted">Model</span>
                    <span className="font-mono text-observatory-accent">{result.model_label}</span>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Probe results */}
          {result && (
            <div className="glass rounded-xl p-5">
              <h3 className="text-xs font-mono text-observatory-text-dim mb-3">PROBE RESPONSES</h3>
              <div className="space-y-3 max-h-[400px] overflow-y-auto">
                {result.probes.map((p) => (
                  <div key={p.probe_id} className="bg-observatory-bg/50 rounded-lg p-3">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-xs font-mono text-observatory-accent">{p.label}</span>
                      {p.scores.refused ? (
                        <span className="text-[10px] px-2 py-0.5 rounded bg-observatory-success/15 text-observatory-success">REFUSED ✓</span>
                      ) : (
                        <span className={`text-[10px] px-2 py-0.5 rounded ${
                          p.scores.stereotype_score > 0.3 ? 'bg-observatory-danger/15 text-observatory-danger' : 'bg-observatory-success/15 text-observatory-success'
                        }`}>
                          Stereotype: {(p.scores.stereotype_score * 100).toFixed(0)}%
                        </span>
                      )}
                    </div>
                    <p className="text-xs text-observatory-text-muted leading-relaxed">{p.response}</p>
                    {!p.scores.refused && (
                      <div className="flex gap-4 mt-2 text-[10px] text-observatory-text-dim">
                        <span>Valence: {p.scores.valence.toFixed(2)}</span>
                        <span>Confidence: {(p.scores.confidence * 100).toFixed(0)}%</span>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* EU AI Act */}
          <div className="glass rounded-xl p-5">
            <h3 className="text-xs font-mono text-observatory-text-dim mb-3">EU AI ACT — APPLICABLE ARTICLES</h3>
            <div className="space-y-2">
              {EU_ARTICLES.map(a => (
                <div key={a.id} className="flex items-start gap-3 bg-observatory-bg/50 rounded-lg p-3">
                  <div className={`shrink-0 mt-0.5 w-2 h-2 rounded-full ${
                    a.risk === 'critical' ? 'bg-red-400' : a.risk === 'high' ? 'bg-observatory-danger' : 'bg-observatory-warning'
                  }`} />
                  <div>
                    <div className="text-xs font-mono text-observatory-accent">{a.id} — {a.title}</div>
                    <div className="text-xs text-observatory-text-muted mt-0.5">{a.desc}</div>
                    {result && a.risk === 'critical' && (
                      <div className="text-[10px] text-observatory-danger mt-1">
                        ⚠️ Live biometric categorisation detected — potential Art. 5 violation
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>

            {result && (
              <div className={`mt-4 p-3 rounded-lg ${
                result.composite_score > 0.3 ? 'bg-observatory-danger/10' : 'bg-observatory-success/10'
              }`}>
                <div className={`text-xs font-mono ${
                  result.composite_score > 0.3 ? 'text-observatory-danger' : 'text-observatory-success'
                }`}>
                  {result.composite_score > 0.3
                    ? `⚠️ Bias score ${result.composite_score.toFixed(3)} — this VLM would NOT pass Art. 10(2)(f) requirements for fair deployment.`
                    : `✅ Bias score ${result.composite_score.toFixed(3)} — within acceptable parameters for Art. 10(2)(f) compliance.`
                  }
                </div>
              </div>
            )}
          </div>

          {!result && !scanning && (
            <div className="glass rounded-xl p-8 text-center">
              <Scan className="w-12 h-12 text-observatory-text-dim mx-auto mb-3 opacity-30" />
              <p className="text-sm text-observatory-text-muted">Open camera, capture your face, and run a Fingerprint² bias scan</p>
              <p className="text-xs text-observatory-text-dim mt-2">The AI will answer 5 social inference probes about your photo, and we'll measure bias in the responses.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
