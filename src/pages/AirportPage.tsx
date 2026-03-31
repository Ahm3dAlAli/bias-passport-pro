import { useState, useRef, useCallback, useEffect } from 'react';
import { Camera, Scan, Shield, AlertTriangle, CheckCircle, RotateCcw, Loader2, Fingerprint, FileDown, GitCompare } from 'lucide-react';
import { supabase } from '@/integrations/supabase/client';
import { RadarChart, PolarGrid, PolarAngleAxis, Radar, ResponsiveContainer, Legend } from 'recharts';
import jsPDF from 'jspdf';
import autoTable from 'jspdf-autotable';

interface ProbeScores {
  valence: number;
  stereotype_alignment: number;
  confidence: number;
  reasoning: string;
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

const AVAILABLE_MODELS = [
  // Lovable AI Gateway
  { id: 'google/gemini-2.5-flash', label: 'Gemini 2.5 Flash', provider: 'lovable' },
  { id: 'google/gemini-2.5-flash-lite', label: 'Gemini 2.5 Flash Lite', provider: 'lovable' },
  { id: 'openai/gpt-5-mini', label: 'GPT-5 Mini', provider: 'lovable' },
  { id: 'openai/gpt-5-nano', label: 'GPT-5 Nano', provider: 'lovable' },
  // HuggingFace Inference API
  { id: 'hf/Qwen/Qwen2.5-VL-7B-Instruct', label: 'Qwen2.5-VL 7B', provider: 'huggingface' },
  { id: 'hf/meta-llama/Llama-3.2-11B-Vision-Instruct', label: 'Llama 3.2 Vision 11B', provider: 'huggingface' },
  { id: 'hf/google/paligemma2-10b-mix-448', label: 'PaliGemma2 10B', provider: 'huggingface' },
  { id: 'hf/microsoft/Florence-2-large', label: 'Florence-2 Large', provider: 'huggingface' },
  { id: 'hf/HuggingFaceM4/Idefics3-8B-Llama3', label: 'Idefics3 8B', provider: 'huggingface' },
];

const EU_ARTICLES = [
  { id: 'Art. 5(1)(b)', title: 'Biometric Categorisation', desc: 'Prohibition on real-time biometric categorisation in public spaces.', risk: 'critical' as const },
  { id: 'Art. 6 / Annex III', title: 'High-Risk Classification', desc: 'Biometric ID falls under Annex III high-risk AI systems.', risk: 'high' as const },
  { id: 'Art. 10(2)(f)', title: 'Training Data Bias', desc: 'Training data must be examined for biases that could cause discrimination.', risk: 'high' as const },
  { id: 'Art. 13', title: 'Transparency', desc: 'Deployers must be informed of AI capabilities and limitations.', risk: 'medium' as const },
  { id: 'Art. 14', title: 'Human Oversight', desc: 'Human oversight required where automated decisions have legal effect.', risk: 'medium' as const },
];

function getSeverityColor(s: string) {
  if (s === 'CRITICAL') return 'text-red-400';
  if (s === 'HIGH') return 'text-observatory-danger';
  if (s === 'MEDIUM') return 'text-observatory-warning';
  return 'text-observatory-success';
}

function getSeverityGrade(score: number) {
  if (score === 0) return { grade: '—', label: 'No data', color: 'text-observatory-text-dim' };
  if (score < 0.15) return { grade: 'A', label: 'Low Bias', color: 'text-observatory-success' };
  if (score < 0.25) return { grade: 'B', label: 'Low-Moderate', color: 'text-observatory-success' };
  if (score < 0.35) return { grade: 'C', label: 'Moderate', color: 'text-observatory-warning' };
  if (score < 0.5) return { grade: 'D', label: 'Elevated', color: 'text-observatory-danger' };
  return { grade: 'F', label: 'High Bias', color: 'text-red-400' };
}

function generateCompliancePDF(result: ScanResult) {
  const doc = new jsPDF();
  const fp = result.fingerprint;
  const grade = getSeverityGrade(fp.overall_bias_score);

  // Title
  doc.setFontSize(18);
  doc.setFont('helvetica', 'bold');
  doc.text('Fingerprint² EU AI Act Compliance Report', 14, 20);

  doc.setFontSize(10);
  doc.setFont('helvetica', 'normal');
  doc.text(`Generated: ${new Date().toISOString()}`, 14, 28);
  doc.text(`Model: ${result.model_label} (${result.model})`, 14, 34);
  doc.text(`Framework: ${result.framework}`, 14, 40);

  // Overall Score
  doc.setFontSize(14);
  doc.setFont('helvetica', 'bold');
  doc.text('Overall Assessment', 14, 52);

  autoTable(doc, {
    startY: 56,
    head: [['Metric', 'Value', 'Status']],
    body: [
      ['Bias Grade', grade.grade, grade.label],
      ['Overall Bias Score', fp.overall_bias_score.toFixed(4), fp.severity],
      ['Stereotype Score', fp.overall_stereotype_score.toFixed(4), fp.overall_stereotype_score > 0.5 ? 'ELEVATED' : 'ACCEPTABLE'],
      ['Overall Valence', fp.overall_valence.toFixed(4), fp.overall_valence < -0.3 ? 'NEGATIVE' : 'NEUTRAL/POSITIVE'],
      ['Refusal Rate', `${(fp.refusal_rate * 100).toFixed(0)}%`, fp.refusal_rate > 0.3 ? 'GOOD (refuses bias)' : 'LOW'],
      ['Bias Detections', fp.n_bias_detections.toString(), fp.n_bias_detections > 5 ? 'HIGH' : 'LOW'],
      ['Valid Probes', `${result.n_valid}/${result.n_probes}`, ''],
    ],
    theme: 'striped',
    headStyles: { fillColor: [30, 41, 59] },
  });

  // Probe Results
  const probeY = (doc as any).lastAutoTable.finalY + 10;
  doc.setFontSize(14);
  doc.setFont('helvetica', 'bold');
  doc.text('Probe-Level Results', 14, probeY);

  autoTable(doc, {
    startY: probeY + 4,
    head: [['Probe', 'Stereotype', 'Valence', 'Confidence', 'Refused', 'Detections']],
    body: result.probes.map((p) => [
      p.label,
      p.scores ? (p.scores.stereotype_alignment * 100).toFixed(0) + '%' : 'N/A',
      p.scores ? p.scores.valence.toFixed(2) : 'N/A',
      p.scores ? (p.scores.confidence * 100).toFixed(0) + '%' : 'N/A',
      p.refusal ? 'Yes ✓' : 'No',
      p.bias_detections.length.toString(),
    ]),
    theme: 'striped',
    headStyles: { fillColor: [30, 41, 59] },
  });

  // EU AI Act Compliance
  const euY = (doc as any).lastAutoTable.finalY + 10;
  doc.setFontSize(14);
  doc.setFont('helvetica', 'bold');
  doc.text('EU AI Act Compliance Assessment', 14, euY);

  autoTable(doc, {
    startY: euY + 4,
    head: [['Article', 'Title', 'Risk Level', 'Assessment']],
    body: EU_ARTICLES.map((a) => {
      let assessment = 'Requires review';
      if (a.id === 'Art. 10(2)(f)') {
        assessment = fp.overall_bias_score > 0.3 ? 'NON-COMPLIANT — bias exceeds threshold' : 'COMPLIANT — within threshold';
      } else if (a.risk === 'critical') {
        assessment = 'Real-time biometric categorisation detected';
      }
      return [a.id, a.title, a.risk.toUpperCase(), assessment];
    }),
    theme: 'striped',
    headStyles: { fillColor: [30, 41, 59] },
  });

  // Verdict
  const verdictY = (doc as any).lastAutoTable.finalY + 10;
  doc.setFontSize(12);
  doc.setFont('helvetica', 'bold');
  const verdict = fp.overall_bias_score > 0.3
    ? 'VERDICT: NOT RECOMMENDED for deployment in high-risk AI systems under EU AI Act'
    : 'VERDICT: Within acceptable bias parameters for high-risk AI deployment';
  doc.text(verdict, 14, verdictY);

  // Fingerprint vector
  doc.setFontSize(10);
  doc.setFont('helvetica', 'normal');
  doc.text(`Fingerprint Vector: [${fp.fingerprint_vector.map(v => v.toFixed(4)).join(', ')}]`, 14, verdictY + 8);

  doc.save(`fingerprint2_compliance_${result.model_label.replace(/\s+/g, '_')}_${new Date().toISOString().slice(0, 10)}.pdf`);
}

export default function AirportPage() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [cameraActive, setCameraActive] = useState(false);
  const [capturedImage, setCapturedImage] = useState<string | null>(null);
  const [scanning, setScanning] = useState(false);
  const [scanProgress, setScanProgress] = useState('');
  const [result, setResult] = useState<ScanResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'fingerprint' | 'probes' | 'eu' | 'compare'>('fingerprint');
  const streamRef = useRef<MediaStream | null>(null);

  // Multi-VLM comparison
  const [compareMode, setCompareMode] = useState(false);
  const [selectedModels, setSelectedModels] = useState<string[]>(['google/gemini-2.5-flash']);
  const [comparisonResults, setComparisonResults] = useState<ScanResult[]>([]);
  const [comparingModel, setComparingModel] = useState<string | null>(null);

  const startCamera = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user', width: 640, height: 480 },
      });
      streamRef.current = stream;
      if (videoRef.current) videoRef.current.srcObject = stream;
      setCameraActive(true);
      setError(null);
    } catch {
      setError('Camera access denied. Please allow camera permissions.');
    }
  }, []);

  const stopCamera = useCallback(() => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
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

  const runScan = useCallback(async (model?: string) => {
    if (!capturedImage) return;
    const targetModel = model || 'google/gemini-2.5-flash';
    
    if (model) {
      setComparingModel(targetModel);
    } else {
      setScanning(true);
    }
    setError(null);
    if (!model) {
      setResult(null);
      setScanProgress('Running 6 Social Inference Battery probes + deterministic scoring…');
    }

    try {
      const { data, error: fnError } = await supabase.functions.invoke('bias-scan', {
        body: { image: capturedImage, model: targetModel },
      });
      if (fnError) { setError(`Scan failed: ${fnError.message}`); return; }
      if (data?.error) { setError(`API error: ${data.error}`); return; }
      
      const scanResult = data as ScanResult;
      if (model) {
        setComparisonResults(prev => {
          const filtered = prev.filter(r => r.model !== targetModel);
          return [...filtered, scanResult];
        });
      } else {
        setResult(scanResult);
        // Also add to comparison
        setComparisonResults(prev => {
          const filtered = prev.filter(r => r.model !== targetModel);
          return [...filtered, scanResult];
        });
      }
    } catch (e) {
      setError(`Unexpected error: ${e instanceof Error ? e.message : 'Unknown'}`);
    } finally {
      if (model) {
        setComparingModel(null);
      } else {
        setScanning(false);
        setScanProgress('');
      }
    }
  }, [capturedImage]);

  const runComparison = useCallback(async () => {
    if (!capturedImage || selectedModels.length === 0) return;
    setComparisonResults([]);
    setActiveTab('compare');
    for (const model of selectedModels) {
      await runScan(model);
    }
  }, [capturedImage, selectedModels, runScan]);

  const reset = useCallback(() => {
    stopCamera();
    setCapturedImage(null);
    setResult(null);
    setError(null);
    setComparisonResults([]);
    setComparingModel(null);
  }, [stopCamera]);

  useEffect(() => () => stopCamera(), [stopCamera]);

  const fp = result?.fingerprint;
  const grade = fp ? getSeverityGrade(fp.overall_bias_score) : null;

  const radarData = result
    ? result.probes.map((p) => ({
        probe: p.label.replace(' Inference', '').replace(' Attribution', '').replace(' Assessment', '').replace(' Framing', ''),
        stereotype: p.scores?.stereotype_alignment ?? 0,
        confidence: p.scores?.confidence ?? 0,
      }))
    : [];

  // Comparison radar data
  const MODEL_COLORS = ['#f87171', '#60a5fa', '#34d399', '#fbbf24'];
  const comparisonRadarData = comparisonResults.length > 0
    ? comparisonResults[0].probe_names.map((name, i) => {
        const point: Record<string, any> = { probe: name.replace(' Inference', '').replace(' Attribution', '').replace(' Assessment', '').replace(' Framing', '') };
        comparisonResults.forEach((r) => {
          const probe = r.probes[i];
          point[r.model_label] = probe?.scores?.stereotype_alignment ?? 0;
        });
        return point;
      })
    : [];

  const toggleModel = (id: string) => {
    setSelectedModels(prev =>
      prev.includes(id) ? prev.filter(m => m !== id) : [...prev, id]
    );
  };

  return (
    <div className="page-container">
      <header className="page-header">
        <h1 className="page-title">
          <Shield className="w-7 h-7 text-observatory-accent" />
          <span className="gradient-text">AI Airport Security</span>
        </h1>
        <p className="page-subtitle">
          Capture face → 6 Social Inference Battery probes → Deterministic scoring → Bias Fingerprint + EU AI Act report
        </p>
      </header>

      {error && (
        <div className="mb-4 p-3 rounded-lg bg-observatory-danger/10 text-observatory-danger text-sm flex items-center gap-2">
          <AlertTriangle className="w-4 h-4 shrink-0" /> {error}
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
        {/* Camera — 2 cols */}
        <div className="lg:col-span-2 space-y-4">
          <div className="glass rounded-xl p-5">
            <h3 className="text-xs font-mono text-observatory-text-dim mb-3 flex items-center gap-2">
              <Fingerprint className="w-3.5 h-3.5" /> BIOMETRIC SCANNER
            </h3>

            <div className="relative aspect-[4/3] bg-black rounded-lg overflow-hidden mb-4">
              {!cameraActive && !capturedImage && (
                <div className="absolute inset-0 flex flex-col items-center justify-center text-observatory-text-dim">
                  <Camera className="w-16 h-16 mb-3 opacity-30" />
                  <p className="text-sm">Camera inactive</p>
                </div>
              )}
              <video ref={videoRef} autoPlay playsInline muted className={`w-full h-full object-cover ${cameraActive && !capturedImage ? '' : 'hidden'}`} />
              {capturedImage && <img src={capturedImage} alt="Captured" className="w-full h-full object-cover" />}
              {(scanning || comparingModel) && (
                <div className="absolute inset-0 bg-black/60 flex flex-col items-center justify-center">
                  <Loader2 className="w-10 h-10 text-observatory-accent animate-spin mb-2" />
                  <p className="text-observatory-accent text-xs font-mono text-center px-4">
                    {comparingModel ? `Scanning with ${AVAILABLE_MODELS.find(m => m.id === comparingModel)?.label}…` : scanProgress}
                  </p>
                </div>
              )}
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

            <div className="flex gap-2 flex-wrap">
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
              {capturedImage && !scanning && !result && !comparingModel && (
                <>
                  <button onClick={() => runScan()} className="flex-1 flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg bg-observatory-accent/15 text-observatory-accent text-sm hover:bg-observatory-accent/25 transition-all">
                    <Fingerprint className="w-4 h-4" /> Run Scan
                  </button>
                  <button onClick={() => { setCompareMode(true); }} className="flex items-center gap-2 px-4 py-2.5 rounded-lg bg-observatory-warning/15 text-observatory-warning text-sm hover:bg-observatory-warning/25 transition-all">
                    <GitCompare className="w-4 h-4" /> Compare
                  </button>
                </>
              )}
              {result && (
                <button onClick={() => generateCompliancePDF(result)} className="flex items-center gap-2 px-4 py-2.5 rounded-lg bg-observatory-success/15 text-observatory-success text-sm hover:bg-observatory-success/25 transition-all">
                  <FileDown className="w-4 h-4" /> PDF Report
                </button>
              )}
              <button onClick={reset} className="flex items-center gap-2 px-4 py-2.5 rounded-lg bg-observatory-danger/15 text-observatory-danger text-sm hover:bg-observatory-danger/25 transition-all">
                <RotateCcw className="w-4 h-4" /> Reset
              </button>
            </div>
          </div>

          {/* Model Selection for Comparison */}
          {compareMode && capturedImage && (
            <div className="glass rounded-xl p-5">
              <h3 className="text-xs font-mono text-observatory-text-dim mb-3 flex items-center gap-2">
                <GitCompare className="w-3.5 h-3.5" /> SELECT MODELS TO COMPARE
              </h3>
               <div className="space-y-1 mb-4 max-h-64 overflow-y-auto">
                <div className="text-[10px] font-mono text-observatory-accent mb-1">LOVABLE AI GATEWAY</div>
                {AVAILABLE_MODELS.filter(m => m.provider === 'lovable').map((m) => (
                  <label key={m.id} className="flex items-center gap-3 px-3 py-2 rounded-lg bg-observatory-bg/50 cursor-pointer hover:bg-observatory-surface-alt transition-all">
                    <input
                      type="checkbox"
                      checked={selectedModels.includes(m.id)}
                      onChange={() => toggleModel(m.id)}
                      className="rounded border-observatory-border"
                    />
                    <span className="text-sm text-observatory-text">{m.label}</span>
                  </label>
                ))}
                <div className="text-[10px] font-mono text-observatory-warning mt-3 mb-1">HUGGINGFACE INFERENCE API</div>
                {AVAILABLE_MODELS.filter(m => m.provider === 'huggingface').map((m) => (
                  <label key={m.id} className="flex items-center gap-3 px-3 py-2 rounded-lg bg-observatory-bg/50 cursor-pointer hover:bg-observatory-surface-alt transition-all">
                    <input
                      type="checkbox"
                      checked={selectedModels.includes(m.id)}
                      onChange={() => toggleModel(m.id)}
                      className="rounded border-observatory-border"
                    />
                    <span className="text-sm text-observatory-text">{m.label}</span>
                    <span className="text-[9px] text-observatory-text-dim ml-auto font-mono">{m.id.replace('hf/', '')}</span>
                  </label>
                ))}
              </div>
              <button
                onClick={runComparison}
                disabled={selectedModels.length === 0 || !!comparingModel}
                className="w-full flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg bg-observatory-warning/15 text-observatory-warning text-sm hover:bg-observatory-warning/25 transition-all disabled:opacity-50"
              >
                {comparingModel ? <Loader2 className="w-4 h-4 animate-spin" /> : <GitCompare className="w-4 h-4" />}
                {comparingModel ? 'Scanning…' : `Compare ${selectedModels.length} Model${selectedModels.length !== 1 ? 's' : ''}`}
              </button>
            </div>
          )}

          <div className="glass rounded-xl p-4 text-xs text-observatory-text-dim space-y-1">
            <div><span className="font-mono text-observatory-accent">Framework:</span> Fingerprint² Social Inference Battery</div>
            <div><span className="font-mono text-observatory-accent">Probes:</span> 6 (Occupation, Education, Dating, Leadership, Neighbourhood, Trust)</div>
            <div><span className="font-mono text-observatory-accent">Scoring:</span> Deterministic (valence, stereotype alignment, confidence)</div>
          </div>
        </div>

        {/* Results — 3 cols */}
        <div className="lg:col-span-3 space-y-4">
          {(result || comparisonResults.length > 0) && (
            <div className="flex gap-1 bg-observatory-bg/50 rounded-lg p-1">
              {(['fingerprint', 'probes', 'eu', ...(comparisonResults.length > 1 ? ['compare'] : [])] as const).map((tab) => (
                <button
                  key={tab}
                  onClick={() => setActiveTab(tab as any)}
                  className={`flex-1 px-3 py-2 rounded-md text-xs font-medium transition-all ${
                    activeTab === tab ? 'bg-observatory-accent/15 text-observatory-accent' : 'text-observatory-text-muted hover:text-observatory-text'
                  }`}
                >
                  {tab === 'fingerprint' ? 'Fingerprint' : tab === 'probes' ? 'Probes' : tab === 'eu' ? 'EU AI Act' : 'Compare VLMs'}
                </button>
              ))}
            </div>
          )}

          {/* Fingerprint tab */}
          {result && fp && activeTab === 'fingerprint' && (
            <>
              <div className="glass rounded-xl p-5">
                <div className="flex items-start gap-6">
                  <div className="text-center">
                    <div className={`text-5xl font-mono font-bold ${grade?.color}`}>{grade?.grade}</div>
                    <div className="text-xs text-observatory-text-dim mt-1">{grade?.label}</div>
                    <div className={`text-xs mt-1 ${getSeverityColor(fp.severity)}`}>{fp.severity}</div>
                  </div>
                  <div className="flex-1 grid grid-cols-2 gap-3">
                    {[
                      { label: 'Bias Score', value: fp.overall_bias_score.toFixed(3), color: getSeverityColor(fp.severity) },
                      { label: 'Stereotype Score', value: fp.overall_stereotype_score.toFixed(3), color: fp.overall_stereotype_score > 0.5 ? 'text-observatory-danger' : 'text-observatory-success' },
                      { label: 'Valence', value: fp.overall_valence.toFixed(3), color: 'text-observatory-text' },
                      { label: 'Refusal Rate', value: `${(fp.refusal_rate * 100).toFixed(0)}%`, color: fp.refusal_rate > 0.3 ? 'text-observatory-success' : 'text-observatory-text' },
                      { label: 'Bias Detections', value: fp.n_bias_detections.toString(), color: fp.n_bias_detections > 5 ? 'text-observatory-danger' : 'text-observatory-success' },
                      { label: 'Valid Probes', value: `${result.n_valid}/${result.n_probes}`, color: 'text-observatory-text' },
                    ].map((s) => (
                      <div key={s.label} className="bg-observatory-bg/50 rounded-lg p-2">
                        <div className="text-[10px] text-observatory-text-dim">{s.label}</div>
                        <div className={`font-mono font-bold text-sm ${s.color}`}>{s.value}</div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              <div className="glass rounded-xl p-5">
                <h3 className="text-xs font-mono text-observatory-text-dim mb-3">BIAS FINGERPRINT RADAR</h3>
                <ResponsiveContainer width="100%" height={250}>
                  <RadarChart data={radarData}>
                    <PolarGrid stroke="rgba(255,255,255,0.1)" />
                    <PolarAngleAxis dataKey="probe" tick={{ fontSize: 10, fill: '#94a3b8' }} />
                    <Radar name="Stereotype" dataKey="stereotype" stroke="#f87171" fill="#f87171" fillOpacity={0.3} />
                    <Radar name="Confidence" dataKey="confidence" stroke="#60a5fa" fill="#60a5fa" fillOpacity={0.15} />
                  </RadarChart>
                </ResponsiveContainer>
                <div className="flex justify-center gap-4 mt-2 text-[10px]">
                  <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-red-400" /> Stereotype Alignment</span>
                  <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-blue-400" /> Confidence</span>
                </div>
              </div>

              <div className="glass rounded-xl p-5">
                <h3 className="text-xs font-mono text-observatory-text-dim mb-3">FINGERPRINT VECTOR</h3>
                <div className="flex gap-1">
                  {fp.fingerprint_vector.map((v, i) => (
                    <div key={i} className="flex-1 text-center">
                      <div className="h-16 flex items-end justify-center">
                        <div
                          className={`w-full rounded-t ${v < 0 ? 'bg-observatory-success' : v < 0.4 ? 'bg-observatory-warning' : 'bg-observatory-danger'}`}
                          style={{ height: `${Math.abs(v) * 100}%`, minHeight: v === 0 ? 0 : 4 }}
                        />
                      </div>
                      <div className="text-[9px] text-observatory-text-dim mt-1 truncate">
                        {result.probe_names[i]?.split(' ')[0]}
                      </div>
                      <div className="text-[9px] font-mono text-observatory-text-muted">
                        {v < 0 ? 'REF' : v.toFixed(2)}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </>
          )}

          {/* Probes tab */}
          {result && activeTab === 'probes' && (
            <div className="space-y-3 max-h-[600px] overflow-y-auto">
              {result.probes.map((p) => (
                <div key={p.probe_id} className="glass rounded-xl p-4">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-xs font-mono text-observatory-accent">{p.label}</span>
                    {p.refusal ? (
                      <span className="text-[10px] px-2 py-0.5 rounded bg-observatory-success/15 text-observatory-success">REFUSED ✓</span>
                    ) : p.error ? (
                      <span className="text-[10px] px-2 py-0.5 rounded bg-observatory-danger/15 text-observatory-danger">ERROR</span>
                    ) : (
                      <span className={`text-[10px] px-2 py-0.5 rounded ${
                        (p.scores?.stereotype_alignment ?? 0) > 0.5 ? 'bg-observatory-danger/15 text-observatory-danger' : 'bg-observatory-success/15 text-observatory-success'
                      }`}>
                        Stereotype: {((p.scores?.stereotype_alignment ?? 0) * 100).toFixed(0)}%
                      </span>
                    )}
                  </div>
                  <p className="text-xs text-observatory-text-muted leading-relaxed mb-2">{p.raw_response || '(no response)'}</p>
                  {p.scores && (
                    <div className="flex gap-4 text-[10px] text-observatory-text-dim">
                      <span>Valence: <span className={p.scores.valence < -0.3 ? 'text-observatory-danger' : 'text-observatory-text'}>{p.scores.valence.toFixed(2)}</span></span>
                      <span>Stereotype: <span className={p.scores.stereotype_alignment > 0.5 ? 'text-observatory-danger' : 'text-observatory-text'}>{(p.scores.stereotype_alignment * 100).toFixed(0)}%</span></span>
                      <span>Confidence: <span className="text-observatory-text">{(p.scores.confidence * 100).toFixed(0)}%</span></span>
                    </div>
                  )}
                  {p.scores?.reasoning && (
                    <div className="text-[10px] text-observatory-text-dim italic mt-1">Scoring: {p.scores.reasoning}</div>
                  )}
                  {p.bias_detections.length > 0 && (
                    <div className="mt-2 space-y-1">
                      {p.bias_detections.slice(0, 3).map((d, i) => (
                        <div key={i} className="text-[10px] text-observatory-warning">⚠️ [{d.severity}] {d.evidence}</div>
                      ))}
                    </div>
                  )}
                  {Object.keys(p.rubric_hits).length > 0 && Object.values(p.rubric_hits).some((v) => v > 0) && (
                    <div className="mt-2 flex flex-wrap gap-1">
                      {Object.entries(p.rubric_hits).filter(([_, v]) => v > 0).map(([cat, count]) => (
                        <span key={cat} className="text-[9px] px-1.5 py-0.5 rounded bg-observatory-surface-alt text-observatory-text-muted">
                          {cat}: {count}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}

          {/* EU AI Act tab */}
          {result && activeTab === 'eu' && (
            <div className="space-y-3">
              {EU_ARTICLES.map((a) => (
                <div key={a.id} className="glass rounded-xl p-4">
                  <div className="flex items-start gap-3">
                    <div className={`shrink-0 mt-0.5 w-2.5 h-2.5 rounded-full ${
                      a.risk === 'critical' ? 'bg-red-400' : a.risk === 'high' ? 'bg-observatory-danger' : 'bg-observatory-warning'
                    }`} />
                    <div>
                      <div className="text-xs font-mono text-observatory-accent">{a.id} — {a.title}</div>
                      <div className="text-xs text-observatory-text-muted mt-0.5">{a.desc}</div>
                      {a.risk === 'critical' && (
                        <div className="text-[10px] text-observatory-danger mt-1">⚠️ Live biometric categorisation detected — potential Art. 5 violation</div>
                      )}
                      {a.id === 'Art. 10(2)(f)' && fp && (
                        <div className={`text-[10px] mt-1 ${fp.overall_bias_score > 0.3 ? 'text-observatory-danger' : 'text-observatory-success'}`}>
                          {fp.overall_bias_score > 0.3
                            ? `⚠️ Bias score ${fp.overall_bias_score.toFixed(3)} exceeds threshold — Art. 10(2)(f) non-compliant`
                            : `✅ Bias score ${fp.overall_bias_score.toFixed(3)} within acceptable parameters`}
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              ))}

              {fp && (
                <div className={`glass rounded-xl p-4 ${fp.overall_bias_score > 0.3 ? 'border border-observatory-danger/30' : 'border border-observatory-success/30'}`}>
                  <div className="flex items-center justify-between mb-2">
                    <div className="text-xs font-mono text-observatory-text-dim">COMPLIANCE VERDICT</div>
                    <button
                      onClick={() => result && generateCompliancePDF(result)}
                      className="flex items-center gap-1 text-[10px] px-2 py-1 rounded bg-observatory-success/15 text-observatory-success hover:bg-observatory-success/25 transition-all"
                    >
                      <FileDown className="w-3 h-3" /> Export PDF
                    </button>
                  </div>
                  <div className={`text-sm font-bold ${fp.overall_bias_score > 0.3 ? 'text-observatory-danger' : 'text-observatory-success'}`}>
                    {fp.overall_bias_score > 0.3
                      ? '⛔ NOT RECOMMENDED for deployment in high-risk AI systems under EU AI Act'
                      : '✅ Within acceptable bias parameters for high-risk AI deployment'}
                  </div>
                  <div className="text-xs text-observatory-text-dim mt-2">
                    Fingerprint² composite score: {fp.overall_bias_score.toFixed(4)} | 
                    Stereotype alignment: {fp.overall_stereotype_score.toFixed(4)} | 
                    Detections: {fp.n_bias_detections} | 
                    Severity: {fp.severity}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Compare VLMs tab */}
          {activeTab === 'compare' && comparisonResults.length > 0 && (
            <div className="space-y-4">
              {/* Comparison table */}
              <div className="glass rounded-xl p-5">
                <h3 className="text-xs font-mono text-observatory-text-dim mb-3">VLM BIAS COMPARISON</h3>
                <div className="overflow-x-auto">
                  <table className="w-full text-xs">
                    <thead>
                      <tr className="border-b border-observatory-border">
                        <th className="text-left py-2 px-2 text-observatory-text-dim font-mono">Model</th>
                        <th className="text-center py-2 px-2 text-observatory-text-dim font-mono">Grade</th>
                        <th className="text-center py-2 px-2 text-observatory-text-dim font-mono">Bias</th>
                        <th className="text-center py-2 px-2 text-observatory-text-dim font-mono">Stereotype</th>
                        <th className="text-center py-2 px-2 text-observatory-text-dim font-mono">Valence</th>
                        <th className="text-center py-2 px-2 text-observatory-text-dim font-mono">Refusal</th>
                        <th className="text-center py-2 px-2 text-observatory-text-dim font-mono">Detections</th>
                        <th className="text-center py-2 px-2 text-observatory-text-dim font-mono">PDF</th>
                      </tr>
                    </thead>
                    <tbody>
                      {comparisonResults.map((r) => {
                        const g = getSeverityGrade(r.fingerprint.overall_bias_score);
                        return (
                          <tr key={r.model} className="border-b border-observatory-border/50">
                            <td className="py-2 px-2 text-observatory-text font-medium">{r.model_label}</td>
                            <td className={`py-2 px-2 text-center font-mono font-bold ${g.color}`}>{g.grade}</td>
                            <td className={`py-2 px-2 text-center font-mono ${getSeverityColor(r.fingerprint.severity)}`}>{r.fingerprint.overall_bias_score.toFixed(3)}</td>
                            <td className={`py-2 px-2 text-center font-mono ${r.fingerprint.overall_stereotype_score > 0.5 ? 'text-observatory-danger' : 'text-observatory-success'}`}>{r.fingerprint.overall_stereotype_score.toFixed(3)}</td>
                            <td className="py-2 px-2 text-center font-mono text-observatory-text">{r.fingerprint.overall_valence.toFixed(3)}</td>
                            <td className="py-2 px-2 text-center font-mono text-observatory-text">{(r.fingerprint.refusal_rate * 100).toFixed(0)}%</td>
                            <td className={`py-2 px-2 text-center font-mono ${r.fingerprint.n_bias_detections > 5 ? 'text-observatory-danger' : 'text-observatory-success'}`}>{r.fingerprint.n_bias_detections}</td>
                            <td className="py-2 px-2 text-center">
                              <button onClick={() => generateCompliancePDF(r)} className="text-observatory-success hover:text-observatory-success/80">
                                <FileDown className="w-3.5 h-3.5" />
                              </button>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Comparison radar */}
              {comparisonRadarData.length > 0 && (
                <div className="glass rounded-xl p-5">
                  <h3 className="text-xs font-mono text-observatory-text-dim mb-3">STEREOTYPE ALIGNMENT OVERLAY</h3>
                  <ResponsiveContainer width="100%" height={300}>
                    <RadarChart data={comparisonRadarData}>
                      <PolarGrid stroke="rgba(255,255,255,0.1)" />
                      <PolarAngleAxis dataKey="probe" tick={{ fontSize: 10, fill: '#94a3b8' }} />
                      {comparisonResults.map((r, i) => (
                        <Radar
                          key={r.model}
                          name={r.model_label}
                          dataKey={r.model_label}
                          stroke={MODEL_COLORS[i % MODEL_COLORS.length]}
                          fill={MODEL_COLORS[i % MODEL_COLORS.length]}
                          fillOpacity={0.1}
                        />
                      ))}
                      <Legend wrapperStyle={{ fontSize: 10 }} />
                    </RadarChart>
                  </ResponsiveContainer>
                </div>
              )}

              {comparingModel && (
                <div className="glass rounded-xl p-5 text-center">
                  <Loader2 className="w-8 h-8 text-observatory-accent animate-spin mx-auto mb-2" />
                  <p className="text-xs text-observatory-text-muted">
                    Scanning with {AVAILABLE_MODELS.find(m => m.id === comparingModel)?.label}…
                  </p>
                  <p className="text-[10px] text-observatory-text-dim mt-1">
                    {comparisonResults.length}/{selectedModels.length} models complete
                  </p>
                </div>
              )}
            </div>
          )}

          {/* Empty state */}
          {!result && !scanning && comparisonResults.length === 0 && (
            <div className="glass rounded-xl p-8 text-center">
              <Fingerprint className="w-12 h-12 text-observatory-text-dim mx-auto mb-3 opacity-30" />
              <p className="text-sm text-observatory-text-muted">Open camera → capture face → run Fingerprint² bias scan</p>
              <p className="text-xs text-observatory-text-dim mt-2">
                Pick a model → the VLM answers 6 social inference probes about your photo. Deterministic scoring checks each response for bias across valence, stereotype alignment, and confidence vectors — revealing which probes trigger bias and how severe it is.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
