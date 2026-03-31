import { useState } from 'react';
import { MODEL_RESULTS, getSeverityGrade } from '@/data/benchmarkData';
import type { ModelResult } from '@/data/benchmarkData';
import ComplianceRadarChart from './ComplianceRadarChart';
import ArticleCompliance from './ArticleCompliance';
import DemographicBreakdown from './DemographicBreakdown';
import { Shield, ChevronDown } from 'lucide-react';

const SCANNABLE_MODELS = MODEL_RESULTS.filter(m => m.composite_score > 0);

export default function ModelComplianceReport() {
  const [selectedModel, setSelectedModel] = useState<ModelResult | null>(null);
  const [dropdownOpen, setDropdownOpen] = useState(false);

  return (
    <div className="space-y-6">
      {/* Model Selector */}
      <div className="card">
        <div className="card-header">Select Model for Compliance Assessment</div>
        <p className="text-sm text-observatory-text-muted mb-4">
          Choose a VLM from the Fingerprint² benchmark. Results are based on FHIBE dataset evaluation across 3,000 images and 6 geographic regions.
        </p>
        <div className="relative">
          <button
            onClick={() => setDropdownOpen(!dropdownOpen)}
            className="w-full flex items-center justify-between bg-observatory-bg/60 border border-observatory-border rounded-xl px-5 py-3.5 text-sm text-observatory-text hover:border-observatory-accent/40 transition-colors"
          >
            {selectedModel ? (
              <div className="flex items-center gap-3">
                <Shield className="w-4 h-4 text-observatory-accent" />
                <span className="font-medium">{selectedModel.name}</span>
                <span className="text-observatory-text-dim">· {selectedModel.provider} · {selectedModel.params}</span>
                <span className={`text-xs px-2 py-0.5 rounded ${getSeverityGrade(selectedModel.composite_score).color} bg-observatory-surface-alt`}>
                  {getSeverityGrade(selectedModel.composite_score).grade}
                </span>
              </div>
            ) : (
              <span className="text-observatory-text-dim">Choose a model…</span>
            )}
            <ChevronDown className={`w-4 h-4 text-observatory-text-dim transition-transform ${dropdownOpen ? 'rotate-180' : ''}`} />
          </button>

          {dropdownOpen && (
            <div className="absolute top-full left-0 right-0 mt-2 bg-observatory-surface border border-observatory-border rounded-xl overflow-hidden z-50 shadow-xl">
              {SCANNABLE_MODELS.map(model => {
                const grade = getSeverityGrade(model.composite_score);
                return (
                  <button
                    key={model.id}
                    onClick={() => { setSelectedModel(model); setDropdownOpen(false); }}
                    className="w-full flex items-center gap-3 px-5 py-3 text-left hover:bg-observatory-surface-alt transition-colors text-sm"
                  >
                    <span className="font-medium text-observatory-text flex-1">{model.name}</span>
                    <span className="text-observatory-text-dim text-xs">{model.provider}</span>
                    <span className="text-observatory-text-dim text-xs font-mono">{model.params}</span>
                    <span className={`text-xs font-mono font-bold ${grade.color}`}>{grade.grade}</span>
                    <span className="text-xs text-observatory-text-dim">{(model.composite_score * 100).toFixed(1)}%</span>
                  </button>
                );
              })}
            </div>
          )}
        </div>
      </div>

      {/* Report */}
      {selectedModel && (
        <>
          {/* Summary Banner */}
          <div className={`card border ${selectedModel.composite_score > 0.25 ? 'border-observatory-danger/30 bg-observatory-danger/5' : selectedModel.composite_score > 0.12 ? 'border-observatory-warning/30 bg-observatory-warning/5' : 'border-observatory-success/30 bg-observatory-success/5'}`}>
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-lg font-bold text-observatory-text">{selectedModel.name}</h3>
                <p className="text-xs text-observatory-text-dim mt-1">{selectedModel.provider} · {selectedModel.family} · {selectedModel.params} · HF: {selectedModel.hf_id}</p>
              </div>
              <div className="text-right">
                <div className={`text-2xl font-mono font-bold ${getSeverityGrade(selectedModel.composite_score).color}`}>
                  {getSeverityGrade(selectedModel.composite_score).grade}
                </div>
                <div className="text-xs text-observatory-text-dim">
                  Composite: {(selectedModel.composite_score * 100).toFixed(1)}% · {selectedModel.severity}
                </div>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <ComplianceRadarChart model={selectedModel} />
            <ArticleCompliance model={selectedModel} />
          </div>

          <DemographicBreakdown model={selectedModel} />

          {/* Fingerprint² Methodology Note */}
          <div className="card bg-observatory-accent/5 border border-observatory-accent/20">
            <div className="text-xs font-mono text-observatory-accent mb-2">METHODOLOGY</div>
            <p className="text-sm text-observatory-text-muted">
              Compliance assessment based on Fingerprint² Social Inference Battery (5 probes × 3,000 FHIBE images × 6 regions).
              Scores are deterministic: VADER sentiment, TF-IDF stereotype alignment, and Cohen's d effect sizes.
              This is an indicative assessment — formal EU AI Act conformity requires accredited third-party audits.
            </p>
          </div>
        </>
      )}
    </div>
  );
}
