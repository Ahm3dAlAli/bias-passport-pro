import React, { useRef } from 'react'
import { FileText, Download, Printer, Award, AlertCircle, CheckCircle } from 'lucide-react'
import BiasRadarChart from './BiasRadarChart'

export default function BiasPassport({ fingerprint }) {
  const passportRef = useRef(null)

  const handlePrint = () => {
    window.print()
  }

  const getBiasGrade = (score) => {
    if (score < 0.2) return { grade: 'A', label: 'Excellent', color: 'text-green-600 bg-green-50' }
    if (score < 0.35) return { grade: 'B', label: 'Good', color: 'text-blue-600 bg-blue-50' }
    if (score < 0.5) return { grade: 'C', label: 'Fair', color: 'text-yellow-600 bg-yellow-50' }
    if (score < 0.65) return { grade: 'D', label: 'Poor', color: 'text-orange-600 bg-orange-50' }
    return { grade: 'F', label: 'Failing', color: 'text-red-600 bg-red-50' }
  }

  const grade = getBiasGrade(fingerprint.overall_bias_score)

  const findings = [
    {
      type: fingerprint.valence_bias > 0.4 ? 'concern' : 'positive',
      text: fingerprint.valence_bias > 0.4
        ? `High valence disparity (${(fingerprint.valence_bias * 100).toFixed(0)}%) indicates differential positive/negative treatment across demographics.`
        : `Relatively balanced valence scores (${(fingerprint.valence_bias * 100).toFixed(0)}%) across demographic groups.`,
    },
    {
      type: fingerprint.stereotype_bias > 0.4 ? 'concern' : 'positive',
      text: fingerprint.stereotype_bias > 0.4
        ? `Elevated stereotype alignment (${(fingerprint.stereotype_bias * 100).toFixed(0)}%) suggests reinforcement of existing stereotypes.`
        : `Lower stereotype alignment (${(fingerprint.stereotype_bias * 100).toFixed(0)}%) indicates less reliance on stereotypical patterns.`,
    },
    {
      type: fingerprint.refusal_rate > 0.2 ? 'positive' : 'neutral',
      text: fingerprint.refusal_rate > 0.2
        ? `High refusal rate (${(fingerprint.refusal_rate * 100).toFixed(0)}%) suggests strong safety guardrails on sensitive questions.`
        : `Moderate refusal rate (${(fingerprint.refusal_rate * 100).toFixed(0)}%) on demographic-sensitive probes.`,
    },
  ]

  // Find most/least biased probe areas
  const probeScores = Object.entries(fingerprint.radar_dimensions)
    .map(([probe, score]) => ({ probe, score }))
    .sort((a, b) => b.score - a.score)

  return (
    <div className="space-y-6">
      {/* Action Bar */}
      <div className="flex justify-end gap-3">
        <button
          onClick={handlePrint}
          className="flex items-center gap-2 px-4 py-2 bg-white border border-slate-300 rounded-lg text-sm font-medium text-slate-700 hover:bg-slate-50"
        >
          <Printer className="w-4 h-4" />
          Print Passport
        </button>
        <button
          className="flex items-center gap-2 px-4 py-2 bg-indigo-600 rounded-lg text-sm font-medium text-white hover:bg-indigo-700"
        >
          <Download className="w-4 h-4" />
          Export PDF
        </button>
      </div>

      {/* Passport Document */}
      <div
        ref={passportRef}
        className="bg-white rounded-xl shadow-lg border-2 border-slate-200 overflow-hidden print:shadow-none print:border-0"
      >
        {/* Header */}
        <div className="bg-gradient-to-r from-indigo-600 to-purple-600 text-white p-8">
          <div className="flex items-start justify-between">
            <div>
              <div className="flex items-center gap-2 text-indigo-200 text-sm mb-2">
                <FileText className="w-4 h-4" />
                BIAS PASSPORT
              </div>
              <h1 className="text-3xl font-bold">{fingerprint.model_name}</h1>
              <p className="text-indigo-200 mt-1">Model ID: {fingerprint.model_id}</p>
            </div>
            <div className={`px-6 py-4 rounded-xl ${grade.color}`}>
              <p className="text-xs font-medium opacity-70">BIAS GRADE</p>
              <p className="text-5xl font-bold">{grade.grade}</p>
              <p className="text-sm font-medium">{grade.label}</p>
            </div>
          </div>
        </div>

        {/* Summary Stats */}
        <div className="grid grid-cols-4 divide-x divide-slate-200 border-b border-slate-200">
          <div className="p-6 text-center">
            <p className="text-sm text-slate-500">Overall Bias</p>
            <p className="text-3xl font-bold text-slate-900">
              {(fingerprint.overall_bias_score * 100).toFixed(0)}%
            </p>
          </div>
          <div className="p-6 text-center">
            <p className="text-sm text-slate-500">Valence Bias</p>
            <p className="text-3xl font-bold text-slate-900">
              {(fingerprint.valence_bias * 100).toFixed(0)}%
            </p>
          </div>
          <div className="p-6 text-center">
            <p className="text-sm text-slate-500">Stereotype Bias</p>
            <p className="text-3xl font-bold text-slate-900">
              {(fingerprint.stereotype_bias * 100).toFixed(0)}%
            </p>
          </div>
          <div className="p-6 text-center">
            <p className="text-sm text-slate-500">Refusal Rate</p>
            <p className="text-3xl font-bold text-slate-900">
              {(fingerprint.refusal_rate * 100).toFixed(0)}%
            </p>
          </div>
        </div>

        {/* Content Grid */}
        <div className="grid grid-cols-2 divide-x divide-slate-200">
          {/* Left: Radar Chart */}
          <div className="p-6">
            <h2 className="text-lg font-semibold text-slate-900 mb-4">Bias Profile</h2>
            <BiasRadarChart data={fingerprint.radar_dimensions} />
          </div>

          {/* Right: Findings */}
          <div className="p-6">
            <h2 className="text-lg font-semibold text-slate-900 mb-4">Key Findings</h2>

            <div className="space-y-4">
              {findings.map((finding, index) => (
                <div
                  key={index}
                  className={`flex gap-3 p-3 rounded-lg ${
                    finding.type === 'concern' ? 'bg-red-50' :
                    finding.type === 'positive' ? 'bg-green-50' : 'bg-slate-50'
                  }`}
                >
                  {finding.type === 'concern' ? (
                    <AlertCircle className="w-5 h-5 text-red-500 flex-shrink-0 mt-0.5" />
                  ) : (
                    <CheckCircle className="w-5 h-5 text-green-500 flex-shrink-0 mt-0.5" />
                  )}
                  <p className="text-sm text-slate-700">{finding.text}</p>
                </div>
              ))}
            </div>

            {/* Probe Rankings */}
            <div className="mt-6">
              <h3 className="text-md font-semibold text-slate-900 mb-3">Probe Analysis</h3>
              <div className="space-y-2">
                {probeScores.slice(0, 3).map((item, index) => (
                  <div key={item.probe} className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <span className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold ${
                        index === 0 ? 'bg-red-100 text-red-700' :
                        index === 1 ? 'bg-orange-100 text-orange-700' :
                        'bg-yellow-100 text-yellow-700'
                      }`}>
                        {index + 1}
                      </span>
                      <span className="text-sm text-slate-700 capitalize">
                        {item.probe.replace('_', ' ')}
                      </span>
                    </div>
                    <span className="text-sm font-semibold text-slate-900">
                      {(item.score * 100).toFixed(0)}%
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="bg-slate-50 px-8 py-4 border-t border-slate-200">
          <div className="flex items-center justify-between text-xs text-slate-500">
            <div className="flex items-center gap-2">
              <Award className="w-4 h-4" />
              <span>Fingerprint Squared - Ethical AI Assessment Framework</span>
            </div>
            <div>
              <span>Generated: {new Date().toLocaleDateString()}</span>
              <span className="mx-2">|</span>
              <span>Total Probes: {fingerprint.total_probes}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Methodology Note */}
      <div className="bg-slate-50 rounded-xl border border-slate-200 p-6">
        <h3 className="text-md font-semibold text-slate-900 mb-2">Methodology</h3>
        <p className="text-sm text-slate-600">
          This bias passport was generated using the Social Inference Battery, a set of 6 probes
          designed to elicit bias patterns in vision-language models. Each response was scored
          by an LLM-as-judge on three dimensions: valence (positive/negative treatment),
          stereotype alignment, and confidence. The overall bias score is a weighted combination
          of disparities across demographic groups.
        </p>
      </div>
    </div>
  )
}
