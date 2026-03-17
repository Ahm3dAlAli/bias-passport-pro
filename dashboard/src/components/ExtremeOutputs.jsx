import React, { useState } from 'react'
import { AlertTriangle, ArrowUpRight, ArrowDownRight, Filter } from 'lucide-react'

export default function ExtremeOutputs({ outputs }) {
  const [filterProbe, setFilterProbe] = useState('all')
  const [sortBy, setSortBy] = useState('stereotype')

  const probeTypes = ['all', ...new Set(outputs.map((o) => o.probe_type))]

  const filteredOutputs = outputs
    .filter((o) => filterProbe === 'all' || o.probe_type === filterProbe)
    .sort((a, b) => {
      if (sortBy === 'stereotype') return b.stereotype - a.stereotype
      if (sortBy === 'valence') return Math.abs(b.valence) - Math.abs(a.valence)
      if (sortBy === 'confidence') return b.confidence - a.confidence
      return 0
    })

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-amber-50 border border-amber-200 rounded-xl p-4">
        <div className="flex items-center gap-3">
          <AlertTriangle className="w-5 h-5 text-amber-600" />
          <div>
            <h3 className="font-semibold text-amber-900">Extreme Output Viewer</h3>
            <p className="text-sm text-amber-700">
              These responses show the most biased outputs from the model. Use these for qualitative analysis.
            </p>
          </div>
        </div>
      </div>

      {/* Filters */}
      <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-4">
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-2">
            <Filter className="w-4 h-4 text-slate-400" />
            <span className="text-sm font-medium text-slate-700">Probe Type:</span>
            <select
              value={filterProbe}
              onChange={(e) => setFilterProbe(e.target.value)}
              className="px-3 py-1.5 border border-slate-300 rounded-lg text-sm"
            >
              {probeTypes.map((type) => (
                <option key={type} value={type}>
                  {type === 'all' ? 'All Probes' : type.replace('_', ' ')}
                </option>
              ))}
            </select>
          </div>

          <div className="flex items-center gap-2">
            <span className="text-sm font-medium text-slate-700">Sort by:</span>
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value)}
              className="px-3 py-1.5 border border-slate-300 rounded-lg text-sm"
            >
              <option value="stereotype">Highest Stereotype</option>
              <option value="valence">Extreme Valence</option>
              <option value="confidence">Highest Confidence</option>
            </select>
          </div>
        </div>
      </div>

      {/* Output Cards */}
      <div className="space-y-4">
        {filteredOutputs.map((output, index) => (
          <OutputCard key={index} output={output} index={index + 1} />
        ))}
      </div>

      {filteredOutputs.length === 0 && (
        <div className="bg-slate-50 rounded-xl p-8 text-center">
          <p className="text-slate-500">No outputs match the current filters.</p>
        </div>
      )}
    </div>
  )
}

function OutputCard({ output, index }) {
  const [expanded, setExpanded] = useState(false)

  const getValenceIndicator = (valence) => {
    if (valence > 0.3) {
      return { icon: ArrowUpRight, color: 'text-green-500', label: 'Positive' }
    } else if (valence < -0.3) {
      return { icon: ArrowDownRight, color: 'text-red-500', label: 'Negative' }
    }
    return { icon: null, color: 'text-slate-500', label: 'Neutral' }
  }

  const valenceInfo = getValenceIndicator(output.valence)
  const ValenceIcon = valenceInfo.icon

  return (
    <div className="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-slate-100">
        <div className="flex items-center gap-3">
          <span className="text-lg font-bold text-slate-300">#{index}</span>
          <div>
            <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-indigo-100 text-indigo-800 capitalize">
              {output.probe_type.replace('_', ' ')}
            </span>
            <span className="ml-2 text-sm text-slate-500">Image: {output.image_id}</span>
          </div>
        </div>

        {/* Demographic Badge */}
        <div className="flex items-center gap-2">
          <span className="text-xs text-slate-500">
            {output.demographic.gender}, {output.demographic.age_range}, {output.demographic.race}
          </span>
        </div>
      </div>

      {/* Scores */}
      <div className="grid grid-cols-3 divide-x divide-slate-100 border-b border-slate-100">
        <div className="p-4 text-center">
          <p className="text-xs text-slate-500 mb-1">Valence</p>
          <div className="flex items-center justify-center gap-1">
            {ValenceIcon && <ValenceIcon className={`w-4 h-4 ${valenceInfo.color}`} />}
            <span className={`text-lg font-semibold ${valenceInfo.color}`}>
              {output.valence.toFixed(2)}
            </span>
          </div>
        </div>
        <div className="p-4 text-center">
          <p className="text-xs text-slate-500 mb-1">Stereotype</p>
          <span className={`text-lg font-semibold ${
            output.stereotype > 0.7 ? 'text-red-500' :
            output.stereotype > 0.5 ? 'text-orange-500' : 'text-green-500'
          }`}>
            {(output.stereotype * 100).toFixed(0)}%
          </span>
        </div>
        <div className="p-4 text-center">
          <p className="text-xs text-slate-500 mb-1">Confidence</p>
          <span className={`text-lg font-semibold ${
            output.confidence > 0.8 ? 'text-slate-900' : 'text-slate-500'
          }`}>
            {(output.confidence * 100).toFixed(0)}%
          </span>
        </div>
      </div>

      {/* Response */}
      <div className="p-4">
        <p className="text-sm text-slate-500 mb-2">Model Response:</p>
        <div
          className={`bg-slate-50 rounded-lg p-3 text-sm text-slate-700 ${
            !expanded && 'line-clamp-3'
          }`}
        >
          "{output.response}"
        </div>
        {output.response.length > 150 && (
          <button
            onClick={() => setExpanded(!expanded)}
            className="text-sm text-indigo-600 hover:text-indigo-700 mt-2"
          >
            {expanded ? 'Show less' : 'Show more'}
          </button>
        )}
      </div>

      {/* Analysis Footer */}
      <div className="bg-slate-50 px-4 py-3 text-xs text-slate-600">
        <strong>Why flagged:</strong>{' '}
        {output.stereotype > 0.7 && 'High stereotype alignment. '}
        {Math.abs(output.valence) > 0.3 && 'Extreme valence score. '}
        {output.confidence > 0.8 && 'High confidence in potentially biased assertion.'}
      </div>
    </div>
  )
}
