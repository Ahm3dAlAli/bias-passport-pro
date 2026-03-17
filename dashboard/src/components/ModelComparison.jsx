import React from 'react'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
} from 'recharts'
import { Trophy, TrendingUp, TrendingDown } from 'lucide-react'

const modelColors = {
  'gpt-4o': '#10b981',
  'claude-sonnet': '#6366f1',
  'llava-1.5': '#f59e0b',
}

const probeLabels = {
  occupation: 'Occupation',
  education: 'Education',
  dating_profile: 'Dating',
  leadership: 'Leadership',
  neighborhood: 'Neighborhood',
  trustworthiness: 'Trust',
}

export default function ModelComparison({ fingerprints }) {
  const models = Object.values(fingerprints)

  // Prepare overall comparison data
  const overallData = models.map((fp) => ({
    name: fp.model_name,
    'Overall Bias': (fp.overall_bias_score * 100).toFixed(1),
    'Valence': (fp.valence_bias * 100).toFixed(1),
    'Stereotype': (fp.stereotype_bias * 100).toFixed(1),
    'Confidence': (fp.confidence_bias * 100).toFixed(1),
  }))

  // Prepare radar comparison data
  const radarData = Object.keys(probeLabels).map((probe) => {
    const dataPoint = { probe: probeLabels[probe] }
    models.forEach((fp) => {
      dataPoint[fp.model_name] = (fp.radar_dimensions[probe] || 0) * 100
    })
    return dataPoint
  })

  // Rankings
  const sortedByBias = [...models].sort((a, b) => a.overall_bias_score - b.overall_bias_score)

  return (
    <div className="space-y-8">
      {/* Rankings */}
      <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
        <h2 className="text-lg font-semibold text-slate-900 mb-4 flex items-center gap-2">
          <Trophy className="w-5 h-5 text-yellow-500" />
          Model Rankings (Lower Bias = Better)
        </h2>
        <div className="space-y-3">
          {sortedByBias.map((fp, index) => (
            <div
              key={fp.model_id}
              className={`flex items-center justify-between p-4 rounded-lg ${
                index === 0 ? 'bg-green-50 border border-green-200' :
                index === sortedByBias.length - 1 ? 'bg-red-50 border border-red-200' :
                'bg-slate-50 border border-slate-200'
              }`}
            >
              <div className="flex items-center gap-4">
                <span className={`text-2xl font-bold ${
                  index === 0 ? 'text-green-600' :
                  index === sortedByBias.length - 1 ? 'text-red-600' :
                  'text-slate-400'
                }`}>
                  #{index + 1}
                </span>
                <div>
                  <p className="font-semibold text-slate-900">{fp.model_name}</p>
                  <p className="text-sm text-slate-500">
                    Refusal rate: {(fp.refusal_rate * 100).toFixed(1)}%
                  </p>
                </div>
              </div>
              <div className="text-right">
                <p className="text-2xl font-bold" style={{ color: modelColors[fp.model_id] }}>
                  {(fp.overall_bias_score * 100).toFixed(0)}%
                </p>
                <p className="text-sm text-slate-500">overall bias</p>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Radar Comparison */}
      <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
        <h2 className="text-lg font-semibold text-slate-900 mb-4">Probe-by-Probe Comparison</h2>
        <ResponsiveContainer width="100%" height={400}>
          <RadarChart cx="50%" cy="50%" outerRadius="80%" data={radarData}>
            <PolarGrid stroke="#e2e8f0" />
            <PolarAngleAxis dataKey="probe" tick={{ fill: '#475569', fontSize: 11 }} />
            <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fill: '#94a3b8', fontSize: 10 }} />
            {models.map((fp) => (
              <Radar
                key={fp.model_id}
                name={fp.model_name}
                dataKey={fp.model_name}
                stroke={modelColors[fp.model_id]}
                fill={modelColors[fp.model_id]}
                fillOpacity={0.15}
                strokeWidth={2}
              />
            ))}
            <Legend />
            <Tooltip
              content={({ active, payload, label }) => {
                if (active && payload && payload.length) {
                  return (
                    <div className="bg-white shadow-lg rounded-lg p-3 border border-slate-200">
                      <p className="font-semibold text-slate-900 mb-2">{label}</p>
                      {payload.map((entry) => (
                        <div key={entry.name} className="flex justify-between gap-4 text-sm">
                          <span style={{ color: entry.color }}>{entry.name}</span>
                          <span className="font-medium">{Number(entry.value).toFixed(1)}%</span>
                        </div>
                      ))}
                    </div>
                  )
                }
                return null
              }}
            />
          </RadarChart>
        </ResponsiveContainer>
      </div>

      {/* Bar Chart Comparison */}
      <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
        <h2 className="text-lg font-semibold text-slate-900 mb-4">Bias Dimension Breakdown</h2>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={overallData} layout="horizontal">
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis dataKey="name" tick={{ fill: '#475569', fontSize: 12 }} />
            <YAxis domain={[0, 100]} tick={{ fill: '#94a3b8', fontSize: 10 }} unit="%" />
            <Tooltip
              contentStyle={{
                backgroundColor: 'white',
                border: '1px solid #e2e8f0',
                borderRadius: '8px',
              }}
            />
            <Legend />
            <Bar dataKey="Valence" fill="#ef4444" radius={[4, 4, 0, 0]} />
            <Bar dataKey="Stereotype" fill="#f59e0b" radius={[4, 4, 0, 0]} />
            <Bar dataKey="Confidence" fill="#6366f1" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Key Insights */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
          <h3 className="text-md font-semibold text-slate-900 mb-3 flex items-center gap-2">
            <TrendingDown className="w-4 h-4 text-green-500" />
            Lowest Bias Areas
          </h3>
          <ul className="space-y-2 text-sm text-slate-600">
            <li className="flex justify-between">
              <span>Leadership (Claude Sonnet)</span>
              <span className="font-medium text-green-600">22%</span>
            </li>
            <li className="flex justify-between">
              <span>Trustworthiness (Claude Sonnet)</span>
              <span className="font-medium text-green-600">25%</span>
            </li>
            <li className="flex justify-between">
              <span>Education (Claude Sonnet)</span>
              <span className="font-medium text-green-600">28%</span>
            </li>
          </ul>
        </div>

        <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
          <h3 className="text-md font-semibold text-slate-900 mb-3 flex items-center gap-2">
            <TrendingUp className="w-4 h-4 text-red-500" />
            Highest Bias Areas
          </h3>
          <ul className="space-y-2 text-sm text-slate-600">
            <li className="flex justify-between">
              <span>Dating Profile (LLaVA)</span>
              <span className="font-medium text-red-600">72%</span>
            </li>
            <li className="flex justify-between">
              <span>Occupation (LLaVA)</span>
              <span className="font-medium text-red-600">68%</span>
            </li>
            <li className="flex justify-between">
              <span>Neighborhood (LLaVA)</span>
              <span className="font-medium text-red-600">65%</span>
            </li>
          </ul>
        </div>
      </div>
    </div>
  )
}
