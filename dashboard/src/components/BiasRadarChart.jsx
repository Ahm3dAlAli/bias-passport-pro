import React from 'react'
import {
  Radar,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ResponsiveContainer,
  Tooltip,
} from 'recharts'

const probeLabels = {
  occupation: 'Occupation',
  education: 'Education',
  dating_profile: 'Dating Profile',
  leadership: 'Leadership',
  neighborhood: 'Neighborhood',
  trustworthiness: 'Trustworthiness',
}

const probeDescriptions = {
  occupation: 'Economic stereotype attribution based on appearance',
  education: 'Class and education level inference bias',
  dating_profile: 'Desirability and gendered language patterns',
  leadership: 'Authority and competence attribution bias',
  neighborhood: 'Geographic and economic stereotyping',
  trustworthiness: 'Direct prejudice and trust attribution',
}

export default function BiasRadarChart({ data, color = '#6366f1' }) {
  const chartData = Object.entries(data).map(([key, value]) => ({
    probe: probeLabels[key] || key,
    value: value * 100,
    fullMark: 100,
    description: probeDescriptions[key] || '',
  }))

  return (
    <div className="w-full">
      <ResponsiveContainer width="100%" height={400}>
        <RadarChart cx="50%" cy="50%" outerRadius="80%" data={chartData}>
          <PolarGrid stroke="#e2e8f0" />
          <PolarAngleAxis
            dataKey="probe"
            tick={{ fill: '#475569', fontSize: 12 }}
          />
          <PolarRadiusAxis
            angle={30}
            domain={[0, 100]}
            tick={{ fill: '#94a3b8', fontSize: 10 }}
            tickFormatter={(v) => `${v}%`}
          />
          <Radar
            name="Bias Score"
            dataKey="value"
            stroke={color}
            fill={color}
            fillOpacity={0.3}
            strokeWidth={2}
          />
          <Tooltip
            content={({ active, payload }) => {
              if (active && payload && payload.length) {
                const data = payload[0].payload
                return (
                  <div className="bg-white shadow-lg rounded-lg p-3 border border-slate-200">
                    <p className="font-semibold text-slate-900">{data.probe}</p>
                    <p className="text-sm text-slate-600">{data.description}</p>
                    <p className="text-lg font-bold mt-1" style={{ color }}>
                      {data.value.toFixed(1)}% bias
                    </p>
                  </div>
                )
              }
              return null
            }}
          />
        </RadarChart>
      </ResponsiveContainer>

      {/* Legend */}
      <div className="mt-4 grid grid-cols-2 md:grid-cols-3 gap-3">
        {chartData.map((item) => (
          <div key={item.probe} className="flex items-center gap-2 text-sm">
            <div
              className="w-3 h-3 rounded-full"
              style={{ backgroundColor: getBiasLevelColor(item.value) }}
            />
            <span className="text-slate-600">{item.probe}:</span>
            <span className="font-medium text-slate-900">{item.value.toFixed(0)}%</span>
          </div>
        ))}
      </div>
    </div>
  )
}

function getBiasLevelColor(value) {
  if (value < 25) return '#22c55e'
  if (value < 50) return '#eab308'
  if (value < 75) return '#f97316'
  return '#ef4444'
}
