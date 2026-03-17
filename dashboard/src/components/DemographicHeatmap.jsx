import React, { useState } from 'react'

const demographics = {
  genders: ['male', 'female'],
  ages: ['18-30', '31-50', '51+'],
  races: ['white', 'black', 'asian', 'hispanic'],
}

export default function DemographicHeatmap({ fingerprint }) {
  const [metric, setMetric] = useState('valence')

  const getScore = (gender, age, race) => {
    const key = `${gender}_${age}_${race}`
    const scores = fingerprint.demographic_scores[key]
    if (!scores) return null

    if (metric === 'valence') return scores.mean_valence
    if (metric === 'stereotype') return scores.mean_stereotype_alignment
    return null
  }

  const getColor = (value, type) => {
    if (value === null) return 'bg-slate-100'

    if (type === 'valence') {
      // Valence: -1 (red) to +1 (green), 0 (yellow)
      if (value < -0.3) return 'bg-red-500 text-white'
      if (value < -0.1) return 'bg-red-300'
      if (value < 0.1) return 'bg-yellow-300'
      if (value < 0.3) return 'bg-green-300'
      return 'bg-green-500 text-white'
    } else {
      // Stereotype: 0 (green) to 1 (red)
      if (value < 0.3) return 'bg-green-400 text-white'
      if (value < 0.5) return 'bg-yellow-400'
      if (value < 0.7) return 'bg-orange-400'
      return 'bg-red-500 text-white'
    }
  }

  return (
    <div className="space-y-6">
      {/* Metric Selector */}
      <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-4">
        <div className="flex items-center gap-4">
          <span className="text-sm font-medium text-slate-700">Metric:</span>
          <div className="flex gap-2">
            <button
              onClick={() => setMetric('valence')}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                metric === 'valence'
                  ? 'bg-indigo-100 text-indigo-700'
                  : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
              }`}
            >
              Valence (Positive/Negative)
            </button>
            <button
              onClick={() => setMetric('stereotype')}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                metric === 'stereotype'
                  ? 'bg-indigo-100 text-indigo-700'
                  : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
              }`}
            >
              Stereotype Alignment
            </button>
          </div>
        </div>
      </div>

      {/* Heatmap by Age Group */}
      {demographics.ages.map((age) => (
        <div key={age} className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
          <h3 className="text-md font-semibold text-slate-900 mb-4">
            Age Group: {age}
          </h3>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr>
                  <th className="text-left text-sm font-medium text-slate-500 pb-3 pr-4">
                    Gender / Race
                  </th>
                  {demographics.races.map((race) => (
                    <th
                      key={race}
                      className="text-center text-sm font-medium text-slate-500 pb-3 px-2 capitalize"
                    >
                      {race}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {demographics.genders.map((gender) => (
                  <tr key={gender}>
                    <td className="text-sm font-medium text-slate-700 py-2 pr-4 capitalize">
                      {gender}
                    </td>
                    {demographics.races.map((race) => {
                      const value = getScore(gender, age, race)
                      return (
                        <td key={race} className="p-1">
                          <div
                            className={`rounded-lg p-3 text-center ${getColor(value, metric)}`}
                            title={`${gender}, ${age}, ${race}`}
                          >
                            {value !== null ? (
                              <span className="text-sm font-semibold">
                                {metric === 'valence'
                                  ? value.toFixed(2)
                                  : (value * 100).toFixed(0) + '%'}
                              </span>
                            ) : (
                              <span className="text-slate-400 text-xs">N/A</span>
                            )}
                          </div>
                        </td>
                      )
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      ))}

      {/* Legend */}
      <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-4">
        <h4 className="text-sm font-medium text-slate-700 mb-3">Legend</h4>
        {metric === 'valence' ? (
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-xs text-slate-500">Negative</span>
            <div className="flex">
              <div className="w-8 h-6 bg-red-500 rounded-l" />
              <div className="w-8 h-6 bg-red-300" />
              <div className="w-8 h-6 bg-yellow-300" />
              <div className="w-8 h-6 bg-green-300" />
              <div className="w-8 h-6 bg-green-500 rounded-r" />
            </div>
            <span className="text-xs text-slate-500">Positive</span>
          </div>
        ) : (
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-xs text-slate-500">Low stereotype</span>
            <div className="flex">
              <div className="w-8 h-6 bg-green-400 rounded-l" />
              <div className="w-8 h-6 bg-yellow-400" />
              <div className="w-8 h-6 bg-orange-400" />
              <div className="w-8 h-6 bg-red-500 rounded-r" />
            </div>
            <span className="text-xs text-slate-500">High stereotype</span>
          </div>
        )}
      </div>

      {/* Insights */}
      <div className="bg-gradient-to-r from-indigo-50 to-purple-50 rounded-xl border border-indigo-100 p-6">
        <h3 className="text-md font-semibold text-slate-900 mb-3">Key Insights</h3>
        <ul className="space-y-2 text-sm text-slate-700">
          <li className="flex items-start gap-2">
            <span className="text-indigo-500 mt-1">•</span>
            <span>
              {metric === 'valence'
                ? 'Higher valence disparity between demographic groups indicates differential treatment.'
                : 'High stereotype alignment scores suggest the model is reinforcing existing stereotypes.'}
            </span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-indigo-500 mt-1">•</span>
            <span>
              Look for patterns across rows (gender) and columns (race) to identify systematic biases.
            </span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-indigo-500 mt-1">•</span>
            <span>
              Intersectional disparities (e.g., Black women vs White men) often reveal compounding biases.
            </span>
          </li>
        </ul>
      </div>
    </div>
  )
}
