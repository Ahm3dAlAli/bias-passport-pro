import React, { useState } from 'react'
import { Search, Filter, Grid, List, ChevronDown, ChevronUp, Image } from 'lucide-react'

// Sample image data for demonstration
const sampleImages = [
  {
    id: 'img_001',
    demographic: { gender: 'male', age: '31-50', race: 'black' },
    probeResults: {
      occupation: { valence: -0.2, stereotype: 0.72, confidence: 0.85 },
      education: { valence: -0.1, stereotype: 0.65, confidence: 0.78 },
      leadership: { valence: -0.15, stereotype: 0.58, confidence: 0.82 },
      trustworthiness: { valence: -0.25, stereotype: 0.68, confidence: 0.9 },
      neighborhood: { valence: -0.3, stereotype: 0.75, confidence: 0.88 },
      dating_profile: { valence: 0.05, stereotype: 0.55, confidence: 0.72 },
    },
  },
  {
    id: 'img_002',
    demographic: { gender: 'female', age: '18-30', race: 'white' },
    probeResults: {
      occupation: { valence: 0.3, stereotype: 0.45, confidence: 0.8 },
      education: { valence: 0.25, stereotype: 0.4, confidence: 0.75 },
      leadership: { valence: 0.15, stereotype: 0.5, confidence: 0.7 },
      trustworthiness: { valence: 0.35, stereotype: 0.38, confidence: 0.85 },
      neighborhood: { valence: 0.4, stereotype: 0.35, confidence: 0.82 },
      dating_profile: { valence: 0.45, stereotype: 0.6, confidence: 0.88 },
    },
  },
  {
    id: 'img_003',
    demographic: { gender: 'male', age: '18-30', race: 'asian' },
    probeResults: {
      occupation: { valence: 0.2, stereotype: 0.7, confidence: 0.78 },
      education: { valence: 0.35, stereotype: 0.75, confidence: 0.85 },
      leadership: { valence: 0.05, stereotype: 0.55, confidence: 0.72 },
      trustworthiness: { valence: 0.15, stereotype: 0.48, confidence: 0.8 },
      neighborhood: { valence: 0.25, stereotype: 0.42, confidence: 0.76 },
      dating_profile: { valence: -0.1, stereotype: 0.58, confidence: 0.7 },
    },
  },
]

const probeTypes = ['occupation', 'education', 'leadership', 'trustworthiness', 'neighborhood', 'dating_profile']

export default function ImageExplorer({ fingerprint }) {
  const [viewMode, setViewMode] = useState('grid')
  const [searchTerm, setSearchTerm] = useState('')
  const [filterGender, setFilterGender] = useState('all')
  const [filterRace, setFilterRace] = useState('all')
  const [sortBy, setSortBy] = useState('stereotype')
  const [expandedImage, setExpandedImage] = useState(null)

  const filteredImages = sampleImages
    .filter((img) => {
      if (filterGender !== 'all' && img.demographic.gender !== filterGender) return false
      if (filterRace !== 'all' && img.demographic.race !== filterRace) return false
      if (searchTerm && !img.id.includes(searchTerm)) return false
      return true
    })
    .sort((a, b) => {
      const getAvgScore = (img, metric) => {
        const scores = Object.values(img.probeResults).map((p) => p[metric])
        return scores.reduce((a, b) => a + b, 0) / scores.length
      }

      if (sortBy === 'stereotype') return getAvgScore(b, 'stereotype') - getAvgScore(a, 'stereotype')
      if (sortBy === 'valence') return Math.abs(getAvgScore(b, 'valence')) - Math.abs(getAvgScore(a, 'valence'))
      return 0
    })

  return (
    <div className="space-y-6">
      {/* Search and Filters */}
      <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-4">
        <div className="flex flex-wrap items-center gap-4">
          {/* Search */}
          <div className="relative flex-1 min-w-[200px]">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-slate-400" />
            <input
              type="text"
              placeholder="Search by image ID..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pl-10 pr-4 py-2 border border-slate-300 rounded-lg text-sm focus:ring-2 focus:ring-indigo-500"
            />
          </div>

          {/* Filters */}
          <div className="flex items-center gap-2">
            <Filter className="w-4 h-4 text-slate-400" />
            <select
              value={filterGender}
              onChange={(e) => setFilterGender(e.target.value)}
              className="px-3 py-2 border border-slate-300 rounded-lg text-sm"
            >
              <option value="all">All Genders</option>
              <option value="male">Male</option>
              <option value="female">Female</option>
            </select>
            <select
              value={filterRace}
              onChange={(e) => setFilterRace(e.target.value)}
              className="px-3 py-2 border border-slate-300 rounded-lg text-sm"
            >
              <option value="all">All Races</option>
              <option value="white">White</option>
              <option value="black">Black</option>
              <option value="asian">Asian</option>
              <option value="hispanic">Hispanic</option>
            </select>
          </div>

          {/* Sort */}
          <div className="flex items-center gap-2">
            <span className="text-sm text-slate-500">Sort:</span>
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value)}
              className="px-3 py-2 border border-slate-300 rounded-lg text-sm"
            >
              <option value="stereotype">Highest Stereotype</option>
              <option value="valence">Extreme Valence</option>
            </select>
          </div>

          {/* View Toggle */}
          <div className="flex border border-slate-300 rounded-lg overflow-hidden">
            <button
              onClick={() => setViewMode('grid')}
              className={`p-2 ${viewMode === 'grid' ? 'bg-indigo-100 text-indigo-600' : 'bg-white text-slate-600'}`}
            >
              <Grid className="w-4 h-4" />
            </button>
            <button
              onClick={() => setViewMode('list')}
              className={`p-2 ${viewMode === 'list' ? 'bg-indigo-100 text-indigo-600' : 'bg-white text-slate-600'}`}
            >
              <List className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>

      {/* Results Count */}
      <div className="text-sm text-slate-500">
        Showing {filteredImages.length} of {sampleImages.length} images
      </div>

      {/* Image Grid/List */}
      {viewMode === 'grid' ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {filteredImages.map((img) => (
            <ImageCard
              key={img.id}
              image={img}
              isExpanded={expandedImage === img.id}
              onToggle={() => setExpandedImage(expandedImage === img.id ? null : img.id)}
            />
          ))}
        </div>
      ) : (
        <div className="space-y-3">
          {filteredImages.map((img) => (
            <ImageListItem
              key={img.id}
              image={img}
              isExpanded={expandedImage === img.id}
              onToggle={() => setExpandedImage(expandedImage === img.id ? null : img.id)}
            />
          ))}
        </div>
      )}

      {filteredImages.length === 0 && (
        <div className="bg-slate-50 rounded-xl p-12 text-center">
          <Image className="w-12 h-12 text-slate-300 mx-auto mb-3" />
          <p className="text-slate-500">No images match the current filters.</p>
        </div>
      )}
    </div>
  )
}

function ImageCard({ image, isExpanded, onToggle }) {
  const avgStereotype = Object.values(image.probeResults)
    .map((p) => p.stereotype)
    .reduce((a, b) => a + b, 0) / Object.keys(image.probeResults).length

  const avgValence = Object.values(image.probeResults)
    .map((p) => p.valence)
    .reduce((a, b) => a + b, 0) / Object.keys(image.probeResults).length

  return (
    <div className="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden">
      {/* Placeholder Image */}
      <div className="h-40 bg-gradient-to-br from-slate-200 to-slate-300 flex items-center justify-center">
        <div className="text-center">
          <Image className="w-8 h-8 text-slate-400 mx-auto mb-1" />
          <span className="text-xs text-slate-500">{image.id}</span>
        </div>
      </div>

      {/* Info */}
      <div className="p-4">
        <div className="flex items-center justify-between mb-3">
          <div className="flex gap-1 flex-wrap">
            <span className="px-2 py-0.5 bg-slate-100 rounded text-xs text-slate-600 capitalize">
              {image.demographic.gender}
            </span>
            <span className="px-2 py-0.5 bg-slate-100 rounded text-xs text-slate-600">
              {image.demographic.age}
            </span>
            <span className="px-2 py-0.5 bg-slate-100 rounded text-xs text-slate-600 capitalize">
              {image.demographic.race}
            </span>
          </div>
        </div>

        {/* Quick Stats */}
        <div className="grid grid-cols-2 gap-2 text-center">
          <div className="bg-slate-50 rounded-lg p-2">
            <p className="text-xs text-slate-500">Avg Stereotype</p>
            <p className={`text-lg font-semibold ${avgStereotype > 0.6 ? 'text-red-500' : 'text-green-500'}`}>
              {(avgStereotype * 100).toFixed(0)}%
            </p>
          </div>
          <div className="bg-slate-50 rounded-lg p-2">
            <p className="text-xs text-slate-500">Avg Valence</p>
            <p className={`text-lg font-semibold ${avgValence < 0 ? 'text-red-500' : 'text-green-500'}`}>
              {avgValence.toFixed(2)}
            </p>
          </div>
        </div>

        {/* Expand Button */}
        <button
          onClick={onToggle}
          className="w-full mt-3 flex items-center justify-center gap-1 text-sm text-indigo-600 hover:text-indigo-700"
        >
          {isExpanded ? (
            <>
              <ChevronUp className="w-4 h-4" />
              Hide Details
            </>
          ) : (
            <>
              <ChevronDown className="w-4 h-4" />
              Show Details
            </>
          )}
        </button>
      </div>

      {/* Expanded Details */}
      {isExpanded && (
        <div className="border-t border-slate-200 p-4 bg-slate-50">
          <p className="text-xs font-medium text-slate-500 mb-2">Probe Results:</p>
          <div className="space-y-2">
            {probeTypes.map((probe) => {
              const result = image.probeResults[probe]
              return (
                <div key={probe} className="flex items-center justify-between text-sm">
                  <span className="text-slate-600 capitalize">{probe.replace('_', ' ')}</span>
                  <div className="flex gap-3">
                    <span className={`${result.valence < 0 ? 'text-red-500' : 'text-green-500'}`}>
                      V:{result.valence.toFixed(2)}
                    </span>
                    <span className={`${result.stereotype > 0.6 ? 'text-red-500' : 'text-slate-500'}`}>
                      S:{(result.stereotype * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      )}
    </div>
  )
}

function ImageListItem({ image, isExpanded, onToggle }) {
  const avgStereotype = Object.values(image.probeResults)
    .map((p) => p.stereotype)
    .reduce((a, b) => a + b, 0) / Object.keys(image.probeResults).length

  return (
    <div className="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden">
      <div className="flex items-center p-4 gap-4">
        {/* Thumbnail */}
        <div className="w-16 h-16 bg-gradient-to-br from-slate-200 to-slate-300 rounded-lg flex items-center justify-center flex-shrink-0">
          <Image className="w-6 h-6 text-slate-400" />
        </div>

        {/* Info */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <span className="font-medium text-slate-900">{image.id}</span>
            <div className="flex gap-1">
              <span className="px-2 py-0.5 bg-slate-100 rounded text-xs text-slate-600 capitalize">
                {image.demographic.gender}
              </span>
              <span className="px-2 py-0.5 bg-slate-100 rounded text-xs text-slate-600 capitalize">
                {image.demographic.race}
              </span>
            </div>
          </div>
          <p className="text-sm text-slate-500">Age: {image.demographic.age}</p>
        </div>

        {/* Score */}
        <div className="text-right">
          <p className="text-xs text-slate-500">Avg Stereotype</p>
          <p className={`text-xl font-bold ${avgStereotype > 0.6 ? 'text-red-500' : 'text-green-500'}`}>
            {(avgStereotype * 100).toFixed(0)}%
          </p>
        </div>

        {/* Expand */}
        <button onClick={onToggle} className="p-2 hover:bg-slate-100 rounded-lg">
          {isExpanded ? <ChevronUp className="w-5 h-5 text-slate-400" /> : <ChevronDown className="w-5 h-5 text-slate-400" />}
        </button>
      </div>

      {isExpanded && (
        <div className="border-t border-slate-200 p-4 bg-slate-50">
          <div className="grid grid-cols-3 md:grid-cols-6 gap-3">
            {probeTypes.map((probe) => {
              const result = image.probeResults[probe]
              return (
                <div key={probe} className="text-center">
                  <p className="text-xs text-slate-500 capitalize mb-1">{probe.replace('_', ' ')}</p>
                  <p className={`text-sm font-semibold ${result.stereotype > 0.6 ? 'text-red-500' : 'text-green-600'}`}>
                    {(result.stereotype * 100).toFixed(0)}%
                  </p>
                </div>
              )
            })}
          </div>
        </div>
      )}
    </div>
  )
}
