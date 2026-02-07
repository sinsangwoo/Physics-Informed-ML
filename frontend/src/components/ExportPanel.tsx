import { Download, FileJson, FileSpreadsheet, Image } from 'lucide-react'
import { useSimulationStore } from '../store/simulationStore'

export function ExportPanel() {
  const { predictions, inputData, type, resolution } = useSimulationStore()

  const exportJSON = () => {
    if (!predictions) return

    const data = {
      metadata: {
        type,
        resolution,
        timestamp: new Date().toISOString(),
        frames: predictions.length,
      },
      inputData,
      predictions,
    }

    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
    downloadBlob(blob, `simulation-${type}-${Date.now()}.json`)
  }

  const exportCSV = () => {
    if (!predictions) return

    // CSV header
    let csv = 'frame,position,value\n'

    // Data rows
    predictions.forEach((frame, frameIndex) => {
      frame.forEach((value, posIndex) => {
        csv += `${frameIndex},${posIndex},${value}\n`
      })
    })

    const blob = new Blob([csv], { type: 'text/csv' })
    downloadBlob(blob, `simulation-${type}-${Date.now()}.csv`)
  }

  const exportImage = () => {
    // Capture canvas
    const canvas = document.querySelector('canvas')
    if (!canvas) return

    canvas.toBlob((blob) => {
      if (blob) {
        downloadBlob(blob, `visualization-${type}-${Date.now()}.png`)
      }
    })
  }

  const downloadBlob = (blob: Blob, filename: string) => {
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = filename
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  const hasData = predictions && predictions.length > 0

  return (
    <div className="space-y-2">
      <h3 className="text-sm font-medium mb-3">Export Results</h3>
      
      <button
        onClick={exportJSON}
        disabled={!hasData}
        className="w-full btn-secondary flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
      >
        <FileJson className="w-4 h-4" />
        Export JSON
      </button>

      <button
        onClick={exportCSV}
        disabled={!hasData}
        className="w-full btn-secondary flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
      >
        <FileSpreadsheet className="w-4 h-4" />
        Export CSV
      </button>

      <button
        onClick={exportImage}
        className="w-full btn-secondary flex items-center justify-center gap-2"
      >
        <Image className="w-4 h-4" />
        Screenshot
      </button>

      {!hasData && (
        <p className="text-xs text-slate-500 text-center mt-2">
          Run simulation to enable export
        </p>
      )}
    </div>
  )
}
