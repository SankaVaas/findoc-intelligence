import { useRef, useState } from 'react'
import { Upload, FileText, CheckCircle, XCircle, Loader2 } from 'lucide-react'
import { ingestFile, type IngestResponse } from '../lib/api'
import clsx from 'clsx'

interface UploadResult {
  file: string
  status: 'ok' | 'error'
  chunks?: number
  latency?: number
  error?: string
}

export function UploadPanel({ onIngested }: { onIngested?: () => void }) {
  const inputRef               = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)
  const [loading, setLoading]   = useState(false)
  const [results, setResults]   = useState<UploadResult[]>([])

  const handleFiles = async (files: FileList | null) => {
    if (!files || files.length === 0) return
    setLoading(true)
    const newResults: UploadResult[] = []

    for (const file of Array.from(files)) {
      try {
        const res: IngestResponse = await ingestFile(file)
        newResults.push({ file: file.name, status: 'ok', chunks: res.chunks_ingested, latency: res.latency_ms })
      } catch (e: any) {
        newResults.push({ file: file.name, status: 'error', error: e.message })
      }
    }

    setResults(r => [...newResults, ...r].slice(0, 20))
    setLoading(false)
    onIngested?.()
  }

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault(); setDragging(false)
    handleFiles(e.dataTransfer.files)
  }

  return (
    <div className="card flex flex-col gap-4">
      <div className="flex items-center gap-2 font-semibold text-slate-200">
        <Upload size={16} className="text-brand-400" />
        Document Ingestion
      </div>

      {/* Drop zone */}
      <div
        onDragOver={e => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        onClick={() => inputRef.current?.click()}
        className={clsx(
          'border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all',
          dragging ? 'border-brand-500 bg-brand-950' : 'border-slate-700 hover:border-slate-500 hover:bg-slate-800/50'
        )}
      >
        {loading
          ? <Loader2 size={28} className="mx-auto text-brand-400 animate-spin mb-2" />
          : <FileText size={28} className="mx-auto text-slate-500 mb-2" />
        }
        <p className="text-sm text-slate-400">
          {loading ? 'Processing...' : 'Drop files here or click to upload'}
        </p>
        <p className="text-xs text-slate-600 mt-1">PDF · DOCX · HTML · TXT · MP3 · WAV</p>
        <input
          ref={inputRef}
          type="file"
          multiple
          accept=".pdf,.docx,.doc,.html,.htm,.txt,.mp3,.wav,.m4a"
          className="hidden"
          onChange={e => handleFiles(e.target.files)}
        />
      </div>

      {/* Upload results */}
      {results.length > 0 && (
        <div className="flex flex-col gap-1.5 max-h-48 overflow-y-auto">
          {results.map((r, i) => (
            <div key={i} className="flex items-center gap-2 text-sm bg-slate-800/60 rounded-lg px-3 py-2">
              {r.status === 'ok'
                ? <CheckCircle size={14} className="text-emerald-400 shrink-0" />
                : <XCircle    size={14} className="text-red-400 shrink-0" />
              }
              <span className="text-slate-300 truncate flex-1">{r.file}</span>
              {r.status === 'ok'
                ? <span className="text-emerald-400 text-xs shrink-0">{r.chunks} chunks · {r.latency}ms</span>
                : <span className="text-red-400 text-xs shrink-0 truncate max-w-32">{r.error}</span>
              }
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
