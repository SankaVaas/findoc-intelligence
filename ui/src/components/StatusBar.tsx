import { useEffect, useState } from 'react'
import { Activity, Database, Cpu, AlertCircle } from 'lucide-react'
import { fetchHealth, fetchStats, type HealthResponse, type StatsResponse } from '../lib/api'
import clsx from 'clsx'

export function StatusBar() {
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [stats, setStats]   = useState<StatsResponse | null>(null)
  const [error, setError]   = useState(false)

  const refresh = async () => {
    try {
      const [h, s] = await Promise.all([fetchHealth(), fetchStats()])
      setHealth(h); setStats(s); setError(false)
    } catch {
      setError(true)
    }
  }

  useEffect(() => { refresh(); const t = setInterval(refresh, 10000); return () => clearInterval(t) }, [])

  const dot = (ok: boolean) => (
    <span className={clsx('inline-block w-2 h-2 rounded-full mr-1.5', ok ? 'bg-emerald-400' : 'bg-red-400')} />
  )

  if (error) return (
    <div className="flex items-center gap-2 px-4 py-2 bg-red-950 border-b border-red-900 text-red-300 text-sm">
      <AlertCircle size={14} />
      Backend not reachable — is the API running? <code className="ml-1 text-xs">uvicorn api.app:app --port 8000</code>
    </div>
  )

  return (
    <div className="flex items-center gap-6 px-4 py-2 bg-slate-900 border-b border-slate-800 text-xs text-slate-400">
      <div className="flex items-center gap-1.5">
        <Activity size={12} className={health?.status === 'ok' ? 'text-emerald-400' : 'text-red-400'} />
        <span className={health?.status === 'ok' ? 'text-emerald-400' : 'text-red-400'}>
          {health?.status ?? '...'}
        </span>
      </div>
      <div className="flex items-center gap-1.5">
        <Database size={12} />
        {dot(!!health?.components?.chromadb?.startsWith('ok'))}
        {stats ? `${stats.chunks_in_store.toLocaleString()} chunks` : '...'}
      </div>
      <div className="flex items-center gap-1.5">
        <Cpu size={12} />
        {dot(!!health?.components?.llm?.startsWith('ok'))}
        {stats?.llm_model ?? '...'}
      </div>
      <div className="ml-auto">
        {stats ? `${stats.queries_processed} queries processed` : ''}
      </div>
    </div>
  )
}
