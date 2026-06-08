import { useEffect, useState } from 'react'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts'
import { TrendingUp, DollarSign, Building2, RefreshCw } from 'lucide-react'
import { fetchResults, type FinancialMetric } from '../lib/api'

function StatCard({ label, value, icon: Icon, color }: {
  label: string; value: string | number; icon: any; color: string
}) {
  return (
    <div className="card flex items-center gap-3">
      <div className={`p-2.5 rounded-xl ${color}`}>
        <Icon size={18} />
      </div>
      <div>
        <p className="text-slate-400 text-xs">{label}</p>
        <p className="text-slate-100 font-semibold text-lg leading-tight">{value}</p>
      </div>
    </div>
  )
}

const COLORS = ['#0ea5e9', '#6366f1', '#10b981', '#f59e0b', '#ef4444']

export function MetricsDashboard({ refreshKey }: { refreshKey: number }) {
  const [metrics, setMetrics] = useState<FinancialMetric[]>([])
  const [loading, setLoading] = useState(false)

  const load = async () => {
    setLoading(true)
    try {
      const res = await fetchResults(50)
      setMetrics(res.results)
    } catch { /* ignore */ }
    setLoading(false)
  }

  useEffect(() => { load() }, [refreshKey])

  // Derived stats
  const companies = [...new Set(metrics.map(m => m.company_name).filter(Boolean))]
  const withRevenue = metrics.filter(m => m.revenue)
  const avgRevenue = withRevenue.length
    ? withRevenue.reduce((s, m) => s + (m.revenue ?? 0), 0) / withRevenue.length
    : 0
  const avgConfidence = metrics.length
    ? metrics.reduce((s, m) => s + m.confidence, 0) / metrics.length
    : 0

  // Revenue chart data
  const revenueData = metrics
    .filter(m => m.company_name && m.revenue)
    .slice(0, 8)
    .map(m => ({
      name: (m.company_name ?? '').split(' ')[0],
      revenue: Math.round((m.revenue ?? 0) / 1000),
    }))

  // D/E ratio chart
  const deData = metrics
    .filter(m => m.company_name && m.debt_to_equity)
    .slice(0, 8)
    .map(m => ({
      name: (m.company_name ?? '').split(' ')[0],
      ratio: m.debt_to_equity,
    }))

  const fmt = (n: number) =>
    n >= 1_000_000 ? `€${(n / 1_000_000).toFixed(1)}M`
    : n >= 1_000 ? `€${(n / 1_000).toFixed(0)}K`
    : `€${n.toFixed(0)}`

  return (
    <div className="flex flex-col gap-4">
      {/* Stat cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <StatCard label="Extractions"    value={metrics.length}                icon={TrendingUp}  color="bg-brand-900 text-brand-400" />
        <StatCard label="Companies"      value={companies.length}              icon={Building2}   color="bg-violet-900 text-violet-400" />
        <StatCard label="Avg Revenue"    value={avgRevenue ? fmt(avgRevenue) : '—'} icon={DollarSign} color="bg-emerald-900 text-emerald-400" />
        <StatCard label="Avg Confidence" value={`${(avgConfidence * 100).toFixed(0)}%`} icon={RefreshCw} color="bg-amber-900 text-amber-400" />
      </div>

      {metrics.length === 0 ? (
        <div className="card flex flex-col items-center justify-center py-16 gap-3 text-center">
          <TrendingUp size={32} className="text-slate-600" />
          <p className="text-slate-400">No extracted metrics yet</p>
          <p className="text-slate-600 text-sm">Upload documents and run queries to see financial data here</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {/* Revenue chart */}
          {revenueData.length > 0 && (
            <div className="card">
              <p className="text-sm font-medium text-slate-300 mb-4">Revenue by Company (€K)</p>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={revenueData} margin={{ top: 0, right: 0, left: -20, bottom: 0 }}>
                  <XAxis dataKey="name" tick={{ fill: '#94a3b8', fontSize: 11 }} axisLine={false} tickLine={false} />
                  <YAxis tick={{ fill: '#94a3b8', fontSize: 11 }} axisLine={false} tickLine={false} />
                  <Tooltip
                    contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8 }}
                    labelStyle={{ color: '#e2e8f0' }}
                    itemStyle={{ color: '#0ea5e9' }}
                    formatter={(v: number) => [`€${v}K`, 'Revenue']}
                  />
                  <Bar dataKey="revenue" radius={[4, 4, 0, 0]}>
                    {revenueData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* D/E ratio chart */}
          {deData.length > 0 && (
            <div className="card">
              <p className="text-sm font-medium text-slate-300 mb-4">Debt-to-Equity Ratio</p>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={deData} margin={{ top: 0, right: 0, left: -20, bottom: 0 }}>
                  <XAxis dataKey="name" tick={{ fill: '#94a3b8', fontSize: 11 }} axisLine={false} tickLine={false} />
                  <YAxis tick={{ fill: '#94a3b8', fontSize: 11 }} axisLine={false} tickLine={false} />
                  <Tooltip
                    contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8 }}
                    labelStyle={{ color: '#e2e8f0' }}
                    itemStyle={{ color: '#6366f1' }}
                    formatter={(v: number) => [v?.toFixed(2), 'D/E Ratio']}
                  />
                  <Bar dataKey="ratio" radius={[4, 4, 0, 0]}>
                    {deData.map((_, i) => <Cell key={i} fill={COLORS[(i + 1) % COLORS.length]} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Recent extractions table */}
          <div className="card lg:col-span-2">
            <p className="text-sm font-medium text-slate-300 mb-3">Recent Extractions</p>
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="text-slate-500 border-b border-slate-800">
                    {['Company','Revenue','EBITDA','D/E','Confidence','Query'].map(h => (
                      <th key={h} className="text-left pb-2 pr-4 font-medium">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {metrics.slice(0, 10).map((m, i) => (
                    <tr key={i} className="border-b border-slate-800/50 hover:bg-slate-800/30 transition-colors">
                      <td className="py-2 pr-4 text-slate-200 font-medium">{m.company_name ?? '—'}</td>
                      <td className="py-2 pr-4 text-emerald-400">{m.revenue ? fmt(m.revenue) : '—'}</td>
                      <td className="py-2 pr-4 text-brand-400">{m.ebitda ? fmt(m.ebitda) : '—'}</td>
                      <td className="py-2 pr-4 text-violet-400">{m.debt_to_equity?.toFixed(2) ?? '—'}</td>
                      <td className="py-2 pr-4">
                        <span className={`badge ${m.confidence > 0.7 ? 'bg-emerald-950 text-emerald-400' : 'bg-amber-950 text-amber-400'}`}>
                          {(m.confidence * 100).toFixed(0)}%
                        </span>
                      </td>
                      <td className="py-2 text-slate-400 truncate max-w-48">{m.query}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
