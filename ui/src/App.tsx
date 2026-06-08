import { useState } from 'react'
import { BarChart2, MessageSquare, Upload, Cpu } from 'lucide-react'
import { StatusBar } from './components/StatusBar'
import { UploadPanel } from './components/UploadPanel'
import { ChatPanel } from './components/ChatPanel'
import { MetricsDashboard } from './components/MetricsDashboard'
import clsx from 'clsx'

type Tab = 'chat' | 'dashboard' | 'upload'

const TABS: { id: Tab; label: string; icon: any }[] = [
  { id: 'chat',      label: 'Chat',      icon: MessageSquare },
  { id: 'dashboard', label: 'Dashboard', icon: BarChart2 },
  { id: 'upload',    label: 'Upload',    icon: Upload },
]

export default function App() {
  const [tab, setTab]           = useState<Tab>('chat')
  const [refreshKey, setRefresh] = useState(0)

  const refresh = () => setRefresh(k => k + 1)

  return (
    <div className="flex flex-col h-screen bg-slate-950 overflow-hidden">
      {/* Header */}
      <header className="flex items-center px-4 py-3 bg-slate-900 border-b border-slate-800 gap-4 shrink-0">
        <div className="flex items-center gap-2">
          <div className="p-1.5 bg-brand-600 rounded-lg">
            <Cpu size={16} className="text-white" />
          </div>
          <span className="font-bold text-slate-100">findoc</span>
          <span className="text-slate-500 font-light">intelligence</span>
        </div>

        {/* Tabs */}
        <nav className="flex gap-1 ml-4">
          {TABS.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              onClick={() => setTab(id)}
              className={clsx(
                'flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium transition-all',
                tab === id
                  ? 'bg-brand-600 text-white'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800'
              )}
            >
              <Icon size={14} />
              {label}
            </button>
          ))}
        </nav>

        <div className="ml-auto text-xs text-slate-500">
          Multilingual Financial Document Intelligence
        </div>
      </header>

      {/* Status bar */}
      <StatusBar />

      {/* Main content */}
      <main className="flex-1 min-h-0 overflow-hidden">
        {/* Chat tab */}
        <div className={clsx('h-full p-4', tab !== 'chat' && 'hidden')}>
          <ChatPanel onQueryDone={refresh} />
        </div>

        {/* Dashboard tab */}
        <div className={clsx('h-full overflow-y-auto p-4', tab !== 'dashboard' && 'hidden')}>
          <MetricsDashboard refreshKey={refreshKey} />
        </div>

        {/* Upload tab */}
        <div className={clsx('h-full overflow-y-auto p-4', tab !== 'upload' && 'hidden')}>
          <div className="max-w-2xl mx-auto">
            <UploadPanel onIngested={refresh} />
            <div className="mt-4 card text-sm text-slate-400">
              <p className="font-medium text-slate-300 mb-2">Supported formats</p>
              <div className="grid grid-cols-2 gap-1">
                {[
                  ['PDF', 'Digital + scanned (OCR)'],
                  ['DOCX', 'Word documents + tables'],
                  ['HTML', 'Web pages (nav/footer stripped)'],
                  ['TXT', 'Plain text files'],
                  ['MP3 / WAV', 'Audio → transcript (Whisper)'],
                  ['Multilingual', 'EN, FR, DE, ES, IT + more'],
                ].map(([fmt, desc]) => (
                  <div key={fmt} className="flex gap-2">
                    <span className="text-brand-400 font-medium w-20 shrink-0">{fmt}</span>
                    <span className="text-slate-500">{desc}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  )
}
