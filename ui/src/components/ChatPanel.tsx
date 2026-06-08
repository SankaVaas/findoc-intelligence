import { useState, useRef, useEffect } from 'react'
import { Send, Bot, User, Loader2, ChevronDown, ChevronUp, Clock } from 'lucide-react'
import { streamQuery, type RetrievedChunk } from '../lib/api'
import clsx from 'clsx'

interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  chunks?: RetrievedChunk[]
  latency?: number
  streaming?: boolean
}

const SUGGESTED = [
  "What is the debt-to-equity ratio?",
  "Summarise the key financial highlights",
  "What are the main risk factors?",
  "What was the revenue growth this year?",
]

export function ChatPanel({ onQueryDone }: { onQueryDone?: () => void }) {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput]        = useState('')
  const [loading, setLoading]    = useState(false)
  const bottomRef                = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const send = async (query: string) => {
    if (!query.trim() || loading) return
    setInput('')
    setLoading(true)

    const userMsg: Message = { id: Date.now().toString(), role: 'user', content: query }
    const asstId = (Date.now() + 1).toString()
    const asstMsg: Message = { id: asstId, role: 'assistant', content: '', streaming: true }
    setMessages(m => [...m, userMsg, asstMsg])

    try {
      const start = Date.now()
      let full = ''
      for await (const token of streamQuery(query)) {
        full += token
        setMessages(m => m.map(msg =>
          msg.id === asstId ? { ...msg, content: full } : msg
        ))
      }
      const latency = Date.now() - start
      setMessages(m => m.map(msg =>
        msg.id === asstId ? { ...msg, streaming: false, latency } : msg
      ))
      onQueryDone?.()
    } catch (e: any) {
      setMessages(m => m.map(msg =>
        msg.id === asstId
          ? { ...msg, content: `Error: ${e.message}`, streaming: false }
          : msg
      ))
    }
    setLoading(false)
  }

  return (
    <div className="card flex flex-col h-full min-h-0 gap-0 p-0 overflow-hidden">
      {/* Header */}
      <div className="flex items-center gap-2 px-4 py-3 border-b border-slate-800">
        <Bot size={16} className="text-brand-400" />
        <span className="font-semibold text-slate-200">Financial Intelligence Chat</span>
        <span className="ml-auto badge bg-emerald-950 text-emerald-400">Streaming</span>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 flex flex-col gap-4 min-h-0">
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full gap-6 text-center">
            <div className="p-4 rounded-2xl bg-brand-950 border border-brand-900">
              <Bot size={32} className="text-brand-400" />
            </div>
            <div>
              <p className="text-slate-300 font-medium">Ask anything about your documents</p>
              <p className="text-slate-500 text-sm mt-1">Upload documents first, then ask questions</p>
            </div>
            <div className="grid grid-cols-1 gap-2 w-full max-w-sm">
              {SUGGESTED.map(q => (
                <button
                  key={q}
                  onClick={() => send(q)}
                  className="text-left text-sm bg-slate-800 hover:bg-slate-700 border border-slate-700
                             rounded-lg px-3 py-2 text-slate-300 transition-colors"
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map(msg => (
          <MessageBubble key={msg.id} msg={msg} />
        ))}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="p-3 border-t border-slate-800">
        <div className="flex gap-2">
          <input
            className="input text-sm"
            placeholder="Ask a question about your documents..."
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && !e.shiftKey && send(input)}
            disabled={loading}
          />
          <button
            className="btn-primary flex items-center gap-1.5 shrink-0"
            onClick={() => send(input)}
            disabled={loading || !input.trim()}
          >
            {loading
              ? <Loader2 size={15} className="animate-spin" />
              : <Send size={15} />
            }
          </button>
        </div>
      </div>
    </div>
  )
}

function MessageBubble({ msg }: { msg: Message }) {
  const [showChunks, setShowChunks] = useState(false)

  return (
    <div className={clsx('flex gap-3', msg.role === 'user' && 'flex-row-reverse')}>
      {/* Avatar */}
      <div className={clsx(
        'w-7 h-7 rounded-full flex items-center justify-center shrink-0 mt-0.5',
        msg.role === 'user' ? 'bg-brand-600' : 'bg-slate-700'
      )}>
        {msg.role === 'user' ? <User size={13} /> : <Bot size={13} />}
      </div>

      {/* Bubble */}
      <div className={clsx('flex flex-col gap-1 max-w-[80%]', msg.role === 'user' && 'items-end')}>
        <div className={clsx(
          'rounded-2xl px-4 py-2.5 text-sm leading-relaxed whitespace-pre-wrap',
          msg.role === 'user'
            ? 'bg-brand-600 text-white rounded-tr-sm'
            : 'bg-slate-800 text-slate-200 rounded-tl-sm'
        )}>
          {msg.content || (msg.streaming && <span className="animate-pulse">▋</span>)}
        </div>

        {/* Metadata */}
        {msg.role === 'assistant' && !msg.streaming && (
          <div className="flex items-center gap-2 text-xs text-slate-500 px-1">
            {msg.latency && (
              <span className="flex items-center gap-1">
                <Clock size={10} />
                {(msg.latency / 1000).toFixed(1)}s
              </span>
            )}
            {msg.chunks && msg.chunks.length > 0 && (
              <button
                onClick={() => setShowChunks(s => !s)}
                className="flex items-center gap-1 hover:text-slate-300 transition-colors"
              >
                {showChunks ? <ChevronUp size={10} /> : <ChevronDown size={10} />}
                {msg.chunks.length} sources
              </button>
            )}
          </div>
        )}

        {/* Source chunks */}
        {showChunks && msg.chunks && (
          <div className="flex flex-col gap-1.5 mt-1">
            {msg.chunks.map((c, i) => (
              <div key={i} className="bg-slate-800/60 border border-slate-700 rounded-lg p-2.5 text-xs">
                <div className="flex items-center gap-2 mb-1 text-slate-400">
                  <span className="badge bg-slate-700 text-slate-300">#{i + 1}</span>
                  <span>score {c.rerank_score?.toFixed(3) ?? c.vector_score.toFixed(3)}</span>
                  {c.chunk.page_number && <span>p.{c.chunk.page_number}</span>}
                </div>
                <p className="text-slate-300 line-clamp-3">{c.chunk.text}</p>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
