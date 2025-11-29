'use client'

import { useState, useEffect, useRef, useCallback } from 'react'
import { ArrowUp } from 'lucide-react'
import { checkAllAgentsHealth, sendMessage } from '@/lib/api'
import { AgentHealth, AgentName, A2AResponse, OrchestratorOutput } from '@/types'
import { AGENTS } from '@/lib/agents'

interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  agentResults?: Record<string, AgentResult>
  thinking?: boolean
}

interface AgentResult {
  agent: string
  time: number
  output: Record<string, unknown>
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [health, setHealth] = useState<Record<AgentName, AgentHealth> | null>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    checkAllAgentsHealth().then(setHealth).catch(console.error)
  }, [])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const adjustTextareaHeight = useCallback(() => {
    const textarea = textareaRef.current
    if (textarea) {
      textarea.style.height = 'auto'
      textarea.style.height = Math.min(textarea.scrollHeight, 200) + 'px'
    }
  }, [])

  const handleSubmit = async () => {
    if (!input.trim() || isLoading) return

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input.trim(),
    }

    const thinkingMessage: Message = {
      id: 'thinking',
      role: 'assistant',
      content: '',
      thinking: true,
    }

    setMessages(prev => [...prev, userMessage, thinkingMessage])
    setInput('')
    setIsLoading(true)

    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
    }

    try {
      const response: A2AResponse = await sendMessage(input.trim())

      if (response.error) {
        throw new Error(response.error.message)
      }

      const output = response.result?.output as unknown as OrchestratorOutput | undefined

      const agentResults: Record<string, AgentResult> = {}
      if (output?.task_results) {
        Object.entries(output.task_results).forEach(([name, result]) => {
          agentResults[name] = {
            agent: AGENTS[name as AgentName]?.displayName || name,
            time: result.executionTime,
            output: result.output,
          }
        })
      }

      const assistantMessage: Message = {
        id: Date.now().toString(),
        role: 'assistant',
        content: output?.synthesis || 'Done.',
        agentResults: Object.keys(agentResults).length > 0 ? agentResults : undefined,
      }

      setMessages(prev => prev.filter(m => m.id !== 'thinking').concat(assistantMessage))
    } catch (error) {
      const errorMessage: Message = {
        id: Date.now().toString(),
        role: 'assistant',
        content: `Something went wrong. ${error instanceof Error ? error.message : 'Please try again.'}`,
      }
      setMessages(prev => prev.filter(m => m.id !== 'thinking').concat(errorMessage))
    } finally {
      setIsLoading(false)
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
  }

  const onlineCount = health
    ? Object.values(health).filter(h => h.status === 'healthy').length
    : 0

  return (
    <div className="min-h-screen flex flex-col bg-bg-primary">
      {/* Header */}
      <header className="flex-shrink-0 border-b border-border-subtle">
        <div className="max-w-chat mx-auto px-5 py-3 flex items-center justify-between">
          <span className="text-text-primary font-medium text-[15px]">Agent</span>
          <div className="flex items-center gap-2">
            <span className={`w-1.5 h-1.5 rounded-full ${onlineCount > 0 ? 'bg-emerald-500' : 'bg-text-muted'}`} />
            <span className="text-[13px] text-text-muted">
              {onlineCount > 0 ? `${onlineCount} online` : 'offline'}
            </span>
          </div>
        </div>
      </header>

      {/* Messages */}
      <main className="flex-1 overflow-y-auto">
        <div className="max-w-chat mx-auto px-5 py-8">
          {messages.length === 0 ? (
            <div className="text-center py-24">
              <h1 className="text-xl font-medium text-text-primary mb-2">
                What can I help you with?
              </h1>
              <p className="text-text-secondary text-[15px]">
                Analyze data, write code, validate content, or process images.
              </p>
            </div>
          ) : (
            <div className="space-y-5">
              {messages.map(message => (
                <div key={message.id} className="animate-in">
                  {message.role === 'user' ? (
                    <div className="flex justify-end">
                      <div className="bg-bg-elevated text-text-primary rounded-2xl rounded-br-sm px-4 py-2.5 max-w-[85%]">
                        <p className="text-[15px] whitespace-pre-wrap">{message.content}</p>
                      </div>
                    </div>
                  ) : message.thinking ? (
                    <div className="flex gap-1 py-2">
                      <span className="w-1.5 h-1.5 bg-text-muted rounded-full thinking-dot" />
                      <span className="w-1.5 h-1.5 bg-text-muted rounded-full thinking-dot" />
                      <span className="w-1.5 h-1.5 bg-text-muted rounded-full thinking-dot" />
                    </div>
                  ) : (
                    <div className="space-y-3">
                      <p className="text-[15px] text-text-primary whitespace-pre-wrap leading-relaxed">
                        {message.content}
                      </p>

                      {message.agentResults && (
                        <details className="group">
                          <summary className="text-[13px] text-text-muted cursor-pointer hover:text-text-secondary transition-colors">
                            {Object.keys(message.agentResults).length} agent{Object.keys(message.agentResults).length !== 1 ? 's' : ''} contributed
                          </summary>
                          <div className="mt-3 pl-3 border-l border-border-subtle space-y-3">
                            {Object.entries(message.agentResults).map(([key, result]) => (
                              <div key={key} className="text-[13px]">
                                <div className="flex items-center gap-2 text-text-muted">
                                  <span className="text-text-secondary">{result.agent}</span>
                                  <span className="text-text-muted">·</span>
                                  <span>{(result.time * 1000).toFixed(0)}ms</span>
                                </div>
                                {result.output && (
                                  <ResultPreview output={result.output} />
                                )}
                              </div>
                            ))}
                          </div>
                        </details>
                      )}
                    </div>
                  )}
                </div>
              ))}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>
      </main>

      {/* Input */}
      <footer className="flex-shrink-0 border-t border-border-subtle bg-bg-primary">
        <div className="max-w-chat mx-auto px-5 py-4">
          <div className="relative">
            <textarea
              ref={textareaRef}
              value={input}
              onChange={e => {
                setInput(e.target.value)
                adjustTextareaHeight()
              }}
              onKeyDown={handleKeyDown}
              placeholder="Message..."
              disabled={isLoading}
              rows={1}
              className="w-full resize-none bg-bg-tertiary rounded-xl px-4 py-3 pr-12
                         text-[15px] text-text-primary placeholder:text-text-muted
                         border border-border-default
                         focus:border-accent focus:outline-none
                         transition-colors duration-150
                         disabled:opacity-50"
              style={{ minHeight: '48px', maxHeight: '200px' }}
            />
            <button
              onClick={handleSubmit}
              disabled={!input.trim() || isLoading}
              aria-label="Send message"
              className="absolute right-2 bottom-2 w-8 h-8 rounded-lg
                         flex items-center justify-center
                         bg-text-primary text-bg-primary
                         hover:bg-text-secondary
                         disabled:bg-bg-elevated disabled:text-text-muted
                         transition-colors duration-150"
            >
              <ArrowUp className="w-4 h-4" />
            </button>
          </div>
        </div>
      </footer>
    </div>
  )
}

function ResultPreview({ output }: { output: Record<string, unknown> }) {
  const insights = output.insights as string[] | undefined
  const code = output.code as string | undefined
  const isValid = output.is_valid as boolean | undefined
  const description = output.visual_description as string | undefined

  if (insights && insights.length > 0) {
    return (
      <ul className="mt-1.5 text-text-secondary space-y-0.5">
        {insights.slice(0, 2).map((insight, i) => (
          <li key={i} className="flex items-start gap-2">
            <span className="text-text-muted">·</span>
            <span>{insight}</span>
          </li>
        ))}
      </ul>
    )
  }

  if (code) {
    return (
      <pre className="mt-2 text-xs overflow-x-auto">
        <code className="text-text-secondary">{code.slice(0, 200)}{code.length > 200 ? '...' : ''}</code>
      </pre>
    )
  }

  if (typeof isValid === 'boolean') {
    return (
      <p className="mt-1.5 text-text-secondary">
        {isValid ? '✓ Valid' : '✗ Invalid'}
      </p>
    )
  }

  if (description) {
    return (
      <p className="mt-1.5 text-text-secondary line-clamp-2">{description}</p>
    )
  }

  return null
}
