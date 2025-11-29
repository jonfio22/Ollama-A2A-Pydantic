'use client'

import { useState, useEffect, useRef, useCallback } from 'react'
import { ArrowUp, Sparkles, BrainIcon, CheckCircle2 } from 'lucide-react'
import { checkAllAgentsHealth, sendMessage } from '@/lib/api'
import { AgentHealth, AgentName, A2AResponse, OrchestratorOutput } from '@/types'
import { AGENTS } from '@/lib/agents'
import { Shimmer } from '@/components/ai-elements/shimmer'
import { Loader } from '@/components/ai-elements/loader'
import { cn } from '@/lib/utils'

// Reasoning steps with slower, more deliberate timing
const REASONING_PHASES = [
  { id: 'understand', label: 'Understanding your request', thinkingText: 'Reading and parsing...' },
  { id: 'analyze', label: 'Breaking down the problem', thinkingText: 'Analyzing context...' },
  { id: 'delegate', label: 'Routing to specialists', thinkingText: 'Selecting agents...' },
  { id: 'synthesize', label: 'Synthesizing response', thinkingText: 'Combining insights...' },
]

interface ReasoningStep {
  id: string
  label: string
  thinkingText: string
  status: 'hidden' | 'spawning' | 'active' | 'complete'
}

interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  agentResults?: Record<string, AgentResult>
  thinking?: boolean
  thinkingDuration?: number
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
  const [reasoningSteps, setReasoningSteps] = useState<ReasoningStep[]>([])
  const [thinkingStartTime, setThinkingStartTime] = useState<number | null>(null)
  const [currentThinkingText, setCurrentThinkingText] = useState('')
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const animationCleanupRef = useRef<(() => void) | null>(null)

  useEffect(() => {
    checkAllAgentsHealth().then(setHealth).catch(console.error)
  }, [])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, reasoningSteps])

  const adjustTextareaHeight = useCallback(() => {
    const textarea = textareaRef.current
    if (textarea) {
      textarea.style.height = 'auto'
      textarea.style.height = Math.min(textarea.scrollHeight, 200) + 'px'
    }
  }, [])

  // Organic, slow animation - one step at a time
  const animateReasoningSteps = useCallback(() => {
    setThinkingStartTime(Date.now())
    setReasoningSteps([])

    const timeouts: NodeJS.Timeout[] = []

    // Slower, more deliberate timing
    // Each step: spawn (shimmer) -> active (working) -> complete
    // Total time per step: ~2.5 seconds before next begins
    const SPAWN_DURATION = 600      // Time showing shimmer while spawning
    const ACTIVE_DURATION = 1800    // Time showing as active/working
    const STEP_TOTAL = SPAWN_DURATION + ACTIVE_DURATION

    REASONING_PHASES.forEach((phase, index) => {
      const stepStartTime = index * STEP_TOTAL

      // Step 1: Spawn with shimmer
      const spawnTimeout = setTimeout(() => {
        setCurrentThinkingText(phase.thinkingText)
        setReasoningSteps(prev => [
          ...prev,
          {
            id: phase.id,
            label: phase.label,
            thinkingText: phase.thinkingText,
            status: 'spawning',
          }
        ])
      }, stepStartTime)
      timeouts.push(spawnTimeout)

      // Step 2: Become active (stop shimmer, show working state)
      const activeTimeout = setTimeout(() => {
        setReasoningSteps(prev => prev.map(step =>
          step.id === phase.id ? { ...step, status: 'active' } : step
        ))
      }, stepStartTime + SPAWN_DURATION)
      timeouts.push(activeTimeout)

      // Step 3: Mark as complete (only if not the last step)
      if (index < REASONING_PHASES.length - 1) {
        const completeTimeout = setTimeout(() => {
          setReasoningSteps(prev => prev.map(step =>
            step.id === phase.id ? { ...step, status: 'complete' } : step
          ))
        }, stepStartTime + STEP_TOTAL - 200) // Slightly before next spawns
        timeouts.push(completeTimeout)
      }
    })

    const cleanup = () => {
      timeouts.forEach(clearTimeout)
    }

    animationCleanupRef.current = cleanup
    return cleanup
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

    animateReasoningSteps()

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

      // Complete all steps gracefully
      setReasoningSteps(prev => prev.map(step => ({ ...step, status: 'complete' as const })))
      setCurrentThinkingText('Done')

      const duration = thinkingStartTime ? Math.ceil((Date.now() - thinkingStartTime) / 1000) : 0

      // Pause to let user see completion
      await new Promise(resolve => setTimeout(resolve, 600))

      const assistantMessage: Message = {
        id: Date.now().toString(),
        role: 'assistant',
        content: output?.synthesis || 'Done.',
        agentResults: Object.keys(agentResults).length > 0 ? agentResults : undefined,
        thinkingDuration: duration,
      }

      setReasoningSteps([])
      setCurrentThinkingText('')
      setMessages(prev => prev.filter(m => m.id !== 'thinking').concat(assistantMessage))
    } catch (error) {
      animationCleanupRef.current?.()
      setReasoningSteps([])
      setCurrentThinkingText('')
      const errorMessage: Message = {
        id: Date.now().toString(),
        role: 'assistant',
        content: `Something went wrong. ${error instanceof Error ? error.message : 'Please try again.'}`,
      }
      setMessages(prev => prev.filter(m => m.id !== 'thinking').concat(errorMessage))
    } finally {
      setIsLoading(false)
      setThinkingStartTime(null)
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
    <div className="min-h-screen flex flex-col bg-background">
      <div className="vignette" />

      {/* Header */}
      <header className="flex-shrink-0 border-b border-border">
        <div className="max-w-2xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-xl bg-primary flex items-center justify-center">
              <Sparkles className="w-4 h-4 text-primary-foreground" />
            </div>
            <span className="text-[15px] font-medium text-foreground" style={{ fontFamily: 'var(--font-display)' }}>
              Agent
            </span>
          </div>
          <div className="flex items-center gap-2">
            <span className={cn(
              "w-2 h-2 rounded-full transition-colors",
              onlineCount > 0 ? "bg-emerald-500" : "bg-muted-foreground"
            )} />
            <span className="text-[13px] text-muted-foreground">
              {onlineCount > 0 ? `${onlineCount} agents ready` : 'offline'}
            </span>
          </div>
        </div>
      </header>

      {/* Messages */}
      <main className="flex-1 overflow-y-auto">
        <div className="max-w-2xl mx-auto px-6 py-12">
          {messages.length === 0 ? (
            <EmptyState />
          ) : (
            <div className="space-y-8">
              {messages.map(message => (
                <div key={message.id}>
                  {message.role === 'user' ? (
                    <UserMessage content={message.content} />
                  ) : message.thinking ? (
                    <ThinkingState
                      steps={reasoningSteps}
                      currentThinkingText={currentThinkingText}
                    />
                  ) : (
                    <AssistantMessage
                      content={message.content}
                      agentResults={message.agentResults}
                      thinkingDuration={message.thinkingDuration}
                    />
                  )}
                </div>
              ))}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>
      </main>

      {/* Input */}
      <footer className="flex-shrink-0 border-t border-border bg-background">
        <div className="max-w-2xl mx-auto px-6 py-5">
          <div className="input-wrapper">
            <textarea
              ref={textareaRef}
              value={input}
              onChange={e => {
                setInput(e.target.value)
                adjustTextareaHeight()
              }}
              onKeyDown={handleKeyDown}
              placeholder="Ask me anything..."
              disabled={isLoading}
              rows={1}
              style={{ minHeight: '56px', maxHeight: '200px' }}
            />
            <button
              onClick={handleSubmit}
              disabled={!input.trim() || isLoading}
              aria-label="Send message"
              className="send-button"
            >
              {isLoading ? (
                <Loader size={18} />
              ) : (
                <ArrowUp className="w-5 h-5" />
              )}
            </button>
          </div>
          <p className="text-center mt-3 text-[12px] text-muted-foreground/50">
            Powered by a multi-agent system
          </p>
        </div>
      </footer>
    </div>
  )
}

function EmptyState() {
  return (
    <div className="text-center py-20 fade-in">
      <h1
        className="text-3xl mb-4 text-foreground"
        style={{
          fontFamily: 'var(--font-display)',
          fontWeight: 400,
          letterSpacing: '-0.02em',
        }}
      >
        What can I help you with?
      </h1>
      <p className="text-[15px] max-w-md mx-auto text-muted-foreground leading-relaxed">
        I can analyze data, write code, validate content, or process images.
        Just ask, and I'll coordinate the right specialists.
      </p>

      <div className="flex flex-wrap justify-center gap-2 mt-8">
        {[
          'Analyze this data',
          'Write a function',
          'Review my code',
        ].map((suggestion) => (
          <button
            key={suggestion}
            className="px-4 py-2 rounded-full text-[13px] bg-secondary text-secondary-foreground border border-border hover:bg-accent hover:text-accent-foreground transition-all duration-200"
          >
            {suggestion}
          </button>
        ))}
      </div>
    </div>
  )
}

function UserMessage({ content }: { content: string }) {
  return (
    <div className="flex justify-end fade-in">
      <div className="user-message">
        <p className="whitespace-pre-wrap">{content}</p>
      </div>
    </div>
  )
}

// Organic thinking state - steps appear one at a time with shimmer
function ThinkingState({
  steps,
  currentThinkingText
}: {
  steps: ReasoningStep[]
  currentThinkingText: string
}) {
  // Initial state before any steps
  if (steps.length === 0) {
    return (
      <div className="fade-in">
        <div className="flex items-center gap-3">
          <div className="w-5 h-5 rounded-full border-2 border-primary/30 flex items-center justify-center">
            <div className="w-2 h-2 rounded-full bg-primary animate-pulse" />
          </div>
          <Shimmer duration={2} className="text-sm text-muted-foreground">
            Thinking...
          </Shimmer>
        </div>
      </div>
    )
  }

  return (
    <div className="fade-in space-y-1">
      {/* Header */}
      <div className="flex items-center gap-2 text-xs text-muted-foreground mb-4">
        <BrainIcon className="w-3.5 h-3.5" />
        <span>Reasoning</span>
      </div>

      {/* Steps - appear one at a time */}
      <div className="space-y-3 pl-1">
        {steps.map((step, index) => (
          <ReasoningStepRow
            key={step.id}
            step={step}
            isLast={index === steps.length - 1}
          />
        ))}
      </div>

      {/* Current action indicator */}
      {currentThinkingText && steps.some(s => s.status === 'active' || s.status === 'spawning') && (
        <div className="mt-4 pl-7">
          <Shimmer duration={2.5} className="text-xs text-muted-foreground/70">
            {currentThinkingText}
          </Shimmer>
        </div>
      )}
    </div>
  )
}

// Individual reasoning step with organic animations
function ReasoningStepRow({ step, isLast }: { step: ReasoningStep; isLast: boolean }) {
  return (
    <div
      className={cn(
        "flex items-start gap-3 transition-all duration-500",
        step.status === 'spawning' && "animate-in fade-in-0 slide-in-from-left-2",
        step.status === 'complete' && "opacity-60"
      )}
    >
      {/* Step indicator */}
      <div className="mt-0.5 flex-shrink-0">
        {step.status === 'complete' ? (
          <CheckCircle2 className="w-4 h-4 text-emerald-500 animate-in zoom-in-50 duration-300" />
        ) : step.status === 'spawning' ? (
          <div className="w-4 h-4 rounded-full border-2 border-primary animate-pulse" />
        ) : (
          <div className="w-4 h-4 rounded-full border-2 border-primary bg-primary/20">
            <div className="w-full h-full rounded-full animate-ping bg-primary/40" />
          </div>
        )}
      </div>

      {/* Step label */}
      <div className="flex-1 min-w-0">
        {step.status === 'spawning' ? (
          <Shimmer duration={1.5} className="text-sm text-foreground">
            {step.label}
          </Shimmer>
        ) : (
          <span className={cn(
            "text-sm transition-colors duration-300",
            step.status === 'active' && "text-foreground",
            step.status === 'complete' && "text-muted-foreground"
          )}>
            {step.label}
          </span>
        )}
      </div>

      {/* Active spinner */}
      {step.status === 'active' && (
        <Loader size={14} className="text-primary flex-shrink-0" />
      )}
    </div>
  )
}

function AssistantMessage({
  content,
  agentResults,
  thinkingDuration,
}: {
  content: string
  agentResults?: Record<string, AgentResult>
  thinkingDuration?: number
}) {
  const paragraphs = content.split('\n\n').filter(Boolean)

  return (
    <div className="response-container max-w-prose">
      {thinkingDuration !== undefined && thinkingDuration > 0 && (
        <div className="flex items-center gap-2 text-xs text-muted-foreground mb-4">
          <BrainIcon className="w-3 h-3" />
          <span>Thought for {thinkingDuration}s</span>
        </div>
      )}

      <div className="space-y-4">
        {paragraphs.map((paragraph, index) => (
          <p
            key={index}
            className="response-paragraph text-[15px] leading-relaxed text-foreground"
          >
            {paragraph}
          </p>
        ))}
      </div>

      {agentResults && Object.keys(agentResults).length > 0 && (
        <div className="mt-6 pt-4 border-t border-border">
          <p className="text-[11px] uppercase tracking-wider mb-3 text-muted-foreground">
            Sources
          </p>
          <div className="flex flex-wrap gap-2">
            {Object.entries(agentResults).map(([key, result]) => (
              <AgentSource key={key} name={key} result={result} />
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

function AgentSource({ name, result }: { name: string; result: AgentResult }) {
  const [expanded, setExpanded] = useState(false)

  return (
    <div className="relative">
      <button
        onClick={() => setExpanded(!expanded)}
        className="source-chip"
      >
        <span className="indicator" />
        <span>{result.agent}</span>
        <span className="text-muted-foreground/50">
          {(result.time * 1000).toFixed(0)}ms
        </span>
      </button>

      {expanded && (
        <div className="absolute top-full left-0 mt-2 p-4 rounded-xl shadow-lg z-10 min-w-[280px] bg-popover border border-border fade-in">
          <ResultPreview output={result.output} />
        </div>
      )}
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
      <ul className="space-y-2">
        {insights.slice(0, 3).map((insight, i) => (
          <li key={i} className="flex items-start gap-2 text-[13px] text-muted-foreground">
            <span className="text-primary">•</span>
            <span>{insight}</span>
          </li>
        ))}
      </ul>
    )
  }

  if (code) {
    return (
      <pre className="text-xs overflow-x-auto rounded-lg">
        <code>{code.slice(0, 300)}{code.length > 300 ? '...' : ''}</code>
      </pre>
    )
  }

  if (typeof isValid === 'boolean') {
    return (
      <p className={cn("text-[13px] flex items-center gap-2", isValid ? "text-emerald-500" : "text-red-500")}>
        <span>{isValid ? '✓' : '✗'}</span>
        <span>{isValid ? 'Validation passed' : 'Validation failed'}</span>
      </p>
    )
  }

  if (description) {
    return (
      <p className="text-[13px] line-clamp-3 text-muted-foreground">
        {description}
      </p>
    )
  }

  return (
    <p className="text-[13px] text-muted-foreground">
      Processing complete
    </p>
  )
}
