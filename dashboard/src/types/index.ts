// Agent Types
export type AgentName = 'orchestrator' | 'analyst' | 'coder' | 'validator' | 'vision'

export interface AgentConfig {
  name: AgentName
  displayName: string
  port: number
  model: string
  modelSize: string
  description: string
  tools: string[]
  color: {
    glow: string
    core: string
    dim: string
  }
}

export interface AgentHealth {
  name: AgentName
  status: 'healthy' | 'unhealthy' | 'unknown'
  responseTime?: number
  lastCheck: number
}

// Task Types
export type TaskStatus = 'pending' | 'delegating' | 'processing' | 'completed' | 'failed'

export interface TaskResult {
  agent: AgentName
  output: Record<string, unknown>
  executionTime: number
  success: boolean
}

export interface Task {
  id: string
  message: string
  status: TaskStatus
  createdAt: number
  updatedAt: number
  activeAgent?: AgentName
  results?: Record<string, TaskResult>
  synthesis?: string
  nextActions?: string[]
  executionStrategy?: 'direct' | 'sequential' | 'parallel'
  totalTime?: number
  error?: string
}

// Message Types for Chat
export interface ChatMessage {
  id: string
  role: 'user' | 'assistant' | 'system'
  content: string
  timestamp: number
  task?: Task
  isThinking?: boolean
}

// A2A Protocol Types
export interface A2ARequest {
  jsonrpc: '2.0'
  id: string
  method: 'run'
  params: {
    message: string
    context_id?: string
    artifacts?: unknown[]
  }
}

export interface A2AResponse {
  jsonrpc: '2.0'
  id: string
  result?: {
    output: Record<string, unknown>
    metadata: {
      model: string
      messages_count: number
      context_id?: string
      cost?: {
        total_tokens: number
        request_tokens: number
        response_tokens: number
      }
    }
  }
  error?: {
    code: number
    message: string
    data?: unknown
  }
}

// Agent Output Types
export interface OrchestratorOutput {
  task_results: Record<string, TaskResult>
  synthesis: string
  next_actions: string[]
  execution_strategy: string
  total_time: number
}

export interface AnalysisOutput {
  insights: string[]
  metrics: Record<string, number>
  recommendations: string[]
  confidence_score: number
  reasoning?: string
}

export interface CodeOutput {
  code: string
  explanation: string
  tests?: string
  dependencies: string[]
  confidence: number
}

export interface ValidationOutput {
  is_valid: boolean
  issues: string[]
  suggestions: string[]
  score: number
}

export interface VisionOutput {
  visual_description: string
  detected_objects: string[]
  extracted_text?: string
  confidence_score: number
  reasoning?: string
  recommendations: string[]
}
