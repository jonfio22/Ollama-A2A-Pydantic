import { A2ARequest, A2AResponse, AgentHealth, AgentName } from '@/types'
import { AGENTS } from './agents'

// Generate unique request IDs
export function generateRequestId(): string {
  return `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
}

// Check agent health
export async function checkAgentHealth(agent: AgentName): Promise<AgentHealth> {
  const config = AGENTS[agent]
  const startTime = Date.now()

  try {
    const response = await fetch(`/api/${agent}/health`, {
      method: 'GET',
      signal: AbortSignal.timeout(5000),
    })

    if (response.ok) {
      return {
        name: agent,
        status: 'healthy',
        responseTime: Date.now() - startTime,
        lastCheck: Date.now(),
      }
    }
    throw new Error('Health check failed')
  } catch {
    return {
      name: agent,
      status: 'unhealthy',
      lastCheck: Date.now(),
    }
  }
}

// Check all agents health
export async function checkAllAgentsHealth(): Promise<Record<AgentName, AgentHealth>> {
  const agents = Object.keys(AGENTS) as AgentName[]
  const results = await Promise.all(agents.map(checkAgentHealth))

  return results.reduce((acc, health) => {
    acc[health.name] = health
    return acc
  }, {} as Record<AgentName, AgentHealth>)
}

// Send message to orchestrator
export async function sendMessage(
  message: string,
  contextId?: string,
  onUpdate?: (update: { status: string; agent?: AgentName }) => void
): Promise<A2AResponse> {
  const requestId = generateRequestId()

  const payload: A2ARequest = {
    jsonrpc: '2.0',
    id: requestId,
    method: 'run',
    params: {
      message,
      context_id: contextId,
    },
  }

  onUpdate?.({ status: 'sending' })

  try {
    const response = await fetch('/api/orchestrator/a2a/run', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    })

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`)
    }

    const data: A2AResponse = await response.json()

    onUpdate?.({ status: 'completed' })

    return data
  } catch (error) {
    onUpdate?.({ status: 'error' })
    throw error
  }
}

// Get agent metadata
export async function getAgentMetadata(agent: AgentName) {
  try {
    const response = await fetch(`/api/${agent}/.well-known/agent.json`)
    if (response.ok) {
      return await response.json()
    }
    return null
  } catch {
    return null
  }
}
