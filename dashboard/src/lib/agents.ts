import { AgentName } from '@/types'

export const AGENTS: Record<AgentName, { displayName: string; port: number }> = {
  orchestrator: { displayName: 'Orchestrator', port: 8000 },
  analyst: { displayName: 'Analyst', port: 8001 },
  coder: { displayName: 'Coder', port: 8002 },
  validator: { displayName: 'Validator', port: 8003 },
  vision: { displayName: 'Vision', port: 8004 },
}
