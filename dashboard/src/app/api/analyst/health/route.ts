import { NextResponse } from 'next/server'

const ANALYST_URL = process.env.ANALYST_URL || 'http://localhost:8001'

export async function GET() {
  try {
    const response = await fetch(`${ANALYST_URL}/health`, {
      method: 'GET',
      signal: AbortSignal.timeout(5000),
    })

    if (response.ok) {
      return NextResponse.json({ status: 'healthy' })
    }

    return NextResponse.json({ status: 'unhealthy' }, { status: response.status })
  } catch {
    return NextResponse.json({ status: 'unhealthy' }, { status: 503 })
  }
}
