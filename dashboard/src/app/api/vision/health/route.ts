import { NextResponse } from 'next/server'

const VISION_URL = process.env.VISION_URL || 'http://localhost:8004'

export async function GET() {
  try {
    const response = await fetch(`${VISION_URL}/health`, {
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
