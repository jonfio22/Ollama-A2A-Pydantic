/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  async rewrites() {
    return [
      // Proxy API requests to the orchestrator
      {
        source: '/api/orchestrator/:path*',
        destination: 'http://localhost:8000/:path*',
      },
      {
        source: '/api/analyst/:path*',
        destination: 'http://localhost:8001/:path*',
      },
      {
        source: '/api/coder/:path*',
        destination: 'http://localhost:8002/:path*',
      },
      {
        source: '/api/validator/:path*',
        destination: 'http://localhost:8003/:path*',
      },
      {
        source: '/api/vision/:path*',
        destination: 'http://localhost:8004/:path*',
      },
    ]
  },
}

module.exports = nextConfig
