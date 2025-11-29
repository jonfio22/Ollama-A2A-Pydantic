import type { Config } from 'tailwindcss'

const config: Config = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        // Refined dark palette - easy on the eyes
        bg: {
          primary: '#131316',
          secondary: '#1c1c1f',
          tertiary: '#232326',
          elevated: '#2a2a2e',
        },
        border: {
          subtle: '#2a2a2e',
          default: '#3a3a3e',
        },
        text: {
          primary: '#ececec',
          secondary: '#a0a0a0',
          muted: '#6a6a6a',
        },
        // Subtle accent - muted blue
        accent: '#6b8afd',
      },
      fontFamily: {
        sans: ['var(--font-sans)', 'system-ui', 'sans-serif'],
      },
      maxWidth: {
        'chat': '680px',
      },
    },
  },
  plugins: [],
}
export default config
