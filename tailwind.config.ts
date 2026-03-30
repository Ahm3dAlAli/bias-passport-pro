import type { Config } from 'tailwindcss';

export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        observatory: {
          bg: 'hsl(var(--obs-bg))',
          surface: 'hsl(var(--obs-surface))',
          'surface-alt': 'hsl(var(--obs-surface-alt))',
          border: 'hsl(var(--obs-border))',
          text: 'hsl(var(--obs-text))',
          'text-muted': 'hsl(var(--obs-text-muted))',
          'text-dim': 'hsl(var(--obs-text-dim))',
          accent: 'hsl(var(--obs-accent))',
          'accent-glow': 'hsl(var(--obs-accent-glow))',
          danger: 'hsl(var(--obs-danger))',
          warning: 'hsl(var(--obs-warning))',
          success: 'hsl(var(--obs-success))',
          info: 'hsl(var(--obs-info))',
        },
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'monospace'],
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
    },
  },
  plugins: [],
} satisfies Config;
