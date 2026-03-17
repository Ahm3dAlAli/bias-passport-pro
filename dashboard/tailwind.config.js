/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'bias-red': '#ef4444',
        'bias-orange': '#f97316',
        'bias-yellow': '#eab308',
        'bias-green': '#22c55e',
        'bias-blue': '#3b82f6',
      },
    },
  },
  plugins: [],
}
