/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      fontFamily: {
        sans: ['"Space Grotesk"', 'sans-serif'],
        display: ['"Sora"', 'sans-serif'],
      },
      colors: {
        night: '#061018',
        glass: 'rgba(11, 29, 43, 0.58)',
        borderglass: 'rgba(154, 194, 228, 0.22)',
        accent: '#22d3ee',
        mint: '#34d399',
        skyline: '#38bdf8',
        danger: '#fb7185',
      },
      boxShadow: {
        glass: '0 18px 40px rgba(0, 0, 0, 0.32)',
      },
      backdropBlur: {
        xs: '2px',
      },
    },
  },
  plugins: [],
}

