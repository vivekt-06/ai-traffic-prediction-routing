/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: { 950: "#080d18", 900: "#0b1220", 800: "#111a2e", 700: "#18233b", 600: "#22304d", 500: "#31425f" },
        accent: { DEFAULT: "#2dd4bf", dim: "#14b8a6", blue: "#4f8df7" },
        lv: { low: "#2fbf71", moderate: "#f2c94c", high: "#f2994a", severe: "#eb5757" },
      },
      fontFamily: { sans: ["Inter", "ui-sans-serif", "system-ui", "Segoe UI", "sans-serif"] },
    },
  },
  plugins: [],
};
