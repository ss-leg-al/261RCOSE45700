/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      animation: {
        "gradient-x":  "gradientX 5s ease infinite",
        "fade-in":     "fadeIn 0.2s ease-out",
        "slide-up":    "slideUp 0.25s ease-out",
        "slide-in-left": "slideInLeft 0.25s ease-out",
        "glow-pulse":  "glowPulse 2.5s ease-in-out infinite",
        "spin-slow":   "spin 3s linear infinite",
      },
      keyframes: {
        gradientX: {
          "0%, 100%": { backgroundPosition: "0% 50%" },
          "50%":      { backgroundPosition: "100% 50%" },
        },
        fadeIn: {
          from: { opacity: "0" },
          to:   { opacity: "1" },
        },
        slideUp: {
          from: { opacity: "0", transform: "translateY(10px)" },
          to:   { opacity: "1", transform: "translateY(0)" },
        },
        slideInLeft: {
          from: { opacity: "0", transform: "translateX(-8px)" },
          to:   { opacity: "1", transform: "translateX(0)" },
        },
        glowPulse: {
          "0%, 100%": { opacity: "0.4", transform: "scale(1)" },
          "50%":      { opacity: "1",   transform: "scale(1.15)" },
        },
      },
    },
  },
  plugins: [],
};
