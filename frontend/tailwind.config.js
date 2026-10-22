/** @type {import('tailwindcss').Config} */
export default {
  darkMode: "class",
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: {
          DEFAULT: "#F8F9F4",
          card: "#FFFFFF",
        },
        accent: {
          real: "#28734D",
          fake: "#B94B45",
          cyan: "#164D3B",
        },
      },
    },
  },
  plugins: [],
};
