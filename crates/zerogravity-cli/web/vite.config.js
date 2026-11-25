import { defineConfig } from "vite";
import solid from "vite-plugin-solid";

export default defineConfig({
  base: "/ui/",
  plugins: [solid()],
  build: {
    outDir: "dist",
    emptyOutDir: true,
    cssCodeSplit: false,
    sourcemap: false,
    minify: "esbuild",
    license: true,
    rollupOptions: {
      output: {
        entryFileNames: "app.js",
        assetFileNames: "style[extname]",
      },
    },
  },
});
