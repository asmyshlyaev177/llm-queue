import { defineConfig } from 'tsup'

export default defineConfig({
  entry: ['src/core.ts', 'src/browser.ts', 'src/node.ts'],
  format: ['esm'],
  dts: true,
  clean: true,
  sourcemap: true,
  // `ollama` is an optional peer; never bundle it. Browser/SW consumers only
  // import ./core and ./browser, so this import is never pulled into them.
  external: ['ollama'],
})
