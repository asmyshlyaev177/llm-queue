import { defineConfig } from 'tsup'

export default defineConfig({
  entry: ['src/core.ts', 'src/browser.ts', 'src/server.ts', 'src/cli.ts'],
  format: ['esm'],
  dts: true,
  clean: true,
  sourcemap: true,
})
