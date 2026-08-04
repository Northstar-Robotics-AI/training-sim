import { defineConfig } from 'vite';

// Cross-origin isolation headers.
//
// The single-threaded MuJoCo build at the root of @mujoco/mujoco works without
// these. The multi-threaded build in @mujoco/mujoco/mt uses SharedArrayBuffer,
// which browsers gate behind COOP/COEP. Setting them here from day one means
// switching to /mt later is a one-line import change rather than a deployment
// problem discovered at the worst moment.
//
// Note this also means any cross-origin asset you pull in needs CORP headers,
// which is why the meshes are served from /public rather than a CDN.
const isolation = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
};

export default defineConfig({
  server: { headers: isolation, https: false, allowedHosts: ['.ngrok-free.dev', '.ngrok-free.app'] },
  preview: { headers: isolation },
  optimizeDeps: { exclude: ['@mujoco/mujoco'] },
  assetsInclude: ['**/*.wasm', '**/*.stl'],
});
