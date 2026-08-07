// Operator HUD.
//
// Two surfaces for the same state: a DOM overlay for desktop preview, and a
// canvas texture on a world-locked quad for inside the headset. DOM overlays do
// not composite into an immersive WebXR session, so the canvas is not optional.
//
// The panel is deliberately parked at the back edge of the table rather than
// head-locked. Head-locked text in VR is read constantly and pulls attention
// off the hands; a fixed panel gets glanced at between attempts, which is when
// feedback is actually useful.

import * as THREE from 'three';
import { metricLabel, formatMetricValue } from '../levels/Level.js';

const W = 640;
const H = 440;

// Must match curriculum.js's TABLE_Z (not shared/exported -- that file has no
// other reason to expose it). The panel's bottom edge is pinned to this.
const TABLE_Z = 0.74;
const PANEL_W = 0.78;
const PANEL_H = PANEL_W * (H / W);

export class HUD {
  // Parked further back than the old spot, which sat close enough that
  // lookAt()'s off-axis target (aimed near the arm base rather than at the
  // operator) read as a visible tilt. Bigger physical size compensates for
  // the added distance -- and buys the room the failure readout and live
  // metric rows below need. Bottom edge sits at table height so the panel
  // reads as standing on the table rather than floating above it.
  constructor(scene, { position = [0.9, TABLE_Z + PANEL_H / 2, 0] } = {}) {
    this.canvas = document.createElement('canvas');
    this.canvas.width = W;
    this.canvas.height = H;
    this.ctx = this.canvas.getContext('2d');
    this.texture = new THREE.CanvasTexture(this.canvas);
    this.texture.colorSpace = THREE.SRGBColorSpace;

    const geo = new THREE.PlaneGeometry(PANEL_W, PANEL_H);
    const mat = new THREE.MeshBasicMaterial({ map: this.texture, transparent: true, side: THREE.DoubleSide });
    this.panel = new THREE.Mesh(geo, mat);
    this.panel.position.set(...position);
    // Pure yaw, no pitch/roll -- squarely faces back down -X, the side both
    // the desktop camera and the VR spawn sit on, instead of lookAt()'s
    // skewed target which baked in an off-axis tilt.
    this.panel.quaternion.setFromAxisAngle(new THREE.Vector3(0, 1, 0), -Math.PI / 2);
    scene.add(this.panel);

    this.dom = document.getElementById('overlay');
    this._last = '';
  }

  /**
   * @param {object} s
   * @param {string} s.levelTitle
   * @param {string} s.hint
   * @param {number} s.progress   0..1
   * @param {number} s.timeLeft   seconds
   * @param {string} s.readout    level-specific line
   * @param {string} s.status     'running' | 'success' | 'failure' | 'timeout'
   * @param {object} s.gate       {passed, window, needed}
   * @param {string[]} s.reasons  why the last finished episode didn't count
   *   toward the gate -- empty for a clean success. Persists through the
   *   post-episode pause, cleared the moment the next episode starts.
   * @param {{key, value, limit}[]} s.metrics  live quality-metric readout,
   *   `limit` undefined wherever this level doesn't gate on that key.
   * @param {number} s.physHz
   */
  update(s) {
    const c = this.ctx;
    const M = 24; // left/right content margin
    c.clearRect(0, 0, W, H);
    c.fillStyle = 'rgba(12,14,18,0.88)';
    roundRect(c, 0, 0, W, H, 16);
    c.fill();

    const accent = s.status === 'success' ? '#4ade80'
      : s.status === 'failure' || s.status === 'timeout' ? '#f87171' : '#38bdf8';

    c.fillStyle = accent;
    c.font = '600 15px ui-monospace, Menlo, monospace';
    c.fillText(s.levelId.toUpperCase(), M, 34);

    c.fillStyle = '#f8fafc';
    c.font = '600 28px system-ui, sans-serif';
    c.fillText(s.levelTitle, M, 76);

    c.fillStyle = '#94a3b8';
    c.font = '16px system-ui, sans-serif';
    wrap(c, s.hint, M, 106, W - 2 * M, 22, 2);

    // progress
    c.fillStyle = '#1e293b';
    roundRect(c, M, 172, W - 2 * M, 14, 7); c.fill();
    c.fillStyle = accent;
    roundRect(c, M, 172, Math.max((W - 2 * M) * (s.progress || 0), 14), 14, 7); c.fill();

    c.fillStyle = '#cbd5e1';
    c.font = '600 20px ui-monospace, Menlo, monospace';
    c.fillText(`${Math.max(s.timeLeft, 0).toFixed(0)}s`, M, 216);
    if (s.readout) {
      c.fillStyle = '#94a3b8';
      c.font = '16px ui-monospace, Menlo, monospace';
      c.fillText(s.readout, M + 60, 216);
    }

    // Why the last episode didn't count, if it didn't. Fixed slot (blank on a
    // clean success) so the metrics rows below don't jump around between draws.
    if (s.reasons && s.reasons.length) {
      c.fillStyle = s.status === 'success' ? '#fbbf24' : '#f87171';
      c.font = '600 15px ui-monospace, Menlo, monospace';
      wrap(c, s.reasons.join('  ·  '), M, 250, W - 2 * M, 18, 2);
    }

    // Live quality metrics -- the numbers a failure/no-count is actually made
    // of, updated every frame so a rising jerk or overforce total is visible
    // *during* the attempt, not just explained after the fact.
    if (s.metrics) {
      c.font = '14px ui-monospace, Menlo, monospace';
      s.metrics.forEach((m, i) => {
        const y = 302 + i * 21;
        const val = formatMetricValue(m.key, m.value);
        const over = m.limit !== undefined && m.value > m.limit;
        const near = m.limit !== undefined && !over && m.value > m.limit * 0.8;
        c.fillStyle = '#64748b';
        c.fillText(metricLabel(m.key), M, y);
        c.fillStyle = over ? '#f87171' : near ? '#fbbf24' : m.limit !== undefined ? '#4ade80' : '#94a3b8';
        c.textAlign = 'right';
        const label = m.limit !== undefined
          ? `${val} / ${formatMetricValue(m.key, m.limit)}`
          : val;
        c.fillText(label, W - M, y);
        c.textAlign = 'left';
      });
    }

    c.fillStyle = '#64748b';
    c.font = '13px ui-monospace, Menlo, monospace';
    c.fillText(`gate ${s.gate.passed}/${s.gate.needed} of last ${s.gate.window}`, M, 400);
    c.textAlign = 'right';
    c.fillText(`${s.physHz | 0} Hz phys`, W - M, 400);
    c.textAlign = 'left';

    if (s.debug) {
      c.fillStyle = '#fbbf24';
      c.font = '11px ui-monospace, Menlo, monospace';
      const lines = Array.isArray(s.debug) ? s.debug : [s.debug];
      lines.forEach((line, i) => c.fillText(line, M, 418 + i * 12));
    }

    this.texture.needsUpdate = true;

    if (this.dom) {
      const line = `${s.levelId} · ${s.levelTitle} · ${(s.progress * 100) | 0}% · `
        + `${Math.max(s.timeLeft, 0).toFixed(0)}s · ${s.physHz | 0} Hz`;
      if (line !== this._last) { this.dom.textContent = line; this._last = line; }
    }
  }
}

function roundRect(c, x, y, w, h, r) {
  c.beginPath();
  c.moveTo(x + r, y);
  c.arcTo(x + w, y, x + w, y + h, r);
  c.arcTo(x + w, y + h, x, y + h, r);
  c.arcTo(x, y + h, x, y, r);
  c.arcTo(x, y, x + w, y, r);
  c.closePath();
}

function wrap(c, text, x, y, maxW, lh, maxLines) {
  const words = String(text || '').split(' ');
  let line = '';
  let n = 0;
  for (const w of words) {
    const test = line ? `${line} ${w}` : w;
    if (c.measureText(test).width > maxW && line) {
      c.fillText(line, x, y + n * lh);
      line = w;
      if (++n >= maxLines) return;
    } else {
      line = test;
    }
  }
  c.fillText(line, x, y + n * lh);
}
