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

const W = 512;
const H = 256;

export class HUD {
  constructor(scene, { position = [0.55, 1.15, 0], lookAt = [0, 1.2, 0.4] } = {}) {
    this.canvas = document.createElement('canvas');
    this.canvas.width = W;
    this.canvas.height = H;
    this.ctx = this.canvas.getContext('2d');
    this.texture = new THREE.CanvasTexture(this.canvas);
    this.texture.colorSpace = THREE.SRGBColorSpace;

    const geo = new THREE.PlaneGeometry(0.44, 0.22);
    const mat = new THREE.MeshBasicMaterial({ map: this.texture, transparent: true });
    this.panel = new THREE.Mesh(geo, mat);
    this.panel.position.set(...position);
    this.panel.lookAt(new THREE.Vector3(...lookAt));
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
   * @param {number} s.physHz
   */
  update(s) {
    const c = this.ctx;
    c.clearRect(0, 0, W, H);
    c.fillStyle = 'rgba(12,14,18,0.88)';
    roundRect(c, 0, 0, W, H, 16);
    c.fill();

    const accent = s.status === 'success' ? '#4ade80'
      : s.status === 'failure' || s.status === 'timeout' ? '#f87171' : '#38bdf8';

    c.fillStyle = accent;
    c.font = '600 13px ui-monospace, Menlo, monospace';
    c.fillText(s.levelId.toUpperCase(), 20, 30);

    c.fillStyle = '#f8fafc';
    c.font = '600 26px system-ui, sans-serif';
    c.fillText(s.levelTitle, 20, 62);

    c.fillStyle = '#94a3b8';
    c.font = '15px system-ui, sans-serif';
    wrap(c, s.hint, 20, 90, W - 40, 20, 2);

    // progress
    c.fillStyle = '#1e293b';
    roundRect(c, 20, 148, W - 40, 12, 6); c.fill();
    c.fillStyle = accent;
    roundRect(c, 20, 148, Math.max((W - 40) * (s.progress || 0), 12), 12, 6); c.fill();

    c.fillStyle = '#cbd5e1';
    c.font = '600 18px ui-monospace, Menlo, monospace';
    c.fillText(`${Math.max(s.timeLeft, 0).toFixed(0)}s`, 20, 196);
    if (s.readout) {
      c.fillStyle = '#94a3b8';
      c.font = '15px ui-monospace, Menlo, monospace';
      c.fillText(s.readout, 76, 196);
    }

    c.fillStyle = '#64748b';
    c.font = '13px ui-monospace, Menlo, monospace';
    c.fillText(`gate ${s.gate.passed}/${s.gate.needed} of last ${s.gate.window}`, 20, 226);
    c.textAlign = 'right';
    c.fillText(`${s.physHz | 0} Hz phys`, W - 20, 226);
    c.textAlign = 'left';

    if (s.debug) {
      c.fillStyle = '#fbbf24';
      c.font = '11px ui-monospace, Menlo, monospace';
      const lines = Array.isArray(s.debug) ? s.debug : [s.debug];
      lines.forEach((line, i) => c.fillText(line, 20, 238 + i * 12));
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
