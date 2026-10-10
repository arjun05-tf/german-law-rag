/* ===========================================================================
   Corpus field — the index drawn as a plane.

   Every coordinate here is real: `xy` comes from serve.py, which fits a 2-D PCA
   on the actual 768-dim passage vectors in Qdrant and projects the query with
   the same fitted transform. Point area is the real character count of the
   Absatz. Spokes connect each Absatz to the centroid of its Abschnitt, which is
   real metadata parsed out of the XML.

   What it is NOT: the space retrieval runs in. Ranking uses cosine over all 768
   dimensions. These two components retain ~18% of the variance, so distance on
   screen is a lossy shadow of distance in the index. The plate says so.

   One canvas instance is moved between mount points rather than recreated, so
   the drift state survives the hero -> results transition.
   =========================================================================== */

const SECTION_TINT = [
  '#6f8fbe', '#b8795a', '#6fa58c', '#a8779f',
  '#8e9a5f', '#c2452c', '#5f8fa5', '#9a8458',
];

const reduced = matchMedia('(prefers-reduced-motion: reduce)');

export class CorpusField {
  constructor() {
    this.canvas = document.createElement('canvas');
    this.canvas.className = 'field-canvas';
    Object.assign(this.canvas.style, {
      position: 'absolute', inset: '0', width: '100%', height: '100%', display: 'block',
    });
    this.ctx = this.canvas.getContext('2d');

    this.chunks = [];
    this.sections = [];
    this.mode = 'hero';
    this.result = null;
    this.w = 0; this.h = 0; this.dpr = 1;
    this.t = 0;
    this.reveal = 0;          // 0..1 — drives the query-state entrance
    this.raf = null;
    this.pointer = { x: 0, y: 0, active: false };
    this._lastFrame = 0;

    this.ro = new ResizeObserver(() => this.resize());
    this._onPointer = (e) => {
      const r = this.canvas.getBoundingClientRect();
      this.pointer.x = (e.clientX - r.left) / r.width * 2 - 1;
      this.pointer.y = (e.clientY - r.top) / r.height * 2 - 1;
      this.pointer.active = true;
    };
    this._onLeave = () => { this.pointer.active = false; };
    this._onVis = () => { document.hidden ? this.stop() : this.start(); };
    document.addEventListener('visibilitychange', this._onVis);
  }

  setData(chunks) {
    this.chunks = chunks.map((c) => ({ ...c, x: c.xy[0], y: c.xy[1] }));

    const names = [...new Set(this.chunks.map((c) => c.abschnitt))];
    this.sections = names.map((name, i) => {
      const members = this.chunks.filter((c) => c.abschnitt === name);
      return {
        name,
        short: shortSection(name),
        tint: SECTION_TINT[i % SECTION_TINT.length],
        cx: avg(members.map((m) => m.x)),
        cy: avg(members.map((m) => m.y)),
        n: members.length,
      };
    });
    this.fit();
    const byName = new Map(this.sections.map((s) => [s.name, s]));
    const maxChars = Math.max(...this.chunks.map((c) => c.chars));
    for (const c of this.chunks) {
      c.sec = byName.get(c.abschnitt);
      // area proportional to length, so radius goes as sqrt
      c.r = 1.9 + 3.1 * Math.sqrt(c.chars / maxChars);
      c.seed = Math.random() * Math.PI * 2;
    }
    this.draw();
  }

  /** Bounding box of everything that must be visible, incl. the query marker.
      The PCA output is only normalised so max|xy| = 1; the cloud itself is
      lopsided, so assuming it fills the box wastes most of the plate. */
  fit() {
    const pts = this.chunks.map((c) => [c.x, c.y]);
    if (this.result?.query_xy) pts.push(this.result.query_xy);
    if (!pts.length) { this.bounds = { x0: -1, x1: 1, y0: -1, y1: 1 }; return; }
    const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
    const mx = (Math.max(...xs) - Math.min(...xs)) * 0.06 || 0.1;
    const my = (Math.max(...ys) - Math.min(...ys)) * 0.06 || 0.1;
    this.bounds = {
      x0: Math.min(...xs) - mx, x1: Math.max(...xs) + mx,
      y0: Math.min(...ys) - my, y1: Math.max(...ys) + my,
    };
  }

  mount(slot, mode) {
    this.mode = mode;
    if (this.canvas.parentNode !== slot) slot.appendChild(this.canvas);
    this.ro.disconnect();
    this.ro.observe(slot);
    this.canvas.removeEventListener('pointermove', this._onPointer);
    this.canvas.removeEventListener('pointerleave', this._onLeave);
    if (mode === 'plate') {
      this.canvas.addEventListener('pointermove', this._onPointer);
      this.canvas.addEventListener('pointerleave', this._onLeave);
      this.canvas.style.pointerEvents = 'auto';
    } else {
      this.canvas.style.pointerEvents = 'none';
    }
    this.resize();
    this.start();
  }

  /** Real query result: query_xy plus candidates carrying dense_rank/final_rank. */
  setResult(result) {
    this.result = result;
    this.reveal = reduced.matches ? 1 : 0;
    const keep = new Set((result?.candidates ?? []).map((c) => c.id));
    const rankById = new Map((result?.candidates ?? []).map((c) => [c.id, c]));
    for (const c of this.chunks) {
      c.cand = keep.has(c.id) ? rankById.get(c.id) : null;
    }
    this.fit();
    this.draw();
  }

  clearResult() { this.setResult(null); }

  resize() {
    const slot = this.canvas.parentNode;
    if (!slot) return;
    const r = slot.getBoundingClientRect();
    if (!r.width || !r.height) return;
    // cap DPR: a 3x backing store on a mobile panel costs far more than it shows
    this.dpr = Math.min(devicePixelRatio || 1, 2);
    this.w = r.width; this.h = r.height;
    this.canvas.width = Math.round(r.width * this.dpr);
    this.canvas.height = Math.round(r.height * this.dpr);
    this.draw();
  }

  start() {
    if (this.raf || !this.canvas.parentNode) return;
    if (reduced.matches && this.reveal >= 1) { this.draw(); return; }
    this._lastFrame = performance.now();
    const loop = (now) => {
      const dt = Math.min((now - this._lastFrame) / 1000, 0.05);
      this._lastFrame = now;
      this.t += dt;
      if (this.reveal < 1) this.reveal = Math.min(1, this.reveal + dt / 1.1);
      this.draw();
      // once settled and motion is off, stop burning frames
      if (reduced.matches && this.reveal >= 1) { this.raf = null; return; }
      this.raf = requestAnimationFrame(loop);
    };
    this.raf = requestAnimationFrame(loop);
  }

  stop() {
    if (this.raf) cancelAnimationFrame(this.raf);
    this.raf = null;
  }

  destroy() {
    this.stop();
    this.ro.disconnect();
    document.removeEventListener('visibilitychange', this._onVis);
    this.canvas.remove();
  }

  // ---------------------------------------------------------------- drawing

  project(x, y) {
    const pad = this.mode === 'plate' ? 54 : 46;
    const b = this.bounds || { x0: -1, x1: 1, y0: -1, y1: 1 };
    // one scale for both axes: the projection's aspect must not be distorted
    const s = Math.max(40, Math.min(
      (this.w - pad * 2) / (b.x1 - b.x0 || 1),
      (this.h - pad * 2) / (b.y1 - b.y0 || 1),
    ));
    let px = this.w / 2 + (x - (b.x0 + b.x1) / 2) * s;
    let py = this.h / 2 + (y - (b.y0 + b.y1) / 2) * s;
    if (this.pointer.active && this.mode === 'plate') {
      px += this.pointer.x * 5;
      py += this.pointer.y * 5;
    }
    return [px, py];
  }

  draw() {
    const { ctx } = this;
    if (!this.w || !this.h) return;
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.clearRect(0, 0, this.w, this.h);

    const dark = this.mode === 'plate';
    const idle = !this.result;
    const drift = reduced.matches ? 0 : 1;

    // --- per-point live position -----------------------------------------
    for (const c of this.chunks) {
      const wob = drift * 0.004 * Math.sin(this.t * 0.35 + c.seed);
      const [px, py] = this.project(c.x + wob, c.y + wob * 0.7);
      c.px = px; c.py = py;
    }
    for (const s of this.sections) {
      const [px, py] = this.project(s.cx, s.cy);
      s.px = px; s.py = py;
    }

    // --- spokes: Absatz -> its Abschnitt centroid (real metadata) ---------
    ctx.lineWidth = dark ? 0.75 : 0.65;
    for (const c of this.chunks) {
      const dim = !idle && !c.cand;
      ctx.strokeStyle = hexA(c.sec.tint, dark ? (dim ? 0.1 : 0.3) : (dim ? 0.08 : 0.22));
      ctx.beginPath();
      ctx.moveTo(c.sec.px, c.sec.py);
      ctx.lineTo(c.px, c.py);
      ctx.stroke();
    }

    // --- section centroids ------------------------------------------------
    for (const s of this.sections) {
      ctx.fillStyle = hexA(s.tint, dark ? 0.85 : 0.55);
      ctx.beginPath(); ctx.arc(s.px, s.py, 2.1, 0, 6.284); ctx.fill();
      ctx.strokeStyle = hexA(s.tint, dark ? 0.35 : 0.22);
      ctx.lineWidth = 0.7;
      ctx.beginPath(); ctx.arc(s.px, s.py, 5, 0, 6.284); ctx.stroke();
    }

    // --- the 80 indexed Absätze -------------------------------------------
    for (const c of this.chunks) {
      const dim = !idle && !c.cand;
      ctx.fillStyle = hexA(c.sec.tint, dark ? (dim ? 0.26 : 0.95) : (dim ? 0.3 : 0.75));
      ctx.beginPath(); ctx.arc(c.px, c.py, c.r, 0, 6.284); ctx.fill();
    }

    if (this.result) this.drawResult(dark);
    if (dark) this.drawPlateFurniture();
  }

  drawResult(dark) {
    const { ctx } = this;
    const r = this.result;
    const e = ease(this.reveal);

    const survivors = this.chunks
      .filter((c) => c.cand?.final_rank)
      .sort((a, b) => a.cand.final_rank - b.cand.final_rank);
    const [qx, qy] = this.project(r.query_xy[0], r.query_xy[1]);

    // candidate rings — slate blue, the bi-encoder's shortlist
    for (const c of this.chunks) {
      if (!c.cand) continue;
      const a = 0.55 * e;
      ctx.strokeStyle = hexA('#5b7fb0', dark ? a : a * 0.7);
      ctx.lineWidth = 0.9;
      ctx.beginPath(); ctx.arc(c.px, c.py, c.r + 3.5, 0, 6.284); ctx.stroke();
    }

    // query -> surviving evidence
    for (const [i, c] of survivors.entries()) {
      const p = clamp01((e - i * 0.07) / 0.5);
      if (p <= 0) continue;
      ctx.strokeStyle = hexA('#c2452c', 0.5 * p);
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(qx, qy);
      ctx.lineTo(qx + (c.px - qx) * p, qy + (c.py - qy) * p);
      ctx.stroke();
    }

    // survivors: filled, ranked, labelled
    const labels = [];
    for (const [i, c] of survivors.entries()) {
      const p = clamp01((e - i * 0.07) / 0.5);
      if (p <= 0) continue;
      ctx.fillStyle = hexA('#c2452c', 0.95 * p);
      ctx.beginPath(); ctx.arc(c.px, c.py, c.r + 1.4, 0, 6.284); ctx.fill();
      ctx.strokeStyle = hexA('#e0694e', 0.55 * p);
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.arc(c.px, c.py, c.r + 6.5, 0, 6.284); ctx.stroke();

      if (this.mode === 'plate') {
        const away = Math.sign(c.px - qx) || 1;
        labels.push({
          text: `${c.cand.final_rank}· ${c.citation}`,
          x: c.px + away * (c.r + 10), y: c.py, ty: c.py, away, p, cx: c.px, cy: c.py,
        });
      }
    }

    // Evidence passages can sit on top of each other in the projection, and
    // then so do their labels. Push them apart vertically and draw a leader
    // back to the point, rather than letting two citations overprint.
    if (labels.length) {
      ctx.font = '500 9px "IBM Plex Mono", monospace';
      ctx.textBaseline = 'middle';
      for (const side of [-1, 1]) {
        const col = labels.filter((l) => l.away === side).sort((a, b) => a.ty - b.ty);
        for (let i = 1; i < col.length; i++) {
          const gap = col[i].ty - col[i - 1].ty;
          if (gap < 12) col[i].ty = col[i - 1].ty + 12;
        }
        // keep the stack inside the plate
        const over = (col.at(-1)?.ty ?? 0) - (this.h - 16);
        if (over > 0) for (const l of col) l.ty -= over;
      }
      for (const l of labels) {
        if (Math.abs(l.ty - l.cy) > 2) {
          ctx.strokeStyle = hexA('#e0694e', 0.35 * l.p);
          ctx.lineWidth = 0.7;
          ctx.beginPath();
          ctx.moveTo(l.cx + l.away * 5, l.cy);
          ctx.lineTo(l.x - l.away * 2, l.ty);
          ctx.stroke();
        }
        ctx.textAlign = l.away > 0 ? 'left' : 'right';
        label(ctx, l.text, l.x, l.ty, hexA('#e8e6e1', 0.95 * l.p), l.p);
      }
    }

    // the query itself — a crosshair, not another dot
    const qp = ease(clamp01(this.reveal / 0.35));
    ctx.strokeStyle = hexA('#c2452c', 0.9 * qp);
    ctx.lineWidth = 1.1;
    const arm = 7;
    ctx.beginPath();
    ctx.moveTo(qx - arm, qy); ctx.lineTo(qx + arm, qy);
    ctx.moveTo(qx, qy - arm); ctx.lineTo(qx, qy + arm);
    ctx.stroke();
    ctx.beginPath(); ctx.arc(qx, qy, 3.2, 0, 6.284); ctx.stroke();

    // one expanding ring, settling — the query entering the space
    if (!reduced.matches && this.reveal < 1) {
      const rr = 6 + 54 * ease(this.reveal);
      ctx.strokeStyle = hexA('#c2452c', 0.3 * (1 - this.reveal));
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.arc(qx, qy, rr, 0, 6.284); ctx.stroke();
    }

    if (this.mode === 'plate') {
      ctx.font = '500 9px "IBM Plex Mono", monospace';
      ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
      label(ctx, 'FRAGE', qx + 11, qy - 10, hexA('#c2452c', 0.95 * qp), qp);
    }
  }

  /** Axis hairlines, section labels and the honesty caption. */
  drawPlateFurniture() {
    const { ctx } = this;
    ctx.strokeStyle = '#ffffff0d';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(14, this.h / 2); ctx.lineTo(this.w - 14, this.h / 2);
    ctx.moveTo(this.w / 2, 14); ctx.lineTo(this.w / 2, this.h - 14);
    ctx.stroke();

    ctx.font = '400 8px "IBM Plex Mono", monospace';
    ctx.fillStyle = '#666c75';
    ctx.textBaseline = 'middle';
    ctx.textAlign = 'left';
    ctx.fillText('PC 1', this.w / 2 + 5, this.h - 18);
    ctx.save();
    ctx.translate(16, this.h / 2 - 5);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('PC 2', 0, 0);
    ctx.restore();

    // section labels at their real centroids, only where there is room
    ctx.font = '400 8px "IBM Plex Mono", monospace';
    for (const s of this.sections) {
      if (s.n < 4) continue;
      ctx.textAlign = 'center';
      label(ctx, s.short, s.px, s.py - 7, hexA(s.tint, 0.75), 1);
    }
  }
}

/** Text over a dense scatter needs to carve its own space out of the plot. */
function label(ctx, text, x, y, fill, alpha = 1) {
  ctx.save();
  ctx.lineJoin = 'round';
  ctx.lineWidth = 3;
  ctx.strokeStyle = `rgba(20, 22, 25, ${(0.9 * alpha).toFixed(3)})`;
  ctx.strokeText(text, x, y);
  ctx.fillStyle = fill;
  ctx.fillText(text, x, y);
  ctx.restore();
}

function shortSection(name) {
  // "Zweiter Abschnitt - Werktägliche Arbeitszeit\nund arbeitsfreie Zeiten"
  const ord = (name.split(' ')[0] || '').slice(0, 3).toUpperCase();
  return ord ? `${ord}.` : '—';
}

export function sectionLegend(field) {
  return field.sections
    .slice()
    .sort((a, b) => b.n - a.n)
    .map((s) => {
      const label = s.name.replace(/\s+/g, ' ').split(' - ').slice(1).join(' - ') || s.name;
      return `<span><i style="background:${s.tint}"></i>${esc(label)} <span style="opacity:.6">(${s.n})</span></span>`;
    })
    .join('');
}

const avg = (xs) => xs.reduce((a, b) => a + b, 0) / (xs.length || 1);
const clamp01 = (v) => Math.max(0, Math.min(1, v));
const ease = (t) => 1 - Math.pow(1 - clamp01(t), 3);

function hexA(hex, a) {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a.toFixed(3)})`;
}

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}
