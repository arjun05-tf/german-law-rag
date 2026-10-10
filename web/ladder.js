/* ===========================================================================
   Rerank ladder — what the cross-encoder changed.

   All values are measured, per query, by serve.py:
     left  : dense_rank + cosine     (bi-encoder, vectors compared separately)
     right : final_rank + ce_score   (cross-encoder, query and passage read together)

   The point of the plate is the two scale strips in the header. The cosine
   scores of all 20 candidates sit inside a sliver of [0,1]; the cross-encoder
   scores spread across most of it. That is the project's central finding, and
   it is visible in every single query rather than only in the eval table.

   The cosine bars are drawn on a stretched scale (min..max of this query)
   because on a true 0..1 axis all 20 would be indistinguishable. The header
   strip shows exactly how much stretching that is, and the absolute value is
   printed next to every bar.
   =========================================================================== */

const NS = 'http://www.w3.org/2000/svg';
const reduced = matchMedia('(prefers-reduced-motion: reduce)');

export function renderLadder(host, candidates, k) {
  host.textContent = '';
  if (!candidates?.length) return;

  // A hidden or not-yet-laid-out host measures 0. Falling through to the wide
  // layout there renders a 1000-unit viewBox into a phone and the labels become
  // unreadable, so fall back to the viewport instead of to "wide".
  const w = host.clientWidth || host.parentElement?.clientWidth || window.innerWidth;
  const narrow = w < 560;   // keep in sync with LADDER_BREAK in app.js
  const cands = candidates.slice().sort((a, b) => a.dense_rank - b.dense_rank);
  const byCe = cands.filter((c) => c.ce_score != null)
    .slice().sort((a, b) => b.ce_score - a.ce_score);

  const cos = cands.map((c) => c.cosine);
  const cosMin = Math.min(...cos), cosMax = Math.max(...cos);
  const ces = byCe.map((c) => c.ce_score);
  const ceMin = ces.length ? Math.min(...ces) : 0;
  const ceMax = ces.length ? Math.max(...ces) : 1;

  const L = narrow ? layoutNarrow(cands.length) : layoutWide(cands.length);
  const svg = el('svg', {
    viewBox: `0 0 ${L.W} ${L.H}`,
    role: 'img',
    'aria-label': ariaSummary(cands, byCe, k, cosMin, cosMax, ceMin, ceMax),
  });

  // ---------------------------------------------------------------- header
  header(svg, L, { cosMin, cosMax, ceMin, ceMax, narrow, n: cands.length, k });

  const rowsL = new Map();
  const rowsR = new Map();
  cands.forEach((c, i) => rowsL.set(c.id, L.rowY(i)));
  byCe.forEach((c, i) => rowsR.set(c.id, L.rowY(i)));

  // ------------------------------------------------- ribbons (wide only)
  if (!narrow) {
    const gr = el('g', {});
    svg.appendChild(gr);
    for (const c of cands) {
      const y0 = rowsL.get(c.id);
      const y1 = rowsR.get(c.id);
      if (y1 == null) continue;
      const kept = c.final_rank != null && c.final_rank <= k;
      const x0 = L.leftEnd, x1 = L.rightStart;
      const mx = (x0 + x1) / 2;
      const p = el('path', {
        class: 'lad-ribbon',
        d: `M${x0} ${y0} C${mx} ${y0} ${mx} ${y1} ${x1} ${y1}`,
        stroke: kept ? 'var(--ce)' : 'var(--dense)',
        'stroke-width': kept ? 1.7 : 0.7,
        'stroke-opacity': kept ? 0.9 : 0.28,
        'data-id': c.id,
      });
      gr.appendChild(p);
      if (!reduced.matches) animateRibbon(p, kept ? 0.12 : 0.02);
    }
  }

  // -------------------------------------------------------------- left side
  cands.forEach((c, i) => {
    const y = L.rowY(i);
    const kept = c.final_rank != null && c.final_rank <= k;
    const g = el('g', { class: `lad-row${kept ? ' is-kept' : ''}`, 'data-id': c.id });
    svg.appendChild(g);

    g.appendChild(txt(L.lRank, y, String(c.dense_rank), 'lad-rank', 'end'));
    g.appendChild(txt(L.lCite, y, c.citation, 'lad-cite', 'start'));

    const frac = cosMax > cosMin ? (c.cosine - cosMin) / (cosMax - cosMin) : 0.5;
    bar(g, L.lBar, y, L.barW, frac, 'var(--dense)', kept ? 0.95 : 0.4);
    g.appendChild(txt(L.lVal, y, c.cosine.toFixed(3), 'lad-val', 'start'));

    if (narrow) {
      // no room for ribbons: state the move explicitly instead
      const move = c.final_rank == null
        ? '→ aus'
        : `→ ${c.final_rank}`;
      const t = txt(L.nMove, y, move, 'lad-val', 'start');
      t.setAttribute('fill', c.final_rank == null ? 'var(--plate-ink-3)'
        : c.final_rank < c.dense_rank ? '#5aa07a' : c.final_rank > c.dense_rank ? '#c99a3f' : 'var(--plate-ink-2)');
      g.appendChild(t);
      if (c.ce_score != null) {
        bar(g, L.nCeBar, y, L.nCeW, clamp01(c.ce_score), 'var(--ce)', kept ? 0.95 : 0.35);
        g.appendChild(txt(L.nCeVal, y, c.ce_score.toFixed(3), 'lad-val', 'start'));
      }
    }
    if (!reduced.matches) fadeIn(g, i * 0.018);
  });

  // ------------------------------------------------------------- right side
  if (!narrow) {
    byCe.forEach((c, i) => {
      const y = L.rowY(i);
      const kept = c.final_rank != null && c.final_rank <= k;
      const g = el('g', { class: `lad-row${kept ? ' is-kept' : ''}`, 'data-id': c.id });
      svg.appendChild(g);

      g.appendChild(txt(L.rRank, y, kept ? String(c.final_rank) : '·', 'lad-rank', 'end'));
      bar(g, L.rBar, y, L.barW, clamp01(c.ce_score), 'var(--ce)', kept ? 0.95 : 0.3);
      g.appendChild(txt(L.rVal, y, c.ce_score.toFixed(3), 'lad-val', 'start'));
      const cite = txt(L.rCite, y, c.citation, 'lad-cite', 'start');
      g.appendChild(cite);
      if (!reduced.matches) fadeIn(g, 0.25 + i * 0.018);
    });

    // the cut line: everything above it becomes the answer's evidence
    const cutY = L.rowY(k - 1) + L.rowH / 2;
    svg.appendChild(el('line', {
      x1: L.rightStart - 6, y1: cutY, x2: L.W - 4, y2: cutY,
      stroke: 'var(--ce)', 'stroke-width': 0.8, 'stroke-dasharray': '3 3', 'stroke-opacity': 0.75,
    }));
    const cut = txt(L.W - 4, cutY - 7, `k = ${k} → Antwort`, 'lad-note', 'end');
    cut.setAttribute('fill', 'var(--ce-lt)');
    svg.appendChild(cut);
  }

  host.appendChild(svg);
  wireHover(svg);
  return svg;
}

// ------------------------------------------------------------------ layouts

function layoutWide(n) {
  const rowH = 19, top = 74, bottom = 26;
  return {
    W: 1000, H: top + n * rowH + bottom, rowH, barW: 104,
    rowY: (i) => top + i * rowH + rowH / 2,
    lRank: 20, lCite: 28, lBar: 128, lVal: 238,
    leftEnd: 282, rightStart: 600,
    rRank: 628, rBar: 636, rVal: 748, rCite: 790,
    headY: 30,
  };
}

function layoutNarrow(n) {
  const rowH = 23, top = 92, bottom = 22;
  return {
    W: 420, H: top + n * rowH + bottom, rowH, barW: 56,
    rowY: (i) => top + i * rowH + rowH / 2,
    lRank: 14, lCite: 20, lBar: 118, lVal: 178,
    nMove: 214, nCeBar: 262, nCeW: 76, nCeVal: 346,
    headY: 30,
  };
}

// ------------------------------------------------------------------- header

function header(svg, L, o) {
  const { cosMin, cosMax, ceMin, ceMax, narrow, n, k } = o;
  const g = el('g', {});
  svg.appendChild(g);

  if (narrow) {
    g.appendChild(txt(L.lRank, 22, 'RANG', 'lad-axis', 'end'));
    g.appendChild(txt(L.lCite, 22, 'ABSATZ', 'lad-axis', 'start'));
    g.appendChild(txt(L.lBar, 22, 'KOSINUS', 'lad-axis', 'start'));
    g.appendChild(txt(L.nCeBar, 22, 'CROSS-ENCODER', 'lad-axis', 'start'));
    // both strips share one 0-1 axis and sit stacked, so each carries its own
    // label - otherwise the lower one reads as belonging to the column above it
    g.appendChild(txt(14, 44, 'KOS', 'lad-note', 'start'));
    strip(g, 46, 44, 120, cosMin, cosMax, 'var(--dense)', `${cosMin.toFixed(3)}–${cosMax.toFixed(3)}`);
    g.appendChild(txt(14, 70, 'CE', 'lad-note', 'start'));
    strip(g, 46, 70, 120, ceMin, ceMax, 'var(--ce)', `${ceMin.toFixed(3)}–${ceMax.toFixed(3)}`);
    return;
  }

  g.appendChild(txt(L.lRank, 22, 'RANG', 'lad-axis', 'end'));
  g.appendChild(txt(L.lCite, 22, `BI-ENCODER · KOSINUS · ${n} KANDIDATEN`, 'lad-axis', 'start'));
  g.appendChild(txt(L.rBar, 22, `CROSS-ENCODER · RELEVANZ · TOP ${k} BLEIBEN`, 'lad-axis', 'start'));

  // the two scale strips, same 0..1 axis — this comparison is the whole plate
  strip(g, L.lCite, 42, 236, cosMin, cosMax, 'var(--dense)',
    `Bandbreite ${(cosMax - cosMin).toFixed(3)} auf der Achse 0–1`);
  strip(g, L.rBar, 42, 236, ceMin, ceMax, 'var(--ce)',
    `Bandbreite ${(ceMax - ceMin).toFixed(3)} auf der Achse 0–1`);

  g.appendChild(txt(L.lCite, 66, 'Balken links gedehnt auf diese Bandbreite', 'lad-note', 'start'));
  g.appendChild(txt(L.rBar, 66, 'Balken rechts auf echter Achse 0–1', 'lad-note', 'start'));
}

/** A 0..1 reference axis with the actual value window marked on it. */
function strip(g, x, y, w, min, max, color, label) {
  g.appendChild(el('line', {
    x1: x, y1: y, x2: x + w, y2: y,
    stroke: 'var(--plate-3)', 'stroke-width': 3, 'stroke-linecap': 'butt',
  }));
  const a = x + clamp01(min) * w;
  const b = x + clamp01(max) * w;
  g.appendChild(el('line', {
    x1: a, y1: y, x2: Math.max(b, a + 0.8), y2: y,
    stroke: color, 'stroke-width': 3, 'stroke-linecap': 'butt',
  }));
  g.appendChild(txt(x, y - 7, '0', 'lad-note', 'start'));
  g.appendChild(txt(x + w, y - 7, '1', 'lad-note', 'end'));
  const t = txt(x + w + 8, y, label, 'lad-note', 'start');
  t.setAttribute('fill', color);
  g.appendChild(t);
}

// -------------------------------------------------------------------- bits

function bar(g, x, y, w, frac, color, opacity) {
  g.appendChild(el('rect', {
    x, y: y - 2.5, width: w, height: 5, rx: 0.5,
    fill: 'var(--plate-2)',
  }));
  const fw = Math.max(clamp01(frac) * w, 0.8);
  const r = el('rect', {
    class: 'lad-bar',
    x, y: y - 2.5, width: fw, height: 5, rx: 0.5,
    fill: color, 'fill-opacity': opacity,
  });
  g.appendChild(r);
  if (!reduced.matches) {
    // SVG geometry properties animate through CSS, so these need units -
    // bare numbers are rejected as invalid keyframe values and nothing moves.
    r.animate([{ width: '0px' }, { width: `${fw}px` }],
      { duration: 620, easing: 'cubic-bezier(.22,.61,.36,1)', fill: 'backwards' });
  }
  return r;
}

function txt(x, y, s, cls, anchor) {
  const t = el('text', { x, y, class: cls, 'text-anchor': anchor, 'dominant-baseline': 'middle' });
  t.textContent = s;
  return t;
}

function el(tag, attrs) {
  const n = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  return n;
}

function animateRibbon(path, delay) {
  const len = path.getTotalLength?.() || 0;
  if (!len) return;
  path.style.strokeDasharray = `${len}`;
  path.animate(
    [{ strokeDashoffset: len }, { strokeDashoffset: 0 }],
    { duration: 900, delay: delay * 1000, easing: 'cubic-bezier(.22,.61,.36,1)', fill: 'both' },
  ).finished.then(() => { path.style.strokeDasharray = ''; }).catch(() => {});
}

function fadeIn(g, delay) {
  g.animate([{ opacity: 0, transform: 'translateY(3px)' }, { opacity: 1, transform: 'none' }],
    { duration: 420, delay: delay * 1000, easing: 'cubic-bezier(.22,.61,.36,1)', fill: 'backwards' });
}

/** Hovering either end of a candidate lights its ribbon and both rows. */
function wireHover(svg) {
  const set = (id, on) => {
    for (const n of svg.querySelectorAll(`[data-id="${cssEsc(id)}"]`)) {
      if (n.classList.contains('lad-ribbon')) {
        n.setAttribute('stroke-opacity', on ? 1 : n.dataset.o0);
        n.setAttribute('stroke-width', on ? 2.6 : n.dataset.w0);
      } else {
        n.classList.toggle('is-hot', on);
      }
    }
  };
  for (const n of svg.querySelectorAll('.lad-ribbon')) {
    n.dataset.o0 = n.getAttribute('stroke-opacity');
    n.dataset.w0 = n.getAttribute('stroke-width');
  }
  for (const n of svg.querySelectorAll('[data-id]')) {
    const id = n.getAttribute('data-id');
    n.addEventListener('pointerenter', () => set(id, true));
    n.addEventListener('pointerleave', () => set(id, false));
  }
}

function ariaSummary(cands, byCe, k, cosMin, cosMax, ceMin, ceMax) {
  const kept = byCe.filter((c) => c.final_rank != null && c.final_rank <= k);
  const moved = kept.filter((c) => c.dense_rank !== c.final_rank).length;
  return `Neuordnung von ${cands.length} Kandidaten. `
    + `Kosinus-Werte liegen zwischen ${cosMin.toFixed(3)} und ${cosMax.toFixed(3)}, `
    + `Bandbreite ${(cosMax - cosMin).toFixed(3)}. `
    + `Cross-Encoder-Werte liegen zwischen ${ceMin.toFixed(3)} und ${ceMax.toFixed(3)}, `
    + `Bandbreite ${(ceMax - ceMin).toFixed(3)}. `
    + `${moved} der ${kept.length} ausgewählten Absätze haben den Rang gewechselt. `
    + `Die Werte stehen als Liste unter "Fundstellen".`;
}

const clamp01 = (v) => Math.max(0, Math.min(1, v));
const cssEsc = (s) => (window.CSS?.escape ? CSS.escape(s) : String(s).replace(/["\\]/g, '\\$&'));
