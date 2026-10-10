/* ===========================================================================
   Pipeline view.

   renderIngest draws the offline half, and it is built from the real corpus:
   the tag path is the one parse_law.py actually walks in the GII schema, the
   § rows are the real paragraphs in the index, and every cell is one real
   Absatz with its width proportional to its real character count on a single
   shared scale. Nothing is a placeholder shape.

   This stage does not run per query. The plate says so, and the diagram is
   built once and never animated on a loop, so it cannot be mistaken for live
   processing.
   =========================================================================== */

const NS = 'http://www.w3.org/2000/svg';
const reduced = matchMedia('(prefers-reduced-motion: reduce)');

const SECTION_TINT = [
  '#6f8fbe', '#b8795a', '#6fa58c', '#a8779f',
  '#8e9a5f', '#c2452c', '#5f8fa5', '#9a8458',
];

// The element path parse_law.py walks. Real tags, in document order.
const XML_TREE = [
  [0, 'dokumente'],
  [1, 'norm'],
  [2, 'metadaten'],
  [3, 'amtabk', 'ArbZG'],
  [3, 'enbez', '§ 3'],
  [3, 'titel', 'Arbeitszeit…'],
  [3, 'gliederungseinheit'],
  [2, 'textdaten'],
  [3, 'text'],
  [4, 'Content'],
  [5, 'P', '(1) …'],
  [5, 'P', '(2) …'],
];

export function renderIngest(host, meta, chunks) {
  if (!host || !chunks?.length) return;
  host.textContent = '';

  // Same trap as the ladder: this is first called while the pipeline view is
  // still hidden, so the host measures 0. Falling back to "wide" there renders
  // a 1000-unit viewBox into a phone.
  const hw = host.clientWidth || host.parentElement?.clientWidth || window.innerWidth;
  const narrow = hw < 620;

  // real grouping: § -> its Absätze, in index order
  const order = [];
  const groups = new Map();
  for (const c of chunks) {
    if (!groups.has(c.paragraph)) { groups.set(c.paragraph, []); order.push(c.paragraph); }
    groups.get(c.paragraph).push(c);
  }
  const sections = [...new Set(chunks.map((c) => c.abschnitt))];
  const tintOf = (c) => SECTION_TINT[sections.indexOf(c.abschnitt) % SECTION_TINT.length];

  // The index is keyed on a string id, so /api/corpus comes back in
  // lexicographic order: § 1, § 10, § 11, § 2. Correct for the store, wrong for
  // a diagram of a statute. Sort for display only; nothing downstream reads it.
  order.sort(byParagraph);
  const rows = order.map((p) => ({ paragraph: p, cells: groups.get(p) }));
  const gap = 1.6;
  const widest = Math.max(...rows.map((r) =>
    r.cells.reduce((a, c) => a + c.chars, 0) + gap * (r.cells.length - 1)));

  const W = narrow ? 460 : 1000;
  const colX = narrow ? 92 : 268;          // where the Absatz cells start
  const avail = W - colX - 14;
  let scale = avail / widest;              // one shared px-per-char scale
  // A § made of many short Absätze is widened by the minimum cell width, so
  // the longest row on screen is not always the one with the most characters.
  // Re-fit against what will actually be drawn.
  const drawnWidth = (sc) => Math.max(...rows.map((r) =>
    r.cells.reduce((a, c) => a + Math.max(c.chars * sc, 1.2) + gap, -gap)));
  const over = drawnWidth(scale);
  if (over > avail) scale *= avail / over;
  const rowH = narrow ? 9.5 : 11;
  const top = narrow ? 56 : 62;
  const H = top + rows.length * rowH + 26;

  const svg = el('svg', {
    viewBox: `0 0 ${W} ${H}`, role: 'img',
    'aria-label': `Parsing-Diagramm: eine XML-Datei wird zu ${order.length} Paragraphen `
      + `und ${chunks.length} absatzgenauen Abschnitten. Zellenbreite entspricht der Textlänge.`,
  });

  // ---------------------------------------------------------- column heads
  const headY = narrow ? 18 : 22;
  if (!narrow) {
    svg.appendChild(txt(16, headY, 'AMTLICHE XML-FASSUNG', 'ing-label', 'start'));
    svg.appendChild(txt(200, headY, 'PARAGRAPHEN', 'ing-label', 'end'));
    svg.appendChild(txt(colX, headY, 'ABSÄTZE — EINZELN ZITIERBAR, BREITE = TEXTLÄNGE', 'ing-label', 'start'));
    svg.appendChild(count(16, headY + 20, '1', 'Datei · 55 KB'));
    svg.appendChild(count(200, headY + 20, String(order.length), '§§', 'end'));
    svg.appendChild(count(colX, headY + 20, String(chunks.length), 'Vektoren · 768 dim'));
  } else {
    // the XML tag tree is not drawn at this width, so its label would dangle
    svg.appendChild(txt(14, headY, `${order.length} §§ → ${chunks.length} ABSÄTZE`, 'ing-label', 'start'));
    svg.appendChild(count(14, headY + 20, String(chunks.length), 'Vektoren · 768 dim'));
  }

  // ------------------------------------------------------------- xml tree
  if (!narrow) {
    const treeTop = top + 6;
    const lh = 10.5;
    XML_TREE.forEach(([depth, tag, sample], i) => {
      const y = treeTop + i * lh;
      const x = 18 + depth * 7;
      const g = el('g', {});
      g.appendChild(txt(x, y, `<${tag}>`, 'ing-tag', 'start'));
      if (sample) g.appendChild(txt(x + 7 + tag.length * 3.5, y, sample, 'ing-xml', 'start'));
      if (depth > 0) {
        g.appendChild(el('path', {
          d: `M${x - 4} ${y - lh + 3} V${y} H${x - 1}`,
          stroke: 'var(--plate-3)', 'stroke-width': 0.5, fill: 'none',
        }));
      }
      svg.appendChild(g);
      stagger(g, i * 0.03);
    });
    const treeH = XML_TREE.length * lh;
    svg.appendChild(txt(18, treeTop + treeH + 8,
      'P-Elemente, die mit "(N)" beginnen,', 'ing-xml', 'start'));
    svg.appendChild(txt(18, treeTop + treeH + 16,
      'eröffnen Absatz N — die Gliederung', 'ing-xml', 'start'));
    svg.appendChild(txt(18, treeTop + treeH + 24,
      'steht im Text, nicht im Schema.', 'ing-xml', 'start'));
  }

  // ------------------------------------------------------ rows: § -> cells
  rows.forEach((r, i) => {
    const y = top + i * rowH + rowH / 2;
    const g = el('g', {});
    svg.appendChild(g);

    if (!narrow) {
      g.appendChild(txt(200, y, r.paragraph, 'ing-cite', 'end'));
      // fan from the XML column into this § row
      g.appendChild(el('path', {
        d: `M150 ${top + rows.length * rowH / 2} C${178} ${top + rows.length * rowH / 2}, ${178} ${y}, ${206} ${y}`,
        stroke: 'var(--plate-3)', 'stroke-width': 0.4, fill: 'none', 'stroke-opacity': 0.8,
      }));
      g.appendChild(el('line', {
        x1: 206, y1: y, x2: colX - 5, y2: y,
        stroke: 'var(--plate-3)', 'stroke-width': 0.4,
      }));
    } else {
      g.appendChild(txt(colX - 6, y, r.paragraph, 'ing-cite', 'end'));
    }

    let x = colX;
    for (const c of r.cells) {
      const w = Math.max(c.chars * scale, 1.2);
      const cell = el('rect', {
        x, y: y - rowH / 2 + 1.4, width: w, height: rowH - 2.8, rx: 0.6,
        fill: tintOf(c), 'fill-opacity': 0.5,
      });
      cell.appendChild(title(`${c.citation} — ${c.paragraph_title} · ${c.chars} Zeichen`));
      g.appendChild(cell);
      // the Absatz number inside the cell, where it fits
      if (w > 11 && c.absatz !== '-') {
        g.appendChild(txt(x + 2.5, y, c.absatz, 'ing-xml', 'start'));
      }
      x += w + gap;
    }
    stagger(g, 0.25 + i * 0.022);
  });

  svg.appendChild(txt(colX, H - 10,
    `Zellenbreite auf gemeinsamer Skala — ${Math.round(1 / scale)} Zeichen je Pixel. `
    + 'Jede Zelle ist eine eigene Fundstelle.', 'ing-sub', 'start'));

  host.appendChild(svg);
}

/** "§ 21a" -> [21, "a"], so § 3 precedes § 21a precedes § 22. */
function byParagraph(a, b) {
  const pa = /(\d+)\s*([a-z]*)/i.exec(a) || [];
  const pb = /(\d+)\s*([a-z]*)/i.exec(b) || [];
  return (+pa[1] || 0) - (+pb[1] || 0) || (pa[2] || '').localeCompare(pb[2] || '');
}

function count(x, y, big, small, anchor = 'start') {
  const g = el('g', {});
  const b = txt(x, y, big, 'ing-count', anchor);
  g.appendChild(b);
  const off = anchor === 'end' ? -big.length * 8 - 5 : big.length * 8 + 5;
  g.appendChild(txt(x + off, y + 1, small, 'ing-sub', anchor));
  return g;
}

/** One-shot build-in when the plate first scrolls into view. Never loops. */
function stagger(g, delay) {
  if (reduced.matches) return;
  g.style.opacity = '0';
  const io = new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (!e.isIntersecting) continue;
      io.disconnect();
      g.animate([{ opacity: 0, transform: 'translateX(-4px)' }, { opacity: 1, transform: 'none' }],
        { duration: 500, delay: delay * 1000, easing: 'cubic-bezier(.22,.61,.36,1)', fill: 'forwards' });
      setTimeout(() => { g.style.opacity = ''; }, delay * 1000 + 520);
    }
  }, { rootMargin: '80px' });
  io.observe(g.ownerSVGElement || g);
}

// -------------------------------------------------------------- online row

export function renderOnlineStages(host, meta, timing) {
  if (!host || !meta) return;
  const m = (s) => String(s || '').split('/').pop();
  const t = (k) => (timing?.[k] == null ? null
    : timing[k] < 1000 ? `${timing[k]} ms` : `${(timing[k] / 1000).toFixed(1)} s`);

  const stages = [
    {
      h: 'Frage einbetten',
      p: 'Die Frage wird mit dem Präfix „query:" in denselben Vektorraum abgebildet wie die '
        + 'Absätze. Derselbe mehrsprachige Encoder, deshalb treffen englische Fragen deutschen Text.',
      model: `${m(meta.embed_model)} · ${meta.dim} dim`, ms: t('embed_ms'),
    },
    {
      h: 'Kandidaten suchen',
      p: 'Kosinus-Ähnlichkeit über alle indexierten Absätze. Es kommen immer k Treffer zurück — '
        + 'auch wenn keiner passt. Diese Stufe kennt kein „nichts passt".',
      model: `${meta.collection} · top 20`, ms: t('search_ms'),
    },
    {
      h: 'Neu ordnen',
      p: 'Der Cross-Encoder liest Frage und Absatz gemeinsam und bewertet jedes Paar einzeln. '
        + '20 Durchläufe statt eines Vektorvergleichs — deshalb die Laufzeit.',
      model: `${m(meta.reranker)} · 20 Paare`, ms: t('rerank_ms'), heavy: true,
    },
    {
      h: 'Antwort erzeugen',
      p: 'Nur die fünf bestbewerteten Absätze gehen in den Prompt. Das Modell ist angewiesen, '
        + 'ausschließlich daraus zu antworten und fehlende Deckung zu benennen.',
      model: meta.generation_available ? `${m(meta.gen_model)} · k=5` : 'nicht konfiguriert',
      ms: t('generate_ms'),
    },
    {
      h: 'Aussagen belegen',
      p: 'Die Antwort wird in Einzelaussagen zerlegt, jede gegen jeden abgerufenen Absatz '
        + 'bewertet. Der beste Treffer wird die Fundstelle der Aussage.',
      model: `${m(meta.reranker)} · Aussagen × 5`, ms: t('attribute_ms'), heavy: true,
    },
  ];

  host.innerHTML = stages.map((s, i) => `
    <div class="ost">
      <span class="ost__n">ONLINE ${String(i + 1).padStart(2, '0')}</span>
      <h3 class="ost__h">${esc(s.h)}</h3>
      <p class="ost__p">${esc(s.p)}</p>
      <span class="ost__model">${esc(s.model)}</span>
      <span class="ost__t">${s.ms
    ? `letzte Abfrage: ${s.heavy ? `<b>${esc(s.ms)}</b>` : esc(s.ms)}`
    : 'noch nicht gemessen'}</span>
    </div>`).join('');
}

// ------------------------------------------------------------------ tables

export function renderTables(host) {
  if (!host) return;
  host.innerHTML = `
  <div class="tbl">
    <h3>Retrieval</h3>
    <p>91 Fragen, deutsch und englisch, mit Soll-Fundstelle.</p>
    <table>
      <thead><tr><th>Konfiguration</th><th>recall@1</th><th>recall@5</th><th>MRR</th></tr></thead>
      <tbody>
        <tr><td>Dense (top-5)</td><td>0.44</td><td>0.84</td><td>0.585</td></tr>
        <tr class="is-best"><td>Dense top-20 + Reranker</td><td>0.60</td><td>0.90</td><td>0.722</td></tr>
      </tbody>
    </table>
    <p class="tbl__note">Vorbehalt: recall@5 steigt mit. Umsortieren von fünf Treffern kann
      nicht ändern, ob die richtige Fundstelle dabei ist — dieser Teil kommt davon, dass der
      Kandidatenpool vor dem Reranking von 5 auf 20 erweitert wurde. Beide Effekte sind in
      diesem Durchlauf vermischt.</p>
  </div>

  <div class="tbl">
    <h3>Nach Sprache der Frage</h3>
    <p>recall@1, vor und nach dem Reranking.</p>
    <table>
      <thead><tr><th>Sprache</th><th>dense</th><th>reranked</th><th>r@5 dense</th><th>r@5 rerank</th></tr></thead>
      <tbody>
        <tr><td>Deutsch</td><td>0.59</td><td>0.59</td><td>0.83</td><td>0.89</td></tr>
        <tr class="is-best"><td>Englisch</td><td>0.29</td><td>0.60</td><td>0.84</td><td>0.91</td></tr>
      </tbody>
    </table>
    <p class="tbl__note">Der Reranker verdoppelte recall@1 für englische Fragen und ließ
      deutsche unverändert — ein Abstand von 30 Punkten schrumpfte auf einen. Ein Cross-Encoder
      kann nur umsortieren, also war der richtige Absatz vorher schon gefunden und bloß
      schlecht platziert.</p>
  </div>

  <div class="tbl">
    <h3>Abstention</h3>
    <p>Trennt ein Retrieval-Signal beantwortbare von unbeantwortbaren Fragen?
      91 beantwortbar, 22 nicht. AUROC: 1.0 perfekt, 0.5 nutzlos.</p>
    <table>
      <thead><tr><th>Signal</th><th>AUROC</th><th>beantwortbar</th><th>nicht</th></tr></thead>
      <tbody>
        <tr><td>Kosinus top-1</td><td>0.660</td><td>0.846</td><td>0.828</td></tr>
        <tr><td>Abstand top-1 zu top-2</td><td>0.682</td><td>0.007</td><td>0.003</td></tr>
        <tr><td>Mittel top-5 Kosinus</td><td>0.644</td><td>0.836</td><td>0.821</td></tr>
        <tr class="is-best"><td>Cross-Encoder top-1</td><td>0.828</td><td>0.594</td><td>0.159</td></tr>
      </tbody>
    </table>
    <p class="tbl__note">Kosinus trennt die beiden Gruppen um 0.018, der Cross-Encoder um
      0.435 — etwa vierundzwanzigmal so weit. Einbettungs-Ähnlichkeit misst thematische Nähe,
      nicht ob ein Absatz als Beleg ausreicht. Vorbehalt: 22 Negativbeispiele zeigen, dass eine
      Trennung existiert, nicht wo der Schwellwert liegen sollte.</p>
  </div>`;
}

// ------------------------------------------------------------------- utils

function el(tag, attrs) {
  const n = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  return n;
}

function txt(x, y, s, cls, anchor) {
  const t = el('text', { x, y, class: cls, 'text-anchor': anchor, 'dominant-baseline': 'middle' });
  t.textContent = s;
  return t;
}

function title(s) {
  const t = document.createElementNS(NS, 'title');
  t.textContent = s;
  return t;
}

function esc(s) {
  return String(s ?? '').replace(/[&<>"']/g, (c) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}
