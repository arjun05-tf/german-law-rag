/* ===========================================================================
   Orchestration.

   The client renders what the server measured and nothing else. Every number on
   screen arrives from /api/meta, /api/corpus or a stage event on
   /api/query/stream. Where a value has not been computed yet, the UI shows it as
   pending rather than guessing at it; where a stage does not run (no API key,
   so no generation) the rail marks it skipped instead of pretending.
   =========================================================================== */

import { CorpusField, sectionLegend } from './field.js';
import { renderIngest, renderOnlineStages, renderTables } from './pipeline.js';

const $ = (id) => document.getElementById(id);

/* The backend can live on another host - the models do not fit on a static one.
   A local server always wins over the configured base, so `uvicorn serve:app`
   keeps hitting its own API rather than the deployed one. */
const API_BASE = ['localhost', '127.0.0.1'].includes(location.hostname)
  ? ''
  : (document.querySelector('meta[name="api-base"]')?.content || '').replace(/\/+$/, '');
const api = (path) => (API_BASE ? `${API_BASE}/${path}` : path);

const LADDER_BREAK = 560;   // must match ladder.js

const STAGES = [
  { key: 'embedded',   label: 'Einbetten',  ms: 'embed_ms',     detail: '768 dim' },
  { key: 'retrieved',  label: 'Suchen',     ms: 'search_ms',    detail: 'Kosinus' },
  { key: 'reranked',   label: 'Reranking',  ms: 'rerank_ms',    detail: 'Paarweise' },
  { key: 'answered',   label: 'Antwort',    ms: 'generate_ms',  detail: 'gpt-4o-mini', gen: true },
  { key: 'attributed', label: 'Attribution', ms: 'attribute_ms', detail: 'pro Claim', gen: true },
];

const state = {
  meta: null,
  field: new CorpusField(),
  es: null,
  result: null,
  stageTimer: null,
  stageStart: 0,
  ladderNarrow: null,
  chunks: null,
};


async function boot() {
  wireForm();
  wireTabs();
  state.field.mount($('field-slot-hero'), 'hero');

  try {
    const [meta, corpus] = await Promise.all([
      fetch(api('api/meta')).then(j),
      fetch(api('api/corpus')).then(j),
    ]);
    state.meta = meta;
    state.chunks = corpus.chunks;
    renderBarFacts(meta);
    state.field.setData(corpus.chunks);
    $('field-legend').innerHTML = sectionLegend(state.field);
    $('field-note').textContent = fieldNote(meta);
    document.title = `${meta.gesetz} · Retrieval`;
    $('bar-gesetz').textContent = meta.gesetz || 'ArbZG';

    renderIngest($('ingest'), meta, corpus.chunks);
    renderOnlineStages($('online-stages'), meta, null);
    renderTables($('tables'));
  } catch (err) {
    showError('Index nicht erreichbar', String(err));
  }
}

const j = (r) => {
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return r.json();
};

function fieldNote(meta) {
  const [a = 0, b = 0] = meta.pca_explained || [];
  const pct = ((a + b) * 100).toFixed(1);
  return `Jeder Punkt ist einer von ${meta.chunks} indexierten Absätzen; die Fläche `
    + `ist eine 2-D-Hauptkomponentenprojektion der echten ${meta.dim}-dimensionalen Vektoren `
    + `und behält ${pct} % der Varianz. Nur zur Anschauung — gesucht wird per Kosinus über `
    + `alle ${meta.dim} Dimensionen. Striche verbinden jeden Absatz mit dem Schwerpunkt `
    + `seines Abschnitts, Punktfläche entspricht der Textlänge.`;
}

function renderBarFacts(m) {
  const facts = [
    ['Korpus', `${m.chunks} Absätze · ${m.paragraphs} §§`],
    ['Index', `${m.collection} · ${m.dim} dim`],
    ['Einbettung', short(m.embed_model)],
    ['Reranker', short(m.reranker)],
    ['Generierung', m.generation_available
      ? `<span class="dot dot--on"></span>${short(m.gen_model)}`
      : '<span class="dot dot--off"></span>aus'],
  ];
  $('bar-facts').innerHTML = facts
    .map(([k, v]) => `<div><dt>${esc(k)}</dt><dd>${v}</dd></div>`).join('');
}

const short = (s) => String(s || '').split('/').pop();

// ------------------------------------------------------------------- tabs

function wireTabs() {
  for (const tab of document.querySelectorAll('.tab')) {
    tab.addEventListener('click', () => {
      const view = tab.dataset.view;
      for (const t of document.querySelectorAll('.tab')) {
        const on = t === tab;
        t.classList.toggle('is-active', on);
        on ? t.setAttribute('aria-current', 'page') : t.removeAttribute('aria-current');
      }
      $('view-query').hidden = view !== 'query';
      $('view-pipeline').hidden = view !== 'pipeline';
      // the diagram was first drawn while this view was hidden and had no
      // width; now that it is on screen, draw it at the size it will be read at
      if (view === 'pipeline' && state.chunks) {
        renderIngest($('ingest'), state.meta, state.chunks);
      }
      // the canvas lives in whichever view is on screen; nothing animates off-screen
      if (view === 'query') {
        state.field.mount(state.result ? $('field-slot-results') : $('field-slot-hero'),
          state.result ? 'plate' : 'hero');
      } else {
        state.field.stop();
      }
      scrollTo({ top: 0, behavior: reducedMotion() ? 'auto' : 'smooth' });
    });
  }
}

const reducedMotion = () => matchMedia('(prefers-reduced-motion: reduce)').matches;

// ------------------------------------------------------------------- form

function wireForm() {
  const form = $('ask-form');
  const input = $('q');

  const autosize = () => {
    input.style.height = 'auto';
    input.style.height = `${Math.min(input.scrollHeight, 144)}px`;
  };
  input.addEventListener('input', autosize);
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); form.requestSubmit(); }
  });

  form.addEventListener('submit', (e) => {
    e.preventDefault();
    const q = input.value.trim();
    if (q) ask(q);
  });

  for (const chip of document.querySelectorAll('.chip')) {
    chip.addEventListener('click', () => {
      input.value = chip.textContent.trim();
      autosize();
      ask(input.value);
    });
  }

  $('rail-cancel').addEventListener('click', () => {
    stopStream();
    setRailIdle();
    $('ask-go').disabled = false;
    $('rail-wrap').hidden = true;
  });
}

// ------------------------------------------------------------------ query

function ask(question) {
  stopStream();
  $('error-plate').hidden = true;
  $('ask-go').disabled = true;
  $('hero').classList.add('is-collapsed');
  $('field-slot-hero').setAttribute('aria-hidden', 'true');

  // keep the previous answer visible while the new one computes? no — it would
  // sit next to a rail describing a different question. clear it.
  $('results').hidden = true;
  state.result = null;
  state.field.clearResult();
  state.field.mount($('field-slot-hero'), 'hero');

  $('asked').innerHTML = `<span>Frage</span>${esc(question)}`;

  const genPossible = !!state.meta?.generation_available;
  buildRail(genPossible);
  $('rail-wrap').hidden = false;
  markStage('embedded', 'active');
  startStageTimer();

  const url = api(`api/query/stream?question=${encodeURIComponent(question)}`)
    + `&k_retrieve=20&k=5&generate=${genPossible}`;
  const es = new EventSource(url);
  state.es = es;

  const partial = { question, candidates: [], timing: {} };

  es.addEventListener('embedded', (e) => {
    const d = JSON.parse(e.data);
    Object.assign(partial, { query_xy: d.query_xy });
    partial.timing.embed_ms = d.embed_ms;
    doneStage('embedded', d.embed_ms);
    markStage('retrieved', 'active');
  });

  es.addEventListener('retrieved', (e) => {
    const d = JSON.parse(e.data);
    partial.candidates = d.candidates;
    partial.timing.search_ms = d.search_ms;
    doneStage('retrieved', d.search_ms);
    markStage('reranked', 'active');
    // show the dense result immediately — it is real and already final for
    // this stage. the reranked view replaces it when the reranker lands.
    state.result = partial;
    showResults(partial, { stage: 'dense' });
  });

  es.addEventListener('reranked', (e) => {
    const d = JSON.parse(e.data);
    partial.candidates = d.candidates;
    partial.evidence = d.evidence;
    partial.timing.rerank_ms = d.rerank_ms;
    doneStage('reranked', d.rerank_ms);
    markStage(genPossible ? 'answered' : null, 'active');
    state.result = partial;
    showResults(partial, { stage: 'reranked' });
  });

  es.addEventListener('answered', (e) => {
    const d = JSON.parse(e.data);
    partial.answer = d.answer;
    partial.timing.generate_ms = d.generate_ms;
    doneStage('answered', d.generate_ms);
    markStage('attributed', 'active');
    showResults(partial, { stage: 'answered' });
  });

  es.addEventListener('attributed', (e) => {
    const d = JSON.parse(e.data);
    partial.claims = d.claims;
    partial.timing.attribute_ms = d.attribute_ms;
    doneStage('attributed', d.attribute_ms);
    showResults(partial, { stage: 'attributed' });
  });

  es.addEventListener('generate_error', (e) => {
    const d = JSON.parse(e.data);
    markStage('answered', 'skip');
    markStage('attributed', 'skip');
    partial.timing.generate_error = d.error;
    showResults(partial, { stage: 'reranked' });
    showError('Antwortgenerierung fehlgeschlagen — Retrieval ist davon unberührt', d.error);
  });

  es.addEventListener('done', (e) => {
    const d = JSON.parse(e.data);
    state.result = d;
    showResults(d, { stage: 'done' });
    renderOnlineStages($('online-stages'), state.meta, d.timing);
    finish();
  });

  es.addEventListener('error', (e) => {
    // SSE fires `error` both for a server-sent error event and for a dropped
    // connection. A payload means the former.
    let msg = 'Verbindung zum Index unterbrochen.';
    try { if (e.data) msg = JSON.parse(e.data).error; } catch {}
    if (!state.result) showError('Abfrage fehlgeschlagen', msg);
    finish();
  });

  es.onerror = () => {
    if (es.readyState === EventSource.CLOSED && !state.result) {
      showError('Abfrage fehlgeschlagen', 'Der Stream wurde geschlossen, bevor Ergebnisse kamen.');
    }
    finish();
  };
}

function finish() {
  stopStream();
  stopStageTimer();
  $('ask-go').disabled = false;
  $('rail-cancel').hidden = true;
  for (const li of document.querySelectorAll('.rail__step[data-state="active"]')) {
    li.dataset.state = 'skip';
  }
}

function stopStream() {
  if (state.es) { state.es.close(); state.es = null; }
  stopStageTimer();
}

// -------------------------------------------------------------- stage rail

function buildRail(genPossible) {
  $('rail-cancel').hidden = false;
  $('rail').innerHTML = STAGES.map((s) => `
    <li class="rail__step" data-stage="${s.key}" data-state="${s.gen && !genPossible ? 'skip' : 'wait'}">
      <span class="rail__pip"></span>
      <span>${esc(s.label)}</span>
      <span class="rail__ms" data-ms>${s.gen && !genPossible ? 'kein Schlüssel' : esc(s.detail)}</span>
    </li>`).join('');
}

function setRailIdle() {
  for (const li of document.querySelectorAll('.rail__step')) li.dataset.state = 'wait';
}

function markStage(key, st) {
  if (!key) return;
  const li = document.querySelector(`.rail__step[data-stage="${key}"]`);
  if (!li || li.dataset.state === 'skip') return;
  li.dataset.state = st;
  if (st === 'active') { state.stageStart = performance.now(); }
}

function doneStage(key, ms) {
  const li = document.querySelector(`.rail__step[data-stage="${key}"]`);
  if (!li) return;
  li.dataset.state = 'done';
  li.querySelector('[data-ms]').textContent = fmtMs(ms);
}

/** Elapsed time on the running stage. Measured, not a predicted progress bar. */
function startStageTimer() {
  stopStageTimer();
  state.stageTimer = setInterval(() => {
    const li = document.querySelector('.rail__step[data-state="active"]');
    if (!li) return;
    const s = (performance.now() - state.stageStart) / 1000;
    if (s > 1.2) li.querySelector('[data-ms]').textContent = `${s.toFixed(1)} s…`;
  }, 100);
}

function stopStageTimer() {
  if (state.stageTimer) clearInterval(state.stageTimer);
  state.stageTimer = null;
}

const fmtMs = (ms) => (ms == null ? '—' : ms < 1000 ? `${ms} ms` : `${(ms / 1000).toFixed(1)} s`);

// ----------------------------------------------------------------- results

async function showResults(r, { stage }) {
  const wrap = $('results');
  const first = wrap.hidden;
  wrap.hidden = false;
  if (first) wrap.classList.add('reveal');

  const reranked = stage !== 'dense';

  // --- field -------------------------------------------------------------
  state.field.setResult(r);
  if (state.field.canvas.parentNode !== $('field-slot-results')) {
    state.field.mount($('field-slot-results'), 'plate');
  }
  state.field.start();

  // --- ladder ------------------------------------------------------------
  const ladderPlate = $('ladder-plate');
  if (reranked) {
    ladderPlate.hidden = false;
    const { renderLadder } = await import('./ladder.js');
    state.ladderNarrow = ($('ladder').clientWidth || innerWidth) < LADDER_BREAK;
    renderLadder($('ladder'), r.candidates, r.config?.k ?? 5);
    renderSpread(r.candidates);
  } else {
    // the plate would be half-empty and the spread comparison meaningless
    // before the cross-encoder has scored anything.
    ladderPlate.hidden = true;
  }

  renderAnswer(r, stage);
  renderEvidence(r, reranked);
  renderTimings(r.timing);
}

/** Re-render the ladder only when the width crosses the layout breakpoint.
    Re-rendering on every resize restarts the staggered entrance animation, so
    an incidental resize (a scrollbar appearing, a full-page capture, a mobile
    URL bar collapsing) would blank the plate for the length of the stagger. */
let resizeT = null;
addEventListener('resize', () => {
  clearTimeout(resizeT);
  resizeT = setTimeout(async () => {
    const r = state.result;
    if (!r?.candidates?.length || $('ladder-plate').hidden) return;
    if (!r.candidates.some((c) => c.ce_score != null)) return;
    const narrow = ($('ladder').clientWidth || innerWidth) < LADDER_BREAK;
    if (narrow === state.ladderNarrow) return;
    state.ladderNarrow = narrow;
    const { renderLadder } = await import('./ladder.js');
    renderLadder($('ladder'), r.candidates, r.config?.k ?? 5);
  }, 200);
});

function renderSpread(cands) {
  const cos = cands.map((c) => c.cosine);
  const ce = cands.map((c) => c.ce_score).filter((v) => v != null);
  const dc = Math.max(...cos) - Math.min(...cos);
  const dce = ce.length ? Math.max(...ce) - Math.min(...ce) : 0;
  const factor = dc > 0 ? (dce / dc) : 0;
  $('spread').innerHTML = `
    <div><dt>Bandbreite Kosinus</dt><dd class="is-dense">${dc.toFixed(3)}</dd></div>
    <div><dt>Bandbreite Cross-Encoder</dt><dd class="is-ce">${dce.toFixed(3)}</dd></div>
    <div><dt>Verhältnis</dt><dd>${factor >= 1 ? `${factor.toFixed(0)}×` : '—'}
      <span class="spread__hint">breiter</span></dd></div>`;
}

function renderAnswer(r, stage) {
  const body = $('answer-body');
  const note = $('answer-note');
  const claims = $('claims');

  if (r.answer) {
    body.classList.remove('is-empty');
    body.innerHTML = linkCitations(r.answer, r.candidates);
    note.textContent = `Erzeugt von ${short(state.meta?.gen_model)} ausschließlich aus den `
      + `${r.config?.k ?? 5} unten aufgeführten Absätzen.`;
    wireCiteJumps();
  } else if (!state.meta?.generation_available) {
    body.classList.add('is-empty');
    body.innerHTML = 'Keine Antwortgenerierung konfiguriert — ohne <code>OPENAI_API_KEY</code> '
      + 'läuft nur das Retrieval. Die Fundstellen unten sind das vollständige Ergebnis.';
    note.textContent = '';
  } else if (r.timing?.generate_error) {
    body.classList.add('is-empty');
    body.textContent = 'Die Antwortgenerierung ist fehlgeschlagen. Das Retrieval unten ist davon unberührt.';
    note.textContent = '';
  } else {
    body.classList.add('is-empty');
    body.textContent = stage === 'dense'
      ? 'Wartet auf das Reranking — die Antwort wird erst aus den endgültigen Absätzen erzeugt.'
      : 'Antwort wird erzeugt…';
    note.textContent = '';
  }

  if (r.claims?.length) {
    const weak = r.claims.filter((c) => !c.supported).length;
    claims.hidden = false;
    claims.innerHTML = `
      <p class="claims__h">Belegprüfung je Aussage</p>
      <p class="claims__note">Die Antwort wird in einzelne Aussagen zerlegt; jede wird vom
        Cross-Encoder gegen jeden abgerufenen Absatz bewertet. Angegeben ist der beste Treffer.
        ${plural(r.claims.length, 'Aussage', 'Aussagen')}, ${weak} ohne ausreichenden Beleg.</p>
      ${r.claims.map((c) => {
    const ok = c.supported;
    return `<div class="claim ${ok ? 'claim--ok' : 'claim--weak'}">
          <span class="claim__mark">${ok ? esc(c.citation ?? '—') : 'kein Beleg'}</span>
          <span class="claim__text">${esc(c.claim)}</span>
          <span class="claim__score">${c.score.toFixed(3)}</span>
        </div>`;
  }).join('')}
      <p class="claims__legend">Schwelle ${'0.35'} — vorläufig, nicht kalibriert.
        Ein Wert darunter heißt: dieser Satz ist in den abgerufenen Absätzen nicht gedeckt,
        nicht dass er falsch ist. Meta-Sätze über das Gesetz selbst landen hier regelmäßig.</p>`;
  } else {
    claims.hidden = true;
    claims.innerHTML = '';
  }
}

/** Turn citations that really appear among the retrieved passages into jumps. */
function linkCitations(text, cands) {
  const cites = [...new Set((cands || []).map((c) => c.citation))]
    .sort((a, b) => b.length - a.length);
  let html = esc(text);
  for (const c of cites) {
    const re = new RegExp(reEsc(esc(c)), 'g');
    html = html.replace(re, `<a class="cite" href="#ev-${slug(c)}" data-cite="${esc(c)}">${esc(c)}</a>`);
  }
  return html.split(/\n{2,}/).map((p) => `<p>${p.replace(/\n/g, '<br>')}</p>`).join('');
}

function wireCiteJumps() {
  for (const a of $('answer-body').querySelectorAll('.cite')) {
    a.addEventListener('click', (e) => {
      const id = a.getAttribute('href').slice(1);
      const target = document.getElementById(id);
      if (!target) return;           // demoted candidate: no card to jump to
      e.preventDefault();
      target.open = true;
      target.scrollIntoView({ behavior: reducedMotion() ? 'auto' : 'smooth', block: 'center' });
      target.classList.add('is-target');
      setTimeout(() => target.classList.remove('is-target'), 1600);
    });
  }
}

function renderEvidence(r, reranked) {
  const cands = r.candidates || [];
  const kept = cands.filter((c) => c.final_rank != null)
    .sort((a, b) => a.final_rank - b.final_rank);
  const list = reranked ? kept : cands.slice(0, r.config?.k ?? 5);

  const cos = cands.map((c) => c.cosine);
  const cosMin = Math.min(...cos), cosMax = Math.max(...cos);

  $('evidence').innerHTML = list.map((c, i) => {
    const rank = reranked ? c.final_rank : c.dense_rank;
    const moved = reranked && c.dense_rank !== c.final_rank;
    const up = reranked && c.final_rank < c.dense_rank;
    return `
      <li><details class="ev" id="ev-${slug(c.citation)}">
        <summary class="ev__top">
          <span class="ev__rank">${rank}</span>
          <span class="ev__id">
            <span class="ev__cite">${esc(c.citation)}</span>
            <span class="ev__title">${esc(c.paragraph_title)}</span>
          </span>
          <span class="ev__scores">
            ${moved ? `<span class="move ${up ? 'move--up' : 'move--down'}">${
      up ? '↑' : '↓'} von ${c.dense_rank}</span>`
      : reranked ? '<span class="move">Rang gehalten</span>' : ''}
            ${metric('Kosinus', c.cosine, frac(c.cosine, cosMin, cosMax), 'dense')}
            ${c.ce_score != null
      ? metric('Cross-Enc.', c.ce_score, clamp01(c.ce_score), 'ce')
      : '<span class="metric metric--ce"><span class="metric__k">Cross-Enc.</span><span class="metric__v">…</span></span>'}
            <svg class="ev__caret" viewBox="0 0 8 8" aria-hidden="true"><path d="M2 1l4 3-4 3" fill="none" stroke="currentColor" stroke-width="1.3"/></svg>
          </span>
        </summary>
        <div class="ev__body">
          <p class="ev__law">${esc(c.text)}</p>
          <div class="ev__meta">
            <span>${esc(oneLine(c.abschnitt))}</span>
            <span>${c.absatz === '-' ? 'ohne Absatzgliederung' : `Absatz ${esc(c.absatz)}`}</span>
            <span>${c.chars} Zeichen</span>
            <span>Dense-Rang ${c.dense_rank}${reranked ? ` · Endrang ${c.final_rank}` : ''}</span>
          </div>
        </div>
      </details></li>`;
  }).join('');

  const out = cands.filter((c) => c.final_rank == null);
  const wrap = $('demoted-wrap');
  if (reranked && out.length) {
    wrap.hidden = false;
    $('demoted-count').textContent = out.length === 1
      ? '1 weiterer Kandidat wurde abgewertet und fließt nicht in die Antwort ein'
      : `${out.length} weitere Kandidaten wurden abgewertet und fließen nicht in die Antwort ein`;
    $('demoted').innerHTML = out
      .sort((a, b) => a.dense_rank - b.dense_rank)
      .map((c) => `<li class="dem">
          <span class="dem__rank">#${c.dense_rank}</span>
          <span class="dem__cite">${esc(c.citation)}</span>
          <span class="dem__title">${esc(c.paragraph_title)}</span>
          <span class="dem__v" style="color:var(--dense-dk)">${c.cosine.toFixed(3)}</span>
          <span class="dem__v" style="color:var(--ce)">${c.ce_score?.toFixed(3) ?? '—'}</span>
        </li>`).join('');
  } else {
    wrap.hidden = true;
  }
}

function metric(label, value, barFrac, kind) {
  return `<span class="metric metric--${kind}">
    <span class="metric__k">${esc(label)}</span>
    <span class="metric__v">${value.toFixed(3)}</span>
    <span class="metric__bar"><i style="width:${(barFrac * 100).toFixed(1)}%"></i></span>
  </span>`;
}

function renderTimings(t) {
  if (!t) return;
  const rows = [
    ['Einbetten', t.embed_ms, false],
    ['Vektorsuche', t.search_ms, false],
    ['Reranking', t.rerank_ms, true],
    ['Generierung', t.generate_ms, false],
    ['Attribution', t.attribute_ms, true],
  ].filter(([, v]) => v != null);
  const total = rows.reduce((a, [, v]) => a + v, 0);
  $('timings').innerHTML = rows
    .map(([k, v, heavy]) => `<div class="${heavy ? 'is-heavy' : ''}">
        <dt>${esc(k)}</dt><dd>${fmtMs(v)}</dd></div>`).join('')
    + `<div><dt>Summe</dt><dd>${fmtMs(total)}</dd></div>`;
}

// ------------------------------------------------------------------ errors

function showError(title, detail) {
  const p = $('error-plate');
  p.hidden = false;
  p.innerHTML = `<strong>${esc(title)}</strong>${detail ? `<code>${esc(detail)}</code>` : ''}`;
}

// ------------------------------------------------------------------- utils

const plural = (n, one, many) => `${n} ${n === 1 ? one : many}`;
const clamp01 = (v) => Math.max(0, Math.min(1, v));
const frac = (v, lo, hi) => (hi > lo ? (v - lo) / (hi - lo) : 0.5);
const oneLine = (s) => String(s || '').replace(/\s+/g, ' ').trim();
const slug = (s) => String(s).replace(/[^\w]+/g, '-').replace(/^-|-$/g, '');
const reEsc = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

function esc(s) {
  return String(s ?? '').replace(/[&<>"']/g, (c) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

export { esc };

// started last: every const in this module must be initialised before boot runs.
boot();
