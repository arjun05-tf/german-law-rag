"""Generate the animated SVGs used by README.md.

Reads the real 768-dim passage vectors out of the local Qdrant store, so the
hero point cloud is a PCA of the actual index rather than decoration. The
retrieval numbers are the ones in the README eval tables.

    python .github/assets/make_assets.py
"""
import math
import pathlib

import numpy as np

OUT = pathlib.Path(__file__).parent
ROOT = OUT.parent.parent

BG, PANEL, INK, MUTED, DIM = "#0b0f14", "#1b2735", "#e6edf3", "#8b98a6", "#5a6673"
GOLD, RED, CYAN = "#f2b53c", "#e0564f", "#4cc9f0"
SECTION_COLORS = [GOLD, CYAN, RED, "#a78bfa", "#4ade80", "#fb923c", "#f472b6", "#2dd4bf"]
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"
SANS = "-apple-system, BlinkMacSystemFont, Segoe UI, Inter, Helvetica, sans-serif"

FRAMES = 30
SPIN_SECONDS = 30



def load_index():
    """Real vectors + payloads from the on-disk Qdrant collection."""
    from qdrant_client import QdrantClient

    client = QdrantClient(path=str(ROOT / "qdrant_data"))
    points, offset = [], None
    while True:
        batch, offset = client.scroll(
            "gesetze", limit=256, offset=offset, with_vectors=True, with_payload=True
        )
        points += batch
        if offset is None:
            break
    vectors = np.array([p.vector for p in points], dtype=np.float64)
    return vectors, [p.payload for p in points]


def pca3(vectors):
    centered = vectors - vectors.mean(0)
    _, singular, components = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ components[:3].T
    variance = (singular[:3] ** 2).sum() / (singular**2).sum()
    # Scale on the 90th percentile of the radii, not the max: one outlier Absatz would
    # otherwise collapse the other 79 into the middle of the sphere. Clamp radially
    # rather than per-axis so the cloud keeps its shape and stays inside the frame.
    norms = np.linalg.norm(coords, axis=1)
    coords /= np.percentile(norms, 90)
    norms = np.linalg.norm(coords, axis=1, keepdims=True)
    coords *= np.minimum(1.0, 1.1 / norms)
    return coords, variance


def neighbour_edges(vectors, coords, limit=32):
    """Top-1 cosine neighbour per chunk, deduped, strongest `limit` kept.

    Pairs that PCA throws far apart are dropped: the line would cross the whole
    cloud and read as noise rather than as a neighbour link.
    """
    unit = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    similarity = unit @ unit.T
    np.fill_diagonal(similarity, -1)
    edges = {}
    for i, j in enumerate(similarity.argmax(1)):
        edges[tuple(sorted((i, int(j))))] = float(similarity[i, j])
    span = {e: float(np.linalg.norm(coords[e[0]] - coords[e[1]])) for e in edges}
    cutoff = np.percentile(list(span.values()), 60)
    kept = {e: s for e, s in edges.items() if span[e] <= cutoff}
    return [edge for edge, _ in sorted(kept.items(), key=lambda kv: -kv[1])[:limit]]


def rotate_frames(coords, tilt=math.radians(17)):
    """Spin about Y, fixed tilt about X, perspective-project. One list per frame."""
    frames = []
    for frame in range(FRAMES + 1):  # +1 closes the loop seamlessly
        angle = 2 * math.pi * (frame % FRAMES) / FRAMES
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        cos_t, sin_t = math.cos(tilt), math.sin(tilt)
        projected = []
        for x, y, z in coords:
            rx, rz = x * cos_a + z * sin_a, -x * sin_a + z * cos_a
            ry, rz = y * cos_t - rz * sin_t, y * sin_t + rz * cos_t
            scale = 4.5 / (4.5 - rz)  # weak perspective: enough depth cue, stays in frame
            projected.append((rx * scale, -ry * scale, scale))
        frames.append(projected)
    return frames


def trim(value, decimals):
    """Shortest exact text for a keyframe number; only a fraction may lose trailing zeros."""
    text = f"{value:.{decimals}f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def spin(attr, series, decimals=1):
    """Evenly spaced keyframes, so SMIL's default keyTimes applies and stays out of the file."""
    values = ";".join(trim(v, decimals) for v in series)
    return (
        f'<animate attributeName="{attr}" dur="{SPIN_SECONDS}s" repeatCount="indefinite" '
        f'values="{values}"/>'
    )


def panel(width, height, title=None, subtitle=None):
    parts = [
        f'<rect width="{width}" height="{height}" rx="14" fill="{BG}"/>',
        f'<rect width="{width}" height="{height}" rx="14" fill="url(#grid)"/>',
        f'<rect x="0.5" y="0.5" width="{width - 1}" height="{height - 1}" rx="14" '
        f'fill="none" stroke="{PANEL}"/>',
    ]
    if title:
        parts.append(
            f'<text x="28" y="36" font-family="{MONO}" font-size="12" fill="{MUTED}" '
            f'letter-spacing="2.4">{title}</text>'
        )
    if subtitle:
        parts.append(
            f'<text x="28" y="58" font-family="{SANS}" font-size="13" fill="{DIM}">{subtitle}</text>'
        )
    return "".join(parts)


def svg(width, height, body):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img"><defs>'
        f'<pattern id="grid" width="36" height="36" patternUnits="userSpaceOnUse">'
        f'<path d="M36 0H0V36" fill="none" stroke="#131c26" stroke-width="1"/></pattern>'
        f'<radialGradient id="halo" cx="50%" cy="50%" r="50%">'
        f'<stop offset="0%" stop-color="{CYAN}" stop-opacity="0.17"/>'
        f'<stop offset="70%" stop-color="{CYAN}" stop-opacity="0"/></radialGradient>'
        f'<linearGradient id="goldfill" x1="0" y1="0" x2="1" y2="0">'
        f'<stop offset="0%" stop-color="{GOLD}"/><stop offset="50%" stop-color="#fff0c0"/>'
        f'<stop offset="100%" stop-color="{GOLD}"/>'
        f'<animateTransform attributeName="gradientTransform" type="translate" from="-1 0" '
        f'to="1 0" dur="7s" repeatCount="indefinite"/></linearGradient>'
        f'</defs>{body}</svg>'
    )


# ---------------------------------------------------------------- hero


def hero(frames, payloads, variance, edges):
    width, height = 1200, 430
    cx, cy, radius = 884, 210, 122
    sections = sorted({p["abschnitt"] for p in payloads})
    colour = {s: SECTION_COLORS[i % len(SECTION_COLORS)] for i, s in enumerate(sections)}

    body = [panel(width, height)]
    body.append(
        f'<clipPath id="cloudclip"><rect x="628" y="14" width="{width - 642}" '
        f'height="{height - 56}" rx="12"/></clipPath><g clip-path="url(#cloudclip)">'
    )
    body.append(f'<ellipse cx="{cx}" cy="{cy}" rx="215" ry="215" fill="url(#halo)"/>')

    for i, j in edges:
        body.append(
            f'<line stroke="{CYAN}" stroke-opacity="0.15" stroke-width="1">'
            + spin("x1", [cx + frames[f][i][0] * radius for f in range(FRAMES + 1)], 0)
            + spin("y1", [cy + frames[f][i][1] * radius for f in range(FRAMES + 1)], 0)
            + spin("x2", [cx + frames[f][j][0] * radius for f in range(FRAMES + 1)], 0)
            + spin("y2", [cy + frames[f][j][1] * radius for f in range(FRAMES + 1)], 0)
            + "</line>"
        )

    for i in sorted(range(len(payloads)), key=lambda n: frames[0][n][2]):
        body.append(
            f'<circle fill="{colour[payloads[i]["abschnitt"]]}">'
            + spin("cx", [cx + frames[f][i][0] * radius for f in range(FRAMES + 1)], 0)
            + spin("cy", [cy + frames[f][i][1] * radius for f in range(FRAMES + 1)], 0)
            + spin("r", [1.6 + (frames[f][i][2] - 0.78) * 8.0 for f in range(FRAMES + 1)], 2)
            + spin(
                "fill-opacity",
                [min(1.0, max(0.3, (frames[f][i][2] - 0.74) * 2.2)) for f in range(FRAMES + 1)],
                2,
            )
            + "</circle>"
        )

    body.append("</g>")
    body.append(
        f'<text x="{cx}" y="{height - 26}" text-anchor="middle" font-family="{MONO}" '
        f'font-size="10.5" fill="{DIM}">80 Absatz vectors &#183; PCA of the live 768-dim index '
        f'&#183; {variance * 100:.0f}% variance &#183; colour = Abschnitt</text>'
    )

    body.append(
        f'<text x="60" y="92" font-family="{MONO}" font-size="12" fill="{CYAN}" '
        f'letter-spacing="3.4">RETRIEVAL &#183; RERANK &#183; ATTRIBUTION</text>'
    )
    body.append(
        f'<text x="60" y="152" font-family="{MONO}" font-size="44" font-weight="700" '
        f'fill="url(#goldfill)">german-law-rag</text>'
    )
    body.append(
        f'<line x1="60" y1="176" x2="60" y2="176" stroke="{GOLD}" stroke-width="2" '
        f'stroke-opacity="0.55"><animate attributeName="x2" from="60" to="500" dur="1.4s" '
        f'fill="freeze"/></line>'
    )
    body.append(
        f'<text x="60" y="206" font-family="{SANS}" font-size="16" fill="{MUTED}">'
        f'Answers over German employment law, cited to the exact Absatz.</text>'
    )

    query = '--answer "Wie lange sind meine Ruhepausen?"'
    answer = "§ 4 ArbZG — 30 min bei 6–9 h Arbeitszeit"
    qw, steps = 416, 32
    tick_times = ";".join(f"{i / 64:.4f}" for i in range(steps))
    widths = ";".join(f"{qw * min(i, len(query)) / len(query):.1f}" for i in range(steps))
    carets = ";".join(f"{94 + qw * min(i, len(query)) / len(query):.1f}" for i in range(steps))

    body.append(
        f'<rect x="60" y="238" width="524" height="94" rx="10" fill="#0e151d" stroke="{PANEL}"/>'
        f'<text x="76" y="270" font-family="{MONO}" font-size="13" fill="{GOLD}">$</text>'
        f'<clipPath id="typeclip"><rect x="94" y="254" height="22" width="0">'
        f'<animate attributeName="width" dur="9s" repeatCount="indefinite" calcMode="discrete" '
        f'keyTimes="{tick_times}" values="{widths}"/></rect></clipPath>'
        f'<g clip-path="url(#typeclip)"><text x="94" y="270" font-family="{MONO}" font-size="13" '
        f'fill="{INK}" textLength="{qw}" lengthAdjust="spacingAndGlyphs">{query}</text></g>'
        f'<rect y="256" width="7" height="16" fill="{GOLD}">'
        f'<animate attributeName="x" dur="9s" repeatCount="indefinite" calcMode="discrete" '
        f'keyTimes="{tick_times}" values="{carets}"/>'
        f'<animate attributeName="opacity" values="1;0;1" dur="0.9s" repeatCount="indefinite"/>'
        f'</rect>'
        f'<text x="94" y="308" font-family="{MONO}" font-size="13" fill="{CYAN}" '
        f'textLength="356" lengthAdjust="spacingAndGlyphs" opacity="0">{answer}'
        f'<animate attributeName="opacity" dur="9s" repeatCount="indefinite" '
        f'keyTimes="0;0.52;0.6;0.93;1" values="0;0;1;1;0"/></text>'
    )

    x = 60
    for i, (value, label) in enumerate(
        [("80", "Absätze"), ("0.60", "recall@1"), ("0.90", "recall@5"), ("0.83", "AUROC")]
    ):
        body.append(
            f'<g opacity="0"><animate attributeName="opacity" from="0" to="1" dur="0.5s" '
            f'begin="{1.2 + i * 0.14}s" fill="freeze"/>'
            f'<rect x="{x}" y="356" width="118" height="46" rx="9" fill="#0e151d" '
            f'stroke="{PANEL}"/>'
            f'<text x="{x + 14}" y="379" font-family="{MONO}" font-size="17" font-weight="700" '
            f'fill="{INK}">{value}</text>'
            f'<text x="{x + 14}" y="394" font-family="{MONO}" font-size="9.5" fill="{DIM}" '
            f'letter-spacing="1.2">{label}</text></g>'
        )
        x += 130
    return svg(width, height, "".join(body))


# ---------------------------------------------------------------- pipeline

STAGES = [
    ("01", "Parse", "parse_law.py", "80 Absatz chunks", GOLD),
    ("02", "Embed", "multilingual-e5-base", "768-dim vectors", "#a78bfa"),
    ("03", "Index", "Qdrant", "cosine search", CYAN),
    ("04", "Rerank", "cross-encoder", "top-20 reordered", "#4ade80"),
    ("05", "Generate", "gpt-4o-mini", "answer + § claims", RED),
]


def pipeline():
    width, height = 1200, 250
    box_w, gap, y, cycle = 196, 30, 96, 7.5
    total = len(STAGES) * box_w + (len(STAGES) - 1) * gap
    x0 = (width - total) / 2

    body = [panel(width, height, "PIPELINE", "each stage runs on the output of the one before it")]
    body.append(
        f'<line x1="{x0}" y1="{y + 40}" x2="{x0 + total}" y2="{y + 40}" stroke="{PANEL}" '
        f'stroke-width="2"/>'
    )
    for i, (num, name, tool, out, colour) in enumerate(STAGES):
        x = x0 + i * (box_w + gap)
        begin = f"{i * cycle / len(STAGES):.2f}s"
        body.append(
            f'<rect x="{x}" y="{y}" width="{box_w}" height="80" rx="11" fill="#0e151d" '
            f'stroke="{PANEL}"><animate attributeName="stroke" values="{PANEL};{colour};{PANEL}" '
            f'dur="{cycle}s" begin="{begin}" repeatCount="indefinite" keyTimes="0;0.08;0.3"/>'
            f'</rect>'
            f'<rect x="{x}" y="{y}" width="{box_w}" height="80" rx="11" fill="{colour}" '
            f'opacity="0"><animate attributeName="opacity" values="0;0.11;0" dur="{cycle}s" '
            f'begin="{begin}" repeatCount="indefinite" keyTimes="0;0.08;0.3"/></rect>'
            f'<text x="{x + 16}" y="{y + 26}" font-family="{MONO}" font-size="11" fill="{colour}" '
            f'letter-spacing="1.6">{num}</text>'
            f'<text x="{x + 16}" y="{y + 48}" font-family="{SANS}" font-size="17" '
            f'font-weight="600" fill="{INK}">{name}</text>'
            f'<text x="{x + 16}" y="{y + 66}" font-family="{MONO}" font-size="10.5" '
            f'fill="{MUTED}">{tool}</text>'
            f'<text x="{x + 16}" y="{y + 108}" font-family="{MONO}" font-size="10.5" '
            f'fill="{DIM}">{out}</text>'
        )
        if i < len(STAGES) - 1:
            mid = x + box_w + gap / 2
            body.append(
                f'<path d="M{mid - 7} {y + 34} l7 6 -7 6" fill="none" stroke="{DIM}" '
                f'stroke-width="1.6"/>'
            )
    body.append(
        f'<circle cy="{y + 40}" r="4.5" fill="{GOLD}"><animate attributeName="cx" from="{x0}" '
        f'to="{x0 + total}" dur="{cycle}s" repeatCount="indefinite"/>'
        f'<animate attributeName="opacity" values="0;1;1;0" keyTimes="0;0.04;0.96;1" '
        f'dur="{cycle}s" repeatCount="indefinite"/></circle>'
    )
    return svg(width, height, "".join(body))


# ---------------------------------------------------------------- results

ROWS = [
    ("All queries", "recall@1", 0.44, 0.60),
    ("All queries", "recall@5", 0.84, 0.90),
    ("German", "recall@1", 0.59, 0.59),
    ("German", "recall@5", 0.83, 0.89),
    ("English", "recall@1", 0.29, 0.60),
    ("English", "recall@5", 0.84, 0.91),
]


def results():
    width, height = 1200, 470
    left, top, bar_w, row_h = 212, 118, 700, 44
    baseline = top + row_h * len(ROWS)

    body = [
        panel(
            width,
            height,
            "RETRIEVAL · 91 LABELLED QUESTIONS",
            "dense top-5 against dense top-20 + cross-encoder rerank",
        )
    ]
    x = width - 28
    for label, colour in (("+ cross-encoder rerank", GOLD), ("dense top-5", DIM)):
        x -= len(label) * 6.6
        body.append(
            f'<text x="{x}" y="{36}" font-family="{MONO}" font-size="11" fill="{MUTED}">'
            f'{label}</text>'
            f'<rect x="{x - 20}" y="27" width="11" height="11" rx="3" fill="{colour}"/>'
        )
        x -= 44
    for tick in range(0, 11, 2):
        x = left + bar_w * tick / 10
        body.append(
            f'<line x1="{x}" y1="{top - 12}" x2="{x}" y2="{baseline}" stroke="#141d27"/>'
            f'<text x="{x}" y="{baseline + 20}" text-anchor="middle" font-family="{MONO}" '
            f'font-size="10" fill="{DIM}">{tick / 10:.1f}</text>'
        )
    for i, (group, metric, dense, reranked) in enumerate(ROWS):
        y = top + i * row_h
        if metric == "recall@1":
            body.append(
                f'<text x="28" y="{y + 24}" font-family="{SANS}" font-size="14" '
                f'font-weight="600" fill="{INK}">{group}</text>'
            )
        body.append(
            f'<text x="{left - 14}" y="{y + 22}" text-anchor="end" font-family="{MONO}" '
            f'font-size="11" fill="{MUTED}">{metric}</text>'
        )
        for row, (value, colour) in enumerate(
            ((dense, DIM), (reranked, GOLD))
        ):
            by = y + row * 15
            body.append(
                f'<rect x="{left}" y="{by}" height="11" rx="3" width="0" fill="{colour}" '
                f'fill-opacity="{0.55 if row == 0 else 1}">'
                f'<animate attributeName="width" from="0" to="{bar_w * value:.1f}" dur="1.1s" '
                f'begin="{0.25 + i * 0.12 + row * 0.08}s" fill="freeze" calcMode="spline" '
                f'keySplines="0.2 0.8 0.2 1" keyTimes="0;1"/></rect>'
                f'<text x="{left + bar_w * value + 10:.1f}" y="{by + 10}" font-family="{MONO}" '
                f'font-size="11" fill="{INK if row else MUTED}" opacity="0">{value:.2f}'
                f'<animate attributeName="opacity" from="0" to="1" dur="0.4s" '
                f'begin="{1.3 + i * 0.12 + row * 0.08}s" fill="freeze"/></text>'
            )
    body.append(
        f'<g opacity="0"><animate attributeName="opacity" from="0" to="1" dur="0.6s" begin="2.4s" '
        f'fill="freeze"/>'
        f'<rect x="{left}" y="{baseline + 34}" width="{bar_w}" height="40" rx="9" fill="#0e151d" '
        f'stroke="{PANEL}"/>'
        f'<text x="{left + 16}" y="{baseline + 59}" font-family="{SANS}" font-size="13" '
        f'fill="{MUTED}">English recall@1 <tspan fill="{RED}">0.29</tspan> to '
        f'<tspan fill="{GOLD}" font-weight="700">0.60</tspan> &#8212; the 30-point cross-lingual '
        f'gap closes to one point.</text></g>'
    )
    return svg(width, height, "".join(body))


# ---------------------------------------------------------------- abstention

SIGNALS = [
    ("Cosine top-1", 0.660, 0.846, 0.828, False),
    ("Cross-encoder top-1", 0.828, 0.594, 0.159, True),
]


def abstention():
    width, height = 1200, 330
    left, axis_w = 250, 690

    body = [
        panel(
            width,
            height,
            "ABSTENTION · 91 ANSWERABLE vs 22 UNANSWERABLE",
            "mean top-1 score per population, on one 0–1 axis",
        )
    ]
    for i, (name, auroc, yes, no, strong) in enumerate(SIGNALS):
        y = 132 + i * 112
        accent = GOLD if strong else DIM
        body.append(
            f'<text x="28" y="{y - 24}" font-family="{SANS}" font-size="15" font-weight="600" '
            f'fill="{INK}">{name}</text>'
            f'<text x="28" y="{y - 6}" font-family="{MONO}" font-size="11" fill="{accent}">'
            f'AUROC {auroc:.3f}</text>'
            f'<line x1="{left}" y1="{y}" x2="{left + axis_w}" y2="{y}" stroke="{PANEL}" '
            f'stroke-width="2"/>'
        )
        for tick in range(0, 11, 2):
            x = left + axis_w * tick / 10
            body.append(
                f'<line x1="{x}" y1="{y - 5}" x2="{x}" y2="{y + 5}" stroke="{DIM}"/>'
                f'<text x="{x}" y="{y + 22}" text-anchor="middle" font-family="{MONO}" '
                f'font-size="10" fill="{DIM}">{tick / 10:.1f}</text>'
            )
        xy, xn = left + axis_w * yes, left + axis_w * no
        # When the two means nearly coincide — which is the finding for cosine — their
        # value labels have to stack instead of overlap.
        crowded = abs(xy - xn) < 64
        for value, x, colour, label, dy, label_y in (
            (no, xn, RED, "unanswerable", 20, y - (36 if crowded else 18)),
            (yes, xy, CYAN, "answerable", 4, y - 18),
        ):
            body.append(
                f'<circle cx="{left}" cy="{y}" r="8" fill="{colour}" fill-opacity="0.9">'
                f'<animate attributeName="cx" from="{left}" to="{x:.1f}" dur="1.3s" '
                f'begin="{0.3 + i * 0.25}s" fill="freeze" calcMode="spline" '
                f'keySplines="0.2 0.8 0.2 1" keyTimes="0;1"/></circle>'
                f'<circle cx="{x:.1f}" cy="{y}" r="8" fill="none" stroke="{colour}" opacity="0">'
                f'<animate attributeName="r" values="8;22" dur="2.4s" begin="{1.6 + i * 0.25}s" '
                f'repeatCount="indefinite"/>'
                f'<animate attributeName="opacity" values="0.5;0" dur="2.4s" '
                f'begin="{1.6 + i * 0.25}s" repeatCount="indefinite"/></circle>'
                f'<text x="{x:.1f}" y="{label_y}" text-anchor="middle" font-family="{MONO}" '
                f'font-size="11" fill="{colour}" opacity="0">{value:.3f}'
                f'<animate attributeName="opacity" from="0" to="1" dur="0.5s" '
                f'begin="{1.5 + i * 0.25}s" fill="freeze"/></text>'
                f'<text x="{left + axis_w + 16}" y="{y + dy}" font-family="{MONO}" font-size="10" '
                f'fill="{colour}" opacity="0.85">{label}</text>'
            )
        body.append(
            f'<g opacity="0"><animate attributeName="opacity" from="0" to="1" dur="0.5s" '
            f'begin="{1.8 + i * 0.25}s" fill="freeze"/>'
            f'<line x1="{min(xy, xn):.1f}" y1="{y + 42}" x2="{max(xy, xn):.1f}" y2="{y + 42}" '
            f'stroke="{accent}" stroke-width="1.5"/>'
            f'<text x="{(xy + xn) / 2:.1f}" y="{y + 58}" text-anchor="middle" font-family="{MONO}" '
            f'font-size="11" fill="{accent}">gap {abs(yes - no):.3f}</text></g>'
        )
    return svg(width, height, "".join(body))


if __name__ == "__main__":
    vectors, payloads = load_index()
    coords, variance = pca3(vectors)
    frames = rotate_frames(coords)
    edges = neighbour_edges(vectors, coords)
    for name, markup in (
        ("hero.svg", hero(frames, payloads, variance, edges)),
        ("pipeline.svg", pipeline()),
        ("results.svg", results()),
        ("abstention.svg", abstention()),
    ):
        (OUT / name).write_text(markup, encoding="utf-8")
        print(f"{name:16} {len(markup) / 1024:6.1f} KB")
