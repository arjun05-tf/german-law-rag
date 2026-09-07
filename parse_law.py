"""
Parse a Gesetze-im-Internet XML law file into Absatz-level chunks.

Why Absatz and not fixed-size: a legal citation points at "§ 3 Abs. 1 ArbZG".
If we chunk on token count we cut across that boundary and can no longer say
which subsection an answer came from. Absatz is the smallest unit that is
independently citable, so it's the natural chunk.

Get the source file:
    curl -O https://www.gesetze-im-internet.de/arbzg/xml.zip
    unzip xml.zip          # -> BJNR117100994.xml (name varies per law)

Usage:
    python parse_law.py BJNR117100994.xml --inspect     # look before parsing
    python parse_law.py BJNR117100994.xml -o arbzg.jsonl
"""

import argparse
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict

# Absätze are marked inline as "(1)", "(2)" at the start of a paragraph.
# There is no XML element for them - the structure is in the text. Annoying,
# but consistent across every law on GII, so it's parseable.
ABSATZ_MARKER = re.compile(r"^\((\d+)\)\s*")


@dataclass
class Chunk:
    id: str
    gesetz: str
    paragraph: str        # "§ 3"
    paragraph_title: str  # "Arbeitszeit der Arbeitnehmer"
    abschnitt: str        # "Erster Abschnitt - Allgemeine Vorschriften"
    absatz: str           # "1", or "-" where the § has no numbered Absätze
    citation: str         # "§ 3 Abs. 1 ArbZG" - this is what we show the user
    text: str


def inspect(path):
    """Print the tag layout before trusting any assumption about it.

    GII's schema is stable but not identical across laws - older ones have
    quirks. Cheaper to look first than to debug silent empty output.
    """
    root = ET.parse(path).getroot()
    norms = root.findall("norm")
    print(f"root=<{root.tag}>  norms={len(norms)}")
    for norm in norms[:5]:
        enbez = norm.findtext("metadaten/enbez", "(none)")
        titel = norm.findtext("metadaten/titel", "(none)")
        paras = len(norm.findall("textdaten/text/Content/P"))
        print(f"  {enbez:<12} P={paras:<3} {titel[:50]}")
    print("  ...")


def node_text(node):
    """Flatten an element to plain text.

    itertext() pulls in nested markup - <DL>/<DT>/<DD> numbered lists inside an
    Absatz, footnote refs, emphasis. We want all of it as running text; the list
    structure doesn't survive embedding anyway.
    """
    return re.sub(r"\s+", " ", "".join(node.itertext())).strip()


def split_absaetze(paragraphs):
    """Group <P> elements into Absätze.

    A <P> starting with "(N)" opens Absatz N. Any <P> after it without a marker
    is a continuation - usually a Satz that got its own element, or a list
    trailer. Those belong to the Absatz above, not to a new one.
    """
    out, current_num, buffer = [], None, []

    def flush():
        if buffer:
            out.append((current_num or "-", " ".join(buffer)))

    for p in paragraphs:
        text = node_text(p)
        if not text:
            continue
        match = ABSATZ_MARKER.match(text)
        if match:
            flush()
            current_num, buffer = match.group(1), [ABSATZ_MARKER.sub("", text)]
        else:
            buffer.append(text)
    flush()
    return out


def parse(path):
    root = ET.parse(path).getroot()
    chunks = []
    gesetz = ""
    abschnitt = ""

    for norm in root.findall("norm"):
        meta = norm.find("metadaten")
        if meta is None:
            continue

        # First norm in the file is the law's header - no <enbez>, but it carries
        # the abbreviation we need for every citation downstream.
        if not gesetz:
            gesetz = meta.findtext("amtabk") or meta.findtext("jurabk") or ""

        # Section headings arrive as their own norm and apply to everything that
        # follows until the next one. Carrying it forward gives each chunk its
        # thematic context - useful later for filtering and for the router.
        gl = meta.find("gliederungseinheit")
        if gl is not None:
            bez = (gl.findtext("gliederungsbez") or "").strip()
            titel = (gl.findtext("gliederungstitel") or "").strip()
            abschnitt = f"{bez} - {titel}".strip(" -")

        enbez = (meta.findtext("enbez") or "").strip()
        # Skip Inhaltsübersicht, Eingangsformel, Schlussformel and the header
        # norm. Only "§ N" and "Art N" entries carry actual legal text.
        if not enbez.startswith(("§", "Art")):
            continue

        content = norm.find("textdaten/text/Content")
        if content is None:
            continue  # repealed ("weggefallen") - the norm exists, the text doesn't

        titel = (meta.findtext("titel") or "").strip()

        for num, text in split_absaetze(content.findall("P")):
            if len(text) < 20:
                continue  # cross-reference stubs, not answerable content
            absatz_part = f" Abs. {num}" if num != "-" else ""
            chunks.append(Chunk(
                id=f"{gesetz}-{enbez.replace(' ', '')}-{num}",
                gesetz=gesetz,
                paragraph=enbez,
                paragraph_title=titel,
                abschnitt=abschnitt,
                absatz=num,
                citation=f"{enbez}{absatz_part} {gesetz}",
                text=text,
            ))

    return chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xml")
    ap.add_argument("-o", "--out", default="chunks.jsonl")
    ap.add_argument("--inspect", action="store_true")
    args = ap.parse_args()

    if args.inspect:
        inspect(args.xml)
        return

    chunks = parse(args.xml)
    with open(args.out, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")

    print(f"{len(chunks)} chunks -> {args.out}")
    lengths = sorted(len(c.text) for c in chunks)
    print(f"chars: min={lengths[0]} median={lengths[len(lengths)//2]} max={lengths[-1]}")
    print("\nsample:")
    for c in chunks[:3]:
        print(f"  [{c.citation}] {c.text[:90]}...")


if __name__ == "__main__":
    main()
