#!/usr/bin/env python3
"""Egyptian Civil Code (bilingual EN/AR table PDF) -> pages + article chunks.

Offline data-pipeline step (extract -> normalize -> chunk). It implements:

* ADR-0006  two-text model (``raw_text`` quoted, ``norm_text`` embedded), a monotonic
            norm->raw offset map produced by span-preserving normalization rules
            (each individually toggleable and logged), OCR detection + coverage
            reporting, and an optional CER measurement against hand transcriptions.
* ADR-0007  article-aware chunking (primary) driven by the corpus' own markers
            (``Article N`` / ``مادة (N)``) on table-row boundaries, optional token cap
            with in-article splitting at paragraph/clause boundaries (``part_index``),
            a fixed-size baseline arm, and the article token-length distribution
            (the measured prerequisite for choosing caps).
* ADR-0005  provenance on every chunk (doc_id, pages, article, section, raw+norm
            offsets, jurisdiction, doc_status, effective_date, source_checksum), an
            ``extraction_config_id`` that feeds ``index_build_id``, and a queryable
            source-to-chunk manifest.
* ADR-0004  chunks ready for one BGE-M3 adapter (dense + sparse from ``norm_text``);
            article numbers are folded to ASCII so ``مادة 147`` is lexically matchable;
            the BGE-M3 max sequence length is checked when its tokenizer is available.

Why geometry and not ``page.get_text()``: the PDF (MS Word export) stores Arabic in
*visual* order. MuPDF's own reordering leaves three defects that silently corrupt legal
text: multi-digit numbers come out reversed (١٨٨٣ -> ٣٨٨١, so article 120 -> 021), the
lam-alef ligature is split into the wrong order (لا يجوز -> ال يجوز, 
       الإصدار -> اإلصدار),
and punctuation/brackets drift across lines. We rebuild every line from per-glyph
geometry: cluster by baseline, order by x (RTL for Arabic cells), swap split lam-alef
ligatures (the alef half is a zero-width glyph), and re-reverse embedded LTR runs
(digits/Latin). Codepoints are never altered at this stage - only their order - so
``raw_text`` remains the faithful, citable source text.

Usage:
    uv run python data/corpus/extract_civil_code.py <args>
    uv run python data/corpus/extract_civil_code.py --strategy both \
        --fixed-size-chars 1200 --fixed-overlap-chars 200
    uv run python data/corpus/extract_civil_code.py --max-tokens 512 \
        --tokenizer Qwen/Qwen2.5-7B-Instruct
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import re
import statistics
import sys
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pymupdf

PIPELINE_VERSION = "1.0.0"
SCHEMA_VERSION = "1.0"
HERE = Path(__file__).resolve().parent

# --------------------------------------------------------------------------------------
# Document metadata (document manifest, ADR-0005)
# --------------------------------------------------------------------------------------
DOC_ID = "egyptian-civil-law-1948"
DOC_META = {
    "title": "Egyptian Civil Code (Law No. 131 of 1948)",
    "title_ar": "القانون المدني المصري (القانون رقم ١٣١ لسنة ١٩٤٨)",
    "authority": "Arab Republic of Egypt",
    "jurisdiction": "EG",
    "doc_status": "in_force",
    # Promulgation law art. 2: in force from 15 October 1949.
    "publication_date": "1948-07-16",
    "effective_date": "1949-10-15",
    "language": ["ar", "en"],
    "doc_type": "statute",
}
EXPECTED_LAST_ARTICLE = 1149


# --------------------------------------------------------------------------------------
# ADR-0006 normalization with an explicit offset map
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class NormConfig:
    """ADR-0006 rules, applied in this order. Each is a run parameter."""

    decompose_presentation_forms: bool = True  # rule 1 (NFKC, lam-alef -> lam+alef)
    strip_tatweel: bool = True  # rule 2
    strip_tashkeel: bool = True  # rule 3 (+ superscript alef)
    unify_letters: bool = True  # rule 4 (alef forms, alef-maksura, teh-marbuta)
    fold_digits: bool = True  # rule 5 (Arabic-Indic + extended -> ASCII)
    collapse_whitespace_punct: bool = True  # rule 6

    def applied(self) -> list[str]:
        return [k for k, v in asdict(self).items() if v]


# A token is (char, raw_start, raw_end): the raw span it was produced from.
Token = tuple[str, int, int]

_PRESENTATION = ((0xFB50, 0xFDFF), (0xFE70, 0xFEFC))
_TATWEEL = "\u0640"
_TASHKEEL = {chr(c) for c in range(0x064B, 0x0653)} | {"\u0670"}
_UNIFY = {"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ة": "ه"}
_DIGITS = {chr(0x0660 + i): str(i) for i in range(10)} | {
    chr(0x06F0 + i): str(i) for i in range(10)
}
_AR_PUNCT = {"،": ",", "؛": ";", "؟": "?"}
_INVISIBLE = {
    *(chr(c) for c in range(0x200B, 0x2010)),
    *(chr(c) for c in range(0x202A, 0x202F)),
    *(chr(c) for c in range(0x2066, 0x206A)),
    "\ufeff",
}


def _is_presentation(ch: str) -> bool:
    o = ord(ch)
    return any(a <= o <= b for a, b in _PRESENTATION)


def normalize_tokens(text: str, cfg: NormConfig, base: int = 0) -> list[Token]:
    """Span-preserving normalization: every output char knows its raw span."""
    toks: list[Token] = [(c, base + i, base + i + 1) for i, c in enumerate(text)]
    if cfg.decompose_presentation_forms:
        out: list[Token] = []
        for c, s, e in toks:
            if _is_presentation(c):
                out.extend((d, s, e) for d in unicodedata.normalize("NFKC", c))
            else:
                out.append((c, s, e))
        toks = out
    if cfg.strip_tatweel:
        toks = [t for t in toks if t[0] != _TATWEEL]
    if cfg.strip_tashkeel:
        toks = [t for t in toks if t[0] not in _TASHKEEL]
    if cfg.unify_letters:
        toks = [(_UNIFY.get(c, c), s, e) for c, s, e in toks]
    if cfg.fold_digits:
        toks = [(_DIGITS.get(c, c), s, e) for c, s, e in toks]
    if cfg.collapse_whitespace_punct:
        toks = [(_AR_PUNCT.get(c, c), s, e) for c, s, e in toks if c not in _INVISIBLE]
        out = []
        run: list[Token] = []

        def flush() -> None:
            if run:
                nl = any(t[0] == "\n" for t in run)
                out.append(("\n" if nl else " ", run[0][1], run[-1][2]))
                run.clear()

        for t in toks:
            if t[0].isspace():
                run.append(t)
            else:
                flush()
                out.append(t)
        flush()
        while out and out[0][0].isspace():
            out.pop(0)
        while out and out[-1][0].isspace():
            out.pop()
        toks = out
    return toks


def normalize(text: str, cfg: NormConfig) -> str:
    return "".join(t[0] for t in normalize_tokens(text, cfg))


def compress_offset_map(toks: list[Token]) -> list[list[int]]:
    """Group tokens into [norm_start, norm_end, raw_start, raw_end].

    A group is *linear* (1:1 char mapping) iff norm_len == raw_len, otherwise it is
    *atomic* (the whole norm range maps to the whole raw range: expansions such as
    lam-alef -> lam+alef, or merges such as collapsed whitespace).
    """
    groups: list[list[int]] = []
    prev: Token | None = None
    for n, (_, s, e) in enumerate(toks):
        g = groups[-1] if groups else None
        if prev is not None and (s, e) == (prev[1], prev[2]):
            # Expansion of the previous raw span: make it atomic.
            if g[1] - g[0] == g[3] - g[2] and g[1] - g[0] > 1:  # linear: split last
                g[1] -= 1
                g[3] -= 1
                groups.append([n - 1, n + 1, s, e])
            else:
                g[1] = n + 1
        elif (
            e == s + 1
            and g is not None
            and g[1] - g[0] == g[3] - g[2]
            and g[3] == s
            and g[1] == n
        ):
            g[1] += 1
            g[3] += 1
        else:
            groups.append([n, n + 1, s, e])
        prev = (_, s, e)
    return groups


def map_norm_span(groups: list[list[int]], ns: int, ne: int) -> tuple[int, int]:
    """Resolve a norm span [ns, ne) to its raw span through the offset map."""
    if ne <= ns:
        raise ValueError(f"empty span {ns}:{ne}")
    starts = [g[0] for g in groups]
    gs = groups[bisect.bisect_right(starts, ns) - 1]
    ge = groups[bisect.bisect_right(starts, ne - 1) - 1]
    lin_s = gs[1] - gs[0] == gs[3] - gs[2]
    lin_e = ge[1] - ge[0] == ge[3] - ge[2]
    rs = gs[2] + (ns - gs[0]) if lin_s else gs[2]
    re_ = ge[2] + (ne - ge[0]) if lin_e else ge[3]
    return rs, re_


# --------------------------------------------------------------------------------------
# Geometry-based extraction (bidi repair)
# --------------------------------------------------------------------------------------
@dataclass
class Glyph:
    c: str
    x0: float
    x1: float
    y: float  # baseline (origin y)
    bold: bool
    idx: int  # stream order (tie-breaker)
    font: str = ""

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2


_ALEFS = set("اأإآ")
_MIRROR = {"(": ")", ")": "(", "[": "]", "]": "[", "{": "}", "}": "{"}
# Fonts whose glyphs encode the *shape* of a bracket (LTR-only fonts embedded by Word
# in Arabic runs). In an RTL line their brackets must be mirrored to recover the
# logical codepoint (UBA rule L4 in reverse). Measured on this corpus: mirroring
# Calibri brackets balances 99% of Arabic lines with brackets vs 3% without.
LTR_ONLY_FONTS = ("Calibri",)
_LTR_SEPARATORS = set("./,:-")


def _bidi(ch: str) -> str:
    b = unicodedata.bidirectional(ch)
    if b in ("AL", "R"):
        return "R"
    if b in ("EN", "AN"):
        return "D"
    if b == "L":
        return "L"
    if b == "NSM":
        return "M"
    return "N"


def _reverse_ltr_runs(chars: list[str]) -> list[str]:
    """In x-descending (RTL visual) order, LTR runs come out reversed: undo that.

    A run = digits/Latin letters, joined by separators between digits (dates, 1/2)
    and by neutrals between Latin letters (multi-word Latin phrases). Numbers separated
    by spaces are distinct runs (UBA rule N1 treats numbers as R for neutrals).
    """
    out = list(chars)
    kinds = [_bidi(c) for c in out]
    i, n = 0, len(out)
    while i < n:
        if kinds[i] not in ("D", "L"):
            i += 1
            continue
        j = i
        while j + 1 < n:
            k = kinds[j + 1]
            if k in ("D", "L"):
                j += 1
                continue
            # look-ahead over neutrals to the next strong char
            m = j + 1
            while m < n and kinds[m] == "N":
                m += 1
            if m < n and kinds[m] in ("D", "L"):
                gap = out[j + 1 : m]
                if kinds[j] == "L" and kinds[m] == "L":
                    j = m
                    continue
                if (
                    kinds[j] == "D"
                    and kinds[m] == "D"
                    and len(gap) == 1
                    and gap[0] in _LTR_SEPARATORS
                ):
                    j = m
                    continue
            break
        out[i : j + 1] = out[i : j + 1][::-1]
        i = j + 1
    return out


@dataclass
class LineStats:
    lam_alef_swaps: int = 0
    ltr_runs_reversed: int = 0
    brackets_mirrored: int = 0
    overlapping_spaces_dropped: int = 0


def build_lines(glyphs: list[Glyph], rtl: bool, stats: LineStats) -> list[str]:
    """Cluster glyphs into baseline lines and order them for reading."""
    if not glyphs:
        return []
    lines: list[list[Glyph]] = []
    ys: list[float] = []
    for g in sorted(glyphs, key=lambda g: (g.y, g.idx)):
        if ys and abs(g.y - ys[-1]) <= 2.0:
            lines[-1].append(g)
        else:
            lines.append([g])
            ys.append(g.y)
    result = []
    for line in lines:
        # Word sometimes emits a space glyph whose box covers neighbouring glyphs
        # (e.g. inside "٦٠١"); its position is meaningless and splits number runs.
        ink = [(g.x0, g.x1) for g in line if not g.c.isspace()]
        kept = []
        for g in line:
            w = g.x1 - g.x0
            if g.c.isspace() and w > 0.5:
                cover = sum(max(0.0, min(g.x1, b) - max(g.x0, a)) for a, b in ink)
                if cover > 0.5 * w:
                    stats.overlapping_spaces_dropped += 1
                    continue
            kept.append(g)
        line = kept
        if rtl:
            line.sort(key=lambda g: (-round(g.cx, 2), g.idx))
            chars: list[str] = []
            k = 0
            while k < len(line):
                g = line[k]
                nxt = line[k + 1] if k + 1 < len(line) else None
                # Split lam-alef ligature: the alef half is a zero-width glyph that
                # sorts before the lam; logical order is lam then alef.
                if (
                    g.c in _ALEFS
                    and g.x1 - g.x0 < 0.05
                    and nxt is not None
                    and nxt.c == "ل"
                ):
                    chars.extend(["ل", g.c])
                    stats.lam_alef_swaps += 1
                    k += 2
                    continue
                c = g.c
                if c in _MIRROR and g.font.startswith(LTR_ONLY_FONTS):
                    c = _MIRROR[c]
                    stats.brackets_mirrored += 1
                chars.append(c)
                k += 1
            rev = _reverse_ltr_runs(chars)
            if rev != chars:
                stats.ltr_runs_reversed += 1
            text = "".join(rev)
        else:
            line.sort(key=lambda g: (round(g.cx, 2), g.idx))
            text = "".join(g.c for g in line)
        text = text.strip()
        if text:
            result.append(text)
    return result


def is_rtl(glyphs: list[Glyph]) -> bool:
    c = Counter(_bidi(g.c) for g in glyphs)
    return c["R"] > c["L"]


# --------------------------------------------------------------------------------------
# Markers (ADR-0007: the corpus' own article markers)
# --------------------------------------------------------------------------------------
# Source typos exist: "rticle 452", "Article1022".
EN_MARKER = re.compile(r"^\s*A?rticle\s*(\d+)\b")
EN_MARKER_LINE = re.compile(r"^\s*A?rticle\s*(\d+)\s*$")
AR_MARKER = re.compile(r"^\s*(?:ال)?ماد[ةه]\s*[()\[\]\s]*(\d+)\s*[()\[\]\s]*")
AR_MARKER_LINE = re.compile(r"^\s*(?:ال)?ماد[ةه]\s*[()\[\]\s]*(\d+)\s*[()\[\]\s]*$")
EN_REPEAL = re.compile(
    r"Articles?\s+(\d+)\s*-\s*(\d+)\s+(?:have\s+been\s+)?repealed", re.I
)
AR_REPEAL = re.compile(r"^\s*المواد\s+من\s+(\d+)\s+(?:الي|إلى|الى)\s+(\d+)")
AR_REPEAL_WORD = re.compile(r"الغيت|ملغاه|ملغي")

HEADING_LEVELS_EN = [
    # (regex, level_name, number_extractor)
    (re.compile(r"^(FIRST|SECOND|THIRD|FOURTH)\s+PART\b", re.I), "part", None),
    (re.compile(r"^PRELIMINARY\s+TITLE\b", re.I), "part", None),
    (re.compile(r"^BOOK\s+([IVX]+)\b", re.I), "book", lambda m: m.group(1).upper()),
    (
        re.compile(r"^CHAPTER\s+([IVX]+)\b", re.I),
        "chapter",
        lambda m: m.group(1).upper(),
    ),
    (
        re.compile(r"^SECTION\s+([IVX]+)\b", re.I),
        "section",
        lambda m: m.group(1).upper(),
    ),
    (re.compile(r"^(\d+)\s*[-.]"), "subsection", lambda m: m.group(1)),
]
HEADING_LEVELS_AR = [
    (re.compile(r"^القسم\s*(?:التمهيدي|الأول|الثاني|الأول)"), "part", None),
    (re.compile(r"^الباب\s+التمهيدي"), "part", None),
    (re.compile(r"^الكتاب\s+([ء-ي]+)\b"), "book", lambda m: m.group(1)),
    (re.compile(r"^الباب\s+([ء-ي]+)\b"), "chapter", lambda m: m.group(1)),
    (re.compile(r"^الفصل\s+([ء-ي]+)\b"), "section", lambda m: m.group(1)),
    (re.compile(r"^(\d+)\s*-"), "subsection", lambda m: m.group(1)),
]

_QUICK = NormConfig()  # used only for marker/heading detection, never stored


def ar_marker(text: str, line_only: bool = False) -> int | None:
    m = (AR_MARKER_LINE if line_only else AR_MARKER).match(normalize(text, _QUICK))
    return int(m.group(1)) if m else None


def en_marker(text: str, line_only: bool = False) -> int | None:
    m = (EN_MARKER_LINE if line_only else EN_MARKER).match(text)
    return int(m.group(1)) if m else None


def repeal_range(en: str, ar: str) -> tuple[int, int] | None:
    """A row that *is* a repeal notice (not an article that merely cites a range)."""
    m = EN_REPEAL.search("\n".join(en.split("\n")[:3]))
    if m is None:
        ar_n = normalize(ar, _QUICK)
        m = AR_REPEAL.match(ar_n) if AR_REPEAL_WORD.search(ar_n) else None
    return (int(m.group(1)), int(m.group(2))) if m else None


def heading_level(title_en: str, title_ar: str) -> tuple[str, str]:
    """Return (level_name, number) for the heading."""
    for rx, lvl_name, extractor in HEADING_LEVELS_EN:
        if title_en:
            m = rx.search(title_en)
            if m:
                num = extractor(m) if extractor else ""
                return lvl_name, num or ""
    ar = normalize(title_ar, _QUICK)
    for rx, lvl_name, extractor in HEADING_LEVELS_AR:
        if ar:
            m = rx.search(ar)
            if m:
                num = extractor(m) if extractor else ""
                return lvl_name, num or ""
    return "unknown", ""


# --------------------------------------------------------------------------------------
# Page model
# --------------------------------------------------------------------------------------
@dataclass
class Cell:
    lang: str  # "en" (left column) | "ar" (right column)
    side: str
    bbox: list[float]
    lines: list[str]
    bold_ratio: float
    detected_lang: str

    @property
    def raw(self) -> str:
        return "\n".join(self.lines)


@dataclass
class Row:
    page: int
    index: int
    bbox: list[float]
    cells: dict[str, Cell]

    def text(self, lang: str) -> str:
        return self.cells[lang].raw if lang in self.cells else ""


@dataclass
class Block:
    block_id: str
    page: int
    lang: str
    row_index: int
    role: str = "unassigned"
    article: str | None = None
    bbox: list[float] = field(default_factory=list)
    raw_text: str = ""
    raw_start: int = 0
    raw_end: int = 0
    norm_start: int = 0
    norm_end: int = 0


@dataclass
class Column:
    raw_text: str = ""
    norm_text: str = ""
    offset_map: list[list[int]] = field(default_factory=list)
    blocks: list[Block] = field(default_factory=list)


@dataclass
class Page:
    number: int
    rows: list[Row]
    columns: dict[str, Column]
    outside: list[dict]
    needs_ocr: bool
    chars_total: int
    chars_in_tables: int
    has_images: bool


def _glyphs_from_page(page: pymupdf.Page) -> list[Glyph]:
    raw = page.get_text("rawdict")
    out: list[Glyph] = []
    idx = 0
    for b in raw["blocks"]:
        for line in b.get("lines", []):
            for span in line["spans"]:
                bold = bool(span["flags"] & 16) or "Bold" in span["font"]
                for ch in span["chars"]:
                    if not ch["c"]:
                        continue
                    x0, _, x1, _ = ch["bbox"]
                    out.append(
                        Glyph(ch["c"], x0, x1, ch["origin"][1], bold, idx, span["font"])
                    )
                    idx += 1
    return out


def extract_page(page: pymupdf.Page, number: int, stats: LineStats) -> Page:
    glyphs = _glyphs_from_page(page)
    mid = page.rect.width / 2
    cells_geom: list[tuple[int, list[float]]] = []  # (row_global, bbox)
    row_bboxes: list[list[float]] = []
    tables = sorted(page.find_tables().tables, key=lambda t: t.bbox[1])
    for t in tables:
        for r in t.rows:
            ri = len(row_bboxes)
            row_bboxes.append(list(r.bbox))
            for c in r.cells:
                if c:
                    cells_geom.append((ri, list(c)))
    assigned: dict[int, list[Glyph]] = {i: [] for i in range(len(cells_geom))}
    outside: list[Glyph] = []
    for g in glyphs:
        cy = g.y - 2.0  # slightly above the baseline: inside the glyph body
        for ci, (_, (x0, y0, x1, y1)) in enumerate(cells_geom):
            if x0 <= g.cx <= x1 and y0 <= cy <= y1:
                assigned[ci].append(g)
                break
        else:
            outside.append(g)

    rows: list[Row] = []
    for ri, rb in enumerate(row_bboxes):
        cells: dict[str, Cell] = {}
        for ci, (rj, bbox) in enumerate(cells_geom):
            if rj != ri:
                continue
            gl = assigned[ci]
            side = "left" if (bbox[0] + bbox[2]) / 2 < mid else "right"
            lang = "en" if side == "left" else "ar"
            rtl = is_rtl(gl) if gl else lang == "ar"
            lines = build_lines(gl, rtl, stats)
            non_space = [g for g in gl if not g.c.isspace()]
            bold = sum(g.bold for g in non_space) / len(non_space) if non_space else 0
            cells[lang] = Cell(
                lang,
                side,
                [round(v, 2) for v in bbox],
                lines,
                bold,
                "ar" if rtl else "en",
            )
        rows.append(Row(number, ri, [round(v, 2) for v in rb], cells))

    outside_blocks = []
    if outside:
        rtl = is_rtl(outside)
        lines = build_lines(outside, rtl, stats)
        if lines:
            outside_blocks.append(
                {"lang": "ar" if rtl else "en", "raw_text": "\n".join(lines)}
            )
    total = sum(1 for g in glyphs if not g.c.isspace())
    in_tables = total - sum(1 for g in outside if not g.c.isspace())
    has_images = bool(page.get_images(full=False))
    return Page(
        number=number,
        rows=rows,
        columns={},
        outside=outside_blocks,
        needs_ocr=total < 30,  # no usable text layer (e.g. scanned page)
        chars_total=total,
        chars_in_tables=in_tables,
        has_images=has_images,
    )


def assemble_columns(page: Page, cfg: NormConfig) -> None:
    """Per language column: raw page text, norm page text, offset map, blocks."""
    for lang, side in (("en", "left"), ("ar", "right")):
        col = Column()
        raw_parts: list[str] = []
        toks: list[Token] = []
        pos = 0
        for row in page.rows:
            cell = row.cells.get(lang)
            if cell is None or not cell.lines:
                continue
            if raw_parts:  # block separator, mapped 1:1
                sep = "\n\n"
                toks.extend((c, pos + i, pos + i + 1) for i, c in enumerate(sep))
                raw_parts.append(sep)
                pos += len(sep)
            raw = cell.raw
            b = Block(
                block_id=f"p{page.number}-{side}-b{len(col.blocks) + 1:02d}",
                page=page.number,
                lang=lang,
                row_index=row.index,
                bbox=cell.bbox,
                raw_text=raw,
                raw_start=pos,
                raw_end=pos + len(raw),
            )
            b.norm_start = len(toks)
            toks.extend(normalize_tokens(raw, cfg, base=pos))
            b.norm_end = len(toks)
            col.blocks.append(b)
            raw_parts.append(raw)
            pos += len(raw)
        col.raw_text = "".join(raw_parts)
        col.norm_text = "".join(t[0] for t in toks)
        col.offset_map = compress_offset_map(toks)
        page.columns[lang] = col


# --------------------------------------------------------------------------------------
# Article assembly (structure) - ADR-0007
# --------------------------------------------------------------------------------------
@dataclass
class Article:
    number: str
    status: str  # in_force | repealed
    section_path: list[dict]
    # Explicit hierarchy fields
    part_name: str = ""
    book_no: str = ""
    chapter_no: str = ""
    section_no: str = ""
    section_title: dict = field(default_factory=lambda: {"en": "", "ar": ""})
    blocks: dict[str, list[Block]] = field(default_factory=lambda: {"en": [], "ar": []})
    en_number: int | None = None
    ar_number: int | None = None
    first_page: int = 0


@dataclass
class Issues:
    errors: list[dict] = field(default_factory=list)
    warnings: list[dict] = field(default_factory=list)

    def err(self, code: str, **kw) -> None:
        self.errors.append({"code": code, **kw})

    def warn(self, code: str, **kw) -> None:
        self.warnings.append({"code": code, **kw})


def _block_for(page: Page, lang: str, row_index: int) -> Block | None:
    col = page.columns.get(lang)
    if not col:
        return None
    for b in col.blocks:
        if b.row_index == row_index:
            return b
    return None


def _clean_title(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def assemble_articles(pages: list[Page], issues: Issues) -> list[Article]:
    articles: list[Article] = []
    # Hierarchy stack: part -> book -> chapter -> section -> subsection
    LEVEL_ORDER = {"part": 0, "book": 1, "chapter": 2, "section": 3, "subsection": 4}
    stack: list[dict] = []

    # Scan outside_table text for PART headings (Preliminary Title, etc.)
    for page in pages:
        for o in page.outside:
            txt = o["raw_text"]
            # Check for PRELIMINARY TITLE / الباب التمهيدي / باب تمهيدي
            if (
                "PRELIMINARY TITLE" in txt.upper()
                or "باب تمهيدي" in txt
                or "الباب التمهيدي" in txt
            ):
                stack.append(
                    {
                        "level_name": "part",
                        "level_order": 0,
                        "number": "",
                        "title_en": "Preliminary Title: General Provisions",
                        "title_ar": "الباب التمهيدي: أحكام عامة",
                    }
                )
                break
            # FIRST PART / القسم الأول
            elif (
                "FIRST PART" in txt.upper()
                or "القسم الأول" in txt
                or "قسم الأول" in txt
            ):
                stack.append(
                    {
                        "level_name": "part",
                        "level_order": 0,
                        "number": "",
                        "title_en": "First Part: Obligations in General",
                        "title_ar": "القسم الأول: التزامات بوجه عامة",
                    }
                )
                break
            # SECOND PART / القسم الثاني
            elif (
                "SECOND PART" in txt.upper()
                or "القسم الثاني" in txt
                or "قسم الثاني" in txt
            ):
                stack.append(
                    {
                        "level_name": "part",
                        "level_order": 0,
                        "number": "",
                        "title_en": "Second Part: Real Rights",
                        "title_ar": "القسم الثاني: الحقوق العينية",
                    }
                )
                break

    current: Article | None = None
    for page in pages:
        if page.needs_ocr:
            current = None  # a gap breaks continuity; never stitch across it
            continue
        for row in page.rows:
            en, ar = row.text("en"), row.text("ar")
            for lang, cell in row.cells.items():
                if cell.detected_lang != lang and cell.lines:
                    issues.err(
                        "column_language_mismatch",
                        page=page.number,
                        row=row.index,
                        column=lang,
                        detected=cell.detected_lang,
                    )
            blocks = {
                lang: b
                for lang in ("en", "ar")
                if (b := _block_for(page, lang, row.index)) is not None
            }
            if not blocks:
                continue
            bold_vals = [c.bold_ratio for c in row.cells.values() if c.lines]
            bold = max(bold_vals) if bold_vals else 0.0
            rr = repeal_range(en, ar)
            n_en, n_ar = en_marker(en), ar_marker(ar)

            # Embedded markers = a missed row boundary (two articles in one cell).
            for lang, text, fn in (("en", en, en_marker), ("ar", ar, ar_marker)):
                for ln in text.split("\n")[1:]:
                    if fn(ln, line_only=True) is not None:
                        issues.err(
                            "embedded_article_marker",
                            page=page.number,
                            row=row.index,
                            lang=lang,
                            line=ln,
                        )

            if rr is not None:
                number, status = f"{rr[0]}-{rr[1]}", "repealed"
            elif n_en is not None or n_ar is not None:
                number, status = str(n_en if n_en is not None else n_ar), "in_force"
            else:
                number = status = None

            if number is not None:
                if n_en is not None and n_ar is not None and n_en != n_ar:
                    issues.err(
                        "en_ar_article_number_mismatch",
                        page=page.number,
                        row=row.index,
                        en=n_en,
                        ar=n_ar,
                    )
                elif rr is None and (n_en is None or n_ar is None):
                    issues.warn(
                        "article_marker_one_column_only",
                        page=page.number,
                        article=number,
                        missing="en" if n_en is None else "ar",
                    )
                # Build explicit hierarchy fields from stack
                part_name = book_no = chapter_no = section_no = ""
                section_title = {"en": "", "ar": ""}
                for h in stack:
                    if h["level_name"] == "part":
                        part_name = h.get("title_en") or h.get("title_ar") or ""
                    elif h["level_name"] == "book":
                        book_no = h["number"]
                    elif h["level_name"] == "chapter":
                        chapter_no = h["number"]
                    elif h["level_name"] == "section":
                        section_no = h["number"]
                        section_title = {
                            "en": h.get("title_en", ""),
                            "ar": h.get("title_ar", ""),
                        }
                    elif h["level_name"] == "subsection":
                        if not section_title["en"] and not section_title["ar"]:
                            section_title = {
                                "en": h.get("title_en", ""),
                                "ar": h.get("title_ar", ""),
                            }
                current = Article(
                    number=number,
                    status=status,
                    section_path=[dict(h) for h in stack],
                    part_name=part_name,
                    book_no=book_no,
                    chapter_no=chapter_no,
                    section_no=section_no,
                    section_title=section_title,
                    en_number=n_en,
                    ar_number=n_ar,
                    first_page=page.number,
                )
                articles.append(current)
                role = "repealed_range" if rr else "article_start"
            elif bold >= 0.6:
                title_en = _clean_title(en)
                title_ar = _clean_title(ar)
                lvl_name, lvl_num = heading_level(title_en, title_ar)
                lvl_order = LEVEL_ORDER.get(lvl_name, 5)
                # Pop stack elements with >= order
                while (
                    stack and LEVEL_ORDER.get(stack[-1]["level_name"], 5) >= lvl_order
                ):
                    stack.pop()
                stack.append(
                    {
                        "level_name": lvl_name,
                        "level_order": lvl_order,
                        "number": lvl_num,
                        "title_en": title_en,
                        "title_ar": title_ar,
                    }
                )
                current = None
                for b in blocks.values():
                    b.role = "heading"
                continue
            elif current is not None:
                role = "continuation"
            else:
                issues.err(
                    "orphan_row", page=page.number, row=row.index, text=(en or ar)[:80]
                )
                for b in blocks.values():
                    b.role = "orphan"
                continue

            for lang, b in blocks.items():
                b.role = role
                b.article = current.number
                current.blocks[lang].append(b)
    return articles


# --------------------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------------------
@dataclass
class Unit:
    """One line of a block, in page-norm coordinates."""

    block: Block
    ns: int
    ne: int
    para_start: bool
    text: str


_AR_CLAUSE = re.compile(r"^[(\[]?\s*(\d+|[\u0621-\u064A])\s*[)\]\-–]")
_EN_CLAUSE = re.compile(r"^(\(?[a-z]\)|\(?[ivx]+\)|\d+\.)\s")


def units_for(blocks: list[Block], pages: dict[int, Page]) -> list[Unit]:
    units: list[Unit] = []
    for b in blocks:
        norm = pages[b.page].columns[b.lang].norm_text
        pos = b.norm_start
        prev_text = units[-1].text if units else ""
        for ln in norm[b.norm_start : b.norm_end].split("\n"):
            if ln:
                if b.lang == "ar":
                    starts = bool(_AR_CLAUSE.match(ln))
                else:
                    starts = bool(_EN_CLAUSE.match(ln)) or prev_text.endswith(
                        (".", ";", ":")
                    )
                units.append(Unit(b, pos, pos + len(ln), starts, ln))
                prev_text = ln
            pos += len(ln) + 1
    if units:
        units[0].para_start = True
        if len(units) > 1:
            units[1].para_start = False  # never split right after the header line
    return units


def paragraphs(units: list[Unit]) -> list[list[Unit]]:
    paras: list[list[Unit]] = []
    for u in units:
        if u.para_start or not paras:
            paras.append([u])
        else:
            paras[-1].append(u)
    return paras


def split_paragraphs(paras: list[list[Unit]], n: int, measure) -> list[list[Unit]]:
    """Contiguous, length-balanced partition of paragraphs into <= n parts."""
    if n <= 1 or len(paras) <= 1:
        return [[u for p in paras for u in p]]
    n = min(n, len(paras))
    sizes = [measure(" ".join(u.text for u in p)) for p in paras]
    target = sum(sizes) / n
    parts: list[list[Unit]] = [[]]
    acc = 0.0
    for i, (p, sz) in enumerate(zip(paras, sizes)):
        remaining_paras = len(paras) - i
        remaining_parts = n - len(parts)
        if parts[-1] and (acc + sz / 2 > target or remaining_paras <= remaining_parts):
            if len(parts) < n:
                parts.append([])
                acc = 0.0
        parts[-1].extend(p)
        acc += sz
    return [p for p in parts if p]


def segments_for(units: list[Unit], pages: dict[int, Page]) -> list[dict]:
    by_block: dict[str, list[Unit]] = {}
    order: list[Block] = []
    for u in units:
        if u.block.block_id not in by_block:
            by_block[u.block.block_id] = []
            order.append(u.block)
        by_block[u.block.block_id].append(u)
    segs = []
    for i, b in enumerate(order):
        us = by_block[b.block_id]
        ns, ne = us[0].ns, us[-1].ne
        rs, re_ = map_norm_span(pages[b.page].columns[b.lang].offset_map, ns, ne)
        segs.append(
            {
                "page_number": b.page,
                "block_id": b.block_id,
                "continues_from_previous_page": i > 0 and order[i - 1].page < b.page,
                "continues_to_next_page": i + 1 < len(order)
                and order[i + 1].page > b.page,
                "page_raw_start": rs,
                "page_raw_end": re_,
                "page_norm_start": ns,
                "page_norm_end": ne,
            }
        )
    return segs


def _seg_norm(seg: dict, lang: str, pages: dict[int, Page]) -> str:
    col = pages[seg["page_number"]].columns[lang]
    return col.norm_text[seg["page_norm_start"] : seg["page_norm_end"]]


def _section(path: list[dict]) -> dict | None:
    if not path:
        return None
    return {"title_en": path[-1]["title_en"], "title_ar": path[-1]["title_ar"]}


def _common(doc: dict, status: str) -> dict:
    return {
        "doc_id": DOC_ID,
        "jurisdiction": DOC_META["jurisdiction"],
        "doc_status": status,
        "effective_date": DOC_META["effective_date"],
        "source_checksum": doc["source_checksum"],
        "extraction_config_id": doc["extraction_config_id"],
    }


def article_chunks(
    articles: list[Article],
    pages: dict[int, Page],
    doc: dict,
    max_tokens: int | None,
    count_tokens,
) -> list[dict]:
    chunks = []
    for art in articles:
        units = {lang: units_for(art.blocks[lang], pages) for lang in ("ar", "en")}
        n_parts = 1
        if max_tokens:
            for lang in ("ar", "en"):
                txt = "\n".join(u.text for u in units[lang])
                if txt:
                    n_parts = max(n_parts, math.ceil(count_tokens(txt) / max_tokens))
        parts = {
            lang: split_paragraphs(paragraphs(units[lang]), n_parts, count_tokens)
            if units[lang]
            else []
            for lang in ("ar", "en")
        }
        n_final = max(len(parts["ar"]), len(parts["en"]), 1)
        for lang in ("ar", "en"):  # keep both languages on the same part count
            while parts[lang] and len(parts[lang]) < n_final:
                parts[lang].append([])
        for pi in range(n_final):
            texts = {}
            page_numbers: set[int] = set()
            for lang in ("ar", "en"):
                us = parts[lang][pi] if pi < len(parts[lang]) else []
                if not us:
                    texts[lang] = None
                    continue
                segs = segments_for(us, pages)
                page_numbers.update(s["page_number"] for s in segs)
                texts[lang] = {
                    "norm_text": "\n".join(_seg_norm(s, lang, pages) for s in segs),
                    "source_segments": segs,
                }
            chunks.append(
                {
                    "record_type": "chunk",
                    "schema_version": SCHEMA_VERSION,
                    "chunk_id": f"{DOC_ID}:article:{art.number}:part:{pi + 1}",
                    "strategy": "article-aware",
                    "article": art.number,
                    "articles": [art.number],
                    "part_name": art.part_name,
                    "book_no": art.book_no,
                    "chapter_no": art.chapter_no,
                    "section_no": art.section_no,
                    "section_title": art.section_title,
                    "section_path": art.section_path,
                    "part_index": pi + 1,
                    "part_count": n_final,
                    "page_numbers": sorted(page_numbers),
                    "parallel_text": {
                        "relationship": "side_by_side_translation",
                        "primary_language": "ar",
                        "alignment_level": "article"
                        if n_final == 1
                        else "article_part",
                    },
                    "texts": texts,
                    **_common(doc, art.status),
                }
            )
    return chunks


def fixed_size_chunks(
    pages: list[Page], articles: list[Article], doc: dict, size: int, overlap: int
) -> list[dict]:
    """ADR-0007 baseline arm: fixed windows over each language's content stream."""
    status_of = {a.number: a.status for a in articles}
    section_of = {a.number: a.section_path for a in articles}
    pmap = {p.number: p for p in pages}
    chunks = []
    for lang in ("ar", "en"):
        stream: list[tuple[int, Block]] = []  # (stream_start, block)
        text_parts: list[str] = []
        pos = 0
        for p in pages:
            for b in p.columns.get(lang, Column()).blocks:
                if b.role not in ("article_start", "continuation", "repealed_range"):
                    continue
                if text_parts:
                    text_parts.append("\n")
                    pos += 1
                t = p.columns[lang].norm_text[b.norm_start : b.norm_end]
                stream.append((pos, b))
                text_parts.append(t)
                pos += len(t)
        text = "".join(text_parts)
        starts = [s for s, _ in stream]
        start, idx = 0, 0
        while start < len(text):
            end = min(start + size, len(text))
            if end < len(text):
                cut = text.rfind(" ", start + int(size * 0.8), end)
                cut = max(cut, text.rfind("\n", start + int(size * 0.8), end))
                if cut > start:
                    end = cut
            s, e = start, end
            while s < e and text[s].isspace():
                s += 1
            while e > s and text[e - 1].isspace():
                e -= 1
            if e > s:
                segs, arts = [], []
                bi = bisect.bisect_right(starts, s) - 1
                while bi < len(stream) and stream[bi][0] < e:
                    bstart, b = stream[bi]
                    blen = b.norm_end - b.norm_start
                    ls, le = max(s, bstart) - bstart, min(e, bstart + blen) - bstart
                    if le > ls:
                        ns, ne = b.norm_start + ls, b.norm_start + le
                        omap = pmap[b.page].columns[lang].offset_map
                        rs, re_ = map_norm_span(omap, ns, ne)
                        segs.append(
                            {
                                "page_number": b.page,
                                "block_id": b.block_id,
                                "page_raw_start": rs,
                                "page_raw_end": re_,
                                "page_norm_start": ns,
                                "page_norm_end": ne,
                            }
                        )
                        if b.article and b.article not in arts:
                            arts.append(b.article)
                    bi += 1
                for i, sg in enumerate(segs):
                    sg["continues_from_previous_page"] = (
                        i > 0 and segs[i - 1]["page_number"] < sg["page_number"]
                    )
                    sg["continues_to_next_page"] = (
                        i + 1 < len(segs)
                        and segs[i + 1]["page_number"] > sg["page_number"]
                    )
                idx += 1
                statuses = {status_of.get(a, "in_force") for a in arts}
                status = "repealed" if statuses == {"repealed"} else "in_force"
                path = section_of.get(arts[0], []) if arts else []
                chunks.append(
                    {
                        "record_type": "chunk",
                        "schema_version": SCHEMA_VERSION,
                        "chunk_id": f"{DOC_ID}:fixed-size:{lang}:"
                        f"s{size}-o{overlap}:{idx:05d}",
                        "strategy": "fixed-size",
                        "article": arts[0] if len(arts) == 1 else None,
                        "articles": arts,
                        "section": _section(path),
                        "section_path": path,
                        "part_index": idx,
                        "page_numbers": sorted({sg["page_number"] for sg in segs}),
                        "parallel_text": None,
                        "texts": {
                            lang: {
                                "norm_text": "\n".join(
                                    _seg_norm(sg, lang, pmap) for sg in segs
                                ),
                                "source_segments": segs,
                            }
                        },
                        **_common(doc, status),
                    }
                )
            if end >= len(text):
                break
            nxt = max(end - overlap, start + 1)
            ws = text.find(" ", nxt)  # do not start mid-word
            start = ws + 1 if 0 <= ws < end else nxt
    return chunks


# --------------------------------------------------------------------------------------
# Measurements (ADR-0006 CER, ADR-0007 token distribution / detection rate)
# --------------------------------------------------------------------------------------
def load_tokenizer(name: str | None):
    if not name:
        return None
    try:
        from transformers import AutoTokenizer  # noqa: PLC0415

        return AutoTokenizer.from_pretrained(name, local_files_only=True)
    except Exception:  # noqa: BLE001 - absence is reported, not fatal
        # print(f"[warn] tokenizer {name!r} unavailable locally: {type(e).__name__}")
        return None


def distribution(values: list[int]) -> dict:
    if not values:
        return {}
    v = sorted(values)

    def pct(p: float) -> int:
        return v[min(len(v) - 1, int(round(p * (len(v) - 1))))]

    return {
        "n": len(v),
        "min": v[0],
        "p50": pct(0.5),
        "p95": pct(0.95),
        "max": v[-1],
        "mean": round(statistics.fmean(v), 1),
    }


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def measure_cer(ref_dir: Path | None, pages: dict[int, Page]) -> dict:
    """CER of raw_text vs hand transcriptions named p<NNN>_<ar|en>.txt (ADR-0006)."""
    if ref_dir is None:
        return {"status": "unmeasured", "reason": "no --cer-reference-dir given"}
    results = []
    for f in sorted(ref_dir.glob("p*_*.txt")):
        m = re.match(r"p(\d+)_(ar|en)\.txt$", f.name)
        if not m or int(m.group(1)) not in pages:
            continue
        pno, lang = int(m.group(1)), m.group(2)
        ref = re.sub(r"\s+", " ", f.read_text(encoding="utf-8")).strip()
        hyp = re.sub(r"\s+", " ", pages[pno].columns[lang].raw_text).strip()
        cer = levenshtein(ref, hyp) / max(1, len(ref))
        results.append({"page": pno, "lang": lang, "cer": round(cer, 4)})
    n_pages = len({r["page"] for r in results})
    return {
        "status": "measured" if n_pages >= 10 else "insufficient_sample",
        "pages_measured": n_pages,
        "required_pages": 10,
        "mean_cer": round(statistics.fmean(r["cer"] for r in results), 4)
        if results
        else None,
        "per_page": results,
    }


# --------------------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------------------
_PRES_RE = re.compile("[\ufb50-\ufdff\ufe70-\ufefc]")
_TASHKEEL_RE = re.compile("[\u064b-\u0652\u0670]")
_INDIC_RE = re.compile("[\u0660-\u0669\u06f0-\u06f9]")


def _letters(text: str) -> tuple[int, int]:
    ar = sum(1 for c in text if _bidi(c) == "R")
    la = sum(1 for c in text if c.isascii() and c.isalpha())
    return ar, la


def validate(
    pages: list[Page],
    articles: list[Article],
    chunks: list[dict],
    cfg: NormConfig,
    issues: Issues,
    expected_last: int,
) -> dict:
    pmap = {p.number: p for p in pages}

    # All article numbers covered by the corpus, used so 4-digit article
    # numbers (e.g. 1002) are not mistaken for reversed years during validation.
    article_nums: set[str] = set()
    for a in articles:
        lo, _, hi = a.number.partition("-")
        for n in range(int(lo), int(hi or lo) + 1):
            article_nums.add(str(n))

    # 1. offset maps + per-block round trip (ADR-0006: a test, not an aspiration)
    blocks_checked = 0
    for p in pages:
        for lang, col in p.columns.items():
            om = col.offset_map
            if col.norm_text:
                if om[0][0] != 0 or om[-1][1] != len(col.norm_text):
                    issues.err("offset_map_coverage", page=p.number, lang=lang)
                for a, b in zip(om, om[1:]):
                    if b[0] != a[1] or b[2] < a[3]:
                        issues.err("offset_map_not_monotonic", page=p.number, lang=lang)
                        break
            for b in col.blocks:
                blocks_checked += 1
                raw = col.raw_text[b.raw_start : b.raw_end]
                norm = col.norm_text[b.norm_start : b.norm_end]
                if raw != b.raw_text or normalize(raw, cfg) != norm:
                    issues.err("block_round_trip", block=b.block_id)
                elif norm and map_norm_span(om, b.norm_start, b.norm_end) != (
                    b.raw_start,
                    b.raw_end,
                ):
                    issues.err("block_offset_map", block=b.block_id)
                if b.role == "unassigned":
                    issues.err("unassigned_block", block=b.block_id)

    # 2. article sequence / detection completeness (ADR-0007)
    seen: list[tuple[int, int]] = []
    for a in articles:
        lo, _, hi = a.number.partition("-")
        seen.append((int(lo), int(hi or lo)))
    expected = 1
    missing, out_of_order = [], []
    for lo, hi in seen:
        if lo > expected:
            missing.extend(range(expected, lo))
        elif lo < expected:
            out_of_order.append(lo)
        expected = max(expected, hi + 1)
    if expected - 1 < expected_last:
        missing.extend(range(expected, expected_last + 1))
    for n in missing:
        issues.err("missing_article", article=n)
    for n in out_of_order:
        issues.err("duplicate_or_out_of_order_article", article=n)
    covered = sum(hi - lo + 1 for lo, hi in seen)

    # 3. chunk-level checks
    ids = Counter(c["chunk_id"] for c in chunks)
    for cid, n in ids.items():
        if n > 1:
            issues.err("duplicate_chunk_id", chunk_id=cid)
    bidi_reversed, years_ok, years_bad = 0, 0, 0
    for c in chunks:
        if not c.get("section_title", {}).get("en") and not c.get(
            "section_title", {}
        ).get("ar"):
            issues.warn("chunk_without_section", chunk_id=c["chunk_id"])
        for lang in ("ar", "en"):
            t = c["texts"].get(lang)
            if t is None:
                if c["doc_status"] != "repealed":
                    issues.warn(
                        "missing_language_text", chunk_id=c["chunk_id"], lang=lang
                    )
                continue
            segs = t["source_segments"]
            rebuilt = []
            for i, s in enumerate(segs):
                col = pmap[s["page_number"]].columns[lang]
                ns, ne = s["page_norm_start"], s["page_norm_end"]
                if map_norm_span(col.offset_map, ns, ne) != (
                    s["page_raw_start"],
                    s["page_raw_end"],
                ):
                    issues.err("segment_offset_map", chunk_id=c["chunk_id"], lang=lang)
                raw = col.raw_text[s["page_raw_start"] : s["page_raw_end"]]
                if normalize(raw, cfg) != col.norm_text[ns:ne]:
                    issues.err("segment_round_trip", chunk_id=c["chunk_id"], lang=lang)
                rebuilt.append(col.norm_text[ns:ne])
                if i + 1 < len(segs):
                    nxt = segs[i + 1]
                    if (
                        s["continues_to_next_page"]
                        != nxt["continues_from_previous_page"]
                    ):
                        issues.err("continuity_flags", chunk_id=c["chunk_id"])
                    if nxt["page_number"] - s["page_number"] > 1:
                        issues.err("non_adjacent_pages", chunk_id=c["chunk_id"])
            norm = t["norm_text"]
            if norm != "\n".join(rebuilt):
                issues.err("chunk_text_mismatch", chunk_id=c["chunk_id"], lang=lang)
            if len(norm) < 15 and c["doc_status"] != "repealed":
                issues.warn("tiny_chunk", chunk_id=c["chunk_id"], lang=lang, text=norm)
            ar, la = _letters(norm)
            letters = max(1, ar + la)
            if lang == "ar" and ar / letters < 0.9:
                issues.err(
                    "ar_text_not_arabic", chunk_id=c["chunk_id"], ratio=ar / letters
                )
            if lang == "en" and la / letters < 0.98:
                issues.err(
                    "en_text_not_latin", chunk_id=c["chunk_id"], ratio=la / letters
                )
            if lang == "ar":
                words = re.findall(r"[\u0621-\u064A]+", norm)
                starts = sum(w.startswith("ال") for w in words)
                ends = sum(w.endswith("لا") and len(w) > 3 for w in words)
                # A genuine whole-chunk visual-order reversal flips *many*
                # definite articles; 1-2 words ending in -الا/ولا is normal
                # Arabic accusative grammar, not a bidi bug.
                if ends >= 3 and ends > starts:
                    bidi_reversed += 1
                    issues.err("ar_visual_order_suspected", chunk_id=c["chunk_id"])
                for y in re.findall(r"(?<!\d)\d{4}(?!\d)", norm):
                    if y in article_nums:
                        continue
                    if y.startswith(("18", "19", "20")):
                        years_ok += 1
                    elif y[::-1].startswith(("18", "19", "20")):
                        years_bad += 1
                if cfg.decompose_presentation_forms and _PRES_RE.search(norm):
                    issues.err("presentation_forms_left", chunk_id=c["chunk_id"])
                if cfg.strip_tatweel and "\u0640" in norm:
                    issues.err("tatweel_left", chunk_id=c["chunk_id"])
                if cfg.strip_tashkeel and _TASHKEEL_RE.search(norm):
                    issues.err("tashkeel_left", chunk_id=c["chunk_id"])
                if cfg.fold_digits and _INDIC_RE.search(norm):
                    issues.err("arabic_indic_digits_left", chunk_id=c["chunk_id"])
    if years_bad > max(2, 0.05 * (years_ok + years_bad)):
        issues.err("digit_order_suspected", ok=years_ok, reversed=years_bad)

    total = sum(p.chars_total for p in pages)
    in_tables = sum(p.chars_in_tables for p in pages)
    ocr = [p.number for p in pages if p.needs_ocr]
    for p in pages:
        for ob in p.outside:
            issues.warn(
                "text_outside_tables_not_indexed",
                page=p.number,
                chars=len(ob["raw_text"]),
                preview=ob["raw_text"][:60],
            )
    return {
        "blocks_round_tripped": blocks_checked,
        "articles_detected": len(articles),
        "article_numbers_covered": covered,
        "expected_last_article": expected_last,
        "article_detection_rate_vs_sequence": round(covered / expected_last, 4),
        "missing_articles": missing,
        "bidi_reversed_chunks": bidi_reversed,
        "four_digit_years": {"plausible": years_ok, "reversed": years_bad},
        "corpus_coverage": {
            "pages_total": len(pages),
            "pages_needs_ocr": ocr,
            "pages_indexed": len(pages) - len(ocr),
            "page_coverage": round((len(pages) - len(ocr)) / max(1, len(pages)), 4),
            "chars_total": total,
            "chars_in_tables": in_tables,
            "char_coverage_tables": round(in_tables / max(1, total), 4),
        },
    }


def detection_rate_vs_hand_count(path: Path | None, articles: list[Article]) -> dict:
    """ADR-0007: article detection rate against a hand-counted page sample.

    The JSON file maps page number -> number of article starts counted by hand.
    """
    if path is None:
        return {"status": "unmeasured", "reason": "no --article-count-reference given"}
    ref = {int(k): int(v) for k, v in json.loads(path.read_text()).items()}
    got = Counter(a.first_page for a in articles)
    detected = sum(min(got.get(p, 0), n) for p, n in ref.items())
    return {
        "status": "measured",
        "pages": len(ref),
        "hand_counted": sum(ref.values()),
        "detected": detected,
        "rate": round(detected / max(1, sum(ref.values())), 4),
        "per_page": {p: {"hand": n, "detected": got.get(p, 0)} for p, n in ref.items()},
    }


# --------------------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def sha256_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_jsonl(path: Path, records) -> int:
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=False) + "\n")
            n += 1
    return n


def page_record(p: Page) -> dict:
    cols = {}
    for lang, col in p.columns.items():
        cols[lang] = {
            "raw_text": col.raw_text,
            "norm_text": col.norm_text,
            "raw_checksum": sha256_text(col.raw_text),
            "norm_to_raw_offset_map": col.offset_map,
            "blocks": [
                {
                    "block_id": b.block_id,
                    "role": b.role,
                    "article": b.article,
                    "row_index": b.row_index,
                    "bbox": b.bbox,
                    "raw_start": b.raw_start,
                    "raw_end": b.raw_end,
                    "norm_start": b.norm_start,
                    "norm_end": b.norm_end,
                }
                for b in col.blocks
            ],
        }
    return {
        "record_type": "page",
        "schema_version": SCHEMA_VERSION,
        "doc_id": DOC_ID,
        "page_number": p.number,
        "text_source": "native_pdf",
        "needs_ocr": p.needs_ocr,
        "has_images": p.has_images,
        "offset_map_format": "[norm_start, norm_end, raw_start, raw_end]; "
        "linear (1:1) iff norm_len == raw_len, else atomic",
        "columns": cols,
        "outside_table": p.outside,
        "stats": {"chars_total": p.chars_total, "chars_in_tables": p.chars_in_tables},
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pdf", type=Path, default=HERE / "Egyptian_Civil_Law.pdf")
    ap.add_argument("--out-dir", type=Path, default=HERE / "processed")
    ap.add_argument(
        "--strategy",
        choices=["article-aware", "fixed-size", "both"],
        default="article-aware",
    )
    ap.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="ADR-0007 article cap (needs --tokenizer); default: no split",
    )
    ap.add_argument(
        "--tokenizer",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="token-length distribution + cap unit (local files only)",
    )
    ap.add_argument("--embedding-tokenizer", default="BAAI/bge-m3")
    ap.add_argument("--embedding-max-seq", type=int, default=8192)
    ap.add_argument("--fixed-size-chars", type=int, default=1200)
    ap.add_argument("--fixed-overlap-chars", type=int, default=200)
    ap.add_argument("--expected-last-article", type=int, default=EXPECTED_LAST_ARTICLE)
    ap.add_argument("--cer-reference-dir", type=Path, default=None)
    ap.add_argument("--article-count-reference", type=Path, default=None)
    for rule in NormConfig.__dataclass_fields__:
        ap.add_argument(
            f"--no-{rule.replace('_', '-')}",
            action="store_true",
            help=f"disable ADR-0006 rule: {rule}",
        )
    ap.add_argument(
        "--allow-errors",
        action="store_true",
        help="exit 0 even if validation errors are found",
    )
    args = ap.parse_args(argv)

    cfg = NormConfig(
        **{r: not getattr(args, f"no_{r}") for r in NormConfig.__dataclass_fields__}
    )
    tok = load_tokenizer(args.tokenizer)
    emb_tok = load_tokenizer(args.embedding_tokenizer)
    if args.max_tokens and tok is None:
        print("[error] --max-tokens requires a locally available --tokenizer")
        return 2

    def count_tokens(text: str) -> int:
        return len(tok.encode(text, add_special_tokens=False)) if tok else len(text)

    source_checksum = sha256_file(args.pdf)
    pdf = pymupdf.open(args.pdf)
    config = {
        "pipeline_version": PIPELINE_VERSION,
        "source_checksum": source_checksum,
        "extraction": {
            "tool": "pymupdf",
            "tool_version": pymupdf.VersionBind,
            "table_detection": "page.find_tables() (ruling rectangles)",
            "line_baseline_tolerance_pt": 2.0,
            "bidi_repair": [
                "geometric_rtl_order",
                "lam_alef_ligature_order",
                "ltr_run_reversal",
            ],
        },
        "normalization": asdict(cfg),
        "chunking": {
            "strategy": args.strategy,
            "article_aware": {
                "boundary": "table_row + article marker (Article N | مادة (N))",
                "max_tokens": args.max_tokens,
                "tokenizer": args.tokenizer if args.max_tokens else None,
                "split_at": "paragraph/clause boundaries",
                "packing": False,
            },
            "fixed_size": {
                "unit": "norm_chars",
                "size": args.fixed_size_chars,
                "overlap": args.fixed_overlap_chars,
            }
            if args.strategy in ("fixed-size", "both")
            else None,
        },
    }
    config_id = sha256_text(json.dumps(config, sort_keys=True, ensure_ascii=False))

    stats = LineStats()
    pages: list[Page] = []
    for i, page in enumerate(pdf, start=1):
        p = extract_page(page, i, stats)
        assemble_columns(p, cfg)
        pages.append(p)
    pmap = {p.number: p for p in pages}

    issues = Issues()
    articles = assemble_articles(pages, issues)
    doc = {
        "record_type": "document",
        "schema_version": SCHEMA_VERSION,
        "doc_id": DOC_ID,
        **DOC_META,
        "source_path": str(args.pdf.relative_to(HERE.parent.parent))
        if args.pdf.is_relative_to(HERE.parent.parent)
        else str(args.pdf),
        "source_checksum": source_checksum,
        "corpus_revision": f"{DOC_ID}@{source_checksum.split(':')[1][:12]}",
        "page_count": pdf.page_count,
        "extraction_config_id": config_id,
        "run_config": config,
    }

    chunks: list[dict] = []
    if args.strategy in ("article-aware", "both"):
        chunks = article_chunks(articles, pmap, doc, args.max_tokens, count_tokens)
    fixed: list[dict] = []
    if args.strategy in ("fixed-size", "both"):
        fixed = fixed_size_chunks(
            pages, articles, doc, args.fixed_size_chars, args.fixed_overlap_chars
        )

    summary = validate(
        pages, articles, chunks + fixed, cfg, issues, args.expected_last_article
    )

    # ADR-0007 measured prerequisite: article length distribution.
    lengths: dict[str, dict] = {}
    for lang in ("ar", "en"):
        texts = [
            c["texts"][lang]["norm_text"]
            for c in chunks
            if c["texts"].get(lang) and c["doc_status"] != "repealed"
        ]
        lengths[lang] = {"chars": distribution([len(t) for t in texts])}
        if tok:
            lengths[lang][f"tokens[{args.tokenizer}]"] = distribution(
                [len(tok.encode(t, add_special_tokens=False)) for t in texts]
            )
        if emb_tok:
            emb = [len(emb_tok.encode(t)) for t in texts]
            lengths[lang][f"tokens[{args.embedding_tokenizer}]"] = distribution(emb)
            over = sum(n > args.embedding_max_seq for n in emb)
            if over:
                issues.err("exceeds_embedding_max_seq", lang=lang, chunks=over)
    token_status = (
        "measured"
        if tok
        else (
            "unmeasured: tokenizer not available locally; char counts only "
            "(ADR-0007 requires the Qwen2.5 token distribution before choosing caps)"
        )
    )

    doc["extraction"] = {
        "cer": measure_cer(args.cer_reference_dir, pmap),
        "article_detection_rate": detection_rate_vs_hand_count(
            args.article_count_reference, articles
        ),
        "corpus_coverage": summary["corpus_coverage"],
        "bidi_repairs": asdict(stats),
        "corpus_gaps": {
            "pages_needs_ocr": summary["corpus_coverage"]["pages_needs_ocr"],
            "text_outside_tables_not_indexed": [
                {"page": p.number, "chars": len(o["raw_text"])}
                for p in pages
                for o in p.outside
            ],
        },
    }
    doc["article_length_distribution"] = {"status": token_status, **lengths}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir
    (out / "document.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_jsonl(out / "pages.jsonl", (page_record(p) for p in pages))
    if chunks:
        write_jsonl(out / "chunks.article_aware.jsonl", chunks)
    if fixed:
        write_jsonl(out / "chunks.fixed_size.jsonl", fixed)
    write_jsonl(
        out / "source_to_chunk.jsonl",
        (
            {
                "chunk_id": c["chunk_id"],
                "lang": lang,
                "article": c["article"],
                "doc_status": c["doc_status"],
                "page_number": s["page_number"],
                "block_id": s["block_id"],
                "raw_start": s["page_raw_start"],
                "raw_end": s["page_raw_end"],
                "norm_start": s["page_norm_start"],
                "norm_end": s["page_norm_end"],
            }
            for c in chunks + fixed
            for lang, t in c["texts"].items()
            if t
            for s in t["source_segments"]
        ),
    )
    report = {
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "extraction_config_id": config_id,
        "summary": summary,
        "error_count": len(issues.errors),
        "warning_count": len(issues.warnings),
        "error_codes": dict(Counter(e["code"] for e in issues.errors)),
        "warning_codes": dict(Counter(w["code"] for w in issues.warnings)),
        "errors": issues.errors,
        "warnings": issues.warnings,
    }
    (out / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(
        f"pages: {len(pages)}  articles: {len(articles)}  "
        f"chunks(article-aware): {len(chunks)}  chunks(fixed-size): {len(fixed)}"
    )
    print(
        f"articles covered: {summary['article_numbers_covered']}/"
        f"{args.expected_last_article}  missing: {summary['missing_articles'][:20]}"
    )
    print(f"coverage: {summary['corpus_coverage']}")
    print(f"bidi repairs: {asdict(stats)}")
    print(f"length distribution: {json.dumps(lengths, ensure_ascii=False)}")
    print(f"token distribution: {token_status}")
    print(f"CER: {doc['extraction']['cer']['status']}")
    print(f"errors: {report['error_codes']}")
    print(f"warnings: {report['warning_codes']}")
    print(f"outputs -> {out}")
    if issues.errors and not args.allow_errors:
        print("VALIDATION FAILED (see validation_report.json)")
        return 1
    print("VALIDATION PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
