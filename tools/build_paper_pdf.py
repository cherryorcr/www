from __future__ import annotations

import html
import hashlib
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib.font_manager import FontProperties
from matplotlib.mathtext import math_to_image
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    LongTable,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ.get(
    "PAPER_SOURCE",
    str(ROOT / "docs" / "paper" / "TAG_CONTINUAL_CONDENSATION_DRAFT_ZH_2026-09-16.md"),
))
OUTPUT = Path(os.environ.get(
    "PAPER_OUTPUT",
    str(ROOT / "output" / "pdf" / "TAG_CONTINUAL_CONDENSATION_DRAFT_ZH_2026-09-16.pdf"),
))
PAGE_HEADER = os.environ.get(
    "PAPER_HEADER",
    "T+degree - 块保持持续 TAG 凝聚论文草稿 | 2026-09-20",
)
FONT_DIR = Path("C:/Windows/Fonts")
MATH_DIR = ROOT / "tmp" / "pdfs" / "paper_export" / "math"
MATH_DPI = 300
matplotlib.rcParams["mathtext.fontset"] = "stix"


def register_fonts() -> tuple[str, str, str]:
    body_path = FONT_DIR / "simhei.ttf"
    bold_path = FONT_DIR / "simsunb.ttf"
    mono_path = FONT_DIR / "simsun.ttc"
    if not body_path.exists():
        raise FileNotFoundError(f"Chinese font not found: {body_path}")
    pdfmetrics.registerFont(TTFont("ZHBody", str(body_path)))
    # SimHei is reliably readable for Chinese glyphs in both regular and heading text.
    # Some Windows bold font collections expose only a partial cmap to ReportLab.
    pdfmetrics.registerFont(TTFont("ZHBold", str(body_path)))
    pdfmetrics.registerFontFamily(
        "ZHBody", normal="ZHBody", bold="ZHBold", italic="ZHBody", boldItalic="ZHBold"
    )
    if mono_path.exists():
        pdfmetrics.registerFont(TTFont("ZHMono", str(mono_path), subfontIndex=0))
    else:
        pdfmetrics.registerFont(TTFont("ZHMono", str(body_path)))
    return "ZHBody", "ZHBold", "ZHMono"


BODY_FONT, BOLD_FONT, MONO_FONT = register_fonts()


def normalize_text(value: str) -> str:
    replacements = {
        "\u2013": "-",
        "\u2014": "-",
        "\u2011": "-",
        "\u2212": "-",
        "\u00a0": " ",
    }
    for old, new in replacements.items():
        value = value.replace(old, new)
    return value


def inline_markup(value: str) -> str:
    value = normalize_text(value)
    tokens: list[str] = []

    def protect(markup: str) -> str:
        tokens.append(markup)
        return f"PDFTOKEN{len(tokens) - 1}END"

    def link(match):
        label, target = match.groups()
        if not re.match(r"https?://", target):
            target = (SOURCE.parent / target).resolve().as_uri()
        return protect(
            f'<link href="{html.escape(target, quote=True)}" color="#315A72">'
            f'{html.escape(label)}</link>'
        )

    value = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", link, value)
    value = re.sub(
        r"`([^`]+)`",
        lambda match: protect(f"<font name='ZHMono'>{html.escape(match[1])}</font>"),
        value,
    )
    # The source uses parenthesized TeX rather than dollar-delimited inline math.
    index, parts = 0, []
    while index < len(value):
        if value[index] != "(":
            parts.append(value[index])
            index += 1
            continue
        end, depth = index + 1, 1
        while end < len(value) and depth:
            depth += (value[end] == "(") - (value[end] == ")")
            end += 1
        expr = value[index:end]
        if depth == 0 and re.search(r"[\\_^]", expr) and not re.search(r"[\u4e00-\u9fff]", expr):
            power = re.match(r"\^\{[^}]+\}", value[end:])
            if power:
                expr += power[0]
                end += len(power[0])
            asset, width, height = math_asset(expr, 9)
            parts.append(protect(
                f'<img src="{asset.as_posix()}" width="{width:.3f}" '
                f'height="{height:.3f}" valign="middle"/>'
            ))
        else:
            parts.append(expr)
        index = end
    value = "".join(parts)
    value = html.escape(value, quote=False)
    value = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", value)
    value = re.sub(r"PDFTOKEN(\d+)END", lambda match: tokens[int(match[1])], value)
    return value


def math_asset(expression: str, size: float):
    expression = " ".join(expression.split())
    expression = expression.replace("\\mathcal", "")
    expression = re.sub(r"\\operatorname\{([^{}]*)\}", r"\1", expression)
    expression = re.sub(r"\\mathrm\{([^{}]*)\}", r"\1", expression)
    expression = expression.replace("\\mathrm ", "")
    expression = re.sub(r"\\ge(?![A-Za-z])", r"\\geq", expression)
    expression = re.sub(r"\\le(?![A-Za-z])", r"\\leq", expression)
    key = hashlib.sha256(f"{size}:{expression}".encode()).hexdigest()[:20]
    MATH_DIR.mkdir(parents=True, exist_ok=True)
    asset = MATH_DIR / f"{key}.png"
    if not asset.exists():
        math_to_image(
            f"${expression}$", str(asset), prop=FontProperties(size=size),
            dpi=MATH_DPI, format="png", color="#252525",
        )
    with PILImage.open(asset) as bitmap:
        width, height = bitmap.size
    return asset, width * 72 / MATH_DPI, height * 72 / MATH_DPI


def make_formula(expression: str, available_width: float):
    asset, width, height = math_asset(expression, 11)
    scale = min(1, (available_width - 12 * mm) / width)
    return Image(str(asset), width=width * scale, height=height * scale, hAlign="CENTER")


styles = getSampleStyleSheet()
styles.add(
    ParagraphStyle(
        name="ZHTitle",
        fontName=BOLD_FONT,
        fontSize=20,
        leading=25,
        alignment=TA_LEFT,
        textColor=colors.HexColor("#222222"),
        spaceAfter=7 * mm,
    )
)
styles.add(
    ParagraphStyle(
        name="ZHSubtitle",
        fontName=BODY_FONT,
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#666666"),
        spaceAfter=4 * mm,
    )
)
styles.add(
    ParagraphStyle(
        name="ZHSection",
        fontName=BOLD_FONT,
        fontSize=15,
        leading=19,
        textColor=colors.HexColor("#292929"),
        spaceBefore=7 * mm,
        spaceAfter=3 * mm,
        keepWithNext=True,
    )
)
styles.add(
    ParagraphStyle(
        name="ZHSubsection",
        fontName=BOLD_FONT,
        fontSize=11.5,
        leading=15,
        textColor=colors.HexColor("#333333"),
        spaceBefore=4 * mm,
        spaceAfter=2 * mm,
        keepWithNext=True,
    )
)
styles.add(
    ParagraphStyle(
        name="ZHBody",
        fontName=BODY_FONT,
        fontSize=9.2,
        leading=15.2,
        splitLongWords=True,
        alignment=TA_LEFT,
        textColor=colors.HexColor("#333333"),
        spaceAfter=3.2 * mm,
        firstLineIndent=0,
    )
)
styles.add(
    ParagraphStyle(
        name="ZHAbstract",
        parent=styles["ZHBody"],
        fontSize=9.4,
        leading=15.6,
        leftIndent=2 * mm,
        rightIndent=2 * mm,
        spaceAfter=3.5 * mm,
    )
)
styles.add(
    ParagraphStyle(
        name="ZHList",
        parent=styles["ZHBody"],
        leftIndent=6 * mm,
        firstLineIndent=-4 * mm,
        spaceAfter=1.6 * mm,
    )
)
styles.add(
    ParagraphStyle(
        name="ZHCaption",
        fontName=BODY_FONT,
        fontSize=8.2,
        leading=11,
        alignment=TA_LEFT,
        textColor=colors.HexColor("#666666"),
        spaceBefore=1 * mm,
        spaceAfter=4 * mm,
    )
)
styles.add(
    ParagraphStyle(
        name="ZHFormula",
        fontName=MONO_FONT,
        fontSize=8.3,
        leading=12,
        leftIndent=8 * mm,
        rightIndent=8 * mm,
        textColor=colors.HexColor("#333333"),
        backColor=colors.HexColor("#F7F7F7"),
        borderColor=colors.HexColor("#E5E5E5"),
        borderWidth=0.4,
        borderPadding=4,
        spaceBefore=1.5 * mm,
        spaceAfter=3 * mm,
    )
)
styles.add(
    ParagraphStyle(
        name="ZHCode",
        fontName=MONO_FONT,
        fontSize=7.2,
        leading=9.4,
        leftIndent=5 * mm,
        rightIndent=5 * mm,
        textColor=colors.HexColor("#333333"),
        backColor=colors.HexColor("#F5F6F7"),
        borderColor=colors.HexColor("#E0E0E0"),
        borderWidth=0.4,
        borderPadding=5,
        spaceBefore=2 * mm,
        spaceAfter=4 * mm,
    )
)
styles.add(
    ParagraphStyle(
        name="ZHTableCell",
        fontName=BODY_FONT,
        fontSize=7.6,
        leading=10.5,
        splitLongWords=True,
        textColor=colors.HexColor("#333333"),
    )
)
styles.add(
    ParagraphStyle(
        name="ZHTableHead",
        fontName=BOLD_FONT,
        fontSize=7.6,
        leading=10.5,
        splitLongWords=True,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#242424"),
    )
)


def split_table_row(line: str) -> list[str]:
    return [part.strip() for part in line.strip().strip("|").split("|")]


def is_separator(line: str) -> bool:
    cells = split_table_row(line)
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) for cell in cells)


def make_table(lines: list[str], available_width: float):
    rows = [split_table_row(line) for line in lines]
    if len(rows) < 2:
        return None
    header = rows[0]
    body = rows[2:] if is_separator(lines[1]) else rows[1:]
    ncol = max(len(row) for row in rows)
    normalized = []
    for row in [header] + body:
        row = row + [""] * (ncol - len(row))
        normalized.append(row[:ncol])

    desired = []
    for col in range(ncol):
        plain = [re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", row[col]) for row in normalized]
        measured = [pdfmetrics.stringWidth(text.replace("**", "").replace("`", ""), BODY_FONT, 7.6) for text in plain]
        desired.append(min(280, max(45, max(measured) + 10)))
    minimum = min(48, available_width / ncol)
    extra = available_width - ncol * minimum
    weights = [max(1, width - minimum) for width in desired]
    widths = [minimum + extra * weight / sum(weights) for weight in weights]

    table_data = []
    for ridx, row in enumerate(normalized):
        table_data.append(
            [
                Paragraph(inline_markup(cell), styles["ZHTableHead" if ridx == 0 else "ZHTableCell"])
                for cell in row
            ]
        )
    table = LongTable(table_data, colWidths=widths, repeatRows=1, hAlign="LEFT", splitByRow=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F1F3F4")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#222222")),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D4D8DC")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#FAFAFA")]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def make_image(path: Path, available_width: float):
    with PILImage.open(path) as image:
        width, height = image.size
    display_width = min(available_width, 174 * mm)
    display_height = display_width * height / width
    max_height = 92 * mm
    if display_height > max_height:
        display_height = max_height
        display_width = display_height * width / height
    return Image(str(path), width=display_width, height=display_height, hAlign="LEFT")


def parse_markdown(path: Path, available_width: float):
    lines = path.read_text(encoding="utf-8").splitlines()
    story = []
    index = 0
    paragraph_buffer: list[str] = []
    first_heading = True

    def flush_paragraph():
        nonlocal paragraph_buffer
        if not paragraph_buffer:
            return
        text = " ".join(part.strip() for part in paragraph_buffer).strip()
        paragraph_buffer = []
        if not text:
            return
        style = styles["ZHAbstract"] if story and any("摘要" in str(item) for item in story[-2:]) else styles["ZHBody"]
        story.append(Paragraph(inline_markup(text), style))

    while index < len(lines):
        raw = lines[index]
        stripped = raw.strip()

        if not stripped:
            flush_paragraph()
            index += 1
            continue

        if stripped.startswith("```"):
            flush_paragraph()
            index += 1
            code_lines = []
            while index < len(lines) and not lines[index].strip().startswith("```"):
                code_lines.append(normalize_text(lines[index]))
                index += 1
            if index < len(lines):
                index += 1
            story.append(Preformatted("\n".join(code_lines), styles["ZHCode"]))
            continue

        image_match = re.match(r"!\[([^]]*)\]\(([^)]+)\)", stripped)
        if image_match:
            flush_paragraph()
            alt, relpath = image_match.groups()
            image_path = (path.parent / relpath).resolve()
            if image_path.exists():
                story.append(Spacer(1, 2 * mm))
                story.append(make_image(image_path, available_width))
                story.append(Paragraph(inline_markup(alt), styles["ZHCaption"]))
            else:
                story.append(Paragraph(inline_markup(f"[图像缺失：{relpath}]"), styles["ZHCaption"]))
            index += 1
            continue

        if stripped.startswith("|") and index + 1 < len(lines) and lines[index + 1].strip().startswith("|") and is_separator(lines[index + 1]):
            flush_paragraph()
            table_lines = [stripped, lines[index + 1].strip()]
            index += 2
            while index < len(lines) and lines[index].strip().startswith("|"):
                table_lines.append(lines[index].strip())
                index += 1
            story.append(Spacer(1, 1.5 * mm))
            story.append(make_table(table_lines, available_width))
            story.append(Spacer(1, 2 * mm))
            continue

        heading = re.match(r"^(#{1,4})\s+(.+)$", stripped)
        if heading:
            flush_paragraph()
            level, title = len(heading.group(1)), heading.group(2)
            if level == 1:
                story.append(Paragraph(inline_markup(title), styles["ZHTitle"]))
                first_heading = False
            elif level == 2:
                story.append(Paragraph(inline_markup(title), styles["ZHSection"]))
            else:
                story.append(Paragraph(inline_markup(title), styles["ZHSubsection"]))
            index += 1
            continue

        if stripped.startswith("\\[") or stripped == "\\]":
            flush_paragraph()
            formula_lines = [stripped]
            if not stripped.endswith("\\]"):
                index += 1
                while index < len(lines):
                    formula_lines.append(lines[index].strip())
                    if lines[index].strip().endswith("\\]"):
                        index += 1
                        break
                    index += 1
            else:
                index += 1
            formula = " ".join(formula_lines).removeprefix("\\[").removesuffix("\\]").strip()
            story.append(Spacer(1, 2 * mm))
            story.append(make_formula(formula, available_width))
            story.append(Spacer(1, 3 * mm))
            continue

        if re.match(r"^(?:[-*])\s+", stripped):
            flush_paragraph()
            text = re.sub(r"^(?:[-*])\s+", "- ", stripped)
            story.append(Paragraph(inline_markup(text), styles["ZHList"]))
            index += 1
            continue

        numbered = re.match(r"^(\d+)\.\s+(.+)$", stripped)
        if numbered:
            flush_paragraph()
            story.append(Paragraph(inline_markup(f"{numbered.group(1)}. {numbered.group(2)}"), styles["ZHList"]))
            index += 1
            continue

        if stripped == "---":
            flush_paragraph()
            story.append(Spacer(1, 1 * mm))
            index += 1
            continue

        paragraph_buffer.append(stripped.removeprefix("> "))
        index += 1

    flush_paragraph()
    return story


def decorate_page(canvas, doc):
    canvas.saveState()
    width, height = A4
    if doc.page > 1:
        canvas.setStrokeColor(colors.HexColor("#D9DDE1"))
        canvas.setLineWidth(0.5)
        canvas.line(doc.leftMargin, height - 15 * mm, width - doc.rightMargin, height - 15 * mm)
        canvas.setFont(BODY_FONT, 7.5)
        canvas.setFillColor(colors.HexColor("#777777"))
        canvas.drawString(doc.leftMargin, height - 11.5 * mm, PAGE_HEADER)
    canvas.setStrokeColor(colors.HexColor("#D9DDE1"))
    canvas.setLineWidth(0.5)
    canvas.line(doc.leftMargin, 13 * mm, width - doc.rightMargin, 13 * mm)
    canvas.setFont(BODY_FONT, 8)
    canvas.setFillColor(colors.HexColor("#777777"))
    canvas.drawRightString(width - doc.rightMargin, 8.5 * mm, f"{doc.page}")
    canvas.restoreState()


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=21 * mm,
        bottomMargin=18 * mm,
        title=SOURCE.read_text(encoding="utf-8").splitlines()[0].removeprefix("# "),
        author="TAG continual condensation project",
        subject="Chinese research draft",
    )
    story = parse_markdown(SOURCE, doc.width)
    doc.build(story, onFirstPage=decorate_page, onLaterPages=decorate_page)
    print(OUTPUT)


if __name__ == "__main__":
    main()
