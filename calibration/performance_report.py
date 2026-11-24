"""Typeset the Chinese research report; report-only dependencies are documented in docs/paper."""
import argparse
from base64 import b64encode
from functools import lru_cache
from html import escape
from io import BytesIO
from os.path import relpath
from pathlib import Path
import re
from urllib.parse import quote, unquote, urlsplit

from markdown_it import MarkdownIt
from markdown_it.token import Token
from matplotlib import rc_context
from matplotlib.font_manager import FontProperties
from matplotlib.mathtext import math_to_image
from pydyf import Dictionary, String
from weasyprint import CSS, HTML
from weasyprint.text.fonts import FontConfiguration

DOC = Path(__file__).resolve().parents[1] / "docs"
PAPER = DOC / "paper"
INLINE_MATH = re.compile(r"\$(?![\s$])((?:\\.|[^$\\\n])+?)(?<!\s)\$(?!\$)")


def inline_math(state, silent):
    expression = INLINE_MATH.match(state.src, state.pos)
    if expression is None or expression.end() > state.posMax:
        return False
    if not silent:
        state.push("inline_math", "", 0).content = expression[1]
    state.pos = expression.end()
    return True


def display_math(state, start, end, silent):
    if state.is_code_block(start) or state.getLines(start, start + 1, 0, False).strip() != "$$":
        return False
    if silent:
        return True
    for close in range(start + 1, end):
        if state.getLines(close, close + 1, 0, False).strip() == "$$":
            token = state.push("display_math", "", 0)
            token.content = state.getLines(start + 1, close, 0, False).strip()
            token.map = [start, close + 1]
            state.line = close + 1
            return True
    raise ValueError(f"Unclosed display equation on line {start + 1}")


@lru_cache(maxsize=512)
def equation_image(expression, block):
    if not expression:
        raise ValueError("Empty equation")
    stream = BytesIO()
    prop = FontProperties(size=12 if block else 11, math_fontfamily="stix")
    with rc_context({"svg.fonttype": "path", "savefig.transparent": True}):
        depth = math_to_image(f"${expression}$", stream, prop=prop, dpi=72, format="svg",
                              color="#142a38")
    data = b64encode(stream.getvalue()).decode("ascii")
    baseline = "" if block else f' style="vertical-align: {-depth:g}pt"'
    return f'<img src="data:image/svg+xml;base64,{data}" alt="{escape(expression)}"{baseline}>'


def render_inline_math(_renderer, tokens, index, _options, _env):
    return f'<span class="math-inline">{equation_image(tokens[index].content, False)}</span>'


def render_display_math(_renderer, tokens, index, _options, env):
    env["equation"] += 1
    label = f'{env["chapter"]}.{env["equation"]}'
    image = equation_image(tokens[index].content, True)
    return (f'<div class="equation"><div class="equation-lines">{image}</div>'
            f'<span class="equation-number">({label})</span></div>')


def render_chapter(path, sections):
    md = MarkdownIt("commonmark", {"html": True}).enable("table")
    md.inline.ruler.before("escape", "inline_math", inline_math)
    md.block.ruler.before("fence", "display_math", display_math, {"alt": ["paragraph"]})
    md.add_render_rule("inline_math", render_inline_math)
    md.add_render_rule("display_math", render_display_math)
    tokens = md.parse(path.read_text())
    for index, token in enumerate(tokens):
        if token.type == "inline" and token.children and len(token.children) == 1:
            element = token.children[0]
            if element.type == "image" and tokens[index - 1].type == "paragraph_open":
                tokens[index - 1].tag = tokens[index + 1].tag = "figure"
                caption = md.renderer.renderInline(element.children, md.options, {})
                token.children.append(Token("html_inline", "", 0, content=f"<figcaption>{caption}</figcaption>"))
        for child in token.children or ():
            if child.type not in ("image", "link_open"):
                continue
            attr = "src" if child.type == "image" else "href"
            url = child.attrGet(attr)
            parsed = urlsplit(url)
            if parsed.scheme or not parsed.path:
                continue
            target = (path.parent / unquote(parsed.path)).resolve()
            if not target.is_file():
                raise ValueError(f"{path.name}: missing {target}")
            if child.type == "image":
                dest = target.as_uri()
            elif target in sections and not parsed.fragment:
                dest = "#" + sections[target]
            else:
                dest = target.as_uri()
                if parsed.fragment:
                    dest += "#" + parsed.fragment
            child.attrSet(attr, dest)
    env = {"chapter": int(path.stem.split("-")[0]), "equation": 0}
    return md.renderer.render(tokens, md.options, env)


def relative_evidence_links(document, pdf):
    """Keep evidence links usable when a checkout is moved or served at a different URL."""
    for obj in pdf.objects:
        if not isinstance(obj, Dictionary) or obj.get("Subtype") != "/Link":
            continue
        action = obj.get("A", {})
        if action.get("S") != "/URI":
            continue
        uri = urlsplit(action["URI"].string)
        if uri.scheme == "file":
            path = quote(relpath(unquote(uri.path), DOC))
            action["URI"] = String(path + ("#" + uri.fragment if uri.fragment else ""))


def build(font_dir=None):
    sources = sorted(PAPER.glob("[0-9][0-9]-*.md"))
    chapters = {p: "chapter-" + p.stem.split("-")[0] for p in sources}
    nav = []
    body = []
    for source, anchor in chapters.items():
        title = source.read_text().splitlines()[0].lstrip("# ")
        nav.append(f'<li><a href="#{anchor}">{escape(title)}</a></li>')
        contents = render_chapter(source, chapters)
        body.append(f'<section id="{anchor}">{contents}</section>')
    cover = """<header class="cover">
    <div class="kicker">OPERATOR-SIM <span>性能建模研究报告</span></div>
    <h1>昇腾算子性能建模</h1>
    <div class="subtitle">从单核指令流水到长上下文推理</div>
    <p class="deck">基于 950DT 与 950PR 的 CA 校准<br>及面向 970DT 的架构外推</p>
    <dl class="coverfoot">
      <dt>发布版本</dt><dd class="release">v2.0.0</dd>
      <dt>实验冻结版</dt><dd>v1.13.1</dd>
      <dt>成稿日期</dt><dd>2026 年 11 月</dd>
    </dl>
    </header>"""
    nav = '<nav><h1>目录</h1><ol>' + "".join(nav) + '</ol></nav>'
    html = '<!DOCTYPE html><html lang="zh-CN"><meta charset="utf-8"><title>昇腾算子性能建模：从单核指令流水到长上下文推理</title><meta name="author" content="operator-sim"><body>' + cover + nav + "".join(body) + "</body></html>"
    fonts = FontConfiguration()
    styles = [CSS(filename=PAPER / "print.css", font_config=fonts)]
    if font_dir is not None:
        faces = []
        for family, stem in [("Noto Serif CJK SC", "NotoSerifCJKsc"), ("Noto Sans CJK SC", "NotoSansCJKsc")]:
            path = font_dir / f"{stem}-Regular.otf"
            if not path.is_file():
                raise FileNotFoundError(path)
            faces.append(f'@font-face {{ font-family: "{family}"; src: url("{path.resolve().as_uri()}"); }}')
        styles.append(CSS(string="\n".join(faces), font_config=fonts))
    document = HTML(string=html, base_url=str(PAPER)).render(stylesheets=styles, font_config=fonts)
    out = DOC / "operator-sim-report-zh.pdf"
    document.write_pdf(out, finisher=relative_evidence_links, pdf_tags=True)
    print(f"{out}: {len(document.pages)} pages")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--font-dir", type=Path, help="directory with the two Noto CJK Regular OTF files")
    build(parser.parse_args().font_dir)
