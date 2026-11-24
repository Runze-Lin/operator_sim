"""Typeset the Chinese research report; report-only dependencies are documented in docs/paper."""
from html import escape
from os.path import relpath
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit

from markdown_it import MarkdownIt
from pydyf import Dictionary, String
from weasyprint import CSS, HTML
from weasyprint.text.fonts import FontConfiguration

DOC = Path(__file__).resolve().parents[1] / "docs"
ROOT = DOC.parent
PAPER = DOC / "paper"


def render_chapter(path, sections):
    md = MarkdownIt("commonmark").enable("table")
    tokens = md.parse(path.read_text())
    for token in tokens:
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
    return md.renderer.render(tokens, md.options, {})


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


def build():
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
    <div class="kicker">OPERATOR-SIM　 /　 性能建模研究报告</div>
    <h1>昇腾算子性能建模</h1>
    <div class="subtitle">从单核指令流水到长上下文推理</div>
    <p class="deck">基于 950DT 与 950PR 的 CA 校准<br>及面向 970DT 的架构外推</p>
    <div class="coverfoot">模型版本　v1.13.1<br>2026 年 11 月</div>
    </header>"""
    nav = '<nav><h1>目录</h1><ol>' + "".join(nav) + '</ol></nav>'
    html = '<!DOCTYPE html><html lang="zh-CN"><meta charset="utf-8"><title>昇腾算子性能建模：从单核指令流水到长上下文推理</title><meta name="author" content="operator-sim"><body>' + cover + nav + "".join(body) + "</body></html>"
    fonts = FontConfiguration()
    sheet = CSS(filename=PAPER / "print.css", font_config=fonts)
    document = HTML(string=html, base_url=str(PAPER)).render(stylesheets=[sheet], font_config=fonts)
    out = DOC / "operator-sim-report-zh.pdf"
    document.write_pdf(out, finisher=relative_evidence_links)
    print(f"{out}: {len(document.pages)} pages")


if __name__ == "__main__":
    build()
