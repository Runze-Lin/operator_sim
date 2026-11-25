"""Typeset the Chinese research report; report-only dependencies are documented in docs/paper."""
import argparse
import csv
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
                if element.attrGet("src").endswith("/system-vector-e2e.png"):
                    tokens[index - 1].attrSet("class", "system-comparison")
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


def request_appendix():
    """Generate the compact, standalone paper tables from the released DES ledger."""
    ledger = DOC.parent / "results/v2.3/system-e2e/requests.csv"
    with ledger.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    names = {"v4-pro": "DeepSeek-V4 Pro", "v41-flash": "DeepSeek-V4.1 Flash", "kimi-k31": "Kimi K3.1 Pro"}
    with ledger.with_name("original27.csv").open(encoding="utf-8", newline="") as f:
        rounds = list(csv.DictReader(f))
    if (len(rounds) != 27 or len({(r["model"], r["layout"], r["batch"]) for r in rounds}) != 27
            or any(sum(r["model"] == m for r in rounds) != 9 for m in names)):
        raise ValueError("Expected nine distinct decode rounds per model")
    keys = {(r["model"], r["prompt"], r["batch"], r["ep"]) for r in rows}
    if len(keys) != 33 or len(rows) != 33 or any(sum(r["model"] == m for r in rows) != 11 for m in names):
        raise ValueError(f"{ledger}: expected 11 distinct configurations for each of the three models")
    out = ["# 附录 H　当前三种策略：33 组整批请求与 27 组单轮 decode", "",
           "以下均为 950DT rank 0 的 DES 预测。每个模型的第一张表以秒描述整批请求（TP=8，至 1,024 个输出 token），第二张以毫秒描述原始 128Ki 图中一个推测轮次。两张表使用不同的配置，不逐行配对。带 † 的 1Mi 输入，其后 decode 超出位置预置，含域外推算。", ""]
    modes = ("legacy", "peak", "effective", "calibrated", "auto", "instruction_classes")
    for key, title in names.items():
        out += [f"## {title}", "", "**整批请求（秒）**", "", "| 输入 / B / EP | 历史 | 峰值 | 原效率 | 原法 | 默认 | 指令法 |",
                "|---|---:|---:|---:|---:|---:|---:|"]
        for r in sorted((r for r in rows if r["model"] == key), key=lambda r: tuple(int(r[k]) for k in ("prompt", "batch", "ep"))):
            length = "1Mi†" if int(r["prompt"]) == 1048576 else f"{int(r['prompt']) // 1024}Ki"
            numbers = " | ".join(f"{float(r[f'request_{m}_s']):.3f}" for m in modes)
            out.append(f"| {length} / {r['batch']} / {r['ep']} | {numbers} |")
        out += ["", "表头说明：B 为本卡 batch，EP 为专家并行度。历史是原始系统 Roofline；峰值和原效率采用相同执行精度与工作量，后者另含系统原效率系数。右三列只改 Vector 定价，依次为强制原法、逐合同默认、强制指令类别法。", ""]
        out += ["**单轮 target + draft（毫秒）**", "", "| 并行 / B | 显存 GB | 历史 | 峰值 | 原法 | 默认 | 指令法 |",
                "|---|---:|---:|---:|---:|---:|---:|"]
        for r in sorted((r for r in rounds if r["model"] == key), key=lambda r: (r["layout"], int(r["batch"]))):
            values = " | ".join(f"{float(r[m + '_us']) / 1000:.3f}" for m in ("historical", "peak", "calibrated", "auto", "instruction_classes"))
            gb = float(r["peak_gb"])
            out.append(f"| {r['layout']} / {r['batch']} | {gb:.2f}{'‡' if gb > 144 else ''} | {values} |")
        out += ["", "表头说明：B 为本卡 batch，显存为该行静态峰值；‡ 表示超过 144 GB，这一行仅作容量外比较。历史、峰值为两种 Roofline；最后三列是同一个 decode 轮次的三次重排，不含 prefill。", ""]
    out += ["[含 prefill、decode、请求的原始 CSV](../../results/v2.3/system-e2e/requests.csv) · [逐阶段图与归因](../system-vector-e2e.md) · [另一个计时边界：原始 27 个 decode 单轮](../decode-vector-comparison.md)", ""]
    return "\n".join(out)


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
      <dt>报告版本</dt><dd class="release">v2.5</dd>
      <dt>系统实验</dt><dd>三模型端到端 · 13 个文本预设的算子覆盖</dd>
      <dt>对照维度</dt><dd>原法、指令类别法与默认策略</dd>
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
    parser.add_argument("--refresh-request-tables", action="store_true", help="regenerate Appendix H from the versioned system request ledger before typesetting")
    args = parser.parse_args()
    if args.refresh_request_tables:
        (PAPER / "16-system-comparison.md").write_text(request_appendix(), encoding="utf-8")
    build(args.font_dir)
