import base64
import html
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence

"""
Assemble charts into one self-contained HTML page.

Every image is inlined as a data URI rather than linked, so the file can be
moved, attached to a message or opened from a different machine and still be
the same document. A dashboard whose images live in a sibling directory is one
`mv` away from being a page of broken icons, and the report from a run that
finished last month is exactly the thing most likely to get moved.

The page is deliberately plain: this is Module 3's "make it legible" applied to
one run, not a feature browser. Each panel carries a caption saying what the
chart is for, because a chart nobody remembers the purpose of is a decoration.

A common pipe could be: plot_* | save_figure | Panel | render
"""

@dataclass
class Panel:
    """One chart on the page: a title, why it is there, and the file it came from"""
    title: str
    caption: str
    path: Path
    wide: bool = False

@dataclass
class Section:
    """A named group of panels, rendered as one band of the page"""
    title: str
    panels: List[Panel] = field(default_factory=list)

_STYLE = """
:root {
  --bg: #ffffff; --fg: #1b1f24; --muted: #5b6470; --line: #e2e6eb;
  --card: #f7f9fb; --accent: #4c72b0;
}
@media (prefers-color-scheme: dark) {
  :root { --bg: #14171b; --fg: #e6e9ed; --muted: #9aa4b0; --line: #2a2f36;
          --card: #1b1f25; --accent: #7aa2d6; }
}
* { box-sizing: border-box; }
body { margin: 0; padding: 2.5rem 1.5rem 4rem; background: var(--bg); color: var(--fg);
       font: 15px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, sans-serif; }
main { max-width: 1180px; margin: 0 auto; }
header { border-bottom: 1px solid var(--line); padding-bottom: 1.25rem; margin-bottom: 2rem; }
h1 { margin: 0 0 .35rem; font-size: 1.6rem; letter-spacing: -.01em; }
.subtitle { color: var(--muted); font-size: .95rem; }
.meta { color: var(--muted); font-size: .8rem; margin-top: .6rem; font-variant-numeric: tabular-nums; }
h2 { font-size: 1.05rem; text-transform: uppercase; letter-spacing: .07em; color: var(--muted);
     margin: 2.5rem 0 1rem; font-weight: 600; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(430px, 1fr)); gap: 1.25rem; }
.panel { background: var(--card); border: 1px solid var(--line); border-radius: 10px;
         padding: 1.1rem 1.1rem 1.25rem; overflow: hidden; }
.panel.wide { grid-column: 1 / -1; }
.panel h3 { margin: 0 0 .3rem; font-size: 1rem; }
.panel p { margin: 0 0 .9rem; color: var(--muted); font-size: .85rem; }
.panel img { width: 100%; height: auto; display: block; border-radius: 6px; background: #fff; }
footer { margin-top: 3rem; padding-top: 1.25rem; border-top: 1px solid var(--line);
         color: var(--muted); font-size: .8rem; }
code { background: var(--line); padding: .1rem .35rem; border-radius: 4px; font-size: .85em; }
"""

def _embed(path: Path) -> str:
    """Read a PNG and return it as a data URI"""
    encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"

def _panel_html(panel: Panel) -> str:
    """One card: title, caption, and the chart itself"""
    classes = "panel wide" if panel.wide else "panel"
    return (
        f'<div class="{classes}">'
        f"<h3>{html.escape(panel.title)}</h3>"
        f"<p>{html.escape(panel.caption)}</p>"
        f'<img alt="{html.escape(panel.title)}" src="{_embed(panel.path)}">'
        f"</div>"
    )

def render(
    sections: Sequence[Section],
    path,
    title: str = "mi-lab",
    subtitle: str = "",
    provenance: Optional[Sequence[str]] = None,
):
    """Write the sections out as one self-contained HTML file

    Panels whose image file is missing are skipped rather than fatal: a
    dashboard assembled from a run where one chart failed should still show
    the ones that worked.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    body = []
    drawn = 0
    for section in sections:
        available = [panel for panel in section.panels if Path(panel.path).exists()]
        if not available:
            continue
        drawn += len(available)
        body.append(f"<h2>{html.escape(section.title)}</h2>")
        body.append('<div class="grid">' + "".join(_panel_html(panel) for panel in available) + "</div>")

    stamped = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = "".join(f"<div>{html.escape(line)}</div>" for line in (provenance or []))
    document = (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>{html.escape(title)}</title><style>{_STYLE}</style></head><body><main>"
        f"<header><h1>{html.escape(title)}</h1>"
        f"<div class='subtitle'>{html.escape(subtitle)}</div>"
        f"<div class='meta'>{lines}<div>{drawn} charts &middot; built {stamped}</div></div></header>"
        + "".join(body)
        + "<footer>Generated by <code>python -m src.cli viz dashboard</code>. "
        "Images are embedded, so this file is self-contained and can be moved anywhere.</footer>"
        "</main></body></html>"
    )
    path.write_text(document, encoding="utf-8")
    return path
