"""Shared CSS and small HTML helpers for the operator dashboard.

Kept separate from app.py so the page script stays focused on layout and
run flow. The base dark palette itself comes from `.streamlit/config.toml`
(the officially supported mechanism for Streamlit's own widget chrome);
this module only adds the operator-console layer on top — badges, KPI
cards, panels, empty states — using CSS classes this app controls
directly rather than fragile selectors on Streamlit's internal markup.
"""

from __future__ import annotations

import html

import streamlit as st

_CSS = """
<style>
.block-container {
    padding-top: 1.75rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}

.zds-header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 0.75rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 1.25rem;
}
.zds-title { font-size: 1.55rem; font-weight: 700; letter-spacing: 0.01em; margin: 0; }
.zds-subtitle { color: rgba(230,237,243,0.55); font-size: 0.85rem; margin-top: 0.2rem; }
.zds-badges { display: flex; gap: 0.5rem; flex-wrap: wrap; align-items: center; }

.badge {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.28rem 0.7rem;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    border: 1px solid transparent;
    white-space: nowrap;
}
.badge-idle      { background: rgba(148,163,184,0.12); color: #94a3b8; border-color: rgba(148,163,184,0.35); }
.badge-ready     { background: rgba(58,166,255,0.12);  color: #3aa6ff; border-color: rgba(58,166,255,0.35); }
.badge-running   { background: rgba(242,183,5,0.14);   color: #f2b705; border-color: rgba(242,183,5,0.4); }
.badge-completed { background: rgba(53,208,127,0.14);  color: #35d07f; border-color: rgba(53,208,127,0.4); }
.badge-error     { background: rgba(242,65,65,0.14);   color: #f24141; border-color: rgba(242,65,65,0.4); }
.badge-neutral   { background: rgba(148,163,184,0.10); color: #cbd5e1; border-color: rgba(148,163,184,0.25); }

.zds-panel {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 1rem 1.1rem;
    margin-bottom: 1rem;
}
.zds-panel h4 {
    margin: 0 0 0.7rem 0;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: rgba(230,237,243,0.55);
}

.kpi-row { display: flex; gap: 0.7rem; flex-wrap: wrap; margin-bottom: 1rem; }
.kpi-card {
    flex: 1 1 130px;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 0.7rem 0.9rem;
}
.kpi-label { color: rgba(230,237,243,0.5); font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.06em; }
.kpi-value { font-size: 1.3rem; font-weight: 700; margin-top: 0.15rem; line-height: 1.2; }
.kpi-sub { color: rgba(230,237,243,0.45); font-size: 0.7rem; margin-top: 0.15rem; }

.zds-empty {
    border: 1px dashed rgba(255,255,255,0.15);
    border-radius: 12px;
    padding: 3rem 1.5rem;
    text-align: center;
    color: rgba(230,237,243,0.55);
}
.zds-empty .zds-empty-title { color: #e6edf3; font-size: 1.05rem; font-weight: 600; margin-bottom: 0.4rem; }

.event-log { max-height: 320px; overflow-y: auto; }
.event-line {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 0.76rem;
    padding: 0.2rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.event-line .badge { padding: 0.1rem 0.5rem; font-size: 0.62rem; }
</style>
"""


def inject_theme() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def badge(text: str, kind: str = "neutral") -> str:
    return f'<span class="badge badge-{kind}">{html.escape(text)}</span>'


def kpi_card(label: str, value: str, sub: str | None = None) -> str:
    sub_html = f'<div class="kpi-sub">{html.escape(sub)}</div>' if sub else ""
    return (
        f'<div class="kpi-card"><div class="kpi-label">{html.escape(label)}</div>'
        f'<div class="kpi-value">{html.escape(str(value))}</div>{sub_html}</div>'
    )


def kpi_row(cards: list[str]) -> str:
    return f'<div class="kpi-row">{"".join(cards)}</div>'


def panel(title: str, body_html: str) -> str:
    return f'<div class="zds-panel"><h4>{html.escape(title)}</h4>{body_html}</div>'


def empty_state(title: str, message: str) -> str:
    return (
        '<div class="zds-empty">'
        f'<div class="zds-empty-title">{html.escape(title)}</div>'
        f"<div>{html.escape(message)}</div>"
        "</div>"
    )


def event_line(kind: str, label: str, text: str) -> str:
    return f'<div class="event-line">{badge(label, kind)}<span>{html.escape(text)}</span></div>'
