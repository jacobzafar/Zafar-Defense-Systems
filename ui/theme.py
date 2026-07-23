"""Shared CSS and small HTML helpers for the operator dashboard.

Kept separate from app.py so the page script stays focused on layout and
run flow. The base dark palette itself comes from `.streamlit/config.toml`
(the officially supported mechanism for Streamlit's own widget chrome);
this module only adds the operator-console layer on top — badges, KPI
cards, panels, empty states — using CSS classes this app controls
directly rather than fragile selectors on Streamlit's internal markup.

Color language (deliberately small): the base UI stays neutral slate/gray
so it reads as instrumentation, not decoration. Exactly one accent color
(`--zds-accent`, a restrained amber-orange) is reserved for the two things
that actually need an operator's attention — live detections/tracks (the
overlay boxes drawn in `ui/overlay.py` use the same color) and the "LOST"
event kind. Green/red stay reserved for unambiguous success/failure
(ACQUIRED, COMPLETED vs. DROPPED, ERROR) rather than being reused as
general decoration.
"""

from __future__ import annotations

import html

import streamlit as st

_CSS = """
<style>
:root {
    --zds-accent: #ff9142;
    --zds-accent-dim: rgba(255, 145, 66, 0.14);
    --zds-accent-border: rgba(255, 145, 66, 0.4);
    --zds-success: #35d07f;
    --zds-error: #f24141;
    --zds-radius: 10px;
    --zds-radius-lg: 12px;
    --zds-border: rgba(255,255,255,0.08);
    --zds-surface: rgba(255,255,255,0.03);
    --zds-text-dim: rgba(230,237,243,0.55);
    --zds-text-dimmer: rgba(230,237,243,0.4);
    --zds-gap: 0.75rem;
}

.block-container {
    /* Streamlit's own fixed header (data-testid="stHeader") is ~60px tall
       and sits at a very high z-index; anything less than this clears its
       bottom edge only partially, so the title/badges' top few pixels get
       painted over — clipped ascenders, not a font-rendering issue. */
    padding-top: 4.5rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}

.zds-header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: var(--zds-gap);
    padding-bottom: 1rem;
    border-bottom: 1px solid var(--zds-border);
    margin-bottom: 1.25rem;
}
.zds-title { font-size: 1.55rem; font-weight: 700; letter-spacing: 0.01em; margin: 0; }
.zds-subtitle { color: var(--zds-text-dim); font-size: 0.85rem; margin-top: 0.2rem; }
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
.badge-completed { background: rgba(53,208,127,0.14);  color: var(--zds-success); border-color: rgba(53,208,127,0.4); }
.badge-acquired  { background: rgba(53,208,127,0.14);  color: var(--zds-success); border-color: rgba(53,208,127,0.4); }
.badge-error     { background: rgba(242,65,65,0.14);   color: var(--zds-error); border-color: rgba(242,65,65,0.4); }
.badge-dropped   { background: rgba(242,65,65,0.14);   color: var(--zds-error); border-color: rgba(242,65,65,0.4); }
.badge-lost      { background: var(--zds-accent-dim);  color: var(--zds-accent); border-color: var(--zds-accent-border); }
.badge-neutral   { background: rgba(148,163,184,0.10); color: #cbd5e1; border-color: rgba(148,163,184,0.25); }
.badge-frame     { background: rgba(148,163,184,0.06); color: rgba(203,213,225,0.4); border-color: rgba(148,163,184,0.12); }

/* Primary status strip: the four live-operational readouts an operator
   should be able to read from across the room (state, FPS, latency,
   active tracks). Deliberately larger and higher-contrast than the
   secondary run-total cards below it. */
.stat-row { display: flex; gap: var(--zds-gap); flex-wrap: wrap; margin-bottom: var(--zds-gap); }
.stat-tile {
    flex: 1 1 160px;
    background: var(--zds-surface);
    border: 1px solid rgba(255,255,255,0.14);
    border-radius: var(--zds-radius-lg);
    padding: 0.85rem 1.1rem;
}
.stat-label { color: var(--zds-text-dim); font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.07em; }
.stat-value { font-size: 1.9rem; font-weight: 700; margin-top: 0.2rem; line-height: 1.15; font-variant-numeric: tabular-nums; }
.stat-value .stat-unit { font-size: 1rem; font-weight: 500; color: var(--zds-text-dim); margin-left: 0.2rem; }
.stat-tile-state .stat-value { font-size: 1.4rem; text-transform: uppercase; letter-spacing: 0.02em; }
.stat-tile-idle      .stat-value { color: #94a3b8; }
.stat-tile-ready     .stat-value { color: #3aa6ff; }
.stat-tile-running   .stat-value { color: #f2b705; }
.stat-tile-completed .stat-value { color: var(--zds-success); }
.stat-tile-error     .stat-value { color: var(--zds-error); }
/* The one place the primary strip uses the detection accent: active
   tracks *are* live detections, so a non-zero count is highlighted the
   same color as the boxes drawn on the video. Zero stays neutral —
   nothing to draw the eye to. */
.stat-tile-tracks-active .stat-value { color: var(--zds-accent); }

.zds-panel {
    background: var(--zds-surface);
    border: 1px solid var(--zds-border);
    border-radius: var(--zds-radius);
    padding: 1rem 1.1rem;
    margin-bottom: var(--zds-gap);
}
.zds-panel h4 {
    margin: 0 0 0.7rem 0;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--zds-text-dim);
}

.zds-panel-label {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--zds-text-dim);
    margin-bottom: 0.4rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

/* Frames the live video pane so it reads as the page's focal panel
   rather than a bare image floating in whitespace. */
div[data-testid="stImage"] {
    border: 1px solid var(--zds-border);
    border-radius: var(--zds-radius-lg);
    overflow: hidden;
    background: #000;
}

.kpi-row { display: flex; gap: 0.6rem; flex-wrap: wrap; margin-bottom: var(--zds-gap); }
.kpi-card {
    flex: 1 1 120px;
    background: var(--zds-surface);
    border: 1px solid var(--zds-border);
    border-radius: var(--zds-radius);
    padding: 0.6rem 0.8rem;
}
.kpi-label { color: var(--zds-text-dim); font-size: 0.66rem; text-transform: uppercase; letter-spacing: 0.06em; }
.kpi-value { font-size: 1.05rem; font-weight: 600; margin-top: 0.15rem; line-height: 1.2; font-variant-numeric: tabular-nums; }
.kpi-sub { color: var(--zds-text-dimmer); font-size: 0.7rem; margin-top: 0.15rem; }

.zds-empty {
    border: 1px dashed rgba(255,255,255,0.15);
    border-radius: var(--zds-radius-lg);
    padding: 3rem 1.5rem;
    text-align: center;
    color: var(--zds-text-dim);
}
.zds-empty .zds-empty-title { color: #e6edf3; font-size: 1.05rem; font-weight: 600; margin-bottom: 0.4rem; }

.event-log { max-height: 320px; overflow-y: auto; }
.event-line {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 0.76rem;
    padding: 0.25rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.event-line .badge { padding: 0.1rem 0.5rem; font-size: 0.62rem; flex-shrink: 0; }
/* Routine per-frame telemetry is real data, kept for anyone who wants it,
   but visually recedes so ACQUIRED/LOST/DROPPED — the events an operator
   actually needs to react to — are what the eye catches while scanning. */
.event-line-frame { opacity: 0.5; }
.event-line-frame span:last-child { font-size: 0.72rem; }
</style>
"""


def inject_theme() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def badge(text: str, kind: str = "neutral") -> str:
    return f'<span class="badge badge-{kind}">{html.escape(text)}</span>'


def stat_tile(label: str, value: str, unit: str | None = None, kind: str = "") -> str:
    """One tile in the primary status strip (state / FPS / latency / tracks)."""
    unit_html = f'<span class="stat-unit">{html.escape(unit)}</span>' if unit else ""
    extra_class = f" stat-tile-{kind}" if kind else ""
    return (
        f'<div class="stat-tile{extra_class}"><div class="stat-label">{html.escape(label)}</div>'
        f'<div class="stat-value">{html.escape(str(value))}{unit_html}</div></div>'
    )


def stat_row(tiles: list[str]) -> str:
    return f'<div class="stat-row">{"".join(tiles)}</div>'


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


def panel_label(text: str) -> str:
    return f'<div class="zds-panel-label"><span>{html.escape(text)}</span></div>'


def empty_state(title: str, message: str) -> str:
    return (
        '<div class="zds-empty">'
        f'<div class="zds-empty-title">{html.escape(title)}</div>'
        f"<div>{html.escape(message)}</div>"
        "</div>"
    )


def event_line(kind: str, label: str, text: str) -> str:
    muted_class = " event-line-frame" if kind == "frame" else ""
    return f'<div class="event-line{muted_class}">{badge(label, kind)}<span>{html.escape(text)}</span></div>'
