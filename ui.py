"""Shared look for the app: Libre Baskerville, paper background, one accent colour.

Rules kept on purpose: no gradients, no glass, no shadows-as-decoration, no left-border
cards, no fading hover states, one spacing scale (8 px steps).
"""
import streamlit as st

PAPER = "#F6F3EC"
SURFACE = "#FFFFFF"
INK = "#16181D"
MUTED = "#4A4F59"
RULE = "#D6CFBF"
ACCENT = "#6B2D4A"        # plum
ACCENT_DARK = "#4D1F35"
POSITIVE = "#2F6B3A"
NEGATIVE = "#A33A24"

FONT_URL = "https://fonts.googleapis.com/css2?family=Libre+Baskerville:wght@400;700&display=swap"


def apply(page_title: str):
    st.set_page_config(page_title=page_title, layout="wide", initial_sidebar_state="collapsed")
    st.markdown(
        f"""
<style>
@import url('{FONT_URL}');
:root {{ --s1: 8px; --s2: 16px; --s3: 24px; --s4: 32px; --s6: 48px; }}
html, body, [class*="css"], .stApp, .stMarkdown, button, input, select, textarea, label,
[data-testid="stMetricValue"], [data-baseweb] {{
  font-family: 'Libre Baskerville', Georgia, serif !important;
}}
.stApp {{ background: {PAPER}; color: {INK}; }}
.block-container {{ padding-top: var(--s6); padding-bottom: var(--s6); max-width: 1180px; }}
h1, h2, h3, h4 {{ color: {INK}; font-weight: 700; letter-spacing: -0.01em; font-style: normal; }}
h1 {{ font-size: 2.4rem; line-height: 1.15; margin-bottom: var(--s1); }}
h2 {{ font-size: 1.45rem; margin-top: var(--s4); }}
h3 {{ font-size: 1.1rem; }}
p, li, label {{ color: {INK}; line-height: 1.6; }}
em, i {{ font-style: normal; }}
.lede {{ color: {MUTED}; font-size: 1.05rem; max-width: 720px; margin-bottom: var(--s4); }}
.rule {{ border: 0; border-top: 1px solid {RULE}; margin: var(--s4) 0; }}
.kicker {{ color: {MUTED}; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: var(--s1); }}
.panel {{ background: {SURFACE}; border: 1px solid {RULE}; border-radius: 4px; padding: var(--s3); margin-bottom: var(--s2); }}
.figure {{ font-size: 2.6rem; font-weight: 700; color: {INK}; line-height: 1.1; }}
.figure.accent {{ color: {ACCENT}; }}
.note {{ color: {MUTED}; font-size: 0.88rem; }}
.stat-row {{ display: flex; gap: var(--s2); flex-wrap: wrap; margin-bottom: var(--s3); }}
.stat {{ flex: 1 1 200px; background: {SURFACE}; border: 1px solid {RULE}; border-radius: 4px; padding: var(--s2) var(--s3); }}
.stat .v {{ font-size: 1.6rem; font-weight: 700; }}
.stat .l {{ color: {MUTED}; font-size: 0.82rem; margin-top: 4px; }}
table.plain {{ width: 100%; border-collapse: collapse; font-size: 0.92rem; }}
table.plain th {{ text-align: left; color: {MUTED}; font-weight: 400; border-bottom: 1px solid {INK}; padding: var(--s1); }}
table.plain td {{ border-bottom: 1px solid {RULE}; padding: var(--s1); }}
table.plain td.num, table.plain th.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
.pos {{ color: {POSITIVE}; }} .neg {{ color: {NEGATIVE}; }}
/* buttons: solid, darken on hover, never fade */
.stButton > button, .stFormSubmitButton > button {{
  background: {ACCENT}; color: #FFFFFF; border: 1px solid {ACCENT}; border-radius: 4px;
  padding: var(--s1) var(--s3); font-weight: 700; opacity: 1; transition: none;
}}
.stButton > button:hover, .stFormSubmitButton > button:hover {{
  background: {ACCENT_DARK}; border-color: {ACCENT_DARK}; color: #FFFFFF; opacity: 1;
}}
.stButton button *, .stFormSubmitButton button * {{ color: #FFFFFF !important; font-weight: 700; }}
.stButton > button:focus-visible {{ outline: 2px solid {INK}; outline-offset: 2px; }}
[data-testid="stTabs"] button p {{ font-size: 0.95rem; }}
[data-testid="stTabs"] [aria-selected="true"] p {{ color: {ACCENT}; font-weight: 700; }}
[data-testid="stHeader"] {{ background: {PAPER}; }}
#MainMenu, footer {{ visibility: hidden; }}
</style>
""",
        unsafe_allow_html=True,
    )


def plotly_layout(fig, height=380):
    fig.update_layout(
        height=height,
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family="Libre Baskerville, Georgia, serif", color=INK, size=13),
        margin=dict(l=16, r=16, t=24, b=16),
        xaxis=dict(gridcolor="#ECE7DC", zerolinecolor=RULE, linecolor=INK, tickfont=dict(color=MUTED)),
        yaxis=dict(gridcolor="#ECE7DC", zerolinecolor=RULE, linecolor=INK, tickfont=dict(color=INK)),
        showlegend=False,
    )
    return fig


def stats(items):
    """items: list of (value, label)."""
    cells = "".join(f'<div class="stat"><div class="v">{v}</div><div class="l">{l}</div></div>' for v, l in items)
    st.markdown(f'<div class="stat-row">{cells}</div>', unsafe_allow_html=True)
