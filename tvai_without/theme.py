"""
TerraVision AI — visual theme ("Abyss").

One place for every colour, the global stylesheet, and the small HTML components (hero with radar,
KPI cards, gauges, badges, legend). Deep ocean-black with an electric-blue → cyan signature
gradient, frosted-glass panels, an animated aurora background, and a crisp light variant.

Alert colours are semantic and deliberately NOT blue, so they never blend with the brand:
  red = illegal encroachment · orange = wildfire · amber = agricultural burn · steel = anomaly
"""

from __future__ import annotations

import html
from typing import Iterable, Optional, Tuple

DARK = {
    "bg": "#040912", "bg2": "#071428", "surface": "rgba(120,170,255,0.055)", "surface2": "rgba(120,170,255,0.10)",
    "solid": "#0a1730", "border": "rgba(120,170,255,0.16)", "border_hi": "rgba(96,165,250,0.65)",
    "text": "#e8f1ff", "muted": "#93a9cf", "faint": "#5f7aa5",
    "primary": "#3b82f6", "primary2": "#7db4ff", "accent": "#22d3ee", "good": "#34d399", "warn": "#fbbf24",
    "bad": "#fb7185", "shadow": "0 12px 44px rgba(1,6,20,0.6)", "glow": "rgba(59,130,246,0.40)",
    "grid": "rgba(120,170,255,0.11)", "input": "#091630", "orb1": "rgba(59,130,246,0.26)", "orb2": "rgba(34,211,238,0.17)",
    "orb3": "rgba(99,102,241,0.14)", "title_from": "#ffffff", "plotly": "plotly_dark",
}
LIGHT = {
    "bg": "#f3f8ff", "bg2": "#e3eeff", "surface": "rgba(255,255,255,0.82)", "surface2": "#ffffff",
    "solid": "#ffffff", "border": "rgba(29,99,237,0.16)", "border_hi": "rgba(29,99,237,0.55)",
    "text": "#0a1b3a", "muted": "#3d5a8a", "faint": "#7890b6",
    "primary": "#1d63ed", "primary2": "#3b82f6", "accent": "#0891b2", "good": "#059669", "warn": "#d97706",
    "bad": "#e11d48", "shadow": "0 12px 34px rgba(20,60,140,0.13)", "glow": "rgba(29,99,237,0.28)",
    "grid": "rgba(29,99,237,0.12)", "input": "#ffffff", "orb1": "rgba(59,130,246,0.20)", "orb2": "rgba(8,145,178,0.14)",
    "orb3": "rgba(99,102,241,0.10)", "title_from": "#0a1b3a", "plotly": "plotly_white",
}

# Semantic alert colours (shared by the map, charts and exports)
CATEGORY_COLORS = {
    "illegal_encroachment": "#ff4d6d", "natural_wildfire": "#ff8a3d",
    "agricultural_burn": "#facc15", "anomaly": "#a5b4d4", "unclassified": "#64748b",
}
# Ordered categorical colours for charts: distinct, colour-blind-friendlier, none equal to a category colour
CHART_COLORWAY = ["#3b82f6", "#22d3ee", "#a78bfa", "#34d399", "#f472b6", "#fbbf24"]


def palette(is_dark: bool) -> dict:
    return DARK if is_dark else LIGHT


def plotly_layout(is_dark: bool) -> dict:
    """Shared Plotly look: transparent panels, soft grid, Inter type."""
    t = palette(is_dark)
    return dict(
        template=t["plotly"], colorway=CHART_COLORWAY, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=t["muted"], size=12),
        title=dict(font=dict(family="Space Grotesk, sans-serif", color=t["text"], size=15)),
        xaxis=dict(gridcolor=t["grid"], zerolinecolor=t["grid"], linecolor=t["border"]),
        yaxis=dict(gridcolor=t["grid"], zerolinecolor=t["grid"], linecolor=t["border"]),
        hoverlabel=dict(bgcolor=t["solid"], bordercolor=t["border_hi"], font=dict(color=t["text"])),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )


def css(is_dark: bool) -> str:
    t = palette(is_dark)
    grad = f"linear-gradient(135deg, {t['primary']} 0%, {t['accent']} 100%)"
    return f"""
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
:root {{
  --bg:{t['bg']}; --bg2:{t['bg2']}; --surface:{t['surface']}; --surface2:{t['surface2']}; --solid:{t['solid']};
  --border:{t['border']}; --border-hi:{t['border_hi']}; --text:{t['text']}; --muted:{t['muted']}; --faint:{t['faint']};
  --primary:{t['primary']}; --primary2:{t['primary2']}; --accent:{t['accent']}; --good:{t['good']};
  --warn:{t['warn']}; --bad:{t['bad']}; --shadow:{t['shadow']}; --glow:{t['glow']}; --grad:{grad};
  --input:{t['input']}; --grid:{t['grid']};
  /* names used by older components */
  --text-main:{t['text']}; --text-muted:{t['muted']}; --teal-primary:{t['primary2']};
  --border-color:{t['border']}; --border-hover:{t['border_hi']}; --bg-mid:{t['solid']}; --bg-dark:{t['bg2']};
  --sidebar-info-color:{t['faint']};
}}
html, body, [class*="css"] {{ font-family:'Inter',sans-serif; color:var(--text); }}

/* ── atmosphere: aurora orbs that drift slowly behind a fixed gradient ───────────────── */
.stApp {{
  background:
    radial-gradient(900px 520px at 6% -10%, {t['orb1']} 0%, transparent 60%),
    radial-gradient(820px 500px at 102% 2%, {t['orb2']} 0%, transparent 55%),
    radial-gradient(700px 520px at 60% 108%, {t['orb3']} 0%, transparent 60%),
    linear-gradient(180deg, var(--bg) 0%, var(--bg2) 100%) !important;
  background-attachment: fixed !important;
}}
.stApp::before {{ content:''; position:fixed; inset:0; pointer-events:none; z-index:0; opacity:.55;
  background-image: linear-gradient(var(--grid) 1px, transparent 1px), linear-gradient(90deg, var(--grid) 1px, transparent 1px);
  background-size:56px 56px; -webkit-mask-image:radial-gradient(ellipse at 50% 0%, #000 0%, transparent 72%);
  mask-image:radial-gradient(ellipse at 50% 0%, #000 0%, transparent 72%); }}
[data-testid="stHeader"] {{ background: transparent !important; }}
.block-container {{ padding-top: 2rem !important; max-width: 1480px; position:relative; z-index:1; }}
h1,h2,h3,h4 {{ font-family:'Space Grotesk',sans-serif !important; letter-spacing:-0.015em; color:var(--text) !important; }}
p, li, span, label, div {{ color: var(--text); }}
a {{ color: var(--accent) !important; text-decoration:none; }}
a:hover {{ text-decoration:underline; }}

/* ── sidebar ─────────────────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {{ background: linear-gradient(190deg, {t['solid']} 0%, var(--bg) 100%) !important;
  border-right:1px solid var(--border) !important; }}
[data-testid="stSidebar"] > div {{ padding-top: .6rem; }}
[data-testid="stSidebar"] label, [data-testid="stSidebar"] p {{ color: var(--muted) !important; font-size:.8rem; }}
.section-title {{ font-family:'Space Grotesk',sans-serif; font-size:.7rem; font-weight:600; letter-spacing:.16em;
  text-transform:uppercase; color:var(--faint) !important; margin:22px 0 8px 2px; display:flex; align-items:center; gap:8px; }}
.section-title::before {{ content:''; width:6px; height:6px; border-radius:2px; background:var(--grad); box-shadow:0 0 10px var(--glow); }}
.section-title::after {{ content:''; flex:1; height:1px; background:linear-gradient(90deg,var(--border),transparent); }}

/* navigation: vertical pills with an active glow bar */
[data-testid="stSidebar"] [role="radiogroup"] {{ gap:6px; }}
[data-testid="stSidebar"] [role="radiogroup"] > label {{ position:relative; background:var(--surface); border:1px solid var(--border);
  border-radius:12px; padding:10px 14px 10px 18px; transition:all .22s ease; width:100%; box-sizing:border-box; cursor:pointer; overflow:hidden; }}
[data-testid="stSidebar"] [role="radiogroup"] > label::before {{ content:''; position:absolute; left:0; top:20%; bottom:20%; width:3px;
  border-radius:3px; background:var(--grad); opacity:0; transition:opacity .2s ease; }}
[data-testid="stSidebar"] [role="radiogroup"] > label:hover {{ border-color:var(--border-hi); transform:translateX(3px); }}
[data-testid="stSidebar"] [role="radiogroup"] > label:hover::before {{ opacity:.6; }}
[data-testid="stSidebar"] [role="radiogroup"] > label > div:first-child {{ display:none; }}
[data-testid="stSidebar"] [role="radiogroup"] > label p {{ color:var(--text) !important; font-size:.92rem !important; font-weight:500; margin:0; }}
[data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) {{ background:var(--grad); border-color:transparent;
  box-shadow:0 10px 28px var(--glow); }}
[data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) p {{ color:#fff !important; }}
[data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked)::before {{ opacity:0; }}

/* brand block */
.brand {{ display:flex; align-items:center; gap:12px; padding:6px 4px 14px; }}
.brand-mark {{ width:42px; height:42px; border-radius:13px; background:var(--grad); display:grid; place-items:center;
  box-shadow:0 8px 24px var(--glow), inset 0 1px 0 rgba(255,255,255,.35); position:relative; }}
.brand-mark::after {{ content:''; position:absolute; inset:-4px; border-radius:16px; border:1px solid var(--border-hi); opacity:.5; }}
.brand-name {{ font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:1.18rem; line-height:1.1; color:var(--text) !important; }}
.brand-sub {{ font-size:.66rem; color:var(--faint) !important; letter-spacing:.12em; text-transform:uppercase; }}

/* ── hero with radar ─────────────────────────────────────────────────────────────────── */
.hero {{ position:relative; overflow:hidden; border:1px solid var(--border); border-radius:24px; padding:32px 38px;
  margin-bottom:22px; background: linear-gradient(135deg, var(--surface2), var(--surface)); box-shadow:var(--shadow);
  backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px); }}
.hero::before {{ content:''; position:absolute; inset:-1px; border-radius:24px; padding:1px;
  background:linear-gradient(135deg, {t['primary']}88, transparent 38%, {t['accent']}77);
  -webkit-mask:linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0); -webkit-mask-composite:xor; mask-composite:exclude;
  pointer-events:none; }}
.hero::after {{ content:''; position:absolute; left:0; right:0; height:90px; top:-90px; pointer-events:none;
  background:linear-gradient(180deg, transparent, {t['accent']}22, transparent); animation:scan 7s linear infinite; }}
@keyframes scan {{ 0% {{ top:-90px; }} 100% {{ top:110%; }} }}
.hero-grid {{ position:absolute; inset:0; opacity:.55; pointer-events:none;
  background-image: linear-gradient(var(--grid) 1px, transparent 1px), linear-gradient(90deg, var(--grid) 1px, transparent 1px);
  background-size:34px 34px; -webkit-mask-image:radial-gradient(ellipse at 88% 10%, #000 0%, transparent 62%);
  mask-image:radial-gradient(ellipse at 88% 10%, #000 0%, transparent 62%); }}
.hero-body {{ position:relative; z-index:1; max-width:calc(100% - 210px); }}
.hero-eyebrow {{ font-family:'JetBrains Mono',monospace; font-size:.7rem; letter-spacing:.2em; text-transform:uppercase;
  color:var(--accent) !important; font-weight:500; }}
.hero-title {{ font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:2.6rem; line-height:1.04; margin:10px 0 8px;
  background:linear-gradient(120deg, {t['title_from']} 0%, {t['primary2']} 48%, {t['accent']} 100%);
  -webkit-background-clip:text; background-clip:text; color:transparent !important; }}
.hero-sub {{ color:var(--muted) !important; font-size:1rem; max-width:720px; line-height:1.55; }}
.chips {{ display:flex; flex-wrap:wrap; gap:8px; margin-top:18px; position:relative; z-index:1; }}
.chip {{ display:inline-flex; align-items:center; gap:8px; padding:5px 13px; border-radius:999px; font-size:.74rem;
  background:var(--surface); border:1px solid var(--border); color:var(--muted) !important; backdrop-filter:blur(6px); }}
.chip b {{ color:var(--text) !important; font-weight:600; }}
.pulse-dot {{ width:8px; height:8px; border-radius:50%; background:var(--good); animation:pulse 2s ease-in-out infinite; }}
@keyframes pulse {{ 0%,100% {{ box-shadow:0 0 0 0 {t['good']}aa; }} 50% {{ box-shadow:0 0 0 7px transparent; }} }}

.radar {{ position:absolute; right:34px; top:50%; transform:translateY(-50%); width:158px; height:158px; border-radius:50%;
  border:1px solid var(--border-hi); z-index:1; overflow:hidden;
  background:
    radial-gradient(circle, transparent 0 24%, var(--grid) 24.5% 25.5%, transparent 26% 49%, var(--grid) 49.5% 50.5%, transparent 51% 74%, var(--grid) 74.5% 75.5%, transparent 76%),
    linear-gradient(var(--grid), var(--grid)) center/1px 100% no-repeat,
    linear-gradient(var(--grid), var(--grid)) center/100% 1px no-repeat,
    radial-gradient(circle, {t['accent']}14, transparent 70%);
  box-shadow:0 0 40px {t['accent']}22, inset 0 0 30px {t['accent']}18; }}
.radar::before {{ content:''; position:absolute; inset:0; border-radius:50%;
  background:conic-gradient(from 0deg, {t['accent']}00 0deg, {t['accent']}00 270deg, {t['accent']}88 355deg, {t['accent']}00 360deg);
  animation:sweep 4.2s linear infinite; }}
@keyframes sweep {{ to {{ transform:rotate(360deg); }} }}
.blip {{ position:absolute; width:7px; height:7px; border-radius:50%; background:#ff4d6d; box-shadow:0 0 10px #ff4d6d;
  animation:blip 4.2s ease-out infinite; opacity:0; }}
.blip.b1 {{ left:62%; top:30%; animation-delay:.7s; }} .blip.b2 {{ left:30%; top:62%; background:#ff8a3d; box-shadow:0 0 10px #ff8a3d; animation-delay:2.1s; }}
.blip.b3 {{ left:70%; top:66%; background:#facc15; box-shadow:0 0 10px #facc15; animation-delay:3.2s; }}
@keyframes blip {{ 0% {{ opacity:1; transform:scale(1.6); }} 60% {{ opacity:.35; }} 100% {{ opacity:0; transform:scale(.8); }} }}

/* ── KPI cards ───────────────────────────────────────────────────────────────────────── */
.metric-grid {{ display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin:18px 0; }}
.metric-card {{ position:relative; overflow:hidden; background:linear-gradient(160deg, var(--surface2), var(--surface));
  border:1px solid var(--border); border-radius:18px; padding:18px 20px 16px; box-shadow:var(--shadow);
  backdrop-filter:blur(10px); -webkit-backdrop-filter:blur(10px);
  transition:transform .22s ease, border-color .22s ease, box-shadow .22s ease; }}
.metric-card::before {{ content:''; position:absolute; left:0; right:0; top:0; height:2px;
  background:linear-gradient(90deg, var(--kpi, var(--primary)), transparent 85%); }}
.metric-card::after {{ content:''; position:absolute; right:-30px; top:-30px; width:110px; height:110px; border-radius:50%;
  background:radial-gradient(circle, var(--kpi, var(--primary)), transparent 70%); opacity:.14; transition:opacity .25s ease; }}
.metric-card:hover {{ transform:translateY(-4px); border-color:var(--border-hi); box-shadow:0 18px 44px var(--glow); }}
.metric-card:hover::after {{ opacity:.28; }}
.metric-val {{ font-family:'Space Grotesk',sans-serif; font-size:2.05rem; font-weight:700; line-height:1; margin-bottom:8px;
  font-variant-numeric:tabular-nums; }}
.metric-label {{ font-size:.68rem; text-transform:uppercase; letter-spacing:.11em; color:var(--muted) !important; font-weight:500; }}

/* gauge ring */
.gauge-wrap {{ display:flex; align-items:center; gap:18px; background:linear-gradient(160deg, var(--surface2), var(--surface));
  border:1px solid var(--border); border-radius:18px; padding:16px 20px; box-shadow:var(--shadow); }}
.gauge {{ --pct:50; --gc:var(--primary); width:84px; height:84px; border-radius:50%; flex-shrink:0; position:relative;
  background:conic-gradient(var(--gc) calc(var(--pct) * 1%), var(--grid) 0); display:grid; place-items:center;
  box-shadow:0 0 26px color-mix(in srgb, var(--gc) 35%, transparent); }}
.gauge::before {{ content:''; position:absolute; inset:9px; border-radius:50%; background:var(--solid); }}
.gauge span {{ position:relative; font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:1.2rem; color:var(--text) !important; }}
.gauge-title {{ font-family:'Space Grotesk',sans-serif; font-weight:600; font-size:1rem; color:var(--text) !important; }}
.gauge-sub {{ font-size:.78rem; color:var(--muted) !important; margin-top:3px; line-height:1.45; }}

/* badges */
.badge {{ display:inline-flex; align-items:center; gap:6px; padding:3px 10px; border-radius:999px; font-size:.7rem; font-weight:600;
  letter-spacing:.04em; border:1px solid color-mix(in srgb, var(--bc) 45%, transparent);
  background:color-mix(in srgb, var(--bc) 14%, transparent); color:var(--bc) !important; }}
.badge::before {{ content:''; width:6px; height:6px; border-radius:50%; background:var(--bc); box-shadow:0 0 8px var(--bc); }}

/* section header */
.sec-head {{ display:flex; align-items:baseline; gap:14px; margin:26px 0 10px; }}
.sec-head h3 {{ margin:0; font-size:1.15rem; }}
.sec-head span {{ color:var(--faint) !important; font-size:.82rem; }}
.sec-head::after {{ content:''; flex:1; height:1px; background:linear-gradient(90deg,var(--border-hi),transparent); align-self:center; }}

/* legend */
.legend-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:10px; margin:6px 0 16px; }}
.legend-item {{ display:flex; align-items:center; gap:12px; background:var(--surface); border:1px solid var(--border);
  border-radius:13px; padding:10px 14px; font-size:.8rem; color:var(--muted) !important; transition:border-color .2s ease; }}
.legend-item:hover {{ border-color:var(--border-hi); }}
.legend-item span {{ color:var(--muted) !important; }}
.legend-dot {{ width:11px; height:11px; border-radius:50%; flex-shrink:0; box-shadow:0 0 12px currentColor; }}

/* ── buttons ─────────────────────────────────────────────────────────────────────────── */
.stButton > button, .stDownloadButton > button, .stLinkButton > a {{ position:relative; overflow:hidden;
  background:var(--grad) !important; color:#fff !important; border:0 !important; border-radius:13px !important; font-weight:600 !important;
  letter-spacing:.02em; padding:.62rem 1.25rem !important; box-shadow:0 8px 24px var(--glow), inset 0 1px 0 rgba(255,255,255,.3);
  transition:all .22s ease !important; }}
.stButton > button::after, .stDownloadButton > button::after {{ content:''; position:absolute; top:0; left:-60%; width:40%; height:100%;
  background:linear-gradient(100deg, transparent, rgba(255,255,255,.35), transparent); transform:skewX(-20deg); transition:left .6s ease; }}
.stButton > button:hover::after, .stDownloadButton > button:hover::after {{ left:130%; }}
.stButton > button:hover, .stDownloadButton > button:hover, .stLinkButton > a:hover {{ transform:translateY(-2px);
  box-shadow:0 16px 36px var(--glow); filter:brightness(1.08); color:#fff !important; }}
.stButton > button p, .stDownloadButton > button p {{ color:#fff !important; }}
.stButton > button:disabled {{ opacity:.45; box-shadow:none; }}

/* ── inputs ──────────────────────────────────────────────────────────────────────────── */
[data-baseweb="select"] > div, [data-baseweb="input"] > div, .stTextInput input, .stTextArea textarea, .stNumberInput input {{
  background:var(--input) !important; border:1px solid var(--border) !important; border-radius:11px !important; color:var(--text) !important; }}
[data-baseweb="select"] > div:hover, .stTextInput input:focus, .stTextArea textarea:focus {{ border-color:var(--border-hi) !important;
  box-shadow:0 0 0 3px var(--glow) !important; }}
[data-baseweb="select"] svg {{ fill:var(--primary2) !important; }}
[data-baseweb="tag"] {{ background:var(--grad) !important; border-radius:8px !important; }}
[data-baseweb="tag"] span {{ color:#fff !important; }}
[data-baseweb="popover"] ul, [data-baseweb="menu"] {{ background:var(--solid) !important; border:1px solid var(--border) !important;
  border-radius:13px !important; box-shadow:var(--shadow) !important; }}
[data-baseweb="menu"] li, [role="option"] {{ background:transparent !important; color:var(--text) !important; }}
[data-baseweb="menu"] li:hover, [role="option"]:hover, [aria-selected="true"] {{ background:var(--surface2) !important; color:var(--primary2) !important; }}
[data-testid="stSlider"] [role="slider"] {{ background:var(--primary2) !important; box-shadow:0 0 0 6px var(--glow); }}
[data-testid="stCheckbox"] label span {{ color:var(--text) !important; }}
label[data-baseweb="checkbox"] span:first-child {{ border-radius:6px !important; }}
[data-testid="stWidgetLabel"] p {{ color:var(--muted) !important; font-size:.78rem; font-weight:500; letter-spacing:.02em; }}

/* ── tabs ────────────────────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {{ gap:6px; background:var(--surface); border:1px solid var(--border); padding:5px; border-radius:15px;
  backdrop-filter:blur(8px); }}
.stTabs [data-baseweb="tab"] {{ border-radius:11px; padding:8px 16px; color:var(--muted) !important; background:transparent; height:auto;
  transition:all .2s ease; }}
.stTabs [data-baseweb="tab"]:hover {{ color:var(--text) !important; background:var(--surface2); }}
.stTabs [data-baseweb="tab"] p {{ color:inherit !important; font-weight:500; }}
.stTabs [aria-selected="true"] {{ background:var(--grad) !important; color:#fff !important; box-shadow:0 6px 20px var(--glow); }}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {{ display:none; }}

/* ── expanders, alerts, tables, charts, map ─────────────────────────────────────────── */
[data-testid="stExpander"] {{ background:var(--surface); border:1px solid var(--border) !important; border-radius:15px !important; overflow:hidden; }}
[data-testid="stExpander"] summary {{ font-weight:500; }}
.stAlert {{ border-radius:15px !important; border:1px solid var(--border) !important; backdrop-filter:blur(8px); }}
[data-testid="stDataFrame"], [data-testid="stDataEditor"] {{ border:1px solid var(--border); border-radius:15px; overflow:hidden; box-shadow:var(--shadow); }}
[data-testid="stPlotlyChart"] {{ background:linear-gradient(160deg, var(--surface2), var(--surface)); border:1px solid var(--border);
  border-radius:18px; padding:8px; box-shadow:var(--shadow); transition:border-color .2s ease; }}
[data-testid="stPlotlyChart"]:hover {{ border-color:var(--border-hi); }}
iframe[title*="streamlit_folium"] {{ border-radius:20px; border:1px solid var(--border-hi); box-shadow:var(--shadow), 0 0 40px {t['glow']}; }}
[data-testid="stStatusWidget"], [data-testid="stStatus"] {{ border-radius:15px !important; }}
hr {{ border-color:var(--border) !important; }}
.stCaption, [data-testid="stCaptionContainer"] {{ color:var(--faint) !important; }}
code {{ background:var(--surface2) !important; color:var(--accent) !important; border-radius:6px; font-family:'JetBrains Mono',monospace !important; }}
::selection {{ background:var(--glow); }}
::-webkit-scrollbar {{ width:8px; height:8px; }} ::-webkit-scrollbar-track {{ background:transparent; }}
::-webkit-scrollbar-thumb {{ background:var(--border-hi); border-radius:8px; }}

@media (prefers-reduced-motion: reduce) {{ *, *::before, *::after {{ animation:none !important; transition:none !important; }} }}
@media (max-width: 900px) {{
  .metric-grid {{ grid-template-columns:repeat(2,1fr); }} .hero-title {{ font-size:1.9rem; }}
  .radar {{ display:none; }} .hero-body {{ max-width:100%; }} .hero {{ padding:24px 22px; }}
}}
"""


def apply(st, is_dark: bool) -> None:
    st.markdown(f"<style>{css(is_dark)}</style>", unsafe_allow_html=True)


BRAND_SVG = ('<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="1.8" '
             'stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/>'
             '<path d="M3 12h18M12 3c3 3.2 3 14.8 0 18M12 3c-3 3.2-3 14.8 0 18"/></svg>')


def brand_html() -> str:
    return (f'<div class="brand"><div class="brand-mark">{BRAND_SVG}</div><div>'
            '<div class="brand-name">TerraVision</div><div class="brand-sub">Forest intelligence</div></div></div>')


def hero_html(title: str, subtitle: str, chips: Iterable[Tuple[str, str]], eyebrow: str = "Satellite forest guardian") -> str:
    chip_html = "".join(f'<span class="chip">{html.escape(k)} <b>{html.escape(v)}</b></span>' for k, v in chips)
    return (f'<div class="hero"><div class="hero-grid"></div>'
            '<div class="radar"><span class="blip b1"></span><span class="blip b2"></span><span class="blip b3"></span></div>'
            f'<div class="hero-body"><div class="hero-eyebrow">{html.escape(eyebrow)}</div>'
            f'<div class="hero-title">{html.escape(title)}</div><div class="hero-sub">{html.escape(subtitle)}</div>'
            f'<div class="chips"><span class="chip"><span class="pulse-dot"></span><b>VIIRS feed live</b></span>{chip_html}</div></div></div>')


def kpi_html(cards: Iterable[Tuple[object, str, str]], min_width: Optional[int] = None) -> str:
    """cards: (value, label, colour)."""
    body = "".join(
        f'<div class="metric-card" style="--kpi:{c}"><div class="metric-val" style="color:{c}">{html.escape(str(v))}</div>'
        f'<div class="metric-label">{html.escape(label)}</div></div>' for v, label, c in cards)
    style = f' style="grid-template-columns:repeat(auto-fit,minmax({min_width}px,1fr));"' if min_width else ""
    return f'<div class="metric-grid"{style}>{body}</div>'


def gauge_html(percent: float, title: str, subtitle: str = "", color: str = "var(--primary)") -> str:
    """A ring gauge (0–100) with a title and one line of explanation."""
    pct = max(0, min(100, float(percent)))
    return (f'<div class="gauge-wrap"><div class="gauge" style="--pct:{pct:.0f};--gc:{color}"><span>{pct:.0f}%</span></div>'
            f'<div><div class="gauge-title">{html.escape(title)}</div><div class="gauge-sub">{html.escape(subtitle)}</div></div></div>')


def badge_html(text: str, color: str = "var(--primary2)") -> str:
    return f'<span class="badge" style="--bc:{color}">{html.escape(text)}</span>'


def section_html(title: str, sub: str = "") -> str:
    return f'<div class="sec-head"><h3>{html.escape(title)}</h3><span>{html.escape(sub)}</span></div>'


def legend_html(items: Iterable[Tuple[str, str]]) -> str:
    return '<div class="legend-grid">' + "".join(
        f'<div class="legend-item"><div class="legend-dot" style="background:{c};color:{c}"></div><span>{html.escape(t)}</span></div>'
        for c, t in items) + "</div>"
