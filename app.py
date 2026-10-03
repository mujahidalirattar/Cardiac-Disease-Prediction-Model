"""
app.py : Cardiac Disease Prediction Dashboard (Streamlit UI)

This file is ONLY the user interface. The machine-learning logic lives in
backend.py. The original prediction flow (label mapping, blood-pressure
validation and predict_heart_disease) is kept exactly as it was. The only
backend addition is evaluate_live_model(), used by the Insights page to measure
the live model on an uploaded dataset; no existing backend logic was changed.

Media is read from an `assets/` folder next to this file:
    assets/heart_monitor.mp4   -> ambient hero background video
    assets/heart_pump.mp4       -> pumping heart video on the right of the hero
    assets/banner_ecg.jpg       -> hero fallback
    assets/bg_assessment.jpg    -> Assessment page background
    assets/bg_insights.jpg      -> Insights page background
    assets/bg_guide.jpg         -> Guide page background
    assets/bg_training.jpg      -> Train Model page background
If a file is missing the app still runs (it falls back to gradients).
"""

import os
import base64

import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go

import backend as be

# --------------------------------------------------------------------------- #
# PATHS AND ASSET LOADING
# --------------------------------------------------------------------------- #

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSET_DIR = os.path.join(BASE_DIR, "assets")


@st.cache_data(show_spinner=False)
def _b64(filename: str) -> str:
    path = os.path.join(ASSET_DIR, filename)
    try:
        with open(path, "rb") as fh:
            return base64.b64encode(fh.read()).decode()
    except Exception:
        return ""


def _img_uri(filename: str) -> str:
    data = _b64(filename)
    return "data:image/jpeg;base64," + data if data else ""


def _video_uri(filename: str) -> str:
    data = _b64(filename)
    return "data:video/mp4;base64," + data if data else ""


PAGE_BG = {
    "assess": "bg_assessment.jpg",
    "insights": "bg_insights.jpg",
    "guide": "bg_guide.jpg",
    "train": "bg_training.jpg",
}

FRIENDLY_ALGO = {
    "RandomForestClassifier": "Random Forest",
    "HistGradientBoostingClassifier": "Gradient Boosting",
    "GradientBoostingClassifier": "Gradient Boosting",
    "LogisticRegression": "Logistic Regression",
}


def short_algo(name: str) -> str:
    return FRIENDLY_ALGO.get(name, name.replace("Classifier", ""))


# --------------------------------------------------------------------------- #
# PAGE CONFIG + SESSION
# --------------------------------------------------------------------------- #

st.set_page_config(page_title="Cardiac Disease Prediction", page_icon="🫀",
                   layout="wide", initial_sidebar_state="collapsed")

if "page" not in st.session_state:
    st.session_state.page = "assess"
if "theme" not in st.session_state:
    st.session_state.theme = "light"
THEME = st.session_state.theme


# --------------------------------------------------------------------------- #
# THEME VARIABLES (light / dark) + GLOBAL STYLE
# --------------------------------------------------------------------------- #

def theme_root(theme: str) -> str:
    if theme == "dark":
        v = {
            "panel": "#171c24", "panel_tint": "#1e242e", "line": "#2b3340",
            "ink900": "#eef2f7", "ink700": "#c4ccd6", "ink500": "#93a0b0", "ink300": "#6b7585",
            "input_bg": "#141922", "chip_bg": "#1a2029", "tag_bg": "#2a1416", "tag_fg": "#ff8f9b",
            "app_bg": "#0e1116",
            "rh_bg": "rgba(226,40,59,.14)", "rh_bd": "rgba(226,40,59,.42)", "rh_fg": "#ff8893",
            "rl_bg": "rgba(31,181,126,.14)", "rl_bd": "rgba(31,181,126,.42)", "rl_fg": "#5fe0a8",
            "glow_blend": "screen",
            "sh_sm": "0 2px 10px rgba(0,0,0,.35)", "sh_md": "0 14px 36px rgba(0,0,0,.5)",
        }
    else:
        v = {
            "panel": "#ffffff", "panel_tint": "#f6f8fb", "line": "#e6eaf0",
            "ink900": "#171a1f", "ink700": "#333a44", "ink500": "#5b6675", "ink300": "#8a94a3",
            "input_bg": "#fcfdff", "chip_bg": "#fcfdff", "tag_bg": "#fdeef0", "tag_fg": "#c01a2b",
            "app_bg": "#f7f8fa",
            "rh_bg": "#fdecec", "rh_bd": "#f3bcbc", "rh_fg": "#a3121f",
            "rl_bg": "#e9f8f0", "rl_bd": "#b7e6cf", "rl_fg": "#0f7a52",
            "glow_blend": "multiply",
            "sh_sm": "0 2px 8px rgba(20,20,40,.06)", "sh_md": "0 12px 32px rgba(20,20,40,.10)",
        }
    return (
        ":root{"
        f"--panel:{v['panel']};--panel-tint:{v['panel_tint']};--line:{v['line']};"
        f"--ink-900:{v['ink900']};--ink-700:{v['ink700']};--ink-500:{v['ink500']};--ink-300:{v['ink300']};"
        f"--input-bg:{v['input_bg']};--chip-bg:{v['chip_bg']};--tag-bg:{v['tag_bg']};--tag-fg:{v['tag_fg']};"
        f"--rh-bg:{v['rh_bg']};--rh-bd:{v['rh_bd']};--rh-fg:{v['rh_fg']};"
        f"--rl-bg:{v['rl_bg']};--rl-bd:{v['rl_bd']};--rl-fg:{v['rl_fg']};"
        f"--glow-blend:{v['glow_blend']};--shadow-sm:{v['sh_sm']};--shadow-md:{v['sh_md']};"
        "--crimson:#c01a2b;--crimson-bright:#e2283b;--crimson-glow:#ff4d5e;"
        "--good:#15926b;--watch:#c9820a;--alert:#cf3030;--shadow-glow:0 12px 44px rgba(201,26,43,.30);"
        "--font:\"Segoe UI\",system-ui,-apple-system,Roboto,\"Helvetica Neue\",Arial,sans-serif;}"
        f"[data-testid=\"stAppViewContainer\"]{{background:{v['app_bg']};}}"
    )


MAIN_CSS = """
html, body, [class*="css"]{ font-family:var(--font); }
.stApp{ color:var(--ink-900); }

#MainMenu, [data-testid="stToolbar"], [data-testid="stDecoration"], footer{ display:none !important; }
[data-testid="stHeader"]{ background:transparent !important; height:0 !important; }
.block-container{ padding-top:1.2rem !important; padding-bottom:3rem !important; max-width:1200px !important; }

h1,h2,h3,h4{ color:var(--ink-900); letter-spacing:-.2px; }
[data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li{ color:var(--ink-700); }
[data-testid="stMarkdownContainer"] strong{ color:var(--ink-900); }
[data-testid="stMarkdownContainer"] code{ background:var(--panel-tint); color:var(--crimson-bright); padding:1px 6px; border-radius:5px; }
[data-testid="stCaptionContainer"] p{ color:var(--ink-300) !important; }

/* cursor-following glow */
#heart-cursor-glow{
  position:fixed; left:-500px; top:-500px; width:460px; height:460px; margin:-230px 0 0 -230px;
  border-radius:50%; pointer-events:none; z-index:9998; opacity:.5; mix-blend-mode:var(--glow-blend);
  background:radial-gradient(circle, rgba(226,40,59,.22) 0%, rgba(226,40,59,.10) 36%, rgba(226,40,59,0) 70%);
}

.page-head{ margin:22px 2px 4px; }
.page-head .kicker{ font-size:11px; font-weight:800; letter-spacing:1.4px; text-transform:uppercase; color:var(--tag-fg); }
.page-head h2{ margin:4px 0 2px; font-size:26px; font-weight:800; }
.page-head p{ margin:0; color:var(--ink-500); font-size:14.5px; }

[data-testid="stVerticalBlockBorderWrapper"]{
  background:var(--panel) !important; border:1px solid var(--line) !important;
  border-radius:16px !important; box-shadow:var(--shadow-sm) !important; animation:rise .45s ease both;
}
@keyframes rise{ from{opacity:0; transform:translateY(12px)} to{opacity:1; transform:none} }

[data-testid="stForm"]{
  background:var(--panel) !important; border:1px solid var(--line) !important;
  border-radius:16px !important; box-shadow:var(--shadow-sm) !important; padding:22px 24px !important;
}

button[kind="primary"], [data-testid="stBaseButton-primary"],
[data-testid="stFormSubmitButton"] button, [data-testid="stDownloadButton"] button{
  background:linear-gradient(135deg,#e2283b,#9a0f1c) !important; color:#fff !important;
  border:none !important; border-radius:12px !important; font-weight:750 !important;
  letter-spacing:.3px !important; box-shadow:var(--shadow-glow) !important;
  transition:transform .18s ease, filter .2s ease, box-shadow .2s ease !important;
}
button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover,
[data-testid="stFormSubmitButton"] button:hover, [data-testid="stDownloadButton"] button:hover{
  transform:translateY(-2px) !important; filter:brightness(1.07) !important; box-shadow:0 16px 50px rgba(201,26,43,.42) !important;
}
button[kind="secondary"], [data-testid="stBaseButton-secondary"]{
  background:var(--panel) !important; color:var(--ink-700) !important;
  border:1px solid var(--line) !important; border-radius:12px !important; font-weight:650 !important;
  transition:transform .18s ease, box-shadow .2s ease, color .2s ease !important;
}
button[kind="secondary"]:hover, [data-testid="stBaseButton-secondary"]:hover{
  color:var(--crimson-bright) !important; transform:translateY(-2px) !important; box-shadow:var(--shadow-md) !important;
}

[data-testid="stNumberInput"] input, [data-testid="stTextInput"] input{
  border-radius:10px !important; background:var(--input-bg) !important; color:var(--ink-900) !important;
}
[data-baseweb="select"] > div{ border-radius:10px !important; background:var(--input-bg) !important; }
[data-baseweb="select"]{ color:var(--ink-900); }
label, [data-testid="stWidgetLabel"] p{ font-weight:600 !important; color:var(--ink-700) !important; }

[data-testid="stMetric"]{ background:var(--panel-tint); border:1px solid var(--line); border-radius:12px; padding:14px 16px; }
[data-testid="stMetricValue"]{ color:var(--ink-900) !important; font-size:1.7rem !important; white-space:normal !important; overflow-wrap:anywhere; line-height:1.2; }
[data-testid="stMetricLabel"] p{ color:var(--ink-500) !important; }

[data-testid="stFileUploaderDropzone"]{ background:var(--panel-tint) !important; border:1.5px dashed var(--line) !important; border-radius:14px !important; }
[data-testid="stExpander"]{ border:1px solid var(--line) !important; border-radius:12px !important; background:var(--panel) !important; }
[data-testid="stAlert"]{ border-radius:12px !important; }
[data-testid="stSlider"] [role="slider"]{ background:#e2283b !important; border-color:#e2283b !important; }

.sect-tag{ display:inline-block; font-size:11px; font-weight:800; letter-spacing:1.2px; text-transform:uppercase;
  color:var(--tag-fg); background:var(--tag-bg); padding:5px 11px; border-radius:7px; margin-bottom:10px; }
.risk-head{ display:flex; align-items:center; gap:14px; padding:15px 17px; border-radius:14px; margin:2px 0 16px; }
.risk-high{ background:var(--rh-bg); border:1px solid var(--rh-bd); }
.risk-low{ background:var(--rl-bg); border:1px solid var(--rl-bd); }
.ricon{ width:46px; height:46px; border-radius:12px; display:grid; place-items:center; font-size:22px; color:#fff; flex:none; animation:beat 1.5s ease-in-out infinite; }
.risk-high .ricon{ background:linear-gradient(145deg,#e2283b,#8b0f1b); }
.risk-low .ricon{ background:linear-gradient(145deg,#1fb57e,#0f7a52); }
.risk-head .rt{ font-weight:800; font-size:16px; }
.risk-high .rt{ color:var(--rh-fg); } .risk-low .rt{ color:var(--rl-fg); }
.risk-head .rs{ font-size:13px; color:var(--ink-500); margin-top:1px; }
@keyframes beat{ 0%,100%{transform:scale(1)} 14%{transform:scale(1.16)} 28%{transform:scale(1)} 42%{transform:scale(1.1)} }

.metrics{ display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin:4px 0 14px; }
.metric{ background:var(--panel-tint); border:1px solid var(--line); border-radius:12px; padding:12px 8px; text-align:center; }
.metric .mv{ font-size:20px; font-weight:800; color:var(--ink-900); }
.metric .ml{ font-size:10.5px; color:var(--ink-500); text-transform:uppercase; letter-spacing:.5px; margin-top:3px; }

.chips{ display:flex; flex-direction:column; gap:9px; }
.chip{ display:flex; gap:12px; align-items:flex-start; padding:11px 13px; border-radius:12px; border:1px solid var(--line); background:var(--chip-bg); transition:transform .16s ease, box-shadow .16s ease; }
.chip:hover{ transform:translateX(3px); box-shadow:var(--shadow-sm); }
.chip .ci{ width:28px; height:28px; border-radius:8px; display:grid; place-items:center; font-size:14px; font-weight:800; color:#fff; flex:none; }
.chip.good .ci{ background:var(--good);} .chip.watch .ci{ background:var(--watch);} .chip.alert .ci{ background:var(--alert);}
.chip .ct{ font-weight:700; font-size:13.5px; color:var(--ink-900); } .chip .cx{ font-size:12.5px; color:var(--ink-500); }

.placeholder{ display:flex; flex-direction:column; align-items:center; justify-content:center; min-height:340px; text-align:center; color:var(--ink-300); padding:20px; }
.placeholder .ph-heart{ font-size:58px; animation:beat 1.6s ease-in-out infinite; filter:drop-shadow(0 10px 22px rgba(226,40,59,.3)); }
.placeholder .ph-t{ font-weight:700; color:var(--ink-500); margin-top:14px; font-size:15px; }
.placeholder .ph-x{ font-size:13px; max-width:280px; margin-top:4px; }

.glossary{ display:grid; grid-template-columns:1fr 1fr; gap:12px; }
.gl-item{ border:1px solid var(--line); border-radius:12px; padding:13px 15px; background:var(--panel); transition:transform .16s ease, box-shadow .16s ease; }
.gl-item:hover{ transform:translateY(-2px); box-shadow:var(--shadow-sm); }
.gl-item .gt{ font-weight:750; font-size:14px; color:var(--ink-900); }
.gl-item .gr{ font-size:12px; color:var(--tag-fg); font-weight:700; margin:2px 0 4px; }
.gl-item .gx{ font-size:12.5px; color:var(--ink-500); }

.footnote{ margin:26px 2px 4px; font-size:12px; color:var(--ink-300); text-align:center; line-height:1.6; }

::-webkit-scrollbar{ width:11px; height:11px; }
::-webkit-scrollbar-thumb{ background:linear-gradient(#e2283b,#9a0f1c); border-radius:10px; }
::-webkit-scrollbar-track{ background:var(--panel-tint); }
"""

st.markdown(f"<style>{theme_root(THEME)}</style>", unsafe_allow_html=True)
st.markdown(f"<style>{MAIN_CSS}</style>", unsafe_allow_html=True)


def set_page_background(filename: str, theme: str):
    """Inject the active page's background image with a theme-aware wash so the heart image shows through."""
    data = _b64(filename)
    if not data:
        return
    wash = "14,17,22" if theme == "dark" else "247,248,250"
    a1 = 0.80 if theme == "dark" else 0.74   # lower = image more visible
    a2 = a1 + 0.04
    css = (
        '[data-testid="stAppViewContainer"]{'
        "background-image:linear-gradient(rgba(" + wash + "," + str(a1) + "),rgba(" + wash + "," + str(a2) + ")),"
        "url(\"data:image/jpeg;base64," + data + "\");"
        "background-size:cover;background-attachment:fixed;background-repeat:no-repeat;"
        "background-position:calc(50% + var(--par-x,0px)) calc(50% + var(--par-y,0px));}"
    )
    st.markdown("<style>" + css + "</style>", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# HERO (two columns: ambient video + pumping heart on the right)
# --------------------------------------------------------------------------- #

HERO_TMPL = """
<!DOCTYPE html><html><head><meta charset="utf-8"><style>
*{margin:0;padding:0;box-sizing:border-box;}
.hero{position:relative;height:330px;overflow:hidden;border-radius:22px;
  font-family:"Segoe UI",system-ui,Roboto,Arial,sans-serif;
  background:linear-gradient(120deg,#160306,#3a0810 55%,#120305);
  box-shadow:0 18px 50px rgba(60,6,12,.35);}
.bgvid{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;opacity:.28;}
.ov{position:absolute;inset:0;z-index:1;
  background:linear-gradient(100deg,rgba(16,3,5,.96) 0%,rgba(30,5,8,.9) 38%,rgba(60,10,16,.35) 66%,rgba(60,10,16,.02) 100%);}
.rightwrap{position:absolute;top:0;right:0;height:100%;width:54%;z-index:2;overflow:hidden;}
.rightwrap video,.rightwrap .pumpimg{width:100%;height:100%;object-fit:cover;}
.rightwrap .pumpimg{background-size:cover;background-position:center;}
.rightfade{position:absolute;inset:0;background:linear-gradient(to right,#160306 0%,rgba(22,3,6,.72) 18%,rgba(22,3,6,0) 52%);}
.pumpglow{position:absolute;inset:0;background:radial-gradient(circle at 70% 50%,rgba(255,45,65,.20),transparent 55%);
  mix-blend-mode:screen;animation:glowpulse 1.5s ease-in-out infinite;}
@keyframes glowpulse{0%,100%{opacity:.45}50%{opacity:.9}}
.inner{position:relative;z-index:4;height:100%;padding:0 42px;max-width:56%;display:flex;flex-direction:column;justify-content:center;}
.brow{display:flex;align-items:center;gap:15px;}
.badge{width:50px;height:50px;border-radius:14px;display:grid;place-items:center;font-size:25px;
  background:linear-gradient(145deg,#e2283b,#8b0f1b);box-shadow:0 0 0 1px rgba(255,255,255,.14),0 10px 26px rgba(226,40,59,.5);
  animation:beat 1.4s ease-in-out infinite;}
@keyframes beat{0%,100%{transform:scale(1)}14%{transform:scale(1.16)}28%{transform:scale(1)}42%{transform:scale(1.1)}}
h1{color:#fff;font-size:34px;font-weight:800;letter-spacing:-.5px;line-height:1.05;text-shadow:0 2px 18px rgba(0,0,0,.5);}
.accent{color:#ff6b78;}
.sub{color:#f3d6da;font-size:14.5px;margin-top:9px;}
.pill{display:inline-flex;align-items:center;gap:9px;margin-top:16px;width:max-content;padding:8px 15px;border-radius:999px;
  font-size:12px;font-weight:700;letter-spacing:.4px;text-transform:uppercase;}
.pill.ok{color:#eafff3;background:rgba(13,42,28,.5);border:1px solid rgba(60,220,140,.42);}
.pill.warn{color:#fff3e0;background:rgba(60,40,8,.5);border:1px solid rgba(230,170,60,.5);}
.dot{width:9px;height:9px;border-radius:50%;}
.pill.ok .dot{background:#34d87f;animation:ping 1.6s infinite;}
.pill.warn .dot{background:#f0a93d;}
@keyframes ping{0%{box-shadow:0 0 0 0 rgba(52,216,127,.65)}70%{box-shadow:0 0 0 10px rgba(52,216,127,0)}100%{box-shadow:0 0 0 0 rgba(52,216,127,0)}}
.ecg{position:absolute;left:0;width:60%;bottom:16px;z-index:3;height:60px;opacity:.9;}
.ecg svg{width:100%;height:100%;}
.ecg path{fill:none;stroke:#ff5566;stroke-width:2.4;filter:drop-shadow(0 0 6px rgba(255,85,102,.9));
  stroke-dasharray:1400;stroke-dashoffset:1400;animation:draw 3.6s linear infinite;}
@keyframes draw{to{stroke-dashoffset:0;}}
</style></head><body>
<div class="hero">
  @@BGVID@@
  <div class="ov"></div>
  <div class="rightwrap">@@PUMP@@<div class="pumpglow"></div><div class="rightfade"></div></div>
  <div class="ecg"><svg viewBox="0 0 960 60" preserveAspectRatio="none">
    <path d="M0,30 L300,30 L326,30 L344,8 L362,52 L380,30 L700,30 L726,30 L744,6 L762,54 L780,30 L960,30"/>
  </svg></div>
  <div class="inner">
    <div class="brow"><div class="badge">🫀</div>
      <h1>Cardiac Disease <span class="accent">Prediction</span></h1></div>
    <div class="sub">Clinical risk screening powered by a machine learning model trained on cardiovascular health records.</div>
    <div class="pill @@STATUS_CLASS@@"><span class="dot"></span> @@STATUS_TEXT@@</div>
  </div>
</div></body></html>
"""


def render_hero(model_ready: bool):
    bg_video = _video_uri("heart_monitor.mp4")
    pump_video = _video_uri("heart_pump.mp4")
    poster = _img_uri("banner_ecg.jpg")

    bgvid = ('<video class="bgvid" autoplay muted loop playsinline><source src="' + bg_video + '" type="video/mp4"></video>') if bg_video else ""
    if pump_video:
        pump = '<video autoplay muted loop playsinline><source src="' + pump_video + '" type="video/mp4"></video>'
    elif poster:
        pump = '<div class="pumpimg" style="background-image:url(\'' + poster + "')\"></div>"
    else:
        pump = ""

    status_class = "ok" if model_ready else "warn"
    status_text = "System online &nbsp;&bull;&nbsp; Model ready" if model_ready else "System online &nbsp;&bull;&nbsp; Model not loaded"
    html = (HERO_TMPL.replace("@@BGVID@@", bgvid).replace("@@PUMP@@", pump)
                     .replace("@@STATUS_CLASS@@", status_class).replace("@@STATUS_TEXT@@", status_text))
    components.html(html, height=350)


def inject_mouse_tracking():
    js = """
<script>
(function(){
  try{
    var pdoc = window.parent.document, pwin = window.parent;
    if(!pdoc.getElementById('heart-cursor-glow')){
      var g = pdoc.createElement('div'); g.id='heart-cursor-glow'; pdoc.body.appendChild(g);
    }
    if(!pwin.__cardioGlow){
      pwin.__cardioGlow = true;
      pwin.addEventListener('mousemove', function(e){
        var el = pdoc.getElementById('heart-cursor-glow');
        if(el){ el.style.left = e.clientX + 'px'; el.style.top = e.clientY + 'px'; }
        pdoc.documentElement.style.setProperty('--par-x', ((e.clientX/pwin.innerWidth - .5) * -14) + 'px');
        pdoc.documentElement.style.setProperty('--par-y', ((e.clientY/pwin.innerHeight - .5) * -14) + 'px');
      }, {passive:true});
    }
  }catch(err){}
})();
</script>
"""
    components.html(js, height=0)


# --------------------------------------------------------------------------- #
# SMALL HTML HELPERS
# --------------------------------------------------------------------------- #

def page_head(kicker, title, subtitle):
    st.markdown(f'<div class="page-head"><div class="kicker">{kicker}</div><h2>{title}</h2><p>{subtitle}</p></div>',
                unsafe_allow_html=True)


def chips_html(observations):
    icon = {"good": "✓", "watch": "~", "alert": "!"}
    items = ""
    for level, title, text in observations:
        lvl = level if level in icon else "watch"
        items += (f'<div class="chip {lvl}"><div class="ci">{icon[lvl]}</div>'
                  f'<div><div class="ct">{title}</div><div class="cx">{text}</div></div></div>')
    return f'<div class="chips">{items}</div>'


def metrics_html(cards):
    body = "".join(f'<div class="metric"><div class="mv">{v}</div><div class="ml">{l}</div></div>' for v, l in cards)
    return f'<div class="metrics">{body}</div>'


def risk_head_html(high: bool):
    if high:
        return ('<div class="risk-head risk-high"><div class="ricon">🚨</div>'
                '<div><div class="rt">Elevated cardiovascular risk</div>'
                '<div class="rs">Further clinical evaluation and diagnostic testing are recommended.</div></div></div>')
    return ('<div class="risk-head risk-low"><div class="ricon">✓</div>'
            '<div><div class="rt">Low cardiovascular risk</div>'
            '<div class="rs">Patient parameters are within normal indicators.</div></div></div>')


# plotly helpers (dark-aware font colour)
def _ink():
    return "#c4ccd6" if THEME == "dark" else "#333a44"


def _grid():
    return "#2b3340" if THEME == "dark" else "#eef0f3"


def gauge_figure(risk_pct: float):
    high = risk_pct >= 50
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=risk_pct,
        number={"suffix": "%", "font": {"size": 40, "color": "#ff8893" if (high and THEME == "dark") else ("#8b0f1b" if high else ("#5fe0a8" if THEME == "dark" else "#0f7a52"))}},
        gauge={"axis": {"range": [0, 100], "tickcolor": _ink(), "tickwidth": 1},
               "bar": {"color": "#e2283b" if high else "#1fb57e", "thickness": 0.28},
               "bgcolor": "rgba(0,0,0,0)", "borderwidth": 0,
               "steps": [{"range": [0, 40], "color": "rgba(31,181,126,.18)"},
                         {"range": [40, 70], "color": "rgba(240,169,61,.20)"},
                         {"range": [70, 100], "color": "rgba(226,40,59,.20)"}],
               "threshold": {"line": {"color": "#e2283b", "width": 3}, "thickness": 0.85, "value": risk_pct}}))
    fig.update_layout(height=230, margin=dict(l=22, r=22, t=12, b=8),
                      paper_bgcolor="rgba(0,0,0,0)", font={"color": _ink(), "family": "Segoe UI"})
    return fig


def themed_bar(names, vals, title):
    fig = go.Figure(go.Bar(x=vals, y=names, orientation="h",
                           marker={"color": vals, "colorscale": [[0, "#f3a1ab"], [1, "#c01a2b"]]}))
    fig.update_layout(height=max(300, 26 * len(names) + 90), title=title, margin=dict(l=10, r=20, t=50, b=20),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font={"color": _ink(), "family": "Segoe UI"},
                      xaxis={"gridcolor": _grid()}, yaxis={"gridcolor": "rgba(0,0,0,0)"})
    return fig


def roc_figure(fpr, tpr):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", line={"color": "#e2283b", "width": 3},
                             fill="tozeroy", fillcolor="rgba(226,40,59,.10)", name="Model"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line={"color": "#8a94a3", "dash": "dash"}, name="Chance"))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"color": _ink()},
                      xaxis={"title": "False positive rate", "gridcolor": _grid()},
                      yaxis={"title": "True positive rate", "gridcolor": _grid()})
    return fig


def confusion_figure(cm):
    fig = go.Figure(go.Heatmap(z=cm, x=["Pred: No", "Pred: Yes"], y=["Actual: No", "Actual: Yes"],
                               text=cm, texttemplate="%{text}", textfont={"size": 18}, colorscale="Reds", showscale=False))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", font={"color": _ink()})
    return fig


def hist_figure(probs):
    fig = go.Figure(go.Histogram(x=probs, nbinsx=25, marker={"color": "#e2283b", "line": {"color": "#9a0f1c", "width": 1}}))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), bargap=0.05,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"color": _ink()},
                      xaxis={"title": "Predicted risk (%)", "gridcolor": _grid()},
                      yaxis={"title": "Patients", "gridcolor": _grid()})
    return fig


# --------------------------------------------------------------------------- #
# HERO + NAV
# --------------------------------------------------------------------------- #

model_ready, model_msg = be.model_status()
render_hero(model_ready)

PAGES = [("assess", "🩺 Assessment"), ("insights", "📊 Insights"),
         ("guide", "📖 Guide"), ("train", "🧠 Train Model")]

st.write("")
nav_cols = st.columns([1, 1, 1, 1, 0.8])
for i, (key, label) in enumerate(PAGES):
    with nav_cols[i]:
        is_active = st.session_state.page == key
        if st.button(label, key=f"nav_{key}", use_container_width=True,
                     type="primary" if is_active else "secondary"):
            st.session_state.page = key
            st.rerun()
with nav_cols[4]:
    toggle_label = "🌙 Dark" if THEME == "light" else "☀️ Light"
    if st.button(toggle_label, key="theme_toggle", use_container_width=True, type="secondary"):
        st.session_state.theme = "dark" if THEME == "light" else "light"
        st.rerun()

page = st.session_state.page
set_page_background(PAGE_BG.get(page, "bg_assessment.jpg"), THEME)

DISCLAIMER = ("This tool provides a statistical risk estimate for screening and education only. "
              "It is not a medical diagnosis. Always consult a qualified clinician for any health decision.")


# =========================================================================== #
# PAGE: ASSESSMENT
# =========================================================================== #

def page_assessment():
    page_head("Patient risk screening", "Cardiovascular Assessment",
              "Enter the patient's clinical parameters to estimate cardiovascular disease risk.")

    if not model_ready:
        st.warning("Model file not found. Add cardio_model.pkl next to the app, or build one on the "
                   "Train Model tab. Details: " + str(model_msg))

    col_form, col_res = st.columns([1.1, 0.9], gap="large")

    with col_form:
        with st.form("patient_form"):
            st.markdown('<span class="sect-tag">Patient clinical data</span>', unsafe_allow_html=True)
            r1 = st.columns(3)
            with r1[0]:
                age = st.number_input("Age (Years)", min_value=18, max_value=100, value=50)
            with r1[1]:
                gender_label = st.selectbox("Gender", ["Female", "Male"])
            with r1[2]:
                height = st.number_input("Height (cm)", min_value=100, max_value=230, value=165)
            r2 = st.columns(3)
            with r2[0]:
                weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0)
            with r2[1]:
                ap_hi = st.number_input("Systolic BP (ap_hi)", min_value=70, max_value=240, value=120)
            with r2[2]:
                ap_lo = st.number_input("Diastolic BP (ap_lo)", min_value=40, max_value=160, value=80)
            r3 = st.columns(3)
            with r3[0]:
                cholesterol_label = st.selectbox("Cholesterol Level", ["Normal", "Above Normal", "Well Above Normal"])
            with r3[1]:
                gluc_label = st.selectbox("Glucose Level", ["Normal", "Above Normal", "Well Above Normal"])
            with r3[2]:
                smoke_label = st.selectbox("Smoker?", ["No", "Yes"])
            r4 = st.columns(3)
            with r4[0]:
                alco_label = st.selectbox("Alcohol Consumption?", ["No", "Yes"])
            with r4[1]:
                active_label = st.selectbox("Physically Active?", ["Yes", "No"])
            with r4[2]:
                st.write("")
            submit_btn = st.form_submit_button("🔍 Assess Cardiovascular Risk", use_container_width=True, type="primary")

    # ----- original mapping and prediction logic, kept exactly as before -----
    gender = 1 if gender_label == "Female" else 2
    mapping_level = {"Normal": 1, "Above Normal": 2, "Well Above Normal": 3}
    cholesterol = mapping_level[cholesterol_label]
    gluc = mapping_level[gluc_label]
    smoke = 1 if smoke_label == "Yes" else 0
    alco = 1 if alco_label == "Yes" else 0
    active = 1 if active_label == "Yes" else 0

    with col_res:
        if not submit_btn:
            with st.container(border=True):
                st.markdown('<div class="placeholder"><div class="ph-heart">🫀</div>'
                            '<div class="ph-t">Awaiting assessment</div>'
                            '<div class="ph-x">Fill in the patient parameters and run the assessment to see the risk score and health indicators.</div></div>',
                            unsafe_allow_html=True)
            return
        if ap_hi < ap_lo:
            st.error("⚠️ Systolic BP (ap_hi) cannot be lower than Diastolic BP (ap_lo). Please fix inputs.")
            return

        patient_data = {"age": age, "gender": gender, "height": height, "weight": weight,
                        "ap_hi": ap_hi, "ap_lo": ap_lo, "cholesterol": cholesterol,
                        "gluc": gluc, "smoke": smoke, "alco": alco, "active": active}
        try:
            result = be.predict_heart_disease(patient_data)
        except Exception as e:
            st.error(f"Prediction Error: {e}")
            return

        risk_pct = result["risk_percentage"]
        high = result["has_disease"]
        ind = be.health_indicators(patient_data)

        with st.container(border=True):
            st.markdown('<span class="sect-tag">Assessment result</span>', unsafe_allow_html=True)
            st.markdown(risk_head_html(high), unsafe_allow_html=True)
            st.plotly_chart(gauge_figure(risk_pct), use_container_width=True, config={"displayModeBar": False})
            cards = [(f"{risk_pct}%", "Risk"), (f"{result['confidence']}%", "Confidence"),
                     (f"{ind['bmi']:.1f}", "BMI"), (f"{ind['pulse_pressure']}", "Pulse Pr.")]
            st.markdown(metrics_html(cards), unsafe_allow_html=True)
            st.markdown(chips_html(ind["observations"]), unsafe_allow_html=True)

    st.markdown(f'<div class="footnote">{DISCLAIMER}</div>', unsafe_allow_html=True)


# =========================================================================== #
# PAGE: INSIGHTS
# =========================================================================== #

def page_insights():
    page_head("Under the hood", "Model Insights",
              "What the live model is, which factors it relies on most, and how well it scores on your data.")

    if not model_ready:
        st.warning("Model file not found, so there is nothing to inspect yet. Build a model on the Train Model tab. "
                   "Details: " + str(model_msg))
        return

    info = be.model_info()
    meta = info.get("meta", {}) or {}

    c = st.columns(4)
    c[0].metric("Algorithm", short_algo(info["algorithm"]))
    c[1].metric("Features", len(info["features"]))
    c[2].metric("Scaler", info["scaler"].replace("Scaler", " Scaler"))
    c[3].metric("Model size", f"{info['file_mb']} MB" if info.get("file_mb") is not None else "n/a")

    acc, auc = meta.get("accuracy"), meta.get("roc_auc")
    if isinstance(acc, (int, float)) and isinstance(auc, (int, float)):
        st.caption(f"Reported at training time: accuracy {acc*100:.1f}%, ROC AUC {auc:.3f}, "
                   f"on {meta.get('rows', 'n/a')} rows ({meta.get('trained_at', 'n/a')}).")
    else:
        st.caption("This model file has no stored training scores. Use the evaluation section below to measure them "
                   "on a labelled dataset.")

    left, right = st.columns([1.1, 0.9], gap="large")
    with left:
        with st.container(border=True):
            st.markdown('<span class="sect-tag">Feature importance</span>', unsafe_allow_html=True)
            pairs = be.feature_importance()
            if pairs:
                names = [p[0] for p in pairs]
                vals = [round(float(p[1]), 4) for p in pairs]
                st.plotly_chart(themed_bar(names, vals, "How much each factor drives the prediction"),
                                use_container_width=True, config={"displayModeBar": False})
            else:
                st.info("This model type does not expose feature importances.")
    with right:
        with st.container(border=True):
            st.markdown('<span class="sect-tag">Model card</span>', unsafe_allow_html=True)
            st.markdown(f"**Algorithm:** {short_algo(info['algorithm'])}  \n"
                        f"**Scaler:** {info['scaler']}  \n"
                        f"**Trained on:** {meta.get('rows', 'n/a')} rows  \n"
                        f"**Last trained:** {meta.get('trained_at', 'n/a')}  \n"
                        f"**Model file size:** {info.get('file_mb', 'n/a')} MB")
            st.markdown("**Input features**")
            st.markdown(", ".join(f"`{f}`" for f in info["features"]))

    # ---- Evaluate the live model on a dataset (real accuracy / ROC / charts) ----
    with st.container(border=True):
        st.markdown('<span class="sect-tag">Measure performance on a dataset</span>', unsafe_allow_html=True)
        st.markdown("Upload a labelled `.csv` / `.xlsx` (same columns as training, including the `cardio` target) "
                    "to score the current live model. Nothing is retrained or saved.")
        colu = st.columns([1, 1], gap="large")
        with colu[0]:
            st.download_button("⬇ Download CSV template", data=be.dataset_template(),
                               file_name="cardio_template.csv", mime="text/csv", key="eval_tmpl")
        with colu[1]:
            eval_file = st.file_uploader("Dataset to evaluate", type=["csv", "txt", "xlsx", "xls"], key="eval_up")

        if eval_file is not None:
            try:
                df_raw = be.read_table(eval_file.getvalue(), eval_file.name)
            except Exception as e:
                st.error(f"Could not read the file: {e}")
                df_raw = None
            if df_raw is not None:
                try:
                    Xe, ye, rep = be.prepare_training_data(df_raw)
                except ValueError as e:
                    st.warning(f"Dataset not usable: {e}")
                else:
                    st.caption(f"Usable rows: {rep['rows_used']:,} (positive rate {rep['positive_rate']:.1f}%). "
                               + " ".join(rep.get("notes", [])))
                    if st.button("📏 Evaluate current model", type="primary", key="eval_btn"):
                        with st.spinner("Scoring the live model..."):
                            st.session_state["eval_result"] = be.evaluate_live_model(Xe, ye)

    ev = st.session_state.get("eval_result")
    if ev:
        with st.container(border=True):
            st.markdown('<span class="sect-tag">Performance on your data</span>', unsafe_allow_html=True)
            mc = st.columns(5)
            mc[0].metric("Accuracy", f"{ev['accuracy']*100:.1f}%")
            mc[1].metric("Precision", f"{ev['precision']*100:.1f}%")
            mc[2].metric("Recall", f"{ev['recall']*100:.1f}%")
            mc[3].metric("F1", f"{ev['f1']*100:.1f}%")
            mc[4].metric("ROC AUC", f"{ev['roc_auc']:.3f}")
            st.caption(f"Measured on {ev.get('n_rows', 'n/a'):,} rows.")
        g = st.columns(2, gap="large")
        with g[0]:
            with st.container(border=True):
                st.markdown('<span class="sect-tag">ROC curve</span>', unsafe_allow_html=True)
                st.plotly_chart(roc_figure(ev["fpr"], ev["tpr"]), use_container_width=True, config={"displayModeBar": False})
        with g[1]:
            with st.container(border=True):
                st.markdown('<span class="sect-tag">Confusion matrix</span>', unsafe_allow_html=True)
                st.plotly_chart(confusion_figure(ev["cm"]), use_container_width=True, config={"displayModeBar": False})
        with st.container(border=True):
            st.markdown('<span class="sect-tag">Predicted risk distribution</span>', unsafe_allow_html=True)
            st.plotly_chart(hist_figure(ev["probabilities"]), use_container_width=True, config={"displayModeBar": False})

    st.markdown(f'<div class="footnote">{DISCLAIMER}</div>', unsafe_allow_html=True)


# =========================================================================== #
# PAGE: GUIDE
# =========================================================================== #

def page_guide():
    page_head("How to read it", "Guide and Reference",
              "What each input means, the healthy reference ranges, and how to use this dashboard.")
    glossary = [
        ("Age", "18 to 100 yrs", "Risk rises with age; one of the strongest non-modifiable factors."),
        ("Systolic BP (ap_hi)", "under 120 mmHg", "Pressure while the heart beats. 130+ is stage 1 hypertension."),
        ("Diastolic BP (ap_lo)", "under 80 mmHg", "Pressure between beats. 80+ counts toward hypertension."),
        ("BMI (height and weight)", "18.5 to 24.9", "Body mass index; 25+ overweight, 30+ obese."),
        ("Cholesterol", "Normal", "Higher levels relate to arterial plaque and heart disease."),
        ("Glucose", "Normal", "Elevated glucose signals diabetes risk, a cardiovascular factor."),
        ("Smoking", "No", "Smoking strongly and directly raises cardiovascular risk."),
        ("Physical activity", "Yes", "Regular activity is protective for the heart."),
    ]
    with st.container(border=True):
        st.markdown('<span class="sect-tag">Input glossary and healthy ranges</span>', unsafe_allow_html=True)
        items = ""
        for title, rng, desc in glossary:
            items += (f'<div class="gl-item"><div class="gt">{title}</div>'
                      f'<div class="gr">Healthy: {rng}</div><div class="gx">{desc}</div></div>')
        st.markdown(f'<div class="glossary">{items}</div>', unsafe_allow_html=True)

    left, right = st.columns(2, gap="large")
    with left:
        with st.container(border=True):
            st.markdown('<span class="sect-tag">How to use</span>', unsafe_allow_html=True)
            st.markdown("1. Open **Assessment** and enter the patient's clinical values.  \n"
                        "2. Press **Assess Cardiovascular Risk** for a risk score, gauge and observations.  \n"
                        "3. Open **Insights** to see what the model weighs most and to measure accuracy on your data.  \n"
                        "4. Open **Train Model** to retrain on a new dataset when you have more data.")
    with right:
        with st.container(border=True):
            st.markdown('<span class="sect-tag">Important</span>', unsafe_allow_html=True)
            st.markdown("This is a **screening and education** tool, not a diagnostic device. A risk percentage is a "
                        "statistical estimate from population data, not a statement about any one person's health. "
                        "Decisions about testing, diagnosis or treatment must be made by a qualified clinician.")
            st.caption("Blood pressure categories follow the AHA 2017 reference ranges.")
    st.markdown(f'<div class="footnote">{DISCLAIMER}</div>', unsafe_allow_html=True)


# =========================================================================== #
# PAGE: TRAIN MODEL
# =========================================================================== #

def page_train():
    page_head("Keep the model current", "Train on New Data",
              "Upload a labelled dataset to train a fresh model, review its scores, then deploy it when you are happy.")
    st.info("The current live model is never changed until you explicitly deploy a new one. "
            "When you deploy, the previous model is automatically backed up as cardio_model.prev.pkl.")

    top = st.columns([1, 1], gap="large")
    with top[0]:
        with st.container(border=True):
            st.markdown('<span class="sect-tag">1. Dataset</span>', unsafe_allow_html=True)
            st.markdown("Need the layout? Download a tiny template with the exact columns.")
            st.download_button("⬇ Download CSV template", data=be.dataset_template(),
                               file_name="cardio_template.csv", mime="text/csv")
            uploaded = st.file_uploader("Upload a .csv or .xlsx dataset", type=["csv", "txt", "xlsx", "xls"])
    with top[1]:
        with st.container(border=True):
            st.markdown('<span class="sect-tag">2. Training settings</span>', unsafe_allow_html=True)
            algorithm = st.selectbox("Algorithm", be.ALGORITHMS)
            n_estimators = st.slider("Trees / iterations", 50, 500, 200, step=50)
            max_depth = st.slider("Max depth (0 = unlimited)", 0, 30, 0)
            test_size = st.slider("Hold-out test share", 0.1, 0.4, 0.2, step=0.05)

    if uploaded is not None:
        try:
            df_raw = be.read_table(uploaded.getvalue(), uploaded.name)
        except Exception as e:
            st.error(f"Could not read the file: {e}")
            df_raw = None
        if df_raw is not None:
            try:
                X, y, report = be.prepare_training_data(df_raw)
            except ValueError as e:
                st.warning(f"Dataset not usable yet: {e}")
            else:
                with st.container(border=True):
                    st.markdown('<span class="sect-tag">Data check</span>', unsafe_allow_html=True)
                    rc = st.columns(4)
                    rc[0].metric("Rows in file", f"{report['rows_in']:,}")
                    rc[1].metric("Usable rows", f"{report['rows_used']:,}")
                    rc[2].metric("Dropped (missing)", f"{report.get('dropped_missing', 0):,}")
                    rc[3].metric("Dropped (outliers)", f"{report.get('dropped_outliers', 0):,}")
                    st.caption(f"Positive (disease) rate: {report['positive_rate']:.1f}%. " + " ".join(report.get("notes", [])))
                if st.button("🧠 Train model", type="primary"):
                    with st.spinner("Training and evaluating on a hold-out split. This can take a moment..."):
                        res = be.train_model(X, y, algorithm=algorithm, n_estimators=n_estimators,
                                             max_depth=max_depth, test_size=test_size)
                    st.session_state["train_result"] = res
                    st.success("Training complete. Review the scores below.")

    res = st.session_state.get("train_result")
    if res:
        m = res["metrics"]
        with st.container(border=True):
            st.markdown('<span class="sect-tag">3. New model performance</span>', unsafe_allow_html=True)
            mc = st.columns(5)
            mc[0].metric("Accuracy", f"{m['accuracy']*100:.1f}%")
            mc[1].metric("Precision", f"{m['precision']*100:.1f}%")
            mc[2].metric("Recall", f"{m['recall']*100:.1f}%")
            mc[3].metric("F1", f"{m['f1']*100:.1f}%")
            mc[4].metric("ROC AUC", f"{m['roc_auc']:.3f}")
            cur = res.get("current_metrics")
            if cur:
                st.caption(f"Versus the live model on the same split: accuracy {(m['accuracy']-cur['accuracy'])*100:+.1f} pts, "
                           f"ROC AUC {m['roc_auc']-cur['roc_auc']:+.3f}.")
        g = st.columns(2, gap="large")
        with g[0]:
            with st.container(border=True):
                st.markdown('<span class="sect-tag">ROC curve</span>', unsafe_allow_html=True)
                st.plotly_chart(roc_figure(m["fpr"], m["tpr"]), use_container_width=True, config={"displayModeBar": False})
        with g[1]:
            with st.container(border=True):
                st.markdown('<span class="sect-tag">Confusion matrix</span>', unsafe_allow_html=True)
                st.plotly_chart(confusion_figure(m["cm"]), use_container_width=True, config={"displayModeBar": False})
        with st.container(border=True):
            st.markdown('<span class="sect-tag">Feature importance (new model)</span>', unsafe_allow_html=True)
            imp = res.get("importances") or []
            if imp:
                names = [p[0] for p in imp]
                vals = [round(float(p[1]), 4) for p in imp]
                st.plotly_chart(themed_bar(names, vals, ""), use_container_width=True, config={"displayModeBar": False})
        with st.container(border=True):
            st.markdown('<span class="sect-tag">4. Deploy</span>', unsafe_allow_html=True)
            st.markdown("Make this trained model the live one used on the Assessment tab.")
            confirm = st.checkbox("I understand this replaces the live model (the current one is backed up).")
            if st.button("🚀 Deploy this model", type="primary", disabled=not confirm):
                try:
                    be.save_artifacts(res["artifacts"])
                    st.success("Deployed. The Assessment tab now uses this model. Previous model saved as cardio_model.prev.pkl.")
                    st.session_state.pop("train_result", None)
                except Exception as e:
                    st.error(f"Could not save the model: {e}")

    st.markdown(f'<div class="footnote">{DISCLAIMER}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# ROUTER
# --------------------------------------------------------------------------- #

if page == "assess":
    page_assessment()
elif page == "insights":
    page_insights()
elif page == "guide":
    page_guide()
elif page == "train":
    page_train()

inject_mouse_tracking()