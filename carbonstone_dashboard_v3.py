"""
CarbonStone Digital Twin v3 - conceptual process simulator
==========================================================
Run:   pip install streamlit plotly pandas numpy
       streamlit run carbonstone_dashboard_v3.py
Theme: copy config.toml to a folder named .streamlit next to this file (light theme).

What is in here
  * Stoichiometric capacity from reactive CaO / Ca(OH)2 (theoretical, exact chemistry)
  * Time-stepped carbonation model with ASSUMED absorption-efficiency parameters
  * Adaptive (stage-switching, endpoint-stopping) vs fixed-flow control, actually simulated
  * Feedstock-variability study (fixed recipe vs adaptive dosing)
  * Auto-blend: computes the CKD:CCR ratio that hits a target capacity
  * Reactor playback of the simulated run: 3D (three.js, needs internet) with a 2D fallback

What this is NOT: a validated kinetic model, CFD, AI model, or measured performance.
Every number is tagged Theoretical, Assumed or Simulated in the UI.
"""
import json
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from plotly.subplots import make_subplots

# ------------------------------------------------------------
# Constants (fixed chemistry)
# ------------------------------------------------------------
M_CO2, M_CAO, M_CAOH2, M_CACO3 = 44.01, 56.08, 74.09, 100.09
K_CAO = M_CO2 / M_CAO      # 0.785 kg CO2 per kg CaO
K_CAOH = M_CO2 / M_CAOH2   # 0.594 kg CO2 per kg Ca(OH)2
R_GAS = 0.082057           # L atm / (mol K)
PH_FINISH_SIGNAL = 10.5    # pH below this counts as the "pH finished" signal
BREAKTHROUGH_ETA = 0.35    # absorption efficiency below this counts as CO2 breakthrough

# Palette (light instrument-panel theme)
INK, MUTED, BORDER = "#14202B", "#5B6B7A", "#D3DBE3"
BLUE, GREEN, AMBER, RED, PURPLE = "#2F6FB3", "#1E8E5A", "#D98E04", "#C8372D", "#6B5FC7"

st.set_page_config(page_title="CarbonStone | Intelligent Mineralisation", page_icon="🌱",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown(f"""
<style>
.stApp {{ background: #EDF1F5; color: {INK}; }}
.block-container {{padding-top: 1.3rem; padding-bottom: 2rem; max-width: 1500px;}}
section[data-testid="stSidebar"] {{ background: #FFFFFF; border-right: 1px solid {BORDER}; }}
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp p, .stApp li, .stApp label {{ color: {INK}; }}
[data-testid="stMetric"] {{
  background: #FFFFFF; border: 1px solid {BORDER}; border-left: 4px solid {BLUE};
  padding: 12px 14px; border-radius: 10px;
}}
[data-testid="stMetricLabel"] p {{ font-size: .82rem; color: {MUTED}; }}
[data-testid="stMetricValue"] {{ color: {INK}; }}
button[data-baseweb="tab"] {{ font-weight: 600; }}
button[data-baseweb="tab"][aria-selected="true"] {{ color: {BLUE}; }}
.hero {{ background:#FFFFFF; border:1px solid {BORDER}; border-left:6px solid {GREEN}; border-radius:12px; padding:14px 18px; margin-bottom:12px; }}
.hero h1 {{ margin:0; font-size:1.7rem; letter-spacing:.3px; }}
.hero p {{ margin:2px 0 0; color:{MUTED}; }}
.pill {{display:inline-block; padding:2px 9px; margin-right:6px; border-radius:999px; font-size:.78rem; font-weight:600; border:1px solid;}}
.pill-t {{color:#1F5FA8; border-color:#A9C6E6; background:#E3EEF9;}}
.pill-a {{color:#8A5A00; border-color:#E6C779; background:#FBF1D9;}}
.pill-s {{color:#14693F; border-color:#9ED3B6; background:#E0F3E9;}}
.banner {{border:1px solid {BORDER}; background:#FFFFFF; border-radius:10px; padding:10px 14px; margin:6px 0 14px;}}
</style>
""", unsafe_allow_html=True)


def pill(kind: str, text: str) -> str:
    return f'<span class="pill pill-{kind}">{text}</span>'


# ------------------------------------------------------------
# Model functions (scalar math in the loop: fast)
# ------------------------------------------------------------
def ph_scalar(X):
    """Indicative slurry pH versus conversion. Assumed shape, to be calibrated with a real probe."""
    if X <= 0.88:
        return 12.4
    return max(12.4 - (X - 0.88) / (0.995 - 0.88) * (12.4 - 8.0), 8.0)


def eta_scalar(X, eta_max, p, x_cap):
    """Fraction of injected CO2 absorbed at conversion X (assumed curve)."""
    if X >= x_cap:
        return 0.0
    return max(eta_max * (1.0 - (X / x_cap) ** p), 0.0)


def ph_of(X):
    return np.array([ph_scalar(v) for v in np.asarray(X, dtype=float)])


def eta_of(X, eta_max, p, x_cap):
    return np.array([eta_scalar(v, eta_max, p, x_cap) for v in np.asarray(X, dtype=float)])


@st.cache_data(show_spinner=False)
def simulate(n0, flow, y, T_K, duration_min, adaptive, x_target, x_switch, ratio2,
             eta_max, p, x_cap, dt=2.0):
    """Time-stepped carbonation of n0 mol of reactive lime (as mol CO2 capacity).

    adaptive=True : stage 1 at full flow, stage 2 at ratio2*flow once X >= x_switch,
                    stop when uptake reaches target OR pH and breakthrough signals agree.
    adaptive=False: constant flow for the whole run time (fixed-recipe reference).
    """
    n0, flow, y, T_K = float(n0), float(flow), float(y), float(T_K)
    steps = int(duration_min * 60 / dt)
    mol_per_L = 1.0 / (R_GAS * T_K)
    X = ab = inj = 0.0
    stopped, switched, stop_t = False, False, None
    T, XS, F, STG, INJ, AB, PH, OUT, ETA, S1, S2, S3 = ([] for _ in range(12))
    events = [(0.0, "Run starts at full flow" if adaptive else "Run starts at fixed flow")]
    limit = min(x_target, x_cap) if adaptive else x_cap

    for i in range(steps + 1):
        t = i * dt
        eta = eta_scalar(X, eta_max, p, x_cap)
        ph = ph_scalar(X)
        s1, s2, s3 = ph < PH_FINISH_SIGNAL, eta < BREAKTHROUGH_ETA, X >= x_target - 1e-3

        if adaptive and not stopped and (s3 or (s1 and s2)):
            stopped, stop_t = True, t
            events.append((t, "Valve closed: " + ("uptake reached target" if s3 else "pH and CO₂ breakthrough agree")))

        if adaptive:
            stage = 0 if stopped else (2 if X >= x_switch else 1)
            if stage == 2 and not switched:
                switched = True
                events.append((t, "Stage 2: flow reduced near endpoint"))
        else:
            stage = 1
        f = 0.0 if stage == 0 else (flow * ratio2 if stage == 2 else flow)
        out = (y * (1 - eta) / (1 - y * eta)) if f > 0 else 0.0

        T.append(t); XS.append(X); F.append(f); STG.append(stage); INJ.append(inj); AB.append(ab)
        PH.append(ph); OUT.append(out); ETA.append(eta); S1.append(s1); S2.append(s2); S3.append(s3)

        inj_dt = f * (dt / 60.0) * y * mol_per_L
        ab_dt = min(eta * inj_dt, max(0.0, limit - X) * n0)
        inj += inj_dt
        ab += ab_dt
        X = ab / n0

    if adaptive and not stopped:
        events.append((steps * dt, "Time limit reached before endpoint"))
    if not adaptive:
        events.append((steps * dt, "Fixed run ends (no endpoint detection)"))

    return {
        "t": np.array(T), "X": np.array(XS), "f": np.array(F), "stage": np.array(STG),
        "inj": np.array(INJ), "ab": np.array(AB), "ph": np.array(PH), "out": np.array(OUT),
        "eta": np.array(ETA), "s1": np.array(S1), "s2": np.array(S2), "s3": np.array(S3),
        "events": events, "stop_t": stop_t, "n0": n0,
        "end_t": stop_t if stop_t is not None else steps * dt,
        "inj_end": inj, "ab_end": ab,
    }


def summarise(sim, n0):
    inj_kg = sim["inj_end"] * M_CO2 / 1000
    up_kg = sim["ab_end"] * M_CO2 / 1000
    return dict(
        inj_kg=inj_kg, up_kg=up_kg, off_kg=max(inj_kg - up_kg, 0.0),
        util=(up_kg / inj_kg) if inj_kg > 0 else 0.0,
        conv=sim["ab_end"] / n0, t_end_min=sim["end_t"] / 60.0,
        caco3_kg=up_kg * M_CACO3 / M_CO2,
    )


def style(fig, h=380, **kw):
    fig.update_layout(
        template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=INK), height=h, margin=dict(t=40, b=30, l=10, r=10),
        legend=dict(orientation="h", y=1.14, x=0), **kw,
    )
    fig.update_xaxes(gridcolor="#E1E7ED", zerolinecolor="#C9D2DB")
    fig.update_yaxes(gridcolor="#E1E7ED", zerolinecolor="#C9D2DB")
    return fig


def mmss(seconds):
    s = int(round(seconds))
    return f"{s // 60:d}:{s % 60:02d}"


# ------------------------------------------------------------
# Header
# ------------------------------------------------------------
st.markdown(
    '<div class="hero"><h1>🌱 CARBONSTONE</h1>'
    '<p>Intelligent mineralisation control system · digital twin prototype · '
    'feedstock intelligence · adaptive carbonation · live reactor playback</p></div>',
    unsafe_allow_html=True)

# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Process configuration")
    st.caption("Every default is an illustrative assumption. Replace with measured values (free-lime test, TGA, XRF).")

    st.subheader("Feedstock (dry basis)")
    total_mass = st.number_input("Total dry feed mass (kg)", min_value=0.05, max_value=500.0, value=0.25, step=0.05)
    cao_pct = st.number_input("Reactive free CaO in kiln dust (%)", 0.0, 100.0, 9.4, 0.1)
    caoh_ckd_pct = st.number_input("Reactive Ca(OH)₂ in kiln dust (%)", 0.0, 100.0, 0.0, 0.1)
    caoh_pct = st.number_input("Reactive Ca(OH)₂ in carbide residue (%)", 0.0, 100.0, 90.0, 0.1)

    c_ckd = K_CAO * cao_pct / 100 + K_CAOH * caoh_ckd_pct / 100   # kg CO2 per kg kiln dust
    c_ccr = K_CAOH * caoh_pct / 100                                # kg CO2 per kg carbide residue

    blend_mode = st.radio("Blend selection", ["Manual ratio", "Auto-blend to target capacity"])
    target_clipped, target_c = False, 0.30
    if blend_mode == "Manual ratio":
        ckd_w = st.slider("Kiln dust share of blend (%)", 0, 100, 50) / 100
    else:
        target_c = st.number_input("Target capacity (kg CO₂ per kg blend)", 0.0, 1.0, 0.30, 0.01)
        if abs(c_ccr - c_ckd) < 1e-9:
            ckd_w = 0.5
        else:
            raw = (c_ccr - target_c) / (c_ccr - c_ckd)
            ckd_w = float(min(max(raw, 0.0), 1.0))
            target_clipped = ckd_w != raw

    st.subheader("Operating conditions")
    x_target = st.slider("Target conversion of reactive lime (%)", 50, 99, 95) / 100
    co2_flow = st.slider("Gas inlet flow (L/min)", 0.1, 10.0, 2.0, 0.1)
    co2_frac = st.slider("CO₂ share of inlet gas (%)", 10, 100, 100, 5) / 100
    st.caption("100% = cylinder CO₂. Cement kiln exhaust is much lower; check the plant's data.")
    duration = st.slider("Maximum run time (min)", 5, 240, 90)
    temperature = st.slider("Gas temperature (°C)", 15, 80, 25)
    adaptive_on = st.toggle("Adaptive CO₂ control", value=True)

    with st.expander("Model assumptions (advanced)"):
        eta_max = st.slider("Peak absorption efficiency", 0.50, 1.00, 0.92, 0.01)
        p_sharp = st.slider("Efficiency fall-off sharpness", 2.0, 10.0, 6.0, 0.5)
        x_cap = st.slider("Practical maximum conversion", 0.90, 1.00, 0.97, 0.01)
        x_switch = st.slider("Stage 2 starts at conversion (%)", 40, 94, 75) / 100
        ratio2 = st.slider("Stage 2 flow (fraction of stage 1)", 0.1, 1.0, 0.40, 0.05)
        st.caption("These control the SHAPE of the assumed reaction curve. They are not measured; "
                   "fit them to your own pH-versus-time data when you have it.")

# ------------------------------------------------------------
# Stoichiometry (theoretical)
# ------------------------------------------------------------
ckd_mass = total_mass * ckd_w
ccr_mass = total_mass - ckd_mass
cao_kg = ckd_mass * cao_pct / 100
caoh_kg = ckd_mass * caoh_ckd_pct / 100 + ccr_mass * caoh_pct / 100
capacity_kg = K_CAO * cao_kg + K_CAOH * caoh_kg
n0 = capacity_kg * 1000 / M_CO2   # mol CO2 capacity

if n0 <= 0:
    st.error("The reactive lime content is zero, so there is nothing to carbonate. "
             "Enter reactive CaO or Ca(OH)₂ above 0%.")
    st.stop()
if cao_pct + caoh_ckd_pct > 100:
    st.warning("Reactive CaO plus Ca(OH)₂ in kiln dust exceeds 100%. Check your inputs.")
if blend_mode != "Manual ratio" and target_clipped:
    st.sidebar.warning(f"Target is outside what these feedstocks can reach "
                       f"({min(c_ckd, c_ccr):.3f} to {max(c_ckd, c_ccr):.3f} kg CO₂/kg). Using the nearest blend.")

T_K = temperature + 273.15
args = (n0, co2_flow, co2_frac, T_K, float(duration))
tail = (x_target, x_switch, ratio2, eta_max, p_sharp, x_cap)
t_sim0 = time.perf_counter()
sim_a = simulate(*args, True, *tail)
sim_f = simulate(*args, False, *tail)
t_sim = time.perf_counter() - t_sim0
act = sim_a if adaptive_on else sim_f
S_a, S_f, S_act = summarise(sim_a, n0), summarise(sim_f, n0), summarise(act, n0)

st.markdown(
    '<div class="banner">'
    + pill("t", "Theoretical") + "fixed chemistry &nbsp;"
    + pill("a", "Assumed") + "feedstock fractions, reaction curve, stage settings &nbsp;"
    + pill("s", "Simulated") + "every time-dependent result &nbsp;·&nbsp; "
    "<b>Nothing on this page is measured.</b></div>", unsafe_allow_html=True)

# ------------------------------------------------------------
# KPI strip
# ------------------------------------------------------------
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Theoretical CO₂ capacity", f"{capacity_kg:.3f} kg", help="0.785 × reactive CaO + 0.594 × reactive Ca(OH)₂")
k2.metric("CO₂ mineralised (sim)", f"{S_act['up_kg']:.3f} kg",
          f"{S_a['up_kg'] - S_f['up_kg']:+.3f} kg vs fixed" if adaptive_on else None)
k3.metric("Lime conversion (sim)", f"{S_act['conv'] * 100:.1f} %")
k4.metric("Gas utilisation (sim)", f"{S_act['util'] * 100:.1f} %",
          f"{(S_a['util'] - S_f['util']) * 100:+.1f} pts vs fixed" if adaptive_on else None)
k5.metric("Time to stop (sim)", f"{S_act['t_end_min']:.1f} min")
k6.metric("CO₂ gas injected (sim)", f"{S_act['inj_kg']:.3f} kg",
          f"{S_a['inj_kg'] - S_f['inj_kg']:+.3f} kg vs fixed" if adaptive_on else None, delta_color="inverse")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Reactor playback", "Feedstock & blend", "Control & kinetics", "Carbon & product", "Assumptions & evidence",
])

# ------------------------------------------------------------
# Reactor playback (three.js with a 2D canvas fallback)
# ------------------------------------------------------------
TWIN_HTML = r'''<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<style>
html,body{margin:0;height:100%;background:transparent;color:#14202B;font:13px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
#wrap{display:flex;flex-direction:column;height:100%;gap:8px}
#bar,#pb{display:flex;flex-wrap:wrap;gap:8px 12px;align-items:center}
.seg{display:flex;border:1px solid #D3DBE3;border-radius:8px;overflow:hidden;background:#fff}
.seg button{font:inherit;padding:6px 11px;border:0;border-right:1px solid #D3DBE3;background:transparent;color:#14202B;cursor:pointer}
.seg button:last-child{border-right:0}
.seg button[aria-pressed=true]{background:#2F6FB3;color:#fff;font-weight:600}
#play{font:inherit;padding:6px 11px;border:0;background:#2F6FB3;color:#fff;border-radius:8px;font-weight:600;min-width:74px;cursor:pointer}
label{display:flex;gap:5px;align-items:center;color:#5B6B7A;cursor:pointer}
select{background:#fff;color:#14202B;border:1px solid #D3DBE3;border-radius:6px;padding:4px}
#v{position:relative;flex:1;min-height:360px;border:1px solid #D3DBE3;border-radius:12px;background:linear-gradient(#FFFFFF,#DCE4EC);overflow:hidden;touch-action:none;cursor:grab}
#v canvas{display:block;width:100%;height:100%}
#lbs{position:absolute;inset:0;pointer-events:none}
.lb{position:absolute;left:0;top:0;font-size:11px;line-height:1;padding:3px 6px;border-radius:4px;background:rgba(255,255,255,.92);color:#14202B;border:1px solid #D3DBE3;white-space:nowrap}
#hud{position:absolute;left:10px;top:10px;min-width:206px;padding:10px 12px;border-radius:10px;background:rgba(255,255,255,.94);border:1px solid #D3DBE3;pointer-events:none;box-shadow:0 1px 3px rgba(20,32,43,.08)}
.row{display:flex;justify-content:space-between;gap:16px}.k{color:#5B6B7A}
#stg{font-weight:700;margin-bottom:4px}
.chips{display:flex;flex-wrap:wrap;gap:5px;margin-top:7px}
.chip{padding:2px 7px;border-radius:999px;border:1px solid #D3DBE3;color:#5B6B7A;font-size:11px;background:#fff}
.chip.on{background:#1E8E5A;color:#fff;border-color:#1E8E5A}
#sl{flex:1;min-width:140px;accent-color:#2F6FB3}
.tag{position:absolute;right:10px;bottom:8px;font-size:11px;color:#5B6B7A;background:rgba(255,255,255,.85);padding:2px 8px;border-radius:6px;border:1px solid #D3DBE3}
</style></head><body>
<div id="wrap">
<div id="bar">
 <div class="seg" id="views"><button data-v="iso" aria-pressed="true">Isometric</button><button data-v="front" aria-pressed="false">Front</button><button data-v="top" aria-pressed="false">Top</button></div>
 <label><input type="checkbox" id="cb-lab" checked> Labels</label>
 <label><input type="checkbox" id="cb-glass" checked> See-through vessel</label>
 <label><input type="checkbox" id="cb-rot"> Auto-rotate</label>
</div>
<div id="v"><div id="lbs"></div>
 <div id="hud"><div id="stg">Stage 1</div>
  <div class="row"><span class="k">Time</span><b id="h-t">0:00</b></div>
  <div class="row"><span class="k">Lime converted</span><b id="h-x">0.0%</b></div>
  <div class="row"><span class="k">pH (indicative)</span><b id="h-ph">12.4</b></div>
  <div class="row"><span class="k">Gas flow</span><b id="h-f">0.0 L/min</b></div>
  <div class="row"><span class="k">Outlet CO₂</span><b id="h-o">0.0%</b></div>
  <div class="row"><span class="k">CO₂ mineralised</span><b id="h-u">0.000 kg</b></div>
  <div class="chips"><span class="chip" id="c1">pH finished</span><span class="chip" id="c2">CO₂ breakthrough</span><span class="chip" id="c3">Uptake at target</span></div>
 </div>
 <div class="tag" id="tag">Simulated playback · conceptual, not CFD</div>
</div>
<div id="pb"><button id="play">Pause</button><input type="range" id="sl" min="0" max="1" value="0" step="1" aria-label="Simulated time">
 <span id="tl">0:00</span>
 <select id="spd" aria-label="Playback speed"><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option><option value="4">4×</option></select></div>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
window.TWIN=__TWIN_DATA__;
(function(){
window.addEventListener('error',function(e){var t=document.getElementById('tag');if(t)t.textContent='Script error: '+e.message;});
var D=window.TWIN,W=document.getElementById('v'),LB=document.getElementById('lbs');
var $=function(id){return document.getElementById(id);};
var COL={1:0x1E8E5A,2:0xD98E04,0:0xC8372D},HEX={1:'#1E8E5A',2:'#B87803',0:'#C8372D'};
var NAME={1:'Stage 1: full flow',2:'Stage 2: reduced flow',0:'Valve closed: endpoint'};
var n=D.t.length,pos=0,playing=true,spd=1,lastI=-1,PLAY_S=24,sl=$('sl'),S={stage:1,ratio:1,X:0};
var xmax=Math.max.apply(null,D.X)||1;
sl.max=n-1;
function fmt(s){s=Math.round(s);return Math.floor(s/60)+':'+('0'+(s%60)).slice(-2);}
function apply(i){
 var st=D.stage[i];S.stage=st;S.ratio=D.fmax>0?D.f[i]/D.fmax:0;S.X=Math.min(1,D.X[i]/xmax);
 $('stg').textContent=NAME[st];$('stg').style.color=HEX[st];
 $('h-t').textContent=fmt(D.t[i]);$('tl').textContent=fmt(D.t[i]);
 $('h-x').textContent=(D.X[i]*100).toFixed(1)+'%';$('h-ph').textContent=D.ph[i].toFixed(1);
 $('h-f').textContent=D.f[i].toFixed(2)+' L/min';$('h-o').textContent=(D.out[i]*100).toFixed(1)+'%';
 $('h-u').textContent=D.upt[i].toFixed(3)+' kg';
 $('c1').className='chip'+(D.s1[i]?' on':'');$('c2').className='chip'+(D.s2[i]?' on':'');$('c3').className='chip'+(D.s3[i]?' on':'');
 sl.value=i;lastI=i;}
sl.addEventListener('input',function(){pos=+sl.value;playing=false;$('play').textContent='Play';apply(pos);});
$('play').onclick=function(){if(!playing&&pos>=n-1)pos=0;playing=!playing;$('play').textContent=playing?'Pause':'Play';};
$('spd').onchange=function(){spd=+this.value;};
function tick(dt){
 if(!playing)return;
 if(!(dt>0))dt=0;
 pos+=dt*(n/PLAY_S)*spd;
 if(pos<0)pos=0;
 if(pos>=n-1){pos=n-1;playing=false;$('play').textContent='Replay';}
 var i=Math.min(n-1,Math.max(0,Math.floor(pos)));if(i!==lastI)apply(i);}
apply(0);

var gl=null;
try{if(typeof THREE!=='undefined'){gl=new THREE.WebGLRenderer({antialias:true,alpha:true});}}catch(e){gl=null;}
if(gl){try{init3D(gl);}catch(e){try{W.removeChild(gl.domElement);}catch(x){}init2D(String(e&&e.message||e));}}else{init2D();}

/* ---------------- 3D ---------------- */
function init3D(R){
R.setPixelRatio(Math.min(window.devicePixelRatio||1,2));W.insertBefore(R.domElement,LB);
var Sc=new THREE.Scene(),C=new THREE.PerspectiveCamera(38,1,.1,100);
Sc.add(new THREE.AmbientLight(0xffffff,.85));
var d1=new THREE.DirectionalLight(0xffffff,.8);d1.position.set(6,10,8);Sc.add(d1);
var d2=new THREE.DirectionalLight(0xffffff,.35);d2.position.set(-6,4,-6);Sc.add(d2);
function M(c,o){var m=new THREE.MeshStandardMaterial({color:c,metalness:.2,roughness:.55});if(o!==undefined){m.transparent=true;m.opacity=o;m.depthWrite=false;m.side=THREE.DoubleSide;}return m;}
var G=new THREE.Group();Sc.add(G);
function add(g,m,x,y,z,p){var o=new THREE.Mesh(g,m);o.position.set(x,y,z);(p||G).add(o);return o;}
function cyl(rt,rb,h,m,x,y,z,p){var g=new THREE.CylinderGeometry(rt,rb,h,32);g.translate(0,h/2,0);return add(g,m,x,y,z,p);}
function box(w,h,d,m,x,y,z,p){return add(new THREE.BoxGeometry(w,h,d),m,x,y,z,p);}
function cylX(r,x1,x2,m,y,z){var len=Math.abs(x2-x1),o=add(new THREE.CylinderGeometry(r,r,len,16),m,(x1+x2)/2,y,z);o.rotation.z=Math.PI/2;return o;}
var glassM=M(0x7fb0dc,.28),slurM=M(0xa8a493,.7);
add(new THREE.CylinderGeometry(1.2,1.2,3.2,48,1,true).translate(0,1.6,0),glassM,0,0,0);
var rimM=M(0x4a5a68);
[0,3.2].forEach(function(y){var r=add(new THREE.TorusGeometry(1.2,.025,8,64),rimM,0,y,0);r.rotation.x=Math.PI/2;});
cyl(1.2,1.2,.1,M(0x6f7780),0,-.1,0);cyl(1.28,1.28,.12,M(0x8a929a),0,3.2,0);cyl(1.17,1.17,2.1,slurM,0,.05,0);
cyl(.4,.4,.8,M(0x39414b),0,3.5,0);cyl(.22,.22,.3,M(0x55606b),0,3.3,0);cyl(.06,.06,3.6,M(0x9aa3ab),0,.3,0);
var imp=new THREE.Group();imp.position.y=.5;G.add(imp);
box(1.4,.12,.18,M(0x9aa3ab),0,0,0,imp);box(.18,.12,1.4,M(0x9aa3ab),0,0,0,imp);cyl(.1,.1,.3,M(0x9aa3ab),0,-.15,0,imp);
var sp=add(new THREE.TorusGeometry(.6,.05,10,40),M(0x2F6FB3),0,.35,0);sp.rotation.x=Math.PI/2;
for(var i=0;i<4;i++){var a=Math.PI/4+i*Math.PI/2;cyl(.06,.06,1.95,M(0x6f7780),Math.cos(a),-1.95,Math.sin(a));}
var blue=M(0x2F6FB3),valveM=M(0x1E8E5A);
cyl(.4,.4,1.8,M(0x7f8a86),-5,-.4,0);cyl(.15,.15,.2,M(0x5e6965),-5,1.4,0);
cylX(.06,-4.6,-.6,blue,.35,0);cylX(.2,-4.1,-3.7,M(0xD98E04),.35,0);
var cA=new THREE.ConeGeometry(.22,.4,20);
var cv1=new THREE.Mesh(cA,valveM);cv1.rotation.z=-Math.PI/2;cv1.position.set(-2.65,.35,0);G.add(cv1);
var cv2=new THREE.Mesh(cA,valveM);cv2.rotation.z=Math.PI/2;cv2.position.set(-2.35,.35,0);G.add(cv2);
cyl(.05,.05,3.0,M(0xC8372D),.55,1.1,0);cyl(.12,.12,.25,M(0xC8372D),.55,4.1,0);
cyl(.04,.04,2.7,M(0x5e6670),0,1.3,.55);cyl(.1,.1,.2,M(0x5e6670),0,4.0,.55);
cyl(.07,.07,.9,M(0x2f8f55),.85,3.2,0);add(new THREE.SphereGeometry(.2,20,16),M(0x2f8f55),.85,4.3,0);
cyl(.6,.12,1.1,M(0xD98E04),3.0,3.4,0);cyl(.6,.12,1.1,M(0xaab2ba),4.2,3.4,0);
cyl(.07,.07,.8,M(0x7a838c),3.0,2.6,0);cyl(.07,.07,.8,M(0x7a838c),4.2,2.6,0);
cyl(.8,.8,1.2,M(0xb9c0c7,.4),3.6,1.4,0);cyl(.04,.04,1.5,M(0x9aa3ab),3.6,1.4,0);
var mix=new THREE.Group();mix.position.set(3.6,1.65,0);G.add(mix);box(1.0,.06,.12,M(0x9aa3ab),0,0,0,mix);
cylX(.07,1.2,2.8,M(0x7a838c),1.7,0);
cyl(.12,.12,.9,M(0x7a838c),0,-1.0,0);box(1.8,.8,1.2,M(0x4f5b66),0,-1.4,0);box(.9,.15,.8,M(0xf4f2ea),0,-1.9,0);
box(1.8,1.1,.5,M(0x6B5FC7),0,2.5,-3.0);
var ledM=new THREE.MeshStandardMaterial({color:0x1E8E5A,emissive:0x1E8E5A,emissiveIntensity:.9});
add(new THREE.SphereGeometry(.09,16,12),ledM,.65,2.5,-2.72);
function line(pts){var g=new THREE.BufferGeometry().setFromPoints(pts.map(function(p){return new THREE.Vector3(p[0],p[1],p[2]);}));var l=new THREE.Line(g,new THREE.LineDashedMaterial({color:0x6B5FC7,dashSize:.16,gapSize:.1}));l.computeLineDistances();G.add(l);}
line([[.55,4.25,0],[.55,4.8,0],[.55,4.8,-3.0],[.55,3.05,-3.0]]);
line([[-.9,2.5,-3.0],[-2.5,2.5,-3.0],[-2.5,2.5,0],[-2.5,.65,0]]);
var grid=new THREE.GridHelper(16,16,0x9fb0bf,0x9fb0bf);grid.position.y=-1.98;grid.material.transparent=true;grid.material.opacity=.55;Sc.add(grid);
var N=140,bp=new Float32Array(N*3),bs=[];
function rb(i,top){var r=Math.sqrt(Math.random())*.55,a=Math.random()*6.283;bp[i*3]=Math.cos(a)*r;bp[i*3+1]=top?.4:.4+Math.random()*1.7;bp[i*3+2]=Math.sin(a)*r;}
for(var i=0;i<N;i++){rb(i,false);bs.push(.5+Math.random()*.7);}
var bg=new THREE.BufferGeometry();bg.setAttribute('position',new THREE.BufferAttribute(bp,3));
var bm=new THREE.PointsMaterial({color:0x2F6FB3,size:.1,transparent:true,opacity:.85});G.add(new THREE.Points(bg,bm));
var L=[['Kiln dust hopper',3.0,4.6,0],['Carbide residue hopper',4.2,4.6,0],['Slurry mixer',3.6,2.7,0],['Slurry line',2.0,1.85,0],['Reactor (glass)',-1.2,3.0,0],['Impeller',.7,.5,0],['Gas sparger',-.55,.2,0],['pH probe',.55,4.6,0],['Temperature probe',0,4.3,.55],['Vent + gas sensor',.85,4.7,0],['CO₂ source',-5,1.7,0],['Flow meter',-3.9,.65,0],['Solenoid valve',-2.5,.75,0],['Controller',0,3.25,-3.0],['Filter',0,-1.0,0],['Product (CaCO₃)',0,-1.75,0]];
var LE=L.map(function(l){var e=document.createElement('div');e.className='lb';e.textContent=l[0];LB.appendChild(e);return {e:e,v:new THREE.Vector3(l[1],l[2],l[3])};});
var c0=new THREE.Color(0xa8a493),c1=new THREE.Color(0xf4f2ea);
var T=new THREE.Vector3(.4,1.3,-.3),th=.75,ph=1.15,rad=17,dragging=false;
var views={iso:[.75,1.15,17],front:[0,1.5,17],top:[.75,.18,18]};
[].forEach.call(document.querySelectorAll('[data-v]'),function(b){b.onclick=function(){var v=views[b.dataset.v];th=v[0];ph=v[1];rad=v[2];
 [].forEach.call(document.querySelectorAll('[data-v]'),function(x){x.setAttribute('aria-pressed',x===b);});};});
var P={},pd=0;
W.addEventListener('pointerdown',function(e){P[e.pointerId]=[e.clientX,e.clientY];dragging=true;W.style.cursor='grabbing';try{W.setPointerCapture(e.pointerId);}catch(x){}});
W.addEventListener('pointermove',function(e){if(!P[e.pointerId])return;var ids=Object.keys(P);
 if(ids.length===2){var o=P[ids[0]==e.pointerId?ids[1]:ids[0]],d=Math.hypot(e.clientX-o[0],e.clientY-o[1]);if(pd)rad=Math.min(30,Math.max(6,rad*pd/d));pd=d;}
 else{th-=(e.clientX-P[e.pointerId][0])*.008;ph=Math.min(1.75,Math.max(.15,ph-(e.clientY-P[e.pointerId][1])*.008));}
 P[e.pointerId]=[e.clientX,e.clientY];});
function up(e){delete P[e.pointerId];pd=0;if(!Object.keys(P).length){dragging=false;W.style.cursor='grab';}}
W.addEventListener('pointerup',up);W.addEventListener('pointercancel',up);
W.addEventListener('wheel',function(e){e.preventDefault();rad=Math.min(30,Math.max(6,rad*Math.exp(e.deltaY*.001)));},{passive:false});
var cbL=$('cb-lab'),cbG=$('cb-glass'),cbR=$('cb-rot');
function size(){var w=W.clientWidth,h=W.clientHeight;R.setSize(w,h,false);C.aspect=w/h;C.updateProjectionMatrix();}
if(window.ResizeObserver)new ResizeObserver(size).observe(W);else window.addEventListener('resize',size);size();
var last=0,tv=new THREE.Vector3(),failed=false;
function frame(t){try{var dt=last?Math.min(.05,Math.max(0,(t-last)/1000)):0;last=t;
 if(R.domElement.width<2)size();
 tick(dt);
 valveM.color.setHex(COL[S.stage]);ledM.color.setHex(COL[S.stage]);ledM.emissive.setHex(COL[S.stage]);
 slurM.color.setRGB(c0.r+(c1.r-c0.r)*S.X,c0.g+(c1.g-c0.g)*S.X,c0.b+(c1.b-c0.b)*S.X);bm.visible=S.ratio>0.001;
 if(cbR.checked&&!dragging)th+=dt*.2;
 C.position.set(T.x+rad*Math.sin(ph)*Math.sin(th),T.y+rad*Math.cos(ph),T.z+rad*Math.sin(ph)*Math.cos(th));C.lookAt(T);
 glassM.opacity=cbG.checked?.28:.6;
 imp.rotation.y+=dt*(S.ratio>0?3:1.0);mix.rotation.y+=dt*2;
 if(S.ratio>0.001){var s=Math.max(S.ratio,.2);for(var k=0;k<N;k++){bp[k*3+1]+=bs[k]*dt*s*1.4;if(bp[k*3+1]>2.1)rb(k,true);}bg.attributes.position.needsUpdate=true;}
 R.render(Sc,C);
 var w=W.clientWidth,h=W.clientHeight;LB.style.display=cbL.checked?'block':'none';
 if(cbL.checked)LE.forEach(function(l){tv.copy(l.v).project(C);l.e.style.display=tv.z<1?'block':'none';
  l.e.style.transform='translate('+((tv.x*.5+.5)*w-l.e.offsetWidth/2)+'px,'+((-tv.y*.5+.5)*h-l.e.offsetHeight/2)+'px)';});
 }catch(err){if(!failed){failed=true;bail(err);}return;}
 requestAnimationFrame(frame);}
requestAnimationFrame(frame);
function bail(err){try{W.removeChild(R.domElement);}catch(e){}LE.forEach(function(l){if(l.e.parentNode)l.e.parentNode.removeChild(l.e);});init2D(String(err&&err.message||err));}
}

/* ---------------- 2D fallback (no internet / no WebGL) ---------------- */
function init2D(why){
['views','cb-glass','cb-rot','cb-lab'].forEach(function(id){var e=$(id),h=e&&(id==='views'?e:e.parentNode);if(h&&h.style)h.style.display='none';});
$('tag').textContent='2D view ('+(why?('3D error: '+why):'3D graphics unavailable, needs internet and WebGL')+'). Simulation is unaffected.';
var cv=document.createElement('canvas');W.insertBefore(cv,LB);var g=cv.getContext('2d');
var dpr=Math.min(window.devicePixelRatio||1,2),cw=1,ch=1;
function size(){cw=W.clientWidth;ch=W.clientHeight;cv.width=cw*dpr;cv.height=ch*dpr;g.setTransform(dpr,0,0,dpr,0,0);}
if(window.ResizeObserver)new ResizeObserver(size).observe(W);else window.addEventListener('resize',size);size();
var B=[];for(var i=0;i<70;i++)B.push({x:Math.random(),y:Math.random(),s:.5+Math.random()*.7});
var ang=0,last=0;
function mixc(a,b,t){return 'rgb('+Math.round(a[0]+(b[0]-a[0])*t)+','+Math.round(a[1]+(b[1]-a[1])*t)+','+Math.round(a[2]+(b[2]-a[2])*t)+')';}
function frame(t){try{var dt=last?Math.min(.05,Math.max(0,(t-last)/1000)):0;last=t;tick(dt);
 g.clearRect(0,0,cw,ch);
 var vx=cw*.40,vw=cw*.20,vy=ch*.16,vh=ch*.56,sy=vy+vh*.22,hex=HEX[S.stage];
 g.fillStyle=mixc([168,164,147],[244,242,234],S.X);g.fillRect(vx,sy,vw,vy+vh-sy);
 g.strokeStyle='#4a5a68';g.lineWidth=2;g.strokeRect(vx,vy,vw,vh);
 g.strokeStyle='#9aa3ab';g.lineWidth=3;g.beginPath();g.moveTo(vx+vw/2,vy-ch*.06);g.lineTo(vx+vw/2,vy+vh*.86);g.stroke();
 ang+=dt*(S.ratio>0?3:1);var bl=vw*.34*Math.abs(Math.cos(ang));g.beginPath();g.moveTo(vx+vw/2-bl,vy+vh*.86);g.lineTo(vx+vw/2+bl,vy+vh*.86);g.stroke();
 if(S.ratio>0.001){g.fillStyle='rgba(47,111,179,.85)';B.forEach(function(b){b.y-=dt*b.s*Math.max(S.ratio,.2)*.5;if(b.y<0){b.y=1;b.x=Math.random();}
  var bx=vx+vw*(.12+b.x*.76),by=sy+(vy+vh*.92-sy)*b.y;g.beginPath();g.arc(bx,by,2.4,0,6.283);g.fill();});}
 var py=vy+vh*.92;g.strokeStyle='#2F6FB3';g.lineWidth=4;g.beginPath();g.moveTo(cw*.12,py);g.lineTo(vx,py);g.stroke();
 g.fillStyle='#7f8a86';g.fillRect(cw*.08,py-ch*.1,cw*.05,ch*.2);
 g.fillStyle=hex;g.fillRect(cw*.27,py-9,18,18);
 g.fillStyle='#6B5FC7';g.fillRect(cw*.72,vy+vh*.25,cw*.14,ch*.14);g.fillStyle=hex;g.beginPath();g.arc(cw*.84,vy+vh*.25+ch*.07,5,0,6.283);g.fill();
 g.setLineDash([6,5]);g.strokeStyle='#6B5FC7';g.lineWidth=1.5;g.beginPath();g.moveTo(cw*.72,vy+vh*.32);g.lineTo(cw*.27+9,vy+vh*.32);g.lineTo(cw*.27+9,py-9);g.stroke();
 g.beginPath();g.moveTo(vx+vw*.9,vy-ch*.05);g.lineTo(vx+vw*.9,vy+vh*.4);g.lineTo(cw*.72,vy+vh*.3);g.stroke();g.setLineDash([]);
 g.fillStyle='#14202B';g.font='12px system-ui,sans-serif';g.textAlign='center';
 g.fillText('CO₂ source',cw*.105,py+ch*.13);g.fillText('Valve',cw*.27+9,py+30);g.fillText('Reactor',vx+vw/2,vy+vh+18);g.fillText('Controller',cw*.79,vy+vh*.25-6);g.fillText('pH probe',vx+vw*.9,vy-ch*.05-6);
 }catch(err){$('tag').textContent='Drawing error: '+String(err&&err.message||err);}
 requestAnimationFrame(frame);}
requestAnimationFrame(frame);
}
})();
</script></body></html>'''


def twin_payload(sim, flow):
    idx = np.unique(np.linspace(0, len(sim["t"]) - 1, 240).astype(int))

    def g(key, nd):
        return [round(float(v), nd) for v in sim[key][idx]]

    return {
        "t": g("t", 1), "X": g("X", 4), "f": g("f", 3), "stage": [int(v) for v in sim["stage"][idx]],
        "ph": g("ph", 2), "out": g("out", 4),
        "upt": [round(float(v) * M_CO2 / 1000, 5) for v in sim["ab"][idx]],
        "s1": [int(v) for v in sim["s1"][idx]], "s2": [int(v) for v in sim["s2"][idx]],
        "s3": [int(v) for v in sim["s3"][idx]], "fmax": float(flow),
    }


# ------------------------------------------------------------
# Tab 1: reactor playback
# ------------------------------------------------------------
with tab1:
    left, right = st.columns([2.4, 1])
    with left:
        mode_txt = "adaptive control" if adaptive_on else "fixed-flow reference"
        st.markdown(f"### Reactor playback: {mode_txt}")
        st.caption("Press play to watch the simulated run. The valve, bubbles, slurry colour and readouts follow the "
                   "model. Drag to rotate, scroll or pinch to zoom. Switch control mode in the sidebar.")
        try:
            components.html(TWIN_HTML.replace("__TWIN_DATA__", json.dumps(twin_payload(act, co2_flow))),
                            height=730, scrolling=False)
        except Exception as exc:  # keep the rest of the dashboard alive
            st.error(f"The reactor view could not be drawn ({exc}). The charts in the other tabs still work.")
    with right:
        st.markdown("### Controller console")
        st.markdown(pill("s", "Simulated") + "event log", unsafe_allow_html=True)
        ev = pd.DataFrame([(mmss(t), e) for t, e in act["events"]], columns=["Time", "Event"])
        st.dataframe(ev, hide_index=True, use_container_width=True)
        st.markdown("**Batch recipe**")
        recipe = pd.DataFrame({
            "Item": ["Kiln dust", "Carbide residue", "Blend capacity", "Blend selection"],
            "Value": [f"{ckd_mass:.3f} kg ({ckd_w * 100:.0f}%)", f"{ccr_mass:.3f} kg ({(1 - ckd_w) * 100:.0f}%)",
                      f"{capacity_kg / total_mass:.3f} kg CO₂/kg",
                      "Manual" if blend_mode == "Manual ratio" else "Auto to target"],
        })
        st.dataframe(recipe, hide_index=True, use_container_width=True)
        st.markdown("**Endpoint rule**")
        st.caption(f"Stop when uptake reaches the {x_target * 100:.0f}% target, or when pH is below "
                   f"{PH_FINISH_SIGNAL} and outlet CO₂ shows breakthrough together. "
                   "In a real reactor these come from calibrated pH and gas sensors; here they are model outputs.")

# ------------------------------------------------------------
# Tab 2: feedstock & blend
# ------------------------------------------------------------
with tab2:
    st.markdown("### CKD–CCR blend screening")
    st.markdown(pill("t", "Theoretical") + "stoichiometric capacity only. It does not predict rate or product quality.",
                unsafe_allow_html=True)
    st.info("Blend capacity is a straight-line average of the two feedstocks. Blending cannot beat the better feedstock "
            "on capacity alone; any benefit must come from consistency, reaction rate or product quality, and must be tested.")

    rows = []
    for a_, b_ in [(100, 0), (75, 25), (50, 50), (25, 75), (0, 100)]:
        w = a_ / 100
        cao_r = w * cao_pct / 100
        caoh_r = w * caoh_ckd_pct / 100 + (1 - w) * caoh_pct / 100
        rows.append({
            "CKD:CCR": f"{a_}:{b_}",
            "Reactive CaO (kg/kg)": cao_r, "Reactive Ca(OH)₂ (kg/kg)": caoh_r,
            "CO₂ capacity (kg/kg blend)": K_CAO * cao_r + K_CAOH * caoh_r,
            "Reactive lime (%)": (cao_r + caoh_r) * 100,
        })
    df = pd.DataFrame(rows)
    st.dataframe(df.style.format({"Reactive CaO (kg/kg)": "{:.3f}", "Reactive Ca(OH)₂ (kg/kg)": "{:.3f}",
                                  "CO₂ capacity (kg/kg blend)": "{:.3f}", "Reactive lime (%)": "{:.1f}"}),
                 use_container_width=True, hide_index=True)

    shares = np.linspace(0, 100, 101)
    cap_line = shares / 100 * c_ckd + (1 - shares / 100) * c_ccr
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=shares, y=cap_line, mode="lines", name="Blend capacity", line=dict(color=GREEN, width=4)))
    cur_c = capacity_kg / total_mass
    fig.add_trace(go.Scatter(x=[ckd_w * 100], y=[cur_c], mode="markers", name="Current blend",
                             marker=dict(size=14, color=AMBER, line=dict(color="#fff", width=2))))
    if blend_mode != "Manual ratio":
        fig.add_hline(y=target_c, line=dict(color=PURPLE, dash="dash"),
                      annotation_text="Target capacity", annotation_position="top right")
    style(fig, 380, xaxis_title="Kiln dust share of blend (%)", yaxis_title="CO₂ capacity (kg per kg blend)")
    st.plotly_chart(fig, use_container_width=True)
    if blend_mode != "Manual ratio":
        st.success(f"Auto-blend: **{ckd_w * 100:.1f}% kiln dust : {(1 - ckd_w) * 100:.1f}% carbide residue** gives "
                   f"{cur_c:.3f} kg CO₂ per kg blend. Formula: w = (c_CCR − c*) ÷ (c_CCR − c_CKD).")

# ------------------------------------------------------------
# Tab 3: control & kinetics
# ------------------------------------------------------------
with tab3:
    st.markdown("### Adaptive versus fixed control (simulated)")
    st.markdown(pill("s", "Simulated") + pill("a", "Assumed") +
                "Reaction curve and stage settings are assumptions, not measurements.", unsafe_allow_html=True)

    tm = act["t"] / 60
    figA = make_subplots(specs=[[{"secondary_y": True}]])
    figA.add_trace(go.Scatter(x=tm, y=act["X"] * 100, name="Lime converted (%)", line=dict(color=GREEN, width=4)))
    figA.add_trace(go.Scatter(x=tm, y=act["ph"], name="pH (indicative)", line=dict(color=AMBER, width=3, dash="dot")),
                   secondary_y=True)
    for t_ev, _label in act["events"][1:]:
        figA.add_vline(x=t_ev / 60, line=dict(color=PURPLE, dash="dash", width=1))
    style(figA, 400, xaxis_title="Simulated time (min)")
    figA.update_yaxes(title_text="Conversion (%)", range=[0, 100], secondary_y=False)
    figA.update_yaxes(title_text="pH", range=[7, 13], secondary_y=True)
    st.plotly_chart(figA, use_container_width=True)
    st.caption("Dashed lines mark controller events from the console log.")

    c1_, c2_ = st.columns(2)
    with c1_:
        fig_f = go.Figure()
        fig_f.add_trace(go.Scatter(x=sim_a["t"] / 60, y=sim_a["f"], name="Adaptive", line=dict(color=GREEN, width=4, shape="hv")))
        fig_f.add_trace(go.Scatter(x=sim_f["t"] / 60, y=sim_f["f"], name="Fixed flow", line=dict(color=AMBER, width=3, dash="dash", shape="hv")))
        style(fig_f, 350, xaxis_title="Simulated time (min)", yaxis_title="Gas flow (L/min)", title="Gas flow profile")
        st.plotly_chart(fig_f, use_container_width=True)
    with c2_:
        fig_c = go.Figure()
        fig_c.add_trace(go.Scatter(x=sim_a["t"] / 60, y=sim_a["inj"] * M_CO2 / 1000, name="Adaptive: injected", line=dict(color=GREEN, width=3, dash="dot")))
        fig_c.add_trace(go.Scatter(x=sim_a["t"] / 60, y=sim_a["ab"] * M_CO2 / 1000, name="Adaptive: mineralised", line=dict(color=GREEN, width=4)))
        fig_c.add_trace(go.Scatter(x=sim_f["t"] / 60, y=sim_f["inj"] * M_CO2 / 1000, name="Fixed: injected", line=dict(color=AMBER, width=3, dash="dot")))
        fig_c.add_trace(go.Scatter(x=sim_f["t"] / 60, y=sim_f["ab"] * M_CO2 / 1000, name="Fixed: mineralised", line=dict(color=AMBER, width=4)))
        style(fig_c, 350, xaxis_title="Simulated time (min)", yaxis_title="Cumulative CO₂ (kg)", title="CO₂ injected versus mineralised")
        st.plotly_chart(fig_c, use_container_width=True)

    fig_o = go.Figure()
    fig_o.add_trace(go.Scatter(x=tm, y=act["out"] * 100, name="Outlet CO₂ (%)", line=dict(color=RED, width=3)))
    fig_o.add_hline(y=co2_frac * 100, line=dict(color=MUTED, dash="dash"), annotation_text="Inlet CO₂ share")
    style(fig_o, 300, xaxis_title="Simulated time (min)", yaxis_title="CO₂ in off-gas (%)", title="Breakthrough signal")
    st.plotly_chart(fig_o, use_container_width=True)

    st.markdown("### Feedstock variability study")
    st.write("A fixed recipe is set for the nominal batch (it runs for the time the nominal batch needs under adaptive "
             "control). Then the reactive lime content is changed. Adaptive control re-doses each batch.")
    nom_min = max(sim_a["end_t"] / 60.0, 0.5)
    vrows = []
    for dev in (-30, -15, 0, 15, 30):
        n_i = n0 * (1 + dev / 100)
        a_ = summarise(simulate(n_i, co2_flow, co2_frac, T_K, float(duration), True, *tail), n_i)
        f_ = summarise(simulate(n_i, co2_flow, co2_frac, T_K, float(nom_min), False, *tail), n_i)
        vrows.append({"Reactive lime vs nominal": f"{dev:+d}%",
                      "Fixed: conversion (%)": f_["conv"] * 100, "Adaptive: conversion (%)": a_["conv"] * 100,
                      "Fixed: off-gas CO₂ (kg)": f_["off_kg"], "Adaptive: off-gas CO₂ (kg)": a_["off_kg"],
                      "Fixed: utilisation (%)": f_["util"] * 100, "Adaptive: utilisation (%)": a_["util"] * 100})
    vdf = pd.DataFrame(vrows)
    v1, v2 = st.columns(2)
    with v1:
        fv = go.Figure()
        fv.add_trace(go.Bar(x=vdf["Reactive lime vs nominal"], y=vdf["Fixed: conversion (%)"], name="Fixed recipe", marker_color=AMBER))
        fv.add_trace(go.Bar(x=vdf["Reactive lime vs nominal"], y=vdf["Adaptive: conversion (%)"], name="Adaptive", marker_color=GREEN))
        style(fv, 340, barmode="group", yaxis_title="Final conversion (%)", title="Lime converted", yaxis_range=[0, 100])
        st.plotly_chart(fv, use_container_width=True)
    with v2:
        fo = go.Figure()
        fo.add_trace(go.Bar(x=vdf["Reactive lime vs nominal"], y=vdf["Fixed: off-gas CO₂ (kg)"], name="Fixed recipe", marker_color=AMBER))
        fo.add_trace(go.Bar(x=vdf["Reactive lime vs nominal"], y=vdf["Adaptive: off-gas CO₂ (kg)"], name="Adaptive", marker_color=GREEN))
        style(fo, 340, barmode="group", yaxis_title="Unreacted CO₂ (kg)", title="Gas wasted")
        st.plotly_chart(fo, use_container_width=True)
    st.dataframe(vdf.style.format("{:.2f}", subset=[c for c in vdf.columns if c != "Reactive lime vs nominal"]),
                 hide_index=True, use_container_width=True)
    st.warning("Caveat: in this model the adaptive run stops at its endpoint, so it can finish with slightly lower "
               "conversion than a long fixed run, in exchange for much less wasted gas. That trade-off comes from the "
               "stopping rule and the assumed efficiency curve. It shows the control concept, not a validated "
               "improvement. Fit the curve to your own pH and gas data before quoting any percentage.")

# ------------------------------------------------------------
# Tab 4: carbon & product
# ------------------------------------------------------------
with tab4:
    st.markdown("### Carbon mass balance and product verification")
    st.markdown(pill("s", "Simulated") + f"results for the {'adaptive' if adaptive_on else 'fixed-flow'} run.",
                unsafe_allow_html=True)
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("CO₂ injected", f"{S_act['inj_kg']:.3f} kg")
    b2.metric("CO₂ mineralised", f"{S_act['up_kg']:.3f} kg")
    b3.metric("Unreacted / off-gas", f"{S_act['off_kg']:.3f} kg")
    b4.metric("CO₂ per kg dry waste", f"{S_act['up_kg'] / total_mass:.3f} kg/kg")

    sk = go.Figure(go.Sankey(
        node=dict(label=["CO₂ injected", "Mineralised as CaCO₃", "Unreacted off-gas"], pad=24, thickness=22,
                  color=[BLUE, GREEN, AMBER]),
        link=dict(source=[0, 0], target=[1, 2], value=[max(S_act["up_kg"], 1e-9), max(S_act["off_kg"], 1e-9)],
                  color=["rgba(30,142,90,.35)", "rgba(217,142,4,.35)"])))
    style(sk, 320)
    st.plotly_chart(sk, use_container_width=True)

    m1, m2 = st.columns(2)
    m1.metric("Fraction of theoretical capacity used", f"{S_act['up_kg'] / capacity_kg * 100:.1f} %")
    m2.metric("CaCO₃ equivalent formed (sim)", f"{S_act['caco3_kg']:.3f} kg")

    st.markdown("#### Product verification checklist")
    checks = pd.DataFrame([
        ["XRD", "Identify calcite, vaterite and remaining phases", "Required"],
        ["TGA/DTG", "Estimate carbonate content and mass-loss profile", "Required"],
        ["XRF", "Check bulk chemistry and impurities (chloride, sulphate, alkalis)", "Required"],
        ["Free-lime / available-lime test", "Measure reactive calcium before and after", "Required"],
        ["Particle-size analysis", "Measure size distribution", "Recommended"],
        ["Leaching test", "Assess soluble contaminants", "Required before application"],
    ], columns=["Test", "Purpose", "Priority"])
    st.dataframe(checks, use_container_width=True, hide_index=True)
    st.info("Carbon accounting: this is gross CO₂ mineralised, not net climate benefit. Energy, transport, drying, water "
            "treatment and product end-of-life must be included. If the product is heated again in a cement kiln, the "
            "CO₂ is released.")

# ------------------------------------------------------------
# Tab 5: assumptions & evidence
# ------------------------------------------------------------
with tab5:
    st.markdown("### What is theory, what is assumed, what is simulated")
    ev_df = pd.DataFrame([
        ["CO₂ capacity per kg of CaO and Ca(OH)₂ (0.785, 0.594)", "Theoretical", "Fixed stoichiometry"],
        ["Reactive CaO / Ca(OH)₂ fractions of each feedstock", "Assumed", "Replace with free-lime test, TGA, XRF"],
        ["Peak absorption efficiency, fall-off sharpness, practical maximum conversion", "Assumed", "Fit to your own bench data"],
        ["Stage 2 start point, stage 2 flow ratio, endpoint thresholds", "Assumed", "Design settings to tune"],
        ["pH versus conversion curve", "Assumed", "Calibrate with a real probe"],
        ["Conversion, uptake, utilisation, pH, outlet CO₂, time to stop", "Simulated", "Outputs of the time-stepped model"],
        ["Measured experimental data", "None", "This dashboard contains no lab measurements"],
    ], columns=["Quantity", "Status", "How to firm it up"])
    st.dataframe(ev_df, hide_index=True, use_container_width=True)

    xs = np.linspace(0, 1, 200)
    fa = make_subplots(specs=[[{"secondary_y": True}]])
    fa.add_trace(go.Scatter(x=xs * 100, y=eta_of(xs, eta_max, p_sharp, x_cap), name="Absorption efficiency",
                            line=dict(color=GREEN, width=4)))
    fa.add_trace(go.Scatter(x=xs * 100, y=ph_of(xs), name="pH (indicative)", line=dict(color=AMBER, width=3, dash="dot")),
                 secondary_y=True)
    style(fa, 340, xaxis_title="Lime converted (%)", title="The assumed reaction curves in use")
    fa.update_yaxes(title_text="Fraction of injected CO₂ absorbed", range=[0, 1], secondary_y=False)
    fa.update_yaxes(title_text="pH", range=[7, 13], secondary_y=True)
    st.plotly_chart(fa, use_container_width=True)

    with st.expander("Self-check (for debugging)"):
        inj_a, ab_a = sim_a["inj_end"], sim_a["ab_end"]
        checks_list = [
            ("Uptake never exceeds theoretical capacity (adaptive)", ab_a <= n0 + 1e-9),
            ("Uptake never exceeds theoretical capacity (fixed)", sim_f["ab_end"] <= n0 + 1e-9),
            ("Absorbed CO₂ never exceeds injected CO₂", ab_a <= inj_a + 1e-12 and sim_f["ab_end"] <= sim_f["inj_end"] + 1e-12),
            ("Time arrays have equal length", len(sim_a["t"]) == len(sim_a["X"]) == len(sim_a["f"])),
            ("No NaN values in results", not any(np.isnan(sim_a[k]).any() for k in ("X", "f", "ph", "out"))),
        ]
        for label, ok in checks_list:
            st.write(("✅ " if ok else "❌ ") + label)
        st.caption(f"Streamlit {st.__version__} · {len(sim_a['t'])} time steps per run · "
                   f"model run time {t_sim * 1000:.0f} ms (cached runs show about 0 ms)")

st.divider()
st.markdown(
    "**CarbonStone prototype statement:** This dashboard demonstrates transparent stoichiometric calculations, "
    "blend screening, a time-stepped carbonation model with assumed parameters, and an adaptive-control concept played "
    "back in 3D. It is not yet an AI model, a validated kinetic model, a CFD simulation or an experimentally calibrated "
    "digital twin.")
