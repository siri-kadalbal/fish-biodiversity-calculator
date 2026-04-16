from flask import Flask, render_template_string, request, jsonify
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io, base64
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# ══════════════════════════════════════════════════════
# FEATURE LISTS
# ══════════════════════════════════════════════════════
HUMAN_FEATURES = [
    "pH", "turbidity", "total_nitrogen_mgl", "ammonia_mgl",
    "doc_mgl", "chloride_mgl", "tss_mgl",
    "dissolved_oxygen", "water_temp_c",
    "substrate_lmm", "pct_fast_water", "pct_slow_water",
    "canopy_pct", "riparian_veg", "embeddedness", "pool_pct", "lwd",
    "air_temp_c", "precip_mm",
    "pct_urban", "pct_forest", "pct_agriculture", "pct_wetland",
    "pct_impervious", "pct_shrub", "pct_grassland",
]

ALL_FEATURES = [
    "water_temp_c", "dissolved_oxygen", "spec_conductance",
    "pH", "turbidity", "total_nitrogen_mgl", "ammonia_mgl",
    "doc_mgl", "chloride_mgl",
    "pct_fast_water", "pct_slow_water", "substrate_lmm", "embeddedness",
    "sinuosity", "canopy_pct", "riparian_veg",
    "pct_urban", "pct_forest", "pct_agriculture", "pct_wetland",
    "pct_impervious", "pct_shrub", "pct_grassland",
    "air_temp_c", "precip_mm", "stream_order",
    "width_m", "elevation_m", "tss_mgl", "pool_pct", "lwd",
]

FEATURE_LABELS = {
    "water_temp_c":       "Water Temperature (C)",
    "dissolved_oxygen":   "Dissolved Oxygen (mg/L)",
    "spec_conductance":   "Specific Conductance (uS/cm)",
    "pH":                 "pH",
    "turbidity":          "Turbidity (NTU)",
    "total_nitrogen_mgl": "Total Nitrogen (mg/L)",
    "ammonia_mgl":        "Ammonia (mg/L)",
    "doc_mgl":            "DOC (mg/L)",
    "chloride_mgl":       "Chloride (mg/L)",
    "tss_mgl":            "TSS (mg/L)",
    "pct_fast_water":     "% Fast Water",
    "pct_slow_water":     "% Slow Water",
    "substrate_lmm":      "Substrate (lmm)",
    "embeddedness":       "Embeddedness",
    "sinuosity":          "Sinuosity",
    "canopy_pct":         "Canopy Cover (%)",
    "riparian_veg":       "Riparian Vegetation",
    "pool_pct":           "Pool Cover (%)",
    "lwd":                "Large Woody Debris",
    "pct_urban":          "% Urban",
    "pct_forest":         "% Forest",
    "pct_agriculture":    "% Agriculture",
    "pct_wetland":        "% Wetland",
    "pct_impervious":     "% Impervious Surface",
    "pct_shrub":          "% Shrubland",
    "pct_grassland":      "% Grassland",
    "air_temp_c":         "Air Temperature (C)",
    "precip_mm":          "Precipitation (mm)",
    "stream_order":       "Stream Order",
    "width_m":            "Stream Width (m)",
    "elevation_m":        "Elevation (m)",
}

XGB_PARAMS = dict(
    n_estimators=400, max_depth=4, learning_rate=0.04,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0,
    min_child_weight=3, random_state=42,
    tree_method="hist", max_bin=256,
)

# ══════════════════════════════════════════════════════
# STARTUP
# ══════════════════════════════════════════════════════
import xgboost as xgb

CSV_PATH    = "nrsa_complete_cleaned.csv"
JOBLIB_PATH = "fish_final_model.joblib"

print("Loading and preprocessing data...")
df_raw = pd.read_csv(CSV_PATH, low_memory=False)
CAT = ["SITE_ID", "AG_ECO9", "AG_ECO9_NM", "STATE", "HUC2", "HUC8"]
df = df_raw.copy()
for col in df.columns:
    if col not in CAT:
        df[col] = pd.to_numeric(df[col], errors="coerce")
df = df[df["shannon_diversity"].notna()].copy()

def build_X(feature_list, df):
    feats = [f for f in feature_list if f in df.columns and not df[f].isnull().all()]
    dff = df.copy()
    dff["_fc"] = dff[feats].notna().sum(axis=1)
    dff = dff[dff["_fc"] >= 5].copy()
    y = dff["shannon_diversity"].values
    scaler  = StandardScaler()
    imputer = KNNImputer(n_neighbors=5, weights="distance")
    X_s = scaler.fit_transform(dff[feats].values)
    X_i = imputer.fit_transform(X_s)
    X_orig = pd.DataFrame(dff[feats].values, columns=feats)
    return feats, X_orig, y, scaler, imputer

print("Building full feature set...")
FEATS_F, X_ORIG_F, Y_F, SCALER_F, IMPUTER_F = build_X(ALL_FEATURES, df)

print("Building management feature set...")
FEATS_H, X_ORIG_H, Y_H, SCALER_H, IMPUTER_H = build_X(HUMAN_FEATURES, df)

print("Loading full model...")
saved   = joblib.load(JOBLIB_PATH)
MODEL_F = saved["model"]

print("Fitting management model...")
MODEL_H = xgb.XGBRegressor(**XGB_PARAMS)
_X_H_s  = SCALER_H.transform(X_ORIG_H.values)
_X_H_i  = IMPUTER_H.transform(_X_H_s)
MODEL_H.fit(_X_H_i, Y_H)

print("Ready.\n")

# ══════════════════════════════════════════════════════
# THRESHOLD DETECTION
# ══════════════════════════════════════════════════════
N_GRID    = 80
EDGE_FRAC = 0.03
SIG_SD    = 1.5

# Feature direction overrides
# "upper_only" — ecologically, more is always worse (e.g. turbidity, pollution).
#                Lower threshold suppressed — zero/low values are ideal, not a minimum limit.
# "lower_only" — ecologically, more is always better (e.g. canopy cover, forest, DO).
#                Upper threshold suppressed — no ceiling on benefit.
# Features not listed here use the full peak-aware algorithm (unimodal allowed).
FEATURE_DIRECTION = {
    # Stressors — higher is always worse, no meaningful lower limit
    "turbidity":          "upper_only",
    "total_nitrogen_mgl": "upper_only",
    "ammonia_mgl":        "upper_only",
    "chloride_mgl":       "upper_only",
    "tss_mgl":            "upper_only",
    "doc_mgl":            "upper_only",
    "pct_urban":          "upper_only",
    "pct_agriculture":    "upper_only",
    "pct_impervious":     "upper_only",
    "water_temp_c":       "upper_only",
    "embeddedness":       "upper_only",
    # Beneficial — higher is always better, no meaningful upper limit
    "canopy_pct":         "lower_only",
    "riparian_veg":       "lower_only",
    "pct_forest":         "lower_only",
    "lwd":                "lower_only",
    "pool_pct":           "lower_only",
    "substrate_lmm":      "lower_only",
    # Unimodal / unrestricted — algorithm decides (pH, DO, substrate, etc.)
    # "dissolved_oxygen", "pH", "spec_conductance", "substrate_lmm",
    # "sinuosity", "pct_fast_water", "pct_slow_water", "pct_wetland",
    # "pct_shrub", "pct_grassland", "air_temp_c", "precip_mm",
    # "stream_order", "width_m", "elevation_m"
}

def detect_thresholds(grid_orig, curve, feat=None):
    diffs = np.diff(curve)
    n     = len(diffs)
    edge  = max(1, int(n * EDGE_FRAC))

    diffs_in    = diffs[edge: n - edge]
    mu          = diffs_in.mean()
    sd          = diffs_in.std() if diffs_in.std() > 1e-9 else 1e-9
    drop_cutoff = mu - SIG_SD * sd
    rise_cutoff = mu + SIG_SD * sd

    interior_curve = curve[edge: n - edge + 1]
    peak_idx       = int(np.argmax(interior_curve)) + edge

    left_idx  = np.arange(edge, min(peak_idx, n - edge))
    right_idx = np.arange(max(peak_idx, edge), n - edge)

    def find_climb(idx_array):
        if len(idx_array) == 0:
            return None, None
        vals = diffs[idx_array]
        best = int(np.argmax(vals))
        if vals[best] > rise_cutoff:
            gi = idx_array[best]
            return float(grid_orig[gi]), float(curve[gi])
        return None, None

    def find_drop(idx_array):
        if len(idx_array) == 0:
            return None, None
        vals = diffs[idx_array]
        best = int(np.argmin(vals))
        if vals[best] < drop_cutoff:
            gi = idx_array[best]
            return float(grid_orig[gi]), float(curve[gi])
        return None, None

    lower_x, lower_y = find_climb(left_idx)
    if lower_x is None and peak_idx <= edge + 2:
        lower_x, lower_y = find_climb(np.arange(edge, n - edge))

    upper_x, upper_y = find_drop(right_idx)
    if upper_x is None and peak_idx >= n - edge - 2:
        upper_x, upper_y = find_drop(np.arange(edge, n - edge))

    # Apply ecological direction override
    direction = FEATURE_DIRECTION.get(feat, None)
    if direction == "upper_only":
        lower_x, lower_y = None, None
    elif direction == "lower_only":
        upper_x, upper_y = None, None

    if upper_x is not None and lower_x is not None:
        shape = "unimodal"
    elif upper_x is not None:
        shape = "upper_only"
    elif lower_x is not None:
        shape = "lower_only"
    else:
        shape = "flat"

    return {"upper_x": upper_x, "upper_y": upper_y,
            "lower_x": lower_x, "lower_y": lower_y, "shape": shape}


def classify_status(curr_val, thr):
    upper_x = thr["upper_x"]
    lower_x = thr["lower_x"]
    shape   = thr["shape"]
    if shape == "flat":
        return "no_threshold", False
    if shape == "upper_only":
        return ("above_upper", True) if (upper_x and curr_val > upper_x) else ("optimal", False)
    if shape == "lower_only":
        return ("below_lower", True) if (lower_x and curr_val < lower_x) else ("optimal", False)
    if upper_x and curr_val > upper_x:
        return "above_upper", True
    if lower_x and curr_val < lower_x:
        return "below_lower", True
    return "optimal", False


def compute_ice_and_threshold(model, feats, X_orig, scaler, imputer, user_input_orig):
    row_orig = {}
    for f in feats:
        val = user_input_orig.get(f, np.nan)
        try:
            v = float(val)
        except (TypeError, ValueError):
            v = np.nan
        row_orig[f] = v if not np.isnan(v) else float(X_orig[f].median())

    row_arr     = np.array([[row_orig[f] for f in feats]])
    row_scaled  = scaler.transform(row_arr)
    row_imputed = imputer.transform(row_scaled)
    base_pred   = float(model.predict(row_imputed)[0])

    results = {}
    for i, feat in enumerate(feats):
        q01 = float(X_orig[feat].quantile(0.01))
        q99 = float(X_orig[feat].quantile(0.99))
        grid_orig   = np.linspace(q01, q99, N_GRID)
        grid_scaled = (grid_orig - scaler.mean_[i]) / scaler.scale_[i]

        curve = []
        base_row = row_imputed.copy()
        for gv in grid_scaled:
            r = base_row.copy()
            r[0, i] = gv
            curve.append(float(model.predict(r)[0]))
        curve = np.array(curve)

        thr             = detect_thresholds(grid_orig, curve, feat=feat)
        status, flagged = classify_status(float(row_orig[feat]), thr)

        results[feat] = {
            "grid_orig":   grid_orig.tolist(),
            "curve":       curve.tolist(),
            "upper_x":     thr["upper_x"],
            "upper_y":     thr["upper_y"],
            "lower_x":     thr["lower_x"],
            "lower_y":     thr["lower_y"],
            "shape":       thr["shape"],
            "status":      status,
            "is_flagged":  flagged,
            "current_val": float(row_orig[feat]),
            "feat_min":    q01,
            "feat_max":    q99,
        }

    return base_pred, results

# ══════════════════════════════════════════════════════
# HTML
# ══════════════════════════════════════════════════════
HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Stream Biodiversity Predictor</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: #f0f4f8;
  color: #1a1a2e;
  min-height: 100vh;
}
header {
  background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
  color: white;
  padding: 24px 40px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 12px;
}
header h1 { font-size: 1.55rem; font-weight: 800; letter-spacing: -0.3px; }
header p  { font-size: 0.85rem; opacity: 0.65; margin-top: 4px; }
.badges { display: flex; gap: 8px; flex-wrap: wrap; }
.badge {
  padding: 5px 14px; border-radius: 20px;
  font-size: 0.72rem; font-weight: 700; letter-spacing: 0.4px;
}
.badge-full  { background: rgba(59,130,246,0.25); border: 1px solid #3b82f6; color: #93c5fd; }
.badge-human { background: rgba(16,185,129,0.2);  border: 1px solid #10b981; color: #6ee7b7; }
.container { max-width: 1400px; margin: 0 auto; padding: 28px 20px; }
.tabs { display: flex; border-bottom: 2px solid #e2e8f0; margin-bottom: 24px; }
.tab-btn {
  padding: 13px 30px; border: none; background: transparent;
  cursor: pointer; font-size: 0.92rem; font-weight: 600; color: #64748b;
  border-bottom: 3px solid transparent; margin-bottom: -2px; transition: all 0.2s;
}
.tab-btn.active { color: #1d4ed8; border-bottom-color: #1d4ed8; }
.tab-btn:hover:not(.active) { color: #334155; background: #f8fafc; }
.card {
  background: white; border-radius: 14px; padding: 26px 28px;
  box-shadow: 0 2px 10px rgba(0,0,0,0.06); margin-bottom: 20px;
  border: 1px solid #e8ecf0;
}
.card h2 { font-size: 1.05rem; font-weight: 700; margin-bottom: 18px; color: #0f172a; }
.section-note { font-size: 0.82rem; color: #64748b; margin-bottom: 16px; line-height: 1.5; }
.feature-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(190px, 1fr));
  gap: 12px;
}
.feature-input label {
  display: block; font-size: 0.72rem; font-weight: 700;
  color: #475569; margin-bottom: 4px;
  text-transform: uppercase; letter-spacing: 0.4px;
}
.feature-input input {
  width: 100%; padding: 8px 11px; border: 1.5px solid #e2e8f0;
  border-radius: 8px; font-size: 0.88rem; color: #1a1a2e;
  background: #f8fafc; transition: border 0.15s, background 0.15s;
}
.feature-input input:focus { outline: none; border-color: #3b82f6; background: white; }
.hint { font-size: 0.68rem; color: #94a3b8; margin-top: 3px; }
.predict-btn {
  margin-top: 18px;
  background: linear-gradient(135deg, #1d4ed8, #2563eb);
  color: white; border: none; padding: 13px 38px;
  border-radius: 10px; font-size: 0.98rem; font-weight: 700;
  cursor: pointer; box-shadow: 0 4px 14px rgba(37,99,235,0.35);
  transition: transform 0.15s, box-shadow 0.15s;
}
.predict-btn:hover { transform: translateY(-1px); box-shadow: 0 6px 20px rgba(37,99,235,0.45); }
.predict-btn:disabled { opacity: 0.55; cursor: not-allowed; transform: none; }
.loader { display:none; text-align:center; padding:40px; color:#64748b; }
.spinner {
  width:36px; height:36px; border:3px solid #e2e8f0; border-top-color:#3b82f6;
  border-radius:50%; animation: spin 0.75s linear infinite; margin: 0 auto 12px;
}
@keyframes spin { to { transform: rotate(360deg); } }
#results { display: none; }
.prediction-hero { display:flex; align-items:center; gap:32px; flex-wrap:wrap; }
.shannon-display { text-align:center; min-width:120px; }
.shannon-value { font-size: 3.8rem; font-weight:900; line-height:1; }
.shannon-label { font-size:0.8rem; color:#64748b; margin-top:4px; font-weight:600; }
.health-badge {
  padding:7px 18px; border-radius:20px; font-weight:700;
  font-size:0.85rem; margin-top:10px; display:inline-block;
}
.health-high   { background:#dcfce7; color:#166534; }
.health-medium { background:#fef9c3; color:#854d0e; }
.health-low    { background:#fee2e2; color:#991b1b; }
.threshold-table { width:100%; border-collapse:collapse; font-size:0.86rem; }
.threshold-table th {
  background:#f1f5f9; padding:10px 14px; text-align:left;
  font-weight:700; color:#475569; font-size:0.74rem;
  text-transform:uppercase; letter-spacing:0.4px;
}
.threshold-table td { padding:10px 14px; border-bottom:1px solid #f1f5f9; color:#334155; }
.threshold-table tr:hover td { background:#f8fafc; }
.status-dot { width:9px; height:9px; border-radius:50%; display:inline-block; margin-right:6px; }
.dot-safe   { background:#22c55e; }
.dot-danger { background:#ef4444; }
.row-danger td { background:#fff5f5 !important; }
.row-danger td:first-child { border-left: 3px solid #ef4444; }
.plots-grid {
  display: grid; grid-template-columns: repeat(auto-fill, minmax(310px, 1fr)); gap: 14px;
}
.plot-card {
  background:white; border-radius:12px; padding:14px;
  box-shadow:0 2px 8px rgba(0,0,0,0.05); border:1.5px solid #e2e8f0;
}
.plot-card h4 { font-size:0.82rem; font-weight:700; color:#0f172a; margin-bottom:8px; }
.plot-card img { width:100%; border-radius:6px; }
.reco-box {
  background:#f0f9ff; border:1.5px solid #bae6fd; border-radius:12px;
  padding:22px; color:#0c4a6e; font-size:0.88rem; line-height:1.65;
}
.reco-box h3 { font-size:0.98rem; font-weight:700; margin-bottom:12px; color:#0369a1; }
.reco-item {
  padding:8px 0; border-bottom:1px solid #e0f2fe;
  display:flex; gap:10px; align-items:flex-start;
}
.reco-item:last-child { border-bottom:none; }
.reco-label { font-size:0.78rem; font-weight:700; min-width:60px; padding-top:2px; }
.reco-label.danger  { color:#ef4444; }
.reco-label.warning { color:#f97316; }
.reco-label.monitor { color:#854d0e; }
.reco-label.ok      { color:#166534; }
</style>
</head>
<body>

<header>
  <div>
    <h1>Stream Biodiversity Predictor</h1>
    <p>Predict Shannon Diversity H' and identify ecological thresholds for your stream site</p>
  </div>
  <div class="badges">
    <span class="badge badge-full">Full Model R² = 0.813</span>
    <span class="badge badge-human">Management Model R² = 0.774</span>
  </div>
</header>

<div class="container">

  <div class="tabs">
    <button class="tab-btn active" id="btn-full" onclick="setModel('full')">
      Full Model (31 features)
    </button>
    <button class="tab-btn" id="btn-human" onclick="setModel('human')">
      Management-Relevant Model (26 features)
    </button>
  </div>

  <div class="card">
    <h2>Enter Stream Features</h2>
    <p class="section-note">
      Enter your measured values below. Any field left blank will use the dataset median for that feature.
    </p>
    <div class="feature-grid" id="feature-grid"></div>
    <br>
    <button class="predict-btn" onclick="runPredict()" id="predict-btn">
      Predict Shannon Diversity
    </button>
  </div>

  <div class="loader" id="loader">
    <div class="spinner"></div>
    Computing ICE curves and thresholds — this may take 15–30 seconds...
  </div>

  <div id="results">

    <div class="card">
      <h2>Prediction Result</h2>
      <div class="prediction-hero">
        <div class="shannon-display">
          <div class="shannon-value" id="shannon-val">—</div>
          <div class="shannon-label">Shannon H'</div>
          <div class="health-badge" id="health-badge">—</div>
        </div>
        <div style="flex:1; min-width:260px;">
          <p style="font-size:0.9rem; color:#475569; line-height:1.65;" id="prediction-context"></p>
        </div>
      </div>
    </div>

    <div class="card">
      <h2>Feature Thresholds</h2>
      <p class="section-note">
        Thresholds are computed from ICE curves specific to this stream's conditions — all other features
        are held at your entered values. The upper threshold marks where increasing a feature causes the
        sharpest drop in diversity. The lower threshold marks the value below which diversity has not yet
        risen to its beneficial level. Red rows indicate your current value has passed a threshold.
      </p>
      <table class="threshold-table">
        <thead>
          <tr>
            <th>Feature</th>
            <th>Your Value</th>
            <th>Upper Threshold</th>
            <th>Lower Threshold</th>
            <th>Delta from Threshold</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody id="threshold-tbody"></tbody>
      </table>
    </div>

    <div class="card">
      <h2>ICE Curves — Feature Response</h2>
      <p class="section-note">
        Each curve shows how predicted Shannon H' changes as one feature varies across its observed range,
        with all other features held at your inputs. The red dashed line marks the upper threshold,
        the orange dashed line marks the lower threshold, and the dotted line marks your current value.
      </p>
      <div class="plots-grid" id="plots-grid"></div>
    </div>

    <div class="card">
      <h2>Management Recommendations</h2>
      <div class="reco-box">
        <h3>Priority Actions for This Stream Site</h3>
        <div id="reco-list"></div>
      </div>
    </div>

  </div>
</div>

<script>
let currentModel = "full";
const featureData = {{ feature_data|tojson }};

function setModel(m) {
  currentModel = m;
  document.getElementById("btn-full").classList.toggle("active",  m === "full");
  document.getElementById("btn-human").classList.toggle("active", m === "human");
  buildForm();
  document.getElementById("results").style.display = "none";
}

function buildForm() {
  const features = featureData[currentModel].features;
  const medians  = featureData[currentModel].medians;
  const grid     = document.getElementById("feature-grid");
  grid.innerHTML = "";
  features.forEach(feat => {
    const label = featureData.labels[feat] || feat;
    const med   = medians[feat] !== undefined ? medians[feat].toFixed(3) : "";
    const div   = document.createElement("div");
    div.className = "feature-input";
    div.innerHTML = `
      <label>${label}</label>
      <input type="number" step="any" id="inp_${feat}" placeholder="${med}">
      <div class="hint">median: ${med}</div>`;
    grid.appendChild(div);
  });
}

async function runPredict() {
  const features = featureData[currentModel].features;
  const medians  = featureData[currentModel].medians;
  const values   = {};
  features.forEach(feat => {
    const el  = document.getElementById("inp_" + feat);
    const val = el ? parseFloat(el.value) : NaN;
    values[feat] = isNaN(val) ? medians[feat] : val;
  });

  document.getElementById("loader").style.display  = "block";
  document.getElementById("results").style.display = "none";
  document.getElementById("predict-btn").disabled  = true;

  const resp = await fetch("/predict", {
    method:  "POST",
    headers: {"Content-Type": "application/json"},
    body:    JSON.stringify({ model: currentModel, values })
  });
  const data = await resp.json();

  document.getElementById("loader").style.display  = "none";
  document.getElementById("predict-btn").disabled  = false;
  document.getElementById("results").style.display = "block";

  const h = data.prediction;
  document.getElementById("shannon-val").textContent = h.toFixed(3);
  document.getElementById("shannon-val").style.color =
    h >= 2.0 ? "#166534" : h >= 1.2 ? "#854d0e" : "#991b1b";
  const badge = document.getElementById("health-badge");
  if      (h >= 2.0) { badge.textContent = "High Diversity";     badge.className = "health-badge health-high"; }
  else if (h >= 1.2) { badge.textContent = "Moderate Diversity"; badge.className = "health-badge health-medium"; }
  else               { badge.textContent = "Low Diversity";       badge.className = "health-badge health-low"; }

  const nFlagged = data.thresholds.filter(t => t.is_flagged).length;
  document.getElementById("prediction-context").textContent =
    `Predicted Shannon H' = ${h.toFixed(3)}. ` +
    `${nFlagged} of ${data.thresholds.length} features have exceeded or fallen below ` +
    `their ecological thresholds for this stream's specific conditions. ` +
    `Features shown in red are priority restoration targets.`;

  const tbody = document.getElementById("threshold-tbody");
  tbody.innerHTML = "";
  data.thresholds.forEach(t => {
    const label    = featureData.labels[t.feature] || t.feature;
    const flagged  = t.is_flagged;
    const upperStr = t.upper_x !== null ? t.upper_x.toFixed(3) : "—";
    const lowerStr = t.lower_x !== null ? t.lower_x.toFixed(3) : "—";

    let deltaStr = "—";
    if (t.status === "above_upper" && t.upper_x !== null) {
      const d = (t.current_val - t.upper_x).toFixed(3);
      deltaStr = `<span style="color:#ef4444;font-weight:700;">+${d} above upper</span>`;
    } else if (t.status === "below_lower" && t.lower_x !== null) {
      const d = (t.lower_x - t.current_val).toFixed(3);
      deltaStr = `<span style="color:#f97316;font-weight:700;">-${d} below lower</span>`;
    }

    let statusStr = t.status === "above_upper" ? "Above upper threshold"
                  : t.status === "below_lower" ? "Below lower threshold"
                  : t.status === "optimal"      ? "Within safe range"
                  :                               "No threshold detected";

    tbody.innerHTML += `
      <tr class="${flagged ? 'row-danger' : ''}">
        <td><strong>${label}</strong></td>
        <td>${t.current_val.toFixed(3)}</td>
        <td style="color:#ef4444">${upperStr}</td>
        <td style="color:#f97316">${lowerStr}</td>
        <td>${deltaStr}</td>
        <td><span class="status-dot ${flagged ? 'dot-danger' : 'dot-safe'}"></span>${statusStr}</td>
      </tr>`;
  });

  const plotsGrid = document.getElementById("plots-grid");
  plotsGrid.innerHTML = "";
  data.plots.forEach(p => {
    const label = featureData.labels[p.feature] || p.feature;
    const div   = document.createElement("div");
    div.className = "plot-card";
    div.innerHTML = `<h4>${label}</h4><img src="data:image/png;base64,${p.img}" alt="${label}">`;
    plotsGrid.appendChild(div);
  });

  const recoList = document.getElementById("reco-list");
  recoList.innerHTML = "";
  const flagged = data.thresholds.filter(t => t.is_flagged);
  const monitor = data.thresholds.filter(t => !t.is_flagged && t.shape !== "flat").slice(0, 3);

  if (flagged.length === 0) {
    recoList.innerHTML = `<div class="reco-item">
      <span class="reco-label ok">All clear</span>
      <span>All features are within their predicted safe ranges for this stream's conditions.
      Continue monitoring and maintaining current conditions.</span>
    </div>`;
  } else {
    flagged.forEach(t => {
      const label = featureData.labels[t.feature] || t.feature;
      let msg = "";
      if (t.status === "above_upper" && t.upper_x !== null) {
        const excess = (t.current_val - t.upper_x).toFixed(3);
        msg = `<strong>${label}</strong> exceeds its upper threshold (current: ${t.current_val.toFixed(3)}, threshold: ${t.upper_x.toFixed(3)}, excess: ${excess}). Reducing toward ${t.upper_x.toFixed(3)} is a priority.`;
      } else if (t.status === "below_lower" && t.lower_x !== null) {
        const deficit = (t.lower_x - t.current_val).toFixed(3);
        msg = `<strong>${label}</strong> is below its beneficial range (current: ${t.current_val.toFixed(3)}, threshold: ${t.lower_x.toFixed(3)}, deficit: ${deficit}). Increasing toward ${t.lower_x.toFixed(3)} or above is recommended.`;
      }
      const cls = t.status === "above_upper" ? "danger" : "warning";
      recoList.innerHTML += `<div class="reco-item">
        <span class="reco-label ${cls}">Action</span>
        <span>${msg}</span>
      </div>`;
    });
    if (monitor.length > 0) {
      recoList.innerHTML += `<div class="reco-item">
        <span class="reco-label monitor">Monitor</span>
        <span>Continue monitoring: ${monitor.map(t => featureData.labels[t.feature] || t.feature).join(", ")}.</span>
      </div>`;
    }
  }

  window.scrollTo({ top: document.getElementById("results").offsetTop - 20, behavior: "smooth" });
}

buildForm();
</script>
</body>
</html>
"""

# ══════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════
@app.route("/")
def index():
    feature_data = {
        "full":  {"features": FEATS_F, "medians": {f: float(X_ORIG_F[f].median()) for f in FEATS_F}},
        "human": {"features": FEATS_H, "medians": {f: float(X_ORIG_H[f].median()) for f in FEATS_H}},
        "labels": FEATURE_LABELS,
    }
    return render_template_string(HTML, feature_data=feature_data)


@app.route("/predict", methods=["POST"])
def predict():
    data   = request.get_json()
    m_type = data["model"]
    values = data["values"]

    if m_type == "full":
        model, feats, X_orig, scaler, imputer = MODEL_F, FEATS_F, X_ORIG_F, SCALER_F, IMPUTER_F
    else:
        model, feats, X_orig, scaler, imputer = MODEL_H, FEATS_H, X_ORIG_H, SCALER_H, IMPUTER_H

    user_input = {f: float(values.get(f, X_orig[f].median())) for f in feats}
    base_pred, ice_results = compute_ice_and_threshold(
        model, feats, X_orig, scaler, imputer, user_input)

    STATUS_LABEL = {
        "above_upper":  "Above upper threshold",
        "below_lower":  "Below lower threshold",
        "optimal":      "Within safe range",
        "no_threshold": "No threshold detected",
    }

    thresholds = []
    for feat, res in ice_results.items():
        curr = res["current_val"]
        excess = (abs(curr - res["upper_x"]) if res["status"] == "above_upper" and res["upper_x"]
                  else abs(curr - res["lower_x"]) if res["status"] == "below_lower" and res["lower_x"]
                  else 0.0)
        thresholds.append({
            "feature":      feat,
            "current_val":  curr,
            "upper_x":      res["upper_x"],
            "upper_y":      res["upper_y"],
            "lower_x":      res["lower_x"],
            "lower_y":      res["lower_y"],
            "shape":        res["shape"],
            "status":       res["status"],
            "status_label": STATUS_LABEL.get(res["status"], res["status"]),
            "is_flagged":   res["is_flagged"],
            "excess":       excess,
        })
    thresholds.sort(key=lambda x: (not x["is_flagged"], -x["excess"]))

    plots = []
    for feat, res in ice_results.items():
        fig, ax = plt.subplots(figsize=(4.2, 3.0))
        grid  = np.array(res["grid_orig"])
        curve = np.array(res["curve"])

        ax.plot(grid, curve, color="#3b82f6", lw=2.2, zorder=3)

        if res["upper_x"] is not None:
            ax.axvline(res["upper_x"], color="#ef4444", lw=1.8, ls="--", alpha=0.9,
                       label="Upper threshold")
            ax.axvspan(res["upper_x"], grid.max(), color="#fee2e2", alpha=0.4, zorder=1)

        if res["lower_x"] is not None:
            ax.axvline(res["lower_x"], color="#f97316", lw=1.8, ls="--", alpha=0.9,
                       label="Lower threshold")
            ax.axvspan(grid.min(), res["lower_x"], color="#fef3c7", alpha=0.4, zorder=1)

        curr_color = "#ef4444" if res["is_flagged"] else "#166534"
        ax.axvline(res["current_val"], color=curr_color, lw=1.8, ls=":",
                   label="Current value")

        ax.set_xlabel(FEATURE_LABELS.get(feat, feat), fontsize=7)
        ax.set_ylabel("Predicted Shannon H'", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=5.5, loc="best", framealpha=0.85)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.15, lw=0.5)
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        plots.append({"feature": feat, "img": base64.b64encode(buf.read()).decode()})

    return jsonify({"prediction": float(base_pred), "thresholds": thresholds, "plots": plots})


if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  Stream Biodiversity Predictor")
    print("  http://localhost:5000")
    print("=" * 55 + "\n")
    app.run(debug=False, port=5000)