import streamlit as st

hide_style = """
    <style>
    header {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .block-container { padding-top: 2rem; }
    </style>
    """
st.markdown(hide_style, unsafe_allow_html=True)

import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import warnings
from plausibility_checks import check_all_plausibility
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Stream Biodiversity Predictor", layout="wide")

# ══════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════
FEATURE_LABELS = {
    "water_temp_c": "Water Temperature (C)", "dissolved_oxygen": "Dissolved Oxygen (mg/L)",
    "spec_conductance": "Specific Conductance (uS/cm)", "pH": "pH", "turbidity": "Turbidity (NTU)",
    "total_nitrogen_mgl": "Total Nitrogen (mg/L)", "ammonia_mgl": "Ammonia (mg/L)",
    "doc_mgl": "DOC (mg/L)", "chloride_mgl": "Chloride (mg/L)", "tss_mgl": "TSS (mg/L)",
    "pct_fast_water": "% Fast Water", "pct_slow_water": "% Slow Water", "substrate_lmm": "Substrate (lmm)",
    "embeddedness": "Embeddedness", "sinuosity": "Sinuosity", "canopy_pct": "Canopy Cover (%)",
    "riparian_veg": "Riparian Vegetation", "pool_pct": "Pool Cover (%)", "lwd": "Large Woody Debris",
    "pct_urban": "% Urban", "pct_forest": "% Forest", "pct_agriculture": "% Agriculture",
    "pct_wetland": "% Wetland", "pct_impervious": "% Impervious Surface", "pct_shrub": "% Shrubland",
    "pct_grassland": "% Grassland", "air_temp_c": "Air Temperature (C)", "precip_mm": "Precipitation (mm)",
    "stream_order": "Stream Order", "width_m": "Stream Width (m)", "elevation_m": "Elevation (m)",
    # NEW: invertebrate labels
    "invert_total_abund": "Invertebrate Total Abundance", "invert_richness": "Invertebrate Taxa Richness",
    "invert_ept_richness": "EPT Taxa Richness", "invert_pct_ept": "% EPT Abundance",
    "invert_hbi": "Hilsenhoff Biotic Index", "invert_pct_dominant": "% Dominant Taxon",
    "pct_ffg_CF": "% Collector-Filterers", "pct_ffg_SH": "% Shredders",
    "pct_ffg_CG": "% Collector-Gatherers", "pct_ffg_SC": "% Scrapers", "pct_ffg_PR": "% Predators",
}

FEATURE_DIRECTION = {
    "turbidity": "upper_only", "total_nitrogen_mgl": "upper_only", "ammonia_mgl": "upper_only",
    "chloride_mgl": "upper_only", "tss_mgl": "upper_only", "doc_mgl": "upper_only",
    "pct_urban": "upper_only", "pct_agriculture": "upper_only", "pct_impervious": "upper_only",
    "water_temp_c": "upper_only", "embeddedness": "upper_only", "canopy_pct": "lower_only",
    "riparian_veg": "lower_only", "pct_forest": "lower_only", "lwd": "lower_only",
    "pool_pct": "lower_only", "substrate_lmm": "lower_only",
    "invert_hbi": "upper_only", "invert_pct_dominant": "upper_only",  # NEW
    "invert_richness": "lower_only", "invert_ept_richness": "lower_only",  # NEW
}

ECO_NAMES = {
    "SAP": "Southern Appalachians", "NAP": "Northern Appalachians",
    "CPL": "Coastal Plains",        "UMW": "Upper Midwest",
    "TPL": "Temperate Plains",      "SPL": "Southern Plains",
    "NPL": "Northern Plains",       "WMT": "Western Mountains",
    "XER": "Xeric",
}

# ── NEW: physiological DO floor, same rule used in retraining ──
DO_LETHAL_FLOOR  = 0.5
DO_SEVERE_STRESS = 2.0

def apply_do_floor(do_value, raw_pred):
    if do_value is None:
        return raw_pred, None
    if do_value <= DO_LETHAL_FLOOR:
        return 0.0, (f"DO ({do_value:.2f} mg/L) is at or below the lethal floor. "
                      f"Diversity forced to 0 regardless of model output, the model "
                      f"was not trained on true anoxic conditions.")
    if do_value <= DO_SEVERE_STRESS:
        capped = min(raw_pred, 1.0)
        if capped < raw_pred:
            return capped, f"DO ({do_value:.2f} mg/L) is in severe hypoxic stress range; prediction capped at {capped:.2f}."
    return raw_pred, None


# ══════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════
class MockDF:
    def __init__(self, medians, quantiles, mins, maxs):
        self.medians, self.quantiles, self.mins, self.maxs = medians, quantiles, mins, maxs
    def median(self): return pd.Series(self.medians)
    def __getitem__(self, key):
        class Col:
            def __init__(self, outer, key): self.outer, self.key = outer, key
            def quantile(self, val): return self.outer.quantiles[self.key][0] if val < 0.5 else self.outer.quantiles[self.key][1]
            def median(self): return self.outer.medians[self.key]
            def min(self): return self.outer.mins[self.key]
            def max(self): return self.outer.maxs[self.key]
        return Col(self, key)

@st.cache_resource
def load_bundle():
    bundle = joblib.load("deployable_bundle.joblib")
    for key in ("full", "human"):
        b = bundle[key]
        # NEW: fall back gracefully if bundle predates the mins/maxs addition
        mins = b.get("mins", {f: b["quantiles"][f][0] for f in b["features"]})
        maxs = b.get("maxs", {f: b["quantiles"][f][1] for f in b["features"]})
        b["X_orig"] = MockDF(b["medians"], b["quantiles"], mins, maxs)
    return bundle

def detect_thresholds(grid_orig, curve, feat=None):
    diffs = np.diff(curve)
    n = len(diffs)
    edge = max(1, int(n * 0.03))
    diffs_in = diffs[edge: n - edge]
    mu, sd = diffs_in.mean(), (diffs_in.std() if diffs_in.std() > 1e-9 else 1e-9)
    drop_cutoff, rise_cutoff = mu - 1.5 * sd, mu + 1.5 * sd
    peak_idx = int(np.argmax(curve[edge: n - edge + 1])) + edge

    def find_val(idx_array, is_drop):
        if len(idx_array) == 0: return None, None
        vals = diffs[idx_array]
        best = int(np.argmin(vals)) if is_drop else int(np.argmax(vals))
        if (is_drop and vals[best] < drop_cutoff) or (not is_drop and vals[best] > rise_cutoff):
            gi = idx_array[best]
            return float(grid_orig[gi]), float(curve[gi])
        return None, None

    lower_x, lower_y = find_val(np.arange(edge, min(peak_idx, n - edge)), False)
    upper_x, upper_y = find_val(np.arange(max(peak_idx, edge), n - edge), True)

    direction = FEATURE_DIRECTION.get(feat)
    if direction == "upper_only": lower_x, lower_y = None, None
    elif direction == "lower_only": upper_x, upper_y = None, None

    shape = "flat"
    if upper_x and lower_x: shape = "unimodal"
    elif upper_x: shape = "upper_only"
    elif lower_x: shape = "lower_only"

    return {"upper_x": upper_x, "upper_y": upper_y, "lower_x": lower_x, "lower_y": lower_y, "shape": shape}

def check_range(feat, val, X_orig):
    """NEW: returns a warning string if val is outside the training range."""
    q01, q99 = X_orig[feat].quantile(0.01), X_orig[feat].quantile(0.99)
    fmin, fmax = X_orig[feat].min(), X_orig[feat].max()
    if val < fmin or val > fmax:
        return f"**{FEATURE_LABELS.get(feat, feat)}** = {val} is outside the full training range ({fmin:.3g} to {fmax:.3g}). This is extrapolation."
    if val < q01 or val > q99:
        return f"**{FEATURE_LABELS.get(feat, feat)}** = {val} is outside the well-supported range ({q01:.3g} to {q99:.3g})."
    return None


# ══════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════
data_bundle = load_bundle()

st.title("Stream Biodiversity Predictor")
st.markdown("Predict Shannon Diversity H' and identify ecological thresholds specific to your stream's ecoregion.")

m_mode = st.radio("Select Model Scope:", ["Full", "Management (controllable features)"], horizontal=True)
m_key = "full" if "Full" in m_mode else "human"
bundle = data_bundle[m_key]
feats, X_orig = bundle["features"], bundle["X_orig"]

# NEW: split out ecoregion dummy columns from the typed-input features
eco_feats  = [f for f in feats if f.startswith("eco_")]
base_feats = [f for f in feats if not f.startswith("eco_")]
available_ecoregions = sorted({f.replace("eco_", "") for f in eco_feats})

st.subheader("1. Select Ecoregion")
if available_ecoregions:
    eco_choice = st.selectbox(
        "Ecoregion",
        available_ecoregions,
        format_func=lambda code: f"{ECO_NAMES.get(code, code)} ({code})")
else:
    eco_choice = None
    st.info("This model version doesn't have ecoregion data. Retrain with build_deployable_bundle.py to enable it.")

st.subheader("2. Enter Stream Features")
st.info("Blank fields default to the median for the SELECTED ecoregion where available, otherwise the national median.")

def median_for_ecoregion(feat):
    # Falls back to bundle median (national) since MockDF doesn't carry raw
    # per-ecoregion rows; for a true per-ecoregion median, compute this in
    # build_deployable_bundle.py and store it in the bundle per ecoregion.
    return bundle["medians"][feat]

with st.container(border=True):
    cols = st.columns(4)
    user_vals = {}
    for i, f in enumerate(base_feats):
        with cols[i % 4]:
            label = FEATURE_LABELS.get(f, f)
            med = float(median_for_ecoregion(f))
            val = st.text_input(label, placeholder=f"{med:.3f}", key=f)
            user_vals[f] = float(val) if val else med

# NEW: set the one-hot ecoregion columns from the selectbox
for eco_col in eco_feats:
    user_vals[eco_col] = 1.0 if eco_choice and eco_col == f"eco_{eco_choice}" else 0.0

if st.button("Run Prediction & Analysis", type="primary", use_container_width=True):
    # NEW: range warnings on typed values before predicting
    warnings_list = [w for f in base_feats
                      if (w := check_range(f, user_vals[f], X_orig)) is not None]
    warnings_list += check_all_plausibility(user_vals)
    if warnings_list:
        with st.expander(f"⚠️ {len(warnings_list)} input(s) outside training range", expanded=True):
            for w in warnings_list:
                st.warning(w)

    with st.spinner("Calculating ICE curves and thresholds..."):
        row_arr = np.array([[user_vals[f] for f in feats]])
        row_scaled = bundle["scaler"].transform(row_arr)
        row_imputed = bundle["imputer"].transform(row_scaled)
        base_pred = float(bundle["model"].predict(row_imputed)[0])

        # NEW: apply DO floor to the headline prediction
        base_pred, do_note = apply_do_floor(user_vals.get("dissolved_oxygen"), base_pred)

        st.divider()
        c1, c2 = st.columns([1, 3])
        with c1:
            st.metric("Shannon H'", f"{base_pred:.3f}")
        with c2:
            if base_pred >= 2.0: st.success("High Diversity Site")
            elif base_pred >= 1.2: st.warning("Moderate Diversity Site")
            else: st.error("Low Diversity Site")
        if do_note:
            st.error(f"**Dissolved oxygen override:** {do_note}")

        st.subheader("3. Ecological Thresholds")
        threshold_data = []
        plot_figs = []

        for i, f in enumerate(feats):
            if f.startswith("eco_"):
                continue  # don't build an ICE curve for a dummy indicator column
            q01, q99 = float(X_orig[f].quantile(0.01)), float(X_orig[f].quantile(0.99))
            grid_orig = np.linspace(q01, q99, 80)
            grid_scaled = (grid_orig - bundle["scaler"].mean_[i]) / bundle["scaler"].scale_[i]

            curve = []
            base_row = row_imputed.copy()
            for grid_idx, gv in enumerate(grid_scaled):
                r = base_row.copy()
                r[0, i] = gv
                pred = float(bundle["model"].predict(r)[0])
                if f == "dissolved_oxygen":
                    # apply the floor using the real (unscaled) DO value at this grid point
                    pred, _ = apply_do_floor(grid_orig[grid_idx], pred)
                curve.append(pred)
            curve = np.array(curve)

            thr = detect_thresholds(grid_orig, curve, feat=f)

            status = "optimal"
            flagged = False
            if thr["shape"] != "flat":
                if thr["upper_x"] and user_vals[f] > thr["upper_x"]:
                    status, flagged = "Above upper threshold", True
                elif thr["lower_x"] and user_vals[f] < thr["lower_x"]:
                    status, flagged = "Below lower threshold", True

            threshold_data.append({
                "Feature": FEATURE_LABELS.get(f, f),
                "Your Value": user_vals[f],
                "Upper": thr["upper_x"],
                "Lower": thr["lower_x"],
                "Status": status,
                "flagged": flagged
            })

            fig, ax = plt.subplots(figsize=(4, 3))
            ax.plot(grid_orig, curve, color="#3b82f6", lw=2)
            if thr["upper_x"]: ax.axvline(thr["upper_x"], color="#ef4444", ls="--")
            if thr["lower_x"]: ax.axvline(thr["lower_x"], color="#f97316", ls="--")
            ax.axvline(user_vals[f], color=("red" if flagged else "green"), ls=":")
            ax.set_title(FEATURE_LABELS.get(f, f), fontsize=9)
            plt.tight_layout()
            plot_figs.append(fig)

        df_thr = pd.DataFrame(threshold_data)
        st.dataframe(df_thr.drop(columns="flagged"), use_container_width=True)

        st.subheader("4. Feature Response Curves (ICE)")
        p_cols = st.columns(3)
        for idx, fig in enumerate(plot_figs):
            with p_cols[idx % 3]:
                st.pyplot(fig)
