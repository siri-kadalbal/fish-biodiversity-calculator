import streamlit as st

# Custom CSS to hide the GitHub icon, Fork button, and Streamlit menu
hide_style = """
    <style>
    /* Hides the top header bar (contains GitHub icon and Fork button) */
    header {visibility: hidden;}
    
    /* Hides the 'three-dot' main menu on the top right */
    #MainMenu {visibility: hidden;}
    
    /* Optional: Hides the 'Made with Streamlit' footer at the bottom */
    footer {visibility: hidden;}
    
    /* This ensures the app content starts at the top of the page */
    .block-container {
        padding-top: 2rem;
    }
    </style>
    """

st.markdown(hide_style, unsafe_allow_html=True)
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io, base64
import xgboost as xgb
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════
# PAGE CONFIG (MUST BE FIRST)
# ══════════════════════════════════════════════════════
st.set_page_config(page_title="Stream Biodiversity Predictor", layout="wide")

# ══════════════════════════════════════════════════════
# DATA & CONSTANTS
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
}

FEATURE_DIRECTION = {
    "turbidity": "upper_only", "total_nitrogen_mgl": "upper_only", "ammonia_mgl": "upper_only",
    "chloride_mgl": "upper_only", "tss_mgl": "upper_only", "doc_mgl": "upper_only",
    "pct_urban": "upper_only", "pct_agriculture": "upper_only", "pct_impervious": "upper_only",
    "water_temp_c": "upper_only", "embeddedness": "upper_only", "canopy_pct": "lower_only",
    "riparian_veg": "lower_only", "pct_forest": "lower_only", "lwd": "lower_only",
    "pool_pct": "lower_only", "substrate_lmm": "lower_only",
}

# ══════════════════════════════════════════════════════
# HELPER CLASSES & FUNCTIONS
# ══════════════════════════════════════════════════════
class MockDF:
    def __init__(self, medians, quantiles):
        self.medians = medians
        self.quantiles = quantiles
    def median(self): return pd.Series(self.medians)
    def __getitem__(self, key):
        class Col:
            def __init__(self, q): self.q = q
            def quantile(self, val): return self.q[0] if val < 0.5 else self.q[1]
            def median(self): return self.medians[key]
        return Col(self.quantiles[key])

@st.cache_resource
def load_bundle():
    bundle = joblib.load("deployable_bundle.joblib")
    # Wrap objects for easier access
    bundle["full"]["X_orig"] = MockDF(bundle["full"]["medians"], bundle["full"]["quantiles"])
    bundle["human"]["X_orig"] = MockDF(bundle["human"]["medians"], bundle["human"]["quantiles"])
    return bundle

def detect_thresholds(grid_orig, curve, feat=None):
    diffs = np.diff(curve)
    n = len(diffs)
    edge = max(1, int(n * 0.03)) # EDGE_FRAC
    diffs_in = diffs[edge: n - edge]
    mu, sd = diffs_in.mean(), (diffs_in.std() if diffs_in.std() > 1e-9 else 1e-9)
    drop_cutoff, rise_cutoff = mu - 1.5 * sd, mu + 1.5 * sd # SIG_SD
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

# ══════════════════════════════════════════════════════
# MAIN APP INTERFACE
# ══════════════════════════════════════════════════════
data_bundle = load_bundle()

st.title("Stream Biodiversity Predictor")
st.markdown("Predict Shannon Diversity H' and identify ecological thresholds based on model interpretability (ICE).")

# Model Selection
m_mode = st.radio("Select Model Scope:", ["Full (31 features)", "Management (26 features)"], horizontal=True)
m_key = "full" if "Full" in m_mode else "human"
bundle = data_bundle[m_key]
feats, X_orig = bundle["features"], bundle["X_orig"]

# Inputs Card
st.subheader("1. Enter Stream Features")
st.info("Blank fields will automatically use the dataset median.")
with st.container(border=True):
    cols = st.columns(4)
    user_vals = {}
    for i, f in enumerate(feats):
        with cols[i % 4]:
            label = FEATURE_LABELS.get(f, f)
            med = float(bundle["medians"][f])
            val = st.text_input(label, placeholder=f"{med:.3f}", key=f)
            user_vals[f] = float(val) if val else med

# Prediction Logic
if st.button("Run Prediction & Analysis", type="primary", use_container_width=True):
    with st.spinner("Calculating ICE curves and thresholds..."):
        # Predict
        row_arr = np.array([[user_vals[f] for f in feats]])
        row_scaled = bundle["scaler"].transform(row_arr)
        row_imputed = bundle["imputer"].transform(row_scaled)
        base_pred = float(bundle["model"].predict(row_imputed)[0])

        # Hero Result
        st.divider()
        c1, c2 = st.columns([1, 3])
        with c1:
            st.metric("Shannon H'", f"{base_pred:.3f}")
        with c2:
            if base_pred >= 2.0: st.success("High Diversity Site")
            elif base_pred >= 1.2: st.warning("Moderate Diversity Site")
            else: st.error("Low Diversity Site")

        # Thresholds Calculation
        st.subheader("2. Ecological Thresholds")
        threshold_data = []
        plot_figs = []

        for i, f in enumerate(feats):
            q01, q99 = float(X_orig[f].quantile(0.01)), float(X_orig[f].quantile(0.99))
            grid_orig = np.linspace(q01, q99, 80)
            grid_scaled = (grid_orig - bundle["scaler"].mean_[i]) / bundle["scaler"].scale_[i]
            
            curve = []
            base_row = row_imputed.copy()
            for gv in grid_scaled:
                r = base_row.copy()
                r[0, i] = gv
                curve.append(float(bundle["model"].predict(r)[0]))
            curve = np.array(curve)
            
            thr = detect_thresholds(grid_orig, curve, feat=f)
            
            # Status check
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

            # Prepare Plot
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.plot(grid_orig, curve, color="#3b82f6", lw=2)
            if thr["upper_x"]: ax.axvline(thr["upper_x"], color="#ef4444", ls="--")
            if thr["lower_x"]: ax.axvline(thr["lower_x"], color="#f97316", ls="--")
            ax.axvline(user_vals[f], color=("red" if flagged else "green"), ls=":")
            ax.set_title(FEATURE_LABELS.get(f, f), fontsize=9)
            plt.tight_layout()
            plot_figs.append(fig)

        # Show Table
        df_thr = pd.DataFrame(threshold_data)
        st.dataframe(df_thr.drop(columns="flagged"), use_container_width=True)

        # Show Plots
        st.subheader("3. Feature Response Curves (ICE)")
        p_cols = st.columns(3)
        for idx, fig in enumerate(plot_figs):
            with p_cols[idx % 3]:
                st.pyplot(fig)
