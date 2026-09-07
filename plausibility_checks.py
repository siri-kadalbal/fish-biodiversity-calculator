"""
Combined plausibility checker. Catches physically or logically impossible
input COMBINATIONS that per-feature training-range checks (check_range /
bounds_check.py) can't catch, since each individual value can look perfectly
normal in isolation.

Three kinds of checks here:
  1. Absolute bounds  : true regardless of training data (a percentage > 100
                        is impossible no matter what the model has seen)
  2. Group sums       : features that are fractions of the same whole
  3. Ordering/subset  : one feature must logically be <= another

Usage (Streamlit or Flask, same idea either way):

    from plausibility_checks import check_all_plausibility

    warnings_list = check_all_plausibility(user_vals)
    for w in warnings_list:
        st.warning(w)   # or however your app surfaces range_flags today
"""

# ── 1. Absolute bounds: true by definition, not just "unusual" ──
# Add more here if you have other true percentage or physically-bounded features.
ABSOLUTE_BOUNDS = {
    "pH":              (0, 14),     # chemistry scale, not data-driven
    "canopy_pct":      (0, 100),
    "riparian_veg":    (0, 100),    # confirm this is a % scale in your data dictionary
    "pool_pct":        (0, 100),
    "pct_fast_water":  (0, 100),
    "pct_slow_water":  (0, 100),
    "pct_urban":       (0, 100),
    "pct_forest":      (0, 100),
    "pct_agriculture": (0, 100),
    "pct_wetland":     (0, 100),
    "pct_impervious":  (0, 100),
    "pct_shrub":       (0, 100),
    "pct_grassland":   (0, 100),
    "invert_pct_ept":       (0, 100),
    "invert_pct_dominant":  (0, 100),
    "pct_ffg_CF": (0, 100), "pct_ffg_SH": (0, 100), "pct_ffg_CG": (0, 100),
    "pct_ffg_SC": (0, 100), "pct_ffg_PR": (0, 100),
    "stream_order": (1, 12),   # Strahler stream order, always a positive small integer
}

# ── 2. Group sums: fractions of the same whole, real sites rarely exceed ~110% ──
GROUP_SUM_CHECKS = [
    {
        "name": "land use",
        "features": ["pct_urban", "pct_forest", "pct_agriculture",
                     "pct_wetland", "pct_shrub", "pct_grassland"],
        "ceiling": 110.0,
    },
    {
        "name": "streambed flow type",
        "features": ["pct_fast_water", "pct_slow_water"],
        "ceiling": 105.0,
        # NOTE: unconfirmed whether pool_pct belongs in this group too, check your
        # NRSA data dictionary before adding it, left out to avoid a false positive.
    },
    {
        "name": "invertebrate functional feeding groups",
        "features": ["pct_ffg_CF", "pct_ffg_SH", "pct_ffg_CG", "pct_ffg_SC", "pct_ffg_PR"],
        "ceiling": 100.0,   # these are a SUBSET of all FFG codes, so should never exceed 100
    },
]

# ── 3. Ordering/subset constraints: feature A can never exceed feature B ──
ORDERING_CHECKS = [
    {
        "smaller": "invert_ept_richness",
        "larger":  "invert_richness",
        "explanation": "EPT taxa (mayflies, stoneflies, caddisflies) are a SUBSET of all "
                        "invertebrate taxa, richness of the subset cannot exceed the total.",
    },
]


def check_all_plausibility(user_vals):
    """
    Runs all three check types against a dict of feature_name -> value.
    Returns a list of human-readable warning strings (empty list if all clear).
    Silently skips any feature not present in user_vals rather than assuming 0,
    since a blank/missing field isn't the same as a confirmed zero.
    """
    warnings_list = []

    # 1. Absolute bounds
    for feat, (lo, hi) in ABSOLUTE_BOUNDS.items():
        if feat in user_vals and user_vals[feat] is not None:
            val = user_vals[feat]
            if val < lo or val > hi:
                warnings_list.append(
                    f"{feat} = {val} is outside the physically valid range "
                    f"({lo} to {hi}). This isn't just unusual, it's impossible, "
                    f"double check this value."
                )

    # 2. Group sums
    for group in GROUP_SUM_CHECKS:
        present = {f: user_vals[f] for f in group["features"]
                   if f in user_vals and user_vals[f] is not None}
        if len(present) < 2:
            continue
        total = sum(present.values())
        if total > group["ceiling"]:
            breakdown = ", ".join(f"{f}={v:.0f}" for f, v in present.items())
            warnings_list.append(
                f"{group['name'].capitalize()} features sum to {total:.0f}% "
                f"({breakdown}), which exceeds what's physically plausible for a "
                f"single site (~{group['ceiling']:.0f}% ceiling). This input "
                f"combination was likely never seen in training."
            )

    # 3. Ordering/subset constraints
    for check in ORDERING_CHECKS:
        s, l = check["smaller"], check["larger"]
        if s in user_vals and l in user_vals and user_vals[s] is not None and user_vals[l] is not None:
            if user_vals[s] > user_vals[l]:
                warnings_list.append(
                    f"{s} ({user_vals[s]}) cannot be greater than {l} ({user_vals[l]}). "
                    f"{check['explanation']}"
                )

    return warnings_list


if __name__ == "__main__":
    # Quick self-test with the exact scenario that started this conversation
    test_input = {
        "pct_urban": 80, "pct_forest": 80,        # impossible land-use sum
        "pH": 7.2,                                  # fine
        "invert_ept_richness": 15, "invert_richness": 10,  # impossible: subset > total
        "pct_ffg_CF": 40, "pct_ffg_SH": 40, "pct_ffg_CG": 40,  # sums to 120%, impossible
    }
    for w in check_all_plausibility(test_input):
        print("-", w)
