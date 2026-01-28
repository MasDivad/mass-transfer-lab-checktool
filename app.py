import math
from io import BytesIO
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Mass Transfer Lab – Data Verification",
    layout="centered",
)

# =========================================================
# GLOBAL HEADER (as you requested)
# =========================================================
st.markdown(
    """
<div style="text-align:center; padding-top:10px; padding-bottom:10px;">
  <div style="font-size:20px; font-weight:700;">Department of Chemical Engineering</div>
  <div style="font-size:18px; font-weight:600;">SRM Institute of Science and Technology</div>
  <div style="font-size:18px; font-weight:700; margin-top:10px;">Mass Transfer Experiment Data Verification</div>
  <div style="font-size:16px; font-weight:600;">(2025 – 26 Even Semester)</div>
</div>
<hr/>
""",
    unsafe_allow_html=True
)

# =========================================================
# HELPERS
# =========================================================
def parse_number_list(raw: str, min_len: int = 1):
    """Parse comma/space/newline separated numbers; disallow negatives."""
    if raw is None:
        return []
    parts = []
    for token in raw.replace(",", " ").split():
        token = token.strip()
        if token:
            parts.append(token)
    vals = []
    for p in parts:
        try:
            v = float(p)
        except ValueError:
            raise ValueError(f"Could not parse '{p}' as a number.")
        if v < 0:
            raise ValueError("Negative values are not allowed.")
        vals.append(v)
    if len(vals) < min_len:
        raise ValueError(f"Need at least {min_len} values.")
    return vals


def df_to_excel_bytes(dfs: dict):
    """
    dfs: {sheet_name: dataframe}
    Returns BytesIO for download.
    """
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        for name, df in dfs.items():
            safe = str(name)[:31]
            df.to_excel(writer, sheet_name=safe, index=False)
    output.seek(0)
    return output


def fig_to_png_bytes(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=200)
    buf.seek(0)
    return buf


def constants_panel(exp_title: str, const_key: str, defaults: dict):
    """
    - Constants retained per experiment using session_state[const_key]
    - Locked view + optional Advanced edit expander
    """
    if const_key not in st.session_state:
        st.session_state[const_key] = defaults.copy()

    consts = st.session_state[const_key]

    st.subheader("Fixed Experimental Conditions Used")
    st.caption("Constants are retained per experiment. Inputs are cleared when you switch experiments.")

    # Display constants (locked view)
    df_const = pd.DataFrame(
        [{"Parameter": k, "Value": v} for k, v in consts.items()]
    )
    st.dataframe(df_const, use_container_width=True, hide_index=True)

    # Advanced edit
    with st.expander("Advanced: Edit constants (only if needed)"):
        st.warning("Edit only if your lab manual uses different fixed values. Otherwise, keep as it is.")
        edited = {}

        for k, v in consts.items():
            if isinstance(v, (int, float)):
                edited_val = st.number_input(
                    f"{k}",
                    value=float(v),
                    step=0.1,
                    format="%.6f",
                    key=f"{const_key}_{k}"
                )
                # preserve int if originally int
                if isinstance(v, int) and abs(edited_val - int(edited_val)) < 1e-12:
                    edited_val = int(edited_val)
                edited[k] = edited_val
            elif isinstance(v, list):
                txt = st.text_area(
                    f"{k} (edit as comma/space separated numbers)",
                    value=" ".join(map(str, v)),
                    key=f"{const_key}_{k}_list"
                )
                try:
                    edited[k] = parse_number_list(txt, min_len=1)
                except Exception:
                    st.error(f"Invalid list for {k}. Keeping old value.")
                    edited[k] = v
            else:
                edited[k] = st.text_input(
                    f"{k}",
                    value=str(v),
                    key=f"{const_key}_{k}_txt"
                )

        if st.button("Save constants", key=f"save_{const_key}"):
            st.session_state[const_key] = edited
            st.success("Constants updated for this experiment.")

    st.divider()
    return st.session_state[const_key]


def clear_inputs_for_experiment(exp_key: str):
    """
    Clears ONLY experimental input widgets for the selected experiment.
    Does NOT touch constants.
    """
    input_keys_map = {
        "5) Drying (Tray Dryer) — Variable readings": [
            "dry_m1", "dry_m2", "dry_L", "dry_B", "dry_m3_raw"
        ],
        # When we enable other experiments, we will add their input keys here.
    }

    for k in input_keys_map.get(exp_key, []):
        if k in st.session_state:
            del st.session_state[k]


# =========================================================
# EXPERIMENT 5: DRYING (FULLY IMPLEMENTED)
# =========================================================
def page_drying():
    st.title("Experiment 5: Drying Characteristics (Tray Dryer)")

    defaults = {
        "Sample": "Calcium carbonate (CaCO3)",
        "DEFAULT_TIME_STEP_min": 5,
    }
    consts = constants_panel("Drying", "const_drying", defaults)

    st.subheader("Experimental Data (Enter readings only)")
    col1, col2 = st.columns(2)
    with col1:
        m1 = st.number_input(
            "Empty mass of plate, m1 (g)",
            min_value=0.0, value=0.0, step=0.1, format="%.3f",
            key="dry_m1"
        )
        m2 = st.number_input(
            "Mass of plate + dry solid, m2 (g)",
            min_value=0.0, value=0.0, step=0.1, format="%.3f",
            key="dry_m2"
        )
    with col2:
        L = st.number_input(
            "Length of plate, L (m)",
            min_value=0.0, value=0.0, step=0.001, format="%.4f",
            key="dry_L"
        )
        B = st.number_input(
            "Breadth of plate, B (m)",
            min_value=0.0, value=0.0, step=0.001, format="%.4f",
            key="dry_B"
        )

    dt_min = int(consts.get("DEFAULT_TIME_STEP_min", 5))
    st.info(f"Time interval is set to **{dt_min} min** (change under Advanced constants if required).")

    m3_raw = st.text_area(
        "Enter `m3` readings (mass of plate + sample) in grams.\n"
        "Paste values separated by space/comma/new line (10–15 values OR 25 values, etc.).",
        height=120,
        key="dry_m3_raw"
    )

    if st.button("Run Calculation (Drying)"):
        # Validation
        if m2 <= m1:
            st.error("m2 must be greater than m1.")
            return
        if L <= 0 or B <= 0:
            st.error("Plate dimensions (L and B) must be > 0.")
            return

        try:
            m3_list = parse_number_list(m3_raw, min_len=2)
        except Exception as e:
            st.error(str(e))
            return

        # Computations
        n = len(m3_list)
        theta_min = np.array([i * dt_min for i in range(n)], dtype=float)
        theta_s = theta_min * 60.0

        Ss_g = m2 - m1
        Ss_kg = Ss_g / 1000.0
        A = L * B

        m3 = np.array(m3_list, dtype=float)
        m = m3 - m1
        X = (m - Ss_g) / Ss_g  # kg/kg

        dX = np.zeros_like(X)
        dtheta = np.zeros_like(theta_s)
        Nflux = np.full_like(X, np.nan, dtype=float)

        for i in range(1, n):
            dX[i] = X[i] - X[i - 1]
            dtheta[i] = theta_s[i] - theta_s[i - 1]
            if dtheta[i] > 0 and A > 0:
                Nflux[i] = -(Ss_kg / A) * (dX[i] / dtheta[i])

        # Tables (match lab record style)
        st.subheader("Observation Table (Table 1)")
        df_obs = pd.DataFrame({
            "S.No": np.arange(1, n + 1),
            "Time, θ (min)": theta_min.astype(int),
            "Time, θ (s)": theta_s.astype(int),
            "Mass of sample + plate, m3 (g)": np.round(m3, 3),
            "Sample mass, m = m3 − m1 (g)": np.round(m, 3),
            "Moisture content, X = (m − Ss)/Ss (kg/kg)": np.round(X, 6),
        })
        st.dataframe(df_obs, use_container_width=True, hide_index=True)

        st.subheader("Calculation Table (Table 2)")
        df_calc = pd.DataFrame({
            "S.No": np.arange(1, n + 1),
            "Time, θ (s)": theta_s.astype(int),
            "X (kg/kg)": np.round(X, 6),
            "dX (kg/kg)": np.round(dX, 6),
            "dθ (s)": dtheta.astype(int),
            "Drying flux, N = −(Ss/A)(dX/dθ) (kg/m²·s)": np.round(Nflux, 10),
        })
        st.dataframe(df_calc, use_container_width=True, hide_index=True)

        # Results
        Xi = float(X[0])
        X_star = float(np.mean(X[-min(3, n):]))
        total_time_min = float(theta_min[-1])

        st.subheader("Result")
        st.markdown(
            f"""
**Result (Lab record style):**  
Drying characteristics were determined for the given sample.  
Initial moisture content, **Xi = {Xi:.6f} kg/kg**.  
Equilibrium moisture content (approx.), **X\* = {X_star:.6f} kg/kg**.  
Total observation time = **{total_time_min:.0f} min**.
"""
        )

        # Graphs
        st.subheader("Graphs")

        # Graph 1: X vs time
        fig1, ax1 = plt.subplots()
        ax1.plot(theta_min / 60.0, X, marker="o")
        ax1.set_xlabel("Time, θ (h)")
        ax1.set_ylabel("Moisture content, X (kg moisture/kg dry solid)")
        ax1.set_title("Moisture Content (X) vs Time (θ)")
        ax1.grid(True)
        st.pyplot(fig1)

        # Graph 2: N vs X
        valid = ~np.isnan(Nflux)
        fig2, ax2 = plt.subplots()
        ax2.plot(X[valid], Nflux[valid], marker="o")
        ax2.set_xlabel("Moisture content, X (kg/kg)")
        ax2.set_ylabel("Drying flux, N (kg/m²·s)")
        ax2.set_title("Drying Flux (N) vs Moisture Content (X)")
        ax2.grid(True)
        st.pyplot(fig2)

        # Sample calculation (first interval)
        st.subheader("Sample Calculation (First interval)")
        if n >= 2:
            st.markdown(
                f"""
- **Ss = m2 − m1 = {m2:.3f} − {m1:.3f} = {Ss_g:.3f} g**
- **A = L × B = {L:.4f} × {B:.4f} = {A:.6f} m²**
- At θ = 0: m3 = {m3[0]:.3f} g → m = {m[0]:.3f} g → X0 = {X[0]:.6f}
- At θ = {dt_min} min: m3 = {m3[1]:.3f} g → m = {m[1]:.3f} g → X1 = {X[1]:.6f}
- dX = X1 − X0 = {dX[1]:.6f} ; dθ = {int(dtheta[1])} s  
- **N = −(Ss/A)(dX/dθ) = {Nflux[1]:.10f} kg/m²·s**
"""
            )

        # Downloads
        st.subheader("Download")
        excel_bytes = df_to_excel_bytes({
            "Fixed_Constants": pd.DataFrame([consts]),
            "Observation_Table": df_obs,
            "Calculation_Table": df_calc,
            "Result": pd.DataFrame([{
                "Xi (kg/kg)": Xi,
                "X* approx (kg/kg)": X_star,
                "Total time (min)": total_time_min,
                "Ss (g)": float(Ss_g),
                "A (m²)": float(A),
            }])
        })
        st.download_button(
            "Download Excel (Drying)",
            data=excel_bytes,
            file_name="Experiment5_Drying_Verification.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        st.download_button(
            "Download Graph 1 (PNG) — X vs Time",
            data=fig_to_png_bytes(fig1),
            file_name="Exp5_Drying_X_vs_Time.png",
            mime="image/png"
        )
        st.download_button(
            "Download Graph 2 (PNG) — N vs X",
            data=fig_to_png_bytes(fig2),
            file_name="Exp5_Drying_N_vs_X.png",
            mime="image/png"
        )


# =========================================================
# PLACEHOLDER PAGES (for now)
# =========================================================
def page_placeholder(exp_no: str, title: str):
    st.title(f"Experiment {exp_no}: {title}")
    st.info("This experiment page will be enabled next with the same lab-manual structure:\n"
            "Fixed constants → Observation table → Calculation table → Graphs → Result → Downloads.")
    st.write("✅ Drying is already enabled as the working template.")


# =========================================================
# EXPERIMENT MENU
# =========================================================
EXPERIMENTS = {
    "1) Simple Distillation (Acetone–Water) — Rayleigh": lambda: page_placeholder("1", "Simple Distillation (Acetone–Water) — Rayleigh"),
    "2) Multi-stage Leaching (Na2CO3) — % recovery vs stages": lambda: page_placeholder("2", "Multi-stage Leaching (Na2CO3)"),
    "3) Single-stage Leaching (Na2CO3) — % recovery vs solvent/feed": lambda: page_placeholder("3", "Single-stage Leaching (Na2CO3)"),
    "4) Adsorption Isotherm (Freundlich) — Acetic acid on carbon": lambda: page_placeholder("4", "Adsorption Isotherm (Freundlich)"),
    "5) Drying (Tray Dryer) — Variable readings": page_drying,
    "6) Steam Distillation (Turpentine–Water) — Vapour efficiency": lambda: page_placeholder("6", "Steam Distillation (Turpentine–Water)"),
}

EXPERIMENT_KEYS = list(EXPERIMENTS.keys())

# =========================================================
# NAVIGATION (constants retained per experiment, inputs cleared)
# =========================================================
if "exp_idx" not in st.session_state:
    st.session_state["exp_idx"] = 4  # default index -> Drying

# Sidebar selector
st.sidebar.title("Select Experiment")
selected = st.sidebar.selectbox("Experiment", EXPERIMENT_KEYS, index=st.session_state["exp_idx"])

# If changed by dropdown -> clear only that experiment inputs
prev_key = st.session_state.get("prev_exp_key", None)
if prev_key is None:
    st.session_state["prev_exp_key"] = selected
elif prev_key != selected:
    clear_inputs_for_experiment(selected)
    st.session_state["prev_exp_key"] = selected

# Sync exp_idx
st.session_state["exp_idx"] = EXPERIMENT_KEYS.index(selected)

# Navigation buttons
c1, c2, c3 = st.columns([1, 2, 1])
with c1:
    if st.button("⬅ Previous Experiment"):
        new_idx = max(0, st.session_state["exp_idx"] - 1)
        st.session_state["exp_idx"] = new_idx
        new_key = EXPERIMENT_KEYS[new_idx]
        clear_inputs_for_experiment(new_key)
        st.session_state["prev_exp_key"] = new_key
        st.rerun()

with c3:
    if st.button("Next Experiment ➡"):
        new_idx = min(len(EXPERIMENT_KEYS) - 1, st.session_state["exp_idx"] + 1)
        st.session_state["exp_idx"] = new_idx
        new_key = EXPERIMENT_KEYS[new_idx]
        clear_inputs_for_experiment(new_key)
        st.session_state["prev_exp_key"] = new_key
        st.rerun()

st.sidebar.caption("Mobile-friendly: inputs reset on experiment switch; constants retained per experiment.")

# Run selected page
EXPERIMENTS[EXPERIMENT_KEYS[st.session_state["exp_idx"]]]()
