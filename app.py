
import math
from io import BytesIO
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Mass Transfer Lab – Cross-check Tool", layout="centered")

# -----------------------------
# Helpers
# -----------------------------
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

def constants_panel(title: str, key: str, defaults: dict):
    """
    Locked by default. Optional 'Advanced: Edit constants' expander.
    """
    if key not in st.session_state:
        st.session_state[key] = defaults.copy()

    consts = st.session_state[key]

    st.subheader("Fixed Experimental Conditions Used")
    st.caption("These constants are used internally for calculations. By default they are locked for cross-checking.")
    st.json(consts)

    with st.expander("Advanced: Edit constants (only if needed)"):
        st.warning("Edit only if your lab manual uses different fixed values. Otherwise, keep locked.")
        edited = {}
        for k, v in consts.items():
            # Choose widget based on type
            if isinstance(v, (int, float)):
                edited[k] = st.number_input(k, value=float(v), step=0.1, format="%.6f")
                # keep int if original was int and user entered whole number
                if isinstance(v, int) and abs(edited[k] - int(edited[k])) < 1e-12:
                    edited[k] = int(edited[k])
            elif isinstance(v, list):
                # Allow editing lists as text
                txt = st.text_area(k + " (edit as comma/space separated numbers)", value=" ".join(map(str, v)))
                try:
                    edited[k] = parse_number_list(txt, min_len=1)
                except Exception:
                    st.error(f"Invalid list format for {k}. Keeping previous value.")
                    edited[k] = v
            else:
                edited[k] = st.text_input(k, value=str(v))

        if st.button("Save updated constants", key=f"save_{key}"):
            st.session_state[key] = edited
            st.success("Constants updated.")

    st.divider()
    return st.session_state[key]


# -----------------------------
# Experiment 5: Drying (Prototype + fully functional)
# -----------------------------
def page_drying():
    st.title("Drying Characteristics (Tray Dryer)")

    defaults = {
        "Sample": "Calcium carbonate (CaCO3)",
        "DEFAULT_TIME_STEP_min": 5,
    }
    consts = constants_panel("DRYING", "const_drying", defaults)

    st.subheader("Enter ONLY experimental readings")
    col1, col2 = st.columns(2)
    with col1:
        m1 = st.number_input("Empty mass of plate, m1 (g)", min_value=0.0, value=0.0, step=0.1, format="%.3f")
        m2 = st.number_input("Mass of plate + dry solid, m2 (g)", min_value=0.0, value=0.0, step=0.1, format="%.3f")
    with col2:
        L = st.number_input("Length of plate, L (m)", min_value=0.0, value=0.0, step=0.001, format="%.4f")
        B = st.number_input("Breadth of plate, B (m)", min_value=0.0, value=0.0, step=0.001, format="%.4f")

    dt_min = int(consts.get("DEFAULT_TIME_STEP_min", 5))
    st.info(f"Time step is set to **{dt_min} min** (changeable under Advanced constants).")

    m3_raw = st.text_area(
        "Mass readings: paste `m3` values (mass of plate + sample) in grams.\n"
        "Enter 10–15 values (or 25 etc.). Separate by space/comma/new line.",
        height=120
    )

    if st.button("Run Drying Calculations"):
        try:
            m3_list = parse_number_list(m3_raw, min_len=2)
        except Exception as e:
            st.error(str(e))
            return

        if m2 <= m1:
            st.error("m2 must be greater than m1.")
            return
        if L <= 0 or B <= 0:
            st.error("Plate dimensions must be > 0.")
            return

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
            dX[i] = X[i] - X[i-1]
            dtheta[i] = theta_s[i] - theta_s[i-1]
            if dtheta[i] > 0 and A > 0:
                Nflux[i] = -(Ss_kg / A) * (dX[i] / dtheta[i])

        df_obs = pd.DataFrame({
            "S.No": np.arange(1, n+1),
            "Time θ (min)": theta_min.astype(int),
            "Time θ (s)": theta_s.astype(int),
            "m3 (g) (plate + sample)": np.round(m3, 3),
            "m (g) = m3 - m1": np.round(m, 3),
            "X (kg/kg) = (m - Ss)/Ss": np.round(X, 6),
        })

        df_calc = pd.DataFrame({
            "S.No": np.arange(1, n+1),
            "Time θ (s)": theta_s.astype(int),
            "X (kg/kg)": np.round(X, 6),
            "dX (kg/kg)": np.round(dX, 6),
            "dθ (s)": dtheta.astype(int),
            "N (kg/m²·s)": np.round(Nflux, 10),
        })

        st.subheader("Observation Table")
        st.dataframe(df_obs, use_container_width=True)

        st.subheader("Calculation Table")
        st.dataframe(df_calc, use_container_width=True)

        Xi = float(X[0])
        X_star = float(np.mean(X[-min(3, n):]))
        st.subheader("Final Results")
        st.write({
            "Dry solid mass, Ss (g)": float(Ss_g),
            "Plate area, A (m²)": float(A),
            "Initial moisture content, Xi (kg/kg)": Xi,
            "Equilibrium moisture content, X* (kg/kg) [avg last points]": X_star,
            "Total observation time (min)": float(theta_min[-1]),
        })

        # Graph 1: X vs time
        fig1, ax1 = plt.subplots()
        ax1.plot(theta_min/60.0, X, marker="o")
        ax1.set_xlabel("Time, θ (h)")
        ax1.set_ylabel("Moisture content, X (kg moisture/kg dry solid)")
        ax1.set_title("Moisture Content (X) vs Time (θ)")
        ax1.grid(True)
        st.pyplot(fig1)

        # Graph 2: N vs X (valid points)
        valid = ~np.isnan(Nflux)
        fig2, ax2 = plt.subplots()
        ax2.plot(X[valid], Nflux[valid], marker="o")
        ax2.set_xlabel("Moisture content, X (kg/kg)")
        ax2.set_ylabel("Drying flux, N (kg/m²·s)")
        ax2.set_title("Drying Flux (N) vs Moisture Content (X)")
        ax2.grid(True)
        st.pyplot(fig2)

        # Downloads
        excel_bytes = df_to_excel_bytes({
            "Fixed_Constants": pd.DataFrame([consts]),
            "Observation_Table": df_obs,
            "Calculation_Table": df_calc,
        })
        st.download_button(
            "Download Excel (Drying)",
            data=excel_bytes,
            file_name="drying_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        st.download_button("Download Graph 1 (PNG)", data=fig_to_png_bytes(fig1), file_name="drying_X_vs_time.png", mime="image/png")
        st.download_button("Download Graph 2 (PNG)", data=fig_to_png_bytes(fig2), file_name="drying_N_vs_X.png", mime="image/png")

        # Sample calculation display (first interval)
        st.subheader("Sample Calculation (Trial 1 / First Interval)")
        if n >= 2:
            st.markdown(
                f"""
- **Ss** = m2 − m1 = {m2:.3f} − {m1:.3f} = **{Ss_g:.3f} g**
- **A** = L×B = {L:.4f}×{B:.4f} = **{A:.6f} m²**
- At θ=0: m3={m3[0]:.3f} g → m={m[0]:.3f} g → X0={X[0]:.6f}
- At θ={dt_min} min: m3={m3[1]:.3f} g → m={m[1]:.3f} g → X1={X[1]:.6f}
- dX = X1 − X0 = {dX[1]:.6f}; dθ = {int(dtheta[1])} s
- N = −(Ss/A)·(dX/dθ) = **{Nflux[1]:.10f} kg/m²·s**
                """
            )


# -----------------------------
# Menu skeleton for all 6 (Drying is fully done now)
# You can progressively plug-in the remaining experiments.
# For faculty cross-checking, Drying usually is the most error-prone: it’s already ready.
# -----------------------------
def page_placeholder(title: str):
    st.title(title)
    st.info("This module will be enabled next in the same interface (same style: constants + readings + tables + graphs + downloads).")
    st.write("If you want, tell me which experiment your faculty uses most frequently (1–4 or 6), and I will prioritize it next.")

EXPERIMENTS = {
    "1) Simple Distillation (Acetone–Water) — Rayleigh": lambda: page_placeholder("Simple Distillation (Acetone–Water) — Rayleigh"),
    "2) Multi-stage Leaching (Na2CO3) — % recovery vs stages": lambda: page_placeholder("Multi-stage Leaching (Na2CO3)"),
    "3) Single-stage Leaching (Na2CO3) — % recovery vs solvent/feed": lambda: page_placeholder("Single-stage Leaching (Na2CO3)"),
    "4) Adsorption Isotherm (Freundlich) — Acetic acid on carbon": lambda: page_placeholder("Adsorption Isotherm (Freundlich)"),
    "5) Drying (Tray Dryer) — Variable readings": page_drying,
    "6) Steam Distillation (Turpentine–Water) — Vapour efficiency": lambda: page_placeholder("Steam Distillation (Turpentine–Water)"),
}

st.sidebar.title("Select Experiment")
choice = st.sidebar.selectbox("Experiment", list(EXPERIMENTS.keys()), index=4)

st.sidebar.caption("Designed for WhatsApp link usage: mobile-friendly, constants locked by default.")
EXPERIMENTS[choice]()
