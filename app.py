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
# GLOBAL HEADER
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
    if raw is None:
        return []
    parts = raw.replace(",", " ").split()
    vals = []
    for p in parts:
        v = float(p)
        if v < 0:
            raise ValueError("Negative values not allowed")
        vals.append(v)
    if len(vals) < min_len:
        raise ValueError("Not enough values")
    return vals


def df_to_excel_bytes(dfs: dict):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        for name, df in dfs.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)
    output.seek(0)
    return output


def fig_to_png_bytes(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)
    return buf


def constants_panel(key, defaults):
    if key not in st.session_state:
        st.session_state[key] = defaults.copy()

    consts = st.session_state[key]

    st.subheader("Fixed Experimental Conditions Used")
    st.dataframe(
        pd.DataFrame({"Parameter": consts.keys(), "Value": consts.values()}),
        hide_index=True,
        use_container_width=True
    )

    with st.expander("Advanced: Edit constants (only if required)"):
        updated = {}
        for k, v in consts.items():
            updated[k] = st.number_input(k, value=float(v))
        if st.button("Save constants"):
            st.session_state[key] = updated
            st.success("Constants updated")

    st.divider()
    return st.session_state[key]


# =========================================================
# EXPERIMENT 5 – DRYING (WORKING)
# =========================================================
def experiment_drying():
    st.title("Experiment 5: Drying Characteristics (Tray Dryer)")

    consts = constants_panel(
        "const_drying",
        {"Time interval (min)": 5}
    )
    dt = int(consts["Time interval (min)"])

    m1 = st.number_input("Empty mass of plate, m1 (g)", key="m1")
    m2 = st.number_input("Mass of plate + dry solid, m2 (g)", key="m2")
    L = st.number_input("Length of plate, L (m)", key="L")
    B = st.number_input("Breadth of plate, B (m)", key="B")

    m3_raw = st.text_area(
        "Enter m3 readings (plate + sample mass, g)",
        key="m3"
    )

    if st.button("Run Drying Calculation"):
        m3 = parse_number_list(m3_raw, min_len=2)
        Ss = m2 - m1
        A = L * B

        theta = np.arange(len(m3)) * dt * 60
        m = np.array(m3) - m1
        X = (m - Ss) / Ss

        df = pd.DataFrame({
            "Time (min)": theta / 60,
            "Moisture content X (kg/kg)": X
        })

        st.subheader("Observation Table")
        st.dataframe(df, use_container_width=True)

        fig, ax = plt.subplots()
        ax.plot(theta / 3600, X, marker="o")
        ax.set_xlabel("Time (h)")
        ax.set_ylabel("Moisture content X")
        ax.grid(True)
        st.pyplot(fig)

        st.subheader("Result")
        st.write(f"Initial moisture content, Xi = {X[0]:.4f}")

        st.download_button(
            "Download Excel",
            data=df_to_excel_bytes({"Drying": df}),
            file_name="Drying.xlsx"
        )


# =========================================================
# PLACEHOLDERS
# =========================================================
def placeholder(title):
    st.title(title)
    st.info("This experiment will be enabled next.")


# =========================================================
# EXPERIMENT REGISTRY
# =========================================================
EXPERIMENTS = {
    "1. Simple Distillation": lambda: placeholder("Simple Distillation"),
    "2. Multi-stage Leaching": lambda: placeholder("Multi-stage Leaching"),
    "3. Single-stage Leaching": lambda: placeholder("Single-stage Leaching"),
    "4. Adsorption Isotherm": lambda: placeholder("Adsorption Isotherm"),
    "5. Drying": experiment_drying,
    "6. Steam Distillation": lambda: placeholder("Steam Distillation"),
}

# =========================================================
# MAIN PAGE EXPERIMENT SELECTION (MOBILE SAFE)
# =========================================================
st.subheader("Select Experiment")

exp_list = list(EXPERIMENTS.keys())

if "exp_idx" not in st.session_state:
    st.session_state["exp_idx"] = 4

selected = st.selectbox(
    "Experiment",
    exp_list,
    index=st.session_state["exp_idx"]
)

st.session_state["exp_idx"] = exp_list.index(selected)

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("⬅ Previous"):
        st.session_state["exp_idx"] = max(0, st.session_state["exp_idx"] - 1)
        st.experimental_rerun()
with col3:
    if st.button("Next ➡"):
        st.session_state["exp_idx"] = min(len(exp_list) - 1, st.session_state["exp_idx"] + 1)
        st.experimental_rerun()

EXPERIMENTS[exp_list[st.session_state["exp_idx"]]]()
