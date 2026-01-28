import math
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(page_title="Mass Transfer Lab – Data Verification", layout="centered")

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
    unsafe_allow_html=True,
)

# =========================================================
# UTILITIES
# =========================================================
def parse_number_list(raw: str, min_len: int = 1):
    if raw is None:
        return []
    parts = [p for p in raw.replace(",", " ").split() if p.strip()]
    vals = []
    for p in parts:
        try:
            v = float(p)
        except ValueError:
            raise ValueError(f"Invalid number: {p}")
        if v < 0:
            raise ValueError("Negative values not allowed.")
        vals.append(v)
    if len(vals) < min_len:
        raise ValueError(f"Enter at least {min_len} values.")
    return vals


def df_to_excel_bytes(dfs: dict):
    out = BytesIO()
    with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
        for name, df in dfs.items():
            df.to_excel(writer, sheet_name=str(name)[:31], index=False)
    out.seek(0)
    return out


def fig_to_png_bytes(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)
    return buf


def constants_panel(const_key: str, defaults: dict):
    """
    Constants retained per experiment (session_state[const_key]).
    Editable in Advanced mode.
    """
    if const_key not in st.session_state:
        st.session_state[const_key] = defaults.copy()

    consts = st.session_state[const_key]

    st.subheader("Fixed Experimental Conditions Used")
    dfc = pd.DataFrame({"Parameter": list(consts.keys()), "Value": list(consts.values())})
    st.dataframe(dfc, use_container_width=True, hide_index=True)

    with st.expander("Advanced: Edit constants (only if required)"):
        updated = {}
        for k, v in consts.items():
            if isinstance(v, (int, float)):
                updated[k] = st.number_input(k, value=float(v), step=0.1, format="%.6f", key=f"{const_key}_{k}")
            else:
                updated[k] = st.text_input(k, value=str(v), key=f"{const_key}_{k}_txt")

        if st.button("Save constants", key=f"save_{const_key}"):
            # preserve int when possible
            cleaned = {}
            for k, v in updated.items():
                if isinstance(consts.get(k), int):
                    try:
                        iv = int(round(float(v)))
                        cleaned[k] = iv
                    except:
                        cleaned[k] = updated[k]
                else:
                    cleaned[k] = updated[k]
            st.session_state[const_key] = cleaned
            st.success("Constants updated for this experiment.")

    st.divider()
    return st.session_state[const_key]


def clear_inputs_for_experiment(exp_key: str):
    """
    Clears ONLY experimental inputs (not constants).
    """
    input_keys = {
        "1. Simple Distillation (Acetone–Water) — Rayleigh": [
            "sd_V",
            "sd_a_F", "sd_b_F",
            "sd_a_D", "sd_b_D",
            "sd_a_W", "sd_b_W",
        ],
        "2. Multi-stage Leaching (Na2CO3)": ["ms_vhcl_raw"],
        "3. Single-stage Leaching (Na2CO3)": ["ss_vhcl_raw"],
        "4. Adsorption Isotherm (Freundlich) — Acetic acid/Carbon": ["ads_vnaoh_raw"],
        "5. Drying (Tray Dryer)": ["dry_m1","dry_m2","dry_L","dry_B","dry_m3_raw"],
        "6. Steam Distillation (Turpentine–Water) — Vapour efficiency": ["steam_trials_raw"],
    }
    for k in input_keys.get(exp_key, []):
        if k in st.session_state:
            del st.session_state[k]

# =========================================================
# EXP 1: SIMPLE DISTILLATION — RAYLEIGH
# =========================================================
def exp_simple_distillation():
    st.title("Experiment 1: Simple Distillation (Acetone–Water) — Rayleigh’s Equation")

    # fixed constants / tables (as per your provided code)
    defaults = {
        "MW_Acetone (g/mol)": 58.0,
        "MW_Water (g/mol)": 18.0,
    }
    consts = constants_panel("const_sd", defaults)

    MW_ACETONE = float(consts["MW_Acetone (g/mol)"])
    MW_WATER = float(consts["MW_Water (g/mol)"])

    # Density–composition table
    phiA_table = np.array([1.0, 0.8, 0.6, 0.4, 0.2, 0.0])
    rho_table  = np.array([0.791, 0.8328, 0.8746, 0.9164, 0.9582, 1.0000])
    idx = np.argsort(rho_table)
    rho_sorted = rho_table[idx]
    phi_sorted = phiA_table[idx]

    # VLE data
    x_eq = np.array([0.00,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,
                     0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00])
    y_eq = np.array([0.0000,0.6381,0.7301,0.7716,0.7916,0.8034,0.8124,0.8201,
                     0.8269,0.8376,0.8387,0.8455,0.8532,0.8615,0.8712,0.8817,
                     0.8950,0.9118,0.9335,0.9627,1.0000])

    st.subheader("Experimental Data (Enter readings only)")
    V = st.number_input("Volume of specific gravity bottle, V (cc)", min_value=0.0, value=0.0, step=0.1, key="sd_V")

    st.markdown("Enter **a** (empty bottle mass) and **b** (bottle + sample mass) for:")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Feed (F)**")
        aF = st.number_input("a_F (g)", min_value=0.0, value=0.0, step=0.1, key="sd_a_F")
        bF = st.number_input("b_F (g)", min_value=0.0, value=0.0, step=0.1, key="sd_b_F")
    with c2:
        st.markdown("**Distillate (D)**")
        aD = st.number_input("a_D (g)", min_value=0.0, value=0.0, step=0.1, key="sd_a_D")
        bD = st.number_input("b_D (g)", min_value=0.0, value=0.0, step=0.1, key="sd_b_D")
    with c3:
        st.markdown("**Residue (W)**")
        aW = st.number_input("a_W (g)", min_value=0.0, value=0.0, step=0.1, key="sd_a_W")
        bW = st.number_input("b_W (g)", min_value=0.0, value=0.0, step=0.1, key="sd_b_W")

    if st.button("Run Calculation (Simple Distillation)"):
        if V <= 0:
            st.error("Bottle volume V must be > 0.")
            return

        data = [("Feed (F)", aF, bF), ("Distillate (D)", aD, bD), ("Residue (W)", aW, bW)]
        rows = []

        def phi_from_density(rho):
            return float(np.interp(rho, rho_sorted, phi_sorted))

        for name, a, b in data:
            m = b - a
            if m <= 0:
                st.error(f"Mass (b-a) must be positive for {name}.")
                return
            rho = m / V
            phiA = phi_from_density(rho)
            phiW = 1 - phiA

            nA = (V * phiA * rho) / MW_ACETONE
            nW = (V * phiW * rho) / MW_WATER
            nt = nA + nW
            xA = nA / nt

            rows.append([name, a, b, m, rho, phiA, nA, nW, nt, xA])

        df = pd.DataFrame(rows, columns=[
            "Mixture","a (g)","b (g)","m (g)","ρ (g/cc)","φA",
            "nA (mol)","nW (mol)","Total moles","x"
        ])

        F = df.loc[0, "Total moles"]
        W = df.loc[2, "Total moles"]
        xF = df.loc[0, "x"]
        xWv = df.loc[2, "x"]

        ln_FW = float(np.log(F / W))

        # Rayleigh integral
        x_int = np.linspace(xWv, xF, 400)
        y_int = np.interp(x_int, x_eq, y_eq)
        integrand = 1.0 / (y_int - x_int)
        I = float(np.trapz(integrand, x_int))

        err = abs((ln_FW - I) / ln_FW) * 100 if ln_FW != 0 else float("nan")

        st.subheader("Observation & Calculation Table")
        st.dataframe(df.round(6), use_container_width=True, hide_index=True)

        st.subheader("Final Results")
        res = pd.DataFrame([{
            "F (total moles)": F,
            "W (total moles)": W,
            "xF": xF,
            "xW": xWv,
            "ln(F/W)": ln_FW,
            "Rayleigh Integral I": I,
            "% Error": err
        }])
        st.dataframe(res.round(6), use_container_width=True, hide_index=True)

        st.subheader("Graphs")
        fig1, ax1 = plt.subplots()
        ax1.plot(x_eq, y_eq, marker="o")
        ax1.set_xlabel("x (liquid mole fraction acetone)")
        ax1.set_ylabel("y* (vapor mole fraction acetone)")
        ax1.set_title("VLE: y* vs x (Acetone–Water)")
        ax1.grid(True)
        st.pyplot(fig1)

        st.subheader("Result (Lab record style)")
        st.markdown(
            f"**Result:** Rayleigh’s equation was verified for acetone–water system.\n\n"
            f"ln(F/W) = **{ln_FW:.6f}**,  Integral(I) = **{I:.6f}**,  % error = **{err:.2f}%**."
        )

        st.subheader("Download")
        excel = df_to_excel_bytes({
            "Fixed_Constants": pd.DataFrame([consts]),
            "Table": df,
            "Final_Results": res
        })
        st.download_button("Download Excel (Exp1)", data=excel,
                           file_name="Exp1_Simple_Distillation.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        st.download_button("Download VLE Graph (PNG)", data=fig_to_png_bytes(fig1),
                           file_name="Exp1_VLE_y_vs_x.png", mime="image/png")


# =========================================================
# EXP 2: MULTI-STAGE LEACHING
# =========================================================
def exp_multistage_leaching():
    st.title("Experiment 2: Multi-stage Leaching (Na₂CO₃)")

    defaults = {
        "N_HCl (N)": 0.1,
        "V_titrated (mL)": 10.0,
        "V_dilution_taken (mL)": 10.0,
        "V_madeup (mL)": 100.0,
        "Total solvent volume used, V_solvent_total (mL)": 200.0,
        "Eq. wt of Na2CO3": 53.0,
        "Mass of Na2CO3 taken in feed (g)": 5.0,
        "Stages count": 5,
        # solvent distribution shown in table (manual)
        "Solvent per stage (mL) for stages 1..5": "200, 100, 67, 50, 40"
    }
    consts = constants_panel("const_ms", defaults)

    N_HCL = float(consts["N_HCl (N)"])
    V_TITRATED = float(consts["V_titrated (mL)"])
    V_DIL_TAKEN = float(consts["V_dilution_taken (mL)"])
    V_MADEUP = float(consts["V_madeup (mL)"])
    V_SOLVENT_TOTAL = float(consts["Total solvent volume used, V_solvent_total (mL)"])
    EQWT = float(consts["Eq. wt of Na2CO3"])
    M_TAKEN = float(consts["Mass of Na2CO3 taken in feed (g)"])

    try:
        solvent_stage = parse_number_list(str(consts["Solvent per stage (mL) for stages 1..5"]), min_len=5)[:5]
    except:
        solvent_stage = [200.0, 100.0, 67.0, 50.0, 40.0]

    st.subheader("Experimental Data (Enter readings only)")
    st.caption("Enter ONLY HCl volumes used for Beakers 1–5 (one value each).")
    vhcl_raw = st.text_area("Volumes of 0.1 N HCl used (mL) for Beakers 1–5 (space/comma separated)",
                            key="ms_vhcl_raw", height=90)

    if st.button("Run Calculation (Multi-stage Leaching)"):
        try:
            v_hcl = parse_number_list(vhcl_raw, min_len=5)[:5]
        except Exception as e:
            st.error(str(e))
            return

        stages = [1,2,3,4,5]

        Ndilute = [(vh * N_HCL) / V_TITRATED for vh in v_hcl]
        Noriginal = [nd * (V_MADEUP / V_DIL_TAKEN) for nd in Ndilute]
        m_rec = [(no * EQWT * V_SOLVENT_TOTAL) / 1000.0 for no in Noriginal]
        rec_pct = [(m / M_TAKEN) * 100.0 for m in m_rec]

        table1 = pd.DataFrame({
            "Beaker No.": [1,2,3,4,5],
            "No. of stages": stages,
            "Solvent per stage (mL)": solvent_stage,
            "Volume of 0.1 N HCl used (mL)": v_hcl
        })

        table2 = pd.DataFrame({
            "Beaker No.": [1,2,3,4,5],
            "No. of stages": stages,
            "Ndilute (N)": Ndilute,
            "Noriginal (N)": Noriginal,
            "Mass Na2CO3 recovered, m (g)": m_rec,
            "Recovery of Na2CO3 (%)": rec_pct
        })

        st.subheader("Observation Table (Table 1)")
        st.dataframe(table1.round(4), use_container_width=True, hide_index=True)

        st.subheader("Calculation Table (Table 2)")
        st.dataframe(table2.round(6), use_container_width=True, hide_index=True)

        st.subheader("Graph")
        fig, ax = plt.subplots()
        ax.plot(stages, rec_pct, marker="o")
        ax.set_xlabel("Number of stages")
        ax.set_ylabel("Percentage Recovery of Na2CO3 (%)")
        ax.set_title("% Recovery vs Number of Stages")
        ax.grid(True)
        st.pyplot(fig)

        best_stage = stages[int(np.argmax(rec_pct))]
        st.subheader("Result (Lab record style)")
        st.markdown(
            f"**Result:** Percentage recovery of Na₂CO₃ was determined for multi-stage leaching and % recovery "
            f"vs number of stages was plotted. Maximum recovery was obtained at **{best_stage} stage(s)**: "
            f"**{max(rec_pct):.2f}%**."
        )

        st.subheader("Download")
        excel = df_to_excel_bytes({
            "Fixed_Constants": pd.DataFrame([consts]),
            "Observation_Table": table1,
            "Calculation_Table": table2,
            "Result": pd.DataFrame({"Stages": stages, "%Recovery": rec_pct})
        })
        st.download_button("Download Excel (Exp2)", data=excel,
                           file_name="Exp2_MultiStage_Leaching.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        st.download_button("Download Graph (PNG)", data=fig_to_png_bytes(fig),
                           file_name="Exp2_Recovery_vs_Stages.png", mime="image/png")


# =========================================================
# EXP 3: SINGLE-STAGE LEACHING
# =========================================================
def exp_singlestage_leaching():
    st.title("Experiment 3: Single-stage Leaching (Na₂CO₃)")

    defaults = {
        "N_HCl (N)": 0.1,
        "Eq. wt of Na2CO3": 53.0,
        "V_titration_sample (mL)": 10.0,
        "V_dilution_taken (mL)": 10.0,
        "V_madeup (mL)": 100.0,
        "Na2CO3 taken (g)": 5.0,
        "Solvent volumes used for extraction (mL) [5 values]": "50, 100, 150, 200, 250",
        "Solvent:Feed ratio labels [5 values]": "0.5:1, 1:1, 1.5:1, 2:1, 2.5:1"
    }
    consts = constants_panel("const_ss", defaults)

    N_HCL = float(consts["N_HCl (N)"])
    EQWT = float(consts["Eq. wt of Na2CO3"])
    V_SAMPLE = float(consts["V_titration_sample (mL)"])
    V_DIL = float(consts["V_dilution_taken (mL)"])
    V_MADEUP = float(consts["V_madeup (mL)"])
    M_TAKEN = float(consts["Na2CO3 taken (g)"])

    try:
        solvent_vols = parse_number_list(str(consts["Solvent volumes used for extraction (mL) [5 values]"]), min_len=5)[:5]
    except:
        solvent_vols = [50,100,150,200,250]

    ratio_labels = [s.strip() for s in str(consts["Solvent:Feed ratio labels [5 values]"]).split(",")]
    if len(ratio_labels) < 5:
        ratio_labels = ["0.5:1","1:1","1.5:1","2:1","2.5:1"]
    ratio_labels = ratio_labels[:5]

    st.subheader("Experimental Data (Enter readings only)")
    st.caption("Enter ONLY HCl volumes used for the 5 beakers (in the same order as ratios).")
    vhcl_raw = st.text_area("Volumes of HCl used (mL) for 5 beakers (space/comma separated)",
                            key="ss_vhcl_raw", height=90)

    if st.button("Run Calculation (Single-stage Leaching)"):
        try:
            v_hcl = parse_number_list(vhcl_raw, min_len=5)[:5]
        except Exception as e:
            st.error(str(e))
            return

        N_dilute = [(vh * N_HCL) / V_SAMPLE for vh in v_hcl]
        dilution_factor = V_MADEUP / V_DIL
        N_original = [nd * dilution_factor for nd in N_dilute]
        m_recovered = [(no * EQWT * Vsol) / 1000.0 for no, Vsol in zip(N_original, solvent_vols)]
        pct = [(m / M_TAKEN) * 100.0 for m in m_recovered]

        df = pd.DataFrame({
            "Beaker No.": [1,2,3,4,5],
            "Solvent used for extraction (mL)": solvent_vols,
            "Solvent:Feed ratio": ratio_labels,
            "Volume of HCl used (mL)": v_hcl,
            "N_dilute (N)": N_dilute,
            "N_original (N)": N_original,
            "Na2CO3 recovered, m (g)": m_recovered,
            "Recovery of Na2CO3 (%)": pct
        })

        st.subheader("Observation & Calculation Table")
        st.dataframe(df.round(6), use_container_width=True, hide_index=True)

        st.subheader("Graph")
        fig, ax = plt.subplots()
        ax.plot(ratio_labels, pct, marker="o")
        ax.set_xlabel("Solvent to Feed Ratio")
        ax.set_ylabel("Percentage Recovery of Na2CO3 (%)")
        ax.set_title("% Recovery vs Solvent:Feed Ratio")
        ax.grid(True)
        st.pyplot(fig)

        max_i = int(np.argmax(pct))
        st.subheader("Result (Lab record style)")
        st.markdown(
            f"**Result:** As solvent-to-feed ratio increases, percentage recovery increases. "
            f"Maximum recovery observed at ratio **{ratio_labels[max_i]}**: **{pct[max_i]:.2f}%**."
        )

        st.subheader("Download")
        excel = df_to_excel_bytes({
            "Fixed_Constants": pd.DataFrame([consts]),
            "Table": df,
            "Result": pd.DataFrame({"Ratio": ratio_labels, "%Recovery": pct})
        })
        st.download_button("Download Excel (Exp3)", data=excel,
                           file_name="Exp3_SingleStage_Leaching.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        st.download_button("Download Graph (PNG)", data=fig_to_png_bytes(fig),
                           file_name="Exp3_Recovery_vs_Ratio.png", mime="image/png")


# =========================================================
# EXP 4: ADSORPTION ISOTHERM — FREUNDLICH
# =========================================================
def exp_adsorption():
    st.title("Experiment 4: Adsorption Isotherm (Freundlich) — Acetic Acid on Activated Carbon")

    defaults = {
        "N_NaOH (N)": 0.1,
        "Eq. wt of acetic acid": 60.0,
        "Filtrate used for titration (mL)": 10.0,
        "Solution volume used for adsorption, V (mL)": 200.0,
        "Mass of adsorbent, m (g)": 5.0,
        "Stock mass (g)": 25.0,
        "Stock volume (mL)": 250.0,
        "Pipetted volumes from stock (mL) [5]": "10, 20, 30, 40, 50",
        "Final dilution volume (mL)": 250.0
    }
    consts = constants_panel("const_ads", defaults)

    N_NAOH = float(consts["N_NaOH (N)"])
    EQWT = float(consts["Eq. wt of acetic acid"])
    V_FIL = float(consts["Filtrate used for titration (mL)"])
    V_ADS = float(consts["Solution volume used for adsorption, V (mL)"])
    M_ADS = float(consts["Mass of adsorbent, m (g)"])

    STOCK_MASS = float(consts["Stock mass (g)"])
    STOCK_VOL = float(consts["Stock volume (mL)"])
    STOCK_CONC = STOCK_MASS / STOCK_VOL  # g/mL

    try:
        pip = parse_number_list(str(consts["Pipetted volumes from stock (mL) [5]"]), min_len=5)[:5]
    except:
        pip = [10,20,30,40,50]
    FINAL_VOL = float(consts["Final dilution volume (mL)"])

    C0_LIST = [(v * STOCK_CONC) / FINAL_VOL for v in pip]  # g/mL

    st.subheader("Experimental Data (Enter readings only)")
    st.caption("Enter ONLY NaOH volumes used for each bottle (5 values).")
    vnaoh_raw = st.text_area("NaOH used (mL) for bottles 1–5 (space/comma separated)",
                             key="ads_vnaoh_raw", height=90)

    if st.button("Run Calculation (Adsorption)"):
        try:
            Vnaoh_used = parse_number_list(vnaoh_raw, min_len=5)[:5]
        except Exception as e:
            st.error(str(e))
            return

        def Ce_g_per_mL(vnaoh):
            return (vnaoh * N_NAOH * EQWT) / (V_FIL * 1000.0)

        calc_rows = []
        for i in range(5):
            C0 = C0_LIST[i]
            vna = Vnaoh_used[i]
            Ce = Ce_g_per_mL(vna)
            dC = C0 - Ce
            x = dC * V_ADS
            xm = x / M_ADS
            calc_rows.append({
                "Bottle": i+1,
                "C0 (g/mL)": C0,
                "NaOH used (mL)": vna,
                "Ce (g/mL)": Ce,
                "C0 - Ce (g/mL)": dC,
                "x = (C0-Ce)*V (g)": x,
                "x/m (g/g)": xm
            })

        calc_df = pd.DataFrame(calc_rows)

        log_df = pd.DataFrame({
            "Bottle": calc_df["Bottle"],
            "Ce (g/mL)": calc_df["Ce (g/mL)"],
            "log Ce": calc_df["Ce (g/mL)"].apply(lambda z: math.log10(z) if z > 0 else np.nan),
            "x/m (g/g)": calc_df["x/m (g/g)"],
            "log(x/m)": calc_df["x/m (g/g)"].apply(lambda z: math.log10(z) if z > 0 else np.nan),
        })

        st.subheader("Observation Table")
        obs = pd.DataFrame({
            "Bottle": [1,2,3,4,5],
            "Filtrate taken (mL)": [V_FIL]*5,
            "NaOH used (mL)": Vnaoh_used
        })
        st.dataframe(obs, use_container_width=True, hide_index=True)

        st.subheader("Calculation Table")
        st.dataframe(calc_df.round(6), use_container_width=True, hide_index=True)

        st.subheader("Final Results Table (Freundlich plot data)")
        st.dataframe(log_df.round(6), use_container_width=True, hide_index=True)

        # fit line log(x/m) = log k + (1/n) log Ce
        xy = log_df.dropna()
        if len(xy) >= 2:
            xs = xy["log Ce"].to_numpy()
            ys = xy["log(x/m)"].to_numpy()
            slope, intercept = np.polyfit(xs, ys, 1)  # slope = 1/n
            yhat = intercept + slope*xs
            ss_res = np.sum((ys - yhat)**2)
            ss_tot = np.sum((ys - np.mean(ys))**2)
            r2 = 1 - ss_res/ss_tot if ss_tot != 0 else np.nan
            one_over_n = slope
            n_val = 1/one_over_n if one_over_n != 0 else np.inf
            k_val = 10**intercept
        else:
            slope = intercept = r2 = one_over_n = n_val = k_val = np.nan

        st.subheader("Graphs")

        fig1, ax1 = plt.subplots()
        ax1.plot(calc_df["Ce (g/mL)"], calc_df["x/m (g/g)"], marker="o")
        ax1.set_xlabel("Ce (g/mL)")
        ax1.set_ylabel("x/m (g/g)")
        ax1.set_title("Adsorption Isotherm: x/m vs Ce")
        ax1.grid(True)
        st.pyplot(fig1)

        fig2, ax2 = plt.subplots()
        ax2.scatter(log_df["log Ce"], log_df["log(x/m)"])
        if not (np.isnan(slope) or np.isnan(intercept)):
            xline = np.array([np.nanmin(log_df["log Ce"])-0.05, np.nanmax(log_df["log Ce"])+0.05])
            yline = intercept + slope*xline
            ax2.plot(xline, yline)
        ax2.set_xlabel("log Ce")
        ax2.set_ylabel("log (x/m)")
        ax2.set_title("Freundlich Plot: log(x/m) vs log Ce")
        ax2.grid(True)
        st.pyplot(fig2)

        st.subheader("Result (Lab record style)")
        st.markdown(
            f"**Result:** Freundlich isotherm constants were obtained from log(x/m) vs log(Ce).\n\n"
            f"Slope = 1/n = **{one_over_n:.4f}**, Intercept = log k = **{intercept:.4f}**\n\n"
            f"Therefore, **k = {k_val:.4f}**, **n = {n_val:.4f}** (R² = **{r2:.4f}**)."
        )

        st.subheader("Download")
        excel = df_to_excel_bytes({
            "Fixed_Constants": pd.DataFrame([consts]),
            "Observation": obs,
            "Calculation": calc_df,
            "Freundlich_Table": log_df,
            "Fit": pd.DataFrame([{"slope(1/n)": one_over_n, "intercept(logk)": intercept, "k": k_val, "n": n_val, "R2": r2}])
        })
        st.download_button("Download Excel (Exp4)", data=excel,
                           file_name="Exp4_Adsorption_Freundlich.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        st.download_button("Download Graph 1 (PNG)", data=fig_to_png_bytes(fig1),
                           file_name="Exp4_xm_vs_Ce.png", mime="image/png")
        st.download_button("Download Graph 2 (PNG)", data=fig_to_png_bytes(fig2),
                           file_name="Exp4_Freundlich_plot.png", mime="image/png")


# =========================================================
# EXP 5: DRYING (variable readings allowed)
# =========================================================
def exp_drying():
    st.title("Experiment 5: Drying Characteristics (Tray Dryer)")

    defaults = {
        "Time interval (min)": 5
    }
    consts = constants_panel("const_dry", defaults)
    dt_min = int(float(consts["Time interval (min)"]))

    st.subheader("Experimental Data (Enter readings only)")
    c1, c2 = st.columns(2)
    with c1:
        m1 = st.number_input("Empty mass of plate, m1 (g)", min_value=0.0, value=0.0, step=0.1, key="dry_m1")
        m2 = st.number_input("Mass of plate + dry solid, m2 (g)", min_value=0.0, value=0.0, step=0.1, key="dry_m2")
    with c2:
        L = st.number_input("Length of plate, L (m)", min_value=0.0, value=0.0, step=0.001, format="%.4f", key="dry_L")
        B = st.number_input("Breadth of plate, B (m)", min_value=0.0, value=0.0, step=0.001, format="%.4f", key="dry_B")

    m3_raw = st.text_area(
        "Enter m3 readings (mass of plate + sample, g). Paste 10–15 values OR 25 values etc.\n"
        "Separate by space / comma / new line.",
        height=120,
        key="dry_m3_raw"
    )

    if st.button("Run Calculation (Drying)"):
        if m2 <= m1:
            st.error("m2 must be greater than m1.")
            return
        if L <= 0 or B <= 0:
            st.error("L and B must be > 0.")
            return
        try:
            m3_list = parse_number_list(m3_raw, min_len=2)
        except Exception as e:
            st.error(str(e))
            return

        n = len(m3_list)
        theta_min = np.array([i * dt_min for i in range(n)], dtype=float)
        theta_s = theta_min * 60.0

        Ss_g = m2 - m1
        Ss_kg = Ss_g / 1000.0
        A = L * B

        m3 = np.array(m3_list, dtype=float)
        m = m3 - m1
        X = (m - Ss_g) / Ss_g

        dX = np.zeros_like(X)
        dtheta = np.zeros_like(theta_s)
        Nflux = np.full_like(X, np.nan, dtype=float)

        for i in range(1, n):
            dX[i] = X[i] - X[i - 1]
            dtheta[i] = theta_s[i] - theta_s[i - 1]
            if dtheta[i] > 0:
                Nflux[i] = -(Ss_kg / A) * (dX[i] / dtheta[i])

        st.subheader("Observation Table")
        df_obs = pd.DataFrame({
            "S.No": np.arange(1, n+1),
            "Time, θ (min)": theta_min.astype(int),
            "m3 (g)": np.round(m3, 3),
            "m = m3 − m1 (g)": np.round(m, 3),
            "X = (m − Ss)/Ss (kg/kg)": np.round(X, 6)
        })
        st.dataframe(df_obs, use_container_width=True, hide_index=True)

        st.subheader("Calculation Table")
        df_calc = pd.DataFrame({
            "S.No": np.arange(1, n+1),
            "θ (s)": theta_s.astype(int),
            "X (kg/kg)": np.round(X, 6),
            "dX": np.round(dX, 6),
            "dθ (s)": dtheta.astype(int),
            "N = −(Ss/A)(dX/dθ) (kg/m²·s)": np.round(Nflux, 10)
        })
        st.dataframe(df_calc, use_container_width=True, hide_index=True)

        st.subheader("Graphs")
        fig1, ax1 = plt.subplots()
        ax1.plot(theta_min/60.0, X, marker="o")
        ax1.set_xlabel("Time, θ (h)")
        ax1.set_ylabel("Moisture content, X (kg/kg)")
        ax1.set_title("X vs Time")
        ax1.grid(True)
        st.pyplot(fig1)

        valid = ~np.isnan(Nflux)
        fig2, ax2 = plt.subplots()
        ax2.plot(X[valid], Nflux[valid], marker="o")
        ax2.set_xlabel("X (kg/kg)")
        ax2.set_ylabel("Drying flux, N (kg/m²·s)")
        ax2.set_title("N vs X")
        ax2.grid(True)
        st.pyplot(fig2)

        Xi = float(X[0])
        Xstar = float(np.mean(X[-min(3, n):]))
        st.subheader("Result (Lab record style)")
        st.markdown(
            f"**Result:** Drying characteristics were determined and graphs were plotted. "
            f"Initial moisture content **Xi = {Xi:.6f} kg/kg**. "
            f"Equilibrium moisture content (approx.) **X* = {Xstar:.6f} kg/kg**."
        )

        st.subheader("Download")
        excel = df_to_excel_bytes({
            "Fixed_Constants": pd.DataFrame([consts]),
            "Observation": df_obs,
            "Calculation": df_calc,
            "Result": pd.DataFrame([{"Xi": Xi, "X* approx": Xstar, "Ss(g)": Ss_g, "Area(m2)": A, "dt(min)": dt_min}])
        })
        st.download_button("Download Excel (Exp5)", data=excel,
                           file_name="Exp5_Drying.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        st.download_button("Download Graph 1 (PNG)", data=fig_to_png_bytes(fig1),
                           file_name="Exp5_X_vs_Time.png", mime="image/png")
        st.download_button("Download Graph 2 (PNG)", data=fig_to_png_bytes(fig2),
                           file_name="Exp5_N_vs_X.png", mime="image/png")


# =========================================================
# EXP 6: STEAM DISTILLATION — VAPOUR EFFICIENCY
# =========================================================
def exp_steam_distillation():
    st.title("Experiment 6: Steam Distillation (Turpentine–Water) — Vapour Efficiency")

    defaults = {
        "ρA Turpentine (g/mL)": 0.87,
        "ρB Water (g/mL)": 1.00,
        "MW Turpentine (g/mol)": 136.0,
        "MW Water (g/mol)": 18.0,
        "pA (Pa)": 27000.0,
        "P total (Pa)": 101325.0
    }
    consts = constants_panel("const_steam", defaults)

    rhoA = float(consts["ρA Turpentine (g/mL)"])
    rhoB = float(consts["ρB Water (g/mL)"])
    MWA = float(consts["MW Turpentine (g/mol)"])
    MWB = float(consts["MW Water (g/mol)"])
    pA = float(consts["pA (Pa)"])
    P = float(consts["P total (Pa)"])

    if not (0 < pA < P):
        st.error("Invalid pressures: must satisfy 0 < pA < P. Fix in Advanced constants.")
        return
    pB = P - pA
    theory_ratio = (pA/pB) * (MWA/MWB)

    st.subheader("Experimental Data (Enter readings only)")
    st.caption("Enter trials as pairs: VA VB (mL). One trial per line.\nExample:\n10 40\n12 38\n...")

    raw = st.text_area("Trials (VA VB per line)", height=120, key="steam_trials_raw")

    if st.button("Run Calculation (Steam Distillation)"):
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        if len(lines) < 1:
            st.error("Enter at least 1 trial.")
            return

        rows = []
        for i, ln in enumerate(lines, start=1):
            parts = ln.replace(",", " ").split()
            if len(parts) != 2:
                st.error(f"Trial {i}: enter exactly 2 numbers (VA VB).")
                return
            VA = float(parts[0])
            VB = float(parts[1])
            if VB == 0:
                st.error(f"Trial {i}: VB cannot be zero.")
                return
            mA = VA * rhoA
            mB = VB * rhoB
            actual = mA/mB
            eff = (actual/theory_ratio) * 100.0
            rows.append({
                "Trial": i,
                "VA (mL)": VA,
                "VB (mL)": VB,
                "mA=VA*ρA (g)": mA,
                "mB=VB*ρB (g)": mB,
                "(mA/mB) actual": actual,
                "(mA/mB) theoretical": theory_ratio,
                "Vapour efficiency (%)": eff
            })

        df = pd.DataFrame(rows)

        st.subheader("Observation Table")
        st.dataframe(df[["Trial","VA (mL)","VB (mL)"]].round(4), use_container_width=True, hide_index=True)

        st.subheader("Calculation Table")
        st.dataframe(df.round(6), use_container_width=True, hide_index=True)

        avg_eff = float(np.mean(df["Vapour efficiency (%)"].values))

        st.subheader("Graphs")
        fig1, ax1 = plt.subplots()
        x = df["Trial"].values
        ax1.bar(x - 0.18, df["(mA/mB) actual"].values, 0.36, label="Actual")
        ax1.bar(x + 0.18, np.full_like(x, theory_ratio, dtype=float), 0.36, label="Theoretical (fixed)")
        ax1.set_xlabel("Trial")
        ax1.set_ylabel("mA/mB")
        ax1.set_title("Actual vs Theoretical Mass Ratio")
        ax1.grid(True)
        ax1.legend()
        st.pyplot(fig1)

        fig2, ax2 = plt.subplots()
        ax2.plot(x, df["Vapour efficiency (%)"].values, marker="o")
        ax2.set_xlabel("Trial")
        ax2.set_ylabel("Vapour efficiency (%)")
        ax2.set_title("Vapour Efficiency per Trial")
        ax2.grid(True)
        st.pyplot(fig2)

        st.subheader("Result (Lab record style)")
        if len(df) == 1:
            st.markdown(f"**Result:** Vapour efficiency for steam distillation (Trial 1) = **{df.loc[0,'Vapour efficiency (%)']:.2f}%**.")
        else:
            st.markdown(f"**Result:** Average vapour efficiency for steam distillation (n={len(df)}) = **{avg_eff:.2f}%**.")

        st.subheader("Download")
        excel = df_to_excel_bytes({
            "Fixed_Constants": pd.DataFrame([consts]),
            "Observation_Calc": df,
            "Summary": pd.DataFrame([{"Avg vapour efficiency (%)": avg_eff, "Theory ratio": theory_ratio}])
        })
        st.download_button("Download Excel (Exp6)", data=excel,
                           file_name="Exp6_Steam_Distillation.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        st.download_button("Download Graph 1 (PNG)", data=fig_to_png_bytes(fig1),
                           file_name="Exp6_Actual_vs_Theory.png", mime="image/png")
        st.download_button("Download Graph 2 (PNG)", data=fig_to_png_bytes(fig2),
                           file_name="Exp6_Efficiency.png", mime="image/png")


# =========================================================
# EXPERIMENT REGISTRY
# =========================================================
EXPERIMENTS = {
    "1. Simple Distillation (Acetone–Water) — Rayleigh": exp_simple_distillation,
    "2. Multi-stage Leaching (Na2CO3)": exp_multistage_leaching,
    "3. Single-stage Leaching (Na2CO3)": exp_singlestage_leaching,
    "4. Adsorption Isotherm (Freundlich) — Acetic acid/Carbon": exp_adsorption,
    "5. Drying (Tray Dryer)": exp_drying,
    "6. Steam Distillation (Turpentine–Water) — Vapour efficiency": exp_steam_distillation,
}

exp_list = list(EXPERIMENTS.keys())

# =========================================================
# MAIN PAGE NAVIGATION (MOBILE SAFE)
# =========================================================
st.subheader("Select Experiment")

if "exp_idx" not in st.session_state:
    st.session_state["exp_idx"] = 4  # default -> Exp5 Drying

selected = st.selectbox("Experiment", exp_list, index=st.session_state["exp_idx"])

# detect switch -> clear inputs only
prev = st.session_state.get("prev_exp_key")
if prev is None:
    st.session_state["prev_exp_key"] = selected
elif prev != selected:
    clear_inputs_for_experiment(selected)
    st.session_state["prev_exp_key"] = selected

st.session_state["exp_idx"] = exp_list.index(selected)

c1, c2, c3 = st.columns(3)
with c1:
    if st.button("⬅ Previous"):
        st.session_state["exp_idx"] = max(0, st.session_state["exp_idx"] - 1)
        new_key = exp_list[st.session_state["exp_idx"]]
        clear_inputs_for_experiment(new_key)
        st.session_state["prev_exp_key"] = new_key
        st.rerun()

with c3:
    if st.button("Next ➡"):
        st.session_state["exp_idx"] = min(len(exp_list) - 1, st.session_state["exp_idx"] + 1)
        new_key = exp_list[st.session_state["exp_idx"]]
        clear_inputs_for_experiment(new_key)
        st.session_state["prev_exp_key"] = new_key
        st.rerun()

st.caption("Inputs reset on experiment switch. Constants are retained per experiment (editable in Advanced mode).")

# Run selected experiment
EXPERIMENTS[exp_list[st.session_state["exp_idx"]]]()
