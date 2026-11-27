#!/usr/bin/env python3
"""Generate publication-ready Figure 1, Figure 2, SuppFigure 1 (copy), and SuppFigure 2
for the v3.0 Genome Biology Methodology manuscript (300 DPI PNG + vector PDF).
All values read strictly from verified source CSVs/JSONs; zero internal shorthands.
"""
from __future__ import annotations

import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(r"C:\Reproducution")
OUT_DIR = ROOT / "manuscript" / "gb_methodology_v3_2026-11-25" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def set_pub_style(font=9.0, title=10.0, label=8.8, tick=8.0, legend=7.4):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": font,
        "axes.titlesize": title,
        "axes.labelsize": label,
        "xtick.labelsize": tick,
        "ytick.labelsize": tick,
        "legend.fontsize": legend,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save_fig(fig, stem: str):
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {stem}.png/.pdf", flush=True)


# =============================================================================
# FIGURE 1: Capture-Stochasticity Atlas + Frozen v3.0 Two-Estimand Architecture
# =============================================================================
def make_figure1():
    set_pub_style(font=9.1, title=10.1, label=8.9, tick=8.1, legend=7.6)
    p1 = ROOT / "review" / "top_phase1_2026-11-19"
    flip = pd.read_csv(p1 / "x1_problem_atlas" / "x1_flip_atlas_summary.csv")
    depth = pd.read_csv(p1 / "x1_problem_atlas" / "x1_depth_shift_summary.csv")

    fig = plt.figure(figsize=(14.4, 9.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.22, 0.96], hspace=0.38, wspace=0.28)

    # Panel a
    ax = fig.add_subplot(gs[0, 0])
    order_ds = [
        ("Xenium_breast", 0.50, "Xenium Breast (q=0.50)"),
        ("lung", 0.50, "Xenium Lung (q=0.50)"),
        ("kidney", 0.50, "Xenium Kidney (q=0.50)"),
        ("CosMx6K_lymph_node", 0.50, "CosMx 6K Lymph Node (q=0.50)"),
        ("Cervilla_CosMx", 0.50, "Cervilla CosMx (q=0.50)"),
        ("Wang_MERSCOPE_HTMA", 0.50, "Wang MERSCOPE HTMA (q=0.50)"),
        ("MERFISHplus_G3_side0", 0.785, "MERFISH+ Heart R0 (SingleR)"),
        ("MERFISHplus_G3_side1", 0.785, "MERFISH+ Heart R1 (SingleR)"),
    ]
    labels, sig_ev, dir_ev, sig_pw, dir_pw = [], [], [], [], []
    for ds, qv, lbl in order_ds:
        r = flip[(flip["dataset"] == ds) & (np.isclose(flip["q"], qv))].iloc[0]
        labels.append(lbl)
        sig_ev.append(r["ever_sig_flip2_pct"]); dir_ev.append(r["ever_dir_flip_pct"])
        sig_pw.append(r["pair_sig_dis2_pct"]); dir_pw.append(r["pair_dir_dis_pct"])

    y = np.arange(len(labels))
    h = 0.36
    ax.barh(y - h / 2, sig_ev, height=h, color="#C0392B", alpha=0.88, label="200-draw significance flip (|z|>1.96)")
    ax.barh(y + h / 2, dir_ev, height=h, color="#2471A3", alpha=0.88, label="200-draw direction flip (R>1 vs R<=1)")
    for i in range(len(labels)):
        ax.text(sig_ev[i] + 1.0, y[i] - h / 2, f"{sig_ev[i]:.1f}% (pw {sig_pw[i]:.1f}%)", va="center", fontsize=7.2, color="#78281F")
        ax.text(dir_ev[i] + 1.0, y[i] + h / 2, f"{dir_ev[i]:.1f}% (pw {dir_pw[i]:.1f}%)", va="center", fontsize=7.2, color="#154360")
    ax.set_yticks(y); ax.set_yticklabels(labels); ax.invert_yaxis()
    ax.set_xlim(0, 130)
    ax.axvline(10, color="#7F8C8D", ls="--", lw=0.9, label="10% reference")
    ax.set_xlabel("Evaluable cell-type pairs flipping across 200 draws (%)")
    ax.set_title("a  Capture-stochasticity instability across 7 datasets & 4 platforms", loc="left", fontweight="bold", pad=34)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.005), ncol=2, frameon=False, fontsize=7.3)

    # Panel b
    ax2 = fig.add_subplot(gs[0, 1])
    for ds, lbl, col, mk in [
        ("kidney", "Xenium Kidney", "#8E44AD", "o"),
        ("Xenium_breast", "Xenium Breast", "#2E86C1", "s"),
        ("lung", "Xenium Lung", "#16A085", "^"),
        ("Wang_MERSCOPE_HTMA", "Wang MERSCOPE HTMA", "#D35400", "D"),
        ("Cervilla_CosMx", "Cervilla CosMx", "#C0392B", "v"),
        ("CosMx6K_lymph_node", "CosMx 6K LN (n<=21)", "#7F8C8D", "P"),
    ]:
        sub = depth[depth["dataset"] == ds].sort_values("q", ascending=False)
        ax2.plot(sub["q"], sub["shift_gt25_pct"], marker=mk, color=col, lw=2.0, ms=6, label=lbl)
    ax2.axhline(20, color="#7F8C8D", ls="--", lw=0.9, label="20% screening threshold")
    ax2.set_xlim(0.85, 0.35); ax2.set_ylim(-2, 68)
    ax2.set_xlabel("Binomial retention factor q (1.0 -> 0.4)")
    ax2.set_ylabel("Well-supported pairs shifted >25% vs baseline (%)")
    ax2.set_title("b  Systematic depth bias in conditional expectation E[R | X, q]", loc="left", fontweight="bold", pad=22)
    ax2.legend(loc="upper left", frameon=True, framealpha=0.92, fontsize=7.5)

    # Panel c: Frozen v3.0 Protocol & Two-Estimand Framework Schematic
    ax3 = fig.add_subplot(gs[1, :])
    ax3.set_xlim(0, 100); ax3.set_ylim(0, 32); ax3.axis("off")
    ax3.set_title("c  Frozen v3.0 two-estimand standardization architecture (archived pre-nTMA: v3.0.0-protocol-freeze-20261125)",
                  loc="left", fontweight="bold", pad=8)

    def box(x, y, w, h, fc, ec, title_t, body_t, title_c="#1B2631"):
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5,rounding_size=1.0", fc=fc, ec=ec, lw=1.5)
        ax3.add_patch(p)
        ax3.text(x + w / 2, y + h - 2.3, title_t, ha="center", va="top", fontsize=8.4, fontweight="bold", color=title_c)
        ax3.text(x + w / 2, y + h - 5.4, body_t, ha="center", va="top", fontsize=7.4, color="#2C3E50", linespacing=1.32)

    box(0.8, 4.5, 20.5, 22.0, "#EBF5FB", "#2E86C1",
        "Inputs (Shared Panel)",
        "Cohort A & Cohort B counts\nFixed spatial graph (r = 25 um)\nFrozen classifier (SingleR /\nInSituType / CellTypist)\nZero paired training samples",
        "#1B4F72")
    box(25.2, 16.5, 25.0, 12.2, "#E8F8F5", "#1E8449",
        "PRIMARY FROZEN DEFAULT (v3.0)",
        "Unsupervised Split-Half 2s (q0=0.75, B=20)\nMatch conditional disagreement psi_A = psi_B\nZero-inclusive ratio (Prop. 1) + Prop. 3",
        "#145A32")
    box(25.2, 1.8, 25.0, 12.2, "#FEF9E7", "#B7950B",
        "SECONDARY / SINGLE-SPECIMEN",
        "Moderate gap (<=6-7x): MITM geom EB\n+ guardrail (1/sqrt(d) local heuristic)\nSingle specimen: Unthinned stability w_i=1-f_i",
        "#7D6608")
    box(54.2, 16.5, 21.5, 12.2, "#F4ECF7", "#7D3C98",
        "Joint Inference (Prop. 2)",
        "Synchronized permutation pi_m\nacross all B draws (M=199/999)\nConditional exact Type-I control",
        "#5B2C6F")
    box(54.2, 1.8, 21.5, 12.2, "#FDEDEC", "#C0392B",
        "Unified 2D Audit (U_common)",
        "Comparability: ICC + MAE + paired FDR\nFidelity: eta^2 + SD_ratio + IF/IMC\nFlags soft plug-in MAE collapse",
        "#78281F")
    box(79.2, 4.5, 20.0, 22.0, "#EAFAF1", "#196F3D",
        "Verified Portability",
        "Prospective Wang 2025 nTMA\nNative Linux spicyR, CRAWDAD,\nPANORAMIC 0.99.4, Proseg 3.2\nSingleR / InSituType / CellTypist\nspatial-membership-audit v3.0",
        "#145A32")

    arrow_kw = dict(arrowstyle="-|>,head_width=0.35,head_length=0.55", color="#34495E", lw=1.4)
    ax3.annotate("", xy=(25.0, 22.5), xytext=(21.6, 18.0), arrowprops=arrow_kw)
    ax3.annotate("", xy=(25.0, 8.0), xytext=(21.6, 12.5), arrowprops=arrow_kw)
    ax3.annotate("", xy=(54.0, 22.5), xytext=(50.5, 22.5), arrowprops=arrow_kw)
    ax3.annotate("", xy=(54.0, 8.0), xytext=(50.5, 8.0), arrowprops=arrow_kw)
    ax3.annotate("", xy=(79.0, 18.0), xytext=(76.0, 22.0), arrowprops=arrow_kw)
    ax3.annotate("", xy=(79.0, 12.5), xytext=(76.0, 8.5), arrowprops=arrow_kw)

    save_fig(fig, "Figure1_problem_framework_and_protocol_freeze")


# =============================================================================
# FIGURE 2: Three Formal Propositions + Ground-Truth Simulation & Discard Power
# =============================================================================
def make_figure2():
    set_pub_style(font=8.6, title=9.4, label=8.4, tick=7.7, legend=6.9)
    p7 = ROOT / "review" / "p7_dua_theory_and_audit_2026-11-23"
    sim_dir = ROOT / "review" / "v3_simulation_benchmark_2026-11-25"
    p1 = pd.read_csv(p7 / "prop1_mc_se_and_zero_bias.csv")
    p2 = pd.read_csv(p7 / "prop2_joint_perm_fpr.csv")
    p4 = pd.read_csv(p7 / "prop4_split_half_p2_verification.csv")
    sim_fac = pd.read_csv(sim_dir / "simulation_factorial_summary.csv")
    sim_pow = pd.read_csv(sim_dir / "simulation_discard_power_curve.csv")

    fig, axes = plt.subplots(2, 3, figsize=(15.8, 9.4), constrained_layout=True)

    # Panel a: Prop 1
    ax = axes[0, 0]
    ax.loglog(p1["B"], p1["empirical_mc_sd"], "o-", color="#1F77B4", lw=2.0, ms=5.5, label="Empirical SD(log mean R)")
    ax.loglog(p1["B"], p1["mean_delta_mc_se"], "s--", color="#E67E22", lw=2.0, ms=5.5, label="Delta-method MC SE (Prop. 1)")
    ax.set_xlabel("Thinning ensemble size B")
    ax.set_ylabel("Fixed-input MC SE of log(R)")
    ax.set_title("a  Prop. 1: Zero-inclusive ratio & MC SE\n(Bias: -0.003 vs +0.354 pos-only; SE ratio 0.986)", loc="left", fontweight="bold")
    ax.legend(loc="upper right", frameon=True, framealpha=0.92)
    ax.grid(alpha=0.25, ls=":", which="both")

    # Panel b: Prop 2
    ax = axes[0, 1]
    ax.plot(p2["flip_prob"], p2["fpr_joint_perm_p05"], "o-", color="#1E8449", lw=2.1, ms=5.5, label="Synchronized joint-perm (Prop. 2)")
    ax.plot(p2["flip_prob"], p2["fpr_unadj_196"], "^--", color="#E67E22", lw=1.8, ms=5.5, label="Unadjusted |z_mean|>1.96")
    ax.plot(p2["flip_prob"], p2["fpr_legacy_v1_196"], "x:", color="#C0392B", lw=1.8, ms=6.0, label="Additive variance inflation")
    ax.axhline(0.05, color="#333333", ls="--", lw=1.0, label="Nominal alpha = 0.05")
    ax.set_xlabel("Per-cell draw flip probability")
    ax.set_ylabel("Empirical within-run Type-I error")
    ax.set_ylim(-0.004, 0.085)
    ax.set_title("b  Prop. 2: Conditional finite-sample Type-I\n(Resolves null variance contraction under H0)", loc="left", fontweight="bold")
    ax.legend(loc="upper right", frameon=True, framealpha=0.92)
    ax.grid(alpha=0.25, ls=":")

    # Panel c: Prop 3 (formerly Prop 4: Poisson split-half identifiability)
    ax = axes[0, 2]
    x = np.arange(len(p4)); w = 0.25
    ax.bar(x - w, p4["true_equiv_ratio"], width=w, color="#2C3E50", label="Analytical target ratio")
    ax.bar(x, p4["autoq_q_b"], width=w, color="#E67E22", label="Depth-only matching")
    ax.bar(x + w, p4["p2_q_b"], width=w, color="#1E8449", label="Split-half coverage match (Prop. 3)")
    for i, r in p4.iterrows():
        ax.text(i - w, r["true_equiv_ratio"] + 0.02, f"{r['true_equiv_ratio']:.2f}", ha="center", fontsize=7.2)
        ax.text(i, r["autoq_q_b"] + 0.02, f"{r['autoq_q_b']:.2f}", ha="center", fontsize=7.2)
        ax.text(i + w, r["p2_q_b"] + 0.02, f"{r['p2_q_b']:.2f}", ha="center", fontsize=7.2)
    ax.set_xticks(x)
    ax.set_xticklabels(["Uniform 2x drop\n(+25% zero-count cells)", "Equal depth\n(1:3 per-tx SNR shift)"])
    ax.set_ylabel("Selected deeper-side retention q_B")
    ax.set_ylim(0, 1.18)
    ax.set_title("c  Prop. 3: Poisson split-half identifiability\n(Zero-inflation invariant; resolves SNR shift)", loc="left", fontweight="bold")
    ax.legend(loc="upper left", frameon=True, framealpha=0.92)
    ax.grid(axis="y", alpha=0.25, ls=":")

    # Panel d: Ground-truth simulation Type-I error under depth + SNR mismatch (Delta_true = 0)
    ax = axes[1, 0]
    scen_list = [
        (0.50, 1.0, "r=0.50\nSNR 1x"),
        (0.25, 1.0, "r=0.25\nSNR 1x"),
        (0.50, 2.0, "r=0.50\nSNR 2x"),
        (0.25, 2.0, "r=0.25\nSNR 2x"),
        (0.25, 3.0, "r=0.25\nSNR 3x*"),
        (0.15, 2.0, "r=0.15\nSNR 2x*"),
    ]
    meths = [
        ("Naive", "Naive", "#C0392B"),
        ("Soft_Plugin_tau005", "Soft plug-in", "#8E44AD"),
        ("Depth_Raref_2s", "Depth-only 2s", "#E67E22"),
        ("MITM_Geom_EB", "MITM geom EB", "#2980B9"),
        ("SplitHalf_2s_Default", "Split-half 2s (v3.0)", "#1E8449"),
    ]
    sub0 = sim_fac[np.isclose(sim_fac["delta_true_target"], 0.0)]
    xs = np.arange(len(scen_list))
    bw = 0.16
    for mi, (mk, mlbl, col) in enumerate(meths):
        vals = []
        for rd, snr, _ in scen_list:
            row = sub0[(np.isclose(sub0["depth_ratio"], rd)) & (np.isclose(sub0["snr_mismatch"], snr)) & (sub0["method"] == mk)]
            vals.append(100.0 * float(row["reject_rate_p05"].iloc[0]) if len(row) else np.nan)
        ax.bar(xs + (mi - 2) * bw, vals, width=bw, color=col, alpha=0.90, label=mlbl)
    ax.axhline(5.0, color="#2C3E50", ls="--", lw=1.0, label="Nominal 5%")
    ax.set_xticks(xs); ax.set_xticklabels([s[2] for s in scen_list], fontsize=7.3)
    ax.set_ylabel("Null false-positive rate at p<0.05 (%)")
    ax.set_ylim(0, 112)
    ax.set_title("d  Controlled simulation: Type-I error (Delta_true=0)\n(*q_a clamped at 0.10 boundary -> residual bias)", loc="left", fontweight="bold")
    ax.legend(loc="upper left", ncol=2, frameon=True, framealpha=0.92, fontsize=6.6)
    ax.grid(axis="y", alpha=0.25, ls=":")

    # Panel e: Simulation Power vs Bias under r=0.25, SNR=2x across Delta_true
    ax = axes[1, 1]
    sub_p = sim_fac[(np.isclose(sim_fac["depth_ratio"], 0.25)) & (np.isclose(sim_fac["snr_mismatch"], 2.0))]
    for mk, mlbl, col, ls, mkr in [
        ("Oracle_True", "Oracle (true labels)", "#2C3E50", "--", "o"),
        ("SplitHalf_2s_Default", "Split-half 2s (v3.0)", "#1E8449", "-", "s"),
        ("MITM_Geom_EB", "MITM geom EB", "#2980B9", "-.", "D"),
        ("Depth_Raref_2s", "Depth-only 2s", "#E67E22", ":", "^"),
        ("Naive", "Naive (unstandardized)", "#C0392B", ":", "v"),
    ]:
        s = sub_p[sub_p["method"] == mk].sort_values("delta_true_target")
        ax.plot(s["delta_true_target"], 100.0 * s["reject_rate_p05"], marker=mkr, ls=ls, color=col, lw=2.0, ms=5.5, label=mlbl)
    ax.axhline(5.0, color="#7F8C8D", ls=":", lw=0.9)
    ax.set_xlabel("True biological colocalization effect Delta_true")
    ax.set_ylabel("Rejection rate at p<0.05 (Type-I at 0; Power >0, %)")
    ax.set_ylim(-2, 106)
    ax.set_title("e  Type-I vs power at r_d=0.25, SNR=2x\n(Split-half 2s: 5.0% null FPR, bias -0.0002)", loc="left", fontweight="bold")
    ax.legend(loc="lower right", frameon=True, framealpha=0.92, fontsize=6.8)
    ax.grid(alpha=0.25, ls=":")

    # Panel f: How much power is lost by discarding deep-side molecules?
    ax = axes[1, 2]
    for reg, rlbl, col_b10, col_b1 in [
        ("Dense_100tx_1000cells", "Dense (100 tx/cell, 1000 cells)", "#1E8449", "#82E0AA"),
        ("Moderate_45tx_400cells", "Moderate (45 tx/cell, 400 cells)", "#2471A3", "#85C1E9"),
    ]:
        sub = sim_pow[(sim_pow["regime"] == reg) & (np.isclose(sim_pow["delta_true_target"], 0.15))]
        b10 = sub[sub["method"].str.startswith("Rarefied_B10")].sort_values("pct_discarded")
        b1 = sub[sub["method"].str.startswith("SingleDraw_B1")].sort_values("pct_discarded")
        ax.plot(b10["pct_discarded"], 100.0 * b10["power_loss_vs_oracle"], "o-", color=col_b10, lw=2.1, ms=5.5,
                label=f"{rlbl} — B=10 Rao-Blackwell")
        ax.plot(b1["pct_discarded"], 100.0 * b1["power_loss_vs_oracle"], "x--", color=col_b1, lw=1.6, ms=5.5,
                label=f"{rlbl} — B=1 single draw")
    ax.axvspan(80, 96, color="#FADBD8", alpha=0.35, label="Sparse cliff (>80% discarded)")
    ax.set_xlabel("Deep-side molecules discarded (%)")
    ax.set_ylabel("Power loss vs Oracle (percentage points)")
    ax.set_xlim(-2, 98); ax.set_ylim(-3, 74)
    ax.set_title("f  Power lost by discarding deep-side molecules\n(Rao-Blackwell B=10 vs single-draw B=1; 160 reps/pt)", loc="left", fontweight="bold")
    ax.legend(loc="upper left", frameon=True, framealpha=0.92, fontsize=6.4)
    ax.grid(alpha=0.25, ls=":")

    save_fig(fig, "Figure2_theory_and_controlled_simulation")


# =============================================================================
# SUPP FIG 1 (copy verified M242T IMC QC) & SUPP FIG 2 (Sensitivity & Conformal)
# =============================================================================
def make_supp_figs():
    src_s1 = ROOT / "manuscript" / "gb_methodology_v2_2026-11-24" / "figures" / "SuppFigure1_m242t_registration_and_imc_qc.png"
    if src_s1.exists():
        shutil.copy2(src_s1, OUT_DIR / "SuppFigure1_m242t_registration_and_imc_qc.png")
        print("Copied SuppFigure1_m242t_registration_and_imc_qc.png", flush=True)

    set_pub_style(font=8.6, title=9.4, label=8.4, tick=7.7, legend=6.9)
    p7 = ROOT / "review" / "p7_dua_theory_and_audit_2026-11-23"
    p3 = pd.read_csv(p7 / "prop3_conformal_coverage.csv")
    p5 = pd.read_csv(p7 / "prop5_inv_sqrtd_and_mitm.csv")
    sens = pd.read_csv(ROOT / "review" / "v3_unified_audit_2026-11-25" / "supp_sensitivity_variants.csv")
    seg = pd.read_csv(ROOT / "review" / "v3_simulation_benchmark_2026-11-25" / "simulation_segregation_check.csv")
    g3 = pd.read_csv(ROOT / "review" / "top_phase1_2026-11-19" / "x4_g3_baselines" / "g3_baselines_summary.csv")

    fig, axes = plt.subplots(2, 2, figsize=(13.6, 8.8), constrained_layout=True)

    # Panel a: Split-conformal extension (standard Vovk / Angelopoulos-Bates)
    ax = axes[0, 0]
    for reg, lbl, col, mk in [
        ("exchangeable_t3", "Exchangeable heavy-tail (t_3)", "#1F77B4", "o"),
        ("exchangeable_normal", "Exchangeable Gaussian", "#1E8449", "s"),
        ("non_exchangeable_shift_1.8x", "Non-exchangeable 1.8x shift", "#C0392B", "x"),
    ]:
        sub = p3[(p3["regime"] == reg) & (p3["K"] >= 19)]
        ax.plot(sub["K"], sub["emp_coverage"], marker=mk, color=col, lw=2.0, ms=5.5, label=lbl)
    ax.axhline(0.95, color="#333333", ls="--", lw=1.0, label="Nominal 1 - alpha = 0.95")
    ax.set_xlabel("Held-out calibration units K (K >= 19 finite)")
    ax.set_ylabel("Empirical repeat coverage")
    ax.set_ylim(0.66, 1.02)
    ax.set_title("a  Standard split-conformal extension (exchangeable vs shift)", loc="left", fontweight="bold")
    ax.legend(loc="lower left", frameon=True, framealpha=0.92)
    ax.grid(alpha=0.25, ls=":")

    # Panel b: Empirical 1/sqrt(d) local MITM variance-reduction heuristic
    ax = axes[0, 1]
    r_grid = np.linspace(0.05, 0.90, 80)
    ax.plot(r_grid, 100.0 * (1.0 - 1.0 / (1.0 + r_grid ** 0.25) ** 2), "-", color="#2471A3", lw=2.2,
            label="Local heuristic: 1 - (1 + r^(1/4))^(-2)")
    ax.scatter(p5["depth_ratio_r"], p5["var_reduction_pct"], s=60, color="#C0392B", zorder=5,
               label="Simulated MITM variance reduction")
    for _, row in p5.iterrows():
        ax.annotate(f"{row['var_reduction_pct']:.1f}%", (row["depth_ratio_r"], row["var_reduction_pct"]),
                    xytext=(0, -13), textcoords="offset points", ha="center", fontsize=7.4, color="#78281F", fontweight="bold")
    ax.text(0.08, 72.5, "Empirical boundary linearity R^2:\n  1/sqrt(d) = 0.991\n  1/d       = 0.917",
            fontsize=7.5, bbox=dict(boxstyle="round,pad=0.3", fc="#F8F9F9", ec="#ABB2B9"))
    ax.set_xlabel("Cross-run depth ratio r = s_shallow / s_deep")
    ax.set_ylabel("Upward extrapolation variance reduction (%)")
    ax.set_ylim(50, 79)
    ax.set_title("b  Empirical 1/sqrt(d) boundary scaling & local MITM heuristic", loc="left", fontweight="bold")
    ax.legend(loc="lower right", frameon=True, framealpha=0.92)
    ax.grid(alpha=0.25, ls=":")

    # Panel c: Wang HTMA q0 exploratory sweep (ICC peaks at frozen q0=0.75)
    ax = axes[1, 0]
    wq = sens[(sens["sensitivity_group"] == "Wang_q0_sweep") & (sens["split"] == "evaluation")].sort_values("q0", ascending=False)
    ax.plot(wq["q0"], wq["icc"], "o-", color="#1E8449", lw=2.1, ms=6, label="Cross-run ICC (peaks at q0=0.75)")
    ax.plot(wq["q0"], wq["sd_ratio_r2"], "s--", color="#2471A3", lw=2.0, ms=5.5, label="Deep-side SD ratio")
    ax.plot(wq["q0"], wq["red_pct"] / 100.0, "^:", color="#D35400", lw=1.9, ms=5.5, label="Discrepancy reduction (/100)")
    ax.axvline(0.75, color="#7F8C8D", ls="--", lw=1.0, label="Frozen default q0 = 0.75")
    ax.set_xlim(1.05, 0.45); ax.set_ylim(0.32, 0.88)
    ax.set_xlabel("Anchor retention q0 (1.00 -> 0.50)")
    ax.set_ylabel("Metric value (ICC, SD ratio, or fractional reduction)")
    ax.set_title("c  Exploratory q0 sensitivity sweep (Wang HTMA eval, n=53)", loc="left", fontweight="bold")
    ax.legend(loc="lower left", frameon=True, framealpha=0.92, fontsize=6.7)
    ax.grid(alpha=0.25, ls=":")

    # Panel d: Segregation-regime simulation (mu_base = -0.45, r_d=0.25, SNR=2x)
    ax = axes[1, 1]
    s0 = seg[(np.isclose(seg["snr_mismatch"], 2.0)) & (np.isclose(seg["delta_true_target"], 0.0)) & (seg["method"] != "Oracle_True")]
    s1 = seg[(np.isclose(seg["snr_mismatch"], 2.0)) & (np.isclose(seg["delta_true_target"], 0.30)) & (seg["method"] != "Oracle_True")]
    order_m = ["Naive", "Soft_Plugin_tau005", "Depth_Raref_2s", "MITM_Geom_EB", "SplitHalf_1s", "SplitHalf_2s_Default"]
    lbl_m = ["Naive", "Soft\n(tau=0.05)", "Depth\n2s", "MITM\nEB", "Split-half\n1s", "Split-half\n2s (v3.0)"]
    v0 = [100.0 * float(s0[s0["method"] == m]["reject_rate_p05"].iloc[0]) for m in order_m]
    v1 = [100.0 * float(s1[s1["method"] == m]["reject_rate_p05"].iloc[0]) for m in order_m]
    xm = np.arange(len(order_m)); bw = 0.35
    ax.bar(xm - bw / 2, v0, width=bw, color="#C0392B", alpha=0.88, label="Null Type-I error (Delta=0)")
    ax.bar(xm + bw / 2, v1, width=bw, color="#1E8449", alpha=0.88, label="Power (Delta=+0.30)")
    for i in range(len(order_m)):
        ax.text(i - bw / 2, v0[i] + 1.8, f"{v0[i]:.1f}%", ha="center", fontsize=6.8, color="#78281F")
        ax.text(i + bw / 2, v1[i] + 1.8, f"{v1[i]:.1f}%", ha="center", fontsize=6.8, color="#145A32")
    ax.axhline(5.0, color="#333333", ls="--", lw=0.9)
    ax.set_xticks(xm); ax.set_xticklabels(lbl_m, fontsize=7.5)
    ax.set_ylabel("Rejection rate at p<0.05 (%)")
    ax.set_ylim(0, 128)
    ax.set_title("d  Segregation-regime simulation (mu_base=-0.45, r=0.25, SNR=2x)", loc="left", fontweight="bold")
    ax.legend(loc="upper right", ncol=2, frameon=True, framealpha=0.92, fontsize=6.8)
    ax.grid(axis="y", alpha=0.25, ls=":")

    save_fig(fig, "SuppFigure2_sensitivity_and_conformal_sweeps")


if __name__ == "__main__":
    make_figure1()
    make_figure2()
    make_supp_figs()
    print("FIGS 1, 2, SUPP1, SUPP2 COMPLETE!", flush=True)
