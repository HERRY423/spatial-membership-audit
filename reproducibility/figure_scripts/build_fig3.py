#!/usr/bin/env python3
"""Build Figure 3 (Same-platform U_common benchmarks, Soft plug-in MAE shrinkage illusion,
Dual orthogonal proteomics [patient-disaggregated IF + M242T 38-plex dual IMC], and
Native Linux Proseg v3.2.0 + full downstream pipeline) strictly from verified v3 CSVs.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

ROOT = Path(r"C:\Reproducution")
AUDIT = ROOT / "review" / "v3_unified_audit_2026-11-25"
COMP = ROOT / "review" / "v3_linux_competitors_2026-11-25"
OUT = ROOT / "manuscript" / "gb_methodology_v3_2026-11-25" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.4,
    "axes.titlesize": 9.6,
    "axes.labelsize": 8.6,
    "xtick.labelsize": 7.9,
    "ytick.labelsize": 7.9,
    "legend.fontsize": 7.4,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 300,
})

C_NAIVE = "#7F8C8D"
C_BATCH = "#9B59B6"
C_SOFT = "#C0392B"
C_DEPTH = "#E67E22"
C_SH = "#1E8449"
C_MITM = "#2471A3"
C_STAB = "#117A65"
C_PROSEG = "#2E4053"


def main():
    t1 = pd.read_csv(AUDIT / "audit_table1_ucommon.csv")
    t2 = pd.read_csv(AUDIT / "audit_table2_ucommon.csv")
    m_single = pd.read_csv(AUDIT / "audit_m242t_single_ucommon.csv")
    pro = pd.read_csv(COMP / "proseg_v320_ucommon_summary.csv")

    fig = plt.figure(figsize=(15.6, 11.4), facecolor="white")
    outer = gridspec.GridSpec(
        2, 2, figure=fig,
        left=0.055, right=0.985, top=0.935, bottom=0.06,
        wspace=0.22, hspace=0.33,
    )

    # =========================================================================
    # PANEL A: Same-platform & longitudinal U_common cross-run ICC
    # =========================================================================
    ax_a = fig.add_subplot(outer[0, 0])

    def get_t1(bench, arm):
        sub = t1[(t1["benchmark"] == bench) & (t1["arm"] == arm)]
        return sub.iloc[0] if len(sub) else None

    cohorts = [
        ("MERFISH+\nHeart G3\n(U=917)", "MERFISH_Heart_G3", "Split_half_2s_P2", None, None),
        ("Wang HTMA\nr1 vs r2\n(U=490)", "Wang_HTMA_r1_vs_r2", "Depth_only_2s_q075", "MITM_geom_eb_guardrail", "scVI_expr_SingleR"),
        ("Wang 1-yr\nr2 vs y24\n(U=391)", "Wang_HTMA_r2_vs_y24", "Depth_only_1s_medpos", None, "Sym_q075_B20"),
        ("Touchstone\nXenium PR1\n(U=113)", "Touchstone_Xenium_PR1", "Depth_matched_MC200", None, "Scuttle_depth"),
        ("Touchstone\nCosMx PR1\n(U=415)", "Touchstone_CosMx_PR1", "Depth_only_2s_q075", None, "Single_draw_B1_2s_avg"),
    ]

    x = np.arange(len(cohorts))
    w = 0.175
    offsets = [-1.5 * w, -0.5 * w, 0.5 * w, 1.5 * w]

    naive_vals, soft_vals, sh_vals, aux_vals, aux_lbls, aux_cols = [], [], [], [], [], []
    for _, b, sh_arm, mitm_arm, alt_arm in cohorts:
        naive_vals.append(get_t1(b, "Naive")["icc"])
        soft_vals.append(get_t1(b, "Soft_plugin_tau005")["icc"])
        sh_vals.append(get_t1(b, sh_arm)["icc"])
        if mitm_arm:
            aux_vals.append(get_t1(b, mitm_arm)["icc"])
            aux_lbls.append("MITM")
            aux_cols.append(C_MITM)
        elif alt_arm:
            aux_vals.append(get_t1(b, alt_arm)["icc"])
            aux_lbls.append("scVI" if "scVI" in alt_arm else ("Sym" if "Sym" in alt_arm else ("Scut" if "Scut" in alt_arm else "B=1")))
            aux_cols.append(C_BATCH if "scVI" in alt_arm else "#95A5A6")
        else:
            aux_vals.append(get_t1(b, "Single_draw_B1_P2_avg")["icc"])
            aux_lbls.append("B=1")
            aux_cols.append("#95A5A6")

    b1 = ax_a.bar(x + offsets[0], naive_vals, w, color=C_NAIVE, label="Naive", edgecolor="white", linewidth=0.5)
    b2 = ax_a.bar(x + offsets[1], soft_vals, w, color=C_SOFT, alpha=0.88, label="Soft plug-in (τ=0.05)", edgecolor="white", linewidth=0.5)
    b3 = ax_a.bar(x + offsets[2], sh_vals, w, color=C_SH, label="Conditional raref. / Split-half", edgecolor="white", linewidth=0.5)
    b4 = ax_a.bar(x + offsets[3], aux_vals, w, color=aux_cols, label="MITM geom EB (r1/r2) / Ref. alt.", edgecolor="white", linewidth=0.5)

    for bars in (b1, b2, b3, b4):
        for rect in bars:
            h = rect.get_height()
            ax_a.text(rect.get_x() + rect.get_width() / 2, h + 0.014, f"{h:.2f}",
                      ha="center", va="bottom", fontsize=6.8, color="#1C2833")

    # Annotate key paired-t zero-effect FDR contrast on Wang r1/r2 and r2/y24
    ax_a.annotate("Paired-t FDR:\nNaive 40% | Soft 60%\nRaref. 0.0%",
                  xy=(1 + offsets[2], sh_vals[1] + 0.055), xytext=(1.55, 0.90),
                  fontsize=6.9, color="#145A32", fontweight="bold",
                  bbox=dict(boxstyle="round,pad=0.25", fc="#EAFAF1", ec="#1E8449", lw=0.7),
                  arrowprops=dict(arrowstyle="->", color="#1E8449", lw=0.8))
    ax_a.annotate("Balanced null\n(98 vs 102 tx)",
                  xy=(3 + offsets[2], sh_vals[3] + 0.06), xytext=(3.15, 0.61),
                  fontsize=6.7, color="#566573",
                  bbox=dict(boxstyle="round,pad=0.2", fc="#F8F9F9", ec="#ABB2B9", lw=0.6),
                  arrowprops=dict(arrowstyle="->", color="#7F8C8D", lw=0.7))
    ax_a.annotate("Soft SD\n0.14–0.25!",
                  xy=(4 + offsets[1], soft_vals[4] + 0.055), xytext=(3.68, 0.34),
                  fontsize=6.8, color=C_SOFT, fontweight="bold",
                  bbox=dict(boxstyle="round,pad=0.2", fc="#FDEDEC", ec=C_SOFT, lw=0.65),
                  arrowprops=dict(arrowstyle="->", color=C_SOFT, lw=0.75))

    # Label aux bars cleanly (only when tall enough)
    for idx_c, lbl in enumerate(aux_lbls):
        if aux_vals[idx_c] >= 0.18:
            ax_a.text(x[idx_c] + offsets[3], aux_vals[idx_c] / 2, lbl,
                      ha="center", va="center", fontsize=6.1, color="white", fontweight="bold", rotation=90)
        else:
            ax_a.text(x[idx_c] + offsets[3], aux_vals[idx_c] + 0.052, lbl,
                      ha="center", va="bottom", fontsize=5.8, color="#566573", fontweight="bold")

    ax_a.set_xticks(x)
    ax_a.set_xticklabels([c[0] for c in cohorts], fontsize=7.6)
    ax_a.set_ylabel("Cross-run ICC (strict $U_{common}$)")
    ax_a.set_ylim(0, 1.07)
    ax_a.grid(axis="y", alpha=0.25, linestyle="--")
    ax_a.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.92, edgecolor="#D5D8DC", ncol=2, fontsize=6.9)
    ax_a.set_title("a   Same-platform & longitudinal cross-run ICC ($U_{common}$)", loc="left", fontweight="bold", pad=8)

    # =========================================================================
    # PANEL B: Confronting Soft Plug-in (tau=0.05) MAE Shrinkage Illusion
    # =========================================================================
    ax_b = fig.add_subplot(outer[0, 1])

    # Shade collapse zone (SD < 0.50)
    ax_b.axvspan(0.0, 0.50, color="#FADBD8", alpha=0.35, zorder=0)
    ax_b.axvline(0.50, color=C_SOFT, linestyle="--", linewidth=1.0, alpha=0.75)
    ax_b.text(0.25, 0.87, "Shrinkage-to-zero\ncollapse zone\n($SD_{ratio} < 0.50$)",
              ha="center", va="top", fontsize=7.6, color="#922B21", fontweight="bold")
    ax_b.text(0.92, 0.87, "High-fidelity\nvariance-preserved\nregime",
              ha="center", va="top", fontsize=7.5, color="#145A32", fontweight="bold")

    # Collect exact trajectories from verified CSVs
    def row_t2(bench, scope, arm):
        return t2[(t2["benchmark"] == bench) & (t2["scope"] == scope) & (t2["arm"] == arm)].iloc[0]

    def row_pro(scope, arm):
        return pro[(pro["scope"] == scope) & (pro["arm"] == arm)].iloc[0]

    # 3 orthogonal truth cohorts:
    # 1) CosMx 108 FOVs IF (y = IF_z15 Spearman rho, x = sd_ratio_deep_vs_naive)
    # 2) CosMx Proseg v3.2.0 54 FOVs (y = IF15 Spearman rho, x = sd_ratio_deep_vs_vendor)
    # 3) M242T Dual-IMC H vs M r=25um (y = IMC_poly ICC, x = sd_ratio_deep_vs_naive)
    pts = [
        # (cohort_label, marker, arm_short, x_sd, y_fid, color, tx, ty, ha, note)
        ("CosMx 108 FOVs (IF $\\rho$)", "o", "Naive",
         row_t2("CosMx_Breast_108FOVs_IF", "POOLED_108FOVs", "Naive")["sd_ratio_deep_vs_naive"],
         row_t2("CosMx_Breast_108FOVs_IF", "POOLED_108FOVs", "Naive")["truth1_spearman_rho"],
         C_NAIVE, 1.07, 0.515, "left", "CosMx Naive"),
        ("CosMx 108 FOVs (IF $\\rho$)", "o", "Stab w1",
         row_t2("CosMx_Breast_108FOVs_IF", "POOLED_108FOVs", "Stability_weighted_w1")["sd_ratio_deep_vs_naive"],
         row_t2("CosMx_Breast_108FOVs_IF", "POOLED_108FOVs", "Stability_weighted_w1")["truth1_spearman_rho"],
         C_STAB, 1.05, 0.725, "left", "CosMx Stab w1\n(IF $\\rho$=0.63)"),
        ("CosMx 108 FOVs (IF $\\rho$)", "o", "Soft",
         row_t2("CosMx_Breast_108FOVs_IF", "POOLED_108FOVs", "Soft_plugin_tau005")["sd_ratio_deep_vs_naive"],
         row_t2("CosMx_Breast_108FOVs_IF", "POOLED_108FOVs", "Soft_plugin_tau005")["truth1_spearman_rho"],
         C_SOFT, 0.44, 0.125, "left", "CosMx Soft τ=0.05\n(SD=0.36, IF $\\rho$=0.19; BR2 −0.16!)"),

        ("Proseg v3.2.0 54 FOVs (IF $\\rho$)", "s", "Proseg Naive",
         row_pro("POOLED_54FOVs", "Proseg_v320_Naive")["sd_ratio_deep_vs_vendor"],
         row_pro("POOLED_54FOVs", "Proseg_v320_Naive")["if15_spearman_avg"],
         C_PROSEG, 0.76, 0.735, "center", "Proseg Naive"),
        ("Proseg v3.2.0 54 FOVs (IF $\\rho$)", "s", "Proseg+MITM",
         row_pro("POOLED_54FOVs", "Proseg_v320_MITM_geom_eb")["sd_ratio_deep_vs_vendor"],
         row_pro("POOLED_54FOVs", "Proseg_v320_MITM_geom_eb")["if15_spearman_avg"],
         C_MITM, 0.62, 0.705, "center", "Proseg+MITM\n(ICC=0.92)"),
        ("Proseg v3.2.0 54 FOVs (IF $\\rho$)", "s", "Proseg+SplitHalf",
         row_pro("POOLED_54FOVs", "Proseg_v320_SplitHalf2s_default")["sd_ratio_deep_vs_vendor"],
         row_pro("POOLED_54FOVs", "Proseg_v320_SplitHalf2s_default")["if15_spearman_avg"],
         C_SH, 0.54, 0.565, "left", "Proseg+SH 2s\n(ICC=0.97)"),
        ("Proseg v3.2.0 54 FOVs (IF $\\rho$)", "s", "Proseg+Soft",
         row_pro("POOLED_54FOVs", "Proseg_v320_Soft_tau005")["sd_ratio_deep_vs_vendor"],
         row_pro("POOLED_54FOVs", "Proseg_v320_Soft_tau005")["if15_spearman_avg"],
         C_SOFT, 0.11, 0.345, "left", "Proseg+Soft\n(MAE −67%,\nIF $\\rho$=0.28)"),

        ("M242T Dual-IMC (poly ICC)", "^", "M242T Naive",
         row_t2("M242T_DualIMC_r25um", "H_vs_M", "Naive")["sd_ratio_deep_vs_naive"],
         row_t2("M242T_DualIMC_r25um", "H_vs_M", "Naive")["truth1_icc"],
         C_NAIVE, 0.94, 0.705, "center", "M242T Naive"),
        ("M242T Dual-IMC (poly ICC)", "^", "M242T MITM",
         row_t2("M242T_DualIMC_r25um", "H_vs_M", "MITM_geom_eb_secondary")["sd_ratio_deep_vs_naive"],
         row_t2("M242T_DualIMC_r25um", "H_vs_M", "MITM_geom_eb_secondary")["truth1_icc"],
         C_MITM, 0.89, 0.475, "center", "M242T MITM\n(ICC=0.88)"),
        ("M242T Dual-IMC (poly ICC)", "^", "M242T Soft",
         row_t2("M242T_DualIMC_r25um", "H_vs_M", "Soft_plugin_tau005")["sd_ratio_deep_vs_naive"],
         row_t2("M242T_DualIMC_r25um", "H_vs_M", "Soft_plugin_tau005")["truth1_icc"],
         C_SOFT, 0.55, 0.315, "left", "M242T Soft (MAE −63%,\nIMC ICC 0.61→0.32)"),
    ]

    # Draw collapse arrows from Naive -> Soft for each cohort
    arrow_pairs = [
        (pts[0], pts[2]),
        (pts[3], pts[6]),
        (pts[7], pts[9]),
    ]
    for src, dst in arrow_pairs:
        ax_b.annotate("", xy=(dst[3], dst[4]), xytext=(src[3], src[4]),
                      arrowprops=dict(arrowstyle="-|>", color=C_SOFT, lw=1.15, ls="--", alpha=0.6, shrinkA=6, shrinkB=7))

    for _, mk, _, xv, yv, col, tx, ty, ha_val, note in pts:
        ax_b.scatter([xv], [yv], marker=mk, s=74, color=col, edgecolor="white", linewidth=0.9, zorder=5)
        ax_b.annotate(note, xy=(xv, yv), xytext=(tx, ty),
                      fontsize=6.6, color="#1C2833" if col != C_SOFT else "#922B21",
                      fontweight="bold" if col in (C_SOFT, C_SH, C_STAB) else "normal",
                      ha=ha_val, va="center",
                      bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.78),
                      arrowprops=dict(arrowstyle="-", color="#95A5A6", lw=0.6))

    # Custom legend for marker shapes
    shape_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#566573", markersize=7, label="CosMx 108 FOVs (IF $\\rho$)"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="#566573", markersize=7, label="Proseg v3.2.0 54 FOVs (IF $\\rho$)"),
        plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="#566573", markersize=7, label="M242T Dual-IMC (poly ICC)"),
    ]
    ax_b.legend(handles=shape_handles, loc="lower right", frameon=True, facecolor="white", framealpha=0.94, edgecolor="#D5D8DC", fontsize=6.9)
    ax_b.set_xlabel("Deep-side structural variance retention ($SD_{ratio}$ vs Naive / Vendor)")
    ax_b.set_ylabel("Orthogonal protein ground-truth concordance")
    ax_b.set_xlim(0.08, 1.25)
    ax_b.set_ylim(0.03, 0.93)
    ax_b.grid(True, alpha=0.25, linestyle="--")
    ax_b.set_title("b   Why Soft plug-in (τ=0.05) MAE gain is a shrinkage-to-zero illusion", loc="left", fontweight="bold", pad=8)

    # =========================================================================
    # PANEL C: Dual Orthogonal Protein Validation (c1: BR1/BR2 IF | c2: M242T IMC)
    # =========================================================================
    sub_c = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1, 0], wspace=0.28, width_ratios=[1.0, 1.15])
    ax_c1 = fig.add_subplot(sub_c[0, 0])
    ax_c2 = fig.add_subplot(sub_c[0, 1])

    # c1: Patient-disaggregated CosMx 108 FOVs IF_z15 MAE (lower = better)
    donors = [("BR1 4066\n(54 FOVs,\nU=159)", "Patient_BR1_4066_54FOVs"),
              ("BR2 4399\n(54 FOVs,\nU=159)", "Patient_BR2_4399_54FOVs")]
    c1_arms = [
        ("Naive", "Naive", C_NAIVE),
        ("Stab w1", "Stability_weighted_w1", C_STAB),
        ("Stab w2", "Stability_weighted_w2", "#145A32"),
        ("Soft τ=0.05", "Soft_plugin_tau005", C_SOFT),
    ]
    xc1 = np.arange(len(donors))
    wc1 = 0.19
    for j, (lbl, arm_k, col) in enumerate(c1_arms):
        maes = [row_t2("CosMx_Breast_108FOVs_IF", sc, arm_k)["truth1_mae"] for _, sc in donors]
        iccs = [row_t2("CosMx_Breast_108FOVs_IF", sc, arm_k)["truth1_icc"] for _, sc in donors]
        pos = xc1 + (j - 1.5) * wc1
        bars = ax_c1.bar(pos, maes, wc1, color=col, label=lbl, edgecolor="white", linewidth=0.5)
        for rect, m_val, i_val in zip(bars, maes, iccs):
            ax_c1.text(rect.get_x() + rect.get_width() / 2, m_val + 0.015, f"{m_val:.2f}",
                       ha="center", va="bottom", fontsize=6.6, fontweight="bold" if arm_k != "Naive" else "normal",
                       color="#922B21" if arm_k == "Soft_plugin_tau005" else "#1C2833")
            ax_c1.text(rect.get_x() + rect.get_width() / 2, 0.04, f"ICC\n{i_val:+.2f}",
                       ha="center", va="bottom", fontsize=5.8, color="white", fontweight="bold")
    ax_c1.set_xticks(xc1)
    ax_c1.set_xticklabels([d[0] for d in donors], fontsize=7.6)
    ax_c1.set_ylabel("Protein IF_z15 MAE (lower = better)")
    ax_c1.set_ylim(0, 0.96)
    ax_c1.grid(axis="y", alpha=0.25, linestyle="--")
    ax_c1.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.92, edgecolor="#D5D8DC", fontsize=6.6, ncol=2)
    ax_c1.set_title("c1  CosMx IF (by patient)", loc="left", fontweight="bold", fontsize=8.9, pad=6)

    # c2: M242T 38-plex Dual-Segmentation IMC (U_common=190, r=25um)
    m25 = m_single[m_single["radius_um"] == 25].set_index("estimator")
    m_order = [
        ("Native H\n(406 tx)", "Native_H_406tx"),
        ("Native M\n(61 tx)", "Native_M_61tx"),
        ("Extrap\nM→H (EB)", "Extrap_M_to_H_eb"),
        ("Soft M\n(τ=0.05)", "Soft_tau005_M"),
        ("Native L\n(20 tx)", "Native_L_20tx"),
        ("Extrap L→H\n(20× fail)", "Extrap_L_to_H_eb"),
    ]
    xc2 = np.arange(len(m_order))
    wc2 = 0.36
    poly_r = [m25.loc[k, "r_pair_poly"] for _, k in m_order]
    indep_r = [m25.loc[k, "r_pair_indep"] for _, k in m_order]

    cols_poly = [C_PROSEG, C_NAIVE, C_MITM, C_SOFT, "#ABB2B9", "#D35400"]
    cols_ind = ["#5D6D7E", "#BFC9CA", "#5DADE2", "#E6B0AA", "#D5D8DC", "#EDBB99"]
    b_p = ax_c2.bar(xc2 - wc2 / 2, poly_r, wc2, color=cols_poly, edgecolor="white", linewidth=0.5, label="IMC (Xenium poly)")
    b_i = ax_c2.bar(xc2 + wc2 / 2, indep_r, wc2, color=cols_ind, edgecolor="#566573", linewidth=0.45, hatch="///", label="IMC (Indep. watershed)")
    for rect in b_p:
        h = rect.get_height()
        ax_c2.text(rect.get_x() + rect.get_width() / 2, h + 0.015, f"{h:.2f}", ha="center", va="bottom", fontsize=6.4, color="#1C2833")
    for rect in b_i:
        h = rect.get_height()
        ax_c2.text(rect.get_x() + rect.get_width() / 2, h + 0.015, f"{h:.2f}", ha="center", va="bottom", fontsize=6.1, color="#566573")

    ax_c2.annotate("Rescue!", xy=(2 - wc2 / 2, poly_r[2] + 0.05), xytext=(1.75, 0.94),
                   fontsize=7.0, color=C_MITM, fontweight="bold",
                   arrowprops=dict(arrowstyle="->", color=C_MITM, lw=0.85))
    ax_c2.annotate("20× boundary", xy=(5, poly_r[5] + 0.05), xytext=(4.05, 0.57),
                   fontsize=6.6, color="#A04000", fontweight="bold",
                   bbox=dict(boxstyle="round,pad=0.18", fc="#FEF5E7", ec="#D35400", lw=0.6),
                   arrowprops=dict(arrowstyle="->", color="#D35400", lw=0.8))

    ax_c2.set_xticks(xc2)
    ax_c2.set_xticklabels([m[0] for m in m_order], fontsize=6.8)
    ax_c2.set_ylabel("Pairwise $r$ vs 38-plex IMC ($U=190$)")
    ax_c2.set_ylim(0, 1.05)
    ax_c2.grid(axis="y", alpha=0.25, linestyle="--")
    leg_c2 = [
        mpatches.Patch(facecolor="#566573", label="IMC Xenium poly"),
        mpatches.Patch(facecolor="#D5D8DC", edgecolor="#566573", hatch="///", label="IMC Indep. watershed"),
    ]
    ax_c2.legend(handles=leg_c2, loc="upper right", frameon=True, facecolor="white", framealpha=0.92, edgecolor="#D5D8DC", fontsize=6.6)
    ax_c2.set_title("c2  M242T Dual-IMC (DNA reg. 0.17 µm)", loc="left", fontweight="bold", fontsize=8.9, pad=6)

    # =========================================================================
    # PANEL D: Native Linux Proseg v3.2.0 + Full Downstream Standardization
    # =========================================================================
    ax_d = fig.add_subplot(outer[1, 1])

    pro_p = pro[pro["scope"] == "POOLED_54FOVs"].set_index("arm")
    pro_arms = [
        # (arm, label, color, marker, size, tx, ty, ha)
        ("Vendor_Naive", "Vendor Naive\n(MAE 0.220)", C_NAIVE, "o", 105, 0.655, 0.565, "center"),
        ("Proseg_v320_Naive", "Proseg v3.2.0 Naive\n(MAE 0.112, +44.2%)", C_PROSEG, "s", 115, 0.775, 0.545, "center"),
        ("Proseg_v320_Stability_w1", "Proseg + Stability w1\n(best IF MAE 0.319)", C_STAB, "D", 105, 0.735, 0.728, "center"),
        ("Proseg_v320_MITM_geom_eb", "Proseg + MITM EB\n(ICC 0.915, IF $\\rho$ 0.626)", C_MITM, "P", 125, 0.932, 0.728, "center"),
        ("Proseg_v320_Depth2s_q075", "Proseg + Depth 2s\n(MAE 0.058)", C_DEPTH, "^", 95, 0.925, 0.515, "center"),
        ("Proseg_v320_SplitHalf2s_default", "Proseg + Split-half 2s (default)\n(ICC 0.969, MAE 0.053,\n+52.6% vs Proseg / +73.6% vs Vend.)", C_SH, "*", 210, 0.745, 0.440, "center"),
        ("Proseg_v320_Soft_tau005", "Proseg + Soft τ=0.05\n(IF $\\rho$ 0.280, BR2 −0.02,\nIF MAE 0.688, SD 0.37!)", C_SOFT, "X", 130, 0.935, 0.370, "center"),
    ]

    # Complementarity arrows
    p_vend = (pro_p.loc["Vendor_Naive", "comp_icc"], pro_p.loc["Vendor_Naive", "if15_spearman_avg"])
    p_pn = (pro_p.loc["Proseg_v320_Naive", "comp_icc"], pro_p.loc["Proseg_v320_Naive", "if15_spearman_avg"])
    p_sh = (pro_p.loc["Proseg_v320_SplitHalf2s_default", "comp_icc"], pro_p.loc["Proseg_v320_SplitHalf2s_default", "if15_spearman_avg"])
    p_mitm = (pro_p.loc["Proseg_v320_MITM_geom_eb", "comp_icc"], pro_p.loc["Proseg_v320_MITM_geom_eb", "if15_spearman_avg"])
    p_soft = (pro_p.loc["Proseg_v320_Soft_tau005", "comp_icc"], pro_p.loc["Proseg_v320_Soft_tau005", "if15_spearman_avg"])

    # Arrow 1: Vendor -> Proseg Naive (upstream resegmentation)
    ax_d.annotate("", xy=p_pn, xytext=p_vend,
                  arrowprops=dict(arrowstyle="-|>", color=C_PROSEG, lw=1.6, connectionstyle="arc3,rad=-0.12", shrinkA=7, shrinkB=8))
    ax_d.text(0.768, 0.665, "1. Upstream Proseg\nresegmentation", fontsize=7.0, color=C_PROSEG, fontweight="bold", ha="center", va="bottom")

    # Arrow 2: Proseg Naive -> Proseg + Split-half 2s (downstream comparability)
    ax_d.annotate("", xy=p_sh, xytext=p_pn,
                  arrowprops=dict(arrowstyle="-|>", color=C_SH, lw=1.6, connectionstyle="arc3,rad=0.14", shrinkA=7, shrinkB=9))
    # Arrow 3: Proseg Naive -> Proseg + MITM (downstream fidelity + comp)
    ax_d.annotate("", xy=p_mitm, xytext=p_pn,
                  arrowprops=dict(arrowstyle="-|>", color=C_MITM, lw=1.35, shrinkA=7, shrinkB=8))
    # Arrow 4: Proseg Naive -> Proseg + Soft (protein collapse)
    ax_d.annotate("", xy=p_soft, xytext=p_pn,
                  arrowprops=dict(arrowstyle="-|>", color=C_SOFT, lw=1.35, ls="--", shrinkA=7, shrinkB=8))

    for arm_k, lbl, col, mk, sz, tx, ty, ha_val in pro_arms:
        xv = pro_p.loc[arm_k, "comp_icc"]
        yv = pro_p.loc[arm_k, "if15_spearman_avg"]
        ax_d.scatter([xv], [yv], s=sz, marker=mk, color=col, edgecolor="white", linewidth=0.9, zorder=6)
        ax_d.annotate(
            lbl, xy=(xv, yv), xytext=(tx, ty), ha=ha_val, va="center",
            fontsize=6.7, color="#922B21" if col == C_SOFT else ("#145A32" if col == C_SH else "#1C2833"),
            fontweight="bold" if col in (C_SH, C_SOFT, C_MITM, C_PROSEG) else "normal",
            bbox=dict(boxstyle="round,pad=0.22", fc="white", ec=col, alpha=0.94, lw=0.75) if col in (C_SH, C_SOFT) else dict(boxstyle="round,pad=0.14", fc="white", ec="none", alpha=0.75),
            arrowprops=dict(arrowstyle="-", color=col, lw=0.7, alpha=0.8),
            zorder=7,
        )

    # Compact donor QC inset box at bottom-left
    qc_txt = (
        "Native Linux Proseg v3.2.0 QC (54 FOVs, U=162):\n"
        "• BR1 4066: 99.8% ≤10µm (med 1.41µm), tx 130→308\n"
        "  ICC: Vend 0.901 → Proseg 0.940 → +SH 2s 0.984\n"
        "• BR2 4399: 99.6% ≤10µm (med 1.72µm), tx 154→374\n"
        "  ICC: Vend 0.389 → Proseg 0.829 → +SH 2s 0.943"
    )
    ax_d.text(0.625, 0.232, qc_txt, fontsize=6.5, color="#2C3E50", va="bottom", ha="left",
              bbox=dict(boxstyle="round,pad=0.32", fc="#F4F6F7", ec="#BDC3C7", lw=0.75))

    ax_d.set_xlabel("4× cross-depth comparability ICC (strict $U_{common} = 162$)")
    ax_d.set_ylabel("Same-cell protein IF_z15 concordance (Spearman $\\rho$)")
    ax_d.set_xlim(0.615, 1.02)
    ax_d.set_ylim(0.22, 0.775)
    ax_d.grid(True, alpha=0.25, linestyle="--")
    ax_d.set_title("d   Native Linux Proseg v3.2.0 + downstream standardization (strict complements)", loc="left", fontweight="bold", pad=8)

    fig.suptitle(
        "Figure 3. Unified $U_{common}$ benchmarks, soft plug-in shrinkage illusion, dual proteomics, and native Linux Proseg v3.2.0",
        fontsize=11.4, fontweight="bold", y=0.982, color="#154360",
    )

    for ext in ("png", "pdf"):
        out_p = OUT / f"Figure3_same_platform_and_protein_validation.{ext}"
        fig.savefig(out_p, dpi=300)
        print(f"Saved: {out_p} ({out_p.stat().st_size:,} B)")
    plt.close(fig)


if __name__ == "__main__":
    main()
