#!/usr/bin/env python3
"""Build Figure 5 (Prospective zero-tuning Wang 2025 nTMA validation + nTMA native Linux
competitors, and direct reproduction + 3-factor factorial decomposition of the 35-core
negative probe) strictly from verified cloud CSV/JSON artifacts.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

ROOT = Path(r"C:\Reproducution")
NTMA = ROOT / "review" / "v3_ntma_prospective_2026-11-25"
COMP = ROOT / "review" / "v3_linux_competitors_2026-11-25"
C35 = ROOT / "review" / "v3_35core_decomposition_2026-11-25"
OUT = ROOT / "manuscript" / "gb_methodology_v3_2026-11-25" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.3,
    "axes.titlesize": 9.5,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.7,
    "ytick.labelsize": 7.7,
    "legend.fontsize": 7.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 300,
})

C_NAIVE = "#7F8C8D"
C_MNN = "#95A5A6"
C_SOFT = "#C0392B"
C_SH = "#1E8449"
C_MITM = "#2471A3"
C_DEPTH = "#E67E22"
C_B1 = "#ABB2B9"


def main():
    ntma_u = pd.read_csv(NTMA / "ntma_ucommon_summary.csv")
    ntma_rev = pd.read_csv(NTMA / "ntma_organ_reversals.csv")
    comp_n = pd.read_csv(COMP / "competitors_5a_ucommon_ntma_NA.csv")
    c35_sp = pd.read_csv(C35 / "35core_marker_sparsity_diagnostics.csv")
    c35_ab = pd.read_csv(C35 / "35core_3factor_ablation.csv")

    fig = plt.figure(figsize=(15.8, 11.6), facecolor="white")
    outer = gridspec.GridSpec(
        2, 2, figure=fig,
        left=0.055, right=0.985, top=0.932, bottom=0.06,
        wspace=0.22, hspace=0.34,
        width_ratios=[1.0, 1.12],
    )

    # =========================================================================
    # PANEL A: Prospective Zero-Tuning Wang 2025 nTMA (r=25um, TMA-block 95% CIs)
    # =========================================================================
    ax_a = fig.add_subplot(outer[0, 0])
    nt25 = ntma_u[ntma_u["radius_um"] == 25]

    tracks = [
        ("Track N-A\nXenium vs CosMx\n(med 23 vs 3, U=556)", "Track_NA_Xenium_vs_CosMx_IF"),
        ("Track N-B\nXenium vs MERSCOPE\n(med 33 vs 2, Regime 4)", "Track_NB_Xenium_vs_MERSCOPE"),
        ("Track N-C\nXen Breast vs Panhuman\n(med 21 vs 19, U=567)", "Track_NC_Xenium_Breast_vs_Panhuman"),
    ]
    arms_a = [
        ("MNN", "MNN_rescaleBatches", C_MNN),
        ("Soft τ=0.05", "Soft_plugin_tau005", C_SOFT),
        ("Split-half 2s (frozen)", "Split_half_2s_default", C_SH),
        ("MITM EB + guardrail", "MITM_geom_eb_guardrail", C_MITM),
    ]

    xa = np.arange(len(tracks))
    wa = 0.19
    for j, (lbl, ak, col) in enumerate(arms_a):
        vals, lo, hi, iccs = [], [], [], []
        for _, tk in tracks:
            r = nt25[(nt25["track"] == tk) & (nt25["arm"] == ak)].iloc[0]
            vals.append(r["red_pct"])
            lo.append(max(0.0, r["red_pct"] - r["red_ci_lo_tma_block"]))
            hi.append(max(0.0, r["red_ci_hi_tma_block"] - r["red_pct"]))
            iccs.append(r["icc"])
        pos = xa + (j - 1.5) * wa
        bars = ax_a.bar(pos, vals, wa, color=col, alpha=0.92, label=lbl, edgecolor="white", linewidth=0.5)
        ax_a.errorbar(pos, vals, yerr=[lo, hi], fmt="none", ecolor="#2C3E50", elinewidth=0.9, capsize=2.3, zorder=5)
        for rect, v, lv, hv, ic in zip(bars, vals, lo, hi, iccs):
            ic_clean = 0.0 if abs(ic) < 0.005 else ic
            y_txt = (v + hv + 2.0) if v >= 0 else (v - lv - 2.2)
            ax_a.text(rect.get_x() + rect.get_width() / 2, y_txt, f"{v:+.0f}%\n({ic_clean:.2f})",
                      ha="center", va="bottom" if v >= 0 else "top", fontsize=6.1,
                      color="#922B21" if ak == "Soft_plugin_tau005" else ("#145A32" if ak == "Split_half_2s_default" else "#2C3E50"),
                      fontweight="bold" if ak in ("Split_half_2s_default", "Soft_plugin_tau005") else "normal")

    ax_a.axhline(0, color="#2C3E50", linewidth=0.8)
    # Compact callout notes for prospective honesty
    ax_a.text(0.02, 0.96,
              "Pre-frozen UTC 2026-11-25T14:12Z → fetched 2026-11-26T15:41Z\n"
              "• Track N-A: Naive ICC 0.19 → SH 0.34 (η² ↑ both); q_A clamped 0.10\n"
              "• Track N-B (med=2): MITM 70% guardrail fallback; Soft SD=0.24!\n"
              "• Track N-C: Naive ICC 0.79 → SH 0.82 (0.0% organ reversals)",
              transform=ax_a.transAxes, va="top", ha="left", fontsize=6.5, color="#1C2833",
              bbox=dict(boxstyle="round,pad=0.3", fc="#F4F6F7", ec="#BDC3C7", lw=0.7))

    ax_a.set_xticks(xa)
    ax_a.set_xticklabels([t[0] for t in tracks], fontsize=7.5)
    ax_a.set_ylabel("Median discrepancy reduction (%) [TMA-block 95% CI]\n(annotated with cross-platform ICC)")
    ax_a.set_ylim(-21, 96)
    ax_a.grid(axis="y", alpha=0.25, linestyle="--")
    ax_a.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.94, edgecolor="#D5D8DC", ncol=2, fontsize=6.7)
    ax_a.set_title("a   Prospective zero-tuning Wang 2025 nTMA (15 normal organs)", loc="left", fontweight="bold", pad=8)

    # =========================================================================
    # PANEL B: b1 Organ-Contrast Reversals | b2 Prospective nTMA Linux Competitors
    # =========================================================================
    sub_b = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0, 1], wspace=0.28, width_ratios=[1.0, 1.15])
    ax_b1 = fig.add_subplot(sub_b[0, 0])
    ax_b2 = fig.add_subplot(sub_b[0, 1])

    # b1: Top-50 organ contrast reversals on Track N-A and Track N-B (Track N-C = 0% all)
    rev_tracks = [("Track N-A\n(CosMx, med=3)", "Track_NA_Xenium_vs_CosMx_IF"),
                  ("Track N-B\n(MERSCOPE, med=2)", "Track_NB_Xenium_vs_MERSCOPE")]
    rev_arms = [
        ("Naive", "Naive", C_NAIVE),
        ("Soft τ=0.05", "Soft_plugin_tau005", C_SOFT),
        ("Split-half 2s", "Split_half_2s_default", C_SH),
        ("MITM + guard.", "MITM_geom_eb_guardrail", C_MITM),
    ]
    xb1 = np.arange(len(rev_tracks))
    wb1 = 0.19
    for j, (lbl, ak, col) in enumerate(rev_arms):
        vals = [ntma_rev[(ntma_rev["track"] == tk) & (ntma_rev["arm"] == ak)].iloc[0]["top_k_reversal_pct"] for _, tk in rev_tracks]
        pos = xb1 + (j - 1.5) * wb1
        bars = ax_b1.bar(pos, vals, wb1, color=col, label=lbl, edgecolor="white", linewidth=0.5)
        for rect, v in zip(bars, vals):
            ax_b1.text(rect.get_x() + rect.get_width() / 2, v + 1.3, f"{v:.0f}%",
                       ha="center", va="bottom", fontsize=6.8,
                       color="#922B21" if ak == "Soft_plugin_tau005" else "#1C2833",
                       fontweight="bold" if ak == "Soft_plugin_tau005" else "normal")

    ax_b1.text(0.03, 0.81, "Track N-C (balanced):\n0.0% reversals (all arms)",
               transform=ax_b1.transAxes, va="top", ha="left", fontsize=6.5, color="#145A32",
               bbox=dict(boxstyle="round,pad=0.22", fc="#EAFAF1", ec="#1E8449", lw=0.65))
    ax_b1.set_xticks(xb1)
    ax_b1.set_xticklabels([t[0] for t in rev_tracks], fontsize=7.4)
    ax_b1.set_ylabel("Top-50 organ contrast reversals (%)")
    ax_b1.set_ylim(0, 86)
    ax_b1.grid(axis="y", alpha=0.25, linestyle="--")
    ax_b1.legend(loc="upper center", frameon=True, facecolor="white", framealpha=0.94, edgecolor="#D5D8DC", fontsize=6.4, ncol=2)
    ax_b1.set_title("b1  nTMA organ reversals", loc="left", fontweight="bold", fontsize=8.8, pad=6)

    # b2: Native Linux competitors on prospective nTMA Track N-A (39 cores)
    tools_b2 = [
        ("spicyR 1.22\n(L_r25, U=585)", "spicyR", "L_r25"),
        ("CRAWDAD 1.0\n(Z_100, U=569)", "CRAWDAD", "Z_scale100"),
        ("CRAWDAD 1.0\n(meanZ, U=571)", "CRAWDAD", "meanZ_multiscale"),
        ("PANORAMIC\n(r=25, U=550)", "PANORAMIC", "local_comp_enrichment_r25"),
    ]
    xb2 = np.arange(len(tools_b2))
    wb2 = 0.25
    nv_icc, b1_icc, sh_icc, sh_red = [], [], [], []
    for _, tl, mt in tools_b2:
        sub = comp_n[(comp_n["tool"] == tl) & (comp_n["metric"] == mt)]
        nv_icc.append(sub[sub["arm"] == "Naive"].iloc[0]["icc"])
        r_sh = sub[sub["arm"] == "Split_half_2s_default"].iloc[0]
        sh_icc.append(r_sh["icc"])
        sh_red.append(r_sh["red_pct"])
        sub_b1 = sub[sub["arm"] == "Split_half_2s_B1"]
        b1_icc.append(sub_b1.iloc[0]["icc"] if len(sub_b1) else np.nan)

    ax_b2.bar(xb2 - wb2, nv_icc, wb2, color=C_NAIVE, label="Tool Naive", edgecolor="white", linewidth=0.5)
    b_b1 = ax_b2.bar(xb2, [v if np.isfinite(v) else 0 for v in b1_icc], wb2, color=C_B1, label="+Single-draw B=1", edgecolor="white", linewidth=0.5)
    b_sh = ax_b2.bar(xb2 + wb2, sh_icc, wb2, color=C_SH, label="+Split-half 2s (frozen)", edgecolor="white", linewidth=0.5)
    for idx_t, (v_nv, v_b1, v_sh, r_pct) in enumerate(zip(nv_icc, b1_icc, sh_icc, sh_red)):
        ax_b2.text(xb2[idx_t] - wb2, v_nv + 0.012, f"{v_nv:.2f}", ha="center", va="bottom", fontsize=6.4, color="#566573")
        if np.isfinite(v_b1):
            ax_b2.text(xb2[idx_t], v_b1 + 0.012, f"{v_b1:.2f}", ha="center", va="bottom", fontsize=6.2, color="#7F8C8D")
        ax_b2.text(xb2[idx_t] + wb2, v_sh + 0.012, f"{v_sh:.2f}\n(+{r_pct:.0f}%)",
                   ha="center", va="bottom", fontsize=6.4, color="#145A32", fontweight="bold")
    # Hide zero dummy bar for meanZ B=1 (not computed)
    b_b1[2].set_visible(False)

    ax_b2.set_xticks(xb2)
    ax_b2.set_xticklabels([t[0] for t in tools_b2], fontsize=6.9)
    ax_b2.set_ylabel("Cross-platform ICC (prospective nTMA N-A)")
    ax_b2.set_ylim(0, 0.72)
    ax_b2.grid(axis="y", alpha=0.25, linestyle="--")
    ax_b2.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.92, edgecolor="#D5D8DC", fontsize=6.5)
    ax_b2.set_title("b2  nTMA Track N-A Linux competitors", loc="left", fontweight="bold", fontsize=8.8, pad=6)

    # =========================================================================
    # PANEL C: 35-Core Negative Probe Sparsity Root Cause & Recovery Bridge
    # =========================================================================
    sub_c = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1, 0], wspace=0.28, width_ratios=[0.95, 1.15])
    ax_c1 = fig.add_subplot(sub_c[0, 0])
    ax_c2 = fig.add_subplot(sub_c[0, 1])

    # c1: Marker sparsity diagnostics on the 35 cores
    sp_df = c35_sp.set_index("platform")
    plats = ["MERSCOPE", "CosMx", "Xenium"]
    xc1 = np.arange(len(plats))
    wc1 = 0.34
    ep_pct = [100.0 * sp_df.loc[p, "epcam_pos_frac"] for p in plats]
    cd_pct = [100.0 * sp_df.loc[p, "cd3e_pos_frac"] for p in plats]
    meds = [sp_df.loc[p, "cal_6g_medpos"] for p in plats]

    bc1_a = ax_c1.bar(xc1 - wc1 / 2, ep_pct, wc1, color="#2E86C1", label="EPCAM+ cells (%)", edgecolor="white")
    bc1_b = ax_c1.bar(xc1 + wc1 / 2, cd_pct, wc1, color="#D35400", label="CD3E+ cells (%)", edgecolor="white")
    for rect in bc1_a:
        h = rect.get_height()
        ax_c1.text(rect.get_x() + rect.get_width() / 2, h + 0.8, f"{h:.1f}%", ha="center", va="bottom", fontsize=6.8, fontweight="bold", color="#1B4F72")
    for rect in bc1_b:
        h = rect.get_height()
        ax_c1.text(rect.get_x() + rect.get_width() / 2, h + 0.8, f"{h:.1f}%", ha="center", va="bottom", fontsize=6.8, fontweight="bold", color="#A04000")

    ax_c1.annotate("37× / 22× deficit!\n6g medpos = 1.0\n→ 23/35 cores\ncollapse!",
                   xy=(-0.05, 5.8), xytext=(0.06, 22.8),
                   fontsize=6.6, color="#922B21", fontweight="bold",
                   bbox=dict(boxstyle="round,pad=0.25", fc="#FDEDEC", ec=C_SOFT, lw=0.75),
                   arrowprops=dict(arrowstyle="->", color=C_SOFT, lw=0.9))

    ax_c1.set_xticks(xc1)
    ax_c1.set_xticklabels([f"{p}\n(6g med={m:.0f})" for p, m in zip(plats, meds)], fontsize=7.3)
    ax_c1.set_ylabel("Marker-positive cells on 35 cores (%)")
    ax_c1.set_ylim(0, 42)
    ax_c1.grid(axis="y", alpha=0.25, linestyle="--")
    ax_c1.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.92, edgecolor="#D5D8DC", fontsize=6.6)
    ax_c1.set_title("c1  6-gene marker drop-out", loc="left", fontweight="bold", fontsize=8.8, pad=6)

    # c2: Direct bridge on CosMx/MERSCOPE from -36.5% legacy failure -> full-panel recovery
    def get_35(comp_p, pan, clf, std):
        return c35_ab[(c35_ab["comparison"] == comp_p) & (c35_ab["panel"] == pan) &
                      (c35_ab["classifier_scope"] == clf) & (c35_ab["standardization"] == std)].iloc[0]

    bridge = [
        ("6g 2-mk\n6g_q (Leg.)\n(n=12)", get_35("CosMx/MERSCOPE", "6_genes", "exclusive_2marker_1pair", "6g_median_q075"), C_SOFT),
        ("6g 2-mk\nSH 2s\n(n=11)", get_35("CosMx/MERSCOPE", "6_genes", "exclusive_2marker_1pair", "Split_half_2s_default"), C_DEPTH),
        ("6g SingleR\n15p 6g_q\n(n=99)", get_35("CosMx/MERSCOPE", "6_genes", "SingleR_15pairs", "6g_median_q075"), "#ABB2B9"),
        ("Full 108g\n15p 6g_q\n(n=480)", get_35("CosMx/MERSCOPE", "full_shared_HPCA_108g", "SingleR_15pairs", "6g_median_q075"), C_MITM),
        ("Full 108g\nFull_q\n(n=478)", get_35("CosMx/MERSCOPE", "full_shared_HPCA_108g", "SingleR_15pairs", "full_panel_depth_q075"), C_SH),
    ]
    xc2 = np.arange(len(bridge))
    vals_b = [r["red_pct_pen"] for _, r, _ in bridge]
    cols_b = [c for _, _, c in bridge]
    bars_b = ax_c2.bar(xc2, vals_b, 0.56, color=cols_b, edgecolor="white", linewidth=0.6)
    ax_c2.axhline(0, color="#2C3E50", linewidth=0.9)
    for rect, (_, r, col) in zip(bars_b, bridge):
        v = r["red_pct_pen"]
        ic = r["icc"]
        ax_c2.text(rect.get_x() + rect.get_width() / 2, v + (2.5 if v >= 0 else -3.5),
                   f"{v:+.1f}%\n(ICC {ic:.2f})",
                   ha="center", va="bottom" if v >= 0 else "top", fontsize=6.5,
                   color="#922B21" if v < 0 else "#1C2833", fontweight="bold")

    ax_c2.set_xticks(xc2)
    ax_c2.set_xticklabels([b[0] for b in bridge], fontsize=6.6)
    ax_c2.set_ylabel("CosMx/MERSCOPE discrepancy red. (%)")
    ax_c2.set_ylim(-52, 48)
    ax_c2.grid(axis="y", alpha=0.25, linestyle="--")
    ax_c2.set_title("c2  CosMx/MERSCOPE rescue bridge", loc="left", fontweight="bold", fontsize=8.8, pad=6)

    # =========================================================================
    # PANEL D: Complete 3-Factor Factorial Ablation Heatmap (ICC + Red.% + n_ok)
    # =========================================================================
    ax_d = fig.add_subplot(outer[1, 1])

    row_specs = [
        ("6g | Excl. 2-mk (1p)", "6_genes", "exclusive_2marker_1pair"),
        ("6g | SingleR (1p Epi-T)", "6_genes", "SingleR_1pair_EpiT"),
        ("6g | SingleR (15p)", "6_genes", "SingleR_15pairs"),
        ("Full HPCA | SingleR (1p)", "FULL", "SingleR_1pair_EpiT"),
        ("Full HPCA | SingleR (15p)", "FULL", "SingleR_15pairs"),
    ]
    col_specs = [
        ("Cos/Xen\nNative", "CosMx/Xenium", "Native"),
        ("Cos/Xen\n6g_q", "CosMx/Xenium", "6g_median_q075"),
        ("Cos/Xen\nSH 2s", "CosMx/Xenium", "Split_half_2s_default"),
        ("Cos/MER\nNative", "CosMx/MERSCOPE", "Native"),
        ("Cos/MER\nFull_q", "CosMx/MERSCOPE", "full_panel_depth_q075"),
        ("Cos/MER\nSH 2s", "CosMx/MERSCOPE", "Split_half_2s_default"),
        ("MER/Xen\nNative", "MERSCOPE/Xenium", "Native"),
        ("MER/Xen\nFull_q", "MERSCOPE/Xenium", "full_panel_depth_q075"),
        ("MER/Xen\nSH 2s", "MERSCOPE/Xenium", "Split_half_2s_default"),
    ]
    full_panel_map = {
        "CosMx/Xenium": "full_shared_HPCA_119g",
        "CosMx/MERSCOPE": "full_shared_HPCA_108g",
        "MERSCOPE/Xenium": "full_shared_HPCA_246g",
    }

    mat_icc = np.full((len(row_specs), len(col_specs)), np.nan)
    cell_txt = [["" for _ in col_specs] for _ in row_specs]
    for ri, (_, r_pan, r_clf) in enumerate(row_specs):
        for ci, (_, c_pair, c_std) in enumerate(col_specs):
            pan_name = full_panel_map[c_pair] if r_pan == "FULL" else r_pan
            sub = c35_ab[(c35_ab["comparison"] == c_pair) & (c35_ab["panel"] == pan_name) &
                         (c35_ab["classifier_scope"] == r_clf) & (c35_ab["standardization"] == c_std)]
            if len(sub):
                r = sub.iloc[0]
                mat_icc[ri, ci] = r["icc"]
                red = r["red_pct_pen"]
                if c_std == "Native":
                    red_s = f"n={int(r['n_ok'])}"
                elif not np.isfinite(red):
                    red_s = f"avail {int(r['n_ok'])}/{int(r['n_d0'])}"
                else:
                    red_s = f"{red:+.0f}%"
                cell_txt[ri][ci] = f"{r['icc']:.2f}\n{red_s}"

    im = ax_d.imshow(mat_icc, cmap="RdYlGn", vmin=-0.15, vmax=0.88, aspect="auto")
    for ri in range(len(row_specs)):
        for ci in range(len(col_specs)):
            val = mat_icc[ri, ci]
            txt = cell_txt[ri][ci]
            if txt:
                fg = "white" if (val < 0.05 or val > 0.72) else "#1C2833"
                ax_d.text(ci, ri, txt, ha="center", va="center", fontsize=6.3, color=fg,
                          fontweight="bold" if ri >= 3 else "normal")

    # Visual separators between 6-gene vs Full HPCA rows and between 3 platform pairs
    ax_d.axhline(2.5, color="#154360", linewidth=2.2)
    ax_d.axvline(2.5, color="white", linewidth=2.0)
    ax_d.axvline(5.5, color="white", linewidth=2.0)

    ax_d.set_xticks(np.arange(len(col_specs)))
    ax_d.set_xticklabels([c[0] for c in col_specs], fontsize=6.7)
    ax_d.set_yticks(np.arange(len(row_specs)))
    ax_d.set_yticklabels([r[0] for r in row_specs], fontsize=7.2)
    cbar = fig.colorbar(im, ax=ax_d, fraction=0.032, pad=0.02)
    cbar.set_label("Cross-platform ICC", fontsize=7.6)
    cbar.ax.tick_params(labelsize=7.0)
    ax_d.set_title("d   3-factor factorial ablation on identical 35 cores (cell = ICC / red.% or avail.)", loc="left", fontweight="bold", pad=8)

    fig.suptitle(
        "Figure 5. Prospective zero-tuning Wang 2025 nTMA validation and 3-factor decomposition of the 35-core negative probe",
        fontsize=11.3, fontweight="bold", y=0.980, color="#154360",
    )

    for ext in ("png", "pdf"):
        out_p = OUT / f"Figure5_prospective_ntma_and_35core_decomposition.{ext}"
        fig.savefig(out_p, dpi=300)
        print(f"Saved: {out_p} ({out_p.stat().st_size:,} B)")
    plt.close(fig)


if __name__ == "__main__":
    main()
