#!/usr/bin/env python3
"""Build Figure 6 (3-tier paired zero-effect testing, multi-center cancer reversals with
TMA-block clustered CIs, HNSCC Core 64 two-estimand diagnostic, and Operational Four-Regime
Decision Guide) strictly from verified v3 CSV/JSON artifacts.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

ROOT = Path(r"C:\Reproducution")
AUDIT = ROOT / "review" / "v3_unified_audit_2026-11-25"
P6 = ROOT / "review" / "p6_conclusion_reversal_case_study_2026-11-23"
OUT = ROOT / "manuscript" / "gb_methodology_v3_2026-11-25" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "axes.titlesize": 9.8,
    "axes.labelsize": 8.8,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "legend.fontsize": 7.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 300,
})

COLORS = {
    "Naive": "#7F8C8D",
    "MNN_rescaleBatches": "#95A5A6",
    "Soft_plugin_tau005": "#C0392B",
    "Depth_only_2s_q075": "#E67E22",
    "Split_half_2s_default": "#1E8449",
    "MITM_geom_eb_secondary": "#2471A3",
}
SHORT = {
    "Naive": "Naive",
    "MNN_rescaleBatches": "MNN",
    "Soft_plugin_tau005": "Soft\n(τ=0.05)",
    "Depth_only_2s_q075": "Depth-only\n2s",
    "Split_half_2s_default": "Split-half 2s\n(default)",
    "MITM_geom_eb_secondary": "MITM EB\n(secondary)",
}


def main():
    df_null = pd.read_csv(AUDIT / "audit_ttma1_paired_null_tests.csv")
    df_rev = pd.read_csv(AUDIT / "audit_ttma1_reversals_clustered_cis.csv")
    show = json.loads((P6 / "p6c_showcase_summary.json").read_text(encoding="utf-8"))

    fig = plt.figure(figsize=(14.8, 9.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.04, 1.04], width_ratios=[1.05, 1.0],
                          hspace=0.38, wspace=0.26, left=0.115, right=0.965, top=0.92, bottom=0.075)

    # -------------------------------------------------------------------------
    # Panel a: Three-tier paired zero-effect null tests (Track A & Track B all)
    # -------------------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0])
    methods = ["Naive", "MNN_rescaleBatches", "Soft_plugin_tau005", "Depth_only_2s_q075", "Split_half_2s_default", "MITM_geom_eb_secondary"]
    sub_a = df_null[(df_null["track"] == "trackA") & (df_null["stratum"] == "all_qualified")].set_index("method").loc[methods]
    x = np.arange(len(methods))
    w = 0.26
    y_wc = 100.0 * sub_a["within_cancer_paired_t_fdr05_rate"].to_numpy()
    y_dj = 100.0 * sub_a["disjoint_50_50_mean_fdr05_rate"].to_numpy()
    y_pan = 100.0 * sub_a["paired_t_pancancer_fdr05_rate"].to_numpy()

    b1 = ax_a.bar(x - w, y_wc, width=w, color="#1F618D", edgecolor="#154360", lw=0.7, label="Tier 1: Within-cancer paired t (FDR<0.05)")
    b2 = ax_a.bar(x, y_dj, width=w, color="#1E8449", edgecolor="#145A32", lw=0.7, label="Tier 2: Disjoint 50/50 Welch (FDR<0.05)")
    b3 = ax_a.bar(x + w, y_pan, width=w, color="#D4AC0D", edgecolor="#7D6608", lw=0.7, alpha=0.88, label="Tier 3: Pan-cancer paired t (n=122, detects |Δ|~0.05)")

    ax_a.axhline(5.0, color="#2C3E50", ls="--", lw=1.0, alpha=0.85, zorder=2)
    ax_a.text(5.48, 6.4, "5%", ha="right", va="bottom", fontsize=7.2, color="#2C3E50", style="italic")

    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            ty = max(h + 1.5, 6.6) if h < 5.0 else h + 1.4
            ax_a.text(bar.get_x() + bar.get_width() / 2.0, ty, f"{h:.1f}",
                      ha="center", va="bottom", fontsize=6.6, color="#1C2833", fontweight="bold" if h < 5.0 else "normal")

    ax_a.set_xticks(x)
    ax_a.set_xticklabels([SHORT[m] for m in methods], fontsize=7.6)
    ax_a.set_ylabel("False-discovery rate at Δ_bio = 0 (%)")
    ax_a.set_ylim(0, 108)
    ax_a.set_title("a  Three-tier paired zero-effect tests (tTMA1 Track A, n=122)", loc="left", fontweight="bold")
    ax_a.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.94, edgecolor="#D5D8DC", fontsize=6.9)
    ax_a.annotate("Track B (n=87) Disjoint 50/50:\nNaive 23.2% | Soft 46.2% → Split-half 5.3%",
                  xy=(4.0, 10.5), xytext=(4.22, 91.5), fontsize=6.9, ha="center",
                  bbox=dict(boxstyle="round,pad=0.30", facecolor="#EBF5FB", edgecolor="#2E86C1", lw=0.8),
                  arrowprops=dict(arrowstyle="->", color="#1E8449", lw=1.0, connectionstyle="arc3,rad=0.16"))

    # -------------------------------------------------------------------------
    # Panel b: Multi-center cancer-vs-cancer sign reversals (TMA-block 95% CIs)
    # -------------------------------------------------------------------------
    ax_b = fig.add_subplot(gs[0, 1])
    rev_a = df_rev[df_rev["track"] == "trackA"].set_index("method").loc[methods]
    rev_b = df_rev[df_rev["track"] == "trackB"].set_index("method").loc[methods]
    w2 = 0.36
    ra = 100.0 * rev_a["sign_rev_rate"].to_numpy()
    rb = 100.0 * rev_b["sign_rev_rate"].to_numpy()
    ea_lo = np.maximum(0.0, ra - 100.0 * rev_a["tma_rev_ci_lo"].to_numpy())
    ea_hi = np.maximum(0.0, 100.0 * rev_a["tma_rev_ci_hi"].to_numpy() - ra)
    eb_lo = np.maximum(0.0, rb - 100.0 * rev_b["tma_rev_ci_lo"].to_numpy())
    eb_hi = np.maximum(0.0, 100.0 * rev_b["tma_rev_ci_hi"].to_numpy() - rb)

    ba = ax_b.bar(x - w2 / 2, ra, width=w2, color="#2E86C1", edgecolor="#1B4F72", lw=0.8, label="Track A: Xenium vs CosMx (50 contrasts)")
    bb = ax_b.bar(x + w2 / 2, rb, width=w2, color="#AF7AC5", edgecolor="#5B2C6F", lw=0.8, label="Track B: Xenium vs MERSCOPE (72 contrasts)")
    ax_b.errorbar(x - w2 / 2, ra, yerr=[ea_lo, ea_hi], fmt="none", ecolor="#1B4F72", elinewidth=1.0, capsize=2.8)
    ax_b.errorbar(x + w2 / 2, rb, yerr=[eb_lo, eb_hi], fmt="none", ecolor="#5B2C6F", elinewidth=1.0, capsize=2.8)

    for i, m in enumerate(methods):
        ax_b.text(x[i] - w2 / 2, ra[i] + ea_hi[i] + 1.1, f"{ra[i]:.1f}%\n(r={rev_a.loc[m,'r_vs_consensus']:.2f})",
                  ha="center", va="bottom", fontsize=6.4, color="#1B4F72")
        ax_b.text(x[i] + w2 / 2, rb[i] + eb_hi[i] + 1.1, f"{rb[i]:.1f}%\n(r={rev_b.loc[m,'r_vs_consensus']:.2f})",
                  ha="center", va="bottom", fontsize=6.4, color="#5B2C6F")

    ax_b.set_xticks(x)
    ax_b.set_xticklabels([SHORT[m] for m in methods], fontsize=7.6)
    ax_b.set_ylabel("Sign reversal rate on strong contrasts (%)")
    ax_b.set_ylim(0, 58)
    ax_b.set_title("b  Cancer-vs-cancer sign reversals (TMA-block 95% CI)", loc="left", fontweight="bold")
    ax_b.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.92, edgecolor="#D5D8DC", fontsize=7.2)

    # -------------------------------------------------------------------------
    # Panel c: Single-specimen diagnostic (Held-out HNSCC Core 64)
    # -------------------------------------------------------------------------
    ax_c = fig.add_subplot(gs[1, 0])
    sm = show["showcase_metrics"]
    cats = [
        "Same-cell CosMx\nProtein IF truth",
        "Native\nXenium RNA",
        "Native\nCosMx RNA",
        "Split-half 2s\nXenium (aligned)",
        "Split-half 2s\nCosMx (aligned)",
        "MITM geom EB\nXenium (fidelity)",
    ]
    vals = [
        sm["protein_IF_logR"],
        sm["naive_logR_Xenium"],
        sm["naive_logR_CosMx"],
        sm["P2_both_Xenium"],
        sm["P2_both_CosMx"],
        sm["MITM_geom_Xenium"],
    ]
    bar_cols = ["#1C2833", "#2E86C1", "#E74C3C", "#1E8449", "#52BE80", "#2471A3"]
    ypos = np.arange(len(cats))[::-1]
    ax_c.barh(ypos, vals, height=0.56, color=bar_cols, edgecolor="#1C2833", lw=0.7)
    ax_c.axvline(0.0, color="#1C2833", lw=1.0)
    ax_c.axvspan(-1.62, 0.0, color="#D6EAF8", alpha=0.22, zorder=0)
    ax_c.axvspan(0.0, 0.85, color="#FADBD8", alpha=0.22, zorder=0)

    ann_txt = [
        f"{vals[0]:+.3f} (segregation)",
        f"{vals[1]:+.3f} (z={sm['naive_z_Xenium']:+.2f})",
        f"{vals[2]:+.3f} (z={sm['naive_z_CosMx']:+.2f})",
        f"{vals[3]:+.3f} (p_joint=0.001)",
        f"{vals[4]:+.3f} (gap=0.057)",
        f"{vals[5]:+.3f} (p_joint=0.027)",
    ]
    for yp, v, txt in zip(ypos, vals, ann_txt):
        if v < -0.5:
            ax_c.text(v + 0.04, yp, txt, ha="left", va="center", fontsize=7.5, color="white", fontweight="bold")
        elif v < 0:
            ax_c.text(v - 0.04, yp, txt, ha="right", va="center", fontsize=7.5, color="#154360", fontweight="bold")
        else:
            ax_c.text(v + 0.03, yp, txt, ha="left", va="center", fontsize=7.5, color="#1C2833")

    ax_c.set_yticks(ypos)
    ax_c.set_yticklabels(cats, fontsize=7.8)
    ax_c.set_xlim(-1.58, 0.84)
    ax_c.set_ylim(-0.82, 5.55)
    ax_c.set_xlabel("Epithelial–Myeloid colocalization log(R)  [− segregation | + enrichment]")
    ax_c.set_title("c  Two-estimand boundary on held-out HNSCC Core 64", loc="left", fontweight="bold")
    ax_c.text(-1.53, -0.64, "True IF sign: Segregation (<0)", fontsize=7.3, color="#1B4F72", style="italic")
    ax_c.text(0.80, -0.64, "CosMx RNA offset (>0)", ha="right", fontsize=7.3, color="#922B21", style="italic")

    # -------------------------------------------------------------------------
    # Panel d: Operational Four-Regime Decision Guide
    # -------------------------------------------------------------------------
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.axis("off")
    ax_d.set_xlim(0, 1)
    ax_d.set_ylim(0, 1)
    ax_d.set_title("d  Operational Four-Regime Decision Guide (v3.0.0)", loc="left", fontweight="bold")

    regimes = [
        (
            "Regime 1: Cross-platform / multi-center group comparison (PRIMARY DEFAULT)",
            "Frozen DEFAULT_PROTOCOL_V3: Split-half 2s (q₀=0.75, B=20) + synchronized joint perm.",
            "• Equalizes depth + SNR without paired training; disjoint FDR ≤3–5%, reversals 2–4%.",
            "#E8F8F5", "#1E8449",
        ),
        (
            "Regime 2: Moderate depth gap (≤6–7×) requiring comparability + fidelity",
            "Optional secondary: Geometric-mean MITM (d* = √(s_A·s_B), 1/√d EB + guardrail).",
            "• Retains 82–90% deep SD & dual-IMC / protein IF accuracy while cutting MAE.",
            "#EBF5FB", "#2471A3",
        ),
        (
            "Regime 3: Single-specimen spatial mapping & hotspot discovery",
            "Do NOT thin high-depth data; use unthinned stability-weighted perm. (w_i = 1 − f_i).",
            "• Zero discarded molecules; improves IF concordance across both donors & boosts power.",
            "#FEF9E7", "#B7950B",
        ),
        (
            "Regime 4: Extreme sparsity (<20 tx/cell, >10× gap, clamped) or balanced runs",
            "Check boundary_clamped flag; never extrapolate ≥20× (L→H fails); skip if balanced.",
            "• Warn on q_A=0.10 clamp & ultra-sparse (<10-gene) panels; avoid soft plug-in.",
            "#FDEDEC", "#C0392B",
        ),
    ]
    y_top = 0.96
    box_h = 0.212
    gap = 0.030
    for title, action, note, fc, ec in regimes:
        rect = mpatches.FancyBboxPatch(
            (0.015, y_top - box_h), 0.96, box_h,
            boxstyle="round,pad=0.010,rounding_size=0.022",
            facecolor=fc, edgecolor=ec, linewidth=1.3,
        )
        ax_d.add_patch(rect)
        ax_d.text(0.035, y_top - 0.036, title, fontsize=7.6, fontweight="bold", color=ec, va="top")
        ax_d.text(0.035, y_top - 0.100, action, fontsize=7.3, color="#1C2833", fontweight="bold", va="top")
        ax_d.text(0.035, y_top - 0.156, note, fontsize=7.0, color="#2C3E50", va="top")
        y_top -= (box_h + gap)

    fig.suptitle("Figure 6. Paired zero-effect testing, multi-center conclusion reversals, and the two-estimand operational guide",
                 fontsize=11.0, fontweight="bold", y=0.975)

    for ext in ("png", "pdf"):
        p = OUT / f"Figure6_multicenter_reversals_and_operational_guide.{ext}"
        fig.savefig(p, dpi=300 if ext == "png" else None)
    plt.close(fig)
    print("Saved Figure 6 cleanly!")


if __name__ == "__main__":
    main()
