#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ALDH7A1 NIPD (RHDO) - phân tích & vẽ hình, viết lại theo bản thảo.

LƯU Ý QUAN TRỌNG
----------------
Code này được dựng lại từ phần Methods/Results/Figure legends của bản thảo.
Dữ liệu thô (số read từng SNP, FF từng ca...) không có trong file, nên:
  * Phần A (mô hình RHDO, FF, tái tổ hợp, bố, tích hợp, thống kê) là code phân tích
    thật, dùng được ngay cho dữ liệu của bạn.
  * Phần B tạo dữ liệu MÔ PHỎNG khớp các con số trong bài (khoảng FF, số SNP,
    dosage...) CHỈ để vẽ lại hình. Khi có Supplementary Table 2 thật, thay bằng
    cách đọc CSV (xem load_real_data()).

Chạy:  python aldh7a1_nipd_analysis.py [--outdir out]
Yêu cầu: numpy, scipy, pandas, matplotlib
"""
import argparse
import os
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import logsumexp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, PathPatch
from matplotlib.path import Path

# ----------------------------------------------------------------------------
# Ngưỡng reportability (Methods)
# ----------------------------------------------------------------------------
FF_MIN = 4.0          # %
SNP_MIN = 25          # số SNP informative
POST_MIN = 0.98       # posterior confidence
PHASE_MIN = 0.95      # phasing confidence
DEPTH_MIN = 30        # cfDNA depth tối thiểu / SNP
IMP_RSQ_MIN = 0.82    # imputation Rsq
IMP_GP_MIN = 0.90     # genotype posterior
POST_RISK, POST_NONRISK = 0.90, 0.10

COL = {"Affected": "#c8372d", "Carrier": "#e08a14", "Unaffected": "#13803f"}


# ============================================================================
# PHẦN A. PHƯƠNG PHÁP PHÂN TÍCH
# ============================================================================
def estimate_fetal_fraction(B, A):
    """FF = 2*sum(B_i) / sum(A_i + B_i)
    B: số read allele đặc hiệu của bố; A: số read allele của mẹ, tại các SNP
    mẹ & bố đồng hợp tử khác allele."""
    B, A = np.asarray(B, float), np.asarray(A, float)
    return 2.0 * B.sum() / (A + B).sum()


def filter_imputed(rsq, gp):
    """Giữ genotype impute nếu Rsq >= 0.82 và posterior >= 0.90."""
    return (np.asarray(rsq) >= IMP_RSQ_MIN) & (np.asarray(gp) >= IMP_GP_MIN)


def maternal_rhdo_posterior(r, d, snp_type, ff, prior=0.5):
    """Xác suất hậu nghiệm thai nhận haplotype nguy cơ (risk) của mẹ.

    r, d      : số read allele risk và độ sâu tổng tại từng SNP
    snp_type  : 'a' (alpha: allele bố = allele risk mẹ) hoặc 'b' (beta: = non-risk)
    ff        : fetal fraction (0-1)
    Kỳ vọng tỉ lệ read risk:
        alpha: 0.5 + f/2 (thừa kế risk) | 0.5 (non-risk)
        beta : 0.5       (risk)         | 0.5 - f/2 (non-risk)
    r_i ~ Binomial(d_i, p_i); giả định độc lập có điều kiện giữa các SNP.
    """
    r, d = np.asarray(r), np.asarray(d)
    is_a = np.asarray(snp_type) == "a"
    p_risk = np.where(is_a, 0.5 + ff / 2, 0.5)
    p_non = np.where(is_a, 0.5, 0.5 - ff / 2)
    ll_risk = stats.binom.logpmf(r, d, p_risk).sum()
    ll_non = stats.binom.logpmf(r, d, p_non).sum()
    lp = np.log([prior, 1 - prior]) + np.array([ll_risk, ll_non])
    return float(np.exp(lp[0] - logsumexp(lp)))


def recombination_scan(r, d, snp_type, ff, pos, min_flank=3):
    """So sánh mô hình không chuyển (no-switch) với mô hình 1 điểm chuyển giữa
    hai SNP liên tiếp (theo thứ tự vị trí). Trả về (log LR tối đa, vị trí cắt).
    Dùng phân tích 2 chiều: tính log-lik cho (risk->non) và (non->risk)."""
    order = np.argsort(pos)
    r, d, st = np.asarray(r)[order], np.asarray(d)[order], np.asarray(snp_type)[order]
    is_a = st == "a"
    p_risk = np.where(is_a, 0.5 + ff / 2, 0.5)
    p_non = np.where(is_a, 0.5, 0.5 - ff / 2)
    l_risk = stats.binom.logpmf(r, d, p_risk)
    l_non = stats.binom.logpmf(r, d, p_non)
    n = len(r)
    ll0 = max(l_risk.sum(), l_non.sum())          # no switch
    best, best_k = -np.inf, None
    for k in range(min_flank, n - min_flank + 1):
        for left, right in ((l_risk, l_non), (l_non, l_risk)):   # bidirectional
            ll = left[:k].sum() + right[k:].sum()
            if ll - ll0 > best:
                best, best_k = ll - ll0, k
    return best, best_k


def paternal_transmission_posterior(alt, depth, ff, err, prior=0.5):
    """Mẹ đồng hợp tử, bố dị hợp: allele đặc hiệu của bố.
    Truyền (transmission): tỉ lệ ~ FF/2 ; Không truyền: tỉ lệ = nền lỗi `err`."""
    alt, depth = np.asarray(alt), np.asarray(depth)
    ll_t = stats.binom.logpmf(alt, depth, ff / 2).sum()
    ll_n = stats.binom.logpmf(alt, depth, np.broadcast_to(err, depth.shape)).sum()
    lp = np.log([prior, 1 - prior]) + np.array([ll_t, ll_n])
    return float(np.exp(lp[0] - logsumexp(lp)))


def call_direction(post):
    """Gọi hướng thừa kế mẹ: >=0.90 risk, <=0.10 non-risk, còn lại inconclusive."""
    if post >= POST_RISK:
        return "risk"
    if post <= POST_NONRISK:
        return "nonrisk"
    return "inconclusive"


def integrate_genotype(mat_call, pat_call):
    """Mô hình lặn NST thường: 2 haplotype bệnh = Affected; 1 = Carrier;
    0 = Unaffected; thiếu bằng chứng của bất kỳ phía nào = Inconclusive."""
    if "inconclusive" in (mat_call, pat_call):
        return "Inconclusive"
    n = (mat_call == "risk") + (pat_call == "risk")
    return {2: "Affected", 1: "Carrier", 0: "Unaffected"}[n]


def is_reportable(ff_pct, n_snp, posterior, depth_ok=True):
    """Quy tắc reportability duy nhất: đạt TẤT CẢ ngưỡng, nếu không -> no-call.
    posterior confidence = max(p, 1-p)."""
    conf = max(posterior, 1 - posterior)
    return (ff_pct >= FF_MIN) and (n_snp >= SNP_MIN) and (conf >= POST_MIN) and depth_ok


def clopper_pearson(k, n, alpha=0.05):
    """Khoảng tin cậy chính xác (exact binomial) 95%."""
    lo = 0.0 if k == 0 else stats.beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return lo, hi


def diagnostic_performance(truth, pred):
    """Độ nhạy/đặc hiệu/PPV cho 'Affected' (loại no-call trước khi gọi hàm này)."""
    truth, pred = np.asarray(truth), np.asarray(pred)
    tp = ((truth == "Affected") & (pred == "Affected")).sum()
    fn = ((truth == "Affected") & (pred != "Affected")).sum()
    tn = ((truth != "Affected") & (pred != "Affected")).sum()
    fp = ((truth != "Affected") & (pred == "Affected")).sum()
    out = {}
    for name, k, n in (("Sensitivity", tp, tp + fn), ("Specificity", tn, tn + fp),
                       ("PPV (affected)", tp, tp + fp)):
        lo, hi = clopper_pearson(int(k), int(n))
        out[name] = (k / n * 100, lo * 100, hi * 100)
    k = int((truth == pred).sum())
    lo, hi = clopper_pearson(k, len(truth))
    out["Concordance"] = (k / len(truth) * 100, lo * 100, hi * 100)
    return out


def simulate_dilution(ff_levels=(0.5, 2, 4, 6, 10, 15), reps=(4, 5, 5, 5, 5, 5), seed=1):
    """Mô phỏng thí nghiệm pha loãng FF (chỉ minh họa logic phân loại đúng /
    no-call). Thay bằng kết quả thực của Supplementary Table 3."""
    rng = np.random.default_rng(seed)
    rows = []
    for f, n in zip(ff_levels, reps):
        for i in range(n):
            snps = int(rng.integers(150, 260))
            d = rng.integers(200, 300, snps)
            p = 0.5 + f / 200
            r = rng.binomial(d, p)
            ll1 = stats.binom.logpmf(r, d, p).sum()
            ll0 = stats.binom.logpmf(r, d, 0.5).sum()
            post = 1 / (1 + np.exp(ll0 - ll1))
            ok = is_reportable(f, snps, post) and post >= POST_RISK
            rows.append(dict(FF=f, replicate=i + 1, correct=bool(ok)))
    return pd.DataFrame(rows)


# ============================================================================
# PHẦN B. DỮ LIỆU (mô phỏng khớp bản thảo) HOẶC ĐỌC DỮ LIỆU THẬT
# ============================================================================
AFFECTED = [4, 6, 8, 12, 14, 16]
UNAFFECTED = [2, 10]
CARRIER = [1, 3, 5, 7, 9, 11, 13, 15, 17]
NOCALL_8W = [1, 2, 5, 9, 10, 11, 16, 17]
REPORT_10W_NEW = [1, 5, 11, 17]       # reportable từ 10 tuần
REPORT_12W_NEW = [2, 9, 10, 16]       # reportable từ 12 tuần
JERKING = [6, 12, 14, 16, 9, 2]
VENTRICULOMEGALY = [12, 14, 16]
PACC = [14, 16]


def build_simulated_cohort(seed=7):
    """Dữ liệu 17 thai kỳ khớp các khoảng số liệu trong bài. KHÔNG phải số liệu thật."""
    rng = np.random.default_rng(seed)
    rows = []
    for fam in range(1, 18):
        geno = ("Affected" if fam in AFFECTED else
                "Unaffected" if fam in UNAFFECTED else "Carrier")
        nocall = fam in NOCALL_8W
        if nocall:
            ff8 = rng.uniform(2.6, 3.8)
            ff10 = rng.uniform(6.5, 9.4)
            ff12 = rng.uniform(10.9, 15.0)
            snp8 = int(rng.integers(24, 78))
            snp_final = int(rng.integers(125, 186))
        else:
            ff8 = rng.uniform(10.8, 14.2)
            ff10 = rng.uniform(13.4, 17.0)
            ff12 = rng.uniform(15.7, 19.2)
            snp8 = int(rng.integers(187, 247))
            snp_final = snp8
        base = {"Affected": 0.70, "Carrier": 0.565, "Unaffected": 0.27}[geno]
        sd = {"Affected": 0.02, "Carrier": 0.02, "Unaffected": 0.03}[geno]
        dos = base + rng.normal(0, sd, 3)
        first = 8 if fam not in NOCALL_8W else (10 if fam in REPORT_10W_NEW else 12)
        rows.append(dict(family=fam, genotype=geno, ff_8w=ff8, ff_10w=ff10, ff_12w=ff12,
                         snps_8w=snp8, snps_final=snp_final,
                         dos_8w=dos[0], dos_10w=dos[1], dos_12w=dos[2],
                         first_reportable_w=first, nocall_8w=nocall,
                         nipd_call=geno, reference=geno))
    df = pd.DataFrame(rows)
    # Các ca đại diện (Figure 3A–C)
    df.loc[df.family == 14, ["dos_8w", "dos_10w", "dos_12w", "ff_8w", "snps_8w"]] = [.72, .73, .71, 12.8, 238]
    df.loc[df.family == 15, ["dos_8w", "dos_10w", "dos_12w", "ff_8w", "snps_8w"]] = [.60, .61, .60, 11.5, 224]
    df.loc[df.family == 10, ["dos_8w", "dos_10w", "dos_12w", "ff_8w", "snps_8w"]] = [.25, .22, .25, 3.2, 24]
    return df


def load_real_data(path):
    """Đọc Supplementary Table 2 thật (CSV) với đúng các cột như build_simulated_cohort()."""
    return pd.read_csv(path)


# ============================================================================
# PHẦN C. VẼ HÌNH
# ============================================================================
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 10})


def _jitter(rng, n, w=0.08):
    return rng.uniform(-w, w, n)


def fig2(df, perf, outdir):
    rng = np.random.default_rng(3)
    fig, axs = plt.subplots(2, 2, figsize=(11, 8.5))
    axB, axC, axD, axE = axs.ravel()

    # B: informative SNPs
    groups = [("Reportable 8w\n(n=9)", df.loc[~df.nocall_8w, "snps_8w"], "#7fb8e6"),
              ("No-call 8w\n(n=8)", df.loc[df.nocall_8w, "snps_8w"], "#b8bcc4"),
              ("Reclassified\n(final; n=8)", df.loc[df.nocall_8w, "snps_final"], "#8fd0a0")]
    bp = axB.boxplot([g[1] for g in groups], widths=.5, patch_artist=True, showfliers=False)
    for patch, g in zip(bp["boxes"], groups):
        patch.set_facecolor(g[2]); patch.set_alpha(.6)
    for i, g in enumerate(groups, 1):
        axB.scatter(i + _jitter(rng, len(g[1])), g[1], s=14, color="#1f5fa8", zorder=3)
    axB.axhline(SNP_MIN, color="red", ls="-.", lw=1.5)
    axB.text(3.4, SNP_MIN + 6, "25-SNP threshold", color="red", ha="right", weight="bold", fontsize=9)
    axB.set_xticklabels([g[0] for g in groups]); axB.set_ylabel("Informative SNPs (n)")
    axB.set_title("Informative SNP yield by reportability", weight="bold")
    axB.text(-.18, 1.05, "B", transform=axB.transAxes, fontsize=20, weight="bold")

    # C: FF theo thai kỳ
    for xi, w in enumerate((8, 10, 12)):
        for flag, c, lab in ((False, "#1f77d4", "Reportable at 8w (n=9)"),
                             (True, "#ff7f0e", "No-call at 8w (n=8)")):
            v = df.loc[df.nocall_8w == flag, f"ff_{w}w"]
            axC.scatter(xi + (0.12 if flag else -0.12) + _jitter(rng, len(v), .05), v, s=16, color=c,
                        label=lab if xi == 0 else None)
    axC.axhline(FF_MIN, color="#1f77d4", ls="--", lw=1.3)
    axC.text(2.4, FF_MIN + .5, "4% reportability threshold", ha="right", fontsize=9)
    axC.set_xticks([0, 1, 2]); axC.set_xticklabels(["8w\n(n=17)", "10w\n(n=17)", "12w\n(n=17)"])
    axC.set_xlabel("Gestational age", weight="bold"); axC.set_ylabel("Fetal fraction (%)")
    axC.set_ylim(0, 20); axC.legend(frameon=False, loc="upper left", fontsize=8)
    axC.set_title("Fetal fraction across gestation", weight="bold")
    axC.text(-.18, 1.05, "C", transform=axC.transAxes, fontsize=20, weight="bold")

    # D: dosage theo kiểu gen
    for g in ("Affected", "Carrier", "Unaffected"):
        sub = df[df.genotype == g]
        Y = sub[["dos_8w", "dos_10w", "dos_12w"]].values
        for y in Y:
            axD.plot([0, 1, 2], y, color=COL[g], alpha=.25, lw=1)
        m = Y.mean(0)
        axD.plot([0, 1, 2], m, "-o", color=COL[g], lw=2.5, label=f"{g} (n = {len(sub)})")
        for x, v in zip((0, 1, 2), m):
            axD.text(x, v + .03, f"{v:.2f}", color=COL[g], ha="center", fontsize=8)
    axD.axhline(.5, color="grey", ls="--", lw=1, label="Reference = 0.5")
    axD.set_xticks([0, 1, 2]); axD.set_xticklabels(["8w", "10w", "12w"]); axD.set_ylim(0, 1)
    axD.set_xlabel("Gestational age", weight="bold"); axD.set_ylabel("Integrated disease-allele dosage")
    axD.legend(frameon=False, fontsize=8, loc="lower right")
    axD.set_title("Longitudinal integrated dosage\nby final genotype", weight="bold", fontsize=10)
    axD.text(-.18, 1.05, "D", transform=axD.transAxes, fontsize=20, weight="bold")

    # E: forest plot hiệu năng
    items = [("Sensitivity", perf["Sensitivity"], "#1f77d4"),
             ("Specificity", perf["Specificity"], "#2ca02c"),
             ("PPV (affected)", perf["PPV (affected)"], "#9b3fd0"),
             ("First-pass\nreportability", perf["First-pass"], "#ff7f0e")]
    for i, (n, (v, lo, hi), c) in enumerate(items[::-1]):
        axE.plot([lo, hi], [i, i], color=c, lw=5, solid_capstyle="round")
        axE.scatter(v, i, color=c, s=60, zorder=3)
        axE.text(v, i + .25, f"{v:.1f}% (95% CI {lo:.1f}–{hi:.1f})", ha="right" if v > 90 else "center", fontsize=8)
    axE.set_yticks(range(4)); axE.set_yticklabels([x[0] for x in items[::-1]])
    axE.set_ylim(-.6, 3.7); axE.set_xlim(0, 105); axE.set_xticks([0, 25, 50, 75, 100]); axE.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    axE.set_xlabel("Percentage (%)", weight="bold")
    axE.set_title("Diagnostic performance\nand reportability", weight="bold", fontsize=10)
    axE.text(-.42, 1.05, "E", transform=axE.transAxes, fontsize=20, weight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "Figure_2_BE.png"), dpi=300)
    plt.close(fig)


def fig2a_schematic(outdir):
    """Figure 2A: sơ đồ gene ALDH7A1 + 3 biến thể đại diện."""
    fig, ax = plt.subplots(figsize=(10, 5.2)); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 60)
    ax.text(50, 58, "ALDH7A1", ha="center", style="italic", weight="bold", fontsize=16, color="#1f5fa8")
    ax.text(50, 53, "NM_001182.5 · chr5:126,541,841–126,595,219 · minus strand · 18 exons",
            ha="center", fontsize=9, color="grey")
    ax.plot([4, 96], [46, 46], color="#444", lw=1.5)
    exons = [(10, "12–18", "#cfe0f6"), (30, "11", "#2a5fbf"), (47, "10", "#e5605c"),
             (63, "9", "#1f9a52"), (78, "8", "#2a5fbf"), (92, "1–7", "#cfe0f6")]
    for x, lab, c in exons:
        ax.add_patch(FancyBboxPatch((x - 5, 43), 10, 6, boxstyle="round,pad=0.2", fc=c, ec="#333"))
        ax.text(x, 46, lab, ha="center", va="center", weight="bold",
                color="white" if c in ("#2a5fbf", "#e5605c", "#1f9a52") else "#1f5fa8")
    ax.text(4, 51, "3′", weight="bold"); ax.text(96, 51, "5′", weight="bold", ha="right")
    # phóng to exon 10 & 9
    ax.add_patch(FancyBboxPatch((8, 24), 30, 8, boxstyle="round,pad=0.2", fc="#fbe3e2", ec="#e5605c"))
    ax.add_patch(FancyBboxPatch((58, 24), 30, 8, boxstyle="round,pad=0.2", fc="#dff1e5", ec="#1f9a52"))
    ax.text(23, 28, "Exon 10", ha="center", va="center", color="#e5605c", weight="bold", fontsize=12)
    ax.text(73, 28, "Exon 9", ha="center", va="center", color="#1f9a52", weight="bold", fontsize=12)
    ax.text(48, 29, "intron 9", ha="center", style="italic", color="grey")
    ax.annotate("", xy=(13, 36), xytext=(27, 36)); ax.text(20, 37.5, "", ha="center")
    ax.annotate("", xy=(18, 37), xytext=(65, 37), arrowprops=dict(arrowstyle="<->"))
    ax.text(41, 38.5, "125 nt (cDNA)", ha="center", fontsize=9)
    ax.annotate("", xy=(65, 34.5), xytext=(78, 34.5), arrowprops=dict(arrowstyle="<->"))
    ax.text(71.5, 35.5, "12 nt (cDNA)", ha="center", fontsize=8)
    boxes = [(18, "c.906T>A\np.(Cys302Ter)\nexon 10 · nonsense", "#e5605c"),
             (65, "c.781A>T\np.(Arg261Trp)\nexon 9 · missense", "#9b59b6"),
             (80, "c.769del\np.(Met257TrpfsTer29)\nexon 9 · frameshift", "#1f9a52")]
    for x, t, c in boxes:
        ax.plot([x, x], [24, 32], color=c, lw=2)
        ax.text(x, 10, t, ha="center", va="center", fontsize=8.5, weight="bold", color=c,
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=c))
    fig.savefig(os.path.join(outdir, "Figure_2A_schematic.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig3(df, outdir):
    rng = np.random.default_rng(11)
    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.25], hspace=.45)
    reps = [(14, "Affected", "A"), (15, "Carrier", "B"), (10, "Unaffected", "C")]
    for j, (fam, g, L) in enumerate(reps):
        ax = fig.add_subplot(gs[0, j]); row = df[df.family == fam].iloc[0]
        y = [row.dos_8w, row.dos_10w, row.dos_12w]
        nocall_pts = [row.first_reportable_w > w for w in (8, 10, 12)]
        ax.plot([0, 1, 2], y, color=COL[g], lw=2)
        for x, v, nc in zip((0, 1, 2), y, nocall_pts):
            ax.scatter(x, v, s=60, facecolor="white" if nc else COL[g], edgecolor=COL[g], lw=2, zorder=3)
            ax.text(x, v + .05, f"{v:.2f}", color=COL[g], ha="center", weight="bold", fontsize=9)
        ax.axhline(.5, color="grey", ls="--", lw=1); ax.set_ylim(0, 1)
        ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["8w", "10w", "12w"])
        ax.set_xlabel("Gestational age (weeks)", weight="bold"); ax.set_ylabel("Integrated disease-allele dosage")
        ax.set_title(f"Family {fam} ({g})", color=COL[g], weight="bold")
        ax.text(-.25, 1.1, L, transform=ax.transAxes, fontsize=20, weight="bold")
        ax.text(.5, -.38, f"FF: {row.ff_8w:.1f}% | SNPs: {int(row.snps_8w)} (8w)", transform=ax.transAxes,
                ha="center", fontsize=9, color="#444")
        if row.first_reportable_w > 8:
            ax.text(.5, -.5, f"First reportable: {int(row.first_reportable_w)}w", transform=ax.transAxes,
                    ha="center", fontsize=9, color=COL[g])
    ax = fig.add_subplot(gs[1, :])
    for xi, w in enumerate((8, 10, 12)):
        for _, r in df.iterrows():
            nc = r.first_reportable_w > w
            x = xi + rng.uniform(-.14, .14)
            ax.scatter(x, r[f"dos_{w}w"], s=40, facecolor="white" if nc else COL[r.genotype],
                       edgecolor=COL[r.genotype], lw=1.6, zorder=3)
    for g in ("Affected", "Carrier"):
        m = [df[df.genotype == g][f"dos_{w}w"].mean() for w in (8, 10, 12)]
        ax.plot([0, 1, 2], m, color=COL[g], lw=2.5)
    ax.axhline(.5, color="grey", ls="--", lw=1); ax.set_ylim(0, 1)
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["8w", "10w", "12w"])
    rep = [(~(df.first_reportable_w > w)).sum() for w in (8, 10, 12)]
    for xi, n in enumerate(rep):
        ax.text(xi, -.14, f"Reportable: {n}/17", ha="center", weight="bold", transform=ax.get_xaxis_transform())
    ax.set_xlabel("\n\nGestational age (weeks)", weight="bold"); ax.set_ylabel("Integrated disease-allele dosage")
    ax.text(-.04, 1.05, "D", transform=ax.transAxes, fontsize=20, weight="bold")
    handles = [plt.Line2D([], [], marker="o", ls="", color=COL[g], label=g) for g in COL] + \
              [plt.Line2D([], [], marker="o", ls="", color="k", label="Reportable"),
               plt.Line2D([], [], marker="o", ls="", mfc="white", mec="k", label="No-call"),
               plt.Line2D([], [], ls="--", color="grey", label="Reference = 0.5")]
    ax.legend(handles=handles, ncol=2, frameon=False, loc="upper right", fontsize=8)
    fig.savefig(os.path.join(outdir, "Figure_3.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


# --- Sankey tự viết bằng matplotlib -----------------------------------------
def _flow(ax, x0, y0a, y0b, x1, y1a, y1b, color, alpha=.45):
    xm = (x0 + x1) / 2
    verts = [(x0, y0a), (xm, y0a), (xm, y1a), (x1, y1a),
             (x1, y1b), (xm, y1b), (xm, y0b), (x0, y0b), (x0, y0a)]
    codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4,
             Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CLOSEPOLY]
    ax.add_patch(PathPatch(Path(verts, codes), fc=color, ec="none", alpha=alpha))


def _node(ax, x, yc, h, text, ec="#777", fc="#f2f2f2", w=.12, fs=8):
    ax.add_patch(FancyBboxPatch((x - w / 2, yc - h / 2), w, h, boxstyle="round,pad=0.005,rounding_size=0.01",
                                fc=fc, ec=ec, lw=1.6, zorder=3))
    ax.text(x, yc, text, ha="center", va="center", fontsize=fs, zorder=4)


def fig4(outdir):
    fig, (a, b) = plt.subplots(2, 1, figsize=(11, 12), gridspec_kw=dict(height_ratios=[1, 1.05]))
    R, O, G = COL["Affected"], COL["Carrier"], COL["Unaffected"]
    LR, LO, LG = "#fbe3e2", "#fdeedd", "#e6f4ea"
    # ---- A ----
    a.set_xlim(0, 1.02); a.set_ylim(0, 1); a.axis("off")
    a.set_title("A  Clinical actionability pathway after RHDO-based NIPD classification", loc="left", weight="bold", fontsize=13)
    cols = [.07, .21, .36, .50, .65, .79, .92]
    heads = ["Cohort", "RHDO-based\nNIPD\nclassification", "Prenatal\npyridoxine", "",
             "Neonatal\npyridoxine\n(first dose)", "Seizure\nbefore\ntreatment", "NICU\nadmission"]
    for x, h in zip(cols, heads):
        a.text(x, .97, h, ha="center", va="top", weight="bold", fontsize=8, color="#444")
    # nodes
    _node(a, cols[0], .50, .35, "At-risk\npregnancies\nn = 17")
    _node(a, cols[1], .76, .12, "Affected\nn = 6\n(35.3%)", R, LR)
    _node(a, cols[1], .42, .14, "Carrier\nn = 9\n(52.9%)", O, LO)
    _node(a, cols[1], .12, .12, "Unaffected\nn = 2\n(11.8%)", G, LG)
    _node(a, cols[2], .85, .09, "Yes\nn = 5", R, LR); _node(a, cols[2], .70, .09, "No (delivered\nat 29 wk)\nn = 1", R, LR)
    _node(a, cols[2], .42, .12, "Routine\nfollow-up"); _node(a, cols[2], .12, .09, "Routine\nfollow-up")
    _node(a, cols[3], .77, .12, "Affected\nneonates\nn = 6")
    _node(a, cols[4], .76, .09, "≤1 hour\nn = 2", R, LR); _node(a, cols[4], .63, .09, "1–4 hours\nn = 3", R, LR)
    _node(a, cols[4], .50, .09, ">4 hours\nn = 1", R, LR)
    _node(a, cols[5], .70, .09, "No seizure\nn = 5", "#2e9e3f", "#eaf6ea"); _node(a, cols[5], .50, .09, "Seizure\nn = 1", R, LR)
    _node(a, cols[6], .74, .10, "No NICU\nadmission\nn = 4", "#2a4fa8", "#e8eefb")
    _node(a, cols[6], .54, .10, "NICU\nadmission\nn = 2", R, LR)
    h = .06
    for (x0, y0, x1, y1, c) in [(0, .6, 1, .76, R), (0, .42, 1, .42, O), (0, .30, 1, .12, G),
                                (1, .78, 2, .85, R), (1, .74, 2, .70, R), (1, .42, 2, .42, O), (1, .12, 2, .12, G),
                                (2, .85, 3, .79, R), (2, .70, 3, .75, R),
                                (3, .80, 4, .76, R), (3, .77, 4, .63, R), (3, .74, 4, .50, R),
                                (4, .76, 5, .70, R), (4, .63, 5, .70, R), (4, .50, 5, .50, R),
                                (5, .70, 6, .74, R), (5, .70, 6, .54, R), (5, .50, 6, .54, R)]:
        _flow(a, cols[x0] + .06, y0 + h / 2, y0 - h / 2, cols[x1] - .06, y1 + h / 2, y1 - h / 2, c)
    # ---- B ----
    b.set_xlim(0, 1); b.set_ylim(0, 1); b.axis("off")
    b.set_title("B  Prenatal neuroimaging and fetal movement by final fetal genotype", loc="left", weight="bold", fontsize=13)
    cx = [.08, .34, .62, .88]
    for x, t in zip(cx, ["Cohort", "Final fetal genotype", "Structural CNS anomaly", "Fetal movement pattern"]):
        b.text(x, .96, t, ha="center", weight="bold", fontsize=9, color="#444")
    _node(b, cx[0], .45, .35, "At-risk\npregnancies\nn = 17")
    _node(b, cx[1], .78, .12, "Affected\nn = 6", R, LR); _node(b, cx[1], .45, .18, "Carrier\nn = 9", O, LO)
    _node(b, cx[1], .12, .09, "Unaffected\nn = 2", G, LG)
    _node(b, cx[2], .80, .10, "CNS anomaly\npresent\nn = 3", R, LR); _node(b, cx[2], .40, .22, "No structural\nCNS anomaly\nn = 14")
    _node(b, cx[3], .78, .13, "Repetitive\njerking\nn = 6", R, LR); _node(b, cx[3], .25, .22, "Normal\nmovement\nn = 11")
    for (x0, y0, x1, y1, c, hh) in [(0, .62, 1, .78, R, .10), (0, .45, 1, .45, O, .16), (0, .30, 1, .12, G, .05),
                                    (1, .80, 2, .80, R, .08), (1, .72, 2, .46, R, .08), (1, .45, 2, .40, O, .16),
                                    (1, .12, 2, .30, G, .05),
                                    (2, .80, 3, .80, R, .08), (2, .50, 3, .72, R, .02), (2, .46, 3, .70, O, .015),
                                    (2, .44, 3, .68, G, .01), (2, .38, 3, .30, R, .05), (2, .32, 3, .22, O, .12),
                                    (2, .26, 3, .15, G, .03)]:
        _flow(b, cx[x0] + .06, y0 + hh / 2, y0 - hh / 2, cx[x1] - .06, y1 + hh / 2, y1 - hh / 2, c)
    b.text(.5, .01, "Structural CNS anomalies were confined to affected fetuses; repetitive jerking occurred in "
           "4 affected, 1 carrier and 1 unaffected pregnancy.", ha="center", style="italic", fontsize=8.5, color="#444")
    fig.savefig(os.path.join(outdir, "Figure_4.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# MAIN
# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="nipd_output")
    ap.add_argument("--data", default=None, help="CSV Supplementary Table 2 thật (tùy chọn)")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df = load_real_data(args.data) if args.data else build_simulated_cohort()
    if not args.data:
        print("[!] Đang dùng dữ liệu MÔ PHỎNG khớp bản thảo; hãy truyền --data để dùng dữ liệu thật.")

    # Hiệu năng chẩn đoán (loại no-call cuối cùng: ở đây 17/17 đều reportable)
    perf = diagnostic_performance(df.reference, df.nipd_call)
    k = int((~df.nocall_8w).sum())
    lo, hi = clopper_pearson(k, len(df))
    perf["First-pass"] = (k / len(df) * 100, lo * 100, hi * 100)
    print("\nHiệu năng chẩn đoán (value, CI thấp, CI cao; %):")
    for name, v in perf.items():
        print(f"  {name:<16} {v[0]:6.1f}  ({v[1]:.1f}–{v[2]:.1f})")
    print("\nPhân bố kiểu gen:", df.genotype.value_counts().to_dict())

    # Ví dụ chạy mô hình RHDO trên dữ liệu mô phỏng 1 ca (kiểm tra code)
    rng = np.random.default_rng(0)
    n = 220; d = rng.integers(200, 290, n); ff = .128
    types = rng.choice(["a", "b"], n)
    p = np.where(types == "a", .5 + ff / 2, .5)           # thai nhận haplotype risk
    r = rng.binomial(d, p)
    post = maternal_rhdo_posterior(r, d, types, ff)
    llr, k_sw = recombination_scan(r, d, types, ff, np.arange(n))
    print(f"\n[Demo RHDO] posterior(risk)={post:.4f} -> {call_direction(post)}; "
          f"max log-LR tái tổ hợp={llr:.2f} (không có bằng chứng switch nếu thấp)")
    pat = paternal_transmission_posterior(rng.binomial(d[:40], ff / 2), d[:40], ff, err=.002)
    print(f"[Demo bố] P(transmission)={pat:.4f}")
    print("[Demo tích hợp] ->", integrate_genotype(call_direction(post), call_direction(pat)))

    dil = simulate_dilution()
    print("\n[Dilution] tỉ lệ phân loại đúng theo FF:\n",
          dil.groupby("FF").correct.mean().round(2).to_string())

    df.to_csv(os.path.join(args.outdir, "cohort_table.csv"), index=False)
    fig2a_schematic(args.outdir); fig2(df, perf, args.outdir)
    fig3(df, args.outdir); fig4(args.outdir)
    print(f"\nĐã lưu hình và bảng vào: {args.outdir}/")


if __name__ == "__main__":
    main()
