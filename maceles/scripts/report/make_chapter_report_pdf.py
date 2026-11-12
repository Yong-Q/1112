#!/usr/bin/env python3
"""Chapter report with native vector plots (no raster screenshots)."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[2]
FONT = Path("/workspace/group/shared/bza/symmetrix-polar/.tmp_fonts/NotoSansCJKsc-Regular.otf")
fontManager.addfont(str(FONT)); CN = FontProperties(fname=str(FONT))
plt.rcParams["mathtext.fontset"] = "dejavusans"
plt.rcParams["font.family"] = CN.get_name()
# Embed fonts with Unicode maps so the review PDF stays selectable/searchable
# (matplotlib's default Type 3 output loses the character mapping).
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
W, H = 8.27, 11.69
INK = "#1a1a1a"


def page(title):
    fig = plt.figure(figsize=(W, H)); fig.patch.set_facecolor("white")
    fig.text(0.08, 0.945, title, fontsize=15, fontweight="bold", fontproperties=CN, color=INK)
    fig.lines.append(plt.Line2D([0.08, 0.92], [0.925, 0.925], transform=fig.transFigure,
                                color="#999999", linewidth=0.8))
    return fig


def text(fig, y, s, size=10.2, color=INK, weight="normal"):
    fig.text(0.08, y, s, fontsize=size, color=color, fontweight=weight,
             fontproperties=CN, va="top", wrap=True)


def eq(fig, y, s, size=12.5):
    fig.text(0.5, y, s, ha="center", va="top", fontsize=size, fontproperties=CN, color=INK)


def style(ax, xlabel, ylabel, title=None):
    ax.set_xlabel(xlabel, fontproperties=CN, fontsize=9)
    ax.set_ylabel(ylabel, fontproperties=CN, fontsize=9)
    if title:
        ax.set_title(title, fontproperties=CN, fontsize=10)
    ax.tick_params(labelsize=8)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontproperties(CN)
    ax.grid(True, alpha=0.22, linewidth=0.6)


def save(fig, pdf):
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def main():
    rw = json.load(open(ROOT / "examples/water96/reweight/reweight_manifest.json"))
    ver = json.load(open(ROOT / "validation/results/forcefield_c1c2_water96/verification.json"))
    comp = json.load(open(ROOT / "validation/results/forcefield_c1c2_water96/reweight_vs_direct/comparison.json"))
    hv = json.load(open(ROOT / "validation/results/harmonic_validation.json"))
    live = np.genfromtxt(ROOT / "docs/results/ess_live_1152.csv", delimiter=",", names=True)
    lin = rw["linear"]
    slope = lin.get("slope", lin.get("c1", float("nan")))
    intercept = lin.get("intercept_eV", lin.get("c2_total_eV", float("nan")))

    out = ROOT / "docs/reports/RESEARCH_REPORT.pdf"
    with PdfPages(out) as pdf:
        # 1 title
        fig = page("MACE-OFF → MACE-POLAR：c1/c2 重加权的理论与实验")
        text(fig, 0.885, "Technical report · 2026-10-23 · for review", 11, "#555555", "bold")
        text(fig, 0.84,
             "第一章给出重加权、ESS、c1、c2 与 Ornstein–Zernike 的推导；"
             "第二章给出 96 原子与 1,152 原子水体系的实测结果、问题和结论。", 10.5)
        text(fig, 0.745, "摘要", 12, "#333333", "bold")
        text(fig, 0.705,
             "重加权公式与 OZ 反演在数学上正确（解析验证误差 4.9×10^-13）；"
             "但实际相空间重叠不足：96 原子 ESS=10.1%（marginal），"
             "1,152 原子 5 ps 早期检查 ESS=1（collapse，6 帧）。"
             "外推表明把轨迹延长到 1 ns 也不会解决重叠问题。"
             "RDF 归一化修正后，两力场 c2 差异 RMS=0.20，重加权未改善（0.48）。", 10.5)
        text(fig, 0.585, "给老师的判断点", 12, "#333333", "bold")
        text(fig, 0.545,
             "• 是否接受直接 MACE-POLAR 采样（100 ps–1 ns）作为 c2/c1 参照？\n"
             "• 是否为 1,000+ 原子投入 ΔV 代理势（能量+力）？\n"
             "• 非均匀 c1(r) 走“外场密度剖面 + Euler–Lagrange”，还是 PRL 式多密度泛函？", 10)
        # flow diagram
        ax = fig.add_axes([0.10, 0.20, 0.80, 0.20]); ax.axis("off")
        labels = ["MACE-OFF\n轨迹", "逐帧 MACE-POLAR\n单点能量", "权重\n$w_i=e^{-\\beta\\Delta V_i}$", "ESS / verdict"]
        xs = [0.02, 0.28, 0.54, 0.80]
        for x, lab in zip(xs, labels):
            ax.add_patch(FancyBboxPatch((x, 0.35), 0.16, 0.30, boxstyle="round,pad=0.02",
                                        fc="#eef3f8", ec="#5b7c99", lw=1.0,
                                        transform=ax.transAxes))
            ax.text(x + 0.08, 0.50, lab, ha="center", va="center", fontsize=8,
                    fontproperties=CN, transform=ax.transAxes)
        for x in [0.18, 0.44, 0.70]:
            ax.add_patch(FancyArrowPatch((x, 0.50), (x + 0.09, 0.50), arrowstyle="-|>",
                                         mutation_scale=10, color="#5b7c99",
                                         transform=ax.transAxes))
        text(fig, 0.13, "流程本身已跑通；这次报告的重点是它在什么条件下失效。", 9.5, "#555555")
        save(fig, pdf)

        # 2 theory 1
        fig = page("第一章 理论：系综、密度与重加权")
        text(fig, 0.885, "1.1 两个 Boltzmann 系综", 11.5, "#333333", "bold")
        eq(fig, 0.845, r"$p_A(x)=\frac{e^{-\beta V_A(x)}}{Z_A},\qquad p_B(x)=\frac{e^{-\beta V_B(x)}}{Z_B}$")
        text(fig, 0.77, "一体密度算符与密度：", 10)
        eq(fig, 0.73, r"$\hat\rho(\mathbf{r})=\sum_i\delta(\mathbf{r}-\mathbf{r}_i),\qquad \rho_A(\mathbf{r})=\langle\hat\rho(\mathbf{r})\rangle_A$")
        eq(fig, 0.65, r"$\int\rho_A(\mathbf{r})d\mathbf{r}=\int\rho_B(\mathbf{r})d\mathbf{r}=N$")
        text(fig, 0.57, "A 的样本来自 $p_A$；目标是估计 B 的系综量。密度是系综平均，不是单帧量。", 10)
        text(fig, 0.47, "1.2 重要性采样（重加权）", 11.5, "#333333", "bold")
        eq(fig, 0.43, r"$\langle O\rangle_B=\int O p_B=\int O\frac{p_B}{p_A}p_A=\frac{\langle O e^{-\beta(V_B-V_A)}\rangle_A}{\langle e^{-\beta(V_B-V_A)}\rangle_A}$", 11.2)
        text(fig, 0.33, "权重与配分函数比：", 10)
        eq(fig, 0.295, r"$w_i=e^{-\beta[V_B(x_i)-V_A(x_i)]},\qquad \frac{Z_B}{Z_A}=\langle e^{-\beta(V_B-V_A)}\rangle_A$")
        text(fig, 0.19, r"常数偏移 $V_B=V_A+\Delta E$ 在归一化权重里消掉；只有逐帧涨落进入 ESS。", 9.8)
        save(fig, pdf)

        # 3 theory 2
        fig = page("第一章 理论：c1、c2 与 OZ")
        text(fig, 0.885, "1.3 直接相关函数", 11.5, "#333333", "bold")
        eq(fig, 0.845, r"$c_1(\mathbf{r};[\rho])=-\beta\frac{\delta F_{\rm exc}[\rho]}{\delta\rho(\mathbf{r})}$")
        eq(fig, 0.755, r"$c_2(\mathbf{r},\mathbf{r}';[\rho])=\frac{\delta c_1(\mathbf{r})}{\delta\rho(\mathbf{r}')}=-\beta\frac{\delta^2F_{\rm exc}}{\delta\rho(\mathbf{r})\delta\rho(\mathbf{r}')}$")
        text(fig, 0.66, "均匀体相：c1 是常数，c2 只依赖距离。", 10)
        eq(fig, 0.62, r"$c_{1,\rm ex}=-\beta\mu_{\rm ex},\qquad c_2(\mathbf{r},\mathbf{r}')=c_2(|\mathbf{r}-\mathbf{r}'|)$")
        text(fig, 0.53, "1.4 Ornstein–Zernike", 11.5, "#333333", "bold")
        eq(fig, 0.49, r"$h(r)=g(r)-1,\qquad h(k)=4\pi\int r^2h(r)\frac{\sin kr}{kr}dr$")
        eq(fig, 0.385, r"$c_2(k)=\frac{h(k)}{1+\rho h(k)},\qquad c_2(r)=\frac{1}{2\pi^2}\int k^2c_2(k)\frac{\sin kr}{kr}dk$")
        text(fig, 0.28, "1.5 非均匀 c1 的中心化", 11.5, "#333333", "bold")
        eq(fig, 0.235, r"$c_1(\mathbf{r})=\ln[\Lambda^3\rho(\mathbf{r})]-\beta\mu+\beta V_{\rm ext}(\mathbf{r})$")
        text(fig, 0.15, r"减掉空间平均去掉 $-\beta\mu$；不需要 Widom。", 9.8)
        save(fig, pdf)

        # 4 theory 3 + ESS scaling native
        fig = page("第一章 理论：ESS 与尺寸限制")
        text(fig, 0.885, "1.6 有效样本数", 11.5, "#333333", "bold")
        eq(fig, 0.845, r"$\mathrm{ESS}=\frac{(\sum_iw_i)^2}{\sum_iw_i^2}\in[1,N],\qquad \sigma_\beta=\mathrm{std}(\beta\Delta V)$")
        eq(fig, 0.755, r"$\frac{\mathrm{ESS}}{N}\sim e^{-\sigma_\beta^2}$")
        ax = fig.add_axes([0.10, 0.44, 0.36, 0.23])
        ax.loglog([96, 1152], [3.89, 93.8], "o-", color="#C0392B", ms=5)
        ax.axhline(3, color="#E67E22", ls=":")
        ax.axhline(5, color="#8E44AD", ls="--")
        ax.text(100, 3.25, "3: marginal", fontsize=7, color="#E67E22")
        ax.text(100, 5.5, "5: collapse", fontsize=7, color="#8E44AD")
        style(ax, "atoms $N$", r"$\sigma_\beta$", "权重展宽随 N 增大")
        ax = fig.add_axes([0.55, 0.44, 0.36, 0.23])
        bars = ax.bar([0, 1], [3.63, 1.0], width=0.5, color=["#2E86C1", "#C0392B"])
        for x, v, lab in zip([0, 1], [3.63, 1.0], ["3.63/36\n= 10.1%", "1/6 (5 ps)\nESS≈1"]):
            ax.text(x, v + 0.12, lab, ha="center", va="bottom", fontsize=7)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["96 atoms", "1,152 atoms"])
        ax.set_ylim(0, 4.4)
        style(ax, "", "ESS (frames)", "绝对有效样本数")
        text(fig, 0.37, "1.7 为什么随 N 失效", 11.5, "#333333", "bold")
        text(fig, 0.335, r"$\Delta V=\sum_i\delta v_i$：每原子 1 meV，1,152 原子就是 1.15 eV（$\beta\Delta V\approx44$）。", 9.8)
        text(fig, 0.27, "高维相空间重叠随 N 指数下降；增加帧数不改变每帧权重的方差。", 9.8)
        text(fig, 0.205, "1.8 与文献做法的区别", 11.5, "#333333", "bold")
        text(fig, 0.17, "PRL 用 bulk g(r) 训练 c2；PNAS 用非均匀密度学 c1；都不是单条均相轨迹直接重加权。", 9.6)
        save(fig, pdf)

        # 5 Chapter 2 setup + 96 diagnostics
        fig = page("第二章 实验：96 原子重加权诊断")
        text(fig, 0.885, "2.1 体系：32 H2O（96 原子），11.696 Å，0.02 molecules/Å³，300 K，0.5 fs。", 9.8)
        text(fig, 0.845, "源：原始 MACE-OFF23 medium（Symmetrix FP32 引擎）；目标：MACE-POLAR-1-M（FP64 单点）。", 9.8)
        ax = fig.add_axes([0.10, 0.48, 0.36, 0.26])
        frames = [f["index"] for f in rw["frames"]]
        offset = rw["constant_offset_meV_per_atom"]
        de = np.array([f["dE_meV_per_atom"] for f in rw["frames"]])
        ax.plot(frames, de - offset, "o-", ms=2.5, lw=0.8, color="#2E86C1")
        ax.axhline(0, color="#C0392B", ls=":")
        ax.axhspan(-rw["delta_meV_per_atom_std"], rw["delta_meV_per_atom_std"],
                   color="#C0392B", alpha=0.10, lw=0)
        ax.set_ylim(-2.2, 2.2)
        style(ax, "frame", r"$\Delta E-$offset (meV/atom)", f"offset = {offset:.2f} meV/atom")
        ax = fig.add_axes([0.55, 0.48, 0.36, 0.26])
        w = np.array([f["weight"] for f in rw["frames"]])
        ax.bar(np.arange(len(w)), w, color="#5DADE2")
        style(ax, "frame", "weight", f"ESS={rw['ess']:.2f}/{rw['n_frames']}")
        text(fig, 0.41,
             f"偏移 {rw['constant_offset_meV_per_atom']:.2f} meV/atom；涨落 {rw['delta_meV_per_atom_std']:.3f} meV/atom；"
             f"σβ={rw['sigma_beta']:.2f}；ESS={rw['ess']:.2f}/{rw['n_frames']}={100*rw['ess_fraction']:.1f}%。", 9.6)
        text(fig, 0.32,
             f"线性 slope/intercept={slope:.4f}/{intercept:.1f} eV；"
             f"R²={rw['linear']['r2']:.4f}；残差 {rw['linear']['residual_rms_meV_per_atom']:.2f} meV/atom；"
             f"力 MAE {rw['force_mae_meV_per_A_mean']:.1f} meV/Å。", 9.6)
        text(fig, 0.22, "判定：marginal。96 原子已经是边界；这不是“生产级 c2”。", 10, "#B03A2E", "bold")
        save(fig, pdf)

        # 6 c2 comparison native
        fig = page("第二章 实验：重加权 c2 与直接 POLAR 的对照")
        r = np.array(comp["r"])
        rmax = float(comp.get("rmax_c2_A", r.max()))
        ax = fig.add_axes([0.10, 0.46, 0.38, 0.30])
        ax.plot(r, comp["g_off"], label="MACE-OFF", color="#2E86C1", lw=1.2)
        ax.plot(r, comp["g_reweighted"], "--", label="reweighted", color="#E67E22", lw=1.2)
        ax.plot(r, comp["g_direct_polar"], ":", label="MACE-POLAR direct", color="#27AE60", lw=1.2)
        ax.set_xlim(2.0, rmax); ax.set_ylim(0, 4.6)
        style(ax, r"$r$ (Å)", r"$g_{OO}(r)$", "O–O RDF")
        ax.legend(fontsize=8, prop=CN, loc="upper right")
        ax = fig.add_axes([0.55, 0.46, 0.38, 0.30])
        ax.plot(r, comp["c2_off"], label="MACE-OFF", color="#2E86C1", lw=1.2)
        ax.plot(r, comp["c2_reweighted"], "--", label="reweighted", color="#E67E22", lw=1.2)
        ax.plot(r, comp["c2_direct_polar"], ":", label="MACE-POLAR direct", color="#27AE60", lw=1.2)
        ax.set_xlim(1.2, rmax); ax.set_ylim(-6, 4)
        style(ax, r"$r$ (Å)", r"$c_2(r)$", "c2（OZ 反演）")
        ax.legend(fontsize=8, prop=CN, loc="lower right")
        text(fig, 0.38,
             f"81 帧：σβ={comp['sigma_beta']:.2f}，ESS={comp['ess']:.2f}/81={100*comp['ess_fraction']:.1f}%。"
             f"直接 POLAR vs OFF 的 c2 RMS={comp['c2_direct_vs_off_rms']:.2f}；"
             f"直接 POLAR vs 重加权 RMS={comp['c2_direct_vs_reweighted_rms']:.2f}。", 9.6)
        text(fig, 0.27,
             "修正归一化后，两个力场的 c2 本身只差 RMS 0.20；重加权（ESS 2.6%）没有缩小差异，"
             "反而把噪声放大到 0.48。10 ps / 81 帧还不足以区分两者的真实结构差。", 9.6)
        text(fig, 0.16,
             "要得到有意义的对照：同一盒子、至少 100 ps 的 POLAR 轨迹，并在 ESS≥10% 的体系上比较。", 10, "#B03A2E", "bold")
        text(fig, 0.075,
             "本版修正：RDF 球壳体积应为 4π/3·(r³−r$_0$³)，旧版误用 r² 差，g 被放大约 3r/2；旧 c2 数值作废。",
             8.6, "#666666")
        save(fig, pdf)

        # 7 1152 live native
        fig = page("第二章 实验：1,152 原子 5 ps 早期 ESS（6 帧）")
        step = live["step"]; b = live["beta_dE"]
        sig = np.array([b[:i+1].std(ddof=1) if i > 0 else 0.0 for i in range(len(b))])
        w = np.exp(-(b - b.min())); w /= w.sum()
        ess_run = []
        for i in range(len(b)):
            wi = np.exp(-(b[:i+1] - b[:i+1].min())); wi /= wi.sum()
            ess_run.append(1.0 / np.sum(wi ** 2))
        ax = fig.add_axes([0.10, 0.46, 0.38, 0.30])
        ax.plot(step, sig, "o-", color="#C0392B", ms=3)
        ax.axhline(3, color="#E67E22", ls=":"); ax.axhline(5, color="#8E44AD", ls="--")
        ax.text(step.max() * 0.45, 8, "3: marginal", fontsize=7, color="#E67E22")
        ax.text(step.max() * 0.45, 14, "5: collapse", fontsize=7, color="#8E44AD")
        style(ax, "MD step (0.5 fs)", r"$\sigma_\beta$", "实时权重展宽")
        ax = fig.add_axes([0.55, 0.46, 0.38, 0.30])
        ax.plot(step, ess_run, "o-", color="#2E86C1", ms=3)
        ax.axhline(1, color="#7F8C8D", ls=":")
        ax.text(step.max() * 0.45, 1.03, "ESS = 1", fontsize=7, color="#555555")
        ax.set_ylim(0.9, 1.8)
        style(ax, "MD step (0.5 fs)", "ESS", "实时有效样本数")
        text(fig, 0.385,
             "运行 24 min 17 s：10,000 步 × 0.5 fs = 5 ps；dump 每 1 ps 一帧，共 6 帧。这不是 1 ns 结果。", 9.6)
        text(fig, 0.315,
             f"σβ 从 127 降到 {sig[-1]:.1f}，ESS={ess_run[-1]:.1f}/{len(b)}。按实测 6.9 步/s 外推，"
             "1 ns（2,000,000 步）约需 80 h，因此在 5 ps 处停止。", 9.6)
        text(fig, 0.245,
             "6 帧不足以给精确 ESS；但现有帧的 βΔV 跨度约 180 kT，再加几帧不会把 ESS 从 1 拉回 10%。", 9.6)
        text(fig, 0.165,
             "0.02 molecules/Å³ 在 300 K 接近水的两相区，本段还是平衡段，不是稳定单相生产轨迹。", 9.4)
        text(fig, 0.10, "结论：结合 96 原子与 √N 标度，1,000+ 原子的直接重加权不可行。", 10, "#B03A2E", "bold")
        save(fig, pdf)

        # 8 validation + problems
        fig = page("第二章 实验：验证、问题与下一步")
        text(fig, 0.885, "2.5 已严谨验证", 11.5, "#333333", "bold")
        conv = hv["convergence"]
        ax = fig.add_axes([0.10, 0.58, 0.36, 0.22])
        n = np.array([c["n"] for c in conv]); err = np.abs([c["mean_error"] for c in conv])
        ax.loglog(n, err, "o-", color="#2E86C1")
        style(ax, "source samples", r"$|\langle x\rangle-x_0|$ (Å)", "1D harmonic convergence")
        ax = fig.add_axes([0.55, 0.58, 0.36, 0.22])
        m = np.array([s["n_modes"] for s in hv["scaling"]]); sb = np.array([s["sigma_beta_measured"] for s in hv["scaling"]])
        ax.loglog(m, sb, "o-", color="#C0392B")
        ax.loglog(m, [s["sigma_beta_theory"] for s in hv["scaling"]], "--", color="#555555")
        style(ax, "modes N", r"$\sigma_\beta$", r"collapse $\propto\sqrt{N}$")
        text(fig, 0.50,
             "OZ 反演：光滑 c(r) 误差 4.9×10^-13；Percus–Yevick 平滑段 2.1×10^-13；理想气体 c2=0；"
             "RDF 归一化单测通过；共 14 项测试。", 9.6)
        text(fig, 0.455,
             "本版修正：RDF 球壳体积应为 4π/3·(r³−r$_0$³)，旧版误用 r² 差，g 被放大约 3r/2；旧 c2 数值作废。",
             8.8, "#B03A2E")
        text(fig, 0.40, "2.6 当前实验的问题", 11.5, "#333333", "bold")
        text(fig, 0.365,
             "• 96 原子盒子 L=11.7 Å < 2r_cut=12 Å；0.02 密度在 300 K 接近两相区。\n"
             "• 10 ps / 81 帧的 c2 仍受统计噪声限制；修正后 c2≈[−4.8, 3.5]，两力场 RMS 0.20。\n"
             "• c1 绝对值被能量参考主导；Widom 仅 50/100 次插入被接受。\n"
             "• 1,152 原子只跑 5 ps / 6 帧（24 min）；σβ≈94，ESS=1 是早期判据。", 9.5)
        text(fig, 0.20, "2.7 建议", 11.5, "#333333", "bold")
        text(fig, 0.165,
             "• 要 MACE-POLAR 的 c2：直接跑 POLAR 100 ps（1,152 原子约 8.3 h），或用 ΔV 代理势。\n"
             "• 要 c1(r)：外场下非均匀 POLAR 轨迹 + Euler–Lagrange；均相只能给常数。\n"
             "• 要 PRL 式泛函：多密度 bulk c2 训练 c2-matching 泛函。", 9.5)
        save(fig, pdf)

        # 9 references
        fig = page("References")
        text(fig, 0.885,
             "[1] J. Dijkman et al., Learning Neural Free-Energy Functionals with Pair-Correlation Matching,\n"
             "    Phys. Rev. Lett. 134, 056103 (2025). DOI: 10.1103/PhysRevLett.134.056103.\n\n"
             "[2] A. T. Bui, S. J. Cox, A unified machine-learning framework for ab initio multiscale\n"
             "    modeling of liquids, PNAS 123, e2610049123 (2026). DOI: 10.1073/pnas.2610049123.\n\n"
             "[3] R. W. Zwanzig, J. Chem. Phys. 22, 1420 (1954).\n"
             "[4] G. M. Torrie, J. P. Valleau, J. Comput. Phys. 23, 187 (1977).\n"
             "[5] A. M. Ferrenberg, R. H. Swendsen, Phys. Rev. Lett. 63, 1195 (1989).\n"
             "[6] C. H. Bennett, J. Comput. Phys. 22, 245 (1976).\n"
             "[7] M. R. Shirts, J. D. Chodera, J. Chem. Phys. 129, 124105 (2008).", 9.2)
        text(fig, 0.32,
             "代码与数据：https://github.com/Yong-Q/test\n"
             "sampling/direct_correlation.py; sampling/off_polar.py; scripts/monitor_ess.py", 9)
        save(fig, pdf)
    print(out)


if __name__ == "__main__":
    main()
