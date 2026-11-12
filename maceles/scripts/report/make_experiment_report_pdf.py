#!/usr/bin/env python3
"""Build the formula+experiment report PDF."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties, fontManager

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FONT = Path("/workspace/group/shared/bza/symmetrix-polar/.tmp_fonts/NotoSansCJKsc-Regular.otf")
fontManager.addfont(str(FONT))
CN = FontProperties(fname=str(FONT))
plt.rcParams["mathtext.fontset"] = "dejavusans"
plt.rcParams["font.family"] = CN.get_name()
plt.rcParams["axes.unicode_minus"] = False

W, H = 8.27, 11.69


def new_page(title):
    fig = plt.figure(figsize=(W, H))
    fig.patch.set_facecolor("white")
    fig.text(0.08, 0.945, title, fontsize=16, fontweight="bold",
             fontproperties=CN, color="#111111")
    fig.lines.append(plt.Line2D([0.08, 0.92], [0.925, 0.925],
                                transform=fig.transFigure, color="#888888",
                                linewidth=0.8))
    return fig


def text(fig, y, s, size=10.5, color="#222222", weight="normal"):
    fig.text(0.08, y, s, fontsize=size, color=color, fontweight=weight,
             fontproperties=CN, va="top", wrap=True)


def eq(fig, y, s, size=14):
    fig.text(0.5, y, s, ha="center", va="top", fontsize=size,
             fontproperties=CN, color="#000000")


def save(fig, pdf):
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main():
    rw = json.load(open(ROOT / "examples/water96/reweight/reweight_manifest.json"))
    ver = json.load(open(ROOT / "validation/results/forcefield_c1c2_water96/verification.json"))
    off = ver["systems"]["MACE-OFF"]
    pol = ver["systems"]["MACE-POLAR"]
    d1 = pol["c1"]["c1_ex"] - off["c1"]["c1_ex"]
    lin = rw["linear"]
    slope = lin.get("slope", lin.get("c1"))
    intercept = lin.get("intercept_eV", lin.get("c2_total_eV"))

    out = HERE.parents[1] / "docs/reports/EXPERIMENT_REPORT.pdf"
    with PdfPages(out) as pdf:
        # 1
        fig = new_page("MACE-OFF → MACE-POLAR：公式、重加权与实验问题")
        text(fig, 0.885, "统计热力学推导 + 96 原子水实验现状", 14, "#333333", "bold")
        text(fig, 0.845,
             "本报告分两部分：\n"
             "(1) 重加权、密度、c1、c2、OZ、ESS 的公式推导；\n"
             "(2) 当前 MACE-OFF / MACE-POLAR 实验的结果与问题。", 11)
        text(fig, 0.75, "核心结论", 12, "#333333", "bold")
        text(fig, 0.715,
             "• 重加权公式精确，但要求平衡采样、正确权重、足够重叠。\n"
             "• 能量常数偏移只平移 c1 零点，对 c2 没有作用。\n"
             "• c2 由 g(r)+OZ 得到；c1 由密度/化学势或非均匀剖面得到。\n"
             "• 96 原子/10 ps 的实验只能定性：盒子太小、轨迹太短、ESS 仅 10%。\n"
             "• 1,296/5,184 原子直接重加权预计不可行（σβ≈14/29）。", 10.5)
        text(fig, 0.55, "参考", 11, "#333333", "bold")
        text(fig, 0.515,
             "• Dijkman et al., PRL 134, 056103 (2025).\n"
             "• Bui & Cox, PNAS 123, e2610049123 (2026).\n"
             "• Hansen & McDonald, Theory of Simple Liquids.", 9.5)
        save(fig, pdf)

        # 2
        fig = new_page("1. 重要性采样（重加权）的推导")
        text(fig, 0.885, "两个 Boltzmann 分布：", 11, "#333333", "bold")
        eq(fig, 0.845, r"$p_A(x)=\frac{e^{-\beta V_A(x)}}{Z_A},\qquad p_B(x)=\frac{e^{-\beta V_B(x)}}{Z_B}$", 13)
        text(fig, 0.765, "对任意观测量 O：", 11, "#333333", "bold")
        eq(fig, 0.725, r"$\langle O\rangle_B=\int O(x)p_B(x)dx=\int O(x)\frac{p_B(x)}{p_A(x)}p_A(x)dx$", 12)
        eq(fig, 0.635, r"$=\frac{\langle O\,e^{-\beta(V_B-V_A)}\rangle_A}{\langle e^{-\beta(V_B-V_A)}\rangle_A}$", 14)
        text(fig, 0.545, "配分函数比：", 11, "#333333", "bold")
        eq(fig, 0.505, r"$\frac{Z_B}{Z_A}=\langle e^{-\beta(V_B-V_A)}\rangle_A$", 13)
        text(fig, 0.425, "权重：", 11, "#333333", "bold")
        eq(fig, 0.385, r"$w_i=e^{-\beta[V_B(x_i)-V_A(x_i)]},\qquad \hat O_B=\frac{\sum_i w_i O_i}{\sum_i w_i}$", 13)
        text(fig, 0.295,
             "常数偏移：若 V_B=V_A+ΔE，则 w_i 都乘同一个 e^{-βΔE}，\n"
             "归一化后消掉；它不影响相对权重。", 10)
        text(fig, 0.205,
             "有限样本：自归一化估计的偏差为 O(1/ESS)，\n"
             "标准误约为 σ_O/√ESS。", 10)
        save(fig, pdf)

        # 3
        fig = new_page("2. 密度比与常数偏移")
        text(fig, 0.885, "一体密度：", 11, "#333333", "bold")
        eq(fig, 0.845, r"$\hat\rho(\mathbf{r})=\sum_i\delta(\mathbf{r}-\mathbf{r}_i),\qquad \rho_A(\mathbf{r})=\langle\hat\rho(\mathbf{r})\rangle_A$", 13)
        text(fig, 0.765, "密度的重加权：", 11, "#333333", "bold")
        eq(fig, 0.715, r"$\rho_B(\mathbf{r})=\frac{\langle\hat\rho(\mathbf{r})e^{-\beta(V_B-V_A)}\rangle_A}{\langle e^{-\beta(V_B-V_A)}\rangle_A}$", 13)
        text(fig, 0.615, "密度比：", 11, "#333333", "bold")
        eq(fig, 0.575, r"$\frac{\rho_B(\mathbf{r})}{\rho_A(\mathbf{r})}=\frac{\langle e^{-\beta(V_B-V_A)}\rangle_{A,\mathbf{r}}}{\langle e^{-\beta(V_B-V_A)}\rangle_A}$", 13)
        text(fig, 0.485,
             "条件平均：固定一个粒子在 r 的 A 系综平均。", 9.5)
        eq(fig, 0.425, r"$\frac{\rho_B(\mathbf{r})}{\rho_A(\mathbf{r})}\neq e^{-\beta[V_B(\mathbf{r})-V_A(\mathbf{r})]}$", 15)
        text(fig, 0.325, "常数偏移：", 11, "#333333", "bold")
        eq(fig, 0.285, r"$V_B=V_A+\Delta E\Rightarrow p_B=p_A,\ \rho_B=\rho_A,\ c_2^B=c_2^A,\ c_1^B=c_1^A-\beta\Delta E$", 11)
        text(fig, 0.195,
             "一般情况：∫ e^{β(V_A-V_B)} ρ_B dr 没有等于 N 或 ρ_A 的简单恒等式。", 10)
        save(fig, pdf)

        # 4
        fig = new_page("3. c1、c2 与 Ornstein–Zernike")
        text(fig, 0.885, "c1：一体直接相关函数", 11, "#333333", "bold")
        eq(fig, 0.845, r"$c_1(\mathbf{r})=-\beta\frac{\delta F_{\mathrm{exc}}[\rho]}{\delta\rho(\mathbf{r})}$", 14)
        text(fig, 0.765, "均匀体相：", 10)
        eq(fig, 0.725, r"$c_{1,\mathrm{ex}}=-\beta\mu_{\mathrm{ex}}$", 13)
        text(fig, 0.655, "非均匀：", 10)
        eq(fig, 0.615, r"$c_1(\mathbf{r})=\ln[\Lambda^3\rho(\mathbf{r})]-\beta\mu+\beta V_{\mathrm{ext}}(\mathbf{r})$", 12)
        text(fig, 0.525, "c2：二体直接相关函数", 11, "#333333", "bold")
        eq(fig, 0.485, r"$c_2(\mathbf{r},\mathbf{r}')=-\beta\frac{\delta^2F_{\mathrm{exc}}}{\delta\rho(\mathbf{r})\delta\rho(\mathbf{r}')}=\frac{\delta c_1(\mathbf{r})}{\delta\rho(\mathbf{r}')}$", 11)
        text(fig, 0.395, "均匀体相：h(r)=g(r)-1，OZ 关系", 10)
        eq(fig, 0.355, r"$h(k)=c_2(k)+\rho\,c_2(k)h(k)$", 13)
        eq(fig, 0.275, r"$c_2(k)=\frac{h(k)}{1+\rho h(k)}$", 14)
        eq(fig, 0.195, r"$c_2(r)=\frac{1}{2\pi^2}\int_0^\infty k^2\frac{\sin kr}{kr}\frac{h(k)}{1+\rho h(k)}dk$", 12)
        save(fig, pdf)

        # 5
        fig = new_page("4. 用加权样本估计 c1、c2")
        text(fig, 0.885, "c2：加权 g(r) + OZ", 11, "#333333", "bold")
        eq(fig, 0.845, r"$w_i=e^{-\beta[\Delta E_i-\min_j\Delta E_j]}$", 13)
        eq(fig, 0.765, r"$g_B(r)=\frac{\sum_i w_i g_i(r)}{\sum_i w_i}$", 13)
        eq(fig, 0.685, r"$c_{2,B}(k)=\frac{h_B(k)}{1+\rho h_B(k)}$", 13)
        text(fig, 0.605, "c1：加权 Widom（均匀体相）", 11, "#333333", "bold")
        eq(fig, 0.565, r"$c_{1,B}=\ln\frac{\sum_i w_i\langle e^{-\beta\Delta U}\rangle_i}{\sum_i w_i}$", 13)
        text(fig, 0.475, "ESS：", 11, "#333333", "bold")
        eq(fig, 0.435, r"$\mathrm{ESS}=\frac{(\sum_i w_i)^2}{\sum_i w_i^2}\in[1,N]$", 13)
        text(fig, 0.345,
             "ESS=90/100：权重均匀，偏差约 1%，标准误 ~σ/√90；\n"
             "但 ESS 高不代表覆盖完整；还要检查 g(r)、密度、最大权重。", 10)
        text(fig, 0.245,
             "帧相关：先做 ACF/thinning；否则 ESS 会被高估。", 10)
        text(fig, 0.185,
             "Widom 插入本身也有方差；稠密水需要软核/TI。", 10)
        save(fig, pdf)

        # 6
        fig = new_page("5. 当前实验：设置与结果")
        text(fig, 0.885,
             "体系：32 H2O（96 原子），11.696 Å，ρ=0.02 Å^-3，300 K；\n"
             "两套 10 ps NVT 轨迹（1 ps 平衡，0.1 ps 存帧），PyTorch 计算器。", 10)
        text(fig, 0.79, "双力场结果（81 帧/套）：", 11, "#333333", "bold")
        text(fig, 0.75,
             f"MACE-OFF   c1_ex = {off['c1']['c1_ex']:.3f}，"
             f"c2 ∈ [{off['c2_min']:.1f}, {off['c2_max']:.1f}]\n"
             f"MACE-POLAR c1_ex = {pol['c1']['c1_ex']:.3f}，"
             f"c2 ∈ [{pol['c2_min']:.1f}, {pol['c2_max']:.1f}]\n"
             f"c1 差 = {d1:.2f}（与 -βΔE 一致）", 10)
        text(fig, 0.62, "重加权诊断（36 帧）：", 11, "#333333", "bold")
        text(fig, 0.58,
             f"ΔE 偏移 = {rw['constant_offset_meV_per_atom']:.2f} meV/atom\n"
             f"ΔE 涨落 std = {rw['delta_meV_per_atom_std']:.3f} meV/atom\n"
             f"σβ = {rw['sigma_beta']:.2f}\n"
             f"ESS = {rw['ess']:.2f}/{rw['n_frames']} = {100*rw['ess_fraction']:.1f}%\n"
             f"slope = {slope:.4f}，intercept = {intercept:.1f} eV\n"
             f"R² = {rw['linear']['r2']:.4f}，残差 = {rw['linear']['residual_rms_meV_per_atom']:.2f} meV/atom\n"
             f"力 MAE = {rw['force_mae_meV_per_A_mean']:.1f} meV/Å", 10)
        text(fig, 0.27,
             "修正 RDF 归一化后，两套 c2 很接近（RMS 0.20）；但 96 原子/10 ps 的噪声\n"
             "与有限尺寸仍占主导，不能当成真实的 MACE-OFF vs MACE-POLAR 结构差。", 10, "#333333")
        save(fig, pdf)

        # 7
        fig = new_page("6. 当前实验存在的问题")
        text(fig, 0.885, "1. 盒子太小", 11, "#333333", "bold")
        text(fig, 0.85,
             "L=11.696 Å，小于 2r_cut=12 Å；低-k 极限和 c2(0) 受有限尺寸影响。\n"
             "生产应使用 L≥2r_cut；该密度下至少 ~104 个原子，建议 128/192。", 9.8)
        text(fig, 0.76, "2. 轨迹太短", 11, "#333333", "bold")
        text(fig, 0.725,
             "10 ps 的 g(r) 仍有噪声；修正归一化后 c2≈[-4.8, 3.5]\n"
             "（旧版 -293~1337 是 RDF 球壳体积 bug 伪影）。至少需要 100 ps。", 9.8)
        text(fig, 0.64, "3. 温度控制偏弱", 11, "#333333", "bold")
        text(fig, 0.605,
             "恒温器设定 300 K，但生产段平均瞬时温度约 334 K；\n"
             "应提高摩擦系数并延长平衡段。", 9.8)
        text(fig, 0.52, "4. c1 绝对值不可直接比较", 11, "#333333", "bold")
        text(fig, 0.485,
             "c1_ex 约 8×10^4，被绝对能量参考主导；必须对齐零点，\n"
             "或只比较差值/导数。Widom 仅 50/100 次插入被接受，方差大。", 9.8)
        text(fig, 0.395, "5. ESS 只在 96 原子勉强", 11, "#333333", "bold")
        text(fig, 0.36,
             "36 帧 ESS=3.6（10%）；外推 1,296 原子 σβ≈14，5,184≈29，\n"
             "直接重加权会塌缩；增加帧数不能解决。", 9.8)
        text(fig, 0.27, "6. 重算能量不能替代目标构型", 11, "#333333", "bold")
        text(fig, 0.235,
             "在 OFF 帧上算 POLAR 能量不会改变 g(r)；c2 仍是 OFF 的。\n"
             "c1 需要目标系综的化学势/密度剖面，或加权。", 9.8)
        text(fig, 0.15, "7. GPU 成本", 11, "#333333", "bold")
        text(fig, 0.115,
             "96 原子 10 ps：OFF 约 525 s，POLAR 约 2006 s；1 ns 约 15/56 h。\n"
             "1,296 原子更慢；生产应使用引擎或蒸馏模型。", 9.8)
        save(fig, pdf)

        # 8
        fig = new_page("7. 已经严谨验证的部分与下一步")
        text(fig, 0.885, "已经验证：", 11, "#333333", "bold")
        text(fig, 0.85,
             "• OZ 反演：光滑解析 c(r) 误差 4.9×10^-13；\n"
             "• Percus–Yevick 硬球平滑段误差 2.1×10^-13；\n"
             "• 理想气体 c2=0；12 项单元测试通过。\n"
             "• 权重/ESS/线性拟合公式与实现一致。", 10)
        text(fig, 0.71, "尚未验证：", 11, "#333333", "bold")
        text(fig, 0.675,
             "• 96 原子/10 ps 的 c2 不是定量结果；\n"
             "• c1 的 Widom 插入方差大；\n"
             "• 两力场 c2 的真实差异尚未测出；\n"
             "• 1 ns / 5–10 ns 的生产统计未做。", 10)
        text(fig, 0.55, "下一步建议：", 11, "#333333", "bold")
        text(fig, 0.515,
             "1. 盒子 ≥128 原子（L≥2r_cut），温度控制用更强摩擦；\n"
             "2. 至少 100 ps–1 ns 的 MACE-OFF / MACE-POLAR 轨迹；\n"
             "3. 对齐 c1 零点；报告 ESS、最大权重、bootstrap 误差；\n"
             "4. 如果 ESS<10%：直接跑目标轨迹或训练 ΔV 代理势；\n"
             "5. 定量 c2 需要 5–10 ns 或更大盒子。", 10)
        text(fig, 0.30,
             "结论：公式和 OZ 反演是严谨的；当前实验只是可行性 smoke，\n"
             "不能用于定量比较 MACE-OFF 与 MACE-POLAR 的 c2/c1。", 10.5, "#111111", "bold")
        save(fig, pdf)

        # 9
        fig = new_page("References")
        text(fig, 0.885,
             "[1] J. Dijkman, M. Dijkstra, R. van Roij, M. Welling,\n"
             "    J.-W. van de Meent, B. Ensing,\n"
             "    Learning Neural Free-Energy Functionals with Pair-Correlation Matching,\n"
             "    Phys. Rev. Lett. 134, 056103 (2025).\n"
             "    DOI: 10.1103/PhysRevLett.134.056103; arXiv:2403.15007.", 9.5)
        text(fig, 0.76,
             "[2] T. Bui, S. J. Cox,\n"
             "    A unified machine-learning framework for ab initio multiscale modeling of liquids,\n"
             "    PNAS 123, e2610049123 (2026).\n"
             "    DOI: 10.1073/pnas.2610049123; arXiv:2603.20493.", 9.5)
        text(fig, 0.64,
             "[3] R. W. Zwanzig, J. Chem. Phys. 22, 1420 (1954).\n"
             "[4] G. M. Torrie, J. P. Valleau, J. Comput. Phys. 23, 187 (1977).\n"
             "[5] A. M. Ferrenberg, R. H. Swendsen, Phys. Rev. Lett. 63, 1195 (1989).\n"
             "[6] C. H. Bennett, J. Comput. Phys. 22, 245 (1976).\n"
             "[7] M. R. Shirts, J. D. Chodera, J. Chem. Phys. 129, 124105 (2008).\n"
             "[8] J.-P. Hansen, I. R. McDonald, Theory of Simple Liquids.", 9.5)
        text(fig, 0.33,
             "代码与数据：\n"
             "https://github.com/Yong-Q/test\n"
             "sampling/direct_correlation.py, sampling/off_polar.py\n"
             "validation/run_forcefield_c1c2.py", 9.5)
        save(fig, pdf)

    print(out)


if __name__ == "__main__":
    main()
