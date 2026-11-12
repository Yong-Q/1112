#!/usr/bin/env python3
"""Render the density/reweighting/c1-c2 math note to a PDF with matplotlib."""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties, fontManager

HERE = Path(__file__).resolve().parent
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
    fig.text(0.08, 0.945, title, fontsize=17, fontweight="bold",
             fontproperties=CN, color="#111111")
    fig.lines.append(plt.Line2D([0.08, 0.92], [0.925, 0.925],
                                transform=fig.transFigure, color="#888888",
                                linewidth=0.8))
    return fig


def text(fig, y, s, size=11, color="#222222", weight="normal"):
    fig.text(0.08, y, s, fontsize=size, color=color, fontweight=weight,
             fontproperties=CN, va="top", wrap=True)


def eq(fig, y, s, size=15):
    fig.text(0.5, y, s, ha="center", va="top", fontsize=size,
             fontproperties=CN, color="#000000")


def save(fig, pdf):
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main():
    out = HERE.parents[1] / "docs/reports/THEORY_REPORT.pdf"
    with PdfPages(out) as pdf:
        # page 1
        fig = new_page("MACE-OFF → MACE-POLAR：密度、重加权与 c1/c2")
        text(fig, 0.885, "统计热力学关系速查", 15, "#333333", "bold")
        text(fig, 0.845,
             "目的：说明“逐帧用 MACE-POLAR 重算能量/力”能得到什么，"
             "不能得到什么；以及 c1（一体直接相关函数）与 c2（二体直接相关函数）"
             "分别从哪里来。", 11)
        text(fig, 0.785, "参考：", 11, "#333333", "bold")
        text(fig, 0.755,
             "• Dijkman et al., PRL 134, 056103 (2025)：pair-correlation matching。\n"
             "• Bui & Cox, PNAS 123, e2610049123 (2026)：水的 O 密度 cDFT。\n"
             "• Hansen & McDonald, Theory of Simple Liquids：OZ 关系。", 10)
        text(fig, 0.665, "符号：", 11, "#333333", "bold")
        text(fig, 0.635,
             "x：全部原子坐标；r：空间位置；β = 1/(k_B T)；\n"
             "V_A、V_P：MACE-OFF、MACE-POLAR 势能；\n"
             "ρ(r)：一体的数密度；g(r)：径向分布函数；\n"
             "c1(r)：一体直接相关函数；c2(r,r′)：二体直接相关函数。", 10)
        text(fig, 0.50,
             "核心结论：\n"
             "密度不是单帧的量，而是系综平均；\n"
             "能量偏移只能平移 c1 的零点，对 c2 没有作用；\n"
             "逐帧能量差可以做重加权，逐帧能量+力可以拟合代理势再采样。",
             12, "#111111", "bold")
        save(fig, pdf)

        # page 2
        fig = new_page("1. 系综、配分函数与密度")
        text(fig, 0.885, "两个力场的 Boltzmann 分布：", 11, "#333333", "bold")
        eq(fig, 0.835, r"$p_A(x)=\frac{e^{-\beta V_A(x)}}{Z_A}$")
        eq(fig, 0.745, r"$p_P(x)=\frac{e^{-\beta V_P(x)}}{Z_P}$")
        text(fig, 0.665, "配分函数：", 11, "#333333", "bold")
        eq(fig, 0.625, r"$Z_A=\int e^{-\beta V_A(x)}\,dx$")
        eq(fig, 0.545, r"$Z_P=\int e^{-\beta V_P(x)}\,dx$")
        text(fig, 0.465, "一体密度算符与密度：", 11, "#333333", "bold")
        eq(fig, 0.425, r"$\hat\rho(\mathbf{r})=\sum_{i=1}^{N}\delta(\mathbf{r}-\mathbf{r}_i)$")
        eq(fig, 0.345, r"$\rho_A(\mathbf{r})=\langle\hat\rho(\mathbf{r})\rangle_A$")
        eq(fig, 0.275, r"$\rho_P(\mathbf{r})=\langle\hat\rho(\mathbf{r})\rangle_P$")
        text(fig, 0.195, "归一化：", 11, "#333333", "bold")
        eq(fig, 0.155, r"$\int \rho_A(\mathbf{r})\,d\mathbf{r}=\int \rho_P(\mathbf{r})\,d\mathbf{r}=N$")
        save(fig, pdf)

        # page 3
        fig = new_page("2. 从 A 到 B：精确重加权")
        text(fig, 0.885, "重要性采样恒等式：", 11, "#333333", "bold")
        eq(fig, 0.825, r"$\langle O\rangle_P=\frac{\langle O\,e^{-\beta(V_P-V_A)}\rangle_A}{\langle e^{-\beta(V_P-V_A)}\rangle_A}$", 14)
        text(fig, 0.735, "配分函数比：", 11, "#333333", "bold")
        eq(fig, 0.695, r"$\frac{Z_P}{Z_A}=\langle e^{-\beta(V_P-V_A)}\rangle_A$")
        text(fig, 0.615, "密度的重加权：", 11, "#333333", "bold")
        eq(fig, 0.555, r"$\rho_P(\mathbf{r})=\frac{\langle\hat\rho(\mathbf{r})\,e^{-\beta(V_P-V_A)}\rangle_A}{\langle e^{-\beta(V_P-V_A)}\rangle_A}$", 13)
        text(fig, 0.455, "密度比：", 11, "#333333", "bold")
        eq(fig, 0.405, r"$\frac{\rho_P(\mathbf{r})}{\rho_A(\mathbf{r})}=\frac{\langle e^{-\beta(V_P-V_A)}\rangle_{A,\mathbf{r}}}{\langle e^{-\beta(V_P-V_A)}\rangle_A}$", 13)
        text(fig, 0.31,
             "其中尖括号表示在 A 系综中、固定一个粒子在 r 的条件平均。", 10)
        text(fig, 0.255,
             "一般地：", 11, "#333333", "bold")
        eq(fig, 0.205, r"$\frac{\rho_P(\mathbf{r})}{\rho_A(\mathbf{r})}\neq e^{-\beta(V_P(\mathbf{r})-V_A(\mathbf{r}))}$", 15)
        text(fig, 0.125,
             "指数因子必须在系综平均里面；不能逐点乘在 ρ_A(r) 上。", 10, "#333333")
        save(fig, pdf)

        # page 4
        fig = new_page("3. 常数偏移与常见误解")
        text(fig, 0.885, "若两个势只差常数 ΔE：", 11, "#333333", "bold")
        eq(fig, 0.835, r"$V_P(x)=V_A(x)+\Delta E$")
        eq(fig, 0.755, r"$p_P(x)=p_A(x)$")
        eq(fig, 0.685, r"$\rho_P(\mathbf{r})=\rho_A(\mathbf{r})$")
        eq(fig, 0.615, r"$c_2^{P}(\mathbf{r},\mathbf{r}')=c_2^{A}(\mathbf{r},\mathbf{r}')$")
        eq(fig, 0.545, r"$c_1^{P}(\mathbf{r})=c_1^{A}(\mathbf{r})-\beta\Delta E$")
        text(fig, 0.465, "此时：", 11, "#333333", "bold")
        eq(fig, 0.425, r"$\int e^{\beta\Delta E}\rho_P(\mathbf{r})\,d\mathbf{r}=e^{\beta\Delta E}N$")
        text(fig, 0.345, "一般情况：", 11, "#333333", "bold")
        eq(fig, 0.295, r"$\int e^{\beta(V_A-V_P)}\rho_P\,d\mathbf{r}$")
        text(fig, 0.225, "没有等于 N 或 ρ_A 的简单恒等式；", 10)
        text(fig, 0.195, "因为 V_A−V_P 是构型相关的，不能提到积分号外。", 10)
        text(fig, 0.135,
             "能量偏移的真实作用：对齐 c1 的零点；c2 完全不受影响。", 11, "#111111", "bold")
        save(fig, pdf)

        # page 5
        fig = new_page("4. c1 与 c2 分别从哪里来")
        text(fig, 0.885, "c1：一体直接相关函数", 12, "#333333", "bold")
        eq(fig, 0.835, r"$c_1(\mathbf{r})=-\beta\frac{\delta F_{\mathrm{exc}}[\rho]}{\delta\rho(\mathbf{r})}$")
        text(fig, 0.755, "均匀体相：", 10)
        eq(fig, 0.715, r"$c_{1,\mathrm{ex}}=-\beta\mu_{\mathrm{ex}}$")
        text(fig, 0.645, "非均匀体相（Euler–Lagrange）：", 10)
        eq(fig, 0.595, r"$c_1(\mathbf{r})=\ln[\Lambda^3\rho(\mathbf{r})]-\beta\mu+\beta V_{\mathrm{ext}}(\mathbf{r})$", 12)
        text(fig, 0.495, "c2：二体直接相关函数", 12, "#333333", "bold")
        eq(fig, 0.445, r"$c_2(\mathbf{r},\mathbf{r}')=-\beta\frac{\delta^2F_{\mathrm{exc}}}{\delta\rho(\mathbf{r})\delta\rho(\mathbf{r}')}=\frac{\delta c_1(\mathbf{r})}{\delta\rho(\mathbf{r}')}$", 12)
        text(fig, 0.345, "均匀体相：h(r)=g(r)−1，", 10)
        eq(fig, 0.295, r"$c_2(r)=\frac{1}{2\pi^2}\int_0^\infty\frac{\sin(kr)}{kr}\frac{\hat h(k)}{1+\rho\hat h(k)}k^2\,dk$", 13)
        text(fig, 0.195,
             "c1 需要密度/化学势（或外场下的密度剖面）；\n"
             "c2 需要 g(r)（均匀体相）或密度剖面的二阶泛函导数。", 10)
        save(fig, pdf)

        # page 6
        fig = new_page("5. 用 A 的帧估计 P 的量：能做什么")
        text(fig, 0.885, "逐帧能量差 → 重加权：", 11, "#333333", "bold")
        eq(fig, 0.835, r"$w_i=e^{-\beta[V_P(x_i)-V_A(x_i)]}$")
        eq(fig, 0.755, r"$g_P(r)=\frac{\sum_i w_i\,g_i(r)}{\sum_i w_i}$")
        eq(fig, 0.665, r"$\mathrm{ESS}=\frac{(\sum_i w_i)^2}{\sum_i w_i^2}$")
        text(fig, 0.585,
             "ESS 是有效样本数；权重越不均，ESS 越小。", 10)
        text(fig, 0.535, "逐帧能量 + 力 → 代理势：", 11, "#333333", "bold")
        eq(fig, 0.485, r"$L=\sum_i(\Delta E_i-\Delta V_\theta(x_i))^2+\lambda\sum_i\|\Delta F_i-\nabla\Delta V_\theta(x_i)\|^2$", 11)
        text(fig, 0.395,
             "训练后在 V_A+ΔV_θ 上采样，再算 g(r)→c2，以及 Widom/密度→c1。", 10)
        text(fig, 0.335, "数值示例（96 原子水，36 生产帧）：", 10, "#333333", "bold")
        eq(fig, 0.295, r"$\sigma_\beta=3.89,\qquad \mathrm{ESS}=3.63/36=10.1\%$", 13)
        text(fig, 0.215,
             "1,296 原子外推：σ_β≈14.3；5,184 原子：σ_β≈28.6。\n"
             "大体系直接重加权会塌缩；需要直接跑目标轨迹或训练代理势。", 10)
        save(fig, pdf)

    print(out)


if __name__ == "__main__":
    main()
