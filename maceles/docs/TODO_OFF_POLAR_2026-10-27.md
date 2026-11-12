# TODO：OFF→POLAR逐帧校正、重加权与c₁/c₂验证

> **估计器文献与任务：** 四篇PDF及力采样、谱RDF、条件能量差和采样频率的具体工作见[估计器 TODO](TODO_ESTIMATOR_LITERATURE_2026-10-28.md)。

> **2026-10-27结论修订：** 不再以OFF旧帧上的学生评分判定学生生产尺寸失败；不把高斯外推10^41帧当真实算量。当前优先核验MD引擎与源势重算约0.030eV的去常数差异。见[修订报告](reports/REWEIGHT_VARIANCE_REVISED_2026-10-27.pdf)与[学生资格说明](ESS_STUDENT_TO_POLAR_2026-10-27.md)。旧PDF与原始数值保留。

本清单记录2026-10-27对501帧上传包的独立核查，以及对应论文的方法。基线提交为`d89ba5f`。**已验证、已修复与待完成项分别列出；低ESS不是已证明的程序错误，也不是已证明所有状态点都不能重加权。**

## 一、已经完成

- [x] 从501行源／目标整盒总能量独立复算：ESS=**2.3486739510**，与原始结果一致；σ(βΔU)=**13.76160816**。
- [x] 核对常数偏移：给全部ΔU加10000eV，归一化权重不变。整盒平均ΔU约480.864eV不是导致退化的原因；标准差约0.355765eV才影响配权。
- [x] 核对权重集中：403ps一帧占59.60%，前三帧共93.22%。应同时报告权重位置和时间分块，不能只报ESS。
- [x] 核对500ps生产温度：501个thermo记录均值299.840K、标准差7.123K。旧96原子案例的334K分析约定不能直接套到这批数据。
- [x] 比较PyTorch OFF与实际MD日志OFF：差值均值2.92214eV、去常数后标准差0.02998eV。以日志OFF能量作敏感性重加权仍为ESS≈2.08988；日志／坐标精度有限，不把该结果冒充严格校正。
- [x] 确认FP32整盒能量量化：约−799keV处ULP为**0.0625eV**，约2.42kBT；不是原上传说明中的约0.001eV。
- [x] 修复独立500ps脚本的RDF计数：上三角为N(N−1)/2对，归一化缺1/2。主模块之前的球壳修复与这次错误是不同问题。
- [x] 新增回归测试，对独立500ps helper与主模块RDF逐项对照；修复前失败，修复后15项现有测试全部通过。当前CPU环境缺pytest，本次逐项直接执行了全部无参数测试函数，未安装或更改依赖。
- [x] 从完整501帧重新计算配对直方图：正确g(r)与原存结果的2倍一致；新诊断结果单独保存，不覆盖旧数据。
- [x] 检查空间均匀性：时间平均O位点密度CV为x≈0.0117、y≈0.0239、z≈0.8876；z最小／平均≈0.00229，最大／平均≈1.965。与液体薄层及低密度区相符，不能直接解释为已确认的均相单相液体。

证据与诊断图见[结果目录](results/20261027_off_polar_audit/)。这是静态能量和轨迹复算，没有重新调用模型，也没有新MD或新的高精度c₁/c₂标签。

## 二、论文怎样“逐帧修正”

### 2.1 MLPT：校正能量，同时校正统计权重

对同一构型R_i：

    ΔU_i = U_H(R_i) − U_L(R_i)
    U_H(R_i) = U_L(R_i) + ΔU_i
    w_i = exp(−βΔU_i) / Σ_j exp(−βΔU_j)
    <A>_H ≈ Σ_i w_i A_H(R_i)

逐帧能量校正是合法步骤；校正后旧轨迹的构型出现频率仍属于源系综，因此需要权重。不能把每帧ΔU人为扣成零后等权平均，也不能任意按时间块分别去均值以抬ESS。

**Herzog等，JCTC2022，第2节Eq.(2)**：从约19000个构型中选200训练＋25独立测试，计算目标单点能，使用SOAP／核回归学习构型相关ΔU，预测其余构型能量再配权。它节约目标单点计算，不会自动增加源／目标分布重叠。论文验证对象为吸附焓，不是本项目c₁/c₂。

### 2.2 重叠不足：使用校正势重新采样

同文MLMC步骤：提出新构型→计算低层级能量→预测ΔU→用校正能量进行Metropolis接受／拒绝。作者约进行400000步MC；不新增目标层级单点，但仍需新构型的低层级能量计算。

    U_corr(R) = U_OFF(R) + f_theta(R)
    新轨迹来自U_corr，而不是原OFF轨迹
    最后残差 = U_POLAR(R) − U_corr(R)

若残差涨落小且覆盖充分，最后重加权的ESS才可能提高。用于MD时需可微能量与一致的力 `F_corr=−∇U_corr`；用于MC可只需要能量。代理势有误差时，必须用独立POLAR重评估检查残差，不能宣称代理采样等于精确目标采样。

### 2.3 水模型的直接案例

**Hilpert与Kresse，JCP2026，第III.5节**：63个水分子、292K，选5000个大致去相关构型，逐帧计算DFT−MLIP能量差，对体积配权。MACE ESS约4200，KbP不足800。其MACE目标为同一RPBE-D3 DFT，不能把该ESS外推到不同势之间的OFF23→POLAR。

## 三、P0：先完成输入与当前实现验收

- [ ] 在固定抽样构型上对比OFF FP32、OFF FP64、POLAR FP64与实际MD引擎的能量／力。使用统一构型、盒子、原子排序；分别报告常数差与去常数残差。
- [ ] 显式记录MACE／PyTorch／graph-longrange版本、模型SHA256、charge=0、spin=1、external_field及PBC/Ewald设置；本地尚无原模型权重的重新推理证据。
- [ ] 短对照检查邻居表：当前skin2Å、every100、checkno。用更频繁且可检查的邻居表策略核对能量／力及NVE漂移；1ps dump不足以证明100fs重建间隔内没有漏邻居。
- [ ] 确认要研究均相液态还是两相／界面。若目标为体相g(r)/c₂，先用明确液态密度／平衡体积建立均匀参考，不能把当前薄层结构直接当成均匀液体OZ输入。
- [ ] 修复Widom分母、重叠截断与非有限能量处理；定义分子理想内部参考和插入能的能量零点；实现源帧的正确加权插入。
- [ ] 修复OZ实际RDF半格点→r=0的网格错位，检查频带、尾部、有限尺寸、S(k)正性和独立非零S(k)一致性。
- [ ] 修订旧文档的错误表述：统一常数只影响截距，不会解释OLS斜率；`sigma_beta_residual`目前只是去常数后的量，不等于去线性拟合残差（后者本例约12.7264）。
- [ ] 删除或修订“增加帧数必然无效／ESS必饱和”的绝对结论。有限样本的ESS分数不是固定总体定律；更多样本能否实用地改善必须按覆盖、代价和观测量误差判断。

## 四、P1：Δ学习／蒸馏的受控对照

- [ ] 先选小体系、同N/T/V/约束的OFF和POLAR参考采样，避免先投入新的大体系长轨迹。
- [ ] 用构型相关描述符拟合ΔU(R)，按时间块及构型覆盖划分训练／验证／测试，不把相邻帧随机拆分的高R²当作泛化证据。
- [ ] 比较能量差代理、带一致力监督的Δ势和蒸馏POLAR；保留或明确近似目标长程项，不预设短程小模型能表达全部长程差异。
- [ ] 在校正势上新跑MD／MC，再用原POLAR重评残差；独立报告原OFF→POLAR和U_corr→POLAR两种ESS。
- [ ] 若使用多个lambda中间势，实际采样这些中间势后再用BAR/MBAR。给旧帧补算lambda能量不能补出缺失构型。

## 五、P2：按观测量验收

- [ ] 报告ESS、最大权重、权重时间分块、平衡检查与分块不确定度；ESS阈值只作诊断，不独立认证标签。
- [ ] 在相同状态点，用独立POLAR轨迹对照g(r)、S(k)及可观测量，而不只对照势能。
- [ ] 体相c₂：采用一致的密度中心和有效OZ关系；O–O位点标量核不能不加说明地当作完整位置—取向分子核。
- [ ] c₁绝对值：补目标化学势与分子理想参考。外场中心化c₁与体相绝对c₁分开验收。
- [ ] 非均匀c₂：保留密度涨落／响应并审核反演误差，不能由一个体相径向g(r)替代。

## 六、复现与引用

只读复算，输出目录必须不存在：

```bash
python validation/audit_off_polar_501.py --output runs/audit501-new
python validation/audit_off_polar_501_structure.py --output runs/audit501-structure-new
python -m pytest tests -q
```

本集群的数值核查经Slurm：109459能量／轨迹旧例复算完成后因环境无pytest退出；109464完整501帧结构复算成功；109463保留修复前回归反例；109465修复后15项测试成功。失败证据与旧结果不覆盖。

1. Herzog et al. **Assessing the Accuracy of Machine Learning Thermodynamic Perturbation Theory: Density Functional Theory and Beyond.** JCTC18,1382–1394(2022). [DOI](https://doi.org/10.1021/acs.jctc.1c01034) · [开放全文，第2节](https://arxiv.org/html/2110.06818)
2. Chehaibou et al. **Computing RPA Adsorption Enthalpies by Machine Learning Thermodynamic Perturbation Theory.** JCTC15,6333–6342(2019). [DOI](https://doi.org/10.1021/acs.jctc.9b00782)
3. Hilpert & Kresse. **Accurate thermophysical properties of water using machine-learned potentials.** JCP164,194504(2026). [DOI](https://doi.org/10.1063/5.0325444) · [公开全文，第III.5节](https://arxiv.org/html/2601.21103v1)
4. Thaler & Zavadlav. **Learning neural network potentials from experimental data via Differentiable Trajectory Reweighting.** Nature Communications(2021). [论文](https://www.nature.com/articles/s41467-021-27241-4)
5. Shirts & Chodera. **Statistically optimal analysis of samples from multiple equilibrium states.** JCP129,124105(2008). [DOI](https://doi.org/10.1063/1.2978177) · [开放全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC2671659/)

官方模型说明：[MACE-POLAR](https://mace-docs.readthedocs.io/en/latest/guide/polar_mace.html)、[MACE-OFF23](https://github.com/ACEsuit/mace-off)。POLAR加入显式长程静电且训练目标不同，不能仅凭名称认定它只是OFF的数值精度升级。
