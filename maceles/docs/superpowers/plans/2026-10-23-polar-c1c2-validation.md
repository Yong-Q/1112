# POLAR c₁/c₂ 重加权修复与验证计划

> **For agentic workers:** 使用 `superpowers:executing-plans` 逐项实施；只有用户明确要求时才委派子智能体。下列未勾选项是待完成工作，不表示已实现或已运行。

**Goal:** 修正已复现的软件错误，明确统计量和分子参考定义，再判断参考势采样能否提供有不确定度的目标POLAR c₁/c₂。

**Architecture:** 保留现有 `sampling/` 接口与历史输出。按输入/系综校验、RDF、Widom、OZ、统计审核分别修复；以独立反例和解析参考验收后，再运行新的匹配采样。报告允许给出“不适用”的结果，不预设重加权成功。

**Tech Stack:** Python、NumPy、ASE、SciPy、pytest；后续模型评估使用现有MACE计算器和计算节点。

**Spec:** [检查证据与验收范围](../../AUDIT_2026-10-23.md)，尤其F1–F9。[原始复算JSON](../../../validation/results/audit_20261023.json)记录检查基线和输入哈希。

## Global Constraints

- 当前发布范围是问题、计划和CPU审计证据；已合入远端865f203的RDF体积修复，其余代码/统计验收仍待实施。远端另报告500 ps运行，进度与资格本次未核验。
- 首轮为相同粒子数、体积、原子表示、约束及300 K温度的NVT。实际采样Hamiltonian与用于重加权的源能量定义一致。
- 使用整帧总能量差；同一外场可以抵消，外场不同时须计入完整外场差。保留构型相关能量残差，不通过删点、截权重或改β提高ESS。
- BZA分子标签优先采用COM密度；O–O结构/有效投影通道另行注明。c₁无量纲、c₂(r)无量纲、c₂(k)为Å³。
- 固定N数据不冒称巨正则k=0涨落；中心化剖面不冒称绝对化学势或EOS。刚性与柔性分子的理想内部参考不得混用。
- 历史数据与报告保留，新输出使用新目录及模型/输入/代码哈希。未通过门槛的标签标为不可用。

## Review Focus

1. 温度元数据与分析β不一致、源模型哈希缺失：任务1须拒绝认证。
2. RDF半径超出完整球壳最小镜像范围、零/负/非有限权重：任务2须显式拒绝。
3. 重叠插入、不等数量插入及柔性参考未定义：任务3不得条件化平均或生成合格c₁。
4. 半格点径向网格、有限截尾及非正结构因子：任务4不得输出静默有效的c₂。
5. 初始非平衡帧、强自相关、单帧主导：任务5须保留诊断和失败状态，不能仅凭ESS比例判“可用”。

---

## Task 1 — P0：统一系综、能量与输入来源

**Files:** 修改 `sampling/off_polar.py`、`sampling/trajectory.py`、`sampling/cli.py`、`scripts/monitor_ess.py`；新增 `tests/test_ensemble_contract.py`。

**Interfaces:** 保留 `ReweightConfig.temperature_K` 为首轮目标/采样共同温度；在输出manifest补充采样设定温度、动力学温度统计、系综、实际源引擎与精度、原子表示、约束、完整模型哈希和轨迹时间范围。缺来源的旧数据可以诊断，但不能获得合格标签状态。

- [ ] 先写失败测试：`test_analysis_temperature_must_match_source_metadata`、`test_missing_model_provenance_cannot_certify`、`test_total_energy_not_per_atom_enters_weights`、`test_nonfinite_energy_is_rejected`。
- [ ] 按轨迹/采样配置校验300 K，而非用平均瞬时温度替代β；334 K旧结果保留为敏感性案例。若未来支持不同温度，使用 `logw=β_source*U_source-β_target*U_target` 并另行验收。
- [ ] 保存实际动能温度及配置时间序列；核对FP32源引擎与离线源能量评估的等价性和误差。不能仅凭模型同名当作同一采样Hamiltonian。
- [ ] 运行 `python -m pytest tests/test_ensemble_contract.py -v`，上述不一致均被拒绝后提交此项修复。

## Task 2 — P0：修复RDF体积与几何范围

**Files:** 修改 `sampling/off_polar.py::OffPolarReweighter.rdf`；新增 `tests/test_rdf.py`；更新相关CLI默认值/输入校验。

**Interfaces:** 保留 `rdf(frames, weights, pair, rmax, bins)` 的返回形式；首轮认证仅支持固定N/V的正交周期盒、非负有限权重且总权重大于0。

- [ ] 先写失败测试：`test_uniform_gas_rdf_is_one`（固定种子，20 Å盒、rmax=6 Å）、`test_shell_uses_cubic_volume`、`test_rmax_beyond_half_box_rejected`、`test_invalid_weights_rejected`、`test_variable_nvt_cell_or_particle_count_rejected`。
- [x] 球壳体积已由远端865f203改为 `4π/3*(r_hi**3-r_lo**3)`，并新增理想气体/配位数测试。
- [ ] 补充并注明当前N(N−1)的有限N归一化及与OZ关系的转换约定。
- [ ] 对当前最小镜像完整球壳算法强制 `rmax <= min(box_lengths)/2`。非正交盒在完成相应几何支持前拒绝认证；该限制不等同于断言MLIP计算器不能处理多周期镜像。
- [ ] 运行 `python -m pytest tests/test_rdf.py -v`，理想气体RDF均值在1附近、误差符合有限样本波动后提交。旧水RDF须在新目录重算，不能只修改图注。

## Task 3 — P0：修复Widom平均及分子理想参考

**Files:** 修改 `sampling/direct_correlation.py::c1_from_widom`、`validation/run_forcefield_c1c2.py`；新增 `tests/test_widom.py`。

**Interfaces:** 为 `c1_from_widom` 增加源帧 `frame_weights`、明确的分子参考约定和标签资格状态；每帧插入平均先按该帧的全部尝试数归一化，再按源帧权重组合。刚性孤立分子参考与柔性内部构型积分分开声明。

- [ ] 先写失败测试：`test_rejected_zero_factors_stay_in_denominator`（0、1、0、1应得到ln(1/2)）、`test_nonfinite_insertions_fail`、`test_frame_weights_are_used`、`test_rigid_isolated_reference_cancels_reference_energy`。
- [ ] 保留所有尝试在分母内；几何筛查不能未经验证当作严格零Boltzmann因子，非有限MLIP能量必须报错并保留证据。
- [ ] 稳定地累积插入对数权重。刚性测试先用一致的孤立探针参考；明确柔性MLIP还需理想分子内部构型的正确采样与归一化，不能用固定ASE水分子替代后声称得到完整柔性化学势。
- [ ] 柔性参考尚未合格时，均相绝对c₁标为 `unavailable_reference_not_qualified`；允许另走经过验证的软核/TI路径。外场中心化c₁路径不强制增加Widom。
- [ ] 运行 `python -m pytest tests/test_widom.py -v`，反例和参考约定均通过后提交；旧约8×10⁴的数值保留为未认证历史输出。

## Task 4 — P1：修复OZ网格并限定c₂含义

**Files:** 修改 `sampling/direct_correlation.py::c2_from_g`、`direct_correlation_report`；扩展 `tests/test_direct_correlation.py`；更新 `docs/DIRECT_CORRELATION.md`。

**Interfaces:** 计算函数保留原径向坐标和实际积分权重；报告补充坐标通道、网格、截尾方式、有限尺寸、有效频带与 `c2_status`。先加权关联统计，再进行OZ反演，不能平均各帧反演后的c₂。

- [ ] 先写独立测试：`test_midpoint_grid_is_not_shifted_to_zero`；使用已知高斯函数的解析三维傅里叶变换，检查原点网格与半格点网格随dr收敛，避免只用同一离散变换构造并反演测试数据。
- [ ] 增加 `test_nonpositive_structure_factor_is_unqualified`、`test_tail_and_box_truncation_are_reported`。非有限或非正的S(k)不得靠静默截断变为合格结果。
- [ ] 明确BZA的分子COM通道；O–O结果标为另外的结构/有效投影通道，不当作完整位置—取向分子c₂。固定N的k=0不作压缩率真值。
- [ ] 运行 `python -m pytest tests/test_direct_correlation.py -v`；在任务2正确RDF基础上，以新输出目录重算96原子历史数据。有限盒中尾部未收敛时只报告合格频带，不声称完整c₂(r)。

## Task 5 — P1：把ESS变为观测量误差审核

**Files:** 新增 `sampling/diagnostics.py`、`tests/test_diagnostics.py`；修改 `sampling/off_polar.py` 和 `scripts/monitor_ess.py`；新增只读结果审核入口。

**Interfaces:** `weight_diagnostics(log_weights)` 返回权重ESS、最大权重、前3/5帧权重和时间块质量；分块重采样以完整连续轨迹块为单位，每次重新归一化权重并重算密度/关联/OZ。不能给所有观测量共用一个“误差已合格”布尔值。

- [ ] 先写失败测试：`test_logweights_do_not_overflow`、`test_single_dominant_weight_is_flagged`、`test_nonfinite_logweights_rejected`、`test_repeated_correlated_frames_do_not_create_independence`、`test_monitor_csv_contains_actual_normalized_weights`。
- [ ] ESS稳定计算使用 `logw-max(logw)`，明确它不包含MD自相关；不通过删除异常大权重、截权重或只用拟合残差来提高ESS。
- [ ] 监测器增加显式预热边界，记录时间步与输入来源；修正CSV中一直为0的weight列。早期6帧和充分平衡生产段分开报告。
- [ ] 对c₁及各c₂频带分别给出连续分块/bootstrap不确定度和反演失败率。36帧不足以可靠估计长时间相关时应明确说明，不给伪精确的自相关修正ESS。
- [ ] 运行 `python -m pytest tests/test_diagnostics.py -v`；`ESS>=10%`不能单独赋予 `usable`，应输出“诊断可计算”和“标签已认证”两种不同状态。

## Task 6 — P1：修复后再开展匹配采样与可行性判定

**Files:** 更新 `validation/run_forcefield_c1c2.py`、`validation/compare_reweighted_vs_direct.py`；新增 `validation/configs/polar_reweight_pilot.json` 与每次运行的独立结果目录。

**Inputs:** 任务1–5的已验收实现、源/目标模型哈希、相同T/N/V与约束、明确COM或O–O通道；不能直接复用旧c₁/c₂数组当真值。

- [ ] 核验初始几何、模型覆盖、周期镜像计算和所选状态是否出现明显相分离/非均匀性。不能仅凭实验水相图断言这两套MLIP的状态相同。
- [ ] 在计算节点做阶段性独立OFF/POLAR采样，首批配置包含3个种子、300 K、初始20 ps平衡预算与100 ps生产预算、0.1 ps存帧；这些是起始预算，不是收敛保证。检查不通过则延长/改变设计，不能按“已跑100 ps”自动通过。
- [ ] 小体系先验收采样与权重；生产c₂另检查更大盒子及截尾收敛，不能把32个水分子的局部结果扩展为完整低k真值。新增规模和耗时按阶段门槛决定，不自动启动大型长轨迹。
- [ ] 对比未重加权OFF、重加权OFF和独立POLAR的密度、g(r)/S(k)、中心化或绝对c₁及合格c₂频带，给出逐状态不确定度和偏差。
- [ ] 若直接重加权仍由少数帧支配，先实际采样一个λ=0.5中间势；必要时增加λ=0.25/0.75并检查相邻重叠，再用BAR/MBAR。混合势的能量和力都须一致，只有重新打分同一批帧不算桥接采样。
- [ ] 只有模型参考、几何/有限尺寸与误差审核都通过，才导出训练标签；否则发布有证据的“不合格/需要桥接”结论。柔性绝对c₁未解决参考问题时保持不可用。

## Task 7 — P2：修订理论与报告，保留历史记录

**Files:** 修改 `README.md`、`docs/THEORY.md`、`docs/EXPERIMENTS.md`、`docs/DIRECT_CORRELATION.md`、`scripts/report/*.py`；新报告写入独立版本目录。

- [ ] 删除或重新推导与单调性相矛盾的erf覆盖公式；区分权重平方和ESS与Shannon熵，不把两者写成同一指标。
- [ ] 将10%/σβ阈值标为启发式；删除从6个初始帧直接推断“1000帧必为ESS=1”和“更多帧永远无效”的表述。
- [ ] 将旧c₂的实现错误与采样噪声分别说明；解析OZ测试通过不等于完整真实轨迹流程正确。对固定N总能常数与每分子能量参考的c₁影响分别推导。
- [ ] 链接新配置、输入哈希、分块误差、独立目标参照和失败案例。论文引用逐项核对，明确PRL是已知μ的GCMC，NVT绝对c₁另需化学势信息。
- [ ] 重新生成新的报告版本并核对图/表/公式/链接；旧报告保留并标注其被哪些检查发现限制。

## 执行顺序和完成标准

顺序：任务1 → 任务2/3 → 任务4 → 任务5 → 任务6 → 任务7；任务7中更正确定的错误描述可提前，但不得将未实现功能写成完成。

每项提交都附对应测试和真实数据范围。最终交付可以是“经审核的标签”或“在明确预算内不适用的实验结论”；不能只交付新的ESS阈值或数值看似平滑的曲线。当前计划已发布；RDF体积修复已合入，其他未勾选项仍未完成。
