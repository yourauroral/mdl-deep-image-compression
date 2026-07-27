# 理论：为什么线性探针在压缩训练的 AR 模型上有效
# Theory: Why Linear Probing Works in a Compression-Trained AR Model

本文将项目实验结果与其背后的理论联系起来，回答一个研究问题——**为什么一个仅以压缩像素为目标训练的模型，能让线性探针恢复语义类别？**——并以 LLM 可解释性领域的**线性表征假设（LRH, Linear Representation Hypothesis）**为框架加以阐释。

论证分三条支柱：

1. **条件码长（模型学到了什么）**：在预共享模型时，压缩目标迫使模型发现图像的生成结构。这里度量的是 `L(data|model)`；严格 MDL 还需计入 `L(model)`。
2. **LRH（为什么是线性可读的）**：架构（线性权重共享 head + 加性残差流）把这种结构压到**线性方向**上，因此线性分类器就够了。
3. **MDL probing（如何诚实地度量）**：原始探针准确率是弱信号；标签的 prequential/variational codelength 能同时体现提取代价和剩余不确定性，但必须明确是否包含探针模型成本。

一句话纲领：主任务度量预共享压缩模型下的像素条件码长，探针可进一步用标签 MDL 度量；二者都以比特报告，但不是未经说明的同一个完整 MDL quantity。

---

## 0. 项目一段话概览

**CC-iGPT** 是一个双尺度、条件式自回归（AR）无损图像压缩模型，直接在 RGB uint8 域上建模（无色彩前端、无有损 codec）。一个小型 **coarse** iGPT 将下采样后的图像压入独立 bitstream；其量化 token 经反量化、双线性上采样、重新 tokenize 后，以 `α · coarse_ctx` 的形式**加性注入** **fine** iGPT 的残差流。fine 模型用 256-way softmax 预测每个子像素 token `p(x_t | x_{<t})`。在预共享模型和均匀首 token 先验下，理想条件码长为：

```
bpd_total = [8 + CE_coarse·(N_coarse-1)/ln2
             + 8 + CE_fine·(N_fine-1)/ln2] / N_fine
```

fine 架构：N=32 层，d=448，h=7，SwiGLU FFN，RoPE base=500000，QK-Norm，OLMo2 后归一化
（`x = x + RMSNorm(sublayer(x))`），权重共享 256-way 输出头，子像素 AR（pixel-first `[R0,G0,B0,R1,...]`），z-loss。总参数约 82.95M（fine ≈ 78M + coarse ≈ 4.81M + α）。

**历史实验结果（diagnostic protocol，不是正式单模型主表）：**

| 数据集     | 设置                                   | bpd        | 位次                                                          |
|------------|----------------------------------------|------------|---------------------------------------------------------------|
| CIFAR-10   | ensemble(best+SWA+EMA) + TTA hflip     | 2.8296 | 三成员、6 forwards/image；正式单 checkpoint/no-TTA 结果待重评 |
| ImageNet64 | ensemble(best+SWA+EMA) + TTA hflip     | 3.4800 | 三成员、旧 batch-level std 作废；正式结果待单模型重评并记录数据 hash |

**下游研究项**：线性探针、图像补全（AR inpainting）、真实无损算术编解码 roundtrip（bit-identical）。旧 probe 曲线曾使用 test 选层，只作为探索性证据；正式结果需按 validation 选层协议重跑。

---

## 1. 线性探针实际度量的是什么

冻结模型将图像 `x` 映射为隐状态 `h`，线性探针训练 `W` 从 `h` 预测类别 `y` 并报告 top-1 准确率。

直觉的解读——"高准确率意味着类别*存在于*表征中"——是错的，而且错的方式很重要。因为 `h` 是 `x` 的确定性函数，`x` 已经决定了 `y`，**数据处理不等式（data-processing inequality）**表明 `h` 关于 `y` 的信息永远不可能多于原始像素（Pimentel et al. 2020）。信息*始终*都在。所以探针不检验**信息是否存在**，而是检验**可提取性（extractability）**——具体地说，是在*线性*预测族下的可提取性（V-usable information，Xu et al. 2020）。

这将研究问题精确地重新表述为：

> 不是"类别信息是否存在？"（显然是），而是
> **"为什么像素压缩目标重新组织了原始 RGB，使类别落在*线性可读*的方向上？"**

这拆分为两半。§2（MDL）解释为什么*语义*会涌现。§3（LRH）解释为什么它们*线性地*涌现——这才是真正有趣的一半，也是与 LLM 可解释性共享的一半。

---

## 2. 为什么语义会涌现——MDL 部分

为减小 `L(data|model)`，模型需要捕捉使像素可预测的结构：物体边界、表面、光照、纹理和全局布局。这些因素可能也有利于类别预测，因此产生“条件码长与线性可提取性相关”的可检验假设。它不是“压缩越好必然学习越好”的定理，更不能省略模型成本后直接等同于完整 MDL。

两个项目特有的放大因素：

- **子像素 AR**：布局 `[R,G,B]` 迫使模型学 `p(G|R, ctx)` 和 `p(B|R,G, ctx)`。对像素内通道依赖建模，将模型推向产生颜色的物理因子（光照、材质）——而这些因子正好携带语义。
- **粗尺度条件注入（coarse conditioning）**：`α · coarse_ctx` 给 fine 模型一个低频全局先验（模糊缩略图）。为了利用它，fine 模型将计算组织在全局场景结构上，而不只是局部像素统计。

这一半是标准且有据可查的，已是项目 §6 叙述的主轴。它解释了**语义**，但对**线性性**只字未提。

---

## 3. 为什么是线性可读的——LRH 部分（核心）

线性表征假设：高层概念以**线性方向**编码在激活空间中；概念的存在 ≈ 在某方向上的投影；概念强度 ≈ 幅值。在本模型中，四种压力产生了这种几何结构——前三种是**架构特有的**，这正是本文构成一个贡献而非仅是引用的原因。

**1. 线性权重共享输出头（Linear, weight-tied readout）**：输出头是一个 `nn.Linear(d, 256)`，其权重与 `token_embed` *绑定共享*——最终隐状态到 logit 之间**零非线性**（`igpt.py:74`）。为了让模型输出正确的下一像素分布，预测相关的结构必须在最后一层被线性可读。由于类别对像素统计有强预测力，梯度下降将类别相关结构推到线性方向上。这是 LLM 中"线性 unembedding ⇒ 线性概念方向"压力在图像域的类比（Park, Choe & Veitch 2024）。权重共享使 embedding ≡ unembedding——他们因果内积几何的一个干净特例。

**2. 加性残差流（Additive residual stream）**：OLMo2 后归一化意味着每一层都*加入*共享流：表征字面上是各层贡献的累积和。线性方向因此是流的**原生格式**，线性探针直接读取该格式。这正是 Anthropic 残差流/叠加图景（Elhage et al. 2021/2022），本模型的流在结构上与 Transformer LM 的完全相同——所以同样的几何应该、且显然确实涌现了。

**3. 叠加（Superposition）**：d=448，但有用"概念"的数量远超 448。在高维空间中随机向量近似正交，模型将许多特征以近正交线性方向打包（Toy Models of Superposition）。探针恢复其中之一。

**4. `α · coarse_ctx` 是加性上下文向量**：粗尺度通路的输出被加入第 0 层 token embedding（`cc_igpt.py:164`，`igpt.py:134`），因此可以研究其沿残差流传播的线性效应。但旧 `79.33%` 与 iGPT-S `66.93%` 来自不同深度、宽度和训练预算，不能把差值 `+12.4pp` 因果归因于 coarse context；匹配训练消融完成前只作为研究假设。

### 3.1 中层峰值（倒 U 形曲线）

旧逐层探索曲线呈先升后降，但层号由 test curve 选择，需按新协议重跑后确认：

- **CIFAR v2 native（旧协议）**：L19 = 79.33%（共 32 层）。
- **IN64→CIFAR transfer（旧协议）**：L0 41.69% → L16 73.19% → L31 64.79%。

这个曲线形状是将本模型与 iGPT 及 LLM 联系起来的指纹：

- **浅层**保存局部像素统计 → 低抽象级 → 准确率低。
- **中间层**达到最大任务无关语义抽象 → 线性可分性峰值。
- **深层重新专化（re-specialize）**为 256-way *输出*分布，丢弃类别通用语义以锐化局部下一像素预测 → 准确率下降。

这是信息瓶颈（Information Bottleneck）/"隧道（tunnel）"动态（Tishby；Shwartz-Ziv & Tishby 2017）。**同样的倒 U** 出现在 iGPT（Chen et al. 2020）和 LLM probing 中（中间层对语义任务探针效果最好）。这种跨模态巧合回答了*"为什么这像 LLM 解释性？"*：

> 表征几何由**目标的形式**（下一 token 预测 + 线性头）和**架构**（加性残差流）驱动，**而非由模态决定**。像素和 token 收敛到相同的结构。

---

## 4. 如何诚实地度量——MDL probing 部分

原始准确率是弱度量：足够强的探针可以*记忆*任务，因此高准确率不一定反映表征质量（Hewitt & Liang 2019）。与本项目 MDL 论点完美契合的修正方案，是度量**特征条件下标签的描述长度**（Voita & Titov 2020）。

**分解**：表征好 ↔ 标签能用少比特传输：

```
L_total = L_model + L_data
        = （探针的复杂度）+（剩余标签不确定性）
```

两种标准估计器：

- **变分编码（Variational code）**：对探针权重在先验 `p(θ)` 下学一个后验 `q(θ)`；编码长度为**上界**
  `L_var = KL(q‖p) + E[-log p(y|x,θ)]`（即负 ELBO）。KL = 模型描述长度；交叉熵 = 数据描述长度。一个仅需简单权重就能解的任务，后验会靠近先验（KL 小）；需要记忆才能解的任务，KL 爆炸。
- **在线（prequential）编码**（Blier & Ollivier 2018）：用小数据子集训练，用当前探针编码下一批数据（累计 `-log p`），扩大数据，重复。好的表征只需少量数据就能廉价编码未来样本 → 总编码长度小。这度量的是**学习效率**/**提取代价**，是最易手工实现的变体——一个在增长数据子集上的循环，无需新依赖。

**为什么 MDL 优于准确率**：在对照任务（随机标签）下，准确率可以保持高位（探针在记忆），但**编码长度爆炸**，因为随机标签不可压缩而真实结构能被少量数据的小探针捕获。MDL 因此能干净地将*"表征中的信息"*与*"探针记忆的信息"*分离——正是 §1 所要求的区分。

**为什么这契合项目**：主任务和 probing 都可以用比特报告，因此可在统一的信息论语言下讨论。但主任务当前是预共享压缩模型下的 `L(x|model)`，probe 若采用 prequential code 则包含不同的学习协议和模型代价；二者不能只因单位相同就视为同一个 MDL quantity。

---

## 5. 统一视角

对流水线 `x → h → y`，总描述长度为

```
L(h) + L(model) + L(y | h, model)
```

不同研究方向关注不同项——本质上是同一个信息论问题：*"任务信息存在哪里，能以多少比特描述？"*

| 项               | 方向                                      | 本项目                      |
|------------------|-------------------------------------------|-----------------------------|
| `L(h)`           | 稀疏编码 / **SAE**                         | 旁支（仅作背景）             |
| `L(model)`       | 剪枝 / **Bayesian Compression**（Louizos 2017） | 旁支（仅作背景）         |
| `L(y\|h,model)`  | **MDL probing**                           | **本文工作所在的角落**       |

SAE（*表征*空间中的稀疏性）和 Bayesian Compression（*参数*空间中的稀疏性）是有用的背景，但与线性探针论文**正交**。项目的贡献在 `L(y|h,model)`——而 §3 解释了为何其所条件化的 `h` 使这一项**线性地**变小。

---

## 6. 阅读清单 (Reading List)

与 `future.md §10` 阅读路径使用同一 **★（易）→ ★★★★★（难）** 标准，两份清单可在一个量表上读。**⭐ = 直接回答研究问题的核心 5 篇**；**粗体** = 对应节的主干引用。标题/年份来自记忆，可信度高——**引用进论文前务必核对 venue/arXiv 编号**（本清单未经引用核查）。

**A. 探针度量的是什么（方法论——先读）**

| 论文 | 回答什么 · 项目关联 | 难度 |
|---|---|---|
| ⭐ Alain & Bengio 2017, *Understanding intermediate layers using linear classifier probes* (ICLR-W) | 技术来源——字面意义上就是 `linear_probe.py` 的实现 | ★★ |
| ⭐ Hewitt & Liang 2019, *Designing and Interpreting Probes with Control Tasks* (EMNLP) | 必须回答的批评；**selectivity / control tasks** = 实验 **E3** | ★★★ |
| Pimentel et al. 2020, *Information-Theoretic Probing for Linguistic Structure* (ACL) | 探针即互信息 → §1 "**可提取性而非存在性**" | ★★★★ |
| **Voita & Titov 2020, *Information-Theoretic Probing with MDL* (EMNLP)** | §4 主干——标签**编码长度** = 与 bpd 相同的 MDL 标尺；驱动 **E6** | ★★★ |
| Blier & Ollivier 2018, *The Description Length of Deep Learning Models* (NeurIPS) | E6 中实现的**在线/prequential 编码**来源 | ★★★★ |
| Xu et al. 2020, *A Theory of Usable Information (V-information)* (ICLR) | 为什么线性探针数字 = 线性族下的 **V-usable info**（§1）| ★★★★ |

**B. 为什么概念是线性的（LRH 部分——核心）**

| 论文 | 回答什么 · 项目关联 | 难度 |
|---|---|---|
| ⭐ Elhage et al. 2021 *A Mathematical Framework for Transformer Circuits* + 2022 *Toy Models of Superposition*（Anthropic）| 残差流叠加——与本模型 OLMo2 加性流**结构相同**（§3 压力 #2–#3）| ★★★ |
| ⭐ Park, Choe & Veitch 2024, *The Linear Representation Hypothesis and the Geometry of LLMs* (ICML) | 正式 LRH；**权重共享头** = 他们 embedding ≡ unembedding 的特例（§3 压力 #1）| ★★★★ |
| Jiang et al. 2024, *On the Origins of Linear Representations in LLMs* | 线性性源于 **NTP 目标**——直接迁移到像素 NTP | ★★★ |
| Engels et al. 2024, *Not All Language Model Features Are Linear* | 诚实的反例（有些特征是圆形/流形的）——引用让 LRH 论断读起来是"强近似"而非定律 | ★★★ |

**C. 中层峰值（倒 U 形）**

| 论文 | 回答什么 · 项目关联 | 难度 |
|---|---|---|
| ⭐ Chen et al. 2020, *Generative Pretraining from Pixels (iGPT)* (ICML) | 直系祖先；其中层探针峰值**就是实测曲线**（L19 native / L16 transfer）| ★★ |
| Tishby & Zaslavsky 2015 / Shwartz-Ziv & Tishby 2017, *Information Bottleneck* | **深层准确率下降**背后的"压缩—再专化"动态（§3.1）| ★★★★ |
| 2024 work, *"intermediate layers give the best representations in LLMs"*（搜此短语）| 确认峰值是**跨模态**的，不是像素特有的偶然 | ★★ |

**D. 线性方向是因果的（Steering）**

| 论文 | 回答什么 · 项目关联 | 难度 |
|---|---|---|
| Turner et al. 2023 *ActAdd* · Zou et al. 2023 *Representation Engineering* · Anthropic 2024 *Scaling Monosemanticity* | 支持将 `α · coarse_ctx` 视为工程化 **steering vector**（§3 压力 #4）；激励实验 **E4** | ★★★ |

**最短路径——只读这四篇即可破题**：Jiang 2024（NTP→线性）+ Elhage 2022（残差流/叠加）+ iGPT 2020（像素域中层峰值）+ Hewitt & Liang 2019（证明是真信息，不是记忆）。评为 ★★★★ 的是*深度，不是前提*——第一轮可以跳过。

---

## 7. 实验菜单（断言 → 证据）

所有实验复用 `scripts/linear_probe.py`，成本低廉，在 AutoDL 上运行（本文在 WSL 编写——绝不在 WSL 上运行训练/探针）。按叙事价值排序。

- **E2 — 压缩与探针准确率的跨 checkpoint 相关性**：使用 `epoch_6..12.pth`，绘制各 checkpoint 的 val bpd 与 validation-selected probe accuracy。它能提供相关性证据，但 checkpoint 共享训练时间这一混杂因素，不能单独证明因果关系。
- **E6 — MDL / 在线编码探针**：用标签 prequential codelength 补充准确率，对 v2 / iGPT-S / 随机初始化和随机标签对照分别报告，并完整记录数据顺序、首段标签先验与 probe 模型成本。
- **E1 — 线性 vs MLP 探针差距**：加一层单隐层探针。差距小 ⇒ 信息确实*线性*编码（LRH 成立）；差距大 ⇒ 纠缠。核心 LRH 证据。
- **E3 — 对照任务 / 选择性（selectivity）**（Hewitt & Liang）：对随机标签做探针；报告 selectivity = 真实准确率 - 对照准确率。反驳"探针只是学了任务"。
- **E5 — coarse_ctx 消融**：现有 `--no_coarse_ctx` 只能做推理时干预；若要得到训练因果结论，还需相同架构、数据、预算和 seed 的 matched retraining ablation。
- **E4 — Steering**（*工作量最大，可解释性故事最好*）：用探针权重向量作为类别方向，在图像补全时将其加入残差流，检查输出是否向该类别偏移。若是，方向具有**因果性**——图像域的 LLM activation steering 类比。

---

## 8. 一句话总结

模型仅以像素条件码长为目标训练；线性权重共享头和加性残差流提供了形成线性可读特征的结构性压力。中层峰值、coarse context 的贡献以及条件码长与 probe 质量的关系仍是需要按无泄漏协议验证的实验假设。标签 codelength 是比单一准确率更完整的补充指标，但必须单独说明其编码协议。

*条件码长、线性可提取性与编码成本：分别定义、分别测量，再检验它们之间的关系。*
