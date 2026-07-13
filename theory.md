# 理论：为什么线性探针在压缩训练的 AR 模型上有效
# Theory: Why Linear Probing Works in a Compression-Trained AR Model

本文将项目实验结果与其背后的理论联系起来，回答一个研究问题——**为什么一个仅以压缩像素为目标训练的模型，能让线性探针恢复语义类别？**——并以 LLM 可解释性领域的**线性表征假设（LRH, Linear Representation Hypothesis）**为框架加以阐释。

论证分三条支柱：

1. **MDL（模型学到了什么）**：压缩目标迫使模型发现图像的生成结构。*压缩越好 → 学习越好。*
2. **LRH（为什么是线性可读的）**：架构（线性权重共享 head + 加性残差流）把这种结构压到**线性方向**上，因此线性分类器就够了。
3. **MDL probing（如何诚实地度量）**：原始探针准确率是弱信号；**标签的编码长度（codelength）** 才是严格度量——且与主表使用的是*同一把* MDL 标尺（比特）。

一句话纲领：**"MDL all the way down"**：主任务压缩像素，探针评估压缩标签，两者均以比特度量。

---

## 0. 项目一段话概览

**CC-iGPT** 是一个双尺度、条件式自回归（AR）无损图像压缩模型，直接在 RGB uint8 域上建模（无色彩前端、无有损 codec）。一个小型 **coarse** iGPT 将下采样后的图像压入独立 bitstream；其量化 token 经反量化、双线性上采样、重新 tokenize 后，以 `α · coarse_ctx` 的形式**加性注入** **fine** iGPT 的残差流。fine 模型用 256-way softmax 预测每个子像素 token `p(x_t | x_{<t})`。交叉熵*即*最优编码长度，因此：

```
bpd_total = (CE_coarse · N_coarse + CE_fine · N_fine) / ln2 / N_fine
```

fine 架构：N=32 层，d=448，h=7，SwiGLU FFN，RoPE base=500000，QK-Norm，OLMo2 后归一化
（`x = x + RMSNorm(sublayer(x))`），权重共享 256-way 输出头，子像素 AR（pixel-first `[R0,G0,B0,R1,...]`），z-loss。总参数约 82.95M（fine ≈ 78M + coarse ≈ 4.81M + α）。

**主表结果：**

| 数据集     | 设置                                   | bpd        | 位次                                                          |
|------------|----------------------------------------|------------|---------------------------------------------------------------|
| CIFAR-10   | ensemble(best+SWA+EMA) + TTA hflip     | **2.8296** | 超越 PixelSNAIL 380M (2.85)；逼近 Sparse Transformer 59M (2.80) |
| ImageNet64 | ensemble(best+SWA+EMA) + TTA hflip     | **3.4800** | 超 SPN (3.52)；逼近但未达 Sparse Transformer 152M (3.44)       |

**下游证据（MDL 主线，论文 §5）**：线性探针、图像补全（AR inpainting）、真实无损算术编解码 roundtrip（bit-identical）。本文是**第一项**的理论支撑。

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

为最小化 `p(x_t | x_{<t})` 的描述长度，模型必须捕捉*使像素可预测的*结构：物体边界、表面、光照、纹理、全局布局。一个数据集的最短编码，由对其真实生成因子的建模实现——而这些因子恰好也决定了类别。由此得到项目的核心论断：**压缩越好 → 学习越好**，这正是最小描述长度原则（MDL，Rissanen）的具体实例：最好的压缩器就是最好的学习器。

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

**4. `α · coarse_ctx` 是字面意义上的线性概念向量**：粗尺度通路的输出被*加入*第 0 层的 token embedding（`cc_igpt.py:164`，`igpt.py:134`）。它以**加性方式注入残差流**——即一个工程化的 steering vector，同时还改善了压缩性能。这是项目最强的 LRH 论点，也预测了 coarse_ctx 注入在中层探针准确率上**提升 +12.4 pp** 这一实验事实（CIFAR v2: L19 79.33% vs iGPT-S L22 66.93%）。

### 3.1 中层峰值（倒 U 形曲线）

逐层准确率的实测：先升后降——

- **CIFAR v2 native**：峰值 **L19 = 79.33%**（共 32 层）。
- **IN64→CIFAR transfer**：L0 41.69% → **L16 73.19%** → L31 64.79%（完整 32 层；L15–L17 平台区）。

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

**为什么这契合项目**：主表报告 bits-per-dim（压缩像素）。把探针准确率替换成标签编码长度，意味着*下游评估与主表使用同一把 MDL 标尺*（压缩标签）。整个故事闭环：**主任务 = MDL，探针 = MDL，均以比特度量。**

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

- **E2 — 压缩与探针准确率的跨 checkpoint 相关性**（*最高价值*）：使用 `epoch_6..12.pth`，绘制各 checkpoint 的 val bpd vs 最优层探针准确率。紧密单调反相关 = **"压缩越好 → 学习越好"的直接实证**。一张图，收益大。
- **E6 — MDL / 在线编码探针**（*最契合论文主线*）：用**标签编码长度**替代探针准确率（在线编码：对增长数据子集编码，累计 `-log p`）。对 v2 / iGPT-S / 随机初始化 + 随机标签对照各报告。手工可实现，无新依赖，且将探针置于**与主表相同的 MDL 标尺**上——"MDL all the way down"。
- **E1 — 线性 vs MLP 探针差距**：加一层单隐层探针。差距小 ⇒ 信息确实*线性*编码（LRH 成立）；差距大 ⇒ 纠缠。核心 LRH 证据。
- **E3 — 对照任务 / 选择性（selectivity）**（Hewitt & Liang）：对随机标签做探针；报告 selectivity = 真实准确率 - 对照准确率。反驳"探针只是学了任务"。
- **E5 — coarse_ctx 消融**（*已可直接跑——`--no_coarse_ctx` 已存在*）：逐层量化加性方向的贡献。检验压力 #4。
- **E4 — Steering**（*工作量最大，可解释性故事最好*）：用探针权重向量作为类别方向，在图像补全时将其加入残差流，检查输出是否向该类别偏移。若是，方向具有**因果性**——图像域的 LLM activation steering 类比。

---

## 8. 一句话总结

模型仅以压缩像素为目标训练，但加性残差流上的线性权重共享头迫使它必须学习的生成因子（MDL）落在**线性方向**上（LRH），在中间层达到峰值——因此线性探针免费读出类别；而诚实评分该探针的方式是**标签编码长度**，下游评估因此落在与主表完全相同的 MDL 标尺上。

*压缩越好 → 学习越好 → 线性读出 → 以比特度量。*
