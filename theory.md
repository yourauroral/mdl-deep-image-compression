# Theory: Why Linear Probing Works in a Compression-Trained AR Model

This note ties the project's empirical results to the theory behind them. It
answers one research question — **why does a linear probe recover semantic
classes from a model that was only ever trained to compress pixels?** — and
frames it the way LLM interpretability frames the *linear representation
hypothesis* (LRH).

The argument has three legs:

1. **MDL (what the model learns).** A compression objective forces the model to
   discover the generative structure of images. *Compress better → learn better.*
2. **LRH (why it's linearly readable).** The architecture (linear weight-tied
   head + additive residual stream) pushes that structure onto **linear
   directions**, so a linear classifier suffices.
3. **MDL probing (how to measure it honestly).** Raw probe accuracy is a weak
   signal; **codelength** of the labels given the features is the rigorous
   metric — and it is the *same* MDL ruler used for the main table.

The slogan is **"MDL all the way down"**: the main task compresses pixels, the
probe evaluation compresses labels, and both are measured in bits.

---

## 0. The project in one paragraph

**CC-iGPT** is a dual-scale, conditional autoregressive (AR) lossless image
compressor operating directly on RGB uint8 (no color front-end, no lossy
codec). A small **coarse** iGPT compresses a downsampled image into an
independent bitstream; its quantized tokens are dequantized, bilinearly
upsampled, re-tokenized, and injected **additively** into the **fine** iGPT's
residual stream as `α · coarse_ctx`. The fine model predicts every sub-pixel
token `p(x_t | x_{<t})` with a 256-way softmax. Cross-entropy *is* the optimal
code length, so:

```
bpd_total = (CE_coarse · N_coarse + CE_fine · N_fine) / ln2 / N_fine
```

Architecture (fine): N=32 layers, d=448, h=7, SwiGLU FFN, RoPE base=500000,
QK-Norm, OLMo2 post-norm (`x = x + RMSNorm(sublayer(x))`), weight-tied 256-way
output head, sub-pixel AR (pixel-first `[R0,G0,B0,R1,...]`), z-loss. Total
≈ 82.95M params (fine ≈ 78M + coarse ≈ 4.81M + α).

**Main-table results:**

| Dataset  | Setting                              | bpd     | Position |
|----------|--------------------------------------|---------|----------|
| CIFAR-10 | ensemble(best+SWA+EMA) + TTA hflip   | **2.8296** | beats PixelSNAIL 380M (2.85); approaches Sparse Transformer 59M (2.80) |
| ImageNet64 | ensemble(best+SWA+EMA) + TTA hflip | **3.4800** | beats SPN (3.52); approaches (not reaches) Sparse Transformer 152M (3.44) |

**Downstream evidence (MDL main line, paper §5):** linear probe, image
completion (AR inpainting), and a real lossless arithmetic-codec roundtrip
(bit-identical). This document is the theory behind the **first** of those.

---

## 1. What a linear probe actually measures

A frozen model maps an image `x` to a hidden state `h`. A linear probe trains
`W` to predict class `y` from `h` and reports top-1 accuracy.

The naive reading — "high accuracy means the class is *in* the representation" —
is wrong, and the reason matters. Because `h` is a deterministic function of
`x`, and `x` already determines `y`, the **data-processing inequality** says `h`
can never contain *more* information about `y` than the raw pixels do
(Pimentel et al. 2020). The information is *always* there. So the probe does not
test **presence**; it tests **extractability** — specifically, extractability
under a *linear* predictor family (V-usable information, Xu et al. 2020).

That reframes the research question precisely:

> Not "is the class present?" (trivially yes), but
> **"why has a pixel-compression objective reorganized raw RGB so that class
> lies along *linearly accessible* directions?"**

This splits into two halves. §2 (MDL) explains why *semantics* emerge at all.
§3 (LRH) explains why they emerge *linearly* — the genuinely interesting half,
and the one shared with LLM interpretability.

---

## 2. Why semantics emerge — the MDL half

To minimize the description length of `p(x_t | x_{<t})`, the model must capture
the structure that *makes pixels predictable*: object boundaries, surfaces,
lighting, texture, global layout. The shortest code for a dataset is achieved by
modeling its true generative factors — and those factors are exactly what also
determine class. Hence the project's thesis, **compress better → learn better**,
which is the Minimum Description Length principle (Rissanen) instantiated:
the best compressor is the best learner.

Two project-specific amplifiers:

- **Sub-pixel AR.** The layout `[R,G,B]` per pixel forces `p(G|R, ctx)` and
  `p(B|R,G, ctx)`. Modeling intra-pixel channel dependence pushes the model
  toward the physical factors (illumination, material) that generate color —
  the same factors that carry semantics.
- **Coarse conditioning.** `α · coarse_ctx` hands the fine model a low-frequency
  global prior (a blurred thumbnail). To exploit it, the fine model organizes
  its computation around global scene structure, not just local pixel
  statistics.

This half is standard and defensible; it is already the project's §6 narrative.
It explains **semantics**, but says nothing about **linearity**.

---

## 3. Why it's linearly readable — the LRH half (the heart)

The linear representation hypothesis: high-level concepts are encoded as
**linear directions** in activation space; concept presence ≈ projection onto a
direction; concept strength ≈ magnitude. In this model, four pressures produce
that geometry — and the first three are **architecture-specific**, which is what
makes this a contribution rather than a citation.

**1. Linear, weight-tied readout.** The output head is a single
`nn.Linear(d, 256)` whose weight is *tied* to `token_embed` — there is **zero
nonlinearity** between the final hidden state and the logits (`igpt.py:74`). For
the model to emit correct next-pixel distributions, prediction-relevant
structure must be made **linearly** readable at the last layer. Since class is
strongly predictive of pixel statistics, gradient descent pushes
class-correlated structure onto linear directions. This is the image-domain
analogue of the "linear unembedding ⇒ linear concept directions" pressure in
LLMs (Park, Choe & Veitch 2024). Weight-tying makes embedding ≡ unembedding — a
clean special case of their causal-inner-product geometry.

**2. Additive residual stream.** OLMo2 post-norm means every layer *adds* into a
shared stream: the representation is literally a running sum of layer
contributions. Linear directions are therefore the **native format** of the
stream, and a linear probe reads that format directly. This is exactly the
Anthropic residual-stream / superposition picture (Elhage et al. 2021/2022),
and the stream here is structurally identical to a Transformer LM's — so the
same geometry should, and apparently does, emerge.

**3. Superposition.** d=448, but the number of useful "concepts" far exceeds
448. In high dimensions random vectors are near-orthogonal, so the model packs
many features as near-orthogonal linear directions (Toy Models of
Superposition). The probe recovers one of them.

**4. `α · coarse_ctx` is a literal linear concept vector.** The coarse pathway's
output is *added* to the token embedding at layer 0 (`cc_igpt.py:164`,
`igpt.py:134`). It injects a global semantic prior **additively into the
residual stream** — i.e. it is an *engineered* steering vector that also happens
to improve compression. This is the project's single strongest LRH talking
point, and it predicts the empirical fact that coarse-ctx injection **raises
mid-layer probe accuracy by +12.4 pp** (CIFAR v2: L19 79.33% vs iGPT-S L22
66.93%).

### 3.1 The mid-layer peak (the inverted-U)

Measured layer-wise accuracy rises then falls:

- **CIFAR v2 native:** peak at **L19 = 79.33%** (32 layers).
- **IN64→CIFAR transfer:** L0 41.69% → **L16 73.19%** → L31 64.79% (full 32 layers; L15–L17 plateau).

The shape is the fingerprint that ties this model to both iGPT and LLMs:

- **Early layers** hold local pixel statistics → low abstraction → low accuracy.
- **Middle layers** reach maximal task-agnostic semantic abstraction → peak
  linear separability.
- **Late layers re-specialize** toward the 256-way *output* distribution,
  discarding class-general semantics to sharpen local next-pixel prediction →
  accuracy falls.

This is the information-bottleneck / "tunnel" dynamic (Tishby; Shwartz-Ziv &
Tishby 2017). The **same inverted-U** appears in iGPT (Chen et al. 2020) and in
LLM probing (mid-layers probe best for semantic tasks). That cross-modal
coincidence is the answer to *"why is this like LLM interpretation?"*:

> The representational geometry is driven by the **form of the objective**
> (next-token prediction + linear head) and the **architecture** (additive
> residual stream), **not by the modality**. Pixels and tokens converge on the
> same structure.

---

## 4. How to measure it honestly — the MDL probing half

Raw accuracy is a weak metric: a sufficiently expressive probe can *memorize*
the task, so high accuracy need not reflect the representation
(Hewitt & Liang 2019). The fix that aligns perfectly with this project's MDL
thesis is to measure the **description length of the labels given the
features** (Voita & Titov 2020).

**Decomposition.** A representation is good if it lets labels be transmitted in
few bits:

```
L_total = L_model + L_data
        = (complexity of the probe) + (remaining label uncertainty)
```

Two standard estimators:

- **Variational code.** Learn a posterior `q(θ)` over probe weights against a
  prior `p(θ)`; the codelength is an **upper bound**
  `L_var = KL(q‖p) + E[-log p(y|x,θ)]` (this is the negative ELBO). KL = model
  description length; cross-entropy = data description length. A task solvable
  with simple weights keeps the posterior near the prior (small KL); a task
  requiring memorization blows KL up.
- **Online (prequential) code** (Blier & Ollivier 2018). Train on a small data
  fraction, encode the next chunk with the current probe (accumulate `-log p`),
  enlarge, repeat. A good representation needs little data and encodes future
  examples cheaply → small total codelength. This measures **learning
  efficiency** / *effort to extract*, and is the most hand-implementable
  variant — one loop over growing data fractions, no new dependency.

**Why MDL beats accuracy.** Under a control task (random labels), accuracy can
stay high (the probe memorizes) but **codelength explodes**, because random
labels are incompressible while real structure is captured by a small probe
from few examples. MDL therefore cleanly separates *"information in the
representation"* from *"information memorized by the probe"* — the exact
distinction §1 demanded.

**Why this fits the project.** The main table reports bits-per-dim (compressing
pixels). Replacing probe accuracy with label codelength means the *downstream
evaluation is measured on the same MDL ruler* (compressing labels). The whole
story closes: **main task = MDL, probe = MDL, both in bits.**

---

## 5. The unified view

For the pipeline `x → h → y`, a total description length is

```
L(h) + L(model) + L(y | h, model)
```

and different research programs target different terms — all the same
information-theoretic question, *"where is the task information stored, and how
cheaply can it be described?"*:

| Term            | Program                          | This project |
|-----------------|----------------------------------|--------------|
| `L(h)`          | Sparse coding / **SAEs**         | tangential (context only) |
| `L(model)`      | Pruning / **Bayesian Compression** (Louizos 2017) | tangential (context only) |
| `L(y\|h,model)` | **MDL probing**                  | **this is the corner we work in** |

SAEs (sparsity in *representation* space) and Bayesian Compression (sparsity in
*parameter* space) are useful background but **orthogonal** to a linear-probe
thesis. The project's contribution lives in `L(y|h,model)` — and §3 explains why
the `h` it conditions on makes that term **linearly** small.

---

## 6. Reading list

⭐ = core 5 that most directly answer the research question. Titles/years are
reliable from memory; **verify exact venue/arXiv IDs before citing.**

**A. What a probe measures (methodology — read first)**
- ⭐ Alain & Bengio 2017, *Understanding intermediate layers using linear
  classifier probes* (ICLR-W). Origin of the technique; already cited in
  `linear_probe.py`.
- ⭐ Hewitt & Liang 2019, *Designing and Interpreting Probes with Control Tasks*
  (EMNLP). The critique to address; introduces **selectivity / control tasks**.
- Pimentel et al. 2020, *Information-Theoretic Probing for Linguistic Structure*
  (ACL). Probing as mutual information; the extractability-not-presence point.
- **Voita & Titov 2020, *Information-Theoretic Probing with MDL* (EMNLP).** The
  backbone of §4 — codelength probing, variational + online codes.
- Blier & Ollivier 2018, *The Description Length of Deep Learning Models*
  (NeurIPS). Source of the **online/prequential code**.
- Xu et al. 2020, *A Theory of Usable Information / V-information* (ICLR). Linear
  probe accuracy = V-usable info under the linear family.

**B. Why concepts are linear (LRH half — the heart)**
- ⭐ Elhage et al. 2021, *A Mathematical Framework for Transformer Circuits* +
  2022 *Toy Models of Superposition* (Anthropic). Residual stream as linear
  superposition; structurally identical to this model's additive stream.
- ⭐ Park, Choe & Veitch 2024, *The Linear Representation Hypothesis and the
  Geometry of LLMs* (ICML). Formal LRH; weight-tying = embedding ≡ unembedding
  special case.
- Jiang et al. 2024, *On the Origins of Linear Representations in LLMs*. Argues
  linearity arises from the NTP objective — transfers directly to pixel NTP.
- Engels et al. 2024, *Not All Language Model Features Are Linear*. Honest
  counterpoint (some features are circular/manifold); cite so the claim reads
  as "strong approximation," not law.

**C. The mid-layer peak (inverted-U)**
- ⭐ Chen et al. 2020, *Generative Pretraining from Pixels (iGPT)* (ICML). Direct
  lineage; its mid-layer probe peak is the same curve measured here.
- Tishby & Zaslavsky 2015 / Shwartz-Ziv & Tishby 2017, Information Bottleneck.
  The compress-then-specialize dynamic behind the inverted-U.
- 2024 work on *"intermediate layers give the best representations in LLMs"*
  (search the phrase). Confirms the peak is cross-modal.

**D. Linear directions are causal (steering)**
- Turner et al. 2023, *Activation Addition (ActAdd)*; Zou et al. 2023,
  *Representation Engineering*; Anthropic 2024, *Scaling Monosemanticity*.
  Justify treating `α · coarse_ctx` as an engineered steering vector and
  motivate the steering experiment (E4).

**Read only four to crack the question:** Jiang 2024 (NTP→linear) + Elhage 2022
(residual stream/superposition) + iGPT 2020 (pixel-domain mid-layer peak) +
Hewitt & Liang 2019 (prove it's real, not memorized).

---

## 7. Experiment menu (assertion → evidence)

All reuse `scripts/linear_probe.py`, are cheap, and run on AutoDL (written on
WSL — never run training/probe on WSL). Ordered by narrative value.

- **E2 — compression ↔ probe correlation across checkpoints** *(highest value).*
  Using `epoch_6..12.pth`, plot val bpd vs best-layer probe accuracy per
  checkpoint. A tight monotone anti-correlation is **direct empirical proof** of
  *compress better → learn better*. One figure, large payoff.
- **E6 — MDL / online-code probe** *(best thesis fit).* Replace probe accuracy
  with **label codelength** (online code: encode growing data fractions, sum
  `-log p`). Report for v2 / iGPT-S / random-init + a random-label control.
  Hand-implementable, no new dependency, and puts the probe on the **same MDL
  ruler** as the main table — "MDL all the way down."
- **E1 — linear vs MLP probe gap.** Add a 1-hidden-layer probe. Small gap ⇒ info
  is genuinely *linearly* encoded (LRH holds); large gap ⇒ entangled. The
  headline LRH evidence.
- **E3 — control task / selectivity** (Hewitt & Liang). Probe random labels;
  report selectivity = real − control. Rebuts "the probe just learns the task."
- **E5 — coarse_ctx ablation** *(already runnable — `--no_coarse_ctx` exists).*
  Quantify the additive direction's per-layer contribution. Tests pressure #4.
- **E4 — steering** *(most work, best interpretability story).* Use a probe
  weight vector as a class direction, add it to the residual stream during image
  completion, and check whether the output shifts toward the class. If yes, the
  direction is **causal** — the image-domain analogue of LLM activation
  steering.

---

## 8. One-line summary

The model is trained only to compress pixels, but a linear, weight-tied head
over an additive residual stream forces the generative factors it must learn
(MDL) onto **linear directions** (LRH), peaking in the middle layers — so a
linear probe reads class for free; and because the honest way to score that
probe is **label codelength**, the downstream evaluation lands on the very same
MDL ruler as the main table. *Compress better → learn better → read it off
linearly → measure it in bits.*
