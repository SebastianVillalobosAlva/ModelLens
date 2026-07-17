# ModelLens — Interpretability Methods: References & Roadmap

References for revisiting the math/theory behind what's implemented, plus the
prioritized roadmap of candidate additions. Ordering principle: **cheap methods
that directly serve the Stoic-Steering bridge experiments (RQ4 residual-stream
constraints, de-risking Exp 12) come before expensive trained methods**, given
the laptop-CPU / occasional-Colab compute budget.

## Implemented in ModelLens — the papers behind each

- **Logit lens** — nostalgebraist, ["interpreting GPT: the logit lens"](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens) (2020, LessWrong). Assumes intermediate states live in the final-layer basis; biased off GPT-2 (→ tuned lens below).
- **Activation patching / causal tracing** — Vig et al. 2020, ["Causal Mediation Analysis"](https://arxiv.org/abs/2004.12265); Meng et al. 2022, [ROME](https://arxiv.org/abs/2202.05262); best practices: Zhang & Nanda 2023, ["Towards Best Practices of Activation Patching"](https://arxiv.org/abs/2309.16042).
- **Attribution patching** — Nanda 2023, [blog/overview](https://learnmechinterp.com/topics/attribution-patching/); Syed et al. 2023, ["Attribution Patching Outperforms Automated Circuit Discovery"](https://arxiv.org/abs/2310.10348). First-order approx `effect ≈ (a_corrupt − a_clean) · ∂metric/∂a_clean`; **exact when the net + metric are linear** (this is how our test validates it), a Taylor approximation otherwise. `~2` passes vs one-per-layer. Edge Attribution Patching (EAP) extends it to edges — see roadmap item 4.
- **Circuit discovery (node-level)** — Wang et al. 2022, ["Interpretability in the Wild: IOI circuit"](https://arxiv.org/abs/2211.00593); framework: Elhage et al. 2021, ["A Mathematical Framework for Transformer Circuits"](https://transformer-circuits.pub/2021/framework/index.html) (Anthropic).
- **Residual stream** — Elhage et al. 2021, ["A Mathematical Framework for Transformer Circuits"](https://transformer-circuits.pub/2021/framework/index.html) (residual stream as the model's communication channel).
- **Sparse autoencoders / dictionary learning** — Bricken et al. 2023, ["Towards Monosemanticity"](https://transformer-circuits.pub/2023/monosemantic-features/index.html) (Anthropic; the untied + L1 recipe our SAE uses); Cunningham et al. 2023, ["Sparse Autoencoders Find Highly Interpretable Features"](https://arxiv.org/abs/2309.08600); TopK variant: Gao et al. 2024, ["Scaling and evaluating sparse autoencoders"](https://arxiv.org/abs/2406.04093).
- **Linear concept probing** — Alain & Bengio 2016, ["Understanding intermediate layers using linear classifier probes"](https://arxiv.org/abs/1610.01644). **Key caveat — correlational, not causal:** Hewitt & Liang 2019, ["Designing and Interpreting Probes with Control Tasks"](https://aclanthology.org/D19-1275/); survey: Belinkov 2022, ["Probing Classifiers: Promises, Shortcomings, and Advances"](https://arxiv.org/abs/2102.12452). Rigorous use = *probe finds a direction → patch/ablate it → confirm the model causally uses it* — the ablation half of that workflow is roadmap item 1.

## Roadmap — candidates in priority order

Each entry states **why it's here** (what experiment it unblocks) and **done
means** (the acceptance test it must pass before any bridge experiment trusts
it — following the pattern set by attribution patching's exact-on-linear test).

### 1. Directional ablation / residual-stream projection

Remove a single direction from the residual stream — `h ← h − (h·v̂)v̂` at
chosen layers — as opposed to swapping whole activations (`activation_patch`)
or zeroing whole components (item 3). Canonical: Arditi et al. 2024,
["Refusal in Language Models Is Mediated by a Single Direction"](https://arxiv.org/abs/2406.11717)
— the closest published work to the Stoic-Steering thesis.

- **Why first:** it is the intervention RQ4 ("constrain the residual stream
  rather than add to it") actually requires, and it completes the probing
  workflow's causal step (probe finds a direction → project it out → confirm
  the model used it). Trivial on the existing hook infrastructure.
- **Done means:** on a toy model with a planted linear concept direction,
  projecting that direction out drops a trained probe's accuracy on the
  concept to chance and degrades the task that depends on it, while an
  orthogonal control task is unaffected; hooks verified removed after the run
  (same pattern as `test_core_regression`).

### 2. Direct Logit Attribution (DLA)

Decompose the *final logit* into additive per-component (attention head / MLP)
contributions by projecting each component's residual-stream write through the
unembedding ("which component wrote the answer"). Distinct from `logit_lens`
(per-layer) and `attribution_patch` (patch-effect gradient). Cheap add on the
residual-stream + unembedding pieces we already have. Framework: Elhage et al.
2021, ["A Mathematical Framework for Transformer Circuits"](https://transformer-circuits.pub/2021/framework/index.html);
practical: [DLA overview](https://learnmechinterp.com/topics/direct-logit-attribution/).

- **Why now:** cheap, and gives the bridge experiments a per-component "who
  wrote the stoic answer" readout to complement patching.
- **Done means:** per-component contributions sum to the actual final logit
  within numerical tolerance on GPT-2 (document how LayerNorm is handled —
  folded or frozen — since that's the only approximation in the decomposition).

### 3. Zero / mean ablation

A distinct intervention from our clean↔corrupted *swap*: set a component's
activation to zero or its dataset mean and measure the drop. Trivial on the
existing hook infra; standard in circuit workflows (mean ablation used in the
[IOI circuit](https://arxiv.org/abs/2211.00593); corruption-method tradeoffs
in [Zhang & Nanda 2023](https://arxiv.org/abs/2309.16042)).

- **Why now:** completes the intervention menu (swap / project-out / knock-out)
  at near-zero cost.
- **Done means:** on a toy network where exactly one component carries the
  label signal, ablating that component drops the metric to chance while
  ablating any other component doesn't; the mean is computed over a
  caller-supplied reference batch (never silently over the eval batch).

### 4. Edge / path patching (EAP)

Circuits over *connections*, not just nodes. Exact: Goldowsky-Dill et al.
2023, ["Localizing Model Behavior with Path Patching"](https://arxiv.org/abs/2304.05969);
gradient approx = **Edge Attribution Patching (EAP)**, which extends our
node-level attribution patching to edges — Syed et al. 2023,
[arXiv 2310.10348](https://arxiv.org/abs/2310.10348).

- **Why now — this de-risks the flagship result:** `discover_circuit`'s edges
  are currently *heuristic* (attention patterns + sequential order), and
  Exp 12's circuit-topology claim — the "predicted the CAA/LoRA split before
  it showed up behaviorally" result — rests partly on those heuristic edges.
  EAP replaces them with measured causal edge effects, so Exp 12 can be
  re-run with real edges *before* a reviewer asks whether the topology
  survives that upgrade.
- **Done means:** edge attributions equal exact path patching on a linear
  network (same validation trick as `test_attribution_patching`), and the sum
  of a node's incoming-edge scores is consistent with its node-level
  `attribution_patch` score.

### 5. Representational similarity (CKA) — cheap model-diffing first pass

Kornblith et al. 2019, ["Similarity of Neural Network Representations Revisited"](https://arxiv.org/abs/1905.00414).
Architecture-agnostic model/layer comparison.

- **Why now:** answers "which layers differ most between base / CAA / LoRA"
  for near-zero compute — run this *before* committing to crosscoders
  (item 7); it may localize the diff enough that a full crosscoder isn't
  needed on this budget.
- **Done means:** property tests — CKA(X, X) = 1, invariance to orthogonal
  transforms and isotropic scaling; layer-wise CKA between a model and a
  finetuned copy is high early and diverges where the finetune bites.

### 6. Tuned lens — conditional, not scheduled

Belrose et al. 2023, ["Eliciting Latent Predictions … with the Tuned Lens"](https://arxiv.org/abs/2303.08112).
Learned affine probe per layer; needed because plain logit lens is biased on
non-GPT-2 models (i.e. Llama).

- **Build only if** the logit-lens bias actually blocks a Llama-3.2-3B
  experiment. Check first whether the [`tuned-lens` package](https://github.com/AlignmentResearch/tuned-lens)
  ships a pretrained lens for Llama-3.2-3B; if not, translator training at 3B
  is a Colab job, not a laptop job — budget it before starting.
- **Done means:** per-layer cross-entropy of tuned-lens predictions ≤ plain
  logit lens at every layer on held-out text (held-out so the translators
  can't "cheat"), and the two lenses roughly agree at late layers on GPT-2.

### 7. Crosscoders / model diffing — stretch goal

Lindsey et al. 2024 (Anthropic), ["Sparse Crosscoders for Cross-Layer Features and Model Diffing"](https://transformer-circuits.pub/2024/crosscoders/index.html).
Directly compares base vs CAA vs LoRA feature-by-feature — the ideal tool for
the bridge, and the most expensive item here.

- **Open design question (decide before writing code):** everything in
  ModelLens is one model → one lens, but model diffing is inherently
  cross-model — base vs CAA vs LoRA is three activation sources feeding one
  trained dictionary. Does `compare_features` take a list of lenses? It
  trains a network, so by our convention it's a module-level function — but
  where do capability checks live when the models differ, and what's the
  return type of a feature-level diff? Revisit this section when the feature
  is picked up.
- **Feasibility:** activation harvesting + crosscoder training at 3B scale
  exceeds the laptop budget — plan for Colab, and estimate the required
  activation-sample count first. Run CKA (item 5) beforehand to check whether
  the cheap answer is enough.
- **Done means:** on two toy models differing by one planted feature, the
  crosscoder isolates that feature as model-exclusive; dead-feature rate and
  reconstruction loss are reported alongside any diff claim.

### Surveyed, not scheduled

- **ACDC** — Conmy et al. 2023, ["Towards Automated Circuit Discovery"](https://arxiv.org/abs/2304.14997).
- **Transcoders** — Dunefsky et al. 2024, ["Transcoders Find Interpretable LLM Feature Circuits"](https://arxiv.org/abs/2406.11944). Sparse approximation of an MLP's input→output.
- **Sparse feature circuits** — Marks et al. 2024, ["Sparse Feature Circuits"](https://arxiv.org/abs/2403.19647). Circuits whose nodes are SAE features (connects our SAE + circuit modules).

## Steering context (Stoic-Steering)

- CAA: Rimsky et al. 2023, ["Steering Llama 2 via Contrastive Activation Addition"](https://arxiv.org/abs/2312.06681); ActAdd: Turner et al. 2023, ["Steering Language Models With Activation Engineering"](https://arxiv.org/abs/2308.10248).
- Directional ablation / refusal direction: Arditi et al. 2024, ["Refusal in Language Models Is Mediated by a Single Direction"](https://arxiv.org/abs/2406.11717).

> Links verified July 2026 (arXiv / transformer-circuits.pub / ACL / LessWrong / GitHub). Cite the canonical venue (many arXiv papers have ACL/NeurIPS versions) in any formal writeup.
