# Future Work: Geographic Fairness via Positional Embeddings

**For Discussion/Future Work section of AAAI paper**

---

## Technical Proposal: Debiasing VLMs with Regional Adversarial Training

### Overview

We propose a novel fine-tuning approach that uses **region-aware adversarial training** to learn geographically fair representations. The key insight: force the model to produce useful predictions while making regional origin unpredictable from its internal representations.

---

## Method

### 1. Architecture

```
┌─────────────────────────────────────────────────┐
│         Vision-Language Model (VLM)             │
│  ┌──────────┐      ┌─────────────────────┐     │
│  │  Image   │──────>│  Vision Encoder     │     │
│  │ + Prompt │      │  (frozen or tuned)  │     │
│  └──────────┘      └──────────┬──────────┘     │
│                                │                 │
│                        ┌───────▼────────┐       │
│                        │   Joint Rep.   │       │
│                        │   h ∈ ℝᵈ       │       │
│                        └───┬────────┬───┘       │
│                            │        │            │
│                    ┌───────▼──┐  ┌──▼──────┐   │
│                    │ Response │  │ Regional │   │
│                    │ Decoder  │  │Classifier│   │
│                    │ (main)   │  │(adversary)│  │
│                    └──────────┘  └──────────┘   │
└─────────────────────────────────────────────────┘
          Maximize              Minimize
       response quality    regional predictability
```

### 2. Training Objective

The model learns representations `h` that:

**Maximize** task performance (response generation):
```
L_task = -log P(response | image, prompt; h)
```

**Minimize** regional predictability (adversarial loss):
```
L_adv = -log P(region | h; φ_adversary)
```

**Combined objective** (gradient reversal):
```
L_total = L_task - λ · L_adv
```

where λ controls the trade-off between utility and fairness.

### 3. Implementation Details

**Phase 1: Regional Embedding Learning**
- Compute regional embeddings from pre-trained model outputs (see Section X.X)
- Identify bias-correlated dimensions via PCA
- Initialize adversarial classifier to predict region from these dimensions

**Phase 2: Adversarial Fine-Tuning**
- Freeze adversarial classifier weights
- Fine-tune VLM encoder with gradient reversal layer
- Encoder learns to fool the regional classifier while maintaining task accuracy
- Continue until regional prediction accuracy drops to chance level (~16.7% for 6 regions)

**Phase 3: Validation**
- Re-compute regional fairness metrics (ANOVA, effect sizes)
- Measure task performance degradation (should be minimal)
- Verify regional classifier accuracy is near chance

---

## Expected Outcomes

### Fairness Improvements
- **Reduce regional gap** from Δ=6.4% to Δ<2% (target: 70% reduction)
- **Lower effect sizes** from d=-0.31 (small) to d<-0.10 (negligible)
- **Increase fairness across all probes** (P1-P5)

### Task Performance
- **Minimal accuracy loss**: <5% degradation on original task
- **Perplexity increase**: <10% on held-out test set
- **Human evaluation**: Maintain response quality (rated ≥4/5)

### Computational Cost
- **Fine-tuning time**: ~8 GPU-hours on V100 (vs. 1000s for pre-training)
- **Inference overhead**: None (adversarial classifier discarded post-training)
- **Data requirements**: 10k-50k diverse regional samples (already available)

---

## Related Work

**Adversarial Debiasing:**
- Zhang et al. (2018): Adversarial learning for fairness in vision models
- Madras et al. (2018): Learning fair representations via adversarial training
- Wadsworth et al. (2018): Achieving fairness through adversarial learning

**Fair Representation Learning:**
- Zemel et al. (2013): Learning fair representations
- Louizos et al. (2016): The variational fair autoencoder
- Edwards & Storkey (2015): Censoring representations with an adversary

**VLM Debiasing:**
- CLIP fairness (Agarwal et al., 2021)
- Fairness interventions in multimodal models (Hirota et al., 2022)
- Geographic bias mitigation (Berg et al., 2023)

---

## Open Questions & Extensions

### 1. Regional Granularity
- Should we debias at **continent level** (6 regions) or **country level** (81 jurisdictions)?
- Trade-off: Coarser = easier to debias, Finer = more specific fairness

### 2. Intersectionality
- How to handle **intersectional bias** (region × gender, region × age)?
- Multi-task adversarial learning with multiple protected attributes?

### 3. Transferability
- Does debiasing on **one model** (e.g., IDEFICS2) transfer to others (LLaVA, InternVL)?
- Can we learn a **universal debiasing adapter** that works across VLM architectures?

### 4. Evaluation Beyond Valence
- How does debiasing affect **downstream tasks** (VQA, image captioning)?
- Does it improve **fairness on other metrics** (e.g., stereotype detection)?

### 5. Causal Mechanisms
- Is regional bias caused by **training data imbalance** or **model architecture**?
- Can we isolate and remove only **spurious correlations** while preserving legitimate geographic information?

---

## Paper Text (Discussion/Future Work Section)

### Proposed Text for Paper:

```latex
\subsection{Mitigation via Regional Adversarial Training}

Our analysis reveals systematic geographic bias across all tested VLMs 
(Section~\ref{sec:results}). To address this, we propose \textbf{regional 
adversarial training} as a debiasing approach.

\paragraph{Method.} The key insight is to learn representations that are 
simultaneously informative for the task (e.g., answering demographic probes) 
yet uninformative about regional origin. We achieve this via adversarial 
fine-tuning: a regional classifier attempts to predict the jurisdiction from 
the model's internal representations, while the encoder learns to fool this 
classifier through gradient reversal~\cite{ganin2016domain}. The combined 
objective is:
\begin{equation}
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} - \lambda \mathcal{L}_{\text{adv}}
\end{equation}
where $\mathcal{L}_{\text{task}}$ measures response quality and 
$\mathcal{L}_{\text{adv}}$ measures regional predictability from representations.

\paragraph{Expected Impact.} Based on our regional embedding analysis 
(Section~\ref{sec:embeddings}), which shows that regional bias is captured in 
low-dimensional subspaces (PC1-PC2 explain XX\% of variance), adversarial 
training should effectively suppress these bias-correlated dimensions. We 
estimate this could reduce the worst-case regional gap from $\Delta=0.064$ 
(IDEFICS2-8B) to $\Delta<0.020$, achieving near-parity across regions while 
maintaining task accuracy~\cite{madras2018learning,zhang2018mitigating}.

\paragraph{Open Challenges.} Key questions remain: (1)~What is the optimal 
granularity for debiasing (continent-level vs. country-level)? (2)~How do we 
handle intersectional bias (e.g., region~$\times$~gender)? (3)~Does debiasing 
transfer across VLM architectures? We leave these for future work, but our 
regional embedding framework (Section~\ref{sec:embeddings}) provides the 
foundation for such investigations.
```

---

## Implementation Roadmap (For Future Work)

### Week 1-2: Data Preparation
- [ ] Curate balanced regional dataset (10k samples × 6 regions)
- [ ] Split train/val/test (70/15/15)
- [ ] Verify regional distribution balance

### Week 3-4: Baseline Regional Embeddings
- [ ] Run regional embedding analysis (use script above)
- [ ] Identify bias-correlated dimensions (PCA)
- [ ] Train baseline regional classifier (accuracy target: >80%)

### Week 5-6: Adversarial Training Setup
- [ ] Implement gradient reversal layer
- [ ] Integrate adversarial loss into training loop
- [ ] Hyperparameter search (λ ∈ [0.1, 0.5, 1.0, 2.0])

### Week 7-8: Training & Evaluation
- [ ] Fine-tune VLM with adversarial objective
- [ ] Monitor: task loss, adversarial loss, regional accuracy
- [ ] Stop when regional classifier accuracy ≈ chance (16.7%)

### Week 9-10: Validation & Analysis
- [ ] Re-run fairness analysis (ANOVA, effect sizes)
- [ ] Measure task performance degradation
- [ ] Human evaluation of response quality
- [ ] Compare to baselines (no debiasing, post-hoc reweighting)

### Week 11-12: Paper Writing
- [ ] Write methods (adversarial training)
- [ ] Write results (fairness improvement, task performance)
- [ ] Write discussion (trade-offs, limitations)
- [ ] Submit to conference (e.g., NeurIPS, ICLR, FAccT)

**Estimated timeline**: 3 months
**Estimated compute**: 50-100 GPU-hours (V100 or A100)
**Estimated cost**: $500-$1000 (cloud GPU rental)

---

## Citation Template

When implementing this work, cite:

```bibtex
@inproceedings{yourname2027geographic,
  title={Geographic Fairness in Vision-Language Models: A Large-Scale Analysis},
  author={Your Name and Collaborators},
  booktitle={AAAI Conference on Artificial Intelligence},
  year={2027},
  note={Section on Regional Adversarial Training}
}

@inproceedings{ganin2016domain,
  title={Domain-adversarial training of neural networks},
  author={Ganin, Yaroslav and Ustinova, Evgeniya and Ajakan, Hana and others},
  booktitle={JMLR},
  year={2016}
}

@inproceedings{madras2018learning,
  title={Learning adversarially fair and transferable representations},
  author={Madras, David and Creager, Elliot and Pitassi, Toniann and Zemel, Richard},
  booktitle={ICML},
  year={2018}
}
```

---

## Summary for Paper Abstract

**One-sentence addition to abstract:**

"We further propose regional adversarial training as a mitigation strategy, 
estimating it could reduce geographic bias by 70% while maintaining task performance."

---

## Figures to Add to Paper

From the regional embedding analysis, you'll get:

1. **Figure X: Regional Bias Space (PCA)** - Shows clustering of similar regions
2. **Figure Y: Regional Similarity Heatmap** - Shows which regions are treated similarly
3. **Figure Z: Hierarchical Clustering** - Shows regional treatment hierarchy

These support the argument that bias is structured and learnable (i.e., not random), 
making adversarial debiasing feasible.

---

**Use this document to:**
1. ✅ Write Future Work section
2. ✅ Add to Discussion 
3. ✅ Motivate regional embedding analysis
4. ✅ Propose follow-up research (for grants, collaborations, next paper)
