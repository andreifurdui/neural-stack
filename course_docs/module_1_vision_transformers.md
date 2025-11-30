# Module 1: Vision Transformers & Modern Architectures

**Duration**: 8 weeks | **Level**: Foundation | **Time Commitment**: 10-12 hours/week

---

## Module Overview

### Prerequisites Refresher
- **Deep Learning**: CNN architectures (ResNet, VGG), backpropagation, optimization
- **PyTorch**: Model building, training loops, data loading, GPU usage
- **Mathematics**: Linear algebra (matrix multiplication, eigenvalues), probability, basic calculus
- **Vision**: Image preprocessing, data augmentation, classification metrics

### Learning Outcomes
By the end of this module, you will:
- Implement transformer architectures from scratch in PyTorch
- Explain mathematically how self-attention enables long-range dependencies
- Fine-tune pre-trained ViTs on custom datasets using timm/HuggingFace
- Visualize and interpret attention patterns across layers
- Conduct systematic architectural comparisons between ViTs and CNNs
- Analyze what vision transformers learn differently than convolutional networks

### What You'll Build
A comprehensive forensic analysis system that trains both ViT and CNN architectures, visualizes attention mechanisms and activation patterns, tests adversarial robustness, and documents architectural insights through rigorous experimentation.

### Time Commitment Breakdown
- **Reading/Theory**: 3-4 hours/week
- **Implementation**: 6-7 hours/week
- **Experimentation/Analysis**: 2-3 hours/week
- **Documentation**: 1 hour/week

---

## Fundamental Concepts Deep Dive

### 1. Scaled Dot-Product Attention

**Motivation**: Traditional CNNs use fixed receptive fields, limiting their ability to capture long-range dependencies. Attention mechanisms allow each position to directly access every other position, enabling global context aggregation regardless of distance.

**Mathematical Formulation**:
```
Given input X ∈ ℝ^(n×d):
  Q = XW_Q  (queries)
  K = XW_K  (keys)
  V = XW_V  (values)
  
Attention(Q,K,V) = softmax(QK^T / √d_k)V

where:
  - Q, K, V are learned linear projections
  - d_k is the key dimension
  - QK^T produces attention scores matrix ∈ ℝ^(n×n)
  - softmax normalizes scores to attention weights
  - Final output is weighted sum of values
```

**Intuitive Explanation**: 
Imagine searching a library. Your query ("find books about transformers") is matched against book titles (keys). Books with relevant titles get higher scores. You then retrieve content (values) weighted by relevance scores. The √d_k scaling prevents softmax saturation for high dimensions.

**Common Misconceptions**:
- ❌ "Attention is just weighted averaging" → The learned projections Q, K, V are crucial; attention learns *what* to attend to
- ❌ "More attention weight = more important" → Raw attention can be misleading; gradient-weighted attention is more meaningful
- ❌ "Attention has O(n) complexity" → It's O(n²d) due to the QK^T matrix multiplication

**Connection to Module**: 
This mechanism forms the core of every ViT layer. In vision, n is the number of image patches (e.g., 196 for 224×224 images with 16×16 patches), allowing each patch to attend to all others.

---

### 2. Patch Embeddings & Positional Encoding

**Motivation**: 
Transformers operate on sequences, but images are 2D grids. We need to convert spatial image data into sequential token representations while preserving positional information that attention mechanisms naturally lack.

**Mathematical Formulation**:
```
Image: I ∈ ℝ^(H×W×C)
Patch size: P × P
Number of patches: n = HW/P²

Patch Embedding:
  1. Split image into n patches: x_p ∈ ℝ^(n×(P²·C))
  2. Linear projection: E = x_p W_E + b  where W_E ∈ ℝ^((P²·C)×d)
  3. Add learnable class token: E_cls ∈ ℝ^d
  4. Add positional embedding: E_pos ∈ ℝ^((n+1)×d)
  
Final embedding: z_0 = [E_cls; E] + E_pos
```

**Intuitive Explanation**: 
Breaking an image into patches is like cutting a photo into a grid. Each patch becomes a "word" in our visual "sentence". The linear projection embeds patches into a common space. The class token acts as a learnable summary token (like [CLS] in BERT). Positional embeddings tell the model where each patch came from, since attention is position-agnostic.

**Common Misconceptions**:
- ❌ "Patch size doesn't matter much" → Larger patches = fewer tokens = faster but less fine-grained; patch size critically affects performance
- ❌ "Positional encoding must be fixed (sin/cos)" → ViT uses *learnable* positional embeddings that outperform fixed encodings for vision
- ❌ "Class token is optional" → While you can pool patch tokens instead, the class token provides a cleaner interface for classification

**Connection to Module**: 
This is how ViT adapts transformers to vision. You'll implement this to understand the spatial-to-sequential conversion and experiment with different patch sizes (4×4 vs 16×16 vs 32×32) to see accuracy-efficiency tradeoffs.

---

### 3. Multi-Head Attention

**Motivation**: 
Single attention heads can focus on one type of relationship. Multiple heads enable parallel attention to different aspects: one head might focus on colors, another on shapes, another on spatial relationships.

**Mathematical Formulation**:
```
MultiHead(Q,K,V) = Concat(head_1,...,head_h)W_O

where:
  head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
  
  W_i^Q ∈ ℝ^(d_model × d_k)
  W_i^K ∈ ℝ^(d_model × d_k)  
  W_i^V ∈ ℝ^(d_model × d_v)
  W_O ∈ ℝ^(hd_v × d_model)
  
Typically: d_k = d_v = d_model/h

Computational cost: O(n²d) per head, O(hn²d) total
```

**Intuitive Explanation**: 
Like having multiple reviewers read the same document, each focusing on different aspects. One reviewer checks grammar, another checks logic, another checks style. Each attention head learns to focus on different patterns. The final output combines all perspectives.

**Common Misconceptions**:
- ❌ "More heads = better performance" → Diminishing returns; 8-12 heads is typical optimal range
- ❌ "All heads learn distinct patterns" → Some redundancy exists; not all heads are equally important
- ❌ "Multi-head adds no computation" → It's parallelizable but costs h× more computation

**Connection to Module**: 
ViT-Base uses 12 heads, ViT-Large uses 16. You'll visualize individual heads to see specialization (some focus on edges, some on semantics) and conduct ablation studies removing heads to measure importance.

---

### 4. Vision Transformer Complete Architecture

**Motivation**: 
Combining attention, embeddings, normalization, and feed-forward layers into a unified architecture for image recognition without convolutions.

**Mathematical Formulation**:
```
Input: Image I ∈ ℝ^(H×W×3)

1. Patch + Position Embedding:
   z_0 = [x_class; x_p^1E; ...; x_p^nE] + E_pos
   
2. L Transformer Encoder Layers:
   for l = 1 to L:
     z'_l = MSA(LN(z_{l-1})) + z_{l-1}     # Multi-head Self-Attention
     z_l = MLP(LN(z'_l)) + z'_l             # Feed-forward network
     
3. Classification Head:
   y = LN(z_L^0)                            # Take class token from final layer
   ŷ = softmax(W_head · y + b)

where:
  MSA = Multi-head Self-Attention
  LN = Layer Normalization  
  MLP = Two-layer feed-forward: MLP(x) = GELU(xW_1)W_2
```

**Intuitive Explanation**: 
Think of it as a deep reading process. First, parse the image into understandable chunks (patches). Then, repeatedly refine understanding through attention layers—each layer can access all previous information and add new insights. Layer normalization keeps values stable. The feed-forward network processes attended information. Finally, extract the summary (class token) for classification.

**Common Misconceptions**:
- ❌ "ViT has no inductive bias" → It has *less* than CNNs (no locality/translation equivariance) but still has inductive bias from patch embedding
- ❌ "Pre-norm (LN before attention) doesn't matter" → Post-norm can cause training instability; pre-norm is standard for ViT
- ❌ "Any transformer architecture works for vision" → Vision-specific choices matter: learnable pos encodings, MLP ratio, GELU activation

**Connection to Module**: 
This complete architecture is what you'll implement, train, and analyze. Understanding each component's role helps debug training issues and enables architectural innovations.

---

## Weekly Breakdown

### Week 1: Transformer Fundamentals

**Learning Objectives**
- Understand self-attention mechanism mathematically and intuitively
- Implement scaled dot-product attention from scratch in PyTorch
- Build multi-head attention with parallel head computation

**Core Topics**
- Query-Key-Value paradigm and matrix interpretation
- Scaled dot-product: QK^T/√d_k and why scaling matters
- Multi-head attention: parallel representation learning
- Attention weight visualization and interpretation

**Required Reading**
- 📄 Vaswani et al., "Attention Is All You Need" (NeurIPS 2017)
  - [Paper](https://arxiv.org/abs/1706.03762) | Sections 3.2, 3.5 | ~2 hours
- 🌐 Jay Alammar, "The Illustrated Transformer"
  - [Blog](https://jalammar.github.io/illustrated-transformer/) | ~1.5 hours
- 📹 3Blue1Brown, "Attention in transformers, visually explained"
  - [Video](https://www.youtube.com/watch?v=eMlx5fFNoYc) | 45 min

### Week 1 Practical Exercise: The Engine Room

**Goal:** Implement efficient, parallelized Multi-Head Self-Attention (MSA) in PyTorch from first principles, without using `torch.nn.MultiheadAttention`.

Target Time: 8 Hours

Location: `src/attention.py` and `notebooks/01_attention_verification.ipynb`

### Technical Constraints & Rules

1. **NO Loops over Sequence Length:** You are forbidden from using `for` loops to iterate over sequence tokens. All operations must be vectorized using matrix multiplications (use `torch.matmul` or the preferred `torch.einsum`).
2. **Einsum Usage:** You are strongly encouraged to use `torch.einsum` for the core attention multiplication. It requires deeper understanding of the dimensions but yields cleaner, more efficient code.
3. **Masking Support:** Your implementation must support an optional mask tensor to handle variable sequence lengths (padding).

---

### Phase 1: Scaled Dot-Product Attention (The Core Math)

**Time Estimate:** 3 Hours

Implement the mathematical heart of the transformer. This function takes pre-projected queries, keys, and values and performs the mixing.

Formula to implement:

$$

\text{Attention}(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V
$$

**Task Checklist:**

- [ ]  Define class `ScaledDotProductAttention(nn.Module)`.
- [ ]  In `__init__`, define a `nn.Dropout` layer and perform the $\sqrt{d_k}$ scaling pre-calculation.
- [ ]  Implement the `forward` method.
- [ ]  **Critical Step:** Ensure correct tensor reshaping/transposing so that matrix multiplication happens over the sequence dimensions (`L`), not the batch (`B`) or head (`H`) dimensions.
- [ ]  Apply the scaling factor *before* the softmax.
- [ ]  Apply mask (if provided) *before* the softmax (using a large negative number like `1e9` for masked positions).
- [ ]  Apply dropout to the attention weights *after* softmax but *before* multiplying with $V$.
- [ ]  Return both the output values and the attention weights (for later visualization).

**Input/Output Specs for `forward`:**

```python
# Inputs:
# query: (Batch_Size, Heads, Seq_Len_Q, Head_Dim)
# key:   (Batch_Size, Heads, Seq_Len_K, Head_Dim)
# value: (Batch_Size, Heads, Seq_Len_V, Head_Dim)
# mask:  (Batch_Size, 1, 1, Seq_Len_K) - Optional boolean or binary mask

# Outputs:
# context:           (Batch_Size, Heads, Seq_Len_Q, Head_Dim)
# attention_weights: (Batch_Size, Heads, Seq_Len_Q, Seq_Len_K)
```

---

### Phase 2: Multi-Head Attention Wrapper (The Plumbing)

**Time Estimate:** 3 Hours

Implement the layer that learns the projections and manages the multiple heads. This wraps Phase 1.

**Task Checklist:**

- [ ]  Define class `MultiHeadAttention(nn.Module)`.
- [ ]  In `__init__`, take `d_model` (e.g., 512), `n_heads` (e.g., 8), and `dropout` rate.
- [ ]  Assert that `d_model` is divisible by `n_heads`. Calculate `head_dim`.
- [ ]  Define four linear layers: $W^Q, W^K, W^V$ (projecting from `d_model` to `d_model`) and $W^O$ (output projection).
- [ ]  Initialize your Phase 1 `ScaledDotProductAttention` module.
- [ ]  Implement `forward`:
    1. Pass input `X` through linear layers to get $Q, K, V.$
    2. **Reshape and Transpose:** Split the `d_model` dimension into `(n_heads, head_dim)` and move the head dimension so it's ready for Phase 1.
        - *Hint: $(B, L, D)$ $\to$ $(B, L, H, D_h)$ $\to$ $(B, H, L, D_h)$*
    3. Pass projected tensors to your Phase 1 module.
    4. **Transpose and Reshape back:** Concatenate the heads back together.
        - *Hint: $(B, H, L, D_h)$ $\to$ $(B, L, H, D_h)$ $\to$ $(B, L, D)$*
    5. Pass through final output linear layer $W^O$.

**Input/Output Specs for `forward`:**

```python
# Inputs:
# x (query/key/value source): (Batch_Size, Seq_Len, d_model)
# mask: (Batch_Size, 1, 1, Seq_Len) - Optional

# Outputs:
# output: (Batch_Size, Seq_Len, d_model)
# attention_weights: (Batch_Size, n_heads, Seq_Len, Seq_Len)
```

---

### Phase 3: Verification & The Toy Task

Time Estimate: 2 Hours

Location: Jupyter Notebook

Don't trust your code until it learns something. We will use a synthetic "Reverse Sequence" regression task. It's simpler than classification as we don't need a softmax output.

The Task:

Input is a sequence of random 1D vectors. The target is the same sequence in reverse order. The model must learn to look at the end of the input sequence to generate the beginning of the output sequence.

**Setup Checklist (in Notebook):**

- [ ]  **Data Generator:** Create a function that generates a batch of random sequences.
    - Shape: `(Batch=32, SeqLen=10, d_model=64)`
    - Input `X`: random floats. Target `Y`: `torch.flip(X, dims=[1])`.
- [ ]  **The Model:** Instantiate your `MultiHeadAttention` module (d_model=64, heads=4).
- [ ]  **Training Loop:**
    - Optimizer: `Adam(lr=1e-3)`
    - Loss function: `MSELoss` (Mean Squared Error).
    - Train for ~500 steps.
- [ ]  **Success Criterion:** The MSE loss should rapidly drop close to zero (< 0.01). If it stalls above 0.1, your implementation is buggy (likely reshaping issues).

---

### Phase 4: Introspection (Visualization)

**Time Estimate:** <1 Hour (Part of Notebook)

Prove that the model solved the task using attention, not magic.

**Task Checklist:**

- [ ]  Run a single validation sequence through the trained model.
- [ ]  Extract the `attention_weights` returned by forward pass. Shape: `(1, 4, 10, 10)`.
- [ ]  Use `matplotlib.pyplot.imshow` to plot the attention heatmap for Head 0.
- [ ]  **Success Criterion:** Since the task is reversing the sequence, the attention matrix should look like an **anti-diagonal line** (from top-right to bottom-left). Token 0 attends to Token 9, Token 1 attends to Token 8, etc.

---

**Time Estimate**
- Reading: 4 hours
- Implementation: 6 hours
- Debugging & visualization: 2 hours
- **Total: ~12 hours**

---

### Week 2: Vision-Specific Adaptations

**Learning Objectives**
- Convert 2D images into 1D sequences via patch embeddings
- Implement learnable positional encodings for spatial awareness
- Build complete minimal ViT architecture end-to-end

**Core Topics**
- Image to sequence conversion: raster scan vs learned ordering
- Patch embedding: linear projection of flattened patches
- Class token ([CLS]) as aggregation mechanism
- Positional encoding: learnable vs fixed, 1D vs 2D
- Complete ViT forward pass from image to logits

**Required Reading**
- 📄 Dosovitskiy et al., "An Image is Worth 16x16 Words" (ICLR 2021)
  - [Paper](https://arxiv.org/abs/2010.11929) | Sections 3, 4.5, Figure 1 | ~2 hours
- 💻 UvA Deep Learning Tutorials, "Vision Transformers"
  - [Notebook](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial15/Vision_Transformer.html) | ~1.5 hours
- 🌐 AI Summer, "How ViT Works in 10 Minutes"
  - [Blog](https://theaisummer.com/vision-transformer/) | 30 min

**Hands-On Exercise Part 1: Patch Embeddings**

*Tasks*:
- [ ] Implement `PatchEmbedding` module: split image → flatten → linear projection
- [ ] Add learnable positional embeddings
- [ ] Implement class token prepending
- [ ] Visualize: original image, patches, patch embeddings (t-SNE)

*Explainability*: Compare 4×4, 8×8, 16×16 patch sizes. Show visual quality vs computational cost tradeoff.

*Success Criteria*:
- [ ] Correctly splits images maintaining spatial structure
- [ ] Visualization shows relationship between patch size and tokens
- [ ] Can explain why smaller patches increase cost quadratically

**Hands-On Exercise Part 2: Complete ViT**

*Tasks*:
- [ ] Build `VisionTransformer` class: embeddings + transformer + classifier
- [ ] Process CIFAR-10 images with correct dimensions
- [ ] Implement forward pass outputting class logits
- [ ] Document computational cost for different patch sizes

*Success Criteria*:
- [ ] Processes CIFAR-10 (32×32×3) with configurable patch sizes
- [ ] Outputs correct logit dimensions
- [ ] Documented FLOPs analysis for patch size variations

**Checkpoint Deliverable**
Working ViT implementation that:
- Processes CIFAR-10 images (32×32×3) with configurable patch size
- Outputs class logits with correct dimensions
- Includes visualization comparing 4×4, 8×8, and 16×16 patch sizes
- Documents computational cost for each patch size

**Self-Assessment**
- Can you calculate number of patches for any image/patch size?
- What's computational complexity as function of patch size?
- What happens without positional encodings?

**Time**: Reading 4h + Implementation 5h + Visualization 2h = **~11 hours**

---

### Week 3: Training ViTs & Data Efficiency

**Learning Objectives**
- Train ViT from scratch and diagnose common failure modes
- Apply knowledge distillation for data-efficient training
- Fine-tune pre-trained models using timm and HuggingFace

**Core Topics**
- Training dynamics: learning rate warmup, gradient clipping, optimizer choice (AdamW)
- Data augmentation for ViT: RandAugment, Mixup, CutMix
- Knowledge distillation: student-teacher framework, soft targets
- Transfer learning: feature extraction vs fine-tuning strategies
- Comparing training regimes: from-scratch vs pre-trained

**Required Reading**
- 📄 Touvron et al., "DeiT" (ICML 2021)
  - [Paper](https://arxiv.org/abs/2012.12877) | Sections 3, 4.3, 4.4 | ~1.5 hours
- 💻 timm Documentation, "Training Script Overview"
  - [Docs](https://timm.fast.ai/) | ~45 min
- 🌐 HuggingFace, "Fine-tune ViT"
  - [Tutorial](https://huggingface.co/docs/transformers/tasks/image_classification) | 30 min

**Hands-On Exercise: Train Minimal ViT & Fine-tune Pre-trained**

*Part 1 - From Scratch*:
- [ ] Train minimal ViT on CIFAR-10 from scratch (track loss, accuracy)
- [ ] Implement warmup schedule and proper augmentation
- [ ] Document training dynamics and convergence

*Part 2 - Transfer Learning*:
- [ ] Fine-tune `timm.create_model('vit_base_patch16_224', pretrained=True)` on Food-101
- [ ] Compare feature extraction (frozen backbone) vs full fine-tuning
- [ ] Measure: accuracy, training time, convergence speed

*Part 3 - Optional Distillation*:
- [ ] Implement simple distillation using CNN teacher's soft labels
- [ ] Compare distilled vs from-scratch performance

*Explainability*: Attention visualization comparing pre-trained (ImageNet) vs fine-tuned (target dataset) attention patterns. How does attention adapt to new domain?

*Success Criteria*:
- [ ] From-scratch achieves >75% on CIFAR-10
- [ ] Fine-tuned achieves >80% on Food-101
- [ ] Documented comparison: from-scratch vs transfer vs distillation
- [ ] Analysis explains why transfer outperforms from-scratch

**Checkpoint Deliverable**
Training report comparing:
- Accuracy curves for all three training approaches
- Final test accuracy, training time, convergence speed
- Analysis: "Why does transfer learning outperform from-scratch on Food-101?"
- Recommendations for when to use each approach

**Self-Assessment**
- Why do ViTs need more data than CNNs?
- What hyperparameters most affect training stability?
- Why does distillation from CNNs work well?

**Time**: Reading 3h + Implementation 7h + Analysis 2h = **~12 hours**

---

### Week 4: Hierarchical Architectures

**Learning Objectives**
- Understand computational limitations of vanilla ViT (O(n²) complexity)
- Implement Swin Transformer's shifted window attention
- Compare flat vs hierarchical vision transformer architectures

**Core Topics**
- Quadratic complexity problem: why full attention doesn't scale
- Window-based attention: local attention for O(n) complexity
- Shifted windows: cross-window connections without full attention
- Hierarchical feature maps: pyramid representations
- Efficiency metrics: FLOPs, memory, throughput

**Required Reading**
- 📄 Liu et al., "Swin Transformer" (ICCV 2021, Best Paper)
  - [Paper](https://arxiv.org/abs/2103.14030) | Section 3.2, Figure 2 | ~2 hours
- 📄 Liu et al., "EfficientViT" (CVPR 2023)
  - [Paper](https://arxiv.org/abs/2305.07027) | Sections 1-3 | ~1 hour
- 💻 timm Swin Documentation
  - [Docs](https://timm.fast.ai/) | 30 min

**Hands-On Implementation: Swin Components**

*Tasks*:
- [ ] Implement window partitioning: split feature map into non-overlapping windows
- [ ] Implement shifted window mechanism with masking
- [ ] Build simplified Swin block with window attention
- [ ] Profile: FLOPs, memory, inference time for ViT vs Swin
- [ ] Train both on Tiny ImageNet or compare pre-trained models

*Explainability*: Visualize attention patterns - global (ViT) vs windowed (Swin). Show how shifted windows create cross-window communication.

*Success Criteria*:
- [ ] Window partitioning correctly handles feature maps
- [ ] Shifted window mechanism properly implemented
- [ ] Comprehensive performance comparison table
- [ ] Clear documentation of efficiency gains

**Checkpoint Deliverable**
Performance analysis document:
- Architecture comparison table: ViT-Small vs Swin-Tiny
- Metrics: accuracy, FLOPs, parameters, inference time, memory usage
- Visualization of attention patterns: global (ViT) vs windowed (Swin)
- Recommendations: when to use each architecture

**Self-Assessment**
- Can you calculate FLOPs for attention with n patches?
- How do shifted windows enable cross-window communication?
- What's the tradeoff between window size and capacity?

**Time**: Reading 3.5h + Implementation 5h + Profiling 2.5h = **~11 hours**

---

### Week 5: Self-Supervised Learning

**Learning Objectives**
- Understand contrastive vision-language pre-training (CLIP)
- Explore self-supervised visual features without labels (DINOv2)
- Apply pre-trained features to downstream tasks via zero-shot or linear probing

**Core Topics**
- Contrastive learning: InfoNCE loss, positive/negative pairs
- Vision-language alignment: joint embedding spaces
- Zero-shot transfer via natural language prompts
- Self-distillation: learning from own predictions
- Feature quality: linear probing, k-NN retrieval

**Required Reading**
- 📄 Radford et al., "CLIP" (ICML 2021)
  - [Paper](https://arxiv.org/abs/2103.00020) | Sections 2, 3.3 | ~1.5 hours
- 📄 Oquab et al., "DINOv2" (TMLR 2024)
  - [Paper](https://arxiv.org/abs/2304.07193) | Sections 3, 4.2 | ~1.5 hours
- 🌐 Meta AI Blog, "DINOv2"
  - [Blog](https://ai.meta.com/blog/dino-v2-computer-vision-self-supervised-learning/) | 30 min

**Hands-On Exploration: Pre-trained Models**

*Tasks*:
- [ ] Load CLIP via `open_clip`: perform zero-shot classification with text prompts
- [ ] Extract DINOv2 features: `torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')`
- [ ] Implement k-NN classifier using DINOv2 features (no training!)
- [ ] Visualize embeddings: t-SNE comparing supervised vs self-supervised features
- [ ] Compare: supervised ViT vs CLIP vs DINOv2 on same test set

*Explainability*: t-SNE visualizations showing embedding space structure. Test different CLIP prompt templates - how sensitive is zero-shot performance to prompt wording?

*Success Criteria*:
- [ ] Zero-shot CLIP classification working with multiple prompts
- [ ] k-NN accuracy (k=1, 5, 20) using DINOv2 features
- [ ] Clear t-SNE visualizations comparing approaches
- [ ] Analysis of self-supervised vs supervised advantages

**Checkpoint Deliverable**
Feature quality analysis:
- Zero-shot accuracy using CLIP with various prompt templates
- k-NN accuracy (k=1,5,20) using DINOv2 features vs supervised features
- t-SNE visualization showing embedding space structure
- Analysis: "What advantages do self-supervised features provide?"

**Self-Assessment**
- Can you explain contrastive loss objective?
- Why can CLIP do zero-shot classification?
- What's the difference between contrastive and masked learning?

**Time**: Reading 3.5h + Implementation 4h + Experimentation 2.5h = **~10 hours**

---

### Week 6: Attention Visualization & Interpretability

**Learning Objectives**
- Extract and visualize attention weights from trained ViT models
- Implement attention rollout for multi-layer information flow analysis
- Understand what different layers and heads learn

**Core Topics**
- Attention weight extraction from transformer models
- Attention rollout: propagating attention through residual connections
- Head importance: identifying redundant vs critical heads
- Layer-wise analysis: what each transformer layer represents
- Visualization best practices: heatmaps, overlays, aggregation methods

**Required Reading**
- 📄 Chefer et al., "Transformer Interpretability Beyond Attention" (CVPR 2021)
  - [Paper](https://arxiv.org/abs/2005.00928) | Sections 3-4 | ~1.5 hours
- 💻 ViT-Prisma Documentation
  - [Repo](https://github.com/soniajoseph/ViT-Prisma) | README, examples | ~1 hour
- 🌐 Jacob Gildenblat, "Exploring Explainability for ViT"
  - [Blog](https://jacobgil.github.io/deeplearning/vision-transformer-explainability) | 45 min

**Hands-On Exercise: Multi-layer Attention Analysis**

*Tasks*:
- [ ] Extract attention weights: modify forward to return attention matrices
- [ ] Implement attention rollout: recursively multiply accounting for residuals
- [ ] Create multi-layer visualization: attention evolution layer 1→12
- [ ] Analyze head specialization: visualize what each of 12 heads attends to
- [ ] Compare attention patterns across 20+ test images

*Explainability*: This IS the explainability. Create comprehensive visualization suite:
- Layer-by-layer attention heatmaps overlaid on images
- Attention distance plot: how far patches attend at each layer
- Head specialization matrix showing focus patterns
- Quantitative: entropy, sparsity, attention to class token

*Success Criteria*:
- [ ] Correct attention rollout with residual connections
- [ ] Visualizations show progression from local to global attention
- [ ] Quantitative metrics reveal measurable patterns
- [ ] Written analysis explaining observed phenomena

**Checkpoint Deliverable**
Attention analysis notebook containing:
- Attention heatmaps for 10+ test images across all 12 layers
- Attention rollout visualization showing information flow to class token
- Head specialization analysis: "Head 3 focuses on edges, Head 7 on semantic objects"
- Quantitative analysis: attention distance (how far patches attend), entropy (focused vs diffuse)

**Self-Assessment**
- Can you implement attention rollout correctly with residual connections?
- Why do early layers have more local attention than later layers?
- How can you determine if a head is redundant?
- What's the difference between attention visualization and gradient-based attribution?

**Time**: Reading 3h + Implementation 5h + Analysis 2h = **~10 hours**

---

### Week 7: Comparative Analysis – ViT vs CNN

**Learning Objectives**
- Systematically compare vision transformer and convolutional architectures
- Analyze differences in learned representations (shape vs texture bias)
- Test adversarial robustness of both architectures

**Core Topics**
- Architectural comparison: inductive biases and their effects
- Representation analysis: what features drive predictions
- GradCAM for CNNs vs attention rollout for ViTs
- Adversarial robustness: FGSM, PGD, frequency-based attacks
- Error analysis: identifying failure modes and dataset biases

**Required Reading**
- 📄 Mahmood et al., "On Adversarial Robustness of ViTs" (2021)
  - [Paper](https://arxiv.org/abs/2103.15670) | Sections 3-4 | ~1 hour
- 💻 pytorch-grad-cam Documentation
  - [Repo](https://github.com/jacobgil/pytorch-grad-cam) | README, ViT examples | 45 min
- 📄 Explainability Survey (2023)
  - [Paper](https://arxiv.org/abs/2311.06786) | Sections 4-5 | ~1.5 hours

**Hands-On Exercise: Comprehensive Architecture Comparison**

*Part 1 - Training*:
- [ ] Train ViT-Small and ResNet-50 on same dataset (Tiny ImageNet or Food-101)
- [ ] Ensure fair comparison: same data, augmentation, training budget

*Part 2 - Visual Explanations*:
- [ ] Generate GradCAM heatmaps for CNN
- [ ] Generate attention rollout for ViT
- [ ] Create 50+ side-by-side comparisons: Input | ViT Attention | CNN GradCAM

*Part 3 - Adversarial Testing*:
- [ ] Implement FGSM and PGD attacks
- [ ] Test both models: generate robustness curves (accuracy vs ε)
- [ ] Visualize attention on adversarial examples

*Part 4 - Error Analysis*:
- [ ] Categorize mistakes: ViT-specific, CNN-specific, both
- [ ] Identify failure patterns
- [ ] Analyze: do architectures make different mistakes?

*Explainability*: The visualization comparison IS the explainability. Document where ViT focuses (global, semantic) vs CNN focuses (local, texture). Show adversarial examples reveal different failure modes.

*Success Criteria*:
- [ ] Both models trained to convergence on same data
- [ ] Quantitative comparison: accuracy, speed, memory, robustness
- [ ] 50+ visual comparisons clearly showing differences
- [ ] Written analysis: "ViTs focus on X, CNNs on Y because..."

**Checkpoint Deliverable**
Comprehensive comparison report:
- Accuracy metrics: both architectures on clean and adversarial examples
- Visualization comparison: 20+ examples showing where each architecture focuses
- Quantitative analysis: adversarial robustness curves (accuracy vs ε)
- Error taxonomy: what types of images fool ViT vs CNN
- Recommendations: when to choose each architecture

**Self-Assessment**
- Can you explain why ViTs are more robust to some adversarial attacks?
- What's the difference between texture bias and shape bias?
- How do inductive biases affect generalization?
- Can you design an experiment testing whether ViT uses global or local features?

**Time**: Reading 3h + Implementation 6h + Analysis 3h = **~12 hours**

---

### Week 8: Portfolio Project

**Learning Objectives**
- Integrate all module skills into comprehensive project
- Design and execute rigorous experimental analysis
- Communicate technical findings through professional documentation

**Core Topics**
- Experimental design: hypothesis formation, ablation studies
- Systematic evaluation: metrics, baselines, statistical significance
- Technical documentation: reproducibility, clarity, visual communication
- Portfolio presentation: GitHub repos, demos, reports

**Project Selection**
You will complete the **Attention Archaeology** portfolio project (details in Portfolio Project section below).

**Week 8 Tasks**
- [ ] Project setup: repo structure, environment, data prep (2 hours)
- [ ] Model training/fine-tuning with experiment tracking (4 hours)
- [ ] Comprehensive analysis: run all experiments, generate visualizations (6 hours)
- [ ] Documentation: technical report, code comments, README (3 hours)
- [ ] Polish: clean code, reproducibility check, demo notebook (1 hour)

**Checkpoint Deliverable**
Complete portfolio project meeting all requirements specified in project description (see Portfolio Project section)

**Self-Assessment**
- Does your project demonstrate mastery of all module concepts?
- Can someone reproduce your results from your documentation?
- Have you included rigorous quantitative analysis, not just qualitative observations?
- Does your project reveal interesting insights about vision transformers?

**Time**: **~16 hours** (project intensive week)

---

## Portfolio Project: Attention Archaeology

**Deep Forensic Analysis of Vision Transformers**

**The Funky Angle**:
Instead of just using ViT as a black-box classifier, you'll excavate its internals like an archaeologist uncovering ancient structures. Build a complete interpretability pipeline revealing what vision transformers actually learn, how they differ from CNNs, and why they fail.

**Technical Objectives**:
- Fine-tune ViT on domain-specific dataset (MedMNIST for medical imaging OR EuroSAT for satellite imagery)
- Implement comprehensive attention visualization toolkit
- Compare ViT attention patterns to CNN activation maps
- Conduct adversarial analysis showing architectural differences
- Rank attention head importance through systematic ablation

**Implementation Requirements**:

1. **Model Training** (~4 hours)
   - Fine-tune ViT-Base on chosen domain dataset
   - Train ResNet-50 baseline for comparison
   - Achieve competitive accuracy (>85% on chosen dataset)
   - Track all experiments with W&B

2. **Attention Analysis Suite** (~6 hours)
   - Implement attention extraction and rollout
   - Create multi-layer attention visualization (all 12 layers)
   - Implement head specialization analysis
   - Generate attention statistics: distance, entropy, sparsity
   - Compare to CNN GradCAM visualizations

3. **Adversarial Forensics** (~3 hours)
   - Implement FGSM, PGD, C&W attacks
   - Test both ViT and CNN
   - Generate robustness curves
   - Visualize attention on adversarial examples
   - Explain why robustness differs

4. **Failure Mode Analysis** (~2 hours)
   - Categorize mistakes: ViT-specific vs CNN-specific vs both
   - Identify challenging examples
   - Analyze attention patterns on failures
   - Document systematic failure modes

**Explainability & Analysis Requirements**:

- **Attention Progression**: Show attention evolution from layer 1→12 for 20+ examples, documenting how global context emerges
- **Head Specialization Matrix**: Document what each of 144 heads (12 layers × 12 heads) learns through visualization and statistics
- **Comparative Analysis**: 50+ side-by-side comparisons (Input | ViT Attention | CNN GradCAM) with written analysis
- **Quantitative Metrics**: 
  - Attention distance by layer (how far patches attend)
  - Attention entropy (focused vs diffuse)
  - Head importance ranking via ablation
  - Adversarial robustness curves
- **Failure Taxonomy**: Categorized examples with explanations: "ViT fails on X because attention doesn't capture Y"

**Deliverables**:

- [ ] **GitHub Repository** with clean, documented code
  - `models/`: ViT and CNN implementations
  - `analysis/`: Attention extraction, visualization, ablation tools
  - `experiments/`: Training scripts with configs
  - `README.md`: Setup, usage, results summary
  
- [ ] **Technical Report** (Jupyter notebook or markdown, 15-25 pages)
  - Introduction: problem, dataset, approach
  - Methodology: architecture details, training procedure, analysis methods
  - Results:
    - Quantitative: accuracy tables, attention statistics, ablation results
    - Qualitative: attention visualizations with analysis
    - Comparative: ViT vs CNN differences documented
  - Failure Analysis: taxonomy of mistakes with visual examples
  - Insights: "What we learned about how ViTs work"
  - Limitations and future work
  
- [ ] **Interactive Demo** (Streamlit/Gradio app or interactive notebook)
  - Upload image → see ViT attention at all layers
  - Compare to CNN explanations side-by-side
  - Explore attention patterns for different heads
  - Test adversarial robustness interactively

**Extension Ideas**:
- Implement multiple interpretation methods (LRP, Integrated Gradients, Attention Grad)
- Test on multiple ViT architectures (DeiT, Swin, BEiT)
- Add 3D attention visualizations showing layer→head→spatial structure
- Create attention-guided counterfactual explanations

---

## Resource Library

### Foundational Papers (Must Read)

**Transformer Fundamentals**
1. Vaswani et al., "Attention Is All You Need" (NeurIPS 2017)
   - [Paper](https://arxiv.org/abs/1706.03762) | [Annotated](https://nlp.seas.harvard.edu/annotated-transformer/)
   - Sections: 3.2 (Attention), 3.5 (Positional Encoding), 5 (Training)
   - Why: Foundational transformer architecture all ViT work builds on
   - Reading: ~2.5 hours

**Vision Transformers**
2. Dosovitskiy et al., "An Image is Worth 16x16 Words" (ICLR 2021)
   - [Paper](https://arxiv.org/abs/2010.11929) | [Code](https://github.com/google-research/vision_transformer)
   - Sections: 3 (Method), 4.5 (Attention), Figure 1 (Architecture)
   - Why: Original ViT paper, defines the paradigm
   - Reading: ~2 hours

3. Touvron et al., "DeiT: Data-efficient Image Transformers" (ICML 2021)
   - [Paper](https://arxiv.org/abs/2012.12877) | [Code](https://github.com/facebookresearch/deit)
   - Sections: 3 (Distillation), 4.3 (Teachers), 4.4 (Augmentation)
   - Why: Solves data hunger problem, makes ViT practical
   - Reading: ~1.5 hours

**Hierarchical Architectures**
4. Liu et al., "Swin Transformer" (ICCV 2021, Best Paper)
   - [Paper](https://arxiv.org/abs/2103.14030) | [Code](https://github.com/microsoft/Swin-Transformer)
   - Sections: 3.2 (Shifted Windows), 4 (Experiments), Figure 2
   - Why: Solves efficiency problem, enables dense prediction
   - Reading: ~2 hours

**Self-Supervised Learning**
5. Radford et al., "CLIP" (ICML 2021)
   - [Paper](https://arxiv.org/abs/2103.00020) | [Code](https://github.com/openai/CLIP)
   - Sections: 2 (Method), 3.3 (Prompt Engineering)
   - Why: Vision-language pre-training, zero-shot transfer
   - Reading: ~1.5 hours

6. Oquab et al., "DINOv2" (TMLR 2024)
   - [Paper](https://arxiv.org/abs/2304.07193) | [Code](https://github.com/facebookresearch/dinov2)
   - Sections: 3 (Data Curation), 4.2 (Training Improvements)
   - Why: Best self-supervised visual features, no labels needed
   - Reading: ~1.5 hours

---

### Recent Papers (State-of-Art 2023-2025)

**Architecture Improvements**
- Darcet et al., "Vision Transformers Need Registers" (ICLR 2024)
  - [Paper](https://arxiv.org/abs/2309.16588)
  - Why: Elegant solution to ViT artifacts, improves dense tasks

- Liu et al., "EfficientViT: Memory Efficient Vision Transformer" (CVPR 2023)
  - [Paper](https://arxiv.org/abs/2305.07027)
  - Why: Memory is bottleneck, not FLOPs

- Xia et al., "ViT-CoMer" (CVPR 2024, Highlight)
  - [Paper](https://arxiv.org/abs/2403.07392) | [Code](https://github.com/Traffic-X/ViT-CoMer)
  - Why: Hybrid CNN-ViT achieving SOTA on detection/segmentation

**Interpretability & Analysis**
- Chefer et al., "Transformer Interpretability Beyond Attention" (CVPR 2021)
  - [Paper](https://arxiv.org/abs/2005.00928) | [Code](https://github.com/hila-chefer/Transformer-Explainability)
  - Why: Raw attention misleads, gradient-weighted is better

- "Explainability of Vision Transformers: A Comprehensive Review" (2023)
  - [Paper](https://arxiv.org/abs/2311.06786)
  - Why: Complete taxonomy of ViT explainability methods

**Robustness & Scaling**
- Mahmood et al., "On Adversarial Robustness of Vision Transformers" (2021)
  - [Paper](https://arxiv.org/abs/2103.15670)
  - Why: ViTs have different robustness properties than CNNs

- Zhai et al., "Scaling Vision Transformers" (CVPR 2022)
  - [Paper](https://arxiv.org/abs/2106.04560)
  - Why: Scaling laws for ViT, how to design compute-optimal models

---

### Textbook Chapters & Comprehensive Resources

- **MIT 6.390 Course Notes**: Chapter on Transformers
  - [Link](https://introml.mit.edu/notes/transformers.html)
  - Rigorous mathematical treatment with proofs
  
- **Foundations of Computer Vision (MIT)**: Chapter 26 on Transformers
  - [Link](https://visionbook.mit.edu/transformers.html)
  - Computer vision perspective with visual intuition
  
- **Stanford CS231n Spring 2025**: Lecture 9 on Vision Transformers
  - [Slides](https://cs231n.stanford.edu/slides/2025/lecture_9.pdf)
  - Lecture-level overview with recent advances

---

### Tutorials & Guides

**Interactive Tutorials**
- UvA Deep Learning Tutorials: Vision Transformers
  - [Notebook](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial15/Vision_Transformer.html)
  - [Colab](https://colab.research.google.com/github/phlippe/uvadlc_notebooks/blob/master/docs/tutorial_notebooks/tutorial15/Vision_Transformer.ipynb)
  - Complete ViT from scratch with training on CIFAR-10
  
- HuggingFace: Fine-tune ViT Tutorial
  - [Docs](https://huggingface.co/docs/transformers/tasks/image_classification)
  - Production-ready code using Trainer API

**Conceptual Explanations**
- Jay Alammar: The Illustrated Transformer
  - [Blog](https://jalammar.github.io/illustrated-transformer/)
  - Best visual explanation of attention mechanisms
  
- AI Summer: How ViT Works in 10 Minutes
  - [Blog](https://theaisummer.com/vision-transformer/)
  - Concise ViT overview with clear diagrams
  
- 3Blue1Brown: Attention in Transformers
  - [Video](https://www.youtube.com/watch?v=eMlx5fFNoYc)
  - Mathematical intuition with beautiful animations

**Implementation Guides**
- Roboflow: Train ViT on Custom Dataset
  - [Blog](https://blog.roboflow.com/how-to-train-vision-transformer/)
  - [Notebook](https://github.com/roboflow/notebooks/blob/main/notebooks/train-vision-transformer-classification-on-custom-data.ipynb)
  - Practical guide with data preprocessing
  
- DigitalOcean: ViT Implementation Guide
  - [Tutorial](https://www.digitalocean.com/community/tutorials/vision-transformer-for-computer-vision)
  - Step-by-step code walkthrough

---

### Code Repositories

**Official Implementations**
- Google Research: Vision Transformer (Original)
  - [Repo](https://github.com/google-research/vision_transformer)
  - JAX/Flax implementation, pre-trained weights
  
- Facebook Research: DeiT
  - [Repo](https://github.com/facebookresearch/deit)
  - PyTorch, distillation code, pre-trained models
  
- Microsoft: Swin Transformer
  - [Repo](https://github.com/microsoft/Swin-Transformer)
  - Complete implementation for classification/detection/segmentation

**Educational Implementations**
- tintn/vision-transformer-from-scratch
  - [Repo](https://github.com/tintn/vision-transformer-from-scratch)
  - Heavily commented, minimal ViT implementation
  - Great for learning, not for production

**Production Libraries**
- timm (PyTorch Image Models)
  - [Repo](https://github.com/huggingface/pytorch-image-models)
  - [Docs](https://timm.fast.ai/)
  - 1000+ pre-trained models, unified API
  
- HuggingFace Transformers
  - [Repo](https://github.com/huggingface/transformers)
  - [Docs](https://huggingface.co/docs/transformers/)
  - Production-ready, excellent documentation

---

### Tools & Frameworks

**Model Libraries**
- **timm**: `pip install timm`
  - Load any vision transformer: `timm.create_model('vit_base_patch16_224', pretrained=True)`
  - 300+ ViT variants with pre-trained weights
  
- **HuggingFace Transformers**: `pip install transformers`
  - Unified API for vision and language transformers
  - Trainer API simplifies fine-tuning

**Interpretability Tools**
- **ViT-Prisma**: `pip install vit-prisma`
  - [Repo](https://github.com/soniajoseph/ViT-Prisma)
  - Mechanistic interpretability library for ViT
  - Attention visualization, activation patching, SAEs
  
- **pytorch-grad-cam**: `pip install grad-cam`
  - [Repo](https://github.com/jacobgil/pytorch-grad-cam)
  - GradCAM and variants for ViT and CNN
  - 20+ explanation methods
  
- **BertViz**: `pip install bertviz`
  - [Repo](https://github.com/jessevig/bertviz)
  - Multi-scale attention visualization
  - Head view, model view, neuron view

**Experiment Tracking**
- **Weights & Biases**: `pip install wandb`
  - [Docs](https://docs.wandb.ai/)
  - Real-time training visualization, hyperparameter sweeps
  
- **TensorBoard**: Built into PyTorch
  - `from torch.utils.tensorboard import SummaryWriter`
  - Scalars, images, histograms, graphs

**Visualization**
- **Transformer Explainer** (Interactive)
  - [Web App](https://poloclub.github.io/transformer-explainer/)
  - Interactive visualization of transformer internals
  
- **AttentionViz**
  - [Web App](https://attentionviz.com/)
  - Global view of attention patterns

---

### Datasets

**Small Scale (Quick Iteration)**
- **CIFAR-10**: 60K 32×32 images, 10 classes
  - `torchvision.datasets.CIFAR10(root, download=True)`
  - Size: 162 MB | Time: <5 min to download
  
- **CIFAR-100**: 60K 32×32 images, 100 classes
  - `torchvision.datasets.CIFAR100(root, download=True)`
  - Size: 162 MB | Time: <5 min

- **Tiny ImageNet**: 100K 64×64 images, 200 classes
  - [Download](http://cs231n.stanford.edu/tiny-imagenet-200.zip)
  - Size: 237 MB | Time: ~10 min

**Medium Scale (Fine-tuning)**
- **Food-101**: 101K food images, 101 categories
  - `torchvision.datasets.Food101(root, download=True)`
  - Size: 4.65 GB | Time: ~1 hour
  
- **Oxford-IIIT Pets**: 7.4K pet images, 37 breeds
  - `torchvision.datasets.OxfordIIITPet(root, download=True)`
  - Size: 800 MB | Time: ~15 min
  
- **Flowers-102**: 8.2K flower images, 102 species
  - `torchvision.datasets.Flowers102(root, download=True)`
  - Size: 328 MB | Time: ~10 min

**Large Scale (Serious Training)**
- **ImageNet-1K**: 1.28M images, 1000 classes
  - [Kaggle](https://www.kaggle.com/c/imagenet-object-localization-challenge)
  - Size: 160+ GB | Time: many hours
  - Requires preprocessing: resize to 224×224

**Domain-Specific (Portfolio Projects)**
- **MedMNIST Collection**: Medical imaging datasets
  - [Website](https://medmnist.com/) | `pip install medmnist`
  - PathMNIST, ChestMNIST, DermaMNIST, etc.
  - Standardized 28×28 or 64×64 images
  
- **EuroSAT**: Satellite imagery, 27K images, 10 land use classes
  - [Download](https://github.com/phelber/EuroSAT)
  - Size: 3 GB | 64×64 RGB images

---

## Troubleshooting & Extensions

### Common Issues

**Issue 1: ViT training is unstable/diverges**
- **Symptoms**: Loss explodes, NaN gradients, oscillating loss
- **Causes**: Learning rate too high, no warmup, incorrect layer norm position
- **Solutions**:
  - Use learning rate warmup (e.g., 10K steps linear warmup)
  - Clip gradients: `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`
  - Check layer norm position: should be pre-norm (before attention/MLP)
  - Reduce learning rate by 10×, gradually increase

**Issue 2: ViT underfits/poor accuracy on small datasets**
- **Symptoms**: Training accuracy <90%, huge train-test gap
- **Causes**: ViT needs more data than available, weak data augmentation
- **Solutions**:
  - Use pre-trained weights, don't train from scratch
  - Strong augmentation: RandAugment, Mixup, CutMix
  - Try DeiT distillation instead of standard training
  - Reduce model size (ViT-Tiny instead of ViT-Base)

**Issue 3: Attention visualization shows uniform attention**
- **Symptoms**: All attention weights approximately equal, no clear patterns
- **Causes**: Early in training, layer norm issues, visualizing wrong layer
- **Solutions**:
  - Check layer: early layers are more uniform, visualize layer 8-12
  - Remove layer norm before visualizing attention weights
  - Use attention rollout instead of raw attention from single layer
  - Ensure model is properly trained (not random initialization)

**Issue 4: Out of memory during training**
- **Symptoms**: CUDA OOM error, training crashes
- **Causes**: Batch size too large, accumulating gradients, large image size
- **Solutions**:
  - Reduce batch size by 2×, use gradient accumulation to compensate
  - Use mixed precision training: `torch.cuda.amp.autocast()`
  - Reduce image resolution (224 → 192 or 160)
  - Use gradient checkpointing: `model.gradient_checkpointing_enable()`

**Issue 5: Can't reproduce paper results**
- **Symptoms**: Your accuracy is 5-10% lower than paper reports
- **Causes**: Hyperparameters differ, dataset preprocessing wrong, training too short
- **Solutions**:
  - Check official implementation for exact hyperparameters
  - Verify data augmentation matches paper (very important!)
  - Train for full epoch count (ViT often needs 300+ epochs)
  - Use same optimizer and learning rate schedule

---

### Debugging Strategies

**For Attention Mechanism**:
1. Test on simple task first (copy sequence, sort numbers)
2. Visualize attention matrix as heatmap every 10 epochs
3. Check attention weight sums: should sum to 1 along last dimension
4. Verify broadcasting: print shapes at each step

**For Training**:
1. Overfit single batch first (should reach 100% accuracy)
2. Add complexity gradually: more data → more augmentation → full model
3. Compare learning curves to paper/baselines
4. Log everything: loss, accuracy, gradient norms, learning rate

**For Interpretability**:
1. Test visualization on known patterns (e.g., stripe detection)
2. Compare multiple methods: if they all agree, more confident
3. Sanity checks: remove attended regions, does accuracy drop?
4. Use quantitative metrics, not just visual inspection

---

### Going Deeper (Optional)

**Advanced ViT Topics**:
- **Architecture Search**: How to find optimal ViT designs
- **Efficient Training**: FlashAttention, mixed precision, distributed training
- **Video Transformers**: Extending ViT to temporal dimension
- **3D Vision**: ViT for point clouds and volumetric data
- **Multimodal**: Combining vision and language (CLIP, Flamingo)

**Research Directions**:
- Long-context ViT: handling high-resolution images efficiently
- Continual learning: updating ViT without catastrophic forgetting
- Robust ViT: defending against adversarial attacks
- Explainability: new methods for interpreting attention
- Compression: pruning, quantization, knowledge distillation

**Advanced Readings**:
- "How to train your ViT? Data, Augmentation, and Regularization" (NeurIPS 2021)
- "Beit: BERT Pre-Training of Image Transformers" (ICLR 2022)
- "Masked Autoencoders Are Scalable Vision Learners" (CVPR 2022)
- "Learning to Prompt for Vision-Language Models" (IJCV 2022)

---

### Next Steps: Connection to Module 2

This module established transformer fundamentals and vision adaptations. **Module 2: Diffusion Models** will:
- Use transformer backbones in latent diffusion models
- Apply attention mechanisms to denoising process
- Compare discriminative (ViT classification) vs generative (diffusion) modeling
- Leverage pre-trained ViT features for conditioning diffusion models

**Skills from Module 1 that transfer**:
- Attention mechanism understanding → apply to U-Net architectures
- Training large models → diffusion models require similar care
- Visualization techniques → visualize diffusion process and denoising
- Architectural choices → design efficient diffusion architectures

**Preparation for Module 2**:
- Review probability theory: Gaussian distributions, KL divergence
- Understand variational inference basics (ELBO)
- Familiarity with U-Net architecture (encoder-decoder)

---

## Summary & Success Metrics

**By completing this module, you should be able to**:

✅ Explain and implement transformer attention from first principles  
✅ Build and train vision transformers from scratch in PyTorch  
✅ Fine-tune pre-trained ViTs on custom datasets efficiently  
✅ Visualize and interpret attention patterns across layers  
✅ Compare ViT vs CNN architectures systematically  
✅ Analyze what vision transformers learn and when they fail  
✅ Apply modern ViT variants (DeiT, Swin, CLIP, DINOv2)  
✅ Create production-quality ViT applications  

**Success looks like**:
- Complete portfolio project with rigorous analysis
- Deep understanding of attention mechanisms (can teach others)
- Practical skills deploying ViTs in real applications
- Critical thinking about architectural choices
- Ability to read and implement recent ViT papers

**Time Investment**: ~94 hours total
- Weeks 1-7: ~78 hours (11-12 hours/week)
- Week 8: ~16 hours (project intensive)

**You're ready for Module 2 when**:
Your portfolio project demonstrates mastery of vision transformers through both technical implementation and analytical depth. You can confidently explain when and why to use transformers for vision tasks.

---