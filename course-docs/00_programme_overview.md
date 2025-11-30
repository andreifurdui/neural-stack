# Advanced AI Engineering Program
## Self-Directed Curriculum for Modern AI Systems

**Version**: 1.0  
**Duration**: 58 weeks (core) + 8 weeks (optional)  
**Last Updated**: November 2025

---

## 1. PROGRAM OVERVIEW

This curriculum bridges the gap between classical machine learning education and modern AI engineering practice. It covers foundational architectures that emerged post-2020 (transformers, diffusion models), scales to production systems through distributed training and MLOps, explores agentic AI and reinforcement learning for autonomous systems, and addresses privacy-preserving techniques increasingly critical for real-world deployment.

The program is structured as seven core modules plus one optional advanced module, each building progressively on previous knowledge. Every module combines theoretical understanding with hands-on implementation, culminating in a portfolio project that demonstrates practical mastery. The emphasis is on deep comprehension rather than surface-level familiarity—implementing algorithms from scratch, comparing approaches empirically, and applying techniques to novel problems.

This is a self-paced program designed for approximately 10-12 hours per week of focused study. Each module takes 6-10 weeks depending on complexity. The curriculum assumes solid foundations in machine learning, deep learning, and Python programming, along with familiarity with PyTorch or similar frameworks.

---

## 2. CORE MODULES

### Module 1: Vision Transformers & Modern Architectures
**Duration**: 8 weeks | **Level**: Foundation | **Prerequisites**: CNN understanding, basic ML

**Learning Objectives**
- Master transformer architecture fundamentals and attention mechanisms
- Understand vision-specific adaptations (patch embeddings, positional encodings)
- Implement and fine-tune modern vision transformers
- Compare transformer vs CNN architectures empirically

**Core Topics**
- **Transformer Fundamentals**: Self-attention, multi-head attention, positional encodings, encoder-decoder architecture
- **Vision Transformers**: ViT architecture, DeiT (distillation), Swin Transformer (hierarchical), patch embedding strategies
- **Self-Supervised Vision**: CLIP (vision-language), DINOv2 (self-supervised ViT), contrastive learning principles
- **Implementation & Fine-tuning**: Transfer learning strategies, LoRA for efficient fine-tuning, architecture comparison methodologies

**Practical Components**
- Implement attention mechanism from scratch
- Build minimal ViT implementation
- Fine-tune pre-trained vision transformers on custom datasets
- Systematic comparison: ViT vs CNN architectures (accuracy, speed, memory)

**Portfolio Project**
Architecture modernization project: Replace CNN-based system with vision transformer, conduct comprehensive performance analysis (mIoU, inference speed, memory footprint, edge deployment feasibility), document architectural decisions and tradeoffs.

**Module Connections**
Foundation module enabling all subsequent work. Transformer understanding directly supports Module 2 (diffusion uses transformers), Module 5 (LLM architectures), and provides modern architectural thinking throughout. Self-attention concepts apply to sequence modeling in Module 6.

---

### Module 2: Diffusion Models & Modern Generative AI
**Duration**: 8 weeks | **Level**: Intermediate | **Prerequisites**: Module 1, probability theory

**Learning Objectives**
- Understand diffusion process mathematics (forward/reverse processes, score-based models)
- Master denoising objectives and training procedures
- Implement diffusion models from scratch and at scale
- Compare diffusion to alternative generative approaches (GANs, VAEs)

**Core Topics**
- **Diffusion Fundamentals**: Forward diffusion (noise scheduling), reverse diffusion (denoising), score-based models, Langevin dynamics, variational lower bound
- **Modern Architectures**: DDPM (Denoising Diffusion Probabilistic Models), DDIM (faster sampling), Latent Diffusion (Stable Diffusion), cascade diffusion
- **Conditional Generation**: Classifier guidance, classifier-free guidance, conditioning strategies
- **Training & Sampling**: Loss functions, noise schedules, sampling strategies (DDPM vs DDIM), quality evaluation (FID, IS)

**Practical Components**
- Implement DDPM from scratch on toy datasets
- Train conditional diffusion models
- Experiment with sampling strategies and schedules
- Quality evaluation and comparison to GANs

**Portfolio Project**
Comparative generative modeling study: Implement diffusion model for specific generation task, compare training stability and sample quality to GAN baseline, document failure modes and convergence properties, create sampling efficiency analysis.

**Module Connections**
Builds on Module 1 transformer knowledge (modern diffusion uses transformer backbones). Sampling strategies relate to Module 5 LLM generation. Provides generative modeling foundation complementing discriminative models throughout program.

---

### Module 3: Large-Scale Distributed Training
**Duration**: 8 weeks | **Level**: Intermediate-Advanced | **Prerequisites**: Modules 1-2

**Learning Objectives**
- Understand distributed training paradigms (data/model/pipeline parallelism)
- Master gradient synchronization and communication patterns
- Implement FSDP and DeepSpeed for large model training
- Gain practical cloud ML infrastructure experience

**Core Topics**
- **Distributed Training Fundamentals**: Data vs model vs pipeline parallelism, gradient synchronization (AllReduce, Ring-AllReduce), communication bottlenecks, bandwidth analysis
- **Memory Optimization**: Mixed precision training (FP16, BF16), gradient checkpointing, activation recomputation, gradient accumulation
- **Modern Frameworks**: PyTorch DDP, FSDP (Fully Sharded Data Parallel), DeepSpeed ZeRO stages, Megatron-LM concepts
- **Cloud Infrastructure**: AWS/GCP multi-GPU setup, cost analysis, profiling tools, scaling efficiency measurement

**Practical Components**
- Profile single-GPU vs multi-GPU training bottlenecks
- Implement FSDP for large vision or diffusion models
- Cloud deployment and cost-benefit analysis
- Scaling efficiency experiments (1-64 GPUs)

**Portfolio Project**
Scaling analysis project: Benchmark model training across different GPU counts, document scaling efficiency curves, analyze communication vs compute tradeoffs, create cost analysis comparing on-premise vs cloud infrastructure, produce scaling best practices guide.

**Module Connections**
Enables training of large models from Modules 1-2 at scale. Essential infrastructure knowledge for Module 4 MLOps. Understanding distributed systems prepares for Module 5 (LLM training scale) and Module 7 (federated learning distributed nature).

---

### Module 4: Modern MLOps & Production ML Systems
**Duration**: 8 weeks | **Level**: Advanced | **Prerequisites**: Modules 1-3

**Learning Objectives**
- Design end-to-end ML pipelines with modern tools
- Master experiment tracking, versioning, and reproducibility
- Implement production deployment and monitoring systems
- Build systematic evaluation and testing frameworks

**Core Topics**
- **MLOps Principles**: ML lifecycle management, data versioning (DVC), experiment tracking (MLflow, W&B), model registries, feature stores
- **Modern Frameworks**: Ray (distributed ML), Kubeflow (Kubernetes-native workflows), model serving (KServe, Ray Serve, BentoML)
- **Production Deployment**: API serving, batch inference, model monitoring, A/B testing, canary deployments
- **Quality & Testing**: Model validation, performance testing, integration testing, CI/CD for ML

**Practical Components**
- Migrate training pipeline to modern MLOps stack
- Implement end-to-end workflow: data versioning → training → evaluation → deployment
- Set up experiment tracking and hyperparameter optimization (Ray Tune)
- Deploy model with monitoring and logging

**Portfolio Project**
Production ML pipeline: Design and implement complete MLOps workflow for model from previous modules, include data versioning, automated training, systematic evaluation, production deployment with monitoring, document pipeline architecture and operational procedures, create reusable pipeline templates.

**Module Connections**
Operationalizes all previous modules (1-3) into production systems. Provides infrastructure for deploying Module 5 agents and Module 6 RL policies. Creates systematic evaluation framework applicable to Module 7 privacy-preserving systems.

---

### Module 5: Agentic AI & LLM Systems
**Duration**: 8 weeks | **Level**: Intermediate-Advanced | **Prerequisites**: Modules 1-4

**Learning Objectives**
- Master LLM fundamentals and fine-tuning techniques
- Build autonomous agent systems with reasoning and tool use
- Implement RAG and memory systems for context-aware applications
- Understand agent evaluation and reliability considerations

**Core Topics**
- **LLM Fundamentals**: Transformer decoder architecture (GPT-style), tokenization strategies, inference techniques (greedy, sampling, beam search), prompt engineering
- **Fine-tuning & Adaptation**: LoRA/QLoRA (parameter-efficient fine-tuning), RLHF (Reinforcement Learning from Human Feedback), DPO (Direct Preference Optimization), instruction tuning
- **Agentic Systems**: ReAct (Reasoning + Acting), Chain-of-Thought, Tree-of-Thoughts, tool use and function calling, planning algorithms
- **RAG & Memory**: Retrieval-Augmented Generation, vector databases, embedding strategies, short-term and long-term memory systems
- **Production & Evaluation**: Agent deployment, hallucination mitigation, benchmark evaluation, multi-agent systems

**Practical Components**
- Fine-tune small LLM with LoRA on custom task
- Build ReAct agent with multiple tools
- Implement RAG system with vector database
- Agent evaluation and reliability testing

**Portfolio Project**
Autonomous agent system: Design agent for specific domain task requiring multi-step reasoning, implement tool use and memory systems, create RAG knowledge base, evaluate agent performance systematically, document failure modes and reliability analysis, compare to alternative approaches.

**Module Connections**
Builds on Module 1 transformers for LLM architecture understanding. RLHF directly prepares for Module 6 RL fundamentals. Distributed training (Module 3) and MLOps (Module 4) enable LLM deployment. Complements Module 6 by offering LLM reasoning as alternative to learned RL policies.

---

### Module 6: Reinforcement Learning for Embodied AI
**Duration**: 10 weeks | **Level**: Advanced | **Prerequisites**: Modules 1-5

**Learning Objectives**
- Master RL fundamentals (MDPs, value functions, policy gradients)
- Implement modern deep RL algorithms (DQN, PPO, SAC)
- Apply RL to sequential decision-making tasks
- Understand sim-to-real transfer and embodied AI considerations

**Core Topics**
- **RL Fundamentals**: Markov Decision Processes, value functions, Bellman equations, dynamic programming (value iteration, policy iteration), Monte Carlo methods, temporal difference learning
- **Deep RL Algorithms**: DQN and variants (Double DQN, Dueling DQN), policy gradient methods (REINFORCE, Actor-Critic), PPO (Proximal Policy Optimization), SAC (Soft Actor-Critic)
- **Advanced Concepts**: Exploration strategies, reward shaping, curriculum learning, off-policy vs on-policy methods
- **Embodied AI**: Simulation environments (Gymnasium, PyBullet, Isaac Gym), sim-to-real transfer, domain randomization

**Practical Components**
- Implement tabular RL algorithms (Q-learning, SARSA)
- Build DQN for discrete action spaces
- Implement PPO for continuous control
- Custom environment development and policy training

**Portfolio Project**
RL policy development: Design RL environment for specific task, implement and train policy using modern algorithms (PPO or SAC), compare to alternative approaches (rule-based, supervised learning, Module 5 LLM agent), analyze learned behaviors and failure modes, document reward engineering and hyperparameter tuning process.

**Module Connections**
RLHF from Module 5 provides RL foundation. Contrasts LLM reasoning (Module 5) with learned policies for sequential decisions. Distributed training (Module 3) enables large-scale RL training. Can apply Module 7 privacy techniques to RL in sensitive domains.

---

### Module 7: Privacy-Preserving ML & Federated Learning
**Duration**: 8 weeks | **Level**: Advanced | **Prerequisites**: All previous modules

**Learning Objectives**
- Master privacy definitions and guarantees (differential privacy, secure computation)
- Implement federated learning for distributed training
- Apply privacy-preserving techniques to various ML tasks
- Understand privacy-utility tradeoffs and evaluation

**Core Topics**
- **Privacy Fundamentals**: Threat models, differential privacy definitions (ε-δ), composition theorems, privacy budget tracking, privacy-utility tradeoffs
- **Federated Learning**: Federated Averaging (FedAvg), communication efficiency, personalized federated learning, secure aggregation, vertical vs horizontal FL
- **Private Training**: DP-SGD (Differentially Private SGD), gradient clipping and noise addition, privacy accounting, private aggregation protocols
- **Advanced Topics**: Secure multi-party computation basics, homomorphic encryption concepts, on-device learning

**Practical Components**
- Implement basic differential privacy mechanisms
- Simulate federated learning across multiple clients
- Train models with DP-SGD and measure privacy-utility tradeoff
- Privacy auditing and attack evaluation

**Portfolio Project**
Privacy-preserving ML system: Implement federated learning or DP training for model from earlier modules, evaluate privacy guarantees and utility preservation, conduct privacy attack analysis (membership inference, model inversion), document privacy-utility tradeoffs, create deployment guidelines for privacy-sensitive applications.

**Module Connections**
Applies to all previous modules—can add privacy to vision models (Module 1), generative models (Module 2), distributed training (Module 3), production systems (Module 4), LLMs/agents (Module 5), and RL (Module 6). Provides essential capability for real-world deployment in regulated or sensitive domains.

---

### Module 8: 3D Vision & Neural Rendering (Optional)
**Duration**: 8 weeks | **Level**: Advanced | **Prerequisites**: Modules 1-2, linear algebra

**Learning Objectives**
- Understand classical 3D vision fundamentals (geometry, reconstruction)
- Master neural rendering techniques (NeRF, 3D Gaussian Splatting)
- Implement novel view synthesis systems
- Bridge classical and neural 3D approaches

**Core Topics**
- **Classical 3D Vision**: Camera models and calibration, epipolar geometry, Structure from Motion (SfM), multi-view stereo, bundle adjustment
- **Neural 3D Representations**: Neural Radiance Fields (NeRF), volume rendering, ray marching, Instant-NGP (hash encoding)
- **Modern Methods**: 3D Gaussian Splatting, point cloud processing, mesh reconstruction
- **Applications**: Novel view synthesis, 3D reconstruction, scene understanding

**Practical Components**
- Implement camera geometry algorithms
- Build minimal NeRF from scratch
- Train 3D Gaussian Splatting on captured scenes
- Novel view synthesis quality evaluation

**Portfolio Project**
3D reconstruction system: Capture multi-view dataset, implement classical SfM pipeline, train neural rendering model (NeRF or 3DGS), generate novel views, compare classical vs neural approaches, create interactive 3D visualization, document accuracy and rendering performance.

**Module Connections**
Builds on Module 1 vision architectures and Module 2 neural representations. Extends 2D vision to 3D understanding. Relevant for robotics, AR/VR, and spatial AI applications. Can incorporate Module 3 distributed training for large-scale 3D datasets.

---

## 3. PROGRAM ARCHITECTURE

### Module Dependencies

```
                    Module 1: Vision Transformers
                             |
                             v
                    Module 2: Diffusion Models
                             |
                    +--------+--------+
                    v                 v
         Module 3: Distributed --> Module 4: MLOps
                    |                 |
                    +--------+--------+
                             |
                             v
                    Module 5: Agentic AI
                             |
                    +--------+--------+
                    v                 v
         Module 6: RL        Module 7: Privacy ML
                    |                 |
                    +--------+--------+
                             |
                             v
                  Module 8: 3D Vision (Optional)

Legend:
  →  Direct prerequisite
  |  Sequential progression
  ↓  Enables/supports
```

### Timeline Overview

**Phase 1** (Weeks 1-16): Modern Architectures - Modules 1-2  
**Phase 2** (Weeks 17-32): Scale & Production - Modules 3-4  
**Phase 3** (Weeks 33-40): Product Development - Module 5  
**Phase 4** (Weeks 41-58): Algorithmic Depth - Modules 6-7  
**Optional** (Weeks 59-66): Advanced Topics - Module 8

Total: 58 weeks core + 8 weeks optional = 66 weeks maximum

Program allows flexible pacing—modules can be extended for deeper exploration or condensed for faster progression. Module order is optimized for knowledge building but some swaps are possible (e.g., Module 7-8 order is flexible).

---

## 4. LEARNING FRAMEWORK

**Time Commitment**: 10-12 hours per week minimum
- Core module work: 8-10 hours (reading, implementation, projects)
- Supplementary activities: 2-4 hours (papers, documentation, exploration)

**Module Completion Criteria**: Move to next module when you can:
1. Explain core concepts clearly without references
2. Implement key algorithms from scratch
3. Compare approach to 3+ recent papers
4. Apply techniques to novel problems beyond tutorials
5. Deliver professional-quality portfolio artifact

**Assessment**: Self-evaluation through practical demonstration rather than formal testing. Each module ends with portfolio project serving as capability validation. Regular implementation of algorithms from scratch ensures deep understanding beyond framework usage.

**Parallel Learning**: Throughout program, maintain paper reading routine (1-2 papers/week from recent conferences), engage with technical communities, and document learning through technical writing. These activities compound knowledge acquisition and build research fluency.

---

## 5. PORTFOLIO & DELIVERABLES

Each module produces one major portfolio project demonstrating practical mastery. Projects should be:
- Fully implemented and reproducible
- Documented with clear setup instructions
- Compared to baselines or alternative approaches
- Analyzed for performance, limitations, and tradeoffs

**Deliverable Format**: GitHub repository + technical report as Jupyter notebook or GitHub Gist. Reports should synthesize learning, document implementation decisions, present empirical results, and reflect on challenges encountered. Focus on substance over presentation—clarity and technical depth matter more than polish.

**Portfolio Collection**: Maintain central repository or website linking all module projects. This serves as both learning record and demonstration of capabilities. Projects build progressively in sophistication, showing knowledge accumulation across program.

---

## 6. RESOURCES

Resources for each module fall into four categories:

**Foundational Papers**: Seminal works establishing core concepts (2-3 per module)  
**Recent Research**: Papers from 2023-2025 showing current state-of-art (3-5 per module)  
**Implementation Resources**: Official framework docs, tutorial repos, reference implementations  
**Textbooks**: Comprehensive references for deep theoretical understanding (optional)

Resource selection prioritizes:
1. Authority (original papers, official docs, recognized textbooks)
2. Clarity (pedagogical quality, code readability)
3. Recency (up-to-date with current practice)
4. Free availability where possible

Specific resource lists will be developed per module during detailed curriculum expansion. Core frameworks used across program include PyTorch, HuggingFace libraries, standard ML tooling (NumPy, scikit-learn), and visualization tools (matplotlib, tensorboard).

Staying current requires monitoring arXiv (cs.CV, cs.LG, cs.AI), major conference proceedings (NeurIPS, ICML, CVPR, ICLR), and key research labs/groups. Emphasis on understanding principles over chasing trends—foundational knowledge enables quickly absorbing new developments.

---

## DOCUMENT USAGE

This curriculum provides high-level program structure and module definitions. Each module will be expanded into detailed weekly breakdowns with specific papers, exercises, and implementation milestones during study. This document serves as roadmap and reference for maintaining program coherence across extended learning period.

Program is self-directed and self-paced—adapt timeline, depth, and focus areas based on interests and goals. Core structure (module order, dependencies) is optimized for knowledge building, but individual module content can be customized to emphasize different domains or applications.
