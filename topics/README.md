# Research themes

A selective reading guide organized by research question. Each theme offers a short path through representative papers already listed in this repository. Read the entries in order for an introduction followed by complementary approaches or limitations.

- [Foundations and core architectures](#foundations-and-core-architectures)
- [Memory, recall, and length generalization](#memory-recall-and-length-generalization)
- [State tracking and expressivity](#state-tracking-and-expressivity)
- [In-context learning and learning dynamics](#in-context-learning-and-learning-dynamics)
- [Gating, linear attention, and test-time learning](#gating-linear-attention-and-test-time-learning)
- [Stability, initialization, and optimization](#stability-initialization-and-optimization)
- [Hybrid models and applications](#hybrid-models-and-applications)

For broader coverage, browse the [recent papers](../README.md#2026-papers), [selected highlights](../README.md#selected-highlights), and [historical lists](../archive/README.md).

## Foundations and core architectures

How do structured state-space models represent sequences, and how do their algorithms relate to attention?

1. [S4: Efficiently Modeling Long Sequences with Structured State Spaces](https://arxiv.org/abs/2111.00396) — Start with the structured parameterization that makes long-sequence SSM computation practical.
2. [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) — Follow the move from fixed dynamics to input-dependent selection and a hardware-aware recurrent algorithm.
3. [Mamba-2 / SSD: Transformers are SSMs](https://arxiv.org/abs/2405.21060) — Connect SSMs and attention through structured matrices, then see how that connection leads to efficient algorithms.

## Memory, recall, and length generalization

What can a recurrent state retain, and why can performance change beyond the training context length?

1. [Repeat After Me: Transformers are Better than State Space Models at Copying](https://arxiv.org/abs/2402.01032) — Understand the copying and retrieval limitations studied for models with a fixed-size latent state.
2. [Revisiting associative recall in modern recurrent models](https://arxiv.org/abs/2508.19029) — Examine how learning rate and model scaling can change apparent recall gaps between architectures.
3. [Understanding and Improving Length Generalization in Recurrent Models](https://arxiv.org/abs/2507.02782) — Follow the unexplored-states hypothesis and training interventions that expose models to a wider range of recurrent states.

## State tracking and expressivity

Which transition structures allow a model to update a discrete state reliably across a sequence?

1. [The Illusion of State in State-Space Models](https://arxiv.org/abs/2404.08819) — Begin with the state-tracking limitations of the SSM architectures analyzed in this paper.
2. [Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues](https://arxiv.org/abs/2411.12537) — See how changing the allowed eigenvalues alters the capabilities of linear recurrent transitions.
3. [DeltaProduct: Improving State-Tracking in Linear RNNs via Householder Products](https://arxiv.org/abs/2502.10297) — Explore products of transition factors as a way to trade additional per-token updates for greater expressivity.
4. [PD-SSM: Structured Sparse Transition Matrices to Enable State Tracking in State-Space Models](https://arxiv.org/abs/2509.22284) — Compare a sparse transition construction that can emulate finite-state automata while retaining efficient scans.

## In-context learning and learning dynamics

How does a model learn to use information in the current context, rather than relying on what it memorized during training?

1. [Training Dynamics of In-Context Learning in Linear Attention](https://arxiv.org/abs/2501.16265) — Study how parameterization changes the acquisition of in-context linear regression through gradient descent.
2. [From Markov to Laplace: How Mamba In-Context Learns Markov Chains](https://arxiv.org/abs/2502.10178) — Examine a setting where Mamba learns a statistically optimal in-context estimator for Markov chains.
3. [On the Importance of Gating: Memorization vs. In-Context Learning in State Space Models](https://arxiv.org/abs/2609.16540) — Investigate how gating can favor memorization and delay an in-context solution even when state capacity is sufficient.

## Gating, linear attention, and test-time learning

How do models control memory writes, forgetting, and adaptation as new tokens arrive?

1. [Gated Linear Attention Transformers with Hardware-Efficient Training](https://arxiv.org/abs/2312.06635) — Connect input-dependent memory gates with algorithms designed for efficient parallel training.
2. [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464) — Compare the roles of memory erasure and targeted updates in a gated delta recurrence.
3. [Titans: Learning to Memorize at Test Time](https://arxiv.org/abs/2501.00663) — Extend the comparison to a neural memory module that learns from the incoming context at test time.

Continue with the repository's [input-dependent gating notes and implementations](../README.md#input-dependent-gating).

## Stability, initialization, and optimization

How do parameterization and initialization affect the ability to learn long-term dependencies?

1. [StableSSM: Alleviating the Curse of Memory in State-space Models through Stable Reparameterization](https://arxiv.org/abs/2311.14495) — Relate memory-learning limitations to stability boundaries and study reparameterizations that improve training.
2. [Autocorrelation Matters: Understanding the Role of Initialization Schemes for State Space Models](https://arxiv.org/abs/2411.19455) — Connect sequence autocorrelation with initialization timescales and the conditioning of optimization.
3. [HOPE for a Robust Parameterization of Long-memory State Space Models](https://arxiv.org/abs/2405.13975) — Compare a Hankel-operator parameterization for improving initialization and training stability.

## Hybrid models and applications

How are SSMs combined with attention or adapted to data beyond one-dimensional language sequences?

1. [H3: Hungry Hungry Hippos: Towards Language Modeling with State Space Models](https://arxiv.org/abs/2212.14052) — Trace an early approach to closing the SSM-attention gap through a new recurrent layer, hybrid models, and efficient convolution.
2. [Hymba: A Hybrid-head Architecture for Small Language Models](https://arxiv.org/abs/2411.13676) — Examine attention and SSM heads operating in parallel to combine recall with context summarization.
3. [VMamba: Visual State Space Model](https://arxiv.org/abs/2401.10166) — See how two-dimensional selective scanning adapts an SSM to visual data.

Continue with the broader [SSM application list](../README.md#on-the-replacement-of-transformerattention-by-ssms) and [vision highlights](../README.md#vision).
