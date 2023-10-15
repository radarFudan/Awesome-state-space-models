# Awesome-state-space-models
Collection of papers on state-space models. 

## ICLR 2024 submissions

I try to use the most important 2-3 sentences in the abstract to summarize the paper. 

1. Variational **quantization** for state space models(https://openreview.net/forum?id=EAkjVCtRO2)

    In this work, we propose a new forecasting model that combines discrete state space hidden Markov models with recent neural network architectures and training procedures inspired by vector quantized variational autoencoders.
    We introduce a variational discrete posterior distribution of the latent states given the observations and a two-stage training procedure to alternatively train the parameters of the latent states and of the emission distributions.

2. Efficient Long Sequence Modeling via State Space Augmented Transformer(https://openreview.net/forum?id=xuxYaBMd9F)

    We propose SPADE, short for State Space Augmented Transformer. 
    Specifically, we augment a SSM into the bottom layer of SPADE, and we employ efficient local attention methods for the other layers.

    **SSM + Transformer**

3. StableSSM: Alleviating the Curse of Memory in State-space Models through Stable Reparameterization(https://openreview.net/forum?id=BwG8hwohU4)

    Our analysis identifies this ``curse of memory'' as a result of the recurrent weights converging to a stability boundary, suggesting that a reparameterization technique can be effective. 
    To this end, we introduce a class of reparameterization techniques for SSMs that effectively lift its memory limitations. 
    Besides improving approximation capabilities, we further illustrate that a principled choice of reparameterization scheme can also enhance **optimization stability**.

    **Stability**

4. Robustifying State-space Models for Long Sequences via Approximate Diagonalization(https://openreview.net/forum?id=DjeQ39QoLQ)

    We introduce a generic, backward-stable ''perturb-then-diagonalize'' (PTD) methodology, which is based on the pseudospectral theory of non-normal operators, and which may be interpreted as the approximate diagonalization of the non-normal matrices defining SSMs. 
    Based on this, we introduce the S4-PTD and S5-PTD models. 
    Through theoretical analysis of the transfer functions of different initialization schemes, we demonstrate that the S4-PTD/S5-PTD initialization strongly converges to the HiPPO framework, while the S4D/S5 initialization only achieves weak convergences. 

    **Robustness (I need to further compare against 3)**

5. From **generalization** analysis to **optimization** designs for state space models(https://openreview.net/forum?id=EGjvMcKrrl)

    In this paper, we theoretically study the generalization of SSMs and propose improvements to training algorithms based on the generalization results. 
    Specifically, we give a data-dependent generalization bound for SSMs, showing an interplay between the SSM parameters and the temporal dependencies of the training sequences. 
    Leveraging the generalization bound, we (1) set up a **scaling rule for model initialization** based on the proposed generalization measure, which significantly improves the robustness of SSMs to different temporal patterns in the sequence data; (2) introduce a new **regularization method for training SSMs to enhance the generalization performance**. Numerical results are conducted to validate our results.

6. A 2-Dimensional State Space Layer for Spatial Inductive Bias(https://openreview.net/forum?id=BGkqypmGvm)

    We leverage an expressive variation of the multidimensional State Space Model (SSM). 
    Our approach introduces efficient parameterization, accelerated computation, and a suitable normalization scheme. 
    Empirically, we observe that incorporating our layer at the beginning of each transformer block of Vision Transformers (ViT) significantly enhances performance for multiple ViT backbones and across datasets. 
    The new layer is effective even with a negligible amount of additional parameters and inference time.

    **Vision task**

7. Hieros: Hierarchical Imagination on Structured State Space Sequence World Models(https://openreview.net/forum?id=5j6wtOO6Fk)

    We propose HIEROS, a hierarchical policy that learns time abstracted world representations and imagines trajectories at multiple time scales in latent space. HIEROS uses an S5 layer-based world model, which predicts next world states in parallel during training and iteratively during environment interaction. Due to the special properties of S5 layers, our method can train in parallel and predict next world states iteratively during imagination. This allows for more efficient training than RNN-based world models and more efficient imagination than Transformer-based world models.

    **Reinforcement Learning** (Use SSM instead of Transformer)

8. S4++: Elevating Long Sequence Modeling with State Memory Reply(https://openreview.net/forum?id=bdnw4qjfH9)

    1. Non-Stable-States (NSS): Significant state variance discrepancies arise among discrete sampling steps, occasionally resulting in divergence.
    2. Dependency Bias: The unidirectional state space dependency in SSM impedes the effective modeling of intricate dependencies. In this paper, we conduct theoretical analysis of SSM from the even-triggered control (ETC) theory perspective and first propose the presence of NSS Phenomenon.

    Our findings indicate that NSS primarily results from the sampling steps, and the integration of multi-state inputs into the current state significantly contributes to the mitigation of NSS. 
    Building upon these theoretical analyses and findings, we propose a simple, yet effective, theoretically grounded State Memory Reply (SMR) mechanism that leverages learnable memories to **incorporate multi-state information into the current state**.

    **Stability**

9. Mamba: Linear-Time Sequence Modeling with Selective State Spaces(https://openreview.net/forum?id=AL1fq05o7H)

    Many subquadratic-time architectures such as linear attention, gated convolution and recurrent models, and structured state space models (SSMs) have been developed to address Transformers' computational inefficiency on long sequences, but they have not performed as well as attention on important modalities such as language. We identify that a key weakness of such models is their inability to perform content-based reasoning, and make several improvements. First, simply letting the **SSM parameters be functions of the input** addresses their weakness with discrete modalities, allowing the model to selectively propagate or forget information along the sequence length dimension depending on the current token. Second, even though this change prevents the use of efficient convolutions, we design a **hardware-aware parallel algorithm in recurrent mode**. We integrate these selective SSMs into a simplified end-to-end neural network architecture without attention or even MLP blocks (Mamba).

    **Time-dependent or input-dependent state-space models** (This is the classical state-space model approach.) + **Hardware acceleration**

10. Gated recurrent neural networks discover attention(https://openreview.net/forum?id=rfSfDSFrRL)

    These modern RNNs feature a prominent design pattern: linear recurrent layers interconnected by feedforward paths with multiplicative gating. 
    Here, we show how RNNs equipped with these two design elements can exactly implement (linear) self-attention, the main building block of Transformers. 

    By reverse-engineering a set of trained RNNs, we find that gradient descent in practice discovers our construction. 
    In particular, we examine RNNs trained to solve simple in-context learning tasks on which Transformers are known to excel and find that gradient descent instills in our RNNs the same attention-based in-context learning algorithm used by Transformers. 

    *Naive question*: What's the difference in contribution sense against [Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention](https://arxiv.org/abs/2006.16236). 

    **Universality of SSM** + **Optimization verification over ICL**

    **TODO**: I am interested in the reverse-engineering part, further check! 

11. GateLoop: Fully Data-Controlled Linear Recurrence for Sequence Modeling(https://openreview.net/forum?id=02Ug9N8DCI)

    We develop GateLoop, a foundational sequence model that generalizes linear recurrent models such as S4, S5, LRU and RetNet, by employing **data-controlled state transitions**.
    Furthermore, we derive an $O(l^2)$ **surrogate-attention mode**, revealing remarkable implications for Transformer and recently proposed architectures.
    While many existing models solely rely on data-controlled cumulative sums for context aggregation, our findings suggest that incorporating data-controlled complex cumulative products may be a crucial step towards more powerful sequence models.

    **Data-controlled state transitions sound similar to 9, TODO comparison**

## Neurips 2023
1. 

## ICML 2023
1. 

## Before 2023
1. See [State-spaces](https://github.com/HazyResearch/state-spaces) for [S4](https://arxiv.org/abs/2111.00396), including [HiPPO](https://arxiv.org/abs/2008.07669), [LSSL](https://arxiv.org/abs/2110.13985), [SaShiMi](https://arxiv.org/abs/2202.09729), [DSS](https://arxiv.org/abs/2203.14343), [HTTYH](https://arxiv.org/abs/2206.12037), [S4D](https://arxiv.org/abs/2206.11893), and [S4ND](https://arxiv.org/abs/2210.06583).
2. 


## TODO
1. Summarize the submission for ICLR 2024 based on abstracts
2. Collect works from Neurips 2023, ICML 2023, Before 2023
3. Summarize the most important unsolved questions in state-space models. 
4. 

