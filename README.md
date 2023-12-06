# Awesome-state-space-models

Collection of papers on state-space models. 

## ICML 2024 submissions

TODO. 

## On the replacement of transformer by SSMs

1. Pretraining Without Attention (https://arxiv.org/abs/2212.10544) [GitHub](https://github.com/jxiw/BiGS)
   
2. Diffusion Models Without Attention (https://arxiv.org/abs/2311.18257) 

3. Recurrent Distance-Encoding Neural Networks for Graph Representation Learning (https://arxiv.org/abs/2312.01538) [GitHub](https://github.com/skeletondyh/GRED)

## ICLR 2024 submissions

I try to use the most important 2-3 sentences in the abstract to summarize the paper. (https://openreview.net/group?id=ICLR.cc/2024/Conference)

1. FlashFFTConv(https://openreview.net/forum?id=gPKTTAfYBp)

    FlashFFTConv speeds up exact FFT convolutions by up to 8.7 over PyTorch and achieves up to 4.4 speedup end-to-end. [GitHub](https://github.com/HazyResearch/flash-fft-conv). 

2. Variational **quantization** for state space models(https://openreview.net/forum?id=EAkjVCtRO2)

    In this work, we propose a new forecasting model that combines discrete state space hidden Markov models with recent neural network architectures and training procedures inspired by vector quantized variational autoencoders.
    We introduce a variational discrete posterior distribution of the latent states given the observations and a two-stage training procedure to alternatively train the parameters of the latent states and of the emission distributions.

2. Efficient Long Sequence Modeling via State Space Augmented Transformer(https://openreview.net/forum?id=xuxYaBMd9F)

    We propose SPADE, short for State Space Augmented Transformer. 
    Specifically, we augment a SSM into the bottom layer of SPADE, and we employ efficient local attention methods for the other layers.

    **SSM + Transformer**

3. StableSSM: Alleviating the Curse of Memory in State-space Models through Stable **Reparameterization**(https://openreview.net/forum?id=BwG8hwohU4)

    Our analysis identifies this ``curse of memory'' as a result of the recurrent weights converging to a stability boundary, suggesting that a reparameterization technique can be effective. 
    To this end, we introduce a class of reparameterization techniques for SSMs that effectively lift its memory limitations. 
    Besides improving approximation capabilities, we further illustrate that a principled choice of reparameterization scheme can also enhance **optimization stability**.

    **Stability, more on parameterisation**

4. Robustifying State-space Models for Long Sequences via Approximate Diagonalization(https://openreview.net/forum?id=DjeQ39QoLQ)

    We introduce a generic, backward-stable ''perturb-then-diagonalize'' (PTD) methodology, which is based on the pseudospectral theory of non-normal operators, and which may be interpreted as the approximate diagonalization of the non-normal matrices defining SSMs. 
    Based on this, we introduce the S4-PTD and S5-PTD models. 
    Through theoretical analysis of the transfer functions of different initialization schemes, we demonstrate that the S4-PTD/S5-PTD **initialization** strongly converges to the HiPPO framework, while the S4D/S5 initialization only achieves weak convergences. 

    **Robustness, more on initialization**

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

    [GitHub](https://github.com/state-spaces/mamba)

   A very nice analysis *in Chinese*: https://zhuanlan.zhihu.com/p/661237120.

11. Gated recurrent neural networks discover attention(https://openreview.net/forum?id=rfSfDSFrRL)

    These modern RNNs feature a prominent design pattern: linear recurrent layers interconnected by feedforward paths with multiplicative gating. 
    Here, we show how RNNs equipped with these two design elements can exactly implement (linear) self-attention, the main building block of Transformers. 

    By reverse-engineering a set of trained RNNs, we find that gradient descent in practice discovers our construction. 
    In particular, we examine RNNs trained to solve simple in-context learning tasks on which Transformers are known to excel and find that gradient descent instills in our RNNs the same attention-based in-context learning algorithm used by Transformers. 

    *Naive question*: What's the difference in contribution sense against [Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention](https://arxiv.org/abs/2006.16236). 

    **Universality of SSM** + **Optimization verification over ICL**

    **TODO**: I am interested in the reverse-engineering part, further check! 

12. GateLoop: Fully Data-Controlled Linear Recurrence for Sequence Modeling(https://openreview.net/forum?id=02Ug9N8DCI)

    We develop GateLoop, a foundational sequence model that generalizes linear recurrent models such as S4, S5, LRU and RetNet, by employing **data-controlled state transitions**.
    Furthermore, we derive an $O(l^2)$ **surrogate-attention mode**, revealing remarkable implications for Transformer and recently proposed architectures.
    While many existing models solely rely on data-controlled cumulative sums for context aggregation, our findings suggest that incorporating data-controlled complex cumulative products may be a crucial step towards more powerful sequence models.

    **Data-controlled state transitions sound similar to 9, TODO comparison**

13. Never Train from Scratch: Fair Comparison of Long-Sequence Models Requires Data-Driven Priors (https://openreview.net/forum?id=PdaPky8MUn)

    In this work, we show that **random initialization leads to gross overestimation of the differences between architectures** and that pretraining with standard denoising objectives, using only the downstream task data, leads to dramatic gains across multiple architectures and to very small gaps between Transformers and state space models (SSMs).
    In stark contrast to prior works, we find **vanilla Transformers to match the performance of S4** on Long Range Arena when properly pretrained, and we improve the best reported results of SSMs on the PathX-256 task by 20 absolute points.
    Subsequently, we analyze the utility of previously-proposed **structured parameterizations for SSMs** and show they become mostly **redundant** in the presence of data-driven initialization obtained through pretraining.
    Our work shows that, when evaluating different architectures on supervised tasks, incorporation of data-driven priors via pretraining is essential for reliable performance estimation, and can be done efficiently.

    *I don't think **fair** comparison requires data-driven priors but this paper's results are still interesting.*

14. Mastering Memory Tasks with World Models (https://openreview.net/forum?id=1vDArHJ68h)

    To improve temporal coherence, we integrate a new family of state space models (SSMs) in world models of MBRL agents to present a new method, Recall to Imagine (R2I). 
    This integration aims to enhance both long-term memory and long-horizon credit assignment. 
    Through a diverse set of illustrative tasks, we systematically demonstrate that R2I establishes a new state-of-the-art performance in challenging memory and credit assignment RL tasks, such as Memory Maze, BSuite, and POPGym. 
    We also show that R2I is **faster** than the state-of-the-art MBRL method, DreamerV3, resulting in faster wall-time convergence.

    **Reinforcement Learning**


## Arxiv
1. RWKV (https://arxiv.org/abs/2305.13048): https://github.com/BlinkDL/RWKV-LM
2. RetNet (https://arxiv.org/pdf/2307.08621.pdf)

## Neurips 2023
1. State-space Models with Layer-wise Nonlinearity are Universal Approximators with Exponential Decaying Memory (https://arxiv.org/abs/2309.13414)
2. Sparse Modular Activation for Efficient Sequence Modeling (https://arxiv.org/abs/2306.11197)
3. Laughing Hyena Distillery: Extracting Compact Recurrences from Convolutions (https://arxiv.org/abs/2310.18780)
4. Structured State Space Models for In-Context Reinforcement Learning (https://arxiv.org/abs/2303.03982)

     We propose a modification to a variant of S4 that enables us to initialise and reset the hidden state in parallel, allowing us to tackle reinforcement learning tasks.
     We show that our modified architecture runs asymptotically faster than Transformers in sequence length and performs better than RNN's on a simple memory-based task.

5. Convolutional State Space Models for Long-Range Spatiotemporal Modeling (https://arxiv.org/abs/2310.19694)

## ICML 2023
1. Resurrecting Recurrent Neural Networks for Long Sequences (https://icml.cc/virtual/2023/oral/25438)
2. Hyena Hierarchy: Towards Larger Convolutional Language Models (https://arxiv.org/abs/2302.10866)

## Before 2023
1. See [State-spaces](https://github.com/HazyResearch/state-spaces) for [S4](https://arxiv.org/abs/2111.00396), including [HiPPO](https://arxiv.org/abs/2008.07669), [LSSL](https://arxiv.org/abs/2110.13985), [SaShiMi](https://arxiv.org/abs/2202.09729), [DSS](https://arxiv.org/abs/2203.14343), [HTTYH](https://arxiv.org/abs/2206.12037), [S4D](https://arxiv.org/abs/2206.11893), and [S4ND](https://arxiv.org/abs/2210.06583).
2. Simplified State Space Layers for Sequence Modeling (S5) (https://openreview.net/forum?id=Ai8Hw3AXqks) [GitHub](https://github.com/lindermanlab/S5)


## TODO
1. Summarize the submission for ICLR 2024 based on abstracts
2. Collect works from Neurips 2023, ICML 2023, Before 2023
3. Summarize the important unsolved questions in state-space models. (Personal viewpoint)
    1. Scale-up, how to train a larger SSM with better performance such as smaller perplexity in language modelling. Interesting topics include but are not limited to scaling law. Scale-up depth / width or other dimensions. 
    2. Speed-up, how to make the SSM layer faster. (This topic can borrow a lot of idea from [Flash-Attention](https://github.com/Dao-AILab/flash-attention))
    3. Cheaper, given a large model, how to perserve the model performance and run the inference with fewer FLOPs. (Personally I believe the training cost does not matter that much in the cheaper sense.) **Quantization** belongs to this part. 
        1. Maybe we can consider some minimal realization of the state-space models: https://ocw.mit.edu/courses/6-241j-dynamic-systems-and-control-spring-2011/resources/mit6_241js11_lec21/ 
    4. Theoretical guarantees, universality, rates for approximation/generalization/optimization, stability/initialisation in approximation/generalization/optimization...
