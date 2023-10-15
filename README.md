# Awesome-state-space-models
Collection of papers on state-space models

## ICLR 2024 submissions

I try to use the most important 2-3 sentences in the abstract to summarize the paper. 

1. Variational quantization for state space models(https://openreview.net/forum?id=EAkjVCtRO2)

    In this work, we propose a new forecasting model that combines discrete state space hidden Markov models with recent neural network architectures and training procedures inspired by vector quantized variational autoencoders.
    We introduce a variational discrete posterior distribution of the latent states given the observations and a two-stage training procedure to alternatively train the parameters of the latent states and of the emission distributions.
3. Efficient Long Sequence Modeling via State Space Augmented Transformer(https://openreview.net/forum?id=xuxYaBMd9F)

    We propose SPADE, short for State Space Augmented Transformer. 
    Specifically, we augment a SSM into the bottom layer of SPADE, and we employ efficient local attention methods for the other layers.
4. StableSSM: Alleviating the Curse of Memory in State-space Models through Stable Reparameterization(https://openreview.net/forum?id=BwG8hwohU4)

    Our analysis identifies this ``curse of memory'' as a result of the recurrent weights converging to a stability boundary, suggesting that a reparameterization technique can be effective. 
    To this end, we introduce a class of reparameterization techniques for SSMs that effectively lift its memory limitations. 
    Besides improving approximation capabilities, we further illustrate that a principled choice of reparameterization scheme can also enhance optimization stability.
5. Robustifying State-space Models for Long Sequences via Approximate Diagonalization(https://openreview.net/forum?id=DjeQ39QoLQ)

    We introduce a generic, backward-stable ''perturb-then-diagonalize'' (PTD) methodology, which is based on the pseudospectral theory of non-normal operators, and which may be interpreted as the approximate diagonalization of the non-normal matrices defining SSMs. 
    Based on this, we introduce the S4-PTD and S5-PTD models. 
    Through theoretical analysis of the transfer functions of different initialization schemes, we demonstrate that the S4-PTD/S5-PTD initialization strongly converges to the HiPPO framework, while the S4D/S5 initialization only achieves weak convergences. 
6. From generalization analysis to optimization designs for state space models(https://openreview.net/forum?id=EGjvMcKrrl)

    In this paper, we theoretically study the generalization of SSMs and propose improvements to training algorithms based on the generalization results. 
    Specifically, we give a data-dependent generalization bound for SSMs, showing an interplay between the SSM parameters and the temporal dependencies of the training sequences. 
    Leveraging the generalization bound, we (1) set up a scaling rule for model initialization based on the proposed generalization measure, which significantly improves the robustness of SSMs to different temporal patterns in the sequence data; (2) introduce a new regularization method for training SSMs to enhance the generalization performance. Numerical results are conducted to validate our results.
7. A 2-Dimensional State Space Layer for Spatial Inductive Bias(https://openreview.net/forum?id=BGkqypmGvm)

    We leverage an expressive variation of the multidimensional State Space Model (SSM). 
    Our approach introduces efficient parameterization, accelerated computation, and a suitable normalization scheme. 
    Empirically, we observe that incorporating our layer at the beginning of each transformer block of Vision Transformers (ViT) significantly enhances performance for multiple ViT backbones and across datasets. 
    The new layer is effective even with a negligible amount of additional parameters and inference time.
8. Hieros: Hierarchical Imagination on Structured State Space Sequence World Models(https://openreview.net/forum?id=5j6wtOO6Fk)


9. S4++: Elevating Long Sequence Modeling with State Memory Reply(https://openreview.net/forum?id=bdnw4qjfH9)


10. Mamba: Linear-Time Sequence Modeling with Selective State Spaces(https://openreview.net/forum?id=AL1fq05o7H)


11. Gated recurrent neural networks discover attention(https://openreview.net/forum?id=rfSfDSFrRL)

    Here, we show how RNNs equipped with these two design elements can exactly implement (linear) self-attention, the main building block of Transformers.

    Question: What's the difference in contribution sense against [Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention](https://arxiv.org/abs/2006.16236). 

12. GateLoop: Fully Data-Controlled Linear Recurrence for Sequence Modeling(https://openreview.net/forum?id=02Ug9N8DCI)

    We develop GateLoop, a foundational sequence model that generalizes linear recurrent models such as S4, S5, LRU and RetNet, by employing data-controlled state transitions.
    Utilizing this theoretical advance, GateLoop empirically outperforms existing models for auto-regressive language modeling.
13. 

## Neurips 2023
1. 

## ICML 2023
1. 

## Before 2023
1. 

## TODO
1. Summarize the submission for ICLR 2024 based on abstracts
2. Collect works from Neurips 2023, ICML 2023, Before 2023
3. Summarize the most important unsolved questions in state-space models. 
4. 

