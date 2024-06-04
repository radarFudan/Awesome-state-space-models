# Awesome-state-space-models

Collection of papers/repos on state-space models. 


## ICML 2024

1. StableSSM: Alleviating the Curse of Memory in State-space Models through Stable Reparameterization (https://arxiv.org/abs/2311.14495)

2. Gated Linear Attention Transformers with Hardware-Efficient Training (https://arxiv.org/abs/2312.06635)

3. Transformers are SSMs: Generalized Models and Efficient Algorithms with Structured State Space Duality (https://arxiv.org/abs/2405.21060)
    
    By Tri Dao and Albert Gu. 

4. From generalization analysis to optimization designs for state space models (https://arxiv.org/abs/2405.02670)

5. The Illusion of State in State-Space Models (https://arxiv.org/abs/2404.08819)

6. State-Free Inference of State-Space Models: The *Transfer Function* Approach (https://arxiv.org/pdf/2405.06147) [GitHub](https://github.com/ruke1ire/RTF)

7. PAC-Bayesian Error Bound, via Renyi Divergence, for a Class of Linear Time-Invariant State-Space Models

8. Hierarchical State Space Models for Continuous Sequence-to-Sequence Modeling (https://arxiv.org/abs/2402.10211) [GitHub](https://github.com/raunaqbhirangi/hiss/tree/main)

9. Repeat After Me: Transformers are Better than State Space Models at Copying (https://arxiv.org/pdf/2402.01032)

10. SLAB: Efficient Transformers with Simplified Linear Attention and Progressive Re-parameterized Batch Normalization (https://www.arxiv.org/abs/2405.11582)

11. Short-Long Convolutions Help Hardware-Efficient Linear Attention to Focus on Long Sequences

12. When Linear Attention Meets Autoregressive Decoding: Towards More Effective and Efficient Linearized Large Language Models

    Abstract: Autoregressive Large Language Models (LLMs) have achieved impressive performance in language tasks but face significant bottlenecks: (1) quadratic complexity bottleneck in the attention module with increasing token numbers, and (2) efficiency bottleneck due to the sequential processing nature of autoregressive LLMs during generation. Linear attention and speculative decoding emerge as solutions for these challenges, yet their applicability and combinatory potential for autoregressive LLMs remain uncertain. To this end, we embark on the first comprehensive empirical investigation into the efficacy of existing linear attention methods for autoregressive LLMs and their integration with speculative decoding. We introduce an augmentation technique for linear attention and ensure the compatibility between linear attention and speculative decoding for efficient LLM training and serving. Extensive experiments and ablation studies on seven existing linear attention works and five encoder/decoder-based LLMs consistently validate the effectiveness of our augmented linearized LLMs, e.g., achieving up to a 6.67 perplexity reduction on LLaMA and 2x speedups during generation as compared to prior linear attention methods.

13. Simple linear attention language models balance the recall-throughput tradeoff (https://arxiv.org/abs/2402.18668) [GitHub](https://github.com/HazyResearch/based)

## Input-dependent gating. 

1. Mamba (https://arxiv.org/abs/2312.00752) [Official GitHub](https://github.com/state-spaces/mamba)

    $$g_k = \sigma(Linear(x_k)),$$
    $$h_{k+1} = (1-g_k) h_{k} + g_k x_k.$$

    The activation is SiLU / Swish. The continuous form is 
    $$\frac{dh_t}{dt} = g_t (x_t - h_t).$$

    Various (unofficial) implementations: 
    1. [Mamba-minimal-in-JAX](https://github.com/radarFudan/mamba-minimal-jax)
    2. [Mamba-minimal-in-PyTorch](https://github.com/johnma2006/mamba-minimal)
    3. [Mamba.py](https://github.com/alxndrTL/mamba.py)
    4. [Mamba-jax](https://github.com/vvvm23/mamba-jax)
    5. [Mamba-mini](https://github.com/MzeroMiko/mamba-mini)
    6. [LongMamba](https://github.com/jzhang38/LongMamba)
    7. [Mamba the hard way](https://srush.github.io/annotated-mamba/hard.html) [Annotated Mamba](https://github.com/srush/annotated-mamba) 
    8. [Mamba the easy way](https://jackcook.com/2024/02/23/mamba.html)

2. [ICML2024] Gated Linear Attention (GLA) (https://arxiv.org/abs/2312.06635) [Official GitHub](https://github.com/berlino/gated_linear_attention)

    The following repo aims at providing a collection of efficient Triton-based implementations for state-of-the-art linear attention models.
    [Flash-linear-attention](https://github.com/sustcsonglin/flash-linear-attention)


## On the replacement of transformer/attention by SSMs

1. [Video] Long Movie Clip Classification with State-Space Video Models (https://arxiv.org/abs/2204.01692) [GitHub](https://github.com/md-mohaiminul/ViS4mer)

2. [Language model] Pretraining Without Attention (https://arxiv.org/abs/2212.10544) [GitHub](https://github.com/jxiw/BiGS)

    Feature: Bidirectional Language Modeling with State-space Model

3. [Reinforcement Learning] Structured State Space Models for In-Context Reinforcement Learning (https://arxiv.org/abs/2303.03982) [GitHub](https://github.com/luchris429/popjaxrl)
   
4. [Diffusion Model] Diffusion Models Without Attention (https://arxiv.org/abs/2311.18257) (NeurIPS 2023 Workshop on Diffusion Models)

5. [Graph] Recurrent Distance-Encoding Neural Networks for Graph Representation Learning (https://arxiv.org/abs/2312.01538) [GitHub](https://github.com/skeletondyh/GRED)

6. [Mixture of Experts] MoE-Mamba: Efficient Selective State Space Models with Mixture of Experts (https://arxiv.org/abs/2401.04081) [GitHub](https://github.com/llm-random/llm-random)

7. [Bio] U-Mamba, a versatile network designed specifically for biomedical image segmentation. (https://arxiv.org/abs/2401.04722) [GitHub](https://github.com/bowang-lab/U-Mamba)
    
8. [Vision] VMamba: Visual State Space Model. (https://arxiv.org/abs/2401.10166) [GitHub](https://github.com/MzeroMiko/VMamba)

9. [Tabular data] MambaTab: A Simple Yet Effective Approach for Handling Tabular Data (https://arxiv.org/abs/2401.08867)

10. [RWKV-TS] RWKV-TS: Beyond Traditional Recurrent Neural Network for Time Series Tasks (https://arxiv.org/abs/2401.09093) [GitHub](https://github.com/howard-hou/rwkv-ts)

11. [Vision] Vision Mamba (Vim) is 2.8× faster than DeiT and saves 86.8% GPU memory when performing batch inference to extract features on images with a resolution of 1248×1248. (https://arxiv.org/abs/2401.09417) [GitHub](https://github.com/hustvl/Vim)

12. [Vision] SegMamba: Long-range Sequential Modeling Mamba For 3D Medical Image Segmentation. (https://arxiv.org/abs/2401.13560) [GitHub](https://github.com/ge-xing/SegMamba)

13. [Token-free language models] MambaByte: Token-free Selective State Space Model.（https://arxiv.org/abs/2401.13660）[GitHub](https://github.com/kyegomez/MambaByte)

    **Token-free** language models learn directly from raw bytes and remove the bias of subword tokenization.    

14. [Vision] MambaMorph: a Mamba-based Backbone with Contrastive Feature Learning for Deformable MR-CT Registration. (https://arxiv.org/abs/2401.13934) [GitHub](https://github.com/Guo-Stone/MambaMorph)

15. [Video] Vivim: a Video Vision Mamba for Medical Video Object Segmentation (https://arxiv.org/pdf/2401.14168.pdf) [GitHub](https://github.com/scott-yjyang/Vivim)

16. [Document Summarization] LOCOST: State-Space Models for Long Document Abstractive Summarization (https://arxiv.org/abs/2401.17919) [GitHub](https://github.com/flbbb/locost-summarization)

17. [Graph] Graph-Mamba: Towards Long-Range Graph Sequence Modeling with Selective State Spaces (https://arxiv.org/abs/2402.00789) [GitHub](https://github.com/bowang-lab/Graph-Mamba)

18. [Mixture of Experts] BlackMamba: Mixture of Experts for State-Space Models (https://arxiv.org/abs/2402.01771) [GitHub](https://github.com/Zyphra/BlackMamba)

19. [Vision] Swin-UMamba: Mamba-based UNet with ImageNet-based pretraining (https://arxiv.org/abs/2402.03302) [GitHub](https://github.com/JiarunLiu/Swin-UMamba)

20. [Bio] VM-UNet: Vision Mamba UNet for Medical Image Segmentation (https://arxiv.org/abs/2402.02491) [GitHub](https://github.com/JCruan519/VM-UNet)

21. [IN-CONTEXT LEARNING] IS MAMBA CAPABLE OF IN-CONTEXT LEARNING? (https://arxiv.org/abs/2402.03170)

22. [Bio] nnMamba: 3D Biomedical Image Segmentation, Classification and Landmark Detection with State Space Model (https://arxiv.org/abs/2402.03526) [GitHub](https://github.com/lhaof/nnMamba)

23. [IN-CONTEXT LEARNING] Can Mamba Learn How to Learn? A Comparative Study on In-Context Learning Tasks (https://arxiv.org/abs/2402.04248)

24. [Diffusion Model] Scalable Diffusion Models with State Space Backbone (https://arxiv.org/abs/2402.05608) [GitHub](https://github.com/feizc/DiS)

25. [Vision] Mamba-ND: Selective State Space Modeling for Multi-Dimensional Data (https://arxiv.org/abs/2402.05892) [GitHub](https://github.com/jacklishufan/Mamba-ND)

26. [Vision] FD-Vision Mamba for Endoscopic Exposure Correction (https://arxiv.org/pdf/2402.06378.pdf)

27. [Vision] Semi-Mamba-UNet: Pixel-Level Contrastive Cross-Supervised Visual Mamba-based UNet for Semi-Supervised Medical Image Segmentation (https://arxiv.org/abs/2402.07245) [GitHub](https://github.com/ziyangwang007/Mamba-UNet)

28. [Segmentation] P-Mamba: Marrying Perona Malik Diffusion with Mamba for Efficient Pediatric Echocardiographic Left Ventricular Segmentation: (https://arxiv.org/abs/2402.08506)

29. [Graph] Graph Mamba: Towards Learning on Graphs with State Space Models (https://arxiv.org/abs/2402.08678) [GitHub](https://github.com/GraphMamba/GMN)

30. [Theory] Spectral State Space Models (https://arxiv.org/abs/2312.06837v3) [GitHub](https://github.com/google-deepmind/spectral_ssm)

31. [Point Cloud Analysis] PointMamba: A Simple State Space Model for Point Cloud Analysis (https://arxiv.org/abs/2402.10739) [GitHub](https://github.com/LMD0311/PointMamba)

32. [Vision] RES-VMAMBA: FINE-GRAINED FOOD CATEGORY VISUAL CLASSIFICATION USING SELECTIVE STATE SPACE MODELS WITH DEEP RESIDUAL LEARNING (https://arxiv.org/abs/2402.15761) [GitHub](https://github.com/ChiShengChen/ResVMamba)

33. [Theory] Learning method for S4 with Diagonal State Space Layers using Balanced Truncation (https://arxiv.org/abs/2402.15993) 

34. [Financial data] MambaStock: Selective state space model for stock prediction (https://arxiv.org/abs/2402.18959) [GitHub](https://github.com/zshicode/MambaStock)

35. [Theory] Theoretical Foundations of Deep Selective State-Space Models (https://arxiv.org/abs/2402.19047)

    Theoretical analysis from the perspective of rough path theory (sig- nature transform). 

36. [Scale-up] Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models (https://arxiv.org/abs/2402.19427)

37. [Point Cloud Analysis] Point Could Mamba: Point Cloud Learning via State Space Model (https://arxiv.org/abs/2403.00762) [GitHub](https://github.com/SkyworkAI/PointCloudMamba?tab=readme-ov-file)

38. [Language Model] DenseMamba: State Space Models with Dense Hidden Connection for Efficient Large Language Models (https://arxiv.org/abs/2403.00818) [GitHub](https://github.com/WailordHe/DenseSSM)

39. [Vision] The Hidden Attention of Mamba Models (https://arxiv.org/abs/2403.01590) [GitHub](https://github.com/AmeenAli/HiddenMambaAttn)

40. [Target Detection] MiM-ISTD: Mamba-in-Mamba for Efficient Infrared Small Target Detection (https://arxiv.org/abs/2403.02148) [GitHub](https://github.com/txchen-USTC/MiM-ISTD)

41. [Time Series] TimeMachine: A Time Series is Worth 4 Mambas for Long-term Forecasting (https://arxiv.org/abs/2403.09898) [GitHub](https://github.com/Atik-Ahamed/TimeMachine?tab=readme-ov-file)

42. [Time Series] Is Mamba Effective for Time Series Forecasting? (https://arxiv.org/abs/2403.11144) [To-be-updated-GitHub](https://github.com/wzhwzhwzh0921/S-D-Mamba)

43. [Recommendation] Mamba4Rec: Towards Efficient Sequential Recommendation with Selective State Space Models (https://arxiv.org/abs/2403.03900) [GitHub](https://github.com/chengkai-liu/Mamba4Rec)

44. [Speech] Multichannel Long-Term Streaming Neural Speech Enhancement for Static and Moving Speakers (https://arxiv.org/abs/2403.07675) [GitHub](https://github.com/Audio-WestlakeU/NBSS)

45. [Vision] On the low-shot transferability of [V]-Mamba (https://arxiv.org/abs/2403.10696)

46. [Diffusion Model] ZigMa: Zigzag Mamba Diffusion Model (https://arxiv.org/abs/2403.13802) [To-be-updated-GitHub](https://github.com/CompVis/zigma)

47. [Scale-up] Jamba: SSM-Transformer Model (https://www.ai21.com/blog/announcing-jamba)

    Total 52B parameters. SSM-Transformer hybrid architecture, 256K context window

48. [Control] State Space Models as Foundation Models: A Control Theoretic Overview (https://arxiv.org/abs/2403.16899) [GitHub](https://github.com/jsie7/ssm-benchmark)

49. [3D reconstruction] Gamba: Marry Gaussian Splatting with Mamba for single view 3D reconstruction (https://arxiv.org/abs/2403.18795)

50. MambaMixer: Efficient Selective State Space Models with Dual Token and Channel Selection (https://arxiv.org/abs/2403.19888)

51. [Semantic Segmentation] Sigma: Siamese Mamba Network for Multi-Modal Semantic Segmentation (https://arxiv.org/abs/2404.04256) [GitHub](https://github.com/zifuwan/Sigma)

52. [Scale-up] RecurrentGemma: Moving Past Transformers for Efficient Open Language Models (https://storage.googleapis.com/deepmind-media/gemma/recurrentgemma-report.pdf) [GitHub](https://github.com/google-deepmind/recurrentgemma?tab=readme-ov-file)

53. HGRN2: Gated Linear RNNs with State Expansion (https://arxiv.org/abs/2404.07904) [GitHub](https://github.com/OpenNLPLab/HGRN2)

54. [Theory] State-Space Systems as Dynamic Generative Models (https://arxiv.org/abs/2404.08717)

    This paper studies the conditions for stochastic echo state property, which is a generalisation of deterministic case. 

55. [Survey] State Space Model for New-Generation Network Alternative to Transformers: A Survey (https://arxiv.org/abs/2404.09516) [GitHub](https://github.com/Event-AHU/Mamba_State_Space_Model_Paper_List)

56. [DNA] Caduceus: Bi-Directional Equivariant Long-Range DNA Sequence Modeling (https://arxiv.org/pdf/2403.03234) [GitHub](https://caduceus-dna.github.io)

57. [Vision] ViM-UNet: Vision Mamba for Biomedical Segmentation (https://arxiv.org/abs/2404.07705) [GitHub](https://github.com/constantinpape/torch-em/blob/main/vimunet.md)

58. Integrating Mamba and Transformer for Long-Short Range Time Series Forecasting (https://arxiv.org/abs/2404.14757) [GitHub](https://github.com/XiongxiaoXu/Mambaformer-in-Time-Series)

59. xLSTM: Extended Long Short-Term Memory (https://arxiv.org/abs/2405.04517)

60. MambaOut: Do We Really Need Mamba for Vision? (https://arxiv.org/abs/2405.07992) [GitHub](https://github.com/yuweihao/MambaOut)

61. [Transformer_to_Recurrent] Linearizing Large Language Models (https://arxiv.org/abs/2405.06640) [GitHub](https://github.com/TRI-ML/linear_open_lm)

62. Not All Language Model Features Are Linear (https://arxiv.org/abs/2405.14860)

63. Attention as an RNN (https://arxiv.org/abs/2405.13956)

64. I2I-Mamba: Multi-modal medical image synthesis via selective state space modeling (https://arxiv.org/abs/2405.14022) [GitHub](https://github.com/icon-lab/I2I-Mamba)

65. There is HOPE to Avoid HiPPOs for Long-memory State Space Models (https://arxiv.org/abs/2405.13975)

66. Understanding the differences in Foundation Models: Attention, State Space Models, and Recurrent Neural Networks (https://arxiv.org/abs/2405.15731) [GitHub](https://github.com/IntelligentControlSystems/dsf-mqar)

67. The Expressive Capacity of State Space Models: A Formal Language Perspective (https://arxiv.org/abs/2405.17394)

68. Efficient Time Series Processing for Transformers and State-Space Models through Token Merging (https://arxiv.org/abs/2405.17951)

69. DiG: Scalable and Efficient Diffusion Models with Gated Linear Attention (https://arxiv.org/abs/2405.18428)

70. ViG: Linear-complexity Visual Sequence Learning with Gated Linear Attention (https://arxiv.org/abs/2405.18425)

71. State Space Models are Comparable to Transformers in Estimating Functions with Dynamic Smoothness (https://arxiv.org/abs/2405.19036)

72. Recurrent neural networks: vanishing and exploding gradients are not the end of the story (https://arxiv.org/abs/2405.21064)


## ICLR 2024 submissions

I try to use the most important 2-3 sentences in the abstract to summarize the paper. (https://openreview.net/group?id=ICLR.cc/2024/Conference)

1. FlashFFTConv(https://openreview.net/forum?id=gPKTTAfYBp)

    FlashFFTConv speeds up exact FFT convolutions by up to 8.7 over PyTorch and achieves up to 4.4 speedup end-to-end. [GitHub](https://github.com/HazyResearch/flash-fft-conv). 

2. Variational **quantization** for state space models(https://openreview.net/forum?id=EAkjVCtRO2)

    In this work, we propose a new forecasting model that combines discrete state space hidden Markov models with recent neural network architectures and training procedures inspired by vector quantized variational autoencoders.
    We introduce a variational discrete posterior distribution of the latent states given the observations and a two-stage training procedure to alternatively train the parameters of the latent states and of the emission distributions.

3. Efficient Long Sequence Modeling via State Space Augmented Transformer(https://openreview.net/forum?id=xuxYaBMd9F)

    We propose SPADE, short for State Space Augmented Transformer. 
    Specifically, we augment a SSM into the bottom layer of SPADE, and we employ efficient local attention methods for the other layers.

    **SSM + Transformer** [GitHub](https://github.com/microsoft/EfficientLongSequenceModeling)

4. StableSSM: Alleviating the Curse of Memory in State-space Models through Stable **Reparameterization**(https://openreview.net/forum?id=BwG8hwohU4)

    Our analysis identifies this ``curse of memory'' as a result of the recurrent weights converging to a stability boundary, suggesting that a reparameterization technique can be effective. 
    To this end, we introduce a class of reparameterization techniques for SSMs that effectively lift its memory limitations. 
    Besides improving approximation capabilities, we further illustrate that a principled choice of reparameterization scheme can also enhance **optimization stability**.

    **Stability, more on parameterisation** 

5. Robustifying State-space Models for Long Sequences via Approximate Diagonalization(https://openreview.net/forum?id=DjeQ39QoLQ)

    We introduce a generic, backward-stable ''perturb-then-diagonalize'' (PTD) methodology, which is based on the pseudospectral theory of non-normal operators, and which may be interpreted as the approximate diagonalization of the non-normal matrices defining SSMs. 
    Based on this, we introduce the S4-PTD and S5-PTD models. 
    Through theoretical analysis of the transfer functions of different initialization schemes, we demonstrate that the S4-PTD/S5-PTD **initialization** strongly converges to the HiPPO framework, while the S4D/S5 initialization only achieves weak convergences. 

    **Robustness, more on initialization**

6. From **generalization** analysis to **optimization** designs for state space models(https://openreview.net/forum?id=EGjvMcKrrl)

    In this paper, we theoretically study the generalization of SSMs and propose improvements to training algorithms based on the generalization results. 
    Specifically, we give a data-dependent generalization bound for SSMs, showing an interplay between the SSM parameters and the temporal dependencies of the training sequences. 
    Leveraging the generalization bound, we (1) set up a **scaling rule for model initialization** based on the proposed generalization measure, which significantly improves the robustness of SSMs to different temporal patterns in the sequence data; (2) introduce a new **regularization method for training SSMs to enhance the generalization performance**. Numerical results are conducted to validate our results.

7. A 2-Dimensional State Space Layer for Spatial Inductive Bias(https://openreview.net/forum?id=BGkqypmGvm)

    We leverage an expressive variation of the multidimensional State Space Model (SSM). 
    Our approach introduces efficient parameterization, accelerated computation, and a suitable normalization scheme. 
    Empirically, we observe that incorporating our layer at the beginning of each transformer block of Vision Transformers (ViT) significantly enhances performance for multiple ViT backbones and across datasets. 
    The new layer is effective even with a negligible amount of additional parameters and inference time.

    **Vision task**

8. Hieros: Hierarchical Imagination on Structured State Space Sequence World Models(https://openreview.net/forum?id=5j6wtOO6Fk)

    We propose HIEROS, a hierarchical policy that learns time abstracted world representations and imagines trajectories at multiple time scales in latent space. HIEROS uses an S5 layer-based world model, which predicts next world states in parallel during training and iteratively during environment interaction. Due to the special properties of S5 layers, our method can train in parallel and predict next world states iteratively during imagination. This allows for more efficient training than RNN-based world models and more efficient imagination than Transformer-based world models.

    **Reinforcement Learning** (Use SSM instead of Transformer)

9. S4++: Elevating Long Sequence Modeling with State Memory Reply(https://openreview.net/forum?id=bdnw4qjfH9)

    1. Non-Stable-States (NSS): Significant state variance discrepancies arise among discrete sampling steps, occasionally resulting in divergence.
    2. Dependency Bias: The unidirectional state space dependency in SSM impedes the effective modeling of intricate dependencies. In this paper, we conduct theoretical analysis of SSM from the even-triggered control (ETC) theory perspective and first propose the presence of NSS Phenomenon.

    Our findings indicate that NSS primarily results from the sampling steps, and the integration of multi-state inputs into the current state significantly contributes to the mitigation of NSS. 
    Building upon these theoretical analyses and findings, we propose a simple, yet effective, theoretically grounded State Memory Reply (SMR) mechanism that leverages learnable memories to **incorporate multi-state information into the current state**.

    **Stability**

10. Mamba: Linear-Time Sequence Modeling with Selective State Spaces(https://openreview.net/forum?id=AL1fq05o7H)

    Many subquadratic-time architectures such as linear attention, gated convolution and recurrent models, and structured state space models (SSMs) have been developed to address Transformers' computational inefficiency on long sequences, but they have not performed as well as attention on important modalities such as language. We identify that a key weakness of such models is their inability to perform content-based reasoning, and make several improvements. First, simply letting the **SSM parameters be functions of the input** addresses their weakness with discrete modalities, allowing the model to selectively propagate or forget information along the sequence length dimension depending on the current token. Second, even though this change prevents the use of efficient convolutions, we design a **hardware-aware parallel algorithm in recurrent mode**. We integrate these selective SSMs into a simplified end-to-end neural network architecture without attention or even MLP blocks (Mamba).

    **Time-dependent or input-dependent state-space models** + **Hardware acceleration**

    A very nice analysis *in Chinese*: https://zhuanlan.zhihu.com/p/661237120.

11. Gated recurrent neural networks discover attention(https://openreview.net/forum?id=rfSfDSFrRL)

    These modern RNNs feature a prominent design pattern: linear recurrent layers interconnected by feedforward paths with multiplicative gating. 
    Here, we show how RNNs equipped with these two design elements can exactly implement (linear) self-attention, the main building block of Transformers. 

    By reverse-engineering a set of trained RNNs, we find that gradient descent in practice discovers our construction. 
    In particular, we examine RNNs trained to solve simple in-context learning tasks on which Transformers are known to excel and find that gradient descent instills in our RNNs the same attention-based in-context learning algorithm used by Transformers. 

    *Naive question*: What's the difference in contribution sense against [Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention](https://arxiv.org/abs/2006.16236). 

    **Universality of SSM** + **Optimization verification over ICL**

12. GateLoop: Fully Data-Controlled Linear Recurrence for Sequence Modeling(https://openreview.net/forum?id=02Ug9N8DCI)

    We develop GateLoop, a foundational sequence model that generalizes linear recurrent models such as S4, S5, LRU and RetNet, by employing **data-controlled state transitions**.
    Furthermore, we derive an $O(l^2)$ **surrogate-attention mode**, revealing remarkable implications for Transformer and recently proposed architectures.
    While many existing models solely rely on data-controlled cumulative sums for context aggregation, our findings suggest that incorporating data-controlled complex cumulative products may be a crucial step towards more powerful sequence models.

    **Data-controlled state transitions sound similar to 9, TODO comparison** [Official GitHub](https://github.com/tobiaskatsch/GatedLinearRNN) [Unofficial GitHub](https://github.com/lucidrains/gateloop-transformer)

13. Never Train from Scratch: Fair Comparison of Long-Sequence Models Requires Data-Driven Priors (https://openreview.net/forum?id=PdaPky8MUn)

    In this work, we show that **random initialization leads to gross overestimation of the differences between architectures** and that pretraining with standard denoising objectives, using only the downstream task data, leads to dramatic gains across multiple architectures and to very small gaps between Transformers and state space models (SSMs).
    In stark contrast to prior works, we find **vanilla Transformers to match the performance of S4** on Long Range Arena when properly pretrained, and we improve the best reported results of SSMs on the PathX-256 task by 20 absolute points.
    Subsequently, we analyze the utility of previously-proposed **structured parameterizations for SSMs** and show they become mostly **redundant** in the presence of data-driven initialization obtained through pretraining.
    Our work shows that, when evaluating different architectures on supervised tasks, incorporation of data-driven priors via pretraining is essential for reliable performance estimation, and can be done efficiently.

    [GitHub](https://github.com/IdoAmos/not-from-scratch)

14. Mastering Memory Tasks with World Models (https://openreview.net/forum?id=1vDArHJ68h)

    To improve temporal coherence, we integrate a new family of state space models (SSMs) in world models of MBRL agents to present a new method, Recall to Imagine (R2I). 
    This integration aims to enhance both long-term memory and long-horizon credit assignment. 
    Through a diverse set of illustrative tasks, we systematically demonstrate that R2I establishes a new state-of-the-art performance in challenging memory and credit assignment RL tasks, such as Memory Maze, BSuite, and POPGym. 
    We also show that R2I is **faster** than the state-of-the-art MBRL method, DreamerV3, resulting in faster wall-time convergence.

    **Reinforcement Learning** [GitHub](https://github.com/danijar/dreamerv3)


## Arxiv

1. RWKV (https://arxiv.org/abs/2305.13048): [GitHub](https://github.com/BlinkDL/RWKV-LM)

2. RetNet (https://arxiv.org/abs/2307.08621) [GitHub](https://github.com/microsoft/torchscale/blob/main/README.md) 

3. Zoology (https://arxiv.org/abs/2312.04927) [GitHub](https://github.com/HazyResearch/zoology)

4. Structured state-space models are deep Wiener models (https://arxiv.org/abs/2312.06211)


## NeurIPS 2023

1. State-space Models with Layer-wise Nonlinearity are Universal Approximators with Exponential Decaying Memory (https://arxiv.org/abs/2309.13414)

   The authors show that the layer-wise nonlinearity is enough to achieve the universality when the state-space models are multi-layer. 

   It is also shown that similar to traditional nonlinear recurrent neural networks, SSMs also suffer from the aymptotically exponential memory decay. 

2. Sparse Modular Activation for Efficient Sequence Modeling (SMA) (https://arxiv.org/abs/2306.11197) [GitHub](https://github.com/renll/SeqBoat)

    SSM + Attention, SOTA at LRA. 

    We design a novel neural architecture, SeqBoat, which employs SMA to sparsely activate a Gated Attention Unit (GAU) based on the state representations learned from an SSM.

3. Laughing Hyena Distillery: Extracting Compact Recurrences from Convolutions (https://arxiv.org/abs/2310.18780)

   Given a convolution-based Hyena model, the authors want to extract the recurrent weights for the convolution kernel so that the convolution model can be converted into a recurrent models.
   Method used are based on Hankel matrix SVD. 

5. Structured State Space Models for In-Context Reinforcement Learning (https://arxiv.org/abs/2303.03982) [GitHub](https://github.com/luchris429/popjaxrl)

    We propose a modification to a variant of S4 that enables us to initialise and reset the hidden state in parallel, allowing us to tackle reinforcement learning tasks.
    We show that our modified architecture runs asymptotically faster than Transformers in sequence length and performs better than RNN's on a simple memory-based task.

6. Convolutional State Space Models for Long-Range Spatiotemporal Modeling (https://arxiv.org/abs/2310.19694) [GitHub](https://github.com/NVlabs/ConvSSM)

7. Hierarchically Gated Recurrent Neural Network for Sequence Modeling (https://paperswithcode.com/paper/hierarchically-gated-recurrent-neural-network) [GitHub](https://github.com/OpenNLPLab/HGRN)


## ICML 2023

1. Resurrecting Recurrent Neural Networks for Long Sequences (https://icml.cc/virtual/2023/oral/25438)

2. Hyena Hierarchy: Towards Larger Convolutional Language Models (https://arxiv.org/abs/2302.10866) [GitHub](https://github.com/HazyResearch/safari)

3. Neural Continuous-Discrete State Space Models for Irregularly-Sampled Time Series (https://icml.cc/virtual/2023/oral/25554) [GitHub](https://github.com/clear-nus/NCDSSM)


## Before 2023

1. See github repo [State-spaces](https://github.com/state-spaces/s4) for [S4](https://arxiv.org/abs/2111.00396), including [HiPPO](https://arxiv.org/abs/2008.07669), [LSSL](https://arxiv.org/abs/2110.13985), [SaShiMi](https://arxiv.org/abs/2202.09729), [DSS](https://arxiv.org/abs/2203.14343), [HTTYH](https://arxiv.org/abs/2206.12037), [S4D](https://arxiv.org/abs/2206.11893), and [S4ND](https://arxiv.org/abs/2210.06583), [GSS](https://arxiv.org/abs/2206.13947)

2. [S5] Simplified State Space Layers for Sequence Modeling (ICLR 2023) (https://openreview.net/forum?id=Ai8Hw3AXqks) [GitHub](https://github.com/lindermanlab/S5)

3. [Liquid SSM] Liquid Structural State-Space Models (ICLR 2023) (https://openreview.net/forum?id=g4OTKRKfS7R) [GitHub](https://github.com/raminmh/liquid-s4)

4. [Parallel scan] Parallelizing Linear Recurrent Neural Nets Over Sequence Length (ICLR 2018) (https://openreview.net/forum?id=HyUNwulC-) [GitHub](https://github.com/eamartin/parallelizing_linear_rnns)

5. Bayesian state-space models [GitHub](https://github.com/lindermanlab/ssm). 
    
    Another very good note is: http://personal.strath.ac.uk/gary.koop/GSE_Bayesian/Bayesian_State_Space_Methods.pdf

6. Mega: Moving Average Equipped Gated Attention (Mega) [GitHub](https://github.com/facebookresearch/mega)

7. [Annotated S4](https://srush.github.io/annotated-s4/) By [Sasha Rush](http://rush-nlp.com) and [Sidd Karamcheti](https://www.siddkaramcheti.com) [GitHub](https://github.com/srush/annotated-s4)

8. The State Space of Complex Systems (thesis by Frank Heilmann) https://d-nb.info/1212365704/34

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=radarFudan/Awesome-state-space-models&type=Date)](https://star-history.com/#radarFudan/Awesome-state-space-models)

