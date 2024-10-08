## ICLR 2025

1. In-context learning and Occam's razor (https://openreview.net/forum?id=2PKLRmU7ne）


## ICLR 2024

For sequence modelling, the phenomenon of in-context learning is interesting in the sense that foundation models can be applied to help the downstream tasks. 

1. The mechanistic basis of data dependence and abrupt learning in an in-context classification task (https://openreview.net/forum?id=aN4Jf6Cx69)

    We first show that these results are recapitulated in a minimal attention-only network trained on a simplified dataset. In-context learning (ICL) is driven by the abrupt emergence of an induction head, which subsequently competes with in-weights learning. 



2. One Step of Gradient Descent is Provably the Optimal In-Context Learner with One Layer of Linear Self-Attention (https://openreview.net/forum?id=8p3fu56lKc&noteId=n3z3Y6l28Q)

    Then, we find that changing the distribution of the covariates and weight vector to a non-isotropic Gaussian distribution has a strong impact on the learned algorithm: the global minimizer of the pre-training loss now implements a single step of pre-conditioned GD. However, if only the distribution of the responses is changed, then this does not have a large effect on the learned algorithm: even when the response comes from a more general family of nonlinear functions, the global minimizer of the pre-training loss still implements a single step of GD on a least-squares linear regression objective.

    Model: Single-layer linear attention

    Task: Linear regression

    Loss function: Mean squared loss

    Question: Optimal in what sense? It is optimal over the sequences drawn from some distribution? 

    Follow-up question: This work does not show the techniques to generalize the proof nonlinear cases. It can be difficult. 



3. In-Context Learning Dynamics with Random Binary Sequences (https://openreview.net/forum?id=62K7mALO2q)

    We propose a Cognitive Interpretability framework that enables us to analyze in-context learning dynamics to understand latent concepts in LLMs underlying behavioral patterns. This provides a more nuanced understanding than success-or-failure evaluation benchmarks, but does not require observing internal activations as a mechanistic interpretation of circuits would require.
    


4. How Many Pretraining Tasks Are Needed for In-Context Learning of Linear Regression? (https://openreview.net/forum?id=vSh5ePa0ph)

    In this paper, we study ICL in one of its simplest setups: pretraining a single-layer linear attention model for linear regression with a Gaussian prior. We establish a statistical task complexity bound for the attention model pretraining, showing that effective pretraining only requires a small number of independent tasks. 

    Question: Does the paper provide a rate? Can the rate be verified in the numerical experiments? 


5. Understanding In-Context Learning in Transformers and LLMs by Learning to Learn Discrete Functions (https://openreview.net/forum?id=ekeyCgeRfC)

    In this work, we take a step towards answering these questions by demonstrating the following: (a) On a test-bed with a variety of Boolean function classes, we find that Transformers can nearly match the optimal learning algorithm for ‘simpler’ tasks, while their performance deteriorates on more ‘complex’ tasks. Additionally, we find that certain attention-free models perform (almost) identically to Transformers on a range of tasks. (b) When provided a teaching sequence, i.e. a set of examples that uniquely identifies a function in a class, we show that Transformers learn more sample-efficiently. Interestingly, our results show that Transformers can learn to implement two distinct algorithms to solve a single task, and can adaptively select the more sample-efficient algorithm depending on the sequence of in-context examples. (c) Lastly, we show that extant LLMs, e.g. LLaMA-2, GPT-4, can compete with nearest-neighbor baselines on prediction tasks that are guaranteed to not be in their training set.

    This paper comes with a detailed code. 


