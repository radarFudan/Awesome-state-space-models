1. Biased Temporal Convolution Graph Network for Time Series Forecasting with Missing Values (https://openreview.net/forum?id=O9nZCwdGcG)

    A naive employment of imputation methods unavoidably involves error accumulation and leads to suboptimal solutions. Motivated by this, we propose a Biased Temporal Convolution Graph Network that jointly captures the temporal dependencies and spatial structure. 

    This paper provides code at: https://anonymous.4open.science/r/BiaTCGNet-1F80/README.md

    This paper considers the time series with missing values. 
    I am not an expert in this topic but this paper seems to have abundant numerical experiments. 
    However, there is not much theoretical explanation about the guarantee of their methods. 



2. ModernTCN: A Modern Pure Convolution Structure for General Time Series Analysis (https://openreview.net/forum?id=vpJMJerXHU)

    As a pure convolution structure, ModernTCN still achieves the consistent state-of-the-art performance on five mainstream time series analysis tasks (long-term and short-term forecasting, imputation, classification and anomaly detection) while maintaining the efficiency advantage of convolution-based models, therefore providing a better balance of efficiency and performance than state-of-the-art Transformer-based and MLP-based models.

    Our study further reveals that, compared with previous convolution-based models, our ModernTCN has much larger effective receptive fields (ERFs)



3. Efficient and Quantization-Friendly Ternary Fourier Convolution Algorithms (https://openreview.net/forum?id=XXrUarMM20)

    To address this challenge, we present a novel fast convolution algorithm that utilizes ternary matrices (coefficients containing only ±1 and 0) for input and weight transformations before multiplication, thus minimizing quantization errors. This approach is derived from the implementation of symbolic arithmetic on the Fourier transform to eliminate the involvement of irrational numbers. Then, we incorporate correction terms to convert ineffective circular convolution results into efficient ones, thereby enhancing algorithm efficiency. Additionally, we propose a corresponding post-training quantization method that requires only a few samples for calibrating network parameters and restoring accuracy without the heavy cost of retraining. Our algorithms achieve 3.68x, 4.89x, and 4.54x theoretical multiplication complexity reduction for 3x3, 5x5, and 7x7 convolutions, respectively. For models trained on the ImageNet dataset, our algorithms with the post-training method, demonstrate an accuracy drop of less than 0.2% under Int8 quantization, surpassing other approaches with similar multiplication reduction ratios.



4. Dilated convolution neural operator for multiscale partial differential equations (https://openreview.net/forum?id=TBLe2BHBsr)

    This paper presents a data-driven operator learning method for multiscale partial differential equations, where preserving high-frequency information is critical. We propose the Dilated Convolution Neural Operator (DCNO), which combines dilated convolution layers to effectively capture high-frequency features at a low computational cost, along with Fourier layers to handle smooth features. We conduct experiments to evaluate the performance of DCNO on various datasets, including the multiscale elliptic equation, its inverse problem, Navier-Stokes equation, and Helmholtz equation. DCNO stands out with significantly higher accuracy compared to existing neural operator techniques, and strikes an optimal balance between accuracy and computational cost.



5. Spatio-Temporal Approximation: A Training-Free SNN Conversion for Transformers (https://openreview.net/forum?id=XrunSYwoLr)

    However, while existing conversion methods work well on convolution networks, emerging Transformer models introduce unique mechanisms like self-attention and test-time normalization, leading to non-causal non-linear interactions unachievable by current SNNs. To address this, we approximate these operations in both temporal and spatial dimensions, thereby providing the first SNN conversion pipeline for Transformers. We propose \textit{Universal Group Operators} to approximate non-linear operations spatially and a \textit{Temporal-Corrective Self-Attention Layer} that approximates spike multiplications at inference through an estimation-correction approach. 
    
    Our algorithm is implemented on a pretrained ViT-B/32 from CLIP, inheriting its zero-shot classification capabilities, while improving control over conversion losses. To our knowledge, this is the first direct training-free conversion of a pretrained Transformer to a purely event-driven SNN, promising for neuromorphic hardware deployment.



6. RefConv: Re-parameterized Refocusing Convolution for Powerful ConvNets (https://openreview.net/forum?id=You77eOFDv)

    This one works on re-parameterization. What's the relationship with the StableSSM reparameterization? This reparameterization aims to augments **the priors to existing structures by establishing connections to the learned kernels**. 

    For example, a depth-wise RefConv can relate the parameters of a specific channel of convolution kernel to the parameters of the other kernel, i.e., make them refocus on the other parts of the model they have never attended to, rather than focus on the input features only. 

    We propose RefConv to replace the original conv layers and experimentally validate that RefConv can improve the performance of various backbone models on ImageNet by a clear margin without extra inference costs or altering model structure. Moreover, RefConv can also improve the ConvNets on object detection and semantic segmentation


