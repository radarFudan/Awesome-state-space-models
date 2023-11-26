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



4. Dilated convolution neural operator for multiscale partial differential equations (https://openreview.net/forum?id=TBLe2BHBsr)



5. Spatio-Temporal Approximation: A Training-Free SNN Conversion for Transformers (https://openreview.net/forum?id=XrunSYwoLr)



6. RefConv: Re-parameterized Refocusing Convolution for Powerful ConvNets (https://openreview.net/forum?id=You77eOFDv)

    This one works on re-parameterization. What's the relationship with the StableSSM reparameterization? This reparameterization aims to augments **the priors to existing structures by establishing connections to the learned kernels**. 

    For example, a depth-wise RefConv can relate the parameters of a specific channel of convolution kernel to the parameters of the other kernel, i.e., make them refocus on the other parts of the model they have never attended to, rather than focus on the input features only. 

    We propose RefConv to replace the original conv layers and experimentally validate that RefConv can improve the performance of various backbone models on ImageNet by a clear margin without extra inference costs or altering model structure. Moreover, RefConv can also improve the ConvNets on object detection and semantic segmentation

