### Available metrics

|   Class   | Range  | Objective | Year | Metric                                                                                               |
|:---------:|:------:|:---------:|:----:|------------------------------------------------------------------------------------------------------|
|   `TV`    | [0, ∞] |     /     | 1937 | [Total Variation](https://en.wikipedia.org/wiki/Total_variation)                                     |
|  `PSNR`   | [0, ∞] |    max    | /    | [Peak Signal-to-Noise Ratio](https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio)               |
|  `SSIM`   | [0, 1] |    max    | 2004 | [Structural Similarity](https://en.wikipedia.org/wiki/Structural_similarity)                         |
| `MS_SSIM` | [0, 1] |    max    | 2004 | [Multi-Scale Structural Similarity](https://ieeexplore.ieee.org/document/1292216/)                   |
|  `LPIPS`  | [0, ∞] |    min    | 2018 | [Learned Perceptual Image Patch Similarity](https://arxiv.org/abs/1801.03924)                        |
|  `GMSD`   | [0, ∞] |    min    | 2013 | [Gradient Magnitude Similarity Deviation](https://arxiv.org/abs/1308.3052)                           |
| `MS_GMSD` | [0, ∞] |    min    | 2017 | [Multi-Scale Gradient Magnitude Similarity Deviation](https://ieeexplore.ieee.org/document/7952357)  |
|  `MDSI`   | [0, ∞] |    min    | 2016 | [Mean Deviation Similarity Index](https://arxiv.org/abs/1608.07433)                                  |
| `HaarPSI` | [0, 1] |    max    | 2018 | [Haar Perceptual Similarity Index](https://arxiv.org/abs/1607.06140)                                 |
|   `VSI`   | [0, 1] |    max    | 2014 | [Visual Saliency-based Index](https://ieeexplore.ieee.org/document/6873260)                          |
|  `FSIM`   | [0, 1] |    max    | 2011 | [Feature Similarity](https://ieeexplore.ieee.org/document/5705575)                                   |
|   `FID`   | [0, ∞] |    min    | 2017 | [Fréchet Inception Distance](https://arxiv.org/abs/1706.08500)                                       |
|  `NIMA`   |  10输出  |    max    | 2017 | [Neural Image Assessment](https://arxiv.org/abs/1709.05424)                                       |  


| Function | Meaning                                                                     |
|----------|-----------------------------------------------------------------------------|
| PSNR     | Peak Signal-to-Noise Ratio (峰值信噪比)                                          |
| SSIM     | Structural Similarity Index (结构相似性指数)                                       |
| MS_SSIM  | Multi-Scale Structural Similarity Index (多尺度结构相似性指数)                        |
| LPIPS    | Learned Perceptual Image Patch Similarity (学习感知图像块相似性)                      |
| GMSD     | Gradient Magnitude Structural Similarity Index (梯度幅值结构相似性指数)                |
| MS_GMSD  | Multi-Scale Gradient Magnitude Structural Similarity Index (多尺度梯度幅值结构相似性指数) |
| MDSI     | Multi-Scale Dilation Similarity Index (多尺度膨胀相似性指数)                          |
| HAARPSI  | Haar-based Perceptual Similarity Index (基于Haar的感知相似性指数)                     |
| VSI      | Visual Saliency-based Index (基于视觉显著性的指数)                                    |
| TV       | Total Variation (总变差)                                                       |
| FSIM     | Frequency Domain Structural Similarity Index (频域结构相似性指数)                    |
| FID      | Fréchet Inception Distance (Fréchet Inception 距离)                           |
| NIMA     | Neural Image Assessment (神经图像评估)1-10分的概率，无参考图像评估                            |