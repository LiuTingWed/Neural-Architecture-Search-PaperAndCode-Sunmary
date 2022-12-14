# Neural-Architecture-Search-PaperAndCode-Sunmary

This repository collects recent NAS based methods and provide a summary (Paper and Code) by year and task. We hope this repo can help you better understand the trend of NAS era.

---
# Change
:fire: Add 1 NIPS paper 2022


---
# <h1 id='Content'>Content</h1>
+ ## 2022
    - [Object Classfication/Detection](#OC22)
    - [Stereo Matching](#SM22)
    - [Super Resolution](#SR22)
    - [3D Point Clouds](#3PC22)
    - [Medical](#Med22)
    - [Deep Image Prior](#DIP22)
    - [Image Restoration](#IR22)
    - [Multi Task](#MT22)
    - [Loss Function](#LF22)
+ ## 2021
    - [Object Classfication/Detection](#OC21)
    - [Image Segmentation](#Seg21)
    - [Video Action Segmentation](#VASeg21)
    - [Video Recognition](#VR21)
    - [Model Pruning](#MP21)
    - [Image Super Resolution](#SR21)
    - [Point Clouds](#PC21)
    - [Medical](#Med21)
    - [Re-Identification/Detection](#ReID21)
    - [Adversarial Attack](#AA21)
    - [Image Restoration](#IR21)
    - [Image Deblurring](#ID21)
    - [Pose Estimation](#PE21)
    - [Video Pose Estimation](#VPE21)
    - [Loss Function](#LF21)
+ ## 2020
    - [Object Classfication/Detection](#OC20)
    - [Image Segmentation](#Seg20)
    - [Video Classification](#VC20)
    - [Lane Detection](#LD20)
    - [Image Super-Resolution](#ISR20)
    - [Image Restoration](#IR20)
    - [Medical](#Med20)
    - [3D Pose Estimation](#3HPE20)
    - [Face Detection](#FD20)
    - [Face Anti-Spoofing](#FAS20)
    - [Scene Text Recognition](#STR20)
    - [Adversarial Attack](#AA20)
    - [Model Defense](#AA20)
    - [Bad Weather Removal](#BWR20)
    - [Image Denoising](#ID20)
    - [Deep Image Prior](#DIP20)
    - [Stereo Matching](#SM20)
    - [Crowd Counting](#CC20)
    - [Model Pruning](#MP20)
    - [Text Representation](#TR20)
+ ## 2019
    - [Object Classfication/Detection](#OC19)
    - [Image Segmentation](#Seg19)
    - [Re-Identification/Detection](#ReID19)
    - [Stereo Matching](#SM19)
    - [Multimodal Fusion](#MF19)
    - [Model Pruning](#MP19)
+ ## 2018
    - [Object Classfication/Detection](#OC18)

+ ## [Survey](#Sur)
---
# 2022

### **<h5 id='OC22'>Object Classfication/Detection</h5>**
| Name | Pub. | Title | Links |
| --- | --- | --- | --- |
| BMNAS | AAAI | Bilevel Multimodal Neural Architecture Search| [Paper](https://arxiv.org/pdf/2104.09379.pdf)/[Code](https://github.com/Somedaywilldo/BM-NAS)|
| DPNAS | AAAI | Neural Architecture Search for Deep Learning with Differential Privacy| [Paper](https://arxiv.org/pdf/2110.08557.pdf)/Code|
| LFMNAS | AAAI | Learning from Mistakes - A Framework for Neural Architecture Search| [Paper](https://arxiv.org/pdf/2111.06353.pdf)/Code|
| ArchGraph | CVPR | Acyclic Architecture Relation Predictor for Task-Transferable Neural Architecture Search | [Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Huang_Arch-Graph_Acyclic_Architecture_Relation_Predictor_for_Task-Transferable_Neural_Architecture_Search_CVPR_2022_paper.pdf)/[Code](https://github.com/Centaurus982034/Arch-Graph)  |
| PAMKD | CVPR | Performance-Aware Mutual Knowledge Distillation for Improving Neural Architecture Search | [Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Xie_Performance-Aware_Mutual_Knowledge_Distillation_for_Improving_Neural_Architecture_Search_CVPR_2022_paper.pdf)/Code |
| BaLeNAS | CVPR | Differentiable Architecture Search via the Bayesian Learning Rule | [Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhang_BaLeNAS_Differentiable_Architecture_Search_via_the_Bayesian_Learning_Rule_CVPR_2022_paper.pdf)/Code |
| LGA | CVPR | Demystifying the Neural Tangent Kernel from a Practical Perspective: Can it be trusted for Neural Architecture Search without training? | [Paper](https://arxiv.org/pdf/2203.14577.pdf)/[Code](https://github.com/nutellamok/DemystifyingNTK)  |
| TFTAS | CVPR | Training-free Transformer Architecture Search | [Paper](https://arxiv.org/pdf/2203.12217.pdf)/Code|
| ??-DARTS | CVPR | Beta-Decay Regularization for Differentiable Architecture Search | [Paper](https://arxiv.org/pdf/2203.01665.pdf)/Code|
| MetaNTK-NAS | CVPR | Global Convergence of MAML and Theory-Inspired Neural Architecture Search for Few-Shot Learning | [Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Global_Convergence_of_MAML_and_Theory-Inspired_Neural_Architecture_Search_for_CVPR_2022_paper.pdf)/[Code](https://github.com/YiteWang/MetaNTK-NAS)
| GreedyNASv2 | CVPR | Greedier Search with a Greedy Path Filter | [Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Huang_GreedyNASv2_Greedier_Search_With_a_Greedy_Path_Filter_CVPR_2022_paper.pdf)/Code|
| RMI-NAS | CVPR | Neural Architecture Search with Representation Mutual Information | [Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Zheng_Neural_Architecture_Search_With_Representation_Mutual_Information_CVPR_2022_paper.pdf)/[Code](https://git.openi.org.cn/PCL/AutoML/XNAS)|
| Shapley-NAS | CVPR | Discovering Operation Contribution for Neural Architecture Search| [Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Xiao_Shapley-NAS_Discovering_Operation_Contribution_for_Neural_Architecture_Search_CVPR_2022_paper.pdf)/[Code](https://github.com/Euphoria16/Shapley-NAS.git)|
| GPUNet | CVPR | Searching the Deployable Convolution Neural Networks for GPUs| [Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Searching_the_Deployable_Convolution_Neural_Networks_for_GPUs_CVPR_2022_paper.pdf)/Code|
| ViT-Slim | CVPR | Multi-Dimension Searching in Continuous Optimization Space| [Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Chavan_Vision_Transformer_Slimming_Multi-Dimension_Searching_in_Continuous_Optimization_Space_CVPR_2022_paper.pdf)/[Code](https://github.com/Arnav0400/ViT-Slim)|
| SUMNAS | ICLR | SUMNAS: Supernet with Unbiased Meta-Features for Neural Architecture Search| [Paper](https://openreview.net/pdf?id=Z8FzvVU6_Kj)/Code|
| NASViT | ICLR | Neural Architecture Search for Efficient Vision Transformers with Gradient Conflict aware Supernet Training| [Paper](https://openreview.net/pdf?id=Qaw16njk6L)/[Code](https://github.com/facebookresearch/NASViT)|
| ASViT | ICLR | Auto-scaling Vision Transformers without Training | [Paper](https://openreview.net/pdf?id=H94a1_Pyr-6)/[Code]( https://github.com/VITA-Group/AsViT)|
| GMNAS | ICLR | Generalizing Few-Shot NAS with Gradient Matching | [Paper](https://openreview.net/pdf?id=_jMtny3sMKU)/[Code](https://github.com/skhu101/GM-NAS)|
| ScaleNet | ECCV | Searching for the Model to Scale | [Paper](https://arxiv.org/pdf/2207.07267.pdf)/[Code](https://github.com/luminolx/ScaleNet)
| SNNNAS | ECCV | Neural Architecture Search for Spiking Neural Networks| [Paper](https://arxiv.org/pdf/2201.10355.pdf)/[Code](https://github.com/Intelligent-Computing-Lab-Yale/Neural-Architecture-Search-for-Spiking-Neural-Networks)  |
| EAutoDet | ECCV | Efficient Architecture Search for Object Detection| [Paper](https://arxiv.org/pdf/2203.10747.pdf)/Code|
| U-BoostNAS | ECCV | Utilization-Boosted Differentiable Neural Architecture Search| [Paper](https://arxiv.org/pdf/2203.12412.pdf)/[Code](https://github.com/yuezuegu/UBoostNAS)|
| SuperTickets | ECCV | Drawing Task-Agnostic Lottery Tickets from Supernets via Jointly Architecture Searching and Parameter Pruning| [Paper](https://arxiv.org/pdf/2207.03677.pdf)/[Code](https://github.com/RICE-EIC/SuperTickets)
| DistPro | ECCV | Searching A Fast Knowledge Distillation Process via Meta Optimization| [Paper](https://arxiv.org/pdf/2204.05547.pdf)/Code|
EKG | ECCV | Ensemble Knowledge Guided Sub-network Search and Fine-tuning for Filter Pruning| [Paper](https://arxiv.org/pdf/2203.02651.pdf)/[Code](https://github.com/sseung0703/EKG)|
EAGAN | ECCV |Efficient Two-stage Evolutionary Architecture Search for GANs| [Paper](https://arxiv.org/pdf/2111.15097.pdf)/[Code](https://github.com/marsggbo/EAGAN)|
ViTAS | ECCV | Vision Transformer Architecture Search| [Paper](https://arxiv.org/pdf/2106.13700.pdf)/[Code](https://github.com/xiusu/ViTAS)|
DFNAS | ECCV | Data-Free Neural Architecture Search via Recursive Label Calibration| [Paper](https://arxiv.org/pdf/2112.02086.pdf)/[Code](https://github.com/liuzechun/Data-Free-NAS)|
BaLeNAS | ECCV | Differentiable Architecture Search via the Bayesian Learning Rule| [Paper](https://arxiv.org/pdf/2111.13204.pdf)/Code|

### **<h5 id='SM22'>Stereo Matching</h5>**
| Name | Pub. | Title | Links |
| --- | --- | --- | --- |
EASNet| ECCV | Searching Elastic and Accurate Network Architecture for Stereo Matching| [Paper](https://arxiv.org/pdf/2207.09796.pdf)/[Code](https://github.com/HKBU-HPML/EASNet.git)|

### **<h5 id='SR22'>Image Super Resolution</h5>**
| Name | Pub. | Title | Links |
| --- | --- | --- | --- |
CANAS| ECCV | Compiler-Aware Neural Architecture Search for On-Mobile Real-time Super-Resolution| [Paper](https://arxiv.org/pdf/2207.12577.pdf)/[Code](https://github.com/wuyushuwys/compiler-aware-nas-sr)|

### **<h5 id='3PC22'>3D Point Clouds</h5>**
| Name | Pub. | Title | Links |
| --- | --- | --- | --- |
LidarNAS | ECCV | Unifying and Searching Neural Architectures for 3D Point Clouds| Paper/Code|
### **<h5 id='Med22'>Medical</h5>**
| Name | Pub. | Title | Links |
| --- | --- | --- | --- |
HyperSegNAS | CVPR | Bridging One-Shot Neural Architecture Search with 3D Medical Image Segmentation using HyperNet| [Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Peng_HyperSegNAS_Bridging_One-Shot_Neural_Architecture_Search_With_3D_Medical_Image_CVPR_2022_paper.pdf)/Code|

### **<h5 id='DIP22'>Deep Image Prior</h5>**
| Name | Pub. | Title | Links |
| --- | --- | --- | --- |
| ISNAS| CVPR | Image-Specific Neural Architecture Search for Deep Image Prior| [Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Arican_ISNAS-DIP_Image-Specific_Neural_Architecture_Search_for_Deep_Image_Prior_CVPR_2022_paper.pdf)/[Code](https://github.com/ozgurkara99/ISNAS-DIP)|

### **<h5 id='IR22'>Image Restoration</h5>**
| Name | Pub. | Title | Links |
| --- | --- | --- | --- |
|- | ECCV | Spectrum-aware and Transferable Architecture Search for Hyperspectral Image Restoration| Paper/Code|

### **<h5 id='MT22'>Multi Task</h5>**
| Name | Pub. | Title | Links |
| --- | --- | --- | --- |
| :fire:AutoMTL | NeurIPS | A Programming Framework for Automating Efficient Multi-Task Learning|[Paper](https://arxiv.org/pdf/2110.13076.pdf)/[Code](https://github.com/zhanglijun95/AutoMTL)|
### **<h5 id='LR22'>Loss Function</h5>**
| Name | Pub. | Title | Links |
| --- | --- | --- | --- |
|AutoLossZero | CVPR | Searching Loss Functions from Scratch for Generic Tasks| [Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_AutoLoss-Zero_Searching_Loss_Functions_From_Scratch_for_Generic_Tasks_CVPR_2022_paper.pdf)/[Code](https://github.com/cpsxhao/AutoLoss-Zero)|
|AutoLossGMS | CVPR | Searching Generalized Margin-based Softmax Loss Function for Person Re-identification| [Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Gu_AutoLoss-GMS_Searching_Generalized_Margin-Based_Softmax_Loss_Function_for_Person_Re-Identification_CVPR_2022_paper.pdf)/Code|
- [Back to content](#Content)
# 2021
### **<h5 id='OC21'>Object Classfication/Detection</h5>**
| Name | Pub. | Title | Links |
| --- | --- | --- | --- |
| NASTransfer | AAAI | NASTransfer: Analyzing Architecture Transferability in Large Scale Neural Architecture Search | [Paper](https://arxiv.org/pdf/2006.13314.pdf)/Code  |
| BCNet | CVPR | Searching for Network Width with Bilaterally Coupled Network | [Paper](https://arxiv.org/pdf/2105.10533.pdf)/Code  |
| FPNAS | CVPR | Fast Probabilistic Neural Architecture Search| [Paper](https://arxiv.org/pdf/2011.10949.pdf)/Code  |
| TransNAS | CVPR | Improving Transferability and Generalizability of Cross-Task Neural Architecture Search| [Paper](https://arxiv.org/pdf/2105.11871.pdf)/Code  |
| JointDetNAS | CVPR | Upgrade Your Detector with NAS, Pruning and Dynamic Distillation| [Paper](https://arxiv.org/pdf/2105.11871.pdf)/Code  |
| LR | CVPR | Landmark Regularization: Ranking Guided Super-Net Training in Neural Architecture Search| [Paper](https://arxiv.org/pdf/2104.05309.pdf)/Code  |
| NetAdaptV2 | CVPR | Efficient Neural Architecture Search with Fast Super-Network Training and Architecture Optimization| [Paper](https://arxiv.org/pdf/2104.00031.pdf)/[Code](http://web.mit.edu/netadapt/)  |
| NEAS | CVPR | One-Shot Neural Ensemble Architecture Search by Diversity-Guided Search Space Shrinking| [Paper](https://arxiv.org/pdf/2104.00597.pdf)/[Code](https://github.com/researchmm/NEAS)  |
| DSNet | CVPR | Dynamic Slimmable Network| [Paper](https://arxiv.org/pdf/2103.13258.pdf)/[Code](https://github.com/changlin31/DS-Net)  |
| PASNAS | CVPR | Prioritized Architecture Sampling with Monto-Carlo Tree Search| [Paper](https://arxiv.org/pdf/2103.07289.pdf)/[Code](https://github.com/eric8607242/SGNAS)  |
| CTNAS | CVPR | Contrastive Neural Architecture Search with Neural Architecture Comparators| [Paper](https://arxiv.org/pdf/2103.05471.pdf)/[Code](https://github.com/chenyaofo/CTNAS)  |
| OPANAS | CVPR | One-Shot Path Aggregation Network Architecture Search for Object Detection| [Paper](https://arxiv.org/pdf/2103.04507.pdf)/[Code](https://github.com/VDIGPKU/OPANAS)  |
| AttentiveNAS | CVPR | Improving Neural Architecture Search via Attentive Sampling| [Paper](https://arxiv.org/pdf/2011.09011.pdf)/[Code](https://github.com/facebookresearch/AttentiveNAS)  |
| ReNAS | CVPR | Relativistic Evaluation of Neural Architecture Search| [Paper](https://arxiv.org/pdf/1910.01523.pdf)/Code  |
| HourNAS | CVPR | Extremely Fast Neural Architecture| [Paper](https://arxiv.org/pdf/1910.01523.pdf)/Code  |
| ICConv | CVPR |  Inception Convolution with Efficient Dilation Search| [Paper](https://arxiv.org/pdf/2012.13587.pdf)/[Code](https://github.com/yifan123/IC-Conv)  |
| TE-NAS | ICLR |  Neural Architecture Search on ImageNet in Four GPU Hours: A Theoretically Inspired Perspective| [Paper](https://openreview.net/pdf?id=Cnon5ezMHtu)/[Code](https://github.com/VITA-Group/TENAS)|
| DrNAS | ICLR | Dirichlet Neural Architecture Search | [Paper](https://openreview.net/pdf?id=9FWas6YbmB3)/[Code](https://github.com/xiangning-chen/DrNAS)  |
| PiNAS | ICCV |  Improving Neural Architecture Search by Reducing Supernet Training Consistency Shift| [Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Peng_Pi-NAS_Improving_Neural_Architecture_Search_by_Reducing_Supernet_Training_Consistency_ICCV_2021_paper.pdf)/[Code](https://github.com/Ernie1/Pi-NAS)  |
| FairNAS | ICCV | Rethinking Evaluation Fairness of Weight Sharing Neural Architecture Search| [Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Chu_FairNAS_Rethinking_Evaluation_Fairness_of_Weight_Sharing_Neural_Architecture_Search_ICCV_2021_paper.pdf)/[Code](https://github.com/xiaomi-automl/FairNAS)  |
| GLiT | ICCV | Neural Architecture Search for Global and Local Image Transformer| [Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Chen_GLiT_Neural_Architecture_Search_for_Global_and_Local_Image_Transformer_ICCV_2021_paper.pdf)/[Code](https://github.com/bychen515/GLiT)  
| BNNAS | ICCV | Neural Architecture Search with Batch Normalization| [Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Chen_BN-NAS_Neural_Architecture_Search_With_Batch_Normalization_ICCV_2021_paper.pdf)/[Code](https://github.com/bychen515/BNNAS) 
| DONNA | ICCV | Distilling Optimal Neural Networks: Rapid Search in Diverse Spaces| [Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Moons_Distilling_Optimal_Neural_Networks_Rapid_Search_in_Diverse_Spaces_ICCV_2021_paper.pdf)/Code|
| HOP | ICCV | Not All Operations Contribute Equally: Hierarchical Operation-adaptive Predictor for Neural Architecture Search| [Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Chen_Not_All_Operations_Contribute_Equally_Hierarchical_Operation-Adaptive_Predictor_for_Neural_ICCV_2021_paper.pdf)/Code|
| VIMNAS | ICCV | Learning Latent Architectural Distribution in Differentiable Neural Architecture Search via Variational Information Maximization| [Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Wang_Learning_Latent_Architectural_Distribution_in_Differentiable_Neural_Architecture_Search_via_ICCV_2021_paper.pdf)/Code|
| DDAS | ICCV | Direct Differentiable Augmentation Search| [Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Liu_Direct_Differentiable_Augmentation_Search_ICCV_2021_paper.pdf)/[Code](https://github.com/zxcvfd13502/DDAS_code)|
| BossNAS | ICCV | Exploring Hybrid CNN-transformers with Block-wisely Self-supervised Neural Architecture Search| [Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Li_BossNAS_Exploring_Hybrid_CNN-Transformers_With_Block-Wisely_Self-Supervised_Neural_Architecture_Search_ICCV_2021_paper.pdf)/[Code](https://github.com/changlin31/BossNAS)|
| NAS-OoD | ICCV | Neural Architecture Search for Out-of-Distribution Generalization| [Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Bai_NAS-OoD_Neural_Architecture_Search_for_Out-of-Distribution_Generalization_ICCV_2021_paper.pdf)/Code|
| IDARTS | ICCV | Interactive Differentiable Architecture Search| [Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Xue_IDARTS_Interactive_Differentiable_Architecture_Search_ICCV_2021_paper.pdf)/Code|
| AutoSpace | ICCV | Neural Architecture Search with Less Human Interference| [Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhou_AutoSpace_Neural_Architecture_Search_With_Less_Human_Interference_ICCV_2021_paper.pdf)/[Code](https://github.com/zhoudaquan/AutoSpace.git)|
| NSENAS | ICCV | Evolving Search Space for Neural Architecture Search|[Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Ci_Evolving_Search_Space_for_Neural_Architecture_Search_ICCV_2021_paper.pdf)/[Code](https://github.com/orashi/NSE_NAS)|
| MNNAS | ICCV | Meta Navigator:Search for a Good Adaptation Policy for Few-shot Learning|[Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhang_Meta_Navigator_Search_for_a_Good_Adaptation_Policy_for_Few-Shot_ICCV_2021_paper.pdf)/Code|
| OQATNets | ICCV | Once Quantization-Aware Training: High Performance Extremely Low-bit Architecture Search|[Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Shen_Once_Quantization-Aware_Training_High_Performance_Extremely_Low-Bit_Architecture_Search_ICCV_2021_paper.pdf)/[Code](https://github.com/LaVieEnRoseSMZ/OQA)|
| AutoFormer: | ICCV | Searching Transformers for Visual Recognition|[Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Chen_AutoFormer_Searching_Transformers_for_Visual_Recognition_ICCV_2021_paper.pdf)/[Code](https://github.com/microsoft/Cream)|
| NASLib: | NeurIPS | How Powerful are Performance Predictors in Neural Architecture Search?|[Paper](https://papers.nips.cc/paper/2021/file/ef575e8837d065a1683c022d2077d342-Paper.pdf)/[Code](https://github.com/automl/naslib)|
| TANS: | NeurIPS | Task-Adaptive Neural Network Search with Meta-Contrastive Learning|[Paper](https://papers.nips.cc/paper/2021/file/b20bb95ab626d93fd976af958fbc61ba-Paper.pdf)/[Code](https://github.com/wyjeong/TANS)|
| AutoGL | NeurIPS | Graph Differentiable Architecture Search with Structure Learning|[Paper](https://papers.nips.cc/paper/2021/file/8c9f32e03aeb2e3000825c8c875c4edd-Paper.pdf)/[Code](https://github.com/THUMNLab/AutoGL)|
| AutoFormerV2 | NeurIPS | Searching the Search Space of Vision Transformer|[Paper](https://papers.nips.cc/paper/2021/file/48e95c45c8217961bf6cd7696d80d238-Paper.pdf)/[Code](https://github.com/microsoft/Cream)|
| TSE | NeurIPS | Speedy Performance Estimation for Neural Architecture Search|[Paper](https://papers.nips.cc/paper/2021/file/2130eb640e0a272898a51da41363542d-Paper.pdf)/[Code](https://github.com/rubinxin/TSE)|
| FNA++ | TPAMI | Fast Network Adaptation via Parameter Remapping and Architecture Search|[Paper](https://arxiv.org/pdf/2006.12986.pdf)/[Code](hhttps://github.com/JaminFong/FNA)

### **<h5 id='Seg21'>Image Segmentation</h5>**
| Name | Pub. | Title | Links |
| --- | --- | --- | --- |
| DCNAS | CVPR | Densely Connected Neural Architecture Search for Semantic Image Segmentation | [Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_DCNAS_Densely_Connected_Neural_Architecture_Search_for_Semantic_Image_Segmentation_CVPR_2021_paper.pdf)/Code  |
| AutoRTNet | IJCV | Real-Time Semantic Segmentation via Auto Depth, Downsampling Joint Decision and Feature Aggregation| [Paper](https://arxiv.org/pdf/2003.14226.pdf)/Code  |
### **<h5 id='VASeg21'>Video Action Segmentation</h5>**
| Name | Pub. | Title | Links |
| --- | --- | --- | --- |
| Global2Local | CVPR | Efficient Structure Search for Video Action Segmentation | [Paper](https://arxiv.org/pdf/2101.00910.pdf)/Code  |

### **<h5 id='VR21'>Video Recognition</h5>**
| Name | Pub. | Title | Links |
| --- | --- | --- | --- |
| AutoTSNet | ICCV | Searching for Two-Stream Models in Multivariate Space for Video Recognition| [Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Gong_Searching_for_Two-Stream_Models_in_Multivariate_Space_for_Video_Recognition_ICCV_2021_paper.pdf)/Code  |

### **<h5 id='MP21'>Model Pruning</h5>**
| Name | Pub. | Title | Links |
| --- | --- | --- | --- |
| RTSRNAS | ICCV | Achieving on-Mobile Real-Time Super-Resolution with Neural Architecture and Pruning Search | [Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_NAS-FCOS_Fast_Neural_Architecture_Search_for_Object_Detection_CVPR_2020_paper.pdf)/Code

### **<h5 id='PC21'>Point Cloud</h5>**
| Name | Pub. | Title | Links |
| --- | --- | --- | --- |
| PointSeaNet | ICCV | Differentiable Convolution Search for Point Cloud Processing| [Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Nie_Differentiable_Convolution_Search_for_Point_Cloud_Processing_ICCV_2021_paper.pdf)/Code |

### **<h5 id='Med21'>Medical</h5>**
| Name | Pub. | Title | Links |
| --- | --- | --- | --- |
| DiNTS | CVPR | Differentiable Neural Network Topology Search for 3D Medical Image Segmentation | [Paper](https://arxiv.org/pdf/2103.15954.pdf)/Code  |

### **<h5 id='ReID21'>Re-Identification/Detection</h5>**
| Name | Pub. | Title | Links |
| --- | --- | --- | --- |
| CDS | CVPR | Combined Depth Space based Architecture Search For Person Re-identification | [Paper](https://arxiv.org/pdf/2104.04163.pdf)/Code  |
| NFS | CVPR | Neural Feature Search for RGB-Infrared Person Re-Identification | [Paper](https://arxiv.org/pdf/2104.02366.pdf)/Code  |
| CMNAS | ICCV | Cross-Modality Neural Architecture Search for Visible-Infrared Person Re-Identification | [Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Fu_CM-NAS_Cross-Modality_Neural_Architecture_Search_for_Visible-Infrared_Person_Re-Identification_ICCV_2021_paper.pdf)/[Code](https://github.com/JDAI-CV/CM-NAS)  |

### **<h5 id='AA21'>Adversarial Attack</h5>**
| Name | Pub. | Title | Links |
| --- | --- | --- | --- |
| AdvRush | ICCV | Searching for Adversarially Robust Neural Architectures | [Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Mok_AdvRush_Searching_for_Adversarially_Robust_Neural_Architectures_ICCV_2021_paper.pdf)/[Code](https://github.com/nutellamok/advrush)  |

### **<h5 id='IR21'>Image Restoration</h5>**
| Name | Pub. | Title | Links |
| --- | --- | --- | --- |
| TASNet | ICCV | Searching for Controllable Image Restoration Networks| [Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Kim_Searching_for_Controllable_Image_Restoration_Networks_ICCV_2021_paper.pdf)/[Code](https://github.com/ghimhw/TASNet)|

### **<h5 id='ID21'>Image Deblurring</h5>**
| Name | Pub. | Title | Links |
| --- | --- | --- | --- |
| PyNAS | ICCV | Pyramid Architecture Search for Real-Time Image Deblurring| [Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Hu_Pyramid_Architecture_Search_for_Real-Time_Image_Deblurring_ICCV_2021_paper.pdf)/Code|

### **<h5 id='PE21'>Pose Estimation</h5>**
| Name | Pub. | Title | Links |
| --- | --- | --- | --- |
| NPPNet | ICCV | Neural Architecture Search for Joint Human Parsing and Pose Estimation| [Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_NAS-FCOS_Fast_Neural_Architecture_Search_for_Object_Detection_CVPR_2020_paper.pdf)/[Code](https://github.com/GuHuangAI/NPP)|

### **<h5 id='VPE21'>Video Pose Estimation</h5>**

| Name | Pub. | Title | Links |
| --- | --- | --- | --- |
| ViPNAS | CVPR | Efficient Video Pose Estimation via Neural Architecture Search | [Paper](https://arxiv.org/pdf/2105.10154.pdf)/[Code](https://www.notion.so/dd9cc2c1457e4e9fab10ae104685eb81)  |

### **<h5 id='LR21'>Loss Function</h5>**
| Name | Pub. | Title | Links |
| --- | --- | --- | --- |
| AutoLoss | ICLR | Loss Function Discovery for Object Detection via Convergence-Simulation Driven Search  | [Paper](https://openreview.net/pdf?id=5jzlpHvvRk)/[Code](https://github.com/PerdonLiu/CSE-Autoloss) |
| AutoSeg-Loss: | ICLR |Searching Metric Surrogates for Semantic Segmentation   | [Paper](https://openreview.net/pdf?id=MJAqnaC2vO1)/[Code](https://github.com/fundamentalvision/Auto-Seg-Loss) |
| AutoBalance | NeurIP | Optimized Loss Functions for Imbalanced Data | [Paper](https://papers.nips.cc/paper/2021/file/191f8f858acda435ae0daf994e2a72c2-Paper.pdf)/Code |
| APNAS | NeurIP | Searching Parameterized AP Loss for Object Detection | [Paper](https://papers.nips.cc/paper/2021/file/191f8f858acda435ae0daf994e2a72c2-Paper.pdf)/Code |
- [Back to content](#Content)
# 2020
### **<h5 id='OC20'>Object Classfication/Detection</h5>**
| Name | Pub. | Title | Links |
| --- | --- | --- | --- |
| SMNAS | AAAI | SM-NAS: Structural-to-Modular Neural Architecture Search for Object Detection| [Paper](https://arxiv.org/pdf/1911.09929.pdf)/Code|
| KDNAS | AAAI | Towards Oracle Knowledge Distillation with Neural Architecture Search| [Paper](https://arxiv.org/pdf/1911.13019.pdf)/Code|
| NASP | AAAI | Efficient Neural Architecture Search via Proximal Iterations| [Paper](https://arxiv.org/pdf/1905.13577.pdf)/[Code](https://github.com/AutoML-Research/NASP) |
| AlphaX | AAAI | Neural Architecture Search using Deep Neural Networks and Monte Carlo Tree Search| [Paper](https://arxiv.org/pdf/1805.07440.pdf)/[Code](https://github.com/linnanwang/AlphaX-NASBench101) |
| BNAS | AAAI | Binarized Neural Architecture Search| [Paper](https://arxiv.org/pdf/1911.10862.pdf)/Code |
| InstaNAS | AAAI | Instance-aware Neural Architecture Search | [Paper](https://arxiv.org/pdf/1811.10201.pdf)/Code |
| PGNAS | AAAI | Posterior-Guided Neural Architecture Search| [Paper](https://arxiv.org/pdf/1906.09557.pdf)/[Code](https://github.com/scenarios/PGNAS) |
| CR-NAS | ICLR | Computation Reallocation for Object Detectionh| [Paper](https://arxiv.org/pdf/1906.09557.pdf)/Code |
| OFANAS | ICLR | Once for All: Train One Network and Specialize it for Efficient Deployment| [Paper](https://openreview.net/pdf?id=HylxE1HKwS)/[Code](https://github.com/mit-han-lab/once-for-all) |
| PC-DARTS | ICLR | PC-DARTS: Partial Channel Connections for Memory-Efficient Architecture Search| [Paper](https://openreview.net/pdf?id=BJlS634tPr)/[Code](https://github.com/yuhuixu1993/PC-DARTS) |
| AtomNAS | ICLR | Fine-Grained End-to-End Neural Architecture Search| [Paper](https://openreview.net/pdf?id=BylQSxHFwr)/[Code](https://github.com/meijieru/AtomNAS) |
| HitDetector | CVPR | Hierarchical Trinity Architecture Search for Object Detection | [Paper](https://arxiv.org/abs/2003.11818)/[Code](https://github.com/ggjy/HitDet.pytorch)  |
| CARS | CVPR | Continuous Evolution for Efficient Neural Architecture Search | [Paper](https://arxiv.org/pdf/1909.04977.pdf)/[Code](https://github.com/zhaohui-yang/CARS)  |
| TuNAS | CVPR | Can weight sharing outperform random architecture search? An investigation with TuNAS | [Paper](https://arxiv.org/pdf/2008.06120.pdf)/[Code](https://github.com/google-research/google-research/tree/master/tunas)  |
| AOWS | CVPR | Adaptive and optimal network width search with latency constraints | [Paper](https://arxiv.org/pdf/2005.10481.pdf)/[Code](https://github.com/bermanmaxim/AOWS)  |
| RDS | CVPR | Rethinking Differentiable Search for Mixed-Precision Neural Networks | [Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Cai_Rethinking_Differentiable_Search_for_Mixed-Precision_Neural_Networks_CVPR_2020_paper.pdf)/Code  |
| Adnet | CVPR | Network Adjustment: Channel Search Guided by FLOPs Utilization Ratio | [Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Chen_Network_Adjustment_Channel_Search_Guided_by_FLOPs_Utilization_Ratio_CVPR_2020_paper.pdf)/[Code](https://github.com/danczs/NetworkAdjustment)  |
| AdverNAS | CVPR | Network Adjustment: Channel Search Guided by FLOPs Utilization Ratio | [Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Gao_AdversarialNAS_Adversarial_Neural_Architecture_Search_for_GANs_CVPR_2020_paper.pdf)/[Code](https://github.com/chengaopro/AdversarialNAS)  |
| MTLNAS | CVPR | Task-Agnostic Neural Architecture Search towards General-Purpose Multi-Task Learning| [Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Gao_MTL-NAS_Task-Agnostic_Neural_Architecture_Search_Towards_General-Purpose_Multi-Task_Learning_CVPR_2020_paper.pdf)/[Code](https://github.com/bhpfelix/MTLNAS)  
| MiLeNAS | CVPR | Efficient Neural Architecture Search via Mixed-Level Reformulation| [Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/He_MiLeNAS_Efficient_Neural_Architecture_Search_via_Mixed-Level_Reformulation_CVPR_2020_paper.pdf)/[Code](https://github.com/chaoyanghe/MiLeNAS)  |
| DSNAS | CVPR | Direct Neural Architecture Search without Parameter Retraining| [Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Hu_DSNAS_Direct_Neural_Architecture_Search_Without_Parameter_Retraining_CVPR_2020_paper.pdf)/Code |
| SPNAS | CVPR | Serial-to-Parallel Backbone Search for Object Detection| [Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Jiang_SP-NAS_Serial-to-Parallel_Backbone_Search_for_Object_Detection_CVPR_2020_paper.pdf)/Code |
| DNA | CVPR | Block-wisely Supervised Neural Architecture Search with Knowledge Distillation| [Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Li_Block-Wisely_Supervised_Neural_Architecture_Search_With_Knowledge_Distillation_CVPR_2020_paper.pdf)/[Code](https://github.com/changlin31/DNA) |
| GPNAS | CVPR | Gaussian Process based Neural Architecture Search| [Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Li_GP-NAS_Gaussian_Process_Based_Neural_Architecture_Search_CVPR_2020_paper.pdf)/Code |
| AutoNL | CVPR | Neural Architecture Search for Lightweight Non-Local Networks| [Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Li_Neural_Architecture_Search_for_Lightweight_Non-Local_Networks_CVPR_2020_paper.pdf)/[Code](https://github.com/LiYingwei/AutoNL) |
| SGAS | CVPR | Sequential Greedy Architecture Search| [Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Li_SGAS_Sequential_Greedy_Architecture_Search_CVPR_2020_paper.pdf)/[Code](https://www.deepgcns.org/auto/sgas) |
| MemNAS | CVPR | Memory-Efficient Neural Architecture Search with Grow-Trim Learning| [Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Liu_MemNAS_Memory-Efficient_Neural_Architecture_Search_With_Grow-Trim_Learning_CVPR_2020_paper.pdf)/[Code](https://github.com/MemNAS/MemNAS) |
| AKD | CVPR |  Search to Distill: Pearls are Everywhere but not the Eyes| [Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Liu_Search_to_Distill_Pearls_Are_Everywhere_but_Not_the_Eyes_CVPR_2020_paper.pdf)/Code|
| UNAS | CVPR | Differentiable Architecture Search Meets Reinforcement Learning| [Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Vahdat_UNAS_Differentiable_Architecture_Search_Meets_Reinforcement_Learning_CVPR_2020_paper.pdf)/[Code](https://github.com/NVlabs/unas)|
FBNetV2 | CVPR | Differentiable Neural Architecture Search for Spatial and Channel Dimensions| [Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wan_FBNetV2_Differentiable_Neural_Architecture_Search_for_Spatial_and_Channel_Dimensions_CVPR_2020_paper.pdf)/[Code](https://github.com/facebookresearch/mobile-vision)|
EcoNAS | CVPR | Finding Proxies for Economical Neural Architecture Search| [Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhou_EcoNAS_Finding_Proxies_for_Economical_Neural_Architecture_Search_CVPR_2020_paper.pdf)/Code|
DenseNAS | CVPR | Densely Connected Search Space for More Flexible Neural Architecture Search| [Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Fang_Densely_Connected_Search_Space_for_More_Flexible_Neural_Architecture_Search_CVPR_2020_paper.pdf)/[Code](https://github.com/JaminFong/DenseNAS)|
NSGANetV2 | ECCV | Evolutionary Multi-Objective Surrogate-Assisted Neural Architecture Search| [Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123460035.pdf)/[Code](https://github.com/mikelzc1990/nsganetv2)|
S2DNAS | ECCV | Transforming Static CNN Model for Dynamic Inference via Neural Architecture Search| [Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123470171.pdf)/Code|
UnNAS | ECCV | Are Labels Necessary for Neural Architecture Search?| [Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123470171.pdf)/[Code](https://github.com/facebookresearch/unnas)|
E2GAN | ECCV | Off-Policy Reinforcement Learning for Efficient and Effective GAN Architecture Search| [Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123520171.pdf)/[Code](https://github.com/Yuantian013/E2GAN)|
BigNAS | ECCV | Scaling Up Neural Architecture Search with Big Single-Stage Models| [Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123520681.pdf)/[Code](https://github.com/xfey/pytorch-BigNAS)|
BPNAS | ECCV | Search What You Want: Barrier Panelty NAS for Mixed Precision Quantization| [Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123540001.pdf)/Code|
TFNAS | ECCV | Rethinking Three Search Freedoms of Latency-Constrained Differentiable Neural Architecture Search| [Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123600120.pdf)/[Code](https://github.com/AberHu/TF-NAS)|
FairDarts | ECCV | Eliminating Unfair Advantages in Differentiable Architecture Search| [Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123600460.pdf)/[Code](https://github.com/xiaomi-automl/FairDARTS)|
SingleNAS | ECCV | Single Path One-Shot Neural Architecture Search with Uniform Sampling| [Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123610528.pdf)/[Code](https://github.com/ShunLu91/Single-Path-One-Shot-NAS)|
DFA | ECCV | Differentiable Feature Aggregation Search for Knowledge Distillation| [Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123620460.pdf)/Code|
ABS | ECCV | Angle-based Search Space Shrinking for Neural Architecture Search| [Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123640120.pdf)/Code
|FAD | ECCV | Representation Sharing for Fast Object Detector Search and Beyond| [Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123640460.pdf)/[Code](https://github.com/msight-tech/research-fad)|
|BATS | ECCV | Binary ArchitecTure Search| [Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123680307.pdf)/Code|
|DANAS | ECCV | Data Adapted Pruning for Efficient Neural Architecture Search| [Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123720579.pdf)/Code|
|NPNAS | ECCV | Neural Predictor for Neural Architecture Search| [Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123740647.pdf)/[Code](https://github.com/ultmaster/neuralpredictor.pytorch)|
|CreamNAS | NeurIPS | Cream of the Crop: Distilling Prioritized Paths For One-Shot Neural Architecture Search| [Paper](https://proceedings.neurips.cc/paper/2020/file/d072677d210ac4c03ba046120f0802ec-Paper.pdf)/[Code](https://github.com/microsoft/cream.git)|
|ISTA-NAS | NeurIPS | Efficient and Consistent Neural Architecture Search by Sparse Coding| [Paper](https://proceedings.neurips.cc/paper/2020/file/76cf99d3614e23eabab16fb27e944bf9-Paper.pdf)/[Code](https://github.com/iboing/ISTA-NAS)|
|E2NAS | NeurIPS | Differentiable Neural Architecture Search in Equivalent Space with Exploration Enhancement| [Paper](https://proceedings.neurips.cc/paper/2020/file/9a96a2c73c0d477ff2a6da3bf538f4f4-Paper.pdf)/Code|
|SemiNAS | NeurIPS | Semi-Supervised Neural Architecture Search| [Paper](https://proceedings.neurips.cc/paper/2020/file/77305c2f862ad1d353f55bf38e5a5183-Paper.pdf)/[Code](https://github.com/Zumbalamambo/SemiNAS)|
|Arch2vec | NeurIPS | Does Unsupervised Architecture Representation Learning Help Neural Architecture Search?| [Paper](https://proceedings.neurips.cc/paper/2020/file/937936029af671cf479fa893db91cbdd-Paper.pdf)/[Code](https://github.com/MSU-MLSys-Lab/arch2vec)|
|BONAS | NeurIPS |  Bridging the Gap between Sample-based and One-shot Neural Architecture Search with BONAS| [Paper](https://proceedings.neurips.cc/paper/2020/file/13d4635deccc230c944e4ff6e03404b5-Paper.pdf)/[Code](https://github.com/pipilurj/BONAS)|
### **<h5 id='Seg20'>Image Segmentation</h5>**
| Name | Pub. | Title | Links |
| --- | --- | --- | --- |
| GAS | CVPR | Graph-guided Architecture Search for Real-time Semantic Segmentation | [Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Lin_Graph-Guided_Architecture_Search_for_Real-Time_Semantic_Segmentation_CVPR_2020_paper.pdf)/Code |
| FasterSeg | ICLR | Searching for Faster Real-time Semantic Segmentation| [Paper](https://openreview.net/pdf?id=BJgqQ6NYvB)/[Code](https://github.com/TAMU-=VITA/FasterSeg) |
| Auto-Panoptic | NeurIPS | Cooperative Multi-Component Architecture Search for Panoptic Segmentation | [Paper](https://proceedings.neurips.cc/paper/2020/file/ec1f764517b7ffb52057af6df18142b7-Paper.pdf)/[Code](https://github.com/Jacobew/AutoPanoptic) |
### **<h5 id='VC20'>Video Classification</h5>**
| Name | Pub. | Title | Links |
| --- | --- | --- | --- |
| AttentionNAS | ECCV | Spatiotemporal Attention Cell Search for Video Classification | [Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123530443.pdf)/Code |
| AssembleNet | ICLR | Searching for Multi-Stream Neural Connectivity in Video Architectures | [Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123530443.pdf)/[Code](https://github.com/leaderj1001/AssembleNet) |

### **<h5 id='LD20'>Lane Detection</h5>**
| Name | Pub. | Title | Links |
| --- | --- | --- | --- |
| CurveLaneNAS | ECCV | Unifying Lane-Sensitive Architecture Search and Adaptive Point Blending | [Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123600681.pdf)/Code |

### **<h5 id='ISR20'>Image Super-Resolution</h5>**
| Name | Pub. | Title | Links |
| --- | --- | --- | --- |
| ESRNAS | AAAI | Efficient Residual Dense Block Search for Image Super-Resolution| [Paper](https://arxiv.org/pdf/1909.11409.pdf)/Code|
### **<h5 id='IR20'>Image Restoration</h5>**
| Name | Pub. | Title | Links |
| --- | --- | --- | --- |
| CLEARER | NeurIPS | Multi-Scale Neural Architecture Search for Image Restoration| [Paper](https://proceedings.neurips.cc/paper/2020/file/c6e81542b125c36346d9167691b8bd09-Paper.pdf)/[Code](https://github.com/limit-scu)|

### **<h5 id='Med20'>Medical</h5>**
| Name | Pub. | Title | Links |
| --- | --- | --- | --- |
| C2FNAS | CVPR | Coarse-to-Fine Neural Architecture Search for 3D Medical Image Segmentation | [Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yu_C2FNAS_Coarse-to-Fine_Neural_Architecture_Search_for_3D_Medical_Image_Segmentation_CVPR_2020_paper.pdf)/Code  |

### **<h5 id='3HPE20'>3D Pose Estimation</h5>**
| Name | Pub. | Title | Links |
| --- | --- | --- | --- |
| PANAS | ECCV | Towards Part-aware Monocular 3D Human Pose Estimation: An Architecture Search Approach | [Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123480715.pdf)/Code  |

### **<h5 id='FD20'>Face Detection</h5>**
| Name | Pub. | Title | Links |
| --- | --- | --- | --- |
| BFBox | CVPR | BFBox Searching Face-Appropriate Backbone and Feature Pyramid Network for Face | [Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Liu_BFBox_Searching_Face-Appropriate_Backbone_and_Feature_Pyramid_Network_for_Face_CVPR_2020_paper.pdf)/[Code](https://github.com/ZitongYu/CDCN) |

### **<h5 id='FAS20'>Face Anti-Spoofing</h5>**
| Name | Pub. | Title | Links |
| --- | --- | --- | --- |
| CDCNAS | CVPR | Searching Central Difference Convolutional Networks for Face Anti-Spoofing| [Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yu_Searching_Central_Difference_Convolutional_Networks_for_Face_Anti-Spoofing_CVPR_2020_paper.pdf)/Code |

### **<h5 id='STR20'>Scene Text Recognition</h5>**
| Name | Pub. | Title | Links |
| --- | --- | --- | --- |
| AutoSTR | ECCV | Efficient Backbone Search for Scene Text Recognition | [Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123530443.pdf)/[Code](https://github.com/AutoML-Research/AutoSTR)|

### **<h5 id='AA20'>Adversarial Attack</h5>**
| Name | Pub. | Title | Links |
| --- | --- | --- | --- |
| RobNets | CVPR | When NAS Meets Robustness: In Search of Robust Architectures against Adversarial Attacks | [Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Guo_When_NAS_Meets_Robustness_In_Search_of_Robust_Architectures_Against_CVPR_2020_paper.pdf)/[Code](https://github.com/gmh14/RobNets)  |

### **<h5 id='MD20'>Model Defense</h5>**
| Name | Pub. | Title | Links |
| --- | --- | --- | --- |
| ABanditNAS | ECCV | Anti-Bandit Neural Architecture Search for Model Defense | [Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123580069.pdf)/Code |

### **<h5 id='BWR20'>Bad Weather Removal</h5>**
| Name | Pub. | Title | Links |
| --- | --- | --- | --- |
| AIOnet | CVPR | All in One Bad Weather Removal using Architectural Search | [Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/LGuo_When_NAS_Meets_Robustness_In_Search_of_Robust_Architectures_Against_CVPR_2020_paper.pdf)/Code|

### **<h5 id='ID20'>Image Denoising</h5>**
| Name | Pub. | Title | Links |
| --- | --- | --- | --- |
| HiNAS | CVPR | Memory-Efficient Hierarchical Neural Architecture Search for Image Denoising | [Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhang_Memory-Efficient_Hierarchical_Neural_Architecture_Search_for_Image_Denoising_CVPR_2020_paper.pdf)/Code|

### **<h5 id='DIP20'>Deep Image Prior</h5>**
| Name | Pub. | Title | Links |
| --- | --- | --- | --- |
| NASDIP | ECCV | Learning Deep Image Prior with Neural Architecture Search| [Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123630426.pdf)/[Code](https://github.com/YunChunChen/NAS-DIP-pytorch)|

### **<h5 id='SM20'>Stereo Matching</h5>**
| Name | Pub. | Title | Links |
| --- | --- | --- | --- |
| HNAS | NeurIPS | Hierarchical Neural Architecture Search for Deep Stereo Matching| [Paper](https://proceedings.neurips.cc/paper/2020/file/fc146be0b230d7e0a92e66a6114b840d-Paper.pdf)/[Code](https://github.com/XuelianCheng/LEAStereo)|
### **<h5 id='CC20'>Crowd Counting</h5>**
| Name | Pub. | Title | Links |
| --- | --- | --- | --- |
| NASCount | ECCV |Counting-by-Density with Neural Architecture Search| [Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123670749.pdf)/Code|

### **<h5 id='MP20'>Model Pruning</h5>**
| Name | Pub. | Title | Links |
| --- | --- | --- | --- |
| NASFCOS | CVPR | Fast Neural Architecture Search for Object Detection | [Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_NAS-FCOS_Fast_Neural_Architecture_Search_for_Object_Detection_CVPR_2020_paper.pdf)/[Code](https://github.com/Lausannen/NAS-FCOS)
| BPE | CVPR | Rethinking Performance Estimation in Neural Architecture Search | [Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_NAS-FCOS_Fast_Neural_Architecture_Search_for_Object_Detection_CVPR_2020_paper.pdf)/[Code](https://github.com/CVPR2020-ID1073/Rethinking-Performance-Estimation-in-Neural-Architecture-Search)|
| BlockSwap | ICLR | Fisher-guided Block Substitution for Network Compression on a Budget| [Paper](https://openreview.net/pdf?id=SklkDkSFPB)/[Code](https://github.com/BayesWatch/pytorch-blockswap) |

### **<h5 id='TR20'>Text Representation</h5>**
| Name | Pub. | Title | Links |
| --- | --- | --- | --- |
| TextNAS | AAAI | A Neural Architecture Search Space tailored for Text Representation| [Paper](https://arxiv.org/pdf/1912.10729.pdf)/[Code](https://github.com/microsoft/TextNAS)|
- [Back to content](#Content)

# 2019
### **<h5 id='OC19'>Object Classfication/Detection</h5>**
| Name | Pub. | Title | Links |
| --- | --- | --- | --- |
| AENAS | AAAI | Aging Evolution for Image Classifier Architecture Search | [Paper](https://www.cse.fau.edu/~xqzhu/courses/cap6619/aging.evolution.pdf)/Code|
| SNAS | ICLR | Stochastic Neural Architecture Search | [Paper](https://openreview.net/pdf?id=rylqooRqK7)/[Code](https://github.com/Astrodyn94/SNAS-Stochastic-Neural-Architecture-Search-)|
| ProxylessNAS | ICLR | ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware  | [Paper](https://openreview.net/pdf?id=HylVB3AqYm)/[Code]( https://github.com/MIT-HAN-LAB/ProxylessNAS)|
| DARTS | ICLR | Differentiable Architecture Search   | [Paper](https://openreview.net/pdf?id=S1eYHoC5FX)/[Code]( https://github.com/quark0/darts)|
| GDAS | CVPR | Searching for A Robust Neural Architecture in Four GPU Hours | [Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Dong_Searching_for_a_Robust_Neural_Architecture_in_Four_GPU_Hours_CVPR_2019_paper.pdf)/[Code](https://github.com/D-X-Y/GDAS)|
| FBNet | CVPR | Hardware-Aware Efficient ConvNet Design via Differentiable Neural Architecture Search | [Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wu_FBNet_Hardware-Aware_Efficient_ConvNet_Design_via_Differentiable_Neural_Architecture_Search_CVPR_2019_paper.pdf)/[Code](https://github.com/facebookresearch/mobile-vision)|
| MnasNet | CVPR | Platform-Aware Neural Architecture Search for Mobile | [Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Tan_MnasNet_Platform-Aware_Neural_Architecture_Search_for_Mobile_CVPR_2019_paper.pdf)/[Code](https://github.com/tensorflow/tpu/tree/master/models/official/mnasnet)|
| RENAS | CVPR | Reinforced Evolutionary Neural Architecture Search | [Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Chen_RENAS_Reinforced_Evolutionary_Neural_Architecture_Search_CVPR_2019_paper.pdf)/[Code](https://github.com/yukang2017/RENAS)|
| IRLAS | CVPR | Inverse Reinforcement Learning for Architecture Search | [Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Guo_IRLAS_Inverse_Reinforcement_Learning_for_Architecture_Search_CVPR_2019_paper.pdf)/Code|
| EIGEN | CVPR | Ecologically-Inspired GENetic Approach for Neural Network Structure Searching from Scratch | [Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Ren_EIGEN_Ecologically-Inspired_GENetic_Approach_for_Neural_Network_Structure_Searching_From_CVPR_2019_paper.pdf)/Code|
| HMNAS | ICCVW | Efficient Neural Architecture Search via Hierarchical Masking | [Paper](https://openaccess.thecvf.com/content_ICCVW_2019/papers/NeurArch/Yan_HM-NAS_Efficient_Neural_Architecture_Search_via_Hierarchical_Masking_ICCVW_2019_paper.pdf)/Code|
| PDarts | ICCV | Progressive Differentiable Architecture Search: Bridging the Depth Gap between Search and Evaluation| [Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Chen_Progressive_Differentiable_Architecture_Search_Bridging_the_Depth_Gap_Between_Search_ICCV_2019_paper.pdf)/[Code](https://github.com/chenxin061/pdarts)|
| MDENAS | ICCV | Multinomial Distribution Learning for Effective Neural Architecture Search| [Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Zheng_Multinomial_Distribution_Learning_for_Effective_Neural_Architecture_Search_ICCV_2019_paper.pdf)/[Code](https://github.com/tanglang96/MDENAS)|
| MobileNetV3 | ICCV | Searching for MobileNetV3| [Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Howard_Searching_for_MobileNetV3_ICCV_2019_paper.pdf)/[Code](https://github.com/leaderj1001/MobileNetV3-Pytorch)|
| RCNet | ICCV | Resource Constrained Neural Network Architecture Search: Will a Submodularity Assumption Help?| [Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Xiong_Resource_Constrained_Neural_Network_Architecture_Search_Will_a_Submodularity_Assumption_ICCV_2019_paper.pdf)/[Code](https://github.com/yyxiongzju/RCNet)|
| AutoGAN | ICCV | Neural Architecture Search for Generative Adversarial Networks| [Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Gong_AutoGAN_Neural_Architecture_Search_for_Generative_Adversarial_Networks_ICCV_2019_paper.pdf)/[Code](https://github.com/VITA-Group/AutoGAN)|
| SETN | ICCV | One-Shot Neural Architecture Search via Self-Evaluated Template Network| [Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Dong_One-Shot_Neural_Architecture_Search_via_Self-Evaluated_Template_Network_ICCV_2019_paper.pdf)/Code|
| FPNAS | ICCV | Fast and Practical Neural Architecture Search| [Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Dong_One-Shot_Neural_Architecture_Search_via_Self-Evaluated_Template_Network_ICCV_2019_paper.pdf)/[Code](https://github.com/jiequancui/FPNASNet)|
| Auto-FPN | ICCV | Automatic Network Architecture Adaptation for Object Detection Beyond Classification| [Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Xu_Auto-FPN_Automatic_Network_Architecture_Adaptation_for_Object_Detection_Beyond_Classification_ICCV_2019_paper.pdf)/Code|
| SNAS | MICCAI | Scalable Neural Architecture Search for 3D Medical Image Segmentation| [Paper](https://arxiv.org/pdf/1906.05956.pdf)/Code|
| XNAS | NeurIPS | Neural Architecture Search with Expert Advice| [Paper](https://papers.nips.cc/paper/2019/file/00e26af6ac3b1c1c49d7c3d79c60d000-Paper.pdf)/Code|
| DetNAS | NeurIPS | Backbone Search for Object Detection| [Paper](https://papers.nips.cc/paper/2019/file/228b25587479f2fc7570428e8bcbabdc-Paper.pdf)/[Code](https://github.com/megvii-model/DetNAS)|
| NATS | NeurIPS | Efficient Neural Architecture Transformation Search in Channel-Level for Object Detection| [Paper](https://papers.nips.cc/paper/2019/file/3aaa3db6a8983226601cac5dde15a26b-Paper.pdf)/Code|
| EFAS | NeurIPS | Efficient Forward Architecture Search| [Paper](https://papers.nips.cc/paper/2019/file/6c468ec5a41d65815de23ec1d08d7951-Paper.pdf)/[Code](https://github.com/microsoft/petridishnn)|

### **<h5 id='Seg19'>Image Segmentation</h5>**
| Name | Pub. | Title | Links |
| --- | --- | --- | --- |
| FastNAS | CVPR | Fast Neural Architecture Search of Compact Semantic Segmentation Models via Auxiliary Cells | [Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Nekrasov_Fast_Neural_Architecture_Search_of_Compact_Semantic_Segmentation_Models_via_CVPR_2019_paper.pdf)/[Code](https://github.com/DrSleep/nas-segm-pytorch)|
| DFNet | CVPR | Partial Order Pruning: for Best Speed/Accuracy Trade-off in Neural Architecture Search | [Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Partial_Order_Pruning_For_Best_SpeedAccuracy_Trade-Off_in_Neural_Architecture_CVPR_2019_paper.pdf)/[Code](https://github.com/lixincn2015/Partial-Order-Pruning)|
AutoDeepLab | CVPR | Hierarchical Neural Architecture Search for Semantic Image Segmentation | [Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_Auto-DeepLab_Hierarchical_Neural_Architecture_Search_for_Semantic_Image_Segmentation_CVPR_2019_paper.pdf)/[Code](https://github.com/tensorflow/models/tree/master/research/deeplab)|
CASNet | CVPR | Customizable Architecture Search for Semantic Segmentation | [Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhang_Customizable_Architecture_Search_for_Semantic_Segmentation_CVPR_2019_paper.pdf)/Code|

### **<h5 id='ReID19'>Re-Identification/Detection</h5>**
| Name | Pub. | Title | Links |
| --- | --- | --- | --- |
| Auto-ReID | ICCV | Searching for a Part-Aware ConvNet for Person Re-Identification | [Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Quan_Auto-ReID_Searching_for_a_Part-Aware_ConvNet_for_Person_Re-Identification_ICCV_2019_paper.pdf)/Code  |

### **<h5 id='SM19'>Stereo Matching</h5>**
| Name | Pub. | Title | Links |
| --- | --- | --- | --- |
| AutoDispNet | ICCV | AutoDispNet: Improving Disparity Estimation With AutoML| [Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Saikia_AutoDispNet_Improving_Disparity_Estimation_With_AutoML_ICCV_2019_paper.pdf)/Code|

### **<h5 id='MF19'>Multimodal Fusion</h5>**
| Name | Pub. | Title | Links |
| --- | --- | --- | --- |
MFAS | CVPR | Multimodal Fusion Architecture Search | [Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Perez-Rua_MFAS_Multimodal_Fusion_Architecture_Search_CVPR_2019_paper.pdf)/[Code](https://github.com/jperezrua/mfas)|

### **<h5 id='MP19'>Model Pruning</h5>**
| Name | Pub. | Title | Links |
| --- | --- | --- | --- |
| NPTAS | NeurIPS | Network Pruning via Transformable Architecture Search| [Paper](https://papers.nips.cc/paper/2019/file/6c468ec5a41d65815de23ec1d08d7951-Paper.pdf)/[Code](https://github.com/D-X-Y/NAS-Projects)|
- [Back to content](#Content)

# 2018
### **<h5 id='OC18'>Object Classfication/Detection</h5>**
| Name | Pub. | Title | Links |
| --- | --- | --- | --- |
NASNet | CVPR | Learning Transferable Architectures for Scalable Image Recognition | [Paper](https://arxiv.org/pdf/1707.07012.pdf)/[Code](https://github.com/aussetg/nasnet.pytorch)|
BlockQNN | CVPR | Practical Block-wise Neural Network Architecture Generation | [Paper](https://arxiv.org/pdf/1708.05552.pdf)/Code|
PNAS | ECCV | Progressive Neural Architecture Search | [Paper](https://arxiv.org/pdf/1712.00559.pdf)/[Code](https://github.com/titu1994/progressive-neural-architecture-search)|
DPP-Net | ECCV | Device-aware Progressive Search for Pareto-optimal Neural Architectures | [Paper](https://openaccess.thecvf.com/content_ECCV_2018/papers/Jin-Dong_Dong_DPP-Net_Device-aware_Progressive_ECCV_2018_paper.pdf)/Code|
HEAS| ICLR | Hierarchical Representations for Efficient Architecture Search | [Paper](https://openreview.net/pdf?id=BJQRKzbA-)/Code|
- [Back to content](#Content)
**<h5 id='Sur'></h5>**
# Survey
| Pub. | Title | Links |
| --- | --- | --- |
| ICLR20 | NAS evaluation is frustratingly hard| [Paper](https://iclr.cc/virtual_2020/poster_H1gDNyrKDS.html)/[Code](https://github.com/antoyang/NAS-Benchmark) |
| ICLR20 | Understanding and Robustifying Differentiable Architecture Search| [Paper](https://iclr.cc/virtual_2020/poster_H1gDNyrKDS.html)/Code |
| ICLR20 | Understanding Architectures Learnt by Cell-based Neural Architecture Search| [Paper](https://openreview.net/pdf?id=BJxH22EKPS)/Code |
| ICLR22 | On Redundancy and Diversity in Cell-based Neural Architecture Search | [Paper](https://openreview.net/pdf?id=rFJWoYoxrDB)/[Code](https://github.com/xingchenwan/cell-based-NAS-analysis)|
| NeurIPS20 | A Study on Encodings for Neural Architecture Search | [Paper](https://proceedings.neurips.cc/paper/2020/file/ea4eb49329550caaa1d2044105223721-Paper.pdf)/[Code](https://github.com/naszilla/naszilla)|
