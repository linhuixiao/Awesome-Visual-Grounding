[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)
[![PR's Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat)](https://github.com/linhuixiao/Awesome-Visual-Grounding/pulls)
<br />
<p align="center">
  <h1 align="center">Towards Visual Grounding: A Survey</h1>
  <p align="center">
    <b> TPAMI under review, 2024 </b>
    <br />
    <a href="https://scholar.google.com.hk/citations?user=4rTE4ogAAAAJ&hl=zh-CN&oi=sra"><strong> Linhui Xiao </strong></a>
    ·
    <a href="https://yangxs.ac.cn/home"><strong>Xiaoshan Yang </strong></a>
    ·
    <a href="https://scholar.google.com.hk/citations?user=c3iwWRcAAAAJ&hl=zh-CN&oi=ao"><strong>Xiangyuan Lan </strong></a>
    ·
    <a href="https://scholar.google.com.hk/citations?user=o_DllmIAAAAJ&hl=zh-CN"><strong>Yaowei Wang </strong></a>
    ·
    <a href="https://scholar.google.com.hk/citations?user=hI9NRDkAAAAJ&hl=zh-CN"><strong>Changsheng Xu</strong></a>
  </p>
  <p align="center">
    <a href='https://arxiv.org/abs/2412.20206'>
      <img src='https://img.shields.io/badge/arXiv-PDF-green?style=flat&logo=arXiv&logoColor=green' alt='arXiv PDF'>
    </a>
<br />

[//]: # "    <a href='https://ieeexplore.ieee.org/document/10420487'>"
[//]: # "      <img src='https://img.shields.io/badge/TPAMI-PDF-blue?style=flat&logo=IEEE&logoColor=green' alt='TPAMI PDF'>"
[//]: # "    </a>"
[//]: # "  </p>"

<p align="center"> <img src='figs/illustration.jpg' align="center" width="60%"> </p>

**<p align="center"> An Illustration of Visual Grounding  </p>**


<p align="center"> <img src='figs/development_trends_h.jpg' align="center" width="100%"> </p>

**<p align="center"> A Decade of Visual Grounding </p>**

This repo is used for recording, tracking, and benchmarking several recent visual grounding methods to supplement our [Grounding Survey](https://arxiv.org/pdf/2412.20206). 

### 🔥 Add Your Paper in our Repo and Survey!

- If you find any work missing or have any suggestions (papers, implementations, and other resources), feel free to [pull requests](https://github.com/linhuixiao/Awesome-Visual-Grounding/pulls).
We will add the missing papers to this repo as soon as possible.

- You are welcome to give us an issue or PR (pull request) for your visual grounding related works!

- **Note that:** Due to the huge paper in Arxiv, we are sorry to cover all in our survey. You can directly present a PR into this repo and we will record it for next version update of our survey.




### 🔥 New

[//]: # "- [ ] Next version of our survey will be updated in:."

- 🔥 **We made our paper public and created this repository on** **December 26, 2024**.



### 🔥 Highlight!!

- A comprehensive survey for Visual Grounding, including Referring Expression Comprehension and Phrase Grounding.

- It includes the newly concepts, such as Grounding Multi-modal LLMs, Generalized Visual Grounding, and VLP-based grounding transfer works. 

- We list detailed results for the most representative works and give a fairer and clearer comparison of different approaches.

- We provide a list of future research insights.



# Introduction

**we are the first survey in the past five years to systematically track and summarize the development of visual 
grounding over the last decade.** By extracting common technical details, this review encompasses the most representative
work in each subtopic. 

**This survey is also currently the most comprehensive review in the field of visual grounding.** We aim for this article 
to serve as a valuable resource not only for beginners seeking an introduction to grounding but also for researchers 
with an established foundation, enabling them to navigate and stay up-to-date with the latest advancements.


<p align="center"> <img src='figs/timeline.jpg' align="center" width="100%"> </p>

**<p align="center"> A Decade of Visual Grounding  </p>**

<p align="center"> <img src='figs/setting.jpg' align="center" width="100%"> </p>

**<p align="center"> Mainstream Settings in Visual Grounding  </p>**

<p align="center"> <img src='figs/architecture.jpg' align="center" width="100%"> </p>

**<p align="center">  Typical Framework Architectures for Visual Grounding  </p>**

<p align="center"> <img src='figs/paper_structure.jpg' align="center" width="100%"> </p>

**<p align="center"> Our Paper Structure  </p>** 


## Citation

If you find our work helpful for your research, please consider citing the following BibTeX entry.   

```bibtex
@misc{xiao2024visualgroundingsurvey,
      title={Towards Visual Grounding: A Survey}, 
      author={Linhui Xiao and Xiaoshan Yang and Xiangyuan Lan and Yaowei Wang and Changsheng Xu},
      year={2024},
      eprint={2412.20206},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.20206}, 
}
```
**It should be noted that**, due to the typesetting restrictions of the journal, there are small differences in the 
typesetting between the Arxiv version and review version.

**The following will be the relevant grounding papers and associated code links in this paper:**

# Summary of Contents
This content corresponds to the main text.


[//]: # "超链接的语法：默认:,.等符号直接忽略，空格“ ”用“-”代替，“-”继续用“-”"

- [Introduction](#introduction)
  - [Citation](#citation)
- [Summary of Contents](#summary-of-contents)
- [1. Methods: A Survey](#1-methods-a-survey)
  - [1.1 Fully Supervised Setting](#11-fully-supervised-setting)
    - [A. Traditional CNN-based Methods](#a-traditional-cnn-based-methods)
    - [B. Transformer-based Methods](#b-transformer-based-methods)
    - [C. VLP-based Methods](#c-vlp-based-methods)
    - [D. Grounding-oriented Pre-training](#d-grounding-oriented-pre-training)
    - [E. Grounding Multimodal LLMs](#e-grounding-multimodal-llms)
  - [1.2 Weakly Supervised Setting](#12-weakly-supervised-setting)
  - [1.2 Semi-supervised Setting](#12-semi-supervised-setting)
  - [1.3 Unsupervised Setting](#13-unsupervised-setting)
  - [1.4 Zero-shot Setting](#14-zero-shot-setting)
  - [1.5 Multi-task Setting](#15-multi-task-setting)
    - [A. REC with REG Multi-task Setting](#a-rec-with-reg-multi-task-setting)
    - [B. REC with REG Multi-task Setting](#b-rec-with-reg-multi-task-setting)
    - [C. Other Multi-task Setting](#c-other-multi-task-setting)
  - [1.6 Generalized Visual Grounding](#16-generalized-visual-grounding)
- [2. Advanced Topics](#2-advanced-topics)
  - [2.1 NLP Language Structure Parsing](#21-nlp-language-structure-parsing)
  - [2.2 Spatial Relation and Graph Networks](#22-spatial-relation-and-graph-networks)
  - [2.3 Modular Grounding](#23-modular-grounding)
- [3. Applications](#3-applications)
  - [3.1 Grounded Object Detection](#31-grounded-object-detection)
  - [3.2 Referring Counting](#32-referring-counting)
  - [3.3 Remote Sensing Visual Grounding](#33-remote-sensing-visual-grounding)
  - [3.4 Medical Visual Grounding](#34-medical-visual-grounding)
  - [3.5 3D Visual Grounding](#35-3d-visual-grounding)
  - [3.6 Video Object Grounding](#36-video-object-grounding)
  - [3.7 Robotic and Multimodal Agent Applications](#37-robotic-and-multimodal-agent-applications)
- [4. Datasets and Benchmarks](#4-datasets-and-benchmarks)
  - [3.1 The Five Datasets for Classical Visual Grounding](#31-the-five-datasets-for-classical-visual-grounding)
  - [3.2 The Other Datasets for Classical Visual Grounding](#32-the-other-datasets-for-classical-visual-grounding)
  - [3.3 Dataset for the Newly Curated Scenarios](#33-dataset-for-the-newly-curated-scenarios)
    - [A. Dataset for Generalized Visual Grounding](#a-dataset-for-generalized-visual-grounding)
    - [B. Datasets and Benchmarks for GMLLMs](#b-datasets-and-benchmarks-for-gmllms)
    - [C. Dataset for Other Newly Curated Scenarios](#c-dataset-for-other-newly-curated-scenarios)
- [5. Challenges And Outlook](#5-challenges-and-outlook)
- [6. Other Valuable Survey and Project](#6-other-valuable-survey-and-project)
- [Acknowledgement](#acknowledgement)
- [Contact](#contact)
  - [Star History](#star-history)




# 1. Methods: A Survey




## 1.1 Fully Supervised Setting

### A. Traditional CNN-based Methods

| Year | Venue | Work Name | Paper Title / Paper Link                                     | Code / Project                                           |
| ---- | ----- | --------- | ------------------------------------------------------------ | -------------------------------------------------------- |
| 2016 | CVPR  | NMI       | [**Generation and Comprehension of Unambiguous Object Descriptions**](https://openaccess.thecvf.com/content_cvpr_2016/papers/Mao_Generation_and_Comprehension_CVPR_2016_paper.pdf) | [Code](https://github.com/mjhucla/Google_Refexp_toolbox) |
|      |       |           |                                                              | Project                                                  |
|      |       |           |                                                              |                                                          |



### B. Transformer-based Methods


| Year | Venue | Work Name | Paper Title / Paper Link                                                                                                                                                                         | Code / Project                                 |
|------|-------|-----------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------|
| 2021 | ICCV  | TransVG   | [**Transvg: End-to-end Visual Grounding with Transformers**](http://openaccess.thecvf.com/content/ICCV2021/html/Deng_TransVG_End-to-End_Visual_Grounding_With_Transformers_ICCV_2021_paper.html) | [Code](https://github.com/djiajunustc/TransVG) |
|      |       |           |                                                                                                                                                                                                  |                                                |

### C. VLP-based Methods

| Year | Venue | Name    | Paper Title / Paper Link                                                                                     | Code / Project                                |
|------|-------|---------|--------------------------------------------------------------------------------------------------------------|-----------------------------------------------|
| 2023 | TMM   | CLIP-VG | [**CLIP-VG: Self-paced Curriculum Adapting of CLIP for Visual Grounding**](https://arxiv.org/pdf/2305.08685) | [Code](https://github.com/linhuixiao/CLIP-VG) |
|  2023    |   TPAMI    |    D-MDETR       |      [**Dynamic MDETR: A Dynamic Multimodal Transformer Decoder for Visual Grounding**](https://arxiv.org/pdf/2209.13959)                                                                                                                                                                                            |                 [Code](https://github.com/MCG-NJU/Dynamic-MDETR)                               |
|      |       |         |                                                                                                              |                                               |
|      |       |         |                                                                                                              |                                               |

### D. Grounding-oriented Pre-training

| Year | Venue | Name  | Paper Title / Paper Link                                                                                                         | Code / Project                             |
|------|-------|-------|----------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------|
| 2021 | ICCV  | MDETR | [**Transvg: End-to-end Visual Grounding with Transformers**](https://openaccess.thecvf.com/content/ICCV2021/papers/Deng_TransVG_End-to-End_Visual_Grounding_With_Transformers_ICCV_2021_paper.pdf) | [Code](https://github.com/ashkamath/mdetr) |
|  2022    |   ICML    |   OFA    | [**OFA: Unifying Architectures, Tasks, and Modalities Through a Simple Sequence-to-Sequence Learning Framework**](https://proceedings.mlr.press/v162/wang22al/wang22al.pdf) |   [Code](https://github.com/OFA-Sys/OFA) |   
|   2022   |    ECCV   |   UniTAB    |     [**UniTAB: Unifying Text and Box Outputs for Grounded Vision-Language Modeling**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136960514.pdf)                                                                                                                             |         [Code](https://github.com/microsoft/UniTAB)                                   |

### E. Grounding Multimodal LLMs

| Year | Venue | Name   | Paper Title / Paper Link                                                                               | Code / Project                            |
|------|-------|--------|--------------------------------------------------------------------------------------------------------|-------------------------------------------|
| 2023 | Arxiv | Shikra | [**Shikra: Unleashing Multimodal LLM's Referential Dialogue Magic**](https://arxiv.org/pdf/2306.15195) | [Code](https://github.com/shikras/shikra) | 
|   2022   |  NeurIPS     |   Chinchilla     |   [**Training Compute-Optimal Large Language Models**](https://proceedings.neurips.cc/paper_files/paper/2022/file/c1e2faff6f588870935f114ebe04a3e5-Paper-Conference.pdf)                                                                                                     |        N/A                                   |
|   2019   |    OpenAI   |    GPT-2    |    [**Language Models are Unsupervised Multitask Learners**](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)                                                                                                    |   N/A                                        |
|   2020   |    NeurIPS   |    GPT-3    |    [**Language Models are Few-Shot Learners**](https://papers.nips.cc/paper_files/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf)                                                                                                    |   N/A                                        |
|   2024   |    ICLR   |    Ferret    |    [**Ferret: Refer And Ground Anything Anywhere At Any Granularity**](https://openreview.net/pdf?id=2msbbX3ydD)                                                                                                    |   [Code](https://github.com/apple/ml-ferret)                                        |
|   2024   |    CVPR   |    LION    |    [**LION: Empowering Multimodal Large Language Model With Dual-Level Visual Knowledge**](https://openaccess.thecvf.com/content/CVPR2024/papers/Chen_LION_Empowering_Multimodal_Large_Language_Model_with_Dual-Level_Visual_Knowledge_CVPR_2024_paper.pdf)                                                                                                    |   [Code](https://github.com/JiuTian-VL/JiuTian-LION)                                        |
|   2022   |    ECCV   |    YORO    |    [**YORO - Lightweight End to End Visual Grounding**](https://arxiv.org/pdf/2211.07912)                                                                                                    |   [Code](https://github.com/chihhuiho/yoro)                                        |


## 1.2 Weakly Supervised Setting

| Year | Venue | Name    | Paper Title / Paper Link                                                                     | Code / Project |
|------|-------|---------|----------------------------------------------------------------------------------------------|----------------|
| 2016 | ECCV  | GroundR | [**Grounding of Textual Phrases in Images by Reconstruction**](https://arxiv.org/pdf/1511.03745) | N/A            | 
|  2017    |   CVPR    |    N/A     |      [**Weakly-supervised Visual Grounding of Phrases with Linguistic Structures**](https://openaccess.thecvf.com/content_cvpr_2017/papers/Xiao_Weakly-Supervised_Visual_Grounding_CVPR_2017_paper.pdf)                                                                                        |        N/A        |
|      |       |         |                                                                                              |                |


## 1.3 Semi-supervised Setting

| Year | Venue  | Name       | Paper Title / Paper Link                                                                                                                           | Code / Project |
|------|--------|------------|----------------------------------------------------------------------------------------------------------------------------------------------------|----------------|
| 2023 | ICASSP | PQG-Distil | [**Pseudo-Query Generation For Semi-Supervised Visual Grounding With Knowledge Distillation**](https://ieeexplore.ieee.org/abstract/document/10095558) | N/A            | 
|  2021    |   WACV     |      LSEP      |     [**Utilizing Every Image Object for Semi-supervised Phrase Grounding**](https://openaccess.thecvf.com/content/WACV2021/papers/Zhu_Utilizing_Every_Image_Object_for_Semi-Supervised_Phrase_Grounding_WACV_2021_paper.pdf)                                                                                                                                               |     N/A           |
|   2022   |   CRV     |      SS-Ground      |        [**Semi-supervised Grounding Alignment for Multi-modal Feature Learning**](https://www.cs.ubc.ca/~lsigal/Publications/chou2022crv.pdf)                                                                                                                                            |      N/A          |


## 1.4 Unsupervised Setting

| Year | Venue | Name     | Paper Title / Paper Link                                                                                                                                                                                              | Code / Project                                 |
|------|-------|----------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------|
| 2022 | CVPR  | Pseudo-Q | [**Pseudo-Q: Generating pseudo language queries for visual grounding**](https://openaccess.thecvf.com/content/CVPR2022/papers/Jiang_Pseudo-Q_Generating_Pseudo_Language_Queries_for_Visual_Grounding_CVPR_2022_paper.pdf) | [Code](https://github.com/LeapLabTHU/Pseudo-Q) | 
|   2018   |    CVPR   |    N/A      |        [**Unsupervised Textual Grounding: Linking Words to Image Concepts**](https://openaccess.thecvf.com/content_cvpr_2018/papers/Yeh_Unsupervised_Textual_Grounding_CVPR_2018_paper.pdf)                                                                                                                                                                                                               |                N/A                                |
| 2024 | ICME | VG-annotator | [**VG-Annotator: Vision-Language Models as Query Annotators for Unsupervised Visual Grounding**](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10688227) |    N/A      |
| 2023 | TMM | CLIPREC | [**CLIPREC: Graph-Based Domain Adaptive Network for Zero-Shot Referring Expression Comprehension**](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10191020) |    N/A      |





## 1.5 Zero-shot Setting

| Year | Venue | Name   | Paper Title / Paper Link | Code / Project |
|------|-------|--------|--------------------------|----------------|
| 2019 | ICCV  | ZSGNet | [**Zero-shot Grounding of Objects from Natural Language Queries**](https://openaccess.thecvf.com/content_ICCV_2019/papers/Sadhu_Zero-Shot_Grounding_of_Objects_From_Natural_Language_Queries_ICCV_2019_paper.pdf) | [Code](https://github.com/TheShadow29/zsgnet-pytorch) |
| 2022 | ACL   | ReCLIP | [**ReCLIP: A Strong Zero-shot Baseline for Referring Expression Comprehension**](https://arxiv.org/pdf/2204.05991)                                                                                                | [Code](https://www.github.com/allenai/reclip)         | 
|  2024    |    Neurocomputing   |   OV-VG     |   [**OV-VG: A Benchmark for Open-Vocabulary Visual Grounding**](https://arxiv.org/pdf/2310.14374)                                                                                                                                                                                                         |            [Code](https://github.com/cv516Buaa/OV-VG)                                              | 
| 2023 | TMM | CLIPREC | [**CLIPREC: Graph-Based Domain Adaptive Network for Zero-Shot Referring Expression Comprehension**](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10191020) |    N/A      |
| 2024 | Neurocomputing | N/A | [**Zero-shot visual grounding via coarse-to-fine representation learning**](https://www.sciencedirect.com/science/article/pii/S0925231224013924) |   [Code](https://github.com/TheShadow29/zsgnet-pytorch.git)    |
| 2022 | Arxiv | adapting-CLIP | [**Adapting CLIP For Phrase Localization Without Further Training**](https://arxiv.org/pdf/2204.03647) |   [Code](https://github.com/pals-ttic/adapting-CLIP)    |
| 2023 | ICLR | ChatRef | [**Language models can do zero-shot visual referring expression comprehension**](https://openreview.net/pdf?id=F7mdgA7c2zD) |   [Code](https://github.com/xiuchao/LLM4ReC)    |
| 2024 | AI Open | Cpt | [**CPT: Colorful Prompt Tuning for pre-trained vision-language models**](https://www.sciencedirect.com/science/article/pii/S2666651024000056) |   [Code](https://github.com/thunlp/CPT)    |
| 2021 | CVPR | VinVL | [**VinVL: Revisiting Visual Representations in Vision-Language Models**](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_VinVL_Revisiting_Visual_Representations_in_Vision-Language_Models_CVPR_2021_paper.pdf) |   [Code](https://github.com/pzzhang/VinVL)    |
| 2024 | CVPR | VR-VLA | [**Zero-shot referring expression comprehension via structural similarity between images and captions**](https://openaccess.thecvf.com/content/CVPR2024/papers/Han_Zero-shot_Referring_Expression_Comprehension_via_Structural_Similarity_Between_Images_and_CVPR_2024_paper.pdf) |   [Code](https://github.com/Show-han/Zeroshot_REC)    |
| 2024 | AAAI | GroundVLP | [**Groundvlp: Harnessing zeroshot visual grounding from vision-language pre-training and openvocabulary object detection**](https://ojs.aaai.org/index.php/AAAI/article/view/28278/28547) |   [Code](https://github.com/om-ai-lab/GroundVLP)    |
| 2024 | TCSVT | MCCE-REC | [**MCCE-REC: MLLM-driven Cross-modal Contrastive Entropy Model for Zero-shot Referring Expression Comprehension**](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10681116) |   N/A   |
| 2024 | ECCV | CRG | [**Contrastive Region Guidance: Improving Grounding in Vision-Language Models Without Training**](https://link.springer.com/chapter/10.1007/978-3-031-72986-7_12) |   [Code](https://contrastive-region-guidance.github.io/)   |
| 2024 | IJCNN | PSAIR | [**Psair: A neurosymbolic approach to zero-shot visual grounding**](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10649948) |   N/A   |
| 2024 | TPAMI | TransCP | [**Context disentangling and prototype inheriting for robust visual grounding**](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10649948) |  [Code](https://github.com/WayneTomas/TransCP)   |
| 2024 | TPAMI | N/A | [**Towards Open Vocabulary Learning: A Survey**](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10420487) |  [Code](https://github.com/jianzongwu/Awesome-Open-Vocabulary)   |
| 2024 | CVPR |  GEM | [**Grounding everything: Emerging localization properties in vision-language transformers**](https://openaccess.thecvf.com/content/CVPR2024/papers/Bousselham_Grounding_Everything_Emerging_Localization_Properties_in_Vision-Language_Transformers_CVPR_2024_paper.pdf) |  [Code](https://github.com/WalBouss/GEM)   |
| 2023 | Arxiv |  GRILL | [**Grill: Grounded vision-language pre-training via aligning text and image regions**](https://arxiv.org/pdf/2305.14676v1) | N/A  |




## 1.6 Multi-task Setting

### A. REC with REG Multi-task Setting
| Year | Venue | Name   | Paper Title / Paper Link | Code / Project |
|------|-------|--------|--------------------------|----------------|
| 2024 | Arxiv |  VLM-VG | [**Learning visual grounding from generative vision and language model**](https://arxiv.org/pdf/2407.14563) |  N/A   |
| 2024 | Arxiv |  EEVG | [**An efficient and effective transformer decoder-based framework for multi-task visual grounding**](https://arxiv.org/pdf/2408.01120) |  [Code](https://github.com/chenwei746/EEVG)   |
|      |       |        |                          |                |


### B. REC with REG Multi-task Setting
| Year | Venue | Name   | Paper Title / Paper Link | Code / Project |
|------|-------|--------|--------------------------|----------------|
| 2023 | Arxiv | Shikra | []()                     | [Code]()       | 
|      |       |        |                          |                |
|      |       |        |                          |                |

### C. Other Multi-task Setting
| Year | Venue | Name   | Paper Title / Paper Link | Code / Project |
|------|-------|--------|--------------------------|----------------|
| 2023 | Arxiv | Shikra | []()                     | [Code]()       | 
|      |       |        |                          |                |
|      |       |        |                          |                |


## 1.7 Generalized Visual Grounding

| Year | Venue | Name   | Paper Title / Paper Link | Code / Project |
|------|-------|--------|--------------------------|----------------|
| 2023 | Arxiv | Shikra | []()                     | [Code]()       | 
|      |       |        |                          |                |
|      |       |        |                          |                |


# 2. Advanced Topics

| Year | Venue | Name  | Paper Title / Paper Link | Code / Project |
|------|-------|-------|--------------------------|----------------|
| 2021 | Arxiv | RefTR | []()                     | [Code]()       | 
|      |       |       |                          |                |
|      |       |       |                          |                |


## 2.1 NLP Language Structure Parsing

| Year | Venue | Name   | Paper Title / Paper Link | Code / Project |
|------|-------|--------|--------------------------|----------------|
| 2023 | Arxiv | Shikra | []()                     | [Code]()       | 
|      |       |        |                          |                |
|      |       |        |                          |                |

## 2.2 Spatial Relation and Graph Networks

| Year | Venue | Name   | Paper Title / Paper Link | Code / Project |
|------|-------|--------|--------------------------|----------------|
| 2023 | TMM | CLIPREC | [**CLIPREC: Graph-Based Domain Adaptive Network for Zero-Shot Referring Expression Comprehension**](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10191020) |    N/A      |
|      |       |        |                          |                |
|      |       |        |                          |                |

## 2.3 Modular Grounding

| Year | Venue | Name   | Paper Title / Paper Link | Code / Project |
|------|-------|--------|--------------------------|----------------|
| 2023 | Arxiv | Shikra | []()                     | [Code]()       | 
|      |       |        |                          |                |
|      |       |        |                          |                |


# 3. Applications

| Year | Venue | Name   | Paper Title / Paper Link | Code / Project |
|------|-------|--------|--------------------------|----------------|
| 2023 | Arxiv | Shikra | []()                     | [Code]()       | 
|      |       |        |                          |                |
|      |       |        |                          |                |


## 3.1 Grounded Object Detection

| Year | Venue | Name   | Paper Title / Paper Link | Code / Project |
|------|-------|--------|--------------------------|----------------|
| 2023 | Arxiv | Shikra | []()                     | [Code]()       | 
|      |       |        |                          |                |
|      |       |        |                          |                |

## 3.2 Referring Counting

| Year | Venue | Name   | Paper Title / Paper Link | Code / Project |
|------|-------|--------|--------------------------|----------------|
| 2023 | Arxiv | Shikra | []()                     | [Code]()       | 
|      |       |        |                          |                |
|      |       |        |                          |                |


## 3.3 Remote Sensing Visual Grounding

| Year | Venue | Name   | Paper Title / Paper Link | Code / Project |
|------|-------|--------|--------------------------|----------------|
| 2023 | Arxiv | Shikra | []()                     | [Code]()       | 
|      |       |        |                          |                |
|      |       |        |                          |                |

## 3.4 Medical Visual Grounding

| Year | Venue | Name   | Paper Title / Paper Link | Code / Project |
|------|-------|--------|--------------------------|----------------|
| 2023 | Arxiv | Shikra | []()                     | [Code]()       | 
|      |       |        |                          |                |
|      |       |        |                          |                |

## 3.5 3D Visual Grounding

| Year | Venue | Name   | Paper Title / Paper Link | Code / Project |
|------|-------|--------|--------------------------|----------------|
| 2023 | Arxiv | Shikra | []()                     | [Code]()       | 
| 2022 | CVPR | 3D-SPS | [3D-SPS: Single-Stage 3D Visual Grounding via Referred Point Progressive Selection](https://openaccess.thecvf.com/content/CVPR2022/html/Luo_3D-SPS_Single-Stage_3D_Visual_Grounding_via_Referred_Point_Progressive_Selection_CVPR_2022_paper.html) | [Code](https://github.com/fjhzhixi/3D-SPS)|
| 2021 | ACMMM | TransRefer3D | [TransRefer3D: Entity-and-Relation Aware Transformer for Fine-Grained 3D Visual Grounding](https://arxiv.org/abs/2108.02388) | [Code](https://github.com/luo-junyu/TransRefer3D) |
|      |       |        |                          |                |

## 3.6 Video Object Grounding

| Year | Venue | Name   | Paper Title / Paper Link | Code / Project |
|------|-------|--------|--------------------------|----------------|
| 2023 | Arxiv | Shikra | []()                     | [Code]()       | 
|      |       |        |                          |                |
|      |       |        |                          |                |

## 3.7 Robotic and Multimodal Agent Applications

| Year | Venue | Name   | Paper Title / Paper Link | Code / Project |
|------|-------|--------|--------------------------|----------------|
| 2023 | Arxiv | Shikra | []()                     | [Code]()       | 
|      |       |        |                          |                |
|      |       |        |                          |                |



# 4. Datasets and Benchmarks

## 3.1 The Five Datasets for Classical Visual Grounding

| Year | Venue | Name   | Paper Title / Paper Link | Code / Project |
|------|-------|--------|--------------------------|----------------|
| 2023 | Arxiv | Shikra | []()                     | [Code]()       | 
|      |       |        |                          |                |
|      |       |        |                          |                |

## 3.2 The Other Datasets for Classical Visual Grounding

| Year | Venue | Name   | Paper Title / Paper Link | Code / Project |
|------|-------|--------|--------------------------|----------------|
| 2024 | Arxiv |  VLM-VG | [**Learning visual grounding from generative vision and language model**](https://arxiv.org/pdf/2407.14563) |  N/A   |
|      |       |        |                          |                |
|      |       |        |                          |                |


## 3.3 Dataset for the Newly Curated Scenarios

| Year | Venue | Name   | Paper Title / Paper Link | Code / Project |
|------|-------|--------|--------------------------|----------------|
| 2023 | Arxiv | Shikra | []()                     | [Code]()       | 
|   2024   |   NeurIPS    |    D$^3$    |         [**Described Object Detection: Liberating Object Detection with Flexible Expressions**](https://proceedings.neurips.cc/paper_files/paper/2023/file/f9fd24fd32eccc14cd3ecd3716a1cbf8-Paper-Conference.pdf)                 |      [Code](https://github.com/shikras/d-cube)          |
|      |       |        |                          |                |


### A. Dataset for Generalized Visual Grounding

| Year | Venue | Name   | Paper Title / Paper Link | Code / Project |
|------|-------|--------|--------------------------|----------------|
| 2023 | Arxiv | Shikra | []()                     | [Code]()       | 
|      |       |        |                          |                |
|      |       |        |                          |                |

### B. Datasets and Benchmarks for GMLLMs

| Year | Venue | Name   | Paper Title / Paper Link | Code / Project |
|------|-------|--------|--------------------------|----------------|
| 2023 | Arxiv | Shikra | []()                     | [Code]()       | 
|      |       |        |                          |                |
|      |       |        |                          |                |


### C. Dataset for Other Newly Curated Scenarios

| Year | Venue | Name   | Paper Title / Paper Link | Code / Project |
|------|-------|--------|--------------------------|----------------|
| 2023 | Arxiv | Shikra | []()                     | [Code]()       | 
|      |       |        |                          |                |
|      |       |        |                          |                |


# 5. Challenges And Outlook

| Year | Venue | Name   | Paper Title / Paper Link | Code / Project |
|------|-------|--------|--------------------------|----------------|
| 2023 | Arxiv | Shikra | []()                     | [Code]()       | 
|      |       |        |                          |                |
|      |       |        |                          |                |


# 6. Other Valuable Survey and Project

| Year | Venue  | Name                                                                                                    | Paper Title / Paper Link                                                           | Code / Project                                                                                          |
|------|--------|---------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------|
| 2021 | Github | [awesome-grounding](https://github.com/TheShadow29/awesome-grounding)                                   | N/A                                                                                | [https://github.com/TheShadow29/awesome-grounding](https://github.com/TheShadow29/awesome-grounding)    |
| 2023 | TPAMI  | [Awesome-Open-Vocabulary](https://github.com/jianzongwu/Awesome-Open-Vocabulary)                        | [**Towards Open Vocabulary Learning: A Survey**](https://arxiv.org/abs/2306.15880) | [Awesome-Open-Vocabulary](https://github.com/jianzongwu/Awesome-Open-Vocabulary)                        |
| 2024 | Github | [awesome-described-object-detection](https://github.com/Charles-Xie/awesome-described-object-detection) | N/A                                                                                | [awesome-described-object-detection](https://github.com/Charles-Xie/awesome-described-object-detection) |



# Acknowledgement

This survey took half a year to complete, and the process was laborious and burdensome.

Building up this GitHub repository also required significant effort. We would like to thank the following individuals for their contributions 
to completing this project: Baochen Xiong, Xianbing Yang, ......., etc.

# Contact
Email: [xiaolinhui16@mails.ucas.ac.cn](xiaolinhui16@mails.ucas.ac.cn).
Any kind discussions are welcomed!



## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=linhuixiao/Awesome-Visual-Grounding&type=Date)](https://star-history.com/#linhuixiao/Awesome-Visual-Grounding&Date)



