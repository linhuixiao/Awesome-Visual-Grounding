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
    <a href='https://arxiv.org/'>
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

**<p align="center"> A Decade of Visual Grounding  </p>**

This repo is used for recording, tracking, and benchmarking several recent visual grounding methods to supplement our [Grounding Survey](). 

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
@article{xiao2024hivg,
      title={HiVG: Hierarchical Multimodal Fine-grained Modulation for Visual Grounding}, 
      author={Linhui Xiao and Xiaoshan Yang and Fang Peng and Yaowei Wang and Changsheng Xu},
      year={2024},
      eprint={2404.13400},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
**It should be noted that**, due to the typesetting restrictions of the journal, there are small differences in the 
typesetting between the Arxiv version and review version.

**The following will be the relevant grounding papers and associated code links in this paper:**

# Summary of Contents
This content corresponds to the main text.


[//]: # "超链接的语法：默认:,.等符号直接忽略，空格“ ”用“-”代替，“-”继续用“-”"

- [Introduction](#introduction)
- [Summary of Contents](#summary-of-contents)
- [1. Methods: A Survey](#1-Methods-A-Survey)
  - [1.1 Fully Supervised Setting](#11-Fully-Supervised-Setting)
    - [A. Traditional CNN-based Methods](#A-Traditional-CNN-based-Methods)
    - [B. Traditional Transformer-based Methods](#B-Traditional-Transformer-based-Methods)
    - [C. VLP-based Transfer Methods](#C-VLP-based-Transfer-Methods)
    - [D. Grounding-oriented Pre-training](#D-Grounding-oriented-Pre-training)
    - [E. Grounding Multimodal LLMs](#E-Grounding-Multimodal-LLMs)
  - [1.1 Weakly Supervised Setting](#11-Weakly-Supervised-Setting)
  - [1.2 Semi-supervised Setting](#12-Semi-supervised-Setting)
  - [1.3 Unsupervised Setting](#13-Unsupervised-Setting)
  - [1.4 Zero-shot Setting](#14-Zero-shot-Setting)
  - [1.5 Multi-task Setting](#15-Multi-task-Setting)
    - [A. REC with REG Multi-task Setting](#A-REC-with-REG-Multi-task-Setting)
    - [B. REC with REG Multi-task Setting](#B-REC-with-REG-Multi-task-Setting)
    - [C. Other Multi-task Setting](#C-Other-Multi-task-Setting)
  - [1.6 Generalized Visual Grounding](#16-Generalized-Visual-Grounding)
- [2. Advanced Topics](#2-Advanced-Topics)  
  - [2.1 NLP Language Structure Parsing](#21-NLP-Language-Structure-Parsing)
  - [2.2 Spatial Relation and Graph Networks](#22-Spatial-Relation-and-Graph-Networks)
  - [2.3 Modular Grounding](#23-Modular-Grounding)
- [3. Applications](#3-Applications)
  - [3.1 Grounded Object Detection](#31-Grounded-Object-Detection)
  - [3.2 Referring Counting](#32-Referring-Counting)
  - [3.3 Remote Sensing Visual Grounding](#33-Remote-Sensing-Visual-Grounding)
  - [3.4 Medical Visual Grounding](#34-Medical-Visual-Grounding)
  - [3.5 3D Visual Grounding](#35-3D-Visual-Grounding)
  - [3.6 Video Object Grounding](#36-Video-Object-Grounding)
  - [3.7 Robotic and Multimodal Agent Applications](#37-Robotic-and-Multimodal-Agent-Applications)
- [4. Datasets and Benchmarks](#4-Datasets-and-Benchmarks)
  - [3.1 The Five Datasets for Classical Visual Grounding](#31-The-Five-Datasets-for-Classical-Visual-Grounding)
  - [3.2 The Other Datasets for Classical Visual Grounding](#32-The-Other-Datasets-for-Classical-Visual-Grounding)
  - [3.3 Dataset for the Newly Curated Scenarios](#33-Dataset-for-the-Newly-Curated-Scenariosn)
    - [A. Dataset for Generalized Visual Grounding](#A-Dataset-for-Generalized-Visual-Grounding)
    - [B. Datasets and Benchmarks for GMLLMs](#B-Datasets-and-Benchmarks-for-GMLLMs)
    - [C. Dataset for Other Newly Curated Scenarios](#C-Dataset-for-Other-Newly-Curated-Scenarios)
- [5. Challenges And Outlook](#5-Challenges-And-Outlook)
- [6. Other Valuable Survey and Project](#6-Other-Valuable-Survey-and-Project)
- [Acknowledgement](#acknowledgement)
- [Contact](#contact)




# 1. Methods: A Survey



## 1.1 Fully Supervised Setting

### A. Traditional CNN-based Methods

| Year | Venue | Work Name | Paper Title / Paper Link                                                                                                                                                           | Code / Project                                           |
|------|-------|-----------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------|
| 2016 | CVPR  | NMI       | [**Generation and comprehension of unambiguous object descriptions**](https://openaccess.thecvf.com/content_cvpr_2016/papers/Mao_Generation_and_Comprehension_CVPR_2016_paper.pdf) | [Code](https://github.com/mjhucla/Google_Refexp_toolbox) | 
|      |       |           |                                                                                                                                                                                    |                                                          |
|      |       |           |                                                                                                                                                                                    |                                                          |



### B. Transformer-based Methods


| Year | Venue | Work Name | Paper Title / Paper Link                                                                                                                                                                         | Code / Project                                 |
|------|-------|-----------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------|
| 2021 | ICCV  | TransVG   | [**Transvg: End-to-end visual grounding with transformers**](http://openaccess.thecvf.com/content/ICCV2021/html/Deng_TransVG_End-to-End_Visual_Grounding_With_Transformers_ICCV_2021_paper.html) | [Code](https://github.com/djiajunustc/TransVG) |
|      |       |           |                                                                                                                                                                                                  |                                                |
|      |       |           |                                                                                                                                                                                                  |                                                |

### C. VLP-based Methods

| Year | Venue | Name    | Paper Title / Paper Link                                                                                     | Code / Project                                |
|------|-------|---------|--------------------------------------------------------------------------------------------------------------|-----------------------------------------------|
| 2023 | TMM   | CLIP-VG | [**CLIP-VG: Self-paced Curriculum Adapting of CLIP for Visual Grounding**](https://arxiv.org/pdf/2305.08685) | [Code](https://github.com/linhuixiao/CLIP-VG) |
|      |       |         |                                                                                                              |                                               |
|      |       |         |                                                                                                              |                                               |

### D. Grounding-oriented Pre-training

| Year | Venue | Name  | Paper Title / Paper Link                                                                                                         | Code / Project                             |
|------|-------|-------|----------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------|
| 2021 | ICCV  | MDETR | [**Transvg: End-to-end visual grounding with transformers**](Mdetr-modulated detection for end-to-end multi-modal understanding) | [Code](https://github.com/ashkamath/mdetr) |
|      |       |       |                                                                                                                                  |                                            |
|      |       |       |                                                                                                                                  |                                            |

### E. Grounding Multimodal LLMs

| Year | Venue | Name   | Paper Title / Paper Link                                                                               | Code / Project                            |
|------|-------|--------|--------------------------------------------------------------------------------------------------------|-------------------------------------------|
| 2023 | Arxiv | Shikra | [**Shikra: Unleashing multimodal llm's referential dialogue magic**](https://arxiv.org/pdf/2306.15195) | [Code](https://github.com/shikras/shikra) | 
|      |       |        |                                                                                                        |                                           |
|      |       |        |                                                                                                        |                                           |


## 1.2 Weakly Supervised Setting

| Year | Venue | Name    | Paper Title / Paper Link                                                                     | Code / Project |
|------|-------|---------|----------------------------------------------------------------------------------------------|----------------|
| 2016 | ECCV  | GroundR | [Grounding of textual phrases in images by reconstruction](https://arxiv.org/pdf/1511.03745) | N/A            | 
|      |       |         |                                                                                              |                |
|      |       |         |                                                                                              |                |


## 1.2 Semi-supervised Setting

| Year | Venue  | Name       | Paper Title / Paper Link                                                                                                                           | Code / Project |
|------|--------|------------|----------------------------------------------------------------------------------------------------------------------------------------------------|----------------|
| 2023 | ICASSP | PQG-Distil | [Pseudo-Query Generation For Semi-Supervised Visual Grounding With Knowledge Distillation](https://ieeexplore.ieee.org/abstract/document/10095558) | N/A            | 
|      |        |            |                                                                                                                                                    |                |
|      |        |            |                                                                                                                                                    |                |


## 1.3 Unsupervised Setting

| Year | Venue | Name     | Paper Title / Paper Link                                                                                                                                                                                              | Code / Project                                 |
|------|-------|----------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------|
| 2022 | CVPR  | Pseudo-Q | [Pseudo-q: Generating pseudo language queries for visual grounding](https://openaccess.thecvf.com/content/CVPR2022/papers/Jiang_Pseudo-Q_Generating_Pseudo_Language_Queries_for_Visual_Grounding_CVPR_2022_paper.pdf) | [Code](https://github.com/LeapLabTHU/Pseudo-Q) | 
|      |       |          |                                                                                                                                                                                                                       |                                                |
|      |       |          |                                                                                                                                                                                                                       |                                                |


## 1.4 Zero-shot Setting

| Year | Venue | Name   | Paper Title / Paper Link                                                                                                                                                                                      | Code / Project                                        |
|------|-------|--------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------|
| 2019 | ICCV  | ZSGNet | [Zero-shot grounding of objects from natural language queries](https://openaccess.thecvf.com/content_ICCV_2019/papers/Sadhu_Zero-Shot_Grounding_of_Objects_From_Natural_Language_Queries_ICCV_2019_paper.pdf) | [Code](https://github.com/TheShadow29/zsgnet-pytorch) |
| 2022 | ACL   | ReCLIP | [Reclip: A strong zero-shot baseline for referring expression comprehension](https://arxiv.org/pdf/2204.05991)                                                                                                | [Code](https://www.github.com/allenai/reclip)         | 
|      |       |        |                                                                                                                                                                                                               |                                                       |
|      |       |        |                                                                                                                                                                                                               |                                                       |


## 1.5 Multi-task Setting

### A. REC with REG Multi-task Setting
| Year | Venue | Name   | Paper Title / Paper Link | Code / Project |
|------|-------|--------|--------------------------|----------------|
| 2023 | Arxiv | Shikra | []()                     | [Code]()       | 
|      |       |        |                          |                |
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


## 1.6 Generalized Visual Grounding

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
| 2023 | Arxiv | Shikra | []()                     | [Code]()       | 
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
|      |       |        |                          |                |
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
| 2023 | Arxiv | Shikra | []()                     | [Code]()       | 
|      |       |        |                          |                |
|      |       |        |                          |                |


## 3.3 Dataset for the Newly Curated Scenarios

| Year | Venue | Name   | Paper Title / Paper Link | Code / Project |
|------|-------|--------|--------------------------|----------------|
| 2023 | Arxiv | Shikra | []()                     | [Code]()       | 
|      |       |        |                          |                |
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

| Year | Venue  | Name                                                                             | Paper Title / Paper Link                                                           | Code / Project                                                                                       |
|------|--------|----------------------------------------------------------------------------------|------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|
| 2021 | Github | [awesome-grounding](https://github.com/TheShadow29/awesome-grounding)            | N/A                                                                                | [https://github.com/TheShadow29/awesome-grounding](https://github.com/TheShadow29/awesome-grounding) |
| 2023 | TPAMI  | [Awesome-Open-Vocabulary](https://github.com/jianzongwu/Awesome-Open-Vocabulary) | [**Towards Open Vocabulary Learning: A Survey**](https://arxiv.org/abs/2306.15880) | [Awesome-Open-Vocabulary](https://github.com/jianzongwu/Awesome-Open-Vocabulary)                     |




# Acknowledgement

This survey took half a year to complete, and the process was laborious and burdensome.

Building up this GitHub repository also required significant effort. We would like to thank the following individuals for their contributions 
to completing this project: Baochen Xiong, Xianbing Yang, ......., etc.

# Contact
Email: [xiaolinhui16@mails.ucas.ac.cn](xiaolinhui16@mails.ucas.ac.cn).
Any kind discussions are welcomed!



## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=linhuixiao/Awesome-Visual-Grounding&type=Date)](https://star-history.com/#linhuixiao/Awesome-Visual-Grounding&Date)



