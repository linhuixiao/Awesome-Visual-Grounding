[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)
[![PR's Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat)](https://github.com/linhuixiao/Awesome-Visual-Grounding/pulls)
<br />
<p align="center">
  <h1 align="center">Towards Visual Grounding: A Survey</h1>
  <p align="center">
    <b> T-PAMI under review, 2024 </b>
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

[//]: # (    <a href='https://ieeexplore.ieee.org/document/10420487'>)
[//]: # (      <img src='https://img.shields.io/badge/TPAMI-PDF-blue?style=flat&logo=IEEE&logoColor=green' alt='TPAMI PDF'>)
[//]: # (    </a>)
[//]: # (  </p>)
<br />


<p align="center"> <img src='figs/illustration.jpg' align="center" width="60%"> </p>

**<p align="center"> An Illustration of Visual Grounding  </p>**


<p align="center"> <img src='figs/development_trends_h.jpg' align="center" width="100%"> </p>

**<p align="center"> A Decade of Visual Grounding  </p>**

This repo is used for recording, tracking, and benchmarking several recent visual grounding methods to supplement our [survey]().  
If you find any work missing or have any suggestions (papers, implementations, and other resources), feel free to [pull requests](https://github.com/linhuixiao/Awesome-Visual-Grounding/pulls).
We will add the missing papers to this repo as soon as possible.

### 🔥 Add Your Paper in our Repo and Survey!

- You are welcome to give us an issue or PR (pull request) for your open vocabulary learning work !

- Note that: Due to the huge paper in Arxiv, we are sorry to cover all in our survey. You can directly present a PR into this repo and we will record it for next version update of our survey.

[//]: # (- **Our survey will be updated in 2024.3.**)


### 🔥 New



- We made our paper public and created this repository on **December 26, 2024**.



### 🔥 Highlight!!

- A comprehensive survey for Visual Grounding, including Referring Expression Comprehension and Phrase Grounding.

- It includes the most recently Grounding Multi-modal LLMs and VLP-based grounding transfer works. 

- We list detailed results for the most representative works and give a fairer and clearer comparison of different approaches.

- We provide a list of future research insights.


[https://github.com/TheShadow29/awesome-grounding](https://github.com/TheShadow29/awesome-grounding)

[Awesome-Open-Vocabulary](https://github.com/jianzongwu/Awesome-Open-Vocabulary)

# Introduction

we are the first survey in the past five years to systematically track and summarize the development of visual 
grounding over the last decade. By extracting common technical details, this review encompasses the most representative
work in each subtopic. 

This survey is also currently the most comprehensive review in the field of visual grounding. We aim for this article 
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

It should be noted that, due to the typesetting restrictions of the journal, there are differences in the 
typesetting between the review version and Arxiv version.

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


**The following will be the relevant grounding papers and associated code links in this paper:**

# Summary of Contents
This content corresponds to the main text.
- [Introduction](#introduction)
- [Summary of Contents](#summary-of-contents)
- [1. Methods: A Survey](#methods-a-survey)
  - [1.1 Fully Supervised Setting](#open-vocabulary-object-detection)
    - [A. Traditional CNN-based Methods](#semantic-segmentation)
    - [B. Traditional Transformer-based Methods](#instance-segmentation)
    - [C. VLP-based Transfer Methods](#panoptic-segmentation)
    - [D. Grounding-oriented Pre-training](#panoptic-segmentation)
    - [E. Grounding Multimodal LLMs](#panoptic-segmentation)
  - [1.1 Weakly Supervised Setting](#open-vocabulary-object-detection)
  - [1.2 Semi-supervised Setting](#open-vocabulary-object-detection)
  - [1.3 Unsupervised Setting](#open-vocabulary-object-detection)
  - [1.4 Zero-shot Setting](#open-vocabulary-object-detection)
  - [1.5 Multi-task Setting](#open-vocabulary-object-detection)
  - [1.6 Generalized Visual Grounding](#open-vocabulary-object-detection)
- [2. Advanced Topics](#methods-a-survey)  
  - [2.1 NLP Language Structure Parsing](#open-vocabulary-object-detection)
  - [2.2 Spatial Relation and Graph Networks](#open-vocabulary-object-detection)
  - [2.3 Modular Grounding](#open-vocabulary-object-detection)
- [3. Applications](#methods-a-survey)
  - [3.1 Grounded Object Detection](#open-vocabulary-object-detection)
  - [3.2 Referring Counting](#open-vocabulary-object-detection)
  - [3.3 Remote Sensing Visual Grounding](#open-vocabulary-object-detection)
  - [3.4 Medical Visual Grounding](#open-vocabulary-object-detection)
  - [3.5 3D Visual Grounding](#open-vocabulary-object-detection)
  - [3.6 Video Object Grounding](#open-vocabulary-object-detection)
  - [3.7 Robotic and Multimodal Agent Applications](#open-vocabulary-object-detection)
- [4. Datasets and Benchmarks](#methods-a-survey)
  - [3.1 The Five Datasets for Classical Visual Grounding](#open-vocabulary-object-detection)
  - [3.2 The Other Datasets for Classical Visual Grounding](#open-vocabulary-object-detection)
  - [3.3 Dataset for the Newly Curated Scenarios](#open-vocabulary-object-detection)
    - [A. Dataset for Generalized Visual Grounding](#open-vocabulary-object-detection)
    - [B. Datasets and Benchmarks for GMLLMs](#open-vocabulary-object-detection)
    - [C. Dataset for Other Newly Curated Scenarios](#open-vocabulary-object-detection)
- [4. Challenges And Outlook](#methods-a-survey)
- [5. Other Valuable Survey and Project](#methods-a-survey)
- [Acknowledgement](#acknowledgement)
- [Contact](#contact)


一些简称的统一：ACM MM, NeurIPS

# 1. Methods: A Survey



## 1.1 Fully Supervised Setting

### A. Traditional CNN-based Methods

### B. Transformer-based Methods


| Year | Venue | Work Name | Paper Title / Paper Link                                                                                                                                                                         | Code / Project |
|------|-------|-----------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------|
| 2021 | ICCV   | TransVG  | [**Transvg: End-to-end visual grounding with transformers**](http://openaccess.thecvf.com/content/ICCV2021/html/Deng_TransVG_End-to-End_Visual_Grounding_With_Transformers_ICCV_2021_paper.html) | [Code](https://github.com/djiajunustc/TransVG) 
|      |       |           |                                                                                                                                                                                                  |
|      |       |           |                                                                                                                                                                                                  |


### C. VLP-based Methods

### D. Grounding-oriented Pre-training

### E. Grounding Multimodal LLMs


# Acknowledgement

This survey took half a year to complete, and the process was laborious and complicated. Building up this GitHub 
repository also required significant effort. We would like to thank the following individuals for their contributions 
to completing this project: Baochen Xiong, Xianbing Yang, etc.

# Contact
Email: [xiaolinhui16@mails.ucas.ac.cn](xiaolinhui16@mails.ucas.ac.cn).
Any kind discussions are welcomed!



## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=linhuixiao/Awesome-Visual-Grounding&type=Date)](https://star-history.com/#linhuixiao/Awesome-Visual-Grounding&Date)




