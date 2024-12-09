# Awesome Large Vision Language Models (VLLMs)


## 🔥🔥🔥 Visual Large Language Models for Generalized and Specialized Applications

<p align="center">
    <img src="assets/VLM_revolution.png" width="90%" height="90%">
</p>


## 📢 News

🚀 **What's New in This Update**:

2024.12.30: We release our VLLM application paper list repo!

<font size=5><center><b> Table of Contents </b> </center></font>
- [🔥🔥🔥 Visual Large Language Models for Generalized and Specialized Applications](#awesome-vllms)
  - [Existing VLM surveys](#existing-vlm-surveys)
    - [VLM surveys](#vlm-surveys)
    - [MLLM surveys](#mllm-surveys)
  - [Vision-to-text](#vision-to-text)
    - [Image-to-text](#image-to-text)
        - [General domain](#general-domain)
            - [General ability](#general-ability)
            - [REC](#rec)
            - [RES](#res)
            - [OCR](#ocr)
            - [Retrieval](#retrieval)
        - [VLLM+X](#VLLM+X)
            - [Remote sensing](#remote-sensing)
            - [Medical](#medical)
            - [Science and math](#science-and-math)
            - [Graphics and UI](#graphics-and-ui)
            - [Financial analysis](#financial-analysis)
    - [Video-to-text](#video-to-text)
        - [General domain](#general-domain)
        - [Video conversation](#video-conversation)
        - [Egocentric understanding](#egocentric-understanding)
  - [Vision-to-action](#vision-to-action)
      - [Autonomous driving](#autonomous-driving)
      - [Embodied AI](#embodied-ai)
      - [Automated tool management](#automated-tool-management)
  - [Text-to-vision](#text-to-vision)
      - [Text-to-image](#text-to-image)
      - [Text-to-3D](#text-to-3D)
      - [Text-to-video](#text-to-video)
  - [Other applications](#other-applications)
      - [Face](#face)

## Existing VLM surveys
### VLM surveys
|  Title  |   Venue  |   Date   |   Code   |   Project   |
|:--------|:--------:|:--------:|:--------:|:--------:|
| ![Star](https://img.shields.io/github/stars/jingyi0000/VLM_survey.svg?style=social&label=Star) <br>[**Vision-Language Models for Vision Tasks: A Survey**](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10445007) <br> | T-PAMI | 2024-8-8 | Github | [Project](https://github.com/jingyi0000/VLM_survey) |
| [**Vision-and-Language Pretrained Models: A Survey**](https://arxiv.org/pdf/2204.07356.pdf) <br> | IJCAI (survey track) | 2022-5-3 | Github | Project |

### MLLM surveys
|  Title  |   Venue  |   Date   |   Code   |   Project   |
|:--------|:--------:|:--------:|:--------:|:--------:|
| ![Star](https://img.shields.io/github/stars/BradyFU/Awesome-Multimodal-Large-Language-Models.svg?style=social&label=Star) <br> [**A Survey on Multimodal Large Language Models**](https://arxiv.org/pdf/2306.13549.pdf) <br> | T-PAMI | 2024-11-29 | Github | [Project](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models) |
| ![Star](https://img.shields.io/github/stars/BradyFU/Awesome-Multimodal-Large-Language-Models.svg?style=social&label=Star) <br> [**MME-Survey: A Comprehensive Survey on Evaluation of Multimodal LLMs**](https://arxiv.org/pdf/2411.15296) <br> | ArXiv | 2024-11-22 | Github | [Project](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/) |
| [**A Survey on Multimodal Large Language Models**](https://academic.oup.com/nsr/advance-article-pdf/doi/10.1093/nsr/nwae403/60676453/nwae403.pdf) <br> | National Science Review | 2024-11-12 | Github | Project |
| [**Video Understanding with Large Language Models: A Survey**](https://arxiv.org/pdf/2312.17432) <br> | ArXiv | 2024-6-24 | Github | [Project](https://github.com/yunlong10/Awesome-LLMs-for-Video-Understanding) |
| ![Star](https://img.shields.io/github/stars/yunlong10/Awesome-LLMs-for-Video-Understanding.svg?style=social&label=Star) <br> [**The Revolution of Multimodal Large Language Models: A Survey**](https://arxiv.org/pdf/2402.12451) <br> | ArXiv | 2024-6-6 | Github | Project |
| ![Star](https://img.shields.io/github/stars/lhanchao777/LVLM-Hallucinations-Survey.svg?style=social&label=Star) <br> [**A Survey on Hallucination in Large Vision-Language Models**](https://arxiv.org/pdf/2402.00253) <br> | ArXiv | 2024-5-6 | Github | [Project](https://github.com/lhanchao777/LVLM-Hallucinations-Survey) |
| [**Exploring the Frontier of Vision-Language Models: A Survey of Current Methodologies and Future Directions**](https://arxiv.org/pdf/2404.07214) <br> | ArXiv | 2024-4-12 | Github | Project |
| ![Star](https://img.shields.io/github/stars/MM-LLMs/mm-llms.github.io.svg?style=social&label=Star) <br> [**MM-LLMs: Recent Advances in MultiModal Large Language Models**](https://arxiv.org/pdf/2401.13601v4) <br> | ArXiv | 2024-2-20 | Github | [Project](https://mm-llms.github.io/) |
| [**Exploring the Reasoning Abilities of Multimodallarge Language Models (mllms): a Comprehensive survey on Emerging Trends in Multimodal Reasonings**](https://arxiv.org/pdf/2401.06805) <br> | ArXiv | 2024-1-18 | Github | Project |
| [**Visual Instruction Tuning towards General-Purpose Multimodal Model: A Survey**](https://arxiv.org/pdf/2312.16602) <br> | ArXiv | 2023-12-27 | Github | Project |
| [**Multimodal Large Language Models: A Survey**](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10386743) <br> | BigData | 2023-12-15 | Github | Project |
| ![Star](https://img.shields.io/github/stars/Yutong-Zhou-cv/Awesome-Text-to-Image.svg?style=social&label=Star) <br> [**Vision + Language Applications: A Survey**](https://openaccess.thecvf.com/content/CVPR2023W/GCV/papers/Zhou_Vision__Language_Applications_A_Survey_CVPRW_2023_paper.pdf) <br> | CVPRW | 2023-5-24 | Github | [Project](https://github.com/Yutong-Zhou-cv/Awesome-Text-to-Image) |

## Vision-to-text
### Image-to-text
#### General domain
##### General ability
|  Model  |  Title  |   Venue  |   Date   |   Code   |   Project   |
|:--------|:--------|:--------:|:--------:|:--------:|:--------:|

##### REC
|  Model  |  Title  |   Venue  |   Date   |   Code   |   Project   |
|:--------|:--------|:--------:|:--------:|:--------:|:--------:|

##### RES
|  Model  |  Title  |   Venue  |   Date   |   Code   |   Project   |
|:--------|:--------|:--------:|:--------:|:--------:|:--------:|

##### OCR
|  Model  |  Title  |   Venue  |   Date   |   Code   |   Project   |
|:--------|:--------|:--------:|:--------:|:--------:|:--------:|

##### Retrieval
|  Model  |  Title  |   Venue  |   Date   |   Code   |   Project   |
|:--------|:--------|:--------:|:--------:|:--------:|:--------:|

#### VLLM + X
##### Remote sensing
|  Model  |  Title  |   Venue  |   Date   |   Code   |   Project   |
|:--------|:--------|:--------:|:--------:|:--------:|:--------:|
| VHM | ![Star](https://img.shields.io/github/stars/opendatalab/VHM.svg?style=social&label=Star) <br> [**VHM: Versatile and Honest Vision Language Model for Remote Sensing Image Analysis**](https://arxiv.org/pdf/2403.20213) <br> | ArXiv | 2024-11-06 | [Github](https://github.com/opendatalab/VHM) | [Project](https://fitzpchao.github.io/vhm_page/) |
| LHRS-Bot | ![Star](https://img.shields.io/github/stars/NJU-LHRS/LHRS-Bot.svg?style=social&label=Star) <br> [**LHRS-Bot: Empowering Remote Sensing with VGI-Enhanced Large Multimodal Language Model**](https://arxiv.org/pdf/2402.02544) <br> | ECCV | 2024-07-16 | [Github](https://github.com/NJU-LHRS/LHRS-Bot) | Project |
| Popeye | [**Popeye: A Unified Visual-Language Model for Multi-Source Ship Detection from Remote Sensing Imagery**](https://arxiv.org/pdf/2403.03790) <br> | J-STARS | 2024-06-13 | Github | Project |
| RS-LLaVA | ![Star](https://img.shields.io/github/stars/BigData-KSU/RS-LLaVA.svg?style=social&label=Star) <br> [**RS-LLaVA: A Large Vision-Language Model for Joint Captioning and Question Answering in Remote Sensing Imagery**](https://www.mdpi.com/2072-4292/16/9/1477) <br> | Remote Sens. | 2024-04-23 | [Github](https://github.com/BigData-KSU/RS-LLaVA) | Project |
| EarthGPT | ![Star](https://img.shields.io/github/stars/wivizhang/EarthGPT.svg?style=social&label=Star) <br> [**EarthGPT: A Universal Multi-modal Large Language Model for Multi-sensor Image Comprehension in Remote Sensing Domain**](https://arxiv.org/pdf/2401.16822) <br> | TGRS | 2024-03-08 | [Github](https://github.com/wivizhang/EarthGPT) | Project |
| RS-CapRet | [**Large Language Models for Captioning and Retrieving Remote Sensing Images**](https://arxiv.org/pdf/2402.06475) <br> | ArXiv | 2024-02-09 | Github | Project |
| SkyEyeGPT | ![Star](https://img.shields.io/github/stars/ZhanYang-nwpu/SkyEyeGPT.svg?style=social&label=Star) <br> [**SkyEyeGPT: Unifying Remote Sensing Vision-Language Tasks via Instruction Tuning with Large Language Model**](https://arxiv.org/pdf/2401.09712) <br> | ArXiv | 2024-01-18 | [Github](https://github.com/ZhanYang-nwpu/SkyEyeGPT) | Project |
| GeoChat | ![Star](https://img.shields.io/github/stars/mbzuai-oryx/geochat.svg?style=social&label=Star) <br> [**GeoChat: Grounded Large Vision-Language Model for Remote Sensing**](https://arxiv.org/pdf/2311.15826) <br> | CVPR | 2023-11-24 | [Github](https://github.com/mbzuai-oryx/geochat) | [Project](https://mbzuai-oryx.github.io/GeoChat/) |
| RSGPT | ![Star](https://img.shields.io/github/stars/Lavender105/RSGPT.svg?style=social&label=Star) <br> [**RSGPT: A Remote Sensing Vision Language Model and Benchmark**](https://arxiv.org/pdf/2307.15266) <br> | ArXiv | 2023-07-28 | [Github](https://github.com/Lavender105/RSGPT) | Project |

##### Medical
|  Model  |  Title  |   Venue  |   Date   |   Code   |   Project   |
|:--------|:--------|:--------:|:--------:|:--------:|:--------:|
| PMC-VQA | ![Star](https://img.shields.io/github/stars/xiaoman-zhang/PMC-VQA.svg?style=social&label=Star) <br> [**PMC-VQA: Visual Instruction Tuning for Medical Visual Question Answering**](https://arxiv.org/pdf/2305.10415) <br> | ArXiv | 2024-09-08 | [Github](https://github.com/xiaoman-zhang/PMC-VQA) | Project |
| MedVersa | [**A Generalist Learner for Multifaceted Medical Image Interpretation**](https://arxiv.org/pdf/2405.07988) <br> | ArXiv | 2024-05-13 | Github | Project |
| PeFoMed | ![Star](https://img.shields.io/github/stars/jinlHe/PeFoMed.svg?style=social&label=Star) <br> [**PeFoMed: Parameter Efficient Fine-tuning of Multimodal Large Language Models for Medical Imaging**](https://arxiv.org/pdf/2401.02797) <br> | ArXiv | 2024-04-16 | [Github](https://github.com/jinlHe/PeFoMed) | Project |
| RaDialog | ![Star](https://img.shields.io/github/stars/ChantalMP/RaDialog.svg?style=social&label=Star) <br> [**RaDialog: A Large Vision-Language Model for Radiology Report Generation and Conversational Assistance**](https://arxiv.org/pdf/2311.18681) <br> | ArXiv | 2023-11-30 | [Github](https://github.com/ChantalMP/RaDialog) | Project |
| Med-Flamingo | ![Star](https://img.shields.io/github/stars/snap-stanford/med-flamingo.svg?style=social&label=Star) <br> [**Med-Flamingo: a Multimodal Medical Few-shot Learner**](https://arxiv.org/pdf/2307.15189) <br> | ML4H | 2023-07-27 | [Github](https://github.com/snap-stanford/med-flamingo) | Project |
| XrayGPT | ![Star](https://img.shields.io/github/stars/mbzuai-oryx/XrayGPT.svg?style=social&label=Star) <br> [**XrayGPT: Chest Radiographs Summarization using Medical Vision-Language Models**](https://arxiv.org/pdf/2306.07971) <br> | BioNLP | 2023-06-13 | [Github](https://github.com/mbzuai-oryx/XrayGPT) | Project |
| LLaVA-Med | ![Star](https://img.shields.io/github/stars/microsoft/LLaVA-Med.svg?style=social&label=Star) <br> [**LLaVA-Med: Training a Large Language-and-Vision Assistant for Biomedicine in One Day**](https://arxiv.org/pdf/2306.00890) <br> | NeurIPS | 2023-06-01 | [Github](https://github.com/microsoft/LLaVA-Med) | Project |
| CXR-RePaiR-Gen | [**Retrieval Augmented Chest X-Ray Report Generation using OpenAI GPT models**](https://arxiv.org/pdf/2305.03660) <br> | MLHC | 2023-05-05 | Github | Project |

##### Science and math
|  Model  |  Title  |   Venue  |   Date   |   Code   |   Project   |
|:--------|:--------|:--------:|:--------:|:--------:|:--------:|
| MAVIS | ![Star](https://img.shields.io/github/stars/ZrrSkywalker/MAVIS.svg?style=social&label=Star) <br> [**MAVIS: Mathematical Visual Instruction Tuning**](https://arxiv.org/pdf/2407.08739) <br> | ECCV | 2024-11-01 | [Github](https://github.com/ZrrSkywalker/MAVIS) | Project |
| Math-LLaVA | ![Star](https://img.shields.io/github/stars/HZQ950419/Math-LLaVA.svg?style=social&label=Star) <br> [**Math-LLaVA: Bootstrapping Mathematical Reasoning for Multimodal Large Language Models**](https://arxiv.org/pdf/2406.172940) <br> | EMNLP | 2024-10-08 | [Github](https://github.com/HZQ950419/Math-LLaVA) | Project |
| MathVerse | ![Star](https://img.shields.io/github/stars/ZrrSkywalker/MathVerse.svg?style=social&label=Star) <br> [**MathVerse: Does Your Multi-modal LLM Truly See the Diagrams in Visual Math Problems?**](https://arxiv.org/pdf/2403.14624) <br> | ECCV | 2024-08-18 | [Github](https://github.com/ZrrSkywalker/MathVerse) | [Project](https://mathverse-cuhk.github.io/) |
| We-Math | ![Star](https://img.shields.io/github/stars/We-Math/We-Math.svg?style=social&label=Star) <br> [**We-Math: Does Your Large Multimodal Model Achieve Human-like Mathematical Reasoning?**](https://arxiv.org/pdf/2407.01284) <br> | ArXiv | 2024-07-01 | [Github](https://github.com/We-Math/We-Math) | [Project](https://we-math.github.io/) |
| CMMaTH | [**CMMaTH: A Chinese Multi-modal Math Skill Evaluation Benchmark for Foundation Models**](https://arxiv.org/pdf/2407.12023) <br> | ArXiv | 2024-06-28 | Github | Project |
| GeoEval | ![Star](https://img.shields.io/github/stars/GeoEval/GeoEval.svg?style=social&label=Star) <br> [**GeoEval: Benchmark for Evaluating LLMs and Multi-Modal Models on Geometry Problem-Solving**](https://arxiv.org/pdf/2402.10104) <br> | ACL | 2024-05-17 | [Github](https://github.com/GeoEval/GeoEval) | Project |
| FigurA11y | ![Star](https://img.shields.io/github/stars/allenai/figura11y.svg?style=social&label=Star) <br> [**FigurA11y: AI Assistance for Writing Scientific Alt Text**](https://dl.acm.org/doi/10.1145/3640543.3645212) <br> | IUI | 2024-04-05 | [Github](https://github.com/allenai/figura11y) | Project |
| MathVista | ![Star](https://img.shields.io/github/stars/lupantech/MathVista.svg?style=social&label=Star) <br> [**MathVista: Evaluating Mathematical Reasoning of Foundation Models in Visual Contexts**](https://arxiv.org/pdf/2310.02255) <br> | ICLR | 2024-01-21 | [Github](https://github.com/lupantech/MathVista) | [Project](https://mathvista.github.io/) |
| mPLUG-PaperOwl | ![Star](https://img.shields.io/github/stars/X-PLUG/mPLUG-DocOwl.svg?style=social&label=Star) <br> [**mPLUG-PaperOwl: Scientific Diagram Analysis with the Multimodal Large Language Model**](https://arxiv.org/pdf/2311.18248) <br> | sMM | 2024-01-09 | [Github](https://github.com/X-PLUG/mPLUG-DocOwl/tree/main/PaperOwl) | Project |
| G-LLaVA | ![Star](https://img.shields.io/github/stars/pipilurj/G-LLaVA.svg?style=social&label=Star) <br> [**G-LLaVA: Solving Geometric Problem with Multi-Modal Large Language Model**](https://arxiv.org/pdf/2312.11370) <br> | ArXiv | 2023-12-18 | [Github](https://github.com/pipilurj/G-LLaVA) | Project |
| T-SciQ | ![Star](https://img.shields.io/github/stars/T-SciQ/T-SciQ.svg?style=social&label=Star) <br> [**T-SciQ: Teaching Multimodal Chain-of-Thought Reasoning via Mixed Large Language Model Signals for Science Question Answering**](https://arxiv.org/pdf/2305.03453) <br> | AAAI | 2023-12-18 | [Github](https://github.com/T-SciQ/T-SciQ) | Project |
| ScienceQA | ![Star](https://img.shields.io/github/stars/lupantech/ScienceQA.svg?style=social&label=Star) <br> [**Learn to Explain: Multimodal Reasoning via Thought Chains for Science Question Answering**](https://arxiv.org/pdf/2209.09513) <br> | NeurIPS | 2022-10-17 | [Github](https://github.com/lupantech/ScienceQA) | [Project](https://scienceqa.github.io/) |

##### Graphics and UI
|  Model  |  Title  |   Venue  |   Date   |   Code   |   Project   |
|:--------|:--------|:--------:|:--------:|:--------:|:--------:|
| Graphist | ![Star](https://img.shields.io/github/stars/graphic-design-ai/graphist.svg?style=social&label=Star) <br> [**Graphic Design with Large Multimodal Model**](https://arxiv.org/pdf/2404.14368) <br> | ArXiv | 2024-04-22 | [Github](https://github.com/graphic-design-ai/graphist) | Project |
| Ferret-UI | [**Ferret-UI: Grounded Mobile UI Understanding with Multimodal LLMs**](https://arxiv.org/pdf/2404.05719) <br> | ECCV | 2024-04-08 | Github | Project |
| CogAgent | ![Star](https://img.shields.io/github/stars/THUDM/CogVLM.svg?style=social&label=Star) <br> [**CogAgent: A Visual Language Model for GUI Agents**](https://arxiv.org/pdf/2312.08914) <br> | CVPR | 2023-12-21 | [Github](https://github.com/THUDM/CogVLM) | Project |

##### Financial analysis
|  Model  |  Title  |   Venue  |   Date   |   Code   |   Project   |
|:--------|:--------|:--------:|:--------:|:--------:|:--------:|
| FinTral | [**FinTral: A Family of GPT-4 Level Multimodal Financial Large Language Models**](https://arxiv.org/pdf/2402.10986) <br> | ACL | 2024-06-14 | Github | Project |
| FinVis-GPT | ![Star](https://img.shields.io/github/stars/wwwadx/FinVis-GPT.svg?style=social&label=Star) <br> [**FinVis-GPT: A Multimodal Large Language Model for Financial Chart Analysis**](https://arxiv.org/pdf/2308.01430) <br> | ArXiv | 2023-07-31 | [Github](https://github.com/wwwadx/FinVis-GPT) | Project |

### Video-to-text
####  General domain
|  Model  |  Title  |   Venue  |   Date   |   Code   |   Project   |
|:--------|:--------|:--------:|:--------:|:--------:|:--------:|

#### Video conversation
|  Model  |  Title  |   Venue  |   Date   |   Code   |   Project   |
|:--------|:--------|:--------:|:--------:|:--------:|:--------:|

#### Egocentric view
|  Model  |  Title  |   Venue  |   Date   |   Code   |   Project   |
|:--------|:--------|:--------:|:--------:|:--------:|:--------:|


## Vision-to-action

### Autonomous driving
#### Perception
|  Model  |  Title  |   Venue  |   Date   |   Code   |   Project   |
|:--------|:--------|:--------:|:--------:|:--------:|:--------:|

#### Planning
|  Model  |  Title  |   Venue  |   Date   |   Code   |   Project   |
|:--------|:--------|:--------:|:--------:|:--------:|:--------:|

#### Prediction
|  Model  |  Title  |   Venue  |   Date   |   Code   |   Project   |
|:--------|:--------|:--------:|:--------:|:--------:|:--------:|

### Embodied AI
#### Perception
|  Model  |  Title  |   Venue  |   Date   |   Code   |   Project   |
|:--------|:--------|:--------:|:--------:|:--------:|:--------:|

#### Manipulation
|  Model  |  Title  |   Venue  |   Date   |   Code   |   Project   |
|:--------|:--------|:--------:|:--------:|:--------:|:--------:|

#### Planning
|  Model  |  Title  |   Venue  |   Date   |   Code   |   Project   |
|:--------|:--------|:--------:|:--------:|:--------:|:--------:|

#### Navigation
|  Model  |  Title  |   Venue  |   Date   |   Code   |   Project   |
|:--------|:--------|:--------:|:--------:|:--------:|:--------:|

### Automated tool management
|  Model  |  Title  |   Venue  |   Date   |   Code   |   Project   |
|:--------|:--------|:--------:|:--------:|:--------:|:--------:|


## Text-to-vision
|  Model  |  Title  |   Venue  |   Date   |   Code   |   Project   |
|:--------|:--------|:--------:|:--------:|:--------:|:--------:|


### Text-to-image
|  Model  |  Title  |   Venue  |   Date   |   Code   |   Project   |
|:--------|:--------|:--------:|:--------:|:--------:|:--------:|


### Text-to-3D
|  Model  |  Title  |   Venue  |   Date   |   Code   |   Project   |
|:--------|:--------|:--------:|:--------:|:--------:|:--------:|


### Text-to-video
|  Model  |  Title  |   Venue  |   Date   |   Code   |   Project   |
|:--------|:--------|:--------:|:--------:|:--------:|:--------:|

## Other applications
### Face
|  Model  |  Title  |   Venue  |   Date   |   Code   |   Project   |
|:--------|:--------|:--------:|:--------:|:--------:|:--------:|

