
# Awesome Visual Large Language Models (VLLMs)


## 🔥🔥🔥 Visual Large Language Models for Generalized and Specialized Applications

<p align="center">
    <img src="assets/VLM_revolution.png" width="90%" height="90%">
</p>


## 📢 News

🚀 **What's New in This Update**:

* [2024.12.30]: 🔥 We release our VLLM application paper list repo!

<span id="head-content"><font size=5><center><b> Table of Contents </b> </center></font></span>
- [Visual Large Language Models for Generalized and Specialized Applications](#awesome-vllms)
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
        - [Perception](#perception)
        - [Planning](#planning)
        - [Prediction](#prediction)
      - [Embodied AI](#embodied-ai)
        - [Perception](#perception)
        - [Manipulation](#manipulation)
        - [Planning](#planning)
        - [Navigation](#navigation)
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
| ![Star](https://img.shields.io/github/stars/Yutong-Zhou-cv/Awesome-Text-to-Image.svg?style=social&label=Star) <br> [**Vision + Language Applications: A Survey**](https://openaccess.thecvf.com/content/CVPR2023W/GCV/papers/Zhou_Vision__Language_Applications_A_Survey_CVPRW_2023_paper.pdf) <br> | CVPRW | 2023-5-24 | Github | [Project](https://github.com/Yutong-Zhou-cv/Awesome-Text-to-Image) |
| [**Vision-and-Language Pretrained Models: A Survey**](https://arxiv.org/pdf/2204.07356.pdf) <br> | IJCAI (survey track) | 2022-5-3 | Github | Project |

[<u><🎯Back to Top></u>](#head-content)

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

[<u><🎯Back to Top></u>](#head-content)

## Vision-to-text
### Image-to-text
#### General domain
##### General ability
|  Name  |  Title  |   Venue  |   Date   |   Code   |   Project   |
|:--------|:--------|:--------:|:--------:|:--------:|:--------:|

[<u><🎯Back to Top></u>](#head-content)

##### REC
|  Name  |  Title  |   Venue  |   Date   |   Code   |   Project   |
|:--------|:--------|:--------:|:--------:|:--------:|:--------:|

[<u><🎯Back to Top></u>](#head-content)

##### RES
|  Name  |  Title  |   Venue  |   Date   |   Code   |   Project   |
|:--------|:--------|:--------:|:--------:|:--------:|:--------:|

[<u><🎯Back to Top></u>](#head-content)

##### OCR
|  Name  |  Title  |   Venue  |   Date   |   Code   |   Project   |
|:--------|:--------|:--------:|:--------:|:--------:|:--------:|

[<u><🎯Back to Top></u>](#head-content)

##### Retrieval
|  Name  |  Title  |   Venue  |   Date   |   Code   |   Project   |
|:--------|:--------|:--------:|:--------:|:--------:|:--------:|

[<u><🎯Back to Top></u>](#head-content)

#### VLLM + X
##### Remote sensing
|  Name  |  Title  |   Venue  |   Date   |   Code   |   Project   |
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

[<u><🎯Back to Top></u>](#head-content)

##### Medical
|  Name  |  Title  |   Venue  |   Date   |   Code   |   Project   |
|:--------|:--------|:--------:|:--------:|:--------:|:--------:|
| PMC-VQA | ![Star](https://img.shields.io/github/stars/xiaoman-zhang/PMC-VQA.svg?style=social&label=Star) <br> [**PMC-VQA: Visual Instruction Tuning for Medical Visual Question Answering**](https://arxiv.org/pdf/2305.10415) <br> | ArXiv | 2024-09-08 | [Github](https://github.com/xiaoman-zhang/PMC-VQA) | Project |
| MedVersa | [**A Generalist Learner for Multifaceted Medical Image Interpretation**](https://arxiv.org/pdf/2405.07988) <br> | ArXiv | 2024-05-13 | Github | Project |
| PeFoMed | ![Star](https://img.shields.io/github/stars/jinlHe/PeFoMed.svg?style=social&label=Star) <br> [**PeFoMed: Parameter Efficient Fine-tuning of Multimodal Large Language Models for Medical Imaging**](https://arxiv.org/pdf/2401.02797) <br> | ArXiv | 2024-04-16 | [Github](https://github.com/jinlHe/PeFoMed) | Project |
| RaDialog | ![Star](https://img.shields.io/github/stars/ChantalMP/RaDialog.svg?style=social&label=Star) <br> [**RaDialog: A Large Vision-Language Model for Radiology Report Generation and Conversational Assistance**](https://arxiv.org/pdf/2311.18681) <br> | ArXiv | 2023-11-30 | [Github](https://github.com/ChantalMP/RaDialog) | Project |
| Med-Flamingo | ![Star](https://img.shields.io/github/stars/snap-stanford/med-flamingo.svg?style=social&label=Star) <br> [**Med-Flamingo: a Multimodal Medical Few-shot Learner**](https://arxiv.org/pdf/2307.15189) <br> | ML4H | 2023-07-27 | [Github](https://github.com/snap-stanford/med-flamingo) | Project |
| XrayGPT | ![Star](https://img.shields.io/github/stars/mbzuai-oryx/XrayGPT.svg?style=social&label=Star) <br> [**XrayGPT: Chest Radiographs Summarization using Medical Vision-Language Models**](https://arxiv.org/pdf/2306.07971) <br> | BioNLP | 2023-06-13 | [Github](https://github.com/mbzuai-oryx/XrayGPT) | Project |
| LLaVA-Med | ![Star](https://img.shields.io/github/stars/microsoft/LLaVA-Med.svg?style=social&label=Star) <br> [**LLaVA-Med: Training a Large Language-and-Vision Assistant for Biomedicine in One Day**](https://arxiv.org/pdf/2306.00890) <br> | NeurIPS | 2023-06-01 | [Github](https://github.com/microsoft/LLaVA-Med) | Project |
| CXR-RePaiR-Gen | [**Retrieval Augmented Chest X-Ray Report Generation using OpenAI GPT models**](https://arxiv.org/pdf/2305.03660) <br> | MLHC | 2023-05-05 | Github | Project |

[<u><🎯Back to Top></u>](#head-content)

##### Science and math
|  Name  |  Title  |   Venue  |   Date   |   Code   |   Project   |
|:--------|:--------|:--------:|:--------:|:--------:|:--------:|
| MAVIS | ![Star](https://img.shields.io/github/stars/ZrrSkywalker/MAVIS.svg?style=social&label=Star) <br> [**MAVIS: Mathematical Visual Instruction Tuning**](https://arxiv.org/pdf/2407.08739) <br> | ECCV | 2024-11-01 | [Github](https://github.com/ZrrSkywalker/MAVIS) | Project |
| Math-LLaVA | ![Star](https://img.shields.io/github/stars/HZQ950419/Math-LLaVA.svg?style=social&label=Star) <br> [**Math-LLaVA: Bootstrapping Mathematical Reasoning for Multimodal Large Language Models**](https://arxiv.org/pdf/2406.172940) <br> | EMNLP | 2024-10-08 | [Github](https://github.com/HZQ950419/Math-LLaVA) | Project |
| MathVerse | ![Star](https://img.shields.io/github/stars/ZrrSkywalker/MathVerse.svg?style=social&label=Star) <br> [**MathVerse: Does Your Multi-modal LLM Truly See the Diagrams in Visual Math Problems?**](https://arxiv.org/pdf/2403.14624) <br> | ECCV | 2024-08-18 | [Github](https://github.com/ZrrSkywalker/MathVerse) | [Project](https://mathverse-cuhk.github.io/) |
| We-Math | ![Star](https://img.shields.io/github/stars/We-Math/We-Math.svg?style=social&label=Star) <br> [**We-Math: Does Your Large Multimodal Model Achieve Human-like Mathematical Reasoning?**](https://arxiv.org/pdf/2407.01284) <br> | ArXiv | 2024-07-01 | [Github](https://github.com/We-Math/We-Math) | [Project](https://we-math.github.io/) |
| CMMaTH | [**CMMaTH: A Chinese Multi-modal Math Skill Evaluation Benchmark for Foundation Models**](https://arxiv.org/pdf/2407.12023) <br> | ArXiv | 2024-06-28 | Github | Project |
| GeoEval | ![Star](https://img.shields.io/github/stars/GeoEval/GeoEval.svg?style=social&label=Star) <br> [**GeoEval: Benchmark for Evaluating LLMs and Multi-Modal Models on Geometry Problem-Solving**](https://arxiv.org/pdf/2402.10104) <br> | ACL | 2024-05-17 | [Github](https://github.com/GeoEval/GeoEval) | Project |
| FigurA11y | ![Star](https://img.shields.io/github/stars/allenai/figura11y.svg?style=social&label=Star) <br> [**FigurA11y: AI Assistance for Writing Scientific Alt Text**](https://dl.acm.org/doi/10.1145/3640543.3645212) <br> | IUI | 2024-04-05 | [Github](https://github.com/allenai/figura11y) | Project |
| MathVista | ![Star](https://img.shields.io/github/stars/lupantech/MathVista.svg?style=social&label=Star) <br> [**MathVista: Evaluating Mathematical Reasoning of Foundation Models in Visual Contexts**](https://arxiv.org/pdf/2310.02255) <br> | ICLR | 2024-01-21 | [Github](https://github.com/lupantech/MathVista) | [Project](https://mathvista.github.io/) |
| mPLUG-PaperOwl | ![Star](https://img.shields.io/github/stars/X-PLUG/mPLUG-DocOwl.svg?style=social&label=Star) <br> [**mPLUG-PaperOwl: Scientific Diagram Analysis with the Multimodal Large Language Model**](https://arxiv.org/pdf/2311.18248) <br> | ACM MM | 2024-01-09 | [Github](https://github.com/X-PLUG/mPLUG-DocOwl/tree/main/PaperOwl) | Project |
| G-LLaVA | ![Star](https://img.shields.io/github/stars/pipilurj/G-LLaVA.svg?style=social&label=Star) <br> [**G-LLaVA: Solving Geometric Problem with Multi-Modal Large Language Model**](https://arxiv.org/pdf/2312.11370) <br> | ArXiv | 2023-12-18 | [Github](https://github.com/pipilurj/G-LLaVA) | Project |
| T-SciQ | ![Star](https://img.shields.io/github/stars/T-SciQ/T-SciQ.svg?style=social&label=Star) <br> [**T-SciQ: Teaching Multimodal Chain-of-Thought Reasoning via Mixed Large Language Model Signals for Science Question Answering**](https://arxiv.org/pdf/2305.03453) <br> | AAAI | 2023-12-18 | [Github](https://github.com/T-SciQ/T-SciQ) | Project |
| ScienceQA | ![Star](https://img.shields.io/github/stars/lupantech/ScienceQA.svg?style=social&label=Star) <br> [**Learn to Explain: Multimodal Reasoning via Thought Chains for Science Question Answering**](https://arxiv.org/pdf/2209.09513) <br> | NeurIPS | 2022-10-17 | [Github](https://github.com/lupantech/ScienceQA) | [Project](https://scienceqa.github.io/) |

[<u><🎯Back to Top></u>](#head-content)

##### Graphics and UI
|  Name  |  Title  |   Venue  |   Date   |   Code   |   Project   |
|:--------|:--------|:--------:|:--------:|:--------:|:--------:|
| Graphist | ![Star](https://img.shields.io/github/stars/graphic-design-ai/graphist.svg?style=social&label=Star) <br> [**Graphic Design with Large Multimodal Model**](https://arxiv.org/pdf/2404.14368) <br> | ArXiv | 2024-04-22 | [Github](https://github.com/graphic-design-ai/graphist) | Project |
| Ferret-UI | [**Ferret-UI: Grounded Mobile UI Understanding with Multimodal LLMs**](https://arxiv.org/pdf/2404.05719) <br> | ECCV | 2024-04-08 | Github | Project |
| CogAgent | ![Star](https://img.shields.io/github/stars/THUDM/CogVLM.svg?style=social&label=Star) <br> [**CogAgent: A Visual Language Model for GUI Agents**](https://arxiv.org/pdf/2312.08914) <br> | CVPR | 2023-12-21 | [Github](https://github.com/THUDM/CogVLM) | Project |

[<u><🎯Back to Top></u>](#head-content)

##### Financial analysis
|  Name  |  Title  |   Venue  |   Date   |   Code   |   Project   |
|:--------|:--------|:--------:|:--------:|:--------:|:--------:|
| FinTral | [**FinTral: A Family of GPT-4 Level Multimodal Financial Large Language Models**](https://arxiv.org/pdf/2402.10986) <br> | ACL | 2024-06-14 | Github | Project |
| FinVis-GPT | ![Star](https://img.shields.io/github/stars/wwwadx/FinVis-GPT.svg?style=social&label=Star) <br> [**FinVis-GPT: A Multimodal Large Language Model for Financial Chart Analysis**](https://arxiv.org/pdf/2308.01430) <br> | ArXiv | 2023-07-31 | [Github](https://github.com/wwwadx/FinVis-GPT) | Project |

[<u><🎯Back to Top></u>](#head-content)

### Video-to-text

####  General domain
|  Name  |  Title  |   Venue  |   Date   |   Code   |   Project   |
|:--------|:--------|:--------:|:--------:|:--------:|:--------:|
| VideoBERT | ![Star](https://img.shields.io/github/stars/ammesatyajit/VideoBERT) <br> [VideoBERT: A joint model for video and language representation learning](https://arxiv.org/pdf/1904.01766) <br> | ICCV | 2019-04-03 | [Github](https://github.com/ammesatyajit/VideoBERT) | Project | 
| LaViLa | ![Star](https://img.shields.io/github/stars/facebookresearch/LaViLa) <br> [Learning Video Representations from Large Language Models](https://arxiv.org/abs/2212.04501) | CVPR | 2022-12-08 | [Github](https://github.com/facebookresearch/LaViLa) | [Project](https://facebookresearch.github.io/LaViLa/) |
| Vid2Seq | ![Star](https://img.shields.io/github/stars/google-research/scenic) <br> [Vid2Seq: Large-Scale Pretraining of a Visual Language Model for Dense Video Captioning](https://arxiv.org/pdf/2302.14115) | CVPR | 2023-02-27 | [Github](https://github.com/google-research/scenic/tree/main/scenic/projects/vid2seq) | [Project](https://antoyang.github.io/vid2seq.html) |
| Video-LLaMA |  ![Star](https://img.shields.io/github/stars/DAMO-NLP-SG/Video-LLaMA) <br> [Video-LLaMA: An Instruction-tuned Audio-Visual Language Model for Video Understanding](https://arxiv.org/pdf/2306.02858) | EMNLP | 2023-06-05 | [Github](https://github.com/DAMO-NLP-SG/Video-LLaMA) | Project |
| Video-LLaMA2 | ![Star](https://img.shields.io/github/stars/DAMO-NLP-SG/VideoLLaMA2) <br> [VideoLLaMA 2: Advancing Spatial-Temporal Modeling and Audio Understanding in Video-LLMs](https://arxiv.org/pdf/2406.07476) | ArXiv | 2024-06-11 | [Github](https://github.com/DAMO-NLP-SG/Video-LLaMA2) | Project | 
| LLaMA-VID | ![Star](https://img.shields.io/github/stars/dvlab-research/LLaMA-VID) <br> [LLaMA-VID: An Image is Worth 2 Tokens in Large Language Models](https://arxiv.org/abs/2311.17043) | ECCV | 2023-11-28 | [Github](https://github.com/dvlab-research/LLaMA-VID) | [Project](https://llama-vid.github.io/) |
| MiniGPT4-Video | ![Star](https://img.shields.io/github/stars/Vision-CAIR/MiniGPT4-video) <br> [MiniGPT4-video: Advancing multimodal llms for video understanding with interleaved visual-textual tokens](https://arxiv.org/pdf/2404.03413) | CVPR Workshop | 2024-04-04 | [Github](https://github.com/Vision-CAIR/MiniGPT4-video) | [Project](https://vision-cair.github.io/MiniGPT4-video/) |
| Goldfish | ![Star](https://img.shields.io/github/stars/Vision-CAIR/MiniGPT4-video) <br> [Goldfish: Vision-Language Understanding of Arbitrarily Long Videos](https://arxiv.org/abs/2407.12679) | ECCV | 2024-07-17 | [Github](https://vision-cair.github.io/Goldfish_website/) | [Project](https://vision-cair.github.io/Goldfish_website/) |
| ST-LLM | ![Star](https://img.shields.io/github/stars/TencentARC/ST-LLM) <br> [ST-LLM: Large language models are effective temporal learners](https://arxiv.org/pdf/2404.00308) | ECCV | 2024-03-30 | [Github](https://github.com/TencentARC/ST-LLM) | Project |
| LongVU | ![Star](https://img.shields.io/github/stars/Vision-CAIR/LongVU) <br> [LongVU: Spatiotemporal Adaptive Compression for Long Video-Language Understanding](https://arxiv.org/pdf/2410.17434) | ArXiv | 2024-10-22 | [Github](https://github.com/Vision-CAIR/LongVU) | [Project](https://vision-cair.github.io/LongVU) |

[<u><🎯Back to Top></u>](#head-content)
#### Video conversation
|  Name  |  Title  |   Venue  |   Date   |   Code   |   Project   |
|:--------|:--------|:--------:|:--------:|:--------:|:--------:|

[<u><🎯Back to Top></u>](#head-content)
#### Egocentric view
|  Name  |  Title  |   Venue  |   Date   |   Code   |   Project   |
|:--------|:--------|:--------:|:--------:|:--------:|:--------:|

[<u><🎯Back to Top></u>](#head-content)

## Vision-to-action

### Autonomous driving

#### Perception
|  Name  |  Title  |   Venue  |   Date   |   Code   |   Project   |
|:--------|:--------|:--------:|:--------:|:--------:|:--------:|
|DriveLM | ![Star](https://img.shields.io/github/stars/OpenDriveLab/DriveLM.svg?style=social&label=Star) <br> [**DriveLM: Driving with Graph Visual Question Answering**](https://arxiv.org/pdf/2312.14150) <br> | ECCV | 2024-7-17 | [Github](https://github.com/OpenDriveLab/DriveLM) | Project |
|Talk2BEV | ![Star](https://img.shields.io/github/stars/llmbev/talk2bev.svg?style=social&label=Star) <br> [**Talk2BEV: Language-enhanced Bird’s-eye View Maps for Autonomous Driving**](https://arxiv.org/pdf/2310.02251) <br> | ICRA | 2024-5-13 | [Github](https://github.com/llmbev/talk2bev) | [Project](https://llmbev.github.io/talk2bev/) |
|Nuscenes-QA | ![Star](https://img.shields.io/github/stars/qiantianwen/NuScenes-QA.svg?style=social&label=Star) <br> [**TNuScenes-QA: A Multi-Modal Visual Question Answering Benchmark for Autonomous Driving Scenario**](https://ojs.aaai.org/index.php/AAAI/article/view/28253/28499) <br> | AAAI | 2024-3-24 | [Github](https://github.com/qiantianwen/NuScenes-QA) | Project |
|DriveMLM | ![Star](https://img.shields.io/github/stars/OpenGVLab/DriveMLM.svg?style=social&label=Star) <br> [**DriveMLM: Aligning Multi-Modal Large Language Models with Behavioral Planning States for Autonomous Driving**](https://arxiv.org/pdf/2312.09245) <br> | ArXiv | 2023-12-25 | [Github](https://github.com/OpenGVLab/DriveMLM) | Project |
|LiDAR-LLM | [**LiDAR-LLM: Exploring the Potential of Large Language Models for 3D LiDAR Understanding**](https://arxiv.org/pdf/2312.14074v1) <br> | CoRR | 2023-12-21 | Github | [Project](https://sites.google.com/view/lidar-llm) |
|Dolphis | ![Star](https://img.shields.io/github/stars/SaFoLab-WISC/Dolphins.svg?style=social&label=Star) <br> [**Dolphins: Multimodal Language Model for Driving**](https://arxiv.org/abs/2312.00438) <br> | ArXiv | 2023-12-1 | [Github](https://github.com/SaFoLab-WISC/Dolphins) | [Project](https://vlm-driver.github.io/) |

[<u><🎯Back to Top></u>](#head-content)

#### Planning
|  Name  |  Title  |   Venue  |   Date   |   Code   |   Project   |
|:--------|:--------|:--------:|:--------:|:--------:|:--------:|
|DriveGPT4 | [**DriveGPT4: Interpretable End-to-End Autonomous Driving Via Large Language Model**](https://arxiv.org/pdf/2311.13549) <br> | RAL | 2024-8-7 | [Github](https://drive.google.com/drive/folders/1PsGL7ZxMMz1ZPDS5dZSjzjfPjuPHxVL5?usp=sharing) | [Project](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10629039) |
|SurrealDriver | ![Star](https://img.shields.io/github/stars/AIR-DISCOVER/Driving-Thinking-Dataset.svg?style=social&label=Star) <br> [**SurrealDriver: Designing LLM-powered Generative Driver Agent Framework based on Human Drivers’ Driving-thinking Data**](https://arxiv.org/pdf/2309.13193) <br> | ArXiv | 2024-7-22 | [Github](https://github.com/AIR-DISCOVER/Driving-Thinking-Dataset) | Project |
|DriveVLM | [**DriveVLM: The Convergence of Autonomous Driving and Large Vision-Language Models**](https://arxiv.org/abs/2402.12289) <br> | CoRL | 2024-6-25 | Github | [Project](https://tsinghua-mars-lab.github.io/DriveVLM/) |
|DiLu | ![Star](https://img.shields.io/github/stars/PJLab-ADG/DiLu.svg?style=social&label=Star) <br> [**DiLu: A Knowledge-Driven Approach to Autonomous Driving with Large Language Models**](https://arxiv.org/pdf/2309.16292) <br> | ICLR | 2024-2-22 | [Github](https://github.com/PJLab-ADG/DiLu) | [Project](https://pjlab-adg.github.io/DiLu/) |
|LMDrive | ![Star](https://img.shields.io/github/stars/opendilab/LMDrive.svg?style=social&label=Star) <br> [**LMDrive: Closed-Loop End-to-End Driving with Large Language Models**](https://arxiv.org/pdf/2309.13193) <br> | CVPR | 2023-12-21 | [Github](https://github.com/opendilab/LMDrive) | [Project](https://hao-shao.com/projects/lmdrive.html) |
|GPT-Driver | ![Star](https://img.shields.io/github/stars/PointsCoder/GPT-Driver.svg?style=social&label=Star) <br> [**DGPT-Driver: Learning to Drive with GPT**](https://arxiv.org/abs/2402.12289) <br> | NeurlPS Workshop | 2023-12-5 | [Github](https://github.com/PointsCoder/GPT-Driver) | [Project](https://pointscoder.github.io/projects/gpt_driver/index.html) |
|ADriver-I | [**ADriver-I: A General World Model for Autonomous Driving**](https://arxiv.org/pdf/2311.13549) <br> | ArXiv | 2023-11-22 | Github | Project |

[<u><🎯Back to Top></u>](#head-content)

#### Prediction
|  Name  |  Title  |   Venue  |   Date   |   Code   |   Project   |
|:--------|:--------|:--------:|:--------:|:--------:|:--------:|
|BEV-InMLLM | ![Star](https://img.shields.io/github/stars/xmed-lab/NuInstruct.svg?style=social&label=Star) <br> [**Holistic autonomous driving understanding by bird’s-eye-view injected multi-Modal large model**](https://openaccess.thecvf.com/content/CVPR2024/papers/Ding_Holistic_Autonomous_Driving_Understanding_by_Birds-Eye-View_Injected_Multi-Modal_Large_Models_CVPR_2024_paper.pdf) <br> | CVPR | 2024-1-2 | [Github](https://github.com/xmed-lab/NuInstruct) | Project |
|Prompt4Driving | ![Star](https://img.shields.io/github/stars/wudongming97/Prompt4Driving.svg?style=social&label=Star) <br> [**Language Prompt for Autonomous Driving**](https://arxiv.org/pdf/2309.04379) <br> | ArXiv | 2023-9-8 | [Github](https://github.com/wudongming97/Prompt4Driving) | Project |

[<u><🎯Back to Top></u>](#head-content)

### Embodied AI
#### Perception
|  Name  |  Title  |   Venue  |   Date   |   Code   |   Project   |
|:--------|:--------|:--------:|:--------:|:--------:|:--------:|
|Wonderful-Team| ![Star](https://img.shields.io/github/stars/wonderful-team-robotics/wonderful_team_robotics.svg?style=social&label=Star) <br> [**Wonderful Team: Zero-Shot Physical Task Planning with Visual LLMs**](https://arxiv.org/pdf/2407.19094) <br> | ArXiv | 2024-12-4 | [Github](https://github.com/wonderful-team-robotics/wonderful_team_robotics) | [Project](https://wonderful-team-robotics.github.io) |
|AffordanceLLM| ![Star](https://img.shields.io/github/stars/wj-on-un/AffordanceLLM_implementation.svg?style=social&label=Star) <br> [**AffordanceLLM: Grounding Affordance from Vision Language Models**](https://arxiv.org/abs/2401.06341) <br> | CVPR | 2024-4-17 | [Github](https://github.com/wj-on-un/AffordanceLLM_implementation) | [Project](https://jasonqsy.github.io/AffordanceLLM/) |
|3DVisProg| ![Star](https://img.shields.io/github/stars/CurryYuan/ZSVG3D.svg?style=social&label=Star) <br> [**Visual Programming for Zero-shot Open-Vocabulary 3D Visual Grounding**](https://openaccess.thecvf.com/content/CVPR2024/papers/Yuan_Visual_Programming_for_Zero-shot_Open-Vocabulary_3D_Visual_Grounding_CVPR_2024_paper.pdf) <br> | CVPR | 2024-3-23 | [Github](https://github.com/CurryYuan/ZSVG3D) | [Project](https://curryyuan.github.io/ZSVG3D/) |
|WREPLAN| [**REPLAN: Robotic Replanning with Perception and Language Models**](https://arxiv.org/pdf/2401.04157) <br> | ArXiv | 2024-2-20 | Github| [Project](https://replan-lm.github.io/replan.github.io/) |
|PaLM-E| [**PaLM-E: An Embodied Multimodal Language Model**](https://arxiv.org/pdf/2303.03378) <br> | ICML | 2023-3-6 | Github | [Project](https://palm-e.github.io) |

[<u><🎯Back to Top></u>](#head-content)

#### Manipulation
|  Name  |  Title  |   Venue  |   Date   |   Code   |   Project   |
|:--------|:--------|:--------:|:--------:|:--------:|:--------:|
|OpenVLA| ![Star](https://img.shields.io/github/stars/openvla/openvla.svg?style=social&label=Star) <br> [**OpenVLA: An Open-Source Vision-Language-Action Model**](https://arxiv.org/pdf/2406.09246) <br> | ArXiv | 2024-9-5 | [Github](https://github.com/openvla/openvla) | [Project](https://openvla.github.io) |
|LLARVA| ![Star](https://img.shields.io/github/stars/Dantong88/LLARVA.svg?style=social&label=Star) <br> [**LLARVA: Vision-Action Instruction Tuning Enhances Robot Learning**](https://arxiv.org/pdf/2406.11815) <br> | CoRL | 2024-6-17 | [Github](https://github.com/Dantong88/LLARVA) | [Project](https://llarva24.github.io) |
|RT-X| ![Star](https://img.shields.io/github/stars/google-deepmind/open_x_embodiment.svg?style=social&label=Star) <br>  [**Open X-Embodiment: Robotic Learning Datasets and RT-X Models**](https://arxiv.org/pdf/2310.08864) <br> | ArXiv | 2024-6-1 | [Github](https://github.com/google-deepmind/open_x_embodiment) | [Project](https://robotics-transformer2.github.io) |
|RoboFlamingo| [**Vision-Language Foundation Models as Effective Robot Imitators**](https://arxiv.org/pdf/2311.01378) <br> | ICLR | 2024-2-5 | Github | Project |
|VoxPoser| ![Star](https://img.shields.io/github/stars/huangwl18/VoxPoser.svg?style=social&label=Star) <br>  [**VoxPoser: Composable 3D Value Maps for Robotic Manipulation with Language Models**](https://arxiv.org/pdf/2307.05973) <br> | CoRL | 2023-11-2 | [Github](https://github.com/huangwl18/VoxPoser) | [Project](https://voxposer.github.io) |
|ManipLLM| ![Star](https://img.shields.io/github/stars/clorislili/ManipLLM.svg?style=social&label=Star) <br>  [**ManipLLM: Embodied Multimodal Large Language Model for Object-Centric Robotic Manipulation**](https://arxiv.org/pdf/2312.16217) <br> | CVPR | 2023-12-24 | [Github](https://github.com/clorislili/ManipLLM) | [Project](https://sites.google.com/view/manipllm) |
|RT-2| [**RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control**](https://robotics-transformer2.github.io/assets/rt2.pdf) <br> | ArXiv | 2023-7-28 | Github | [Project](https://robotics-transformer-x.github.io) |
|Instruct2Act| ![Star](https://img.shields.io/github/stars/OpenGVLab/Instruct2Act.svg?style=social&label=Star) <br> [**Instruct2Act: Mapping Multi-modality Instructions to Robotic Actions with Large Language Model**](https://arxiv.org/pdf/2305.11176) <br> | ArXiv | 2023-5-24 | [Github](https://github.com/OpenGVLab/Instruct2Act) | Project |

[<u><🎯Back to Top></u>](#head-content)

#### Planning
|  Name  |  Title  |   Venue  |   Date   |   Code   |   Project   |
|:--------|:--------|:--------:|:--------:|:--------:|:--------:|
|LLaRP| ![Star](https://img.shields.io/github/stars/apple/ml-llarp.svg?style=social&label=Star) <br> [**Large Language Models as Generalizable Policies for Embodied Tasks**](https://openaccess.thecvf.com/content/CVPR2024/papers/Yuan_Visual_Programming_for_Zero-shot_Open-Vocabulary_3D_Visual_Grounding_CVPR_2024_paper.pdf) <br> | ICLR | 2024-4-16 | [Github](https://github.com/apple/ml-llarp) | [Project](https://llm-rl.github.io) |
|MP5| ![Star](https://img.shields.io/github/stars/IranQin/MP5.svg?style=social&label=Star) <br> [**MP5: A Multi-modal Open-ended Embodied System in Minecraft via Active Perception**](https://arxiv.org/pdf/2312.07472) <br> | CVPR | 2024-3-24 | [Github](https://github.com/IranQin/MP5) | [Project](https://iranqin.github.io/MP5.github.io/) |
|LL3DA| ![Star](https://img.shields.io/github/stars/Open3DA/LL3DA.svg?style=social&label=Star) <br> [**LL3DA: Visual Interactive Instruction Tuning for Omni-3D Understanding, Reasoning, and Planning**](https://arxiv.org/pdf/2311.18651) <br> | CVPR | 2023-11-30 | [Github](https://github.com/Open3DA/LL3DA) | [Project](https://ll3da.github.io/) |
|EmbodiedGPT| ![Star](https://img.shields.io/github/stars/EmbodiedGPT/EmbodiedGPT_Pytorch.svg?style=social&label=Star) <br> [**EmbodiedGPT: Vision-Language Pre-Training via Embodied Chain of Thought**](https://openaccess.thecvf.com/content/CVPR2024/papers/Yuan_Visual_Programming_for_Zero-shot_Open-Vocabulary_3D_Visual_Grounding_CVPR_2024_paper.pdf) <br> | NeurlPS | 2023-11-2 | [Github](https://github.com/EmbodiedGPT/EmbodiedGPT_Pytorch) | [Project](https://embodiedgpt.github.io) |
|ELLM| ![Star](https://img.shields.io/github/stars/yuqingd/ellm.svg?style=social&label=Star) <br> [**Guiding Pretraining in Reinforcement Learning with Large Language Models**](https://proceedings.mlr.press/v202/du23f/du23f.pdf) <br> | ICML | 2023-9-15 | [Github](https://github.com/yuqingd/ellm) | Project |
|3D-LLM| ![Star](https://img.shields.io/github/stars/UMass-Foundation-Model/3D-LLM.svg?style=social&label=Star) <br> [**3D-LLM: Injecting the 3D World into Large Language Models**](https://arxiv.org/abs/2307.12981) <br> | NeurlPS | 2023-7-24 | [Github](https://github.com/UMass-Foundation-Model/3D-LLM) | [Project](hhttps://vis-www.cs.umass.edu/3dllm/) |
|NLMap| ![Star](https://img.shields.io/github/stars/ericrosenbrown/nlmap_spot.svg?style=social&label=Star) <br> [**Open-vocabulary Queryable Scene Representations for Real World Planning**](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10161534) <br> | ICRA | 2023-7-4 | [Github](https://github.com/ericrosenbrown/nlmap_spot) | [Project](https://nlmap-saycan.github.io) |

[<u><🎯Back to Top></u>](#head-content)

#### Navigation
|  Name  |  Title  |   Venue  |   Date   |   Code   |   Project   |
|:--------|:--------|:--------:|:--------:|:--------:|:--------:|
|ConceptGraphs| ![Star](https://img.shields.io/github/stars/concept-graphs/concept-graphs.svg?style=social&label=Star) <br> [**ConceptGraphs: Open-Vocabulary 3D Scene Graphs for Perception and Planning**](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10610243) <br> | ICRA | 2024-5-13 | [Github](https://github.com/concept-graphs/concept-graphs) | [Project](https://concept-graphs.github.io) |
|RILA| [**RILA: Reflective and Imaginative Language Agent for Zero-Shot Semantic Audio-Visual Navigation**](https://openaccess.thecvf.com/content/CVPR2024/papers/Yang_RILA_Reflective_and_Imaginative_Language_Agent_for_Zero-Shot_Semantic_Audio-Visual_CVPR_2024_paper.pdf) <br> | CVPR | 2024-4-27 | Github | Project |
|EMMA| ![Star](https://img.shields.io/github/stars/stevenyangyj/Emma-Alfworld.svg?style=social&label=Star) <br> [**Embodied Multi-Modal Agent trained by an LLM from a Parallel TextWorld**](https://arxiv.org/abs/2401.08577) <br> | CVPR | 2024-3-29 | [Github](https://github.com/stevenyangyj/Emma-Alfworld) | Project |
|VLN-VER| ![Star](https://img.shields.io/github/stars/DefaultRui/VLN-VER.svg?style=social&label=Star) <br> [**Volumetric Environment Representation for Vision-Language Navigation**](https://arxiv.org/pdf/2403.14158) <br> | CVPR | 2024-3-24 | [Github](https://github.com/DefaultRui/VLN-VER) | Project |
|MultiPLY| ![Star](https://img.shields.io/github/stars/UMass-Foundation-Model/MultiPLY.svg?style=social&label=Star) <br> [**MultiPLY: A Multisensory Object-Centric Embodied Large Language Model in 3D World**](https://arxiv.org/abs/2401.08577) <br> | CVPR | 2024-1-16 | [Github](https://github.com/UMass-Foundation-Model/MultiPLY) | [Project](https://vis-www.cs.umass.edu/multiply/) |

[<u><🎯Back to Top></u>](#head-content)

### Automated tool management
|  Name  |  Title  |   Venue  |   Date   |   Code   |   Project   |
|:--------|:--------|:--------:|:--------:|:--------:|:--------:|
| TROVE      | ![Star](https://img.shields.io/github/stars/zorazrw/trove.svg?style=social&label=Star) <br> [**TROVE: Inducing Verifiable and Efficient Toolboxes for Solving Programmatic Tasks**](https://arxiv.org/abs/2401.12869) | arXiv | 2024-01-23 | [Github](https://github.com/zorazrw/trove) | Project |
| Tool-LMM   | ![Star](https://img.shields.io/github/stars/MLLM-Tool/MLLM-Tool.svg?style=social&label=Star) <br> [**Tool-LMM: A Large Multi-Modal Model for Tool Agent Learning**](https://arxiv.org/abs/2401.10727) | arXiv | 2024-1-19 | [Github](https://github.com/MLLM-Tool/MLLM-Tool) | Project |
| CLOVA      | ![Star](https://img.shields.io/github/stars/clova-tool/CLOVA-tool.svg?style=social&label=Star) <br> [**CLOVA: A Closed-loop Visual Assistant with Tool Usage and Update**](https://arxiv.org/abs/2312.10908) | CVPR | 2023-12-18 | [Github](https://github.com/clova-tool/CLOVA-tool) | [Project](https://clova-tool.github.io/) |
| CRAFT      | ![Star](https://img.shields.io/github/stars/lifan-yuan/CRAFT.svg?style=social&label=Star) <br> [**CRAFT: Customizing LLMs by Creating and Retrieving from Specialized Toolsets**](https://arxiv.org/abs/2309.17428) | arXiv | 2023-9-29 | [Github](https://github.com/lifan-yuan/CRAFT) | Project |
| Confucius  | ![Star](https://img.shields.io/github/stars/shizhl/CTL.svg?style=social&label=Star) <br> [**Confucius: Iterative tool learning from introspection feedback by easy-to-difficult curriculum**](https://arxiv.org/abs/2308.14034) | AAAI | 2023-8-27 | [Github](https://github.com/shizhl/CTL) | Project |
| AVIS       | [**Avis: Autonomous visual information seeking with large language model agent**](https://arxiv.org/abs/2306.08129) | NeurIPS | 2023-6-13 | Github | Project |
| GPT4Tools  | ![Star](https://img.shields.io/github/stars/StevenGrove/GPT4Tools.svg?style=social&label=Star) <br> [**GPT4Tools: Teaching large language model to use tools via self-instruction**](https://arxiv.org/abs/2305.18752) | NeurIPS | 2023-5-30 | [Github](https://github.com/StevenGrove/GPT4Tools) | Project |
| ToolkenGPT | ![Star](https://img.shields.io/github/stars/Ber666/ToolkenGPT.svg?style=social&label=Star) <br> [**ToolkenGPT: Augmenting frozen language models with massive tools via tool embeddings**](https://arxiv.org/abs/2305.11554) | NeurIPS | 2023-5-19 | [Github](https://github.com/Ber666/ToolkenGPT) | Project |
| Chameleon  | ![Star](https://img.shields.io/github/stars/lupantech/chameleon-llm.svg?style=social&label=Star) <br> [**Chameleon: Plug-and-play compositional reasoning with large language models**](https://arxiv.org/abs/2304.09842) | NeurIPS | 2023-4-19 | [Github](https://github.com/lupantech/chameleon-llm) | [Project](https://chameleon-llm.github.io/) |
| HuggingGPT | ![Star](https://img.shields.io/github/stars/microsoft/JARVIS.svg?style=social&label=Star) <br> [**HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in Hugging Face**](https://arxiv.org/abs/2303.17580) | NeurIPS | 2023-3-30 | [Github](https://github.com/microsoft/JARVIS) | Project |
| TaskMatrix.AI | [**TaskMatrix.AI: Completing tasks by connecting foundation models with millions of APIs**](https://arxiv.org/abs/2303.16434) | Intelligent Computing (AAAS) | 2023-3-29 | Github | Projecct |
| MM-ReACT   | ![Star](https://img.shields.io/github/stars/microsoft/MM-REACT.svg?style=social&label=Star) <br> [**MM-ReACT: Prompting ChatGPT for Multimodal Reasoning and Action**](https://arxiv.org/abs/2303.11381) | arXiv | 2023-3-20 | [Github](https://github.com/microsoft/MM-REACT) | [Project](https://multimodal-react.github.io/) |
| ViperGPT   | ![Star](https://img.shields.io/github/stars/cvlab-columbia/viper.svg?style=social&label=Star) <br> [**ViperGPT: Visual Inference via Python Execution for Reasoning**](https://arxiv.org/abs/2303.08128) | ICCV | 2023-3-14 | [Github](https://github.com/cvlab-columbia/viper) | Project |
| MIND’S EYE | [**MIND’S EYE: GROUNDED LANGUAGE MODEL REASONING THROUGH SIMULATION**](https://arxiv.org/abs/2210.05359) | arXiv | 2022-10-11 | GitHub | Project |

[<u><🎯Back to Top></u>](#head-content)

## Text-to-vision
|  Name  |  Title  |   Venue  |   Date   |   Code   |   Project   |
|:--------|:--------|:--------:|:--------:|:--------:|:--------:|

[<u><🎯Back to Top></u>](#head-content)

### Text-to-image
|  Name  |  Title  |   Venue  |   Date   |   Code   |   Project   |
|:--------|:--------|:--------:|:--------:|:--------:|:--------:|

[<u><🎯Back to Top></u>](#head-content)

### Text-to-3D
|  Name  |  Title  |   Venue  |   Date   |   Code   |   Project   |
|:--------|:--------|:--------:|:--------:|:--------:|:--------:|

[<u><🎯Back to Top></u>](#head-content)

### Text-to-video
|  Name  |  Title  |   Venue  |   Date   |   Code   |   Project   |
|:--------|:--------|:--------:|:--------:|:--------:|:--------:|

[<u><🎯Back to Top></u>](#head-content)

## Other applications
### Face
|  Name  |  Title  |   Venue  |   Date   |   Code   |   Project   |
|:--------|:--------|:--------:|:--------:|:--------:|:--------:|
| EmoLA   | ![Star](https://img.shields.io/github/stars/JackYFL/EmoLA.svg?style=social&label=Star) <br> [**Facial Affective Behavior Analysis with Instruction Tuning**](https://arxiv.org/pdf/2404.05052) | ECCV | 2024-7-12 | [Github](https://github.com/JackYFL/EmoLA) | [Project](https://johnx69.github.io/FABA/) |

[<u><🎯Back to Top></u>](#head-content)

<!-- ## <span id="head8"> Contributors </span>
Thanks to all the contributors! -->

