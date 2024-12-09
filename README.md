# Awesome Large Vision Language Models (VLLMs)


## 🔥🔥🔥 Visual Large Language Models for Generalized and Specialized Applications

<p align="center">
    <img src="assets/VLM_revolution.png" width="90%" height="90%">
</p>


## 📢 News

🚀 **What's New in This Update**:

2024.12.30: We release our VLLM application paper list repo!
<span id="head-content"><font size=5><center><b> Table of Contents </b> </center></font></span>
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
| ![Star](https://img.shields.io/github/stars/Yutong-Zhou-cv/Awesome-Text-to-Image.svg?style=social&label=Star) <br> [**Vision + Language Applications: A Survey**](https://openaccess.thecvf.com/content/CVPR2023W/GCV/papers/Zhou_Vision__Language_Applications_A_Survey_CVPRW_2023_paper.pdf) <br> | CVPRW | 2023-5-24 | Github | [Project](https://github.com/Yutong-Zhou-cv/Awesome-Text-to-Image) |
| [**Vision-and-Language Pretrained Models: A Survey**](https://arxiv.org/pdf/2204.07356.pdf) <br> | IJCAI (survey track) | 2022-5-3 | Github | Project |

### MLLM surveys
|  Title  |   Venue  |   Date   |   Code   |   Project   |
|:--------|:--------:|:--------:|:--------:|:--------:|
| ![Star](https://img.shields.io/github/stars/BradyFU/Awesome-Multimodal-Large-Language-Models.svg?style=social&label=Star) <br> [**A Survey on Multimodal Large Language Models**](https://arxiv.org/pdf/2306.13549.pdf) <br> | T-PAMI | 2024-11-29 | Github | [Project](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models) |
| ![Star](https://img.shields.io/github/stars/BradyFU/Awesome-Multimodal-Large-Language-Models.svg?style=social&label=Star) <br> [**MME-Survey: A Comprehensive Survey on Evaluation of Multimodal LLMs**](https://arxiv.org/pdf/2411.15296) <br> | AriXv | 2024-11-22 | Github | [Project](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models) |
| [**A Survey on Multimodal Large Language Models**](https://academic.oup.com/nsr/advance-article-pdf/doi/10.1093/nsr/nwae403/60676453/nwae403.pdf) <br> | National Science Review | 2024-11-12 | Github | Project |
| [**Video Understanding with Large Language Models: A Survey**](https://arxiv.org/pdf/2312.17432) <br> | ArXiv | 2024-6-24 | Github | [Project](https://github.com/yunlong10/Awesome-LLMs-for-Video-Understanding) |
| ![Star](https://img.shields.io/github/stars/yunlong10/Awesome-LLMs-for-Video-Understanding.svg?style=social&label=Star) <br> [**The Revolution of Multimodal Large Language Models: A Survey**](https://arxiv.org/pdf/2402.12451) <br> | ArXiv | 2024-6-6 | Github | Project |
| ![Star](https://img.shields.io/github/stars/lhanchao777/LVLM-Hallucinations-Survey.svg?style=social&label=Star) <br> [**A Survey on Hallucination in Large Vision-Language Models**](https://arxiv.org/pdf/2402.00253) <br> | ArXiv | 2024-5-6 | Github | [Project](https://github.com/lhanchao777/LVLM-Hallucinations-Survey) |
| [**Exploring the Frontier of Vision-Language Models: A Survey of Current Methodologies and Future Directions**](https://arxiv.org/pdf/2404.07214) <br> | ArXiv | 2024-4-12 | Github | Project |
| ![Star](https://img.shields.io/github/stars/MM-LLMs/mm-llms.github.io.svg?style=social&label=Star) <br> [**MM-LLMs: Recent Advances in MultiModal Large Language Models**](https://arxiv.org/pdf/2401.13601v4) <br> | ArXiv | 2024-2-20 | Github | [Project](https://mm-llms.github.io/) |
| [**Exploring the Reasoning Abilities of Multimodallarge Language Models (mllms): a Comprehensive survey on Emerging Trends in Multimodal Reasonings**](https://arxiv.org/pdf/2401.06805) <br> | AriXv | 2024-1-18 | Github | Project |
| [**Visual Instruction Tuning towards General-Purpose Multimodal Model: A Survey**](https://arxiv.org/pdf/2312.16602) <br> | ArXiv | 2023-12-27 | Github | Project |
| [**Multimodal Large Language Models: A Survey**](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10386743) <br> | BigData | 2023-12-15 | Github | Project |

## Vision-to-text
### Image-to-text
#### General domain
##### General ability
|  Model  |  Title  |   Venue  |   Date   |   Code   |   Project   |
|:--------|:--------|:--------:|:--------:|:--------:|:--------:|

[<u><🎯Back to Top></u>](#head-content)

##### REC
|  Model  |  Title  |   Venue  |   Date   |   Code   |   Project   |
|:--------|:--------|:--------:|:--------:|:--------:|:--------:|

[<u><🎯Back to Top></u>](#head-content)

##### RES
|  Model  |  Title  |   Venue  |   Date   |   Code   |   Project   |
|:--------|:--------|:--------:|:--------:|:--------:|:--------:|

[<u><🎯Back to Top></u>](#head-content)

##### OCR
|  Model  |  Title  |   Venue  |   Date   |   Code   |   Project   |
|:--------|:--------|:--------:|:--------:|:--------:|:--------:|

[<u><🎯Back to Top></u>](#head-content)

##### Retrieval
|  Model  |  Title  |   Venue  |   Date   |   Code   |   Project   |
|:--------|:--------|:--------:|:--------:|:--------:|:--------:|

[<u><🎯Back to Top></u>](#head-content)

#### VLLM + X
##### Remote sensing
|  Model  |  Title  |   Venue  |   Date   |   Code   |   Project   |
|:--------|:--------|:--------:|:--------:|:--------:|:--------:|

[<u><🎯Back to Top></u>](#head-content)

##### Medical
|  Model  |  Title  |   Venue  |   Date   |   Code   |   Project   |
|:--------|:--------|:--------:|:--------:|:--------:|:--------:|

[<u><🎯Back to Top></u>](#head-content)

##### Science and math
|  Model  |  Title  |   Venue  |   Date   |   Code   |   Project   |
|:--------|:--------|:--------:|:--------:|:--------:|:--------:|

[<u><🎯Back to Top></u>](#head-content)

##### Graphics and UI
|  Model  |  Title  |   Venue  |   Date   |   Code   |   Project   |
|:--------|:--------|:--------:|:--------:|:--------:|:--------:|

[<u><🎯Back to Top></u>](#head-content)

##### Financial analysis
|  Model  |  Title  |   Venue  |   Date   |   Code   |   Project   |
|:--------|:--------|:--------:|:--------:|:--------:|:--------:|

[<u><🎯Back to Top></u>](#head-content)

### Video-to-text

[<u><🎯Back to Top></u>](#head-content)
####  General domain
|  Model  |  Title  |   Venue  |   Date   |   Code   |   Project   |
|:--------|:--------|:--------:|:--------:|:--------:|:--------:|

[<u><🎯Back to Top></u>](#head-content)
#### Video conversation
|  Model  |  Title  |   Venue  |   Date   |   Code   |   Project   |
|:--------|:--------|:--------:|:--------:|:--------:|:--------:|

[<u><🎯Back to Top></u>](#head-content)
#### Egocentric view
|  Model  |  Title  |   Venue  |   Date   |   Code   |   Project   |
|:--------|:--------|:--------:|:--------:|:--------:|:--------:|

[<u><🎯Back to Top></u>](#head-content)

## Vision-to-action

### Autonomous driving

#### Perception
|  Model  |  Title  |   Venue  |   Date   |   Code   |   Project   |
|:--------|:--------|:--------:|:--------:|:--------:|:--------:|
|DriveLM | ![Star](https://img.shields.io/github/stars/OpenDriveLab/DriveLM.svg?style=social&label=Star) <br> [**DriveLM: Driving with Graph Visual Question Answering**](https://arxiv.org/pdf/2312.14150) <br> | ECCV | 2024-7-17 | [Github](https://github.com/OpenDriveLab/DriveLM) | Project |
|Talk2BEV | ![Star](https://img.shields.io/github/stars/llmbev/talk2bev.svg?style=social&label=Star) <br> [**Talk2BEV: Language-enhanced Bird’s-eye View Maps for Autonomous Driving**](https://arxiv.org/pdf/2310.02251) <br> | ICRA | 2024-5-13 | [Github](https://github.com/llmbev/talk2bev) | [Project](https://llmbev.github.io/talk2bev/) |
|Nuscenes-QA | ![Star](https://img.shields.io/github/stars/qiantianwen/NuScenes-QA.svg?style=social&label=Star) <br> [**TNuScenes-QA: A Multi-Modal Visual Question Answering Benchmark for Autonomous Driving Scenario**](https://ojs.aaai.org/index.php/AAAI/article/view/28253/28499) <br> | AAAI | 2024-3-24 | [Github](https://github.com/qiantianwen/NuScenes-QA) | Project |
|DriveMLM | ![Star](https://img.shields.io/github/stars/OpenGVLab/DriveMLM.svg?style=social&label=Star) <br> [**DriveMLM: Aligning Multi-Modal Large Language Models with Behavioral Planning States for Autonomous Driving**](https://arxiv.org/pdf/2312.09245) <br> | ArXiv | 2023-12-25 | [Github](https://github.com/OpenGVLab/DriveMLM) | Project |
|LiDAR-LLM | [**LiDAR-LLM: Exploring the Potential of Large Language Models for 3D LiDAR Understanding**](https://arxiv.org/pdf/2312.14074v1) <br> | CoRR | 2023-12-21 | Github | [Project](https://sites.google.com/view/lidar-llm) |
|Dolphis | ![Star](https://img.shields.io/github/stars/SaFoLab-WISC/Dolphins.svg?style=social&label=Star) <br> [**Dolphins: Multimodal Language Model for Driving**](https://arxiv.org/abs/2312.00438) <br> | ArXiv | 2023-12-1 | [Github](https://github.com/SaFoLab-WISC/Dolphins) | [Project](https://vlm-driver.github.io/) |

[<u><🎯Back to Top></u>](#head-content)

#### Planning
|  Model  |  Title  |   Venue  |   Date   |   Code   |   Project   |
|:--------|:--------|:--------:|:--------:|:--------:|:--------:|
|LanguageMPC | [**LanguageMPC: Large Language Models as Decision Makers for Autonomous Driving**](https://arxiv.org/pdf/2310.03026) <br> | ArXiv | 2023-10-13 | Github | [Project](https://sites.google.com/view/llm-ad) |

[<u><🎯Back to Top></u>](#head-content)

#### Prediction
|  Model  |  Title  |   Venue  |   Date   |   Code   |   Project   |
|:--------|:--------|:--------:|:--------:|:--------:|:--------:|

[<u><🎯Back to Top></u>](#head-content)

### Embodied AI
#### Perception
|  Model  |  Title  |   Venue  |   Date   |   Code   |   Project   |
|:--------|:--------|:--------:|:--------:|:--------:|:--------:|

[<u><🎯Back to Top></u>](#head-content)

#### Manipulation
|  Model  |  Title  |   Venue  |   Date   |   Code   |   Project   |
|:--------|:--------|:--------:|:--------:|:--------:|:--------:|

[<u><🎯Back to Top></u>](#head-content)

#### Planning
|  Model  |  Title  |   Venue  |   Date   |   Code   |   Project   |
|:--------|:--------|:--------:|:--------:|:--------:|:--------:|

[<u><🎯Back to Top></u>](#head-content)

#### Navigation
|  Model  |  Title  |   Venue  |   Date   |   Code   |   Project   |
|:--------|:--------|:--------:|:--------:|:--------:|:--------:|

[<u><🎯Back to Top></u>](#head-content)

### Automated tool management
| Model      | Title                                                                                                                                          | Venue                          |  Date  |   Code    |  Project                           |
|:-----------|:-----------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------:|:------:|:---------:|:-----------------------------------:|
| TROVE      | ![Star](https://img.shields.io/github/stars/zorazrw/trove.svg?style=social&label=Star) <br> [**TROVE: Inducing Verifiable and Efficient Toolboxes for Solving Programmatic Tasks**](https://arxiv.org/abs/2401.12869) | arXiv | 2024 | [Github](https://github.com/zorazrw/trove) | Project |
| Tool-LMM   | ![Star](https://img.shields.io/github/stars/MLLM-Tool/MLLM-Tool.svg?style=social&label=Star) <br> [**Tool-LMM: A Large Multi-Modal Model for Tool Agent Learning**](https://arxiv.org/abs/2401.10727) | arXiv | 2024 | [Github](https://github.com/MLLM-Tool/MLLM-Tool) | Project |
| CLOVA      | ![Star](https://img.shields.io/github/stars/clova-tool/CLOVA-tool.svg?style=social&label=Star) <br> [**CLOVA: A Closed-loop Visual Assistant with Tool Usage and Update**](https://arxiv.org/abs/2312.10908) | CVPR | 2024 | [Github](https://github.com/clova-tool/CLOVA-tool) | [Project](https://clova-tool.github.io/) |
| CRAFT      | ![Star](https://img.shields.io/github/stars/lifan-yuan/CRAFT.svg?style=social&label=Star) <br> [**CRAFT: Customizing LLMs by Creating and Retrieving from Specialized Toolsets**](https://arxiv.org/abs/2309.17428) | arXiv | 2023 | [Github](https://github.com/lifan-yuan/CRAFT) | Project |
| Confucius  | ![Star](https://img.shields.io/github/stars/shizhl/CTL.svg?style=social&label=Star) <br> [**Confucius: Iterative tool learning from introspection feedback by easy-to-difficult curriculum**](https://arxiv.org/abs/2308.14034) | AAAI | 2023 | [Github](https://github.com/shizhl/CTL) | Project |
| AVIS       | [**Avis: Autonomous visual information seeking with large language model agent**](https://arxiv.org/abs/2306.08129) | NeurIPS | 2024 | N/A | N/A |
| GPT4Tools  | ![Star](https://img.shields.io/github/stars/StevenGrove/GPT4Tools.svg?style=social&label=Star) <br> [**GPT4Tools: Teaching large language model to use tools via self-instruction**](https://arxiv.org/abs/2305.18752) | NeurIPS | 2024 | [Github](https://github.com/StevenGrove/GPT4Tools) | Project |
| ToolkenGPT | ![Star](https://img.shields.io/github/stars/Ber666/ToolkenGPT.svg?style=social&label=Star) <br> [**ToolkenGPT: Augmenting frozen language models with massive tools via tool embeddings**](https://arxiv.org/abs/2305.11554) | NeurIPS | 2024 | [Github](https://github.com/Ber666/ToolkenGPT) | Project |
| Chameleon  | ![Star](https://img.shields.io/github/stars/lupantech/chameleon-llm.svg?style=social&label=Star) <br> [**Chameleon: Plug-and-play compositional reasoning with large language models**](https://arxiv.org/abs/2304.09842) | NeurIPS | 2024 | [Github](https://github.com/lupantech/chameleon-llm) | [Project](https://chameleon-llm.github.io/) |
| HuggingGPT | ![Star](https://img.shields.io/github/stars/microsoft/JARVIS.svg?style=social&label=Star) <br> [**HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in Hugging Face**](https://arxiv.org/abs/2303.17580) | NeurIPS | 2024 | [Github](https://github.com/microsoft/JARVIS) | Project |
| TaskMatrix.AI | [**TaskMatrix.AI: Completing tasks by connecting foundation models with millions of APIs**](https://arxiv.org/abs/2303.16434) | Intelligent Computing (AAAS) | 2024 | N/A | N/A |
| MM-ReACT   | ![Star](https://img.shields.io/github/stars/microsoft/MM-REACT.svg?style=social&label=Star) <br> [**MM-ReACT: Prompting ChatGPT for Multimodal Reasoning and Action**](https://arxiv.org/abs/2303.11381) | arXiv | 2023 | [Github](https://github.com/microsoft/MM-REACT) | [Project](https://multimodal-react.github.io/) |
| ViperGPT   | ![Star](https://img.shields.io/github/stars/cvlab-columbia/viper.svg?style=social&label=Star) <br> [**ViperGPT: Visual Inference via Python Execution for Reasoning**](https://arxiv.org/abs/2303.08128) | ICCV | 2023 | [Github](https://github.com/cvlab-columbia/viper) | Project |
| MIND’S EYE | [**MIND’S EYE: GROUNDED LANGUAGE MODEL REASONING THROUGH SIMULATION**](https://arxiv.org/abs/2210.05359) | arXiv | 2023 | N/A | N/A |




[<u><🎯Back to Top></u>](#head-content)

## Text-to-vision
|  Model  |  Title  |   Venue  |   Date   |   Code   |   Project   |
|:--------|:--------|:--------:|:--------:|:--------:|:--------:|

[<u><🎯Back to Top></u>](#head-content)

### Text-to-image
|  Model  |  Title  |   Venue  |   Date   |   Code   |   Project   |
|:--------|:--------|:--------:|:--------:|:--------:|:--------:|

[<u><🎯Back to Top></u>](#head-content)

### Text-to-3D
|  Model  |  Title  |   Venue  |   Date   |   Code   |   Project   |
|:--------|:--------|:--------:|:--------:|:--------:|:--------:|

[<u><🎯Back to Top></u>](#head-content)

### Text-to-video
|  Model  |  Title  |   Venue  |   Date   |   Code   |   Project   |
|:--------|:--------|:--------:|:--------:|:--------:|:--------:|

[<u><🎯Back to Top></u>](#head-content)

## Other applications
### Face
|  Model  |  Title  |   Venue  |   Date   |   Code   |   Project   |
|:--------|:--------|:--------:|:--------:|:--------:|:--------:|

[<u><🎯Back to Top></u>](#head-content)
