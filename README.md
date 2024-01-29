# Prompting Segmentation with Sound Is Generalizable Audio-Visual Source Localizer
## GAVS: Generalizable-Audio-Visual-Segmentation
Official repository of "Prompting Segmentation with Sound is Generalizable Audio-Visual Source Localizer", AAAI 2024

arXiv: https://arxiv.org/abs/2309.07929

# Abstract
Never having seen an object and heard its sound simultaneously, can the model still accurately localize its visual position from the input audio? In this work, we concentrate on the Audio-Visual Localization and Segmentation tasks but under the demanding zero-shot and few-shot scenarios. To achieve this goal, different from existing approaches that mostly employ the encoder-fusion-decoder paradigm to decode localization information from the fused audio-visual feature, we introduce the encoder-prompt-decoder paradigm, aiming to better fit the data scarcity and varying data distribution dilemmas with the help of abundant knowledge from pre-trained models. Specifically, we first propose to construct a Semantic-aware Audio Prompt (SAP) to help the visual foundation model focus on sounding objects, meanwhile, the semantic gap between the visual and audio modalities is also encouraged to shrink. Then, we develop a Correlation Adapter (ColA) to keep minimal training efforts as well as maintain adequate knowledge of the visual foundation model. By equipping with these means, extensive experiments demonstrate that this new paradigm outperforms other fusion-based methods in both the unseen class and cross-dataset settings. We hope that our work can further promote the generalization study of Audio-Visual Localization and Segmentation in practical application scenarios.

# Highlight

### Motivation
Due to the scarcity of AVS data and the varying data distribution in real-world scenarios, the model is hard to learn strong audio-visual correlation well. We expect to use audio information to `prompt` the powerful `visual foundation model` like Segment Anything Model (SAM) by utilizing the inlined visual priors to adapt to downstream data, thereby achieving generalizable audio-visual segmentation (GAVS). We lso develop ColA to keep minimal training efforts as well as maintain adequate knowledge of the visual foundation model.  

### AVS-V3 dataset
We develop the V3 dataset for analyzing the generalization ability of audio-visual segmentation models.   
- For zero-shot, you should train with "meta_v3_seen_train", eval with "meta_v3_seen_val", then test with "meta_v3_unseen".

- For other few-shot settings, the val set is remained (i.e., "meta_v3_seen_val"), training samples are picked up from  "meta_v3_unseen". In summary, train with "/v3_x_shot/train", test with "/v3_x_shot/test".

# 1. Comparison with traditional AVS models

# 2. Model architecture

# 3. Experiments
## 3.1 AVS-Benchmarks

## 3.2 Data efficiency

# 4. Qualitative results

# 5. Run
## 5.1 run scripts
\> cd segment_anything  
\> sh run_v1m.sh

## 5.2 Path
All path configured should be found in dataset/avs_bench.py  

## 5.3 SAM checkpoint  

## 5.4 VGGish for audio feature extraction  

# 6. Citation
We appreciate your citation if you found our work is helpful:
```
@article{wang2023prompting,
  title={Prompting Segmentation with Sound is Generalizable Audio-Visual Source Localizer},
  author={Wang, Yaoting and Liu, Weisong and Li, Guangyao and Ding, Jian and Hu, Di and Li, Xi},
  journal={arXiv preprint arXiv:2309.07929},
  year={2023}
}
```
