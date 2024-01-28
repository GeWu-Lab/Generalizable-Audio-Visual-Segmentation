# GAVS: Generalizable-Audio-Visual-Segmentation
Official repository of "Prompting Segmentation with Sound is Generalizable Audio-Visual Source Localizer", AAAI 2024

arXiv: https://arxiv.org/abs/2309.07929

The project page and code are expected to be released before March, 2024.

### AVS-V3 dataset
- For zero-shot, you should train with "meta_v3_seen_train", eval with "meta_v3_seen_val", then test with "meta_v3_unseen".

- For other few-shot settings, the val set is remained (i.e., "meta_v3_seen_val"), training samples are picked up from  "meta_v3_unseen". In summary, train with "/v3_x_shot/train", test with "/v3_x_shot/test".

# 1. Run
\> cd segment_anything  
\> sh run_v1m.sh

# 2. Path
All path configured should be found in dataset/avs_bench.py  
