# Patch Ranking: Token Pruning as Ranking Prediction for Efficient CLIP (WACV 2025)  

> **Patch Ranking: Token Pruning as Ranking Prediction for Efficient CLIP**  
> **Cheng-En Wu, Jinhong Lin, Yu Hen Hu, Pedro Morgado**  
> **University of Wisconsin–Madison**  
> Contact: {cwu356, jlin398, yhhu, pmorgado}@wisc.edu  

[![arXiv](https://img.shields.io/badge/arXiv-Paper-B31B1B.svg)](https://arxiv.org/abs/2409.14607) 

---

## 🔥 Introduction  

Contrastive image-text pre-trained models like CLIP have demonstrated exceptional adaptability to various downstream tasks. However, the computational demands of the Vision Transformer (ViT) backbone limit their deployment efficiency. **Patch Ranking** proposes a novel approach to token pruning in CLIP models by leveraging a **Golden Ranking**—a method that systematically ranks patch tokens based on their importance.  

We introduce a **lightweight predictor** that efficiently approximates the Golden Ranking, allowing for **significant computational savings with minimal accuracy loss**. Our method integrates **learnable tokens** to restore model performance post-pruning, making **CLIP models up to 40% more efficient while maintaining accuracy within 0.3% of the original model**.  

---

## 🚀 Key Contributions  

1️⃣ **Golden Ranking for Token Pruning**: A systematic ranking strategy to identify the most informative patch tokens in CLIP.  
2️⃣ **Lightweight Predictor**: A predictor trained to approximate the Golden Ranking, guiding real-time token pruning.  
3️⃣ **Efficient Model Tuning**: The integration of **learnable text and visual tokens** helps recover accuracy post-pruning.  
4️⃣ **Computational Efficiency**: Achieves **40% token reduction** in CLIP’s ViT with only **0.3% accuracy loss** across seven datasets.  

---

## 📊 **Experimental Results**  

We evaluate **Patch Ranking** against **state-of-the-art token pruning methods** on **eight benchmark datasets**, demonstrating **superior efficiency while maintaining high classification accuracy**.  

### 🔥 **Comparison with Existing Token Pruning Methods**  

| Method                | Caltech101 | OxfordPets | Flowers102 | Food101 | FGVCAircraft | DTD  | UCF101 | ImageNet | Avg. GFLOPs |
|----------------------|-----------|------------|------------|---------|-------------|------|--------|----------|-------------|
| **EViT**      | 92.5      | 87.1       | 67.1       | 80.3    | 23.3        | 43.3 | 63.3   | 58.1     | 16.8        |
| **A-ViT**     | 91.4      | 83.2       | 67.7       | 82.3    | 21.7        | 43.5 | 63.3   | 57.6     | 16.8        |
| **ToMe**      | 91.5      | 87.2       | 67.7       | 82.4    | 20.4        | 41.5 | 64.9   | 58.3     | 16.8        |
| **Ours** | **93.4**  | **88.1**   | **67.9**   | **82.8**  | **22.9**  | **43.7** | **65.2** | **59.5** | **16.8** |
| **Ours-(ImageNet-trained)** | **93.6**  | **87.7**   | **69.6**   | **84.0**  | **21.6**  | **44.3** | **63.0** | **59.5** | **16.8** |
---
## 🛠 Installation & Setup  

To set up the repository, install dependencies:  

```bash
git clone https://github.com/CEWu/PatchRanking.git
cd PatchRanking
pip install -r requirements.txt
```

---

## 📖 Usage  

### 1️⃣ Data Preparation  
Follow the steps in [DATASETS.md](docs/DATASETS.md) to prepare datasets.  

### 2️⃣ Training the Golden Ranking Predictor  
To train the predictor for ranking patch tokens:  

```bash
python train_predictor.py --dataset <dataset_name> --config configs/train_predictor.yaml
```

## 📜 Citation  

If you find our work useful, please cite:  

```bibtex
@article{wu2025patchranking,
  title={Patch Ranking: Token Pruning as Ranking Prediction for Efficient CLIP},
  author={Wu, Cheng-En and Lin, Jinhong and Hu, Yu Hen and Morgado, Pedro},
  journal={IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year={2025}
}
```