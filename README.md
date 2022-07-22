# KECRS


Towards **K**nowledge-**E**nriched **C**onversational **R**ecommendation  **S**ystem.<br>

## Prerequisites
- Python 3.6
- PyTorch 1.4.0
- Torch-Geometric 1.4.2

## Getting Started

### Installation

Clone this repo.

```bash
git clone https://github.com/xiaolan98/KECRS.git
cd KECRS/parlai/task/crs/
```

### Dataset
All the data are in * ./KECRS/data/crs/ *folder
- **ReDial** dataset
- The Movie Domain Knowledge Graph, TMDKG

### Training

To train the recommender part, run:

```bash
python train_kecrs.py
```

To train the dialog part, run:

```bash
python train_transformer_rec.py
```

### Logging

TensorBoard logs and models will be saved in `saved/` folder.

### Evaluation

All results on testing set will be shown after training.

### Discussion

If you have difficulties to get things working in the above steps, please let us know.

### Cite
Please cite our paper if you use this code in your own work:
```bash
@inproceedings{zhang-etal-2022-toward,
    title = "Toward Knowledge-Enriched Conversational Recommendation Systems",
    author = "Zhang, Tong  and
      Liu, Yong  and
      Li, Boyang  and
      Zhong, Peixiang  and
      Zhang, Chen  and
      Wang, Hao  and
      Miao, Chunyan",
    booktitle = "Proceedings of the 4th Workshop on NLP for Conversational AI",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.nlp4convai-1.17",
    doi = "10.18653/v1/2022.nlp4convai-1.17",
    pages = "212--217",
    }

```
