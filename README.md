# superlim-word-level-eval

A simple repo to evaluate the word-level tasks on superlim.

## Results

#### swesim_similarity

A) With no training, we can compare the cosine similarity between the word embeddings for `word_1` and `word_2` and
computing
the pearson correlation coeff.

| pearson   | spearman |
|-----------|----------|
| 0.43      | 0.50     |

B) With training,

#####

## Installation

#### Install deps and download word embeddings for Swedish

```
pip install -r requirements.txt
python -m spacy download sv_core_news_lg
```