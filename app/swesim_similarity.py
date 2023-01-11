import spacy
from datasets import load_dataset
from scipy.stats import pearsonr, spearmanr, zscore

nlp = spacy.load("sv_core_news_lg")

dataset = load_dataset("AI-Sweden/SuperLim", 'swesim_similarity')['test']

sims_pred, sims_label = [], []
for row in dataset:
    v1, v2 = nlp(row['word_1']), nlp(row['word_2'])
    sims_pred.append(v1.similarity(v2))
    sims_label.append(float(row['similarity']) / 10.0)

print(pearsonr(sims_pred, sims_label))
print(spearmanr(sims_pred, sims_label))
