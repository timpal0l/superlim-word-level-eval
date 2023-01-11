import spacy
from datasets import load_dataset
from scipy.stats import pearsonr, spearmanr

nlp = spacy.load("sv_core_news_lg")

dataset = load_dataset("AI-Sweden/SuperLim", 'swesim_relatedness')['test']

sims_pred, sims_label = [], []
for row in dataset:
    v1, v2 = nlp(row['word_1']), nlp(row['word_2'])
    sims_pred.append(v1.similarity(v2))
    sims_label.append(float(row['relatedness']) / 10.0)

print(round(pearsonr(sims_pred, sims_label).statistic, 4))
print(round(spearmanr(sims_pred, sims_label).statistic, 4))
