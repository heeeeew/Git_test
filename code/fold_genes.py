import argparse
import time
import numpy as np
import pandas as pd
import networkx as nx
import os
from scipy.spatial.distance import pdist, squareform
from scipy.stats import ttest_1samp, ttest_ind, ranksums, ttest_rel, wilcoxon
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn import metrics
import pickle
from sklearn.utils import shuffle
from sklearn.model_selection import KFold

def sigmoid(x):
    return 1 / (1 +np.exp(-x))


cancer_type = "LIHC"
DATA_PATH = f"./PRODIGY/{cancer_type}/"
OUTPUT_PATH = DATA_PATH

genes = pd.read_csv(DATA_PATH + "genes.csv").values
genes = genes.reshape(-1)
KO = pd.read_csv(f"./{cancer_type}/{cancer_type}_driver.csv")['Gene_symbol'].values
KO = set(KO) & set(genes)
KO = np.array(sorted(list(KO)))
notKO = np.array(sorted(list(set(genes)-set(KO))))
KO = shuffle(KO)
notKO = shuffle(notKO)

print(f"KO : {len(KO)} notKO : {len(notKO)}")

kf = KFold(n_splits=5)
kf2 = KFold(n_splits=5)

train_genes = {}
test_genes = {}

i = 0
for train_index, test_index in kf.split(KO):
    train_genes[i] = {}
    test_genes[i] = {}
    train_genes[i]['KO'] = KO[train_index]
    test_genes[i]['KO'] = KO[test_index]
    i += 1
    
i = 0
for train_index, test_index in kf.split(notKO):
    train_genes[i]['notKO'] = notKO[train_index]
    test_genes[i]['notKO'] = notKO[test_index]
    i += 1

with open(OUTPUT_PATH + "train_genes.pkl","wb") as f:
    pickle.dump(train_genes,f)
    
with open(OUTPUT_PATH + "test_genes.pkl","wb") as f:
    pickle.dump(test_genes,f)
    
