import pandas as pd, numpy as np, glob
from sklearn.metrics import roc_auc_score

low = []
for f in sorted(glob.glob('runs/multi_dens/seed_*_probs.csv')):
    d = pd.read_csv(f)
    y = d.true_level.values
    P = d[['prob_l1','prob_l2','prob_l3']].values
    aucs = [roc_auc_score(y == L, P[:, L-1]) for L in (1,2,3)]
    low.append(int(np.argmin(aucs)) + 1)
print(pd.Series(low).value_counts().sort_index())

