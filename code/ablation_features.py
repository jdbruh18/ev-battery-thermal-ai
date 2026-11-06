"""Feature-set ablation for XGBoost under LOBO: 5 basic electrical features vs full 9-feature set."""
import sys; sys.argv=['x']
import numpy as np, pandas as pd
exec(open('run_experiments.py').read().split("def models()")[0].replace('print(summ)',''))
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
sets={"basic5":["V","I","P","E_cum","dV"],"basic5+t":["V","I","P","E_cum","dV","t"],"full9":FEATS,"full9-cycle_idx":[f for f in FEATS if f!="cycle_idx"]}
out=[]
for nm,fs in sets.items():
  for c in CELLS:
    tr=df[df.cell!=c]; te=df[df.cell==c]
    m=XGBRegressor(n_estimators=300,max_depth=6,learning_rate=0.05,subsample=0.8,colsample_bytree=0.9,n_jobs=-1,random_state=0).fit(tr[fs],tr["T"])
    p=m.predict(te[fs]); e=np.abs(p-te["T"])
    out.append(dict(feature_set=nm,test_cell=c,MAE=e.mean(),RMSE=np.sqrt((e**2).mean()),MaxAE=e.max(),R2=r2_score(te["T"],p)))
o=pd.DataFrame(out); o.to_csv(RES+"/ablation_features_xgb.csv",index=False); print(o.groupby("feature_set").mean(numeric_only=True).round(3))
