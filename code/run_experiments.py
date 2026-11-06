"""ICCAQI EV thermal paper: LOBO temperature prediction on NASA PCoE B0005/6/7/18 discharge cycles."""
import numpy as np, pandas as pd, scipy.io as sio, time, pickle, os, json, platform
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
B=os.path.dirname(os.path.abspath(__file__))+"/.."; RAW=B+"/data/raw"; RES=B+"/results"; os.makedirs(RES,exist_ok=True)
CELLS=["B0005","B0006","B0007","B0018"]
FEATS=["V","I","P","t","E_cum","Q_cum","dV","amb","cycle_idx"]
def load(c):
    m=sio.loadmat(f"{RAW}/{c}.mat",squeeze_me=True,struct_as_record=False)[c]; rows=[]; k=0
    for cy in m.cycle:
        if cy.type!="discharge": continue
        d=cy.data; V=np.atleast_1d(d.Voltage_measured).astype(float); I=np.atleast_1d(d.Current_measured).astype(float)
        T=np.atleast_1d(d.Temperature_measured).astype(float); t=np.atleast_1d(d.Time).astype(float)
        if len(t)<10: continue
        k+=1; P=V*np.abs(I); dt=np.diff(t,prepend=t[0])
        rows.append(pd.DataFrame(dict(cell=c,cycle_idx=k,t=t,V=V,I=I,P=P,E_cum=np.cumsum(P*dt)/3600,
            Q_cum=np.cumsum(np.abs(I)*dt)/3600,dV=V[0]-V,amb=float(cy.ambient_temperature),T=T)))
    return pd.concat(rows)
df=pd.concat([load(c) for c in CELLS]); df=df.replace([np.inf,-np.inf],np.nan).dropna()
df.to_csv(RES+"/processed_dataset_summary.csv",index=False) if False else None
summ=df.groupby("cell").agg(cycles=("cycle_idx","max"),samples=("T","size"),T_min=("T","min"),T_mean=("T","mean"),T_max=("T","max"))
summ.to_csv(RES+"/dataset_summary.csv"); print(summ)
def models(): return {
 "RandomForest":RandomForestRegressor(n_estimators=100,max_depth=12,min_samples_leaf=5,n_jobs=-1,random_state=0),
 "XGBoost":XGBRegressor(n_estimators=300,max_depth=6,learning_rate=0.05,subsample=0.8,colsample_bytree=0.9,n_jobs=-1,random_state=0),
 "MLP":make_pipeline(StandardScaler(),MLPRegressor(hidden_layer_sizes=(32,16),max_iter=200,early_stopping=True,random_state=0))}
def nparams(name,m):
    if name=="MLP": r=m[-1]; return int(sum(w.size for w in r.coefs_)+sum(b.size for b in r.intercepts_))
    if name=="RandomForest": return int(sum(e.tree_.node_count for e in m.estimators_))  # nodes
    return int(sum(len(t) for t in m.get_booster().get_dump()))  # trees -> leaf/node lines
res=[]; preds={}
for test in CELLS:
    tr=df[df.cell!=test]; te=df[df.cell==test]
    for name,m in models().items():
        t0=time.perf_counter(); m.fit(tr[FEATS].values,tr["T"].values); fit=time.perf_counter()-t0
        X=te[FEATS].values; m.predict(X[:100])
        t0=time.perf_counter(); p=m.predict(X); bt=time.perf_counter()-t0
        x1=X[:1]; lat=[]
        for _ in range(200): s=time.perf_counter(); m.predict(x1); lat.append(time.perf_counter()-s)
        size=len(pickle.dumps(m)) if name!="XGBoost" else len(m.get_booster().save_raw("ubj"))
        e=np.abs(p-te["T"].values)
        res.append(dict(test_cell=test,model=name,MAE=e.mean(),RMSE=np.sqrt(mean_squared_error(te["T"],p)),MaxAE=e.max(),
            R2=r2_score(te["T"],p),train_s=fit,batch_us_per_sample=bt/len(X)*1e6,single_ms_median=np.median(lat)*1e3,
            size_kB=size/1024,complexity=nparams(name,m),n_test=len(X)))
        preds[(test,name)]=p; print(res[-1])
r=pd.DataFrame(res); r.to_csv(RES+"/metrics_per_fold.csv",index=False)
s=r.groupby("model")[["MAE","RMSE","MaxAE","R2","batch_us_per_sample","single_ms_median","size_kB","train_s"]].agg(["mean","std"]); s.to_csv(RES+"/metrics_summary.csv"); print(s)
# naive baseline: predict training-set mean temperature
bl=[]
for test in CELLS:
    mu=df[df.cell!=test]["T"].mean(); y=df[df.cell==test]["T"].values
    bl.append(dict(test_cell=test,MAE=np.abs(y-mu).mean(),RMSE=np.sqrt(((y-mu)**2).mean()),R2=r2_score(y,np.full_like(y,mu))))
pd.DataFrame(bl).to_csv(RES+"/baseline_mean_predictor.csv",index=False)
json.dump(dict(platform=platform.platform(),cpu=os.popen("lscpu | grep 'Model name'").read().strip(),cores=os.cpu_count(),python=platform.python_version()),open(RES+"/hardware.json","w"),indent=1)
# figures
te=df[df.cell=="B0018"]; cyc=sorted(te.cycle_idx.unique()); pick=[cyc[0],cyc[len(cyc)//2],cyc[-1]]
fig,ax=plt.subplots(1,3,figsize=(12,3.4),sharey=True)
for a,c in zip(ax,pick):
    msk=(te.cycle_idx==c).values; a.plot(te.t.values[msk],te["T"].values[msk],"k-",label="Measured")
    for name,st in [("RandomForest","--"),("XGBoost",":"),("MLP","-.")]: a.plot(te.t.values[msk],preds[("B0018",name)][msk],st,label=name)
    a.set_title(f"B0018 discharge cycle {c}"); a.set_xlabel("Time (s)")
ax[0].set_ylabel("Temperature (°C)"); ax[0].legend(fontsize=7); plt.tight_layout(); plt.savefig(RES+"/fig1_pred_vs_actual_B0018.png",dpi=200)
plt.figure(figsize=(6,3.4)); pv=r.pivot(index="test_cell",columns="model",values="MAE"); pv.plot.bar(ax=plt.gca()); plt.ylabel("MAE (°C)"); plt.xlabel("Held-out cell"); plt.xticks(rotation=0); plt.tight_layout(); plt.savefig(RES+"/fig2_mae_by_cell_model.png",dpi=200)
best=s[("MAE","mean")].idxmin(); allr=np.concatenate([preds[(c,best)]-df[df.cell==c]["T"].values for c in CELLS])
plt.figure(figsize=(6,3.4)); plt.hist(allr,bins=100); plt.xlabel(f"Residual (pred − meas, °C), {best}"); plt.ylabel("Count"); plt.tight_layout(); plt.savefig(RES+"/fig3_residuals_best.png",dpi=200)
print("best",best)
