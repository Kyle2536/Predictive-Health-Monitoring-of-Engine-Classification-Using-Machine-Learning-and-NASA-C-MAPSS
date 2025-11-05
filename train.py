from __future__ import annotations
import argparse, json
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
RANDOM_STATE=42

def main(args):
    df = pd.read_csv(args.input_csv)
    y = df["status"].values
    X = df.drop(columns=["status"])
    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("rf", RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)),
    ])
    params = {"rf__n_estimators":[200],"rf__max_depth":[10],"rf__min_samples_split":[2,5]}
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=RANDOM_STATE,stratify=y)
    gs=GridSearchCV(pipe,param_grid=params,scoring="recall",cv=5,n_jobs=-1,verbose=1)
    gs.fit(Xtr,ytr)
    ypred=gs.predict(Xte)
    yprob=getattr(gs,"predict_proba",lambda X: np.c_[1-gs.predict(X),gs.predict(X)])(Xte)[:,1]
    auc=roc_auc_score(yte,yprob)
    res={"best_params":gs.best_params_,"best_score_cv_recall":float(gs.best_score_),
         "report":classification_report(yte,ypred,output_dict=True),
         "confusion_matrix":confusion_matrix(yte,ypred).tolist(),"roc_auc":float(auc)}
    with open(args.output_json,"w") as f: json.dump(res,f,indent=2)
    print(json.dumps(res,indent=2))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--input-csv",required=True);p.add_argument("--output-json",default="models/rf_results.json");
    main(p.parse_args())
