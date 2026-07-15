import os,sys,json
import numpy as np
sys.path.insert(0,'tools'); sys.path.insert(0,'.')
import lightgbm as lgb
from eval_submission import score as le_score
import lambdamart_map_ablation2 as A   # reuse feats/build/le_build/DEFAULT

rows = A.build_syn(1)   # rich (level 1)
X=np.concatenate([r[1] for r in rows]); Y=np.concatenate([r[2] for r in rows]); G=[len(r[3]) for r in rows]
model=lgb.train(A.DEFAULT, lgb.Dataset(X,label=Y,group=G), num_boost_round=200)
le_data, gt = A.le_build(1)
sub={}; pool={}
for q,(cand,Xq) in le_data.items():
    lm=model.predict(Xq); o=np.argsort(-lm)
    sub[q]=[cand[j] for j in o][:10]; pool[q]=[[cand[j],float(lm[j])] for j in o]
m=le_score(sub,gt)
os.makedirs('submissions/lambdamart_pool_3enc_rich',exist_ok=True)
json.dump(pool,open('submissions/lambdamart_pool_3enc_rich/scores.json','w'))
print(f"COMPLIANT rich full-train | LE R@1={m['R@1']:.2f} R@5={m['R@5']:.2f} R@10={m['R@10']:.2f} mAP={m['mAP']:.2f}")
print("dumped -> submissions/lambdamart_pool_3enc_rich/scores.json")
