const fs=require('fs');const ML=require('../src/ml.js');
eval(fs.readFileSync('src/concrete.js','utf8').replace(/^var /gm,'global.'));
const core=fs.readFileSync('src/core.js','utf8');const a=core.indexOf('var FEAT=');const b=core.indexOf('/* ══ 등장인물');eval(core.slice(a,b).replace(/^var /gm,'global.').replace(/^function (\w+)/gm,'global.$1=function'));
const R=concRows(),TR=R.filter(r=>!r.test),TE=R.filter(r=>r.test);
console.log('manual test',TE.filter(r=>manualRule(r)===r.pass).length,'/',TE.length,'train',(TR.filter(r=>manualRule(r)===r.pass).length/TR.length).toFixed(3));
const fv=r=>FEAT.map(f=>r[f]);const sc=ML.scaler(TR.map(fv)),A=sc.t(TR.map(fv)),B=sc.t(TE.map(fv));const y=TR.map(r=>r.pass),yt=TE.map(r=>r.pass);
for(const it of [2000,4000,8000]){const t0=Date.now();const st=ML.binFit(A,y,{C:1,iters:it});const pr=x=>ML.sigmoid(st.b+st.w.reduce((s,w,j)=>s+w*x[j],0));
 console.log(it,Date.now()-t0+'ms','tr',ML.acc(y,A.map(x=>pr(x)>=.5?1:0)).toFixed(3),'te',ML.acc(yt,B.map(x=>pr(x)>=.5?1:0)).toFixed(3),st.w.map(v=>v.toFixed(2)).join(','),st.b.toFixed(2),'auc',ML.roc(yt,B.map(pr)).auc.toFixed(3));}
const g=TR.map(r=>r.grade),gt=TE.map(r=>r.grade);const m=ML.softmaxLR(A,g,3,{C:1,iters:1500});console.log('3g',ML.acc(g,A.map(m.predict)).toFixed(3),ML.acc(gt,B.map(m.predict)).toFixed(3));
