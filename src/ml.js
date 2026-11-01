/* ══ 작은 학습 엔진 (브라우저·node 공용) ══════════════════════════════ */
var ML=(function(){
function mean(a){var s=0;for(var i=0;i<a.length;i++)s+=a[i];return s/a.length;}
/* StandardScaler와 같다: 평균을 빼고 모표준편차(ddof=0)로 나눈다 */
function scaler(X){var d=X[0].length,mu=[],sd=[];for(var j=0;j<d;j++){var c=X.map(function(r){return r[j]}),m=mean(c),v=0;c.forEach(function(x){v+=(x-m)*(x-m)});mu.push(m);sd.push(Math.sqrt(v/c.length)||1);}
  return {mu:mu,sd:sd,t:function(Z){return Z.map(function(r){return r.map(function(x,j){return (x-mu[j])/sd[j]})})}};}
function sigmoid(z){return z>=0?1/(1+Math.exp(-z)):Math.exp(z)/(1+Math.exp(z));}
function softmax(z){var m=Math.max.apply(null,z),e=z.map(function(v){return Math.exp(v-m)}),s=e.reduce(function(a,b){return a+b},0);return e.map(function(v){return v/s});}
/* 다중(소프트맥스) 로지스틱 회귀. 목적: C·Σ교차엔트로피 + ½‖W‖² (scikit-learn과 같은 꼴, 절편은 규제 안 함)
   을 C·n으로 나눈 평균 손실 + ‖W‖²/(2Cn)을 Nesterov 가속 경사 하강으로 줄인다 */
function softmaxLR(X,y,K,o){o=o||{};var n=X.length,d=X[0].length,C=o.C==null?1:o.C,it=o.iters||3000,lr=o.lr||0.5,lam=isFinite(C)?1/(C*n):0;
  var W=[],b=[],vW=[],vb=[];for(var k=0;k<K;k++){W.push(new Array(d).fill(0));vW.push(new Array(d).fill(0));b.push(0);vb.push(0);}
  function grad(Wl,bl){var gW=Wl.map(function(r){return r.map(function(){return 0})}),gb=bl.map(function(){return 0}),L=0;
    for(var i=0;i<n;i++){var z=[];for(var k=0;k<K;k++){var s=bl[k];for(var j=0;j<d;j++)s+=Wl[k][j]*X[i][j];z.push(s);}var p=softmax(z);L-=Math.log(Math.max(p[y[i]],1e-15));
      for(k=0;k<K;k++){var e=(p[k]-(k===y[i]?1:0))/n;gb[k]+=e;for(j=0;j<d;j++)gW[k][j]+=e*X[i][j];}}
    var R=0;for(k=0;k<K;k++)for(j=0;j<d;j++){gW[k][j]+=lam*Wl[k][j];R+=Wl[k][j]*Wl[k][j];}
    return {gW:gW,gb:gb,loss:L/n+lam*R/2};}
  var mom=0.9,hist=[];
  for(var t=0;t<it;t++){var lW=W.map(function(r,k){return r.map(function(w,j){return w+mom*vW[k][j]})}),lb=b.map(function(v,k){return v+mom*vb[k]});
    var g=grad(lW,lb);for(var k=0;k<K;k++){for(var j=0;j<d;j++){vW[k][j]=mom*vW[k][j]-lr*g.gW[k][j];W[k][j]+=vW[k][j];}vb[k]=mom*vb[k]-lr*g.gb[k];b[k]+=vb[k];}
    if(o.trace&&(t%o.trace===0))hist.push(g.loss);}
  function proba(x){var z=[];for(var k=0;k<K;k++){var s=b[k];for(var j=0;j<d;j++)s+=W[k][j]*x[j];z.push(s);}return softmax(z);}
  function predict(x){var p=proba(x),m=0;for(var k=1;k<K;k++)if(p[k]>p[m])m=k;return m;}
  return {W:W,b:b,proba:proba,predict:predict,hist:hist,loss:grad(W,b).loss};}
/* 이진 로지스틱 회귀 한 걸음 (수습 기간 재생용): 평균 교차엔트로피 + λ/2‖w‖² */
function binStep(st,X,y,lr,lam){var n=X.length,d=X[0].length,gw=new Array(d).fill(0),gb=0,L=0;
  for(var i=0;i<n;i++){var z=st.b;for(var j=0;j<d;j++)z+=st.w[j]*X[i][j];var p=sigmoid(z);L-=y[i]?Math.log(Math.max(p,1e-15)):Math.log(Math.max(1-p,1e-15));var e=(p-y[i])/n;gb+=e;for(j=0;j<d;j++)gw[j]+=e*X[i][j];}
  var R=0;for(j=0;j<d;j++){R+=st.w[j]*st.w[j];st.w[j]-=lr*(gw[j]+lam*st.w[j]);}st.b-=lr*gb;st.t++;return L/n+lam*R/2;}
function binLoss(st,X,y,lam){var n=X.length,L=0,R=0;for(var i=0;i<n;i++){var z=st.b;for(var j=0;j<X[i].length;j++)z+=st.w[j]*X[i][j];var p=sigmoid(z);L-=y[i]?Math.log(Math.max(p,1e-15)):Math.log(Math.max(1-p,1e-15));}st.w.forEach(function(w){R+=w*w});return L/n+lam*R/2;}
function binFit(X,y,o){o=o||{};var st={w:new Array(X[0].length).fill(0),b:0,t:0},lam=o.C==null||!isFinite(o.C)?0:1/(o.C*X.length);for(var t=0;t<(o.iters||4000);t++)binStep(st,X,y,o.lr||0.5,lam);return st;}
function confusion(yt,yp,K){var M=[];for(var a=0;a<K;a++)M.push(new Array(K).fill(0));for(var i=0;i<yt.length;i++)M[yt[i]][yp[i]]++;return M;}
function acc(yt,yp){var c=0;for(var i=0;i<yt.length;i++)if(yt[i]===yp[i])c++;return c/yt.length;}
/* ROC: 점수가 높을수록 양성(1) */
function roc(yt,s){var idx=s.map(function(v,i){return i}).sort(function(a,b){return s[b]-s[a]});var P=yt.filter(function(v){return v===1}).length,N=yt.length-P,tp=0,fp=0,pts=[[0,0]],auc=0,px=0,py=0;
  for(var k=0;k<idx.length;k++){var i=idx[k];if(yt[i]===1)tp++;else fp++;if(k===idx.length-1||s[idx[k+1]]!==s[i]){var x=fp/N,yv=tp/P;auc+=(x-px)*(yv+py)/2;px=x;py=yv;pts.push([x,yv]);}}
  return {pts:pts,auc:auc};}
return {scaler:scaler,sigmoid:sigmoid,softmax:softmax,softmaxLR:softmaxLR,binStep:binStep,binLoss:binLoss,binFit:binFit,confusion:confusion,acc:acc,roc:roc,mean:mean};})();
if(typeof module!=='undefined')module.exports=ML;
