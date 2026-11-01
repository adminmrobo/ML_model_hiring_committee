var BOARD_LINE='5-NN 18.1%'; /* 차원축소팀 합격선: 매뉴얼 씨 2D 지도(잉크 양·위쪽 비율)의 5-NN 정확도 */
/* ══ 5-14 PCA · 손글씨 숫자 실기 과제지 ══════════════════════════════════ */
/* PCA-ENGINE-START */
/* ── 주성분 분석 엔진 (이 편 전용, 브라우저·node 공용) ──
   · PCA: 평균 빼기 → 공분산 행렬(÷ n−1) → 야코비 회전으로 고유값 분해 → 큰 고유값 순서. scikit-learn PCA와 같은 축이 나온다.
     축의 부호는 성분에서 절댓값이 가장 큰 칸이 +가 되게 맞춘다(scikit-learn 1.9의 결과와 같은 방향)
   · 5-NN 채점: scikit-learn cross_val_score(KNeighborsClassifier(5), Z, y, cv=5)와 같은 접기(StratifiedKFold, 섞지 않음)와 다수결(동점이면 작은 숫자)
   · K-Means: k-means++ 시작(탐욕적 시도 2+ln k번) + 로이드 반복, 여러 번 시작해 관성이 가장 작은 것. 시작값의 난수가 scikit-learn과 달라 결과가 조금 다를 수 있다
   · 거듭제곱법: 수습 기간 재생용. 무작위 방향에서 시작해 공분산을 곱하고 길이를 1로 맞추기를 되풀이하면 가장 퍼진 방향으로 모인다 */
var PCAE=(function(){
  function jacobi(S,d){var A=S.map(function(r){return Float64Array.from(r)}),V=[];for(var i=0;i<d;i++){V.push(new Float64Array(d));V[i][i]=1;}
    for(var sw=0;sw<60;sw++){var off=0;for(var p=0;p<d;p++)for(var q=p+1;q<d;q++)off+=A[p][q]*A[p][q];if(off<1e-22)break;
      for(p=0;p<d;p++)for(q=p+1;q<d;q++){var apq=A[p][q];if(Math.abs(apq)<1e-300)continue;var th=(A[q][q]-A[p][p])/(2*apq),t=(th>=0?1:-1)/(Math.abs(th)+Math.sqrt(th*th+1)),c=1/Math.sqrt(t*t+1),s=t*c;
        for(var k=0;k<d;k++){var akp=A[k][p],akq=A[k][q];A[k][p]=c*akp-s*akq;A[k][q]=s*akp+c*akq;}
        for(k=0;k<d;k++){var apk=A[p][k],aqk=A[q][k];A[p][k]=c*apk-s*aqk;A[q][k]=s*apk+c*aqk;}
        for(k=0;k<d;k++){var vkp=V[k][p],vkq=V[k][q];V[k][p]=c*vkp-s*vkq;V[k][q]=s*vkp+c*vkq;}}}
    var vals=[];for(var j=0;j<d;j++)vals.push(A[j][j]);return {vals:vals,V:V};}
  function cov(X,mu){var n=X.length,d=mu.length,C=[];for(var a=0;a<d;a++)C.push(new Float64Array(d));
    var xc=new Float64Array(d);for(var i=0;i<n;i++){var x=X[i];for(a=0;a<d;a++)xc[a]=x[a]-mu[a];for(a=0;a<d;a++){var v=xc[a];if(v===0)continue;var Ca=C[a];for(var b=a;b<d;b++)Ca[b]+=v*xc[b];}}
    for(a=0;a<d;a++)for(var b=a;b<d;b++){C[a][b]/=(n-1);C[b][a]=C[a][b];}return C;}
  function fit(X){var n=X.length,d=X[0].length,mu=new Float64Array(d);X.forEach(function(x){for(var j=0;j<d;j++)mu[j]+=x[j];});for(var j=0;j<d;j++)mu[j]/=n;
    var C=cov(X,mu),E=jacobi(C,d),ord=E.vals.map(function(v,i){return i}).sort(function(a,b){return E.vals[b]-E.vals[a]});
    var comps=ord.map(function(c){var v=new Float64Array(d),bi=0;for(var j=0;j<d;j++){v[j]=E.V[j][c];if(Math.abs(v[j])>Math.abs(v[bi]))bi=j;}if(v[bi]<0)for(j=0;j<d;j++)v[j]=-v[j];return v;});
    var vals=ord.map(function(c){return Math.max(0,E.vals[c])}),tot=vals.reduce(function(a,b){return a+b},0);
    return {mean:mu,comps:comps,vals:vals,total:tot,ratio:vals.map(function(v){return v/tot}),C:C,d:d};}
  function project(P,x,k){var z=new Float64Array(k);for(var c=0;c<k;c++){var s=0,v=P.comps[c];for(var j=0;j<P.d;j++)s+=(x[j]-P.mean[j])*v[j];z[c]=s;}return z;}
  function transform(P,X,k){return X.map(function(x){return project(P,x,k)});}
  function recon(P,z,k){var r=Float64Array.from(P.mean);for(var c=0;c<k;c++){var v=P.comps[c],s=z[c];for(var j=0;j<P.d;j++)r[j]+=s*v[j];}return r;}
  /* StratifiedKFold(n_splits=K, shuffle=False)의 시험 접기 번호 */
  function folds(y,K){var first={},enc=[],cls=0;y.forEach(function(v){if(!(v in first))first[v]=cls++;});var ye=y.map(function(v){return first[v]});
    var so=ye.slice().sort(function(a,b){return a-b}),alloc=[];for(var f=0;f<K;f++){var b=new Array(cls).fill(0);for(var i=f;i<so.length;i+=K)b[so[i]]++;alloc.push(b);}
    var out=new Int32Array(y.length),ptr=new Array(cls).fill(0),seq=[];for(var c=0;c<cls;c++){var s=[];for(f=0;f<K;f++)for(var r=0;r<alloc[f][c];r++)s.push(f);seq.push(s);}
    for(i=0;i<y.length;i++){out[i]=seq[ye[i]][ptr[ye[i]]++];}return out;}
  /* 5겹 k-NN 채점: 각 접기를 시험으로, 나머지에서 가장 가까운 k장의 정답으로 다수결. 반환 {acc(접기 평균), pred} */
  function knnCV(Z,y,k,K){k=k||5;K=K||5;var n=Z.length,d=Z[0].length,F=folds(y,K),pred=new Int32Array(n),corr=new Array(K).fill(0),cnt=new Array(K).fill(0),bd=new Float64Array(k),bi=new Int32Array(k);
    for(var i=0;i<n;i++){var fi=F[i],zi=Z[i];bd.fill(Infinity);bi.fill(-1);
      for(var j=0;j<n;j++){if(F[j]===fi)continue;var zj=Z[j],s=0;for(var c=0;c<d;c++){var t=zi[c]-zj[c];s+=t*t;}
        if(s<bd[k-1]){var p=k-1;while(p>0&&bd[p-1]>s){bd[p]=bd[p-1];bi[p]=bi[p-1];p--;}bd[p]=s;bi[p]=j;}}
      var v=new Array(10).fill(0);for(var q=0;q<k;q++)v[y[bi[q]]]++;var best=0;for(var c2=1;c2<10;c2++)if(v[c2]>v[best])best=c2;pred[i]=best;cnt[fi]++;if(best===y[i])corr[fi]++;}
    var acc=0;for(var f=0;f<K;f++)acc+=corr[f]/cnt[f];return {acc:acc/K,pred:pred};}
  function RNGf(a){return function(){a|=0;a=a+0x6D2B79F5|0;var t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296}}
  function d2(a,b){var s=0;for(var j=0;j<a.length;j++){var t=a[j]-b[j];s+=t*t;}return s;}
  function kmOnce(Z,K,rnd){var n=Z.length,d=Z[0].length,C=[],D=new Float64Array(n),trials=2+Math.floor(Math.log(K));
    C.push(Float64Array.from(Z[Math.floor(rnd()*n)]));for(var i=0;i<n;i++)D[i]=d2(Z[i],C[0]);
    while(C.length<K){var tot=0;for(i=0;i<n;i++)tot+=D[i];var bestC=-1,bestPot=Infinity,bestD=null;
      for(var t=0;t<trials;t++){var r=rnd()*tot,c=0;while(c<n-1&&r>=D[c]){r-=D[c];c++;}var nd=new Float64Array(n),pot=0;for(i=0;i<n;i++){nd[i]=Math.min(D[i],d2(Z[i],Z[c]));pot+=nd[i];}if(pot<bestPot){bestPot=pot;bestC=c;bestD=nd;}}
      C.push(Float64Array.from(Z[bestC]));D=bestD;}
    var L=new Int32Array(n),it=0,inertia=0;
    for(it=0;it<300;it++){inertia=0;for(i=0;i<n;i++){var bk=0,bv=Infinity;for(var k=0;k<K;k++){var v=d2(Z[i],C[k]);if(v<bv){bv=v;bk=k;}}L[i]=bk;inertia+=bv;}
      var S=[],N=new Array(K).fill(0);for(k=0;k<K;k++)S.push(new Float64Array(d));for(i=0;i<n;i++){N[L[i]]++;for(var j=0;j<d;j++)S[L[i]][j]+=Z[i][j];}
      var shift=0;for(k=0;k<K;k++){if(!N[k])continue;for(j=0;j<d;j++){var nv=S[k][j]/N[k];shift+=(nv-C[k][j])*(nv-C[k][j]);C[k][j]=nv;}}if(shift<1e-10)break;}
    inertia=0;for(i=0;i<n;i++){bk=0;bv=Infinity;for(k=0;k<K;k++){v=d2(Z[i],C[k]);if(v<bv){bv=v;bk=k;}}L[i]=bk;inertia+=bv;}
    return {labels:L,centers:C,inertia:inertia,iters:it+1};}
  function kmeans(Z,K,o){o=o||{};var rnd=RNGf(o.seed||7),best=null;for(var r=0;r<(o.nInit||10);r++){var m=kmOnce(Z,K,rnd);if(!best||m.inertia<best.inertia)best=m;}return best;}
  /* 묶음마다 중심에 가장 가까운 한 장(대표)에만 사람이 이름을 붙이고, 그 이름을 묶음 전체에 퍼뜨린다 */
  function repLabel(Z,y,M){var K=M.centers.length,rep=new Array(K).fill(-1),rd=new Array(K).fill(Infinity);
    for(var i=0;i<Z.length;i++){var c=M.labels[i],v=d2(Z[i],M.centers[c]);if(v<rd[c]){rd[c]=v;rep[c]=i;}}
    var ok=0;for(i=0;i<Z.length;i++)if(y[rep[M.labels[i]]]===y[i])ok++;return {acc:ok/Z.length,rep:rep};}
  function ari(a,b){var n=a.length,T={},A={},B={};for(var i=0;i<n;i++){var k=a[i]+'_'+b[i];T[k]=(T[k]||0)+1;A[a[i]]=(A[a[i]]||0)+1;B[b[i]]=(B[b[i]]||0)+1;}
    function c2(x){return x*(x-1)/2;}var s=0,sa=0,sb=0,k2;for(k2 in T)s+=c2(T[k2]);for(k2 in A)sa+=c2(A[k2]);for(k2 in B)sb+=c2(B[k2]);var e=sa*sb/c2(n),mx=(sa+sb)/2;return (s-e)/(mx-e);}
  /* 거듭제곱법 한 걸음: v ← C v / |C v|. 반환 v와 그 방향의 분산(레일리 몫) */
  function powerStep(C,v){var d=v.length,w=new Float64Array(d);for(var a=0;a<d;a++){var s=0,Ca=C[a];for(var b=0;b<d;b++)s+=Ca[b]*v[b];w[a]=s;}var nr=0;for(a=0;a<d;a++)nr+=w[a]*w[a];nr=Math.sqrt(nr)||1;var lam=0;for(a=0;a<d;a++)lam+=v[a]*w[a];for(a=0;a<d;a++)w[a]/=nr;return {v:w,lam:lam};}
  function deflate(C,v,lam){return C.map(function(r,a){var o=Float64Array.from(r);for(var b=0;b<o.length;b++)o[b]-=lam*v[a]*v[b];return o;});}
  return {fit:fit,project:project,transform:transform,recon:recon,folds:folds,knnCV:knnCV,kmeans:kmeans,repLabel:repLabel,ari:ari,powerStep:powerStep,deflate:deflate,RNG:RNGf,cov:cov};})();
/* PCA-ENGINE-END */

/* ══ 데이터 · 공통 도구 ══ */
const ROWS=digitRows(),X=ROWS.map(r=>r.px),Y=ROWS.map(r=>r.y),N=X.length; /* Y는 채점 봉투: 축을 고르는 데는 쓰지 않는다 */
const BASE=Math.max(...[0,1,2,3,4,5,6,7,8,9].map(d=>Y.filter(v=>v===d).length))/N; /* 기준선: 가장 흔한 숫자만 답하기 */
const SEAL={open:false,fns:[]};function onSeal(fn){SEAL.fns.push(fn);}
function setSeal(v){SEAL.open=v;['envOpen','mapEnv'].forEach(id=>{const b=$(id);if(!b)return;b.setAttribute('aria-pressed',v?'true':'false');b.textContent=v?'채점 봉투 닫기 (정답 가리기)':(id==='mapEnv'?'채점 봉투 열기 (숫자별 색)':'채점 봉투 열기 (정답 보기)');});SEAL.fns.forEach(f=>{try{f();}catch(e){}});}
function axes(ctx,C,pl,pt,gw,gh){ctx.strokeStyle=C.rule;ctx.lineWidth=1;ctx.strokeRect(pl,pt,gw,gh);ctx.fillStyle=C.muted;ctx.font=font(12);}
function vlabel(ctx,txt,x,y){ctx.save();ctx.translate(x,y);ctx.rotate(-Math.PI/2);ctx.textAlign='center';ctx.fillText(txt,0,0);ctx.restore();}
function barRows(rows,max){return rows.map(r=>'<div class="row"><span style="text-align:left;font-family:var(--font)">'+r[0]+'</span><div class="tr"><i style="left:0;width:'+Math.max(0,Math.min(100,r[1]/max*100))+'%;background:'+(r[2]||'var(--blue)')+'"></i></div><span>'+r[3]+'</span></div>').join('');}
/* 성분 그림: 파랑 = 그 칸을 밝게(+), 빨강 = 어둡게(−) */
function drawComp(ctx,v,x,y,s,C){const c=s/8;let m=0;for(let j=0;j<64;j++)m=Math.max(m,Math.abs(v[j]));ctx.fillStyle=C.panel;ctx.fillRect(x,y,s,s);
  for(let j=0;j<64;j++){const a=Math.abs(v[j])/(m||1);if(a<0.03)continue;ctx.fillStyle=rgba(v[j]>0?C.blue:C.stamp,a);ctx.fillRect(x+(j%8)*c,y+Math.floor(j/8)*c,Math.ceil(c),Math.ceil(c));}
  ctx.strokeStyle=C.rule;ctx.lineWidth=1;ctx.strokeRect(x+.5,y+.5,s-1,s-1);}
function drawGlyph(ctx,px,x,y,s,C,ink){ctx.fillStyle=C.panel;ctx.fillRect(x,y,s,s);drawDigit(ctx,px.map(v=>clamp(v,0,16)),x,y,s,ink||C.text);ctx.strokeStyle=C.rule;ctx.lineWidth=1;ctx.strokeRect(x+.5,y+.5,s-1,s-1);}
function whenNear(id,fn,margin){const t=$(id);let done=false;function go(){if(done)return;done=true;setTimeout(fn,10);}
  if('IntersectionObserver' in window){const io=new IntersectionObserver(es=>{if(es.some(e=>e.isIntersecting)){go();io.disconnect();}},{rootMargin:(margin||600)+'px'});io.observe(t);}else go();
  window.addEventListener('hashchange',()=>{const h=location.hash.slice(1),el=h&&$(h);if(el&&(el===t||t.contains(el)||t.compareDocumentPosition(el)&Node.DOCUMENT_POSITION_FOLLOWING))go();});
  if(location.hash){const el=$(location.hash.slice(1));if(el&&(el===t||t.contains(el)||t.compareDocumentPosition(el)&Node.DOCUMENT_POSITION_FOLLOWING))go();}}

/* ── 본 모델: 1,797장 전체로 PCA (정답 없이). 2D 지도와 그 채점은 필요할 때 한 번만 계산한다 ── */
let P=null,Z2=null,KNN2=null;
function ensureP(){if(P)return P;const t0=performance.now();P=PCAE.fit(X);P.ms=performance.now()-t0;P.cum=[];let c=0;P.ratio.forEach(r=>{c+=r;P.cum.push(c);});
  P.n90=P.cum.findIndex(v=>v>=0.9)+1;P.n95=P.cum.findIndex(v=>v>=0.95)+1;P.n80=P.cum.findIndex(v=>v>=0.8)+1;
  if($('fitMs'))$('fitMs').textContent=Math.max(1,Math.round(P.ms));return P;}
function ensureZ2(){if(!Z2)Z2=PCAE.transform(ensureP(),X,2);return Z2;}
function ensureKNN2(){if(!KNN2){const t0=performance.now();KNN2=PCAE.knnCV(ensureZ2(),Y,5,5);KNN2.ms=performance.now()-t0;}return KNN2;}
/* 픽셀당 복원 오차(1,797장 평균): 버린 축들의 분산 합 × (n−1)/n ÷ 64 */
function reconMSE(k){const p=ensureP();let s=0;for(let j=k;j<64;j++)s+=p.vals[j];return s*(N-1)/N/64;}
document.querySelectorAll('.pcAcc').forEach(e=>{e.textContent=pct(PRE.colab.acc2);});

/* ── 히어로: 그림자 놀이 ── */
heroRoom($('heroCv'),{sym:'PC',no:'APP-14',name:'PCA\n(주성분)',role:'지원자 · 차원축소',banner:'모델 채용 위원회 · 5-14 면접',doc:'front',asst:'board',fly:'clip',robo:'calc',loop:11,
  beats:[{at:0.4,who:'doc',mood:'magic',move:'pop',say:'64칸을 두 축으로'},{at:2.4,who:'fly',mood:'arms',move:'tilt',say:'그림자로 된다고?'},{at:4.4,who:'asst',mood:'shock',move:'shake',say:'1,797장 전부요?'},
         {at:6.4,who:'robo',mood:'count',move:'bob',say:'숫자 192개면 끝?'},{at:8.4,who:'asst',mood:'spark',move:'jump',say:'가장 길게 퍼졌어요'}],
  motif:(function(){const A0=-0.42,R=RNG(5),pts=[];for(let i=0;i<16;i++){const a=(R()*2-1)*44,b=(R()*2-1)*11;pts.push([a*Math.cos(A0)-b*Math.sin(A0),a*Math.sin(A0)+b*Math.cos(A0)]);}
    let vmax=0;for(let k=0;k<90;k++){const th=k/90*Math.PI,ux=Math.cos(th),uy=Math.sin(th);let s=0;pts.forEach(p=>{const t=p[0]*ux+p[1]*uy;s+=t*t;});vmax=Math.max(vmax,s);}
    return function(ctx,t,W,H,C){const cx=338,cy=112,th=REDUCE?A0:A0+1.1*Math.sin(t*0.6)*Math.max(0,Math.cos(t*0.17)),ux=Math.cos(th),uy=Math.sin(th);
      ctx.strokeStyle=C.c_app;ctx.lineWidth=2.6;ctx.beginPath();ctx.moveTo(cx-64*ux,cy-64*uy);ctx.lineTo(cx+64*ux,cy+64*uy);ctx.stroke();let s=0;
      pts.forEach(p=>{const tt=p[0]*ux+p[1]*uy,fx_=cx+tt*ux,fy=cy+tt*uy;s+=tt*tt;ctx.strokeStyle=rgba(C.muted,.55);ctx.lineWidth=1;ctx.setLineDash([2,3]);ctx.beginPath();ctx.moveTo(cx+p[0],cy+p[1]);ctx.lineTo(fx_,fy);ctx.stroke();ctx.setLineDash([]);
        circ(ctx,cx+p[0],cy+p[1],3.2,C.text,null);circ(ctx,fx_,fy,2.4,C.c_app,null);});
      const f=s/vmax;ctx.fillStyle=rgba(C.muted,.25);ctx.fillRect(280,186,120,7);ctx.fillStyle=f>0.97?C.ok:C.c_app;ctx.fillRect(280,186,120*f,7);
      ctx.fillStyle=C.muted;ctx.font=font(12,500,1);ctx.textAlign='left';ctx.fillText(f>0.97?'그림자 퍼짐 최대 → 축 1':'그림자 퍼짐 '+Math.round(f*100)+'%',280,210);};})()});
$('heroLine').textContent='5-NN '+pct(DR_LINE);

/* ── 01 장면 ── */
comic('comicIntro',[
 {bg:'site',prop:'축 1 · 가장 퍼진 방향',cast:[{who:'doc',mood:'magic',x:30,move:'pop',h:104},{who:'asst',mood:'shock',x:72,move:'shake'}],say:{who:'doc',txt:'벽을 돌려 보세요. 그림자가 가장 길게 늘어지는 쪽, 거기가 첫 번째 축이에요.'},cap:'① 가장 퍼진 방향 — 그림자의 분산이 가장 큰 쪽'},
 {bg:'site',prop:'축 2 · 직각으로',cast:[{who:'asst',mood:'ledger',x:30,move:'nod'},{who:'fly',mood:'point',x:74,move:'tilt',flip:true}],say:{who:'asst',txt:'첫 축과 직각인 방향 중에서 또 가장 퍼진 쪽이 두 번째 축이에요. 둘은 같은 정보를 겹쳐 담지 않아요.'},cap:'② 두 번째 축 — 첫 축과 직각, 그중 가장 퍼진 방향'},
 {bg:'site',prop:'남긴 몫 28.5% · 버린 몫 71.5%',cast:[{who:'fly',mood:'arms',x:28,move:'shake'},{who:'robo',mood:'count',x:72,move:'bob',h:104}],say:{who:'fly',txt:'두 축만 남기면 나머지 62축 몫은 버린 거잖아. 뭘 버렸는지 보여 줘.'},cap:'③ 버린 정보 — 복원하면 흐린 글씨가 된다(05장)'}]);

/* ── 01 자기소개: 두 칸(1행 5열, 2행 5열)의 그림자 놀이 ── */
(function(){const J=13,K=21,R=RNG(11),pts=X.map(x=>[x[J],x[K]]),jit=pts.map(()=>[(R()-0.5)*0.7,(R()-0.5)*0.7]);
  const mx=ML.mean(pts.map(p=>p[0])),my=ML.mean(pts.map(p=>p[1]));let sxx=0,syy=0,sxy=0;pts.forEach(p=>{const a=p[0]-mx,b=p[1]-my;sxx+=a*a;syy+=b*b;sxy+=a*b;});sxx/=N-1;syy/=N-1;sxy/=N-1;
  const varAt=th=>{const c=Math.cos(th),s=Math.sin(th);return c*c*sxx+2*c*s*sxy+s*s*syy;},tot=sxx+syy;
  const best=(0.5*Math.atan2(2*sxy,sxx-syy)+Math.PI)%Math.PI,bestDeg=Math.round(best*180/Math.PI),vmax=varAt(best),vmin=varAt(best+Math.PI/2);
  const st={deg:120,second:false,anim:false},cv=$('shCv'),SAMPLE=[];for(let i=0;i<N;i+=61)SAMPLE.push(i);
  function draw(){const W=cv.clientWidth||cv.parentElement.clientWidth||600,wide=W>=700,H=wide?400:Math.round(Math.min(W,440)+250);fit(cv,H);const ctx=cv.getContext('2d'),C=cols();ctx.clearRect(0,0,W,H);
    const S=wide?Math.min(380,W*0.55):Math.min(W-56,440),pl=46,pt=12,sx=v=>pl+(v+1)/18*S,sy=v=>pt+(1-(v+1)/18)*S,th=st.deg*Math.PI/180,ux=Math.cos(th),uy=Math.sin(th);
    axes(ctx,C,pl,pt,S,S);[0,4,8,12,16].forEach(v=>{ctx.textAlign='center';ctx.fillText(v,sx(v),pt+S+15);ctx.textAlign='right';ctx.fillText(v,pl-5,sy(v)+4);});
    ctx.textAlign='center';ctx.fillText('1행 5열 칸의 밝기',pl+S/2,pt+S+30);vlabel(ctx,'2행 5열 칸의 밝기',12,pt+S/2);
    ctx.save();ctx.beginPath();ctx.rect(pl,pt,S,S);ctx.clip();
    pts.forEach((p,i)=>{circ(ctx,sx(p[0]+jit[i][0]),sy(p[1]+jit[i][1]),1.9,rgba(C.text,.22),null);});
    function axis(dg,col,w,dash){const t2=dg*Math.PI/180,ax=Math.cos(t2),ay=Math.sin(t2);ctx.strokeStyle=col;ctx.lineWidth=w;ctx.setLineDash(dash||[]);ctx.beginPath();ctx.moveTo(sx(mx-30*ax),sy(my-30*ay));ctx.lineTo(sx(mx+30*ax),sy(my+30*ay));ctx.stroke();ctx.setLineDash([]);}
    if(st.second)axis(st.deg+90,C.ok,2.4,[7,5]);axis(st.deg,C.blue,3);
    SAMPLE.forEach(i=>{const p=pts[i],q=[p[0]+jit[i][0],p[1]+jit[i][1]],t=(q[0]-mx)*ux+(q[1]-my)*uy,fx_=mx+t*ux,fy=my+t*uy;ctx.strokeStyle=rgba(C.muted,.7);ctx.lineWidth=1;ctx.setLineDash([2,3]);ctx.beginPath();ctx.moveTo(sx(q[0]),sy(q[1]));ctx.lineTo(sx(fx_),sy(fy));ctx.stroke();ctx.setLineDash([]);circ(ctx,sx(q[0]),sy(q[1]),3,C.text,null);circ(ctx,sx(fx_),sy(fy),3,C.blue,null);});
    ctx.restore();circ(ctx,sx(mx),sy(my),5,C.panel,C.stamp,2.5);ctx.fillStyle=C.stamp;ctx.font=font(12,700);ctx.textAlign='left';ctx.fillText('평균',sx(mx)+8,sy(my)+16);
    /* 그림자 띠: 모든 점의 그림자를 한 줄로 */
    const bx=wide?pl+S+40:pl,by=wide?pt+20:pt+S+60,bw=wide?W-bx-16:W-pl-16;
    ctx.fillStyle=C.muted;ctx.font=font(12,700);ctx.textAlign='left';ctx.fillText('축 위의 그림자 1,797개 (가운데 = 평균)',bx,by-6);
    ctx.strokeStyle=C.rule;ctx.beginPath();ctx.moveTo(bx,by+16);ctx.lineTo(bx+bw,by+16);ctx.stroke();
    const hb=new Array(60).fill(0);pts.forEach(p=>{const t=(p[0]-mx)*ux+(p[1]-my)*uy,k=clamp(Math.floor((t+22)/44*60),0,59);hb[k]++;});const hm=Math.max(...hb);
    hb.forEach((c,k)=>{const h=c/hm*28;ctx.fillStyle=rgba(C.blue,.75);ctx.fillRect(bx+k*bw/60,by+16-h,bw/60-1,h);});
    if(st.second){const hb2=new Array(60).fill(0),t2=th+Math.PI/2,u2x=Math.cos(t2),u2y=Math.sin(t2);pts.forEach(p=>{const t=(p[0]-mx)*u2x+(p[1]-my)*u2y,k=clamp(Math.floor((t+22)/44*60),0,59);hb2[k]++;});const hm2=Math.max(...hb2);
      hb2.forEach((c,k)=>{const h=c/hm2*28;ctx.fillStyle=rgba(C.ok,.6);ctx.fillRect(bx+k*bw/60,by+50-h,bw/60-1,h);});ctx.strokeStyle=C.rule;ctx.beginPath();ctx.moveTo(bx,by+50);ctx.lineTo(bx+bw,by+50);ctx.stroke();ctx.fillStyle=C.ok;ctx.fillText('두 번째 축의 그림자',bx,by+64);}
    /* 각도별 분산 곡선 */
    const cx0=bx,cy0=by+(st.second?96:70),cw=bw,ch=wide?H-cy0-36:150,lx=d=>cx0+d/180*cw,ly=v=>cy0+(1-v/(vmax*1.08))*ch;
    axes(ctx,C,cx0,cy0,cw,ch);[0,45,90,135,180].forEach(d=>{ctx.textAlign='center';ctx.fillText(d+'°',lx(d),cy0+ch+15);});ctx.fillText('축의 각도',cx0+cw/2,cy0+ch+30);
    ctx.strokeStyle=C.blue;ctx.lineWidth=2.4;ctx.beginPath();for(let d=0;d<=180;d++){const X_=lx(d),Y_=ly(varAt(d*Math.PI/180));d?ctx.lineTo(X_,Y_):ctx.moveTo(X_,Y_);}ctx.stroke();
    ctx.strokeStyle=rgba(C.stamp,.7);ctx.setLineDash([4,4]);ctx.beginPath();ctx.moveTo(lx(bestDeg),cy0);ctx.lineTo(lx(bestDeg),cy0+ch);ctx.stroke();ctx.setLineDash([]);
    circ(ctx,lx(st.deg),ly(varAt(th)),6,C.panel,C.blue,3);if(st.second){const d2=(st.deg+90)%180;circ(ctx,lx(d2),ly(varAt(d2*Math.PI/180)),6,C.panel,C.ok,3);}
    ctx.fillStyle=C.muted;ctx.font=font(12,700);ctx.textAlign='left';ctx.fillText('그림자의 분산',cx0+6,cy0+16);ctx.fillStyle=C.stamp;ctx.fillText('최대 '+bestDeg+'°',lx(bestDeg)+5,cy0+ch-8);info();}
  function info(){const th=st.deg*Math.PI/180,v=varAt(th);let h='각도 <b>'+st.deg+'°</b> · 그림자 분산 <b>'+v.toFixed(1)+'</b> (가장 퍼진 방향의 '+Math.round(v/vmax*100)+'%). 두 칸의 분산 합은 '+tot.toFixed(1)+'이다. ';
    if(st.second){const v2=varAt(th+Math.PI/2);h+='두 번째 축(초록, 직각) 분산 <b>'+v2.toFixed(1)+'</b>. 두 축의 분산을 더하면 '+(v+v2).toFixed(1)+' — 축을 어떻게 돌려도 직각인 두 축의 합은 늘 전체와 같다. 그래서 첫 축이 많이 가져가면 둘째 축 몫은 줄어든다. '+(Math.abs(st.deg-bestDeg)<=1?'지금이 PCA의 답이다: 축 1이 '+Math.round(vmax/tot*100)+'%, 축 2가 '+Math.round(vmin/tot*100)+'%를 담는다.':'');}
    else if(Math.abs(st.deg-bestDeg)<=1)h+='<b>가장 퍼진 방향</b>에 왔다. 이 두 칸에서는 한 칸이 밝으면 다른 칸도 밝은 경우가 많아(상관 '+(sxy/Math.sqrt(sxx*syy)).toFixed(2)+') 대각선 쪽으로 길게 퍼진다. 이 한 축이 두 칸 퍼짐의 '+Math.round(vmax/tot*100)+'%를 담는다. 64칸에서도 똑같이 한다.';
    else h+='그림자가 아래 띠에서 '+(v/vmax<0.6?'좁게 몰려 있다. 글씨끼리 구별이 잘 안 되는 방향이다.':'넓게 퍼지고 있다. 조금 더 돌려 보자.');
    $('shInfo').innerHTML=h;}
  function setDeg(d){st.deg=((Math.round(d)%180)+180)%180;$('shAng').value=st.deg;$('shAngO').textContent=st.deg+'°';draw();}
  $('shAng').addEventListener('input',e=>setDeg(+e.target.value));
  $('shFind').addEventListener('click',async()=>{if(st.anim)return;st.anim=true;let d=st.deg,diff=((bestDeg-d+270)%180)-90;const n=REDUCE?1:24;for(let i=1;i<=n;i++){setDeg(d+diff*ease(i/n));await wait(REDUCE?0:30);}setDeg(bestDeg);st.anim=false;});
  $('shSecond').addEventListener('click',()=>{st.second=!st.second;$('shSecond').textContent=st.second?'두 번째 축 감추기':'두 번째 축 보기';draw();});
  draw();onRedraw(draw);})();

/* ── 02 실기 과제지 ── */
$('lineDR').textContent=pct(DR_LINE);
const DG=(function(){const cv=$('digCv'),pv=$('pxCv'),st={off:0,sel:0,cell:40};const COUNT=96;
  function layout(){const W=cv.clientWidth||cv.parentElement.clientWidth||400,cols_=Math.max(6,Math.floor(W/st.cell)),cell=W/cols_,rows=Math.ceil(COUNT/cols_);return {W,cols_,cell,H:Math.ceil(rows*cell)};}
  function draw(){const L=layout();fit(cv,L.H);const ctx=cv.getContext('2d'),C=cols();ctx.clearRect(0,0,L.W,L.H);
    for(let k=0;k<COUNT;k++){const i=(st.off+k)%N,x=(k%L.cols_)*L.cell,y=Math.floor(k/L.cols_)*L.cell,s=L.cell-4;ctx.fillStyle=C.panel;ctx.fillRect(x+2,y+2,s,s);drawDigit(ctx,X[i],x+2,y+2,s,SEAL.open?DIGCOL[Y[i]]:C.text);
      if(SEAL.open){ctx.fillStyle=DIGCOL[Y[i]];ctx.font=font(10.5,700,1);ctx.textAlign='right';ctx.fillText(Y[i],x+s,y+12);}
      ctx.strokeStyle=i===st.sel?C.blue:C.rule;ctx.lineWidth=i===st.sel?2.5:1;ctx.strokeRect(x+2.5,y+2.5,s-1,s-1);}
    $('digNote').textContent='글씨 #'+(st.off+1)+' ~ #'+(st.off+COUNT)+' / 1,797장';drawPx();}
  function drawPx(){const W=pv.clientWidth||pv.parentElement.clientWidth||300,S=Math.min(W,300);fit(pv,S+4);const ctx=pv.getContext('2d'),C=cols(),c=S/8,px=X[st.sel];ctx.clearRect(0,0,W,S+4);
    for(let j=0;j<64;j++){const x=(j%8)*c,y=Math.floor(j/8)*c,v=px[j];ctx.fillStyle=C.panel;ctx.fillRect(x,y,c,c);ctx.fillStyle=rgba(C.text,v/16);ctx.fillRect(x,y,c,c);ctx.strokeStyle=C.rule;ctx.lineWidth=1;ctx.strokeRect(x+.5,y+.5,c-1,c-1);
      ctx.fillStyle=v>8?C.panel:C.muted;ctx.font=font(Math.max(10,c*0.32),700,1);ctx.textAlign='center';ctx.fillText(v,x+c/2,y+c/2+4);}
    $('pxInfo').innerHTML='<b style="font-family:var(--font)">글씨 #'+(st.sel+1)+'</b> · 특징 64개 = ['+px.join(', ')+'] · 정답: '+(SEAL.open?'<b>'+Y[st.sel]+'</b>':'<b>?</b> (봉투 속)');}
  cv.addEventListener('click',e=>{const r=cv.getBoundingClientRect(),L=layout(),k=Math.floor((e.clientY-r.top)/L.cell)*L.cols_+Math.floor((e.clientX-r.left)/L.cell);if(k>=0&&k<COUNT){st.sel=(st.off+k)%N;draw();}});
  $('digMore').addEventListener('click',()=>{st.off=(st.off+COUNT)%N;st.sel=st.off;draw();});
  $('envOpen').addEventListener('click',()=>setSeal(!SEAL.open));
  draw();onRedraw(draw);onSeal(draw);return {draw};})();
/* 매뉴얼 씨의 지도 */
(function(){const cv=$('manCv'),H=320,F=X.map(px=>digitFeat(px)),pts=F.map(f=>[f.ink,f.top]);let acc=null;
  function draw(){const W=fit(cv,H),ctx=cv.getContext('2d'),C=cols(),pl=50,pr=14,pt=12,pb=40,gw=W-pl-pr,gh=H-pt-pb,x0=0,x1=1100,sx=v=>pl+(v-x0)/(x1-x0)*gw,sy=v=>pt+(1-v)*gh;ctx.clearRect(0,0,W,H);
    axes(ctx,C,pl,pt,gw,gh);[0,200,400,600,800,1000].forEach(v=>{ctx.textAlign='center';ctx.fillText(v,sx(v),H-24);});ctx.fillText('잉크 양 (64칸 밝기의 합)',pl+gw/2,H-6);
    [0,0.25,0.5,0.75,1].forEach(v=>{ctx.textAlign='right';ctx.fillText(v.toFixed(2),pl-5,sy(v)+4);});vlabel(ctx,'위쪽 비율',12,pt+gh/2);
    pts.forEach((p,i)=>circ(ctx,sx(p[0]),sy(p[1]),2.3,SEAL.open?rgba(DIGCOL[Y[i]],.75):rgba(C.text,.3),null));}
  function info(){if(acc==null)acc=PCAE.knnCV(pts,Y,5,5).acc;
    $('manInfo').innerHTML='매뉴얼 씨 지도의 5-NN 정확도 <b>'+pct(DR_LINE)+'</b>(Python, 합격선). 이 브라우저에서 다시 재면 '+pct(acc)+'다. 잉크 양이 같은 글씨가 많아 "가장 가까운 다섯 장"을 고르는 동점 처리 순서가 scikit-learn과 달라서 생기는 차이다. 가로축은 0~1,000, 세로축은 0~1이라 거리는 거의 잉크 양 하나로 정해진다(같은 지도를 표준화하면 '+pct(PRE.team.manualStd)+'). '+(SEAL.open?'봉투를 여니 색이 가로로 넓게 겹친다. 1은 잉크가 적어 왼쪽, 0·8은 많아 오른쪽에 몰리지만 나머지는 섞인다.':'봉투를 열면 숫자마다 색이 보인다.');
    $('manLeg').innerHTML=SEAL.open?DIGCOL.map((c,d)=>'<span><i style="background:'+c+'"></i>'+d+'</span>').join(''):'<span><i style="background:var(--muted)"></i>정답을 모를 때는 모두 같은 색</span>';}
  whenNear('manCv',()=>{draw();info();},300);onRedraw(draw);onSeal(()=>{draw();if(acc!=null)info();});})();
/* 표 */
function renderTbl(){const COLS=[];for(let j=0;j<64;j++)COLS.push(Math.floor(j/8)+'_'+(j%8));
  $('digTbl').innerHTML='<table><thead><tr><th>#</th>'+COLS.map(c=>'<th><span class="tg x">X</span>'+c+'</th>').join('')+'<th><span class="tg t">봉투</span>정답</th></tr></thead><tbody>'+
    X.slice(0,40).map((px,i)=>'<tr><td>'+(i+1)+'</td>'+px.map(v=>'<td'+(v?'':' class="muted"')+'>'+v+'</td>').join('')+'<td class="y">'+(SEAL.open?Y[i]:'?')+'</td></tr>').join('')+'</tbody></table>';}
renderTbl();onSeal(renderTbl);

/* ── 03 AI 비서 ── */
const MAIN_PROMPT=`너는 머신러닝 수업 조교다. Google Colab에서 바로 실행되는 파이썬 코드를 써 줘.

[환경] Colab 기본 라이브러리(numpy, scikit-learn, matplotlib)만 쓴다. 다른 패키지는 설치하지 않는다. 첫 줄에 scikit-learn 버전을 출력한다.
[데이터] 〔sklearn.datasets.load_digits〕로 손글씨 숫자 〔1,797장, 8×8 = 64칸, 밝기 0~16〕을 읽는다.
  - 특징 X: 〔64칸 밝기〕. 행이 샘플, 열이 특징이다. 모양을 assert로 확인한다.
  - 정답 y: 〔숫자 0~9〕. PCA를 맞추는 데는 쓰지 않고, 마지막 채점 칸에서만 쓴다.
[규정]
  1) 열의 단위가 〔모두 같으므로(밝기)〕 표준화 여부를 정하고, 그 까닭을 주석으로 적는다. 표준화한 경우와 안 한 경우를 둘 다 채점해 비교한다.
  2) PCA로 설명된 분산 비율(explained_variance_ratio_)과 누적 곡선을 그리고, 80·90·95%에 필요한 성분 수를 출력한다.
  3) 평균 글씨와 성분 1~10을 8×8 그림으로 그린다. 글씨 몇 장을 2·10·21성분으로 복원(inverse_transform)해 원본과 나란히 그린다.
  4) 2성분 지도를 그리고, 채점은 cross_val_score(KNeighborsClassifier(n_neighbors=5), Z, y, cv=5)로 한다.
[출력] 위 그림들, 성분 수별 누적 분산, 2D 지도의 5-NN 정확도와 기준선(가장 흔한 숫자 비율), 합격선 〔18.1%〕와의 비교.
[금지] 정답 y로 축이나 성분 수를 고르지 않는다. 지도를 만든 점 그대로 점수를 매기지 않는다(교차 검증). 데이터를 지어내지 않는다.
[형식] 셀 단위로 나누고, 각 셀 위에 무엇을 하는지 한국어 주석을 단다. 코드 뒤에 내가 확인해야 할 점 3가지를 적어 줘.`;
$('mainPrompt').textContent=MAIN_PROMPT;$('copyMain').addEventListener('click',()=>copyText(MAIN_PROMPT,'프롬프트를 복사했습니다'));
(function(){const LINES=[
 ['import numpy as np'],['from sklearn.datasets import load_digits'],['from sklearn.preprocessing import StandardScaler'],['from sklearn.decomposition import PCA'],['from sklearn.neighbors import KNeighborsClassifier'],[''],
 ['X, y = load_digits(return_X_y=True)     # y는 채점에만 쓴다',0,'데이터를 실제로 읽었고, 정답을 학습에 안 쓰겠다고 밝혔다. 괜찮다.'],
 ['X = StandardScaler().fit_transform(X)   # PCA 전에는 표준화가 필수',1,'<b>"필수"가 아니다.</b> 64칸이 모두 같은 단위(밝기 0~16)인데 표준화하면, 거의 늘 0인 가장자리 칸의 작은 떨림이 다른 칸만큼 커져 축을 끌어당긴다. 이 데이터에서 2D 지도 점수가 60.3% → 51.4%로 떨어진다. 단위가 서로 다른 표라면 표준화가 맞다. 까닭을 보고 정한다.'],
 ['pca = PCA(n_components=2)',0,'지도용으로 축 두 개. 괜찮다.'],
 ['Z = pca.fit_transform(X.T)',2,'<b>행과 열을 뒤집었다.</b> X.T는 64×1,797이라 "칸 64개를 샘플로, 글씨 1,797장을 특징으로" 본다. 오류 없이 돌지만 글씨가 아니라 칸의 지도가 나온다. scikit-learn은 늘 (샘플 수, 특징 수) 모양을 받는다. <code>X.shape</code>를 먼저 확인한다.'],
 ['print("설명된 분산 비율:", pca.explained_variance_)',3,'<b>분산과 분산 비율을 헷갈렸다.</b> <code>explained_variance_</code>는 축마다의 분산(이 데이터에서 179, 164 같은 큰 숫자)이고, 비율은 <code>explained_variance_ratio_</code>(0.149, 0.136)다. 보고서에 "179%"가 적히는 사고가 여기서 난다.'],[''],
 ['# 2성분으로 복원해 보기',0,'주석. 괜찮다.'],
 ['X_back = Z @ pca.components_',4,'<b>평균을 빠뜨렸다.</b> PCA는 평균을 빼고 축에 비췄으므로, 되살릴 때 평균을 다시 더해야 한다. <code>pca.inverse_transform(Z)</code> 또는 <code>Z @ pca.components_ + pca.mean_</code>. 빠뜨리면 평균 글씨만큼 어두운 "유령 글씨"가 나온다.'],[''],
 ['knn = KNeighborsClassifier(n_neighbors=5).fit(Z, y)',0,'채점 도구. 정답은 여기서 처음 쓴다. 이 줄 자체는 괜찮다.'],
 ['print("지도 점수:", knn.score(Z, y))',5,'<b>지도를 만든 점들로 바로 채점했다.</b> 자기 자신이 "가장 가까운 다섯 장" 안에 들어가 점수가 부풀려진다. 이 데이터의 2D 지도에서 73.0%가 나오지만 5겹 교차 검증은 60.3%다. <code>cross_val_score(knn, Z, y, cv=5).mean()</code>을 쓴다.']];
  const host=$('bugCode'),found=new Set();
  host.innerHTML=LINES.map((l,i)=>'<div class="ln" role="listitem" tabindex="0" data-i="'+i+'"><span class="no">'+(i+1)+'</span><span>'+hl(l[0]||' ')+'</span></div>').join('');
  function score(){$('bugScore').textContent=found.size+' / 5';$('bugScore').className='pill'+(found.size===5?' ok':'');}
  function pick(el){const i=+el.getAttribute('data-i'),l=LINES[i];if(!l[0])return;
    if(l[1]){found.add(l[1]);el.classList.add('hit');$('bugMsg').innerHTML=(i+1)+'번 줄: '+l[2]+(found.size===5?'<br><b>다섯 군데를 모두 찾았다.</b> 고친 코드는 A장의 Colab 코드와 같은 순서(모양 확인 → 표준화 여부를 까닭으로 → 비율로 보고 → inverse_transform으로 복원 → 교차 검증으로 채점)를 따른다.':'');}
    else{el.classList.add('miss');$('bugMsg').innerHTML=(i+1)+'번 줄: '+(l[2]||'이 줄은 괜찮다.');}score();}
  host.addEventListener('click',e=>{const el=e.target.closest('.ln');if(el)pick(el);});
  host.addEventListener('keydown',e=>{if((e.key==='Enter'||e.key===' ')&&e.target.classList.contains('ln')){e.preventDefault();pick(e.target);}});
  $('bugReveal').addEventListener('click',()=>{LINES.forEach((l,i)=>{if(l[1]){found.add(l[1]);host.querySelector('[data-i="'+i+'"]').classList.add('hit','reveal');}});score();$('bugMsg').innerHTML='빨간 줄 다섯 군데: ① 8번 같은 단위인데 "표준화 필수" ② 10번 X.T로 행과 열 뒤집기 ③ 11번 분산을 비율로 보고 ④ 14번 복원에서 평균 빠뜨림 ⑤ 17번 지도를 만든 점으로 바로 채점.';});
  $('bugReset').addEventListener('click',()=>{found.clear();host.querySelectorAll('.ln').forEach(x=>x.classList.remove('hit','miss','reveal'));score();$('bugMsg').textContent='줄을 하나 눌러 보세요.';});score();})();
(function(){const CK=[['데이터 모양을 확인했나','X.shape가 (1797, 64)인지. 행이 샘플, 열이 특징이다. X.T를 넣지 않았나.'],['정답 y를 PCA에 넣지 않았나','PCA.fit(X)만 한다. 정답은 그림 색칠과 채점 칸에서만 쓴다. 비지도 지원자의 규칙이다.'],['표준화 여부를 까닭으로 정했나','열의 단위가 같으면 대개 하지 않고, 다르면 한다. "늘 한다"도 "늘 안 한다"도 아니다.'],['비율과 분산을 구분했나','explained_variance_ratio_가 비율(합 1), explained_variance_는 분산이다.'],['성분 수의 근거가 있나','누적 90%(이 데이터 21성분), 엘보, 또는 쓰임새(지도 2, 전처리 20~30).'],['복원에 평균을 더했나','inverse_transform을 쓰거나 mean_을 더한다.'],['지도 점수를 교차 검증으로 쟀나','지도를 만든 점 그대로 knn.score를 하면 부풀려진다(이 데이터 73.0% vs 60.3%).'],['정답 색칠 그림만 보고 결론 내지 않았나','"잘 나뉘어 보인다"는 숫자가 아니다. 5-NN 정확도, 겹치는 숫자 쌍을 숫자로 적는다.'],['합격선·기준선과 비교했나','매뉴얼 씨 지도 18.1%, 가장 흔한 숫자 10.2%보다 낫지 않으면 뽑을 까닭이 없다.']];
  const ul=$('checks');ul.innerHTML=CK.map((c,i)=>'<li><input type="checkbox" id="ck'+i+'"><label for="ck'+i+'"><b>'+c[0]+'</b><small>'+c[1]+'</small></label></li>').join('');
  function upd(){const n=ul.querySelectorAll('input:checked').length;ul.querySelectorAll('li').forEach(li=>li.classList.toggle('done',li.querySelector('input').checked));$('checkScore').textContent='확인 '+n+' / '+CK.length+(n===CK.length?' — 이제 돌려 본다. 결과도 확인 대상이다.':'');}
  ul.addEventListener('change',upd);upd();})();

/* ── 04 수습 기간 ① 거듭제곱법으로 축 찾기 ── */
const POW=(function(){const cv=$('powCv'),MAXK=5,IT=30;let st=null;
  function reset(){const p=ensureP(),R=RNG(3);let v=new Float64Array(64).map(()=>R()-0.5),nr=Math.sqrt(v.reduce((s,a)=>s+a*a,0));v=v.map(a=>a/nr);st={C:p.C,k:0,v,hist:[],found:[],busy:false};draw();info();}
  function step(){const r=PCAE.powerStep(st.C,st.v);st.v=r.v;st.hist.push(r.lam/P.total);}
  function draw(){if(!st)return;const W=cv.clientWidth||cv.parentElement.clientWidth||600,wide=W>=640,S=wide?Math.min(230,W*0.34):Math.min(W-20,240),H=wide?S+60:S+260;fit(cv,H);const ctx=cv.getContext('2d'),C=cols();ctx.clearRect(0,0,W,H);
    drawComp(ctx,st.v,10,24,S,C);ctx.fillStyle=C.text;ctx.font=font(13,700);ctx.textAlign='left';ctx.fillText('지금 찾는 축 '+(st.k+1)+' · 되풀이 '+st.hist.length+'번',10,16);
    const fx0=wide?S+84:58,fy0=wide?24:S+70,fw=W-fx0-14,fh=wide?S:150,lx=i=>fx0+i/IT*fw,ly=v=>fy0+(1-v/0.17)*fh;
    axes(ctx,C,fx0,fy0,fw,fh);[0,10,20,30].forEach(i=>{ctx.textAlign='center';ctx.fillText(i,lx(i),fy0+fh+15);});ctx.fillText('되풀이 횟수',fx0+fw/2,fy0+fh+30);
    [0,0.05,0.1,0.15].forEach(v=>{ctx.textAlign='right';ctx.fillText(pct(v,0),fx0-5,ly(v)+4);});vlabel(ctx,'그 방향의 분산 비율',fx0-46,fy0+fh/2);
    const truth=P.ratio[st.k];ctx.strokeStyle=rgba(C.stamp,.7);ctx.setLineDash([5,4]);ctx.beginPath();ctx.moveTo(fx0,ly(truth));ctx.lineTo(fx0+fw,ly(truth));ctx.stroke();ctx.setLineDash([]);ctx.fillStyle=C.stamp;ctx.font=font(12,700);ctx.textAlign='right';ctx.fillText('고유값 정답 '+pct(truth),fx0+fw-6,ly(truth)-6);
    st.found.forEach((f,i)=>{ctx.fillStyle=rgba(C.muted,.8);ctx.textAlign='left';ctx.font=font(11.5,500,1);ctx.fillText('축 '+(i+1)+' '+pct(f,1),fx0+6+i*Math.min(90,fw/5),fy0+fh-8);});
    if(st.hist.length){ctx.strokeStyle=C.blue;ctx.lineWidth=2.6;ctx.beginPath();st.hist.forEach((v,i)=>{const X_=lx(i+1),Y_=ly(v);i?ctx.lineTo(X_,Y_):ctx.moveTo(X_,Y_);});ctx.stroke();circ(ctx,lx(st.hist.length),ly(st.hist[st.hist.length-1]),5,C.blue,null);}}
  function info(){const n=st.hist.length,v=n?st.hist[n-1]:0,cs=Math.abs(st.v.reduce((s,a,j)=>s+a*P.comps[st.k][j],0));
    let h='축 <b>'+(st.k+1)+'</b> · 되풀이 <b>'+n+'</b>번 · 이 방향의 분산 비율 <b>'+pct(v,1)+'</b> · 정답 성분과 닮은 정도 |cos| <b>'+cs.toFixed(4)+'</b>. ';
    if(!n)h+='무작위 방향에서 출발한다. 왼쪽 그림이 얼룩덜룩한 까닭이다. "축 찾기"를 누르면 공분산 표를 곱하고 길이를 1로 맞추기를 30번 되풀이한다.';
    else if(cs>0.999)h+='다 왔다. 왼쪽 그림이 04장 아래 "성분 '+(st.k+1)+'" 그림과 같다(부호가 반대면 색만 뒤집힌다). 분산 비율도 고유값 정답('+pct(P.ratio[st.k],1)+')과 같다. '+(st.k<MAXK-1?'"다음 축"을 누르면 찾은 몫(λvvᵀ)을 공분산 표에서 빼고 같은 일을 한다. 남은 것 중 가장 퍼진 방향, 곧 직각인 다음 축이 나온다.':'다섯 축까지 찾았다. scikit-learn은 같은 일을 한 번의 분해로 64축 모두 한다.');
    else h+='곱할 때마다 가장 퍼진 방향의 몫이 다른 방향보다 빨리 커져, 방향이 그쪽으로 돌아선다. '+(st.k>0&&n<15?'뒤의 축은 고유값끼리 가까워 조금 더 오래 걸린다.':'');
    $('powInfo').innerHTML=h;}
  async function play(){if(st.busy)return;st.busy=true;['powPlay','powNext'].forEach(i=>$(i).disabled=true);st.hist=[];const R=RNG(3+st.k);let v=new Float64Array(64).map(()=>R()-0.5),nr=Math.sqrt(v.reduce((s,a)=>s+a*a,0));st.v=v.map(a=>a/nr);
    for(let i=0;i<IT;i++){step();draw();info();await wait(REDUCE?0:(i<8?160:60));}st.busy=false;['powPlay','powNext'].forEach(i=>$(i).disabled=false);}
  $('powPlay').addEventListener('click',play);
  $('powNext').addEventListener('click',()=>{if(st.busy)return;if(!st.hist.length){play();return;}if(st.k>=MAXK-1){toast('다섯 축까지만 재생한다');return;}const lam=st.hist[st.hist.length-1]*P.total;st.found.push(st.hist[st.hist.length-1]);st.C=PCAE.deflate(st.C,st.v,lam);st.k++;st.hist=[];
    $('powPlay').textContent='▶ 축 '+(st.k+1)+' 찾기';const R=RNG(3+st.k);let v=new Float64Array(64).map(()=>R()-0.5),nr=Math.sqrt(v.reduce((s,a)=>s+a*a,0));st.v=v.map(a=>a/nr);draw();info();play();});
  $('powReset').addEventListener('click',()=>{if(st.busy)return;$('powPlay').textContent='▶ 축 1 찾기';reset();});
  onRedraw(draw);return {start(){if(!st)reset();}};})();
/* 성분 그림 */
function drawComps(){const cv=$('compCv'),W=cv.clientWidth||cv.parentElement.clientWidth||600,cn=W>=640?8:4,g=10,s=(W-g*(cn-1))/cn,two=s<130,LH=two?44:30,H=Math.ceil(16/cn)*(s+LH);fit(cv,H);const ctx=cv.getContext('2d'),C=cols(),p=ensureP();ctx.clearRect(0,0,W,H);
  for(let k=0;k<16;k++){const x=(k%cn)*(s+g),y=Math.floor(k/cn)*(s+LH);if(k===0)drawGlyph(ctx,Array.from(p.mean),x,y,s,C);else drawComp(ctx,p.comps[k-1],x,y,s,C);
    ctx.fillStyle=k===0?C.text:C.muted;ctx.font=font(12,k<3?700:500);ctx.textAlign='center';if(two){ctx.fillText(k===0?'평균 글씨':'성분 '+k,x+s/2,y+s+15);if(k)ctx.fillText(pct(p.ratio[k-1],1),x+s/2,y+s+30);}else ctx.fillText(k===0?'평균 글씨':'성분 '+k+' ('+pct(p.ratio[k-1],1)+')',x+s/2,y+s+16);}}
function compInfo(){const p=ensureP();$('compInfo').innerHTML='성분 1('+pct(p.ratio[0],1)+')은 가운데 아래 왼쪽 칸(파랑)이 밝고 위·아래 가장자리 획(빨강)이 어두울수록 점수가 커진다. 성분 2('+pct(p.ratio[1],1)+')는 가운데 세로줄(파랑)과 오른쪽 아래(빨강)를 맞바꾼다. 봉투를 열어 확인해 보면 성분 1의 + 쪽에는 4·6이, − 쪽에는 3·9가 모이고, 성분 2의 + 쪽에는 7·1이, − 쪽에는 0·6이 모인다(05장 지도). 정답을 안 보고 고른 축인데도 숫자의 모양 차이를 잡았다. 뒤 성분일수록 무늬가 잘게 쪼개지고 몫이 작다. 얼굴 사진으로 같은 일을 하면 성분이 얼굴 모양으로 보여 "고유 얼굴(eigenface)"이라 부른다. 이 그림들은 "고유 숫자"인 셈이다.';}
/* 설명된 분산 */
const SCR=(function(){const cv=$('scrCv'),H=300;let k=2;
  function draw(){const p=ensureP(),W=fit(cv,H),ctx=cv.getContext('2d'),C=cols(),pl=46,pr=44,pt=12,pb=40,gw=W-pl-pr,gh=H-pt-pb,bw=gw/64,ly=v=>pt+(1-v)*gh,BM=0.2;ctx.clearRect(0,0,W,H);
    axes(ctx,C,pl,pt,gw,gh);[1,10,20,30,40,50,64].forEach(i=>{ctx.textAlign='center';ctx.fillText(i,pl+(i-0.5)*bw,H-24);});ctx.fillText('성분 번호',pl+gw/2,H-6);
    [0,0.25,0.5,0.75,1].forEach(v=>{ctx.textAlign='right';ctx.fillText(Math.round(v*100)+'%',pl-5,ly(v)+4);});vlabel(ctx,'누적 (선, 왼쪽 눈금)',11,pt+gh/2);[0,0.05,0.1,0.15,0.2].forEach(v=>{ctx.fillStyle=C.muted;ctx.textAlign='left';ctx.fillText(Math.round(v*100)+'%',pl+gw+5,pt+(1-v/BM)*gh+4);});ctx.save();ctx.translate(W-6,pt+gh/2);ctx.rotate(Math.PI/2);ctx.textAlign='center';ctx.fillText('막대: 성분마다 (오른쪽 눈금)',0,0);ctx.restore();
    p.ratio.forEach((r,i)=>{const h=Math.min(gh,r/BM*gh);ctx.fillStyle=rgba(i<k?C.blue:C.muted,i<k?.75:.35);ctx.fillRect(pl+i*bw+1,pt+gh-h,Math.max(1,bw-2),h);});
    [[0.8,'80%'],[0.9,'90%'],[0.95,'95%']].forEach(a=>{ctx.strokeStyle=rgba(C.muted,.5);ctx.setLineDash([3,4]);ctx.beginPath();ctx.moveTo(pl,ly(a[0]));ctx.lineTo(pl+gw,ly(a[0]));ctx.stroke();ctx.setLineDash([]);ctx.fillStyle=C.muted;ctx.textAlign='right';ctx.fillText(a[1],pl+gw-4,ly(a[0])-3);});
    ctx.strokeStyle=C.blue;ctx.lineWidth=2.6;ctx.beginPath();p.cum.forEach((v,i)=>{const X_=pl+(i+0.5)*bw,Y_=ly(v);i?ctx.lineTo(X_,Y_):ctx.moveTo(X_,Y_);});ctx.stroke();
    const kx=pl+(k-0.5)*bw;ctx.strokeStyle=C.stamp;ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(kx,pt);ctx.lineTo(kx,pt+gh);ctx.stroke();circ(ctx,kx,ly(p.cum[k-1]),5,C.stamp,null);
    ctx.fillStyle=C.stamp;ctx.font=font(13,700);ctx.textAlign=kx>pl+gw*0.6?'right':'left';ctx.fillText(k+'성분 → '+pct(p.cum[k-1],1),kx+(kx>pl+gw*0.6?-8:8),ly(p.cum[k-1])+(p.cum[k-1]>0.85?22:-10));}
  function info(){const p=ensureP(),c=p.cum[k-1];$('scrInfo').innerHTML='<b>'+k+'성분</b>: 누적 설명된 분산 <b>'+pct(c,1)+'</b> · 버린 몫 '+pct(1-c,1)+' · 복원 오차(픽셀당 제곱 오차, 1,797장 평균) <b>'+reconMSE(k).toFixed(2)+'</b> (0성분 = 평균만이면 '+reconMSE(0).toFixed(2)+') · 외울 숫자 '+(64+64*k).toLocaleString()+'개. '+
    (k===2?'2D 지도는 퍼짐의 3분의 1도 못 담는다. 그래도 지도가 쓸모 있는지는 05장에서 봉투를 열어 잰다.':k<p.n80?'아직 '+p.n80+'성분(80%)에 못 미친다.':k<p.n90?p.n80+'성분에서 80%를 넘었다. 90%는 '+p.n90+'성분.':k<p.n95?p.n90+'성분에서 90%를 넘었다. scikit-learn PCA(n_components=0.9)가 고르는 수다.':k<64?p.n95+'성분에서 95%를 넘었다. 남은 축은 대부분 잡음과 가장자리의 작은 떨림이다.':'64성분이면 100%. 축을 돌리기만 했을 뿐 버린 것이 없다.');}
  function set(v){k=clamp(+v,1,64);$('scrK').value=k;$('scrKO').textContent=k;document.querySelectorAll('#scrQuick button').forEach(b=>b.setAttribute('aria-pressed',+b.getAttribute('data-v')===k?'true':'false'));draw();info();}
  $('scrK').addEventListener('input',e=>set(e.target.value));seg('#scrQuick button',v=>set(v));onRedraw(()=>{if(P)draw();});return {start(){set(k);}};})();
whenNear('m4',()=>{ensureP();$('memN').textContent=(64+64*2);POW.start();drawComps();compInfo();SCR.start();onRedraw(drawComps);},400);

/* ── 05 성과 평가 ── */
function renderEval(){const K=ensureKNN2(),p=ensureP();
  $('evalStat').innerHTML=statGrid([[pct(K.acc),'PCA 2D 지도 5-NN (5겹)','a'],[pct(DR_LINE),'합격선 · 매뉴얼 씨 지도','m'],[pct(BASE),'기준선 · 가장 흔한 숫자'],[pct(p.cum[1]),'설명된 분산 · 정답 없이 재는 점수','s']],4)+
    '<p class="foot">채점은 지도를 다 그린 뒤 처음 봉투를 연다. 1,797장을 숫자 비율을 지키며 다섯 몫으로 나누고, 한 몫씩 돌아가며 "지도에서 나머지 네 몫 중 가장 가까운 다섯 장의 다수결"로 맞힌다(scikit-learn cross_val_score와 같은 나누기). 이 브라우저에서 '+Math.round(K.ms)+'ms. Colab(A장 7번 칸)도 '+pct(PRE.colab.acc2)+'.</p>';}
const MAP=(function(){const cv=$('mapCv'),pk=$('pickCv'),st={mode:'dot',sel:0};let geo=null;
  function draw(){const Z=ensureZ2(),W=cv.clientWidth||cv.parentElement.clientWidth||600,H=W>=700?500:Math.round(W*1.05);fit(cv,H);const ctx=cv.getContext('2d'),C=cols(),pl=44,pr=12,pt=12,pb=38,gw=W-pl-pr,gh=H-pt-pb,R=34;
    const sx=v=>pl+(v+R)/(2*R)*gw,sy=v=>pt+(1-(v+R)/(2*R))*gh;geo={sx,sy};ctx.clearRect(0,0,W,H);axes(ctx,C,pl,pt,gw,gh);
    [-30,-15,0,15,30].forEach(v=>{ctx.textAlign='center';ctx.fillText(v,sx(v),H-22);ctx.textAlign='right';ctx.fillText(v,pl-5,sy(v)+4);});ctx.textAlign='center';ctx.fillText('성분 1 점수',pl+gw/2,H-5);vlabel(ctx,'성분 2 점수',11,pt+gh/2);
    ctx.save();ctx.beginPath();ctx.rect(pl,pt,gw,gh);ctx.clip();
    if(st.mode==='img'){const s=W>=700?11:8;Z.forEach((z,i)=>drawDigit(ctx,X[i],sx(z[0])-s/2,sy(z[1])-s/2,s,SEAL.open?DIGCOL[Y[i]]:C.text));}
    else Z.forEach((z,i)=>circ(ctx,sx(z[0]),sy(z[1]),2.6,SEAL.open?rgba(DIGCOL[Y[i]],.75):rgba(C.text,.28),null));
    if(SEAL.open){for(let d=0;d<10;d++){let a=0,b=0,c=0;Z.forEach((z,i)=>{if(Y[i]===d){a+=z[0];b+=z[1];c++;}});const x=sx(a/c),y=sy(b/c);circ(ctx,x,y,13,C.panel,DIGCOL[d],2.5);ctx.fillStyle=DIGCOL[d];ctx.font=font(15,700,1);ctx.textAlign='center';ctx.fillText(d,x,y+5);}}
    const z=Z[st.sel];circ(ctx,sx(z[0]),sy(z[1]),8,null,C.stamp,2.5);ctx.restore();
    $('mapLeg').innerHTML=SEAL.open?DIGCOL.map((c,d)=>'<span><i style="background:'+c+'"></i>'+d+'</span>').join('')+'<span>큰 동그라미 = 그 숫자의 평균 자리</span>':'<span><i style="background:var(--muted)"></i>지원자가 보는 지도: 정답 없이 모두 같은 색</span>';drawPick();}
  function drawPick(){const Z=ensureZ2(),p=ensureP(),W=pk.clientWidth||pk.parentElement.clientWidth||300,s=Math.min(110,(W-30)/2),H=s+26;fit(pk,H);const ctx=pk.getContext('2d'),C=cols(),i=st.sel,r=PCAE.recon(p,Z[i],2);ctx.clearRect(0,0,W,H);
    drawGlyph(ctx,X[i],0,0,s,C);drawGlyph(ctx,Array.from(r),s+20,0,s,C,C.blue);ctx.fillStyle=C.muted;ctx.font=font(12,700);ctx.textAlign='center';ctx.fillText('원본 #'+(i+1),s/2,s+18);ctx.fillStyle=C.blue;ctx.fillText('2성분 복원',s*1.5+20,s+18);
    let e=0;for(let j=0;j<64;j++)e+=(X[i][j]-r[j])**2;
    const nb=[];if(SEAL.open){const d=Z.map((z,j)=>[(z[0]-Z[i][0])**2+(z[1]-Z[i][1])**2,j]).filter(a=>a[1]!==i).sort((a,b)=>a[0]-b[0]).slice(0,5);d.forEach(a=>nb.push(Y[a[1]]));}
    $('mapInfo').innerHTML='글씨 <b>#'+(i+1)+'</b> · 지도 좌표 ('+fx(Z[i][0],1)+', '+fx(Z[i][1],1)+') · 2성분 복원 오차 '+(e/64).toFixed(2)+'(픽셀당). '+(SEAL.open?'정답 <b>'+Y[i]+'</b>, 지도에서 가장 가까운 다섯 장의 정답 ['+nb.join(', ')+'] → 다수결 <b>'+[0,1,2,3,4,5,6,7,8,9].reduce((b,d)=>nb.filter(v=>v===d).length>nb.filter(v=>v===b).length?d:b,0)+'</b>. ':'정답은 봉투 속. ')+'오른쪽 복원은 이 글씨가 지도에서 가진 정보(점수 두 개)만으로 되살린 모습이다. 지도 위 같은 자리의 글씨는 모두 이 모습으로 보인다.';}
  cv.addEventListener('click',e=>{if(!geo)return;const r=cv.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top,Z=ensureZ2();let b=-1,bd=1e9;Z.forEach((z,i)=>{const d=(geo.sx(z[0])-mx)**2+(geo.sy(z[1])-my)**2;if(d<bd){bd=d;b=i;}});if(b>=0&&bd<400){st.sel=b;draw();}});
  $('mapEnv').addEventListener('click',()=>setSeal(!SEAL.open));seg('#mapMode button',v=>{st.mode=v;draw();});
  onRedraw(()=>{if(Z2)draw();});onSeal(()=>{if(Z2)draw();});return {draw};})();
function renderOverlap(){const K=ensureKNN2(),M=[];for(let a=0;a<10;a++)M.push(new Array(10).fill(0));Y.forEach((y,i)=>M[y][K.pred[i]]++);
  const acc=M.map((r,d)=>r[d]/r.reduce((s,v)=>s+v,0)),pairs=[];for(let a=0;a<10;a++)for(let b=a+1;b<10;b++)pairs.push([M[a][b]+M[b][a],a,b]);pairs.sort((x,y)=>y[0]-x[0]);
  $('digBars').innerHTML=barRows(acc.map((v,d)=>['숫자 '+d,v,DIGCOL[d],pct(v)]),1);
  const ord=acc.map((v,d)=>[v,d]).sort((a,b)=>b[0]-a[0]);
  $('overlapInfo').innerHTML='잘 찾는 숫자: <b>'+ord.slice(0,3).map(a=>a[1]+'('+pct(a[0],0)+')').join(', ')+'</b>. 지도 가장자리에 자기 자리가 있다. 못 찾는 숫자: <b>'+ord.slice(-3).reverse().map(a=>a[1]+'('+pct(a[0],0)+')').join(', ')+'</b>. 가장 많이 서로 헷갈린 쌍은 '+pairs.slice(0,4).map(p=>'<b>'+p[1]+'↔'+p[2]+'</b> '+p[0]+'장').join(', ')+'이다. 이 숫자들은 64칸에서는 갈라지지만, 가장 퍼진 곧은 축 두 개에 비친 그림자로는 지도 한가운데에서 포개진다. 성분 3 이후의 축(04장)이 그 차이를 담고 있다. 05장 아래 "축을 늘리면"에서 확인한다.';}
/* 복원 */
const REC=(function(){const cv=$('recCv'),sel=$('recPick'),IDX=[0,1,2,3,4,5,6,7,8,9,10,11,25,42,77,100,333,512,1000,1500];let k=2;
  IDX.forEach(i=>sel.add(new Option('글씨 #'+(i+1),i)));
  function draw(){const p=ensureP(),i=+sel.value,W=cv.clientWidth||cv.parentElement.clientWidth||600,s=Math.min(150,(W-40)/3),ss=Math.min(70,(W-6*8)/7),H=s+30+ss+30;fit(cv,H);const ctx=cv.getContext('2d'),C=cols();ctx.clearRect(0,0,W,H);
    const z=PCAE.project(p,X[i],64),r=PCAE.recon(p,z,k),diff=X[i].map((v,j)=>v-r[j]);
    drawGlyph(ctx,X[i],0,0,s,C);drawGlyph(ctx,Array.from(r),s+20,0,s,C,C.blue);drawComp(ctx,diff,2*s+40,0,s,C);
    ctx.font=font(12.5,700);ctx.textAlign='center';ctx.fillStyle=C.text;ctx.fillText('원본',s/2,s+18);ctx.fillStyle=C.blue;ctx.fillText(k+'성분 복원',s*1.5+20,s+18);ctx.fillStyle=C.stamp;ctx.fillText(s<130?'버린 정보':'버린 정보 (원본 − 복원)',s*2.5+40,s+18);
    [0,1,2,5,10,21,64].forEach((kk,q)=>{const rr=PCAE.recon(p,z,kk),x=q*(ss+8),y=s+30;drawGlyph(ctx,Array.from(rr),x,y,ss,C,kk===k?C.blue:C.text);ctx.fillStyle=kk===k?C.blue:C.muted;ctx.font=font(11.5,kk===k?700:500);ctx.textAlign='center';ctx.fillText(kk===0?'평균만':kk+'성분',x+ss/2,y+ss+16);});
    let e=0;diff.forEach(v=>e+=v*v);e/=64;
    $('recInfo').innerHTML='글씨 #'+(i+1)+' · <b>'+k+'성분</b> 복원: 이 글씨의 픽셀당 제곱 오차 <b>'+e.toFixed(2)+'</b> · 1,797장 평균 '+reconMSE(k).toFixed(2)+' · 남긴 퍼짐 '+pct(k?p.cum[k-1]:0,1)+'. '+
      (k===0?'점수를 하나도 안 쓰면 모든 글씨가 평균 글씨가 된다.':k<=2?'두 점수로는 "평균에서 어느 쪽으로 조금 기울었나"만 남는다. 흐리고 뭉개져, 다른 숫자와 헷갈리기 쉽다. 오른쪽 빨강·파랑이 버린 정보다.':k<10?'획의 큰 흐름이 돌아온다.':k<21?'거의 알아볼 수 있다. 숫자 구별에 필요한 정보는 앞쪽 축에 몰려 있다.':k<64?'원본과 거의 같다. 남은 차이는 잔물결 같은 작은 떨림이다.':'64성분이면 원본과 똑같다. PCA는 축을 돌린 것뿐이라 다 쓰면 잃는 것이 없다.');}
  function set(v){k=clamp(+v,0,64);$('recK').value=k;$('recKO').textContent=k;draw();}
  $('recK').addEventListener('input',e=>set(e.target.value));sel.addEventListener('change',draw);onRedraw(()=>{if(P)draw();});return {start(){set(k);}};})();
/* 성분 수를 늘리면 */
let KRES=null;
$('kRun').addEventListener('click',async()=>{const b=$('kRun');if(b.disabled)return;b.disabled=true;const p=ensureP(),KS=[1,2,3,5,10,30,64],res=[];
  for(const k of KS){$('kBars').innerHTML='<p class="muted">'+k+'성분 채점 중…</p>';await wait(30);const Z=k===64?X:PCAE.transform(p,X,k);res.push([k,PCAE.knnCV(Z,Y,5,5).acc]);}
  KRES=res;$('kBars').innerHTML=barRows(res.map(r=>[(r[0]===64?'64 (원래 칸)':r[0]+'성분')+(r[0]===2?' · 지도':''),r[1],r[0]===2?'var(--blue)':'var(--muted)',pct(r[1])]),1)+barRows([['매뉴얼 씨 지도',DR_LINE,'var(--c-man)',pct(DR_LINE)]],1);
  $('kInfo').innerHTML='축 두 개에서 '+pct(res[1][1])+', 세 개에서 '+pct(res[2][1])+', 다섯 개에서 '+pct(res[3][1])+', 열 개면 '+pct(res[4][1])+'로 원래 64칸('+pct(res[6][1])+')에 거의 닿는다. 숫자를 가르는 정보는 앞쪽 10축 안에 대부분 들어 있다. 그러나 사람이 볼 수 있는 지도는 2축(기껏해야 3축)뿐이다. 2축 지도의 겹침은 PCA가 정보를 몰라서가 아니라 <b>곧은 축 두 개에 다 담을 수 없어서</b>다. 휘어 있는 무리를 펴서 2차원에 담는 일은 5-15 t-SNE·UMAP이 한다. (1·2·3·5·10·30·64성분 모두 이 브라우저 계산, Colab A장 7번 칸과 같다.)';b.disabled=false;b.textContent='다시 채점하기';});
/* 업무 점수 */
const BIZ=(function(){let seed=3,res=null,r10=null;const cv=$('bizCv');
  function run(){const Z=ensureZ2(),t0=performance.now(),M=PCAE.kmeans(Z,10,{nInit:10,seed}),L=PCAE.repLabel(Z,Y,M),ari=PCAE.ari(Y,Array.from(M.labels));
    const cl=[];for(let c=0;c<10;c++){const mem=[];M.labels.forEach((l,i)=>{if(l===c)mem.push(i);});const cnt=new Array(10).fill(0);mem.forEach(i=>cnt[Y[i]]++);const rep=L.rep[c];cl.push({c,rep,size:mem.length,name:Y[rep],ok:cnt[Y[rep]],maj:Math.max(...cnt)});}
    cl.sort((a,b)=>a.name-b.name||b.size-a.size);const pur=cl.reduce((s,a)=>s+a.maj,0)/N;res={acc:L.acc,ari,pur,cl,ms:performance.now()-t0};}
  function draw(){if(!res)return;const W=cv.clientWidth||cv.parentElement.clientWidth||600,cn=W>=640?10:5,s=Math.min(64,(W-(cn-1)*10)/cn),cw=(W-(cn-1)*10)/cn,H=Math.ceil(10/cn)*(s+58);fit(cv,H);const ctx=cv.getContext('2d'),C=cols();ctx.clearRect(0,0,W,H);
    res.cl.forEach((a,q)=>{const x=(q%cn)*(cw+10)+(cw-s)/2,y=Math.floor(q/cn)*(s+58);drawGlyph(ctx,X[a.rep],x,y,s,C);ctx.textAlign='center';ctx.fillStyle=C.text;ctx.font=font(12.5,700);ctx.fillText('이름 "'+a.name+'"',x+s/2,y+s+16);
      ctx.fillStyle=C.muted;ctx.font=font(11.5,500);ctx.fillText(a.size+'장',x+s/2,y+s+31);const f=a.ok/a.size;ctx.fillStyle=f>=0.6?C.ok:f>=0.35?C.amber:C.stamp;ctx.font=font(11.5,700);ctx.fillText('맞음 '+pct(f,0),x+s/2,y+s+46);});}
  function info(){const names=res.cl.map(a=>a.name),dup=names.filter((v,i)=>names.indexOf(v)!==i),miss=[0,1,2,3,4,5,6,7,8,9].filter(d=>names.indexOf(d)<0);
    if(!r10){const Z10=PCAE.transform(ensureP(),X,10),M=PCAE.kmeans(Z10,10,{nInit:10,seed:1});r10={acc:PCAE.repLabel(Z10,Y,M).acc,ari:PCAE.ari(Y,Array.from(M.labels))};}
    $('bizStat').innerHTML=statGrid([[pct(res.acc),'2D 지도 위 대표 10장으로 이름 붙이기','a'],[pct(res.pur),'묶음마다 다수결 이름이라면 (최대)'],[pct(r10.acc),'PCA 10성분 위에서 (지도 아님)','s'],[pct(PRE.pre.kmRepRaw),'원래 64칸 위에서 (Python)']],4);
    $('bizInfo').innerHTML='사람이 이름을 붙인 글씨는 <b>10장</b>뿐이다. 그 이름을 지도 위 같은 묶음 1,797장에 퍼뜨리면 <b>'+pct(res.acc)+'</b>가 맞는다. 묶음 ARI '+fx(res.ari,3)+'. '+(dup.length?'대표가 같은 이름을 받은 묶음이 있다('+[...new Set(dup)].join(', ')+'). 그 대신 아무 묶음도 못 받은 숫자('+miss.join(', ')+')는 전부 틀린다. ':'')+
      '지도에서 겹친 무리는 대표 한 장이 어느 숫자냐에 따라 결과가 크게 흔들린다. "다른 시작값으로 다시 묶기"를 눌러 보면 대략 40~50% 사이를 오간다. 묶음마다 가장 많은 숫자로 이름을 붙였다면(사람이 묶음을 다 훑어봐야 가능) '+pct(res.pur)+'까지다. 같은 일을 지도가 아니라 PCA 10성분 위에서 하면 '+pct(r10.acc)+'로 원래 64칸('+pct(PRE.pre.kmRepRaw)+')과 거의 같다. 라벨링 비용을 줄이는 데는 <b>사람 눈용 2축 지도보다 기계용 10축</b>이 맞다. <span class="muted">(시작값 씨앗 '+seed+'. 기본 씨앗 3은 Colab A장 8번 칸의 결과 '+pct(PRE.pre.kmRep2)+'와 가까운 것을 골랐다. K-Means는 이 브라우저 계산 '+Math.round(res.ms)+'ms.)</span>';}
  $('bizAgain').addEventListener('click',()=>{seed++;run();draw();info();});onRedraw(draw);return {start(){if(!res){run();draw();info();}}};})();
whenNear('m5',()=>{renderEval();MAP.draw();renderOverlap();REC.start();},600);
whenNear('bizCv',()=>BIZ.start(),300);

/* ── 면접 결과 발표 ── */
Interview('iv14',{title:'PCA는 손글씨 1,797장을 2D 지도로 펼치는 일에 합격할까?',
  q:'조건: 정답을 보지 않고 그린 2D 지도(성분 1·2)에서 5-NN 정확도(5겹 교차 검증)가 합격선(매뉴얼 씨 지도 '+pct(DR_LINE)+')과 기준선(가장 흔한 숫자 '+pct(BASE)+')을 모두 넘으면 합격. 근무 조건은 n_components = 2, 표준화 없음. 축은 브라우저가 1,797장 전체로 직접 구한다.',
  fly:'봉투 안 보고 그린 지도 맞지? 서로 다른 숫자가 지도에서 얼마나 겹쳐?',robo:'외울 숫자 몇 개야? 새 글씨 하나 지도에 올리는 데 얼마 들어?',
  async run(setP){setP(.3);await wait(30);const p=ensureP();setP(.6);await wait(30);const K=ensureKNN2();setP(1);
    const a=K.acc,pass=a>DR_LINE&&a>BASE,M=[];for(let i=0;i<10;i++)M.push(new Array(10).fill(0));Y.forEach((y,i)=>M[y][K.pred[i]]++);const a8=M[8][8]/M[8].reduce((s,v)=>s+v,0);
    return {result:pass?'pass':'hold',flyAns:'5-NN '+pct(a)+' · 축은 정답 없이 1,797장으로만 골랐음 · 숫자 8은 '+pct(a8)+'만 제자리, 1·5·7과 겹침',roboAns:'외울 숫자 192개(평균 64 + 축 2개 × 64) · 학습 '+Math.max(1,Math.round(p.ms))+'ms(이 브라우저) · 새 글씨 1장 = 곱셈 128번',
      flyMood:pass?'coffee':'cry',flyMove:pass?'tilt':'shake',roboMood:pass?'coin':'meh',roboMove:pass?'pop':'tilt',asstMood:pass?'spark':'sigh',docMood:pass?'judge':'scale',
      measure:'채점 봉투를 열었어요. 2D 지도 5-NN <b>'+pct(a)+'</b>로 합격선 '+pct(DR_LINE)+'보다 '+fx((a-DR_LINE)*100,1)+'%p, 기준선 '+pct(BASE)+'보다도 높아요. 정답 없이 재는 설명된 분산은 '+pct(p.cum[1])+'. 64칸 퍼짐의 3분의 1도 안 남겼는데 이만큼 맞힙니다.',
      verdict:pass?'합격입니다. 차원축소팀에 배치합니다. 조건이 둘 있어요. 지도로 이름을 퍼뜨릴 때는 겹치는 숫자(1·5·8·9)를 사람이 더 확인할 것, 다른 지원자에게 넘길 때는 2축이 아니라 90~95%를 담는 20~30축으로 넘길 것.':'보류입니다. 합격선을 넘지 못했어요.',
      key:'그림자 두 축 — 5-NN '+pct(DR_LINE)+' → '+pct(a)+' · 설명된 분산 '+pct(p.cum[1]),
      render(box){box.innerHTML='<div class="bars">'+barRows([['PCA 2D 지도',a,'var(--blue)',pct(a)],['매뉴얼 씨 지도 (합격선)',DR_LINE,'var(--c-man)',pct(DR_LINE)],['가장 흔한 숫자 (기준선)',BASE,'var(--muted)',pct(BASE)]],1)+'</div>';}};},
  alts:[
   {label:'표준화한 뒤 PCA',async run(box){const d=64,mu=new Array(d).fill(0),sd=new Array(d).fill(0);X.forEach(x=>{for(let j=0;j<d;j++)mu[j]+=x[j]/N;});X.forEach(x=>{for(let j=0;j<d;j++)sd[j]+=(x[j]-mu[j])**2/N;});
     const Xs=X.map(x=>x.map((v,j)=>sd[j]>0?(v-mu[j])/Math.sqrt(sd[j]):0)),Ps=PCAE.fit(Xs),Zs=PCAE.transform(Ps,Xs,2),a=PCAE.knnCV(Zs,Y,5,5).acc,a0=ensureKNN2().acc;
     box.innerHTML=statGrid([[pct(a),'표준화 + PCA 2D 지도 5-NN','a'],[pct(a0),'표준화 없음 (본 조건)','s'],[pct(Ps.ratio[0]+Ps.ratio[1]),'설명된 분산 (표준화 공간)'],[pct(DR_LINE),'합격선','m']],4)+'<p class="foot">판정: '+(a>DR_LINE&&a>BASE?/*i18n:verdict*/'합격':'<b>보류</b>')+'. 합격선은 넘지만 본 조건보다 '+fx((a0-a)*100,1)+'%p 낮다. 64칸이 모두 같은 단위(밝기)인데 표준화하면, 거의 늘 0인 가장자리 칸(분산이 아주 작다)의 떨림이 다른 칸만큼 커져서 축을 엉뚱한 쪽으로 끌어당긴다. 5-1 KNN에서 표준화는 약이었지만(단위가 다른 콘크리트 재료) 여기서는 독이다. Colab A장 7번 칸도 '+pct(PRE.colab.accStd)+'.</p>';}},
   {label:'성분 1개만 (1차원 줄)',async run(box){const p=ensureP(),Z1=PCAE.transform(p,X,1),a=PCAE.knnCV(Z1,Y,5,5).acc;
     box.innerHTML=statGrid([[pct(a),'성분 1 하나로 5-NN','a'],[pct(p.ratio[0]),'설명된 분산'],[pct(DR_LINE),'합격선','m']],3)+'<p class="foot">판정: '+(a>DR_LINE&&a>BASE?/*i18n:verdict*/'합격':'<b>보류</b>')+'. 한 줄로 세워도 매뉴얼 씨의 지도(사실상 잉크 양 한 줄)보다 낫다. 잉크 양도 "64칸을 모두 같은 무게로 더한" 축 하나이고, 성분 1도 64칸에 무게를 달리 준 축 하나다. 같은 한 줄인데 무게를 데이터가 고르니 '+pct(DR_LINE)+' → '+pct(a)+'가 된다. 그래도 한 줄에 열 숫자를 늘어세우면 많이 겹친다.</p>';}},
   {label:'성분 10개 (지도가 아닌 전처리)',async run(box){const p=ensureP(),Z10=PCAE.transform(p,X,10),a=PCAE.knnCV(Z10,Y,5,5).acc;
     box.innerHTML=statGrid([[pct(a),'10성분 공간 5-NN','a'],[pct(p.cum[9]),'설명된 분산'],[pct(PRE.team.raw),'원래 64칸 5-NN (참고)']],3)+'<p class="foot">숫자를 가르는 정보는 앞 10축에 거의 다 들어 있다. 다만 10차원은 사람이 볼 수 없어 이 면접의 과제(지도)에는 해당하지 않는다. 판정은 2D 지도로만 한다. 이 10축은 다른 지원자에게 넘기는 <b>전처리</b> 용도다. 계산할 칸이 64개에서 10개로 줄어 5-NN이나 K-Means가 여섯 배 가벼워진다(06장).</p>';}}]});

/* ── 06 인사 카드 ── */
function renderCard(){const K=ensureKNN2(),a=K.acc;
  const T=[['매뉴얼 씨 지도 (5-0)','사람이 고른 두 숫자: 잉크 양, 위쪽 비율',PRE.team.manual,'규칙 그대로 계산','합격선',''],
    ['<b>PCA 2D</b> (이 면접)','가장 퍼진 곧은 축 두 개',a,'평균 빼고 축과 곱하기 (transform)',a>DR_LINE&&a>BASE?'pass':'hold','브라우저 계산'],
    ['t-SNE 2D (5-15 예정)','가까운 이웃 관계를 지키며 휘어 펼치기',PRE.team.tsne,'못 한다. 새 글씨가 오면 다시 그린다','예정','Python 미리 계산 (A장 10번 칸)'],
    ['UMAP 2D (5-15 예정)','이웃 그래프를 지키며 펼치기',PRE.team.umap,'transform으로 가능 (근사)','예정','작업 기록 값, 5-15에서 다시 잰다'],
    ['참고: 원래 64칸','줄이지 않음 (지도가 아님)',PRE.team.raw,'—','—','Python']];
  $('teamTbl').innerHTML='<thead><tr><th>지도</th><th>만드는 법</th><th>5-NN (5겹)</th><th>새 글씨 올리기</th><th>판정</th></tr></thead><tbody>'+
    T.map((r,i)=>'<tr'+(i===1?' style="background:var(--panel2)"':'')+'><td>'+r[0]+'</td><td style="font-size:13px">'+r[1]+'</td><td><b style="color:var(--text)">'+pct(r[2])+'</b><br><small class="muted">'+r[5]+'</small></td><td style="font-size:13px">'+r[3]+'</td><td>'+(r[4]==='pass'?'<span class="pill ok">합격</span>':r[4]==='hold'?'<span class="pill no">보류</span>':'<span class="muted">'+r[4]+'</span>')+'</td></tr>').join('')+'</tbody>';
  $('teamFoot').innerHTML='PCA 줄만 이 브라우저가 계산했다. t-SNE는 Colab A장 10번 칸이 같은 채점으로 '+pct(PRE.team.tsne)+'를 낸다. UMAP은 이 빌드 환경에서 따로 잰 값이며 5-15에서 다시 잰다. t-SNE·UMAP은 PCA보다 훨씬 무겁고(1,797장에 수 초~수십 초), 지도 위 무리 사이의 거리와 무리 크기를 그대로 믿으면 안 된다는 주의가 붙는다. 그 이야기는 다음 면접에서 한다.';
  $('preTbl').innerHTML=[['5-13 GMM','PCA 30성분 (분산 '+pct(ensureP().cum[29])+')','<b>'+fx(PRE.pre.gmm30,3)+'</b> · 군집팀 1위'],['5-10 K-Means','PCA 10성분 (분산 '+pct(ensureP().cum[9])+')',fx(PRE.pre.km10,3)+' · 원래 64칸 '+fx(PRE.pre.kmRaw,3)+'과 거의 같고 계산은 6분의 1'],['5-10 K-Means','PCA 2D 지도 (분산 '+pct(ensureP().cum[1])+')',fx(PRE.pre.km2,3)+' · 지도 위에서 묶으면 겹침 때문에 떨어진다']].map(r=>'<tr><td>'+r[0]+'</td><td>'+r[1]+'</td><td>'+r[2]+'</td></tr>').join('');}
whenNear('m6',renderCard,600);

/* ══ 확인 문제 ══ */
var QUIZ={
q1:[{q:'PCA가 첫 번째 축으로 고르는 방향은?',o:[['데이터를 비춘 그림자(투영)의 분산이 가장 큰 방향',1,'가장 넓게 퍼지는 방향이다. 이 방향은 축 하나로 되살릴 때 잃는 제곱 오차가 가장 작은 방향이기도 하다.'],['정답(숫자)을 가장 잘 가르는 방향',0,'PCA는 정답을 보지 않는다. 정답으로 축을 고르는 것은 LDA 같은 지도 학습 방법이다.'],['원래 특징 중 분산이 가장 큰 칸 하나',0,'칸 하나가 아니라 64칸에 무게를 나눠 준 새 방향이다.']]},
 {q:'두 번째 주성분과 첫 번째 주성분의 관계는?',o:[['서로 직각이고, 두 축의 점수는 상관이 0이다',1,'첫 축과 직각인 방향 중 가장 퍼진 방향이다. 그래서 같은 정보를 겹쳐 담지 않는다.'],['두 번째로 분산이 큰 원래 칸이다',0,'원래 칸이 아니라 새 방향이다.'],['첫 축과 같은 방향이다',0,'같은 방향이면 새 정보가 없다.']]},
 {q:'01장 그림에서 축을 어떻게 돌려도 직각인 두 축의 분산 합이 같은 까닭은?',o:[['축을 돌리는 것은 좌표만 바꿀 뿐, 전체 퍼짐(분산의 합)은 그대로이기 때문',1,'그래서 첫 축이 많이 가져갈수록 나머지 축의 몫은 줄어든다. 64축 모두 더하면 언제나 100%다.'],['우연이다',0,'늘 성립한다. 회전은 거리와 퍼짐을 바꾸지 않는다.'],['두 칸의 밝기 단위가 같아서',0,'단위와 상관없이 성립한다.']]}],
q2:[{q:'이 면접에서 PCA 앞에 표준화를 하지 않은 까닭은?',o:[['64칸이 모두 같은 단위(밝기 0~16)이고, 표준화하면 거의 빈 가장자리 칸의 떨림이 과장되기 때문',1,'표준화하면 2D 지도 점수가 60.3% → 51.4%로 떨어진다. 단위가 서로 다른 표라면 표준화가 맞다.'],['PCA는 표준화를 하면 오류가 나서',0,'오류는 나지 않는다. 결과가 달라질 뿐이다.'],['표준화는 지도 학습에서만 쓰는 것이라서',0,'비지도에서도 쓴다. 까닭을 보고 정한다.']]},
 {q:'매뉴얼 씨의 2D 지도(잉크 양, 위쪽 비율)가 사실상 "잉크 양 한 줄"인 까닭은?',o:[['잉크 양은 0~1,000, 위쪽 비율은 0~1이라 거리를 잉크 양이 독차지하기 때문',1,'같은 지도를 표준화하면 18.1% → 27.0%로 오른다. 그래도 합격선은 원래 지도로 정했다.'],['위쪽 비율이 모든 글씨에서 같아서',0,'글씨마다 다르다. 다만 숫자 범위가 너무 작아 거리에 거의 안 보탠다.'],['5-NN이 첫 번째 특징만 보기 때문',0,'5-NN은 모든 특징으로 거리를 잰다.']]}],
q3:[{q:'PCA(n_components=2).fit_transform(X.T)의 문제는?',o:[['행과 열이 뒤집혀 글씨가 아니라 64칸을 샘플로 본다',1,'scikit-learn은 (샘플 수, 특징 수) 모양을 받는다. 오류 없이 돌아서 더 위험하다.'],['문제없다. 전치해도 같은 결과다',0,'전혀 다른 지도가 나온다.'],['n_components가 너무 작다',0,'지도용 2는 괜찮다.']]},
 {q:'"설명된 분산 비율"을 보고하려면 어느 속성을 쓰나?',o:[['explained_variance_ratio_',1,'합이 1인 비율이다. 이 데이터의 성분 1은 0.149.'],['explained_variance_',0,'축마다의 분산(179 같은 큰 값)이다. 비율이 아니다.'],['components_',0,'축의 방향(64칸 무게)이다.']]},
 {q:'Z @ pca.components_로 복원했더니 글씨가 이상하게 어둡다. 왜?',o:[['평균을 빼고 비췄으니 평균을 다시 더해야 한다',1,'pca.inverse_transform(Z) 또는 Z @ pca.components_ + pca.mean_.'],['성분 수가 부족해서',0,'흐린 것은 그 때문이지만, 전체가 평균만큼 어긋난 것은 평균을 빠뜨려서다.'],['부호가 뒤집혀서',0,'부호가 뒤집혀도 점수와 축이 함께 뒤집혀 복원은 같다.']]}],
q4:[{q:'공분산 행렬의 고유값 λ₁이 뜻하는 것은?',o:[['첫 번째 축에 비친 그림자의 분산',1,'그래서 λ₁ ÷ (λ 전부의 합)이 성분 1의 설명된 분산 비율이다.'],['첫 번째 축의 길이(늘 1)',0,'축 방향 벡터의 길이는 1이고, λ는 그 방향의 분산이다.'],['정답과의 상관',0,'정답은 쓰지 않는다.']]},
 {q:'손글씨 데이터에서 분산의 90%를 담으려면 성분이 몇 개 필요한가?',o:[['21개',1,'PCA(n_components=0.9)가 고르는 수다. 80%는 13개, 95%는 29개.'],['2개',0,'2개는 28.5%다.'],['64개',0,'64개면 100%다.']]},
 {q:'04장 거듭제곱법에서 아무 방향에서 시작해도 첫 축으로 모이는 까닭은?',o:[['공분산을 곱할 때마다 가장 큰 고유값 방향의 몫이 다른 방향보다 빨리 커지기 때문',1,'λ₁배, λ₂배, … 로 늘어나므로 되풀이할수록 λ₁ 방향이 압도한다. 길이는 매번 1로 맞춘다.'],['시작 방향을 정답으로 정해서',0,'무작위로 시작한다.'],['평균 글씨 쪽으로 끌려가서',0,'평균은 미리 뺐다.']]}],
q5:[{q:'PCA 2D 지도의 5-NN 정확도를 교차 검증으로 재는 까닭은?',o:[['지도를 만든 점 그대로 재면 자기 자신이 이웃에 들어가 점수가 부풀려진다',1,'이 데이터에서 그대로 재면 73.0%, 5겹 교차 검증은 60.3%다.'],['PCA가 교차 검증을 요구해서',0,'PCA 자체는 정답이 없다. 채점 방법의 문제다.'],['계산이 빨라서',0,'오히려 다섯 번 잰다.']]},
 {q:'2D 지도에서 5와 8이 많이 겹친 까닭으로 맞는 것은?',o:[['둘을 가르는 정보가 성분 1·2가 아닌 뒤쪽 축에 있어서',1,'10성분 공간에서는 5-NN이 93.7%까지 오른다. 곧은 두 축의 그림자에 다 담을 수 없었다.'],['PCA가 5와 8을 같은 숫자로 학습해서',0,'PCA는 숫자를 학습하지 않는다.'],['5와 8의 글씨가 64칸에서도 똑같아서',0,'64칸에서는 96%가 갈린다.']]},
 {q:'업무 점수에서 대표 10장으로 이름을 퍼뜨릴 때, 2D 지도보다 PCA 10성분이 나은 까닭은?',o:[['묶음이 숫자와 더 잘 맞아 대표 한 장의 이름이 묶음 전체에 맞을 확률이 커서',1,'2D 지도 약 49%(시작값에 따라 40~50%), 10성분 78%, 원래 64칸 79%.'],['10성분이 정답을 써서',0,'둘 다 정답을 쓰지 않는다. 대표 10장의 이름만 사람이 붙인다.'],['10이 10묶음과 같은 숫자라서',0,'성분 수와 묶음 수는 상관이 없다.']]}],
q6:[{q:'PCA가 t-SNE보다 나은 점은?',o:[['새 데이터를 평균 빼고 축과 곱하기만 하면 같은 지도에 올릴 수 있다',1,'transform이 있다. t-SNE는 새 점이 오면 지도를 다시 그려야 한다.'],['언제나 더 잘 나뉜 지도를 그린다',0,'이 데이터에서는 t-SNE가 97.6%로 훨씬 잘 나뉜다.'],['무리 사이 거리가 정확하다',0,'PCA는 곧은 투영이라 거리가 줄어들 뿐 늘지는 않지만, "정확하다"고 할 수는 없다.']]},
 {q:'5-13 GMM이 64칸 그대로가 아니라 PCA 30성분 위에서 일하는 까닭은?',o:[['64칸 공분산을 열 개 맞추기는 불안정하고, 30축이면 분산의 95.9%를 담으면서 안정된다',1,'그 결과 ARI 0.836으로 군집팀 1위다. PCA는 다른 지원자의 전처리로 쓰인다.'],['GMM은 30차원만 받는다',0,'차원 제한은 없다.'],['PCA가 정답을 알려 줘서',0,'PCA도 정답을 모른다.']]}]};

/* ══ 공통 B·C·D·E·F 데이터 ══ */
var PROMPTS=[
 {t:'① 내 데이터로 PCA 코드 받기',use:'내 공장의 측정·센서 데이터로 03장의 프롬프트를 맞출 때',p:MAIN_PROMPT},
 {t:'② AI 코드 검증 요청',use:'받은 코드를 다른 AI(또는 같은 AI의 새 대화)에게 검토받을 때',p:'아래 scikit-learn PCA 코드를 코드 리뷰해 줘. 특히 다음을 하나씩 확인하고, 문제가 있으면 줄 번호와 고친 코드를 보여 줘.\n1) 입력 X의 모양이 (샘플 수, 특징 수)인가. X.T를 넣지 않았나\n2) 표준화 여부가 열의 단위에 맞게 정해졌나 (같은 단위면 대개 하지 않고, 다르면 한다)\n3) explained_variance_와 explained_variance_ratio_를 바르게 썼나\n4) 복원에 inverse_transform을 쓰거나 mean_을 더했나\n5) 정답(라벨)을 PCA를 맞추는 데 쓰지 않았나. 채점에만 썼나\n6) 지도의 점수를 교차 검증으로 쟀나 (지도를 만든 점 그대로 knn.score를 하면 부풀려진다)\n7) 결과를 기준선(가장 흔한 라벨 비율)과 기존 규칙과 비교했나\n추측으로 답하지 말고, 확실하지 않은 부분은 "공식 문서 확인 필요"라고 표시해 줘.'},
 {t:'③ 공정 이상 감시에 PCA 쓰기',use:'센서 수십 개의 정상 데이터로 "평소와 다른 시편·로트"를 찾고 싶을 때',p:'〔센서 30개〕를 〔1분〕마다 기록한 〔정상 운전 3개월〕 데이터가 있다. 이것으로 PCA 기반 이상 감시 코드를 Colab용으로 써 줘.\n규칙:\n1) 열의 단위가 다르므로 정상 데이터로 StandardScaler를 맞추고, 같은 스케일러를 새 데이터에 쓴다\n2) 정상 데이터에서 누적 설명된 분산 〔90%〕가 되는 성분 수를 고르고, 그 성분 수를 출력한다\n3) 새 데이터마다 복원 오차(Q 통계량, SPE)와 주성분 점수의 크기(호텔링 T²)를 계산한다\n4) 경보 문턱은 정상 데이터의 〔99%〕 분위수로 정하고, 문턱을 넘은 시점과 그때 오차에 가장 크게 기여한 센서 3개를 표로 보여 준다\n5) 정상 데이터에 이미 이상이 섞여 있을 수 있다는 점과, 공정 조건이 바뀌면 모델을 다시 맞춰야 한다는 점을 주석으로 적는다\n데이터는 지어내지 말고, 내 CSV 파일 이름 〔process.csv〕를 읽는 칸으로 시작한다.'},
 {t:'④ 수업용 문제 만들기',use:'이 편으로 수업할 때 확인 문제가 더 필요할 때',p:'주성분 분석(PCA) 입문 수업용 문제를 만들어 줘. 대상은 〔공대 2학년, 선형대수는 행렬 곱까지만 앎〕이다. 예시는 손글씨 숫자(8×8 = 64칸)로 통일한다.\n- 투영·분산·직교·설명된 분산·복원을 각각 묻는 객관식 5문제 (보기 3개, 정답 해설 포함)\n- 오해를 드러내는 문제 2개: "PCA 앞에는 늘 표준화한다", "가장 퍼진 방향이 늘 가장 중요한 방향이다"\n- 손으로 풀 수 있는 계산 문제 1개: 점 네 개 (1,1), (2,2), (3,3), (4,2)의 평균을 빼고, 2×2 공분산 행렬을 구한 뒤, 45° 방향과 0° 방향 중 어느 쪽 그림자의 분산이 큰지 비교하기\n정답은 맨 끝에 따로 모아 줘.'}];
var GL=[
 ['모델','PCA','Principal Component Analysis, 주성분 분석. 데이터가 가장 넓게 퍼진 방향부터 차례로 서로 직각인 새 축을 세우고, 앞의 몇 축만 남겨 차원을 줄이는 비지도 방법.'],
 ['모델','주성분','PCA가 찾은 새 축 하나. 64칸에 무게(+/−)를 나눠 준 방향이다. scikit-learn의 components_. 성분이라고도 줄여 부른다.'],
 ['모델','차원 축소','특징(열)의 수를 줄이는 일. 그림으로 보려고(2·3차원), 계산을 가볍게 하려고, 잡음을 줄이려고 한다.'],
 ['모델','비지도 학습','정답(라벨) 없이 데이터의 구조를 찾는 학습. 군집과 차원 축소가 대표적이다.'],
 ['모델','t-SNE','가까운 이웃 관계를 지키도록 점을 2·3차원에 휘어 펼치는 차원 축소. 지도는 잘 나뉘지만 새 데이터를 올릴 수 없다(5-15).'],
 ['모델','UMAP','이웃 그래프를 지키며 펼치는 차원 축소. t-SNE보다 빠르고 새 데이터 배치(transform)를 지원한다(5-15).'],
 ['모델','KernelPCA','커널로 공간을 휜 뒤 PCA를 하는 방법. 곧은 축의 한계를 일부 넘는다.'],
 ['계산','차원','데이터 한 점을 나타내는 숫자의 개수. 손글씨 한 장은 64칸이라 64차원.'],
 ['계산','투영','점에서 축으로 수직으로 내려 그 축 위의 위치(그림자)를 얻는 일. 점과 축 방향의 내적으로 계산한다.'],
 ['계산','분산','값들이 평균에서 얼마나 퍼져 있나. 편차 제곱의 평균. PCA는 이것이 큰 방향을 고른다.'],
 ['계산','공분산 행렬','특징끼리 함께 커지고 작아지는 정도를 모두 모은 표. 64칸이면 64×64. 대각선은 각 칸의 분산이다.'],
 ['계산','고유벡터','행렬을 곱해도 방향이 바뀌지 않고 길이만 λ배 되는 벡터. 공분산 행렬의 고유벡터가 주성분이다.'],
 ['계산','고유값','고유벡터가 늘어나는 배수 λ. 공분산 행렬에서는 그 축에 비친 그림자의 분산과 같다.'],
 ['계산','직교','서로 직각. 직교하는 두 축의 점수는 상관이 0이라 같은 정보를 겹쳐 담지 않는다.'],
 ['계산','평균 중심화','각 특징에서 그 평균을 빼 데이터의 가운데를 원점으로 옮기는 일. PCA의 첫 단계.'],
 ['계산','표준화','각 특징에서 평균을 빼고 표준편차로 나누는 일. 단위가 다른 열을 같은 저울에 올린다. 같은 단위면 오히려 해로울 수 있다.'],
 ['계산','거듭제곱법','아무 벡터에 행렬을 거듭 곱하고 길이를 1로 맞추면 가장 큰 고유값의 고유벡터로 모이는 계산법. 04장 재생에 썼다.'],
 ['계산','특이값 분해','행렬을 회전 · 늘이기 · 회전의 곱으로 나누는 분해(SVD). 공분산 행렬을 만들지 않고 PCA를 계산하는 흔한 방법이다.'],
 ['계산','화이트닝','주성분 점수를 그 축의 표준편차로 나눠 모든 축의 분산을 1로 맞추는 일. whiten=True.'],
 ['평가','설명된 분산 비율','한 주성분의 분산 ÷ 전체 분산. 그 축이 전체 퍼짐의 몇 %를 담나. explained_variance_ratio_.'],
 ['평가','누적 설명된 분산','앞에서부터 k개 주성분의 설명된 분산 비율을 더한 값. 성분 수를 고를 때 본다(이 데이터 21성분에 90%).'],
 ['평가','엘보','설명된 분산(또는 오차) 그래프가 팔꿈치처럼 꺾이는 곳. 성분 수를 고르는 어림 규칙.'],
 ['평가','주성분 점수','점을 주성분 축에 비춘 위치. 2D 지도의 좌표다. transform이 돌려준다.'],
 ['평가','복원','주성분 점수 몇 개로 원래 64칸을 되살리는 일. 평균 + Σ(점수 × 성분). inverse_transform.'],
 ['평가','복원 오차','원본과 복원의 차이(제곱 평균). 버린 축들의 분산 합과 같다. 정답 없이 재는 점수다.'],
 ['평가','채점 봉투','정답(0~9)을 학습에서 가려 두었다가 채점할 때만 여는 것. 이 시리즈 비지도 본부의 규칙.'],
 ['평가','5-NN 정확도','지도에서 가장 가까운 다섯 장의 정답으로 다수결해 맞힌 비율. 차원축소팀의 채점 점수(5겹 교차 검증).'],
 ['평가','교차 검증','데이터를 몇 몫으로 나눠, 한 몫씩 돌아가며 시험으로 쓰고 나머지로 맞혀 점수를 평균하는 방법.'],
 ['평가','ARI','조정 랜드 지수. 군집 결과와 숨긴 정답이 얼마나 겹치나. 1이면 완벽, 무작위면 약 0. 군집팀의 채점 점수.'],
 ['평가','기준선','배우지 않고도 낼 수 있는 점수. 여기서는 가장 흔한 숫자만 답하기(10.2%).'],
 ['평가','합격선','지원자가 넘어야 할 점수. 차원축소팀은 매뉴얼 씨 2D 지도의 5-NN 정확도 18.1%.']];
var ANALOGY=[
 ['주성분 (축 고르기)','물체를 이리저리 돌려 벽에 비친 그림자가 가장 길게 늘어지는 방향 찾기',2,'헤브 학습 규칙을 따르는 뉴런 하나가 입력의 첫 주성분 방향으로 시냅스 무게를 맞춘다는 수학 모형(Oja, 1982)','모형이 그렇게 될 수 있다는 것이지, 실제 뇌가 PCA를 한다는 직접 증거는 약하다. 오야 규칙은 첫 축 하나만 찾고, 여러 축을 직각으로 찾으려면 장치를 더해야 한다'],
 ['차원 축소 (초파리)','64칸을 두 숫자로 요약하는 신입',1,'오히려 반대다. 초파리 버섯체는 냄새 수용체 쪽 약 50종의 신호를 케니언 세포 약 2,000개로 무작위로 넓게 퍼뜨린 뒤 일부만 켠다(Caron 등, 2013; Dasgupta 등, 2017)','초파리의 냄새 회로는 차원을 줄이지 않고 늘린다. 비슷한 냄새를 구별하기 쉽게 하려는 방식으로 해석된다. PCA와 짝지을 대응은 알려지지 않았다'],
 ['저차원 구조','오케스트라 단원 백 명이 몇 개의 악보 흐름을 함께 따라 움직인다',2,'수백 개 뉴런의 활동이 소수의 공통 패턴으로 함께 움직이는 경우가 많아, 연구자들이 PCA 같은 방법으로 그 흐름을 찾는다(Cunningham & Yu, 2014)','PCA를 쓰는 것은 연구자이지 뇌가 아니다. 뇌 활동의 저차원 구조가 곧은 축으로 잘 잡힌다는 보장도 없다'],
 ['성분 그림 (고유 숫자)','"기본 얼굴" 몇 장을 섞어 얼굴을 그리는 몽타주',1,'자연 이미지에서 희소 부호화로 배운 필터는 시각 피질 V1 단순 세포처럼 국소적이고 방향을 고르지만, PCA로 배운 성분은 그렇게 나오지 않는다(Olshausen & Field, 1996)','시각 피질의 필터는 PCA보다 희소 부호화에 가깝다는 것이 이 연구의 요지다'],
 ['복원과 버린 정보','압축한 사진 파일을 풀면 조금 흐려진 사진',1,'','PCA의 복원 오차는 버린 축의 분산 합으로 정확히 계산된다. 사람의 기억이 흐려지는 방식과 대응시킬 근거는 없다'],
 ['설명된 분산','요약문이 원문 내용의 몇 %를 담았나',1,'','"퍼짐"을 담은 비율이지 "중요한 내용"을 담은 비율이 아니다. 가장 퍼진 방향이 정답을 가르는 방향이라는 보장은 없다']];
var REFS=[
 'Pearson, K. (1901). On lines and planes of closest fit to systems of points in space. <i>The London, Edinburgh, and Dublin Philosophical Magazine and Journal of Science</i>, 2(11), 559–572. — PCA의 출발점, 01장 "가장 덜 잃는 방향"',
 'Hotelling, H. (1933). Analysis of a complex of statistical variables into principal components. <i>Journal of Educational Psychology</i>, 24(6), 417–441. — "주성분"이라는 이름, 04장 고유벡터',
 'Jolliffe, I. T. (2002). <i>Principal Component Analysis</i> (2nd ed.). Springer. — 성분 수 고르기, 06장',
 'Jolliffe, I. T., &amp; Cadima, J. (2016). Principal component analysis: A review and recent developments. <i>Philosophical Transactions of the Royal Society A</i>, 374(2065), 20150202. <a href="https://doi.org/10.1098/rsta.2015.0202" target="_blank" rel="noopener">doi:10.1098/rsta.2015.0202</a> — 01·04장',
 'Alpaydin, E., &amp; Kaynak, C. (1998). Optical Recognition of Handwritten Digits [Dataset]. UCI Machine Learning Repository. <a href="https://doi.org/10.24432/C50P49" target="_blank" rel="noopener">doi:10.24432/C50P49</a> (CC BY 4.0) — 02장 데이터(scikit-learn load_digits는 이 데이터의 검증 세트 1,797장)',
 'Turk, M., &amp; Pentland, A. (1991). Eigenfaces for recognition. <i>Journal of Cognitive Neuroscience</i>, 3(1), 71–86. — 04장 성분 그림(고유 얼굴)',
 'Halko, N., Martinsson, P.-G., &amp; Tropp, J. A. (2011). Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions. <i>SIAM Review</i>, 53(2), 217–288. — 06장 svd_solver="randomized"',
 'Minka, T. P. (2000). Automatic choice of dimensionality for PCA. In <i>Advances in Neural Information Processing Systems 13</i>. — 06장 n_components="mle"',
 'Schölkopf, B., Smola, A., &amp; Müller, K.-R. (1998). Nonlinear component analysis as a kernel eigenvalue problem. <i>Neural Computation</i>, 10(5), 1299–1319. — 06장 KernelPCA',
 'Hastie, T., Tibshirani, R., &amp; Friedman, J. (2009). <i>The Elements of Statistical Learning</i> (2nd ed.), 14.5절 Principal Components, Curves and Surfaces. Springer. — 01·04장',
 'van der Maaten, L., &amp; Hinton, G. (2008). Visualizing data using t-SNE. <i>Journal of Machine Learning Research</i>, 9, 2579–2605. — 06장 지도 비교, 5-15 예고',
 'Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. <i>Journal of Machine Learning Research</i>, 12, 2825–2830. — A장 코드',
 'scikit-learn 문서. <a href="https://scikit-learn.org/stable/modules/decomposition.html" target="_blank" rel="noopener">Decomposing signals in components (matrix factorization problems)</a> · <a href="https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html" target="_blank" rel="noopener">PCA</a> — n_components, svd_solver, whiten, 03·06장',
 'Oja, E. (1982). Simplified neuron model as a principal component analyzer. <i>Journal of Mathematical Biology</i>, 15(3), 267–273. — D장 헤브 학습과 주성분',
 'Caron, S. J. C., Ruta, V., Abbott, L. F., &amp; Axel, R. (2013). Random convergence of olfactory inputs in the <i>Drosophila</i> mushroom body. <i>Nature</i>, 497(7447), 113–117. — D장 초파리 버섯체',
 'Dasgupta, S., Stevens, C. F., &amp; Navlakha, S. (2017). A neural algorithm for a fundamental computing problem. <i>Science</i>, 358(6364), 793–796. — D장 초파리의 차원 늘리기',
 'Cunningham, J. P., &amp; Yu, B. M. (2014). Dimensionality reduction for large-scale neural recordings. <i>Nature Neuroscience</i>, 17(11), 1500–1509. — D장 신경 활동의 저차원 구조',
 'Olshausen, B. A., &amp; Field, D. J. (1996). Emergence of simple-cell receptive field properties by learning a sparse code for natural images. <i>Nature</i>, 381(6583), 607–609. — D장 성분 그림과 시각 피질'];
var TIP={who:'머신러닝을 처음 배우는 대학생·직장인·제조 현장 실무자. 5-1 KNN(거리, 표준화)을 먼저 보면 좋다. 선형대수는 "점과 방향의 내적 = 그림자의 위치" 정도만 알면 된다',
 goal:'① "가장 퍼진 방향으로 축을 돌린다"를 01장 그림자 놀이로 설명한다 ② 설명된 분산과 복원으로 "무엇을 얼마나 버렸나"를 숫자와 그림으로 말한다 ③ 성분을 8×8 그림으로 읽는다 ④ 2D 지도를 채점 봉투(5-NN)로 평가하고, 표준화·성분 수에 따라 결과가 어떻게 달라지는지, 겹침이 왜 생기는지 설명한다',
 time:'읽기 25분 · 실습(01·04·05장 그림) 25분 · 틀린 곳 찾기 10분 · Colab 15분(t-SNE 칸은 1분 안팎)',
 plans:[['강의형 50분','00장 지원서 1장을 인쇄해 2~4번 칸을 함께 읽기 8분 → 01장에서 학생들이 각도를 돌려 "그림자가 가장 넓을 때"를 맞히기 대결 10분 → 04장 성분 그림을 보고 "성분 1은 무엇을 보나" 토론 10분 → 05장 지도에서 봉투를 열기 전에 "몇 개 무리가 보이나" 세어 보고 연 뒤 비교 10분 → 면접 결과를 먼저 맞히고 발표, "표준화한 뒤 PCA" 다시 보기 12분.'],['실무 OJT 90분','03장 프롬프트로 각자 AI에게 코드를 받는다 20분 → 체크리스트로 서로의 코드를 바꿔 검사 20분 → Colab에서 실행하고 A장 결과와 비교 20분 → B장 ③ 프롬프트로 우리 공정 센서 데이터에 PCA 이상 감시를 걸어 보고, "표준화할까, 성분 몇 개를 남길까"를 공정 지식으로 토론 30분.'],['1:1 튜터링','01장에서 각도를 돌리며 "그림자 띠가 넓어지나 좁아지나"를 말로 설명하게 한다. 이어 05장 복원 그림에서 성분 수를 0 → 2 → 10 → 64로 올리며 "무엇이 돌아오나"를 묘사하게 하고, 마지막에 2D 지도에서 겹친 점 하나를 눌러 왜 이웃이 다른 숫자인지 복원 그림으로 설명하게 한다.']],
 myths:[['PCA 앞에는 늘 표준화한다','열의 단위가 다를 때 한다. 같은 단위(밝기)인 이 과제에서는 표준화하면 지도 60.3% → 51.4%로 나빠진다'],['가장 퍼진 방향이 가장 중요한 방향이다','PCA는 정답을 모른다. 퍼짐이 큰 방향이 숫자를 가르는 방향과 다를 수 있다. 5와 8을 가르는 정보는 뒤쪽 축에 있었다'],['2축 지도에서 겹치면 그 숫자들은 원래 비슷하다','64칸에서는 96%가 갈린다. 곧은 두 축의 그림자에 다 담지 못했을 뿐이다'],['성분 하나는 원래 특징 하나를 고른 것이다','성분은 64칸 모두에 무게를 나눠 준 새 방향이다'],['PCA는 정답 색칠 그림으로 평가한다','"잘 나뉘어 보인다"는 숫자가 아니다. 5-NN 같은 점수와 복원 오차를 숫자로 적는다'],['성분 부호가 바뀌면 결과가 틀린 것이다','부호(+/−)는 버전과 풀이 방법에 따라 뒤집힐 수 있고 뜻이 없다. 지도가 거울처럼 뒤집힐 뿐이다']],
 tasks:[['과제 1 · 손으로 PCA','점 네 개 (1,1), (2,2), (3,3), (4,2)의 평균을 빼고 2×2 공분산 행렬을 구한 뒤, 0°·45°·90° 방향의 그림자 분산을 계산해 가장 큰 방향을 찾는다. 01장 그림과 같은 순서다.'],['과제 2 · 틀린 곳 찾기 만들기','AI에게 일부러 실수가 섞인 PCA 코드를 받고(B장 ②를 거꾸로 활용), 짝에게 찾게 한다. 실수마다 "왜 틀렸나"를 한 줄로 쓴다. "오류 없이 도는 실수"를 하나 이상 넣는다.'],['과제 3 · 성분 수 지도','A장 7번 칸을 고쳐 성분 수 1~30마다 5-NN 정확도와 누적 설명된 분산을 한 그래프에 그린다. "분산 90%"와 "정확도가 거의 안 오르는 곳"이 같은지 비교한다.'],['과제 4 · 다른 데이터','scikit-learn의 load_wine(단위가 다른 13개 열)으로 같은 면접을 한다. 표준화를 한 경우와 안 한 경우의 2D 지도 점수를 비교하고, 이번에는 왜 결과가 반대인지 설명한다.']],
 ai:'학생이 AI에게 코드를 받는 것은 막지 말고 과정에 넣는다. PCA는 코드가 짧고 오류 없이 도는 실수(행·열 뒤집기, 비율과 분산 혼동, 평균 빠뜨리기, 훈련 점수 보고)가 많아 검증 연습에 특히 좋다. ① 프롬프트를 제출하게 하고 ② 체크리스트 아홉 항목을 직접 표시하게 하고 ③ 실행 결과를 합격선·기준선과 비교한 문장을 쓰게 한다. AI 답의 함수·인자는 설치된 버전의 공식 문서로 확인하게 하고, 과제에 AI 도움을 받았으면 사용 범위를 밝히게 한다.'};
var CODE={title:'PCA 면접 — 그림자 놀이부터 합격 판정까지',sub:'numpy, scikit-learn · 인터넷 필요 없음 · CPU 1분 안팎(t-SNE 칸이 대부분)',file:'pca_interview',
 desc:'이 편의 면접을 그대로 옮겼다. 손글씨 1,797장을 읽고 정답을 채점 봉투에 넣고(1번 칸), 매뉴얼 씨 지도로 합격선을 세우고(2번), PCA를 직접 짜서 scikit-learn과 맞춰 보고(3번), 설명된 분산(4번), 성분 그림(5번), 복원(6번), 2D 지도 채점(7번), 대표 10장으로 이름 붙이기(8번), GMM의 전처리(9번), 다음 지원자 t-SNE 미리 보기(10번), 판정(11번)까지 간다. 정답 y_sealed는 [채점] 표시 칸에서만 쓴다. 노트북으로 저장하면 칸마다 셀이 나뉜다.',
 cap:'실행 결과 (scikit-learn 1.9, CPU). 시간(초)은 컴퓨터마다 다르다. 브라우저 계산과 같은 숫자가 나온다: 설명된 분산 2성분 0.285 · 90%에 21성분, 2D 지도 5-NN 0.603, 표준화하면 0.514, 10성분 0.937. 8번 칸의 "대표 10장으로 이름 붙이기"는 K-Means 시작값의 난수에 따라 달라져 브라우저(05장)와 조금 다를 수 있다. 성분의 부호(+/−)는 버전에 따라 뒤집힐 수 있다.'};
