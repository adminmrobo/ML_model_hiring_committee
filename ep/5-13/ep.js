var BOARD_LINE='ARI 0.108'; /* 군집팀 합격선: 매뉴얼 씨 8묶음의 ARI */
/* ══ 5-13 GMM · 손글씨 실기 과제지 ══════════════════════════════════ */
/* GMM-ENGINE-START */
/* ── 가우시안 혼합 엔진 (이 편 전용, 브라우저·node 공용) ──
   scikit-learn GaussianMixture의 기본 동작을 그대로 따른다.
   · 시작: 묶음 번호(init_params="kmeans"의 K-Means 결과)를 0/1 소속으로 놓고 M 단계 한 번
   · E 단계: 묶음마다 log(비중) + log 정규분포 밀도 → 소속 확률(합 1)
   · M 단계: 소속 확률로 가중한 비중·평균·공분산, 공분산 대각선에 reg_covar(1e-6)를 더한다
   · 멈춤: 평균 로그 가능도의 변화 < tol(1e-3), 최대 max_iter(100)번
   · BIC = −2 × 로그 가능도 합 + 모수 개수 × ln(n). 작을수록 좋다
   공분산 모양: full(묶음마다 마음대로 기운 타원) · tied(모두 같은 타원) · diag(축에 나란한 타원) · spherical(동그라미) */
var GM=(function(){
  var LN2PI=Math.log(2*Math.PI);
  /* ── PCA: 공분산 행렬의 고윳값 분해(야코비 회전). scikit-learn PCA와 같은 축(부호는 다를 수 있다) ── */
  function pca(X,q){var n=X.length,d=X[0].length,mu=new Float64Array(d),i,j,k;
    for(i=0;i<n;i++)for(j=0;j<d;j++)mu[j]+=X[i][j];for(j=0;j<d;j++)mu[j]/=n;
    var A=[];for(j=0;j<d;j++)A.push(new Float64Array(d));
    var c=new Float64Array(d);for(i=0;i<n;i++){for(j=0;j<d;j++)c[j]=X[i][j]-mu[j];for(j=0;j<d;j++){var cj=c[j];if(cj===0)continue;var Aj=A[j];for(k=j;k<d;k++)Aj[k]+=cj*c[k];}}
    for(j=0;j<d;j++)for(k=j;k<d;k++){A[j][k]/=(n-1);A[k][j]=A[j][k];}
    var V=[];for(j=0;j<d;j++){V.push(new Float64Array(d));V[j][j]=1;}
    for(var sweep=0;sweep<60;sweep++){var off=0;for(j=0;j<d;j++)for(k=j+1;k<d;k++)off+=A[j][k]*A[j][k];if(off<1e-18)break;
      for(var p=0;p<d;p++)for(var r=p+1;r<d;r++){var apr=A[p][r];if(Math.abs(apr)<1e-14)continue;
        var th=(A[r][r]-A[p][p])/(2*apr),t=(th>=0?1:-1)/(Math.abs(th)+Math.sqrt(th*th+1)),cs=1/Math.sqrt(t*t+1),sn=t*cs;
        for(k=0;k<d;k++){var akp=A[k][p],akr=A[k][r];A[k][p]=cs*akp-sn*akr;A[k][r]=sn*akp+cs*akr;}
        for(k=0;k<d;k++){var apk=A[p][k],ark=A[r][k];A[p][k]=cs*apk-sn*ark;A[r][k]=sn*apk+cs*ark;}
        for(k=0;k<d;k++){var vkp=V[k][p],vkr=V[k][r];V[k][p]=cs*vkp-sn*vkr;V[k][r]=sn*vkp+cs*vkr;}}}
    var ev=[];for(j=0;j<d;j++)ev.push([A[j][j],j]);ev.sort(function(a,b){return b[0]-a[0]});
    var tot=ev.reduce(function(s,e){return s+Math.max(0,e[0])},0),comp=[],vr=[];
    for(k=0;k<q;k++){var col=ev[k][1],v=new Float64Array(d),big=0;for(j=0;j<d;j++){v[j]=V[j][col];if(Math.abs(v[j])>Math.abs(big))big=v[j];}if(big<0)for(j=0;j<d;j++)v[j]=-v[j];comp.push(v);vr.push(ev[k][0]/tot);}
    function tf(x){var z=new Float64Array(q);for(var k=0;k<q;k++){var s=0,ck=comp[k];for(var j=0;j<d;j++)s+=(x[j]-mu[j])*ck[j];z[k]=s;}return z;}
    function inv(z){var x=new Float64Array(d);for(var j=0;j<d;j++)x[j]=mu[j];for(var k=0;k<z.length;k++){var ck=comp[k];for(j=0;j<d;j++)x[j]+=z[k]*ck[j];}return x;}
    return {mean:mu,comp:comp,ratio:vr,var:ev.slice(0,q).map(function(e){return e[0]}),t:tf,inv:inv,Z:X.map(function(x){return tf(x)})};}
  /* ── 촐레스키 분해 (하삼각 L, C = L·Lᵀ) ── */
  function chol(C,d){var L=new Float64Array(d*d);for(var i=0;i<d;i++)for(var j=0;j<=i;j++){var s=C[i*d+j];for(var k=0;k<j;k++)s-=L[i*d+k]*L[j*d+k];
    if(i===j){if(s<=0)throw new Error('공분산이 양의 정부호가 아님');L[i*d+i]=Math.sqrt(s);}else L[i*d+j]=s/L[j*d+j];}return L;}
  function nParams(K,d,type){var cov=type==='full'?K*d*(d+1)/2:type==='diag'?K*d:type==='tied'?d*(d+1)/2:K;return cov+K*d+K-1;}
  /* ── M 단계: 소속 확률 R(n×K, 한 줄 Float64Array)로 비중·평균·공분산 ── */
  function mstep(Z,R,K,type,reg){var n=Z.length,d=Z[0].length,nk=new Float64Array(K),mu=[],i,j,k,a,b;
    for(k=0;k<K;k++)mu.push(new Float64Array(d));
    for(i=0;i<n;i++){var zi=Z[i];for(k=0;k<K;k++){var r=R[i*K+k];if(r===0)continue;nk[k]+=r;var mk=mu[k];for(j=0;j<d;j++)mk[j]+=r*zi[j];}}
    for(k=0;k<K;k++){nk[k]+=10*2.220446049250313e-16;for(j=0;j<d;j++)mu[k][j]/=nk[k];}
    var w=Array.from(nk,function(v){return v/n}),cov=[],D=new Float64Array(d);
    if(type==='full'||type==='tied'){var T=type==='tied'?new Float64Array(d*d):null;
      for(k=0;k<K;k++){var C=new Float64Array(d*d),mk2=mu[k];
        for(i=0;i<n;i++){var rr=R[i*K+k];if(rr<1e-300)continue;var zz=Z[i];for(a=0;a<d;a++)D[a]=zz[a]-mk2[a];for(a=0;a<d;a++){var ra=rr*D[a];if(ra===0)continue;for(b=a;b<d;b++)C[a*d+b]+=ra*D[b];}}
        if(T){for(a=0;a<d;a++)for(b=a;b<d;b++)T[a*d+b]+=C[a*d+b];}
        else{for(a=0;a<d;a++){for(b=a;b<d;b++){C[a*d+b]/=nk[k];C[b*d+a]=C[a*d+b];}C[a*d+a]+=reg;}cov.push(C);}}
      if(T){for(a=0;a<d;a++){for(b=a;b<d;b++){T[a*d+b]/=n;T[b*d+a]=T[a*d+b];}T[a*d+a]+=reg;}for(k=0;k<K;k++)cov.push(T);}}
    else{for(k=0;k<K;k++){var v=new Float64Array(d),mk3=mu[k];for(i=0;i<n;i++){var r3=R[i*K+k];if(r3<1e-300)continue;var z3=Z[i];for(a=0;a<d;a++){var df=z3[a]-mk3[a];v[a]+=r3*df*df;}}
        for(a=0;a<d;a++)v[a]=v[a]/nk[k]+reg;
        if(type==='spherical'){var s=0;for(a=0;a<d;a++)s+=v[a];s/=d;for(a=0;a<d;a++)v[a]=s;}cov.push(v);}}
    return {w:w,mu:mu,cov:cov,nk:nk};}
  /* ── 묶음마다 log(비중 × 밀도) ── */
  function logProb(Z,P,K,type){var n=Z.length,d=Z[0].length,out=new Float64Array(n*K),D=new Float64Array(d),y=new Float64Array(d);
    for(var k=0;k<K;k++){var mk=P.mu[k],lw=Math.log(P.w[k]);
      if(type==='full'||type==='tied'){var L=chol(P.cov[k],d),ld=0;for(var a=0;a<d;a++)ld+=Math.log(L[a*d+a]);
        for(var i=0;i<n;i++){var zi=Z[i],q=0;for(a=0;a<d;a++)D[a]=zi[a]-mk[a];for(a=0;a<d;a++){var s=D[a];for(var b=0;b<a;b++)s-=L[a*d+b]*y[b];y[a]=s/L[a*d+a];q+=y[a]*y[a];}
          out[i*K+k]=lw-0.5*(d*LN2PI+q)-ld;}}
      else{var v=P.cov[k],ld2=0;for(a=0;a<d;a++)ld2+=0.5*Math.log(v[a]);
        for(i=0;i<n;i++){var z2=Z[i],q2=0;for(a=0;a<d;a++){var e=z2[a]-mk[a];q2+=e*e/v[a];}out[i*K+k]=lw-0.5*(d*LN2PI+q2)-ld2;}}}
    return out;}
  /* E 단계: 소속 확률과 평균 로그 가능도 */
  function estep(Z,P,K,type){var n=Z.length,lp=logProb(Z,P,K,type),R=new Float64Array(n*K),ll=0;
    for(var i=0;i<n;i++){var m=-Infinity,k;for(k=0;k<K;k++)if(lp[i*K+k]>m)m=lp[i*K+k];var s=0;for(k=0;k<K;k++)s+=Math.exp(lp[i*K+k]-m);var lse=m+Math.log(s);ll+=lse;for(k=0;k<K;k++)R[i*K+k]=Math.exp(lp[i*K+k]-lse);}
    return {R:R,ll:ll/n};}
  function oneHot(lab,K){var R=new Float64Array(lab.length*K);for(var i=0;i<lab.length;i++)R[i*K+lab[i]]=1;return R;}
  /* fit(Z, 시작 묶음 번호, {K, type, reg, tol, maxIter, onIter(t, P, R, ll)}) */
  function fit(Z,lab,o){o=o||{};var K=o.K||10,type=o.type||'full',reg=o.reg==null?1e-6:o.reg,tol=o.tol==null?1e-3:o.tol,MX=o.maxIter==null?100:o.maxIter,n=Z.length,d=Z[0].length;
    var P=mstep(Z,oneHot(lab,K),K,type,reg),prev=-Infinity,ll=-Infinity,it=0,conv=false,hist=[],R=null;
    for(it=1;it<=MX;it++){var E=estep(Z,P,K,type);ll=E.ll;R=E.R;P=mstep(Z,R,K,type,reg);hist.push(ll);if(o.onIter)o.onIter(it,P,R,ll);if(Math.abs(ll-prev)<tol){conv=true;break;}prev=ll;}
    var F=estep(Z,P,K,type),lab2=new Int32Array(n);for(var i=0;i<n;i++){var b=0;for(var k=1;k<K;k++)if(F.R[i*K+k]>F.R[i*K+b])b=k;lab2[i]=b;}
    var np=nParams(K,d,type);
    return {K:K,type:type,P:P,R:F.R,ll:F.ll,labels:lab2,iters:Math.min(it,MX),conv:conv,hist:hist,nParams:np,bic:-2*F.ll*n+np*Math.log(n),aic:-2*F.ll*n+2*np};}
  /* fitAsync: 같은 계산을 반복마다 한 번씩 쉬어 가며 한다(화면이 멈추지 않게) */
  async function fitAsync(Z,lab,o){o=o||{};var K=o.K||10,type=o.type||'full',reg=o.reg==null?1e-6:o.reg,tol=o.tol==null?1e-3:o.tol,MX=o.maxIter==null?100:o.maxIter,n=Z.length,d=Z[0].length;
    var P=mstep(Z,oneHot(lab,K),K,type,reg),prev=-Infinity,ll=-Infinity,it=0,conv=false,hist=[],R=null;if(o.onIter)o.onIter(0,P,null,null);
    for(it=1;it<=MX;it++){await new Promise(function(r){setTimeout(r,0)});var E=estep(Z,P,K,type);ll=E.ll;R=E.R;P=mstep(Z,R,K,type,reg);hist.push(ll);if(o.onIter)o.onIter(it,P,R,ll);if(Math.abs(ll-prev)<tol){conv=true;break;}prev=ll;}
    var F=estep(Z,P,K,type),lab2=new Int32Array(n);for(var i=0;i<n;i++){var b=0;for(var k=1;k<K;k++)if(F.R[i*K+k]>F.R[i*K+b])b=k;lab2[i]=b;}
    var np=nParams(K,d,type);
    return {K:K,type:type,P:P,R:F.R,ll:F.ll,labels:lab2,iters:Math.min(it,MX),conv:conv,hist:hist,nParams:np,bic:-2*F.ll*n+np*Math.log(n),aic:-2*F.ll*n+2*np};}
  /* ── K-Means (시작값 만들기용): k-means++ 시작, 로이드 반복 ── */
  function kmeans(Z,K,seed,iters){var rnd=RNG(seed||1),n=Z.length,d=Z[0].length,C=[],dist=new Float64Array(n).fill(Infinity),i,k,j;
    function d2(a,b){var s=0;for(var j=0;j<d;j++){var e=a[j]-b[j];s+=e*e;}return s;}
    C.push(Float64Array.from(Z[Math.floor(rnd()*n)]));
    while(C.length<K){var S=0;for(i=0;i<n;i++){dist[i]=Math.min(dist[i],d2(Z[i],C[C.length-1]));S+=dist[i];}var u=rnd()*S,acc=0;for(i=0;i<n;i++){acc+=dist[i];if(acc>=u)break;}C.push(Float64Array.from(Z[Math.min(i,n-1)]));}
    var lab=new Int32Array(n);
    for(var t=0;t<(iters||100);t++){var ch=0;for(i=0;i<n;i++){var b=0,bd=Infinity;for(k=0;k<K;k++){var v=d2(Z[i],C[k]);if(v<bd){bd=v;b=k;}}if(lab[i]!==b||t===0){ch++;lab[i]=b;}}
      var S2=[],N=new Float64Array(K);for(k=0;k<K;k++)S2.push(new Float64Array(d));for(i=0;i<n;i++){N[lab[i]]++;for(j=0;j<d;j++)S2[lab[i]][j]+=Z[i][j];}
      for(k=0;k<K;k++)if(N[k])for(j=0;j<d;j++)C[k][j]=S2[k][j]/N[k];if(!ch&&t>0)break;}
    return lab;}
  /* ── 채점 도구: ARI · NMI (scikit-learn과 같은 정의, NMI는 산술 평균) ── */
  function table(a,b){var A={},B={},T={},n=a.length;for(var i=0;i<n;i++){A[a[i]]=(A[a[i]]||0)+1;B[b[i]]=(B[b[i]]||0)+1;var key=a[i]+'|'+b[i];T[key]=(T[key]||0)+1;}return {A:A,B:B,T:T,n:n};}
  function c2(x){return x*(x-1)/2;}
  function ari(a,b){var t=table(a,b),s=0,sa=0,sb=0,k;for(k in t.T)s+=c2(t.T[k]);for(k in t.A)sa+=c2(t.A[k]);for(k in t.B)sb+=c2(t.B[k]);var ex=sa*sb/c2(t.n),mx=(sa+sb)/2;return mx===ex?1:(s-ex)/(mx-ex);}
  function nmi(a,b){var t=table(a,b),n=t.n,ha=0,hb=0,mi=0,k;for(k in t.A){var p=t.A[k]/n;ha-=p*Math.log(p);}for(k in t.B){var q=t.B[k]/n;hb-=q*Math.log(q);}
    for(k in t.T){var ab=k.split('|'),pij=t.T[k]/n;mi+=pij*Math.log(pij/((t.A[ab[0]]/n)*(t.B[ab[1]]/n)));}return (ha+hb)?mi/((ha+hb)/2):1;}
  /* 실루엣: 원래 64칸 픽셀의 유클리드 거리로 (정답 없이 재는 점수) */
  async function silhouette(X,lab){var n=X.length,d=X[0].length,K=0,i,j,k;for(i=0;i<n;i++)K=Math.max(K,lab[i]+1);var cnt=new Float64Array(K);for(i=0;i<n;i++)cnt[lab[i]]++;
    X=X.map(function(r){return Float64Array.from(r)});
    var S=new Float64Array(n*K);for(i=0;i<n;i++){if(i%150===149)await new Promise(function(r){setTimeout(r,0)});var xi=X[i];for(j=i+1;j<n;j++){var xj=X[j],s=0;for(k=0;k<d;k++){var e=xi[k]-xj[k];s+=e*e;}s=Math.sqrt(s);S[i*K+lab[j]]+=s;S[j*K+lab[i]]+=s;}}
    var tot=0,per=new Float64Array(n);for(i=0;i<n;i++){var own=lab[i];if(cnt[own]<=1){per[i]=0;continue;}var a=S[i*K+own]/(cnt[own]-1),b=Infinity;for(k=0;k<K;k++)if(k!==own&&cnt[k])b=Math.min(b,S[i*K+k]/cnt[k]);per[i]=(b-a)/Math.max(a,b);tot+=per[i];}
    return {mean:tot/n,per:per};}
  return {pca:pca,chol:chol,mstep:mstep,estep:estep,logProb:logProb,fit:fit,fitAsync:fitAsync,kmeans:kmeans,ari:ari,nmi:nmi,silhouette:silhouette,nParams:nParams,oneHot:oneHot};})();
/* GMM-ENGINE-END */

/* ══ 손글씨 실기 과제지 ══ */
const DIG=digitRows(),N=DIG.length,X=DIG.map(r=>r.px);
const ENVELOPE=DIG.map(r=>r.y); /* 채점 봉투(숨긴 정답). 학습에는 한 번도 쓰지 않고 05장 채점과 면접 결과에서만 연다 */
const MAN=X.map(manualDigit),LINE=GM.ari(ENVELOPE,MAN),LINE_NMI=GM.nmi(ENVELOPE,MAN);
const RAND=(function(){const r=RNG(7);return GM.ari(ENVELOPE,X.map(()=>Math.floor(r()*10)));})();
const INK10=(function(){const ink=X.map(p=>p.reduce((a,b)=>a+b,0)),lo=Math.min(...ink),hi=Math.max(...ink);return GM.ari(ENVELOPE,ink.map(v=>Math.min(9,Math.floor((v-lo)/(hi-lo)*10))));})();
const f3=v=>fx(v,3),KC=10;
const lab=s=>Array.from(s,Number);
function hex2rgb(h){return [parseInt(h.slice(1,3),16),parseInt(h.slice(3,5),16),parseInt(h.slice(5,7),16)];}
const DRGB=DIGCOL.map(hex2rgb);
function axes(ctx,C,pl,pt,gw,gh){ctx.strokeStyle=C.rule;ctx.lineWidth=1;ctx.strokeRect(pl,pt,gw,gh);ctx.fillStyle=C.muted;ctx.font=font(12);}
function vlabel(ctx,txt,x,y){ctx.save();ctx.translate(x,y);ctx.rotate(-Math.PI/2);ctx.textAlign='center';ctx.fillText(txt,0,0);ctx.restore();}
function barRows(rows,max){return rows.map(r=>'<div class="row"><span style="text-align:left;font-family:var(--font)">'+r[0]+'</span><div class="tr"><i style="left:0;width:'+Math.max(0,Math.min(100,r[1]/max*100))+'%;background:'+(r[2]||'var(--blue)')+'"></i></div><span>'+r[3]+'</span></div>').join('');}
/* 종이 위 글씨: 어두운 화면에서도 흰 카드 위에 검은 잉크로 그린다 */
function paper(ctx,x,y,s,C){ctx.fillStyle='#fffdf8';ctx.fillRect(x,y,s,s);ctx.strokeStyle=rgba(C.muted,.35);ctx.lineWidth=1;ctx.strokeRect(x+.5,y+.5,s-1,s-1);}
function clip16(v){return Array.from(v,x=>Math.max(0,Math.min(16,x)));}
function whenNear(id,fn,margin){const t=$(id);let done=false;function go(){if(done)return;done=true;setTimeout(fn,10);}
  if('IntersectionObserver' in window){const io=new IntersectionObserver(es=>{if(es.some(e=>e.isIntersecting)){go();io.disconnect();}},{rootMargin:(margin||600)+'px'});io.observe(t);}else go();
  window.addEventListener('hashchange',()=>{const h=location.hash.slice(1),el=h&&$(h);if(el&&(el===t||t.contains(el)||t.compareDocumentPosition(el)&Node.DOCUMENT_POSITION_FOLLOWING))go();});
  if(location.hash){const el=$(location.hash.slice(1));if(el&&(el===t||t.contains(el)||t.compareDocumentPosition(el)&Node.DOCUMENT_POSITION_FOLLOWING))go();}}

/* ── PCA 30축 (브라우저 계산) ── */
let P30=null;function ensureP(){if(!P30){const t0=performance.now();P30=GM.pca(X,30);P30.ms=performance.now()-t0;P30.Z2=P30.Z.map(z=>[z[0],z[1]]);}return P30;}
function meanDigit(mu){return clip16(ensureP().inv(mu));}

/* ── 본 모델: PCA 30축 · 10묶음 · full 공분산 · 시작점 = scikit-learn random_state=0과 같은 K-Means 묶음 ── */
let MAINP=null;
function ensureMain(){if(MAINP)return MAINP;MAINP=(async()=>{const P=ensureP(),snaps=[],t0=performance.now();
    const M=await GM.fitAsync(P.Z,lab(PRE.init.i30_10),{K:KC,type:'full',onIter:(it,Pp,R,ll)=>{snaps.push({it,ll,w:Pp.w.slice(),mu:Pp.mu.map(m=>meanDigit(m))});}});
    M.ms=performance.now()-t0;M.snaps=snaps;
    /* 묶음마다 가장 전형적인 한 장 = 그 종 모양의 밀도가 가장 높은 글씨 */
    const lp=GM.logProb(P.Z,M.P,KC,'full');M.rep=[];for(let k=0;k<KC;k++){let b=-1,bv=-Infinity;for(let i=0;i<N;i++)if(M.labels[i]===k){const v=lp[i*KC+k]-Math.log(M.P.w[k]);if(v>bv){bv=v;b=i;}}M.rep.push(b);}
    /* 글씨마다 로그 가능도(어느 종에든 얼마나 잘 맞나) */
    M.score=new Float64Array(N);for(let i=0;i<N;i++){let m=-Infinity;for(let k=0;k<KC;k++)m=Math.max(m,lp[i*KC+k]);let s=0;for(let k=0;k<KC;k++)s+=Math.exp(lp[i*KC+k]-m);M.score[i]=m+Math.log(s);}
    M.maxp=new Float64Array(N);for(let i=0;i<N;i++){let m=0;for(let k=0;k<KC;k++)m=Math.max(m,M.R[i*KC+k]);M.maxp[i]=m;}
    M.size=new Array(KC).fill(0);M.labels.forEach(l=>M.size[l]++);
    return M;})();return MAINP;}
/* 채점 봉투를 연 뒤에만 쓰는 값 */
function grade(M){const ari=GM.ari(ENVELOPE,M.labels),nmi=GM.nmi(ENVELOPE,M.labels),name=M.rep?M.rep.map(i=>ENVELOPE[i]):null;let ok=0;if(name)for(let i=0;i<N;i++)if(name[M.labels[i]]===ENVELOPE[i])ok++;return {ari,nmi,name,repAcc:name?ok/N:null};}

/* ── 히어로: 겹친 종 모양 실험실 ── */
heroRoom($('heroCv'),{sym:'N',no:'APP-13',name:'GMM\n(혼합 모델)',role:'지원자 · 군집팀',banner:'모델 채용 위원회 · 5-13 면접',doc:'front',asst:'board',fly:'clip',robo:'calc',loop:11,
  beats:[{at:0.4,who:'doc',mood:'scale',move:'tilt',say:'종이 몇 개일까요?'},{at:2.4,who:'asst',mood:'ledger',move:'nod',say:'확률로 나눠 적어요'},{at:4.4,who:'fly',mood:'point',move:'shake',say:'반반인 글씨는?'},
         {at:6.4,who:'robo',mood:'count',move:'bob',say:'공분산 숫자 세 봐'},{at:8.4,who:'doc',mood:'judge',move:'nod',say:'BIC로 저울질해요'}],
  motif:function(ctx,t,W,H,C){const x0=300,x1=432,y0=196,hh=92,mus=[0.22,0.5,0.78],sds=[0.07,0.09,0.06],ph=REDUCE?0:t*0.9,w=[0.32+0.12*Math.sin(ph),0.36+0.1*Math.sin(ph+2.1),0.32+0.12*Math.sin(ph+4.2)],ws=w[0]+w[1]+w[2];
    const g=(x,m,s)=>Math.exp(-0.5*((x-m)/s)**2)/s,col=[C.c_app,C.stamp,C.ok];
    ctx.strokeStyle=C.muted;ctx.lineWidth=1.5;ctx.beginPath();ctx.moveTo(x0,y0);ctx.lineTo(x1,y0);ctx.stroke();
    for(let k=0;k<3;k++){ctx.strokeStyle=col[k];ctx.lineWidth=2;ctx.beginPath();for(let s=0;s<=60;s++){const u=s/60,v=w[k]/ws*g(u,mus[k],sds[k])*0.055;const X=x0+u*(x1-x0),Y=y0-v*hh;s?ctx.lineTo(X,Y):ctx.moveTo(X,Y);}ctx.stroke();}
    ctx.strokeStyle=C.ink;ctx.lineWidth=2.6;ctx.setLineDash([5,3]);ctx.beginPath();for(let s=0;s<=60;s++){const u=s/60;let v=0;for(let k=0;k<3;k++)v+=w[k]/ws*g(u,mus[k],sds[k])*0.055;const X=x0+u*(x1-x0),Y=y0-v*hh;s?ctx.lineTo(X,Y):ctx.moveTo(X,Y);}ctx.stroke();ctx.setLineDash([]);
    ctx.fillStyle=C.muted;ctx.font=font(12,500,1);ctx.textAlign='left';ctx.fillText('종 3개 섞기 · 점선 = 합',x0-4,y0+16);}});

/* ── 01 장면 ── */
comic('comicIntro',[
 {bg:'lab',prop:'종 하나 · 평균 하나',cast:[{who:'doc',mood:'scale',x:30,move:'tilt',h:104},{who:'fly',mood:'point',x:74,move:'shake',flip:true}],say:{who:'fly',txt:'1,797장을 종 하나로 덮으면 0도 1도 다 한 동네잖아.'},cap:'① 하나의 종 — 평균 하나, 퍼짐 하나로는 숫자 열 종류를 못 나눈다'},
 {bg:'lab',prop:'종 10개 · 비중 · 퍼짐',cast:[{who:'asst',mood:'ledger',x:30,move:'nod'},{who:'robo',mood:'calc',x:72,move:'bob',h:104}],say:{who:'asst',txt:'종 열 개를 섞었어요. 종마다 자리(평균)와 모양(공분산), 몫(비중)이 달라요.'},cap:'② 여러 종 섞기 — 종마다 동네가 생긴다'},
 {bg:'lab',prop:'1일 확률 0.6 · 7일 확률 0.4',cast:[{who:'doc',mood:'judge',x:28,move:'nod',h:104},{who:'asst',mood:'spark',x:72,move:'jump'}],say:{who:'doc',txt:'애매한 글씨는 한 묶음에 딱 넣지 않고, 확률로 나눠 갖게 해요.'},cap:'③ 확률로 나눠 갖기 — K-Means와 다른 점'}]);

/* ── 01 ① 하나의 종 vs 여러 종 (PC1 한 축, 브라우저 1차원 EM) ── */
(function(){const cv=$('oneCv'),H=300;let K=1,fitR=null,xs=null,lo=0,hi=1,pos=0.5,pend=false;
  function prep(){if(xs)return;const P=ensureP();xs=P.Z.map(z=>z[0]);lo=Math.min(...xs)-2;hi=Math.max(...xs)+2;}
  function em1(K){const n=xs.length,sorted=[...xs].sort((a,b)=>a-b),mu=[],sd=[],w=[];let m0=ML.mean(xs),v0=ML.mean(xs.map(x=>(x-m0)**2));
    for(let k=0;k<K;k++){mu.push(sorted[Math.floor((k+0.5)/K*n)]);sd.push(Math.sqrt(v0)/K);w.push(1/K);}
    let ll=0,prev=-Infinity;const R=new Float64Array(n*K);
    for(let it=0;it<300;it++){ll=0;for(let i=0;i<n;i++){let m=-Infinity;const lp=[];for(let k=0;k<K;k++){const z=(xs[i]-mu[k])/sd[k];lp.push(Math.log(w[k])-0.5*z*z-Math.log(sd[k])-0.5*Math.log(2*Math.PI));m=Math.max(m,lp[k]);}let s=0;lp.forEach(v=>s+=Math.exp(v-m));const lse=m+Math.log(s);ll+=lse;for(let k=0;k<K;k++)R[i*K+k]=Math.exp(lp[k]-lse);}
      for(let k=0;k<K;k++){let nk=0,s1=0;for(let i=0;i<n;i++){nk+=R[i*K+k];s1+=R[i*K+k]*xs[i];}mu[k]=s1/nk;let s2=0;for(let i=0;i<n;i++)s2+=R[i*K+k]*(xs[i]-mu[k])**2;sd[k]=Math.sqrt(s2/nk+1e-6);w[k]=nk/n;}
      if(Math.abs(ll-prev)<1e-6*n)break;prev=ll;}
    const np=3*K-1;return {K,mu,sd,w,ll:ll/n,bic:-2*ll+np*Math.log(n)};}
  function dens(F,x,k){const z=(x-F.mu[k])/F.sd[k];return F.w[k]*Math.exp(-0.5*z*z)/(F.sd[k]*Math.sqrt(2*Math.PI));}
  function draw(){prep();const W=fit(cv,H),ctx=cv.getContext('2d'),C=cols(),pl=44,pr=12,pt=12,pb=70,gw=W-pl-pr,gh=H-pt-pb,NB=48,bw=(hi-lo)/NB,b=new Array(NB).fill(0);xs.forEach(x=>b[clamp(Math.floor((x-lo)/bw),0,NB-1)]++);
    const ymax=Math.max(...b)*1.15,sx=x=>pl+(x-lo)/(hi-lo)*gw,sy=c=>pt+gh-c/ymax*gh;ctx.clearRect(0,0,W,H);axes(ctx,C,pl,pt,gw,gh);
    b.forEach((c,k)=>{ctx.fillStyle=rgba(C.muted,.35);ctx.fillRect(sx(lo+k*bw)+1,sy(c),gw/NB-2,pt+gh-sy(c));});
    [0,Math.round(ymax/2)].forEach(v=>{ctx.textAlign='right';ctx.fillStyle=C.muted;ctx.fillText(v,pl-5,sy(v)+4);});vlabel(ctx,'글씨 수',11,pt+gh/2);
    ctx.textAlign='center';ctx.fillText('첫째 축 PC1 (64칸을 가장 많이 퍼진 방향 하나로 본 값)',pl+gw/2,pt+gh+16);
    const F=fitR;if(F){const scale=N*bw;for(let k=0;k<F.K;k++){ctx.strokeStyle=DIGCOL[k];ctx.lineWidth=2;ctx.beginPath();for(let s=0;s<=200;s++){const x=lo+(hi-lo)*s/200,Y=sy(dens(F,x,k)*scale);s?ctx.lineTo(sx(x),Y):ctx.moveTo(sx(x),Y);}ctx.stroke();}
      ctx.strokeStyle=C.text;ctx.lineWidth=2.6;ctx.setLineDash(F.K>1?[6,3]:[]);ctx.beginPath();for(let s=0;s<=200;s++){const x=lo+(hi-lo)*s/200;let v=0;for(let k=0;k<F.K;k++)v+=dens(F,x,k);const Y=sy(v*scale);s?ctx.lineTo(sx(x),Y):ctx.moveTo(sx(x),Y);}ctx.stroke();ctx.setLineDash([]);
      const xp=lo+(hi-lo)*pos;ctx.strokeStyle=C.stamp;ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(sx(xp),pt);ctx.lineTo(sx(xp),pt+gh);ctx.stroke();
      /* 그 자리의 글씨 몇 장 */
      const near=xs.map((x,i)=>[Math.abs(x-xp),i]).sort((a,b)=>a[0]-b[0]).slice(0,Math.max(4,Math.min(10,Math.floor(gw/34))));const s=28;
      near.forEach((q,j)=>{const x=pl+j*(s+6),y=H-s-8;paper(ctx,x,y,s,C);drawDigit(ctx,X[q[1]],x+2,y+2,s-4,'#111827');});
      ctx.fillStyle=C.muted;ctx.textAlign='left';ctx.font=font(11.5);ctx.fillText('빨간 선 자리의 글씨',pl+near.length*(s+6)+2,H-18);}
    info();}
  function info(){const F=fitR;if(!F){$('oneInfo').textContent='계산 중…';return;}const xp=lo+(hi-lo)*pos;let tot=0;const d=[];for(let k=0;k<F.K;k++){d.push(dens(F,xp,k));tot+=d[k];}
    const pr=d.map((v,k)=>'<span style="color:'+DIGCOL[k]+';font-weight:700">종 '+(k+1)+' '+(v/tot).toFixed(2)+'</span>').join(' · ');
    $('oneInfo').innerHTML='종 <b>'+F.K+'개</b> · 평균 로그 가능도 <b>'+F.ll.toFixed(3)+'</b> · BIC <b>'+Math.round(F.bic).toLocaleString()+'</b>(작을수록 좋다). 빨간 선 자리 글씨의 소속 확률: '+pr+'. '+
      (F.K===1?'종 하나는 봉우리가 여럿인 막대그래프를 둥근 언덕 하나로 덮는다. 봉우리 사이의 빈 골짜기에도 글씨가 있다고 믿는다.':F.K===2?'종 둘이 겹치는 곳에서는 소속 확률이 0.5 근처로 나뉜다. 빨간 선을 골짜기로 옮겨 보자.':'종을 늘릴수록 막대그래프를 더 잘 덮고 BIC가 내려간다. 다만 이것은 한 축만 본 그림이다. 숫자 열 종류는 이 한 축에서 서로 많이 겹친다.');}
  async function refit(){if(pend)return;pend=true;$('oneInfo').textContent='계산 중…';await wait(10);prep();fitR=em1(K);pend=false;draw();}
  seg('#oneK button',v=>{K=+v;refit();});$('onePos').addEventListener('input',e=>{pos=+e.target.value/100;draw();});
  whenNear('m1',refit,400);onRedraw(()=>{if(fitR)draw();});})();

/* ── 01 ② 2차원 EM 재생 (PC1·PC2, 브라우저 계산) ── */
const EM2=(function(){const cv=$('emCv'),HH=()=>Math.round(Math.min(420,Math.max(280,(cv.clientWidth||600)*0.8))),st={K:10,seed:1,P:null,R:null,it:0,ll:null,busy:false,pick:-1,hist:[]};let Z2=null,bx=null;
  function init(){Z2=ensureP().Z2;const xs=Z2.map(z=>z[0]),ys=Z2.map(z=>z[1]);bx=[Math.min(...xs)-3,Math.max(...xs)+3,Math.min(...ys)-3,Math.max(...ys)+3];
    const l=GM.kmeans(Z2,st.K,st.seed,3);st.P=GM.mstep(Z2,GM.oneHot(l,st.K),st.K,'full',1e-6);st.R=GM.oneHot(l,st.K);st.it=0;st.ll=null;st.hist=[];}
  function step(){const E=GM.estep(Z2,st.P,st.K,'full');st.R=E.R;st.ll=E.ll;st.P=GM.mstep(Z2,E.R,st.K,'full',1e-6);st.it++;st.hist.push(E.ll);}
  function map(W){const H=HH(),pl=10,pt=10,gw=W-20,gh=H-20,s=Math.min(gw/(bx[1]-bx[0]),gh/(bx[3]-bx[2])),ox=pl+(gw-s*(bx[1]-bx[0]))/2,oy=pt+(gh-s*(bx[3]-bx[2]))/2;
    return {sx:x=>ox+(x-bx[0])*s,sy:y=>oy+(bx[3]-y)*s,s,ox,oy};}
  function draw(){if(!st.P)return;const H=HH(),W=fit(cv,H),ctx=cv.getContext('2d'),C=cols(),m=map(W),K=st.K;ctx.clearRect(0,0,W,H);ctx.strokeStyle=C.rule;ctx.strokeRect(.5,.5,W-1,H-1);
    for(let i=0;i<N;i++){let r=0,g=0,b=0,mx=0;for(let k=0;k<K;k++){const p=st.R[i*K+k];r+=p*DRGB[k][0];g+=p*DRGB[k][1];b+=p*DRGB[k][2];mx=Math.max(mx,p);}
      ctx.fillStyle='rgba('+Math.round(r)+','+Math.round(g)+','+Math.round(b)+','+(0.35+0.5*mx)+')';ctx.beginPath();ctx.arc(m.sx(Z2[i][0]),m.sy(Z2[i][1]),mx<0.9?3.2:2.4,0,7);ctx.fill();}
    for(let k=0;k<K;k++){const c=st.P.cov[k],a=c[0],bb=c[1],d=c[3],tr=(a+d)/2,df=Math.sqrt(((a-d)/2)**2+bb*bb),l1=tr+df,l2=Math.max(tr-df,1e-9),th=0.5*Math.atan2(2*bb,a-d),mu=st.P.mu[k];
      ctx.strokeStyle=DIGCOL[k];ctx.lineWidth=2.2;ctx.beginPath();for(let s=0;s<=64;s++){const u=s/64*2*Math.PI,e1=2*Math.sqrt(l1)*Math.cos(u),e2=2*Math.sqrt(l2)*Math.sin(u),x=mu[0]+e1*Math.cos(th)-e2*Math.sin(th),y=mu[1]+e1*Math.sin(th)+e2*Math.cos(th);s?ctx.lineTo(m.sx(x),m.sy(y)):ctx.moveTo(m.sx(x),m.sy(y));}ctx.stroke();
      circ(ctx,m.sx(mu[0]),m.sy(mu[1]),4+12*st.P.w[k],DIGCOL[k],'#fff',1.5);}
    if(st.pick>=0){const z=Z2[st.pick];circ(ctx,m.sx(z[0]),m.sy(z[1]),7,null,C.text,2.5);}
    ctx.fillStyle=C.text;ctx.font=font(13,700,1);ctx.textAlign='left';ctx.fillText('EM '+st.it+'번'+(st.ll!=null?' · 평균 로그 가능도 '+st.ll.toFixed(3):' · 시작점(K-Means)'),14,28);ctx.font=font(11.5);ctx.fillStyle=C.muted;ctx.fillText('타원 = 종 모양(평균 ± 2 표준편차) · 동그라미 크기 = 비중',14,46);
    info();}
  function info(){let amb=0;for(let i=0;i<N;i++){let mx=0;for(let k=0;k<st.K;k++)mx=Math.max(mx,st.R[i*st.K+k]);if(mx<0.9)amb++;}
    $('emInfo').innerHTML='종 <b>'+st.K+'개</b> · EM <b>'+st.it+'번</b> · 가장 큰 소속 확률이 0.9 미만인 "애매한 글씨" <b>'+amb.toLocaleString()+'장</b>(큰 점, 색이 섞임). '+
      (st.it===0?'지금은 K-Means 묶음 그대로다. 모든 글씨가 한 묶음에 확률 1로 들어 있다. "E·M 한 번"을 눌러 보자.':st.it<5?'E 단계가 소속 확률을 다시 나눴다. 종이 겹치는 곳의 점은 색이 섞인다. M 단계는 그 확률로 가중해 평균과 타원을 옮긴다.':'타원이 거의 움직이지 않으면 가능도도 거의 오르지 않는다. 여기서 멈춘다(scikit-learn은 변화가 0.001보다 작으면 멈춘다).')+
      ' 이 그림은 64칸을 두 축으로만 본 것이라 숫자들이 많이 겹친다. 실제 면접은 30축에서 한다(04장).';}
  function pickInfo(){const i=st.pick;if(i<0){$('emPick').innerHTML='<p class="muted" style="margin:0">그림의 점을 눌러 그 글씨의 소속 확률을 보세요.</p>';return;}
    const pr=[];for(let k=0;k<st.K;k++)pr.push([st.R[i*st.K+k],k]);pr.sort((a,b)=>b[0]-a[0]);
    $('emPick').innerHTML='<div style="display:flex;gap:12px;align-items:center"><canvas id="emPickCv" width="64" height="64" style="width:64px;height:64px;background:#fffdf8;border:1px solid var(--rule);border-radius:6px"></canvas><div class="bars" style="flex:1;min-width:0">'+
      barRows(pr.slice(0,3).map(q=>['종 '+(q[1]+1),q[0],DIGCOL[q[1]],q[0].toFixed(2)]),1)+'</div></div><p class="foot" style="margin-top:6px">글씨 #'+(i+1)+'. 확률의 합은 1이다. '+(pr[0][0]<0.9?'두 종 사이에 있어 나눠 가졌다.':'한 종의 한가운데라 거의 한 묶음 몫이다.')+'</p>';
    const c=$('emPickCv').getContext('2d');drawDigit(c,X[i],0,0,64,'#111827');}
  cv.addEventListener('click',e=>{if(!st.P)return;const r=cv.getBoundingClientRect(),m=map(r.width),px=e.clientX-r.left,py=e.clientY-r.top;let b=-1,bd=1e9;
    for(let i=0;i<N;i++){const d=(m.sx(Z2[i][0])-px)**2+(m.sy(Z2[i][1])-py)**2;if(d<bd){bd=d;b=i;}}if(bd<400){st.pick=b;draw();pickInfo();}});
  $('emStep').addEventListener('click',()=>{if(st.busy||!st.P)return;step();draw();pickInfo();});
  $('emPlay').addEventListener('click',async()=>{if(st.busy||!st.P)return;st.busy=true;$('emPlay').disabled=true;let prev=-Infinity;for(let t=0;t<60;t++){step();draw();if(st.pick>=0)pickInfo();if(Math.abs(st.ll-prev)<1e-3)break;prev=st.ll;await wait(REDUCE?0:220);}st.busy=false;$('emPlay').disabled=false;});
  $('emReset').addEventListener('click',()=>{if(st.busy)return;st.seed++;init();draw();pickInfo();});
  seg('#emK button',v=>{if(st.busy)return;st.K=+v;init();draw();pickInfo();});
  function start(){if(st.P)return;init();draw();pickInfo();}
  whenNear('emCv',start,400);onRedraw(draw);return {start,st};})();

/* ── 02 실기 과제지: 정답 없는 1,797장 ── */
(function(){const cv=$('gridCv');let off=0,pick=-1,cell=34,ncol=10;const PER=()=>ncol*8;
  function draw(){const W=fit(cv,8*cell+10),ctx=cv.getContext('2d'),C=cols();ncol=Math.max(6,Math.floor((W-4)/cell));const H=8*cell+10;fit(cv,H);ctx.clearRect(0,0,W,H);
    for(let j=0;j<PER();j++){const i=(off+j)%N,x=2+(j%ncol)*cell,y=2+Math.floor(j/ncol)*cell;paper(ctx,x,y,cell-3,C);drawDigit(ctx,X[i],x+2,y+2,cell-7,'#111827');if(i===pick){ctx.strokeStyle=C.stamp;ctx.lineWidth=2.5;ctx.strokeRect(x,y,cell-3,cell-3);}}
    $('gridInfo').innerHTML='글씨 #'+(off+1)+' ~ #'+(Math.min(N,off+PER()))+' / 1,797장. 이름표(정답)는 붙어 있지 않다. 정답 1,797개는 <b>채점 봉투</b>에 봉인되어 05장에서만 연다. 글씨를 누르면 64칸의 밝기 숫자가 보인다.';}
  function show(i){pick=i;const px=X[i],f=digitFeat(px),g=manualDigit(px);let h='<table class="tb" style="font-family:var(--mono);font-size:12.5px;width:auto"><tbody>';
    for(let r=0;r<8;r++){h+='<tr>'+px.slice(r*8,r*8+8).map(v=>'<td style="padding:3px 6px;text-align:right;background:'+rgba('#1f6fe5',v/16*0.55)+'">'+v+'</td>').join('')+'</tr>';}h+='</tbody></table>';
    $('pxTbl').innerHTML=h+'<p class="foot">글씨 #'+(i+1)+' · 64칸 = 특징 64개(밝기 0~16). 잉크 합 '+f.ink+' · 위쪽 비율 '+f.top.toFixed(2)+' · 가운데 4칸 '+f.center+' → 매뉴얼 씨의 묶음 <b>'+g+'</b>번 ('+(f.center<12?'가운데 빔':'가운데 참')+', '+(f.ink>300?'잉크 많음':'잉크 적음')+', '+(f.top>0.5?'위가 무거움':'아래가 무거움')+')</p>';draw();}
  cv.addEventListener('click',e=>{const r=cv.getBoundingClientRect(),c=Math.floor((e.clientX-r.left-2)/cell),rw=Math.floor((e.clientY-r.top-2)/cell);if(c<0||c>=ncol||rw<0||rw>=8)return;show((off+rw*ncol+c)%N);});
  $('gridMore').addEventListener('click',()=>{off=(off+PER())%N;draw();});
  draw();show(0);onRedraw(draw);
  const sz=new Array(8).fill(0);MAN.forEach(g=>sz[g]++);
  $('manSizes').innerHTML='매뉴얼 씨의 8묶음 크기: '+sz.map((v,g)=>'<b>'+g+'</b>번 '+v+'장').join(' · ')+'. 묶음이 고르지 않다. 0과 6처럼 가운데가 빈 글씨, 1처럼 잉크가 적은 글씨를 가르는 데까지는 되지만, 나머지는 큰 묶음 몇 개에 몰린다.';})();
/* 64칸 → 30축 */
(function(){const cv=$('pcaCv'),H=200;
  function draw(){const P=ensureP(),W=fit(cv,H),ctx=cv.getContext('2d'),C=cols(),pl=40,pr=10,pt=10,pb=32,gw=W-pl-pr,gh=H-pt-pb;const r=P.ratio,cum=[];r.reduce((s,v,i)=>(cum[i]=s+v),0);
    ctx.clearRect(0,0,W,H);axes(ctx,C,pl,pt,gw,gh);const bw=gw/30;
    r.forEach((v,i)=>{ctx.fillStyle=rgba(C.blue,.75);ctx.fillRect(pl+i*bw+1,pt+gh-v/0.16*gh*0.5,bw-2,v/0.16*gh*0.5);});
    ctx.strokeStyle=C.stamp;ctx.lineWidth=2.4;ctx.beginPath();cum.forEach((v,i)=>{const X_=pl+(i+.5)*bw,Y=pt+gh-v*gh;i?ctx.lineTo(X_,Y):ctx.moveTo(X_,Y);});ctx.stroke();
    [0,0.5,1].forEach(v=>{ctx.fillStyle=C.muted;ctx.textAlign='right';ctx.fillText(v,pl-5,pt+gh-v*gh+4);});[1,10,20,30].forEach(k=>{ctx.textAlign='center';ctx.fillText(k,pl+(k-.5)*bw,H-16);});ctx.fillText('축 번호',pl+gw/2,H-2);
    ctx.fillStyle=C.stamp;ctx.font=font(12,700);ctx.textAlign='right';ctx.fillText('누적 '+pct(cum[29])+' (30축)',pl+gw-6,pt+gh-cum[29]*gh+16);ctx.fillStyle=C.blue;ctx.textAlign='left';ctx.fillText('막대 = 축 하나가 담은 몫',pl+6,pt+gh-8);
    $('pcaInfo').innerHTML='PCA(5-14 편)로 64칸을 "가장 많이 퍼진 방향" 30개로 돌려 본다. 30축이 분산의 <b>'+pct(cum[29])+'</b>를 담는다(이 브라우저 계산 '+Math.round(P.ms)+'ms, scikit-learn과 같은 값). 나머지 34축은 흐린 가장자리처럼 거의 움직이지 않는 방향이다.';}
  whenNear('m2',()=>{draw();onRedraw(draw);},300);})();

$('lineAri').textContent=f3(LINE);
/* ── 03 AI 비서 ── */
const MAIN_PROMPT=`너는 머신러닝 수업 조교다. Google Colab에서 바로 실행되는 파이썬 코드를 써 줘.

[환경] Colab 기본 라이브러리(numpy, scikit-learn)만 쓴다. 다른 패키지는 설치하지 않는다. 첫 줄에 scikit-learn 버전을 출력한다.
[데이터] 〔sklearn.datasets.load_digits〕 의 손글씨 숫자 1,797장(8×8 = 특징 64개). 정답 y는 "채점 봉투"다.
[규정]
  1) 학습에는 X만 쓴다. y는 마지막 채점 칸에서만 쓴다. fit에 y를 넘기지 않는다.
  2) 먼저 PCA로 〔30〕차원까지 줄이고, GaussianMixture(n_components=〔10〕, covariance_type="full", random_state=0)로 묶는다.
  3) 공분산 모양(full, tied, diag, spherical)과 묶음 수(〔2~20〕)는 정답 없이 BIC로 비교한다. BIC는 작을수록 좋다.
  4) 시작점에 따라 결과가 달라지니 random_state를 고정하고, n_init을 바꿔 본 결과도 보여 준다.
[채점] 묶음 번호는 이름이 아니므로 accuracy_score를 쓰지 않는다. adjusted_rand_score와 normalized_mutual_info_score로 잰다. 비교 대상: 무작위 묶음, 〔기존 규칙(ARI 0.108)〕.
[출력] BIC 표, 소속 확률(predict_proba)이 0.9 미만인 글씨 수, ARI·NMI, 묶음마다 대표 한 장에만 이름을 붙였을 때 전체 정답률.
[형식] 셀 단위로 나누고, 각 셀 위에 무엇을 하는지 한국어 주석을 단다. 코드 뒤에 내가 확인해야 할 점 3가지를 적어 줘.`;
$('mainPrompt').textContent=MAIN_PROMPT;$('copyMain').addEventListener('click',()=>copyText(MAIN_PROMPT,'프롬프트를 복사했습니다'));
(function(){const LINES=[
 ['from sklearn.datasets import load_digits'],['from sklearn.preprocessing import StandardScaler',0,'불러오는 것 자체는 괜찮다. 어디에 쓰는지가 문제다(8번 줄).'],['from sklearn.decomposition import PCA'],['from sklearn.mixture import GaussianMixture',0,'지금의 GMM 클래스 이름이다. 괜찮다(옛 이름 GMM은 지워졌다. ④ 버전 메모).'],['from sklearn.metrics import accuracy_score',0,'불러오는 것은 괜찮다. 어디에 쓰는지가 문제다(13번 줄).'],[''],
 ['X, y = load_digits(return_X_y=True)',0,'데이터를 실제로 읽었다. y는 채점 봉투다. 괜찮다.'],
 ['X = StandardScaler().fit_transform(X)      # 표준화는 늘 좋겠지',1,'<b>표준화가 여기서는 해롭다.</b> 가장자리 칸은 거의 늘 0이라 표준편차가 아주 작다. 나누면 우연히 찍힌 점 하나가 크게 부풀어 PCA의 축을 흔든다. A장 8번 칸: 표준화 후 ARI 0.602, 그대로 0.836. 모든 칸이 같은 단위(밝기 0~16)이면 그대로 둔다.'],
 ['Z = PCA(n_components=30).fit_transform(X)',0,'30축으로 줄인다. 괜찮다.'],
 ['gmm = GaussianMixture(n_components=10)',2,'<b>random_state가 없다.</b> GMM은 시작점(K-Means)에 따라 결과가 크게 달라진다. 이 데이터에서는 random_state에 따라 ARI가 0.630~0.836으로 흔들린다(A장 8번 칸). 고정하거나 n_init을 늘린다.'],
 ['gmm.fit(Z, y)                           # 정답도 같이 주자',3,'<b>정답을 학습에 넣으려 했다.</b> 비지도 학습은 y를 쓰지 않는다. scikit-learn의 GaussianMixture.fit은 y를 받기만 하고 <b>오류 없이 무시한다</b>. 그래서 "정답을 줬으니 잘 배웠겠지"라고 착각하기 쉽다. fit(Z)로 쓰고, y는 채점에서만 꺼낸다.'],
 ['labels = gmm.predict(Z)',0,'묶음 번호를 받는다. 괜찮다. 소속 확률이 필요하면 predict_proba.'],
 ['print("정확도:", accuracy_score(y, labels))',4,'<b>묶음 번호를 정답 이름처럼 비교했다.</b> GMM이 붙인 "묶음 3"은 숫자 3이라는 뜻이 아니다. 번호가 뒤섞여 있어 정확도가 0.8%처럼 엉뚱하게 나온다. 번호와 상관없는 ARI·NMI로 잰다.'],[''],
 ['bics = [GaussianMixture(k, random_state=0).fit(Z).bic(Z) for k in range(2, 21)]',0,'묶음 수별 BIC를 잰다. 괜찮다.'],
 ['best_k = range(2, 21)[bics.index(max(bics))]',5,'<b>BIC가 가장 큰 것을 골랐다.</b> BIC는 "−2 × 로그 가능도 + 모수 벌점"이라 <b>작을수록</b> 좋다. max가 아니라 min이다. 반대로 고르면 k = 2나 20처럼 엉뚱한 끝을 고른다.']];
  const host=$('bugCode'),found=new Set();
  host.innerHTML=LINES.map((l,i)=>'<div class="ln" role="listitem" tabindex="0" data-i="'+i+'"><span class="no">'+(i+1)+'</span><span>'+hl(l[0]||' ')+'</span></div>').join('');
  function score(){$('bugScore').textContent=found.size+' / 5';$('bugScore').className='pill'+(found.size===5?' ok':'');}
  function pick(el){const i=+el.getAttribute('data-i'),l=LINES[i];if(!l[0])return;
    if(l[1]){found.add(l[1]);el.classList.add('hit');$('bugMsg').innerHTML=(i+1)+'번 줄: '+l[2]+(found.size===5?'<br><b>다섯 군데를 모두 찾았다.</b> 고친 코드는 A장의 Colab 코드와 같은 순서(그대로 PCA → 시작점 고정 → X만으로 학습 → BIC는 작은 쪽 → 채점 봉투는 마지막에 ARI로)를 따른다.':'');}
    else{el.classList.add('miss');$('bugMsg').innerHTML=(i+1)+'번 줄: '+(l[2]||'이 줄은 괜찮다.');}score();}
  host.addEventListener('click',e=>{const el=e.target.closest('.ln');if(el)pick(el);});
  host.addEventListener('keydown',e=>{if((e.key==='Enter'||e.key===' ')&&e.target.classList.contains('ln')){e.preventDefault();pick(e.target);}});
  $('bugReveal').addEventListener('click',()=>{LINES.forEach((l,i)=>{if(l[1]){found.add(l[1]);host.querySelector('[data-i="'+i+'"]').classList.add('hit','reveal');}});score();$('bugMsg').innerHTML='빨간 줄 다섯 군데: ① 8번 해로운 표준화 ② 10번 시작점 고정 안 함 ③ 11번 정답을 fit에 넣음(조용히 무시됨) ④ 13번 묶음 번호로 정확도 ⑤ 16번 BIC를 크게 고름.';});
  $('bugReset').addEventListener('click',()=>{found.clear();host.querySelectorAll('.ln').forEach(x=>x.classList.remove('hit','miss','reveal'));score();$('bugMsg').textContent='줄을 하나 눌러 보세요.';});score();})();
(function(){const CK=[['데이터를 실제로 읽었나','load_digits가 1,797 × 64인지 본다. AI가 make_blobs 같은 가짜 데이터로 바꾸지 않았는지.'],['정답 y가 학습 칸에 없나','fit, PCA, StandardScaler 어디에도 y가 들어가지 않아야 한다. fit(Z, y)는 오류 없이 무시되어 알아채기 어렵다.'],['줄이고 묶었나','64칸 그대로 full 공분산이면 모수가 21,449개다. PCA로 줄였는지, 줄인 축 수와 담은 분산을 출력했는지.'],['표준화가 필요한 데이터인가','모든 칸이 같은 단위(밝기 0~16)면 하지 않는다. 표준화하면 거의 빈 가장자리 칸이 부풀어 ARI가 0.836 → 0.602로 떨어진다.'],['random_state와 n_init','시작점에 따라 ARI가 0.63~0.84로 흔들린다. 고정하고, n_init으로 여러 번 시작해 가능도가 가장 높은 것을 고른다.'],['BIC는 작은 쪽','bic()는 작을수록 좋다. score()(평균 로그 가능도)는 클수록 좋다. 둘을 섞지 않는다.'],['채점은 ARI·NMI로','묶음 번호는 이름이 아니다. accuracy_score(y, labels)는 뜻이 없다.'],['소속 확률을 그대로 믿지 않았나','30차원에서는 확률이 거의 0 아니면 1로 날카롭다(과신). "확률 0.99"가 정답률 99%라는 뜻이 아니다.'],['결과를 합격선과 비교했나','무작위(ARI ≈ 0)와 매뉴얼 씨(ARI 0.108)보다 낫지 않으면 뽑을 까닭이 없다.']];
  const ul=$('checks');ul.innerHTML=CK.map((c,i)=>'<li><input type="checkbox" id="ck'+i+'"><label for="ck'+i+'"><b>'+c[0]+'</b><small>'+c[1]+'</small></label></li>').join('');
  function upd(){const n=ul.querySelectorAll('input:checked').length;ul.querySelectorAll('li').forEach(li=>li.classList.toggle('done',li.querySelector('input').checked));$('checkScore').textContent='확인 '+n+' / '+CK.length+(n===CK.length?' — 이제 돌려 본다. 결과도 확인 대상이다.':'');}
  ul.addEventListener('change',upd);upd();})();

/* ── 04 수습 기간 ① EM 재생 (30축, 본 모델) ── */
const REPLAY=(function(){const cv=$('rpCv'),cc=$('llCv');let M=null,n=0,busy=false;
  function draw(){if(!M)return;const S=M.snaps[n],cw=cv.clientWidth||600,cell=Math.min(70,(cw-8)/10-6),HH=Math.round(cell+50),W=fit(cv,HH),ctx=cv.getContext('2d'),C=cols(),gap=(W-10*cell)/11;ctx.clearRect(0,0,W,HH);
    const ord=S.w.map((w,k)=>k);ord.forEach((k,j)=>{const x=gap+j*(cell+gap),y=8;paper(ctx,x,y,cell,C);drawDigit(ctx,S.mu[k],x+3,y+3,cell-6,'#111827');ctx.fillStyle=DIGCOL[k];ctx.fillRect(x,y+cell+6,cell*Math.min(1,S.w[k]/0.15),6);ctx.fillStyle=C.muted;ctx.font=font(11,500,1);ctx.textAlign='center';ctx.fillText('종 '+(k+1),x+cell/2,y+cell+26);ctx.fillText(pct(S.w[k],0),x+cell/2,y+cell+40);});
    const W2=fit(cc,200),c2=cc.getContext('2d'),pl=56,pr=12,pt=10,pb=32,gw=W2-pl-pr,gh=200-pt-pb,L=M.snaps.filter(s=>s.ll!=null).map(s=>s.ll),lo=Math.min(...L)-0.5,hi=Math.max(...L)+0.3,last=M.snaps.length-1,lx=i=>pl+i/last*gw,ly=v=>pt+(1-(v-lo)/(hi-lo))*gh;
    c2.clearRect(0,0,W2,200);axes(c2,C,pl,pt,gw,gh);[lo,(lo+hi)/2,hi].forEach(v=>{c2.textAlign='right';c2.fillText(v.toFixed(1),pl-5,ly(v)+4);});vlabel(c2,'평균 로그 가능도',11,pt+gh/2);
    for(let i=0;i<=last;i+=Math.max(1,Math.round(last/8))){c2.textAlign='center';c2.fillText(i,lx(i),200-16);}c2.fillText('EM 반복',pl+gw/2,200-2);
    c2.strokeStyle=C.blue;c2.lineWidth=2.4;c2.beginPath();let first=true;M.snaps.forEach((s,i)=>{if(s.ll==null||i>n)return;first?c2.moveTo(lx(i),ly(s.ll)):c2.lineTo(lx(i),ly(s.ll));first=false;});c2.stroke();
    if(S.ll!=null)circ(c2,lx(n),ly(S.ll),5,C.stamp,null);
    $('rpInfo').innerHTML='EM <b>'+n+'번</b>'+(S.ll!=null?' · 평균 로그 가능도 <b>'+S.ll.toFixed(3)+'</b>':' · 시작점(K-Means 묶음으로 낸 첫 평균)')+'. 위 그림은 종마다 평균(30축 → 64칸으로 되돌린 "평균 글씨")과 비중이다. '+
      (n===0?'시작점은 K-Means가 낸 묶음이다. 벌써 숫자 모양이 보이는 칸이 많다. 그래서 시작점이 결과를 크게 좌우한다(②).':n<last?'반복할수록 가능도가 오르고, 평균 글씨가 조금씩 또렷해진다. 처음 몇 번에 거의 다 오른다.':M.iters+'번째에 가능도 변화가 0.001보다 작아져 멈췄다. scikit-learn도 같은 시작점에서 '+M.iters+'번 만에 멈춘다(A장 4번 칸). 이 브라우저에서 '+Math.round(M.ms)+'ms.');}
  function set(k){n=clamp(k,0,M.snaps.length-1);$('rpN').max=M.snaps.length-1;$('rpN').value=n;$('rpNO').textContent=n;draw();}
  $('rpN').addEventListener('input',e=>{if(M)set(+e.target.value);});
  $('rpPlay').addEventListener('click',async()=>{if(busy||!M)return;busy=true;$('rpPlay').disabled=true;for(let k=0;k<M.snaps.length;k++){set(k);await wait(REDUCE?0:300);}busy=false;$('rpPlay').disabled=false;});
  async function start(){if(M)return;$('rpInfo').textContent='30축에서 EM을 돌리는 중…';M=await ensureMain();set(0);onRedraw(draw);}
  return {start};})();

/* ── 04 ② 시작점 운: 다섯 번 시작해 보기 (정답 없이 가능도·BIC만) ── */
const SEEDS=(function(){const RUNS={},ORDER=['A','B','C','D','E'];
  function lbl(k){return k==='A'?'시작점 A (scikit-learn random_state=0)':'시작점 '+k+' (브라우저 k-means++ '+(ORDER.indexOf(k))+'번)';}
  async function run(k){if(RUNS[k])return RUNS[k];const P=ensureP();let M;if(k==='A')M=await ensureMain();else M=await GM.fitAsync(P.Z,GM.kmeans(P.Z,KC,ORDER.indexOf(k)),{K:KC});RUNS[k]=M;return M;}
  function table(withAri){const ks=ORDER.filter(k=>RUNS[k]);if(!ks.length)return '<p class="muted">아직 돌린 시작점이 없다.</p>';const best=Math.min(...ks.map(k=>RUNS[k].bic));
    return '<div class="tw"><table class="tb" style="font-size:14px"><thead><tr><th>시작점</th><th>EM 반복</th><th>평균 로그 가능도</th><th>BIC</th>'+(withAri?'<th>ARI (채점 봉투)</th>':'')+'</tr></thead><tbody>'+
      ks.map(k=>{const M=RUNS[k],b=M.bic===best;return '<tr'+(b?' style="font-weight:700;color:var(--ok)"':'')+'><td>'+lbl(k)+'</td><td>'+M.iters+'</td><td>'+M.ll.toFixed(3)+'</td><td>'+Math.round(M.bic).toLocaleString()+(b?' ← 가장 작음':'')+'</td>'+(withAri?'<td>'+f3(GM.ari(ENVELOPE,M.labels))+'</td>':'')+'</tr>';}).join('')+'</tbody></table></div>';}
  async function all(box,withAri){for(const k of ORDER){box.innerHTML='<p class="muted">'+lbl(k)+' 돌리는 중…</p>'+table(withAri);await run(k);}box.innerHTML=table(withAri);}
  document.querySelectorAll('[data-seed]').forEach(b=>b.addEventListener('click',async()=>{const k=b.getAttribute('data-seed');$('seedTbl').innerHTML='<p class="muted">계산 중…</p>'+table(false);await run(k);$('seedTbl').innerHTML=table(false);seedNote();}));
  $('seedAll').addEventListener('click',async()=>{$('seedAll').disabled=true;await all($('seedTbl'),false);$('seedAll').disabled=false;seedNote();});
  function seedNote(){const ks=ORDER.filter(k=>RUNS[k]);if(ks.length<2)return;const b=ks.map(k=>RUNS[k].bic);$('seedInfo').innerHTML='같은 데이터, 같은 종 10개인데 시작점만 바꿔도 BIC가 '+Math.round(Math.max(...b)-Math.min(...b)).toLocaleString()+'만큼 달라졌다. EM은 언덕을 오르다 가장 가까운 꼭대기에서 멈추기 때문이다(국소 최적). 그래서 여러 번 시작해 <b>가능도가 가장 높은(BIC가 가장 작은) 것</b>을 고른다. scikit-learn의 <code>n_init</code>이 이 일을 한다. 정답 없이 고르는 방법이다. 이 고르기가 실제로 맞는 글씨 묶음을 골랐는지는 05장에서 채점 봉투로 확인한다.';}
  return {run,table,all,RUNS};})();

/* ── 04 ③ 공분산 모양과 묶음 수 (BIC) ── */
const COV=(function(){const CT=['full','tied','diag','spherical'],RES={},NAME={full:'full · 종마다 마음대로 기운 타원',tied:'tied · 모두 같은 타원',diag:'diag · 축에 나란한 타원',spherical:'spherical · 동그라미'};let cur='full';
  const cv=$('bicCv'),H=280,COL={full:'blue',tied:'ok',diag:'amber',spherical:'muted'};
  async function run(ct){if(RES[ct])return RES[ct];const M=ct==='full'?await ensureMain():await GM.fitAsync(ensureP().Z,lab(PRE.init.i30_10),{K:KC,type:ct});RES[ct]=M;return M;}
  function draw(){const W=fit(cv,H),ctx=cv.getContext('2d'),C=cols(),pl=64,pr=12,pt=12,pb=34,gw=W-pl-pr,gh=H-pt-pb,lo=296000,hi=346000,lx=k=>pl+(k-2)/18*gw,ly=v=>pt+(1-(v-lo)/(hi-lo))*gh;
    ctx.clearRect(0,0,W,H);axes(ctx,C,pl,pt,gw,gh);[300000,320000,340000].forEach(v=>{ctx.textAlign='right';ctx.fillStyle=C.muted;ctx.fillText((v/1000)+'천',pl-5,ly(v)+4);ctx.strokeStyle=rgba(C.muted,.15);ctx.beginPath();ctx.moveTo(pl,ly(v));ctx.lineTo(pl+gw,ly(v));ctx.stroke();});vlabel(ctx,'BIC (작을수록 좋다)',11,pt+gh/2);
    [2,4,6,8,10,12,14,16,18,20].forEach(k=>{ctx.textAlign='center';ctx.fillText(k,lx(k),H-18);});ctx.fillText('묶음 수 k',pl+gw/2,H-3);
    ctx.strokeStyle=rgba(C.stamp,.5);ctx.setLineDash([4,4]);ctx.beginPath();ctx.moveTo(lx(10),pt);ctx.lineTo(lx(10),pt+gh);ctx.stroke();ctx.setLineDash([]);
    CT.forEach((ct,ci)=>{const pts=PRE.bic[ct],on=ct===cur,col=C[COL[ct]];ctx.strokeStyle=rgba(col,on?1:.45);ctx.lineWidth=on?3:1.5;ctx.beginPath();pts.forEach((p,i)=>{i?ctx.lineTo(lx(p[0]),ly(p[1])):ctx.moveTo(lx(p[0]),ly(p[1]));});ctx.stroke();
      const mn=pts.reduce((a,b)=>b[1]<a[1]?b:a);if(on){circ(ctx,lx(mn[0]),ly(mn[1]),6,null,C.stamp,2.5);}
      ctx.fillStyle=rgba(col,on?1:.7);ctx.font=font(12,on?700:500);ctx.textAlign='right';ctx.fillText(ct,pl+gw-8,pt+16+ci*15);
      if(RES[ct])circ(ctx,lx(10),ly(RES[ct].bic),5,col,C.panel,1.5);});}
  async function pickCt(ct){cur=ct;draw();$('covInfo').innerHTML='<p class="muted">'+ct+' 공분산으로 EM을 돌리는 중…</p>';const M=await run(ct);draw();
    const pts=PRE.bic[ct],mn=pts.reduce((a,b)=>b[1]<a[1]?b:a),full=RES.full;
    $('covInfo').innerHTML=statGrid([[M.nParams.toLocaleString(),'외울 숫자(모수)','a'],[Math.round(M.bic).toLocaleString(),'BIC (k = 10, 브라우저)'],[String(M.iters),'EM 반복'],['k = '+mn[0],'이 모양에서 BIC가 가장 작은 k (2~20 중)','s']])+
      '<p class="foot"><b>'+NAME[ct]+'</b>. '+(ct==='full'?'모수가 가장 많지만 BIC도 가장 작다. 글씨 묶음마다 "같이 움직이는 칸"의 방향이 다르기 때문이다(예: 7의 사선 획). 늘어난 모수의 벌점보다 가능도가 훨씬 많이 올랐다.':ct==='tied'?'모든 종이 같은 타원을 나눠 쓴다. 모수는 full의 6분의 1이지만 BIC가 1만 이상 크다. 숫자마다 획이 기우는 방향이 다른데 하나로 묶었기 때문이다.':ct==='diag'?'칸끼리 함께 움직이는 관계(공분산)를 버리고 축마다 퍼짐만 본다. 모수는 적지만 BIC가 크다. 5-5 나이브 베이즈의 "특징끼리 독립" 가정과 같은 모양이다.':'종을 동그라미로만 그린다. 거리만 보는 K-Means와 가장 닮은 모양이다. 모수가 가장 적고 BIC가 가장 크다.')+
      ' 곡선은 Python(scikit-learn, random_state=0)이 k = 2~20을 모두 돌린 값이고, k = 10의 점은 이 브라우저가 같은 시작점에서 직접 낸 값이다(두 값이 같다).'+(full&&ct!=='full'?' full 대비 BIC '+(M.bic>full.bic?'+':'')+Math.round(M.bic-full.bic).toLocaleString()+'.':'')+'</p>';}
  seg('#covChoice button',v=>pickCt(v));
  function start(){draw();pickCt('full');onRedraw(draw);}
  return {start,run,RES};})();
whenNear('m4',()=>{REPLAY.start();COV.start();},300);

/* ── 05 성과 평가 ── */
async function renderEval(){const M=await ensureMain(),G=grade(M);
  $('evalStat').innerHTML=statGrid([['ARI '+f3(G.ari),'GMM full · k = 10 (NMI '+f3(G.nmi)+')','a'],['ARI '+f3(LINE),'군집팀 합격선 · 매뉴얼 씨 8묶음 (NMI '+f3(LINE_NMI)+')','m'],['ARI '+f3(RAND),'무작위 10묶음 (기준선)'],['+'+f3(G.ari-LINE),'합격선과의 차이','s']]);
  /* 묶음 × 숨긴 정답 표: 묶음은 대표 이름 순으로 */
  const ord=[...Array(KC).keys()].sort((a,b)=>G.name[a]-G.name[b]),T=[...Array(KC)].map(()=>new Array(10).fill(0));for(let i=0;i<N;i++)T[M.labels[i]][ENVELOPE[i]]++;
  let h='<div class="cm" style="grid-template-columns:96px repeat(10,minmax(30px,1fr));max-width:560px;font-size:12.5px"><div class="h">묶음 ＼ 정답</div>'+[...Array(10).keys()].map(d=>'<div class="h">'+d+'</div>').join('');
  ord.forEach(k=>{h+='<div class="h" style="justify-content:end;text-align:right;color:'+DIGCOL[k]+'">종 '+(k+1)+' ('+M.size[k]+')</div>'+T[k].map((v,d)=>{const big=v/M.size[k];return '<div class="c'+(v?(d===G.name[k]?' d':' x'):'')+'" style="'+(v&&d===G.name[k]?'background:'+rgba(DIGCOL[k],.25+.6*big)+';color:#fff':'')+'">'+(v||'·')+'</div>';}).join('');});
  $('ctTbl').innerHTML=h+'</div>';
  const mix=ord.map(k=>{const s=[...T[k]].map((v,d)=>[v,d]).sort((a,b)=>b[0]-a[0]);return [k,s[0],s[1]];}).filter(q=>q[2][0]>=10);
  $('ctInfo').innerHTML='줄 = GMM이 만든 묶음(괄호 안은 크기), 칸 = 채점 봉투 속 정답. 묶음 열 개가 숫자 열 개에 거의 하나씩 대응한다. 섞인 곳: '+mix.map(q=>'종 '+(q[0]+1)+'에 숫자 '+q[1][1]+'이(가) '+q[1][0]+'장, 숫자 '+q[2][1]+'이(가) '+q[2][0]+'장').join(', ')+'. 손으로 써도 헷갈릴 만한, 획이 닮은 짝이다.';
  document.querySelectorAll('.gmAri').forEach(e=>e.textContent=f3(G.ari));
  /* 업무 점수 */
  renderWork(M,G);
  /* 군집팀 비교 */
  $('teamBars').innerHTML=barRows([['무작위 10묶음',Math.max(0,RAND),'var(--muted)',f3(RAND)],['잉크 양 10구간',Math.max(0,INK10),'var(--muted)',f3(INK10)],['매뉴얼 씨 8묶음 (합격선)',LINE,'var(--c-man)',f3(LINE)],['5-10 K-Means k=10 (원래 픽셀)',PRE.team.km,'rgba(31,111,229,.55)',f3(PRE.team.km)+' *'],['5-11 계층적 ward',PRE.team.ward,'rgba(31,111,229,.55)',f3(PRE.team.ward)+' *'],['5-13 GMM full (PCA 30)',G.ari,'var(--blue)',f3(G.ari)]],1);
  /* 정답 없이 재는 점수 */
  $('noLabel').innerHTML=statGrid([[M.ll.toFixed(2),'평균 로그 가능도 (클수록 좋다)'],[Math.round(M.bic).toLocaleString(),'BIC (작을수록 좋다)'],['<span id="silV">…</span>','실루엣 (64칸 거리, −1~1)','a']])+'<p class="foot" id="silNote">실루엣은 글씨 1,797장 사이 거리 161만 쌍을 재야 해서 이 브라우저에서 잠깐 걸린다.</p>';
  drawConf(M);
  await wait(30);const S=await GM.silhouette(X,M.labels);$('silV').textContent=f3(S.mean);
  $('silNote').innerHTML='실루엣 '+f3(S.mean)+'은 "내 묶음까지의 평균 거리보다 가장 가까운 다른 묶음까지가 얼마나 먼가"의 평균이다. 1에 가까우면 묶음이 또렷이 떨어져 있다는 뜻이다. 손글씨는 숫자끼리 이어져 있어 값이 낮다. 5-10 K-Means(0.183)와 비슷하다. 실루엣은 둥근 묶음을 좋아하므로, 기운 타원을 쓰는 GMM의 장점을 다 보여 주지는 못한다. 그래서 정답 없는 점수는 하나만 믿지 않는다.';}
/* 2차원 vs 30차원의 소속 확률 */
let M2D=null;
function drawConf(M){if(!M2D){const Z2=ensureP().Z2;M2D=GM.fit(Z2,GM.kmeans(Z2,KC,1,3),{K:KC});M2D.maxp=new Float64Array(N);for(let i=0;i<N;i++){let m=0;for(let k=0;k<KC;k++)m=Math.max(m,M2D.R[i*KC+k]);M2D.maxp[i]=m;}}
  const cv=$('confCv'),H=230,W=fit(cv,H),ctx=cv.getContext('2d'),C=cols(),half=(W-30)/2,NB=10;ctx.clearRect(0,0,W,H);
  [[M2D.maxp,'2축 GMM',C.amber],[M.maxp,'30축 GMM (본 모델)',C.blue]].forEach((q,j)=>{const b=new Array(NB).fill(0);q[0].forEach(v=>b[Math.min(NB-1,Math.floor((v-0.0001)*NB))]++);
    const x0=10+j*(half+10),pt=24,gh=H-pt-40,mx=N;ctx.fillStyle=C.text;ctx.font=font(12.5,700);ctx.textAlign='left';ctx.fillText(q[1],x0,16);
    ctx.strokeStyle=C.rule;ctx.strokeRect(x0,pt,half,gh);b.forEach((c,k)=>{const h=Math.sqrt(c/mx)*gh*0.88;ctx.fillStyle=rgba(q[2],.85);ctx.fillRect(x0+k*half/NB+2,pt+gh-h,half/NB-4,h);if(c){ctx.fillStyle=C.muted;ctx.font=font(10.5,500,1);ctx.textAlign='center';ctx.fillText(c,x0+(k+.5)*half/NB,pt+gh-h-3);}});
    ctx.fillStyle=C.muted;ctx.font=font(11);ctx.textAlign='left';ctx.fillText('0.1',x0,pt+gh+14);ctx.textAlign='right';ctx.fillText('1.0',x0+half,pt+gh+14);ctx.textAlign='center';ctx.fillText('가장 큰 소속 확률',x0+half/2,pt+gh+30);});
  const a2=Array.from(M2D.maxp).filter(v=>v<0.9).length,a30=Array.from(M.maxp).filter(v=>v<0.9).length;
  $('confInfo').innerHTML='가장 큰 소속 확률이 0.9 미만인 글씨: 2축 <b>'+a2.toLocaleString()+'장</b>, 30축 <b>'+a30+'장</b>. 2축에서는 종이 넓게 겹쳐 "반반"인 글씨가 많다. 30축에서는 거의 모든 글씨가 한 종에 확률 0.99 이상으로 붙는다. 차원이 높으면 종과 종 사이가 멀어지고 밀도가 몇십 자릿수씩 차이 나서 확률이 날카로워진다(막대 높이는 제곱근으로 줄였다). <b>소속 확률이 1에 가깝다고 정답일 확률이 그만큼 높은 것은 아니다</b>(과신). 애매한 글씨를 찾을 때는 확률보다 가능도가 쓸모 있다(아래 업무 점수).';}
/* 업무 점수: 대표 한 장 + 가능도 낮은 글씨 다시 보기 */
function renderWork(M,G){const cv=$('repCv'),lc=$('oddCv');const low=[...Array(N).keys()].sort((a,b)=>M.score[a]-M.score[b]);
  function drawReps(){const W=fit(cv,120),ctx=cv.getContext('2d'),C=cols(),cell=Math.min(64,(W-8)/10-8),gap=(W-10*cell)/11,ord=[...Array(KC).keys()].sort((a,b)=>G.name[a]-G.name[b]);ctx.clearRect(0,0,W,120);
    ord.forEach((k,j)=>{const x=gap+j*(cell+gap);paper(ctx,x,6,cell,C);drawDigit(ctx,X[M.rep[k]],x+3,9,cell-6,'#111827');ctx.strokeStyle=DIGCOL[k];ctx.lineWidth=3;ctx.strokeRect(x,6,cell,cell);ctx.fillStyle=C.text;ctx.font=font(13,700,1);ctx.textAlign='center';ctx.fillText('"'+G.name[k]+'"',x+cell/2,cell+26);if(cell>=44){ctx.fillStyle=C.muted;ctx.font=font(11,500,1);ctx.fillText(M.size[k]+'장',x+cell/2,cell+42);}});}
  function drawOdd(){const W=fit(lc,96),ctx=lc.getContext('2d'),C=cols(),cell=38,n=Math.max(8,Math.floor((W-4)/(cell+4))*2),per=Math.floor((W-4)/(cell+4));ctx.clearRect(0,0,W,96);
    low.slice(0,n).forEach((i,j)=>{const x=2+(j%per)*(cell+4),y=2+Math.floor(j/per)*(cell+8);paper(ctx,x,y,cell,C);drawDigit(ctx,X[i],x+2,y+2,cell-4,'#111827');});}
  function upd(){const f=+$('rvN').value/100,c=Math.round(N*f),set=new Set(low.slice(0,c));let ok=0;for(let i=0;i<N;i++){if(set.has(i)||G.name[M.labels[i]]===ENVELOPE[i])ok++;}
    $('rvNO').textContent=Math.round(f*100)+'%';
    $('workStat').innerHTML=statGrid([[pct(G.repAcc),'대표 10장에만 이름 → 전체 정답률','a'],[pct(ok/N),'+ 가능도 하위 '+Math.round(f*100)+'%('+c+'장) 사람이 다시 보기','a'],[(10+c).toLocaleString()+'장','사람이 본 글씨 (1,797장 중)','s'],[pct(1-(10+c)/N,0),'이름표 붙이는 일 절감']]);}
  drawReps();drawOdd();upd();$('rvN').addEventListener('input',upd);onRedraw(()=>{drawReps();drawOdd();});
  $('workInfo').innerHTML='위 열 장은 묶음마다 <b>그 종 모양의 밀도가 가장 높은 글씨</b>(가장 전형적인 한 장)다. 사람은 이 열 장만 보고 이름을 붙인다. 열 장이 0부터 9까지 겹치지 않고 다 나왔다. 그 이름을 묶음 전체에 퍼뜨리면 1,797장 중 <b>'+pct(G.repAcc)+'</b>가 맞는다. 아래 줄은 <b>가능도가 가장 낮은 글씨</b>, 곧 어느 종에도 잘 맞지 않는 글씨다. 굵거나 기울었거나 잘린 글씨가 많다. 틀린 이름의 상당수가 여기 몰려 있어서, 이 글씨만 사람이 다시 보면 정답률이 빠르게 오른다. K-Means는 거리만 주지만, GMM은 "이 글씨가 얼마나 그럴듯한가"(밀도)를 함께 준다.';}
whenNear('m5',renderEval,500);
document.addEventListener('click',async e=>{const b=e.target.closest('#seedGrade');if(!b)return;b.disabled=true;await SEEDS.all($('seedGradeTbl'),true);b.disabled=false;
  const R=SEEDS.RUNS,ks=Object.keys(R),bestK=ks.reduce((a,k)=>R[k].bic<R[a].bic?k:a),aris=ks.map(k=>GM.ari(ENVELOPE,R[k].labels));
  const bA=GM.ari(ENVELOPE,R[bestK].labels),top=bA>=Math.max(...aris)-1e-9;
  $('seedGradeInfo').innerHTML='채점 봉투로 확인하니 ARI가 '+f3(Math.min(...aris))+' ~ '+f3(Math.max(...aris))+'로 흔들렸다. BIC가 가장 작은 시작점('+bestK+')의 ARI는 <b>'+f3(bA)+'</b>'+(top?'로 다섯 중 가장 높다. 가능도(정답 없이 재는 점수)로 고른 것이 정답과도 가장 잘 맞았다.':'다. 가능도로 고른 것이 정답과 가장 잘 맞지는 않았다. 가능도 차이가 작으면 묶음이 꽤 달라도 비슷한 점수를 받는다.')+' 늘 그렇다는 보장은 없지만, 여러 번 시작해 가능도가 가장 높은 것을 고르면 운이 나쁜 시작점(ARI 0.6 언저리)은 피한다(A장 7·8번 칸: random_state 0~5에서 ARI 0.630~0.836, n_init=10은 0.840).';});

/* ── 면접 결과 발표 ── */
Interview('iv13',{title:'GMM은 정답 없이 손글씨 1,797장을 숫자별로 묶을 수 있을까?',
  q:'조건: 채점 봉투를 열어 ARI가 군집팀 합격선(매뉴얼 씨 8묶음 ARI '+f3(LINE)+')과 무작위 묶음(ARI '+f3(RAND)+')을 모두 넘으면 합격. 근무 조건은 PCA 30축 · 종 10개 · full 공분산 · 시작점 random_state=0. EM은 이 브라우저가 직접 돌린다.',
  fly:'정답은 정말 학습 끝나고 열었어? 그리고 확률 0.99라는 말, 믿어도 돼?',robo:'종 하나에 공분산 숫자가 몇 개야? 다 합쳐 외울 숫자는?',
  async run(setP){setP(.2);await wait(30);const M=await ensureMain();setP(.8);await wait(30);const G=grade(M);setP(1);
    const pass=G.ari>LINE&&G.ari>RAND+0.01;
    return {result:pass?'pass':'hold',flyAns:'ARI '+f3(G.ari)+' · NMI '+f3(G.nmi)+' · 정답은 EM이 끝난 뒤에만 열었음 · 30축 확률은 과신(0.9 미만 '+Array.from(M.maxp).filter(v=>v<0.9).length+'장뿐)',
      roboAns:'종 하나에 평균 30 + 공분산 465 · 외울 숫자 '+M.nParams.toLocaleString()+'개 · EM '+M.iters+'번 · '+Math.round(M.ms)+'ms(이 브라우저)',
      flyMood:pass?'arms':'coffee',flyMove:pass?'nod':'shake',roboMood:pass?'wow':'cry',roboMove:pass?'jump':'shake',asstMood:pass?'spark':'sigh',docMood:pass?'judge':'scale',
      measure:'채점 봉투를 열었어요. <b>ARI '+f3(G.ari)+'</b>(NMI '+f3(G.nmi)+')로 합격선 '+f3(LINE)+'보다 '+f3(G.ari-LINE)+' 높고, 무작위('+f3(RAND)+')와는 비교가 안 돼요. 대표 10장에만 이름을 붙여도 '+pct(G.repAcc)+'가 맞아요. 군집팀에서 지금까지 가장 높은 ARI입니다(5-10 K-Means '+f3(PRE.team.km)+', 5-11 ward '+f3(PRE.team.ward)+').',
      verdict:pass?'합격입니다. 군집팀에 배치합니다. 조건이 둘 있어요. 먼저 PCA로 줄인 뒤에 일할 것, 그리고 시작점을 여러 번(n_init) 바꿔 가능도가 가장 높은 결과를 쓸 것.':'보류입니다. 합격선을 넘지 못했어요.',
      key:'겹친 종 모양 열 개 — ARI '+f3(LINE)+' → '+f3(G.ari)+' · 대표 10장으로 '+pct(G.repAcc),
      render(box){box.innerHTML='<div class="bars">'+barRows([['GMM (PCA 30 · full)',G.ari,'var(--blue)',f3(G.ari)],['매뉴얼 씨 8묶음',LINE,'var(--c-man)',f3(LINE)],['무작위 10묶음',Math.max(0,RAND),'var(--muted)',f3(RAND)]],1)+'</div>';}};},
  alts:[
   {label:'공분산을 diag(축에 나란한 타원)로',async run(box){const M=await COV.run('diag'),a=GM.ari(ENVELOPE,M.labels),F=await ensureMain();
     box.innerHTML=statGrid([['ARI '+f3(a),'diag · k = 10','a'],['ARI '+f3(GM.ari(ENVELOPE,F.labels)),'full (본 조건)','s'],[M.nParams.toLocaleString(),'외울 숫자'],['ARI '+f3(LINE),'합격선','m']])+'<p class="foot">칸끼리 함께 움직이는 관계를 버리니 외울 숫자는 '+Math.round(F.nParams/M.nParams)+'분의 1로 줄었지만 ARI가 크게 떨어졌다. 그래도 합격선은 넘으므로 판정은 '+(a>LINE?/*i18n:verdict*/'합격':'<b>보류</b>')+'. 04장 BIC도 diag보다 full을 골랐다. 정답 없이 고른 쪽이 채점에서도 나았다.</p>';}},
   {label:'BIC가 고른 k = 8로',async run(box){const M=await GM.fitAsync(ensureP().Z,lab(PRE.init.i30_8),{K:8}),a=GM.ari(ENVELOPE,M.labels);
     box.innerHTML=statGrid([['ARI '+f3(a),'k = 8 (BIC '+Math.round(M.bic).toLocaleString()+')','a'],['ARI '+f3(GM.ari(ENVELOPE,(await ensureMain()).labels)),'k = 10 (BIC 300,818)','s'],['ARI '+f3(LINE),'합격선','m']])+'<p class="foot">정답 없이 BIC만 보면 k = 8이 가장 작다. 묶음이 여덟이면 숫자 두 쌍이 한 묶음에 들어가야 하므로 ARI가 낮아진다. 판정은 '+(a>LINE?/*i18n:verdict*/'합격':'<b>보류</b>')+'. "숫자는 열 종류"는 사람이 아는 업무 지식이다. 이런 지식이 있으면 k를 정하는 데 쓰고, 없으면 BIC가 고른 k에서 묶음을 사람이 살펴 합치거나 나눈다.</p>';}},
   {label:'PCA 없이 64칸 그대로',async run(box){box.innerHTML='<p class="muted">64칸 full 공분산으로 EM을 돌리는 중… (몇 초 걸린다)</p>';const t0=performance.now(),M=await GM.fitAsync(X,lab(PRE.init.i64_10),{K:KC}),a=GM.ari(ENVELOPE,M.labels);
     box.innerHTML=statGrid([['ARI '+f3(a),'64칸 그대로 (EM '+M.iters+'번, '+(Math.round(performance.now()-t0)/1000).toFixed(1)+'초)','a'],['ARI '+f3(GM.ari(ENVELOPE,(await ensureMain()).labels)),'PCA 30축 (본 조건)','s'],[M.nParams.toLocaleString(),'외울 숫자 (글씨는 1,797장)'],['ARI '+f3(LINE),'합격선','m']])+'<p class="foot">늘 0인 가장자리 칸은 퍼짐이 0이라, 공분산에 아주 작은 값(reg_covar = 10⁻⁶)을 더해야 겨우 계산이 된다. 그 칸 방향으로는 종이 바늘처럼 뾰족해져 밀도가 터무니없이 커지고, 묶음이 그 바늘을 따라 갈린다(BIC도 30축보다 훨씬 작게 나오지만, 차원이 다르면 BIC끼리 비교할 수 없다). 판정은 '+(a>LINE?/*i18n:verdict*/'합격':'<b>보류</b>')+'이지만 ARI가 크게 떨어지고 계산도 무겁다. 그래서 GMM은 줄인 다음에 쓴다. 줄이는 일은 다음 지원자 5-14 PCA의 몫이다.</p>';}}]});

/* ── 06 인사 카드: GMM이 그린 새 글씨 ── */
(function(){const cv=$('genCv');let seed=1,M=null;
  function sample(){const P=ensureP(),r=RNG(seed*97+13),d=30,out=[];function gauss(){let u=0,v=0;while(!u)u=r();v=r();return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v);}
    for(let k=0;k<KC;k++){const L=GM.chol(M.P.cov[k],d);for(let s=0;s<2;s++){const e=[];for(let a=0;a<d;a++)e.push(gauss());const z=new Float64Array(d);for(let a=0;a<d;a++){let v=M.P.mu[k][a];for(let b=0;b<=a;b++)v+=L[a*d+b]*e[b];z[a]=v;}out.push([k,clip16(P.inv(z))]);}}return out;}
  function draw(){if(!M)return;const S=sample(),cw=cv.clientWidth||600,cell=Math.min(56,(cw-8)/10-8),HH=Math.round(2*(cell+8)+4),W=fit(cv,HH),ctx=cv.getContext('2d'),C=cols(),gap=(W-10*cell)/11,G=grade(M),ord=[...Array(KC).keys()].sort((a,b)=>G.name[a]-G.name[b]);ctx.clearRect(0,0,W,HH);
    ord.forEach((k,j)=>{const x=gap+j*(cell+gap);S.filter(q=>q[0]===k).forEach((q,s)=>{const y=4+s*(cell+8);paper(ctx,x,y,cell,C);drawDigit(ctx,q[1],x+3,y+3,cell-6,'#111827');});ctx.fillStyle=DIGCOL[k];ctx.fillRect(x,4+2*(cell+8)-4,cell,4);});}
  $('genBtn').addEventListener('click',async()=>{M=await ensureMain();seed++;draw();});
  whenNear('m6',async()=>{M=await ensureMain();draw();onRedraw(draw);},400);})();

/* ══ 확인 문제 ══ */
var QUIZ={
q1:[{q:'GMM이 K-Means와 가장 크게 다른 점은?',o:[['글씨마다 묶음별 소속 확률을 나눠 갖는다',1,'K-Means는 가장 가까운 중심 하나에 딱 넣는다. GMM은 "종 1일 확률 0.6, 종 2일 확률 0.4"처럼 나눠 갖고, 종마다 모양(공분산)과 몫(비중)도 다르다.'],['묶음 수를 미리 정하지 않는다',0,'GMM도 종의 개수 k를 정해야 한다. 미리 안 정하는 것은 5-12 DBSCAN이다.'],['정답을 보고 배운다',0,'둘 다 비지도다. 정답은 채점에만 쓴다.']]},
 {q:'GMM의 "종 모양" 하나를 정하는 것은?',o:[['평균(자리), 공분산(퍼짐과 기울기), 비중(몫)',1,'평균이 종의 가운데, 공분산이 타원의 크기와 방향, 비중이 전체에서 그 종이 차지하는 몫이다.'],['중심 하나뿐',0,'그것은 K-Means다. GMM은 퍼짐과 몫까지 배운다.'],['문턱값 몇 개',0,'그것은 결정 트리다.']]},
 {q:'01장 2축 그림에서 색이 섞인 큰 점의 뜻은?',o:[['두 종 사이에 있어 소속 확률을 나눠 가진 글씨',1,'가장 큰 소속 확률이 0.9 미만인 글씨다. 확률대로 색을 섞어 칠했다.'],['잘못 읽은 데이터',0,'데이터는 그대로다. 위치가 애매할 뿐이다.'],['정답이 둘인 글씨',0,'정답은 하나다. 모델이 확신하지 못할 뿐이다.']]}],
q2:[{q:'손글씨 64칸 그대로 full GMM 10묶음을 맞추면 외울 숫자(모수)는?',o:[['21,449개 — 글씨 1,797장보다 훨씬 많다',1,'종 하나에 평균 64 + 공분산 64×65/2 = 2,080. 열 개면 2만이 넘는다. 30축으로 줄이면 4,959개다.'],['640개',0,'평균만 센 값이다. 공분산이 대부분을 차지한다.'],['64개',0,'특징 수다.']]},
 {q:'"채점 봉투"를 따로 두는 까닭은?',o:[['비지도 학습은 정답 없이 배우고, 정답은 결과를 채점할 때만 써야 하기 때문',1,'학습에 정답이 섞이면 "정답 없이도 묶을 수 있나"를 잴 수 없다. 그래서 1,797장 전체로 학습하고, 정답은 마지막에 연다.'],['시험지를 25% 떼어야 해서',0,'정답을 학습에 쓰지 않으니 시험지를 따로 떼지 않는다(지도 본부와 다른 점).'],['정답이 틀려서',0,'정답은 맞다. 쓰는 시점을 정한 것이다.']]}],
q3:[{q:'gmm.fit(Z, y)처럼 정답 y를 넘기면?',o:[['오류 없이 y를 무시한다. 그래서 잘못을 알아채기 어렵다',1,'scikit-learn의 비지도 모델은 API를 맞추려고 y를 받기만 한다. 결과는 fit(Z)와 같다.'],['정답을 보고 더 잘 묶는다',0,'GMM은 y를 쓰지 않는다.'],['오류가 난다',0,'나지 않는다. 그래서 더 위험하다.']]},
 {q:'GMM 결과를 accuracy_score(y, labels)로 재면?',o:[['묶음 번호는 이름이 아니라 0.8%처럼 엉뚱한 값이 나온다',1,'"묶음 3"이 숫자 3이라는 보장이 없다. 번호와 상관없는 ARI·NMI로 잰다.'],['정확한 정답률이 나온다',0,'번호가 뒤섞여 있어 뜻이 없다.'],['ARI와 같은 값이 나온다',0,'ARI는 번호를 바꿔도 같은 값을 주도록 만든 점수다.']]},
 {q:'GaussianMixture(k).bic(Z)로 k를 고를 때 맞는 것은?',o:[['BIC가 가장 작은 k를 고른다',1,'BIC = −2 × 로그 가능도 + 모수 개수 × ln(n). 잘 맞으면서 단순할수록 작다.'],['BIC가 가장 큰 k를 고른다',0,'반대다. score()(평균 로그 가능도)는 클수록 좋아서 헷갈리기 쉽다.'],['BIC는 정답이 있어야 잰다',0,'정답 없이 잰다. 그래서 비지도에서 쓴다.']]}],
q4:[{q:'EM의 E 단계와 M 단계가 하는 일은?',o:[['E: 지금의 종 모양으로 소속 확률을 나눈다 · M: 그 확률로 비중·평균·공분산을 다시 잰다',1,'K-Means의 "가까운 중심에 배정 → 평균으로 이동"을 확률로 바꾼 것이다. 반복할수록 가능도가 오른다(내려가지 않는다).'],['E: 정답과 비교한다 · M: 틀린 것을 고친다',0,'정답은 쓰지 않는다.'],['E: 묶음 수를 늘린다 · M: 줄인다',0,'묶음 수는 고정이다.']]},
 {q:'시작점만 바꿨는데 결과가 달라지는 까닭과 대책은?',o:[['EM은 가까운 꼭대기에서 멈춘다(국소 최적). 여러 번 시작해 가능도가 가장 높은 것을 고른다(n_init)',1,'A장: random_state에 따라 ARI 0.630~0.836. 가능도로 고르는 것은 정답 없이 할 수 있다.'],['계산 오류다. 다시 설치한다',0,'오류가 아니라 EM의 성질이다.'],['정답으로 가장 좋은 시작점을 고른다',0,'정답은 채점에만 쓴다. 정답으로 고르면 비지도가 아니다.']]},
 {q:'이 데이터에서 BIC가 가장 작았던 공분산 모양과 묶음 수는?',o:[['full, k = 8',1,'full이 어느 k에서나 가장 작고, full 안에서는 k = 8(299,782)이 k = 10(300,818)보다 조금 작다. 숫자가 열 종류라는 건 채점 봉투 속 정보다.'],['spherical, k = 10',0,'spherical은 모든 k에서 BIC가 가장 크다.'],['diag, k = 20',0,'diag는 칸끼리의 관계를 버려 BIC가 크다.']]}],
q5:[{q:'30축 GMM에서 가장 큰 소속 확률이 0.9 미만인 글씨는 몇 장이었나? 이것이 뜻하는 것은?',o:[['2장 — 고차원에서는 확률이 날카로워져 과신한다. 확률을 정답률로 읽으면 안 된다',1,'2축에서는 1,000장 남짓이 애매했다. 30축에서는 밀도가 몇십 자릿수씩 차이 나서 거의 0 아니면 1이 된다. 그런데 틀린 글씨는 140장이 넘는다.'],['2장 — 거의 모든 글씨를 맞혔다는 뜻이다',0,'대표 이름으로 퍼뜨리면 8%쯤 틀린다. 확신과 정답은 다르다.'],['1,029장 — 반 이상이 애매하다',0,'그것은 2축에서 맞춘 GMM의 숫자다.']]},
 {q:'업무 점수에서 "가능도가 낮은 글씨만 사람이 다시 보기"가 효과적인 까닭은?',o:[['어느 종에도 잘 맞지 않는 글씨에 틀린 이름이 몰려 있기 때문',1,'가능도 하위 10%(180장)에 틀린 이름의 절반 넘게 들어 있다. 이 글씨만 다시 보면 92.0% → 96.2%.'],['가능도가 낮은 글씨는 모두 틀렸기 때문',0,'다 틀린 것은 아니다. 틀릴 가능성이 높을 뿐이다.'],['정답을 보고 골랐기 때문',0,'가능도는 정답 없이 잰다.']]},
 {q:'ARI 0의 뜻은?',o:[['무작위로 묶은 것과 같은 수준',1,'ARI는 우연히 겹칠 몫을 빼고 잰다. 무작위 10묶음은 0.001, 매뉴얼 씨는 0.108, GMM은 0.836.'],['완벽히 반대로 묶었다',0,'ARI는 음수가 될 수도 있지만 0은 "우연 수준"이다.'],['묶음이 0개',0,'묶음 수와는 관계없다.']]}],
q6:[{q:'GMM이 새 글씨를 그릴 수 있는 까닭은?',o:[['데이터가 어떻게 퍼져 있는지(확률 분포)를 배운 생성 모델이라서',1,'종 하나를 비중대로 고르고, 그 종의 평균·공분산에서 점을 뽑아 PCA를 거꾸로 돌리면 새 글씨가 된다.'],['글씨를 외워 두었다가 꺼내서',0,'외운 글씨가 아니라 뽑은 점이다. 그래서 조금씩 다르다.'],['정답 이름을 알아서',0,'정답 없이 배웠다.']]},
 {q:'GMM을 손글씨에 쓸 때 PCA를 먼저 하는 까닭으로 가장 알맞은 것은?',o:[['64칸 그대로는 모수가 데이터보다 많고, 늘 0인 칸 때문에 공분산이 망가지기 때문',1,'64칸 그대로 ARI 0.611, PCA 30축 0.836. 다음 지원자 5-14 PCA가 이 일을 맡는다.'],['PCA를 하면 정답을 알 수 있어서',0,'PCA도 정답을 쓰지 않는다.'],['GMM은 2차원에서만 돌아가서',0,'몇 차원이든 돈다. 다만 차원이 크면 모수가 폭발한다.']]}]};

/* ══ 공통 B·C·D·E·F 데이터 ══ */
var PROMPTS=[
 {t:'① 내 데이터로 GMM 코드 받기',use:'내 공장의 이미지·측정 데이터로 03장 프롬프트를 맞출 때',p:MAIN_PROMPT},
 {t:'② AI 코드 검증 요청',use:'받은 코드를 다른 AI(또는 같은 AI의 새 대화)에게 검토받을 때',p:'아래 scikit-learn GaussianMixture 코드를 코드 리뷰해 줘. 특히 다음을 하나씩 확인하고, 문제가 있으면 줄 번호와 고친 코드를 보여 줘.\n1) 정답 y가 fit, PCA, 전처리 어디에든 들어가지 않았나 (fit(Z, y)는 오류 없이 무시된다)\n2) 모든 특징이 같은 단위인데 StandardScaler를 써서 거의 빈 특징을 부풀리지 않았나\n3) 차원이 큰데 PCA 없이 full 공분산을 쓰지 않았나 (모수 개수를 계산해 데이터 수와 비교해 줘)\n4) random_state와 n_init이 있나\n5) BIC로 k를 고를 때 가장 작은 값을 골랐나 (bic는 작을수록, score는 클수록 좋다)\n6) 묶음 번호를 accuracy_score로 정답과 비교하지 않았나. ARI·NMI를 썼나\n7) predict_proba의 확률을 정답률처럼 해석하지 않았나\n추측으로 답하지 말고, 확실하지 않은 부분은 "공식 문서 확인 필요"라고 표시해 줘.'},
 {t:'③ 이상한 글씨(불량) 찾기로 바꾸기',use:'GMM의 밀도(score_samples)로 "드문 것"을 찾는 업무에 쓸 때',p:'아래 Colab 코드(GaussianMixture, PCA 30축)를 바탕으로 "평소와 다른 글씨(이상치) 찾기" 셀을 덧붙여 줘.\n1) gmm.score_samples(Z)로 글씨마다 로그 가능도를 구하고, 하위 〔2%〕를 "다시 볼 글씨"로 표시한다\n2) 그 글씨들을 8×8 그림으로 격자에 그린다 (matplotlib)\n3) 문턱은 정답을 보지 않고 정한다. 정답(y)은 마지막에 "다시 볼 글씨 중 대표 이름이 틀린 비율"을 재는 데만 쓴다\n4) 같은 일을 IsolationForest로도 해 보고 겹치는 글씨 수를 센다\n5) 해석 주의점 3가지를 적는다 (예: 가능도는 차원 수에 따라 크기가 달라 다른 모델과 직접 비교하지 않는다)'},
 {t:'④ 수업용 문제 만들기',use:'이 편으로 수업할 때 확인 문제가 더 필요할 때',p:'가우시안 혼합 모델(GMM) 입문 수업용 문제를 만들어 줘. 대상은 〔제조 현장 실무자, 수식에 익숙하지 않음〕이다. 예시는 손글씨 숫자로 통일한다.\n- 소속 확률·EM·공분산 모양·BIC를 각각 묻는 객관식 4문제 (보기 3개, 정답 해설 포함)\n- 오해를 드러내는 문제 2개: "확률 0.99면 99% 맞는다", "BIC는 클수록 좋다"\n- 손으로 풀 수 있는 계산 문제 1개: 1차원 점 다섯 개와 두 종(평균·표준편차·비중이 주어짐)으로 한 점의 소속 확률을 구하고, 그 확률로 가중한 새 평균을 구하기 (소수 둘째 자리)\n정답은 맨 끝에 따로 모아 줘.'}];
var GL=[
 ['모델','GMM','Gaussian Mixture Model. 데이터가 종 모양(정규분포) 여러 개를 섞은 데서 나왔다고 보고, 종마다 평균·공분산·비중을 EM으로 맞추는 모델. 가우시안 혼합 모델.'],
 ['모델','가우시안','정규분포의 다른 이름. 가운데가 가장 높고 양쪽으로 갈수록 낮아지는 종 모양 분포. 여러 차원이면 타원 모양 언덕이 된다.'],
 ['모델','정규분포','평균과 퍼짐(분산)으로 정해지는 종 모양 확률 분포. 가우시안 분포.'],
 ['모델','혼합 모델','여러 분포를 비중대로 섞은 모델. 글씨 하나는 "종 하나를 비중대로 고르고 그 종에서 뽑힌 것"으로 본다.'],
 ['모델','공분산','두 특징이 함께 움직이는 정도. 공분산 행렬은 종 모양 타원의 크기와 기울기를 정한다.'],
 ['모델','비중','전체 데이터에서 그 종이 차지하는 몫. 비중을 다 더하면 1이다. 혼합 가중치.'],
 ['모델','생성 모델','데이터가 어떻게 퍼져 있는지(분포)를 배워, 새 데이터를 뽑아 만들 수도 있는 모델. GMM은 새 글씨를 그릴 수 있다.'],
 ['모델','PCA','주성분 분석. 데이터가 가장 많이 퍼진 방향부터 새 축을 잡아 차원을 줄인다. 5-14 편의 지원자.'],
 ['모델','K-Means','중심 k개를 두고 가장 가까운 중심에 배정 → 평균으로 중심 이동을 되풀이하는 군집. GMM의 둥근(spherical) 공분산·딱 잘라 나누기 버전과 닮았다. 5-10 편.'],
 ['학습','EM','Expectation–Maximization. E 단계(지금의 종 모양으로 소속 확률 계산)와 M 단계(그 확률로 비중·평균·공분산 다시 계산)를 번갈아 가능도를 올린다.'],
 ['학습','소속 확률','글씨 하나가 각 종에서 나왔을 확률. 합이 1. responsibility. predict_proba가 돌려준다.'],
 ['학습','가능도','지금의 모델이 이 데이터를 낼 그럴듯함. 로그를 취해 더한 로그 가능도를 쓴다. 클수록 잘 맞는다.'],
 ['학습','국소 최적','언덕을 오르다 가장 가까운 꼭대기에서 멈춘 상태. EM은 시작점에 따라 다른 꼭대기에서 멈출 수 있다.'],
 ['학습','n_init','시작점을 여러 번 바꿔 EM을 돌리고 가능도가 가장 높은 결과를 고르는 횟수.'],
 ['학습','reg_covar','공분산 대각선에 더하는 아주 작은 값(기본 10⁻⁶). 늘 같은 값인 특징 때문에 공분산이 망가지는 것을 막는다.'],
 ['학습','공분산 모양','full(종마다 마음대로 기운 타원), tied(모두 같은 타원), diag(축에 나란한 타원), spherical(동그라미). 모수 개수와 표현력이 다르다.'],
 ['학습','모수','모델이 데이터에서 배워 외우는 숫자. GMM에서는 비중·평균·공분산의 숫자들.'],
 ['학습','차원의 저주','차원이 커질수록 필요한 데이터가 폭발하고, 거리와 밀도가 이상하게 행동하는 일. 64칸 full 공분산은 모수가 2만을 넘는다.'],
 ['평가','BIC','베이즈 정보 기준. −2 × 로그 가능도 + 모수 개수 × ln(데이터 수). 잘 맞으면서 단순할수록 작다. 작은 쪽을 고른다.'],
 ['평가','AIC','아카이케 정보 기준. BIC와 비슷하지만 모수 벌점이 2 × 모수 개수로 더 가볍다.'],
 ['평가','ARI','조정 랜드 지수. 두 묶음 방식이 "같은 묶음인 쌍"을 얼마나 함께 두는지 재고, 우연히 겹칠 몫을 뺀 점수. 1이면 완전히 같고 0이면 무작위 수준.'],
 ['평가','NMI','정규화 상호 정보량. 묶음을 알면 정답에 대한 불확실성이 얼마나 줄어드는지를 0~1로 잰 점수.'],
 ['평가','실루엣 점수','글씨마다 (가장 가까운 다른 묶음까지 평균 거리 − 내 묶음 안 평균 거리) ÷ 둘 중 큰 값의 평균. −1~1. 정답 없이 잰다.'],
 ['평가','채점 봉투','이 시리즈의 비지도 본부에서 숨긴 정답(0~9)을 부르는 말. 학습이 끝난 뒤 채점에만 연다.'],
 ['평가','업무 점수','묶음마다 대표 한 장에만 사람이 이름을 붙이고 그 이름을 묶음 전체에 퍼뜨렸을 때 전체 정답률. 라벨링 비용 절감의 크기를 잰다.'],
 ['평가','과신','모델이 낸 확률이 실제로 맞을 확률보다 높은 것. 30축 GMM은 거의 모든 글씨에 0.99 이상을 주지만 8%쯤 틀린다.'],
 ['평가','합격선','지원자가 넘어야 할 점수. 군집팀은 매뉴얼 씨의 8묶음 규칙이 낸 ARI 0.108.']];
var ANALOGY=[
 ['종 모양 여럿의 섞임','공장에 여러 작업자가 쓴 일지가 섞여 있다. 사람마다 글씨 버릇(평균)과 흔들림(퍼짐)이 달라, 한 장을 보면 "누가 썼을 법한가"를 확률로 짐작한다',1,'','뇌가 감각 입력을 "여러 원인의 섞임"으로 보고 원인을 추론한다는 이론(베이지안 뇌 가설)은 있지만, 가우시안 혼합을 그대로 계산한다는 회로 근거는 없다'],
 ['소속 확률 (나눠 갖기)','애매한 글씨는 "1일 것 같지만 7일 수도"라고 메모해 두는 꼼꼼한 담당자',2,'사람이 애매한 소리를 듣고 두 범주 사이의 확률적 판단을 한다는 범주 지각 연구(Feldman 등, 2009의 지각 자석 효과 모형)','사람의 범주 판단은 경험과 맥락으로 계속 바뀐다. GMM의 확률은 고정된 종 모양에서 계산되고, 고차원에서는 지나치게 확신한다'],
 ['EM 반복','반 배정: 먼저 반을 대충 나누고(E), 반마다 평균 키·몸무게를 다시 재고(M), 그 평균으로 다시 나누기를 되풀이한다',1,'','뇌에서 E·M 단계를 번갈아 계산한다는 직접 근거는 없다'],
 ['밀도와 이상치 (초파리)','처음 맡는 냄새는 기존 냄새 묶음 어디에도 잘 맞지 않는다. 그래서 "새롭다"고 알아챈다',2,'초파리 버섯체 출력 뉴런 MBON-α\'3이 새 냄새에 크게 반응하고 익숙해지면 줄어든다는 새로움 감지 연구(Hattori 등, 2017)','초파리는 냄새마다 드문 케니언 세포 조합으로 반응하고 습관화로 새로움을 판단한다. "종 모양의 밀도가 낮다"를 계산하는 것은 아니다'],
 ['BIC (단순함 벌점)','같은 설명력이면 규칙이 적은 매뉴얼이 낫다. 조항 하나 늘 때마다 벌점을 매기는 감사관',1,'','오컴의 면도날에 해당하는 계산이 뇌 어디서 일어나는지는 알려지지 않았다'],
 ['생성 모델 (새 글씨 그리기)','글씨 버릇을 익힌 사람이 "이 사람이라면 3을 이렇게 쓰겠지" 하고 흉내 내 써 보는 것',2,'뇌가 바깥 세계를 예측하는 생성 모델을 가진다는 예측 부호화 이론(Rao & Ballard, 1999)','예측 부호화는 여러 층의 계층 모델이고 시간에 따라 계속 고친다. GMM은 한 층짜리 정적인 모델이다']];
var REFS=[
 'Alpaydin, E., &amp; Kaynak, C. (1998). Optical Recognition of Handwritten Digits [Dataset]. UCI Machine Learning Repository. <a href="https://doi.org/10.24432/C50P49" target="_blank" rel="noopener">doi:10.24432/C50P49</a> (CC BY 4.0) — 02장 실기 과제지(scikit-learn load_digits는 이 데이터의 테스트 세트 1,797장)',
 'Dempster, A. P., Laird, N. M., &amp; Rubin, D. B. (1977). Maximum likelihood from incomplete data via the EM algorithm. <i>Journal of the Royal Statistical Society: Series B</i>, 39(1), 1–38. — EM, 01·04장',
 'Schwarz, G. (1978). Estimating the dimension of a model. <i>The Annals of Statistics</i>, 6(2), 461–464. — BIC, 04장',
 'McLachlan, G. J., &amp; Peel, D. (2000). <i>Finite Mixture Models</i>. Wiley. — 혼합 모델 전반',
 'Bishop, C. M. (2006). <i>Pattern Recognition and Machine Learning</i>, 9장 Mixture Models and EM. Springer. — K-Means와 GMM의 관계, 00·01장',
 'Fraley, C., &amp; Raftery, A. E. (2002). Model-based clustering, discriminant analysis, and density estimation. <i>Journal of the American Statistical Association</i>, 97(458), 611–631. — 공분산 모양과 BIC로 모델 고르기, 04장',
 'Hubert, L., &amp; Arabie, P. (1985). Comparing partitions. <i>Journal of Classification</i>, 2(1), 193–218. — ARI, 05장',
 'Rousseeuw, P. J. (1987). Silhouettes: A graphical aid to the interpretation and validation of cluster analysis. <i>Journal of Computational and Applied Mathematics</i>, 20, 53–65. — 실루엣, 05장',
 'Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. <i>Journal of Machine Learning Research</i>, 12, 2825–2830. — A장 코드',
 'scikit-learn 문서. <a href="https://scikit-learn.org/stable/modules/mixture.html" target="_blank" rel="noopener">Gaussian mixture models</a> · <a href="https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html" target="_blank" rel="noopener">GaussianMixture</a> — 인자, 시작점, 공분산 모양, 03·06장',
 'Feldman, N. H., Griffiths, T. L., &amp; Morgan, J. L. (2009). The influence of categories on perception: Explaining the perceptual magnet effect as optimal statistical inference. <i>Psychological Review</i>, 116(4), 752–782. — D장 범주 지각',
 'Hattori, D., Aso, Y., Swartz, K. J., Rubin, G. M., Abbott, L. F., &amp; Axel, R. (2017). Representations of novelty and familiarity in a mushroom body compartment. <i>Cell</i>, 169(5), 956–969. — D장 초파리 새로움 감지',
 'Rao, R. P. N., &amp; Ballard, D. H. (1999). Predictive coding in the visual cortex. <i>Nature Neuroscience</i>, 2(1), 79–87. — D장 생성 모델'];
var TIP={who:'머신러닝을 처음 배우는 대학생·직장인·제조 현장 실무자. 5-10 K-Means(가까운 중심에 배정)를 먼저 보면 좋다. 수식은 "종 모양 여럿의 가중합" 하나가 중심이다',
 goal:'① 딱 잘라 나누기(K-Means)와 확률로 나눠 갖기(GMM)의 차이를 01장 그림으로 설명한다 ② EM의 E·M 단계를 말로 설명하고, 시작점에 따라 결과가 달라지는 까닭(국소 최적)과 대책(n_init)을 안다 ③ 공분산 모양과 묶음 수를 정답 없이 BIC로 고르고, 정답으로 채점(ARI)하는 순서를 지킨다 ④ 고차원 확률의 과신을 알고, 업무 점수(대표 이름 + 가능도 낮은 글씨 다시 보기)로 라벨링 비용을 계산한다',
 time:'읽기 25분 · 실습(01·04·05장 그림) 25분 · 틀린 곳 찾기 10분 · Colab 15분',
 plans:[['강의형 50분','00장 지원서 1장을 인쇄해 1~3번 칸을 함께 읽기 8분 → 01장 ① 한 축 그림에서 종 1 → 2 → 4개로 늘리며 BIC가 어떻게 바뀌는지 보기 7분 → ② 2축 EM 재생을 한 번씩 누르며 E·M을 말로 설명하기 10분 → 04장 시작점 다섯 개와 공분산 모양 비교 10분 → 05장 업무 점수 슬라이더로 "몇 장을 사람이 보면 몇 %가 되나" 토론 7분 → 면접 결과를 먼저 맞히고 발표, "BIC가 고른 k = 8" 다시 보기 8분.'],['실무 OJT 90분','03장 프롬프트로 각자 AI에게 코드를 받는다 20분 → 체크리스트로 서로의 코드를 바꿔 검사(특히 fit(Z, y), accuracy_score, BIC max) 20분 → Colab에서 실행하고 A장 결과와 비교 20분 → B장 ③ 프롬프트로 "가능도가 낮은 글씨 = 다시 볼 글씨"를 만들고, 내 현장의 불량 검출에 쓸 수 있을지 토론 30분.'],['1:1 튜터링','01장 ② 그림에서 색이 섞인 점 하나를 골라 "왜 이 글씨는 반반인가"를 설명하게 한다. 이어 05장 2축·30축 확률 막대를 보며 "확률 0.99가 99% 맞는다는 뜻인가?"를 묻고, 틀린 글씨 수와 비교해 과신을 스스로 찾게 한다.']],
 myths:[['GMM의 확률은 정답일 확률이다','모델이 맞다고 칠 때의 계산값이다. 30축에서는 거의 모든 글씨가 0.99 이상이지만 8%쯤 틀린다(과신)'],['BIC는 클수록 좋다','작을수록 좋다. score()(평균 로그 가능도)가 클수록 좋은 것과 헷갈리기 쉽다'],['묶음 수는 BIC가 정답을 알려 준다','BIC는 k = 8을 골랐다. 숫자가 열 종류라는 것은 사람의 업무 지식이다. BIC는 출발점이다'],['차원이 많을수록 GMM이 정확하다','64칸 그대로는 모수가 2만을 넘고 ARI가 0.611로 떨어진다. 줄인 뒤에 쓴다(PCA 30축 0.836)'],['같은 코드면 늘 같은 결과가 나온다','시작점(random_state)에 따라 ARI가 0.630~0.836으로 흔들린다. 고정하고 n_init을 늘린다'],['표준화는 언제나 해야 한다','모든 칸이 같은 단위인 손글씨에서는 거의 빈 가장자리를 부풀려 오히려 해롭다(0.836 → 0.602)']],
 tasks:[['과제 1 · 손으로 EM 한 번','1차원 점 다섯 개(1, 2, 3, 7, 8)와 두 종(평균 2와 7, 표준편차 1, 비중 0.5씩)으로 점마다 소속 확률을 구하고(E), 그 확률로 가중한 새 평균 두 개를 구한다(M). 01장 그림과 같은 순서다.'],['과제 2 · 틀린 곳 찾기 만들기','AI에게 일부러 실수가 섞인 GMM 코드를 받고(B장 ②를 거꾸로 활용), 짝에게 찾게 한다. "오류 없이 조용히 틀리는" 실수(fit(Z, y), accuracy_score)를 하나 이상 넣는다.'],['과제 3 · 축 수 × 공분산 모양 지도','A장 3·6번 칸을 고쳐 PCA 축 수(10, 20, 30, 40)와 공분산 모양(full, diag)마다 BIC와 ARI를 표로 만든다. BIC는 축 수가 다르면 서로 비교할 수 없다는 점에 주의하고, 정답 없이 고를 수 있는 것과 없는 것을 구분해 적는다.'],['과제 4 · 라벨링 비용 계산','05장 업무 점수처럼 "대표 몇 장 + 가능도 하위 몇 %"를 사람이 볼 때 정답률과 사람 작업량을 표로 만들고, 목표 정답률 95%를 가장 적은 작업량으로 내는 조합을 찾는다.']],
 ai:'학생이 AI에게 코드를 받는 것은 막지 말고 과정에 넣는다. GMM은 오류 없이 조용히 틀리는 실수(정답을 fit에 넣기, 묶음 번호로 정확도 재기, BIC를 크게 고르기)가 많아 검증 연습에 특히 좋다. ① 프롬프트를 제출하게 하고 ② 체크리스트 아홉 항목을 직접 표시하게 하고 ③ 실행 결과를 무작위·합격선과 비교한 문장을 쓰게 한다. AI 답의 함수·인자는 설치된 버전의 공식 문서로 확인하게 하고, 과제에 AI 도움을 받았으면 사용 범위를 밝히게 한다.'};
var CODE={title:'GMM 면접 — 겹친 종 모양부터 합격 판정까지',sub:'numpy, scipy, scikit-learn · 인터넷 없이 실행 · CPU 10초 안팎',file:'gmm_interview',
 desc:'이 편의 면접을 그대로 옮겼다. 손글씨 1,797장을 읽고 정답을 채점 봉투에 넣고(1번 칸), 매뉴얼 씨의 합격선을 세우고(2번), PCA로 30축까지 줄이고(3번), EM을 직접 짜서 scikit-learn과 맞춰 보고(4번), 소속 확률(5번), 공분산 모양(6번), BIC로 묶음 수 고르기와 시작점 운(7번), 채점 봉투 열기(8번), 업무 점수(9번), 새 글씨 그리기(10번), 군집팀 비교(11번), 판정(12번)까지 간다. 정답은 8번 칸부터만 쓴다. 노트북으로 저장하면 칸마다 셀이 나뉜다.',
 cap:'실행 결과 (scikit-learn 1.9, CPU). 시간(ms)은 컴퓨터마다 다르다. 브라우저 계산과 같은 숫자(ARI 0.836, BIC 300,818, EM 16번, 대표 10장 92.0%)가 나온다. 브라우저는 scikit-learn과 같은 시작점(GaussianMixture(random_state=0)이 쓰는 K-Means 묶음)에서 EM을 직접 돌리기 때문이다. 4번 칸의 직접 짠 EM은 멈추는 판정을 한 번 일찍 해 15번으로 적히지만 가능도는 같다.'};
