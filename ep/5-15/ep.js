var BOARD_LINE='5-NN 18.1%'; /* 차원축소팀 합격선: 매뉴얼 씨 2D 지도(잉크 양·위쪽 비율)의 5-NN 정확도 */
/* ══ 시리즈 채용 결산표 (마무리의 15명 결산) — 이 배열만 고치면 표와 합계가 함께 바뀐다 ══════════
   n 편 · who 지원자 · hq '공고'|'지도'|'비지도' · team 팀 · model 기준 모델·조건 · score 핵심 숫자(글자 그대로 표에 나온다)
   res 'pass'(합격) | 'hold'(보류) | 'fail'(불합격) · est 1이면 "예상"(그 편의 면접 결과가 확정되기 전 작업 지시서 숫자로 적은 값). 확정되면 0으로
   note 덧붙임(없으면 '') */
var FINAL=[
 {n:'5-0', who:'매뉴얼 씨 (룰 베이스)', hq:'공고', team:'기존 직원', model:'작업 표준서 · 경험식 · 정리 규칙 · 손으로 고른 두 숫자', score:'분류 76.7% · 회귀 R² 0.735 · 군집 ARI 0.108 · 지도 5-NN 18.1%', res:'hold', est:0, note:'모든 팀의 합격선이 되었다'},
 {n:'5-1', who:'KNN', hq:'지도', team:'입문', model:'k = 5, 표준화', score:'분류 85.7% · 회귀 R² 0.767', res:'pass', est:0, note:''},
 {n:'5-2', who:'선형 회귀', hq:'지도', team:'회귀', model:'특징 8개', score:'R² 0.567', res:'hold', est:0, note:'좋은 특징을 넣으면 0.842'},
 {n:'5-3', who:'다항 회귀와 규제', hq:'지도', team:'회귀', model:'3차 다항', score:'R² 0.849', res:'pass', est:0, note:''},
 {n:'5-4', who:'로지스틱 회귀', hq:'지도', team:'분류', model:'특징 8개, 표준화', score:'86.4%', res:'pass', est:0, note:''},
 {n:'5-5', who:'나이브 베이즈', hq:'지도', team:'분류', model:'GaussianNB, 특징 8개', score:'70.9%', res:'hold', est:0, note:'좋은 특징을 넣으면 81.0%'},
 {n:'5-6', who:'결정 트리', hq:'지도', team:'겸용', model:'끝까지 키운 나무', score:'92.6% · R² 0.891', res:'pass', est:0, note:''},
 {n:'5-7', who:'랜덤 포레스트', hq:'지도', team:'겸용', model:'나무 100그루', score:'93.4% · R² 0.927', res:'pass', est:0, note:''},
 {n:'5-8', who:'SVM', hq:'지도', team:'겸용', model:'SVC RBF C=10 · SVR C=100', score:'88.0% · R² 0.907', res:'pass', est:0, note:''},
 {n:'5-9', who:'GBM', hq:'지도', team:'겸용', model:'나무 300그루, η 0.1', score:'93.4% · R² 0.949', res:'pass', est:0, note:''},
 {n:'5-10', who:'K-Means', hq:'비지도', team:'군집', model:'k = 10, 원래 픽셀', score:'ARI 0.666 · 실루엣 0.183', res:'pass', est:0, note:''},
 {n:'5-11', who:'계층적 군집', hq:'비지도', team:'군집', model:'ward 연결, 10묶음', score:'ARI 0.794', res:'pass', est:0, note:'single 연결이면 0.000'},
 {n:'5-12', who:'DBSCAN', hq:'비지도', team:'군집', model:'원래 픽셀, min_samples 5, eps는 k-거리 무릎(25.16)', score:'ARI 0.001 · 96%가 한 묶음', res:'hold', est:0, note:'정답 없이 eps를 고르면 뭉친다. HDBSCAN은 ARI 0.483'},
 {n:'5-13', who:'GMM', hq:'비지도', team:'군집', model:'full 공분산, PCA 30차원 위', score:'ARI 0.836', res:'pass', est:0, note:''},
 {n:'5-14', who:'PCA', hq:'비지도', team:'차원축소', model:'2성분', score:'지도 5-NN 60.3% · 설명된 분산 28.5%', res:'pass', est:0, note:''},
 {n:'5-15', who:'t-SNE · UMAP', hq:'비지도', team:'차원축소', model:'perplexity 30 · n_neighbors 15', score:'지도 5-NN 97.6% · 97.6%', res:'pass', est:0, note:'이 편'}];
/* 팀별 합격선 (결산표 아래 줄) */
var FINAL_LINES=[['회귀팀 · 겸용팀(회귀)','R² 0.735 (매뉴얼 경험식)'],['분류팀 · 겸용팀(분류) · 입문팀','76.7% (매뉴얼 씨 작업 표준서)'],['군집팀','ARI 0.108 (매뉴얼 씨 정리 규칙 8묶음)'],['차원축소팀','지도 5-NN 18.1% (매뉴얼 씨 지도)']];

/* ══ 5-15 t-SNE · UMAP · 손글씨 숫자 실기 과제지 ══════════════════════════ */
(function(){const st=document.createElement('style');st.textContent=
 '.nbstrip{margin-top:8px}.nbstrip canvas{display:block;width:100%}'+
 '.maps4{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px}.maps4 figure{margin:0;min-width:0}.maps4 figcaption{font-size:13.5px;color:var(--muted);text-align:center;margin-top:4px}.maps4 figcaption b{color:var(--text);font-family:var(--mono);margin-left:4px}'+
 'table.pix{border-collapse:collapse;font-family:var(--mono);font-size:12.5px;margin:0 auto}table.pix td{width:30px;height:26px;text-align:center;border:1px solid var(--rule);padding:0}'+
 '.fsum{display:flex;flex-wrap:wrap;gap:8px}.fsum span{border:1px solid var(--rule);border-radius:999px;padding:3px 12px;font-size:14px;background:var(--panel2)}.fsum b{font-family:var(--mono)}'+
 '#finalTbl tr.hq td{background:var(--panel2);font-weight:700;color:var(--text)}#finalTbl tr.me td{background:var(--blue-soft)}#finalTbl .est{font-size:11.5px;border:1px dashed var(--amber);color:var(--amber);border-radius:6px;padding:0 5px;margin-left:4px;white-space:nowrap}';
 document.head.appendChild(st);})();

/* DR-ENGINE-START */
/* ── 차원축소 채점 엔진 (이 편 전용, 브라우저·node 공용) ──
   · knnCV: scikit-learn cross_val_score(KNeighborsClassifier(5), Z, y, cv=5)와 같은 규칙
     (StratifiedKFold 섞지 않음: 숫자를 정렬해 다섯 몫에 돌려 나눈 개수만큼, 각 숫자의 글씨를 번호 순으로 0,0,…,1,1,… 몫에 배정. 표 다수결이 같으면 작은 숫자)
   · trust: sklearn.manifold.trustworthiness와 같은 식. 지도 위 이웃 k장 중 64칸 이웃 k장에 없는 글씨의 64칸 순위로 벌점
   · kmeans: k-means++ 초기화 n번, 로이드 반복, 관성이 가장 작은 것
   · tsne: 정확한 t-SNE (scikit-learn method="exact"와 같은 규칙: 이웃 확률은 제곱 거리의 가우시안을 perplexity에 맞춤, 조기 과장 12배 250걸음, 모멘텀 0.5→0.8, 이득(gains), 학습률 auto, PCA 시작 ×1e-4) */
var DRX=(function(){
  function d2(a,b){let s=0;for(let j=0;j<a.length;j++){const t=a[j]-b[j];s+=t*t;}return s;}
  /* 64칸 거리 행렬 (제곱 거리, Float32) */
  function distMatrix(X){const n=X.length,D=new Float32Array(n*n);for(let i=0;i<n;i++)for(let j=i+1;j<n;j++){const v=d2(X[i],X[j]);D[i*n+j]=v;D[j*n+i]=v;}return D;}
  /* 가장 가까운 k개 (자기 자신 뺌). dist(i,j) 함수. 같은 거리면 번호가 작은 쪽 */
  function knnAll(n,dist,k){const out=new Int32Array(n*k);for(let i=0;i<n;i++){const bd=new Float64Array(k).fill(Infinity),bi=new Int32Array(k).fill(-1);
      for(let j=0;j<n;j++){if(j===i)continue;const v=dist(i,j);if(v<bd[k-1]){let p=k-1;while(p>0&&bd[p-1]>v){bd[p]=bd[p-1];bi[p]=bi[p-1];p--;}bd[p]=v;bi[p]=j;}}
      for(let q=0;q<k;q++)out[i*k+q]=bi[q];}return out;}
  function folds(y,K){const n=y.length,first={},order=[];y.forEach(v=>{if(!(v in first)){first[v]=order.length;order.push(v);}});const enc=y.map(v=>first[v]),C=order.length;
    const srt=enc.slice().sort((a,b)=>a-b),alloc=[];for(let f=0;f<K;f++){const c=new Array(C).fill(0);for(let i=f;i<n;i+=K)c[srt[i]]++;alloc.push(c);}
    const tf=new Int8Array(n);for(let c=0;c<C;c++){const seq=[];for(let f=0;f<K;f++)for(let t=0;t<alloc[f][c];t++)seq.push(f);let q=0;for(let i=0;i<n;i++)if(enc[i]===c)tf[i]=seq[q++];}return tf;}
  /* dist(i,j)로 5겹 교차 검증 k-NN 정확도 (폴드 정확도의 평균) */
  function knnCV(n,dist,y,k,K){k=k||5;K=K||5;const tf=folds(y,K),acc=[];
    for(let f=0;f<K;f++){let ok=0,m=0;for(let i=0;i<n;i++){if(tf[i]!==f)continue;m++;const bd=new Float64Array(k).fill(Infinity),bi=new Int32Array(k).fill(-1);
        for(let j=0;j<n;j++){if(tf[j]===f)continue;const v=dist(i,j);if(v<bd[k-1]){let p=k-1;while(p>0&&bd[p-1]>v){bd[p]=bd[p-1];bi[p]=bi[p-1];p--;}bd[p]=v;bi[p]=j;}}
        const cnt=new Array(10).fill(0);for(let q=0;q<k;q++)cnt[y[bi[q]]]++;let b=0;for(let c=1;c<10;c++)if(cnt[c]>cnt[b])b=c;if(b===y[i])ok++;}
      acc.push(ok/m);}
    return acc.reduce((a,b)=>a+b,0)/K;}
  /* trustworthiness: D 원래 공간 제곱 거리 행렬, nnMap 지도 위 k-NN */
  function trust(n,D,nnMap,nnOrig,k){let t=0;for(let i=0;i<n;i++){const row=i*n;for(let q=0;q<k;q++){const j=nnMap[i*k+q];let inO=false;for(let r=0;r<k;r++)if(nnOrig[i*k+r]===j){inO=true;break;}if(inO)continue;
        const dj=D[row+j];let rank=1;for(let m=0;m<n;m++){if(m===i||m===j)continue;const v=D[row+m];if(v<dj||(v===dj&&m<j))rank++;}t+=rank-k;}}
    return 1-t*(2/(n*k*(2*n-3*k-1)));}
  function overlap(n,nnA,nnB,k){let s=0;for(let i=0;i<n;i++)for(let q=0;q<k;q++){const j=nnA[i*k+q];for(let r=0;r<k;r++)if(nnB[i*k+r]===j){s++;break;}}return s/(n*k);}
  function kmeans(Z,k,o){o=o||{};const n=Z.length,d=Z[0].length,rng=RNG(o.seed||7),nInit=o.nInit||10;let best=null;
    for(let r=0;r<nInit;r++){const C=[Z[Math.floor(rng()*n)].slice()],dm=new Float64Array(n).fill(Infinity);
      while(C.length<k){const c=C[C.length-1];let S=0;for(let i=0;i<n;i++){const v=d2(Z[i],c);if(v<dm[i])dm[i]=v;S+=dm[i];}let u=rng()*S,pick=n-1;for(let i=0;i<n;i++){u-=dm[i];if(u<=0){pick=i;break;}}C.push(Z[pick].slice());}
      const lab=new Int32Array(n).fill(-1);let inertia=0;
      for(let it=0;it<300;it++){let ch=0;inertia=0;for(let i=0;i<n;i++){let b=0,bv=Infinity;for(let c=0;c<k;c++){const v=d2(Z[i],C[c]);if(v<bv){bv=v;b=c;}}inertia+=bv;if(lab[i]!==b){lab[i]=b;ch++;}}
        const S=[],N=new Array(k).fill(0);for(let c=0;c<k;c++)S.push(new Float64Array(d));for(let i=0;i<n;i++){N[lab[i]]++;for(let j=0;j<d;j++)S[lab[i]][j]+=Z[i][j];}
        for(let c=0;c<k;c++)if(N[c])for(let j=0;j<d;j++)C[c][j]=S[c][j]/N[c];if(!ch)break;}
      if(!best||inertia<best.inertia)best={labels:Array.from(lab),centers:C.map(c=>Array.from(c)),inertia};}
    return best;}
  function ari(a,b){const n=a.length,M={},A={},B={};for(let i=0;i<n;i++){const k=a[i]+'|'+b[i];M[k]=(M[k]||0)+1;A[a[i]]=(A[a[i]]||0)+1;B[b[i]]=(B[b[i]]||0)+1;}
    const c2=x=>x*(x-1)/2;let s=0,sa=0,sb=0;for(const k in M)s+=c2(M[k]);for(const k in A)sa+=c2(A[k]);for(const k in B)sb+=c2(B[k]);const e=sa*sb/c2(n),mx=(sa+sb)/2;return mx===e?1:(s-e)/(mx-e);}
  /* 대표 한 장 이름 퍼뜨리기: 묶음 중심에 가장 가까운 글씨의 정답을 묶음 전체에 */
  function propagate(Z,km,y){const reps=km.centers.map(c=>{let b=0,bv=Infinity;Z.forEach((z,i)=>{const v=d2(z,c);if(v<bv){bv=v;b=i;}});return b;});let ok=0;for(let i=0;i<Z.length;i++)if(y[reps[km.labels[i]]]===y[i])ok++;return {acc:ok/Z.length,reps};}
  /* PCA (공분산 행렬 거듭제곱법). 부호는 scikit-learn처럼 성분에서 절댓값이 가장 큰 칸이 양수가 되게 */
  function pca(X,nc){const n=X.length,d=X[0].length,mu=new Float64Array(d);X.forEach(r=>{for(let j=0;j<d;j++)mu[j]+=r[j]/n;});
    const Cv=[];for(let a=0;a<d;a++)Cv.push(new Float64Array(d));X.forEach(r=>{for(let a=0;a<d;a++){const xa=r[a]-mu[a];if(!xa)continue;for(let b=a;b<d;b++)Cv[a][b]+=xa*(r[b]-mu[b]);}});
    for(let a=0;a<d;a++)for(let b=a;b<d;b++){Cv[a][b]/=(n-1);Cv[b][a]=Cv[a][b];}
    const comps=[],vals=[],rng=RNG(3);let tot=0;for(let a=0;a<d;a++)tot+=Cv[a][a];
    for(let c=0;c<nc;c++){let v=new Float64Array(d).map(()=>rng()-0.5),lam=0;
      for(let it=0;it<500;it++){const w=new Float64Array(d);for(let a=0;a<d;a++){let s=0;for(let b=0;b<d;b++)s+=Cv[a][b]*v[b];w[a]=s;}comps.forEach(u=>{let p=0;for(let a=0;a<d;a++)p+=w[a]*u[a];for(let a=0;a<d;a++)w[a]-=p*u[a];});
        let nr=0;for(let a=0;a<d;a++)nr+=w[a]*w[a];nr=Math.sqrt(nr);lam=nr;for(let a=0;a<d;a++)w[a]/=nr;v=w;}
      let mi=0;for(let a=1;a<d;a++)if(Math.abs(v[a])>Math.abs(v[mi]))mi=a;if(v[mi]<0)for(let a=0;a<d;a++)v[a]=-v[a];comps.push(v);vals.push(lam);}
    const Z=X.map(r=>comps.map(u=>{let s=0;for(let a=0;a<d;a++)s+=(r[a]-mu[a])*u[a];return s;}));return {Z,comps,vals,ratio:vals.map(v=>v/tot),mu};}
  /* 정확한 t-SNE. step()을 부를 때마다 한 걸음. 스냅숏은 바깥에서 */
  function tsne(X,o){o=o||{};const n=X.length,perp=o.perplexity||30,EX=12,lr=Math.max(n/EX/4,50),D=new Float64Array(n*n);
    for(let i=0;i<n;i++)for(let j=i+1;j<n;j++){const v=d2(X[i],X[j]);D[i*n+j]=v;D[j*n+i]=v;}
    const P=new Float64Array(n*n),H0=Math.log(perp);
    for(let i=0;i<n;i++){let beta=1,bmin=-Infinity,bmax=Infinity,row=i*n;
      for(let s=0;s<100;s++){let sp=0,sdp=0;for(let j=0;j<n;j++){if(j===i){P[row+j]=0;continue;}const p=Math.exp(-D[row+j]*beta);P[row+j]=p;sp+=p;}
        if(sp===0)sp=1e-8;for(let j=0;j<n;j++){P[row+j]/=sp;sdp+=D[row+j]*P[row+j];}const H=Math.log(sp)+beta*sdp,df=H-H0;if(Math.abs(df)<=1e-5)break;
        if(df>0){bmin=beta;beta=bmax===Infinity?beta*2:(beta+bmax)/2;}else{bmax=beta;beta=bmin===-Infinity?beta/2:(beta+bmin)/2;}}}
    let S=0;for(let i=0;i<n;i++)for(let j=i+1;j<n;j++){const v=P[i*n+j]+P[j*n+i];P[i*n+j]=v;P[j*n+i]=v;S+=2*v;}
    for(let i=0;i<n*n;i++)P[i]=Math.max(P[i]/S,2.220446049250313e-16);
    const Y0=pca(X,2).Z;let m=0,sd=0;Y0.forEach(r=>m+=r[0]/n);Y0.forEach(r=>sd+=(r[0]-m)*(r[0]-m)/n);sd=Math.sqrt(sd)||1;
    const Y=new Float64Array(2*n);for(let i=0;i<n;i++){Y[2*i]=Y0[i][0]/sd*1e-4;Y[2*i+1]=Y0[i][1]/sd*1e-4;}
    let upd=new Float64Array(2*n),gain=new Float64Array(2*n).fill(1),it=0;const G=new Float64Array(2*n),NUM=new Float64Array(n*n);
    function kl(){let sq=0;for(let i=0;i<n;i++)for(let j=i+1;j<n;j++){const dx=Y[2*i]-Y[2*j],dy=Y[2*i+1]-Y[2*j+1];sq+=2/(1+dx*dx+dy*dy);}
      let s=0;for(let i=0;i<n;i++)for(let j=i+1;j<n;j++){const dx=Y[2*i]-Y[2*j],dy=Y[2*i+1]-Y[2*j+1],q=Math.max(1/(1+dx*dx+dy*dy)/sq,2.220446049250313e-16),p=P[i*n+j];s+=2*p*Math.log(p/q);}return s;}
    function step(){if(it===250){upd=new Float64Array(2*n);gain=new Float64Array(2*n).fill(1);}const ex=it<250?EX:1,mom=it<250?0.5:0.8;let sq=0;
      for(let i=0;i<n;i++)for(let j=i+1;j<n;j++){const dx=Y[2*i]-Y[2*j],dy=Y[2*i+1]-Y[2*j+1],v=1/(1+dx*dx+dy*dy);NUM[i*n+j]=v;sq+=2*v;}
      G.fill(0);for(let i=0;i<n;i++)for(let j=i+1;j<n;j++){const v=NUM[i*n+j],w=(ex*P[i*n+j]-v/sq)*v*4,dx=Y[2*i]-Y[2*j],dy=Y[2*i+1]-Y[2*j+1];G[2*i]+=w*dx;G[2*i+1]+=w*dy;G[2*j]-=w*dx;G[2*j+1]-=w*dy;}
      for(let q=0;q<2*n;q++){gain[q]=(upd[q]*G[q]<0)?gain[q]+0.2:gain[q]*0.8;if(gain[q]<0.01)gain[q]=0.01;upd[q]=mom*upd[q]-lr*gain[q]*G[q];Y[q]+=upd[q];}it++;}
    return {Y,step,kl,get it(){return it;},n,lr};}
  /* 숫자별 무리의 중심 · 평균 반지름 · 중심 거리 순위 상관(스피어만) */
  function geometry(Z,y){const d=Z[0].length,c=[],r=[];for(let k=0;k<10;k++){const m=new Float64Array(d);let cnt=0;Z.forEach((z,i)=>{if(y[i]===k){cnt++;for(let j=0;j<d;j++)m[j]+=z[j];}});for(let j=0;j<d;j++)m[j]/=cnt;c.push(Array.from(m));
      let s=0;Z.forEach((z,i)=>{if(y[i]===k)s+=Math.sqrt(d2(z,m));});r.push(s/cnt);}
    const pd=[],pairs=[];for(let a=0;a<10;a++)for(let b=a+1;b<10;b++){pd.push(Math.sqrt(d2(c[a],c[b])));pairs.push([a,b]);}return {c,r,pd,pairs};}
  function rank(a){const o=a.map((v,i)=>[v,i]).sort((p,q)=>p[0]-q[0]),r=new Array(a.length);for(let i=0;i<o.length;){let j=i;while(j+1<o.length&&o[j+1][0]===o[i][0])j++;for(let k=i;k<=j;k++)r[o[k][1]]=(i+j)/2+1;i=j+1;}return r;}
  function spearman(a,b){const ra=rank(a),rb=rank(b),n=a.length,ma=ra.reduce((s,v)=>s+v,0)/n,mb=rb.reduce((s,v)=>s+v,0)/n;let num=0,da=0,db=0;for(let i=0;i<n;i++){num+=(ra[i]-ma)*(rb[i]-mb);da+=(ra[i]-ma)**2;db+=(rb[i]-mb)**2;}return num/Math.sqrt(da*db);}
  return {d2,distMatrix,knnAll,folds,knnCV,trust,overlap,kmeans,ari,propagate,pca,tsne,geometry,spearman};})();
/* DR-ENGINE-END */

/* ══ 공용 준비 ══════════════════════════════════════════════════════ */
const DR=digitRows(),NN=DR.length,PXS=DR.map(r=>r.px);
const SEALED=DR.map(r=>r.y); /* 채점 봉투: 숨긴 정답 0~9. 지도를 그리는 데는 쓰지 않고, 채점과 05장 이후의 색칠에만 쓴다 */
const decodeEmb=(a,s)=>{const Z=[];for(let i=0;i<a.length;i+=2)Z.push([a[i]/s,a[i+1]/s]);return Z;};
const MAPNAME={man:'매뉴얼 씨 (잉크·위쪽)',pca:'PCA 2D (5-14)',ts30:'t-SNE (perplexity 30)',ts5:'t-SNE (perplexity 5)',ts50:'t-SNE (perplexity 50)',ts100:'t-SNE (perplexity 100)',
  um15:'UMAP (n_neighbors 15, min_dist 0.1)',um5:'UMAP (n_neighbors 5)',um50:'UMAP (n_neighbors 50)',um200:'UMAP (n_neighbors 200)',umd0:'UMAP (min_dist 0.0)',umd99:'UMAP (min_dist 0.99)',x64:'원래 64칸 (참고)'};
let PCA_RES=null;function pcaRes(){return PCA_RES||(PCA_RES=DRX.pca(PXS,2));}
const MAPS={};
function getMap(k){if(MAPS[k])return MAPS[k];let Z;
  if(k==='man')Z=PXS.map(p=>{const f=digitFeat(p);return [f.ink,f.top];});
  else if(k==='pca')Z=pcaRes().Z;
  else Z=decodeEmb(EMB[k],k.slice(0,2)==='ts'?EMB_SCALE.ts:EMB_SCALE.um);
  return (MAPS[k]=Z);}
const mapDist=Z=>(i,j)=>DRX.d2(Z[i],Z[j]);
/* 64칸 거리 행렬: 처음 필요할 때 조금씩 나눠 계산한다 */
let D64=null,NN64=null,D64P=null;
function ensureD64(){if(D64P)return D64P;D64P=(async()=>{const n=NN,D=new Float32Array(n*n),X=PXS;
    for(let i=0;i<n;i++){const a=X[i];for(let j=i+1;j<n;j++){const b=X[j];let s=0;for(let q=0;q<64;q++){const t=a[q]-b[q];s+=t*t;}D[i*n+j]=s;D[j*n+i]=s;}if(i%120===119)await wait(0);}
    D64=D;await wait(0);NN64=DRX.knnAll(n,(i,j)=>D[i*n+j],5);return D;})();return D64P;}
/* 지도별 점수 (한 번 계산하면 적어 둔다) */
const ST={};function stOf(k){return ST[k]||(ST[k]={});}
function mapNN(k){const s=stOf(k);if(!s.nn)s.nn=DRX.knnAll(NN,mapDist(getMap(k)),5);return s.nn;}
function mapCV(k){const s=stOf(k);if(s.cv==null)s.cv=DRX.knnCV(NN,mapDist(getMap(k)),SEALED);return s.cv;}   /* [채점] */
async function x64CV(){const s=stOf('x64');if(s.cv==null){const D=await ensureD64();s.cv=DRX.knnCV(NN,(i,j)=>D[i*NN+j],SEALED);}return s.cv;}
async function mapTrust(k){const s=stOf(k);if(s.trust==null){await ensureD64();const nn=mapNN(k);s.trust=DRX.trust(NN,D64,nn,NN64,5);s.ov=DRX.overlap(NN,nn,NN64,5);}return s.trust;}
const KMC={};function kmOf(k,K){const key=k+'|'+K;if(!KMC[key]){const Z=k==='x64'?PXS:getMap(k),km=DRX.kmeans(Z,K,{seed:7,nInit:10}),pr=DRX.propagate(Z,km,SEALED);KMC[key]={acc:pr.acc,reps:pr.reps,ari:DRX.ari(SEALED,km.labels),labels:km.labels};}return KMC[key];}
const GEO={};function geoOf(k){return GEO[k]||(GEO[k]=DRX.geometry(k==='x64'?PXS:getMap(k),SEALED));}
/* 기준선 [채점]: 늘 가장 흔한 숫자만 답하기 (DummyClassifier와 같게 5겹마다 훈련 몫에서 가장 흔한 숫자) */
const BASE_DR=(function(){const tf=DRX.folds(SEALED,5);let s=0;for(let f=0;f<5;f++){const c=new Array(10).fill(0);let m=0,ok=0;SEALED.forEach((v,i)=>{if(tf[i]!==f)c[v]++;});let b=0;for(let k=1;k<10;k++)if(c[k]>c[b])b=k;SEALED.forEach((v,i)=>{if(tf[i]===f){m++;if(v===b)ok++;}});s+=ok/m;}return s/5;})();
const pc1=v=>pct(v,1);
/* 글씨 그림 조각 (8×8 → 작은 캔버스, 색마다 따로 적어 둔다) */
const GLY={};
function hexRgb(c){if(!c||c[0]!=='#')return [30,40,60];const h=c.length===4?c.replace(/#(.)(.)(.)/,'#$1$1$2$2$3$3'):c;return [parseInt(h.slice(1,3),16),parseInt(h.slice(3,5),16),parseInt(h.slice(5,7),16)];}
function glyph(i,color){const set=GLY[color]||(GLY[color]=[]);if(set[i])return set[i];const c=document.createElement('canvas');c.width=c.height=8;const g=c.getContext('2d'),im=g.createImageData(8,8),rgb=hexRgb(color),p=PXS[i];
  for(let j=0;j<64;j++){im.data[4*j]=rgb[0];im.data[4*j+1]=rgb[1];im.data[4*j+2]=rgb[2];im.data[4*j+3]=Math.round(p[j]/16*255);}g.putImageData(im,0,0);return (set[i]=c);}
/* 지도 그리기. o: {mode:'glyph'|'dot', color(i), eq(같은 눈금), idx(그릴 글씨), size, alpha, axes:[가로 이름, 세로 이름], top} */
function drawMap(cv,H,Z,o){o=o||{};const W=fit(cv,H),ctx=cv.getContext('2d'),C=cols(),idx=o.idx||null,pl=o.axes?44:8,pb=o.axes?30:8,pt=8+(o.top||0),pr=8;
  let x0=Infinity,x1=-Infinity,y0=Infinity,y1=-Infinity;const each=f=>{if(idx)idx.forEach(f);else for(let i=0;i<Z.length;i++)f(i);};
  each(i=>{const z=Z[i];if(z[0]<x0)x0=z[0];if(z[0]>x1)x1=z[0];if(z[1]<y0)y0=z[1];if(z[1]>y1)y1=z[1];});
  const gw=W-pl-pr,gh=H-pt-pb,m=o.mode==='glyph'?(o.size||9)/2+2:4;let sx,sy;
  if(o.eq!==false){const s=Math.min((gw-2*m)/((x1-x0)||1),(gh-2*m)/((y1-y0)||1)),ox=pl+(gw-(x1-x0)*s)/2,oy=pt+(gh-(y1-y0)*s)/2;sx=x=>ox+(x-x0)*s;sy=y=>oy+(y1-y)*s;}
  else{sx=x=>pl+m+(x-x0)/((x1-x0)||1)*(gw-2*m);sy=y=>pt+m+(y1-y)/((y1-y0)||1)*(gh-2*m);}
  ctx.clearRect(0,0,W,H);ctx.strokeStyle=C.rule;ctx.lineWidth=1;ctx.strokeRect(pl+.5,pt+.5,gw-1,gh-1);
  if(o.axes){ctx.fillStyle=C.muted;ctx.font=font(11.5);ctx.textAlign='center';for(let k=0;k<=4;k++){const v=x0+(x1-x0)*k/4;ctx.fillText(o.fmtX?o.fmtX(v):v.toFixed(0),sx(v),H-pb+14);}ctx.fillText(o.axes[0],pl+gw/2,H-3);
    ctx.textAlign='right';for(let k=0;k<=4;k++){const v=y0+(y1-y0)*k/4;ctx.fillText(o.fmtY?o.fmtY(v):v.toFixed(2),pl-4,sy(v)+4);}ctx.save();ctx.translate(11,pt+gh/2);ctx.rotate(-Math.PI/2);ctx.textAlign='center';ctx.fillText(o.axes[1],0,0);ctx.restore();}
  const sz=o.size||9;ctx.imageSmoothingEnabled=false;
  each(i=>{const X=sx(Z[i][0]),Y=sy(Z[i][1]),col=o.color?o.color(i):C.text;if(o.mode==='glyph')ctx.drawImage(glyph(i,col),X-sz/2,Y-sz/2,sz,sz);else{ctx.fillStyle=rgba(col,o.alpha||.6);ctx.fillRect(X-1.6,Y-1.6,3.2,3.2);}});
  return {sx,sy,W,H,ctx,C};}
function respH(cv,r,lo,hi){return Math.round(clamp((cv.clientWidth||cv.parentElement.clientWidth||600)*r,lo,hi));}
function whenNear(id,fn,margin){const t=$(id);let done=false;function go(){if(done)return;done=true;setTimeout(fn,10);}
  if('IntersectionObserver' in window){const io=new IntersectionObserver(es=>{if(es.some(e=>e.isIntersecting)){go();io.disconnect();}},{rootMargin:(margin||600)+'px'});io.observe(t);}else go();
  const chk=()=>{const h=location.hash.slice(1),el=h&&$(h);if(el&&(el===t||t.contains(el)||t.compareDocumentPosition(el)&Node.DOCUMENT_POSITION_FOLLOWING))go();};window.addEventListener('hashchange',chk);if(location.hash)chk();}
function barRows(rows,max){return rows.map(r=>'<div class="row"><span style="text-align:left;font-family:var(--font)">'+r[0]+'</span><div class="tr"><i style="left:0;width:'+Math.max(0,Math.min(100,r[1]/max*100))+'%;background:'+(r[2]||'var(--blue)')+'"></i></div><span>'+r[3]+'</span></div>').join('');}

/* ══ 히어로: 지도 완성 파티 ══════════════════════════════════════════ */
(function(){const S=140,idx=[...Array(S).keys()],A=DRX.pca(PXS.slice(0,S),2).Z,B=decodeEmb(EMB.ts30,EMB_SCALE.ts).slice(0,S);
  const norm=Z=>{let x0=Infinity,x1=-Infinity,y0=Infinity,y1=-Infinity;Z.forEach(z=>{x0=Math.min(x0,z[0]);x1=Math.max(x1,z[0]);y0=Math.min(y0,z[1]);y1=Math.max(y1,z[1]);});return Z.map(z=>[(z[0]-x0)/((x1-x0)||1),(z[1]-y0)/((y1-y0)||1)]);};
  const PA=norm(A),PB=norm(B);
  heroRoom($('heroCv'),{sym:'tS',no:'APP-15',name:'t-SNE ·\nUMAP',role:'공동 지원자 · 차원축소',banner:'모델 채용 위원회 · 5-15 마지막 면접',doc:'front',asst:'board',fly:'clip',robo:'calc',loop:12,
    beats:[{at:0.4,who:'robo',mood:'chart',move:'sway',say:'지도 한 장 얼마?'},{at:2.4,who:'fly',mood:'point',move:'tilt',say:'거리도 믿어?'},{at:4.4,who:'doc',mood:'magic',move:'pop',say:'이웃만 지켜요'},
           {at:6.4,who:'asst',mood:'spark',move:'nod',say:'열 무리가 보여요'},{at:8.4,who:'robo',mood:'happy',move:'jump',say:'열다섯 명 끝!'},{at:8.5,who:'fly',mood:'arms',move:'jump'},{at:8.6,who:'doc',mood:'board',move:'jump'},{at:8.7,who:'asst',mood:'spark',move:'jump'},
           {at:9.6,who:'robo',mood:'wow',move:'jump'},{at:9.7,who:'fly',mood:'arms',move:'jump'},{at:9.8,who:'doc',mood:'magic',move:'jump'},{at:9.9,who:'asst',mood:'spark',move:'jump'}],
    motif:function(ctx,t,W,H,C){const tt=REDUCE?6:(t<1.4?0:(t-1.4)%12),f=tt<1.5?0:tt<5?ease((tt-1.5)/3.5):tt<11?1:1-ease((tt-11)/1),bx=262,by=88,bw=112,bh=112;
      ctx.fillStyle=rgba(C.panel,.6);rr(ctx,bx-6,by-6,bw+12,bh+12,8);ctx.fill();ctx.strokeStyle=rgba(C.c_app,.6);ctx.lineWidth=1.5;ctx.stroke();ctx.imageSmoothingEnabled=false;
      idx.forEach(i=>{const x=bx+(PA[i][0]*(1-f)+PB[i][0]*f)*bw,y=by+(1-(PA[i][1]*(1-f)+PB[i][1]*f))*bh;ctx.drawImage(glyph(i,C.text),x-4.5,y-4.5,9,9);});
      ctx.fillStyle=C.muted;ctx.font=font(12,500,1);ctx.textAlign='center';ctx.fillText(f<0.5?'64칸 · 그림자(PCA)':'이웃을 지킨 지도',bx+bw/2,by+bh+22);}});})();

/* ══ 01 장면 ══ */
comic('comicIntro',[
 {bg:'party',prop:'64칸 · 가까운 다섯 장',cast:[{who:'asst',mood:'board',x:30,move:'nod'},{who:'doc',mood:'board',x:72,move:'tilt',h:100}],say:{who:'doc',txt:'64칸에서 가까운 다섯 장을 먼저 적어 둡니다. 지도에서도 곁에 둘 거예요.'},cap:'① 이웃 지키기 — 가까운 글씨부터 적는다'},
 {bg:'party',prop:'끌림 ×12 → 1,000걸음',cast:[{who:'asst',mood:'firm',x:28,move:'jump'},{who:'robo',mood:'chart',x:72,move:'sway',h:104}],say:{who:'asst',txt:'이웃은 당기고 나머지는 밀었더니, 같은 모양끼리 섬이 됐어요!'},cap:'② 펼치기 — 당기고 밀며 점을 하나씩 옮긴다'},
 {bg:'board',prop:'섬 크기 9.2배?',cast:[{who:'fly',mood:'point',x:30,move:'shake'},{who:'doc',mood:'scale',x:72,move:'tilt',h:100}],say:{who:'fly',txt:'섬이 크다고 글씨가 들쭉날쭉한 건 아니야. 섬 사이 거리도 마찬가지고.'},cap:'③ 거리는 믿지 말 것 — 무리는 믿고, 거리와 크기는 의심한다'}]);

/* ══ 01 자기소개: 지도 위에서 64칸 이웃 찾기 ══ */
(function(){const cv=$('nbCv'),st={k:'ts30',mode:'glyph',sel:0,T:null};const strip=document.createElement('canvas');$('nbStrip').appendChild(strip);strip.setAttribute('aria-hidden','true');
  function nb64(i){const a=PXS[i],d=[];for(let j=0;j<NN;j++){if(j===i)continue;let s=0;const b=PXS[j];for(let q=0;q<64;q++){const t=a[q]-b[q];s+=t*t;}d.push([s,j]);}d.sort((p,q)=>p[0]-q[0]||p[1]-q[1]);return d.slice(0,5).map(e=>e[1]);}
  function mapRank(i,Z){const d=[];for(let j=0;j<NN;j++)if(j!==i)d.push([DRX.d2(Z[i],Z[j]),j]);d.sort((p,q)=>p[0]-q[0]||p[1]-q[1]);const r={};d.forEach((e,q)=>r[e[1]]=q+1);return {rank:r,top:d.slice(0,5).map(e=>e[1])};}
  function draw(){const Z=getMap(st.k),H=respH(cv,0.62,300,520),man=st.k==='man';
    const T=drawMap(cv,H,Z,{mode:st.mode,size:cv.clientWidth<520?7:9,eq:!man,axes:man?['잉크 양 (64칸 밝기 합)','위쪽 잉크 비율']:null});st.T=T;
    const {ctx,C}=T,i=st.sel,nb=nb64(i),mr=mapRank(i,Z),X=T.sx(Z[i][0]),Y=T.sy(Z[i][1]);
    ctx.lineWidth=1.6;nb.forEach(j=>{const x=T.sx(Z[j][0]),y=T.sy(Z[j][1]);ctx.strokeStyle=rgba(C.blue,.75);ctx.beginPath();ctx.moveTo(X,Y);ctx.lineTo(x,y);ctx.stroke();circ(ctx,x,y,8,null,C.blue,2.4);});
    circ(ctx,X,Y,11,null,C.stamp,3);
    drawStrip(i,nb,mr);info(i,nb,mr);}
  function drawStrip(i,nb,mr){const W=fit(strip,92),ctx=strip.getContext('2d'),C=cols(),big=64,small=Math.min(46,(W-big-40)/5-10);ctx.clearRect(0,0,W,92);ctx.imageSmoothingEnabled=false;
    ctx.fillStyle=C.panel2;ctx.fillRect(0,10,big+8,big+8);ctx.drawImage(glyph(i,C.text),4,14,big,big);ctx.strokeStyle=C.stamp;ctx.lineWidth=2.5;ctx.strokeRect(1.5,11.5,big+5,big+5);
    ctx.fillStyle=C.muted;ctx.font=font(11.5,700);ctx.textAlign='left';ctx.fillText('고른 글씨',2,8);ctx.fillText('64칸에서 가장 가까운 다섯 장 → 지도 위 순위',big+24,8);
    nb.forEach((j,q)=>{const x=big+24+q*(small+10),ok=mr.rank[j]<=5;ctx.fillStyle=C.panel2;ctx.fillRect(x,16,small,small);ctx.drawImage(glyph(j,C.text),x+2,18,small-4,small-4);ctx.strokeStyle=C.blue;ctx.lineWidth=2;ctx.strokeRect(x+1,17,small-2,small-2);
      ctx.fillStyle=ok?C.ok:C.stamp;ctx.font=font(12,700,1);ctx.textAlign='center';ctx.fillText((ok?'✓ ':'')+mr.rank[j]+'위',x+small/2,16+small+16);});}
  function info(i,nb,mr){const kept=nb.filter(j=>mr.rank[j]<=5).length,worst=Math.max(...nb.map(j=>mr.rank[j])),s=stOf(st.k);
    let h='고른 글씨 <b>#'+(i+1)+'</b> · '+MAPNAME[st.k]+': 64칸 이웃 다섯 장 중 <b>'+kept+'장</b>이 지도 위에서도 가장 가까운 다섯 장 안에 있다(✓). 가장 멀리 떨어진 이웃은 지도 위 <b>'+worst+'위</b>. ';
    h+=st.k==='man'?'매뉴얼 씨 지도에서는 잉크 양이 비슷한 다른 숫자가 이웃 자리를 차지한다.':st.k==='pca'?'곧은 그림자라 64칸의 이웃이 지도 위에서 흩어지기 쉽다.':'지도가 휘고 늘어난 대신 이웃은 곁에 남았다.';
    h+='<br><span class="muted">'+(s.ov!=null?'이 지도 전체 1,797장의 이웃 보존율(64칸 5-NN이 지도 5-NN에 남은 몫): <b>'+pc1(s.ov)+'</b>':'이 지도 전체의 이웃 보존율 계산 중…')+' · 다른 점을 누르거나 "무작위 한 장"을 눌러 보자.</span>';
    $('nbInfo').innerHTML=h;}
  function ovAll(){mapTrust(st.k).then(()=>{if(st.T)info(st.sel,nb64(st.sel),mapRank(st.sel,getMap(st.k)));});}
  cv.addEventListener('click',e=>{if(!st.T)return;const r=cv.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top,Z=getMap(st.k);let b=-1,bv=24*24;
    for(let i=0;i<NN;i++){const dx=st.T.sx(Z[i][0])-mx,dy=st.T.sy(Z[i][1])-my,v=dx*dx+dy*dy;if(v<bv){bv=v;b=i;}}if(b>=0){st.sel=b;draw();}});
  seg('#nbMap button',v=>{st.k=v;draw();ovAll();});seg('#nbDraw button',v=>{st.mode=v;draw();});
  const rng=RNG(15);$('nbRand').addEventListener('click',()=>{st.sel=Math.floor(rng()*NN);draw();});
  st.sel=24;whenNear('m1',()=>{draw();ovAll();},400);onRedraw(()=>{if(st.T)draw();});})();

/* ══ 02 실기 과제지 ══ */
$('lineDr').textContent=pct(DR_LINE);$('heroLine').textContent='5-NN '+pct(DR_LINE);
(function(){const cv=$('gridCv'),COLS=20,ROWS=8;let sel=0,cell=0;
  function draw(){const w=cv.clientWidth||600;cell=Math.floor(w/COLS);const H=cell*ROWS,W=fit(cv,H),ctx=cv.getContext('2d'),C=cols();ctx.clearRect(0,0,W,H);ctx.imageSmoothingEnabled=false;
    for(let k=0;k<COLS*ROWS;k++){const x=(k%COLS)*cell,y=Math.floor(k/COLS)*cell;ctx.fillStyle=k%2?C.panel:C.panel2;ctx.fillRect(x,y,cell,cell);ctx.drawImage(glyph(k,C.text),x+1,y+1,cell-2,cell-2);}
    ctx.strokeStyle=C.stamp;ctx.lineWidth=2.5;ctx.strokeRect((sel%COLS)*cell+1.5,Math.floor(sel/COLS)*cell+1.5,cell-3,cell-3);table();}
  function table(){const p=PXS[sel],C=cols();let h='<tbody>';for(let r=0;r<8;r++){h+='<tr>';for(let c=0;c<8;c++){const v=p[r*8+c];h+='<td style="background:'+rgba(C.blue,v/16*.85)+';color:'+(v>9?'#fff':'var(--text)')+'">'+v+'</td>';}h+='</tr>';}
    $('pixTbl').innerHTML=h+'</tbody>';const f=digitFeat(p);$('pixInfo').innerHTML='글씨 <b>#'+(sel+1)+'</b>: 이 64개 숫자가 이 글씨의 전부다. 밝기 합(잉크 양) <b>'+f.ink+'</b>, 위쪽 절반 비율 <b>'+f.top.toFixed(2)+'</b>. 지원자는 64개를 다 보고, 매뉴얼 씨는 이 두 숫자만 본다. 정답은 적혀 있지 않다.';}
  cv.addEventListener('click',e=>{const r=cv.getBoundingClientRect(),c=Math.floor((e.clientX-r.left)/cell),w=Math.floor((e.clientY-r.top)/cell);if(c>=0&&c<COLS&&w>=0&&w<ROWS){sel=w*COLS+c;draw();}});
  draw();onRedraw(draw);})();
(function(){const cv=$('manCv');function draw(){drawMap(cv,respH(cv,0.5,260,360),getMap('man'),{mode:'dot',eq:false,alpha:.45,axes:['잉크 양 (64칸 밝기 합)','위쪽 잉크 비율']});}whenNear('m2',draw,300);onRedraw(()=>{if(cv.width)draw();});})();
$('baseDr').textContent=pct(BASE_DR);

/* ══ 03 AI 비서 ══ */
const MAIN_PROMPT=`너는 머신러닝 수업 조교다. Google Colab에서 바로 실행되는 파이썬 코드를 써 줘.

[환경] Colab 기본 라이브러리(numpy, scikit-learn, matplotlib)와 umap-learn만 쓴다. umap-learn이 없으면 pip install umap-learn으로 설치한다(이름이 umap이 아니다). 첫 줄에 scikit-learn과 umap-learn 버전을 출력한다.
[데이터] 〔sklearn.datasets.load_digits()〕의 X(〔1,797 × 64〕)만으로 지도를 그린다. 정답 y는 〔채점 칸에서만〕 쓴다.
[규정]
  1) TSNE(n_components=2, perplexity=〔30〕, random_state=0)와 umap.UMAP(n_neighbors=〔15〕, min_dist=0.1, random_state=0)로 2D 지도를 만든다. fit_transform에 y를 넣지 않는다.
  2) 비교용으로 PCA 2D 지도도 만든다.
  3) 채점: 각 지도 좌표 위에서 KNeighborsClassifier(5)의 5겹 교차 검증 정확도(cross_val_score, cv=5).
  4) 정답 없이 재는 점수로 sklearn.manifold.trustworthiness(X, Z, n_neighbors=5)도 낸다.
[출력] 지도 세 장(정답 색칠은 채점 칸에서만), 지도별 5-NN 정확도와 trustworthiness, 지도마다 걸린 시간.
[금지] 정답을 지도 만들기에 쓰지 않는다. 지도 위 무리 사이 거리나 무리 크기로 결론을 내리지 않는다. 지금 버전에 없는 인자 이름을 쓰지 않는다. 데이터를 지어내지 않는다.
[형식] 셀 단위로 나누고, 각 셀 위에 무엇을 하는지 한국어 주석을 단다. 코드 뒤에 내가 확인해야 할 점 3가지를 적어 줘.`;
$('mainPrompt').textContent=MAIN_PROMPT;$('copyMain').addEventListener('click',()=>copyText(MAIN_PROMPT,'프롬프트를 복사했습니다'));
(function(){const LINES=[
 ['!pip install umap',1,'<b>다른 패키지를 설치했다.</b> PyPI의 <code>umap</code>은 이름만 같은 다른 패키지라 <code>umap.UMAP</code>이 없다는 오류가 난다. 설치 이름은 <code>umap-learn</code>, 불러오는 이름은 <code>umap</code>이다.'],
 ['import umap',0,'불러오는 이름은 umap이 맞다. 문제는 설치한 패키지다(1번 줄).'],
 ['from sklearn.datasets import load_digits',0,'손글씨 1,797장. 인터넷 없이 읽는다. 괜찮다.'],['from sklearn.manifold import TSNE',0,'t-SNE는 scikit-learn에 있다. 괜찮다.'],['from sklearn.neighbors import KNeighborsClassifier',0,'채점 잣대(5-NN)를 쓰려고 불렀다. 괜찮다.'],[''],
 ['X, y = load_digits(return_X_y=True)',0,'y를 받아 두는 것 자체는 괜찮다. 어디에 쓰는지가 문제다(11·17번 줄).'],[''],
 ['tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=0)',2,'<b>지금은 없는 인자 이름이다.</b> 반복 횟수는 1.5부터 <code>max_iter</code>이고, <code>n_iter</code>는 1.7에서 지워져 <code>TypeError</code>가 난다. 옛 글을 보고 배운 AI가 자주 낸다.'],
 ['Z_tsne = tsne.fit_transform(X)',0,'정답 없이 X만 넣었다. 괜찮다.'],
 ['Z_umap = umap.UMAP(n_neighbors=15, random_state=0).fit_transform(X, y=y)',3,'<b>정답으로 지도를 그렸다.</b> UMAP의 <code>fit_transform(X, y=y)</code>는 정답이 같은 글씨끼리 끌어당기는 "지도학습 UMAP"이다. 무리가 깨끗해지는 것은 당연하고, 그 지도의 5-NN 점수는 비지도 지원자의 성적이 아니다. 채점 봉투를 미리 연 셈이다.'],[''],
 ['# 다음 주에 들어온 글씨도 같은 지도에 놓기',0,'하려는 일은 좋다. 방법이 문제다(14번 줄).'],
 ['Z_new = tsne.transform(X_new)',4,'<b>t-SNE에는 transform이 없다.</b> <code>AttributeError</code>가 난다. t-SNE는 그린 지도에 새 점을 놓는 규칙을 배우지 않는다. 새 글씨까지 합쳐 다시 그리거나, UMAP의 <code>transform</code>을 쓴다(05장: 450장 96.9%).'],[''],
 ['knn = KNeighborsClassifier(5).fit(Z_umap, y)',5,'<b>채점한 글씨로 그대로 학습했다.</b> 다음 줄과 한 쌍의 문제다.'],
 ['print("지도 정확도:", knn.score(Z_umap, y))',5,'<b>학습에 쓴 점으로 채점했다.</b> 모든 글씨가 자기 자신을 거리 0의 첫째 이웃으로 찾으니 점수가 부풀려진다. <code>cross_val_score(KNeighborsClassifier(5), Z_umap, y, cv=5)</code>처럼 교차 검증으로 잰다.']];
  const host=$('bugCode'),found=new Set();
  host.innerHTML=LINES.map((l,i)=>'<div class="ln" role="listitem" tabindex="0" data-i="'+i+'"><span class="no">'+(i+1)+'</span><span>'+hl(l[0]||' ')+'</span></div>').join('');
  function score(){$('bugScore').textContent=found.size+' / 5';$('bugScore').className='pill'+(found.size===5?' ok':'');}
  function pick(el){const i=+el.getAttribute('data-i'),l=LINES[i];if(!l[0])return;
    if(l[1]){found.add(l[1]);host.querySelectorAll('.ln').forEach(x=>{if(LINES[+x.getAttribute('data-i')][1]===l[1])x.classList.add('hit');});$('bugMsg').innerHTML=(i+1)+'번 줄: '+l[2]+(found.size===5?'<br><b>다섯 군데를 모두 찾았다.</b> 고친 코드는 A장의 Colab 코드와 같은 순서(바른 패키지 → 지금 버전의 인자 → 정답 없이 지도 → 새 글씨는 UMAP transform → 교차 검증으로 채점)를 따른다.':'');}
    else{el.classList.add('miss');$('bugMsg').innerHTML=(i+1)+'번 줄: '+(l[2]||'이 줄은 괜찮다.');}score();}
  host.addEventListener('click',e=>{const el=e.target.closest('.ln');if(el)pick(el);});
  host.addEventListener('keydown',e=>{if((e.key==='Enter'||e.key===' ')&&e.target.classList.contains('ln')){e.preventDefault();pick(e.target);}});
  $('bugReveal').addEventListener('click',()=>{LINES.forEach((l,i)=>{if(l[1]){found.add(l[1]);host.querySelector('[data-i="'+i+'"]').classList.add('hit','reveal');}});score();$('bugMsg').innerHTML='빨간 줄 다섯 군데: ① 1번 다른 패키지 설치 ② 9번 지워진 인자 n_iter ③ 11번 정답으로 그린 지도 ④ 14번 t-SNE에 없는 transform ⑤ 16·17번 학습한 점으로 채점.';});
  $('bugReset').addEventListener('click',()=>{found.clear();host.querySelectorAll('.ln').forEach(x=>x.classList.remove('hit','miss','reveal'));score();$('bugMsg').textContent='줄을 하나 눌러 보세요.';});score();})();
(function(){const CK=[['데이터를 실제로 읽었나','load_digits가 (1797, 64)를 돌려주는지 본다. AI가 예제 데이터를 지어내지 않았는지 확인한다.'],['패키지 이름이 맞나','pip install umap-learn, import umap. pip install umap은 다른 패키지다.'],['인자 이름이 내 버전에 맞나','print(sklearn.__version__). TSNE의 n_iter는 지금 오류다(max_iter). init·learning_rate 기본값도 1.2에서 바뀌었다.'],['지도를 만들 때 정답(y)을 넣지 않았나','fit_transform(X)만. UMAP에 y를 넣으면 지도학습이 된다. 정답은 채점 칸에서만.'],['random_state를 정했나','같은 지도를 다시 얻으려면 고정한다. 결과를 남과 비교할 수 있다.'],['채점을 교차 검증으로 했나','지도 좌표 위 cross_val_score(KNeighborsClassifier(5), Z, y, cv=5). 학습에 쓴 점으로 score하면 부풀려진다.'],['같은 잣대로 비교했나','매뉴얼 씨 지도(18.1%), PCA 지도, 원래 64칸과 같은 5-NN으로 견준다.'],['지도 위 거리·크기로 결론 내지 않았나','"4와 9 무리가 가깝다", "이 무리가 크다"는 지도가 지어낸 것일 수 있다. 근무 조건을 바꿔 여러 장을 함께 본다.'],['새 데이터를 어떻게 놓을지 정했나','t-SNE는 transform이 없다. 매일 들어오는 데이터면 UMAP transform 또는 다시 그리기.']];
  const ul=$('checks');ul.innerHTML=CK.map((c,i)=>'<li><input type="checkbox" id="ck'+i+'"><label for="ck'+i+'"><b>'+c[0]+'</b><small>'+c[1]+'</small></label></li>').join('');
  function upd(){const n=ul.querySelectorAll('input:checked').length;ul.querySelectorAll('li').forEach(li=>li.classList.toggle('done',li.querySelector('input').checked));$('checkScore').textContent='확인 '+n+' / '+CK.length+(n===CK.length?' — 이제 돌려 본다. 결과도 확인 대상이다.':'');}
  ul.addEventListener('change',upd);upd();})();

/* ══ 04 수습 기간 ① t-SNE 1,000걸음 재생 (앞 500장, 이 브라우저가 계산) ══ */
const TSR=(function(){const M=500,X=PXS.slice(0,M),cv=$('tsCv'),RUN={},st={perp:30,mode:'glyph',view:0,busy:false};
  let nn64m=null;function nn64Mini(){if(!nn64m)nn64m=DRX.knnAll(M,(i,j)=>DRX.d2(X[i],X[j]),5);return nn64m;}
  function snapZ(R,k){const Y=R.snaps[k],Z=[];for(let i=0;i<M;i++)Z.push([Y[2*i],Y[2*i+1]]);return Z;}
  function draw(){const R=RUN[st.perp],H=respH(cv,0.62,320,500);const W=fit(cv,H),ctx=cv.getContext('2d'),C=cols(),chartH=74;ctx.clearRect(0,0,W,H);
    if(!R||!R.snaps.length){ctx.fillStyle=C.muted;ctx.font=font(14);ctx.textAlign='center';ctx.fillText('▶ 처음부터 펼치기를 누르면 시작한다',W/2,H/2);return;}
    const k=Math.min(st.view,R.snaps.length-1),Z=snapZ(R,k);
    /* 위: 지도 (그 걸음의 점들에 맞춰 눈금을 매번 다시 잡는다) */
    let x0=Infinity,x1=-Infinity,y0=Infinity,y1=-Infinity;Z.forEach(z=>{x0=Math.min(x0,z[0]);x1=Math.max(x1,z[0]);y0=Math.min(y0,z[1]);y1=Math.max(y1,z[1]);});
    const gh=H-chartH-14,sz=W<520?7:9,s=Math.min((W-20)/((x1-x0)||1e-9),(gh-20)/((y1-y0)||1e-9)),ox=(W-(x1-x0)*s)/2,oy=8+(gh-(y1-y0)*s)/2;
    ctx.strokeStyle=C.rule;ctx.strokeRect(.5,.5,W-1,gh+8);ctx.imageSmoothingEnabled=false;
    Z.forEach((z,i)=>{const X_=ox+(z[0]-x0)*s,Y_=oy+(y1-z[1])*s;if(st.mode==='glyph')ctx.drawImage(glyph(i,C.text),X_-sz/2,Y_-sz/2,sz,sz);else{ctx.fillStyle=rgba(C.text,.6);ctx.fillRect(X_-1.6,Y_-1.6,3.2,3.2);}});
    const stepN=k*10,ex=stepN<250;ctx.fillStyle=ex?C.stamp:C.blue;ctx.font=font(13,700,1);ctx.textAlign='right';ctx.fillText('걸음 '+stepN+' · '+(ex?'끌림 ×12 (과장)':'다듬기')+' · KL '+R.kl[k].toFixed(3),W-10,24);
    ctx.fillStyle=C.muted;ctx.font=font(11.5,500,1);ctx.fillText('지도 폭 '+(x1-x0).toExponential(1),W-10,40);
    /* 아래: KL 발산 곡선 */
    const cy=H-chartH,pl=40,gw=W-pl-12,lx=q=>pl+q/100*gw,kmax=Math.max(...R.kl.filter(isFinite)),kmin=Math.min(...R.kl),ly=v=>cy+6+(1-(v-kmin)/((kmax-kmin)||1))*(chartH-24);
    ctx.strokeStyle=C.rule;ctx.strokeRect(pl+.5,cy+.5,gw,chartH-14);ctx.fillStyle=C.muted;ctx.font=font(11);ctx.textAlign='right';ctx.fillText('KL',pl-6,cy+16);ctx.textAlign='center';[0,250,500,1000].forEach(v=>ctx.fillText(v,lx(v/10),H-2));
    ctx.strokeStyle=rgba(C.stamp,.6);ctx.setLineDash([4,4]);ctx.beginPath();ctx.moveTo(lx(25),cy);ctx.lineTo(lx(25),cy+chartH-14);ctx.stroke();ctx.setLineDash([]);
    ctx.strokeStyle=C.blue;ctx.lineWidth=2.2;ctx.beginPath();for(let q=0;q<=k;q++){const X_=lx(q),Y_=ly(R.kl[q]);q?ctx.lineTo(X_,Y_):ctx.moveTo(X_,Y_);}ctx.stroke();circ(ctx,lx(k),ly(R.kl[k]),4,C.stamp,null);}
  function info(){const R=RUN[st.perp];if(!R)return;const k=Math.min(st.view,R.snaps.length-1),stepN=k*10;let h='perplexity <b>'+st.perp+'</b> · 걸음 <b>'+stepN+'</b> / 1,000 · KL 발산 <b>'+R.kl[k].toFixed(3)+'</b>. ';
    h+=stepN===0?'처음 자리는 PCA 그림자를 아주 작게(폭 0.0001쯤) 줄인 것이다. 모든 글씨가 한 점에 겹쳐 있다.':stepN<250?'과장 구간이다. 이웃 끌림이 12배라 같은 모양의 글씨가 먼저 덩어리로 뭉친다. 지도 폭이 빠르게 커진다.':stepN<400?'과장을 풀었다. KL이 한 번 뛰었다가(과장이 사라진 만큼 P와 Q가 다시 어긋난다) 다시 줄어든다. 무리 사이가 벌어진다.':'다듬기 구간이다. 무리 모양은 거의 정해졌고, 무리 안의 자리를 조금씩 고친다. 같은 숫자의 글씨끼리 섬을 이룬 것이 정답 없이도 보인다.';
    if(R.done){h+='<br><span class="muted">1,000걸음 계산 '+(R.ms/1000).toFixed(1)+'초(이 브라우저, 500장 정확한 t-SNE) · 이웃 보존율(64칸 5-NN이 지도 5-NN에 남은 몫) <b>'+pc1(R.ov)+'</b>'+(st.perp===5?' · perplexity 5는 아주 가까운 몇 장만 봐서 작은 섬이 많다.':st.perp===100?' · perplexity 100은 넓게 봐서 무리가 둥글게 뭉치고 서로 가깝다.':'')+'</span>';}
    $('tsInfo').innerHTML=h;}
  async function run(){if(st.busy)return;st.busy=true;$('tsPlay').disabled=true;$('tsStep').disabled=true;const perp=st.perp;$('tsInfo').textContent='perplexity '+perp+': 500장의 64칸 이웃 확률(P) 계산 중…';await wait(20);
    const t0=performance.now(),T=DRX.tsne(X,{perplexity:perp}),R={snaps:[Float32Array.from(T.Y)],kl:[T.kl()],done:false,ms:0};RUN[perp]=R;st.view=0;draw();info();
    while(T.it<1000){const f0=performance.now();for(let q=0;q<10;q++)T.step();R.snaps.push(Float32Array.from(T.Y));R.kl.push(T.kl());R.ms+=performance.now()-f0;
      if(st.perp===perp){st.view=R.snaps.length-1;$('tsStep').value=st.view*10;$('tsStepO').textContent=st.view*10;draw();if(R.snaps.length%5===1)info();}await wait(REDUCE?0:16);}
    R.done=true;const Zf=snapZ(R,R.snaps.length-1);R.ov=DRX.overlap(M,DRX.knnAll(M,(i,j)=>DRX.d2(Zf[i],Zf[j]),5),nn64Mini(),5);
    st.busy=false;$('tsPlay').disabled=false;$('tsStep').disabled=false;if(st.perp===perp){draw();info();}}
  $('tsPlay').addEventListener('click',()=>{delete RUN[st.perp];run();});
  seg('#tsPerp button',v=>{if(st.busy){toast('지금 계산이 끝나면 바꿀 수 있어요');document.querySelectorAll('#tsPerp button').forEach(b=>b.setAttribute('aria-pressed',+b.getAttribute('data-v')===st.perp?'true':'false'));return;}
    st.perp=+v;const R=RUN[st.perp];if(R){st.view=R.snaps.length-1;$('tsStep').value=st.view*10;$('tsStepO').textContent=st.view*10;draw();info();}else run();});
  seg('#tsDraw button',v=>{st.mode=v;draw();});
  $('tsStep').addEventListener('input',e=>{st.view=Math.round(+e.target.value/10);$('tsStepO').textContent=st.view*10;draw();info();});
  draw();onRedraw(draw);return {run,RUN,st};})();
whenNear('m4',()=>{if(!TSR.RUN[30])TSR.run();},200);

/* ══ 04 수습 기간 ② UMAP 근무 조건 ══ */
(function(){const cv=$('umCv');let key='um15';
  const NOTE={um5:'이웃 5장만 잇는다. 동네만 보니 작은 섬이 많이 생기고, 섬들의 배치는 제멋대로다.',um15:'기본값. 동네와 전체의 균형.',um50:'이웃 50장을 잇는다. 섬이 조금 둥글어지고 전체 배치가 조금 더 안정된다.',um200:'이웃 200장. 넓게 보느라 섬들이 서로 붙고, 섬 안의 세밀한 구조는 흐려진다.',umd0:'점끼리 겹쳐도 된다. 무리가 아주 촘촘한 덩어리가 된다(군집에 넣기 좋다).',umd99:'점끼리 거의 붙지 못한다. 무리가 넓고 고르게 퍼져 보기에는 편하지만 무리 사이 틈이 좁아진다.'};
  function draw(){drawMap(cv,respH(cv,0.6,300,480),getMap(key),{mode:'glyph',size:cv.clientWidth<520?7:9});}
  async function info(){$('umInfo').innerHTML='<b>'+MAPNAME[key]+'</b> — '+NOTE[key]+'<br><span class="muted">이웃 보존율 계산 중…</span>';await mapTrust(key);const s=stOf(key);
    $('umInfo').innerHTML='<b>'+MAPNAME[key]+'</b> — '+NOTE[key]+'<br>정답 없이 재는 점수: 이웃 보존율 <b>'+pc1(s.ov)+'</b> · trustworthiness <b>'+fx(s.trust,3)+'</b>. <span class="muted">정답으로 잰 점수(5-NN)는 05장에서 채점 봉투를 열고 본다.</span>';}
  function set(k){key=k;document.querySelectorAll('#umNn button').forEach(b=>b.setAttribute('aria-pressed',b.getAttribute('data-v')===(k.startsWith('umd')?'um15':k)?'true':'false'));
    document.querySelectorAll('#umMd button').forEach(b=>b.setAttribute('aria-pressed',b.getAttribute('data-v')===(k.startsWith('umd')||k==='um15'?k:'um15')?'true':'false'));draw();info();}
  document.querySelectorAll('#umNn button,#umMd button').forEach(b=>b.addEventListener('click',()=>set(b.getAttribute('data-v'))));
  whenNear('m4',()=>{draw();info();},300);onRedraw(()=>{if(cv.width)draw();});})();

/* ══ 05 성과 평가: 채점 봉투 열기 ══ */
const ENV={open:false};
const M4=[['man','mpMan','sMan'],['pca','mpPca','sPca'],['ts30','mpTs','sTs'],['um15','mpUm','sUm']];
function drawMaps4(){M4.forEach(m=>{const cv=$(m[1]),man=m[0]==='man';drawMap(cv,respH(cv,0.8,220,300),getMap(m[0]),{mode:'dot',eq:!man,alpha:ENV.open?.75:.35,color:ENV.open?(i=>DIGCOL[SEALED[i]]):null});});}
async function openEnvelope(){if(ENV.open)return;ENV.open=true;$('envOpen').disabled=true;$('envOpen').textContent='✉ 봉투를 열었다';$('envNote').textContent='채점 중…';drawMaps4();await wait(20);
  $('digLeg').innerHTML=DIGCOL.map((c,k)=>'<span><i style="background:'+c+'"></i>'+k+'</span>').join('')+'<span class="muted">색 = 숨긴 정답 (채점 봉투)</span>';
  M4.forEach(m=>{$(m[2]).textContent=pct(mapCV(m[0]))+(m[0]==='man'?' (Python '+pct(DR_LINE)+')':'');});await wait(10);const x64=await x64CV();
  const cMan=mapCV('man'),cP=mapCV('pca'),cT=mapCV('ts30'),cU=mapCV('um15');
  $('envStat').innerHTML=statGrid([[pct(cT),'t-SNE 지도 5-NN','a'],[pct(cU),'UMAP 지도 5-NN','a'],[pct(DR_LINE),'합격선 · 매뉴얼 씨 지도','m'],[pct(cP),'PCA 2D 지도 (5-14)'],[pct(x64),'원래 64칸 그대로 (참고)','s'],[pct(BASE_DR),'기준선 · 가장 흔한 숫자']],3)+
    '<p class="foot">모두 같은 잣대: 지도 좌표 위 5-NN, 5겹 교차 검증(섞지 않음). t-SNE·UMAP 지도는 정답 없이 그렸는데도 원래 64칸 그대로 이웃을 찾은 것('+pct(x64)+')과 맞먹는다. 64칸 거리에는 모든 칸의 잡음이 섞이지만, 지도는 이웃 관계만 남겨 오히려 조금 깨끗해졌다. 매뉴얼 씨 지도를 이 브라우저에서 다시 재면 '+pct(cMan)+'다. 잉크 양이 정수라 같은 거리의 이웃이 많고, 다섯째 이웃을 고르는 순서가 scikit-learn과 달라 생기는 차이다. 합격선은 Python 값 '+pct(DR_LINE)+'로 둔다.</p>';
  $('envNote').textContent='색은 숨긴 정답이다. 지도는 이 색을 모르고 그렸다.';}
$('envOpen').addEventListener('click',openEnvelope);
whenNear('m5',()=>{drawMaps4();},400);onRedraw(()=>{if($('mpMan').width)drawMaps4();});
/* 정답 없이 재는 점수 · 업무 점수 · 거리와 크기 · 새 글씨 */
async function renderTrust(){const keys=['man','pca','ts30','um15'];for(const k of keys)await mapTrust(k);
  $('trustBars').innerHTML=barRows(keys.map(k=>[MAPNAME[k].replace(/ \(.*/,''),stOf(k).trust,k==='man'?'var(--c-man)':k==='pca'?'var(--muted)':'var(--blue)',fx(stOf(k).trust,3)]),1);
  $('trustFoot').innerHTML='정답을 쓰지 않는다. 지도 위 이웃 5장 가운데 64칸에서 이웃이 아니었던 글씨가 있으면, 그 글씨가 64칸에서 몇 번째로 가까웠는지만큼 벌점을 준다. 1이면 지도 위 이웃이 모두 진짜 이웃. 이웃 보존율(64칸 5-NN이 지도 5-NN에 남은 몫)은 매뉴얼 '+pc1(stOf('man').ov)+' · PCA '+pc1(stOf('pca').ov)+' · t-SNE '+pc1(stOf('ts30').ov)+' · UMAP '+pc1(stOf('um15').ov)+'. A장(Python)의 trustworthiness는 0.614 · 0.830 · 0.995 · 0.990이다.';}
let REPK=10;
async function renderRep(){$('repBars').innerHTML='<p class="muted">K-Means '+REPK+'묶음 계산 중…</p>';await wait(20);const keys=['man','pca','ts30','um15','x64'],R={};for(const k of keys){R[k]=kmOf(k,REPK);await wait(0);}
  $('repBars').innerHTML=barRows(keys.map(k=>[k==='x64'?'원래 64칸 (참고)':MAPNAME[k].replace(/ \(.*/,''),R[k].acc,k==='man'?'var(--c-man)':k==='x64'?'var(--muted)':'var(--blue)',pct(R[k].acc)]),1);
  $('repFoot').innerHTML='사람이 이름을 붙이는 글씨: <b>'+REPK+'장</b>(1,797장의 '+pct(REPK/NN,1)+'). 나머지 '+(NN-REPK).toLocaleString()+'장은 같은 묶음의 대표 이름을 그대로 받는다. K-Means 묶음과 숨긴 정답의 겹침(ARI)은 t-SNE '+fx(R.ts30.ari,3)+' · UMAP '+fx(R.um15.ari,3)+' · PCA '+fx(R.pca.ari,3)+' · 64칸 '+fx(R.x64.ari,3)+'(5-10 K-Means의 0.666). '+
    (REPK===10?'UMAP은 섬 크기가 들쭉날쭉해 10묶음이면 큰 섬이 둘로 쪼개지고 작은 섬 둘이 한 묶음이 되기 쉽다. 대표를 늘리면 빠르게 따라잡는다. ':'')+'<b>주의:</b> t-SNE·UMAP 지도는 무리를 실제보다 또렷하게 그리므로 그 위의 군집 점수는 후하게 나온다(반칙에 가깝다). 지도에서 찾은 묶음은 원래 64칸에서 다시 확인한다. 이 막대는 브라우저 K-Means(k-means++ 10번)라 A장 숫자와 조금 다를 수 있다.';}
seg('#repK button',v=>{REPK=+v;renderRep();});
(function(){const cv=$('geoCv');let key='ts30';
  function draw(){const g=geoOf(key),H=respH(cv,0.8,260,340),W=fit(cv,H),ctx=cv.getContext('2d'),C=cols(),Z=getMap(key);
    let x0=Infinity,x1=-Infinity,y0=Infinity,y1=-Infinity;Z.forEach(z=>{x0=Math.min(x0,z[0]);x1=Math.max(x1,z[0]);y0=Math.min(y0,z[1]);y1=Math.max(y1,z[1]);});
    const s=Math.min((W-20)/(x1-x0),(H-20)/(y1-y0)),ox=(W-(x1-x0)*s)/2,oy=(H-(y1-y0)*s)/2,sx=x=>ox+(x-x0)*s,sy=y=>oy+(y1-y)*s;ctx.clearRect(0,0,W,H);ctx.strokeStyle=C.rule;ctx.strokeRect(.5,.5,W-1,H-1);
    Z.forEach((z,i)=>{ctx.fillStyle=rgba(DIGCOL[SEALED[i]],.18);ctx.fillRect(sx(z[0])-1,sy(z[1])-1,2,2);});
    g.c.forEach((c,k)=>{const X=sx(c[0]),Y=sy(c[1]);circ(ctx,X,Y,Math.max(6,g.r[k]*s),rgba(DIGCOL[k],.18),DIGCOL[k],2);ctx.fillStyle=C.text;ctx.font=font(15,700,1);ctx.textAlign='center';ctx.fillText(k,X,Y+5);});}
  function table(){const g=geoOf(key),h=geoOf('x64'),rho=DRX.spearman(h.pd,g.pd),ratio=a=>Math.max(...a)/Math.min(...a),big=a=>a.indexOf(Math.max(...a)),small=a=>a.indexOf(Math.min(...a));
    const near=x=>{const q=x.pd.indexOf(Math.min(...x.pd));return x.pairs[q].join('–');},far=x=>{const q=x.pd.indexOf(Math.max(...x.pd));return x.pairs[q].join('–');};
    $('geoTbl').innerHTML='<thead><tr><th></th><th>64칸 (원래)</th><th>이 지도</th></tr></thead><tbody>'+
      '<tr><td>무리 크기 비 (가장 큰 ÷ 가장 작은)</td><td>'+ratio(h.r).toFixed(2)+'배</td><td><b>'+ratio(g.r).toFixed(2)+'배</b></td></tr>'+
      '<tr><td>가장 큰 무리 · 가장 작은 무리</td><td>'+big(h.r)+' · '+small(h.r)+'</td><td>'+big(g.r)+' · '+small(g.r)+'</td></tr>'+
      '<tr><td>가장 가까운 두 무리</td><td>'+near(h)+'</td><td>'+near(g)+'</td></tr>'+
      '<tr><td>가장 먼 두 무리</td><td>'+far(h)+'</td><td>'+far(g)+'</td></tr>'+
      '<tr><td>중심 거리 45쌍의 순위 상관</td><td>1 (기준)</td><td><b>'+fx(rho,3)+'</b></td></tr></tbody>';
    $('geoInfo').innerHTML='<b>'+MAPNAME[key]+'</b>. 원의 크기는 그 숫자 글씨들이 중심에서 평균 얼마나 떨어져 있나(무리 반지름)다. 64칸에서는 가장 큰 무리가 가장 작은 무리의 '+ratio(h.r).toFixed(2)+'배인데, 이 지도에서는 '+ratio(g.r).toFixed(2)+'배다. '+
      (key.startsWith('um')?'UMAP은 이웃 그물을 당겨 그리므로 촘촘한 무리는 더 작게, 듬성한 무리는 더 크게 그리기 쉽다. ':key.startsWith('ts')?'t-SNE는 장마다 동네 반지름을 따로 맞추므로 빽빽한 무리와 듬성한 무리를 비슷한 크기로 펴는 경향이 있다. ':'PCA는 곧은 그림자라 크기와 거리를 비교적 정직하게 남기지만 무리끼리 겹친다. ')+
      '순위 상관이 1보다 작을수록 "어느 무리끼리 가까운가"가 64칸과 다르다. 근무 조건을 바꿔 가며 같은 표를 보자. 바뀌는 것은 믿지 않는다.';}
  seg('#geoMap button',v=>{key=v;draw();table();});
  whenNear('geoCv',()=>{draw();table();},300);onRedraw(()=>{if(cv.width)draw();});})();
function newScore(){const s=stOf('new');if(s.acc==null){const Zo=decodeEmb(EMB.umOld,EMB_SCALE.um),Zn=decodeEmb(EMB.umNew,EMB_SCALE.um);let ok=0;
    Zn.forEach((z,q)=>{const bd=new Float64Array(5).fill(Infinity),bi=new Int32Array(5);for(let r=0;r<Zo.length;r++){const v=DRX.d2(z,Zo[r]);if(v<bd[4]){let p=4;while(p>0&&bd[p-1]>v){bd[p]=bd[p-1];bi[p]=bi[p-1];p--;}bd[p]=v;bi[p]=r;}}
      const c=new Array(10).fill(0);for(let p=0;p<5;p++)c[SEALED[EMB.umOldIdx[bi[p]]]]++;let b=0;for(let k=1;k<10;k++)if(c[k]>c[b])b=k;if(b===SEALED[EMB.umNewIdx[q]])ok++;});
    s.acc=ok/Zn.length;s.Zo=Zo;s.Zn=Zn;}return s;}
(function(){const cv=$('newCv');function draw(){const s=newScore(),H=respH(cv,0.7,240,340),W=fit(cv,H),ctx=cv.getContext('2d'),C=cols(),A=s.Zo.concat(s.Zn);
    let x0=Infinity,x1=-Infinity,y0=Infinity,y1=-Infinity;A.forEach(z=>{x0=Math.min(x0,z[0]);x1=Math.max(x1,z[0]);y0=Math.min(y0,z[1]);y1=Math.max(y1,z[1]);});
    const sc=Math.min((W-20)/(x1-x0),(H-20)/(y1-y0)),ox=(W-(x1-x0)*sc)/2,oy=(H-(y1-y0)*sc)/2,sx=x=>ox+(x-x0)*sc,sy=y=>oy+(y1-y)*sc;ctx.clearRect(0,0,W,H);ctx.strokeStyle=C.rule;ctx.strokeRect(.5,.5,W-1,H-1);
    s.Zo.forEach((z,r)=>{ctx.fillStyle=rgba(DIGCOL[SEALED[EMB.umOldIdx[r]]],.28);ctx.fillRect(sx(z[0])-1.5,sy(z[1])-1.5,3,3);});
    s.Zn.forEach((z,q)=>circ(ctx,sx(z[0]),sy(z[1]),3.2,DIGCOL[SEALED[EMB.umNewIdx[q]]],C.text,0.8));
    ctx.fillStyle=C.text;ctx.font=font(12,700);ctx.textAlign='left';ctx.fillText('흐린 점: 먼저 그린 1,347장 · 진한 점: 나중 450장',10,18);
    $('newInfo').innerHTML='UMAP은 지도를 그릴 때 배운 "64칸 → 지도" 규칙을 남겨 둔다. 나중에 들어온 글씨 450장을 <code>transform</code>으로 기존 지도에 놓고, 먼저 그린 1,347장의 이름으로 5-NN을 하면 <b>'+pct(s.acc)+'</b>가 맞는다(A장 10번 칸 '+pct(PRE.py.umNew)+'). 지도를 다시 그리지 않았으니 기존 점들의 자리는 그대로다.<br><br><b>t-SNE는?</b> scikit-learn의 <code>TSNE</code>에는 <code>transform</code>이 없다(A장 4번 칸: transform 있음? False). 새 글씨를 놓으려면 1,797장을 다시 그려야 하고, 그러면 기존 점들의 자리도 바뀔 수 있다. 매일 글씨가 들어오는 일터라면 이 차이가 크다.';}
  whenNear('newCv',draw,300);onRedraw(()=>{if(cv.width)draw();});})();
whenNear('m5',async()=>{await renderTrust();await renderRep();},200);

/* ══ 면접 결과 발표 ══ */
Interview('iv15',{title:'t-SNE · UMAP은 손글씨 지도 그리기에 합격할까?',
  q:'조건: 정답 없이 그린 2D 지도 위에서 5-NN 정확도(5겹, 채점 봉투로 채점)가 합격선(매뉴얼 씨 지도 '+pct(DR_LINE)+')과 기준선(가장 흔한 숫자 '+pct(BASE_DR)+')을 모두 넘으면 합격. 근무 조건은 기본값(t-SNE perplexity 30 · UMAP n_neighbors 15, min_dist 0.1). 공동 지원이라 둘 다 넘어야 합격이다. 지도는 Python으로 미리 그렸고, 채점은 이 브라우저가 한다.',
  fly:'정답 안 보고 그린 거 맞지? 그리고 지도 위 거리는 믿어도 돼?',robo:'지도 한 장 그리는 데 몇 초야? 새 글씨 들어오면 또 그려?',
  async run(setP){setP(.15);await wait(20);const cT=mapCV('ts30');setP(.35);await wait(20);const cU=mapCV('um15');setP(.55);await wait(20);const x64=await x64CV();setP(.8);const cP=mapCV('pca');setP(1);
    if(!ENV.open)openEnvelope();
    const pass=Math.min(cT,cU)>DR_LINE&&Math.min(cT,cU)>BASE_DR;
    return {result:pass?'pass':'hold',flyAns:'t-SNE '+pct(cT)+' · UMAP '+pct(cU)+' · 지도를 그릴 때 정답은 안 씀(채점에만) · 무리는 믿고, 무리 사이 거리와 크기는 믿지 말 것(05장)',
      roboAns:'1,797장에 t-SNE 약 '+PRE.tsneSec.toFixed(0)+'초 · UMAP 약 '+PRE.umapWarmSec.toFixed(0)+'초(첫 실행 '+PRE.umapSec.toFixed(0)+'초) · 외울 숫자는 좌표 '+(NN*2).toLocaleString()+'개 · 새 글씨: UMAP은 transform, t-SNE는 다시 그리기',
      flyMood:pass?'arms':'coffee',flyMove:pass?'jump':'shake',roboMood:pass?'wow':'cry',roboMove:pass?'jump':'shake',asstMood:pass?'spark':'sigh',docMood:pass?'magic':'scale',
      measure:'채점 봉투로 채점했어요. t-SNE <b>'+pct(cT)+'</b>, UMAP <b>'+pct(cU)+'</b>로 합격선 '+pct(DR_LINE)+'보다 '+fx((Math.min(cT,cU)-DR_LINE)*100,1)+'%p 높아요. 원래 64칸 그대로 이웃을 찾은 것('+pct(x64)+')과 맞먹고, 같은 잣대에서 5-14 PCA 지도는 '+pct(cP)+'였습니다.',
      verdict:pass?'합격입니다. 두 지원자 모두 차원축소팀에 배치합니다. 조건이 있어요. 지도는 "보는 용도"로 쓰고 무리 사이 거리와 크기로 결론 내지 말 것, 근무 조건을 바꾼 지도를 함께 볼 것, 매일 들어오는 글씨는 UMAP으로 놓을 것.':'보류입니다. 지도 위에서 같은 숫자끼리 충분히 모이지 않았어요.',
      key:'이웃을 지킨 지도 — 5-NN '+pct(DR_LINE)+' → t-SNE '+pct(cT)+' · UMAP '+pct(cU),
      render(box){box.innerHTML='<div class="bars">'+barRows([['t-SNE 지도',cT,'var(--blue)',pct(cT)],['UMAP 지도',cU,'var(--blue)',pct(cU)],['원래 64칸 (참고)',x64,'var(--muted)',pct(x64)],['PCA 2D 지도 (5-14)',cP,'var(--muted)',pct(cP)],['매뉴얼 씨 지도 (합격선)',DR_LINE,'var(--c-man)',pct(DR_LINE)],['기준선 (가장 흔한 숫자)',BASE_DR,'var(--muted)',pct(BASE_DR)]],1)+'</div>';
        if(pass){const res=box.parentNode;setTimeout(()=>{react(res.querySelector('.verdict img'),'jump');react(res.querySelector('.say.asst img'),'jump');},700);}}};},
  alts:[
   {label:'근무 조건을 바꾸면 (perplexity 5·100 · n_neighbors 5·200)',async run(box){const ks=['ts5','ts100','um5','um200'],v={};for(const k of ks){v[k]=mapCV(k);await wait(0);}
     box.innerHTML=statGrid(ks.map(k=>[pct(v[k]),MAPNAME[k],'a']),4)+'<p class="foot">네 지도 모두 합격선을 크게 넘는다. 판정: '+(ks.every(k=>v[k]>DR_LINE)?'<b>합격</b>':'보류')+'. 같은 숫자끼리 모인다는 결론은 근무 조건에 거의 흔들리지 않는다. 흔들리는 것은 섬의 모양, 크기, 섬끼리의 배치다(05장 "믿지 말아야 할 것"). 넓게 볼수록(perplexity 100, n_neighbors 200) 세밀한 이웃이 흐려져 점수가 조금 내려간다.</p>';}},
   {label:'새 글씨 450장이 들어오면',async run(box){const s=newScore();
     box.innerHTML=statGrid([[pct(s.acc),'UMAP · transform으로 기존 지도에 놓기','a'],['불가','t-SNE · transform 없음 (다시 그려야 함)'],[pct(DR_LINE),'합격선','m']],3)+'<p class="foot">1,347장으로 지도를 그려 두고, 나중에 들어온 450장을 UMAP <code>transform</code>으로 놓은 뒤 먼저 그린 1,347장의 이름으로 5-NN을 했다. 지도를 다시 그리지 않아도 합격선을 크게 넘는다. 판정: UMAP <b>합격</b> · t-SNE는 이 조건에서 <b>해당 없음</b>(일을 할 수 없다). 매일 새 글씨가 들어오는 일터에는 UMAP을 배치한다.</p>';}},
   {label:'지도 위에서 바로 묶으면 (군집팀 잣대 ARI)',async run(box){const ks=['ts30','um15','pca','x64'],R={};for(const k of ks){R[k]=kmOf(k,10);await wait(0);}
     box.innerHTML=statGrid(ks.map(k=>['ARI '+fx(R[k].ari,3),(k==='x64'?'원래 64칸 K-Means (5-10)':MAPNAME[k].replace(/ \(.*/,'')+' 위 K-Means'),k==='x64'?'s':'a']).concat([['ARI '+fx(CLU_LINE,3),'군집팀 합격선 · 매뉴얼 씨','m']]),5)+'<p class="foot">지도 위에서 K-Means(10묶음)를 돌려 숨긴 정답과 겹침(ARI)을 쟀다. t-SNE 지도 위가 64칸 그대로보다 훨씬 높다. 하지만 이것은 <b>반칙에 가깝다</b>. t-SNE는 무리 사이를 일부러 벌려 그리므로 그 위에서 묶기가 쉬워진 것이고, 지도의 모양은 근무 조건마다 바뀐다. 묶음 수를 정하거나 대표를 고르는 데 도움으로 쓰고, 결과는 원래 64칸에서 확인한다. 이 면접의 판정(지도 5-NN)은 바뀌지 않는다.</p>';}}]});

/* ══ 06 인사 카드: 차원축소팀 정산 ══ */
async function renderDrLedger(){const keys=['man','pca','ts30','um15'];$('drLedger').innerHTML='<tbody><tr><td class="muted">계산 중…</td></tr></tbody>';
  for(const k of keys){mapCV(k);await mapTrust(k);kmOf(k,10);await wait(0);}const x64=await x64CV(),nw=newScore().acc;
  const T={man:'—',pca:'0.01초 미만',ts30:'약 '+PRE.tsneSec.toFixed(0)+'초',um15:'약 '+PRE.umapWarmSec.toFixed(0)+'초 (첫 실행 '+PRE.umapSec.toFixed(0)+'초)'},NEW={man:'규칙 그대로',pca:'가능 (transform)',ts30:'불가',um15:'가능 · '+pct(nw)};
  const NAME={man:'5-0 매뉴얼 씨 · 잉크·위쪽',pca:'5-14 PCA · 2성분',ts30:'5-15 t-SNE · perplexity 30',um15:'5-15 UMAP · n_neighbors 15'};
  $('drLedger').innerHTML='<thead><tr><th>지원자 · 기준 지도</th><th>지도 5-NN</th><th>trustworthiness</th><th>대표 10장 업무 점수</th><th>새 글씨</th><th>시간 (1,797장)</th><th>판정</th></tr></thead><tbody>'+
    keys.map(k=>{const s=stOf(k),cv=s.cv,pass=k!=='man'&&cv>DR_LINE;return '<tr'+(k.startsWith('ts')||k.startsWith('um')?' style="background:var(--panel2)"':'')+'><td><b style="color:var(--text)">'+NAME[k]+'</b></td><td>'+pct(k==='man'?DR_LINE:cv)+'</td><td>'+fx(s.trust,3)+'</td><td>'+pct(kmOf(k,10).acc)+'</td><td>'+NEW[k]+'</td><td>'+T[k]+'</td><td>'+(k==='man'?'<span class="pill no">합격선</span>':'<span class="pill '+(pass?'ok':'no')+'">'+(pass?/*i18n:verdict*/'합격':'보류')+'</span>')+'</td></tr>';}).join('')+
    '<tr class="muted"><td>(참고) 원래 64칸 그대로</td><td>'+pct(x64)+'</td><td>1</td><td>'+pct(kmOf('x64',10).acc)+'</td><td>—</td><td>—</td><td></td></tr></tbody>';
  $('drLedgerFoot').innerHTML='모든 숫자는 이 브라우저가 같은 1,797장, 같은 잣대로 계산했다(t-SNE·UMAP 좌표만 Python에서 미리 그림, 시간은 A장 실행 기록). 매뉴얼 씨 줄의 5-NN은 합격선(Python 18.1%)을 적었다. 5-14 PCA의 판정은 그 편의 면접 결과가 확정되기 전이라 "예상"으로 적었다. 차원축소팀 두 자리 모두 합격선을 넘었다. PCA는 축에 뜻이 있고 빠르며 다른 지원자의 전처리로 쓰이고(5-13 GMM), t-SNE·UMAP은 눈으로 읽는 지도에서 앞선다.';
  document.querySelectorAll('.tsAcc').forEach(e=>e.textContent=pct(stOf('ts30').cv));document.querySelectorAll('.umAcc').forEach(e=>e.textContent=pct(stOf('um15').cv));}
whenNear('m6',renderDrLedger,500);

/* ══ 마무리: 15명 채용 결산 (맨 위 FINAL 배열에서 그린다) ══ */
(function(){const RES={pass:[/*i18n:verdict*/'합격','ok'],hold:['보류','no'],fail:[/*i18n:verdict*/'불합격','no']},HQN={'지도':'지도 본부 — 정답이 있는 업무 (콘크리트 실기 과제지)','비지도':'비지도 본부 — 정답이 없는 업무 (손글씨 숫자 실기 과제지)'};
  const ep=n=>EPISODES.find(e=>e.n===n),link=r=>{const e=ep(r.n);return e?'<a href="'+e.f+'">'+r.n+'</a>':r.n;};
  const row=r=>'<tr'+(r.n===EP.n?' class="me"':'')+'><td style="white-space:nowrap">'+link(r)+'</td><td><b style="color:var(--text)">'+r.who+'</b></td><td style="white-space:nowrap">'+r.team+'</td><td style="font-size:13px">'+r.model+'</td><td>'+r.score+(r.note?'<br><span class="muted" style="font-size:12.5px">'+r.note+'</span>':'')+'</td><td style="white-space:nowrap"><span class="pill '+RES[r.res][1]+'">'+RES[r.res][0]+'</span>'+(r.est?'<span class="est" title="그 편의 면접 결과가 확정되기 전의 예상값">예상</span>':'')+'</td></tr>';
  const cand=FINAL.filter(r=>r.hq!=='공고'),cnt=(arr,res)=>arr.filter(r=>r.res===res).length,by=h=>cand.filter(r=>r.hq===h);
  let h='<thead><tr><th>편</th><th>지원자</th><th>팀</th><th>기준 모델 · 조건</th><th>핵심 숫자</th><th>판정</th></tr></thead><tbody>';
  FINAL.filter(r=>r.hq==='공고').forEach(r=>{h+='<tr class="hq"><td colspan="6">채용 공고 — 기존 직원 (모든 합격선의 출처)</td></tr>'+row(r);});
  ['지도','비지도'].forEach(q=>{const a=by(q);h+='<tr class="hq"><td colspan="6">'+HQN[q]+' · 합격 '+cnt(a,'pass')+' / '+a.length+'</td></tr>'+a.map(row).join('');});
  h+='<tr class="hq"><td colspan="6">팀별 합격선 (매뉴얼 씨)</td></tr>'+FINAL_LINES.map(l=>'<tr><td></td><td colspan="2">'+l[0]+'</td><td colspan="3">'+l[1]+'</td></tr>').join('')+'</tbody>';
  $('finalTbl').innerHTML=h;
  const teams={};cand.forEach(r=>{(teams[r.team]=teams[r.team]||[]).push(r);});
  $('finalSum').innerHTML='<div class="fsum"><span>전체 합격 <b>'+cnt(cand,'pass')+' / '+cand.length+'</b></span><span>보류 <b>'+cnt(cand,'hold')+'</b></span>'+(cnt(cand,'fail')?'<span>불합격 <b>'+cnt(cand,'fail')+'</b></span>':'')+
    Object.keys(teams).map(t=>'<span>'+t+'팀 <b>'+cnt(teams[t],'pass')+'/'+teams[t].length+'</b></span>').join('')+'</div>';
  const est=cand.filter(r=>r.est).map(r=>r.n);
  $('finalFoot').innerHTML='각 편의 기준 모델로 낸 확정 숫자다(지도 본부: 같은 시험지 258건, 비지도 본부: 같은 1,797장). 편 번호를 누르면 그 면접으로 간다. 보류는 "이 일, 이 조건과 맞지 않는다"는 뜻일 뿐 모델의 우열이 아니다. 보류된 지원자도 조건을 바꾸면(좋은 특징, 다른 전처리) 합격선을 넘었다.'+(est.length?' <span class="est">예상</span> 표시('+est.join(', ')+')는 그 편의 면접 발표가 확정되기 전에 작업 지시서의 숫자로 적은 값이다. 특히 5-12 DBSCAN은 잡음으로 남긴 글씨의 비율을 어떻게 셈하느냐에 따라 판정이 달라질 수 있다.':'');})();

/* ══ 확인 문제 ══ */
var QUIZ={
q1:[{q:'t-SNE와 UMAP이 2D 지도를 그릴 때 가장 먼저 지키려는 것은?',o:[['64칸에서 가까운 이웃이 지도에서도 가까이 있는 것',1,'둘 다 "이웃 관계"를 지킨다. 먼 글씨끼리의 거리는 덜 챙긴다.'],['64칸의 모든 거리를 그대로 지키는 것',0,'2칸에 64칸의 모든 거리를 담을 수는 없다. 그나마 비슷하게 하려는 것은 PCA나 MDS 쪽이다.'],['숫자 정답이 같은 글씨끼리 모으는 것',0,'정답을 보지 않는다. 같은 숫자끼리 모인 것은 64칸에서 이미 이웃이었기 때문이다.']]},
 {q:'PCA 지도와 t-SNE 지도의 차이로 맞는 것은?',o:[['PCA는 곧은 축 두 개로 투영하고, t-SNE는 점마다 따로 옮겨 휘고 늘어난 지도를 만든다',1,'PCA는 직선(선형) 투영, t-SNE·UMAP은 비선형이다. 그래서 PCA 지도의 축에는 뜻이 있고, t-SNE 지도의 축에는 뜻이 없다.'],['둘 다 같은 일을 하고 속도만 다르다',0,'지도를 만드는 원리가 다르고, 이 면접에서 5-NN도 60.3% 대 97.6%로 다르다.'],['t-SNE는 정답을 쓰고 PCA는 쓰지 않는다',0,'둘 다 정답을 쓰지 않는 비지도 학습이다.']]},
 {q:'"t"-SNE의 t는 무엇을 뜻하나?',o:[['지도 위 이웃 확률을 잴 때 쓰는 꼬리가 두꺼운 스튜던트 t-분포',1,'먼 점의 확률이 천천히 줄어 무리끼리 뭉개지지 않게 벌려 준다(밀집 문제를 푼다).'],['시간(time)',0,'걸음을 거듭하지만 이름의 t는 분포 이름이다.'],['나무(tree)',0,'반스-헛 근사가 나무 구조를 쓰지만 이름과는 관계없다.']]}],
q2:[{q:'비지도 본부는 왜 시험지를 따로 떼지 않고 1,797장 전체로 지도를 그리나?',o:[['정답을 학습에 쓰지 않으므로 정답이 새어 들어갈 길이 없기 때문',1,'정답은 채점 봉투에만 있다. 지도를 다 그린 뒤 채점할 때 한 번 연다.'],['데이터가 적어서',0,'1,797장은 적지 않다. 이유는 정답을 쓰지 않기 때문이다.'],['t-SNE는 원래 시험지가 필요 없어서',0,'지원자 때문이 아니라 "정답 없이 배운다"는 비지도 학습의 규칙 때문이다.']]},
 {q:'매뉴얼 씨 지도(잉크 양, 위쪽 비율)의 5-NN이 18.1%에 그친 큰 까닭은?',o:[['두 숫자만으로는 다른 숫자끼리 겹치고, 잉크 양이 거리를 거의 다 정하기 때문',1,'잉크 양은 수백 단위, 위쪽 비율은 0~1이라 거리에서 위쪽 비율이 거의 무시된다. 표준화하면 27.0%로 오르지만 여전히 낮다.'],['5-NN이 원래 나쁜 잣대라서',0,'같은 잣대로 t-SNE 지도는 97.6%다.'],['정답을 보지 않아서',0,'t-SNE·UMAP도 정답을 보지 않는다.']]}],
q3:[{q:'Colab에서 pip install umap 뒤 umap.UMAP()을 부르면?',o:[['umap.UMAP이 없다는 오류가 난다. 설치 이름은 umap-learn이다',1,'PyPI의 umap은 다른 패키지다. 불러오는 이름(import umap)만 같다.'],['잘 돈다',0,'이름만 같은 다른 패키지가 깔린다.'],['scikit-learn의 UMAP이 불린다',0,'scikit-learn에는 UMAP이 없다.']]},
 {q:'umap.UMAP().fit_transform(X, y=y)로 그린 지도의 5-NN 점수를 비지도 지원자의 성적으로 보고하면 안 되는 까닭은?',o:[['정답(y)으로 같은 숫자끼리 끌어당긴 지도학습 지도이기 때문',1,'채점 봉투를 미리 연 셈이다. 비지도 면접에서는 fit_transform(X)만 쓴다.'],['y를 넣으면 오류가 나서',0,'오류 없이 돈다. 그래서 알아채기 어렵다.'],['y를 넣으면 느려져서',0,'속도 문제가 아니라 정답이 새어 들어간 문제다.']]},
 {q:'새로 들어온 글씨를 그려 둔 t-SNE 지도에 놓으려면?',o:[['scikit-learn TSNE에는 transform이 없다. 합쳐서 다시 그리거나 UMAP transform을 쓴다',1,'t-SNE는 점의 자리만 정하고 "64칸 → 지도" 규칙을 남기지 않는다.'],['tsne.transform(X_new)',0,'AttributeError가 난다.'],['tsne.fit_transform(X_new)',0,'새 글씨만으로 전혀 다른 지도를 새로 그린다. 기존 지도와 좌표가 맞지 않는다.']]}],
q4:[{q:'t-SNE의 perplexity를 5에서 100으로 키우면 대개?',o:[['더 넓게 이웃을 봐서 무리가 둥글게 뭉치고 섬들이 서로 가까워진다',1,'04장 재생에서 비교해 보자. 작으면 작은 섬이 많고, 크면 큰 덩어리가 된다.'],['정답을 더 많이 본다',0,'perplexity는 정답과 관계없다. 이웃 수의 어림이다.'],['아무것도 바뀌지 않는다',0,'지도의 모양이 꽤 바뀐다. 그래서 여러 값으로 그려 본다.']]},
 {q:'처음 250걸음 동안 이웃 끌림을 12배로 키우는(조기 과장) 까닭은?',o:[['같은 무리를 먼저 뭉쳐 놓아, 무리가 엉뚱한 곳에 갈라져 갇히지 않게 하려고',1,'과장을 풀면 무리 사이가 벌어지며 지도가 정리된다. KL 곡선이 250걸음 즈음 한 번 바뀐다.'],['계산을 빠르게 하려고',0,'걸음 수는 같다. 지도의 모양을 잡는 장치다.'],['정답을 12번 확인하려고',0,'정답은 쓰지 않는다.']]},
 {q:'UMAP의 min_dist를 0.0에서 0.99로 키우면?',o:[['점끼리 덜 붙어 무리가 넓고 고르게 퍼진다',1,'보기에는 편하지만 무리 사이 틈이 좁아진다. 군집에 넣을 지도라면 작게 둔다.'],['이웃 수가 늘어난다',0,'이웃 수는 n_neighbors다.'],['무리가 더 촘촘해진다',0,'반대다. 촘촘한 무리는 min_dist가 작을 때다.']]}],
q5:[{q:'t-SNE 지도에서 숫자 1의 무리가 숫자 4의 무리보다 크게 그려졌다. 맞는 해석은?',o:[['지도의 무리 크기는 믿지 않는다. 64칸에서 확인해야 한다',1,'05장 표: 이 면접의 UMAP은 무리 크기 비 9.2배, 64칸은 1.5배다. 지도가 크기를 부풀리거나 줄인다.'],['1을 쓰는 방식이 4보다 훨씬 다양하다',0,'그럴 수도 있지만 지도만 보고는 알 수 없다.'],['1이 데이터에 더 많다',0,'개수와 무리 크기는 다르다.']]},
 {q:'trustworthiness가 "정답 없이 재는 점수"인 까닭은?',o:[['지도 위 이웃이 원래 64칸에서도 이웃이었는지만 보기 때문',1,'숨긴 정답이 없어도 잴 수 있다. 5-NN 정확도는 정답이 있어야 잰다.'],['정답의 개수를 세기 때문',0,'정답을 쓰지 않는다.'],['지도의 넓이를 재기 때문',0,'넓이는 의미 없다. 이웃의 순위를 본다.']]},
 {q:'t-SNE 지도 위에서 K-Means를 돌려 ARI 0.88이 나왔다. 조심할 점은?',o:[['t-SNE가 무리를 일부러 벌려 그렸으므로 후한 점수다. 원래 공간에서 다시 확인한다',1,'"반칙에 가깝다". 묶음 수를 정하는 도움으로는 좋지만, 지도 모양은 근무 조건마다 바뀐다.'],['아무 문제 없다. 5-10 K-Means보다 좋은 군집이다',0,'지도는 이웃 관계만 지키고 거리·밀도를 바꾼다. 그 위의 군집은 지도에 기대고 있다.'],['ARI는 지도에서 잴 수 없다',0,'잴 수는 있다. 해석이 문제다.']]}],
q6:[{q:'매일 새 손글씨가 들어오는 현장에 지도를 배치하려면?',o:[['UMAP: 그려 둔 지도에 transform으로 새 글씨를 놓는다',1,'05장: 450장을 놓아도 5-NN 96.9%. t-SNE는 다시 그려야 한다.'],['t-SNE: 매번 처음부터 다시 그린다',0,'할 수는 있지만 매번 시간이 들고 기존 점들의 자리도 바뀔 수 있다.'],['어느 쪽이든 같다',0,'새 데이터 처리에서 크게 다르다.']]},
 {q:'"지도에서 4와 9 무리가 가까우니 4와 9는 비슷한 글씨다"라는 보고서 문장은?',o:[['지도 위 무리 사이 거리는 근무 조건마다 바뀌므로 그대로 쓰면 안 된다. 64칸에서 확인한다',1,'05장 표에서 중심 거리 순위 상관은 지도와 조건에 따라 0.45~0.83으로 흔들린다.'],['맞다. 지도는 거리를 지킨다',0,'가까운 이웃만 지킨다. 무리 사이 거리는 지키지 않는다.'],['틀렸다. 4와 9는 전혀 비슷하지 않다',0,'실제로 닮았을 수도 있다. 지도만으로는 판단할 수 없다는 것이 요점이다.']]}]};

/* ══ 공통 B·C·D·E·F 데이터 ══ */
var PROMPTS=[
 {t:'① 내 데이터로 지도 그리기',use:'내 센서·이미지·설문 데이터로 03장의 프롬프트를 맞출 때',p:MAIN_PROMPT},
 {t:'② AI 코드 검증 요청',use:'받은 코드를 다른 AI(또는 같은 AI의 새 대화)에게 검토받을 때',p:'아래 t-SNE · UMAP 코드를 코드 리뷰해 줘. 특히 다음을 하나씩 확인하고, 문제가 있으면 줄 번호와 고친 코드를 보여 줘.\n1) umap-learn을 바른 이름으로 설치했나 (pip install umap은 다른 패키지)\n2) TSNE 인자 이름이 내 버전(〔1.x〕)에 있는 이름인가 (n_iter는 지금 오류, max_iter를 쓴다)\n3) 지도를 만들 때 정답(y)을 넣지 않았나 (UMAP fit_transform(X, y=y)는 지도학습이 된다)\n4) t-SNE에 transform을 쓰지 않았나\n5) 지도 점수를 학습에 쓴 점으로 재지 않았나 (교차 검증을 썼나)\n6) 지도 위 무리 사이 거리나 크기로 결론을 내린 문장이 있나\n추측으로 답하지 말고, 확실하지 않은 부분은 "공식 문서 확인 필요"라고 표시해 줘.'},
 {t:'③ 근무 조건 여러 장 비교하기',use:'지도 한 장에 속지 않으려고 조건을 바꿔 함께 볼 때',p:'아래 Colab 코드와 같은 데이터로, 다음 조건의 지도를 한 화면(2행 4열)에 그리는 셀을 덧붙여 줘.\n- t-SNE perplexity 〔5, 30, 50, 100〕 (random_state=0)\n- UMAP n_neighbors 〔5, 15, 50, 200〕 (min_dist=0.1, random_state=0)\n규칙:\n1) 지도를 그릴 때 정답을 쓰지 않는다. 색칠은 채점용이라고 제목에 적는다\n2) 각 지도 아래에 5-NN 5겹 정확도와 trustworthiness(n_neighbors=5)를 적는다\n3) 숫자별 무리의 평균 반지름으로 "가장 큰 무리 ÷ 가장 작은 무리"를 계산해 적고, 원래 공간의 같은 값과 비교한다\n4) 모든 지도에 공통으로 보이는 무리와, 조건에 따라 바뀌는 배치를 나눠서 세 줄로 정리한다'},
 {t:'④ 수업용 문제 만들기',use:'이 편으로 수업할 때 확인 문제가 더 필요할 때',p:'t-SNE와 UMAP 입문 수업용 문제를 만들어 줘. 대상은 〔제조 현장 실무자, 수식에 익숙하지 않음〕이다. 예시는 손글씨 숫자(8×8 픽셀)로 통일한다.\n- perplexity · n_neighbors · min_dist · transform을 각각 묻는 객관식 4문제 (보기 3개, 정답 해설 포함)\n- 오해를 드러내는 문제 2개: "지도 위 무리 사이 거리는 실제 차이다", "t-SNE 지도의 축에는 뜻이 있다"\n- 손으로 풀 수 있는 문제 1개: 점 다섯 개의 2D 좌표와 원래 공간에서의 가까운 이웃 목록을 주고, 지도 위 가까운 이웃 2개가 원래 이웃과 몇 개 겹치는지(이웃 보존율) 세기\n정답은 맨 끝에 따로 모아 줘.'}];
var GL=[
 ['모델','t-SNE','t-분포 확률적 이웃 임베딩(t-distributed Stochastic Neighbor Embedding). 원래 공간의 이웃 확률과 지도 위 이웃 확률(t-분포)의 차이(KL 발산)를 줄이도록 점을 옮겨 2·3차원 지도를 만든다(van der Maaten & Hinton, 2008).'],
 ['모델','UMAP','균일 매니폴드 근사와 투영(Uniform Manifold Approximation and Projection). 가까운 이웃끼리 이은 그물을 저차원에서 다시 펴는 차원 축소. t-SNE보다 빠르고 새 데이터를 놓을 수 있다(McInnes 등, 2018).'],
 ['모델','PCA','주성분 분석. 가장 넓게 퍼진 방향부터 곧은 축을 골라 투영한다. 5-14 지원자.'],
 ['모델','차원 축소','특징(칸)의 수를 줄이는 일. 여기서는 64칸을 2칸으로.'],
 ['모델','비선형 차원 축소','곧은 축 투영이 아니라 점마다 따로 옮겨, 휘고 접힌 구조를 펴는 차원 축소. t-SNE, UMAP, Isomap 등.'],
 ['모델','매니폴드 학습','고차원 데이터가 사실은 휘어진 낮은 차원의 면(매니폴드) 위에 놓여 있다고 보고 그 면을 펴는 방법들.'],
 ['모델','임베딩','데이터를 다른(대개 더 낮은) 공간의 좌표로 옮긴 것. 이 편의 2D 지도 좌표.'],
 ['모델','비지도 학습','정답 없이 데이터의 구조만으로 배우는 학습. 군집, 차원 축소.'],
 ['모델','K-Means','묶음 중심 k개를 두고, 가까운 중심에 배정하고 중심을 평균으로 옮기기를 되풀이하는 군집. 5-10 지원자.'],
 ['학습','perplexity','t-SNE가 한 점마다 챙기는 이웃 수의 어림. 이 값에 맞게 점마다 가우시안 폭 σ를 정한다. 기본값 30.'],
 ['학습','n_neighbors','UMAP이 한 점마다 줄로 잇는 이웃 수. 작으면 동네, 크면 전체 구조를 본다. 기본값 15.'],
 ['학습','min_dist','UMAP 지도에서 점끼리 붙을 수 있는 최소 간격. 작으면 촘촘한 무리. 기본값 0.1.'],
 ['학습','이웃 확률','두 점이 서로의 이웃일 정도를 확률로 나타낸 것. t-SNE는 원래 공간에서는 가우시안, 지도에서는 t-분포로 잰다.'],
 ['학습','t-분포','스튜던트 t-분포. 가우시안보다 꼬리가 두꺼워 먼 거리에도 작은 확률이 남는다. t-SNE 지도에서 무리끼리 뭉개지지 않게 한다.'],
 ['학습','KL 발산','두 확률 분포가 얼마나 다른지 재는 값. Σ p log(p/q). 0이면 같다. t-SNE가 줄이는 손실.'],
 ['학습','조기 과장','t-SNE의 처음 몇백 걸음 동안 원래 공간의 이웃 확률을 몇 배(기본 12) 부풀려 무리를 먼저 잡는 장치. early exaggeration.'],
 ['학습','반스-헛 근사','먼 점들을 나무 구조로 묶어 한 덩어리처럼 계산해 t-SNE를 빠르게 하는 근사. scikit-learn 기본값.'],
 ['학습','이웃 그래프','점마다 가장 가까운 몇 점과 선으로 이은 그물. k-NN 그래프. UMAP의 출발점.'],
 ['학습','음의 표본','UMAP이 밀어낼 점을 모두 세지 않고 무작위로 몇 개만 뽑아 쓰는 방법. 계산을 크게 줄인다.'],
 ['학습','transform','학습한 규칙으로 새 데이터를 같은 공간에 옮기는 일. UMAP·PCA에는 있고 scikit-learn t-SNE에는 없다.'],
 ['평가','채점 봉투','이 시리즈의 비지도 본부에서 숨긴 정답(0~9)을 부르는 말. 지도를 다 그린 뒤 채점할 때만 연다.'],
 ['평가','5-NN 정확도','지도 좌표만 보고 가장 가까운 다섯 점의 정답 다수결로 맞힌 비율(5겹 교차 검증). 차원축소팀의 잣대.'],
 ['평가','교차 검증','데이터를 몇 몫으로 나눠, 한 몫씩 돌아가며 채점용으로 두고 나머지로 맞히는 방법. 학습한 점으로 채점하는 실수를 막는다.'],
 ['평가','trustworthiness','지도 위 이웃이 원래 공간에서도 이웃이었는지를 재는 점수(Venna & Kaski, 2001). 1이면 완벽. 정답이 필요 없다.'],
 ['평가','이웃 보존율','원래 공간의 가장 가까운 k점 중 지도에서도 가장 가까운 k점 안에 든 비율. 이 페이지에서는 k = 5.'],
 ['평가','ARI','수정 랜드 지수. 군집 결과와 숨긴 정답이 얼마나 겹치나. 1이면 완벽, 무작위면 0 근처. 군집팀의 잣대.'],
 ['평가','업무 점수','묶음마다 대표 한 장에만 사람이 이름을 붙이고 나머지에 퍼뜨렸을 때 맞는 비율. 라벨링 비용을 얼마나 줄이나.'],
 ['평가','순위 상관','두 목록의 크기 순서가 얼마나 같은지(스피어만). 1이면 순서가 같다. 05장에서 무리 사이 거리 순서를 비교한다.'],
 ['평가','기준선','배우지 않고도 낼 수 있는 점수. 여기서는 늘 가장 흔한 숫자를 답하는 10.1%.'],
 ['평가','합격선','지원자가 넘어야 할 점수. 차원축소팀은 매뉴얼 씨 지도의 5-NN 18.1%.'],
 ['데이터','픽셀','그림의 한 칸. 손글씨 숫자 한 장은 8×8 = 64픽셀이고 칸마다 밝기 0~16이 적혀 있다.']];
var ANALOGY=[
 ['이웃 지키기','반 자리 배치표: 친한 친구끼리는 꼭 옆자리에 두고, 먼 사이끼리 몇 칸 떨어지는지는 신경 쓰지 않는 담임',2,'뇌의 감각 지도: 몸이나 망막에서 이웃한 곳이 대뇌 피질에서도 이웃하게 배치된다(체성감각·시각 피질의 지형 지도, Penfield & Rasmussen, 1950)','뇌의 지도는 발달 중 신경 연결이 자라며 생긴다. KL 발산을 줄이는 계산이라는 근거는 없다. 자기 조직화 지도(Kohonen, 1982)가 이 현상을 본뜬 알고리즘이다'],
 ['같은 것끼리 모이기 (초파리)','같은 냄새 정보를 받는 사원들이 한 사무실로 모이는 회사',2,'초파리 더듬이엽: 같은 후각 수용체를 가진 감각 뉴런들이 한 사구체로 모인다(Vosshall 등, 2000)','초파리는 수용체 종류라는 "정해진 이름표"로 모이도록 유전적으로 배선된다. 정답 없이 모이는 t-SNE보다는 오히려 정답을 쓴 지도학습 UMAP에 가깝다'],
 ['무리 크기를 믿지 말 것','지하철 노선도: 도심 구간은 크게, 외곽은 작게 그려 실제 거리와 면적이 맞지 않는다',2,'피질 확대: 손가락과 시야 가운데처럼 자주 쓰는 곳이 피질 지도에서 훨씬 크게 그려진다','뇌는 쓰임(감각 수용기 밀도)에 따라 크기를 바꾸고, t-SNE·UMAP은 이웃의 밀도와 계산 방식 때문에 바꾼다. 까닭이 다르다'],
 ['perplexity · n_neighbors','"이웃"의 범위: 시골에서는 몇 킬로미터 안이 이웃이고, 아파트에서는 같은 층이 이웃이다',1,'','뇌가 이웃의 범위를 이런 값으로 정한다는 근거는 알려지지 않았다'],
 ['t-SNE에 transform이 없다','손으로 그린 동네 약도: 새 가게가 생기면 약도를 처음부터 다시 그려야 하고, 다시 그리면 다른 가게 자리도 조금씩 옮겨진다',1,'','직접 대응하는 뇌 기능은 알려지지 않았다'],
 ['정답 없이 대표 몇 장에 이름 붙이기','창고 정리: 비슷한 상자끼리 쌓아 두고 더미마다 하나만 열어 이름표를 붙인다',1,'','사람이 적은 이름표로 범주를 배우는 일(준지도 학습)과 닮았지만, 뇌의 방식과 같다는 근거는 없다']];
var REFS=[
 'Alpaydin, E., &amp; Kaynak, C. (1998). Optical Recognition of Handwritten Digits [Dataset]. UCI Machine Learning Repository. <a href="https://doi.org/10.24432/C50P49" target="_blank" rel="noopener">doi:10.24432/C50P49</a> (CC BY 4.0) — 실기 과제지(scikit-learn load_digits)',
 'van der Maaten, L., &amp; Hinton, G. (2008). Visualizing data using t-SNE. <i>Journal of Machine Learning Research</i>, 9, 2579–2605. — t-SNE, 00·01·04장',
 'Hinton, G., &amp; Roweis, S. (2002). Stochastic neighbor embedding. In <i>Advances in Neural Information Processing Systems 15</i>. — t-SNE의 앞선 형태 SNE',
 'van der Maaten, L. (2014). Accelerating t-SNE using tree-based algorithms. <i>Journal of Machine Learning Research</i>, 15, 3221–3245. — 반스-헛 근사, 04장',
 'McInnes, L., Healy, J., &amp; Melville, J. (2018). UMAP: Uniform manifold approximation and projection for dimension reduction. <i>arXiv:1802.03426</i>. — UMAP, 00·04·06장',
 'Wattenberg, M., Viégas, F., &amp; Johnson, I. (2016). How to use t-SNE effectively. <i>Distill</i>. <a href="https://doi.org/10.23915/distill.00002" target="_blank" rel="noopener">doi:10.23915/distill.00002</a> — 거리와 크기를 믿지 말 것, 05장',
 'Kobak, D., &amp; Berens, P. (2019). The art of using t-SNE for single-cell transcriptomics. <i>Nature Communications</i>, 10, 5416. — perplexity와 시작 자리, 04·06장',
 'Kobak, D., &amp; Linderman, G. C. (2021). Initialization is critical for preserving global data structure in both t-SNE and UMAP. <i>Nature Biotechnology</i>, 39, 156–157. — PCA 시작, 06장',
 'Venna, J., &amp; Kaski, S. (2001). Neighborhood preservation in nonlinear projection methods: An experimental study. In <i>Artificial Neural Networks — ICANN 2001</i> (LNCS 2130, pp. 485–491). Springer. — trustworthiness, 05장',
 'Hubert, L., &amp; Arabie, P. (1985). Comparing partitions. <i>Journal of Classification</i>, 2, 193–218. — ARI, 05장',
 'Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. <i>Journal of Machine Learning Research</i>, 12, 2825–2830. — A장 코드',
 'scikit-learn 문서. <a href="https://scikit-learn.org/stable/modules/manifold.html" target="_blank" rel="noopener">Manifold learning</a> · <a href="https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html" target="_blank" rel="noopener">TSNE</a> — 인자 이름과 기본값의 변화, 03·06장',
 'umap-learn 문서. <a href="https://umap-learn.readthedocs.io/" target="_blank" rel="noopener">UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction</a> — 설치 이름, n_neighbors·min_dist, transform, 03·06장',
 'Penfield, W., &amp; Rasmussen, T. (1950). <i>The Cerebral Cortex of Man</i>. Macmillan. — D장 감각 지도',
 'Kohonen, T. (1982). Self-organized formation of topologically correct feature maps. <i>Biological Cybernetics</i>, 43, 59–69. — D장 자기 조직화 지도',
 'Vosshall, L. B., Wong, A. M., &amp; Axel, R. (2000). An olfactory sensory map in the fly brain. <i>Cell</i>, 102(2), 147–159. — D장 초파리 더듬이엽'];
var TIP={who:'머신러닝을 처음 배우는 대학생·직장인·제조 현장 실무자. 5-1 KNN(이웃으로 맞히기)과 5-14 PCA(곧은 그림자)를 먼저 보면 좋다. 수식은 이웃 확률 두 개(P, Q)와 KL 발산이 중심이고, 몰라도 그림으로 따라갈 수 있다',
 goal:'① "이웃을 지킨다, 먼 거리는 포기한다"를 01장 그림으로 설명한다 ② 04장 재생으로 조기 과장과 perplexity의 효과를 보인다 ③ 채점 봉투(숨긴 정답)로 지도를 채점하는 법과, 정답 없이 재는 점수(trustworthiness)를 구별한다 ④ 지도 위 무리 사이 거리와 크기를 믿지 않는 습관, 새 데이터(transform)와 비용의 차이를 설명한다 ⑤ AI가 낸 코드에서 패키지 이름, 지워진 인자, 정답 누수를 찾는다',
 time:'읽기 25분 · 실습(01·04·05장 그림) 25분 · 틀린 곳 찾기 10분 · Colab 15분(UMAP 첫 실행 컴파일 포함)',
 plans:[['강의형 50분','00장 지원서 1장을 인쇄해 1~4번 칸을 함께 읽기 8분 → 01장에서 학생마다 점 하나를 골라 t-SNE와 PCA 지도에서 이웃이 몇 장 남는지 세어 발표 10분 → 04장 재생을 perplexity 5·30·100으로 돌리며 "무엇이 바뀌고 무엇이 안 바뀌나" 토론 12분 → 05장 채점 봉투를 함께 열고 "거리와 크기" 표 읽기 12분 → 면접 결과 맞히고 발표, 마무리 15명 결산 8분.'],['실무 OJT 90분','03장 프롬프트로 각자 AI에게 코드를 받는다 20분 → 체크리스트로 서로의 코드 검사(특히 패키지 이름과 y 누수) 15분 → Colab에서 실행하고 A장 결과와 비교 20분 → 자기 부서 데이터(센서 로그, 검사 이미지 특징)로 B장 ③ 프롬프트를 돌려 조건 여러 장을 비교하고 "모든 그림에 공통인 무리"를 찾기 35분.'],['1:1 튜터링','01장에서 점을 하나 고르게 하고, 64칸 이웃 다섯 장이 지도에서 몇 위인지 말로 설명하게 한다. 이어 05장 "믿지 말아야 할 것"에서 지도를 바꿔 가며 무리 크기 비가 어떻게 바뀌는지 예측하고 확인하게 한다. 마지막으로 "이 지도로 보고서에 쓸 수 있는 문장과 쓰면 안 되는 문장"을 하나씩 쓰게 한다.']],
 myths:[['지도 위 무리 사이 거리는 실제 차이다','가까운 이웃만 지킨다. 무리 사이 거리는 근무 조건마다 바뀐다(05장 순위 상관 0.45~0.83)'],['무리가 크면 그 숫자 글씨가 다양하다','지도가 크기를 부풀리거나 줄인다. UMAP 크기 비 9.2배 대 64칸 1.5배'],['t-SNE 지도의 가로·세로 축에 뜻이 있다','축에는 뜻이 없다. 지도를 돌리거나 뒤집어도 같은 지도다. 축에 뜻이 필요하면 PCA'],['정답 색으로 칠해 보니 잘 나뉘었으니 정답을 쓴 것이다','지도는 정답 없이 그렸다. 색은 그린 뒤에 칠한 채점이다. 단, UMAP에 y를 넣으면 정말로 정답을 쓴 것이 된다'],['한 번 그린 지도가 정답이다','조건(perplexity, n_neighbors, min_dist)과 시작 자리에 따라 바뀐다. 여러 장을 함께 보고 공통인 것만 믿는다'],['t-SNE 지도 위에서 군집하면 가장 좋은 군집이다','무리를 일부러 벌린 지도라 점수가 후하다. 원래 공간에서 확인한다']],
 tasks:[['과제 1 · 이웃 세기','01장에서 점 다섯 개를 골라, 지도(t-SNE, UMAP, PCA, 매뉴얼)마다 64칸 이웃 다섯 장이 지도 위 몇 위인지 표로 적는다. 지도별 평균을 내고 05장의 이웃 보존율과 비교한다.'],['과제 2 · 틀린 곳 찾기 만들기','AI에게 일부러 실수가 섞인 t-SNE·UMAP 코드를 받고(B장 ②를 거꾸로 활용), 짝에게 찾게 한다. 실수마다 "왜 틀렸나"를 한 줄로 쓴다. 정답 누수를 하나 이상 넣는다.'],['과제 3 · 조건 여러 장','B장 ③ 프롬프트로 지도 여덟 장을 그리고, 모든 지도에 공통인 무리와 조건마다 바뀌는 배치를 나눠 쓴다. 보고서에 쓸 수 있는 문장 세 개를 고른다.'],['과제 4 · 라벨링 비용 계산','05장 업무 점수를 대표 10·20·30장으로 바꿔 보고, "이름표 한 장 붙이는 데 30초"라 할 때 1,797장 전부와 대표만 붙이는 경우의 시간과 정확도를 비교한다.']],
 ai:'학생이 AI에게 코드를 받는 것은 막지 말고 과정에 넣는다. t-SNE·UMAP은 패키지 이름과 바뀐 인자 이름, 정답 누수(UMAP의 y)가 섞이기 쉬워 검증 연습에 좋다. ① 프롬프트를 제출하게 하고 ② 체크리스트 아홉 항목을 직접 표시하게 하고 ③ 지도마다 5-NN과 trustworthiness를 합격선과 비교한 문장, 그리고 "지도에서 결론 내면 안 되는 것" 한 문장을 쓰게 한다. AI 답의 함수·인자는 설치된 버전의 공식 문서로 확인하게 하고, 과제에 AI 도움을 받았으면 사용 범위를 밝히게 한다.'};
var CODE={title:'t-SNE · UMAP 면접 — 정답 없는 지도부터 합격 판정까지',sub:'numpy, scipy, scikit-learn, umap-learn · CPU 1분 남짓 (UMAP 첫 실행 컴파일 포함)',file:'tsne_umap_interview',
 desc:'이 편의 면접을 그대로 옮겼다. 손글씨 1,797장을 읽고 정답을 채점 봉투에 넣고(1번 칸), 매뉴얼 씨 지도로 합격선을 세우고(2번), PCA(3번), t-SNE(4번), UMAP(5번) 지도를 그린 뒤, 봉투를 열어 같은 잣대로 채점하고(6번), 근무 조건을 바꿔 보고(7번), 무리 사이 거리와 크기를 64칸과 비교하고(8번), 대표 몇 장으로 이름 붙이기 업무 점수(9번), 새 글씨를 UMAP transform으로 놓기(10번), 판정(11번)까지 간다. 정답(y)을 쓰는 칸에는 [채점]이라고 적었다. 노트북으로 저장하면 칸마다 셀이 나뉜다.',
 cap:'실행 결과 (scikit-learn 1.9.1, umap-learn 0.5.12, CPU 2코어). 시간(초)은 컴퓨터마다 다르고, UMAP의 첫 실행에는 numba 컴파일 시간이 섞인다. 이 페이지의 05장·면접 결과는 같은 지도 좌표로 브라우저가 다시 채점해 같은 값(t-SNE 97.6% · UMAP 97.6% · PCA 60.3% · 64칸 96.3% · transform 96.9%)을 낸다. 매뉴얼 씨 지도는 같은 거리의 이웃이 많아 브라우저에서 18.4%로 조금 다르게 나오므로 합격선은 이 결과의 18.1%를 쓴다. 9번 칸의 업무 점수는 브라우저 K-Means와 초기값이 달라 소수 둘째 자리에서 조금 다를 수 있다.'};
