var BOARD_LINE='ARI 0.108'; /* 군집팀 합격선: 매뉴얼 씨 규칙 8더미의 ARI (CLU_LINE) */
/* ══ 5-10 K-Means (+ K-Medoids) · 손글씨 숫자 실기 과제지 ══════════════════════════ */
/* KM-ENGINE-START */
/* ── K-Means · K-Medoids · 채점 엔진 (이 편 전용, 브라우저·node 공용) ──
   · lloyd: scikit-learn KMeans(algorithm="lloyd")와 같은 규칙. 가까운 중심에 배정 → 묶음 평균으로 중심 이동.
     배정이 그대로이거나(엄격한 수렴) 중심 이동량 제곱합 ≤ tol(= 1e-4 × 특징별 분산의 평균)이면 멈춘다
   · kpp: k-means++ 시작(scikit-learn처럼 후보 2 + ln k개 중 관성을 가장 줄이는 점을 고른다). 난수는 이 페이지의 것
   · kmedoids: 대표가 실제 글씨인 K-Medoids. BUILD로 시작하고(PAM 방식, 난수 없음) 배정 ↔ 묶음 안 거리 합이 가장 작은 글씨로 교체를 되풀이(alternate)
   · ari, nmi(산술 평균), silhouette: scikit-learn의 adjusted_rand_score, normalized_mutual_info_score, silhouette_score와 같은 정의 */
var KM=(function(){
  function d2(a,b){var s=0;for(var j=0;j<a.length;j++){var t=a[j]-b[j];s+=t*t;}return s;}
  function flat(X){if(X._f)return X._f;var n=X.length,d=X[0].length,F=new Float64Array(n*d);for(var i=0;i<n;i++)for(var j=0;j<d;j++)F[i*d+j]=X[i][j];X._f=F;return F;}
  /* 가까운 중심에 배정. 반환: 관성(가장 가까운 중심까지 거리 제곱의 합) */
  function assign(X,C,lab){var n=X.length,d=X[0].length,k=C.length,F=flat(X),CF=new Float64Array(k*d),inn=0;for(var c=0;c<k;c++)for(var j=0;j<d;j++)CF[c*d+j]=C[c][j];
    for(var i=0;i<n;i++){var bi=0,bd=Infinity,o=i*d;for(c=0;c<k;c++){var s=0,q=c*d;for(j=0;j<d;j++){var t=F[o+j]-CF[q+j];s+=t*t;}if(s<bd){bd=s;bi=c;}}lab[i]=bi;inn+=bd;}return inn;}
  function means(X,lab,k,old){var d=X[0].length,F=flat(X),S=[],N=new Array(k).fill(0);for(var c=0;c<k;c++)S.push(new Float64Array(d));
    for(var i=0;i<X.length;i++){var c2=lab[i],s=S[c2],o=i*d;N[c2]++;for(var j=0;j<d;j++)s[j]+=F[o+j];}
    return S.map(function(s,c){if(!N[c])return Float64Array.from(old[c]);for(var j=0;j<d;j++)s[j]/=N[c];return s;});}
  function tolOf(X){var n=X.length,d=X[0].length,v=0;for(var j=0;j<d;j++){var m=0,q=0;for(var i=0;i<n;i++)m+=X[i][j];m/=n;for(i=0;i<n;i++)q+=(X[i][j]-m)*(X[i][j]-m);v+=q/n;}return 1e-4*v/d;}
  /* lloyd(X, 시작 중심 C0 또는 시작 글씨 번호, {maxIter, trace}) → {lab, C, inertia, iters, hist:[{C, inertia, changed}]} */
  function lloyd(X,init,o){o=o||{};var k=init.length,C=init.map(function(c){return typeof c==='number'?Float64Array.from(X[c]):Float64Array.from(c);}),n=X.length,
      lab=new Int32Array(n),old=new Int32Array(n).fill(-1),tol=o.tol==null?(X._tol!=null?X._tol:(X._tol=tolOf(X))):o.tol,hist=[],strict=false,it=0;
    for(it=0;it<(o.maxIter||300);it++){var inn=assign(X,C,lab),ch=0;for(var i=0;i<n;i++)if(lab[i]!==old[i])ch++;
      if(o.trace)hist.push({C:C.map(function(c){return Float64Array.from(c)}),inertia:inn,changed:ch,lab:Int32Array.from(lab)});
      var Cn=means(X,lab,k,C),sh=0;for(var c=0;c<k;c++)sh+=d2(C[c],Cn[c]);C=Cn;
      if(ch===0){strict=true;it++;break;}if(sh<=tol){it++;break;}old.set(lab);}
    var inertia=assign(X,C,lab);if(o.trace)hist.push({C:C.map(function(c){return Float64Array.from(c)}),inertia:inertia,changed:0,lab:Int32Array.from(lab)});
    return {lab:lab,C:C,inertia:inertia,iters:it,hist:hist,strict:strict};}
  /* k-means++ (난수 rnd: 0~1) — 시작 글씨 번호를 돌려준다 */
  function kpp(X,k,rnd){var n=X.length,trials=2+Math.floor(Math.log(k)),idx=[Math.floor(rnd()*n)],cl=new Float64Array(n);
    for(var i=0;i<n;i++)cl[i]=d2(X[i],X[idx[0]]);var pot=cl.reduce(function(a,b){return a+b},0);
    for(var c=1;c<k;c++){var best=-1,bp=Infinity,bcl=null;
      for(var t=0;t<trials;t++){var r=rnd()*pot,acc=0,cand=n-1;for(i=0;i<n;i++){acc+=cl[i];if(acc>=r){cand=i;break;}}
        var nc=new Float64Array(n),p=0;for(i=0;i<n;i++){nc[i]=Math.min(cl[i],d2(X[i],X[cand]));p+=nc[i];}if(p<bp){bp=p;best=cand;bcl=nc;}}
      idx.push(best);cl=bcl;pot=bp;}
    return idx;}
  /* 거리 행렬 (유클리드), Float32Array n×n */
  function dmat(X){var n=X.length,d=X[0].length,F=flat(X),D=new Float32Array(n*n);for(var i=0;i<n;i++){var a=i*d;for(var j=i+1;j<n;j++){var b=j*d,s=0;for(var q=0;q<d;q++){var t=F[a+q]-F[b+q];s+=t*t;}var v=Math.sqrt(s);D[i*n+j]=v;D[j*n+i]=v;}}D.n=n;return D;}
  function kmedoids(D,k,o){o=o||{};var n=D.n,med=[],dmin=new Float64Array(n),i,j,c;
    var best=-1,bs=Infinity;for(i=0;i<n;i++){var s=0;for(j=0;j<n;j++)s+=D[i*n+j];if(s<bs){bs=s;best=i;}}med.push(best);for(j=0;j<n;j++)dmin[j]=D[best*n+j];
    for(c=1;c<k;c++){var bg=-1,bi=-1;for(i=0;i<n;i++){if(med.indexOf(i)>=0)continue;var g=0,row=i*n;for(j=0;j<n;j++){var t=dmin[j]-D[row+j];if(t>0)g+=t;}if(g>bg){bg=g;bi=i;}}
      med.push(bi);for(j=0;j<n;j++)dmin[j]=Math.min(dmin[j],D[bi*n+j]);}
    var build=med.slice(),lab=new Int32Array(n),it=0,hist=[];
    function asg(){var cost=0;for(var q=0;q<n;q++){var b=0,bd=Infinity;for(var m=0;m<k;m++){var v=D[q*n+med[m]];if(v<bd){bd=v;b=m;}}lab[q]=b;cost+=bd;}return cost;}
    for(it=0;it<(o.maxIter||100);it++){var cost=asg();hist.push({med:med.slice(),cost:cost});var mem=[];for(c=0;c<k;c++)mem.push([]);for(i=0;i<n;i++)mem[lab[i]].push(i);
      var nw=mem.map(function(M,cc){var bb=med[cc],bv=Infinity;M.forEach(function(a){var s=0,row=a*n;for(var q=0;q<M.length;q++)s+=D[row+M[q]];if(s<bv){bv=s;bb=a;}});return bb;});
      var same=nw.every(function(v,q){return v===med[q]});if(same)break;med=nw;}
    var cost2=asg();return {med:med,build:build,lab:lab,cost:cost2,iters:it,hist:hist};}
  function silhouette(D,lab,k){var n=D.n,cnt=new Array(k).fill(0),s=0;for(var i=0;i<n;i++)cnt[lab[i]]++;
    for(i=0;i<n;i++){if(cnt[lab[i]]<2)continue;var sum=new Float64Array(k),row=i*n;for(var j=0;j<n;j++)sum[lab[j]]+=D[row+j];
      var a=sum[lab[i]]/(cnt[lab[i]]-1),b=Infinity;for(var c=0;c<k;c++)if(c!==lab[i]&&cnt[c])b=Math.min(b,sum[c]/cnt[c]);s+=(b-a)/Math.max(a,b);}
    return s/n;}
  function table(a,b){var A={},B={},T={},n=a.length;for(var i=0;i<n;i++){A[a[i]]=(A[a[i]]||0)+1;B[b[i]]=(B[b[i]]||0)+1;var key=a[i]+','+b[i];T[key]=(T[key]||0)+1;}return {A:A,B:B,T:T,n:n};}
  function c2(x){return x*(x-1)/2;}
  function ari(a,b){var t=table(a,b),sij=0,sa=0,sb=0,k;for(k in t.T)sij+=c2(t.T[k]);for(k in t.A)sa+=c2(t.A[k]);for(k in t.B)sb+=c2(t.B[k]);
    var e=sa*sb/c2(t.n),mx=(sa+sb)/2;return mx===e?1:(sij-e)/(mx-e);}
  /* NMI(산술 평균): 정답과 묶음이 얼마나 많은 정보를 나눠 갖나. 기댓값 보정은 하지 않는다(scikit-learn 기본) */
  function nmi(a,b){var t=table(a,b),n=t.n,ha=0,hb=0,mi=0,k;for(k in t.A){var p=t.A[k]/n;ha-=p*Math.log(p);}for(k in t.B){p=t.B[k]/n;hb-=p*Math.log(p);}
    for(k in t.T){var ab=k.split(','),pij=t.T[k]/n;mi+=pij*Math.log(pij/((t.A[ab[0]]/n)*(t.B[ab[1]]/n)));}var den=(ha+hb)/2;return den?Math.max(0,mi)/den:1;}
  return {d2:d2,assign:assign,lloyd:lloyd,kpp:kpp,dmat:dmat,kmedoids:kmedoids,silhouette:silhouette,ari:ari,nmi:nmi,tolOf:tolOf};})();
if(typeof module!=='undefined')module.exports=KM;
/* KM-ENGINE-END */

/* ── 실기 과제지: 손글씨 1,797장. X(64칸)만 학습에 쓴다 ── */
const DR=digitRows(),X=DR.map(r=>r.px),N=X.length;
/* 채점 봉투: 숨긴 정답. 05장 채점과 면접 판정에서만 연다(openEnvelope) */
const ENVELOPE=DR.map(r=>r.y);
function openEnvelope(){return ENVELOPE;}
const fmt3=v=>fx(v,3),comma=v=>Math.round(v).toLocaleString();
function axes(ctx,C,pl,pt,gw,gh){ctx.strokeStyle=C.rule;ctx.lineWidth=1;ctx.strokeRect(pl,pt,gw,gh);ctx.fillStyle=C.muted;ctx.font=font(12);}
function vlabel(ctx,txt,x,y){ctx.save();ctx.translate(x,y);ctx.rotate(-Math.PI/2);ctx.textAlign='center';ctx.fillText(txt,0,0);ctx.restore();}
function barRows(rows,max){return rows.map(r=>'<div class="row"><span style="text-align:left;font-family:var(--font)">'+r[0]+'</span><div class="tr"><i style="left:0;width:'+Math.max(0,Math.min(100,r[1]/max*100))+'%;background:'+(r[2]||'var(--blue)')+'"></i></div><span>'+r[3]+'</span></div>').join('');}
/* 글씨 한 장(또는 중심)을 s×s 칸에 그린다. col이 있으면 그 색 테두리 */
function thumb(ctx,px,x,y,s,C,col,lw){ctx.fillStyle=C.panel;ctx.fillRect(x,y,s,s);drawDigit(ctx,px,x,y,s,C.text);ctx.lineWidth=col?(lw||2):1;ctx.strokeStyle=col||C.rule;ctx.strokeRect(x+(col?1:.5),y+(col?1:.5),s-(col?2:1),s-(col?2:1));}
function whenNear(id,fn,margin){const t=$(id);let done=false;function go(){if(done)return;done=true;setTimeout(fn,10);}
  if('IntersectionObserver' in window){const io=new IntersectionObserver(es=>{if(es.some(e=>e.isIntersecting)){go();io.disconnect();}},{rootMargin:(margin||600)+'px'});io.observe(t);}else go();
  window.addEventListener('hashchange',()=>{const h=location.hash.slice(1),el=h&&$(h);if(el&&(el===t||t.contains(el)||t.compareDocumentPosition(el)&Node.DOCUMENT_POSITION_FOLLOWING))go();});
  if(location.hash){const el=$(location.hash.slice(1));if(el&&(el===t||t.contains(el)||t.compareDocumentPosition(el)&Node.DOCUMENT_POSITION_FOLLOWING))go();}}

/* ── 공용 계산 (한 번만 하고 기억) ── */
let DM=null;const KRES={};let MAIN=null,STD=null,MED=null;
function ensureD(){if(!DM){const t0=performance.now();DM=KM.dmat(X);DM.ms=performance.now()-t0;}return DM;}
/* k마다 Colab과 같은 시작점(PRE.init[k])에서 K-Means */
function runK(k){if(KRES[k])return KRES[k];const t0=performance.now(),R=KM.lloyd(X,PRE.init[k],{trace:k===10});R.ms=performance.now()-t0;R.k=k;KRES[k]=R;return R;}
function ensureMain(){if(!MAIN){MAIN=runK(10);if($('itN'))$('itN').textContent=MAIN.iters;}return MAIN;}
function silOf(R){if(R.sil==null)R.sil=KM.silhouette(ensureD(),R.lab,R.k);return R.sil;}
/* 업무 점수: 묶음마다 중심에 가장 가까운 글씨 한 장에 사람이 이름을 붙이고, 묶음 전체가 그 이름을 물려받는다 */
function reps(R){const k=R.C.length,best=new Array(k).fill(-1),bd=new Array(k).fill(Infinity);for(let i=0;i<N;i++){const c=R.lab[i],d=KM.d2(X[i],R.C[c]);if(d<bd[c]){bd[c]=d;best[c]=i;}}return best;}
function bizOf(R){if(R.biz)return R.biz;const y=openEnvelope(),rp=reps(R);let ok=0;const per=rp.map(()=>({n:0,ok:0}));
  for(let i=0;i<N;i++){const c=R.lab[i],r=rp[c];per[c].n++;if(r>=0&&y[r]===y[i]){ok++;per[c].ok++;}}return (R.biz={acc:ok/N,reps:rp,per});}
function ariOf(R){if(R.ari==null){const y=openEnvelope(),l=Array.from(R.lab);R.ari=KM.ari(y,l);R.nmi=KM.nmi(y,l);}return R.ari;}
function meanPx(lab,k){const S=[...Array(k)].map(()=>new Float64Array(64)),n=new Array(k).fill(0);for(let i=0;i<N;i++){const c=lab[i];n[c]++;for(let j=0;j<64;j++)S[c][j]+=X[i][j];}return S.map((s,c)=>{if(n[c])for(let j=0;j<64;j++)s[j]/=n[c];return s;});}
/* 기준선들 (채점 봉투가 열린 뒤에만 부른다) */
let BASES=null;
function ensureBases(){if(BASES)return BASES;const y=openEnvelope(),man=X.map(manualDigit),inks=X.map(p=>digitFeat(p).ink),lo=Math.min(...inks),hi=Math.max(...inks);
  const ink10=inks.map(v=>{let b=0;for(let i=1;i<10;i++)if(v>=lo+(hi-lo)*i/10)b++;return b;});const rnd=RNG(20251022);let ra=0;for(let t=0;t<20;t++)ra+=KM.ari(y,X.map(()=>Math.floor(rnd()*10)));
  BASES={man:KM.ari(y,man),manN:KM.nmi(y,man),ink:KM.ari(y,ink10),inkN:KM.nmi(y,ink10),rand:ra/20};return BASES;}
const MANNAME=c=>((c&4)?'가운데 빔':'가운데 참')+' · '+((c&2)?'잉크 많음':'잉크 적음')+' · '+((c&1)?'위가 무거움':'아래가 무거움');

/* ── 히어로: 서류 더미에서 중심 세 개 ── */
heroRoom($('heroCv'),{sym:'μ',no:'APP-10',name:'K-Means\n(군집)',role:'지원자 · 군집팀',banner:'모델 채용 위원회 · 5-10 면접',doc:'front',asst:'board',fly:'clip',robo:'calc',loop:11,
  beats:[{at:0.4,who:'asst',mood:'shock',move:'shake',say:'서류가 1,797장!'},{at:2.4,who:'doc',mood:'board',move:'nod',say:'중심 10개를 찍어요'},{at:4.4,who:'fly',mood:'point',move:'tilt',say:'정답도 없이?'},
         {at:6.4,who:'robo',mood:'count',move:'bob',say:'이름표 10장이면 돼?'},{at:8.4,who:'asst',mood:'spark',move:'jump',say:'평균 글씨가 보여요'}],
  motif:(function(){/* 그림용으로 고른 글씨 18장(세 무리). 흩어진 자리 → 중심 셋이 나타나 배정 → 중심이 평균으로 이동 → 무리 정돈 */
    const IDX=[0,10,20,30,36,48,4,14,24,41,64,68,7,17,27,43,44,52],G=[[300,118],[384,112],[340,182]],S0=[[350,104],[300,186],[392,170]],rnd=RNG(5);
    const P=IDX.map((ix,i)=>{const g=G[Math.floor(i/6)];return {ix,sx:g[0]+(rnd()-.5)*78,sy:g[1]+(rnd()-.5)*56,gx:g[0]+((i%6)%3-1)*23,gy:g[1]+(Math.floor((i%6)/3)-.5)*23};});
    P.forEach(p=>{p.sx=clamp(p.sx,280,402);p.sy=clamp(p.sy,94,198);});
    const near=(x,y,Cs)=>{let b=0,bd=1e9;Cs.forEach((c,k)=>{const d=(x-c[0])**2+(y-c[1])**2;if(d<bd){bd=d;b=k;}});return b;};
    return function(ctx,t,W,H,C){const T=REDUCE?7:(t%8),cl=[C.blue,C.stamp,C.ok];if(!X.length)return;
      const asg=P.map(p=>near(p.sx,p.sy,S0)),mean=S0.map((s,k)=>{const m=P.filter((p,i)=>asg[i]===k);return m.length?[m.reduce((a,p)=>a+p.sx,0)/m.length,m.reduce((a,p)=>a+p.sy,0)/m.length]:s;});
      const mv=ease(clamp((T-4.4)/1.4,0,1)),tidy=ease(clamp((T-6)/1.2,0,1));
      const Cs=S0.map((s,k)=>{const a=[s[0]+(mean[k][0]-s[0])*mv,s[1]+(mean[k][1]-s[1])*mv];return [a[0]+(G[k][0]-a[0])*tidy,a[1]+(G[k][1]-a[1])*tidy];});
      P.forEach((p,i)=>{const x=p.sx+(p.gx-p.sx)*tidy,y=p.sy+(p.gy-p.sy)*tidy,s=21;thumb(ctx,X[p.ix],x-s/2,y-s/2,s,C,T>=3?cl[asg[i]]:null,1.8);});
      if(T>=1.5){const a=REDUCE?1:clamp((T-1.5)/.5,0,1);Cs.forEach((c,k)=>{ctx.globalAlpha=a;ctx.strokeStyle=C.panel;ctx.lineWidth=6;ctx.beginPath();ctx.moveTo(c[0]-9,c[1]);ctx.lineTo(c[0]+9,c[1]);ctx.moveTo(c[0],c[1]-9);ctx.lineTo(c[0],c[1]+9);ctx.stroke();
        ctx.strokeStyle=cl[k];ctx.lineWidth=3.2;ctx.stroke();ctx.fillStyle=cl[k];ctx.font=font(13,700,1);ctx.textAlign='left';ctx.fillText('μ',c[0]+7,c[1]-6);ctx.globalAlpha=1;});}
      ctx.fillStyle=C.muted;ctx.font=font(12,500,1);ctx.textAlign='left';ctx.fillText(T<1.5?'정답 없는 서류 더미':T<3?'① 중심 찍기':T<4.4?'② 가까운 쪽으로':T<6?'③ 평균으로 옮기기':'묶음 3개 완성',272,226);};})()});

/* ── 01 장면 ── */
comic('comicIntro',[
 {bg:'office',prop:'서류 1,797장 · 중심 3개 찍기',cast:[{who:'asst',mood:'shock',x:28,move:'shake'},{who:'doc',mood:'board',x:72,move:'nod',h:104}],say:{who:'doc',txt:'아무 글씨나 세 장 골라 책상마다 하나씩 놓아요. 그게 첫 중심이에요.'},cap:'① 중심 찍기 — 처음엔 아무 데나'},
 {bg:'office',prop:'가장 닮은 책상으로',cast:[{who:'asst',mood:'ledger',x:30,move:'nod'},{who:'fly',mood:'point',x:74,move:'tilt',flip:true}],say:{who:'asst',txt:'한 장씩 보고 가장 닮은 책상에 올려요. 7은 7 옆으로!'},cap:'② 가까운 쪽으로 — 가장 가까운 중심의 묶음에 넣는다'},
 {bg:'office',prop:'책상마다 평균 글씨',cast:[{who:'doc',mood:'magic',x:28,move:'pop',h:104},{who:'robo',mood:'count',x:72,move:'bob',h:100}],say:{who:'robo',txt:'책상마다 평균 한 장만 남기면 되네. 외울 숫자 640개!'},cap:'③ 중심 옮기기 — 모인 글씨의 평균으로. 다시 ②로'}]);

/* ── 01 자기소개: 그림용 2차원 지도 위에서 K-Means 한 단계씩 ── */
/* 64칸을 두 축으로 줄이는 PCA(분산이 가장 큰 두 방향, 거듭제곱법). 그림을 그리려는 것뿐이다 — 5-14에서 자세히 */
function pca2(X){const n=X.length,d=X[0].length,mu=new Float64Array(d);X.forEach(r=>{for(let j=0;j<d;j++)mu[j]+=r[j]/n;});
  const M=[...Array(d)].map(()=>new Float64Array(d));X.forEach(r=>{for(let a=0;a<d;a++){const xa=r[a]-mu[a];if(!xa)continue;const row=M[a];for(let b=0;b<d;b++)row[b]+=xa*(r[b]-mu[b]);}});
  function top(){let v=new Float64Array(d).fill(1/8);for(let it=0;it<300;it++){const w=new Float64Array(d);let nr=0;for(let a=0;a<d;a++){let s=0;for(let b=0;b<d;b++)s+=M[a][b]*v[b];w[a]=s;nr+=s*s;}nr=Math.sqrt(nr);for(let a=0;a<d;a++)w[a]/=nr;v=w;}
    let lam=0;for(let a=0;a<d;a++){let s=0;for(let b=0;b<d;b++)s+=M[a][b]*v[b];lam+=v[a]*s;}let mi=0;for(let a=1;a<d;a++)if(Math.abs(v[a])>Math.abs(v[mi]))mi=a;if(v[mi]<0)v=v.map(x=>-x);return {v,lam};}
  const e1=top();for(let a=0;a<d;a++)for(let b=0;b<d;b++)M[a][b]-=e1.lam*e1.v[a]*e1.v[b];const e2=top();
  return X.map(r=>{let p=0,q=0;for(let j=0;j<d;j++){const c=r[j]-mu[j];p+=c*e1.v[j];q+=c*e2.v[j];}return [p,q];});}
(function(){const P=pca2(X),cv=$('kmCv'),st={k:5,C:null,prev:null,lab:null,phase:'none',moves:0,changed:null,busy:false};
  const xs=P.map(p=>p[0]),ys=P.map(p=>p[1]),X0=Math.min(...xs),X1=Math.max(...xs),Y0=Math.min(...ys),Y1=Math.max(...ys);
  function d2(a,b){return (a[0]-b[0])**2+(a[1]-b[1])**2;}
  function nearest(p){let b=0,bd=Infinity;st.C.forEach((c,k)=>{const v=d2(p,c);if(v<bd){bd=v;b=k;}});return b;}
  function seed(){const idx=[];while(idx.length<st.k){const i=Math.floor(Math.random()*N);if(idx.indexOf(i)<0)idx.push(i);}st.C=idx.map(i=>P[i].slice());st.prev=null;st.lab=null;st.phase='seed';st.moves=0;st.changed=null;}
  function assign(){if(!st.C)seed();const nl=P.map(nearest);st.changed=st.lab?nl.reduce((s,v,i)=>s+(v!==st.lab[i]?1:0),0):N;st.lab=nl;st.prev=null;st.phase='assign';}
  function move(){if(!st.lab)assign();st.prev=st.C.map(c=>c.slice());st.C=st.C.map((c,k)=>{let sx=0,sy=0,n=0;P.forEach((p,i)=>{if(st.lab[i]===k){sx+=p[0];sy+=p[1];n++;}});return n?[sx/n,sy/n]:c;});st.phase='move';st.moves++;}
  function inertia(){if(!st.lab)return null;let s=0;P.forEach((p,i)=>s+=d2(p,st.C[st.lab[i]]));return s;}
  function draw(){const W=fit(cv,10),mh=W<560?300:360,s=Math.max(30,Math.min(56,Math.floor((W-8)/Math.max(st.k,5))-6)),perRow=Math.max(1,Math.floor((W+6)/(s+6))),rows=Math.ceil(st.k/perRow),H=mh+16+rows*(s+22);fit(cv,H);
    const ctx=cv.getContext('2d'),C=cols(),pad=10,sx=x=>pad+(x-X0)/(X1-X0)*(W-2*pad),sy=y=>pad+(1-(y-Y0)/(Y1-Y0))*(mh-2*pad);ctx.clearRect(0,0,W,H);
    ctx.fillStyle=C.panel2;ctx.fillRect(0,0,W,mh);
    if(st.C){const g=W<560?7:9;for(let gx=0;gx<W;gx+=g)for(let gy=0;gy<mh;gy+=g){const px=X0+(gx+g/2-pad)/(W-2*pad)*(X1-X0),py=Y0+(1-(gy+g/2-pad)/(mh-2*pad))*(Y1-Y0);ctx.fillStyle=rgba(DIGCOL[nearest([px,py])],.13);ctx.fillRect(gx,gy,g,g);}}
    P.forEach((p,i)=>{circ(ctx,sx(p[0]),sy(p[1]),2.2,st.lab?rgba(DIGCOL[st.lab[i]],.8):rgba(C.text,.35),null);});
    if(st.prev)st.prev.forEach((c,k)=>{const a=st.C[k];ctx.strokeStyle=C.text;ctx.lineWidth=1.6;ctx.setLineDash([4,3]);ctx.beginPath();ctx.moveTo(sx(c[0]),sy(c[1]));ctx.lineTo(sx(a[0]),sy(a[1]));ctx.stroke();ctx.setLineDash([]);circ(ctx,sx(c[0]),sy(c[1]),5,null,C.muted,1.6);});
    if(st.C)st.C.forEach((c,k)=>{const X_=sx(c[0]),Y_=sy(c[1]);ctx.strokeStyle=C.panel;ctx.lineWidth=7;ctx.beginPath();ctx.moveTo(X_-10,Y_);ctx.lineTo(X_+10,Y_);ctx.moveTo(X_,Y_-10);ctx.lineTo(X_,Y_+10);ctx.stroke();ctx.strokeStyle=DIGCOL[k];ctx.lineWidth=3.6;ctx.stroke();});
    ctx.fillStyle=C.muted;ctx.font=font(12,600);ctx.textAlign='left';ctx.fillText('그림용 지도 · 점 = 글씨 한 장 · 십자 = 중심',pad+2,mh-8);
    /* 아래 줄: 묶음마다 평균 글씨 (64칸 평균) */
    const y0=mh+14;ctx.font=font(12,600);
    if(!st.lab){ctx.fillStyle=C.muted;ctx.fillText(st.C?'② 배정을 누르면 묶음마다 모인 글씨의 64칸 평균("평균 글씨")이 여기에 나온다.':'① 중심 찍기부터.',4,y0+18);return;}
    const mp=meanPx(st.lab,st.k),cnt=new Array(st.k).fill(0);st.lab.forEach(c=>cnt[c]++);
    for(let k=0;k<st.k;k++){const r=Math.floor(k/perRow),c=k%perRow,x=c*(s+6)+2,y=y0+r*(s+22);if(cnt[k])thumb(ctx,mp[k],x,y,s,C,DIGCOL[k],3);else{ctx.strokeStyle=DIGCOL[k];ctx.strokeRect(x,y,s,s);}ctx.fillStyle=C.muted;ctx.textAlign='center';ctx.fillText(cnt[k]+'장',x+s/2,y+s+14);}ctx.textAlign='left';}
  function info(){const I=inertia();let h='k = <b>'+st.k+'</b> · 이동 <b>'+st.moves+'</b>번'+(st.changed!=null?' · 이번 배정에서 묶음이 바뀐 글씨 <b>'+st.changed.toLocaleString()+'</b>장':'')+(I!=null?' · 지도 위 관성 <b>'+comma(I/1000)+'</b>천':'')+'. ';
    if(st.phase==='none')h+='① 중심 찍기를 누르면 아무 글씨 '+st.k+'장의 자리에 중심이 놓인다.';
    else if(st.phase==='seed')h+='중심을 아무 데나 찍었다. 배경 색은 "여기 놓인 글씨는 어느 중심이 가장 가까운가"다. ② 배정을 누르자.';
    else if(st.phase==='assign')h+=st.changed===0?'<b>바뀐 글씨가 0장이다. 멈춘다.</b> 중심이 묶음의 평균 자리에 있고, 글씨도 모두 가장 가까운 중심에 붙어 있다. ① 중심 찍기로 다시 시작하면 다른 끝에 닿을 수도 있다.':st.moves===0?'글씨마다 가장 가까운 중심의 색을 받았다. 아래 평균 글씨가 아직 흐릿하게 섞여 있다. ③ 이동을 누르자.':'중심이 옮겨 가자 경계가 바뀌어 '+st.changed+'장이 이웃 묶음으로 넘어갔다. 다시 ③ 이동.';
    else h+='중심이 묶음의 평균 자리로 옮겨 갔다(점선 = 옛 자리에서 새 자리). 경계(배경 색)도 따라 움직였다. 다시 ② 배정을 누르면 몇몇 글씨가 묶음을 바꾼다.';
    $('kInfo').innerHTML=h;}
  function upd(){draw();info();}
  $('kSeed').addEventListener('click',()=>{if(st.busy)return;seed();upd();});
  $('kAssign').addEventListener('click',()=>{if(st.busy)return;assign();upd();});
  $('kMove').addEventListener('click',()=>{if(st.busy)return;move();upd();});
  $('kAuto').addEventListener('click',async()=>{if(st.busy)return;st.busy=true;$('kAuto').disabled=true;if(!st.C){seed();upd();await wait(REDUCE?0:400);}
    const tok=st.tok;for(let g=0;g<200;g++){if(st.tok!==tok)break;assign();upd();if(st.changed===0)break;await wait(REDUCE?0:300);if(st.tok!==tok)break;move();upd();await wait(REDUCE?0:300);}st.busy=false;$('kAuto').disabled=false;});
  seg('#kChoice button',v=>{st.tok=(st.tok||0)+1;st.k=+v;st.C=null;st.lab=null;st.prev=null;st.phase='none';st.moves=0;st.changed=null;upd();});
  upd();onRedraw(draw);})();

/* ── 02 실기 과제지 ── */
(function(){const cv=$('wallCv'),pc=$('pixCv');let sel=0,cache=null,geo=null;
  function layout(){const W=fit(cv,10),cs=clamp(Math.floor(W/56),7,13),colsN=Math.floor(W/cs),rows=Math.ceil(N/colsN);return {W,cs,colsN,rows,H:rows*cs};}
  function draw(){geo=layout();fit(cv,geo.H);const ctx=cv.getContext('2d'),C=cols(),key=geo.W+'|'+C.text+'|'+C.panel;
    if(!cache||cache.key!==key){const off=document.createElement('canvas'),dpr=window.devicePixelRatio||1;off.width=Math.round(geo.W*dpr);off.height=Math.round(geo.H*dpr);const o=off.getContext('2d');o.setTransform(dpr,0,0,dpr,0,0);o.fillStyle=C.panel;o.fillRect(0,0,geo.W,geo.H);
      for(let i=0;i<N;i++){const x=(i%geo.colsN)*geo.cs,y=Math.floor(i/geo.colsN)*geo.cs;drawDigit(o,X[i],x+.5,y+.5,geo.cs-1,C.text);}cache={key,off};}
    ctx.clearRect(0,0,geo.W,geo.H);ctx.drawImage(cache.off,0,0,geo.W,geo.H);const x=(sel%geo.colsN)*geo.cs,y=Math.floor(sel/geo.colsN)*geo.cs;ctx.strokeStyle=C.stamp;ctx.lineWidth=2;ctx.strokeRect(x-1,y-1,geo.cs+2,geo.cs+2);drawPix();}
  function drawPix(){const W=fit(pc,10),s=Math.min(W,300),c=s/8;fit(pc,s+2);const ctx=pc.getContext('2d'),C=cols(),px=X[sel];ctx.clearRect(0,0,W,s+2);
    for(let j=0;j<64;j++){const x=(j%8)*c,y=Math.floor(j/8)*c,v=px[j];ctx.fillStyle=C.panel;ctx.fillRect(x,y,c,c);ctx.fillStyle=rgba(C.text,v/16);ctx.fillRect(x,y,c,c);ctx.strokeStyle=C.rule;ctx.strokeRect(x+.5,y+.5,c-1,c-1);
      ctx.fillStyle=v>8?C.panel:C.muted;ctx.font=font(Math.max(10,c*.36),600,1);ctx.textAlign='center';ctx.fillText(v,x+c/2,y+c/2+c*.13);}
    const f=digitFeat(px),m=manualDigit(px);
    $('pixInfo').innerHTML='<b>'+(sel+1).toLocaleString()+'번째 글씨</b> · 특징 64개 = 위 숫자를 왼쪽 위부터 한 줄로 편 것. 잉크 합 <b>'+f.ink+'</b> · 위쪽 네 줄 비율 <b>'+pct(f.top,0)+'</b> · 가운데 네 칸 <b>'+f.center+'</b> → 매뉴얼 씨 더미 '+m+' ('+MANNAME(m)+'). 이 글씨가 무슨 숫자인지는 봉투 속에 있다.';}
  cv.addEventListener('click',e=>{const r=cv.getBoundingClientRect(),c=Math.floor((e.clientX-r.left)/geo.cs),rw=Math.floor((e.clientY-r.top)/geo.cs),i=rw*geo.colsN+c;if(c>=0&&c<geo.colsN&&i>=0&&i<N){sel=i;draw();}});
  draw();onRedraw(()=>{cache=null;draw();});
  /* 매뉴얼 씨의 여덟 더미 */
  const mc=$('manCv'),man=X.map(manualDigit),mem=[...Array(8)].map(()=>[]);man.forEach((m,i)=>mem[m].push(i));
  function drawMan(){const W=fit(mc,10),ts=W<560?24:30,rowH=ts+26,H=8*rowH;fit(mc,H);const ctx=mc.getContext('2d'),C=cols(),per=Math.floor((W-4)/(ts+3));ctx.clearRect(0,0,W,H);
    for(let m=0;m<8;m++){const y=m*rowH;ctx.fillStyle=C.text;ctx.font=font(12.5,700);ctx.textAlign='left';ctx.fillText('더미 '+m+' · '+MANNAME(m)+' · '+mem[m].length+'장',2,y+14);
      mem[m].slice(0,per).forEach((i,q)=>thumb(ctx,X[i],2+q*(ts+3),y+19,ts,C));}}
  drawMan();onRedraw(drawMan);})();
$('heroLine').textContent='ARI '+fmt3(CLU_LINE);

/* ── 03 AI 비서 ── */
const MAIN_PROMPT=`너는 머신러닝 수업 조교다. Google Colab에서 바로 실행되는 파이썬 코드를 써 줘.

[환경] Colab 기본 라이브러리(numpy, scikit-learn)만 쓴다. 다른 패키지는 설치하지 않는다. 첫 줄에 scikit-learn 버전을 출력한다.
[데이터] 〔sklearn.datasets.load_digits〕 로 손글씨 숫자 〔1,797장, 8×8 픽셀 = 특징 64개〕를 읽는다.
  - 학습에는 X(픽셀)만 쓴다. 정답 y는 "채점 봉투"로 따로 두고, 마지막 채점 칸에서만 쓴다.
[규정]
  1) KMeans(n_clusters=〔10〕, n_init=10, random_state=0). n_init은 버전마다 기본값이 다르니 꼭 적는다.
  2) k는 정답 없이 고른다: k = 2~15의 관성(엘보)과 실루엣 점수를 표로 낸다.
  3) 전처리(표준화 등)는 넣은 것과 안 넣은 것을 둘 다 돌려 비교한다.
  4) 채점은 묶음 번호와 무관한 지표(adjusted_rand_score, normalized_mutual_info_score)로 한다.
[출력] 중심 〔10〕개를 8×8 그림으로, 묶음 × 정답 표, ARI·NMI·실루엣, 묶음마다 대표 한 장에만 이름을 붙였을 때 맞게 이름 붙는 비율, 기존 규칙(〔매뉴얼 씨 ARI 0.108〕)과 무작위 묶음의 점수.
[금지] 정답 y로 k나 전처리를 고르지 않는다. 묶음 번호와 정답을 그대로 맞대어 정확도를 내지 않는다. 데이터를 지어내지 않는다.
[형식] 셀 단위로 나누고, 각 셀 위에 무엇을 하는지 한국어 주석을 단다. 코드 뒤에 내가 확인해야 할 점 3가지를 적어 줘.`;
$('mainPrompt').textContent=MAIN_PROMPT;$('copyMain').addEventListener('click',()=>copyText(MAIN_PROMPT,'프롬프트를 복사했습니다'));
(function(){const LINES=[
 ['import numpy as np'],['from sklearn.datasets import load_digits'],['from sklearn.preprocessing import StandardScaler',0,'불러오는 것 자체는 괜찮다. 어디에 쓰는지가 문제다(7번 줄).'],
 ['from sklearn.cluster import KMeans'],['from sklearn.metrics import accuracy_score',0,'불러오는 것은 괜찮다. 쓰는 곳이 문제다(14번 줄).'],[''],
 ['X, y = load_digits(return_X_y=True)',0,'scikit-learn 안에 든 손글씨 1,797장을 읽었다. 괜찮다. y를 어디에 쓰는지만 지켜보자.'],
 ['X = StandardScaler().fit_transform(X)   # 거리 모델이니 표준화부터',1,'<b>습관처럼 표준화했다.</b> 단위가 다른 특징(kg과 일)에는 약이지만, 이 글씨들은 64칸 모두 같은 단위(밝기 0~16)다. 가장자리처럼 거의 늘 0인 칸은 표준편차가 아주 작아, 표준화하면 그 칸에 가끔 묻은 잉크가 수십 배로 커진다. 05장: ARI 0.666 → 0.468. 넣은 것과 안 넣은 것을 둘 다 돌려 비교한다.'],[''],
 ['best_k, best = 0, 0'],
 ['for k in range(2, 16):'],
 ['    km = KMeans(n_clusters=k,             # 시작은 알아서 하겠지',2,'<b>n_init과 random_state가 없다.</b> 1.4부터 n_init 기본값은 "auto", 곧 k-means++ 시작 <b>한 번</b>이다. 시작 운에 맡기는 셈이고(04장: 한 번이면 0.615, 열 번이면 0.666), random_state가 없으니 돌릴 때마다 결과가 바뀐다. <code>n_init=10, random_state=0</code>을 적는다.'],
 ['                algorithm="full")',3,'<b>지금은 없는 이름이다.</b> "full"은 1.1에서 "lloyd"로 바뀌었고 1.3에서 지워졌다. 지금 쓰면 <code>InvalidParameterError</code>가 난다. 기본값("lloyd")이면 적지 않아도 된다.'],
 ['    km.fit(X, y)',0,'<b>오류는 아니다.</b> KMeans의 fit은 y를 받아도 쓰지 않고 버린다(형식을 맞추려고 자리만 있다). 하지만 읽는 사람이 "정답으로 학습했나?" 오해하니 <code>km.fit(X)</code>로 쓰는 편이 낫다.'],
 ['    score = accuracy_score(y, km.labels_)',4,'<b>묶음 번호를 정답과 그대로 맞댔다.</b> 묶음 0이 숫자 0이라는 법은 없다. 묶음 번호는 그냥 이름표라서, 완벽하게 묶어도 번호만 엇갈리면 정확도가 0에 가깝게 나온다. 번호와 무관한 <code>adjusted_rand_score</code>나 <code>normalized_mutual_info_score</code>로 잰다.'],
 ['    if score > best: best_k, best = k, score',5,'<b>정답으로 k를 골랐다.</b> 정답이 없어서 군집을 하는 것인데, 정답 점수가 가장 높은 k를 고르면 채점 봉투를 미리 연 셈이다. k는 관성(엘보)·실루엣·업무 지식으로 고르고, 정답은 마지막 채점에만 쓴다.'],[''],
 ['print("최적 k:", best_k, "정확도:", best)',5,'<b>정답으로 고른 k와 그 점수를 그대로 보고했다.</b> 바로 윗줄과 같은 문제다.']];
  const host=$('bugCode'),found=new Set();
  host.innerHTML=LINES.map((l,i)=>'<div class="ln" role="listitem" tabindex="0" data-i="'+i+'"><span class="no">'+(i+1)+'</span><span>'+hl(l[0]||' ')+'</span></div>').join('');
  function score(){$('bugScore').textContent=found.size+' / 5';$('bugScore').className='pill'+(found.size===5?' ok':'');}
  function pick(el){const i=+el.getAttribute('data-i'),l=LINES[i];if(!l[0])return;
    if(l[1]){found.add(l[1]);host.querySelectorAll('.ln').forEach(x=>{if(LINES[+x.getAttribute('data-i')][1]===l[1])x.classList.add('hit');});$('bugMsg').innerHTML=(i+1)+'번 줄: '+l[2]+(found.size===5?'<br><b>다섯 군데를 모두 찾았다.</b> 고친 코드는 A장의 Colab 코드와 같은 순서(전처리는 비교해서 → 시작 10번·난수 고정 → 정답 없이 k 고르기 → 마지막에 봉투 열어 ARI)를 따른다.':'');}
    else{el.classList.add('miss');$('bugMsg').innerHTML=(i+1)+'번 줄: '+(l[2]||'이 줄은 괜찮다.');}score();}
  host.addEventListener('click',e=>{const el=e.target.closest('.ln');if(el)pick(el);});
  host.addEventListener('keydown',e=>{if((e.key==='Enter'||e.key===' ')&&e.target.classList.contains('ln')){e.preventDefault();pick(e.target);}});
  $('bugReveal').addEventListener('click',()=>{LINES.forEach((l,i)=>{if(l[1]){found.add(l[1]);host.querySelector('[data-i="'+i+'"]').classList.add('hit','reveal');}});score();$('bugMsg').innerHTML='빨간 줄 다섯 군데: ① 8번 습관성 표준화 ② 12번 n_init·random_state 없음 ③ 13번 지워진 이름 "full" ④ 15번 묶음 번호로 정확도 ⑤ 16·18번 정답으로 k 고르고 보고. 14번 <code>fit(X, y)</code>는 오류가 아니지만 오해를 부른다.';});
  $('bugReset').addEventListener('click',()=>{found.clear();host.querySelectorAll('.ln').forEach(x=>x.classList.remove('hit','miss','reveal'));score();$('bugMsg').textContent='줄을 하나 눌러 보세요.';});score();})();
(function(){const CK=[['데이터를 실제로 읽었나','1,797 × 64인지, 밝기가 0~16인지 본다. AI가 예제 데이터를 지어내지 않았는지.'],['정답 y를 학습에 쓰지 않았나','fit(X)만. y는 마지막 채점 칸에서만 나온다.'],['n_init과 random_state를 적었나','기본값 n_init="auto"는 시작 한 번이다. 버전마다 결과가 달라진다.'],['인자 이름이 내 버전에 맞나','print(sklearn.__version__). algorithm="full"·"auto", n_jobs, precompute_distances는 지금 오류다.'],['전처리를 비교했나','표준화한 것과 안 한 것을 둘 다 돌린다. 이 데이터에서는 표준화가 해롭다.'],['k를 정답 없이 골랐나','관성(엘보)·실루엣·업무 지식. 정답 점수로 고르면 봉투를 미리 연 것이다.'],['채점 지표가 묶음 번호와 무관한가','accuracy_score가 아니라 ARI·NMI.'],['묶음을 눈으로 확인했나','중심을 8×8로 그려 보고, 묶음마다 몇 장을 직접 본다. 섞인 묶음(1과 8)을 찾는다.'],['결과를 기준선·합격선과 비교했나','무작위 묶음(ARI ≈ 0)과 매뉴얼 씨(0.108)보다 낫지 않으면 뽑을 까닭이 없다.']];
  const ul=$('checks');ul.innerHTML=CK.map((c,i)=>'<li><input type="checkbox" id="ck'+i+'"><label for="ck'+i+'"><b>'+c[0]+'</b><small>'+c[1]+'</small></label></li>').join('');
  function upd(){const n=ul.querySelectorAll('input:checked').length;ul.querySelectorAll('li').forEach(li=>li.classList.toggle('done',li.querySelector('input').checked));$('checkScore').textContent='확인 '+n+' / '+CK.length+(n===CK.length?' — 이제 돌려 본다. 결과도 확인 대상이다.':'');}
  ul.addEventListener('change',upd);upd();})();

/* ── 04 수습 기간 ① 평균 글씨가 생긴다 (64칸 K-Means 반복 재생) ── */
const REP=(function(){const cv=$('cenCv'),cc=$('cenCurve');let r=1,busy=false;
  function frames(){return ensureMain().hist.length-1;} /* 반복 r = r번째 배정 (r−1번 옮긴 중심으로) */
  function draw(){if(!MAIN)return;const W=fit(cv,10),per=W>=560?10:5,s=Math.min(72,Math.floor((W-(per-1)*8)/per)),rows=Math.ceil(10/per),H=rows*(s+24)+2;fit(cv,H);
    const ctx=cv.getContext('2d'),C=cols(),F=MAIN.hist[r-1],cnt=new Array(10).fill(0);F.lab.forEach(c=>cnt[c]++);ctx.clearRect(0,0,W,H);
    for(let k=0;k<10;k++){const x=(k%per)*(s+8),y=Math.floor(k/per)*(s+24);thumb(ctx,F.C[k],x,y,s,C,DIGCOL[k],3);ctx.fillStyle=C.muted;ctx.font=font(12,600,1);ctx.textAlign='center';ctx.fillText('#'+k+' · '+cnt[k],x+s/2,y+s+15);}drawC();}
  function drawC(){const W=fit(cc,170),ctx=cc.getContext('2d'),C=cols(),pl=58,pr=44,pt=10,pb=30,gw=W-pl-pr,gh=170-pt-pb,F=frames(),H=MAIN.hist.slice(0,F),lo=MAIN.inertia*0.98,hi=H[0].inertia,lx=t=>pl+(t-1)/(F-1)*gw,ly=v=>pt+(1-(v-lo)/(hi-lo))*gh,lb=v=>pt+gh-v/N*gh;
    ctx.clearRect(0,0,W,170);axes(ctx,C,pl,pt,gw,gh);
    H.forEach((h,t)=>{const bw=Math.max(3,gw/F*.55),x=lx(t+1)-bw/2,hh=h.changed/N*gh;ctx.fillStyle=rgba(C.stamp,t+1===r?.75:.28);ctx.fillRect(x,pt+gh-hh,bw,hh);});
    ctx.strokeStyle=C.blue;ctx.lineWidth=2.4;ctx.beginPath();H.forEach((h,t)=>{const X_=lx(t+1),Y_=ly(h.inertia);t?ctx.lineTo(X_,Y_):ctx.moveTo(X_,Y_);});ctx.stroke();circ(ctx,lx(r),ly(H[r-1].inertia),5,C.blue,C.panel,1.5);
    ctx.fillStyle=C.muted;ctx.font=font(11.5);[1,5,10,15,F].forEach(t=>{if(t<=F){ctx.textAlign='center';ctx.fillText(t,lx(t),170-14);}});ctx.fillText('반복 (배정 횟수)',pl+gw/2,170-1);
    ctx.textAlign='right';ctx.fillStyle=C.blue;ctx.fillText(Math.round(hi/1e4)/100+'M',pl-4,pt+10);ctx.fillText(Math.round(lo/1e4)/100+'M',pl-4,pt+gh);vlabel(ctx,'관성',12,pt+gh/2);
    ctx.textAlign='left';ctx.fillStyle=C.stamp;ctx.fillText(N.toLocaleString(),pl+gw+4,pt+10);ctx.fillText('0',pl+gw+4,pt+gh);ctx.save();ctx.translate(W-6,pt+gh/2);ctx.rotate(Math.PI/2);ctx.textAlign='center';ctx.fillText('바뀐 글씨',0,0);ctx.restore();}
  function info(){const F=MAIN.hist[r-1],last=frames();
    $('cenInfo').innerHTML='반복 <b>'+r+'</b> / '+last+' · 관성 <b>'+comma(F.inertia)+'</b> · 묶음이 바뀐 글씨 <b>'+F.changed.toLocaleString()+'</b>장. '+
      (r===1?'중심 10개는 아직 k-means++가 고른 <b>실제 글씨 10장</b>이다(그래서 선명하다). 1,797장이 처음으로 가장 가까운 글씨 쪽에 배정되었다.':r<=4?'중심이 모인 글씨의 평균으로 옮겨 가며 흐릿해졌다. 처음 몇 번에 관성이 크게 준다. 같은 숫자끼리 모일수록 평균 글씨가 그 숫자 모양을 띤다.':r<last?'옮겨 다니는 글씨가 몇 장 남지 않았다. 남은 것은 두 묶음의 경계에 있는 애매한 글씨들이다.':'<b>바뀐 글씨 0장 — 멈춘다.</b> 중심 10개가 "평균 글씨"가 되었다. 0, 6, 4, 2, 3, 5, 7은 또렷하지만, #1은 1과 8이 반반 섞여 흐릿하고 #8은 9·8·5가 겹쳐 있다(05장에서 봉투를 열면 확인된다). 이 그림만 보고도 사람은 묶음 대부분에 이름을 붙일 수 있다. 학습 '+Math.round(MAIN.ms)+'ms(이 브라우저).');}
  function set(v){r=clamp(v,1,frames());$('cenN').value=r;$('cenNO').textContent=r;draw();info();}
  $('cenN').addEventListener('input',e=>set(+e.target.value));
  $('cenPlay').addEventListener('click',async()=>{if(busy||!MAIN)return;busy=true;$('cenPlay').disabled=true;for(let t=1;t<=frames();t++){set(t);await wait(REDUCE?0:(t<5?650:280));}busy=false;$('cenPlay').disabled=false;});
  onRedraw(()=>{if(MAIN)draw();});
  return {start(){ensureMain();$('cenN').min=1;$('cenN').max=frames();set(1);}};})();

/* ── 04 ② 시작 운: 시작 10번 + 새 시작 ── */
const STARTS=(function(){const cv=$('startCv');let runs=[],extra=[],rnd=RNG(1010);
  function draw(){const all=runs.concat(extra),H=Math.max(140,all.length*22+34),W=fit(cv,H),ctx=cv.getContext('2d'),C=cols(),pl=86,pr=110,gw=W-pl-pr;ctx.clearRect(0,0,W,H);if(!all.length)return;
    const vals=all.map(r=>r.inertia),lo=Math.min(...vals)*0.995,hi=Math.max(...vals)*1.002,best=runs.reduce((b,r,i)=>r.inertia<runs[b].inertia?i:b,0);
    all.forEach((r,i)=>{const y=6+i*22,w=(r.inertia-lo)/(hi-lo)*gw,isB=i===best,isNew=i>=runs.length;ctx.fillStyle=C.text;ctx.font=font(12,isB?700:500);ctx.textAlign='right';ctx.fillText(isNew?'새 시작 '+(i-runs.length+1):'시작 '+(i+1)+(i===0?' (auto)':''),pl-6,y+13);
      ctx.fillStyle=isNew?rgba(C.amber,.7):isB?C.blue:rgba(C.blue,.45);ctx.fillRect(pl,y+2,Math.max(2,w),15);if(isB){ctx.strokeStyle=C.stamp;ctx.lineWidth=2;ctx.strokeRect(pl-1,y+1,Math.max(2,w)+2,17);}
      ctx.fillStyle=C.muted;ctx.font=font(11.5,500,1);ctx.textAlign='left';ctx.fillText(comma(r.inertia)+' · '+r.iters+'번',pl+Math.max(2,w)+6,y+14);});
    ctx.fillStyle=C.muted;ctx.font=font(11.5);ctx.textAlign='left';ctx.fillText('막대 = 끝났을 때의 관성 (짧을수록 좋다, 가로축은 '+comma(lo)+'부터)',pl,H-6);}
  function info(){if(!runs.length)return;const b=runs.reduce((b,r,i)=>r.inertia<runs[b].inertia?i:b,0),w=runs.reduce((b,r,i)=>r.inertia>runs[b].inertia?i:b,0);
    $('startInfo').innerHTML='같은 데이터, 같은 k = 10인데 시작만 다르다. 가장 좋은 것은 <b>시작 '+(b+1)+'</b>(관성 '+comma(runs[b].inertia)+'), 가장 나쁜 것은 시작 '+(w+1)+'('+comma(runs[w].inertia)+', '+pct(runs[w].inertia/runs[b].inertia-1)+' 더 큼). scikit-learn은 n_init=10이면 시작 '+(b+1)+'번을, n_init="auto"(1.4부터 기본값)면 <b>시작 1 하나만</b> 쓴다. 이 비교에 정답은 쓰지 않았다. 관성이 작은 것이 정답과도 더 잘 맞는지는 면접 결과의 "시작 한 번만" 다시 보기에서 봉투를 열어 확인한다.'+(extra.length?' 새 시작(주황)은 이 페이지가 k-means++로 뽑은 것이다. 누를 때마다 달라진다.':'');}
  async function start(){if(runs.length)return;$('startInfo').textContent='시작 10개에서 K-Means를 차례로 돌리는 중…';for(let i=0;i<PRE.starts10.length;i++){await wait(10);const t0=performance.now(),R=KM.lloyd(X,PRE.starts10[i]);R.ms=performance.now()-t0;runs.push(R);if(i%3===2)draw();}draw();info();}
  $('startNew').addEventListener('click',async()=>{if(!runs.length)return;$('startNew').disabled=true;await wait(20);extra.push(KM.lloyd(X,KM.kpp(X,10,rnd)));draw();info();$('startNew').disabled=false;});
  onRedraw(draw);return {start,get runs(){return runs;}};})();

/* ── 04 ③ k 고르기: 엘보와 실루엣 ── */
const ELBOW=(function(){const ev=$('elbowCv'),sv=$('silCv'),KS=[2,3,4,5,6,7,8,9,10,11,12,13,14,15];let pick=10,ready=false;
  function frame(cv,H){const W=fit(cv,H);return {W,ctx:cv.getContext('2d'),C:cols(),pl:52,pr:10,pt:12,pb:32,gw:W-62,gh:H-44};}
  function xk(f,k){return f.pl+(k-2)/13*f.gw;}
  function draw(){const H=220,a=frame(ev,H),b=frame(sv,H);[a,b].forEach(f=>{f.ctx.clearRect(0,0,f.W,H);axes(f.ctx,f.C,f.pl,f.pt,f.gw,f.gh);KS.forEach(k=>{if(k%2===0||k===15){f.ctx.textAlign='center';f.ctx.fillText(k,xk(f,k),H-18);}});f.ctx.fillText('k (묶음 수)',f.pl+f.gw/2,H-3);});
    if(!ready)return;const I=KS.map(k=>KRES[k].inertia),lo=Math.min(...I)*0.96,hi=Math.max(...I)*1.02,ly=v=>a.pt+(1-(v-lo)/(hi-lo))*a.gh;
    a.ctx.textAlign='right';[lo,(lo+hi)/2,hi].forEach(v=>a.ctx.fillText((v/1e6).toFixed(2)+'M',a.pl-4,ly(v)+4));vlabel(a.ctx,'관성',11,a.pt+a.gh/2);
    a.ctx.strokeStyle=a.C.blue;a.ctx.lineWidth=2.6;a.ctx.beginPath();KS.forEach((k,i)=>{const X_=xk(a,k),Y_=ly(I[i]);i?a.ctx.lineTo(X_,Y_):a.ctx.moveTo(X_,Y_);});a.ctx.stroke();KS.forEach((k,i)=>circ(a.ctx,xk(a,k),ly(I[i]),k===pick?6:3,k===pick?a.C.stamp:a.C.blue,null));
    const S=KS.map(k=>KRES[k].sil),sl=0.10,sh=0.20,sy=v=>b.pt+(1-(v-sl)/(sh-sl))*b.gh,bw=b.gw/14*.6,best=Math.max(...S);
    b.ctx.textAlign='right';[0.10,0.15,0.20].forEach(v=>b.ctx.fillText(v.toFixed(2),b.pl-4,sy(v)+4));vlabel(b.ctx,'실루엣',11,b.pt+b.gh/2);
    KS.forEach((k,i)=>{const x=xk(b,k)-bw/2,y=sy(S[i]);b.ctx.fillStyle=k===pick?b.C.stamp:S[i]===best?b.C.ok:rgba(b.C.blue,.6);b.ctx.fillRect(x,y,bw,b.pt+b.gh-y);});}
  function info(){if(!ready)return;const R=KRES[pick],prev=KRES[pick-1],S=KS.map(k=>KRES[k].sil),best=KS[S.indexOf(Math.max(...S))];
    $('elbowInfo').innerHTML='k = <b>'+pick+'</b>: 관성 <b>'+comma(R.inertia)+'</b>'+(prev?' (k = '+(pick-1)+'보다 '+pct(1-R.inertia/prev.inertia)+' 줄어듦)':'')+' · 실루엣 <b>'+fmt3(R.sil)+'</b>. '+
      '실루엣이 가장 높은 k는 <b>'+best+'</b>('+fmt3(Math.max(...S))+')이고, 10~14도 0.18대로 거의 같다. 관성 곡선은 k가 늘 때마다 꾸준히 줄 뿐 뚜렷한 팔꿈치가 없다. 정답 없이 재는 점수가 "9~14 어딘가"까지만 말해 주므로, 위원회는 업무 지식(숫자는 열 가지)으로 k = 10을 정했다. '+(pick<8?'k가 작으면 서로 다른 숫자가 한 묶음에 억지로 들어가 실루엣이 낮다.':pick>13?'k가 크면 한 숫자가 여러 묶음으로 쪼개지기 시작한다. 쪼개진 묶음끼리는 서로 가까워 실루엣이 내려간다.':'')+' (A장 6번 칸과 같은 값)';}
  async function start(){if(ready)return;$('elbowInfo').textContent='거리 1,797 × 1,797 계산 중…';await wait(20);ensureD();
    for(const k of KS){$('elbowInfo').textContent='k = '+k+' 묶는 중… (Colab과 같은 시작점, 실루엣까지)';await wait(10);silOf(runK(k));}ready=true;draw();info();}
  $('kPick').addEventListener('input',e=>{pick=+e.target.value;$('kPickO').textContent=pick;draw();info();});
  draw();onRedraw(draw);return {start,get ready(){return ready;}};})();
whenNear('m4',async()=>{REP.start();await wait(30);await STARTS.start();await ELBOW.start();},300);

/* ── 05 성과 평가: 채점 봉투를 연다 ── */
function ensureStd(){if(STD)return STD;const mu=[],sd=[];for(let j=0;j<64;j++){let m=0;for(let i=0;i<N;i++)m+=X[i][j];m/=N;let v=0;for(let i=0;i<N;i++)v+=(X[i][j]-m)**2;mu.push(m);sd.push(Math.sqrt(v/N)||1);}
  const Xs=X.map(r=>r.map((x,j)=>(x-mu[j])/sd[j])),t0=performance.now(),R=KM.lloyd(Xs,PRE.initStd);R.ms=performance.now()-t0;R.k=10;let zmax=0,zj=0,zi=0;Xs.forEach((r,i)=>r.forEach((z,j)=>{if(z>zmax){zmax=z;zj=j;zi=i;}}));
  STD={R,mu,sd,zmax,zj,zi,raw:X[zi][zj]};ariOf(R);return STD;}
function ensureMed(){if(MED)return MED;const t0=performance.now();MED=KM.kmedoids(ensureD(),10);MED.ms=performance.now()-t0;const y=openEnvelope(),l=Array.from(MED.lab);MED.ari=KM.ari(y,l);MED.nmi=KM.nmi(y,l);
  let ok=0;for(let i=0;i<N;i++)if(y[MED.med[MED.lab[i]]]===y[i])ok++;MED.biz=ok/N;return MED;}
function renderEval(){const M=ensureMain(),A=ariOf(M),B=ensureBases(),S=silOf(M),y=openEnvelope();
  $('evalStat').innerHTML=statGrid([['ARI '+fmt3(A),'K-Means k = 10 · 숨긴 정답과 겹침','a'],['ARI '+fmt3(CLU_LINE),'합격선 · 매뉴얼 씨 8더미','m'],['NMI '+fmt3(M.nmi),'정보를 얼마나 나눠 갖나 (매뉴얼 '+fmt3(B.manN)+')'],['실루엣 '+fmt3(S),'정답 없이 재는 점수 (매뉴얼 '+fmt3(KM.silhouette(ensureD(),X.map(manualDigit),8))+')']],4);
  $('ariBars').innerHTML='<div style="font-size:13px;margin-bottom:4px;color:var(--muted)">ARI — 0이면 무작위와 같고, 1이면 정답과 똑같이 나눴다</div>'+barRows([['무작위 10묶음 (20번 평균)',Math.max(0,B.rand),'var(--muted)',fmt3(B.rand)],['잉크 양만으로 10구간',B.ink,'var(--muted)',fmt3(B.ink)],['매뉴얼 씨 8더미 (합격선)',B.man,'var(--c-man)',fmt3(B.man)],['K-Means k = 10',A,'var(--blue)',fmt3(A)]],1);
  /* 묶음 × 정답 표 */
  const T=[...Array(10)].map(()=>new Array(10).fill(0));for(let i=0;i<N;i++)T[M.lab[i]][y[i]]++;const C=cols();
  let h='<table class="tb" style="font-size:13px;min-width:640px;font-variant-numeric:tabular-nums"><thead><tr><th>묶음</th><th>평균 글씨</th>'+[...Array(10)].map((_,d)=>'<th style="text-align:center">'+d+'</th>').join('')+'<th>가장 많은 숫자</th></tr></thead><tbody>';
  const mixed=[];T.forEach((row,c)=>{const n=row.reduce((a,b)=>a+b,0),mx=Math.max(...row),top=row.indexOf(mx),srt=row.map((v,d)=>[v,d]).sort((a,b)=>b[0]-a[0]);if(srt[1][0]>=0.2*n)mixed.push([c,srt[0],srt[1],n]);
    h+='<tr><td><b style="color:'+DIGCOL[c]+'">#'+c+'</b> <span class="muted">'+n+'</span></td><td><canvas class="xt" data-c="'+c+'" width="34" height="34" style="width:34px;height:34px;display:block"></canvas></td>'+
      row.map(v=>'<td style="text-align:center;'+(v?'background:'+rgba(C.blue,Math.min(.85,v/n*1.1))+';color:'+(v/n>.45?'#fff':'var(--text)'):'color:var(--muted)')+'">'+(v||'·')+'</td>').join('')+'<td><b>'+top+'</b> ('+pct(mx/n,0)+')</td></tr>';});
  $('xtab').innerHTML=h+'</tbody></table>';
  function drawX(){const C2=cols();document.querySelectorAll('#xtab canvas.xt').forEach(cv=>{const dpr=window.devicePixelRatio||1;cv.width=34*dpr;cv.height=34*dpr;const ctx=cv.getContext('2d');ctx.setTransform(dpr,0,0,dpr,0,0);thumb(ctx,M.C[+cv.getAttribute('data-c')],0,0,34,C2,null);});}
  drawX();onRedraw(drawX);
  const split=[...Array(10)].map((_,d)=>T.map((row,c)=>[row[d],c]).filter(v=>v[0]>=15).sort((a,b)=>b[0]-a[0]));const spl=split.map((s,d)=>[d,s]).filter(v=>v[1].length>=2);
  $('xtabInfo').innerHTML='봉투를 열어 보니 묶음 대부분이 숫자 하나에 모였다(0·4·6은 97% 넘게). 섞인 묶음은 '+mixed.map(m=>'<b>#'+m[0]+'</b>('+m[1][1]+' '+m[1][0]+'장 · '+m[2][1]+' '+m[2][0]+'장)').join(', ')+'이다. 거꾸로 한 숫자가 여러 묶음으로 갈린 것은 '+spl.map(v=>'<b>'+v[0]+'</b>('+v[1].map(q=>'#'+q[1]).join('·')+')').join(', ')+'다. 1은 굵기와 기울기가 제각각이라 64칸 거리로는 8이나 9와 더 가까운 경우가 많다. 묶음 번호 #0~#9는 숫자 0~9와 아무 상관이 없다 — 그래서 정확도가 아니라 ARI로 잰다.';
  renderBiz(10);renderScaleMap();}
/* 업무 점수 */
let BIZK=10;
function renderBiz(k){BIZK=k;const R=runK(k),Bz=bizOf(R),A=ariOf(R),y=openEnvelope(),cv=$('repCv'),D=ensureD();
  if(R.rand1nn==null){const rnd=RNG(77+k);let s=0;const T=100;for(let t=0;t<T;t++){const pick=[];while(pick.length<k){const i=Math.floor(rnd()*N);if(pick.indexOf(i)<0)pick.push(i);}let ok=0;for(let i=0;i<N;i++){let b=pick[0],bd=Infinity;for(const p of pick){const v=D[i*N+p];if(v<bd){bd=v;b=p;}}if(y[b]===y[i])ok++;}s+=ok/N;}R.rand1nn=s/T;}
  function draw(){const W=fit(cv,10),s=k>20?(W<560?30:36):(W<560?40:52),cw=s+(k>20?8:14),per=Math.max(1,Math.floor((W+4)/cw)),rows=Math.ceil(k/per),H=rows*(s+(k>20?30:36));fit(cv,H);const ctx=cv.getContext('2d'),C=cols();ctx.clearRect(0,0,W,H);
    Bz.reps.forEach((ri,c)=>{const x=(c%per)*cw,yy=Math.floor(c/per)*(s+(k>20?30:36)),p=Bz.per[c];thumb(ctx,X[ri],x,yy,s,C,DIGCOL[c%10],2);ctx.fillStyle=C.stamp;ctx.font=font(k>20?13:15,700,1);ctx.textAlign='left';ctx.fillText(y[ri],x+s+2,yy+14);
      ctx.fillStyle=C.muted;ctx.font=font(k>20?10:11.5,500,1);ctx.textAlign='center';ctx.fillText(p.ok+'/'+p.n,x+s/2,yy+s+13);});}
  draw();cv._draw=draw;
  $('bizStat').innerHTML=statGrid([[pct(Bz.acc),'1,797장 중 맞게 이름 붙은 비율','a'],[k+'장','사람이 본 이름표 ('+pct(k/N)+')'],[pct(R.rand1nn),'비교: 무작위 '+k+'장에 이름표 (100번 평균)'],['ARI '+fmt3(A),'같은 묶음의 ARI']],2);
  const t=(k===10?'대표 10장(빨간 숫자 = 사람이 붙인 이름)에게만 이름을 붙이면 묶음 전체가 그 이름을 물려받아 <b>'+pct(Bz.acc)+'</b>가 맞는다. 같은 10장을 무작위로 골라 "가장 가까운 이름표"를 따르게 하면 '+pct(R.rand1nn)+'다. 묶음의 대표가 낫다. 1과 8이 섞인 묶음은 대표가 한쪽 이름만 줄 수 있어 절반이 틀린다.':
    'k = '+k+'로 잘게 나누고 대표 '+k+'장에 이름을 붙이면 <b>'+pct(Bz.acc)+'</b>가 맞는다. 그런데 ARI는 '+fmt3(A)+', k = 10의 '+fmt3(ariOf(runK(10)))+'보다 '+(A<ariOf(runK(10))?'<b>낮다</b>. ARI는 "숫자 하나 = 묶음 하나"를 바라므로 한 숫자를 여러 묶음으로 쪼개면 깎는다. 이름표 일에는 쪼개도 상관없다. 같은 숫자의 두 묶음에 같은 이름을 붙이면 그만이다.':'높다.'))+
    ' <b>점수는 일에 맞게 고른다.</b> 묶음 수를 "숫자 열 개"에 맞추는 것보다, 이름표 몇 장을 쓸 수 있는지가 업무의 질문이다.';
  $('bizInfo').innerHTML=t;if(k===10)document.querySelectorAll('.bizA').forEach(e=>e.textContent=pct(Bz.acc));if(k===20)document.querySelectorAll('.bizB').forEach(e=>e.textContent=pct(Bz.acc));}
seg('#bizChoice button',v=>{if(!MAIN)return;renderBiz(+v);});onRedraw(()=>{const cv=$('repCv');if(cv._draw)cv._draw();});
/* 표준화는 독 */
function renderScaleMap(){const cv=$('sdCv');function draw(){const sd=[];for(let j=0;j<64;j++){let m=0;for(let i=0;i<N;i++)m+=X[i][j];m/=N;let v=0;for(let i=0;i<N;i++)v+=(X[i][j]-m)**2;sd.push(Math.sqrt(v/N));}
    const W=fit(cv,10),s=Math.min(W,300),c=s/8;fit(cv,s+2);const ctx=cv.getContext('2d'),C=cols(),mx=Math.max(...sd);ctx.clearRect(0,0,W,s+2);
    sd.forEach((v,j)=>{const x=(j%8)*c,y=Math.floor(j/8)*c;ctx.fillStyle=C.panel;ctx.fillRect(x,y,c,c);ctx.fillStyle=rgba(C.blue,v/mx*.9);ctx.fillRect(x,y,c,c);ctx.strokeStyle=C.rule;ctx.strokeRect(x+.5,y+.5,c-1,c-1);
      ctx.fillStyle=v/mx>.5?'#fff':(v<0.5?C.stamp:C.text);ctx.font=font(Math.max(9.5,c*.3),600,1);ctx.textAlign='center';ctx.fillText(v===0?'0':v<1?v.toFixed(2):v.toFixed(1),x+c/2,y+c/2+c*.11);});}
  draw();onRedraw(draw);}
$('scaleRun').addEventListener('click',async()=>{$('scaleInfo').textContent='표준화한 64칸으로 K-Means 계산 중…';await wait(20);const S=ensureStd(),M=ensureMain(),A=ariOf(M);
  $('scaleInfo').innerHTML='표준화한 뒤 같은 방법(시작 10번 중 관성 최소, Colab과 같은 시작점)으로 묶으면 ARI <b>'+fmt3(S.R.ari)+'</b>(NMI '+fmt3(S.R.nmi)+'), 표준화 전 '+fmt3(A)+'보다 크게 낮다. 까닭은 표준편차 그림의 빨간 숫자 칸이다. 가장자리 칸은 1,797장 거의 모두 0이라 표준편차가 0.1도 안 된다. 표준화는 칸마다 표준편차로 나누므로, 그런 칸에 어쩌다 묻은 밝기 '+S.raw+'짜리 잉크가 표준화 값 <b>'+S.zmax.toFixed(1)+'</b>로 부풀려진다('+(S.zj+1)+'번째 칸, '+(S.zi+1).toLocaleString()+'번째 글씨). 거리는 제곱의 합이라 이런 칸 하나가 글씨 모양 전체보다 크게 거리를 좌우한다. KNN의 콘크리트 데이터는 kg과 일처럼 단위가 달라 표준화가 약이었고, 이 글씨는 64칸 단위가 같아 독이 된다. <b>전처리는 데이터마다 둘 다 해 보고 비교한다.</b> (표준화해도 합격선 0.108은 넘는다.)';
  document.querySelectorAll('.stdA').forEach(e=>e.textContent=fmt3(S.R.ari));});
/* K-Medoids */
function drawMed(){const cv=$('medCv');if(!MED)return;const M=ensureMain(),y=openEnvelope(),W=fit(cv,10),per=W>=560?10:5,s=Math.min(60,Math.floor((W-(per-1)*8)/per)),rows=Math.ceil(10/per),bh=rows*(s+22),H=2*bh+40;fit(cv,H);const ctx=cv.getContext('2d'),C=cols();ctx.clearRect(0,0,W,H);
  ctx.fillStyle=C.text;ctx.font=font(12.5,700);ctx.textAlign='left';ctx.fillText('K-Means 중심 = 평균 (흐릿하다, 실제로 없는 글씨)',0,13);ctx.fillText('K-Medoids 대표 = 실제 글씨 한 장 (빨간 숫자 = 봉투 속 정답)',0,bh+33);
  for(let k=0;k<10;k++){const x=(k%per)*(s+8),yy=Math.floor(k/per)*(s+22);thumb(ctx,M.C[k],x,20+yy,s,C,DIGCOL[k],2);const m=MED.med[k];thumb(ctx,X[m],x,bh+40+yy,s,C,C.ink,2);ctx.fillStyle=C.stamp;ctx.font=font(14,700,1);ctx.fillText(y[m],x+s-12,bh+40+yy+s+15);}}
$('medRun').addEventListener('click',async()=>{$('medInfo').textContent='1,797장 사이 거리로 대표 고르는 중…';await wait(20);const R=ensureMed(),M=ensureMain(),y=openEnvelope(),ml=MED.med.map(m=>y[m]),miss=[...Array(10).keys()].filter(d=>ml.indexOf(d)<0),dup=[...new Set(ml.filter((d,i)=>ml.indexOf(d)!==i))];
  drawMed();$('medInfo').innerHTML='K-Medoids k = 10: ARI <b>'+fmt3(R.ari)+'</b>(NMI '+fmt3(R.nmi)+') · K-Means '+fmt3(ariOf(M))+'. 대표 10장에 이름을 붙이면 <b>'+pct(R.biz)+'</b>가 맞는다(K-Means '+pct(bizOf(M).acc)+'). 대표 10장의 정답은 '+ml.join(', ')+(dup.length?'로 '+dup.join('·')+'이 두 번 뽑히고 '+miss.join('·')+'은 대표가 없다':'')+'. 대표가 실제 글씨라 선명하고 바로 이름을 붙일 수 있다. 대신 1,797 × 1,797 거리를 모두 재야 한다(거리 '+Math.round(DM.ms)+'ms + 대표 고르기 '+Math.round(R.ms)+'ms, 이 브라우저). 반복은 '+(R.iters+1)+'번(BUILD 시작이 이미 좋았다). 이 데이터에서는 K-Means가 조금 낫다. 튀는 글씨가 많지 않고, 64칸 평균이 숫자 모양을 잘 담기 때문이다.';
  fillKmkm();});
onRedraw(drawMed);
whenNear('m5',renderEval,600);

/* ── 면접 결과 발표 ── */
Interview('iv10',{title:'K-Means는 손글씨 숫자 묶기(군집)에 합격할까?',
  q:'조건: 1,797장을 k = 10묶음으로 나눈 뒤 채점 봉투를 열어, 숨긴 정답과의 ARI가 합격선(매뉴얼 씨 '+fmt3(CLU_LINE)+')과 무작위 묶음(≈ 0)을 모두 넘으면 합격. 근무 조건은 04장에서 정한 k = 10, k-means++ 시작 10번, 원래 밝기 그대로(표준화 없음). 묶기는 브라우저가 직접 한다.',
  fly:'묶을 때 정답은 한 번도 안 본 거지? 봉투는 지금 처음 여는 거 맞아? 그리고 k = 10은 누가 정했어?',robo:'외울 숫자 몇 개야? 이름표는 몇 장 붙이면 돼?',
  async run(setP){setP(.2);await wait(30);const M=ensureMain();setP(.5);await wait(30);const A=ariOf(M),B=ensureBases(),Bz=bizOf(M);setP(1);
    const pass=A>CLU_LINE&&A>B.rand;
    return {result:pass?'pass':'hold',flyAns:'ARI '+fmt3(A)+' · NMI '+fmt3(M.nmi)+' · 학습·시작 고르기·k 고르기에 정답 0번 사용, 봉투는 05장에서 처음 엶 · k = 10은 업무 지식(사람이 정함)',roboAns:'외울 숫자 640개(중심 10 × 64칸) · 반복 '+M.iters+'번 · 학습 '+Math.round(M.ms)+'ms(시작 1번, 이 브라우저) · 이름표 10장 → '+pct(Bz.acc),
      flyMood:pass?'arms':'coffee',flyMove:pass?'nod':'shake',roboMood:pass?'count':'meh',roboMove:pass?'bob':'tilt',asstMood:pass?'spark':'sigh',docMood:pass?'judge':'scale',
      measure:'채점 봉투를 연 결과예요. K-Means의 ARI는 <b>'+fmt3(A)+'</b>, 합격선 '+fmt3(CLU_LINE)+'보다 '+fx(A-CLU_LINE,3)+' 높아요. 무작위 묶음('+fmt3(B.rand)+')과는 비교가 안 되고요. 업무로 보면 대표 10장에만 이름표를 붙여 1,797장의 <b>'+pct(Bz.acc)+'</b>를 맞게 분류해요. 1과 8이 섞인 묶음 하나가 가장 큰 감점입니다.',
      verdict:pass?'합격입니다. 군집팀에 배치합니다. 조건이 셋 있어요. 묶음 수 k는 사람이 정해 줄 것, 시작은 늘 여러 번 할 것, 묶음마다 대표를 사람이 확인할 것.':'보류입니다. 합격선을 넘지 못했어요.',
      key:'중심 10개 · 정답 없이 — ARI '+fmt3(CLU_LINE)+' → '+fmt3(A)+' · 이름표 10장으로 '+pct(Bz.acc),
      render(box){box.innerHTML='<div class="bars">'+barRows([['K-Means k = 10',A,'var(--blue)',fmt3(A)],['매뉴얼 씨 (합격선)',B.man,'var(--c-man)',fmt3(B.man)],['잉크 양 10구간',B.ink,'var(--muted)',fmt3(B.ink)],['무작위 10묶음',Math.max(0,B.rand),'var(--muted)',fmt3(B.rand)]],1)+'</div>';}};},
  alts:[
   {label:'표준화한 64칸으로',async run(box){const S=ensureStd(),A=S.R.ari;
     box.innerHTML=statGrid([['ARI '+fmt3(A),'표준화 후 K-Means','a'],['ARI '+fmt3(ariOf(ensureMain())),'원래 밝기 (본 조건)','s'],['ARI '+fmt3(CLU_LINE),'합격선','m']],3)+'<p class="foot">합격선은 넘으니 판정은 '+(A>CLU_LINE?/*i18n:verdict*/'합격':'<b>보류</b>')+'이지만 점수가 크게 깎인다. 거의 늘 0인 가장자리 칸이 표준화로 부풀려져 거리를 좌우하기 때문이다(05장). 5-1 KNN에서 약이던 표준화가 이 데이터에서는 독이다.</p>';}},
   {label:'시작 한 번만 (n_init="auto")',async run(box){if(!STARTS.runs.length)await STARTS.start();const y=openEnvelope(),R=STARTS.runs.map(r=>({inn:r.inertia,a:KM.ari(y,Array.from(r.lab))})),b=R.reduce((q,r,i)=>r.inn<R[q].inn?i:q,0),aMax=Math.max(...R.map(r=>r.a)),aMin=Math.min(...R.map(r=>r.a));
     box.innerHTML=statGrid([['ARI '+fmt3(R[0].a),'시작 1번만 (1.4부터 기본값)','a'],['ARI '+fmt3(R[b].a),'시작 10번 중 관성 최소 (본 조건)','s'],[fmt3(aMin)+' ~ '+fmt3(aMax),'시작 10개 각각의 ARI']],3)+
      '<div class="tw"><table class="tb" style="font-size:13px"><thead><tr><th>시작</th>'+R.map((r,i)=>'<th>'+(i+1)+'</th>').join('')+'</tr></thead><tbody><tr><td>관성(천)</td>'+R.map((r,i)=>'<td'+(i===b?' style="font-weight:700;color:var(--ok)"':'')+'>'+comma(r.inn/1000)+'</td>').join('')+'</tr><tr><td>ARI</td>'+R.map(r=>'<td>'+fmt3(r.a)+'</td>').join('')+'</tr></tbody></table></div>'+
      '<p class="foot">시작 한 번만 해도 합격선은 넘는다(판정 '+(R[0].a>CLU_LINE?/*i18n:verdict*/'합격':'보류')+'). 다만 시작에 따라 ARI가 '+fmt3(aMin)+'~'+fmt3(aMax)+'로 흔들린다. 관성이 가장 작은 시작이 ARI도 꼭 가장 높지는 않다(관성은 정답을 모르는 점수다). 그래도 정답 없이 고를 수 있는 기준은 관성뿐이고, 관성이 큰 시작은 대체로 ARI도 낮다.</p>';}},
   {label:'K-Medoids로 (대표 = 실제 글씨)',async run(box){const R=ensureMed();drawMed();fillKmkm();
     box.innerHTML=statGrid([['ARI '+fmt3(R.ari),'K-Medoids k = 10','a'],['ARI '+fmt3(ariOf(ensureMain())),'K-Means (본 조건)','s'],[pct(R.biz),'대표 10장 이름표 → 맞는 비율']],3)+'<p class="foot">판정 '+(R.ari>CLU_LINE?/*i18n:verdict*/'합격':'<b>보류</b>')+'. 대표가 실제 글씨라 사람이 바로 이름을 붙일 수 있고, 어떤 거리든 쓸 수 있다. 대신 모든 글씨 쌍의 거리(약 161만 쌍)를 재야 해서 데이터가 커지면 비싸다. 이 데이터에서는 K-Means가 조금 낫다.</p>';}}]});

/* ── 06 인사 카드 ── */
function fillKmkm(){const M=MAIN?ariOf(MAIN):null,R=MED,Bz=MAIN?bizOf(MAIN):null;
  const rows=[['대표','묶음의 평균 (흐릿한 글씨, 실제로 없는 글씨일 수 있다)','묶음 안의 실제 글씨 한 장'],['줄이는 것','거리 제곱의 합 (관성)','거리의 합 (제곱하지 않는다)'],['쓸 수 있는 거리','유클리드 (평균이 뜻을 가지려면)','아무 거리나 (맨해튼, 코사인, 편집 거리…)'],['튀는 값','평균이 끌려간다','덜 끌려간다'],
    ['계산','반복마다 데이터 수 × k × 특징 수','모든 쌍의 거리 (1,797장이면 약 161만 쌍). 데이터가 크면 비싸다'],['이 면접의 ARI',M!=null?fmt3(M):'—',R?fmt3(R.ari):'(05장에서 돌리면 채워진다)'],['대표 10장 이름표',Bz?pct(Bz.acc):'—',R?pct(R.biz):'—'],['scikit-learn','KMeans (본체)','본체에 없음 · scikit-learn-extra의 KMedoids (별도 설치)']];
  $('kmkm').innerHTML=rows.map(r=>'<tr><td style="white-space:nowrap"><b>'+r[0]+'</b></td><td>'+r[1]+'</td><td>'+r[2]+'</td></tr>').join('');}
function renderCard(){ensureMain();fillKmkm();const out=document.querySelectorAll('.outA');out.forEach(e=>e.textContent=fmt3(ariOf(MAIN)));if(!runK(20).biz)bizOf(runK(20));document.querySelectorAll('.bizB').forEach(e=>e.textContent=pct(bizOf(runK(20)).acc));document.querySelectorAll('.bizA').forEach(e=>e.textContent=pct(bizOf(MAIN).acc));}
fillKmkm();whenNear('m6',renderCard,600);

/* ══ 확인 문제 ══ */
var QUIZ={
q1:[{q:'K-Means가 되풀이하는 두 걸음은?',o:[['가장 가까운 중심에 배정하기, 중심을 묶음 평균으로 옮기기',1,'배정 → 이동 → 배정 → … 묶음이 바뀌는 데이터가 없으면 멈춘다.'],['정답을 보고 틀린 것 고치기, 다시 학습하기',0,'K-Means는 정답을 쓰지 않는다. 비지도학습이다.'],['데이터를 둘로 쪼개기, 다시 합치기',0,'그것은 계층적 군집(5-11)에 가깝다.']]},
 {q:'K-Means의 k와 KNN의 k는?',o:[['K-Means는 묶음 수, KNN은 참고할 이웃 수',1,'이름만 비슷하다. KNN은 정답이 있는 지도학습, K-Means는 정답 없는 군집이다.'],['둘 다 이웃 수',0,'K-Means의 k는 묶음 수다.'],['둘 다 묶음 수',0,'KNN은 묶음을 만들지 않는다. 가까운 이웃의 정답을 빌린다.']]},
 {q:'K-Means가 끝나면 묶음 #3이 무슨 숫자인지 알려 주나?',o:[['아니다. 묶음 번호만 줄 뿐, 뜻은 사람이 붙인다',1,'그래서 업무 점수는 "대표 한 장에 사람이 이름 붙이기"로 잰다(05장).'],['그렇다. 묶음 #3은 숫자 3이다',0,'번호는 아무 이름표다. 묶음 #3에는 주로 1이 모였다.'],['정답을 쓰면 알려 준다',0,'정답을 쓰면 비지도가 아니다.']]}],
q2:[{q:'이 면접의 실기 과제지에서 특징 X는?',o:[['8×8 픽셀의 밝기 64개',1,'한 장을 64칸 한 줄로 편 것이 한 샘플이다.'],['글씨의 정답 숫자',0,'정답은 채점 봉투에 있다. 학습에 쓰지 않는다.'],['글씨를 쓴 사람',0,'그런 정보는 없다.']]},
 {q:'정답이 데이터에 들어 있는데도 학습에 쓰지 않는 까닭은?',o:[['실제 업무에는 정답이 없다는 상황을 흉내 내고, 정답은 채점에만 쓰려고',1,'정답 없이 묶은 뒤 봉투를 열어 "얼마나 잘 묶었나"를 잰다.'],['정답이 틀려서',0,'정답은 맞다. 일부러 숨긴다.'],['K-Means가 숫자를 못 읽어서',0,'숫자를 읽지만 정답을 쓰는 방법이 아니다.']]}],
q3:[{q:'KMeans로 묶은 뒤 accuracy_score(y, km.labels_)로 채점하면?',o:[['묶음 번호가 숫자와 맞지 않으니 잘 묶어도 낮게 나올 수 있다. ARI·NMI로 잰다',1,'묶음 #0이 숫자 0이라는 법이 없다. ARI는 번호와 무관하게 "같은 묶음에 든 짝"을 본다.'],['정확하게 채점된다',0,'번호가 엇갈리면 엉뚱한 점수가 나온다.'],['오류가 난다',0,'오류 없이 엉뚱한 숫자가 나와서 더 위험하다.']]},
 {q:'scikit-learn 1.4 이후 KMeans(n_clusters=10)만 쓰면 시작은 몇 번?',o:[['k-means++로 한 번 ("auto")',1,'그 전에는 10번이었다. 결과를 맞추려면 n_init과 random_state를 적는다.'],['10번',0,'1.4부터 기본값이 "auto"로 바뀌었다.'],['데이터 수만큼',0,'그렇지 않다.']]},
 {q:'k를 고를 때 정답 점수(ARI)가 가장 높은 k를 고르면?',o:[['채점 봉투를 미리 연 셈이다. 정답이 없는 실제 업무에서는 할 수 없다',1,'k는 관성·실루엣·업무 지식으로 고른다.'],['가장 좋은 방법이다',0,'정답이 있으면 애초에 군집이 아니라 분류를 하면 된다.'],['상관없다',0,'점수가 부풀려지고, 실제로는 쓸 수 없는 방법이다.']]}],
q4:[{q:'반복 1의 중심 그림이 선명하고, 끝의 중심 그림이 흐릿한 까닭은?',o:[['처음 중심은 실제 글씨 한 장이고, 끝 중심은 수백 장의 평균이라서',1,'평균 글씨는 그 묶음 글씨들이 겹친 모양이다. 흐릿하지만 숫자 모양이 보인다.'],['학습이 잘못되어서',0,'평균은 원래 흐릿하다.'],['해상도가 줄어서',0,'둘 다 8×8이다.']]},
 {q:'시작 10번 중 하나를 고르는 기준은?',o:[['관성이 가장 작은 것',1,'정답 없이 고를 수 있는 기준이다. scikit-learn의 n_init이 이렇게 한다.'],['ARI가 가장 높은 것',0,'그러려면 정답이 필요하다.'],['가장 빨리 끝난 것',0,'빠르기와 좋음은 다르다.']]},
 {q:'관성이 k를 늘릴수록 늘 줄어든다면, 관성으로 k를 고르는 법은?',o:[['줄어드는 폭이 갑자기 작아지는 곳(엘보)을 찾는다',1,'이 데이터는 엘보가 뚜렷하지 않아 실루엣과 업무 지식을 함께 본다.'],['관성이 가장 작은 k를 고른다',0,'그러면 k = 데이터 수(관성 0)가 된다.'],['관성이 가장 큰 k를 고른다',0,'그것은 k = 1이다.']]}],
q5:[{q:'ARI가 0에 가깝다는 뜻은?',o:[['무작위로 나눈 것과 비슷하다',1,'ARI는 우연히 맞을 몫을 빼서 무작위면 0 근처, 정답과 똑같으면 1이 되게 고친 점수다.'],['정답과 똑같다',0,'그때는 1이다.'],['묶음이 0개다',0,'묶음 수와 무관하다.']]},
 {q:'k를 10에서 50으로 늘리자 ARI는 내려갔는데 대표 이름표 업무 점수는 올랐다. 맞는 해석은?',o:[['ARI는 한 숫자를 여러 묶음으로 쪼개면 깎지만, 이름표 일에는 쪼개도 괜찮다',1,'같은 숫자의 두 묶음에 같은 이름을 붙이면 된다. 점수는 일에 맞게 고른다.'],['둘 중 하나가 계산이 틀렸다',0,'둘은 서로 다른 질문에 답한다.'],['k는 늘 클수록 좋다',0,'이름표 장수(비용)도 같이 는다.']]},
 {q:'이 글씨 데이터에서 표준화가 해로운 까닭은?',o:[['거의 늘 0인 가장자리 칸이 아주 작은 표준편차로 나뉘어 크게 부풀려지기 때문',1,'64칸 단위가 같으니 표준화할 까닭이 없었고, 오히려 드문 잉크 몇 칸이 거리를 좌우하게 된다.'],['표준화는 늘 해롭다',0,'콘크리트 KNN(5-1)에서는 약이었다. 데이터마다 다르다.'],['표준화하면 묶음 수가 바뀌어서',0,'k는 그대로 10이다.']]}],
q6:[{q:'K-Medoids의 대표는?',o:[['묶음 안의 실제 데이터 한 장',1,'그래서 선명하고 바로 이름을 붙일 수 있다. 어떤 거리든 쓸 수 있다.'],['묶음의 평균',0,'그것은 K-Means의 중심이다.'],['무작위 점',0,'묶음 안 다른 글씨들과의 거리 합이 가장 작은 글씨다.']]},
 {q:'K-Means가 잘 못 하는 일은?',o:[['길쭉하거나 휜 묶음, 묶음 수를 모르는 일',1,'가까운 중심으로 나누면 경계가 직선이라 둥근 묶음을 가정하는 셈이다. 5-11·5-12·5-13이 이 약점을 메운다.'],['큰 데이터',0,'단순해서 큰 데이터에 오히려 잘 쓴다.'],['새 데이터를 배정하는 일',0,'가장 가까운 중심을 찾으면 된다(predict).']]}]};

/* ══ 공통 B·C·D·E·F 데이터 ══ */
var PROMPTS=[
 {t:'① 내 데이터로 K-Means 코드 받기',use:'내 데이터(로트, 고객, 이미지)로 03장의 프롬프트를 맞출 때',p:MAIN_PROMPT},
 {t:'② AI 코드 검증 요청',use:'받은 군집 코드를 다른 AI(또는 같은 AI의 새 대화)에게 검토받을 때',p:'아래 scikit-learn K-Means 군집 코드를 코드 리뷰해 줘. 특히 다음을 하나씩 확인하고, 문제가 있으면 줄 번호와 고친 코드를 보여 줘.\n1) 정답(레이블)이 있다면 fit이나 k 고르기에 쓰지 않았나. 정답은 마지막 채점에만 써야 한다\n2) n_init과 random_state를 적었나 (내 버전 〔1.x〕의 n_init 기본값은 무엇인가)\n3) algorithm, n_jobs, precompute_distances 같은 인자가 내 버전에 있는 이름인가\n4) 채점에 accuracy_score처럼 묶음 번호에 따라 달라지는 지표를 쓰지 않았나 (ARI, NMI를 써야 한다)\n5) 표준화 같은 전처리를 넣었다면, 넣지 않은 것과 비교했나\n6) k를 고른 근거(엘보, 실루엣, 업무 지식)가 코드와 출력에 있나\n추측으로 답하지 말고, 확실하지 않은 부분은 "공식 문서 확인 필요"라고 표시해 줘.'},
 {t:'③ 묶음 결과를 사람이 확인하는 도구 만들기',use:'묶음마다 대표와 이상한 데이터를 사람이 빨리 훑어볼 때',p:'아래 Colab 코드(K-Means, 손글씨 숫자)에 이어서, 사람이 묶음 결과를 확인하는 셀을 덧붙여 줘.\n- 묶음마다: 크기, 중심을 8×8 그림으로, 중심에 가장 가까운 글씨 〔5〕장, 가장 먼 글씨 〔5〕장을 matplotlib 한 줄 그림으로\n- 가장 먼 글씨는 "묶음이 잘못 받았을 수 있는 글씨"로 따로 목록을 만든다\n- 사람이 묶음마다 이름(숫자)을 입력하면 1,797장 전체에 이름을 물려주는 함수\n- 마지막에만 정답과 비교해 "이름표 몇 장으로 몇 %가 맞았나"를 출력\n규칙: 정답은 마지막 비교 셀에서만 쓴다. Colab 기본 라이브러리만 쓴다.'},
 {t:'④ 수업용 문제 만들기',use:'이 편으로 수업할 때 확인 문제가 더 필요할 때',p:'K-Means 입문 수업용 문제를 만들어 줘. 대상은 〔제조 현장 실무자, 수식에 익숙하지 않음〕이다. 예시는 손글씨 숫자 또는 〔공장의 로트 데이터〕로 통일한다.\n- 배정·이동·관성·시작 운·k 고르기를 각각 묻는 객관식 5문제 (보기 3개, 정답 해설 포함)\n- 오해를 드러내는 문제 2개: "묶음 번호 = 정답 숫자", "표준화는 늘 해야 한다"\n- 손으로 풀 수 있는 계산 문제 1개: 1차원 점 여섯 개(1, 2, 3, 10, 11, 12)와 시작 중심 두 개(1, 2)로 배정 → 이동을 바뀌는 점이 없을 때까지 손으로 되풀이하고, 매 단계의 관성을 구하기\n정답은 맨 끝에 따로 모아 줘.'}];
var GL=[
 ['모델','K-Means','중심 k개를 두고, 데이터를 가장 가까운 중심에 배정하고, 중심을 묶음 평균으로 옮기기를 되풀이하는 군집 방법. 관성을 줄인다.'],
 ['모델','K-Medoids','K-Means와 같은 틀이지만 대표(메도이드)를 묶음 안의 실제 데이터 한 장으로 삼는 군집 방법. 어떤 거리든 쓸 수 있고 튀는 값에 덜 끌려간다. PAM, alternate 같은 풀이법이 있다.'],
 ['모델','메도이드','묶음 안에서 다른 데이터들과의 거리 합이 가장 작은 실제 데이터 한 장.'],
 ['모델','군집','정답 없이 비슷한 데이터끼리 묶는 일. 클러스터링.'],
 ['모델','비지도학습','정답(레이블) 없이 데이터의 구조를 찾는 학습. 군집과 차원 축소가 대표적이다.'],
 ['모델','MiniBatchKMeans','데이터 일부(미니배치)만 써서 중심을 조금씩 옮기는 빠른 K-Means. 큰 데이터에 쓴다.'],
 ['학습','중심','묶음을 대표하는 점. K-Means에서는 묶음에 든 데이터의 평균. 센트로이드.'],
 ['학습','관성','데이터와 자기 중심 사이 거리 제곱의 합(inertia). K-Means가 줄이는 값. 묶음 안 제곱합.'],
 ['학습','k-means++','시작 중심을 고를 때, 이미 고른 중심에서 먼 데이터일수록 잘 뽑히게 하는 방법(Arthur & Vassilvitskii, 2007).'],
 ['학습','n_init','시작을 몇 번 할지. 관성이 가장 작은 결과를 남긴다. 1.4부터 기본값 "auto".'],
 ['학습','지역 최솟값','주변보다는 낮지만 가장 낮지는 않은 골짜기. K-Means는 시작에 따라 여기에 멈출 수 있다.'],
 ['학습','유클리드 거리','두 점의 같은 칸끼리 차이를 제곱해 더한 뒤 제곱근을 씌운 거리. 자로 잰 직선 거리.'],
 ['학습','표준화','특징마다 평균을 빼고 표준편차로 나눠 단위를 맞추는 일. 단위가 같은 데이터에서는 오히려 해로울 수 있다.'],
 ['학습','PCA','데이터가 가장 넓게 퍼진 방향으로 축을 돌려, 앞의 몇 축만 남기는 차원 축소(5-14). 이 편에서는 그림용 지도를 그리는 데만 썼다.'],
 ['평가','엘보 방법','k에 따른 관성 곡선에서 줄어드는 폭이 갑자기 작아지는(팔꿈치처럼 꺾이는) k를 고르는 방법.'],
 ['평가','실루엣 점수','데이터마다 (가장 가까운 다른 묶음까지 평균 거리 − 자기 묶음 평균 거리) ÷ 둘 중 큰 값을 구해 평균한 것. −1~1, 클수록 묶음이 뚜렷하다. 정답이 필요 없다.'],
 ['평가','ARI','조정 랜드 지수. 두 나누기(묶음과 숨긴 정답)에서 "같은 묶음에 든 짝"이 얼마나 일치하는지를, 우연히 맞을 몫을 빼서 잰 점수. 무작위 ≈ 0, 똑같으면 1. 묶음 번호와 무관하다.'],
 ['평가','NMI','정규화 상호 정보량. 묶음을 알면 정답을 얼마나 알 수 있는지를 0~1로 잰 점수.'],
 ['평가','채점 봉투','이 시리즈의 비지도 본부에서 숨긴 정답. 학습이 끝난 뒤 채점에만 연다.'],
 ['평가','업무 점수','묶음마다 대표 한 장에만 사람이 이름을 붙이고 묶음 전체에 물려주었을 때 맞게 이름 붙는 비율. 이름표 비용을 얼마나 줄이나를 잰다.'],
 ['평가','합격선','지원자가 넘어야 할 점수. 군집팀은 매뉴얼 씨 규칙 8더미의 ARI 0.108.'],
 ['평가','기준선','배우지 않고도 낼 수 있는 점수. 군집에서는 무작위로 나눈 묶음(ARI ≈ 0).']];
var ANALOGY=[
 ['중심 = 평균 글씨','같은 부서 사원 사진 수백 장을 겹쳐 만든 "평균 얼굴". 흐릿하지만 부서의 인상이 보인다',2,'원형 이론: 점 무늬의 변형만 본 사람이, 한 번도 보지 못한 원형(평균)을 가장 잘 알아본다(Posner & Keele, 1968)','뇌가 정말 평균을 계산해 저장한다는 직접 근거는 없다. 본 예시들을 그대로 기억해 비교한다는 이론(본보기 이론)도 실험을 잘 설명한다'],
 ['K-Medoids의 대표 = 실제 한 장','부서를 소개할 때 "평균 얼굴" 대신 가장 무난한 실제 사원 한 명을 내세우기',2,'본보기 이론: 범주를 실제로 본 예시들과의 닮음으로 판단한다(Medin & Schaffer, 1978)','본보기 이론은 예시를 여럿(때로 전부) 기억한다. K-Medoids는 묶음마다 하나만 남긴다'],
 ['가장 가까운 중심에 배정','서류를 가장 닮은 책상에 올려놓기',2,'경쟁 학습: 입력에 가장 강하게 반응한 단위만 이겨 자기 가중치를 입력 쪽으로 조금 옮긴다(Rumelhart & Zipser, 1985). 온라인 K-Means와 같은 꼴이다','신경망 모형의 이야기다. 실제 피질이 이 규칙대로 범주를 만든다는 증명은 아니다'],
 ['비슷한 것에 비슷한 표시 (초파리)','닮은 서류에 같은 색 스티커를 붙여 두기',2,'초파리 후각 회로는 냄새 신호를 수천 개의 케니언 세포로 넓게 펼친 뒤 가장 센 몇 개만 남겨, 비슷한 냄새에 비슷한 표시를 준다(Dasgupta 등, 2017)','묶음을 만드는 일이 아니라 비슷한 것을 빨리 찾게 표시를 다는 일이다. 중심도, 배정 ↔ 이동의 반복도 없다'],
 ['시작 운 · n_init','자리 배치를 여러 번 해 보고 가장 덜 붐비는 배치를 고르기',1,'','직접 대응하는 뇌 기능은 알려지지 않았다'],
 ['채점 봉투 (ARI)','시험이 다 끝난 뒤에 봉인한 정답지를 여는 감독관',1,'','뇌 비유가 없다. 평가 절차에 대한 비유다']];
var REFS=[
 'Alpaydin, E., &amp; Kaynak, C. (1998). Optical Recognition of Handwritten Digits [Dataset]. UCI Machine Learning Repository. <a href="https://doi.org/10.24432/C50P49" target="_blank" rel="noopener">doi:10.24432/C50P49</a> (CC BY 4.0) — 실기 과제지 원출처 (scikit-learn load_digits는 그 시험 세트 사본)',
 'MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations. In <i>Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability</i>, Vol. 1, 281–297. — "k-means"라는 이름',
 'Lloyd, S. P. (1982). Least squares quantization in PCM. <i>IEEE Transactions on Information Theory</i>, 28(2), 129–137. — 배정 ↔ 평균 이동(Lloyd 알고리즘), 01·04장',
 'Arthur, D., &amp; Vassilvitskii, S. (2007). k-means++: The advantages of careful seeding. In <i>Proceedings of the 18th Annual ACM-SIAM Symposium on Discrete Algorithms</i> (pp. 1027–1035). — 04장 시작 운',
 'Elkan, C. (2003). Using the triangle inequality to accelerate k-means. In <i>Proceedings of the 20th International Conference on Machine Learning</i> (pp. 147–153). — 06장 algorithm="elkan"',
 'Sculley, D. (2010). Web-scale k-means clustering. In <i>Proceedings of the 19th International Conference on World Wide Web</i> (pp. 1177–1178). — 06장 MiniBatchKMeans',
 'Rousseeuw, P. J. (1987). Silhouettes: A graphical aid to the interpretation and validation of cluster analysis. <i>Journal of Computational and Applied Mathematics</i>, 20, 53–65. — 04장 실루엣 점수',
 'Hubert, L., &amp; Arabie, P. (1985). Comparing partitions. <i>Journal of Classification</i>, 2(1), 193–218. — 05장 ARI',
 'Strehl, A., &amp; Ghosh, J. (2002). Cluster ensembles — A knowledge reuse framework for combining multiple partitions. <i>Journal of Machine Learning Research</i>, 3, 583–617. — 05장 NMI',
 'Kaufman, L., &amp; Rousseeuw, P. J. (1990). <i>Finding Groups in Data: An Introduction to Cluster Analysis</i>. Wiley. — K-Medoids(PAM, BUILD), 05·06장',
 'Park, H.-S., &amp; Jun, C.-H. (2009). A simple and fast algorithm for K-medoids clustering. <i>Expert Systems with Applications</i>, 36(2), 3336–3341. — 배정·교체를 되풀이하는 K-Medoids, 05장',
 'Schubert, E., &amp; Rousseeuw, P. J. (2021). Fast and eager k-medoids clustering: O(k) runtime improvement of the PAM, CLARA, and CLARANS algorithms. <i>Information Systems</i>, 101, 101804. — 06장 FasterPAM',
 'Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. <i>Journal of Machine Learning Research</i>, 12, 2825–2830. — A장 코드',
 'scikit-learn 문서. <a href="https://scikit-learn.org/stable/modules/clustering.html" target="_blank" rel="noopener">Clustering</a> · <a href="https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html" target="_blank" rel="noopener">KMeans</a> — n_init 기본값, algorithm 이름, 03·06장',
 'Posner, M. I., &amp; Keele, S. W. (1968). On the genesis of abstract ideas. <i>Journal of Experimental Psychology</i>, 77(3), 353–363. — D장 원형',
 'Medin, D. L., &amp; Schaffer, M. M. (1978). Context theory of classification learning. <i>Psychological Review</i>, 85(3), 207–238. — D장 본보기',
 'Rumelhart, D. E., &amp; Zipser, D. (1985). Feature discovery by competitive learning. <i>Cognitive Science</i>, 9(1), 75–112. — D장 경쟁 학습',
 'Dasgupta, S., Stevens, C. F., &amp; Navlakha, S. (2017). A neural algorithm for a fundamental computing problem. <i>Science</i>, 358(6364), 793–796. — D장 초파리 후각 회로'];
var TIP={who:'머신러닝을 처음 배우는 대학생·직장인·제조 현장 실무자. 비지도 본부의 첫 편이라 지도 본부를 몰라도 읽을 수 있지만, 5-1 KNN(거리, 표준화)을 먼저 보면 좋다. 수식은 평균과 거리 제곱 합(관성) 둘이 중심이다',
 goal:'① 배정 ↔ 평균 이동을 01장 지도에서 손으로 되풀이해 설명한다 ② 시작 운과 k 고르기를 정답 없이(관성·실루엣) 하고, k = 10이 사람의 지식이었음을 말한다 ③ 정답 봉투를 마지막에 열어 ARI로 채점하고, 정확도가 왜 안 되는지 설명한다 ④ 업무 점수(대표 이름표)로 라벨링 비용 절감을 계산하고, 표준화가 독이 되는 경우를 찾는다',
 time:'읽기 25분 · 실습(01·04·05장 그림) 25분 · 틀린 곳 찾기 10분 · Colab 15분',
 plans:[['강의형 50분','00장 지원서 1장을 인쇄해 1~4번 칸을 함께 읽기 8분 → 01장 지도에서 학생들이 번갈아 ②배정·③이동을 누르며 "다음에 어느 점이 넘어갈까" 맞히기 12분 → 04장 반복 재생으로 평균 글씨가 생기는 것 보기, 시작 운 막대 5분 → k 고르기 그래프로 "정답 없이 k를 정할 수 있나" 토론 10분 → 05장 봉투 열기와 업무 점수, 면접 결과 먼저 맞히고 발표 15분.'],['실무 OJT 90분','03장 프롬프트로 각자 AI에게 코드를 받는다 20분 → 체크리스트로 서로의 코드를 바꿔 검사(특히 정답을 어디에 썼나) 20분 → Colab에서 실행하고 A장 결과와 비교 20분 → 자기 공장 데이터(로트, 설비 로그)로 "대표 몇 개만 사람이 확인하면 되나"를 계산해 보고 B장 ③ 프롬프트로 확인 도구 만들기 30분.'],['1:1 튜터링','01장 지도에서 k = 3으로 한 단계씩 누르며 "왜 이 점이 넘어갔나"를 말로 설명하게 한다. 이어 05장 묶음 × 정답 표에서 섞인 묶음(1과 8)을 찾고, 그 묶음의 평균 글씨(04장)가 왜 흐릿한지 연결하게 한다.']],
 myths:[['묶음 번호 = 정답 숫자','묶음 번호는 아무 이름표다. 번호와 숫자를 맞대어 정확도를 내면 안 된다. ARI·NMI로 잰다'],['K-Means가 묶음 수를 찾아 준다','k는 사람이 준다. 엘보·실루엣은 실마리일 뿐이고, 이 면접의 k = 10도 업무 지식이다'],['한 번 돌리면 답이 정해진다','시작에 따라 끝이 다르다(지역 최솟값). 여러 번 시작해 관성이 가장 작은 것을 남긴다'],['거리 모델이니 표준화는 늘 해야 한다','단위가 같고 0이 많은 데이터(픽셀)에서는 해롭다. 0.666 → 0.468. 둘 다 해 보고 비교한다'],['관성이 작을수록 좋은 군집이다','k를 늘리면 관성은 늘 준다. 같은 k 안에서 시작을 비교할 때만 쓴다'],['ARI가 높을수록 업무에 좋다','이름표 일에는 잘게 쪼갠 묶음(ARI 낮음)이 더 좋을 수 있다. 점수는 일에 맞게 고른다']],
 tasks:[['과제 1 · 손으로 K-Means','1차원 점 여섯 개(1, 2, 3, 10, 11, 12)와 시작 중심 두 개(1, 2)로 배정 → 이동을 바뀌는 점이 없을 때까지 표로 되풀이하고 매 단계 관성을 쓴다. 시작을 (1, 12)로 바꾸면 어떻게 되나?'],['과제 2 · 틀린 곳 찾기 만들기','AI에게 일부러 실수가 섞인 군집 코드를 받고(B장 ②를 거꾸로 활용), 짝에게 찾게 한다. "정답을 몰래 쓴 곳"을 하나 이상 넣는다.'],['과제 3 · 이름표 예산','A장 8번 칸을 고쳐 k = 5, 10, 15, 20, 30, 50, 100에서 업무 점수를 표로 만들고, "이름표 한 장 = 1,000원, 틀린 이름 한 장 = 5,000원"일 때 총비용이 가장 작은 k를 찾는다.'],['과제 4 · 내 데이터로','공장·학교의 숫자 표 하나(로트별 측정값 등)로 K-Means를 돌리고, 표준화 전후, k 고르기 근거, 묶음마다 대표 한 줄을 보고서 한 장으로 낸다. 정답이 없으면 묶음의 뜻을 사람이 붙인다.']],
 ai:'학생이 AI에게 코드를 받는 것은 막지 말고 과정에 넣는다. 군집은 정답이 없어 틀린 코드도 그럴듯한 결과를 내므로 검증 연습에 특히 좋다. ① 프롬프트를 제출하게 하고 ② 체크리스트 아홉 항목을 직접 표시하게 하고(정답을 어디에 썼는지 줄 번호로) ③ 실행 결과를 무작위·합격선과 비교한 문장을 쓰게 한다. AI 답의 인자 이름과 기본값은 설치된 버전의 공식 문서로 확인하게 하고, 과제에 AI 도움을 받았으면 사용 범위를 밝히게 한다.'};
var CODE={title:'K-Means 면접 — 정답 없이 묶고, 마지막에 봉투 열기',sub:'numpy, scikit-learn · 인터넷 필요 없음 · CPU 1분 안팎',file:'kmeans_interview',
 desc:'이 편의 면접을 그대로 옮겼다. 손글씨 1,797장을 읽어 정답을 봉투에 넣고(1번 칸), 매뉴얼 씨의 8더미를 만들고(2번), K-Means로 묶어 평균 글씨를 글자 그림으로 보고(3번), 배정 ↔ 이동을 직접 짜서 scikit-learn과 같은지 확인하고(4번), 시작 운(5번)과 k 고르기(6번)를 정답 없이 한 뒤, 봉투를 열어 ARI로 채점하고(7번), 대표 이름표의 업무 점수(8번), 표준화의 독(9번), K-Medoids(10번), 판정(11번)까지 간다. 정답 y는 7번 칸부터만 쓴다. 노트북으로 저장하면 칸마다 셀이 나뉜다.',
 cap:'실행 결과 (scikit-learn 1.9.1, CPU). 시간(ms)은 컴퓨터마다 다르다. 브라우저 계산과 같은 결론이 나온다: K-Means ARI 0.666 · NMI 0.742 · 실루엣 0.183으로 합격선 0.108을 넘어 합격, 시작 한 번이면 0.615, 표준화하면 0.468, K-Medoids 0.609, 대표 10장 이름표 79.2% · 20장 92.8%. 브라우저는 같은 시작점에서 같은 규칙으로 반복하므로 k = 10의 묶음이 글씨 하나까지 같다.'};
