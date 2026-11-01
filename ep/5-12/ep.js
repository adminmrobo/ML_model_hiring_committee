var BOARD_LINE='ARI 0.108'; /* 군집팀 합격선: 매뉴얼 씨 규칙 8묶음의 ARI */
/* ══ 5-12 DBSCAN · 손글씨 숫자 실기 과제지 ══════════════════════════════ */
/* DB-ENGINE-START */
/* ── DBSCAN 엔진 (이 편 전용, 브라우저·node 공용) ──
   scikit-learn DBSCAN(metric='euclidean')과 같은 규칙:
   · 이웃 = 거리 ≤ eps인 점(자기 자신 포함), 이웃 수 ≥ min_samples면 핵심점
   · 번호 순으로 아직 묶음이 없는 핵심점에서 출발해, 핵심점의 이웃을 차례로 퍼뜨린다(깊이 우선)
   · 어느 핵심점에도 닿지 않는 점은 잡음(−1)
   픽셀 밝기가 정수라 거리의 제곱도 정수다. 그래서 제곱 거리를 Uint16으로 한 번만 재 두고 다시 쓴다.
   두 묶음의 핵심점에 함께 닿는 가장자리 점은 퍼뜨리는 순서에 따라 묶음이 갈릴 수 있다(scikit-learn도 같다) */
var DB=(function(){
  function dist2(P,onRow){var n=P.length,d=P[0].length,D=new Uint16Array(n*n);
    for(var i=0;i<n;i++){var a=P[i];for(var j=i+1;j<n;j++){var b=P[j],s=0;for(var k=0;k<d;k++){var t=a[k]-b[k];s+=t*t;}D[i*n+j]=s;D[j*n+i]=s;}if(onRow)onRow(i);}
    return D;}
  /* 줄마다 가장 가까운 K개의 제곱 거리(자기 자신 0 포함)를 정렬해 둔다 → k-거리 */
  function nearK(D,n,K){var out=new Uint16Array(n*K),row=new Uint16Array(n);for(var i=0;i<n;i++){row.set(D.subarray(i*n,i*n+n));row.sort();out.set(row.subarray(0,K),i*K);}return out;}
  function kdist(NK,n,K,k){var o=new Float64Array(n);for(var i=0;i<n;i++)o[i]=Math.sqrt(NK[i*K+k-1]);return o;}
  /* 무릎: 오름차순으로 늘어놓은 k-거리 곡선에서, 양 끝을 잇는 직선과 가장 멀리 떨어진 점 (A장 4번 칸과 같다) */
  function knee(kd){var s=Array.from(kd).sort(function(a,b){return a-b}),n=s.length,lo=s[0],hi=s[n-1],bi=0,bv=-1;
    for(var i=0;i<n;i++){var x=i/(n-1),y=(s[i]-lo)/(hi-lo),v=Math.abs(y-x);if(v>bv){bv=v;bi=i;}}return {i:bi,eps:Math.round(s[bi]*100)/100,sorted:s};}
  function run(D,n,eps,ms,trace){var e2=eps*eps,NB=new Array(n),core=new Uint8Array(n),lab=new Int32Array(n).fill(-1),order=[];
    for(var i=0;i<n;i++){var r=[],off=i*n;for(var j=0;j<n;j++)if(D[off+j]<=e2)r.push(j);NB[i]=r;if(r.length>=ms)core[i]=1;}
    var c=0,stack=[];
    for(i=0;i<n;i++){if(lab[i]!==-1||!core[i])continue;var p=i;
      while(true){if(lab[p]===-1){lab[p]=c;if(trace)order.push(p);if(core[p]){var nb=NB[p];for(var q=0;q<nb.length;q++)if(lab[nb[q]]===-1)stack.push(nb[q]);}}
        if(!stack.length)break;p=stack.pop();}
      c++;}
    var noise=0;for(i=0;i<n;i++)if(lab[i]<0)noise++;
    return {lab:lab,core:core,k:c,noise:noise,order:order,nbCount:NB.map(function(r){return r.length})};}
  /* 채점: 조정 랜드 지수(ARI)와 정규화 상호 정보량(NMI, 산술 평균). 잡음(−1)은 scikit-learn처럼 한 묶음으로 센다 */
  function cont(a,b){var ia={},ib={},A=[],B=[],M={};for(var i=0;i<a.length;i++){var x=a[i],y=b[i];if(!(x in ia)){ia[x]=A.length;A.push(0);}if(!(y in ib)){ib[y]=B.length;B.push(0);}A[ia[x]]++;B[ib[y]]++;var key=ia[x]*100000+ib[y];M[key]=(M[key]||0)+1;}return {A:A,B:B,M:M,n:a.length};}
  function c2(x){return x*(x-1)/2;}
  function ari(a,b){var t=cont(a,b),s=0,sa=0,sb=0;for(var k in t.M)s+=c2(t.M[k]);t.A.forEach(function(v){sa+=c2(v)});t.B.forEach(function(v){sb+=c2(v)});
    var e=sa*sb/c2(t.n),m=(sa+sb)/2;if(m===e)return 1;return (s-e)/(m-e);}
  function ent(C,n){var h=0;C.forEach(function(v){if(v>0)h-=v/n*Math.log(v/n)});return h;}
  function nmi(a,b){var t=cont(a,b),n=t.n,mi=0;for(var k in t.M){var v=t.M[k],i=Math.floor(k/100000),j=k%100000;mi+=v/n*Math.log(v*n/(t.A[i]*t.B[j]));}
    var ha=ent(t.A,n),hb=ent(t.B,n);if(ha===0&&hb===0)return 1;var d=(ha+hb)/2;return d>0?Math.max(0,mi)/d:0;}
  /* 실루엣 점수: 잡음을 뺀 점들만 (묶음이 2개 이상일 때) */
  function silhouette(D,n,lab){var idx=[],K=0;for(var i=0;i<n;i++)if(lab[i]>=0){idx.push(i);K=Math.max(K,lab[i]+1);}if(K<2)return NaN;
    var cnt=new Array(K).fill(0);idx.forEach(function(i){cnt[lab[i]]++});var S=0;
    for(var a=0;a<idx.length;a++){var i=idx[a],sum=new Float64Array(K),off=i*n;for(var b=0;b<idx.length;b++){var j=idx[b];if(j!==i)sum[lab[j]]+=Math.sqrt(D[off+j]);}
      var li=lab[i],ai=cnt[li]>1?sum[li]/(cnt[li]-1):0,bi=Infinity;for(var k=0;k<K;k++)if(k!==li&&cnt[k])bi=Math.min(bi,sum[k]/cnt[k]);S+=cnt[li]>1?(bi-ai)/Math.max(ai,bi):0;}
    return S/idx.length;}
  /* 업무 점수: 묶음마다 평균 글씨에 가장 가까운 한 장(대표)에만 사람이 이름을 붙이고, 묶음 전체에 그 이름을 퍼뜨린다. 잡음은 이름 없이 남는다 */
  function propagate(P,lab,y){var K=0;for(var i=0;i<lab.length;i++)K=Math.max(K,lab[i]+1);var d=P[0].length,mu=[],cnt=new Array(K).fill(0),rep=new Array(K).fill(-1),best=new Array(K).fill(Infinity);
    for(var k=0;k<K;k++)mu.push(new Float64Array(d));for(i=0;i<lab.length;i++)if(lab[i]>=0){cnt[lab[i]]++;for(var j=0;j<d;j++)mu[lab[i]][j]+=P[i][j];}
    for(k=0;k<K;k++)for(j=0;j<d;j++)mu[k][j]/=cnt[k]||1;
    for(i=0;i<lab.length;i++){var c=lab[i];if(c<0)continue;var s=0;for(j=0;j<d;j++){var t=P[i][j]-mu[c][j];s+=t*t;}if(s<best[c]){best[c]=s;rep[c]=i;}}
    var ok=0,okC=new Array(K).fill(0);for(i=0;i<lab.length;i++){c=lab[i];if(c>=0&&y[i]===y[rep[c]]){ok++;okC[c]++;}}
    return {acc:ok/lab.length,rep:rep,cnt:cnt,okC:okC,K:K};}
  return {dist2:dist2,nearK:nearK,kdist:kdist,knee:knee,run:run,ari:ari,nmi:nmi,silhouette:silhouette,propagate:propagate};})();
/* DB-ENGINE-END */

/* ── 과제지: 1,797장. Y는 채점 봉투 속 정답 — 채점 함수(scoreAll, 봉투 열기, 면접)에서만 쓴다 ── */
const DIG=digitRows(),P=DIG.map(r=>r.px),Y=DIG.map(r=>r.y),N=P.length;
const MAN=P.map(manualDigit);
const PAL=DIGCOL.concat(['#e0a100','#5b7cfa','#b0413e','#3c8d2f','#a65fb0','#2a9bb5','#d1607f','#8a7a2a','#6b7a90','#1d8a7a','#c9772f','#7058c8']);
const CC=c=>c<0?null:PAL[c%PAL.length];
const fmtA=v=>fx(v,3);
function axes(ctx,C,pl,pt,gw,gh){ctx.strokeStyle=C.rule;ctx.lineWidth=1;ctx.strokeRect(pl,pt,gw,gh);ctx.fillStyle=C.muted;ctx.font=font(12);}
function vlabel(ctx,txt,x,y){ctx.save();ctx.translate(x,y);ctx.rotate(-Math.PI/2);ctx.textAlign='center';ctx.fillText(txt,0,0);ctx.restore();}
function barRows(rows,max){return rows.map(r=>'<div class="row"><span style="text-align:left;font-family:var(--font)">'+r[0]+'</span><div class="tr"><i style="left:0;width:'+Math.max(0,Math.min(100,r[1]/max*100))+'%;background:'+(r[2]||'var(--blue)')+'"></i></div><span>'+r[3]+'</span></div>').join('');}
function whenNear(id,fn,margin){const t=$(id);let done=false;function go(){if(done)return;done=true;setTimeout(fn,10);}
  if('IntersectionObserver' in window){const io=new IntersectionObserver(es=>{if(es.some(e=>e.isIntersecting)){go();io.disconnect();}},{rootMargin:(margin||600)+'px'});io.observe(t);}else go();
  const near=()=>{const h=location.hash.slice(1),el=h&&$(h);if(el&&(el===t||t.contains(el)||t.compareDocumentPosition(el)&Node.DOCUMENT_POSITION_FOLLOWING))go();};
  window.addEventListener('hashchange',near);if(location.hash)near();}
function xcross(ctx,x,y,r,col,lw){ctx.strokeStyle=col;ctx.lineWidth=lw||1.4;ctx.beginPath();ctx.moveTo(x-r,y-r);ctx.lineTo(x+r,y+r);ctx.moveTo(x+r,y-r);ctx.lineTo(x-r,y+r);ctx.stroke();}

/* ── 무거운 계산은 한 번만: 제곱 거리 행렬 · 가까운 32개 · 무릎 ── */
let D=null,NK=null,COST_MS=0;const KN={},RUNS={};
function ensureD(){if(D)return D;const t0=performance.now();D=DB.dist2(P);COST_MS=performance.now()-t0;NK=DB.nearK(D,N,32);
  [3,5,10,20].forEach(ms=>{KN[ms]=DB.knee(DB.kdist(NK,N,32,ms));});
  if($('costMs'))$('costMs').textContent=Math.round(COST_MS).toLocaleString();if($('heroEps'))$('heroEps').textContent='eps '+KN[5].eps.toFixed(2);return D;}
function runDB(eps,ms){const key=eps.toFixed(2)+'|'+ms;if(RUNS[key])return RUNS[key];ensureD();const r=DB.run(D,N,eps,ms);r.eps=eps;r.ms=ms;
  let big=0;const cnt=new Array(r.k).fill(0);for(let i=0;i<N;i++)if(r.lab[i]>=0)cnt[r.lab[i]]++;cnt.forEach(c=>big=Math.max(big,c));r.cnt=cnt;r.big=big;r.ncore=r.core.reduce((a,b)=>a+b,0);return (RUNS[key]=r);}
const MAINM=()=>runDB(ensureD()&&KN[5].eps,5),MS3=()=>runDB(ensureD()&&KN[3].eps,3);
const HDB_LAB=Array.from(HDB.lab).map(ch=>ch==='.'?-1:parseInt(ch,36));
/* 채점 (봉투를 연 뒤에만 부른다): ARI · NMI · 업무 점수 */
const SC=new Map();
function scoreAll(lab){if(SC.has(lab))return SC.get(lab);let noise=0,K=0;for(let i=0;i<lab.length;i++){if(lab[i]<0)noise++;else K=Math.max(K,lab[i]+1);}
  const pr=DB.propagate(P,lab,Y),s={ari:DB.ari(Y,lab),nmi:DB.nmi(Y,lab),biz:pr.acc,pr,noise,K,labels:K+noise,bizAll:pr.acc+noise/N};SC.set(lab,s);return s;}
let RAND_ARI=null;function randARI(lab){const a=Array.from(lab),r=RNG(0);for(let i=a.length-1;i>0;i--){const j=Math.floor(r()*(i+1)),t=a[i];a[i]=a[j];a[j]=t;}return DB.ari(Y,a);}
let MAN_S=null;function manS(){return MAN_S||(MAN_S=scoreAll(MAN));}
let ENV=false;

/* ── 히어로: 밀도 지도 ── */
heroRoom($('heroCv'),{sym:'ε',no:'APP-12',name:'DBSCAN\n(밀도 군집)',role:'지원자 · 군집팀',banner:'모델 채용 위원회 · 5-12 면접',doc:'front',asst:'board',fly:'clip',robo:'calc',loop:11,
  beats:[{at:0.4,who:'fly',mood:'point',move:'shake',say:'외톨이는?'},{at:2.4,who:'robo',mood:'dq',move:'pop',say:'잡음 처리비 경고!'},{at:4.4,who:'doc',mood:'board',move:'nod',say:'촘촘하면 한 묶음'},
         {at:6.4,who:'asst',mood:'shock',move:'shake',say:'반지름이 문제예요'},{at:8.4,who:'fly',mood:'type',move:'nod',say:'잡음 수 적어 둬'}],
  motif:(function(){const r=RNG(12),pts=[];for(let i=0;i<16;i++){const a=r()*6.28,d=Math.sqrt(r())*20;pts.push([300+Math.cos(a)*d*1.3,150+Math.sin(a)*d,0]);}
    for(let i=0;i<14;i++){const a=r()*6.28,d=Math.sqrt(r())*16;pts.push([372+Math.cos(a)*d,118+Math.sin(a)*d*1.4,1]);}
    [[272,96],[404,182],[338,200],[420,92]].forEach(p=>pts.push([p[0],p[1],-1]));
    return function(ctx,t,W,H,C){const ph=REDUCE?0.6:(Math.sin(t*1.4)+1)/2,R=10+ph*16,cx=pts[0][0],cy=pts[0][1];
      ctx.fillStyle=rgba(C.blue,.08);ctx.strokeStyle=rgba(C.blue,.55);ctx.lineWidth=1.5;ctx.setLineDash([4,3]);ctx.beginPath();ctx.arc(cx,cy,R,0,7);ctx.fill();ctx.stroke();ctx.setLineDash([]);
      pts.forEach(p=>{if(p[2]<0)xcross(ctx,p[0],p[1],3.5,C.muted,1.8);else circ(ctx,p[0],p[1],3.6,p[2]?C.amber:C.blue,null);});
      ctx.fillStyle=C.muted;ctx.font=font(12,500,1);ctx.textAlign='left';ctx.fillText('ε = '+(R/2).toFixed(1)+' · × = 잡음',276,226);};})()});

/* ── 01 장면 ── */
comic('comicIntro',[
 {bg:'night',prop:'촘촘한 동네',cast:[{who:'doc',mood:'board',x:30,move:'nod',h:104},{who:'asst',mood:'ledger',x:74,flip:true}],say:{who:'doc',txt:'반지름 안에 이웃이 다섯 장 넘게 있으면 동네 한가운데, 핵심점이에요. 핵심점끼리 이으면 한 묶음.'},cap:'① 촘촘한 동네 — 핵심점끼리 이어 한 묶음'},
 {bg:'night',prop:'가장자리',cast:[{who:'asst',mood:'spark',x:30,move:'jump'},{who:'fly',mood:'type',x:74,move:'nod',flip:true}],say:{who:'asst',txt:'이 글씨는 이웃이 셋뿐인데 핵심점 바로 곁이에요. 같은 묶음 가장자리에 넣었어요.'},cap:'② 가장자리 — 스스로는 성기지만 동네에 붙는다'},
 {bg:'night',prop:'잡음 −1',cast:[{who:'fly',mood:'point',x:28,move:'shake'},{who:'robo',mood:'dq',x:72,move:'pop',h:104}],say:{who:'robo',txt:'어디에도 안 닿는 외톨이는 버리는 게 아니라 사람한테 넘기는 거야. 장당 인건비!'},cap:'③ 잡음 — 억지로 넣지 않고 따로 내놓는다'}]);

/* ── 01 자기소개: 2차원 연습 지도 ── */
const TOY=(function(){const r=RNG(7),g=()=>{let u=0,v=0;while(!u)u=r();v=r();return Math.sqrt(-2*Math.log(u))*Math.cos(6.2832*v);},pts=[];
  for(let i=0;i<75;i++){const a=Math.PI*i/74;pts.push([24+16*Math.cos(a)+g()*1.5,26+16*Math.sin(a)+g()*1.5,0]);}
  for(let i=0;i<75;i++){const a=Math.PI+Math.PI*i/74;pts.push([40+16*Math.cos(a)+g()*1.5,20+16*Math.sin(a)+g()*1.5,1]);}
  for(let i=0;i<48;i++)pts.push([80+g()*3.6,40+g()*3.6,2]);
  for(let i=0;i<22;i++)pts.push([4+r()*92,4+r()*52,3]);
  const n=pts.length,D2=new Float64Array(n*n);for(let i=0;i<n;i++)for(let j=0;j<n;j++){const dx=pts[i][0]-pts[j][0],dy=pts[i][1]-pts[j][1];D2[i*n+j]=dx*dx+dy*dy;}
  return {pts,n,D2};})();
(function(){const cv=$('toyCv'),st={eps:5,ms:5,sel:-1,km:false,play:false,shown:Infinity,res:null,kml:null};
  function kmeans(){const X=TOY.pts,c=[X[0],X[75],X[150]].map(p=>[p[0],p[1]]),lab=new Array(TOY.n).fill(0);
    for(let it=0;it<60;it++){X.forEach((p,i)=>{let b=0,bd=1e9;c.forEach((q,k)=>{const d=(p[0]-q[0])**2+(p[1]-q[1])**2;if(d<bd){bd=d;b=k;}});lab[i]=b;});
      c.forEach((q,k)=>{let sx=0,sy=0,m=0;X.forEach((p,i)=>{if(lab[i]===k){sx+=p[0];sy+=p[1];m++;}});if(m){q[0]=sx/m;q[1]=sy/m;}});}return {lab,c};}
  function compute(){st.res=DB.run(TOY.D2,TOY.n,st.eps,st.ms,true);}
  function geo(){const W=cv.clientWidth||600,s=Math.min((W-20)/100,6.2),H=Math.round(60*s+20);return {W,H,s,ox:(W-100*s)/2,oy:10};}
  function draw(){const g=geo(),W=fit(cv,g.H),ctx=cv.getContext('2d'),C=cols(),sx=x=>g.ox+x*g.s,sy=y=>g.oy+(60-y)*g.s,R=st.res;
    ctx.clearRect(0,0,W,g.H);ctx.strokeStyle=C.rule;ctx.strokeRect(g.ox,g.oy,100*g.s,60*g.s);
    const rank=new Int32Array(TOY.n).fill(1e9);R.order.forEach((p,k)=>rank[p]=k);
    if(st.sel>=0){const p=TOY.pts[st.sel];ctx.fillStyle=rgba(C.blue,.10);ctx.strokeStyle=C.blue;ctx.lineWidth=1.6;ctx.setLineDash([5,4]);ctx.beginPath();ctx.arc(sx(p[0]),sy(p[1]),st.eps*g.s,0,7);ctx.fill();ctx.stroke();ctx.setLineDash([]);}
    TOY.pts.forEach((p,i)=>{const x=sx(p[0]),y=sy(p[1]);
      if(st.km){circ(ctx,x,y,4,rgba(PAL[st.kml.lab[i]],.85),null);return;}
      const c=R.lab[i],on=rank[i]<st.shown||c<0&&!st.play;
      if(st.play&&rank[i]>=st.shown){circ(ctx,x,y,3,rgba(C.muted,.35),null);return;}
      if(c<0)xcross(ctx,x,y,3.6,C.muted,1.8);else if(R.core[i])circ(ctx,x,y,4.4,CC(c),null);else circ(ctx,x,y,4,C.panel,CC(c),2);});
    if(st.km){st.kml.c.forEach(q=>{ctx.fillStyle=C.text;ctx.font=font(18,700);ctx.textAlign='center';ctx.fillText('+',sx(q[0]),sy(q[1])+6);});}
    if(st.sel>=0){const p=TOY.pts[st.sel];circ(ctx,sx(p[0]),sy(p[1]),7,null,C.stamp,2.4);}
    ctx.font=font(12,700);ctx.textAlign='left';ctx.fillStyle=C.muted;ctx.fillText(st.km?'K-Means k = 3 (+ = 중심)':'● 핵심점  ○ 가장자리  × 잡음',g.ox+6,g.oy+16);info();}
  function info(){const R=st.res;let core=0,border=0;for(let i=0;i<TOY.n;i++){if(R.lab[i]<0)continue;R.core[i]?core++:border++;}
    let h=st.km?'<b>K-Means(k=3)</b>: 점마다 가장 가까운 중심(+)에 넣는다. 맞물린 초승달 두 개를 가운데로 잘라 서로 섞고, 흩어진 외톨이까지 모두 어딘가에 넣는다. 묶음 수 3은 사람이 정해 준 값이다. 다시 누르면 DBSCAN으로 돌아간다.':
      'eps <b>'+st.eps.toFixed(1)+'</b>, min_samples <b>'+st.ms+'</b> → 묶음 <b>'+R.k+'개</b> · 핵심점 '+core+' · 가장자리 '+border+' · 잡음 <b>'+R.noise+'</b>. ';
    if(!st.km){if(st.sel>=0){const i=st.sel,m=R.nbCount[i];h+='누른 점: 반지름 안 '+m+'개(자기 포함) → '+(R.core[i]?'<b>핵심점</b>':R.lab[i]>=0?'<b>가장자리 점</b>(핵심점 곁이라 묶음 '+(R.lab[i]+1)+'에 붙었다)':'<b>잡음</b>(어느 핵심점의 원에도 들지 않는다)')+'. ';}
      else h+='점을 하나 눌러 보자. ';
      h+=R.k>3&&st.eps<=4?'반지름이 너무 작다. 초승달이 여러 조각으로 끊기고 잡음이 많다.':R.k<3&&st.eps>=6?'반지름이 너무 크다. 맞물린 초승달이 사슬로 이어져 한 묶음이 된다.':R.k===3?'초승달 둘과 둥근 무리가 모양 그대로 셋으로 나뉘었다. 묶음 수 3은 아무도 알려 주지 않았다.':R.k>3?'작은 조각 묶음이 생겼다. min_samples를 바꿔 보자.':'';}
    $('toyInfo').innerHTML=h;}
  function pick(e){const g=geo(),r=cv.getBoundingClientRect(),x=(e.clientX-r.left-g.ox)/g.s,y=60-(e.clientY-r.top-g.oy)/g.s;let b=-1,bd=1e9;TOY.pts.forEach((p,i)=>{const d=(p[0]-x)**2+(p[1]-y)**2;if(d<bd){bd=d;b=i;}});st.sel=bd<25?b:-1;draw();}
  cv.addEventListener('click',pick);
  $('toyEps').addEventListener('input',e=>{st.eps=+e.target.value;$('toyEpsO').textContent=st.eps.toFixed(1);if(st.play)return;compute();draw();});
  seg('#toyMs button',v=>{st.ms=+v;if(st.play)return;compute();draw();});
  $('toyKm').addEventListener('click',()=>{if(st.play)return;st.km=!st.km;if(st.km&&!st.kml)st.kml=kmeans();$('toyKm').setAttribute('aria-pressed',st.km?'true':'false');draw();});
  $('toyPlay').addEventListener('click',async()=>{if(st.play)return;st.km=false;$('toyKm').setAttribute('aria-pressed','false');st.play=true;$('toyPlay').disabled=true;compute();const L=st.res.order.length;
    for(let k=0;k<=L;k+=REDUCE?L:3){st.shown=k;draw();await wait(28);}st.play=false;st.shown=Infinity;$('toyPlay').disabled=false;draw();});
  compute();draw();onRedraw(draw);})();

/* ── 02 실기 과제지: 글씨 격자 · 64칸 · 매뉴얼 씨 8묶음 ── */
(function(){const cv=$('digCv'),pc=$('pixCv');let sel=0;const NS=160;
  function geo(){const W=cv.clientWidth||500,cols_=W<420?10:16,s=Math.floor((W-4)/cols_);return {W,cols_,s,H:Math.ceil(NS/cols_)*s+4};}
  function draw(){const g=geo(),W=fit(cv,g.H),ctx=cv.getContext('2d'),C=cols(),dark=getComputedStyle(document.documentElement).colorScheme==='dark',ink=dark?'#e8ecf3':'#111827';ctx.clearRect(0,0,W,g.H);
    for(let i=0;i<NS;i++){const x=2+(i%g.cols_)*g.s,y=2+Math.floor(i/g.cols_)*g.s;ctx.fillStyle=C.panel2;ctx.fillRect(x+1,y+1,g.s-2,g.s-2);drawDigit(ctx,P[i],x+3,y+3,g.s-6,ink);if(i===sel){ctx.strokeStyle=C.stamp;ctx.lineWidth=2.5;ctx.strokeRect(x+1,y+1,g.s-2,g.s-2);}}
    drawPix();}
  function drawPix(){const s=Math.floor(Math.min(pc.clientWidth||300,300)/8),W=fit(pc,s*8+2),ctx=pc.getContext('2d'),C=cols(),dark=getComputedStyle(document.documentElement).colorScheme==='dark';
    ctx.clearRect(0,0,W,s*8+2);const px=P[sel];for(let j=0;j<64;j++){const x=(j%8)*s,y=Math.floor(j/8)*s,v=px[j]/16;ctx.fillStyle=dark?'rgba(232,236,243,'+v+')':'rgba(17,24,39,'+v+')';ctx.fillRect(x,y,s-1,s-1);ctx.strokeStyle=C.rule;ctx.strokeRect(x,y,s-1,s-1);
      ctx.fillStyle=v>0.5?(dark?'#111827':'#fff'):C.muted;ctx.font=font(Math.max(10,s*0.36),700,1);ctx.textAlign='center';ctx.fillText(px[j],x+s/2,y+s/2+4);}
    const f=digitFeat(px);$('pixInfo').innerHTML='<b>'+(sel+1)+'번째 글씨</b>의 특징 64개(밝기 0~16). 잉크 합 '+f.ink+', 위쪽 비율 '+f.top.toFixed(2)+', 가운데 네 칸 '+f.center+' → 매뉴얼 씨의 '+(manualDigit(px)+1)+'번 묶음. 정답 숫자는 봉투 속에 있다.';}
  cv.addEventListener('click',e=>{const g=geo(),r=cv.getBoundingClientRect(),c=Math.floor((e.clientX-r.left-2)/g.s),rw=Math.floor((e.clientY-r.top-2)/g.s),i=rw*g.cols_+c;if(c>=0&&c<g.cols_&&i>=0&&i<NS){sel=i;draw();}});
  draw();onRedraw(draw);
  /* 매뉴얼 씨 8묶음 */
  const QN=['가운데 참','가운데 빔'],cnt=new Array(8).fill(0);MAN.forEach(m=>cnt[m]++);let g8=4;
  $('manPick').innerHTML=cnt.map((c,k)=>'<button data-v="'+k+'" aria-pressed="'+(k===g8)+'">'+(k+1)+'번 · '+c+'장</button>').join('');
  const mc=$('manCv');
  function drawM(){const W=mc.clientWidth||500,cols_=W<420?10:20,s=Math.floor((W-4)/cols_),ids=[];for(let i=0;i<N&&ids.length<cols_*3;i++)if(MAN[i]===g8)ids.push(i);const H=Math.ceil(ids.length/cols_)*s+4;fit(mc,H);
    const ctx=mc.getContext('2d'),C=cols(),dark=getComputedStyle(document.documentElement).colorScheme==='dark';ctx.clearRect(0,0,W,H);
    ids.forEach((i,k)=>{const x=2+(k%cols_)*s,y=2+Math.floor(k/cols_)*s;ctx.fillStyle=C.panel2;ctx.fillRect(x+1,y+1,s-2,s-2);drawDigit(ctx,P[i],x+3,y+3,s-6,dark?'#e8ecf3':'#111827');});
    $('manInfo').innerHTML='<b>'+(g8+1)+'번 묶음</b> = '+(g8&4?'가운데가 비었고':'가운데가 찼고')+', '+(g8&2?'잉크가 많고(300 초과)':'잉크가 적고')+', '+(g8&1?'위가 무겁다':'아래가 무겁다')+'. '+cnt[g8]+'장 중 앞의 '+ids.length+'장. 같은 묶음에 모양이 다른 숫자가 섞여 있다. 세 질문으로는 0·6·8처럼 "가운데가 빈 굵은 글씨"를 가르지 못한다. 그래도 이 규칙의 ARI 0.108이 합격선이다.';}
  seg('#manPick button',v=>{g8=+v;drawM();});drawM();onRedraw(drawM);})();

/* ── 03 AI 비서 ── */
const MAIN_PROMPT=`너는 머신러닝 수업 조교다. Google Colab에서 바로 실행되는 파이썬 코드를 써 줘.

[환경] Colab 기본 라이브러리(numpy, scikit-learn)만 쓴다. 다른 패키지는 설치하지 않는다. 첫 줄에 scikit-learn 버전을 출력한다.
[데이터] 〔sklearn.datasets.load_digits〕의 X(64칸 픽셀)만으로 묶는다. 정답 y는 채점에만 쓴다.
[규정]
  1) DBSCAN(scikit-learn)을 쓴다. 특징은 〔모두 같은 단위(밝기 0~16)라 표준화하지 않는다〕.
  2) eps는 정답 없이 고른다: min_samples=〔5〕번째 이웃(자기 포함)까지의 거리를 정렬한 k-거리 그래프를 그리고, 무릎(양 끝을 잇는 직선에서 가장 먼 점)을 eps로 삼는다.
  3) 묶음 수는 labels_.max() + 1로 세고, 잡음(-1) 수와 비율을 따로 출력한다.
  4) 정답 없이 재는 점수(잡음을 뺀 실루엣)를 먼저 출력하고, 그다음에 정답 y로 ARI·NMI를 잰다. 잡음도 한 묶음으로 넣어 채점한다.
  5) 업무 점수: 묶음마다 평균에 가장 가까운 한 장의 정답을 묶음 전체에 붙였을 때 맞는 비율(잡음은 틀린 것으로 센다).
[출력] k-거리 그래프, 고른 eps, 묶음 수, 잡음 수, 실루엣, ARI, NMI, 업무 점수, 합격선 〔ARI 0.108〕과 비교.
[금지] 정답 y로 eps나 min_samples를 고르지 않는다. 잡음을 빼고 ARI를 재지 않는다. 데이터를 지어내지 않는다.
[형식] 셀 단위로 나누고, 각 셀 위에 무엇을 하는지 한국어 주석을 단다. 정답 y를 쓰는 줄에는 "# [채점]"을 붙인다. 코드 뒤에 내가 확인해야 할 점 3가지를 적어 줘.`;
$('mainPrompt').textContent=MAIN_PROMPT;$('copyMain').addEventListener('click',()=>copyText(MAIN_PROMPT,'프롬프트를 복사했습니다'));
(function(){const LINES=[
 ['import numpy as np'],['from sklearn.datasets import load_digits'],['from sklearn.cluster import DBSCAN'],['from sklearn.preprocessing import StandardScaler',0,'불러오는 것 자체는 괜찮다. 어디에 쓰는지가 문제다(8번 줄).'],['from sklearn.metrics import adjusted_rand_score',0,'ARI 채점 함수. 괜찮다.'],[''],
 ['X, y = load_digits(return_X_y=True)',0,'X는 64칸 픽셀, y는 정답. 정답을 받아 두는 것은 괜찮다. 어디에 쓰는지가 문제다.'],
 ['X = StandardScaler().fit_transform(X)   # 거리를 재니까 표준화부터',1,'<b>같은 단위의 픽셀을 표준화했다.</b> 64칸이 모두 밝기 0~16이라 표준화할 까닭이 없다. 오히려 가장자리처럼 거의 늘 0인 칸은 표준편차가 아주 작아, 얼룩 하나가 큰 값으로 부풀어 거리를 흔든다. 정답을 엿보고 eps를 골라도 ARI가 원래 픽셀 0.614 → 표준화 0.248로 떨어진다(Python 확인).'],[''],
 ['best = 0',2,'<b>정답으로 eps를 고르는 반복문이 시작된다.</b> 아래 네 줄이 한 덩어리다.'],
 ['for eps in np.arange(0.5, 10, 0.5):',2,'<b>정답으로 eps를 고른다.</b> 비지도 업무에는 채점 봉투가 없다. 이렇게 고른 eps의 점수는 실제로는 낼 수 없는 점수다. eps는 k-거리 그래프처럼 정답 없이 고른다.'],
 ['    labels = DBSCAN(eps=eps).fit_predict(X)',0,'eps를 바꿔 가며 묶는 것 자체는 괜찮다(민감도 보기). 고르는 기준이 문제다.'],
 ['    score = adjusted_rand_score(y, labels)',2,'<b>정답 y로 점수를 매겨 eps를 고른다.</b> 채점 봉투를 근무 조건 고르는 데 썼다.'],
 ['    if score > best: best, best_eps = score, eps',2,'<b>정답 점수가 가장 높은 eps를 골랐다.</b> 위와 같은 문제다.'],[''],
 ['labels = DBSCAN(eps=best_eps, min_samples=1).fit_predict(X)',3,'<b>min_samples=1</b>이면 모든 점이 스스로 핵심점이다. 잡음이 사라지고, 외톨이 한 장도 혼자 한 묶음이 된다. 계층적 군집의 single 연결을 eps 높이에서 자른 것과 같아져 사슬 효과가 그대로 온다. 원래 픽셀에서 eps 21이면 묶음이 233개다. 이름표 233장이 필요하다.'],
 ['n_clusters = len(set(labels))',4,'<b>잡음 −1까지 묶음 하나로 셌다.</b> 잡음이 있으면 묶음 수가 하나 부풀려진다. <code>labels.max() + 1</code>로 세고 잡음 수 <code>(labels == -1).sum()</code>를 따로 보고한다.'],
 ['mask = labels != -1',5,'<b>잡음을 빼고 채점하려 한다.</b> 바로 아랫줄과 한 덩어리다.'],
 ['print("묶음", n_clusters, "ARI", adjusted_rand_score(y[mask], labels[mask]))',5,'<b>잡음을 빼고 ARI를 쟀다.</b> 어려운 글씨를 잡음으로 다 버리면 남은 글씨로는 점수가 높게 나온다. eps 16이면 1,380장(77%)을 버리고 남은 417장의 ARI가 0.84, 전체로 재면 0.04다. 잡음도 한 묶음으로 넣어 채점하고, 잡음 수를 함께 보고한다.']];
  const host=$('bugCode'),found=new Set();
  host.innerHTML=LINES.map((l,i)=>'<div class="ln" role="listitem" tabindex="0" data-i="'+i+'"><span class="no">'+(i+1)+'</span><span>'+hl(l[0]||' ')+'</span></div>').join('');
  function score(){$('bugScore').textContent=found.size+' / 5';$('bugScore').className='pill'+(found.size===5?' ok':'');}
  function pick(el){const i=+el.getAttribute('data-i'),l=LINES[i];if(!l[0])return;
    if(l[1]){found.add(l[1]);host.querySelectorAll('.ln').forEach(x=>{if(LINES[+x.getAttribute('data-i')][1]===l[1])x.classList.add('hit');});$('bugMsg').innerHTML=(i+1)+'번 줄: '+l[2]+(found.size===5?'<br><b>다섯 군데를 모두 찾았다.</b> 고친 코드는 A장의 Colab 코드와 같은 순서(픽셀 그대로 → k-거리로 eps → 묶음 수와 잡음 따로 → 잡음까지 넣어 채점)를 따른다.':'');}
    else{el.classList.add('miss');$('bugMsg').innerHTML=(i+1)+'번 줄: '+(l[2]||'이 줄은 괜찮다.');}score();}
  host.addEventListener('click',e=>{const el=e.target.closest('.ln');if(el)pick(el);});
  host.addEventListener('keydown',e=>{if((e.key==='Enter'||e.key===' ')&&e.target.classList.contains('ln')){e.preventDefault();pick(e.target);}});
  $('bugReveal').addEventListener('click',()=>{LINES.forEach((l,i)=>{if(l[1]){found.add(l[1]);host.querySelector('[data-i="'+i+'"]').classList.add('hit','reveal');}});score();$('bugMsg').innerHTML='빨간 줄 다섯 군데: ① 8번 같은 단위의 픽셀을 표준화 ② 10·11·13·14번 정답으로 eps 고르기 ③ 16번 min_samples=1(잡음이 사라진다) ④ 17번 잡음까지 묶음으로 세기 ⑤ 18·19번 잡음을 빼고 채점.';});
  $('bugReset').addEventListener('click',()=>{found.clear();host.querySelectorAll('.ln').forEach(x=>x.classList.remove('hit','miss','reveal'));score();$('bugMsg').textContent='줄을 하나 눌러 보세요.';});score();})();
(function(){const CK=[['특징 X만으로 묶었나','fit(X)에 정답 y가 섞여 들어가지 않았는지 본다. X의 모양이 (1797, 64)인지 확인한다.'],['정답 y를 채점에만 썼나','eps·min_samples를 y 점수로 고르지 않았는지 본다. y를 쓰는 줄마다 표시를 달게 하면 찾기 쉽다.'],['표준화가 필요한 데이터인가','단위가 같은 픽셀이면 그대로 둔다. 단위가 다른 특징이면 표준화하고 eps를 다시 고른다.'],['eps가 데이터의 거리 단위에 맞나','기본값 0.5는 이 데이터에서 전부 잡음이다. k-거리 그래프의 눈금과 비교한다.'],['min_samples가 1이 아닌가','1이면 잡음이 사라지고 사슬처럼 이어진다. 기본값 5부터 본다.'],['묶음 수를 labels_.max() + 1로 셌나','len(set(labels))는 잡음(-1)까지 하나로 센다.'],['잡음 수와 비율을 보고했나','잡음이 많을수록 사람이 직접 볼 글씨가 많다. 업무 비용이다.'],['잡음을 빼고 채점하지 않았나','잡음까지 넣어 ARI를 잰다. 뺀다면 뺀 수와 비율을 나란히 적는다.'],['결과를 합격선·무작위와 비교했나','ARI 0.108(매뉴얼 씨)과 무작위(≈ 0)보다 낫지 않으면 뽑을 까닭이 없다.']];
  const ul=$('checks');ul.innerHTML=CK.map((c,i)=>'<li><input type="checkbox" id="ck'+i+'"><label for="ck'+i+'"><b>'+c[0]+'</b><small>'+c[1]+'</small></label></li>').join('');
  function upd(){const n=ul.querySelectorAll('input:checked').length;ul.querySelectorAll('li').forEach(li=>li.classList.toggle('done',li.querySelector('input').checked));$('checkScore').textContent='확인 '+n+' / '+CK.length+(n===CK.length?' — 이제 돌려 본다. 결과도 확인 대상이다.':'');}
  ul.addEventListener('change',upd);upd();})();

/* ── 04 수습 기간 ① k-거리 그래프 ── */
const KD=(function(){const cv=$('kdCv'),H=320,st={ms:5,eps:null,ready:false};const toyK={};
  function toyKd(ms){if(toyK[ms])return toyK[ms];const n=TOY.n,o=new Float64Array(n),row=new Float64Array(n);for(let i=0;i<n;i++){row.set(TOY.D2.subarray(i*n,i*n+n));row.sort();o[i]=Math.sqrt(row[ms-1]);}return (toyK[ms]=DB.knee(o));}
  function draw(){const W=fit(cv,H),ctx=cv.getContext('2d'),C=cols(),pl=50,pr=14,pt=12,pb=38,gw=W-pl-pr,gh=H-pt-pb;ctx.clearRect(0,0,W,H);axes(ctx,C,pl,pt,gw,gh);
    if(!st.ready){ctx.textAlign='center';ctx.fillText('거리 계산을 기다리는 중…',pl+gw/2,pt+gh/2);return;}
    const K=KN[st.ms],s=K.sorted,y0=8,y1=36,lx=i=>pl+i/(N-1)*gw,ly=v=>pt+(1-(clamp(v,y0,y1)-y0)/(y1-y0))*gh;
    [10,15,20,25,30,35].forEach(v=>{ctx.textAlign='right';ctx.fillStyle=C.muted;ctx.fillText(v,pl-5,ly(v)+4);ctx.strokeStyle=rgba(C.muted,.15);ctx.beginPath();ctx.moveTo(pl,ly(v));ctx.lineTo(pl+gw,ly(v));ctx.stroke();});
    [0,500,1000,1500,1797].forEach(i=>{ctx.textAlign='center';ctx.fillStyle=C.muted;ctx.fillText(i,lx(i),H-22);});ctx.fillText('글씨 (k-거리가 작은 것부터)',pl+gw/2,H-5);vlabel(ctx,st.ms+'번째 이웃까지 거리',12,pt+gh/2);
    /* eps 위 = 핵심점 못 됨 */
    const e=st.eps,over=s.filter(v=>v>e).length;ctx.fillStyle=rgba(C.stamp,.07);ctx.fillRect(lx(N-over),pt,gw-(lx(N-over)-pl),gh);
    ctx.strokeStyle=rgba(C.muted,.5);ctx.setLineDash([2,3]);ctx.beginPath();ctx.moveTo(lx(0),ly(s[0]));ctx.lineTo(lx(N-1),ly(s[N-1]));ctx.stroke();ctx.setLineDash([]);
    ctx.strokeStyle=C.blue;ctx.lineWidth=2.6;ctx.beginPath();for(let i=0;i<N;i++){const X=lx(i),Y=ly(s[i]);i?ctx.lineTo(X,Y):ctx.moveTo(X,Y);}ctx.stroke();
    circ(ctx,lx(K.i),ly(s[K.i]),7,null,C.stamp,2.5);
    ctx.strokeStyle=C.stamp;ctx.setLineDash([6,4]);ctx.lineWidth=1.6;ctx.beginPath();ctx.moveTo(pl,ly(e));ctx.lineTo(pl+gw,ly(e));ctx.stroke();ctx.setLineDash([]);
    ctx.fillStyle=C.stamp;ctx.font=font(12,700);ctx.textAlign='right';ctx.fillText('eps '+e.toFixed(2)+(Math.abs(e-K.eps)<1e-9?' (무릎)':''),pl+gw-6,ly(e)-6);
    /* 안쪽 작은 그림: 2차원 연습 지도의 k-거리 */
    const T=toyKd(st.ms),ts=T.sorted,iw=Math.min(240,gw*0.42),ih=Math.min(120,gh*0.45),ix=pl+10,iy=pt+10,tn=ts.length,tmax=ts[tn-1];
    ctx.fillStyle=rgba(C.panel,.94);ctx.fillRect(ix,iy,iw,ih);ctx.strokeStyle=C.rule;ctx.strokeRect(ix,iy,iw,ih);
    ctx.strokeStyle=C.muted;ctx.lineWidth=2;ctx.beginPath();for(let i=0;i<tn;i++){const X=ix+6+i/(tn-1)*(iw-12),Y=iy+ih-8-ts[i]/tmax*(ih-26);i?ctx.lineTo(X,Y):ctx.moveTo(X,Y);}ctx.stroke();
    circ(ctx,ix+6+T.i/(tn-1)*(iw-12),iy+ih-8-ts[T.i]/tmax*(ih-26),5,null,C.stamp,2);
    ctx.fillStyle=C.text;ctx.font=font(11.5,700);ctx.textAlign='left';ctx.fillText('비교: 01장 2차원 지도 (무릎이 뚜렷)',ix+6,iy+14);}
  function info(){const K=KN[st.ms],r=runDB(st.eps,st.ms),over=K.sorted.filter(v=>v>st.eps).length;
    $('kdInfo').innerHTML='min_samples <b>'+st.ms+'</b>: k-거리 중앙값 '+K.sorted[(N-1)>>1].toFixed(2)+' · 무릎 <b>'+(K.i+1)+'번째 점</b>(위쪽 '+pct(1-K.i/N,0)+') → eps <b>'+K.eps.toFixed(2)+'</b>. 지금 eps '+st.eps.toFixed(2)+'에서 k-거리가 eps보다 큰 글씨 '+over+'장(빨간 바탕)은 핵심점이 못 된다. 이 eps로 묶으면 <b>묶음 '+r.k+'개, 잡음 '+r.noise+'장('+pct(r.noise/N)+')</b>, 가장 큰 묶음 '+r.big+'장('+pct(r.big/N)+'). 정답은 아직 쓰지 않았다. '+
      (r.big/N>0.8?'<b>가장 큰 묶음이 거의 전부다.</b> 서로 다른 숫자 무리가 사슬로 이어졌다는 신호다. 무릎은 "잡음은 10%쯤"이라는 뜻일 뿐, 숫자 무리를 가르는 반지름이라는 보장은 없다.':r.noise/N>0.4?'잡음이 너무 많다. 사람이 직접 볼 글씨가 절반 가까이 된다.':'묶음과 잡음이 함께 있다. 이 eps가 좋은지는 05장에서 봉투를 열어야 안다.');}
  function setEps(v){st.eps=v;$('kdEps').value=v;$('kdEpsO').textContent=v.toFixed(2);draw();info();}
  $('kdEps').addEventListener('input',e=>{if(!st.ready)return;setEps(+e.target.value);});
  $('kdKnee').addEventListener('click',()=>{if(st.ready)setEps(KN[st.ms].eps);});
  seg('#kdMs button',v=>{st.ms=+v;if(st.ready)setEps(KN[st.ms].eps);});
  draw();onRedraw(draw);
  return {start(){ensureD();st.ready=true;setEps(KN[st.ms].eps);}};})();

/* ── 04 ② 수습 재생: eps를 키우면 ── */
const GR=(function(){const cv=$('grCv'),bc=$('grBar'),EPS=[];for(let e=16;e<=27.001;e+=0.5)EPS.push(+e.toFixed(1));let cur=20,ready=false,busy=false;
  function res(e){return runDB(e,5);}
  function draw(){drawMapOn(cv,ready?res(cur).lab:null,{h:330,note:ready?'eps '+cur.toFixed(1)+' · 묶음 '+res(cur).k+'개 · 잡음 '+res(cur).noise+'장':''});drawBar();}
  function drawBar(){const H=200,W=fit(bc,H),ctx=bc.getContext('2d'),C=cols(),pl=40,pr=10,pt=10,pb=30,gw=W-pl-pr,gh=H-pt-pb;ctx.clearRect(0,0,W,H);axes(ctx,C,pl,pt,gw,gh);if(!ready)return;
    const bw=gw/EPS.length,lx=k=>pl+k*bw+bw/2,ly=v=>pt+(1-v)*gh;[0,0.5,1].forEach(v=>{ctx.textAlign='right';ctx.fillStyle=C.muted;ctx.fillText(pct(v,0),pl-4,ly(v)+4);});
    EPS.forEach((e,k)=>{const r=res(e);ctx.fillStyle=rgba(C.muted,e===cur?.75:.38);ctx.fillRect(pl+k*bw+1,ly(r.noise/N),bw-2,gh-(ly(r.noise/N)-pt));if(k%2===0){ctx.fillStyle=C.muted;ctx.textAlign='center';ctx.fillText(e,lx(k),H-15);}});
    ctx.strokeStyle=C.blue;ctx.lineWidth=2.4;ctx.beginPath();EPS.forEach((e,k)=>{const Y=ly(res(e).big/N);k?ctx.lineTo(lx(k),Y):ctx.moveTo(lx(k),Y);});ctx.stroke();
    EPS.forEach((e,k)=>{if(e===cur)circ(ctx,lx(k),ly(res(e).big/N),5,C.stamp,null);});
    ctx.textAlign='center';ctx.fillStyle=C.muted;ctx.fillText('eps',pl+gw/2,H-2);ctx.font=font(11.5,700);ctx.textAlign='left';ctx.fillStyle=C.blue;ctx.fillText('— 가장 큰 묶음 몫',pl+6,pt+14);ctx.fillStyle=C.muted;ctx.fillText('▮ 잡음 몫',pl+6,pt+28);}
  function info(){const r=res(cur);$('grInfo').innerHTML='eps <b>'+cur.toFixed(1)+'</b>: 묶음 <b>'+r.k+'개</b>, 잡음 <b>'+pct(r.noise/N)+'</b>, 가장 큰 묶음 <b>'+pct(r.big/N)+'</b>. '+
    (cur<=18?'반지름이 작아 촘촘한 한가운데만 묶이고 대부분이 회색 잡음이다.':cur<=21.5?'숫자 무리들이 저마다 색을 얻는다. 잡음이 줄면서 묶음 수가 가장 많아지는 구간이다.':cur<=24?'이웃 무리끼리 사슬로 이어지기 시작한다. 몇몇 색이 한 색으로 합쳐진다.':'거의 모든 글씨가 한 묶음이다. 반지름 몇 칸 차이로 무리 사이의 성긴 곳이 사라졌다.');}
  function set(e){cur=e;$('grEps').value=e;$('grEpsO').textContent=e.toFixed(1);draw();info();}
  $('grEps').addEventListener('input',e=>{if(ready)set(+e.target.value);});
  $('grPlay').addEventListener('click',async()=>{if(!ready||busy)return;busy=true;$('grPlay').disabled=true;for(const e of EPS){set(e);await wait(REDUCE?0:380);}busy=false;$('grPlay').disabled=false;});
  draw();onRedraw(draw);
  return {start(){ensureD();EPS.forEach(res);ready=true;set(cur);}};})();
/* 2D 지도 그리기 (t-SNE 위치는 그림용, 묶음은 64차원 계산) */
function drawMapOn(cv,lab,o){o=o||{};const H=o.h||360,W=fit(cv,H),ctx=cv.getContext('2d'),C=cols(),pad=14,s=Math.min((W-2*pad)/100,(H-2*pad)/100),ox=(W-100*s)/2,oy=(H-100*s)/2;ctx.clearRect(0,0,W,H);
  ctx.strokeStyle=C.rule;ctx.strokeRect(ox-4,oy-4,100*s+8,100*s+8);
  for(let i=0;i<N;i++){const x=ox+MAP[2*i]*s,y=oy+(100-MAP[2*i+1])*s;
    if(!lab){circ(ctx,x,y,1.8,rgba(C.muted,.5),null);continue;}
    const c=o.truth?lab[i]:lab[i];if(!o.truth&&c<0)xcross(ctx,x,y,1.9,rgba(C.muted,.75),1.1);else circ(ctx,x,y,2.4,o.truth?DIGCOL[c]:CC(c),null);}
  if(o.note){ctx.font=font(12,700,1);ctx.textAlign='left';ctx.fillStyle=rgba(C.panel,.85);const tw=ctx.measureText(o.note).width+12;ctx.fillRect(ox,oy,tw,20);ctx.fillStyle=C.text;ctx.fillText(o.note,ox+6,oy+14);}
  if(o.truth){ctx.font=font(13,700);ctx.textAlign='center';for(let d=0;d<10;d++){let sx=0,sy=0,m=0;for(let i=0;i<N;i++)if(lab[i]===d){sx+=MAP[2*i];sy+=MAP[2*i+1];m++;}ctx.fillStyle=C.text;ctx.strokeStyle=C.panel;ctx.lineWidth=3;const X=ox+sx/m*s,Y=oy+(100-sy/m)*s+5;ctx.strokeText(d,X,Y);ctx.fillText(d,X,Y);}}}

/* ── 04 ③ 차원의 저주 ── */
const CU=(function(){const cv=$('cuCv'),H=260,cache={};let cur='dig',ready=false;
  function ratios(k){if(cache[k])return cache[k];let r=[];
    if(k==='dig'){const n=N;for(let i=0;i<n;i++){let mn=1e9,mx=0,off=i*n;for(let j=0;j<n;j++){if(j===i)continue;const v=D[off+j];if(v<mn)mn=v;if(v>mx)mx=v;}r.push(Math.sqrt(mn)/Math.sqrt(mx));}}
    else{const d=+k,n=400,R=RNG(0),U=[];for(let i=0;i<n;i++){const u=new Float64Array(d);for(let j=0;j<d;j++)u[j]=R()*16;U.push(u);}
      for(let i=0;i<n;i++){let mn=1e9,mx=0;for(let j=0;j<n;j++){if(j===i)continue;let s=0;for(let q=0;q<d;q++){const t=U[i][q]-U[j][q];s+=t*t;}if(s<mn)mn=s;if(s>mx)mx=s;}r.push(Math.sqrt(mn/mx));}}
    const sr=r.slice().sort((a,b)=>a-b);return (cache[k]={r,med:sr[sr.length>>1]});}
  function draw(){const W=fit(cv,H),ctx=cv.getContext('2d'),C=cols(),pl=44,pr=12,pt=12,pb=36,gw=W-pl-pr,gh=H-pt-pb,NB=50;ctx.clearRect(0,0,W,H);axes(ctx,C,pl,pt,gw,gh);if(!ready)return;
    const R=ratios(cur),b=new Array(NB).fill(0);R.r.forEach(v=>b[Math.min(NB-1,Math.floor(v*NB))]++);const mx=Math.max(...b);
    b.forEach((c,k)=>{const h=c/mx*gh;ctx.fillStyle=rgba(cur==='dig'?C.blue:C.amber,.8);ctx.fillRect(pl+k*gw/NB+1,pt+gh-h,gw/NB-2,h);});
    [0,0.25,0.5,0.75,1].forEach(v=>{ctx.textAlign='center';ctx.fillStyle=C.muted;ctx.fillText(v,pl+v*gw,H-20);});ctx.fillText('가장 가까운 이웃 거리 ÷ 가장 먼 점 거리 (0이면 이웃이 아주 가깝다, 1이면 모두 같은 거리)',pl+gw/2,H-4);
    vlabel(ctx,'점 수',12,pt+gh/2);ctx.strokeStyle=C.stamp;ctx.setLineDash([5,4]);ctx.beginPath();ctx.moveTo(pl+R.med*gw,pt);ctx.lineTo(pl+R.med*gw,pt+gh);ctx.stroke();ctx.setLineDash([]);
    ctx.fillStyle=C.stamp;ctx.font=font(12,700);ctx.textAlign=R.med>0.6?'right':'left';ctx.fillText('중앙값 '+R.med.toFixed(3),pl+R.med*gw+(R.med>0.6?-6:6),pt+16);}
  function info(){const L=['2','8','16','64','dig'].map(k=>[k,ratios(k).med]);
    $('cuInfo').innerHTML='중앙값 — '+L.map(x=>(x[0]==='dig'?'<b>손글씨 64차원 ':'고른 무작위 '+x[0]+'차원 ')+x[1].toFixed(3)+(x[0]==='dig'?'</b>':'')).join(' · ')+'. 고르게 흩뿌린 점(점 400개, 난수는 A장과 다르다)은 차원이 늘수록 비율이 1 쪽으로 밀린다. 가까운 것과 먼 것의 차이가 사라지니 "반지름 안"과 "밖"을 가르는 eps가 의미를 잃는다. 손글씨는 숫자마다 획이 몰려 있어 8차원 무작위 점 정도로 버티지만, 2차원 지도처럼 뚜렷하지는 않다.';}
  seg('#cuPick button',v=>{cur=v;if(ready){draw();}});draw();onRedraw(draw);
  return {start(){ensureD();ready=true;draw();info();}};})();
whenNear('m4',()=>{ensureD();KD.start();setTimeout(()=>{GR.start();CU.start();},30);},300);

/* ── 05 성과 평가 ── */
function renderPre(){const r=MAINM(),sil=DB.silhouette(D,N,r.lab);r.sil=sil;
  $('preStat').innerHTML=statGrid([[r.k+'개','묶음 수 (미리 정하지 않음)','a'],[r.noise+'장','잡음 ('+pct(r.noise/N)+')'],[pct(r.big/N),'가장 큰 묶음의 몫','s'],[isNaN(sil)?'—':fx(sil,3),'실루엣 (잡음 뺀 '+(N-r.noise)+'장, 정답 없이)']],4);
  $('preFoot').innerHTML='eps <b>'+r.eps.toFixed(2)+'</b>(min_samples 5의 k-거리 무릎), 핵심점 '+r.ncore+'장. 봉투를 열기 전에도 경고등이 켜져 있다. 1,797장 중 '+r.big+'장이 한 묶음이고, 실루엣이 0에 가깝다(−1~1, 묶음 안이 묶음 밖보다 얼마나 가까운가). 정답 없이도 "이 묶기는 쓸모가 적다"고 의심할 수 있다. 그래도 판정은 봉투를 열어 숫자로 한다.';}
function openEnv(){if(ENV)return;ENV=true;const r=MAINM(),s=scoreAll(r.lab),m=manS();RAND_ARI=randARI(r.lab);
  $('envOpen').disabled=true;$('envOpen').textContent='✉ 봉투를 열었다';$('envOut').hidden=false;
  $('envStat').innerHTML=statGrid([[fmtA(s.ari),'ARI (합격선 '+fmtA(m.ari)+')','a'],[fmtA(s.nmi),'NMI (매뉴얼 씨 '+fmtA(m.nmi)+')'],[fmtA(RAND_ARI),'무작위 기준선 ARI'],[pct(s.biz),'업무 점수 (매뉴얼 씨 '+pct(m.biz)+')','s']],4);
  const rows=[];for(let c=0;c<r.k;c++)rows.push(c);rows.sort((a,b)=>r.cnt[b]-r.cnt[a]);const show=rows.slice(0,12);
  const comp=c=>{const b=new Array(10).fill(0);for(let i=0;i<N;i++)if(r.lab[i]===c)b[Y[i]]++;return b;};
  $('envTbl').innerHTML='<thead><tr><th>묶음</th><th>장 수</th>'+[0,1,2,3,4,5,6,7,8,9].map(d=>'<th>'+d+'</th>').join('')+'<th>대표의 숫자</th><th>맞게 퍼짐</th></tr></thead><tbody>'+
    show.map(c=>{const b=comp(c),mx=Math.max(...b),rep=s.pr.rep[c];return '<tr><td><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:'+CC(c)+'"></span> '+(c+1)+'</td><td>'+r.cnt[c]+'</td>'+b.map(v=>'<td'+(v===mx&&v?' style="font-weight:700;color:var(--text)"':' class="muted"')+'>'+(v||'·')+'</td>').join('')+'<td>'+Y[rep]+'</td><td>'+s.pr.okC[c]+'</td></tr>';}).join('')+
    '<tr><td class="muted">잡음</td><td>'+r.noise+'</td>'+(()=>{const b=comp(-1);return b.map(v=>'<td class="muted">'+(v||'·')+'</td>').join('');})()+'<td class="muted">—</td><td class="muted">0</td></tr></tbody>';
  $('envInfo').innerHTML='봉투를 열었다. 기준 모델의 ARI는 <b>'+fmtA(s.ari)+'</b>로 합격선 '+fmtA(m.ari)+'에 한참 못 미치고 무작위('+fmtA(RAND_ARI)+')와 거의 같다. 가장 큰 묶음 '+r.cnt[rows[0]]+'장에는 0부터 9까지 모든 숫자가 고르게 들어 있다. 64차원에서 서로 다른 숫자 무리를 잇는 "다리" 글씨들이 있고, 무릎의 eps는 그 다리를 건너기에 충분히 컸다. 작은 묶음 '+(r.k-1)+'개는 아주 또렷한 2와 7 몇 장뿐이다.';
  renderBiz();renderSens();}
$('envOpen').addEventListener('click',()=>{ensureD();openEnv();});
/* 업무 점수 */
let BIZK='main';
function bizLab(k){return k==='main'?MAINM().lab:k==='ms3'?MS3().lab:k==='man'?MAN:HDB_LAB;}
const BIZNAME={main:'기준 모델 (eps 무릎, min_samples 5)',ms3:'min_samples 3 (eps 무릎)',man:'매뉴얼 씨 8묶음',hdb:'HDBSCAN (Python이 묶음, 채점은 브라우저)'};
function renderBiz(){if(!ENV){$('bizInfo').textContent='먼저 위의 채점 봉투를 연다.';return;}const lab=bizLab(BIZK),s=scoreAll(lab),cv=$('bizCv'),m=manS();
  const ord=[];for(let c=0;c<s.K;c++)ord.push(c);ord.sort((a,b)=>s.pr.cnt[b]-s.pr.cnt[a]);const W=cv.clientWidth||500,cw=W<420?52:62,per=Math.max(1,Math.floor((W-4)/cw)),cells=ord.length+(s.noise?1:0),H=Math.ceil(cells/per)*(cw+30)+6;fit(cv,H);
  const ctx=cv.getContext('2d'),C=cols(),dark=getComputedStyle(document.documentElement).colorScheme==='dark',maxc=Math.max(...s.pr.cnt,s.noise);ctx.clearRect(0,0,W,H);
  ord.concat(s.noise?[-1]:[]).forEach((c,k)=>{const x=2+(k%per)*cw,y=2+Math.floor(k/per)*(cw+30),sz=cw-14;
    ctx.fillStyle=C.panel2;ctx.fillRect(x+2,y,cw-4,cw+26);
    if(c>=0){drawDigit(ctx,P[s.pr.rep[c]],x+7,y+4,sz,dark?'#e8ecf3':'#111827');ctx.fillStyle=C.stamp;ctx.font=font(11,700);ctx.textAlign='left';ctx.fillText(Y[s.pr.rep[c]],x+6,y+14);
      const n=s.pr.cnt[c],ok=s.pr.okC[c],bw=cw-10;ctx.fillStyle=rgba(C.muted,.35);ctx.fillRect(x+5,y+cw-4,bw,6);ctx.fillStyle=C.ok;ctx.fillRect(x+5,y+cw-4,bw*ok/n,6);
      ctx.fillStyle=C.text;ctx.font=font(10.5,500,1);ctx.textAlign='center';ctx.fillText(ok+'/'+n,x+cw/2,y+cw+16);}
    else{xcross(ctx,x+cw/2,y+cw/2-4,10,C.muted,2.4);ctx.fillStyle=C.muted;ctx.font=font(10.5,700);ctx.textAlign='center';ctx.fillText('잡음 '+s.noise,x+cw/2,y+cw+16);}});
  $('bizInfo').innerHTML='<b>'+BIZNAME[BIZK]+'</b>: 이름표 <b>'+s.K+'장</b>만 붙이면 1,797장 중 <b>'+Math.round(s.biz*N)+'장('+pct(s.biz)+')</b>이 맞게 이름 붙는다. 잡음 '+s.noise+'장('+pct(s.noise/N)+')은 이름 없이 남는다. 잡음까지 사람이 한 장씩 붙이면 이름표 '+s.labels+'장으로 '+pct(s.bizAll)+'. '+
    (BIZK==='man'?'매뉴얼 씨는 잡음이 없지만 한 묶음에 여러 숫자가 섞여 대표의 이름이 틀리게 퍼진다. 이 '+pct(m.biz)+'가 업무 점수의 기준이다.':s.biz>m.biz?'매뉴얼 씨('+pct(m.biz)+')보다 높다.':'매뉴얼 씨('+pct(m.biz)+')보다 낮다. 거의 전부가 한 묶음이라 이름표 하나로는 그 묶음의 한 숫자만 맞는다.')+
    ' 각 칸: 대표 글씨(왼쪽 위 빨간 숫자 = 대표의 정답), 초록 막대 = 대표 이름이 맞게 퍼진 몫.';}
seg('#bizPick button',v=>{BIZK=v;if(ENV){ensureD();renderBiz();}else $('bizInfo').textContent='먼저 위의 채점 봉투를 연다.';});
/* 민감도 표 */
let SENSK='ari';const SE=[];for(let e=16;e<=28;e++)SE.push(e);const SM=[3,5,10,20];
function renderSens(){if(!ENV)return;const m=manS(),best={v:-1};
  const val=(lab)=>{const s=scoreAll(lab);return SENSK==='ari'?s.ari:SENSK==='biz'?s.biz:s.noise/N;};
  const col=v=>SENSK==='noise'?'rgba(140,140,150,'+(0.1+0.8*v)+')':v>(SENSK==='ari'?m.ari:m.biz)?'rgba(47,157,99,'+(0.15+0.75*Math.min(1,v/0.8))+')':'rgba(192,57,43,'+(0.12+0.4*(1-Math.max(0,v)/(SENSK==='ari'?m.ari:m.biz)))+')';
  SM.forEach(ms=>SE.forEach(e=>{const s=scoreAll(runDB(e,ms).lab);if(ms===5&&s.ari>best.v){best.v=s.ari;best.e=e;}}));
  const fmt=v=>SENSK==='ari'?v.toFixed(2):pct(v,0);
  $('sensTbl').innerHTML='<table class="tb" style="font-size:12.5px;min-width:720px;text-align:center"><thead><tr><th style="text-align:left">min_samples ＼ eps</th>'+SE.map(e=>'<th>'+e+'</th>').join('')+'<th>무릎 (정답 없이)</th></tr></thead><tbody>'+
    SM.map(ms=>{const K=KN[ms],kr=runDB(K.eps,ms);return '<tr><td style="text-align:left">'+ms+'</td>'+SE.map(e=>{const v=val(runDB(e,ms).lab),pk=ms===5&&e===best.e;return '<td style="background:'+col(v)+';font-family:var(--mono)'+(pk?';outline:2px dashed var(--stamp);outline-offset:-3px':'')+'">'+fmt(v)+'</td>';}).join('')+
      '<td style="font-family:var(--mono);background:'+col(val(kr.lab))+'">'+fmt(val(kr.lab))+' <span class="muted" style="font-size:11px">(eps '+K.eps.toFixed(2)+')</span></td></tr>';}).join('')+'</tbody></table>';
  $('sensInfo').innerHTML=(SENSK==='ari'?'초록 칸 = 합격선 ARI '+fmtA(m.ari)+'를 넘는 칸, 빨간 칸 = 못 넘는 칸.':SENSK==='biz'?'초록 칸 = 업무 점수가 매뉴얼 씨('+pct(m.biz)+')보다 높은 칸.':'진할수록 잡음이 많다(사람이 직접 볼 글씨).')+
    ' 빨간 점선 칸은 min_samples 5에서 정답을 엿봤을 때 가장 좋은 eps('+best.e+', ARI '+fmtA(best.v)+')다. 실제 업무에서는 이 칸을 알 수 없다. 맨 오른쪽 열이 정답 없이 k-거리 무릎으로 고른 자리다. 같은 방법인데 min_samples 3에서는 좋은 칸에, 5·10·20에서는 "다 이어진" 칸에 떨어진다. <b>좋은 칸의 띠가 eps 몇 칸 너비뿐이다.</b> 이 민감함이 이 지원자의 가장 큰 약점이다.';}
seg('#sensPick button',v=>{SENSK=v;renderSens();});
/* 지도 */
let MAPK='main';
function renderMap(){const lab=MAPK==='main'?MAINM().lab:MAPK==='ms3'?MS3().lab:MAPK==='hdb'?HDB_LAB:MAPK==='man'?MAN:Y;let noise=0,K=0;if(MAPK!=='truth')for(let i=0;i<N;i++){if(lab[i]<0)noise++;else K=Math.max(K,lab[i]+1);}
  drawMapOn($('mapCv'),lab,{h:380,truth:MAPK==='truth'});
  $('mapInfo').innerHTML=MAPK==='truth'?'<b>봉투 속 정답</b>(채점용). 숫자마다 지도에서 섬을 이룬다. 1과 8, 9와 3, 4와 7이 군데군데 닿아 있다. 64차원에서도 이런 다리가 있어 반지름을 조금만 키우면 섬들이 이어진다.':
    '<b>'+BIZNAME[MAPK]+'</b>: 묶음 '+K+'개, 잡음 '+noise+'장(회색 ×). '+(MAPK==='main'?'거의 모든 섬이 한 색이다. 무릎의 eps가 섬 사이 다리를 다 건넜다.':MAPK==='ms3'?'섬마다 대체로 다른 색이다. 몇몇 섬(1과 8 근처)은 둘로 쪼개지거나 이웃과 합쳐졌다.':MAPK==='hdb'?'섬마다 색이 다르지만 섬 가장자리의 회색 ×가 많다. 밀도가 낮은 곳을 과감히 잡음으로 둔다.':'매뉴얼 씨의 세 질문은 섬을 가로지른다. 같은 섬(같은 숫자)도 잉크 양에 따라 둘로 갈린다.')+' 지도는 그림용 t-SNE 위치이고 묶음은 64차원에서 계산했다.';}
seg('#mapPick button',v=>{MAPK=v;ensureD();renderMap();});onRedraw(()=>{if(D){renderMap();if(ENV)renderBiz();}});
whenNear('m5',()=>{ensureD();renderPre();renderMap();},500);

/* ── 면접 결과 발표 ── */
Interview('iv12',{title:'DBSCAN은 손글씨 묶기(군집팀)에 합격할까?',
  q:'조건: 기준 모델(min_samples 5 = scikit-learn 기본값, eps = 정답 없이 고른 k-거리 그래프의 무릎)의 묶음이 채점 봉투 속 정답과 겹치는 정도 ARI가 합격선(매뉴얼 씨 0.108)과 무작위 기준선(≈ 0)을 넘고, 업무 점수(묶음마다 대표 한 장에만 이름을 붙였을 때 맞는 비율, 잡음은 이름 없음)도 매뉴얼 씨보다 높으면 합격. 계산은 브라우저가 한다.',
  fly:'eps는 정답 안 보고 고른 거 맞지? 그리고 잡음으로 빼 놓은 글씨는 몇 장이야?',robo:'이름표 몇 장 붙이면 끝나? 잡음은 사람이 하나씩 봐야 하는데, 그 인건비까지 쳐서 말해 봐.',
  async run(setP){setP(.15);await wait(30);ensureD();setP(.5);await wait(30);const r=MAINM();openEnv();setP(.9);await wait(20);const s=scoreAll(r.lab),m=manS();setP(1);
    const pass=s.ari>m.ari&&s.ari>Math.max(0,RAND_ARI)&&s.biz>m.biz;
    return {result:pass?'pass':'hold',flyAns:'eps '+r.eps.toFixed(2)+'는 k-거리 무릎(정답 안 씀) · 잡음 '+r.noise+'장('+pct(r.noise/N)+') · 그런데 '+r.big+'장이 한 묶음',roboAns:'이름표 '+r.k+'장 → 맞게 붙는 글씨 '+pct(s.biz)+' · 잡음까지 사람이 하면 이름표 '+s.labels+'장 · 거리 계산 '+Math.round(COST_MS)+'ms',
      flyMood:pass?'arms':'coffee',flyMove:pass?'nod':'shake',roboMood:pass?'wow':'sigh',roboMove:pass?'jump':'tilt',asstMood:pass?'spark':'sigh',docMood:pass?'judge':'scale',
      measure:'채점 봉투 결과예요. ARI <b>'+fmtA(s.ari)+'</b>(합격선 '+fmtA(m.ari)+', 무작위 '+fmtA(RAND_ARI)+'), 업무 점수 <b>'+pct(s.biz)+'</b>(매뉴얼 씨 '+pct(m.biz)+'). 잡음은 '+pct(r.noise/N)+'로 적었지만, 묶음 '+r.k+'개 중 하나가 '+pct(r.big/N)+'를 삼켰어요. 1,797장에 이름표 '+r.k+'장이면 대부분이 한 숫자 이름을 받습니다.',
      verdict:pass?'합격입니다. 군집팀에 배치합니다.':'보류입니다. 지원자의 규칙대로, 정답 없이 고른 반지름으로는 숫자 무리를 가르지 못했어요. 64차원에서는 k-거리 그래프에 뚜렷한 무릎이 없고, 무릎 자리의 eps는 무리 사이 다리를 건너 버립니다. 차원이 낮고 밀도가 뚜렷한 일(위치, 센서 이상 찾기)이라면 다시 부르겠습니다.',
      key:'정답 없이 고른 eps '+r.eps.toFixed(2)+' — ARI '+fmtA(m.ari)+'(합격선) vs '+fmtA(s.ari)+' · 업무 점수 '+pct(m.biz)+' vs '+pct(s.biz),
      render(box){box.innerHTML='<div class="bars">'+barRows([['DBSCAN ARI',Math.max(0,s.ari),'var(--blue)',fmtA(s.ari)],['매뉴얼 씨 ARI',m.ari,'var(--c-man)',fmtA(m.ari)],['무작위 ARI',Math.max(0,RAND_ARI),'var(--muted)',fmtA(RAND_ARI)],['DBSCAN 업무',s.biz,'var(--blue)',pct(s.biz)],['매뉴얼 씨 업무',m.biz,'var(--c-man)',pct(m.biz)],['DBSCAN 잡음',r.noise/N,'var(--muted)',pct(r.noise/N)]],1)+'</div>';}};},
  alts:[
   {label:'min_samples 3 (eps는 같은 방법, 정답 없이)',async run(box){ensureD();const r=MS3(),s=scoreAll(r.lab),m=manS(),ok=s.ari>m.ari&&s.biz>m.biz;
     box.innerHTML=statGrid([[fmtA(s.ari),'ARI (eps '+r.eps.toFixed(2)+')','a'],[pct(s.biz),'업무 점수','a'],[r.k+'개','묶음 수'],[pct(r.noise/N),'잡음 ('+r.noise+'장)']],4)+'<p class="foot">같은 방법(k-거리 무릎)인데 min_samples만 5 → 3으로 낮췄다. 무릎이 '+KN[5].eps.toFixed(2)+' → '+r.eps.toFixed(2)+'로 내려와 숫자 무리 사이 다리를 건너지 않는다. 판정: <b>'+(ok?/*i18n:verdict*/'합격':'보류')+'</b>(ARI '+fmtA(s.ari)+' > '+fmtA(m.ari)+', 업무 점수 '+pct(s.biz)+' > '+pct(m.biz)+'). 하지만 이 조건을 고른 근거는 무엇인가? 정답 없이는 3과 5 중 어느 쪽이 좋은지 알 수 없다. 근무 조건 하나에 합격과 보류가 갈린다. 그래서 위원회는 기본값으로 판정했다.</p>';}},
   {label:'정답을 엿보고 eps를 고르면 (반칙)',async run(box){ensureD();let b=null;for(let e=16;e<=28.001;e+=0.5){const r=runDB(+e.toFixed(1),5),s=scoreAll(r.lab);if(!b||s.ari>b.s.ari)b={r,s};}const m=manS();
     box.innerHTML=statGrid([[fmtA(b.s.ari),'ARI (eps '+b.r.eps.toFixed(1)+')','a'],[pct(b.s.biz),'업무 점수','a'],[b.r.k+'개','묶음 수'],[pct(b.r.noise/N),'잡음 ('+b.r.noise+'장)']],4)+'<p class="foot">eps 16~28을 0.5 간격으로 돌려 <b>봉투 속 정답</b>으로 가장 좋은 것을 골랐다. ARI '+fmtA(b.s.ari)+', 업무 점수 '+pct(b.s.biz)+'로 숫자만 보면 넉넉히 합격이다. 판정: <b>무효</b>. 실제 업무에는 채점 봉투가 없다. 정답 1,797장이 있었다면 처음부터 지도 학습을 했을 것이다. 03장 틀린 곳 찾기 ②번과 같은 실수다. 이 점수는 "이 지원자가 낼 수 있는 최고치"로만 기록한다.</p>';}},
   {label:'HDBSCAN (eps 없이 · Python이 묶음)',async run(box){const s=scoreAll(HDB_LAB),m=manS(),ok=s.ari>m.ari&&s.biz>m.biz;
     box.innerHTML=statGrid([[fmtA(s.ari),'ARI (Python '+fmtA(HDB.ari)+')','a'],[pct(s.biz),'업무 점수','a'],[s.K+'개','묶음 수'],[pct(s.noise/N),'잡음 ('+s.noise+'장)']],4)+'<p class="foot">HDBSCAN(min_cluster_size=5)은 반지름 하나를 고르지 않는다. 밀도 문턱을 여러 높이에서 보며 오래 버티는 묶음을 고른다. 정답 없이 정해지는 근무 조건은 "묶음의 최소 크기" 하나다. 묶기는 Python이 미리 했고(A장 10번 칸) 채점은 이 브라우저가 했다. 판정: <b>'+(ok?/*i18n:verdict*/'합격':'보류')+'</b>. 다만 잡음이 '+s.noise+'장('+pct(s.noise/N)+')이라, 잡음까지 사람이 붙이면 이름표가 '+s.labels+'장 든다. 같은 밀도 집안의 다른 지원자로 06장에 적는다.</p>';}}]});

/* ── 06 인사 카드 ── */
comic('comicCard',[
 {bg:'night',prop:'eps 21 · 봉투 열고 고름',cast:[{who:'asst',mood:'spark',x:30,move:'jump'},{who:'fly',mood:'point',x:74,move:'shake',flip:true}],say:{who:'fly',txt:'ARI 0.61? 그 반지름, 봉투 열고 고른 거잖아. 무효야.'},cap:'① 엿본 반지름 — 숫자는 좋지만 쓸 수 없다'},
 {bg:'night',prop:'eps 25.16 · 무릎',cast:[{who:'asst',mood:'shock',x:30,move:'shake'},{who:'robo',mood:'cry',x:72,move:'shake',h:104}],say:{who:'asst',txt:'정답 없이 고른 반지름으로는… 1,728장이 한 묶음이에요.'},cap:'② 정답 없이 고른 반지름 — 무리들이 다 이어졌다'},
 {bg:'night',prop:'HDBSCAN',cast:[{who:'doc',mood:'magic',x:30,move:'pop',h:104},{who:'fly',mood:'arms',x:74,move:'nod',flip:true}],say:{who:'doc',txt:'반지름을 하나로 정하지 말고, 여러 높이에서 오래 버티는 동네를 고르면 어떨까요.'},cap:'③ 사촌의 해법 — 밀도 문턱을 여러 높이에서'}]);
function renderTeam(){ensureD();const m=manS(),a=scoreAll(MAINM().lab),b=scoreAll(MS3().lab),h=scoreAll(HDB_LAB);let pk=null;for(let e=16;e<=28.001;e+=0.5){const s=scoreAll(runDB(+e.toFixed(1),5).lab);if(!pk||s.ari>pk.s.ari)pk={e,s};}
  const R=[['매뉴얼 씨 (합격선)','세 질문 규칙, 학습 없음',m,'기준','브라우저'],['<b>DBSCAN · 기준 모델</b>','eps '+KN[5].eps.toFixed(2)+'(무릎), min_samples 5',a,a.ari>m.ari&&a.biz>m.biz?'pass':'hold','브라우저'],['DBSCAN','eps '+KN[3].eps.toFixed(2)+'(무릎), min_samples 3',b,b.ari>m.ari&&b.biz>m.biz?'pass':'hold','브라우저'],['DBSCAN [반칙]','eps '+pk.e.toFixed(1)+'(정답으로 고름), min_samples 5',pk.s,'void','브라우저'],
    ['HDBSCAN','min_cluster_size 5',h,h.ari>m.ari&&h.biz>m.biz?'pass':'hold','Python 묶음 · 브라우저 채점'],['K-Means (5-10)','k = 10, 원래 픽셀',{ari:PRE.kmeans.ari,nmi:PRE.kmeans.nmi,biz:PRE.kmeans.prop,K:10,noise:0,labels:10},'ref','Python'],['계층적 ward (5-11)','10묶음',{ari:PRE.ward.ari,nmi:PRE.ward.nmi,biz:PRE.ward.prop,K:10,noise:0,labels:10},'ref','Python']];
  const tag=v=>v==='pass'?'<span class="pill ok">합격 조건</span>':v==='hold'?'<span class="pill no">보류 조건</span>':v==='void'?'<span class="pill no">무효</span>':v==='ref'?'<span class="pill">그 편에서</span>':'<span class="pill">합격선</span>';
  $('teamTbl').innerHTML='<thead><tr><th>지원자</th><th>설정</th><th>묶음</th><th>잡음</th><th>ARI</th><th>NMI</th><th>업무 점수</th><th>이름표 (잡음 포함)</th><th>비고</th></tr></thead><tbody>'+
    R.map(r=>'<tr'+(r[0].indexOf('기준 모델')>=0?' style="background:var(--panel2)"':'')+'><td>'+r[0]+'</td><td style="font-size:13px">'+r[1]+'</td><td>'+r[2].K+'</td><td>'+(r[2].noise?r[2].noise+' ('+pct(r[2].noise/N,0)+')':'0')+'</td><td>'+fmtA(r[2].ari)+'</td><td>'+fmtA(r[2].nmi)+'</td><td>'+pct(r[2].biz)+'</td><td>'+r[2].labels+'</td><td style="font-size:12.5px">'+tag(r[3])+' <span class="muted">'+r[4]+'</span></td></tr>').join('')+'</tbody>';
  $('teamFoot').innerHTML='업무 점수 = 묶음마다 평균에 가장 가까운 한 장에만 이름을 붙이고 묶음 전체에 퍼뜨렸을 때 1,797장 중 맞게 이름 붙은 비율(잡음은 이름 없음). 이름표 = 묶음 수 + 잡음 수(잡음은 사람이 한 장씩). K-Means와 계층적 군집 줄은 같은 규칙으로 Python(scikit-learn 1.9.1)이 계산했다(A장 11번 칸). 군집팀 지원자의 판정은 각 편에서 한다. 밀도 집안(DBSCAN·HDBSCAN)은 잡음을 내놓는 대신 근무 조건에 민감하고, 64차원 픽셀에서는 중심이나 합치기로 묶는 지원자보다 낮았다.';}
whenNear('m6',renderTeam,500);

/* ══ 확인 문제 ══ */
var QUIZ={
q1:[{q:'DBSCAN에서 핵심점의 조건은?',o:[['반지름 eps 안에 자기 포함 min_samples개 이상의 점이 있다',1,'scikit-learn은 자기 자신도 센다. 01장 그림에서 점을 눌러 이웃 수를 확인해 보자.'],['묶음의 중심에 가장 가깝다',0,'그것은 K-Means의 중심 개념이다. DBSCAN에는 중심이 없다.'],['다른 점들과 가장 멀다',0,'그런 점은 오히려 잡음이 되기 쉽다.']]},
 {q:'가장자리 점이란?',o:[['스스로는 이웃이 모자라지만 어떤 핵심점의 반지름 안에 있는 점',1,'그 핵심점의 묶음에 붙는다. 다만 다른 점으로 더 번지지는 않는다.'],['지도 맨 끝에 있는 점',0,'위치가 아니라 이웃 수와 핵심점과의 거리로 정한다.'],['잡음과 같은 말',0,'잡음은 어느 핵심점의 반지름에도 들지 않는 점이다.']]},
 {q:'맞물린 초승달 두 개를 K-Means(k=2)와 DBSCAN으로 묶으면 대개?',o:[['DBSCAN은 모양대로 둘로 나누고, K-Means는 가운데를 잘라 섞는다',1,'K-Means는 가장 가까운 중심으로 나누므로 둥근 묶음을 가정하는 셈이다. 01장의 "K-Means로 묶으면?" 버튼.'],['둘 다 같은 결과다',0,'모양이 휘면 크게 달라진다.'],['K-Means만 잡음을 찾는다',0,'잡음(−1)을 내놓는 것은 DBSCAN이다.']]}],
q2:[{q:'이 면접에서 정답(0~9)을 쓰는 때는?',o:[['마지막 채점(ARI·업무 점수)에서만',1,'묶기와 근무 조건 고르기는 정답 없이 한다. 그래서 시험지를 따로 떼지 않고 1,797장 전체를 묶는다.'],['eps를 고를 때',0,'그러면 비지도가 아니다. 05장의 "엿본 eps"가 그 예다.'],['특징으로 함께 넣는다',0,'정답을 특징에 넣으면 묶음이 정답을 그대로 따라간다. 반칙이다.']]},
 {q:'손글씨 픽셀을 DBSCAN 앞에서 표준화하지 않는 까닭은?',o:[['64칸 모두 같은 단위(밝기 0~16)이고, 표준화하면 거의 빈 칸이 부풀어 거리를 흔들기 때문',1,'Python 확인: 표준화하면 정답을 엿보고 고른 최고 ARI도 0.614 → 0.248로 떨어진다.'],['DBSCAN은 거리를 쓰지 않아서',0,'DBSCAN은 거리가 전부다.'],['표준화하면 오류가 나서',0,'오류 없이 돈다. 그래서 더 알아채기 어렵다.']]}],
q3:[{q:'DBSCAN()을 기본값 그대로 손글씨 픽셀에 쓰면?',o:[['eps=0.5라 반지름 안에 자기 자신뿐 → 전부 잡음',1,'가장 가까운 이웃도 10 넘게 떨어져 있다(04장 k-거리). eps는 데이터의 거리 단위에 맞춰 고른다.'],['10묶음이 나온다',0,'DBSCAN은 묶음 수를 정하지 않는다. 기본값으로는 0묶음이다.'],['오류가 난다',0,'오류 없이 −1만 1,797개 돌려준다.']]},
 {q:'len(set(labels))로 묶음 수를 세면 생기는 문제는?',o:[['잡음 −1까지 묶음 하나로 센다',1,'labels.max() + 1로 세고, 잡음 수는 (labels == -1).sum()으로 따로 보고한다.'],['묶음을 하나 적게 센다',0,'오히려 하나 많게 센다.'],['아무 문제 없다',0,'잡음이 하나라도 있으면 하나 부풀려진다.']]},
 {q:'잡음을 빼고 ARI를 재면?',o:[['어려운 글씨를 잡음으로 버릴수록 점수가 부풀려진다',1,'eps 16이면 77%를 버리고 남은 글씨의 ARI가 0.84, 전체로는 0.04다. 잡음까지 넣어 재고 잡음 수를 함께 적는다.'],['더 공정한 점수다',0,'버린 글씨는 결국 사람이 처리한다. 그 비용이 점수에서 사라진다.'],['ARI가 낮아진다',0,'대개 높아진다.']]}],
q4:[{q:'k-거리 그래프에서 무릎에 eps를 두면 그 오른쪽 점들은?',o:[['핵심점이 못 된다(잡음이나 가장자리 후보)',1,'무릎은 "잡음으로 볼 몫"을 정하는 일이다. 손글씨 min_samples 5에서는 위쪽 10%쯤이다.'],['모두 한 묶음이 된다',0,'핵심점이 못 될 뿐, 묶음 여부는 다른 핵심점과의 거리로 정해진다.'],['지워진다',0,'데이터는 그대로다. 묶음 번호만 정해진다.']]},
 {q:'64차원에서 eps 고르기가 어려운 까닭은?',o:[['가장 가까운 이웃과 가장 먼 점의 거리가 비슷해져 "반지름 안과 밖"의 차이가 흐려지기 때문',1,'차원의 저주. 04장 ③: 고른 무작위 점은 64차원에서 비율 중앙값이 0.6을 넘는다.'],['계산이 너무 오래 걸려서',0,'1,797장이면 1초 안팎이다. 속도가 아니라 거리의 성질 문제다.'],['픽셀이 정수라서',0,'정수인 것은 상관없다.']]},
 {q:'eps를 조금씩 키우면 대개?',o:[['잡음이 줄고 묶음이 늘다가, 어느 순간 무리끼리 이어져 한 덩어리가 된다',1,'04장 ② 수습 재생. 손글씨에서는 eps 21쯤에서 묶음이 가장 많고, 25를 넘으면 거의 한 묶음이다.'],['묶음 수가 계속 늘어난다',0,'반지름이 커지면 결국 모두 이어진다.'],['결과가 바뀌지 않는다',0,'이 지원자는 eps에 매우 민감하다.']]}],
q5:[{q:'ARI가 0에 가깝다는 뜻은?',o:[['묶음이 정답과 무작위로 묶은 것만큼만 겹친다',1,'ARI는 우연히 겹칠 몫을 빼고 잰다. 기준 모델 ARI ≈ 0은 무작위 기준선과 비슷하다는 뜻이다.'],['묶음이 정답과 완전히 같다',0,'완전히 같으면 1이다.'],['잡음이 없다',0,'잡음 수와는 다른 이야기다.']]},
 {q:'업무 점수(대표 한 장씩 이름 붙이기)가 매뉴얼 씨보다 낮았던 까닭은?',o:[['1,728장이 한 묶음이라 이름표 하나로는 그 안의 한 숫자만 맞기 때문',1,'그 묶음에는 0~9가 고르게 들어 있다. 대표의 이름이 맞는 것은 약 10분의 1이다.'],['잡음이 너무 많아서',0,'기준 모델의 잡음은 2.8%뿐이다. 문제는 반대로 너무 많이 이어진 것이다.'],['대표를 잘못 골라서',0,'대표를 누구로 골라도 한 묶음에 한 이름뿐이다.']]},
 {q:'min_samples 3으로 바꾸면 합격선을 넘는다. 그렇다면 3으로 판정해야 하나?',o:[['아니다. 정답 없이 3을 고를 근거가 없으면, 결과를 보고 고른 셈이다',1,'위원회는 기본값으로 판정하고 다른 조건은 "다시 보기"로 기록했다. 근무 조건에 따라 합격·보류가 갈린다는 것 자체가 이 지원자의 성적이다.'],['그렇다. 더 좋은 쪽으로 판정한다',0,'그 "더 좋은 쪽"을 정답으로 알았다면 반칙이다.'],['min_samples는 결과에 영향이 없다',0,'05장 민감도 표에서 행마다 결과가 크게 다르다.']]}],
q6:[{q:'DBSCAN에 새 글씨 한 장이 들어오면?',o:[['predict가 없어 다시 돌리거나, 가장 가까운 핵심점의 묶음을 따로 붙인다',1,'K-Means·GMM은 새 점을 바로 배정한다. DBSCAN은 지금 가진 점들의 밀도로만 묶는다.'],['predict로 바로 배정한다',0,'scikit-learn DBSCAN에는 predict가 없다.'],['자동으로 잡음이 된다',0,'따로 규칙을 만들어야 한다.']]},
 {q:'HDBSCAN이 DBSCAN보다 나은 점은?',o:[['반지름 하나를 정하지 않고, 밀도가 다른 무리를 함께 다룬다',1,'여러 밀도 문턱에서 오래 버티는 묶음을 고른다. 근무 조건은 묶음 최소 크기(min_cluster_size)가 중심이다.'],['잡음을 만들지 않는다',0,'이 면접에서는 오히려 잡음이 500장이다.'],['묶음 수를 정해 줘야 한다',0,'정하지 않는다.']]}]};

/* ══ 공통 B·C·D·E·F 데이터 ══ */
var PROMPTS=[
 {t:'① 내 데이터로 DBSCAN 코드 받기',use:'내 공장의 위치·센서·이미지 데이터로 03장의 프롬프트를 맞출 때',p:MAIN_PROMPT},
 {t:'② AI 코드 검증 요청',use:'받은 군집 코드를 다른 AI(또는 같은 AI의 새 대화)에게 검토받을 때',p:'아래 scikit-learn DBSCAN 코드를 코드 리뷰해 줘. 특히 다음을 하나씩 확인하고, 문제가 있으면 줄 번호와 고친 코드를 보여 줘.\n1) 정답(라벨)이 fit에 들어가거나, eps·min_samples를 정답 점수로 고르지 않았나\n2) eps가 데이터의 거리 단위에 맞나 (기본값 0.5를 그대로 쓰지 않았나). k-거리 그래프를 그렸나\n3) 특징의 단위가 같은데 표준화했거나, 단위가 다른데 표준화하지 않았나\n4) min_samples가 1이 아닌가\n5) 묶음 수를 len(set(labels))로 세어 잡음(-1)까지 넣지 않았나\n6) 잡음을 빼고 ARI·실루엣을 재면서 잡음 수를 보고하지 않았나\n추측으로 답하지 말고, 확실하지 않은 부분은 "공식 문서 확인 필요"라고 표시해 줘.'},
 {t:'③ 밀도 집안과 차원 줄이기 비교',use:'Colab에서 DBSCAN·HDBSCAN·OPTICS와 PCA 전처리를 같은 데이터로 비교할 때',p:'아래 Colab 코드(load_digits, DBSCAN)를 출발점으로, 같은 1,797장에서 다음을 비교하는 셀을 덧붙여 줘.\n- DBSCAN: eps는 k-거리 무릎(정답 없이), min_samples 3·5·10\n- HDBSCAN(min_cluster_size 5·10·20), OPTICS(min_samples 5)\n- 같은 셋을 PCA 〔10·20·30〕차원으로 줄인 뒤에도 (eps는 줄인 공간에서 다시 k-거리로 고른다)\n규칙:\n1) 정답 y는 마지막 채점에만 쓴다. 근무 조건은 정답 없이 고른다\n2) 표: 묶음 수, 잡음 수와 비율, 실루엣(잡음 제외), ARI, NMI, 업무 점수(묶음마다 평균에 가장 가까운 한 장의 정답을 퍼뜨린 정확도, 잡음은 틀림)\n3) 합격선 ARI 0.108(매뉴얼 씨)과 비교한 문장으로 해석한다\n4) t-SNE·UMAP 위에서 돌린 결과를 넣는다면 "지도 + DBSCAN"이라고 따로 표시한다'},
 {t:'④ 수업용 문제 만들기',use:'이 편으로 수업할 때 확인 문제가 더 필요할 때',p:'밀도 기반 군집(DBSCAN) 입문 수업용 문제를 만들어 줘. 대상은 〔제조 현장 실무자, 수식에 익숙하지 않음〕이다. 예시는 공장 바닥의 고장 신고 위치(2차원 좌표)로 통일한다.\n- 핵심점·가장자리 점·잡음, eps, min_samples를 각각 묻는 객관식 4문제 (보기 3개, 정답 해설 포함)\n- 오해를 드러내는 문제 2개: "eps 기본값이면 된다", "잡음을 빼고 채점하면 공정하다"\n- 손으로 풀 수 있는 문제 1개: 좌표 10개와 eps 2, min_samples 3이 주어졌을 때 핵심점·가장자리·잡음을 가리고 묶음을 만들기\n정답은 맨 끝에 따로 모아 줘.'}];
var GL=[
 ['모델','DBSCAN','Density-Based Spatial Clustering of Applications with Noise. 반지름 eps 안의 이웃 수로 밀도를 재서, 촘촘한 곳을 이어 묶음을 만들고 어디에도 닿지 않는 점은 잡음으로 남기는 군집 방법(Ester 등, 1996).'],
 ['모델','HDBSCAN','DBSCAN을 여러 밀도 문턱에서 한꺼번에 보고, 오래 버티는 묶음을 고르는 방법. eps를 정하지 않는다. scikit-learn 1.3부터 들어 있다.'],
 ['모델','OPTICS','점들을 밀도 순서로 늘어놓아 여러 eps의 결과를 한 번에 보게 하는 DBSCAN의 사촌.'],
 ['모델','K-Means','묶음 수 k를 정하고, 점마다 가장 가까운 중심에 넣은 뒤 중심을 평균으로 옮기기를 되풀이하는 군집 방법(5-10).'],
 ['모델','군집','정답 없이 비슷한 것끼리 묶는 일. 비지도 학습의 한 갈래.'],
 ['모델','비지도 학습','정답(이름표) 없이 데이터의 구조를 찾는 학습. 군집과 차원 축소가 대표적이다.'],
 ['학습','eps','DBSCAN의 이웃 반지름. 이 거리 안의 점을 이웃으로 센다. 데이터의 거리 단위에 따라 알맞은 값이 전혀 다르다.'],
 ['학습','min_samples','핵심점이 되려면 반지름 eps 안에 있어야 할 점의 수. scikit-learn은 자기 자신을 포함해 센다. 기본값 5.'],
 ['학습','핵심점','반지름 eps 안에 min_samples개 이상의 점이 있는 점. 촘촘한 곳의 한가운데. 핵심점끼리 이어져 묶음이 된다.'],
 ['학습','가장자리 점','스스로는 이웃이 모자라지만 어떤 핵심점의 반지름 안에 든 점. 그 핵심점의 묶음에 붙는다.'],
 ['학습','잡음','어느 핵심점의 반지름에도 들지 않는 점. 묶음 번호 −1. 이 시리즈의 업무에서는 사람이 직접 볼 글씨다.'],
 ['학습','밀도 도달','핵심점에서 핵심점으로 반지름 안을 건너 닿을 수 있는 관계. 이렇게 이어진 점들이 한 묶음이다.'],
 ['학습','k-거리 그래프','점마다 k번째로 가까운 이웃까지의 거리를 재서 작은 것부터 늘어놓은 곡선. eps를 정답 없이 고르는 데 쓴다.'],
 ['학습','무릎','곡선이 확 꺾이는 곳. 이 페이지에서는 양 끝을 잇는 직선에서 가장 멀리 떨어진 점(Kneedle 방식과 같다).'],
 ['학습','차원의 저주','차원이 높아지면 점 사이 거리가 서로 비슷해지고 공간이 텅 비어, 거리·밀도로 하는 판단이 흐려지는 현상.'],
 ['학습','유클리드 거리','두 점의 칸별 차이를 제곱해 더하고 제곱근을 씌운 거리. 손글씨에서는 64칸 밝기의 차이로 잰다.'],
 ['학습','사슬 효과','이웃의 이웃으로 계속 이어 붙이다 보면 멀리 떨어진 두 무리가 가는 다리로 이어져 한 묶음이 되는 일. 계층적 군집 single 연결과 같은 문제.'],
 ['평가','채점 봉투','숨겨 둔 정답(0~9). 묶기와 근무 조건 고르기에는 쓰지 않고, 마지막 채점에서 한 번만 연다.'],
 ['평가','ARI','조정 랜드 지수. 두 묶기(지원자의 묶음과 정답)가 "같은 묶음에 넣은 쌍"을 얼마나 함께하나를, 우연히 겹칠 몫을 빼고 잰다. 무작위 ≈ 0, 완전 일치 1.'],
 ['평가','NMI','정규화 상호 정보량. 묶음을 알면 정답을 얼마나 알 수 있나를 0~1로 잰다.'],
 ['평가','실루엣 점수','점마다 "자기 묶음 안 평균 거리"와 "가장 가까운 다른 묶음까지 평균 거리"를 비교한 값(−1~1). 정답 없이 잰다.'],
 ['평가','업무 점수','묶음마다 대표 한 장(묶음 평균에 가장 가까운 글씨)에만 사람이 이름을 붙이고 묶음 전체에 퍼뜨렸을 때, 전체 중 맞게 이름 붙은 비율. 잡음은 이름 없음으로 센다.'],
 ['평가','합격선','지원자가 넘어야 할 점수. 군집팀은 매뉴얼 씨 규칙 8묶음의 ARI 0.108.'],
 ['평가','무작위 기준선','같은 크기의 묶음에 글씨를 무작위로 나눠 넣었을 때의 점수. ARI는 0 근처다.'],
 ['데이터','픽셀','그림을 이루는 작은 칸. 손글씨 한 장은 8×8 = 64칸, 칸마다 밝기 0~16.'],
 ['데이터','t-SNE','가까운 이웃 관계를 지키며 고차원 데이터를 2차원 지도로 펼치는 방법(5-15). 이 페이지에서는 그림의 위치로만 쓴다.']];
var ANALOGY=[
 ['가까운 것끼리 묶기','운동장에 흩어진 학생들을 위에서 보면 모여 있는 무리가 저절로 보인다',2,'사람의 시각은 가까이 있는 점들을 한 무리로 본다(게슈탈트 근접성 원리, Wertheimer, 1923)','사람은 근접성 말고도 모양·색·움직임을 함께 보고, 정해진 반지름 문턱을 쓰지 않는다. 64차원 거리는 사람이 직접 볼 수 없다'],
 ['이웃 찾기 (초파리)','비슷한 냄새를 맡으면 비슷한 기억을 떠올리는 것',2,'초파리 후각 회로는 냄새 신호를 케니언 세포 수천 개로 무작위로 펼친 뒤 소수만 켜서, 비슷한 냄새가 비슷한 패턴이 되게 한다(Dasgupta 등, 2017)','초파리 회로는 "비슷한 것을 빨리 찾는" 쪽에 가깝다. 반지름 안의 이웃 수를 세어 묶음을 만들거나 잡음을 따로 내놓는다는 근거는 없다'],
 ['잡음으로 남기기','어느 팀에도 맞지 않는 서류를 억지로 끼우지 않고 "보류함"에 넣는 담당자',1,'','새롭거나 낯선 것에 뇌가 다르게 반응한다는 연구는 많지만, 밀도 문턱으로 "외톨이"를 가르는 회로는 알려지지 않았다'],
 ['퍼뜨리기','소문이 옆 사람에서 옆 사람으로 번지다 빈 자리에서 멈추는 것',1,'','뉴런 사이 활동 전파와 겉모양은 비슷하지만, DBSCAN의 퍼뜨리기는 정해진 거리 규칙의 계산일 뿐 대응하는 뇌 기능이 알려지지 않았다'],
 ['차원의 저주','항목 64개짜리 체크리스트로 사람을 비교하면 누구나 서로 "조금씩 다 다르다"로 보인다',1,'','뇌가 고차원 신호를 낮은 차원의 구조로 다룬다는 이론은 있으나, 이 현상에 직접 대응하는 뇌 기능을 말하기는 어렵다'],
 ['k-거리 무릎','동네마다 "다섯 번째로 가까운 이웃집"까지의 거리를 재고, 그 거리가 갑자기 멀어지는 곳을 마을 경계로 삼는다',1,'','직접 대응하는 뇌 기능은 알려지지 않았다']];
var REFS=[
 'Ester, M., Kriegel, H.-P., Sander, J., &amp; Xu, X. (1996). A density-based algorithm for discovering clusters in large spatial databases with noise. In <i>Proceedings of the Second International Conference on Knowledge Discovery and Data Mining (KDD-96)</i> (pp. 226–231). AAAI Press. — DBSCAN 원논문, 01·04장',
 'Schubert, E., Sander, J., Ester, M., Kriegel, H.-P., &amp; Xu, X. (2017). DBSCAN revisited, revisited: Why and how you should (still) use DBSCAN. <i>ACM Transactions on Database Systems</i>, 42(3), Article 19. <a href="https://doi.org/10.1145/3068335" target="_blank" rel="noopener">doi:10.1145/3068335</a> — 근무 조건 고르기와 고차원의 한계, 04·06장',
 'Campello, R. J. G. B., Moulavi, D., &amp; Sander, J. (2013). Density-based clustering based on hierarchical density estimates. In <i>Advances in Knowledge Discovery and Data Mining (PAKDD 2013)</i>, LNCS 7819 (pp. 160–172). Springer. — HDBSCAN, 05·06장',
 'Ankerst, M., Breunig, M. M., Kriegel, H.-P., &amp; Sander, J. (1999). OPTICS: Ordering points to identify the clustering structure. <i>ACM SIGMOD Record</i>, 28(2), 49–60. — 03장 버전 메모',
 'Satopää, V., Albrecht, J., Irwin, D., &amp; Raghavan, B. (2011). Finding a "Kneedle" in a haystack: Detecting knee points in system behavior. In <i>31st International Conference on Distributed Computing Systems Workshops</i> (pp. 166–171). IEEE. — 04장 무릎 찾기',
 'Beyer, K., Goldstein, J., Ramakrishnan, R., &amp; Shaft, U. (1999). When is "nearest neighbor" meaningful? In <i>Database Theory — ICDT\'99</i>, LNCS 1540 (pp. 217–235). Springer. — 04장 차원의 저주',
 'Hubert, L., &amp; Arabie, P. (1985). Comparing partitions. <i>Journal of Classification</i>, 2(1), 193–218. — ARI, 05장',
 'Vinh, N. X., Epps, J., &amp; Bailey, J. (2010). Information theoretic measures for clusterings comparison. <i>Journal of Machine Learning Research</i>, 11, 2837–2854. — NMI, 05장',
 'Rousseeuw, P. J. (1987). Silhouettes: A graphical aid to the interpretation and validation of cluster analysis. <i>Journal of Computational and Applied Mathematics</i>, 20, 53–65. — 실루엣, 05장',
 'Alpaydin, E., &amp; Kaynak, C. (1998). Optical Recognition of Handwritten Digits [Dataset]. UCI Machine Learning Repository. <a href="https://doi.org/10.24432/C50P49" target="_blank" rel="noopener">doi:10.24432/C50P49</a> (CC BY 4.0) — 02장 실기 과제지(scikit-learn load_digits)',
 'van der Maaten, L., &amp; Hinton, G. (2008). Visualizing data using t-SNE. <i>Journal of Machine Learning Research</i>, 9, 2579–2605. — 04·05장 그림용 2D 지도',
 'Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. <i>Journal of Machine Learning Research</i>, 12, 2825–2830. · scikit-learn 문서 <a href="https://scikit-learn.org/stable/modules/clustering.html#dbscan" target="_blank" rel="noopener">Clustering: DBSCAN · HDBSCAN · OPTICS</a> — A장 코드, 03·06장',
 'Wertheimer, M. (1923). Untersuchungen zur Lehre von der Gestalt II. <i>Psychologische Forschung</i>, 4, 301–350. — D장 근접성 원리',
 'Dasgupta, S., Stevens, C. F., &amp; Navlakha, S. (2017). A neural algorithm for a fundamental computing problem. <i>Science</i>, 358(6364), 793–796. — D장 초파리 후각 회로'];
var TIP={who:'머신러닝을 처음 배우는 대학생·직장인·제조 현장 실무자. 5-1 KNN(거리와 이웃)을 먼저 보면 좋고, 5-10 K-Means와 나란히 보면 차이가 잘 보인다. 수식은 "반지름 안의 이웃 수 ≥ min_samples" 하나다',
 goal:'① 핵심점·가장자리 점·잡음을 01장 그림으로 가려낸다 ② eps를 정답 없이 k-거리 그래프로 고르고, 그 방법의 한계(뚜렷하지 않은 무릎)를 설명한다 ③ 차원의 저주가 밀도 기반 군집을 어떻게 흔드는지 말한다 ④ AI가 낸 군집 코드에서 정답으로 근무 조건 고르기, 잡음 빼고 채점하기를 찾고, 합격선·업무 점수로 판정한다',
 time:'읽기 25분 · 실습(01·04·05장 그림) 25분 · 틀린 곳 찾기 10분 · Colab 15분',
 plans:[['강의형 50분','00장 지원서 1장을 인쇄해 1~5번 칸 읽기 8분 → 01장 그림에서 eps·min_samples를 바꾸며 핵심점·가장자리·잡음 맞히기, "K-Means로 묶으면?" 비교 12분 → 04장 k-거리 그래프와 수습 재생으로 "반지름 몇 칸 차이" 토론 12분 → 05장 채점 봉투 열기, 민감도 표 8분 → 면접 결과를 먼저 맞히고 발표, 세 가지 다시 보기 10분.'],['실무 OJT 90분','03장 프롬프트로 각자 AI에게 코드를 받는다 20분 → 체크리스트로 서로의 코드를 바꿔 검사(특히 정답 사용 위치) 20분 → Colab에서 실행하고 A장 결과와 비교 20분 → 내 현장의 2차원 데이터(고장 신고 위치, 작업자 동선)를 가정해 DBSCAN이 맞는 일인지, 잡음 한 장의 처리 비용은 얼마인지 토론 30분.'],['1:1 튜터링','01장에서 점 하나를 누를 때마다 "이 점은 핵심점인가, 가장자리인가, 잡음인가"를 먼저 말하게 하고 확인한다. 이어 04장 k-거리 그래프에서 eps를 직접 옮기며 "정답 없이 무엇을 근거로 고를 수 있나"를 말로 정리하게 한다.']],
 myths:[['eps 기본값(0.5)이면 된다','eps는 데이터의 거리 단위에 따라 다르다. 손글씨 픽셀에서 0.5는 전부 잡음이다'],['DBSCAN은 묶음 수를 정하지 않으니 근무 조건이 없다','k 대신 eps와 min_samples를 정한다. 이 면접에서는 그 두 개에 합격·보류가 갈렸다'],['잡음은 버리는 데이터다','업무에서 잡음은 사람이 직접 볼 대상이거나 이상 후보다. 잡음 수는 비용으로 보고한다'],['잡음을 빼고 채점하면 공정하다','어려운 것을 버릴수록 점수가 오른다. 잡음까지 넣어 채점하고 잡음 수를 함께 적는다'],['k-거리 그래프의 무릎은 늘 정답이다','무릎은 "잡음으로 볼 몫"을 정할 뿐이다. 차원이 높으면 무릎이 흐려진다'],['t-SNE 지도 위에서 돌리면 DBSCAN이 잘한다','점수는 오르지만 지도가 이미 이웃을 모아 둔 덕이다. "지도 + DBSCAN"의 성적으로 따로 적는다']],
 tasks:[['과제 1 · 손으로 묶기','좌표 12개와 eps 1.5, min_samples 3이 주어진 모눈종이에서 원을 그려 핵심점·가장자리·잡음을 가리고 묶음을 만든다. 01장 그림으로 답을 확인한다(같은 규칙).'],['과제 2 · 틀린 곳 찾기 만들기','AI에게 일부러 실수가 섞인 군집 코드를 받고(B장 ②를 거꾸로 활용), 짝에게 찾게 한다. "정답을 어디에 썼나"를 하나 이상 넣는다.'],['과제 3 · 무릎 찾기 비교','A장 4번 칸을 고쳐 min_samples 3·5·10에서 무릎 eps를 구하고, PCA 10·20차원으로 줄인 뒤에도 같은 일을 한다. 무릎이 뚜렷해지는지 그래프로 비교한다. 정답은 마지막 채점에만 쓴다.'],['과제 4 · 업무 점수 표','B장 ③ 프롬프트로 DBSCAN·HDBSCAN·OPTICS의 묶음 수, 잡음, ARI, 업무 점수를 표로 만들고, "잡음 한 장 = 이름표 한 장"으로 계산한 총 이름표 수를 덧붙인다. 06장 비교표와 견준다.']],
 ai:'학생이 AI에게 코드를 받는 것은 막지 말고 과정에 넣는다. 비지도 학습은 실수해도 오류가 나지 않아 검증 연습에 특히 좋다. ① 프롬프트를 제출하게 하고 ② 체크리스트 아홉 항목을 직접 표시하게 하고(특히 정답을 쓴 줄을 모두 찾게 한다) ③ 실행 결과를 합격선·업무 점수와 비교한 문장을 쓰게 한다. AI 답의 함수·인자는 설치된 버전의 공식 문서로 확인하게 하고, 과제에 AI 도움을 받았으면 사용 범위를 밝히게 한다.'};
var CODE={title:'DBSCAN 면접 — 촘촘한 동네와 외톨이, 그리고 채점 봉투',sub:'numpy, scikit-learn · 인터넷 필요 없음 · CPU 1분 안팎(t-SNE 포함)',file:'dbscan_interview',
 desc:'이 편의 면접을 그대로 옮겼다. 손글씨 1,797장을 읽고 정답을 봉투에 넣고(1번 칸), 매뉴얼 씨의 합격선을 세우고(2번), 기본값의 함정(3번), k-거리 무릎으로 eps 고르기(4번), 기준 모델과 채점 봉투 열기(5번), eps × min_samples 민감도(6번), 차원의 저주(7번), min_samples 3(8번), 정답을 엿본 eps(9번, 반칙 표시), HDBSCAN(10번), 군집팀 비교와 그림용 t-SNE 지도(11번), 판정(12번)까지 간다. 정답을 쓰는 줄에는 모두 # [채점]이 붙어 있다. 노트북으로 저장하면 칸마다 셀이 나뉜다.',
 cap:'실행 결과 (scikit-learn 1.9.1, CPU). 시간은 컴퓨터마다 다르다. 브라우저 계산과 같은 숫자(무릎 eps 25.16 → 묶음 3개 · 잡음 50장 · ARI 0.001 · 업무 점수 9.8% 보류, min_samples 3이면 ARI 0.513 · 55.3%)가 나온다. 7번 칸의 무작위 점은 난수가 달라 04장 ③ 그림과 소수 둘째 자리가 조금 다르다.'};
