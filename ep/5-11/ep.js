var BOARD_LINE='ARI 0.108'; /* 군집팀 합격선: 매뉴얼 씨 8묶음의 ARI (CLU_LINE) */
/* ══ 5-11 계층적 군집 · 손글씨 숫자 실기 과제지 ══════════════════════════════ */
/* HC-ENGINE-START */
/* ── 계층적 군집 엔진 (이 편 전용, 브라우저·node 공용) ──
   · 거리: 64칸 픽셀의 유클리드 거리. 1,797장이면 쌍 1,614,006개를 한 줄 배열(Float64Array)에 담는다
   · 합치기: NN-chain(가장 가까운 이웃을 사슬처럼 따라가다 서로 가장 가까운 둘을 만나면 합친다). O(n²)
   · 합친 뒤 새 거리: Lance-Williams 식. ward(분산 증가) · average(평균) · complete(가장 먼 쌍) · single(가장 가까운 쌍)
   · 결과: scipy linkage와 같은 꼴 [왼쪽, 오른쪽, 높이, 크기] × (n−1). 높이 순으로 정렬한 뒤 번호를 다시 매긴다 */
var HC=(function(){
  function cidx(n,i,j){if(i>j){var t=i;i=j;j=t;}return n*i-(i*(i+1)>>1)+(j-i-1);}
  function pdist(X){var n=X.length,d=X[0].length,D=new Float64Array(n*(n-1)/2),k=0;
    for(var i=0;i<n;i++){var a=X[i];for(var j=i+1;j<n;j++){var b=X[j],s=0;for(var q=0;q<d;q++){var v=a[q]-b[q];s+=v*v;}D[k++]=Math.sqrt(s);}}return D;}
  function upd(m,dxa,dxb,dab,sa,sb,sx){
    if(m==='ward'){var t=sa+sb+sx;return Math.sqrt(((sa+sx)*dxa*dxa+(sb+sx)*dxb*dxb-sx*dab*dab)/t);}
    if(m==='average')return (sa*dxa+sb*dxb)/(sa+sb);
    if(m==='complete')return dxa>dxb?dxa:dxb;
    return dxa<dxb?dxa:dxb;}
  /* D0: pdist 결과(복사해서 쓴다). 반환 Z: [[a,b,h,size]…] (scipy 번호: 0..n−1은 낱장, n+i는 i번째 합치기) */
  function linkage(D0,n,method){var D=new Float64Array(D0),size=new Int32Array(n).fill(1),act=new Uint8Array(n).fill(1),chain=new Int32Array(n),cl=0,raw=[];
    for(var step=0;step<n-1;step++){
      if(cl===0){for(var s=0;s<n;s++)if(act[s]){chain[0]=s;cl=1;break;}}
      var x,y,cur;
      while(true){x=chain[cl-1];var best=Infinity,by=-1;
        if(cl>1){by=chain[cl-2];best=D[cidx(n,x,by)];}
        for(var i=0;i<n;i++){if(!act[i]||i===x)continue;var dv=D[cidx(n,x,i)];if(dv<best){best=dv;by=i;}}
        y=by;cur=best;
        if(cl>1&&y===chain[cl-2])break;
        chain[cl++]=y;}
      cl-=2;
      if(x>y){var t=x;x=y;y=t;}
      var nx=size[x],ny=size[y];raw.push([x,y,cur,nx+ny]);
      act[x]=0;size[y]=nx+ny;
      for(i=0;i<n;i++){if(!act[i]||i===y)continue;D[cidx(n,i,y)]=upd(method,D[cidx(n,i,x)],D[cidx(n,i,y)],cur,nx,ny,size[i]);}}
    /* 높이 순으로 세우고(같으면 먼저 합친 순) scipy처럼 번호를 다시 매긴다 */
    var ord=raw.map(function(r,i){return i;}).sort(function(a,b){return raw[a][2]-raw[b][2]||a-b;});
    var par=new Int32Array(2*n-1),nxt=n,Z=[];for(var i2=0;i2<2*n-1;i2++)par[i2]=i2;
    function find(a){while(par[a]!==a){par[a]=par[par[a]];a=par[a];}return a;}
    ord.forEach(function(o){var r=raw[o],a=find(r[0]),b=find(r[1]),la=Math.min(a,b),lb=Math.max(a,b);Z.push([la,lb,r[2],r[3]]);par[a]=nxt;par[b]=nxt;nxt++;});
    return Z;}
  /* 앞에서부터 n−k번 합친 상태 = 묶음 k개. 라벨은 0..k−1 (낱장 0이 속한 묶음부터 차례로) */
  function cut(Z,n,k){var par=new Int32Array(2*n-1);for(var i=0;i<2*n-1;i++)par[i]=i;
    function find(a){while(par[a]!==a){par[a]=par[par[a]];a=par[a];}return a;}
    for(var m=0;m<n-k;m++){par[Z[m][0]]=n+m;par[Z[m][1]]=n+m;}
    var map={},c=0,lab=new Int32Array(n);for(i=0;i<n;i++){var r=find(i);if(map[r]==null)map[r]=c++;lab[i]=map[r];}return lab;}
  /* 덴드로그램 잎 순서 (왼쪽 → 오른쪽) */
  function leaves(Z,n,node){var out=[],st=[node==null?2*n-2:node];while(st.length){var v=st.pop();if(v<n)out.push(v);else{st.push(Z[v-n][1]);st.push(Z[v-n][0]);}}return out;}
  /* 채점용: ARI · NMI(산술 평균 정규화, scikit-learn 기본) · 실루엣 */
  function table(a,b){var A={},B={},T={},n=a.length;for(var i=0;i<n;i++){A[a[i]]=(A[a[i]]||0)+1;B[b[i]]=(B[b[i]]||0)+1;var k=a[i]+'|'+b[i];T[k]=(T[k]||0)+1;}return {A:A,B:B,T:T,n:n};}
  function c2(x){return x*(x-1)/2;}
  function ari(a,b){var t=table(a,b),s=0,sa=0,sb=0,k;for(k in t.T)s+=c2(t.T[k]);for(k in t.A)sa+=c2(t.A[k]);for(k in t.B)sb+=c2(t.B[k]);var e=sa*sb/c2(t.n),mx=(sa+sb)/2;return mx===e?1:(s-e)/(mx-e);}
  function nmi(a,b){var t=table(a,b),n=t.n,H=function(O){var h=0;for(var k in O){var p=O[k]/n;h-=p*Math.log(p);}return h;},ha=H(t.A),hb=H(t.B),mi=0;
    for(var k in t.T){var p=k.split('|'),c=t.T[k];mi+=c/n*Math.log(c*n/(t.A[p[0]]*t.B[p[1]]));}var den=(ha+hb)/2;return den<=0?(ha===hb?1:0):Math.max(0,mi)/den;}
  function silhouette(D,n,lab){var K=0;for(var i=0;i<n;i++)if(lab[i]+1>K)K=lab[i]+1;var cnt=new Int32Array(K);for(i=0;i<n;i++)cnt[lab[i]]++;
    var S=new Float64Array(n*K);for(i=0;i<n;i++)for(var j=i+1;j<n;j++){var d=D[cidx(n,i,j)];S[i*K+lab[j]]+=d;S[j*K+lab[i]]+=d;}
    var tot=0;for(i=0;i<n;i++){var c=lab[i];if(cnt[c]<2)continue;var a=S[i*K+c]/(cnt[c]-1),b=Infinity;for(var k=0;k<K;k++)if(k!==c&&cnt[k]){var v=S[i*K+k]/cnt[k];if(v<b)b=v;}tot+=(b-a)/Math.max(a,b);}return tot/n;}
  /* 업무 점수: 묶음마다 평균에 가장 가까운 한 장(대표)에만 사람이 이름을 붙이고, 그 이름을 묶음 전체에 붙인다 */
  function reps(X,lab){var n=X.length,d=X[0].length,K=0;for(var i=0;i<n;i++)if(lab[i]+1>K)K=lab[i]+1;
    var mu=[],cnt=new Int32Array(K);for(var k=0;k<K;k++)mu.push(new Float64Array(d));
    for(i=0;i<n;i++){cnt[lab[i]]++;for(var q=0;q<d;q++)mu[lab[i]][q]+=X[i][q];}
    for(k=0;k<K;k++)for(q=0;q<d;q++)mu[k][q]/=cnt[k]||1;
    var rep=new Int32Array(K).fill(-1),bd=new Float64Array(K).fill(Infinity);
    for(i=0;i<n;i++){var c=lab[i],s=0;for(q=0;q<d;q++){var v=X[i][q]-mu[c][q];s+=v*v;}if(s<bd[c]){bd[c]=s;rep[c]=i;}}
    return {K:K,mu:mu,cnt:cnt,rep:rep};}
  function propagate(X,lab,y){var R=reps(X,lab),ok=0;for(var i=0;i<X.length;i++)if(y[R.rep[lab[i]]]===y[i])ok++;R.acc=ok/X.length;return R;}
  function purity(lab,y){var T={};for(var i=0;i<lab.length;i++){var k=lab[i];T[k]=T[k]||{};T[k][y[i]]=(T[k][y[i]]||0)+1;}var s=0;for(var k2 in T){var m=0;for(var v in T[k2])if(T[k2][v]>m)m=T[k2][v];s+=m;}return s/lab.length;}
  return {cidx:cidx,pdist:pdist,linkage:linkage,cut:cut,leaves:leaves,ari:ari,nmi:nmi,silhouette:silhouette,reps:reps,propagate:propagate,purity:purity};})();
/* HC-ENGINE-END */

const DR=digitRows(),X=DR.map(r=>r.px),N=X.length;
/* 채점 봉투: 숨긴 정답은 ENV.y에만 둔다. 합치기(HC.linkage)는 X로 만든 거리만 받는다 */
const ENV={y:DR.map(r=>r.y)};
const MAN=X.map(manualDigit),LINE_ARI=HC.ari(ENV.y,MAN),LINE_NMI=HC.nmi(ENV.y,MAN);
const RAND_ARI=(function(){const R=RNG(2024);let s=0;for(let t=0;t<20;t++)s+=HC.ari(ENV.y,X.map(()=>Math.floor(R()*10)));return s/20;})();
const LINKS=['ward','average','complete','single'],LCOL={ward:'blue',average:'ok',complete:'amber',single:'stamp'};
const KBASE=10;
const f3=v=>fx(v,3);
/* ── 계산 창고: 거리표 한 번, 연결 방식마다 합치기 한 번, 자르기·실루엣은 필요할 때 ── */
let DIST=null;const ZS={},MS={},LAB={},SIL={},PROP={};
function ensureD(){if(!DIST){const t=performance.now();DIST=HC.pdist(X);MS.d=performance.now()-t;if($('msD'))$('msD').textContent=Math.round(MS.d)+'ms';}return DIST;}
function ensureZ(m){if(!ZS[m]){ensureD();const t=performance.now();ZS[m]=HC.linkage(DIST,N,m);MS[m]=performance.now()-t;if(m==='ward'&&$('msZ'))$('msZ').textContent=Math.round(MS[m])+'ms';}return ZS[m];}
function labK(m,k){const key=m+'|'+k;return LAB[key]||(LAB[key]=HC.cut(ensureZ(m),N,k));}
function silK(m,k){const key=m+'|'+k;if(SIL[key]==null)SIL[key]=HC.silhouette(ensureD(),N,labK(m,k));return SIL[key];}
function propK(m,k){const key=m+'|'+k;return PROP[key]||(PROP[key]=HC.propagate(X,labK(m,k),ENV.y));}
async function prepAll(cb){ensureD();if(cb)cb(0);await wait(15);for(let i=0;i<LINKS.length;i++){ensureZ(LINKS[i]);if(cb)cb(i+1);await wait(15);}}
/* 화면에 가까워지면 계산 */
function whenNear(id,fn,margin){const t=$(id);let done=false;function go(){if(done)return;done=true;setTimeout(fn,10);}
  if('IntersectionObserver' in window){const io=new IntersectionObserver(es=>{if(es.some(e=>e.isIntersecting)){go();io.disconnect();}},{rootMargin:(margin||600)+'px'});io.observe(t);}else go();
  window.addEventListener('hashchange',()=>{const h=location.hash.slice(1),el=h&&$(h);if(el&&(el===t||t.contains(el)||t.compareDocumentPosition(el)&Node.DOCUMENT_POSITION_FOLLOWING))go();});
  if(location.hash){const el=$(location.hash.slice(1));if(el&&(el===t||t.contains(el)||t.compareDocumentPosition(el)&Node.DOCUMENT_POSITION_FOLLOWING))go();}}
function axes(ctx,C,pl,pt,gw,gh){ctx.strokeStyle=C.rule;ctx.lineWidth=1;ctx.strokeRect(pl,pt,gw,gh);ctx.fillStyle=C.muted;ctx.font=font(12);}
function vlabel(ctx,txt,x,y){ctx.save();ctx.translate(x,y);ctx.rotate(-Math.PI/2);ctx.textAlign='center';ctx.fillText(txt,0,0);ctx.restore();}
function barRows(rows,max){return rows.map(r=>'<div class="row"><span style="text-align:left;font-family:var(--font)">'+r[0]+'</span><div class="tr"><i style="left:0;width:'+Math.max(0,Math.min(100,r[1]/max*100))+'%;background:'+(r[2]||'var(--blue)')+'"></i></div><span>'+r[3]+'</span></div>').join('');}
/* 글씨 그리기: 종이 칸 + 잉크 */
function digitTile(ctx,C,px,x,y,s,frame){ctx.fillStyle=C.panel2;ctx.fillRect(x,y,s,s);drawDigit(ctx,px,x,y,s,C.text);if(frame){ctx.strokeStyle=frame;ctx.lineWidth=2;ctx.strokeRect(x+1,y+1,s-2,s-2);}}
function meanPx(idx){const m=new Array(64).fill(0);idx.forEach(i=>{for(let j=0;j<64;j++)m[j]+=X[i][j];});return m.map(v=>v/(idx.length||1));}
/* 덴드로그램 배치: 마지막 p갈래만 그린다. 반환 {leaves:[노드 번호], pos:{번호:[x(잎 단위), 높이]}} */
function dendroLayout(Z,n,p){const thr=n-p,isIn=v=>v>=n&&(v-n)>=thr,leaves=[],pos={};
  (function walk(v){if(!isIn(v)){pos[v]=[leaves.length+0.5,0];leaves.push(v);return;}const z=Z[v-n];walk(z[0]);walk(z[1]);pos[v]=[(pos[z[0]][0]+pos[z[1]][0])/2,z[2]];})(2*n-2);
  return {leaves,pos,thr};}

/* ── 히어로: 가계도 무대 ── */
heroRoom($('heroCv'),{sym:'H',no:'APP-11',name:'계층 군집\n(병합)',role:'지원자 · 군집팀',banner:'모델 채용 위원회 · 5-11 면접',doc:'front',asst:'board',fly:'clip',robo:'calc',loop:11.5,
  beats:[{at:0.4,who:'doc',mood:'board',move:'nod',say:'가까운 둘부터 합쳐요'},{at:2.6,who:'fly',mood:'type',move:'tilt',say:'몇 묶음에서 잘라?'},{at:4.8,who:'robo',mood:'count',move:'bob',say:'거리 161만 쌍?'},
         {at:7.0,who:'asst',mood:'ledger',move:'nod',say:'가계도가 자라요'},{at:9.0,who:'doc',mood:'magic',move:'pop',say:'자르는 높이 = 묶음 수'}],
  motif:(function(){const LX=[0,1,2,3,4,5,6,7].map(i=>304+i*15),MG=[[0,1,10],[4,5,14],[2,3,19],[6,7,24],[8,10,40],[9,11,48],[12,13,76]];
    return function(ctx,t,W,H,C){const m=REDUCE?7:Math.min(7,Math.floor(((t*1.1)%11)));const base=196,sy=h=>base-h*1.35;const P=LX.map(x=>[x,0]);
      ctx.lineWidth=2.4;MG.forEach((g,i)=>{const a=P[g[0]],b=P[g[1]];P.push([(a[0]+b[0])/2,g[2]]);if(i>=m)return;ctx.strokeStyle=i===m-1?C.stamp:C.c_app;ctx.beginPath();ctx.moveTo(a[0],sy(a[1]));ctx.lineTo(a[0],sy(g[2]));ctx.lineTo(b[0],sy(g[2]));ctx.lineTo(b[0],sy(b[1]));ctx.stroke();});
      LX.forEach((x,i)=>circ(ctx,x,base+6,4.5,C.text,null));
      if(m>=7){const k=2+Math.floor((t*0.5)%3),hc=k===2?60:k===3?44:30;ctx.strokeStyle=C.stamp;ctx.setLineDash([5,4]);ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(296,sy(hc));ctx.lineTo(420,sy(hc));ctx.stroke();ctx.setLineDash([]);ctx.fillStyle=C.stamp;ctx.font=font(12,700);ctx.textAlign='left';ctx.fillText('✂ '+k+'묶음',296,sy(hc)-6);}
      ctx.fillStyle=C.muted;ctx.font=font(12,500,1);ctx.textAlign='left';ctx.fillText('합치기 '+m+'번 · 가계도',298,222);};})()});
$('heroLine').textContent='ARI '+f3(LINE_ARI);

/* ── 01 장면 ── */
comic('comicIntro',[
 {bg:'board',prop:'1,797장 · 한 장이 한 묶음',cast:[{who:'doc',mood:'board',x:30,move:'nod',h:104},{who:'asst',mood:'shock',x:74,move:'shake',flip:true}],say:{who:'doc',txt:'처음엔 한 장이 한 묶음이에요. 가장 닮은 두 장부터 합칩니다.'},cap:'① 가장 가까운 둘 합치기 — 1,797묶음에서 1,796묶음으로'},
 {bg:'board',prop:'합친 기록 1,796줄 = 가계도',cast:[{who:'asst',mood:'ledger',x:30,move:'nod'},{who:'fly',mood:'type',x:74,move:'tilt',flip:true}],say:{who:'asst',txt:'합칠 때마다 거리를 높이로 적었더니 가계도가 됐어요. 아래가 닮은 사이, 위가 먼 사이예요.'},cap:'② 나무(덴드로그램) — 합친 순서와 거리가 그대로 남는다'},
 {bg:'board',prop:'✂ 자르는 높이 = 묶음 수',cast:[{who:'fly',mood:'point',x:30,move:'shake'},{who:'robo',mood:'calc',x:72,move:'bob',h:104}],say:{who:'fly',txt:'그래서 어디서 자를 건데? 정답 안 보고 높이를 정할 수 있어?'},cap:'③ 어디서 자를까 — 가로선 하나가 묶음 수를 정한다'}]);

/* ── 01 자기소개: 과제지 맨 앞 20장 합치기 ── */
(function(){const n=20,XS=X.slice(0,n),D0=HC.pdist(XS),cv=$('miniCv'),st={m:'ward',Z:null,ord:null,step:0,play:0};
  function build(){st.Z=HC.linkage(D0,n,st.m);st.ord=HC.leaves(st.Z,n);st.step=0;}
  function draw(){const W=fit(cv,340),ctx=cv.getContext('2d'),C=cols(),pl=40,pr=8,cell=Math.min(46,(W-pl-pr)/n),yb=340-cell-22,yt=16,Z=st.Z,hmax=Z[n-2][2]*1.04,sy=h=>yb-h/hmax*(yb-yt);
    ctx.clearRect(0,0,W,340);const xs={};st.ord.forEach((v,p)=>{xs[v]=pl+(p+0.5)*cell;});const ys={};for(let v=0;v<n;v++)ys[v]=yb;
    ctx.strokeStyle=C.rule;ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(pl,yt);ctx.lineTo(pl,yb);ctx.stroke();ctx.fillStyle=C.muted;ctx.font=font(11.5);ctx.textAlign='right';
    [0,0.5,1].forEach(f=>{const h=hmax*f;ctx.fillText(Math.round(h),pl-4,sy(h)+4);});vlabel(ctx,'높이 (거리)',10,(yt+yb)/2);
    for(let j=0;j<n-1;j++){const z=Z[j],v=n+j;xs[v]=(xs[z[0]]+xs[z[1]])/2;ys[v]=sy(z[2]);if(j>=st.step)continue;
      const last=j===st.step-1;ctx.strokeStyle=last?C.stamp:C.blue;ctx.lineWidth=last?3.2:2;ctx.beginPath();ctx.moveTo(xs[z[0]],ys[z[0]]);ctx.lineTo(xs[z[0]],ys[v]);ctx.lineTo(xs[z[1]],ys[v]);ctx.lineTo(xs[z[1]],ys[z[1]]);ctx.stroke();}
    const lab=HC.cut(Z,n,n-st.step),cnt={};for(let i=0;i<n;i++)cnt[lab[i]]=(cnt[lab[i]]||0)+1;const gid={};let g=0;
    st.ord.forEach((v,p)=>{const x=pl+p*cell,L=lab[v];if(cnt[L]>1&&gid[L]==null)gid[L]=g++;digitTile(ctx,C,X[v],x+1,yb+4,cell-2,null);
      if(cnt[L]>1){ctx.fillStyle=DIGCOL[gid[L]%10];ctx.fillRect(x+1,yb+cell+4,cell-2,5);}
      if(cell>=24){ctx.fillStyle=C.muted;ctx.font=font(10,500,1);ctx.textAlign='center';ctx.fillText('#'+(v+1),x+cell/2,340-3);}});
    ctx.fillStyle=C.text;ctx.font=font(12.5,700);ctx.textAlign='right';ctx.fillText(st.m+' · 합치기 '+st.step+'/19 · 지금 '+(n-st.step)+'묶음',W-pr,yt+4);info();}
  function info(){const Z=st.Z;let h='';
    if(!st.step)h='스무 장이 모두 따로다(20묶음). 지원자는 190쌍의 거리를 먼저 잰다. "한 번 합치기"를 누르면 그중 가장 가까운 두 장이 빨간 선으로 이어진다.';
    else{const z=Z[st.step-1],sa=z[0]<n?1:Z[z[0]-n][3],sb=z[1]<n?1:Z[z[1]-n][3];
      h=st.step+'번째 합치기: <b>'+sa+'장</b> 묶음과 <b>'+sb+'장</b> 묶음을 높이 <b>'+z[2].toFixed(1)+'</b>에서 합쳤다('+st.m+'로 잰 두 묶음 사이 거리). 지금 '+(n-st.step)+'묶음. ';
      if(st.step<=6)h+='처음 몇 번은 낱장 두 장끼리 합친다. 아래 그림을 눈으로 보면, 같은 숫자 두 장이 먼저 만나는 일이 많다. 받은 순서대로 0~9가 두 번씩 들어 있다.';
      else if(st.step<19)h+=st.m==='single'?'single은 이미 커진 묶음에 낱장을 하나씩 매단다. 높이가 비슷한 가로줄이 계단처럼 쌓인다. 이것이 1,797장에서 사슬 효과로 커진다(04장).':st.m==='ward'?'ward는 큰 묶음끼리 합칠수록 값이 커져서, 위로 갈수록 가로줄 사이가 벌어진다. 가로줄 사이가 크게 벌어진 곳이 "자르기 좋은 높이"다.':'뒤로 갈수록 묶음끼리 합친다. 높이가 점점 커진다.';
      else h+='모두 한 묶음이 되었다. 이 그림이 덴드로그램이다. 어떤 높이에 가로선을 그으면, 그 선에 걸린 세로줄 수가 묶음 수다.';}
    $('mInfo').innerHTML=h;}
  $('mStep').addEventListener('click',()=>{if(st.play||st.step>=n-1)return;st.step++;draw();});
  $('mPlay').addEventListener('click',async()=>{if(st.play)return;st.play=1;$('mPlay').disabled=true;if(st.step>=n-1)st.step=0;while(st.step<n-1){st.step++;draw();await wait(REDUCE?0:320);}st.play=0;$('mPlay').disabled=false;});
  $('mReset').addEventListener('click',()=>{if(st.play)return;st.step=0;draw();});
  seg('#mLink button',v=>{if(st.play)return;st.m=v;build();draw();});
  build();draw();onRedraw(draw);})();

/* ── 02 실기 과제지: 1,797장 격자와 64칸 숫자 ── */
$('lineAri').textContent='ARI '+f3(LINE_ARI);$('lineRand').textContent=f3(RAND_ARI);
(function(){const cv=$('gridCv'),pc=$('pxCv'),st={order:'raw',sel:0,cols:40,cell:12,idx:null};
  function setOrder(){st.idx=[...Array(N).keys()];if(st.order==='man')st.idx.sort((a,b)=>MAN[a]-MAN[b]||a-b);
    const cnt=new Array(8).fill(0);MAN.forEach(g=>cnt[g]++);
    $('gNote').textContent=st.order==='man'?'규칙 묶음 8개: '+cnt.map((c,g)=>'#'+g+' '+c+'장').join(' · '):'받은 순서 그대로. 이름표는 없다.';}
  function draw(){const w=cv.clientWidth||cv.parentElement.clientWidth||600;st.cols=w>560?40:w>360?30:24;st.cell=Math.floor(w/st.cols);const rows=Math.ceil(N/st.cols),H=rows*st.cell+2;
    const W=fit(cv,H),ctx=cv.getContext('2d'),C=cols(),c=st.cell;ctx.clearRect(0,0,W,H);
    st.idx.forEach((i,p)=>{const x=(p%st.cols)*c,y=Math.floor(p/st.cols)*c;ctx.fillStyle=C.panel2;ctx.fillRect(x,y,c-1,c-1);drawDigit(ctx,X[i],x,y,c-1,C.text);
      if(st.order==='man'){ctx.fillStyle=DIGCOL[MAN[i]];ctx.fillRect(x,y+c-3,c-1,2);}
      if(i===st.sel){ctx.strokeStyle=C.stamp;ctx.lineWidth=2;ctx.strokeRect(x+1,y+1,c-3,c-3);}});
    drawPx();}
  function drawPx(){const W=fit(pc,Math.min(300,(pc.clientWidth||300))),ctx=pc.getContext('2d'),C=cols(),s=Math.min(W,300),c=s/8,px=X[st.sel];ctx.clearRect(0,0,W,s);
    for(let j=0;j<64;j++){const x=(j%8)*c,y=Math.floor(j/8)*c,v=px[j];ctx.fillStyle=C.panel2;ctx.fillRect(x,y,c-1,c-1);ctx.fillStyle=rgba(C.text,v/16);ctx.fillRect(x,y,c-1,c-1);
      ctx.fillStyle=v>8?C.panel:C.muted;ctx.font=font(Math.max(10,c*0.32),700,1);ctx.textAlign='center';ctx.fillText(v,x+c/2,y+c/2+4);
      if([27,28,35,36].indexOf(j)>=0){ctx.strokeStyle=C.blue;ctx.lineWidth=2;ctx.strokeRect(x+1,y+1,c-3,c-3);}}
    ctx.strokeStyle=C.amber;ctx.setLineDash([4,3]);ctx.beginPath();ctx.moveTo(0,s/2);ctx.lineTo(s,s/2);ctx.stroke();ctx.setLineDash([]);
    const f=digitFeat(px),g=manualDigit(px);
    $('pxInfo').innerHTML='글씨 <b>#'+(st.sel+1)+'</b> · 64칸 합(잉크) <b>'+f.ink+'</b> · 위쪽 절반 비율 <b>'+f.top.toFixed(2)+'</b> · 가운데 네 칸(파란 테두리) <b>'+f.center+'</b>.<br>매뉴얼 씨의 규칙: '+(f.center<12?'가운데가 비었다':'가운데가 찼다')+' · '+(f.ink>300?'잉크가 많다':'잉크가 적다')+' · '+(f.top>0.5?'위가 무겁다':'아래가 무겁다')+' → <b>규칙 묶음 #'+g+'</b>. 지원자에게 이 글씨는 숫자 64개짜리 한 줄일 뿐이다.';}
  cv.addEventListener('click',e=>{const r=cv.getBoundingClientRect(),c=st.cell,p=Math.floor((e.clientY-r.top)/c)*st.cols+Math.floor((e.clientX-r.left)/c);if(p>=0&&p<N){st.sel=st.idx[p];draw();}});
  seg('#gOrder button',v=>{st.order=v;setOrder();draw();});
  setOrder();draw();onRedraw(draw);})();

/* ── 03 AI 비서 ── */
const MAIN_PROMPT=`너는 머신러닝 수업 조교다. Google Colab에서 바로 실행되는 파이썬 코드를 써 줘.

[환경] Colab 기본 라이브러리(numpy, scipy, scikit-learn, matplotlib)만 쓴다. 다른 패키지는 설치하지 않는다. 첫 줄에 scikit-learn 버전을 출력한다.
[데이터] 〔from sklearn.datasets import load_digits〕 로 손글씨 숫자 1,797장(8×8 = 64칸)을 읽는다.
  - 특징 X: 〔64칸 밝기(0~16)〕
  - 정답 y(0~9)는 군집에 쓰지 않는다. 마지막 채점 칸에서만 쓴다고 주석으로 밝힌다.
[규정]
  1) AgglomerativeClustering으로 연결 방식 〔ward, average, complete, single〕을 같은 묶음 수 〔10〕으로 비교한다.
  2) scipy.cluster.hierarchy의 linkage와 dendrogram으로 가계도를 그린다(truncate_mode="lastp", p=30).
  3) 묶음 수는 정답 없이 고른 근거를 함께 적는다(업무 지식, 높이가 크게 뛰는 곳, 실루엣 점수).
  4) 채점: adjusted_rand_score, normalized_mutual_info_score. 묶음 번호를 정답과 그대로 맞대는 정확도는 쓰지 않는다.
  5) 업무 점수: 묶음마다 평균에 가장 가까운 한 장에만 정답 이름을 붙여 묶음 전체에 퍼뜨렸을 때 맞은 비율.
[출력] 연결 방식별 ARI·NMI·실루엣·가장 큰 묶음 크기 표, 덴드로그램, 묶음 × 정답 표, 업무 점수, 기존 규칙(〔매뉴얼 씨 8묶음 ARI 0.108〕)과의 비교.
[금지] 정답으로 묶음 수나 연결 방식을 고르지 않는다. 표준화를 습관처럼 넣지 않는다(넣는다면 넣을 때와 뺄 때를 비교한다). 데이터를 지어내지 않는다.
[형식] 셀 단위로 나누고, 각 셀 위에 무엇을 하는지 한국어 주석을 단다. 코드 뒤에 내가 확인해야 할 점 3가지를 적어 줘.`;
$('mainPrompt').textContent=MAIN_PROMPT;$('copyMain').addEventListener('click',()=>copyText(MAIN_PROMPT,'프롬프트를 복사했습니다'));
(function(){const LINES=[
 ['from sklearn.datasets import load_digits',0,'scikit-learn에 들어 있는 손글씨 데이터. 인터넷 없이 읽는다. 괜찮다.'],['from sklearn.preprocessing import StandardScaler',0,'불러오는 것 자체는 괜찮다. 어디에 쓰는지가 문제다(7번 줄).'],['from sklearn.cluster import AgglomerativeClustering',0,'계층적 군집(병합). 괜찮다.'],['from sklearn.metrics import accuracy_score',0,'불러오는 것은 괜찮다. 군집 채점에 쓰는 것이 문제다(12번 줄).'],[''],
 ['X, y = load_digits(return_X_y=True)',0,'X와 정답 y를 함께 읽는다. y를 채점에만 쓰면 괜찮다.'],
 ['X = StandardScaler().fit_transform(X)        # 거리 재기 전 표준화는 필수',1,'<b>표준화는 "필수"가 아니다.</b> 64칸은 모두 같은 밝기 단위(0~16)다. 표준화하면 가장자리처럼 거의 늘 비어 있는 칸의 작은 차이가 다른 칸만큼 크게 부풀려진다. 이 과제에서 ward ARI가 0.794 → 0.664로 떨어진다(A장 11번 칸). 단위가 섞인 데이터라면 검토하되, 넣을 때와 뺄 때를 비교한다.'],[''],
 ['model = AgglomerativeClustering(n_clusters=10,',0,'묶음 10개(숫자 열 가지라는 업무 지식). 괜찮다.'],
 ['    linkage="ward", affinity="cosine")',2,'<b>두 가지가 틀렸다.</b> <code>affinity</code>는 1.4에서 지워진 이름이라 지금은 <code>TypeError</code>다(지금 이름은 <code>metric</code>). 이름을 고쳐도 ward는 유클리드 거리만 받으므로 <code>metric="cosine"</code>이면 <code>ValueError</code>다. ward를 쓰려면 metric을 빼고, cosine을 쓰려면 average·complete 연결로 바꾼다.'],
 ['labels = model.fit_predict(X)',0,'정답 없이 X만으로 묶는다. 괜찮다.'],
 ['print("정확도:", accuracy_score(y, labels))',3,'<b>묶음 번호를 정답처럼 채점했다.</b> 지원자가 붙인 "묶음 3"이 숫자 3이라는 보장은 없다. 완벽하게 묶어도 번호가 어긋나면 정확도는 0에 가까울 수 있다. 번호와 상관없이 "같이 묶였나"를 보는 <code>adjusted_rand_score</code>(ARI)나 NMI로 채점한다.'],
 ['heights = model.distances_                   # 덴드로그램 높이',4,'<b>이 속성은 기본으로 생기지 않는다.</b> <code>distances_</code>는 <code>compute_distances=True</code>를 주거나 <code>distance_threshold</code>를 쓸 때만 남는다. 지금은 <code>AttributeError</code>다. 덴드로그램은 scipy <code>linkage</code>로 그리는 편이 쉽다.'],[''],
 ['new_digits = X[:5]                           # 새로 들어온 글씨라고 치고',0,'예시용. 괜찮다.'],
 ['print(model.predict(new_digits))',5,'<b>계층적 군집에는 predict가 없다.</b> 가계도는 가진 글씨로만 만든다. <code>AttributeError</code>가 난다. 새 글씨는 다시 합치거나, 묶음 평균(또는 이웃)에 붙이는 규칙을 따로 만든다(06장).']];
  const host=$('bugCode'),found=new Set();
  host.innerHTML=LINES.map((l,i)=>'<div class="ln" role="listitem" tabindex="0" data-i="'+i+'"><span class="no">'+(i+1)+'</span><span>'+hl(l[0]||' ')+'</span></div>').join('');
  function score(){$('bugScore').textContent=found.size+' / 5';$('bugScore').className='pill'+(found.size===5?' ok':'');}
  function pick(el){const i=+el.getAttribute('data-i'),l=LINES[i];if(!l[0])return;
    if(l[1]){found.add(l[1]);host.querySelectorAll('.ln').forEach(x=>{if(LINES[+x.getAttribute('data-i')][1]===l[1])x.classList.add('hit');});$('bugMsg').innerHTML=(i+1)+'번 줄: '+l[2]+(found.size===5?'<br><b>다섯 군데를 모두 찾았다.</b> 고친 코드는 A장의 Colab 코드와 같은 순서(원래 픽셀 그대로 → 연결 방식에 맞는 거리 → ARI로 채점 → scipy로 가계도 → 새 글씨는 덧붙인 규칙)를 따른다.':'');}
    else{el.classList.add('miss');$('bugMsg').innerHTML=(i+1)+'번 줄: '+(l[2]||'이 줄은 괜찮다.');}score();}
  host.addEventListener('click',e=>{const el=e.target.closest('.ln');if(el)pick(el);});
  host.addEventListener('keydown',e=>{if((e.key==='Enter'||e.key===' ')&&e.target.classList.contains('ln')){e.preventDefault();pick(e.target);}});
  $('bugReveal').addEventListener('click',()=>{LINES.forEach((l,i)=>{if(l[1]){found.add(l[1]);host.querySelector('[data-i="'+i+'"]').classList.add('hit','reveal');}});score();$('bugMsg').innerHTML='빨간 줄 다섯 군데: ① 7번 습관적인 표준화 ② 10번 지워진 affinity와 ward + cosine ③ 12번 묶음 번호를 정확도로 채점 ④ 13번 기본으로 없는 distances_ ⑤ 16번 없는 predict.';});
  $('bugReset').addEventListener('click',()=>{found.clear();host.querySelectorAll('.ln').forEach(x=>x.classList.remove('hit','miss','reveal'));score();$('bugMsg').textContent='줄을 하나 눌러 보세요.';});score();})();
(function(){const CK=[['데이터를 실제로 읽었나','load_digits가 1,797 × 64인지, AI가 make_blobs 같은 가짜 데이터로 바꾸지 않았는지 본다.'],['정답 y를 학습에 넣지 않았나','fit(X)·fit_predict(X)에 y가 들어가지 않는다. y는 채점 칸에서만 쓴다.'],['표준화를 습관처럼 넣지 않았나','같은 단위의 픽셀이면 원래 값이 낫다. 넣었다면 넣을 때와 뺄 때를 비교한다.'],['인자 이름이 내 버전에 맞나','print(sklearn.__version__). affinity는 지금 오류, metric을 쓴다. ward는 euclidean만.'],['채점을 ARI·NMI로 했나','묶음 번호에는 뜻이 없다. accuracy_score(y, labels)는 틀린 채점이다.'],['묶음 수를 정답으로 고르지 않았나','ARI가 가장 높은 k를 고르면 정답을 엿본 것이다. 업무 지식, 높이의 도약, 실루엣으로 고른다.'],['연결 방식을 비교했나','ward 하나만 돌리면 single의 사슬 효과 같은 실패를 모른다. 가장 큰 묶음의 크기를 꼭 본다.'],['덴드로그램을 scipy로 그렸나','scikit-learn 모델에서 그리려면 compute_distances=True가 필요하다.'],['predict를 부르지 않았나','계층적 군집에는 predict가 없다. 새 데이터 배정 규칙은 따로 만든다.'],['합격선·무작위와 비교했나','매뉴얼 씨 8묶음 ARI 0.108과 무작위 0.000을 함께 적는다.']];
  const ul=$('checks');ul.innerHTML=CK.map((c,i)=>'<li><input type="checkbox" id="ck'+i+'"><label for="ck'+i+'"><b>'+c[0]+'</b><small>'+c[1]+'</small></label></li>').join('');
  function upd(){const n=ul.querySelectorAll('input:checked').length;ul.querySelectorAll('li').forEach(li=>li.classList.toggle('done',li.querySelector('input').checked));$('checkScore').textContent='확인 '+n+' / '+CK.length+(n===CK.length?' — 이제 돌려 본다. 결과도 확인 대상이다.':'');}
  ul.addEventListener('change',upd);upd();})();

/* ── 04 수습 기간 ① 가장 큰 묶음의 몫 (사슬 효과) ── */
const CHAIN=(function(){const cv=$('chainCv'),st={m:'ward',v:69,busy:false,ready:false},MX={},LN=Math.log(N);
  const kOf=v=>Math.max(1,Math.min(N,Math.round(Math.exp(LN*(1-v/100)))));
  function maxShare(m){if(MX[m])return MX[m];const Z=ensureZ(m),a=new Float64Array(N);let mx=1;a[0]=1/N;for(let j=0;j<N-1;j++){if(Z[j][3]>mx)mx=Z[j][3];a[j+1]=mx/N;}return (MX[m]=a);}
  function draw(){const w=cv.clientWidth||600,small=w<520,H=small?400:360,W=fit(cv,H),ctx=cv.getContext('2d'),C=cols(),pl=44,pr=12,pt=12,gh=small?190:200,gw=W-pl-pr,lx=k=>pl+(LN-Math.log(k))/LN*gw,ly=s=>pt+(1-s)*gh;
    ctx.clearRect(0,0,W,H);axes(ctx,C,pl,pt,gw,gh);
    [1797,500,100,30,10,3,1].forEach(k=>{if(small&&(k===500||k===3))return;ctx.textAlign='center';ctx.fillText(k.toLocaleString(),lx(k),pt+gh+15);});ctx.fillText('남은 묶음 수 (합칠수록 오른쪽 →, 로그 눈금)',pl+gw/2,pt+gh+30);
    [0,0.5,1].forEach(s=>{ctx.textAlign='right';ctx.fillText(s===0?'0':s===1?'100%':'50%',pl-5,ly(s)+4);});vlabel(ctx,'가장 큰 묶음의 몫',10,pt+gh/2);
    if(!st.ready)return;const k=kOf(st.v);
    LINKS.forEach((m,li)=>{const a=maxShare(m),on=m===st.m,col=C[LCOL[m]];ctx.strokeStyle=rgba(col,on?1:.45);ctx.lineWidth=on?3:1.6;ctx.beginPath();
      for(let j=0;j<N;j++){const X_=lx(N-j),Y_=ly(a[j]);j?ctx.lineTo(X_,Y_):ctx.moveTo(X_,Y_);}ctx.stroke();
      ctx.fillStyle=rgba(col,on?1:.7);ctx.font=font(12,on?700:500);ctx.textAlign='left';ctx.fillText(m,pl+8,pt+16+li*15);});
    ctx.strokeStyle=C.stamp;ctx.setLineDash([4,4]);ctx.beginPath();ctx.moveTo(lx(k),pt);ctx.lineTo(lx(k),pt+gh);ctx.stroke();ctx.setLineDash([]);
    const sh=maxShare(st.m)[N-k];circ(ctx,lx(k),ly(sh),5,C[LCOL[st.m]],C.panel,1.5);
    /* 아래: 고른 연결 방식에서 가장 큰 묶음 10개의 평균 글씨 */
    const lab=labK(st.m,k),R=HC.reps(X,lab),ord=[...Array(R.K).keys()].sort((a,b)=>R.cnt[b]-R.cnt[a]).slice(0,10),y0=pt+gh+44,cs=Math.min(64,(gw-9*8)/10);
    ctx.fillStyle=C.text;ctx.font=font(12.5,700);ctx.textAlign='left';ctx.fillText(st.m+' · '+k.toLocaleString()+'묶음일 때 가장 큰 묶음 '+Math.min(10,R.K)+'개 (평균 글씨 · 장수)',pl,y0-6);
    ord.forEach((c,i)=>{const x=pl+i*(cs+8);digitTile(ctx,C,R.mu[c],x,y0,cs,i===0?C[LCOL[st.m]]:null);ctx.fillStyle=C.muted;ctx.font=font(11.5,700,1);ctx.textAlign='center';ctx.fillText(R.cnt[c],x+cs/2,y0+cs+14);});
    info(k,sh,R);}
  function info(k,sh,R){const big=Math.round(sh*N);let h='<b>'+st.m+'</b> · 남은 묶음 <b>'+k.toLocaleString()+'</b>개 · 가장 큰 묶음 <b>'+big.toLocaleString()+'장('+pct(sh)+')</b>. ';
    const s100=maxShare('single')[N-100],w10=maxShare('ward')[N-10];
    h+=st.m==='single'?'single은 묶음이 100개나 남았을 때 이미 한 묶음이 '+pct(s100,0)+'를 차지한다. 가장 가까운 한 쌍만 있으면 합치니, 글씨 사이를 잇는 징검다리를 따라 큰 묶음이 낱장을 하나씩 삼킨다. 이것이 <b>사슬 효과</b>다.':
      st.m==='ward'?'ward는 10묶음이 남을 때까지 가장 큰 묶음이 '+pct(w10,0)+'에 머문다. 큰 묶음끼리 합치면 흩어짐이 크게 늘어 값이 비싸지므로, 크기가 고른 묶음이 끝까지 남는다.':
      st.m==='average'?'average는 ward와 single 사이다. 10묶음일 때 몇 장짜리 작은 묶음이 따로 남는다(평균 글씨가 흐릿하지 않은 칸).':'complete는 가장 먼 쌍으로 재서 묶음이 너무 커지는 것을 막지만, 글씨 한두 장의 모양에 끌려 묶음 경계가 흔들린다.';
    h+='<br><span class="muted">합치기 시간(이 브라우저): 거리 '+Math.round(MS.d||0)+'ms · '+LINKS.map(m=>m+' '+Math.round(MS[m]||0)+'ms').join(' · ')+'. 채점 봉투는 아직 봉인되어 있다.</span>';$('cInfo').innerHTML=h;}
  function setV(v){st.v=v;$('cK').value=v;$('cKO').textContent=kOf(v).toLocaleString();draw();}
  $('cK').addEventListener('input',e=>setV(+e.target.value));
  $('cPlay').addEventListener('click',async()=>{if(st.busy||!st.ready)return;st.busy=true;$('cPlay').disabled=true;for(let v=0;v<=100;v+=4){setV(v);await wait(REDUCE?0:140);}setV(69);st.busy=false;$('cPlay').disabled=false;});
  seg('#cLink button',v=>{st.m=v;draw();});
  async function start(){if(st.ready)return;$('cInfo').textContent='거리 1,613,706쌍을 재는 중…';await prepAll(i=>{$('cInfo').textContent=i?'연결 방식 '+i+'/4 합치는 중…':'거리를 다 쟀다. 합치는 중…';});st.ready=true;setV(st.v);DEND.start();}
  draw();onRedraw(draw);return {start};})();

/* ── 04 수습 기간 ② 가계도 자르기 ── */
const DEND=(function(){const cv=$('dendCv'),P=30,st={m:'ward',k:KBASE,ready:false},LAY={},MEAN={};
  function lay(m){return LAY[m]||(LAY[m]=dendroLayout(ensureZ(m),N,P));}
  function leafMean(m,v){const key=m+v;if(!MEAN[key]){const idx=v<N?[v]:HC.leaves(ensureZ(m),N,v);MEAN[key]={px:meanPx(idx),n:idx.length,first:idx[0]};}return MEAN[key];}
  function draw(){const w=cv.clientWidth||600,small=w<520,cell0=(w-50)/P,H=(small?300:360)+Math.min(cell0,40),W=fit(cv,H),ctx=cv.getContext('2d'),C=cols(),pl=44,pr=6,gw=W-pl-pr,cell=gw/P,pt=14,yb=H-Math.min(cell,40)-26;
    ctx.clearRect(0,0,W,H);if(!st.ready)return;const Z=ensureZ(st.m),L=lay(st.m),hmax=Z[N-2][2]*1.04,hmin=Z[N-P-1][2]*0.85,sy=h=>h<=0?yb:yb-8-(h-hmin)/(hmax-hmin)*(yb-8-pt);
    ctx.strokeStyle=C.rule;ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(pl,pt);ctx.lineTo(pl,yb);ctx.stroke();ctx.fillStyle=C.muted;ctx.font=font(11.5);ctx.textAlign='right';
    [hmin,(hmin+hmax)/2,hmax].forEach(h=>ctx.fillText(Math.round(h),pl-4,sy(h)+4));vlabel(ctx,'높이',10,(pt+yb)/2);
    const lab=labK(st.m,st.k),gcol={};let gi=0;const colOf=v=>{const L_=lab[leafMean(st.m,v).first];if(gcol[L_]==null)gcol[L_]=DIGCOL[(gi++)%10];return gcol[L_];};
    L.leaves.forEach(v=>colOf(v));
    const xOf=v=>pl+L.pos[v][0]*cell;
    for(let j=N-P;j<N-1;j++){const z=Z[j],v=N+j,below=j<N-st.k,c=below?colOf(v):C.text;ctx.strokeStyle=c;ctx.lineWidth=below?2.6:1.8;const yv=sy(z[2]);
      ctx.beginPath();ctx.moveTo(xOf(z[0]),sy(L.pos[z[0]][1]));ctx.lineTo(xOf(z[0]),yv);ctx.lineTo(xOf(z[1]),yv);ctx.lineTo(xOf(z[1]),sy(L.pos[z[1]][1]));ctx.stroke();}
    const cutH=(Z[N-st.k-1][2]+Z[N-st.k][2])/2;ctx.strokeStyle=C.stamp;ctx.lineWidth=2;ctx.setLineDash([6,4]);ctx.beginPath();ctx.moveTo(pl,sy(cutH));ctx.lineTo(pl+gw,sy(cutH));ctx.stroke();ctx.setLineDash([]);
    ctx.fillStyle=C.stamp;ctx.font=font(12,700);ctx.textAlign='right';ctx.fillText('✂ 높이 '+Math.round(cutH)+' → '+st.k+'묶음',pl+gw-4,sy(cutH)-6);
    const cs=Math.min(cell-2,40);L.leaves.forEach((v,i)=>{const M=leafMean(st.m,v),x=pl+i*cell+(cell-cs)/2;digitTile(ctx,C,M.px,x,yb+2,cs,null);ctx.fillStyle=colOf(v);ctx.fillRect(x,yb+cs+4,cs,3);
      if(cell>=22){ctx.fillStyle=C.muted;ctx.font=font(cell>=30?10.5:9,500,1);ctx.textAlign='center';ctx.fillText(M.n,x+cs/2,yb+cs+18);}});
    info(Z,cutH);}
  function info(Z,cutH){const lab=labK(st.m,st.k),cnt=new Array(st.k).fill(0);for(let i=0;i<N;i++)cnt[lab[i]]++;cnt.sort((a,b)=>b-a);
    let gj=0,gk=2;for(let k=2;k<=30;k++){const g=Z[N-k][2]-Z[N-k-1][2];if(g>gj){gj=g;gk=k;}}
    $('dInfo').innerHTML='<b>'+st.m+'</b> 가계도를 높이 <b>'+Math.round(cutH)+'</b>에서 자르면 <b>'+st.k+'묶음</b>. 묶음 크기: '+cnt.join(' · ')+'. 정답 없이 재는 실루엣 <b id="dSil">계산 중…</b>. '+
      '높이가 가장 크게 뛰는 곳은 '+gk+'묶음 → '+(gk-1)+'묶음으로 합칠 때('+Math.round(Z[N-gk-1][2])+' → '+Math.round(Z[N-gk][2])+')다. 그 기준이면 '+gk+'묶음에서 자른다.'+
      (st.m==='single'?' single 가계도는 한 줄기에 낱장이 하나씩 매달린 빗 모양이다. 어디서 잘라도 큰 묶음 하나와 낱장 몇 장이 된다.':st.m==='ward'&&st.k===KBASE?' 위원회는 "숫자는 열 가지"라는 업무 지식으로 10묶음에서 자른다.':'')+
      '<br><span class="muted">갈래 끝 그림은 그 갈래에 든 글씨들의 평균이고, 아래 숫자는 장수다. 나무는 다시 키우지 않는다. 슬라이더는 가로선만 옮긴다.</span>';
    setTimeout(()=>{const s=silK(st.m,st.k),e=$('dSil');if(e)e.textContent=f3(s);},20);}
  function set(k){st.k=k;$('dK').value=k;$('dKO').textContent=k;draw();}
  $('dK').addEventListener('input',e=>set(+e.target.value));seg('#dLink button',v=>{st.m=v;draw();});
  onRedraw(draw);return {start(){st.ready=true;set(st.k);}};})();
whenNear('m4',()=>CHAIN.start(),300);

/* ── 05 성과 평가 ── */
function renderEval(){const lab=labK('ward',KBASE),A=HC.ari(ENV.y,lab),M=HC.nmi(ENV.y,lab),S=silK('ward',KBASE);
  $('evalStat').innerHTML=statGrid([['ARI '+f3(A),'채점 봉투와 겹침 (1 = 완벽)','a'],['NMI '+f3(M),'정보 겹침 (1 = 완벽)'],['실루엣 '+f3(S),'정답 없이 잰 갈림 정도'],['ARI '+f3(LINE_ARI),'합격선 · 매뉴얼 씨 8묶음','m'],['ARI '+f3(RAND_ARI),'무작위 10묶음 (20번 평균)']]);
  ENVL.init();BIZ.init();SILC.start();renderLinkTbl();}
/* 채점 봉투 */
const ENVL=(function(){const cv=$('envCv');let open=false,T=null,rows=null;
  function prep(){const lab=labK('ward',KBASE);T=[];for(let c=0;c<KBASE;c++)T.push({c,n:0,d:new Array(10).fill(0),idx:[]});
    for(let i=0;i<N;i++){const r=T[lab[i]];r.n++;r.d[ENV.y[i]]++;r.idx.push(i);}T.forEach(r=>{r.px=meanPx(r.idx);r.maj=r.d.indexOf(Math.max(...r.d));});}
  function order(){rows=T.slice().sort(open?(a,b)=>a.maj-b.maj||b.n-a.n:(a,b)=>b.n-a.n);}
  function draw(){if(!T)return;const w=cv.clientWidth||600,rh=Math.max(24,Math.min(40,(w-20)/13)),H=rh*(KBASE+1)+6,W=fit(cv,H),ctx=cv.getContext('2d'),C=cols(),iw=rh,sw=rh*1.5,cw=(W-iw-sw-6)/10;
    ctx.clearRect(0,0,W,H);ctx.font=font(Math.min(13,rh*0.42),700,1);ctx.textAlign='center';ctx.fillStyle=C.muted;ctx.fillText('묶음',iw/2,rh*0.62);ctx.fillText('장수',iw+sw/2,rh*0.62);
    for(let d=0;d<10;d++){ctx.fillStyle=open?C.text:C.muted;ctx.fillText(open?d:'?',iw+sw+cw*(d+.5),rh*0.62);}
    rows.forEach((r,ri)=>{const y=rh*(ri+1)+2;digitTile(ctx,C,r.px,1,y,rh-3,null);ctx.fillStyle=C.text;ctx.font=font(Math.min(13,rh*0.4),700,1);ctx.textAlign='center';ctx.fillText(r.n,iw+sw/2,y+rh*0.62);
      for(let d=0;d<10;d++){const x=iw+sw+cw*d,v=r.d[d];if(open){const a=v/r.n;ctx.fillStyle=a>0?rgba(d===r.maj?C.blue:C.stamp,0.12+0.8*a):C.panel2;ctx.fillRect(x+1,y,cw-2,rh-3);if(v){ctx.fillStyle=a>0.45?'#fff':C.text;ctx.font=font(Math.min(12.5,cw*0.36),700,1);ctx.fillText(v,x+cw/2,y+rh*0.6);}}
        else{ctx.fillStyle=C.panel2;ctx.fillRect(x+1,y,cw-2,rh-3);}}});
    info();}
  function info(){if(!open){$('envInfo').innerHTML='지원자가 만든 묶음 10개다. 맨 앞 그림은 묶음의 평균 글씨, 둘째 칸은 장수. 평균 글씨만 봐도 몇 묶음은 무슨 숫자인지 짐작이 간다. 봉투를 열면 칸마다 숨긴 정답 0~9의 장수가 나온다.';return;}
    const mixed=T.filter(r=>{const s=r.d.slice().sort((a,b)=>b-a);return s[1]>=0.25*s[0];}),split=[];for(let d=0;d<10;d++){const parts=T.map(r=>r.d[d]).filter(v=>v>=0.12*T.reduce((s,r)=>s+r.d[d],0));if(parts.length>=2)split.push([d,parts.sort((a,b)=>b-a)]);}
    const pure=T.reduce((s,r)=>s+r.d[r.maj],0)/N;let h='파란 칸이 그 묶음에서 가장 많은 숫자, 빨간 칸이 섞여 든 숫자다. 묶음마다 가장 많은 숫자만 세면 1,797장 중 <b>'+pct(pure)+'</b>(순도). ';
    mixed.forEach(r=>{const o=r.d.map((v,d)=>[v,d]).sort((a,b)=>b[0]-a[0]);const pr=[o[0][1],o[1][1]].sort().join('');h+='한 묶음에 <b>숫자 '+o[0][1]+' '+o[0][0]+'장과 숫자 '+o[1][1]+' '+o[1][0]+'장</b>이 섞였다. '+(pr==='39'?'8×8 칸에서는 위쪽 고리 모양이 같고 아래쪽만 조금 다르다. ':'8×8 칸에서는 닮은 모양이다. ');});
    split.forEach(s=>{h+='숫자 <b>'+s[0]+'</b>'+('2459'.indexOf(s[0])>=0?'는':'은')+' '+s[1].length+'묶음으로 나뉘었다('+s[1].join('·')+'장). 같은 숫자라도 쓰는 법(곧게, 기울여, 받침을 붙여)이 달라 64칸이 서로 먼 글씨가 된다. ';});
    h+='지원자는 숫자를 몰랐다. 64칸이 닮은 글씨끼리 묶었을 뿐이다. 이 겹침을 번호와 상관없이 하나로 잰 것이 ARI '+f3(HC.ari(ENV.y,labK('ward',KBASE)))+'다.';$('envInfo').innerHTML=h;}
  $('envOpen').addEventListener('click',()=>{open=!open;$('envOpen').textContent=open?'✉ 봉투 다시 봉인하기':'✉ 채점 봉투 열기';$('envNote').textContent=open?'숨긴 정답 0~9가 열렸다. 줄은 가장 많은 숫자 순으로 다시 놓였다.':'지금은 묶음마다 평균 글씨와 장수만 보인다.';order();draw();});
  onRedraw(draw);return {init(){if(!T){prep();order();}draw();}};})();
/* 업무 점수 */
const BIZ=(function(){const rc=$('repCv'),cc=$('costCv'),KM=60;let k=KBASE,curve=null;
  function prep(){curve={};['ward','average','single'].forEach(m=>{const a=[];for(let q=2;q<=KM;q++)a[q]=propK(m,q).acc;curve[m]=a;});}
  function drawR(){const P=propK('ward',k),w=rc.clientWidth||600,tile=w<500?54:68,colsN=Math.max(1,Math.floor(w/tile)),ord=[...Array(P.K).keys()].sort((a,b)=>P.cnt[b]-P.cnt[a]).slice(0,Math.min(P.K,colsN*2)),rows=Math.ceil(ord.length/colsN),H=rows*(tile+30)+4;
    const W=fit(rc,H),ctx=rc.getContext('2d'),C=cols(),s=tile-14;ctx.clearRect(0,0,W,H);const lab=labK('ward',k);
    ord.forEach((c,i)=>{const x=(i%colsN)*tile+7,y=Math.floor(i/colsN)*(tile+30)+2,r=P.rep[c],nm=ENV.y[r];let ok=0;for(let q=0;q<N;q++)if(lab[q]===c&&ENV.y[q]===nm)ok++;
      digitTile(ctx,C,X[r],x,y,s,C.blue);ctx.fillStyle=C.blue;ctx.font=font(13,700);ctx.textAlign='center';ctx.fillText('"'+nm+'"',x+s/2,y+s+14);ctx.fillStyle=C.muted;ctx.font=font(10.5,500,1);ctx.fillText(ok+'/'+P.cnt[c],x+s/2,y+s+27);});}
  function drawC(){const W=fit(cc,250),ctx=cc.getContext('2d'),C=cols(),pl=44,pr=12,pt=10,gh=200,gw=W-pl-pr,lx=q=>pl+(q-2)/(KM-2)*gw,ly=v=>pt+(1-v)*gh;ctx.clearRect(0,0,W,250);axes(ctx,C,pl,pt,gw,gh);
    [2,10,20,30,40,50,60].forEach(q=>{ctx.textAlign='center';ctx.fillText(q,lx(q),pt+gh+15);});ctx.fillText('사람이 이름 붙인 장수 (= 묶음 수)',pl+gw/2,pt+gh+31);
    [0,0.5,1].forEach(v=>{ctx.textAlign='right';ctx.fillText(pct(v,0),pl-5,ly(v)+4);ctx.strokeStyle=rgba(C.muted,.18);ctx.beginPath();ctx.moveTo(pl,ly(v));ctx.lineTo(pl+gw,ly(v));ctx.stroke();});
    [['ward',3],['average',1.6],['single',1.6]].forEach(([m,lw])=>{const a=curve[m],col=C[LCOL[m]];ctx.strokeStyle=col;ctx.lineWidth=lw;ctx.beginPath();for(let q=2;q<=KM;q++){q===2?ctx.moveTo(lx(q),ly(a[q])):ctx.lineTo(lx(q),ly(a[q]));}ctx.stroke();
      ctx.fillStyle=col;ctx.font=font(12,700);ctx.textAlign='right';ctx.fillText(m,pl+gw-6,ly(a[KM])+(m==='average'?14:-6));});
    const man=HC.propagate(X,MAN,ENV.y).acc;circ(ctx,lx(8),ly(man),6,C.c_man,C.panel,1.5);ctx.fillStyle=C.c_man;ctx.font=font(11.5,700);ctx.textAlign='left';ctx.fillText('매뉴얼 씨 '+pct(man),lx(8)+8,ly(man)+4);
    circ(ctx,lx(10),ly(PRE.km.prop),5,C.muted,C.panel,1.5);ctx.fillStyle=C.muted;ctx.textAlign='left';ctx.fillText('K-Means '+pct(PRE.km.prop),lx(10)+8,ly(PRE.km.prop)+14);
    ctx.strokeStyle=C.stamp;ctx.setLineDash([4,4]);ctx.beginPath();ctx.moveTo(lx(k),pt);ctx.lineTo(lx(k),pt+gh);ctx.stroke();ctx.setLineDash([]);circ(ctx,lx(k),ly(curve.ward[k]),5,C.blue,C.panel,1.5);}
  function info(){const a=curve.ward[k],man=HC.propagate(X,MAN,ENV.y).acc;
    $('bizInfo').innerHTML='ward를 <b>'+k+'묶음</b>으로 자르고, 묶음마다 평균에 가장 가까운 대표 한 장(파란 테두리)에만 사람이 이름을 붙였다. 그 이름을 묶음 전체에 퍼뜨리면 1,797장 중 <b>'+Math.round(a*N).toLocaleString()+'장('+pct(a)+')</b>이 맞게 붙는다. 사람이 한 일은 1,797장 중 '+k+'장(<b>'+pct(k/N)+'</b>)이다. '+
      (k===KBASE?'열두 장이면 '+pct(curve.ward[12])+', 스무 장이면 '+pct(curve.ward[20])+'. 같은 열 장이라도 K-Means(5-10)는 '+pct(PRE.km.prop)+', 매뉴얼 씨는 여덟 장으로 '+pct(man)+'다. ':'')+
      'single은 몇 장을 붙여도 큰 묶음 하나가 한 이름을 받아 바닥에 머문다. 위 그림에는 큰 묶음부터 최대 두 줄만 그렸다.'+(k>=20?' 묶음이 많아지면 ARI는 내려가지만(한 숫자가 여러 묶음으로 나뉘니까) 업무 점수는 오른다. 이름 붙일 장수와 맞는 비율을 맞바꾸는 것이다.':'');}
  function set(q){k=q;$('bK').value=q;$('bKO').textContent=q;drawR();drawC();info();}
  $('bK').addEventListener('input',e=>set(+e.target.value));onRedraw(()=>{if(curve){drawR();drawC();}});
  return {init(){if(!curve)prep();set(k);}};})();
/* 실루엣 vs ARI */
const SILC=(function(){const cv=$('silCv'),K0=2,K1=20,S=[],A=[];let done=false;
  function draw(){const W=fit(cv,260),ctx=cv.getContext('2d'),C=cols(),pl=44,pr=44,pt=12,gh=200,gw=W-pl-pr,lx=q=>pl+(q-K0)/(K1-K0)*gw,lyS=v=>pt+(1-v/0.25)*gh,lyA=v=>pt+(1-v)*gh;ctx.clearRect(0,0,W,260);axes(ctx,C,pl,pt,gw,gh);
    for(let q=K0;q<=K1;q+=2){ctx.textAlign='center';ctx.fillText(q,lx(q),pt+gh+15);}ctx.fillText('묶음 수 (ward 가계도를 자른 곳)',pl+gw/2,pt+gh+31);
    [0,0.1,0.2].forEach(v=>{ctx.fillStyle=C.blue;ctx.textAlign='right';ctx.fillText(v.toFixed(1),pl-5,lyS(v)+4);});[0,0.5,1].forEach(v=>{ctx.fillStyle=C.stamp;ctx.textAlign='left';ctx.fillText(v.toFixed(1),pl+gw+5,lyA(v)+4);});
    ctx.strokeStyle=C.muted;ctx.setLineDash([3,4]);ctx.beginPath();ctx.moveTo(lx(KBASE),pt);ctx.lineTo(lx(KBASE),pt+gh);ctx.stroke();ctx.setLineDash([]);
    [[S,lyS,C.blue,'실루엣 (정답 없이)'],[A,lyA,C.stamp,'ARI (채점 봉투)']].forEach(([arr,ly,col,nm],i)=>{ctx.strokeStyle=col;ctx.lineWidth=2.6;ctx.beginPath();let f=true;for(let q=K0;q<=K1;q++){if(arr[q]==null)continue;f?ctx.moveTo(lx(q),ly(arr[q])):ctx.lineTo(lx(q),ly(arr[q]));f=false;}ctx.stroke();
      for(let q=K0;q<=K1;q++)if(arr[q]!=null)circ(ctx,lx(q),ly(arr[q]),3,col,null);ctx.fillStyle=col;ctx.font=font(12,700);ctx.textAlign='left';ctx.fillText(nm,pl+8,pt+16+i*15);});}
  function info(){const bs=best(S),ba=best(A);$('silInfo').innerHTML='정답 없이 재는 실루엣은 <b>'+bs+'묶음('+f3(S[bs])+')</b>에서 가장 높고, 채점 봉투로 잰 ARI는 <b>'+ba+'묶음('+f3(A[ba])+')</b>에서 가장 높다. 위원회가 업무 지식으로 정한 10묶음은 실루엣 '+f3(S[KBASE])+', ARI '+f3(A[KBASE])+'로 둘 다 꼭대기 근처다. 실루엣 그래프만 보고 9묶음을 골랐어도 ARI '+f3(A[9])+'로 합격선을 크게 넘는다. 다만 ARI가 가장 높은 12묶음을 "골랐다"고 하면 정답을 엿보고 고른 것이다. 근무 조건은 정답 없이 정하고, 채점 봉투는 확인에만 쓴다.';}
  function best(a){let b=K0;for(let q=K0;q<=K1;q++)if(a[q]>a[b])b=q;return b;}
  async function start(){if(done)return;done=true;for(let q=K0;q<=K1;q++){S[q]=silK('ward',q);A[q]=HC.ari(ENV.y,labK('ward',q));if(q%3===0){draw();$('silInfo').textContent='실루엣 계산 중… '+q+'/'+K1;await wait(10);}}draw();info();}
  onRedraw(draw);return {start};})();
function renderLinkTbl(){const R=LINKS.map(m=>{const l=labK(m,KBASE),c=new Array(KBASE).fill(0);for(let i=0;i<N;i++)c[l[i]]++;return [m,HC.ari(ENV.y,l),HC.nmi(ENV.y,l),silK(m,KBASE),Math.max(...c),propK(m,KBASE).acc,MS[m]];});
  $('linkTbl').innerHTML='<thead><tr><th>연결 방식</th><th>ARI</th><th>NMI</th><th>실루엣</th><th>가장 큰 묶음</th><th>업무 점수 (이름 10장)</th><th>합치기 (이 브라우저)</th><th>합격선 0.108</th></tr></thead><tbody>'+
    R.map(r=>'<tr><td><b style="color:var(--'+LCOL[r[0]]+')">'+r[0]+'</b></td><td>'+f3(r[1])+'</td><td>'+f3(r[2])+'</td><td>'+f3(r[3])+'</td><td>'+r[4].toLocaleString()+'장</td><td>'+pct(r[5])+'</td><td>'+Math.round(r[6])+'ms</td><td><span class="pill '+(r[1]>LINE_ARI&&r[1]>RAND_ARI?'ok':'no')+'">'+(r[1]>LINE_ARI&&r[1]>RAND_ARI?'넘음':'못 넘음')+'</span></td></tr>').join('')+'</tbody>';}
whenNear('m5',()=>{prepAll().then(renderEval);},500);

/* ── 면접 결과 발표 ── */
function ivPanel(fly,flyMv,robo,roboMv){const el=$('iv11'),fi=el.querySelector('.fly img'),ri=el.querySelector('.robo img');if(fi){fi.src=sprite('fly',fly);react(fi,flyMv);}if(ri){ri.src=sprite('robo',robo);react(ri,roboMv);}}
Interview('iv11',{title:'계층적 군집은 정답 없는 손글씨 묶기에 합격할까?',
  q:'조건: 1,797장 전체를 ward 연결, 원래 픽셀(유클리드 거리)로 합치고 가계도를 10묶음(숫자 열 가지라는 업무 지식)에서 자른다. 채점 봉투를 열어 ARI가 합격선(매뉴얼 씨 8묶음 '+f3(LINE_ARI)+')과 무작위 기준선('+f3(RAND_ARI)+')을 모두 넘으면 합격. 합치기는 브라우저가 직접 한다.',
  fly:'정답 안 보고 묶은 거 맞지? 그리고 10묶음은 누가 정했어? 새 글씨가 오면 어떻게 해?',robo:'161만 쌍 거리표면 장부가 몇 MB야? 합치는 데 얼마나 걸려?',
  async run(setP){setP(.2);await wait(30);ensureD();setP(.5);await wait(30);ensureZ('ward');setP(.8);await wait(20);const lab=labK('ward',KBASE),a=HC.ari(ENV.y,lab),nm=HC.nmi(ENV.y,lab),P=propK('ward',KBASE);setP(1);
    const pass=a>LINE_ARI&&a>RAND_ARI;
    return {result:pass?'pass':'hold',flyAns:'합치기는 X만 썼음 · 10묶음은 업무 지식(숫자 열 가지), 실루엣은 9묶음 근처를 가리킴 · 새 글씨 배정은 predict가 없어 덧붙인 규칙으로(06장)',roboAns:'거리 1,613,706쌍 ≈ 13MB · 거리 '+Math.round(MS.d)+'ms + ward 합치기 '+Math.round(MS.ward)+'ms(이 브라우저) · 장수가 10배면 100배',
      flyMood:pass?'type':'coffee',flyMove:pass?'nod':'shake',roboMood:pass?'happy':'meh',roboMove:pass?'jump':'shake',asstMood:pass?'spark':'sigh',docMood:pass?'magic':'scale',
      measure:'채점 봉투 결과예요. ward 10묶음의 <b>ARI '+f3(a)+'</b>(NMI '+f3(nm)+')로 합격선 '+f3(LINE_ARI)+'보다 '+fx(a-LINE_ARI,3)+' 높고, 무작위로 나눈 점수('+f3(RAND_ARI)+')와는 비교가 안 돼요. 업무 점수로는 대표 10장에만 이름을 붙여 <b>'+pct(P.acc)+'</b>가 맞게 붙습니다. 5-10 K-Means(ARI '+f3(PRE.km.ari)+')보다 높아요.',
      verdict:pass?'합격입니다. 군집팀에 배치합니다. 조건이 둘 있어요. 연결 방식은 ward로 두고 다른 방식으로 바꿀 때는 가장 큰 묶음의 크기부터 볼 것, 새 글씨를 받을 때는 배정 규칙을 따로 정해 함께 적을 것.':'보류입니다. 합격선을 넘지 못했어요.',
      key:'가장 닮은 둘부터 1,796번 — ARI '+f3(LINE_ARI)+' → '+f3(a)+' · 이름 10장으로 '+pct(P.acc),
      render(box){box.innerHTML='<div class="bars">'+barRows([['계층적 군집 (ward)',a,'var(--blue)',f3(a)],['K-Means k=10 (5-10, Python)',PRE.km.ari,'var(--muted)',f3(PRE.km.ari)],['매뉴얼 씨 8묶음 (합격선)',LINE_ARI,'var(--c-man)',f3(LINE_ARI)],['무작위 10묶음',Math.max(0,RAND_ARI),'var(--muted)',f3(RAND_ARI)]],1)+'</div>';}};},
  alts:[
   {label:'single 연결로 바꾸면',async run(box){ensureZ('single');const l=labK('single',KBASE),a=HC.ari(ENV.y,l),c=new Array(KBASE).fill(0);for(let i=0;i<N;i++)c[l[i]]++;const pass=a>LINE_ARI&&a>RAND_ARI;
     ivPanel(pass?'arms':'coffee',pass?'jump':'shake',pass?'happy':'meh',pass?'pop':'shake');
     box.innerHTML=statGrid([['ARI '+f3(a),'single · 10묶음','a'],['ARI '+f3(LINE_ARI),'합격선','m'],[Math.max(...c).toLocaleString()+'장','가장 큰 묶음'],[pct(propK('single',KBASE).acc),'업무 점수 (이름 10장)']])+'<p class="foot">판정: <b>'+(pass?/*i18n:verdict*/'합격':'보류')+'</b>. 같은 지원자, 같은 거리인데 "묶음 사이 = 가장 가까운 한 쌍"으로 바꿨다. 1,797장 중 '+Math.max(...c).toLocaleString()+'장이 한 묶음이고 나머지 아홉 묶음은 낱장 한두 장이다. 글씨 사이를 잇는 징검다리를 따라 사슬처럼 이어진 것이다(04장 ①). 실루엣도 '+f3(silK('single',KBASE))+'로 음수다. 정답 없이도 "잘못 묶였다"는 신호가 있었다는 뜻이다.</p>';}},
   {label:'average 연결로 바꾸면',async run(box){ensureZ('average');const l=labK('average',KBASE),a=HC.ari(ENV.y,l),c=new Array(KBASE).fill(0);for(let i=0;i<N;i++)c[l[i]]++;const pass=a>LINE_ARI&&a>RAND_ARI;
     ivPanel(pass?'type':'coffee',pass?'nod':'shake',pass?'calc':'meh',pass?'tilt':'shake');
     box.innerHTML=statGrid([['ARI '+f3(a),'average · 10묶음','a'],['ARI '+f3(LINE_ARI),'합격선','m'],[c.slice().sort((p,q)=>q-p).slice(0,3).join(' · ')+'…','묶음 크기 (큰 순)'],[pct(propK('average',KBASE).acc),'업무 점수 (이름 10장)']])+'<p class="foot">판정: <b>'+(pass?/*i18n:verdict*/'합격':'보류')+'</b>(합격선은 넘는다). 하지만 ward보다 '+fx(HC.ari(ENV.y,labK('ward',KBASE))-a,3)+' 낮다. 묶음 크기가 '+Math.min(...c)+'장짜리부터 '+Math.max(...c)+'장짜리까지 고르지 않다. 큰 묶음 두 개에 여러 숫자가 함께 들어갔다. 연결 방식은 성적을 바꾸는 근무 조건이다.</p>';}},
   {label:'20묶음으로 자르면',async run(box){const l=labK('ward',20),a=HC.ari(ENV.y,l),P=propK('ward',20),pass=a>LINE_ARI&&a>RAND_ARI;
     ivPanel(pass?'arms':'coffee',pass?'nod':'shake',pass?'wow':'meh',pass?'jump':'shake');
     box.innerHTML=statGrid([['ARI '+f3(a),'ward · 20묶음','a'],['ARI '+f3(HC.ari(ENV.y,labK('ward',KBASE))),'ward · 10묶음 (본 조건)','s'],[pct(P.acc),'업무 점수 (이름 20장)'],[pct(propK('ward',KBASE).acc),'업무 점수 (이름 10장)']])+'<p class="foot">판정: <b>'+(pass?/*i18n:verdict*/'합격':'보류')+'</b>. 같은 가계도를 더 낮게 잘랐다. 한 숫자가 두세 묶음으로 나뉘니 ARI는 내려가지만, 묶음 하나하나는 더 깨끗해져서 대표 20장에 이름을 붙이면 '+pct(P.acc)+'가 맞는다. 채점 점수(ARI)와 업무 점수가 다른 방향으로 움직인다. 이름 붙일 사람의 시간과 맞는 비율을 맞바꾸는 선택이다. 나무는 다시 키우지 않았다.</p>';}}]});

/* ── 06 인사 카드: 그림판 · 중간 비교 ── */
const PAD=(function(){const cv=$('padCv'),oc=$('padOut'),pad=new Array(64).fill(0);let pen=16,down=false,cent=null,R=null,sampleI=0;const RS=RNG(511);
  function prep(){if(cent)return;const lab=labK('ward',KBASE);R=HC.reps(X,lab);cent=R.mu;}
  function draw(){const W=fit(cv,Math.min(300,cv.clientWidth||300)),ctx=cv.getContext('2d'),C=cols(),s=Math.min(W,300),c=s/8;ctx.clearRect(0,0,W,s);
    for(let j=0;j<64;j++){const x=(j%8)*c,y=Math.floor(j/8)*c;ctx.fillStyle=C.panel2;ctx.fillRect(x,y,c-1,c-1);ctx.fillStyle=rgba(C.text,pad[j]/16);ctx.fillRect(x,y,c-1,c-1);}}
  function paint(e){const r=cv.getBoundingClientRect(),c=Math.min(r.width,300)/8,cx=Math.floor((e.clientX-r.left)/c),cy=Math.floor((e.clientY-r.top)/c);if(cx<0||cx>7||cy<0||cy>7)return;const j=cy*8+cx;
    if(pen){pad[j]=16;[[1,0],[-1,0],[0,1],[0,-1]].forEach(([dx,dy])=>{const x=cx+dx,y=cy+dy;if(x>=0&&x<8&&y>=0&&y<8)pad[y*8+x]=Math.max(pad[y*8+x],6);});}else pad[j]=0;draw();assign();}
  function assign(){const W=fit(oc,190),ctx=oc.getContext('2d'),C=cols();ctx.clearRect(0,0,W,190);const ink=pad.reduce((a,b)=>a+b,0);
    if(!ink){$('padInfo').innerHTML='왼쪽 칸에 숫자를 그려 보자. 칸을 누르면 진해지고 옆 칸이 조금 번진다. "과제지에서 한 장 불러오기"로 실제 글씨를 넣어 봐도 된다.';return;}
    if(!cent){$('padInfo').textContent='04장 계산이 끝나면 배정한다.';return;}
    const d=cent.map((m,c)=>{let s=0;for(let j=0;j<64;j++){const v=pad[j]-m[j];s+=v*v;}return [Math.sqrt(s),c];}).sort((a,b)=>a[0]-b[0]);
    let nn=0,nd=Infinity;for(let i=0;i<N;i++){let s=0;for(let j=0;j<64;j++){const v=pad[j]-X[i][j];s+=v*v;}if(s<nd){nd=s;nn=i;}}
    const t=Math.min(70,(W-40)/4.6);ctx.font=font(12,700);ctx.textAlign='center';
    d.slice(0,3).forEach((q,i)=>{const x=10+i*(t+16);digitTile(ctx,C,cent[q[1]],x,20,t,i===0?C.blue:null);ctx.fillStyle=i===0?C.blue:C.muted;ctx.fillText((i+1)+'위 · "'+ENV.y[R.rep[q[1]]]+'"',x+t/2,14);ctx.font=font(11,500,1);ctx.fillText('거리 '+q[0].toFixed(1),x+t/2,t+36);ctx.font=font(12,700);});
    const x=10+3*(t+16);digitTile(ctx,C,X[nn],x,20,t,C.amber);ctx.fillStyle=C.amber;ctx.fillText('가장 닮은 한 장',x+t/2,14);ctx.font=font(11,500,1);ctx.fillText('#'+(nn+1)+' · 거리 '+Math.sqrt(nd).toFixed(1),x+t/2,t+36);
    $('padInfo').innerHTML='가장 가까운 묶음 평균(파란 테두리)은 대표에게 <b>"'+ENV.y[R.rep[d[0][1]]]+'"</b>라는 이름이 붙은 묶음이다(거리 '+d[0][0].toFixed(1)+'). 2위와의 거리 차이는 '+(d[1][0]-d[0][0]).toFixed(1)+'. 차이가 작으면 애매한 글씨다. 1,797장 중 가장 닮은 한 장(주황)은 #'+(nn+1)+'이다.<br><span class="muted">이 배정은 계층적 군집의 기능이 아니다. 가계도를 10묶음에서 자른 뒤 묶음 평균 10개를 따로 저장해 두고 "가장 가까운 평균"에 붙이는, 우리가 덧붙인 규칙(K-Means의 배정과 같다)이다. A장 10번 칸: 앞 1,500장으로 묶고 뒤 297장을 이렇게 배정하면 ARI '+f3(PRE.split.assigned)+'(1,500장 자체는 '+f3(PRE.split.fit)+').</span>';}
  cv.addEventListener('pointerdown',e=>{down=true;try{cv.setPointerCapture(e.pointerId);}catch(_){}paint(e);});cv.addEventListener('pointermove',e=>{if(down)paint(e);});['pointerup','pointercancel','pointerleave'].forEach(t=>cv.addEventListener(t,()=>{down=false;}));
  $('padClear').addEventListener('click',()=>{pad.fill(0);draw();assign();});
  $('padSample').addEventListener('click',()=>{sampleI=Math.floor(RS()*N);for(let j=0;j<64;j++)pad[j]=X[sampleI][j];draw();assign();});
  seg('#padPen button',v=>{pen=+v;});
  draw();assign();onRedraw(()=>{draw();assign();});return {start(){prep();assign();}};})();
function renderCard(){PAD.start();const lab=labK('ward',KBASE),ls=labK('single',KBASE);
  const rowsL=[['5-0','매뉴얼 씨','규칙 3줄 → 8묶음',LINE_ARI,LINE_NMI,HC.silhouette(ensureD(),N,Int32Array.from(MAN)),HC.propagate(X,MAN,ENV.y).acc,8,'line','브라우저'],
    ['5-10','K-Means','k = 10, 원래 픽셀',PRE.km.ari,PRE.km.nmi,PRE.km.sil,PRE.km.prop,10,'ref','Python'],
    ['5-11','계층적 군집','ward · 10묶음 (본 조건)',HC.ari(ENV.y,lab),HC.nmi(ENV.y,lab),silK('ward',KBASE),propK('ward',KBASE).acc,10,'me','브라우저'],
    ['5-11','계층적 군집 (참고)','single · 10묶음',HC.ari(ENV.y,ls),HC.nmi(ENV.y,ls),silK('single',KBASE),propK('single',KBASE).acc,10,'alt','브라우저'],
    ['—','무작위','10묶음 20번 평균',RAND_ARI,null,null,null,null,'base','브라우저']];
  const st=r=>{if(r[8]==='line')return '<span class="pill">합격선</span>';if(r[8]==='base')return '<span class="pill">기준선</span>';const p=r[3]>LINE_ARI&&r[3]>RAND_ARI;return '<span class="pill '+(p?'ok':'no')+'">'+(p?(r[8]==='ref'?'합격선 넘음':'합격'):'보류')+'</span>';};
  $('cluLedger').innerHTML='<thead><tr><th>편</th><th>지원자</th><th>조건</th><th>ARI</th><th>NMI</th><th>실루엣</th><th>업무 점수 (이름 장수)</th><th>계산</th><th>결과</th></tr></thead><tbody>'+
    rowsL.map(r=>'<tr'+(r[8]==='me'?' style="background:var(--panel2)"':'')+'><td>'+r[0]+'</td><td><b style="color:var(--text)">'+r[1]+'</b></td><td style="font-size:13px">'+r[2]+'</td><td>'+f3(r[3])+'</td><td>'+(r[4]==null?'—':f3(r[4]))+'</td><td>'+(r[5]==null?'—':f3(r[5]))+'</td><td>'+(r[6]==null?'—':pct(r[6])+' ('+r[7]+'장)')+'</td><td style="font-size:12.5px">'+r[9]+'</td><td>'+st(r)+'</td></tr>').join('')+'</tbody>';
  $('cluFoot').innerHTML='5-10 K-Means 줄은 Python(scikit-learn 1.9.1, n_init=10, random_state=0)이 미리 계산한 값이고(A장 11번 칸), 판정은 5-10 면접에서 한다. 나머지 줄은 이 브라우저 계산이다. 표준화한 뒤 ward로 묶으면 ARI '+f3(PRE.wardStd)+'(A장 11번 칸)로 원래 픽셀보다 낮다. 같은 10묶음이라도 ward가 K-Means보다 높은 것은 이 데이터의 성질이다. 실루엣은 K-Means가 조금 높다. K-Means는 실루엣이 좋아하는 "둥근 묶음"을 직접 만들기 때문이다. 정답 없이 재는 점수 하나로 지원자를 줄 세우지 않는 까닭이다.';}
whenNear('m6',()=>{prepAll().then(renderCard);},500);

/* ══ 확인 문제 ══ */
var QUIZ={
q1:[{q:'병합(agglomerative) 계층적 군집의 첫 상태와 마지막 상태는?',o:[['처음엔 한 장이 한 묶음, 끝에는 모두가 한 묶음',1,'n장이면 n−1번 합친다. 그 기록 전체가 가계도다.'],['처음엔 k개의 중심, 끝에는 중심이 멈춘 상태',0,'그것은 5-10 K-Means다.'],['처음엔 모두 한 묶음, 끝에는 한 장이 한 묶음',0,'그것은 위에서 쪼개 내려가는 분할(divisive) 군집이다.']]},
 {q:'계층적 군집에서 묶음 수는 언제 정하나?',o:[['다 합친 뒤, 가계도를 자르는 높이로 정한다',1,'나무는 한 번만 키우고, 가로선을 옮겨 2묶음이든 30묶음이든 바로 얻는다.'],['합치기 전에 반드시 정해야 한다',0,'scikit-learn이 n_clusters를 받기는 하지만, 그것은 다 합친 나무를 어디서 자를지 알려 주는 것이다.'],['정할 수 없다',0,'자르는 높이를 고르면 된다.']]},
 {q:'01장에서 한 번 합친 두 묶음을 나중에 떼어 낼 수 있나?',o:[['없다. 욕심쟁이(greedy) 방식이라 되돌리지 않는다',1,'그래서 초반의 잘못된 합치기가 끝까지 따라간다.'],['더 좋은 짝이 나타나면 자동으로 떼어 낸다',0,'그런 단계가 없다. K-Means는 배정을 되풀이해 고치지만 계층적 군집은 아니다.'],['ward일 때만 떼어 낸다',0,'어떤 연결 방식도 되돌리지 않는다.']]}],
q2:[{q:'이 면접에서 정답(0~9)을 쓰는 곳은?',o:[['채점(ARI·NMI·업무 점수)에서만',1,'합치기는 64칸 픽셀 X로 만든 거리만 쓴다. 정답은 채점 봉투에 넣어 둔다.'],['거리를 잴 때',0,'거리는 픽셀만으로 잰다.'],['묶음 수를 고를 때',0,'정답으로 묶음 수를 고르면 정답을 엿본 것이다. 업무 지식·높이의 도약·실루엣으로 고른다.']]},
 {q:'군집 결과를 accuracy_score(y, labels)로 채점하면 안 되는 까닭은?',o:[['묶음 번호에는 뜻이 없어서, 완벽히 묶어도 번호가 어긋나면 점수가 낮게 나온다',1,'ARI는 "같이 묶였나"만 보므로 번호를 바꿔 붙여도 같은 점수를 준다.'],['계산이 느려서',0,'속도 문제가 아니다.'],['군집은 정확도가 늘 100%라서',0,'그렇지 않다.']]},
 {q:'매뉴얼 씨의 8묶음 ARI 0.108의 뜻은?',o:[['무작위(≈ 0)보다는 정답과 겹치지만, 많이 겹치지는 않는다',1,'규칙 세 줄은 잉크 양과 위아래 무게처럼 숫자를 가르기 어려운 특징만 본다. 업무 점수로는 이름 8장으로 25.6%.'],['10.8%를 맞혔다',0,'ARI는 정확도가 아니다. 무작위 0, 완벽 1로 맞춘 겹침 점수다.'],['정답과 거의 같다',0,'1에 가까워야 그렇다.']]}],
q3:[{q:'AgglomerativeClustering(linkage="ward", affinity="cosine")을 지금 버전에서 실행하면?',o:[['affinity는 지워진 이름이라 TypeError. metric으로 고쳐도 ward는 유클리드만 받아 ValueError',1,'1.2에서 예고, 1.4에서 지워졌다. cosine을 쓰려면 average·complete 연결로 바꾼다.'],['코사인 거리로 ward가 잘 돈다',0,'ward는 유클리드 거리의 흩어짐(제곱 거리 합)을 쓰므로 다른 거리와 맞지 않는다.'],['경고만 뜨고 유클리드로 돈다',0,'오류가 난다.']]},
 {q:'model.predict(new_X)를 부르면?',o:[['AttributeError. 계층적 군집에는 predict가 없다',1,'가계도는 가진 데이터로만 만든다. 새 데이터 배정 규칙(가까운 묶음 평균 등)은 따로 정한다.'],['가장 가까운 갈래를 돌려준다',0,'그런 기능이 없다.'],['다시 합쳐서 돌려준다',0,'그런 기능이 없다. fit_predict는 X 전체를 새로 합친다.']]},
 {q:'model.distances_가 없다는 오류가 난다. 고치는 법은?',o:[['compute_distances=True를 주거나 distance_threshold를 쓴다. 또는 scipy linkage로 높이를 얻는다',1,'0.24부터 compute_distances가 있다. 덴드로그램은 scipy가 쉽다.'],['n_clusters를 크게 한다',0,'묶음 수와 상관없다.'],['표준화한다',0,'상관없다.']]}],
q4:[{q:'single 연결로 10묶음을 만들었더니 한 묶음에 1,788장이 들어갔다. 까닭은?',o:[['가장 가까운 한 쌍만 있으면 합치므로, 징검다리처럼 이어진 글씨를 따라 큰 묶음이 낱장을 계속 삼킨다(사슬 효과)',1,'04장 ①의 single 선: 묶음이 100개 남았을 때 이미 한 묶음이 대부분이다.'],['거리 계산이 틀려서',0,'거리는 같다. 묶음 사이 거리의 정의가 다를 뿐이다.'],['10묶음이 너무 적어서',0,'30묶음으로 잘라도 큰 묶음 하나는 그대로다.']]},
 {q:'ward 연결이 합치는 두 묶음은?',o:[['합쳤을 때 묶음 안 제곱 거리 합(흩어짐)이 가장 적게 느는 두 묶음',1,'Δ = nAnB/(nA+nB)·‖μA−μB‖². 큰 묶음끼리 합치면 비싸져 크기가 고른 묶음이 남는다.'],['평균끼리 가장 가까운 두 묶음',0,'평균 사이 거리만 보면 centroid 연결이다. ward는 크기까지 곱한다.'],['가장 먼 두 글씨가 가장 가까운 두 묶음',0,'그것은 complete 연결이다.']]},
 {q:'1,797장의 계층적 군집이 미리 재 두는 거리는 몇 쌍인가? 10만 장이면?',o:[['1,613,706쌍 · 약 50억 쌍',1,'n(n−1)/2. 장수가 10배면 쌍은 100배다. 거리표만 수십 GB가 된다.'],['1,797쌍 · 10만 쌍',0,'장수만큼이 아니라 장수의 제곱에 비례한다.'],['64쌍 · 64쌍',0,'64는 특징(픽셀) 수다.']]}],
q5:[{q:'ward 가계도에서 실루엣은 9묶음, ARI는 12묶음에서 가장 높았다. 근무 조건으로 12묶음을 고르면?',o:[['정답을 엿보고 고른 것이 된다. 묶음 수는 정답 없이(업무 지식·실루엣) 정하고 채점 봉투는 확인에만 쓴다',1,'05장의 위원회는 업무 지식으로 10을 정했고, 실루엣도 9~10 근처를 가리켰다.'],['ARI가 가장 높으니 당연히 12다',0,'현장에는 채점 봉투가 없다. 정답이 있으면 애초에 지도학습을 한다.'],['상관없다',0,'평가가 부풀려진다.']]},
 {q:'채점 봉투를 열었더니 숫자 3과 9가 한 묶음에 섞였다. 맞는 해석은?',o:[['8×8 칸에서는 두 숫자의 픽셀 모양이 많이 겹쳐 거리가 가깝다',1,'지원자는 숫자를 모른다. 64칸이 닮은 글씨를 묶었을 뿐이다.'],['지원자가 3과 9를 같은 숫자로 안다',0,'지원자는 숫자의 뜻을 모른다.'],['정답이 틀렸다',0,'정답은 맞다. 픽셀 거리로는 가깝다는 뜻이다.']]},
 {q:'"묶음마다 대표 한 장에만 이름 붙이기"에서 묶음 수를 10 → 20으로 늘리면?',o:[['사람이 붙일 이름은 두 배가 되지만 맞게 붙는 비율은 오른다(84.9% → 94.5%)',1,'묶음이 작아질수록 깨끗해진다. 대신 ARI는 0.794 → 0.669로 내려간다. 업무 점수와 채점 점수는 다른 질문에 답한다.'],['ARI와 업무 점수가 함께 오른다',0,'ARI는 한 숫자가 여러 묶음으로 쪼개지면 내려간다.'],['아무것도 바뀌지 않는다',0,'05장 그래프를 보자.']]}],
q6:[{q:'K-Means(5-10)와 비교한 계층적 군집의 장점은?',o:[['묶음 수를 나중에 정하고, 같은 데이터면 늘 같은 가계도가 나온다',1,'무작위 시작점이 없다. 묶음 사이의 친척 관계도 보인다.'],['큰 데이터에서 더 빠르다',0,'거꾸로다. 거리표가 장수의 제곱이다.'],['새 데이터를 바로 배정한다',0,'predict가 없다. K-Means는 가장 가까운 중심으로 바로 배정한다.']]},
 {q:'10만 장을 계층적으로 묶고 싶다. 현실적인 방법은?',o:[['K-Means 등으로 먼저 수백 묶음으로 줄이고 그 중심들로 가계도를 그리거나, connectivity로 이웃끼리만 합친다',1,'가계도의 장점을 큰 데이터에서도 쓰는 흔한 2단계다.'],['그냥 돌린다',0,'거리표만 약 40GB다.'],['single 연결로 바꾼다',0,'single은 최소 신장 나무로 빨리 돌릴 수 있지만 사슬 효과가 심하다.']]}]};

/* ══ 공통 B·C·D·E·F 데이터 ══ */
var PROMPTS=[
 {t:'① 내 데이터로 계층적 군집 코드 받기',use:'정답 없는 내 데이터(서식 스캔, 센서 패턴, 불량 사진)로 03장의 프롬프트를 맞출 때',p:MAIN_PROMPT},
 {t:'② AI 코드 검증 요청',use:'받은 코드를 다른 AI(또는 같은 AI의 새 대화)에게 검토받을 때',p:'아래 scikit-learn·scipy 계층적 군집 코드를 코드 리뷰해 줘. 특히 다음을 하나씩 확인하고, 문제가 있으면 줄 번호와 고친 코드를 보여 줘.\n1) 정답 y가 fit/fit_predict에 들어가지 않았나 (y는 채점에만)\n2) affinity 같은 지워진 인자 이름을 쓰지 않았나, ward에 유클리드가 아닌 metric을 주지 않았나 (내 버전: 〔1.x〕)\n3) 채점을 accuracy_score(y, labels)로 하지 않았나 (ARI·NMI를 써야 한다)\n4) 묶음 수나 연결 방식을 정답 점수(ARI)로 고르지 않았나\n5) 표준화를 넣었다면, 넣을 때와 뺄 때를 비교했나\n6) 없는 predict나 compute_distances 없이 distances_를 쓰지 않았나\n7) 연결 방식별 가장 큰 묶음의 크기를 출력해 사슬 효과를 확인했나\n추측으로 답하지 말고, 확실하지 않은 부분은 "공식 문서 확인 필요"라고 표시해 줘.'},
 {t:'③ 큰 데이터로 넓히기',use:'데이터가 수만 건 이상이라 거리표를 만들 수 없을 때',p:'〔약 5만 건, 특징 〔30〕개〕의 정답 없는 데이터를 계층적으로 묶고 싶다. Colab 메모리(약 12GB) 안에서 돌도록 다음 세 방법을 각각 셀로 짜고 비교해 줘.\n1) 무작위 표본 〔3,000〕건으로 AgglomerativeClustering(ward) → 나머지는 가장 가까운 묶음 평균에 배정\n2) MiniBatchKMeans로 〔300〕묶음 → 그 중심 300개로 ward 가계도 → 자른 결과를 원래 데이터에 되돌려 붙이기\n3) kneighbors_graph(n_neighbors=〔10〕)를 connectivity로 준 AgglomerativeClustering\n규칙:\n- 각 방법의 메모리 사용 추정(거리표 크기)과 실행 시간을 출력\n- 정답이 있다면 채점 칸에서만 ARI를 내고, 없으면 실루엣(표본 5,000건)만 낸다\n- 세 방법의 묶음이 서로 얼마나 같은지 ARI로 비교\n- scipy·scikit-learn 인자 이름은 설치된 버전의 공식 문서 기준으로 쓴다'},
 {t:'④ 수업용 문제 만들기',use:'이 편으로 수업할 때 확인 문제가 더 필요할 때',p:'계층적 군집 입문 수업용 문제를 만들어 줘. 대상은 〔제조 현장 실무자, 수식에 익숙하지 않음〕이다. 예시는 손글씨 숫자나 공장 서식으로 통일한다.\n- 병합 순서·연결 방식·자르는 높이·사슬 효과를 각각 묻는 객관식 4문제 (보기 3개, 정답 해설 포함)\n- 오해를 드러내는 문제 2개: "군집 결과는 정확도로 채점한다", "계층적 군집도 새 데이터를 predict 한다"\n- 손으로 풀 수 있는 계산 문제 1개: 1차원 점 다섯 개(1, 2, 4, 8, 9)를 single 연결과 complete 연결로 각각 합치는 순서와 높이를 표로 쓰고 덴드로그램을 그리기\n정답은 맨 끝에 따로 모아 줘.'}];
var GL=[
 ['모델','계층적 군집','데이터를 한 줄의 가계도(계층)로 묶는 방법. 아래에서 위로 합치는 병합형이 흔하다. 묶음 수는 가계도를 자르는 높이로 나중에 정한다.'],
 ['모델','병합 군집','한 장이 한 묶음인 상태에서 시작해 가장 가까운 두 묶음을 합치기를 되풀이하는 계층적 군집(agglomerative). scikit-learn AgglomerativeClustering.'],
 ['모델','분할 군집','모두 한 묶음인 상태에서 시작해 둘로 쪼개기를 되풀이하는 계층적 군집(divisive).'],
 ['모델','덴드로그램','합친 순서와 높이를 나무 모양으로 그린 그림. 두 갈래가 만나는 높이가 합칠 때의 거리다. 가계도.'],
 ['모델','연결 방식','묶음과 묶음 사이 거리를 재는 규칙(linkage). single(가장 가까운 쌍), complete(가장 먼 쌍), average(모든 쌍의 평균), ward(합친 뒤 늘어나는 흩어짐).'],
 ['모델','와드 연결','두 묶음을 합쳤을 때 묶음 안 제곱 거리 합이 가장 적게 느는 쌍을 합치는 연결 방식(Ward, 1963). 크기가 고른 묶음을 만든다.'],
 ['모델','사슬 효과','single 연결에서, 가장 가까운 한 쌍만 있으면 합치므로 징검다리처럼 이어진 데이터가 모두 한 묶음이 되는 현상(chaining).'],
 ['모델','K-Means','묶음 수 k를 정하고 중심 k개를 옮겨 가며 묶는 군집 방법. 5-10 면접.'],
 ['학습','군집','정답 없이 비슷한 것끼리 묶는 일(clustering). 비지도 학습의 한 갈래.'],
 ['학습','비지도 학습','정답(이름표) 없이 데이터의 구조를 찾는 학습. 군집과 차원 축소가 대표적이다.'],
 ['학습','유클리드 거리','두 점의 특징 차이를 제곱해 더하고 제곱근을 씌운 직선거리. 이 면접에서는 64칸 픽셀 사이의 거리.'],
 ['학습','거리표','모든 두 샘플 사이 거리를 적은 표. n장이면 n(n−1)/2쌍. 계층적 군집의 메모리 비용이다.'],
 ['학습','Lance-Williams 식','두 묶음을 합친 뒤, 새 묶음과 나머지 묶음 사이 거리를 옛 거리들로 바로 계산하는 한 줄 식. 연결 방식마다 계수만 다르다.'],
 ['학습','NN-chain','가장 가까운 이웃을 사슬처럼 따라가다 서로 가장 가까운 둘을 만나면 합치는 순서. 거리표 크기(n²)의 시간으로 끝난다.'],
 ['학습','자르는 높이','가계도에 그은 가로선의 높이. 선에 걸린 갈래 수가 묶음 수다. scikit-learn distance_threshold.'],
 ['학습','표준화','특징마다 평균을 빼고 표준편차로 나누는 일. 단위가 섞인 데이터에서는 돕지만, 같은 단위의 픽셀에서는 오히려 해로울 수 있다.'],
 ['평가','채점 봉투','이 시리즈 비지도 본부에서 숨긴 정답(0~9)을 부르는 말. 학습에는 쓰지 않고 채점에만 연다.'],
 ['평가','ARI','수정 랜드 지수(Adjusted Rand Index). 두 나누기가 "같이 묶인 쌍"에서 얼마나 일치하는지를, 무작위면 0·완벽하면 1이 되게 맞춘 점수. 묶음 번호와 상관없다.'],
 ['평가','NMI','정규화 상호 정보량. 묶음을 알면 정답을 얼마나 알 수 있는지를 0~1로 잰 점수.'],
 ['평가','실루엣 점수','정답 없이 재는 점수. 글씨마다 (가장 가까운 다른 묶음까지 평균 거리 − 내 묶음 안 평균 거리) ÷ 둘 중 큰 값. 1에 가까울수록 잘 갈렸다.'],
 ['평가','업무 점수','이 시리즈의 비지도 본부 점수. 묶음마다 대표 한 장에만 사람이 이름을 붙여 묶음 전체에 퍼뜨렸을 때 맞게 붙은 비율. 라벨링 비용 절감의 크기.'],
 ['평가','순도','묶음마다 가장 많은 정답의 장수를 모두 더해 전체로 나눈 값. 업무 점수의 위쪽 한계에 가깝다.'],
 ['평가','코페네틱 상관','가계도에서 두 글씨가 만나는 높이와 원래 거리 사이의 상관. 가계도가 원래 거리를 얼마나 닮았나.'],
 ['평가','합격선','지원자가 넘어야 할 점수. 군집팀은 매뉴얼 씨 8묶음의 ARI 0.108.'],
 ['평가','기준선','배우지 않고도 낼 수 있는 점수. 군집에서는 무작위로 나눈 묶음의 ARI(≈ 0).']];
var ANALOGY=[
 ['가장 가까운 둘부터 합친다','책상 위 서류 더미에서 가장 닮은 두 장을 클립으로 묶고, 묶음끼리 또 묶어 가는 서류 정리',2,'원숭이·사람의 하측두 피질 반응 패턴의 비슷함을 덴드로그램으로 그리면 생물/무생물, 얼굴/몸으로 갈래가 지어진다(Kriegeskorte 등, 2008)','그것은 연구자가 뇌 반응에 계층적 군집을 적용해 본 분석 결과다. 뇌가 실제로 "가장 가까운 둘부터 합치기"를 한다는 근거는 없다'],
 ['덴드로그램 (가계도)','족보·조직도: 아래로 갈수록 가까운 사이, 위로 갈수록 먼 사이',2,'사람의 범주는 "동물 > 개 > 진돗개"처럼 층을 이루고, 그중 "개" 같은 기본 수준을 가장 먼저 떠올린다(Rosch 등, 1976)','사람의 범주 층은 쓰임새와 문화가 정한다. 높이(거리)라는 숫자가 없고, 한 물건이 여러 층에 동시에 속하기도 한다'],
 ['비슷함을 잰다 (초파리)','냄새가 비슷한 두 와인을 같은 칸에 꽂는 소믈리에',2,'초파리 버섯체는 비슷한 냄새에 비슷한 케니언 세포 꼬리표를 붙여 "닮은 것 찾기"를 한다는 해석(Dasgupta, Stevens, Navlakha, 2017)','초파리 회로는 이웃 찾기에 가깝고, 합친 기록(가계도)을 남기지 않는다. 계층적 군집과는 일하는 방식이 다르다'],
 ['연결 방식','두 반이 얼마나 가까운지를 "가장 친한 두 사람"으로 볼지, "가장 먼 두 사람"으로 볼지, "평균"으로 볼지 정하기',1,'','대응하는 뇌 기능은 알려지지 않았다'],
 ['사슬 효과 (single)','친구의 친구의 친구도 친구라 치면, 반 전체가 한 무리가 된다',1,'','대응하는 뇌 기능은 알려지지 않았다. 사람은 친구 관계를 한 쌍만 보고 판단하지 않는다'],
 ['예측이 없다','회의록(가계도)은 남았지만, 내일 들어올 신입을 어느 팀에 보낼지는 따로 정해야 한다',1,'','대응하는 뇌 기능은 알려지지 않았다. 뇌는 새 경험을 기존 범주에 늘 바로 붙인다']];
var REFS=[
 'Alpaydin, E., &amp; Kaynak, C. (1998). Optical Recognition of Handwritten Digits [Dataset]. UCI Machine Learning Repository. <a href="https://doi.org/10.24432/C50P49" target="_blank" rel="noopener">doi:10.24432/C50P49</a> (CC BY 4.0) — 실기 과제지(scikit-learn load_digits는 이 데이터의 8×8 검증 세트)',
 'Ward, J. H., Jr. (1963). Hierarchical grouping to optimize an objective function. <i>Journal of the American Statistical Association</i>, 58(301), 236–244. — 와드 연결, 00·04장',
 'Sneath, P. H. A. (1957). The application of computers to taxonomy. <i>Journal of General Microbiology</i>, 17(1), 201–226. — single 연결(최단 거리법)',
 'Sokal, R. R., &amp; Michener, C. D. (1958). A statistical method for evaluating systematic relationships. <i>University of Kansas Science Bulletin</i>, 38, 1409–1438. — average 연결(UPGMA)',
 'Lance, G. N., &amp; Williams, W. T. (1967). A general theory of classificatory sorting strategies: 1. Hierarchical systems. <i>The Computer Journal</i>, 9(4), 373–380. — 합친 뒤 거리 고쳐 쓰기, 04장',
 'Gower, J. C., &amp; Ross, G. J. S. (1969). Minimum spanning trees and single linkage cluster analysis. <i>Journal of the Royal Statistical Society: Series C (Applied Statistics)</i>, 18(1), 54–64. — single 연결과 최소 신장 나무',
 'Murtagh, F. (1983). A survey of recent advances in hierarchical clustering algorithms. <i>The Computer Journal</i>, 26(4), 354–359. — NN-chain, 04장 브라우저 계산',
 'Müllner, D. (2011). Modern hierarchical, agglomerative clustering algorithms. <i>arXiv:1109.2378</i>. — scipy linkage의 알고리즘, 04장',
 'Hubert, L., &amp; Arabie, P. (1985). Comparing partitions. <i>Journal of Classification</i>, 2(1), 193–218. — ARI, 05장',
 'Strehl, A., &amp; Ghosh, J. (2002). Cluster ensembles — A knowledge reuse framework for combining multiple partitions. <i>Journal of Machine Learning Research</i>, 3, 583–617. — NMI, 05장',
 'Rousseeuw, P. J. (1987). Silhouettes: A graphical aid to the interpretation and validation of cluster analysis. <i>Journal of Computational and Applied Mathematics</i>, 20, 53–65. — 실루엣 점수, 04·05장',
 'Hastie, T., Tibshirani, R., &amp; Friedman, J. (2009). <i>The Elements of Statistical Learning</i> (2nd ed.), 14.3.12 Hierarchical Clustering. Springer. — 01·04장',
 'Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. <i>Journal of Machine Learning Research</i>, 12, 2825–2830. · Virtanen, P. et al. (2020). SciPy 1.0: Fundamental algorithms for scientific computing in Python. <i>Nature Methods</i>, 17, 261–272. — A장 코드',
 'scikit-learn 문서. <a href="https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering" target="_blank" rel="noopener">Clustering: Hierarchical clustering</a> · <a href="https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html" target="_blank" rel="noopener">AgglomerativeClustering</a> — metric·compute_distances, 03·06장',
 'Kriegeskorte, N., Mur, M., Ruff, D. A., Kiani, R., Bodurka, J., Esteky, H., Tanaka, K., &amp; Bandettini, P. A. (2008). Matching categorical object representations in inferior temporal cortex of man and monkey. <i>Neuron</i>, 60(6), 1126–1141. — D장',
 'Rosch, E., Mervis, C. B., Gray, W. D., Johnson, D. M., &amp; Boyes-Braem, P. (1976). Basic objects in natural categories. <i>Cognitive Psychology</i>, 8(3), 382–439. — D장',
 'Dasgupta, S., Stevens, C. F., &amp; Navlakha, S. (2017). A neural algorithm for a fundamental computing problem. <i>Science</i>, 358(6364), 793–796. — D장 초파리'];
var TIP={who:'머신러닝을 처음 배우는 대학생·직장인·제조 현장 실무자. 5-1 KNN(거리)을 먼저 보면 좋고, 5-10 K-Means와 나란히 보면 더 좋다. 수식은 와드의 Δ 하나와 ARI의 뜻(무작위 0, 완벽 1)이 중심이다',
 goal:'① "가장 가까운 둘부터 합치기"와 가계도를 01장 그림으로 설명한다 ② 연결 방식이 묶음 사이 거리의 정의이며, single의 사슬 효과가 왜 생기는지 04장 그림으로 보인다 ③ 자르는 높이 = 묶음 수이고, 묶음 수를 정답 없이 고르는 세 근거를 말한다 ④ 채점 봉투(ARI·NMI)와 업무 점수(대표 한 장 이름 붙이기)를 구별하고, AI가 낸 군집 코드의 채점 실수를 찾는다',
 time:'읽기 25분 · 실습(01·04·05장 그림) 25분 · 틀린 곳 찾기 10분 · Colab 15분',
 plans:[['강의형 50분','00장 지원서 1장을 인쇄해 1·5·6번 칸 함께 읽기 8분 → 01장 그림에서 스무 장을 한 번씩 합치며 "다음엔 어떤 둘이 합쳐질까" 맞히기 10분(ward와 single을 번갈아) → 04장 ①에서 single 선이 치솟는 것을 보고 사슬 효과 토론 10분 → 05장 채점 봉투를 열기 전에 "3과 9 중 무엇이 섞일까" 예상, 열어 보기 10분 → 면접 결과를 먼저 맞히고 발표, "single로 바꾸면" 다시 보기 12분.'],['실무 OJT 90분','03장 프롬프트로 각자 AI에게 코드를 받는다 20분 → 체크리스트로 서로의 코드를 바꿔 검사(특히 채점 방법과 정답 사용 위치) 20분 → Colab에서 실행하고 A장 결과와 비교 20분 → B장 ③ 프롬프트로 "내 데이터가 5만 건이면" 세 방법을 비교하고 업무 점수(대표 몇 장에 이름 붙이면 되나)로 라벨링 비용을 계산 30분.'],['1:1 튜터링','01장에서 합치기를 한 번 누를 때마다 "왜 이 둘인가, 높이는 무엇인가"를 말로 설명하게 한다. 이어 05장 업무 점수 그래프에서 "이름을 몇 장 붙일지"를 학습자의 현장 서식 수로 바꿔 계산해 보게 한다.']],
 myths:[['군집 결과도 정확도로 채점한다','묶음 번호에는 뜻이 없다. ARI·NMI로 채점하고, 현장에서는 대표에 이름을 붙여 업무 점수로 본다'],['계층적 군집은 묶음 수가 필요 없다','나무를 만들 때는 필요 없지만 쓰려면 결국 자르는 높이(묶음 수)를 정해야 한다'],['연결 방식은 세부 설정일 뿐이다','같은 데이터에서 ward 0.794, single 0.000. 성적을 가장 크게 바꾸는 근무 조건이다'],['거리 계산 전에는 늘 표준화한다','같은 단위의 픽셀에서는 표준화가 해롭다(0.794 → 0.664). 단위가 섞였을 때 검토한다'],['실루엣이 가장 높은 묶음 수가 정답이다','실루엣은 9, ARI는 12를 가리켰다. 여러 근거(업무 지식 포함)를 함께 본다'],['계층적 군집으로 새 데이터를 예측한다','predict가 없다. 배정 규칙을 덧붙이거나 다시 합친다']],
 tasks:[['과제 1 · 손으로 합치기','1차원 점 다섯 개(1, 2, 4, 8, 9)를 single과 complete로 각각 합치는 순서와 높이를 표로 쓰고 덴드로그램을 그린다. 두 나무가 어디서 달라지는지 한 줄로 쓴다.'],['과제 2 · 틀린 곳 찾기 만들기','AI에게 일부러 실수가 섞인 군집 코드를 받고(B장 ②를 거꾸로 활용), 짝에게 찾게 한다. 실수마다 "왜 틀렸나"를 한 줄로 쓴다. 채점 실수를 하나 이상 넣는다.'],['과제 3 · 라벨링 비용 계산','A장 9번 칸을 고쳐 묶음 수 5~60에서 업무 점수를 표로 만들고, "90%를 넘기려면 대표 몇 장에 이름을 붙여야 하나"를 찾는다. 같은 계산을 5-10 K-Means로 해 비교한다.'],['과제 4 · 큰 데이터 2단계','B장 ③ 프롬프트로 K-Means 300묶음 → ward 가계도 2단계를 짜고, 1,797장 전체 ward와 결과가 얼마나 같은지 ARI로 비교한다.']],
 ai:'학생이 AI에게 코드를 받는 것은 막지 말고 과정에 넣는다. 군집은 코드가 오류 없이 돌아가도 채점이 틀리기 쉬운 주제라 검증 연습에 특히 좋다. ① 프롬프트를 제출하게 하고 ② 체크리스트 열 항목을 직접 표시하게 하고 ③ 결과를 합격선(0.108)·무작위(0)와 비교한 문장과, 묶음 수를 정답 없이 고른 근거를 쓰게 한다. AI 답의 함수·인자는 설치된 버전의 공식 문서로 확인하게 하고, 과제에 AI 도움을 받았으면 사용 범위를 밝히게 한다.'};
var CODE={title:'계층적 군집 면접 — 스무 장 합치기부터 합격 판정까지',sub:'numpy, scipy, scikit-learn, matplotlib · 인터넷 불필요 · CPU 15초 안팎',file:'hierarchical_interview',
 desc:'이 편의 면접을 그대로 옮겼다. 손글씨 1,797장을 읽고 정답을 채점 봉투에 넣고(1번 칸), 매뉴얼 씨의 합격선을 세우고(2번), 합치기를 직접 짜서 scikit-learn과 맞춰 보고(3번), ward로 10묶음(4번), 연결 방식 네 가지(5번), 덴드로그램(6번), 자르는 높이와 실루엣(7번), 채점 봉투 열기(8번), 업무 점수(9번), 새 글씨 배정(10번), 표준화와 K-Means 비교(11번), 판정(12번)까지 간다. 정답 y는 "[채점]"이라고 적힌 칸에서만 쓴다. 노트북으로 저장하면 칸마다 셀이 나뉜다.',
 cap:'실행 결과 (scikit-learn 1.9.1, CPU). 시간(ms)은 컴퓨터마다 다르다. 6번 칸의 덴드로그램 그림은 Colab에서만 보인다. 브라우저 계산과 같은 숫자(ward ARI 0.794 · NMI 0.868 · 실루엣 0.178, average 0.514, complete 0.429, single 0.000, 업무 점수 이름 10장 84.9%)가 나온다. 3번 칸의 직접 짠 합치기는 느린 판이라 앞 100장만 쓴다.'};
