var BOARD_LINE='R² 0.735 · 76.7%'; /* 회귀 합격선(매뉴얼 경험식 R²) · 분류 합격선(매뉴얼 씨 정확도) */
/* ══ 5-9 GBM · 콘크리트 실기 과제지 ══════════════════════════════════ */
/* GB-ENGINE-START */
/* ── 그레이디언트 부스팅 엔진 (이 편 전용, 브라우저·node 공용) ──
   scikit-learn GradientBoostingRegressor/Classifier의 기본 동작을 그대로 따른다.
   · 나무: 회귀 나무, 깊이 max_depth, 분할 기준 friedman_mse(= 두 쪽 평균 차이로 고르는 MSE 감소), 문턱은 이웃한 두 값의 가운데
   · 회귀(squared_error): 시작값 = 훈련 평균, 잎 값 = 잔차 평균
   · 이진 분류(log_loss): 시작값 = log(p/(1−p)), 잔차 = y − p, 잎 값 = Σ잔차 / Σp(1−p) (뉴턴 한 걸음)
   · 다중 분류: 단계마다 클래스 수만큼 나무, 잎 값 = (K−1)/K · Σ잔차 / Σp(1−p)
   같은 점수의 분할이 여럿이면 특징 순서가 scikit-learn(무작위 순서)과 달라 다른 분할을 고를 수 있다 */
var GB=(function(){
  function presort(X){var n=X.length,d=X[0].length,O=[];for(var j=0;j<d;j++){var o=new Int32Array(n);for(var i=0;i<n;i++)o[i]=i;var col=X.map(function(r){return r[j]});Array.prototype.sort.call(o,function(a,b){return col[a]-col[b]||a-b});O.push(o);}return O;}
  /* 회귀 나무 한 그루: r(잔차)에 맞춘다. 반환 {f,t,l,r,v,n,imp}; 잎은 f=-1 */
  function tree(X,O,r,depth,minLeaf){var n=X.length,d=X[0].length,T={f:[],t:[],l:[],r:[],v:[],n:[],gain:[]},node=new Int32Array(n);minLeaf=minLeaf||1;
    function add(){T.f.push(-1);T.t.push(0);T.l.push(-1);T.r.push(-1);T.v.push(0);T.n.push(0);T.gain.push(0);return T.f.length-1;}
    function build(id,members,dp){var m=members.length,s=0,ss=0;for(var i=0;i<m;i++){var v=r[members[i]];s+=v;ss+=v*v;}T.v[id]=s/m;T.n[id]=m;
      var imp=ss/m-(s/m)*(s/m);if(dp>=depth||m<2*minLeaf||m<2||imp<=1e-12)return;
      var inN=new Uint8Array(n);for(i=0;i<m;i++)inN[members[i]]=1;
      var best=-1,bf=-1,bt=0;
      for(var j=0;j<d;j++){var o=O[j],sl=0,nl=0,prev=-1;
        for(var k=0;k<n;k++){var a=o[k];if(!inN[a])continue;
          if(prev>=0&&nl>=minLeaf&&m-nl>=minLeaf&&X[a][j]>X[prev][j]+1e-7){var nr=m-nl,sr=s-sl,df=sl*nr-sr*nl,p=df*df/(nl*nr);if(p>best){best=p;bf=j;bt=(X[prev][j]+X[a][j])/2;}}
          sl+=r[a];nl++;prev=a;}}
      if(bf<0)return;
      var L=[],R=[];for(i=0;i<m;i++){var q=members[i];(X[q][bf]<=bt?L:R).push(q);}
      var li=add(),ri=add();T.f[id]=bf;T.t[id]=bt;T.l[id]=li;T.r[id]=ri;T.gain[id]=best/m/n; /* = 이 분할이 줄인 제곱 오차 ÷ 훈련 표본 수 */
      build(li,L,dp+1);build(ri,R,dp+1);}
    var root=add(),all=[];for(var i=0;i<n;i++)all.push(i);build(root,all,0);return T;}
  function leaf(T,x){var i=0;while(T.f[i]>=0)i=x[T.f[i]]<=T.t[i]?T.l[i]:T.r[i];return i;}
  function sig(z){return z>=0?1/(1+Math.exp(-z)):Math.exp(z)/(1+Math.exp(z));}
  /* fit(X, y, {loss:'sq'|'log'|'multi', K, n, lr, depth, evals:[X…], onStage(t, raws)}) */
  function fit(X,y,o){o=o||{};var loss=o.loss||'sq',K=loss==='multi'?(o.K||3):1,N=o.n||100,lr=o.lr==null?0.1:o.lr,depth=o.depth||3,n=X.length,O=presort(X);
    var init=[];if(loss==='sq')init=[y.reduce(function(a,b){return a+b},0)/n];
    else if(loss==='log'){var p1=y.reduce(function(a,b){return a+b},0)/n;p1=Math.min(Math.max(p1,2.2e-16),1-2.2e-16);init=[Math.log(p1/(1-p1))];}
    else{var cnt=new Array(K).fill(0);y.forEach(function(v){cnt[v]++});var lg=cnt.map(function(c){return Math.log(Math.max(c/n,2.2e-16))}),mu=lg.reduce(function(a,b){return a+b},0)/K;init=lg.map(function(v){return v-mu});}
    var F=[];for(var k=0;k<K;k++)F.push(new Float64Array(n).fill(init[k]));
    var EV=(o.evals||[]).map(function(E){var A=[];for(var k=0;k<K;k++)A.push(new Float64Array(E.length).fill(init[k]));return A;});
    var trees=[];
    for(var t=0;t<N;t++){var P=null;
      if(loss==='multi'){P=[];for(var i=0;i<n;i++){var mx=-1e300;for(k=0;k<K;k++)mx=Math.max(mx,F[k][i]);var e=[],se=0;for(k=0;k<K;k++){e.push(Math.exp(F[k][i]-mx));se+=e[k];}P.push(e.map(function(v){return v/se}));}}
      var stage=[];
      for(k=0;k<K;k++){var r=new Float64Array(n),pr=null;
        if(loss==='sq')for(i=0;i<n;i++)r[i]=y[i]-F[0][i];
        else if(loss==='log'){pr=new Float64Array(n);for(i=0;i<n;i++){pr[i]=sig(F[0][i]);r[i]=y[i]-pr[i];}}
        else{pr=new Float64Array(n);for(i=0;i<n;i++){pr[i]=P[i][k];r[i]=(y[i]===k?1:0)-pr[i];}}
        var T=tree(X,O,r,depth,o.minLeaf);
        if(loss!=='sq'){var num={},den={};for(i=0;i<n;i++){var lf=leaf(T,X[i]);num[lf]=(num[lf]||0)+r[i];den[lf]=(den[lf]||0)+pr[i]*(1-pr[i]);}
          for(var key in num){var nu=num[key];if(loss==='multi')nu*=(K-1)/K;T.v[+key]=Math.abs(den[key])<1e-150?0:nu/den[key];}}
        for(i=0;i<n;i++)F[k][i]+=lr*T.v[leaf(T,X[i])];
        (o.evals||[]).forEach(function(E,ei){for(var q=0;q<E.length;q++)EV[ei][k][q]+=lr*T.v[leaf(T,E[q])];});
        stage.push(T);}
      trees.push(stage);if(o.onStage)o.onStage(t+1,F,EV);}
    return {loss:loss,K:K,lr:lr,init:init,trees:trees,F:F,EV:EV};}
  function raw(M,x,upto){var K=M.K,z=M.init.slice(),U=upto==null?M.trees.length:Math.min(upto,M.trees.length);for(var t=0;t<U;t++)for(var k=0;k<K;k++){var T=M.trees[t][k];z[k]+=M.lr*T.v[leaf(T,x)];}return z;}
  function predict(M,x,upto){var z=raw(M,x,upto);if(M.loss==='sq')return z[0];if(M.loss==='log')return z[0]>0?1:0;var b=0;for(var k=1;k<z.length;k++)if(z[k]>z[b])b=k;return b;}
  /* 특징 중요도: 분할이 줄인 제곱 오차를 특징별로 모두 더해 합이 1이 되게 나눈다 (scikit-learn feature_importances_와 같은 정의) */
  function importance(M,d){var tot=new Array(d).fill(0);M.trees.forEach(function(st){st.forEach(function(T){T.f.forEach(function(f,i){if(f>=0)tot[f]+=T.gain[i];});});});var S=tot.reduce(function(a,b){return a+b},0)||1;return tot.map(function(v){return v/S});}
  return {presort:presort,tree:tree,leaf:leaf,fit:fit,raw:raw,predict:predict,importance:importance,sig:sig};})();
/* GB-ENGINE-END */

const ROWS=concRows(),TR=ROWS.filter(r=>!r.test),TE=ROWS.filter(r=>r.test);
const fv=r=>FEAT.map(f=>r[f]);
const A=TR.map(fv),B=TE.map(fv); /* 나무는 표준화가 필요 없다: kg·일 그대로 */
const YTR=TR.map(r=>r.strength),YTE=TE.map(r=>r.strength),PTR=TR.map(r=>r.pass),PTE=TE.map(r=>r.pass),GTR=TR.map(r=>r.grade),GTE=TE.map(r=>r.grade);
const PNAME=['불합격','합격'];
function r2(y,p){const m=ML.mean(y);let a=0,b=0;for(let i=0;i<y.length;i++){a+=(y[i]-p[i])*(y[i]-p[i]);b+=(y[i]-m)*(y[i]-m);}return 1-a/b;}
function mae(y,p){let s=0;for(let i=0;i<y.length;i++)s+=Math.abs(y[i]-p[i]);return s/y.length;}
/* 매뉴얼 씨: 회귀 경험식(28일 강도 96.5/7^물-결합재비 × 재령 보정 t/(4+0.85t)), 분류 작업 표준서 */
const manualMPa=(wb,t)=>96.5/Math.pow(7,wb)*t/(4+0.85*t);
const MAN=TE.map(r=>manualMPa(r.wb,r.age)),LINE_R2=r2(YTE,MAN),LINE_MAE=mae(YTE,MAN);
const BASE=ML.mean(YTR),BASE_R2=r2(YTE,YTE.map(()=>BASE)),BASE_MAE=mae(YTE,YTE.map(()=>BASE));
const BASE_ACC=PTE.filter(v=>v===0).length/PTE.length,BASE3=GTE.filter(v=>v===1).length/GTE.length;
const fmtR=v=>fx(v,3),fmtM=v=>v.toFixed(2);
function cmGrid(M,labels,colsFor){const K=labels.length;let h='<div class="cm" style="grid-template-columns:86px repeat('+K+',minmax(48px,1fr));max-width:'+(110+K*90)+'px"><div class="h">정답 ＼ 예측</div>'+labels.map(k=>'<div class="h">'+k+'</div>').join('');
  M.forEach((row,a)=>{h+='<div class="h" style="justify-content:end;text-align:right">'+labels[a]+'</div>'+row.map((v,b)=>a===b?'<div class="c'+(v?' d':'')+'" style="'+(v?'background:'+rgba(colsFor(a),.85):'')+'">'+v+'</div>':'<div class="c'+(v?' x':'')+'">'+(v||'·')+'</div>').join('');});return h+'</div>';}
function barRows(rows,max){return rows.map(r=>'<div class="row"><span style="text-align:left;font-family:var(--font)">'+r[0]+'</span><div class="tr"><i style="left:0;width:'+Math.max(0,Math.min(100,r[1]/max*100))+'%;background:'+(r[2]||'var(--blue)')+'"></i></div><span>'+r[3]+'</span></div>').join('');}
function axes(ctx,C,pl,pt,gw,gh){ctx.strokeStyle=C.rule;ctx.lineWidth=1;ctx.strokeRect(pl,pt,gw,gh);ctx.fillStyle=C.muted;ctx.font=font(12);}
function vlabel(ctx,txt,x,y){ctx.save();ctx.translate(x,y);ctx.rotate(-Math.PI/2);ctx.textAlign='center';ctx.fillText(txt,0,0);ctx.restore();}
/* 나무 수 세기 (예산): 문턱값 + 잎 값 */
function countNums(M){let th=0,lf=0;M.trees.forEach(st=>st.forEach(T=>T.f.forEach(f=>{if(f>=0)th++;else lf++;})));return {th,lf,all:th+lf};}

/* ── 본 모델: 특징 8개 · 300그루 · η 0.1 · 깊이 3. 나무마다 훈련·시험 예측을 모두 적어 둔다(04·05장·면접 공용) ── */
const NMAIN=300;let MAIN=null,CLS=null,G3=null;
function ensureMain(){if(MAIN)return MAIN;const t0=performance.now(),trS=[new Float64Array(YTR.length).fill(BASE)],teS=[new Float64Array(YTE.length).fill(BASE)];
  const M=GB.fit(A,YTR,{n:NMAIN,lr:0.1,depth:3,evals:[B],onStage:(t,F,EV)=>{trS.push(Float64Array.from(F[0]));teS.push(Float64Array.from(EV[0][0]));}});
  M.ms=performance.now()-t0;M.trS=trS;M.teS=teS;M.teR2=teS.map(p=>r2(YTE,p));M.trR2=trS.map(p=>r2(YTR,p));M.imp=GB.importance(M,8);M.nums=countNums(M);trS[0]=new Float64Array(YTR.length).fill(M.init[0]);teS[0]=new Float64Array(YTE.length).fill(M.init[0]);
  MAIN=M;if($('memN'))$('memN').textContent=M.nums.all.toLocaleString();return M;}
function ensureCls(){if(CLS)return CLS;const t0=performance.now(),teZ=[];
  const M=GB.fit(A,PTR,{loss:'log',n:NMAIN,lr:0.1,depth:3,evals:[B],onStage:(t,F,EV)=>{teZ.push(Float64Array.from(EV[0][0]));}});
  M.ms=performance.now()-t0;teZ.unshift(new Float64Array(YTE.length).fill(M.init[0]));M.teZ=teZ;
  M.acc=teZ.map(z=>{let c=0;for(let i=0;i<z.length;i++)if((z[i]>0?1:0)===PTE[i])c++;return c/z.length;});
  M.trAcc=ML.acc(PTR,M.F[0].map(z=>z>0?1:0));M.pred=Array.from(teZ[NMAIN]).map(z=>z>0?1:0);CLS=M;return M;}
function ensureG3(){if(G3)return G3;const M=GB.fit(A,GTR,{loss:'multi',K:3,n:100,lr:0.1,depth:3});M.acc=ML.acc(GTE,B.map(x=>GB.predict(M,x)));G3=M;return M;}
/* 화면에 가까워지면 계산 */
function whenNear(id,fn,margin){const t=$(id);let done=false;function go(){if(done)return;done=true;setTimeout(fn,10);}
  if('IntersectionObserver' in window){const io=new IntersectionObserver(es=>{if(es.some(e=>e.isIntersecting)){go();io.disconnect();}},{rootMargin:(margin||600)+'px'});io.observe(t);}else go();
  window.addEventListener('hashchange',()=>{const h=location.hash.slice(1),el=h&&$(h);if(el&&(el===t||t.contains(el)||t.compareDocumentPosition(el)&Node.DOCUMENT_POSITION_FOLLOWING))go();});
  if(location.hash){const el=$(location.hash.slice(1));if(el&&(el===t||t.contains(el)||t.compareDocumentPosition(el)&Node.DOCUMENT_POSITION_FOLLOWING))go();}}

/* ── 히어로: 오답 노트 릴레이 ── */
heroRoom($('heroCv'),{sym:'GB',no:'APP-09',name:'GBM\n(부스팅)',role:'지원자 · 겸용팀',banner:'모델 채용 위원회 · 5-9 면접',doc:'front',asst:'board',fly:'clip',robo:'calc',loop:11,
  beats:[{at:0.4,who:'asst',mood:'ledger',move:'lean',say:'오답 노트 넘겨요'},{at:2.4,who:'robo',mood:'chart',move:'pop',say:'300그루면 얼마?'},{at:4.4,who:'fly',mood:'point',move:'shake',say:'언제 멈추는데?'},
         {at:6.4,who:'doc',mood:'magic',move:'pop',say:'조금씩 고쳐요'},{at:8.4,who:'asst',mood:'spark',move:'jump',say:'잔차가 줄었어요'}],
  motif:(function(){const R0=[30,-22,18,-34,26,-12,20,-28,14];
    return function(ctx,t,W,H,C){const m=Math.floor((REDUCE?6:t*1.6))%14,sc=Math.pow(0.78,m),x0=278,y0=150;
      ctx.strokeStyle=C.muted;ctx.lineWidth=1.5;ctx.beginPath();ctx.moveTo(x0-6,y0);ctx.lineTo(x0+9*15+4,y0);ctx.stroke();
      R0.forEach((v,i)=>{const h=v*sc*1.4;ctx.fillStyle=rgba(C.stamp,.85);ctx.fillRect(x0+i*15,h>0?y0-h:y0,10,Math.abs(h));});
      for(let k=0;k<Math.min(m,10);k++){const tx=x0+4+k*13,ty=88;ctx.fillStyle=C.c_app;ctx.beginPath();ctx.moveTo(tx,ty-9);ctx.lineTo(tx-5,ty);ctx.lineTo(tx+5,ty);ctx.closePath();ctx.fill();ctx.fillRect(tx-1,ty,2,4);}
      ctx.fillStyle=C.muted;ctx.font=font(12,500,1);ctx.textAlign='left';ctx.fillText('나무 '+m+'그루 · 잔차',x0-4,206);};})()});
$('heroLine').textContent='R² '+fmtR(LINE_R2)+' · '+pct(PASSLINE);

/* ── 01 장면 ── */
comic('comicIntro',[
 {bg:'office',prop:'나무 1 · 평균에서 출발',cast:[{who:'asst',mood:'ledger',x:30,move:'nod'},{who:'fly',mood:'point',x:74,move:'tilt',flip:true}],say:{who:'fly',txt:'첫 나무가 센 배합을 다 낮게 봤잖아. 틀린 게 수두룩해.'},cap:'① 첫 나무의 실수 — 질문 세 개로는 대충만 맞힌다'},
 {bg:'office',prop:'나무 2 · 잔차만 배운다',cast:[{who:'asst',mood:'spark',x:30,move:'jump'},{who:'robo',mood:'chart',x:72,move:'sway',h:104}],say:{who:'asst',txt:'틀린 만큼만 적은 오답 노트를 다음 나무에게 넘겼어요. 남은 오차가 줄었습니다.'},cap:'② 다음 나무가 고친다 — 앞 나무들이 남긴 오차만 배운다'},
 {bg:'office',prop:'η = 1.0 · 600그루',cast:[{who:'asst',mood:'shock',x:28,move:'shake'},{who:'doc',mood:'scale',x:72,move:'tilt',h:100}],say:{who:'doc',txt:'시편 하나의 우연까지 고치면 외운 거예요. 조금씩, 그리고 멈출 때를 정합시다.'},cap:'③ 너무 많이 고치면 — 우연까지 외운다(과대적합)'}]);

/* ── 01 자기소개: 특징 하나로 한 그루씩 덧대기 (28일 훈련 배합) ── */
(function(){const T=TR.filter(r=>r.age===28),X1=T.map(r=>[r.wb]),Y1=T.map(r=>r.strength),O1=GB.presort(X1),n=T.length;
  const xs=X1.map(r=>r[0]),X0=Math.min(...xs)-0.02,X9=Math.max(...xs)+0.02,NG=240,grid=[...Array(NG)].map((_,i)=>X0+(X9-X0)*i/(NG-1));
  const st={lr:0.3,F:null,G:null,k:0,lastR:null,lastH:null,lastT:null,play:0};const cv=$('bstCv'),H=430,m0=ML.mean(Y1);
  function reset(){st.F=new Float64Array(n).fill(m0);st.G=new Float64Array(NG).fill(m0);st.k=0;st.lastR=null;st.lastH=null;st.lastT=null;}
  function add(){const r=Y1.map((y,i)=>y-st.F[i]),T_=GB.tree(X1,O1,r,2);st.lastR=r;st.lastT=T_;st.lastH=grid.map(x=>T_.v[GB.leaf(T_,[x])]);
    for(let i=0;i<n;i++)st.F[i]+=st.lr*T_.v[GB.leaf(T_,X1[i])];for(let g=0;g<NG;g++)st.G[g]+=st.lr*st.lastH[g];st.k++;}
  function draw(){const W=fit(cv,H),ctx=cv.getContext('2d'),C=cols(),pl=48,pr=14,gw=W-pl-pr,t1=10,h1=230,t2=290,h2=110;
    const sx=x=>pl+(x-X0)/(X9-X0)*gw,sy=y=>t1+(1-y/85)*h1,sr=v=>t2+h2/2-v/45*(h2/2);ctx.clearRect(0,0,W,H);
    axes(ctx,C,pl,t1,gw,h1);[0,20,40,60,80].forEach(v=>{ctx.textAlign='right';ctx.fillText(v,pl-5,sy(v)+4);});vlabel(ctx,'강도 (MPa)',11,t1+h1/2);
    for(let x=0.3;x<=X9;x+=0.1){ctx.textAlign='center';ctx.fillText(x.toFixed(1),sx(x),t1+h1+15);}
    ctx.textAlign='center';ctx.fillText('물-결합재비 (28일 배합만)',pl+gw/2,t1+h1+30);
    X1.forEach((x,i)=>circ(ctx,sx(x[0]),sy(Y1[i]),2.8,rgba(C.text,.5),null));
    ctx.strokeStyle=C.blue;ctx.lineWidth=2.8;ctx.beginPath();grid.forEach((x,g)=>{const X=sx(x),Y=sy(st.G[g]);g?ctx.lineTo(X,Y):ctx.moveTo(X,Y);});ctx.stroke();
    ctx.fillStyle=C.blue;ctx.font=font(12.5,700);ctx.textAlign='right';ctx.fillText('예측 F (나무 '+st.k+'그루, η = '+st.lr+')',pl+gw-8,t1+18);
    /* 아래: 이번 나무가 배운 잔차 */
    axes(ctx,C,pl,t2,gw,h2);ctx.strokeStyle=C.muted;ctx.setLineDash([3,3]);ctx.beginPath();ctx.moveTo(pl,sr(0));ctx.lineTo(pl+gw,sr(0));ctx.stroke();ctx.setLineDash([]);
    [-40,0,40].forEach(v=>{ctx.textAlign='right';ctx.fillStyle=C.muted;ctx.fillText((v>0?'+':'')+v,pl-5,sr(v)+4);});vlabel(ctx,'잔차',11,t2+h2/2);
    const R=st.lastR||Y1.map((y,i)=>y-st.F[i]);ctx.strokeStyle=rgba(C.stamp,.55);ctx.lineWidth=1.6;
    X1.forEach((x,i)=>{const v=clamp(R[i],-45,45);ctx.beginPath();ctx.moveTo(sx(x[0]),sr(0));ctx.lineTo(sx(x[0]),sr(v));ctx.stroke();});
    if(st.lastH){ctx.strokeStyle=C.amber;ctx.lineWidth=3;ctx.beginPath();grid.forEach((x,g)=>{const X=sx(x),Y=sr(st.lastH[g]);g?ctx.lineTo(X,Y):ctx.moveTo(X,Y);});ctx.stroke();}
    ctx.font=font(12,700);ctx.textAlign='left';ctx.fillStyle=rgba(C.panel,.9);ctx.fillRect(pl+4,t2+3,Math.min(gw-8,330),st.lastH?32:18);ctx.fillStyle=C.stamp;ctx.fillText(st.lastH?'빨간 막대: '+st.k+'번째 나무가 받은 오답 노트(잔차)':'빨간 막대: 지금 남은 잔차 (실제 − 평균)',pl+8,t2+15);
    if(st.lastH){ctx.fillStyle=C.amber;ctx.fillText('주황 선: 그 나무의 답 h (곱하기 η 해서 위에 더함)',pl+8,t2+30);}
    info();}
  function info(){const R2=r2(Y1,Array.from(st.F)),ms=Y1.reduce((s,y,i)=>s+(y-st.F[i])**2,0)/n;let h='나무 <b>'+st.k+'그루</b> · 훈련 R² <b>'+fmtR(R2)+'</b> · 잔차 제곱 평균 <b>'+ms.toFixed(1)+'</b>. ';
    if(!st.k)h+='지금은 324건 모두에게 평균 '+m0.toFixed(1)+'MPa를 답한다. R²가 0인 까닭이다. "덧대기"를 눌러 첫 나무를 세워 보자.';
    else{const T_=st.lastT,q=T_.f[0]>=0?'물-결합재비 ≤ '+T_.t[0].toFixed(3)+'?':'(나눌 곳 없음)';
      h+=st.k+'번째 나무의 첫 질문은 "'+q+'"였고, 잎 '+T_.f.filter(f=>f<0).length+'개에 잔차 평균을 적었다. 그 값의 '+Math.round(st.lr*100)+'%만 예측에 더했다. ';
      h+=st.lr>=1&&st.k>=15?'η = 1.0은 한 그루의 답을 통째로 믿는다. 파란 선이 몇 안 되는 점을 쫓아 들쭉날쭉해진다 — 시편의 우연까지 외우기 시작한 것이다. 이 위험을 04장에서 검증 몫으로 잰다.':
        st.lr<=0.1&&st.k<15?'η = 0.1은 걸음이 작다. 같은 성적에 나무가 더 필요하지만 선이 차분하다.':
        st.k>=25?'잔차 막대가 고르게 짧아졌다. 물-결합재비 하나로는 더 나아지기 어렵다. 나머지는 다른 특징(시멘트 종류, 감수제)의 몫이다.':'오른쪽(물이 많은 배합)과 왼쪽(물이 적은 배합)의 높이 차이부터 잡는다. 가장 큰 오차를 먼저 줄이는 질문이 뽑힌다.';}
    $('bInfo').innerHTML=h;}
  $('bAdd').addEventListener('click',()=>{if(st.play)return;add();draw();});
  $('bPlay').addEventListener('click',async()=>{if(st.play)return;st.play=1;$('bPlay').disabled=true;while(st.k<30){add();draw();await wait(REDUCE?0:160);}st.play=0;$('bPlay').disabled=false;});
  $('bReset').addEventListener('click',()=>{if(st.play)return;reset();draw();});
  seg('#bLr button',v=>{if(st.play)return;st.lr=+v;reset();draw();});
  reset();draw();onRedraw(draw);})();

/* ── 02 실기 과제지 ── */
$('lineR2').textContent='R² '+fmtR(LINE_R2);$('lineAcc').textContent=pct(PASSLINE);
$('scaleRun').addEventListener('click',async()=>{$('scaleInfo').textContent='계산 중…';await wait(20);
  const SC=ML.scaler(A),As=SC.t(A),M1=GB.fit(A,YTR,{n:30}),M2=GB.fit(As,YTR,{n:30});let dmax=0;
  A.forEach((x,i)=>{dmax=Math.max(dmax,Math.abs(GB.raw(M1,x)[0]-GB.raw(M2,As[i])[0]));});
  const T1=M1.trees[0][0],T2=M2.trees[0][0],f=T1.f[0],back=T2.t[0]*SC.sd[T2.f[0]]+SC.mu[T2.f[0]];
  $('scaleInfo').innerHTML='첫 나무의 첫 질문 — 원래 단위 그대로: <b>"'+FEATK[FEAT[f]]+' ≤ '+T1.t[0].toFixed(1)+'?"</b> · 표준화: <b>"'+FEATK[FEAT[T2.f[0]]]+' ≤ '+fx(T2.t[0],3)+'?"</b>. 표준화 값 '+fx(T2.t[0],3)+'을 원래 단위로 되돌리면 '+back.toFixed(1)+'. 같은 문턱이다. 30그루를 다 덧댄 뒤 훈련 772건의 예측 차이는 최대 <b>'+(dmax<1e-9?'0':dmax.toExponential(1))+'MPa</b>. 나무에게 표준화는 아무것도 바꾸지 않는다(문턱의 숫자만 바뀐다).';});
/* 산점도와 표 */
(function(){const xs=$('dxSel'),ys=$('dySel'),OPT=FEAT.concat(['wb','strength']),LAB=Object.assign({},FEATK,{wb:'물-결합재비',strength:'압축 강도(MPa)'});
  OPT.forEach(f=>{xs.add(new Option(LAB[f],f));ys.add(new Option(LAB[f],f));});xs.value='age';ys.value='strength';
  const cv=$('dataCv'),H=340;
  function draw(){const W=fit(cv,H),ctx=cv.getContext('2d'),C=cols(),fx_=xs.value,fy=ys.value,pl=52,pr=14,pt=14,pb=40,gw=W-pl-pr,gh=H-pt-pb;
    const xv=ROWS.map(r=>r[fx_]),yv=ROWS.map(r=>r[fy]),x0=Math.min(0,...xv),x1=Math.max(...xv)*1.04,y0=Math.min(0,...yv),y1=Math.max(...yv)*1.05,sx=x=>pl+(x-x0)/(x1-x0)*gw,sy=y=>pt+(1-(y-y0)/(y1-y0))*gh;
    ctx.clearRect(0,0,W,H);ctx.strokeStyle=C.rule;ctx.fillStyle=C.muted;ctx.font=font(12);
    for(let k=0;k<=5;k++){const gx=x0+(x1-x0)*k/5,gy=y0+(y1-y0)*k/5;ctx.beginPath();ctx.moveTo(sx(gx),pt);ctx.lineTo(sx(gx),pt+gh);ctx.moveTo(pl,sy(gy));ctx.lineTo(pl+gw,sy(gy));ctx.stroke();ctx.textAlign='center';ctx.fillText(gx.toFixed(gx<5?2:0),sx(gx),H-22);ctx.textAlign='right';ctx.fillText(gy.toFixed(gy<5?2:0),pl-6,sy(gy)+4);}
    ctx.textAlign='center';ctx.fillText(LAB[fx_],pl+gw/2,H-4);vlabel(ctx,LAB[fy],12,pt+gh/2);
    if(fy==='strength'){[25,35,45].forEach(v=>{ctx.setLineDash(v===35?[6,4]:[2,4]);ctx.strokeStyle=v===35?C.stamp:C.muted;ctx.beginPath();ctx.moveTo(pl,sy(v));ctx.lineTo(pl+gw,sy(v));ctx.stroke();ctx.setLineDash([]);ctx.fillStyle=v===35?C.stamp:C.muted;ctx.textAlign='right';ctx.fillText(v+'MPa'+(v===35?' 합격선':''),pl+gw-4,sy(v)-4);});}
    ROWS.forEach(r=>{const c=GRADECOL[r.grade];if(r.test)circ(ctx,sx(r[fx_]),sy(r[fy]),3.6,C.panel,c,1.6);else circ(ctx,sx(r[fx_]),sy(r[fy]),3,rgba(c,.75),null);});}
  xs.addEventListener('change',draw);ys.addEventListener('change',draw);draw();onRedraw(draw);
  const cnt=[0,0,0],tst=[0,0,0];ROWS.forEach(r=>{cnt[r.grade]++;if(r.test)tst[r.grade]++;});
  $('dataLeg').innerHTML=GRADES.map((g,k)=>'<span><i style="background:'+GRADECOL[k]+'"></i>'+g+' ('+(k===0?'25MPa 미만':k===1?'25~45':'45 이상')+') '+cnt[k]+'건 · 시험 '+tst[k]+'</span>').join('')+'<span>○ 속이 빈 점 = 봉인된 시험지</span>';
  $('concTbl').innerHTML='<table><thead><tr><th>#</th>'+FEAT.map(f=>'<th><span class="tg x">특징 X</span>'+FEATK[f]+'</th>').join('')+'<th><span class="tg t">과제 A</span>압축 강도(MPa)</th><th><span class="tg t">과제 B</span>35MPa 합격</th><th>용도</th></tr></thead><tbody>'+
    ROWS.map(r=>'<tr><td>'+(r.i+1)+'</td>'+FEAT.map(f=>'<td>'+r[f]+'</td>').join('')+'<td class="y">'+r.strength+'</td><td>'+PNAME[r.pass]+'</td><td>'+(r.test?'<span class="pill no">봉인</span>':'훈련')+'</td></tr>').join('')+'</tbody></table>';})();

/* ── 03 AI 비서 ── */
const MAIN_PROMPT=`너는 머신러닝 수업 조교다. Google Colab에서 바로 실행되는 파이썬 코드를 써 줘.

[환경] Colab 기본 라이브러리(numpy, pandas, scikit-learn)와 ucimlrepo만 쓴다. 다른 패키지는 설치하지 않는다. 첫 줄에 scikit-learn 버전을 출력한다.
[데이터] 〔ucimlrepo의 fetch_ucirepo(id=165)〕 로 콘크리트 압축 강도 데이터를 읽는다.
  - 특징 X: 〔시멘트, 고로 슬래그, 플라이 애시, 물, 고성능 감수제, 굵은 골재, 잔골재, 재령〕 8개 (열 순서대로 이름을 붙인다)
  - 타겟: 과제 A는 〔압축 강도(MPa)〕(회귀), 과제 B는 〔35MPa 이상이면 1〕(분류)
[규정]
  1) 강도를 〔25, 45〕MPa로 나눈 3등급으로 stratify 해서 train_test_split(test_size=0.25, random_state=42)로 먼저 나눈다.
  2) 과제 A는 GradientBoostingRegressor, 과제 B는 GradientBoostingClassifier. 깊이는 max_depth=3으로 얕게 둔다.
  3) 학습률과 나무 수는 훈련 데이터 안에서 고른다. 훈련 데이터의 일부를 검증 몫으로 떼어 staged_predict로 나무 수별 검증 점수를 그리거나, n_iter_no_change로 조기 종료한다.
  4) 손실 이름은 지금 버전의 이름("squared_error", "log_loss")을 쓴다.
[출력] 나무 수별 검증 R² 그래프, 고른 근무 조건, 시험 R²·MAE와 정확도, 기준선과 기존 규칙의 점수.
[금지] 시험 데이터로 나무 수나 학습률을 고르지 않는다. 데이터를 지어내지 않는다.
[형식] 셀 단위로 나누고, 각 셀 위에 무엇을 하는지 한국어 주석을 단다. 코드 뒤에 내가 확인해야 할 점 3가지를 적어 줘.`;
$('mainPrompt').textContent=MAIN_PROMPT;$('copyMain').addEventListener('click',()=>copyText(MAIN_PROMPT,'프롬프트를 복사했습니다'));
(function(){const LINES=[
 ['import pandas as pd'],['from ucimlrepo import fetch_ucirepo'],['from sklearn.model_selection import train_test_split'],
 ['from sklearn.ensemble import GradientBoostingClassifier',0,'불러오는 것 자체는 괜찮다. 어디에 쓰는지가 문제다(11번 줄).'],['from sklearn.metrics import r2_score',0,'회귀 점수 R²를 쓴다. 괜찮다.'],[''],
 ['uci = fetch_ucirepo(id=165)',0,'UCI에서 콘크리트 데이터를 실제로 받았다. 괜찮다.'],['X = uci.data.features',0,'특징 여덟 열. 나무는 표준화가 필요 없으니 그대로 써도 된다.'],['y = uci.data.targets.iloc[:, 0]   # 압축 강도(MPa)',0,'타겟은 강도 숫자다. 회귀 과제다. 괜찮다.'],[''],
 ['X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)',1,'<b>stratify와 random_state가 없다.</b> 돌릴 때마다 시험지가 바뀌어 점수를 비교할 수 없고, 고강도 배합이 한쪽으로 쏠릴 수 있다. 강도 등급으로 <code>stratify</code>하고 <code>random_state=42</code>를 넣는다.'],
 ['model = GradientBoostingClassifier(',2,'<b>회귀 과제에 분류기를 불렀다.</b> 강도(MPa)는 연속된 숫자라 <code>ValueError: Unknown label type: continuous</code>가 난다. 분류 GBM도 안쪽은 회귀 나무지만, 겉의 손실과 출력이 다르다. <code>GradientBoostingRegressor</code>를 쓴다.'],
 ['    loss="ls",',3,'<b>지금은 없는 손실 이름이다.</b> "ls"는 1.2에서 지워졌고 지금 이름은 <code>"squared_error"</code>(기본값)다. 옛 코드를 보고 배운 AI가 자주 낸다. 분류의 "deviance"도 같은 처지다(지금은 "log_loss").'],
 ['    n_estimators=1000, learning_rate=0.1,',0,'나무 수를 넉넉히 두고 나중에 고르는 것은 괜찮다. 어떻게 고르는지가 문제다(15~17번 줄).'],
 ['    max_depth=None)          # 깊을수록 정확하겠지',4,'<b>끝까지 키운 나무로 부스팅했다.</b> 첫 나무가 훈련 데이터를 거의 외워 잔차가 0에 가까워지고, 다음 나무들은 배울 것이 없다. A장 8번 칸에서 깊이 None은 시험 R² 0.890, 깊이 3은 0.949다. 부스팅은 <b>얕은</b> 나무(2~5)를 쓴다.'],
 ['model.fit(X_train, y_train)',0,'학습. 이 줄 자체는 괜찮다.'],[''],
 ['scores = [r2_score(y_test, p) for p in model.staged_predict(X_test)]',5,'<b>시험지로 나무 수를 골랐다.</b> staged_predict로 나무 수별 점수를 보는 것은 좋은 도구지만, 시험지에 쓰면 시험지에 맞춘 나무 수가 된다. 훈련 데이터에서 뗀 검증 몫에 쓰거나 <code>n_iter_no_change</code>로 조기 종료한다.'],
 ['best_n = scores.index(max(scores)) + 1',5,'<b>시험지로 나무 수를 골랐다.</b> 바로 윗줄과 같은 문제다.'],
 ['print("나무 수:", best_n, "R²:", max(scores))',5,'<b>시험지로 고른 최고점을 그대로 보고했다.</b> 고른 뒤의 모델로 시험지를 한 번만 열어 채점한 점수를 보고한다.']];
  const host=$('bugCode'),found=new Set();
  host.innerHTML=LINES.map((l,i)=>'<div class="ln" role="listitem" tabindex="0" data-i="'+i+'"><span class="no">'+(i+1)+'</span><span>'+hl(l[0]||' ')+'</span></div>').join('');
  function score(){$('bugScore').textContent=found.size+' / 5';$('bugScore').className='pill'+(found.size===5?' ok':'');}
  function pick(el){const i=+el.getAttribute('data-i'),l=LINES[i];if(!l[0])return;
    if(l[1]){found.add(l[1]);host.querySelectorAll('.ln').forEach(x=>{if(LINES[+x.getAttribute('data-i')][1]===l[1])x.classList.add('hit');});$('bugMsg').innerHTML=(i+1)+'번 줄: '+l[2]+(found.size===5?'<br><b>다섯 군데를 모두 찾았다.</b> 고친 코드는 A장의 Colab 코드와 같은 순서(규정대로 나누기 → 회귀·분류에 맞는 지원자 → 얕은 나무 → 검증 몫으로 나무 수 → 시험지 한 번)를 따른다.':'');}
    else{el.classList.add('miss');$('bugMsg').innerHTML=(i+1)+'번 줄: '+(l[2]||'이 줄은 괜찮다.');}score();}
  host.addEventListener('click',e=>{const el=e.target.closest('.ln');if(el)pick(el);});
  host.addEventListener('keydown',e=>{if((e.key==='Enter'||e.key===' ')&&e.target.classList.contains('ln')){e.preventDefault();pick(e.target);}});
  $('bugReveal').addEventListener('click',()=>{LINES.forEach((l,i)=>{if(l[1]){found.add(l[1]);host.querySelector('[data-i="'+i+'"]').classList.add('hit','reveal');}});score();$('bugMsg').innerHTML='빨간 줄 다섯 군데: ① 11번 나누기 규정 ② 12번 회귀에 분류기 ③ 13번 지워진 손실 이름 ④ 15번 끝까지 키운 나무 ⑤ 18·19·20번 시험지로 나무 수 고르고 그 점수 보고.';});
  $('bugReset').addEventListener('click',()=>{found.clear();host.querySelectorAll('.ln').forEach(x=>x.classList.remove('hit','miss','reveal'));score();$('bugMsg').textContent='줄을 하나 눌러 보세요.';});score();})();
(function(){const CK=[['데이터를 실제로 읽었나','AI가 예제 데이터를 지어내거나 다른 데이터로 바꾸지 않았는지, 행이 1,030개인지 본다.'],['타겟에 맞는 지원자를 불렀나','강도(MPa)는 GradientBoostingRegressor, 35MPa 합격 여부는 GradientBoostingClassifier다.'],['stratify와 random_state를 넣었나','등급 비율이 시험지에도 유지되고, 점수가 매번 같게 나온다.'],['손실·인자 이름이 내 버전에 맞나','print(sklearn.__version__). "ls"·"deviance"·criterion="mse"는 지금 오류다.'],['나무가 얕은가','max_depth는 2~5. None(끝까지)은 부스팅의 뜻을 없앤다.'],['나무 수와 학습률을 훈련 데이터 안에서 골랐나','검증 몫의 staged_predict, 교차 검증, n_iter_no_change. 시험지로 고르면 점수가 부풀려진다.'],['다른 패키지를 몰래 설치하거나 섞지 않았나','XGBoost·LightGBM을 쓴다면 버전과 인자 이름을 그 문서로 확인한다. eta, num_leaves 같은 이름은 scikit-learn에 없다.'],['훈련 점수를 성적으로 보고하지 않았나','이 면접에서 분류 훈련 정확도는 99.9%다. 나무를 덧댈수록 1에 가까워지므로 뜻이 없다.'],['결과를 기준선·합격선과 비교했나','회귀는 R² 0.735(매뉴얼 경험식), 분류는 76.7%(매뉴얼 씨)와 기준선(늘 평균, 늘 불합격)보다 낫지 않으면 뽑을 까닭이 없다.']];
  const ul=$('checks');ul.innerHTML=CK.map((c,i)=>'<li><input type="checkbox" id="ck'+i+'"><label for="ck'+i+'"><b>'+c[0]+'</b><small>'+c[1]+'</small></label></li>').join('');
  function upd(){const n=ul.querySelectorAll('input:checked').length;ul.querySelectorAll('li').forEach(li=>li.classList.toggle('done',li.querySelector('input').checked));$('checkScore').textContent='확인 '+n+' / '+CK.length+(n===CK.length?' — 이제 돌려 본다. 결과도 확인 대상이다.':'');}
  ul.addEventListener('change',upd);upd();})();

/* ── 04 수습 기간 ① 오답 노트가 얇아진다 (훈련 잔차 분포) ── */
const RH=(function(){const cv=$('rhCv'),H=280,st={n:0,busy:false},LO=-40,HI=50,BW=2.5,NB=(HI-LO)/BW;let ymax=0;
  function hist(n){const F=ensureMain().trS[n],b=new Array(NB).fill(0);for(let i=0;i<F.length;i++){const r=YTR[i]-F[i],k=clamp(Math.floor((r-LO)/BW),0,NB-1);b[k]++;}return b;}
  function draw(){if(!MAIN)return;const W=fit(cv,H),ctx=cv.getContext('2d'),C=cols(),pl=44,pr=12,pt=12,pb=36,gw=W-pl-pr,gh=H-pt-pb,b=hist(st.n);
    if(!ymax)ymax=Math.max(...hist(NMAIN))*1.05;const sx=v=>pl+(v-LO)/(HI-LO)*gw;
    ctx.clearRect(0,0,W,H);axes(ctx,C,pl,pt,gw,gh);
    [-40,-20,0,20,40].forEach(v=>{ctx.textAlign='center';ctx.fillText((v>0?'+':'')+v,sx(v),H-20);});ctx.fillText('잔차 = 실제 − 예측 (MPa)',pl+gw/2,H-4);
    [0,0.5,1].forEach(f=>{const v=Math.round(ymax*f);ctx.textAlign='right';ctx.fillText(v,pl-5,pt+gh-f*gh+4);});vlabel(ctx,'배합 수',11,pt+gh/2);
    b.forEach((c,k)=>{const h=Math.min(gh,c/ymax*gh),x=sx(LO+k*BW);ctx.fillStyle=rgba(Math.abs(LO+k*BW+BW/2)<=5?C.blue:C.stamp,.8);ctx.fillRect(x+1,pt+gh-h,gw/NB-2,h);});
    ctx.strokeStyle=C.muted;ctx.setLineDash([3,3]);ctx.beginPath();ctx.moveTo(sx(0),pt);ctx.lineTo(sx(0),pt+gh);ctx.stroke();ctx.setLineDash([]);
    ctx.fillStyle=C.text;ctx.font=font(13,700,1);ctx.textAlign='right';ctx.fillText('나무 '+st.n+'그루 · 훈련 R² '+fmtR(MAIN.trR2[st.n]),pl+gw-8,pt+18);info(b);}
  function info(){const F=MAIN.trS[st.n],res=YTR.map((y,i)=>y-F[i]),in5=res.filter(v=>Math.abs(v)<=5).length,ma=res.reduce((s,v)=>s+Math.abs(v),0)/res.length;
    $('rhInfo').innerHTML='나무 <b>'+st.n+'그루</b>: 훈련 잔차의 절댓값 평균 <b>'+ma.toFixed(2)+'MPa</b>, ±5MPa 안에 든 배합 <b>'+in5+'/'+YTR.length+'</b>건(파란 막대). '+
      (st.n===0?'처음에는 모두에게 평균 '+MAIN.init[0].toFixed(1)+'MPa를 답하니 잔차가 −30~+45MPa로 넓게 퍼진다.':st.n<=30?'나무마다 가장 큰 오차부터 메운다. 양 끝의 빨간 막대가 빠르게 줄어든다.':st.n<300?'남은 오차가 작아질수록 한 그루가 줄이는 몫도 작아진다.':'300그루 뒤의 오답 노트는 얇다. 하지만 이것은 훈련 데이터의 성적이다. 처음 보는 배합에서도 그런지는 아래 ②와 05장이 잰다.');}
  function set(n){st.n=n;$('rhN').value=n;$('rhNO').textContent=n;draw();}
  $('rhN').addEventListener('input',e=>set(+e.target.value));
  $('rhPlay').addEventListener('click',async()=>{if(st.busy)return;st.busy=true;$('rhPlay').disabled=true;ensureMain();const SEQ=[0,1,2,3,5,8,12,18,25,35,50,70,100,140,200,250,300];for(const n of SEQ){set(n);await wait(REDUCE?0:280);}st.busy=false;$('rhPlay').disabled=false;});
  onRedraw(draw);return {start(){ensureMain();set(0);}};})();

/* ── 04 수습 기간 ② 학습률 × 나무 수 (검증 몫) ── */
const VAL=(function(){const FI=[],VA=[];TR.forEach((r,j)=>(j%5?FI:VA).push(r));
  const XF=FI.map(fv),YF=FI.map(r=>r.strength),XV=VA.map(fv),YV=VA.map(r=>r.strength),LRS=[1,0.3,0.1,0.03],NV=600,RES={};let cur=0.1,ready=false;
  const cv=$('valCv'),H=300,COL={1:'amber',0.3:'ok',0.1:'blue',0.03:'muted'};
  function run(lr){if(RES[lr])return RES[lr];const va=[],tr=[];GB.fit(XF,YF,{n:NV,lr,depth:3,evals:[XV],onStage:(t,F,EV)=>{va.push(r2(YV,EV[0][0]));tr.push(r2(YF,F[0]));}});
    const mx=Math.max(...va),best=va.indexOf(mx)+1,first=va.findIndex(v=>v>=mx-0.005)+1;return (RES[lr]={va,tr,mx,best,first});}
  function draw(){const W=fit(cv,H),ctx=cv.getContext('2d'),C=cols(),pl=46,pr=14,pt=12,pb=36,gw=W-pl-pr,gh=H-pt-pb,y0=0.5,y1=1,lx=n=>pl+n/NV*gw,ly=v=>pt+(1-(clamp(v,y0,y1)-y0)/(y1-y0))*gh;
    ctx.clearRect(0,0,W,H);axes(ctx,C,pl,pt,gw,gh);
    [0,100,200,300,400,500,600].forEach(n=>{ctx.textAlign='center';ctx.fillText(n,lx(n),H-20);});ctx.fillText('나무 수',pl+gw/2,H-4);
    [0.5,0.6,0.7,0.8,0.9,1].forEach(v=>{ctx.textAlign='right';ctx.fillText(v.toFixed(1),pl-5,ly(v)+4);ctx.strokeStyle=rgba(C.muted,.18);ctx.beginPath();ctx.moveTo(pl,ly(v));ctx.lineTo(pl+gw,ly(v));ctx.stroke();});vlabel(ctx,'R²',11,pt+gh/2);
    ctx.strokeStyle=rgba(C.stamp,.55);ctx.setLineDash([4,4]);ctx.beginPath();ctx.moveTo(lx(300),pt);ctx.lineTo(lx(300),pt+gh);ctx.stroke();ctx.setLineDash([]);ctx.fillStyle=C.stamp;ctx.font=font(12,700);ctx.textAlign='left';ctx.fillText('300그루',lx(300)+4,pt+gh-8);
    if(!ready)return;
    LRS.forEach((lr,li)=>{const R=RES[lr];if(!R)return;const on=lr===cur,col=C[COL[lr]]||C.blue;
      function line(arr,dash,w,a){ctx.strokeStyle=rgba(col,a);ctx.lineWidth=w;ctx.setLineDash(dash);ctx.beginPath();arr.forEach((v,i)=>{const X=lx(i+1),Y=ly(v);i?ctx.lineTo(X,Y):ctx.moveTo(X,Y);});ctx.stroke();ctx.setLineDash([]);}
      if(on){line(R.tr,[5,4],2,0.9);line(R.va,[],3,1);circ(ctx,lx(R.first),ly(R.va[R.first-1]),6,null,C.stamp,2.5);circ(ctx,lx(R.best),ly(R.mx),4,col,null);}
      else line(R.va,[],1.4,0.35);
      ctx.fillStyle=rgba(col,on?1:.6);ctx.font=font(12,on?700:500);ctx.textAlign='right';ctx.fillText('η = '+lr+(on&&W>520?' · 검증(실선) · 학습 몫(점선)':''),pl+gw-8,pt+gh-12-(LRS.length-1-li)*15);});}
  function info(){const R=RES[cur];if(!R)return;const t=[];LRS.forEach(lr=>{const Q=RES[lr];if(Q)t.push('η '+lr+': 최고 '+fmtR(Q.mx)+'('+Q.best+'그루)');});
    $('valInfo').innerHTML='<b>η = '+cur+'</b> · 검증 R² 최고 <b>'+fmtR(R.mx)+'</b>('+R.best+'그루) · 최고점의 0.005 안에 처음 든 곳 <b>'+R.first+'그루</b>(빨간 원) · 300그루에서 검증 '+fmtR(R.va[299])+', 학습 몫 '+fmtR(R.tr[299])+'. '+
      (cur>=1?'한 그루를 통째로 믿으면 수십 그루 만에 오르고, 그 뒤로는 학습 몫 R²만 1에 붙고 검증은 제자리이거나 내려간다. 고칠수록 우연을 외운다.':cur>=0.3?'빨리 오르지만 최고점이 η = 0.1보다 낮다. 큰 걸음은 골짜기 바닥을 지나치기 쉽다.':cur>=0.1?'천천히 오르고 300그루 무렵부터 거의 평평하다. 위원회는 이 평평한 구간의 어림수 300그루를 근무 조건으로 정했다.':'걸음이 너무 작아 600그루로도 아직 오르는 중이다. 시간을 더 들이면 더 좋아질 수 있지만, 그만큼 나무가 많이 든다.')+
      '<br><span class="muted">네 학습률의 검증 최고점 — '+t.join(' · ')+'. 이 그래프는 훈련 772건만 쓴다. 시험지는 아직 봉인되어 있다.</span>';}
  async function start(){if(ready)return;$('valInfo').textContent='학습률 네 가지로 600그루씩 학습하는 중…';for(const lr of LRS){await wait(15);run(lr);}ready=true;draw();info();}
  seg('#lrChoice button',v=>{cur=+v;draw();info();});draw();onRedraw(draw);return {start,run,RES};})();
whenNear('m4',()=>{RH.start();VAL.start();},300);

/* ── 05 성과 평가 ── */
function renderEval(){const M=ensureMain(),p=M.teS[NMAIN],R=M.teR2[NMAIN],E=mae(YTE,p);
  $('evalStat').innerHTML=statGrid([['R² '+fmtR(R),'시험 R² (MAE '+fmtM(E)+'MPa)','a'],['R² '+fmtR(LINE_R2),'회귀 합격선 · 매뉴얼 경험식 (MAE '+fmtM(LINE_MAE)+')','m'],['R² '+fmtR(BASE_R2),'기준선 · 늘 훈련 평균 '+BASE.toFixed(1)+'MPa (MAE '+fmtM(BASE_MAE)+')'],['+'+fx(R-LINE_R2,3),'합격선과의 차이 (R²)','s']],4);
  document.querySelectorAll('.gbR2').forEach(e=>e.textContent=fmtR(R));
  /* 덧대기 재생 */
  const cv=$('rpCv'),cc=$('rpCurve');let n=10,busy=false;
  function drawS(){const W=fit(cv,300),ctx=cv.getContext('2d'),C=cols(),pl=44,pt=10,s=Math.min(W-60,250),m=90,sx=v=>pl+v/m*s,sy=v=>pt+s-v/m*s,q=M.teS[n];
    ctx.clearRect(0,0,W,300);ctx.strokeStyle=C.rule;ctx.strokeRect(pl,pt,s,s);ctx.fillStyle=C.muted;ctx.font=font(12);
    [0,30,60,90].forEach(v=>{ctx.textAlign='center';ctx.fillText(v,sx(v),pt+s+15);ctx.textAlign='right';ctx.fillText(v,pl-5,sy(v)+4);});
    ctx.textAlign='center';ctx.fillText('실제 강도 (MPa)',pl+s/2,pt+s+31);vlabel(ctx,'예측 강도 (MPa)',11,pt+s/2);
    ctx.setLineDash([4,4]);ctx.strokeStyle=C.muted;ctx.beginPath();ctx.moveTo(sx(0),sy(0));ctx.lineTo(sx(m),sy(m));ctx.stroke();ctx.setLineDash([]);
    YTE.forEach((y,i)=>circ(ctx,sx(y),sy(clamp(q[i],0,m)),3.2,rgba(C.blue,.7),null));
    ctx.fillStyle=C.blue;ctx.font=font(12,700,1);ctx.textAlign='left';ctx.fillText(n+'그루 · R² '+fmtR(M.teR2[n])+' · MAE '+fmtM(mae(YTE,q)),pl+8,pt+16);}
  function drawC(){const W=fit(cc,180),ctx=cc.getContext('2d'),C=cols(),pl=40,pr=10,pt=8,pb=30,gw=W-pl-pr,gh=180-pt-pb,lx=k=>pl+k/NMAIN*gw,ly=v=>pt+(1-clamp(v,0,1))*gh;
    ctx.clearRect(0,0,W,180);axes(ctx,C,pl,pt,gw,gh);[0,100,200,300].forEach(k=>{ctx.textAlign='center';ctx.fillText(k,lx(k),180-14);});ctx.fillText('나무 수',pl+gw/2,180-1);
    [0,0.5,1].forEach(v=>{ctx.textAlign='right';ctx.fillText(v,pl-5,ly(v)+4);});
    ctx.strokeStyle=C.stamp;ctx.setLineDash([5,4]);ctx.beginPath();ctx.moveTo(pl,ly(LINE_R2));ctx.lineTo(pl+gw,ly(LINE_R2));ctx.stroke();ctx.setLineDash([]);ctx.fillStyle=C.stamp;ctx.textAlign='left';ctx.fillText('합격선 '+fmtR(LINE_R2),pl+6,ly(LINE_R2)+14);
    ctx.strokeStyle=C.blue;ctx.lineWidth=2.4;ctx.beginPath();for(let k=0;k<=n;k++){const X=lx(k),Y=ly(M.teR2[k]);k?ctx.lineTo(X,Y):ctx.moveTo(X,Y);}ctx.stroke();
    [10,100,300].forEach(k=>{if(k<=n)circ(ctx,lx(k),ly(M.teR2[k]),4,C.blue,C.panel,1.5);});circ(ctx,lx(n),ly(M.teR2[n]),5,C.stamp,null);}
  function info(){const first=M.teR2.findIndex(v=>v>LINE_R2);
    $('rpInfo').innerHTML='나무 <b>'+n+'그루</b>: 시험 R² <b>'+fmtR(M.teR2[n])+'</b> (10그루 '+fmtR(M.teR2[10])+' → 100그루 '+fmtR(M.teR2[100])+' → 300그루 '+fmtR(M.teR2[300])+'). '+
      (n<first?'아직 합격선 아래다. 나무가 적으면 예측이 평균 쪽으로 눌려 점들이 가로로 납작하다. 학습률 0.1이면 한 그루가 남은 오차의 10%쯤만 메우기 때문이다.':n<100?first+'그루에서 합격선을 넘었다. 점들이 대각선 쪽으로 펴지는 중이다.':'대각선 위에 모였다. 100그루 뒤로는 오르는 폭이 작아진다. 04장의 검증 곡선과 같은 모양이다.')+' 오른쪽 위 끝(70MPa 이상)은 여전히 조금 낮게 예측한다. 나무의 답은 훈련 데이터 안의 계단이라 범위 끝을 넘지 못한다.';}
  function set(k){n=clamp(k,1,NMAIN);$('rpN').value=n;$('rpNO').textContent=n;drawS();drawC();info();}
  $('rpN').addEventListener('input',e=>set(+e.target.value));
  document.querySelectorAll('[data-rp]').forEach(b=>b.addEventListener('click',()=>set(+b.getAttribute('data-rp'))));
  $('rpPlay').addEventListener('click',async()=>{if(busy)return;busy=true;$('rpPlay').disabled=true;for(const k of [1,2,3,5,8,10,15,20,30,45,60,80,100,130,170,220,260,300]){set(k);await wait(REDUCE?0:260);}busy=false;$('rpPlay').disabled=false;});
  set(10);onRedraw(()=>{drawS();drawC();});
  /* 과제 B: 분류 */
  const K=ensureCls(),Mc=ML.confusion(PTE,K.pred,2);
  $('cmB').innerHTML=cmGrid(Mc,PNAME,k=>k?cols().blue:cols().muted);
  $('clsStat').innerHTML=statGrid([[pct(K.acc[NMAIN]),'시험 정확도 (300그루)','a'],[pct(K.trAcc),'훈련 정확도'],[pct(PASSLINE),'분류 합격선 · 매뉴얼 씨','m'],[pct(BASE_ACC),'기준선 · 늘 불합격']],2)+
    '<p class="foot">나무 10 → 100 → 300그루: '+pct(K.acc[10])+' → '+pct(K.acc[100])+' → '+pct(K.acc[300])+'. 강도 미달인데 "합격"이라 한 배합 '+Mc[0][1]+'건, 합격인데 "불합격" '+Mc[1][0]+'건. 훈련 정확도가 거의 100%인 것은 외운 몫이다. 성적은 시험 정확도로 본다.</p>';
  /* 분류의 속: 로그 오즈 */
  const sel=$('loPick'),lc=$('loCv');
  TE.forEach((r,i)=>{const pf=GB.sig(K.teZ[NMAIN][i]);sel.add(new Option('시험 #'+(r.i+1)+' · 재령 '+r.age+'일 · 실제 '+r.strength.toFixed(1)+'MPa('+PNAME[r.pass]+') · 최종 확률 '+pf.toFixed(2),i));});
  let def=TE.findIndex((r,i)=>r.pass===1&&K.teZ[30][i]<0&&K.teZ[NMAIN][i]>1.5);if(def<0)def=0;sel.value=def;
  function drawL(){const i=+sel.value,W=fit(lc,240),ctx=lc.getContext('2d'),C=cols(),pl=40,pr=44,pt=10,pb=32,gw=W-pl-pr,gh=240-pt-pb,zs=K.teZ.map(z=>z[i]),zm=Math.max(4,...zs.map(Math.abs))*1.05,lx=k=>pl+k/NMAIN*gw,ly=z=>pt+(1-(z+zm)/(2*zm))*gh;
    ctx.clearRect(0,0,W,240);axes(ctx,C,pl,pt,gw,gh);[0,100,200,300].forEach(k=>{ctx.textAlign='center';ctx.fillText(k,lx(k),240-16);});ctx.fillText('나무 수',pl+gw/2,240-2);
    [-zm,0,zm].forEach(z=>{ctx.textAlign='right';ctx.fillText(z.toFixed(1),pl-5,ly(z)+4);ctx.textAlign='left';ctx.fillText(GB.sig(z).toFixed(2),pl+gw+5,ly(z)+4);});
    vlabel(ctx,'로그 오즈 F',11,pt+gh/2);ctx.save();ctx.translate(W-6,pt+gh/2);ctx.rotate(Math.PI/2);ctx.textAlign='center';ctx.fillText('합격 확률',0,0);ctx.restore();
    ctx.strokeStyle=C.muted;ctx.setLineDash([4,4]);ctx.beginPath();ctx.moveTo(pl,ly(0));ctx.lineTo(pl+gw,ly(0));ctx.stroke();ctx.setLineDash([]);
    ctx.strokeStyle=C.blue;ctx.lineWidth=2.4;ctx.beginPath();zs.forEach((z,k)=>{const X=lx(k),Y=ly(z);k?ctx.lineTo(X,Y):ctx.moveTo(X,Y);});ctx.stroke();
    circ(ctx,lx(0),ly(zs[0]),4,C.text,null);circ(ctx,lx(NMAIN),ly(zs[NMAIN]),5,zs[NMAIN]>0?C.blue:C.stamp,C.panel,1.5);
    const r=TE[i],cross=zs.findIndex(z=>(z>0)!==(zs[NMAIN]>0)),last=zs.length-1-[...zs].reverse().findIndex(z=>(z>0)!==(zs[NMAIN]>0));
    $('loInfo').innerHTML='시작값 F₀ = log(합격 비율 ÷ 불합격 비율) = <b>'+fx(K.init[0],3)+'</b>(확률 '+GB.sig(K.init[0]).toFixed(3)+'). 나무마다 회귀 나무의 잎 값 × 0.1을 더한다. 300그루 뒤 F = <b>'+fx(zs[NMAIN],2)+'</b> → 시그모이드 → 합격 확률 <b>'+GB.sig(zs[NMAIN]).toFixed(3)+'</b> → <b>'+PNAME[zs[NMAIN]>0?1:0]+'</b>. 실제는 '+r.strength.toFixed(1)+'MPa('+PNAME[r.pass]+')'+((zs[NMAIN]>0?1:0)===r.pass?', 맞혔다.':', 틀렸다.')+
      (cross>=0&&last<NMAIN?' 확률 0.5 선을 마지막으로 넘은 것은 '+(last+1)+'그루째다.':'')+' 5-4 로지스틱 회귀는 로그 오즈를 직선(가중치 × 특징)으로 만들었고, GBM은 나무 계단을 쌓아 만든다.';}
  sel.addEventListener('change',drawL);drawL();onRedraw(drawL);
  /* 특징 중요도 */
  const ord=FEAT.map((f,j)=>[M.imp[j],j]).sort((a,b)=>b[0]-a[0]);
  $('impBars').innerHTML='<div style="font-size:13px;margin-bottom:4px"><span style="color:var(--blue);font-weight:700">■ GBM</span> &nbsp; <span style="color:var(--muted);font-weight:700">■ 랜덤 포레스트</span></div>'+ord.map(o=>{const f=FEAT[o[1]],rf=PRE.rfImp[f];
    return '<div class="row"><span style="text-align:left;font-family:var(--font)">'+FEATK[f]+'</span><div><div class="tr" style="height:9px;margin-bottom:2px"><i style="left:0;width:'+(o[0]/0.4*100)+'%"></i></div><div class="tr" style="height:9px"><i style="left:0;width:'+(rf/0.4*100)+'%;background:var(--muted)"></i></div></div><span>'+o[0].toFixed(3)+'<br><span style="color:var(--muted)">'+rf.toFixed(3)+'</span></span></div>';}).join('');
  $('impInfo').innerHTML='GBM의 1·2위는 <b>'+FEATK[FEAT[ord[0][1]]]+'</b>, <b>'+FEATK[FEAT[ord[1][1]]]+'</b>(합 '+pct(ord[0][0]+ord[1][0],0)+')이고, 랜덤 포레스트도 같은 두 특징이 순서만 바꿔 1·2위다. 매뉴얼 씨가 쓰던 두 숫자(물-결합재비, 재령)와 겹친다. 재령은 그대로, 물-결합재비는 시멘트와 물로 나뉘어 들어갔다. 중요도는 "분할로 오차를 얼마나 줄였나"의 몫이다. 원인을 뜻하지 않고, 서로 얽힌 특징(시멘트·슬래그·물)끼리는 몫을 나눠 가질 수 있다.';}
whenNear('m5',renderEval,600);

/* ── 면접 결과 발표 ── */
Interview('iv9',{title:'GBM은 콘크리트 강도 예측(회귀)과 35MPa 판정(분류)에 합격할까?',
  q:'조건: 봉인된 시험지 258건에서 회귀 R²가 합격선(매뉴얼 경험식 R² '+fmtR(LINE_R2)+')과 기준선(R² '+fmtR(BASE_R2)+')을, 분류 정확도가 합격선(매뉴얼 씨 '+pct(PASSLINE)+')과 기준선(늘 불합격 '+pct(BASE_ACC)+')을 모두 넘으면 합격. 근무 조건은 04장에서 정한 300그루, η = 0.1, 깊이 3. 나무는 브라우저가 직접 키운다.',
  fly:'처음 보는 258건에서도 돼? 몇 그루에서 멈출지는 시험지 안 보고 정한 거 맞지?',robo:'나무 300그루면 외울 숫자 몇 개야? 줄 서서 키우는 데 얼마나 걸려?',
  async run(setP){setP(.2);await wait(30);const M=ensureMain();setP(.6);await wait(30);const K=ensureCls();setP(1);
    const R=M.teR2[NMAIN],E=mae(YTE,M.teS[NMAIN]),a=K.acc[NMAIN],pass=R>LINE_R2&&R>BASE_R2&&a>PASSLINE&&a>BASE_ACC;
    return {result:pass?'pass':'hold',flyAns:'회귀 R² '+fmtR(R)+' · MAE '+fmtM(E)+'MPa · 분류 '+pct(a)+' · 나무 수는 훈련 데이터의 검증 몫으로 정함',roboAns:'회귀 나무 300그루 · 외울 숫자 '+M.nums.all.toLocaleString()+'개 · 학습 '+Math.round(M.ms)+'ms(이 브라우저) · 예측 1건에 질문 최대 900번',
      flyMood:pass?'type':'coffee',flyMove:pass?'nod':'shake',roboMood:pass?'wow':'cry',roboMove:pass?'jump':'shake',asstMood:pass?'spark':'sigh',docMood:pass?'verdict':'scale',
      measure:'봉인된 시험지 결과예요. 회귀 <b>R² '+fmtR(R)+'</b>(MAE '+fmtM(E)+'MPa)로 합격선 '+fmtR(LINE_R2)+'보다 '+fx(R-LINE_R2,3)+' 높고, 분류 <b>'+pct(a)+'</b>로 합격선 '+pct(PASSLINE)+'보다 '+fx((a-PASSLINE)*100,1)+'%p 높아요. 지금까지 지원자 중 회귀 R²가 가장 높습니다(06장 중간 정산).',
      verdict:pass?'합격입니다. 겸용팀에 배치합니다. 조건이 둘 있어요. 나무 수와 학습률은 늘 검증 몫으로 다시 고를 것, 데이터가 크게 늘면 HistGradientBoosting 같은 빠른 구현으로 옮길 것.':'보류입니다. 합격선 하나를 넘지 못했어요.',
      key:'오답 노트 릴레이 300그루 — 회귀 R² '+fmtR(LINE_R2)+' → '+fmtR(R)+' · 분류 '+pct(PASSLINE)+' → '+pct(a),
      render(box){box.innerHTML='<div class="bars">'+barRows([['GBM 회귀 R²',R,'var(--blue)',fmtR(R)],['매뉴얼 경험식',LINE_R2,'var(--c-man)',fmtR(LINE_R2)],['기준선 (늘 평균)',Math.max(0,BASE_R2),'var(--muted)',fmtR(BASE_R2)],['GBM 분류 정확도',a,'var(--blue)',pct(a)],['매뉴얼 씨 (분류)',PASSLINE,'var(--c-man)',pct(PASSLINE)],['기준선 (늘 불합격)',BASE_ACC,'var(--muted)',pct(BASE_ACC)]],1)+'</div>';}};},
  alts:[
   {label:'나무 10그루에서 멈추면',async run(box){const M=ensureMain(),K=ensureCls(),R=M.teR2[10],a=K.acc[10];
     box.innerHTML=statGrid([['R² '+fmtR(R),'시험 R² (MAE '+fmtM(mae(YTE,M.teS[10]))+')','a'],['R² '+fmtR(LINE_R2),'회귀 합격선','m'],[pct(a),'분류 정확도','a'],[pct(PASSLINE),'분류 합격선','m']],4)+'<p class="foot">같은 지원자인데 덧대기를 일찍 멈췄다. 분류는 합격선을 넘지만 회귀는 합격선 아래다. 판정: '+(R>LINE_R2&&a>PASSLINE?/*i18n:verdict*/'합격':'<b>보류</b>')+'. η = 0.1이면 한 그루가 남은 오차의 10%쯤만 메우므로 나무가 적으면 예측이 평균 쪽에 눌려 있다(과소적합). 학습률과 나무 수는 한 쌍이다.</p>';}},
   {label:'학습률 1.0 (한 그루를 통째로 믿으면)',async run(box){const t0=performance.now();let trR=0;const M=GB.fit(A,YTR,{n:NMAIN,lr:1,depth:3,onStage:(t,F)=>{if(t===NMAIN)trR=r2(YTR,F[0]);}}),p=B.map(x=>GB.raw(M,x)[0]),R=r2(YTE,p),R1=VAL.run(1);
     box.innerHTML=statGrid([['R² '+fmtR(R),'시험 R² (300그루, η 1.0)','a'],['R² '+fmtR(trR),'훈련 R²'],['R² '+fmtR(MAIN.teR2[NMAIN]),'η 0.1 (본 조건)','s']],3)+'<p class="foot">훈련 R²는 거의 1인데 시험 R²는 본 조건보다 '+fx(MAIN.teR2[NMAIN]-R,3)+' 낮다. 합격선은 넘으니 판정은 '+(R>LINE_R2?/*i18n:verdict*/'합격':'보류')+'이지만, 한 그루의 답을 통째로 믿어 시편의 우연까지 외운 몫이 크다. 04장 검증 곡선에서도 η = 1.0은 '+R1.best+'그루에서 최고('+fmtR(R1.mx)+')를 찍고 더 오르지 않았다. 학습 '+Math.round(performance.now()-t0)+'ms.</p>';}},
   {label:'3등급 분류 (단계마다 등급별 나무 3그루)',async run(box){const G=ensureG3(),P=B.map(x=>GB.predict(G,x)),M3=ML.confusion(GTE,P,3);
     box.innerHTML='<div class="tw">'+cmGrid(M3,GRADES,k=>GRADECOL[k])+'</div>'+statGrid([[pct(G.acc),'시험 정확도 (100단계 × 3그루)','a'],[pct(BASE3),'기준선 · 늘 보통 강도']],2)+'<p class="foot">등급마다 "이 등급일 로그 오즈"를 따로 덧대고 소프트맥스로 확률을 낸다. 단계마다 나무가 세 그루라 100단계면 300그루다. 고강도를 보통으로, 저강도를 보통으로 본 칸이 주로 틀린다. 등급 경계(25·45MPa) 근처의 배합이다.</p>';}}]});

/* ── 06 인사 카드: 같은 집안의 구현 · 중간 정산 ── */
function renderCard(){const M=ensureMain(),K=ensureCls(),G=ensureG3(),R=M.teR2[NMAIN],E=mae(YTE,M.teS[NMAIN]),a=K.acc[NMAIN];
  const I=[['<b>GradientBoosting</b> (scikit-learn) · 이 면접의 지원자','정렬한 값 사이 전부를 분할 후보로 본다. 정확하지만 데이터가 수만 건을 넘으면 느리다. 나무는 깊이로 크기를 정한다.','회귀 R² '+fmtR(R)+' (MAE '+fmtM(E)+') · 분류 '+pct(a)+' · 브라우저 계산'],
    ['<b>HistGradientBoosting</b> (scikit-learn)','특징 값을 최대 255칸으로 묶어 칸 경계만 분할 후보로 본다(히스토그램). 수만 건 이상에서 훨씬 빠르고, 빈칸과 범주형 특징을 바로 받는다. 데이터가 1만 건을 넘으면 조기 종료가 기본으로 켜진다.','회귀 R² '+fmtR(PRE.hgb.r2)+' (MAE '+PRE.hgb.mae.toFixed(2)+') · 분류 '+pct(PRE.hgb.acc)+' · Python 계산(기본 설정)'],
    ['<b>XGBoost</b> (Chen &amp; Guestrin, 2016)','손실의 기울기와 곡률(2차 근사)로 잎 값과 분할을 정하고, 잎 수와 잎 값에 규제를 건다. 빈칸이 갈 방향을 데이터에서 배운다. 여러 코어·GPU로 분할 찾기를 나눠 한다.','이 빌드에서 실행 안 함'],
    ['<b>LightGBM</b> (Ke 등, 2017)','히스토그램에 더해, 오차를 가장 많이 줄이는 잎부터 키운다(잎 우선 성장). 잔차가 큰 표본 위주로 뽑고(GOSS), 함께 0인 희소 특징을 묶어(EFB) 빠르다. 나무 크기를 잎 수(num_leaves)로 정한다.','이 빌드에서 실행 안 함'],
    ['<b>CatBoost</b> (Prokhorenkova 등, 2018)','글자로 된 범주형 특징을 "앞선 행들의 타겟 평균"으로 바꿔 누수를 막고, 순서 부스팅으로 예측 편향을 줄인다. 층마다 같은 질문을 하는 대칭 나무를 쓴다.','이 빌드에서 실행 안 함']];
  $('implTbl').innerHTML=I.map(r=>'<tr><td style="min-width:140px">'+r[0]+'</td><td>'+r[1]+'</td><td style="min-width:120px">'+r[2]+'</td></tr>').join('');
  const L=PRE.ledger.concat([['5-9','GBM','나무 300그루, η 0.1, 깊이 3',a,G.acc,R,E,(R>LINE_R2&&a>PASSLINE)?'pass':'hold',1]]);
  const best=k=>Math.max(...L.map(r=>r[k]==null?-1:r[k]));const bA=best(3),b3=best(4),bR=best(5);
  const cell=(v,b,fm)=>v==null?'<td class="muted">—</td>':'<td'+(v===b?' style="font-weight:700;color:var(--ok)"':'')+'>'+fm(v)+'</td>';
  const st=s=>'<td><span class="pill '+(s==='pass'?'ok':'no')+'">'+(s==='pass'?/*i18n:verdict*/'합격':'보류')+'</span></td>';
  $('ledger').innerHTML='<thead><tr><th>편</th><th>지원자</th><th>기준 모델</th><th>분류 35MPa</th><th>3등급</th><th>회귀 R² (MAE)</th><th>판정</th></tr></thead><tbody>'+
    L.map(r=>'<tr'+(r[8]?' style="background:var(--panel2)"':'')+'><td>'+r[0]+'</td><td><b style="color:var(--text)">'+r[1]+'</b></td><td style="font-size:13px">'+r[2]+'</td>'+cell(r[3],bA,pct)+cell(r[4],b3,pct)+(r[5]==null?'<td class="muted">— (분류 전용)</td>':'<td'+(r[5]===bR?' style="font-weight:700;color:var(--ok)"':'')+'>'+fmtR(r[5])+(r[6]!=null?' ('+r[6].toFixed(2)+')':'')+'</td>')+st(r[7])+'</tr>').join('')+
    '<tr><td colspan="7" class="muted" style="font-size:12.5px;padding-top:12px">앞선 지원자 (참고)</td></tr>'+
    PRE.earlier.map(r=>'<tr class="muted" style="font-size:13px"><td>'+r[0]+'</td><td>'+r[1]+'</td><td>'+r[2]+'</td><td>'+(r[3]==null?'—':pct(r[3]))+'</td><td>—</td><td>'+(r[5]==null?'—':fmtR(r[5]))+'</td><td>'+(r[7]==='pass'?/*i18n:verdict*/'합격':'보류')+'</td></tr>').join('')+
    '<tr><td colspan="3"><b>합격선</b> · 매뉴얼 씨</td><td>'+pct(PASSLINE)+'</td><td>기준선 '+pct(BASE3)+'</td><td>'+fmtR(LINE_R2)+' ('+fmtM(LINE_MAE)+')</td><td></td></tr></tbody>';
  $('ledgerFoot').innerHTML='초록 굵은 글씨가 그 열의 1등이다. 5-9 줄만 이 브라우저가 계산했고(분류·3등급·회귀 모두), 5-5 ~ 5-8과 앞선 지원자 줄은 각 편의 기준 모델로 Python(scikit-learn 1.9)이 미리 계산한 숫자다(A장 12번 칸이 같은 값을 다시 낸다). 5-9의 회귀 R²는 Python에서 0.949(MAE 2.78)다. 브라우저 사본의 강도가 소수 둘째 자리로 반올림되어 있어 셋째 자리가 조금 다르다. 겸용팀 넷(5-6 ~ 5-9)은 회귀와 분류를 모두 했고, 5-5 나이브 베이즈는 분류 전용이다. 나무 집안(5-6·5-7·5-9)이 위쪽을 차지했다. 이 데이터처럼 특징이 수십 개 이하인 표에서 흔한 모습이다.';}
whenNear('m6',renderCard,600);

/* ══ 확인 문제 ══ */
var QUIZ={
q1:[{q:'GBM의 두 번째 나무가 배우는 것은?',o:[['첫 나무까지의 예측이 아직 틀린 만큼(잔차)',1,'다음 나무의 타겟은 실제 값이 아니라 "실제 − 지금까지의 예측"이다. 오답 노트를 넘겨받는 셈이다.'],['원래 타겟(강도) 그대로',0,'그것은 랜덤 포레스트의 나무들이 하는 일이다. 나무마다 같은 타겟을 따로 배운다.'],['첫 나무가 맞힌 배합만',0,'모든 배합을 쓴다. 다만 잘 맞힌 배합은 잔차가 작아 다음 나무에게 거의 영향을 주지 않는다.']]},
 {q:'GBM과 랜덤 포레스트의 차이로 맞는 것은?',o:[['GBM은 얕은 나무를 차례로 더하고, 랜덤 포레스트는 깊은 나무를 나란히 평균한다',1,'부스팅은 치우침(편향)을, 배깅·랜덤 포레스트는 흔들림(분산)을 주로 줄인다.'],['GBM은 나무를 하나만 쓴다',0,'수백 그루를 쓴다. 이 면접은 300그루다.'],['둘은 이름만 다르고 같은 방법이다',0,'나무를 세우는 순서와 답을 모으는 방법이 다르다.']]},
 {q:'학습률 η = 0.1의 뜻은?',o:[['새 나무의 답을 10%만 예측에 더한다',1,'한 그루를 다 믿지 않고 조금씩 고친다. 대신 나무가 더 필요하다.'],['훈련 데이터의 10%만 쓴다',0,'그것은 subsample(서브샘플)이다.'],['나무 깊이를 10%로 줄인다',0,'깊이는 max_depth가 정한다.']]}],
q2:[{q:'GBM 앞에서 특징을 표준화하면 예측은?',o:[['그대로다. 나무는 값의 크기 순서로 문턱을 고르기 때문이다',1,'02장에서 확인했다. 문턱의 숫자만 표준화 단위로 바뀐다. KNN·SVM과 다른 점이다.'],['좋아진다',0,'나무에게는 아무것도 바뀌지 않는다.'],['나빠진다',0,'바뀌지 않는다. 다른 지원자와 같은 파이프라인에 넣어도 해롭지 않다.']]},
 {q:'GBM이 훈련 데이터에 없던 아주 낮은 물-결합재비(0.2)의 강도를 예측하면?',o:[['맨 끝 계단의 값을 그대로 답한다. 범위 밖으로 나가지 못한다',1,'나무의 답은 잎에 적힌 값의 합이다. 문턱 바깥은 가장 가까운 잎이 맡는다. 외삽을 못 한다.'],['곡선을 이어 더 높은 값을 낸다',0,'그것은 선형·다항 회귀의 방식이다.'],['오류가 난다',0,'오류 없이 끝 계단의 값을 낸다. 그래서 더 위험할 수 있다.']]}],
q3:[{q:'GradientBoostingRegressor(loss="ls")를 지금 버전에서 실행하면?',o:[['"ls"는 지워진 이름이라 오류가 난다. "squared_error"를 쓴다',1,'1.0에서 예고, 1.2에서 지워졌다. 분류의 "deviance"는 "log_loss"가 되었다.'],['최소제곱으로 잘 돈다',0,'옛 버전에서만 됐다.'],['자동으로 분류 손실로 바뀐다',0,'바뀌지 않고 오류가 난다.']]},
 {q:'staged_predict(X_test)로 시험 점수가 가장 높은 나무 수를 골라 보고했다. 문제는?',o:[['시험지에 맞춘 나무 수라 처음 보는 데이터에서는 더 낮게 나온다',1,'나무 수는 훈련 데이터에서 뗀 검증 몫이나 n_iter_no_change로 고른다.'],['문제없다. staged_predict는 그러라고 있는 도구다',0,'도구는 좋지만 쓰는 데이터가 틀렸다.'],['계산이 너무 느리다',0,'속도 문제가 아니라 점수가 부풀려지는 문제다.']]},
 {q:'부스팅에 max_depth=None(끝까지 키운 나무)을 쓰면?',o:[['첫 나무가 훈련 데이터를 외워 잔차가 사라지고, 시험 성적이 떨어진다',1,'A장 8번 칸: 깊이 3은 시험 R² 0.949, 끝까지 키우면 0.890. 부스팅의 부품은 약한(얕은) 나무다.'],['더 정확해진다',0,'훈련 점수만 오른다.'],['오류가 난다',0,'오류 없이 돈다. 그래서 알아채기 어렵다.']]}],
q4:[{q:'나무를 덧댈수록 훈련 R²가 계속 오르는 까닭은?',o:[['나무는 늘 남은 오차를 배우므로, 끝까지 가면 시편의 우연까지 메운다',1,'그래서 훈련 R²로는 멈출 때를 정할 수 없다. 검증 몫으로 잰다.'],['모델이 점점 일반화를 잘해서',0,'검증 R²는 어느새 평평해지거나 내려간다.'],['학습률이 점점 커져서',0,'학습률은 고정이다.']]},
 {q:'학습률을 0.1에서 0.03으로 줄이면 대개?',o:[['같은 성적까지 나무가 더 많이 필요하다',1,'04장 그래프: η = 0.03은 600그루에서도 아직 오르는 중이다. 학습률과 나무 수는 한 쌍이다.'],['나무가 덜 필요하다',0,'걸음이 작아지니 더 많이 걸어야 한다.'],['결과가 똑같다',0,'곡선의 모양과 필요한 나무 수가 달라진다.']]},
 {q:'조기 종료의 검증 몫은 어디서 떼나?',o:[['훈련 데이터에서',1,'scikit-learn의 validation_fraction도 훈련 데이터에서 뗀다. 시험지는 마지막에 한 번만 연다.'],['시험 데이터에서',0,'그러면 시험지로 나무 수를 고르는 셈이다.'],['훈련과 시험을 합쳐서',0,'시험지 정보가 새어 들어간다.']]}],
q5:[{q:'GBM 분류기(35MPa 합격) 안의 나무는?',o:[['회귀 나무. 로그 오즈에 더할 숫자를 맞힌다',1,'잔차 = 실제(0/1) − 확률을 회귀 나무로 배우고, 덧댄 로그 오즈를 시그모이드로 확률로 바꾼다.'],['분류 나무. 잎마다 합격/불합격을 적는다',0,'GradientBoostingClassifier의 estimators_는 DecisionTreeRegressor다(A장 9번 칸).'],['나무를 쓰지 않는다',0,'나무를 쓴다.']]},
 {q:'분류 GBM의 시작값 F₀는?',o:[['log(합격 비율 ÷ 불합격 비율)',1,'아무것도 모를 때의 로그 오즈다. 훈련 데이터의 합격 비율 0.488이면 F₀ ≈ −0.05다.'],['0.5',0,'확률 0.5는 로그 오즈 0이다. 시작값은 훈련 데이터의 합격 비율에서 나온다.'],['강도의 평균',0,'그것은 회귀 GBM의 시작값이다.']]},
 {q:'GBM과 랜덤 포레스트 모두 특징 중요도 1·2위가 재령과 시멘트였다. 맞는 해석은?',o:[['두 특징으로 나눌 때 오차가 가장 많이 줄었다는 뜻이다. 원인이라는 뜻은 아니다',1,'중요도는 분할이 줄인 오차의 몫이다. 얽힌 특징끼리는 몫을 나눠 갖기도 한다.'],['시멘트를 늘리면 반드시 강해진다',0,'중요도는 방향(늘리면 강해지나)을 말하지 않는다.'],['나머지 특징은 빼도 된다',0,'몫이 작아도 성적에 보탬이 될 수 있다. 빼고 다시 재 봐야 안다.']]}],
q6:[{q:'HistGradientBoosting이 GradientBoosting보다 큰 데이터에서 빠른 까닭은?',o:[['특징 값을 최대 255칸으로 묶어 칸 경계만 분할 후보로 보기 때문',1,'정렬된 값 사이 전부 대신 몇백 개 경계만 본다. LightGBM·XGBoost도 비슷한 방법을 쓴다.'],['나무를 하나만 키워서',0,'여러 그루를 차례로 키우는 것은 같다.'],['GPU를 꼭 써서',0,'CPU에서도 빠르다.']]},
 {q:'XGBoost·LightGBM·CatBoost와 scikit-learn GBM의 관계는?',o:[['잔차(기울기)에 나무를 덧댄다는 같은 원리의 다른 구현이다',1,'분할 찾기, 규제, 범주형 처리, 나무 키우는 순서가 다르다. 인자 이름도 다르니 문서로 확인한다.'],['전혀 다른 알고리즘이다',0,'같은 집안이다(4-10 앙상블 계보).'],['딥러닝 모델이다',0,'나무 앙상블이다.']]}]};

/* ══ 공통 B·C·D·E·F 데이터 ══ */
var PROMPTS=[
 {t:'① 내 공정 데이터로 GBM 코드 받기',use:'내 공장의 품질 데이터로 03장의 프롬프트를 맞출 때',p:MAIN_PROMPT},
 {t:'② AI 코드 검증 요청',use:'받은 코드를 다른 AI(또는 같은 AI의 새 대화)에게 검토받을 때',p:'아래 scikit-learn 그레이디언트 부스팅 코드를 코드 리뷰해 줘. 특히 다음을 하나씩 확인하고, 문제가 있으면 줄 번호와 고친 코드를 보여 줘.\n1) 타겟이 연속값인데 GradientBoostingClassifier를 쓰지 않았나 (또는 그 반대)\n2) loss, criterion 같은 인자 이름이 내 버전(〔1.x〕)에 있는 이름인가 ("ls", "deviance", "mse"는 지금 오류)\n3) train_test_split에 stratify와 random_state가 있나\n4) max_depth가 얕은가 (None이면 부스팅이 제 구실을 못 한다)\n5) 나무 수(n_estimators)와 learning_rate를 시험 데이터 점수(staged_predict 포함)로 고르지 않았나. 검증 몫이나 조기 종료를 썼나\n6) 결과를 기준선(DummyRegressor/DummyClassifier)과 기존 규칙과 비교했나\n추측으로 답하지 말고, 확실하지 않은 부분은 "공식 문서 확인 필요"라고 표시해 줘.'},
 {t:'③ XGBoost·LightGBM과 같은 시험지로 비교하기',use:'Colab에서 같은 집안의 다른 구현을 직접 돌려 볼 때',p:'아래 Colab 코드(scikit-learn GradientBoosting)와 똑같은 데이터 나누기(train_test_split(test_size=0.25, stratify=강도 3등급, random_state=42))를 그대로 쓰고, 같은 시험지에서 다음 넷을 비교하는 셀을 덧붙여 줘.\n- sklearn GradientBoostingRegressor(n_estimators=300, learning_rate=0.1, max_depth=3)\n- sklearn HistGradientBoostingRegressor(기본값)\n- xgboost.XGBRegressor, lightgbm.LGBMRegressor (Colab에 설치된 버전을 먼저 출력한다. 없으면 설치 명령을 따로 적는다)\n규칙:\n1) 학습률·나무 수는 훈련 데이터에서 뗀 검증 몫으로 조기 종료한다. 시험 데이터는 마지막 채점에만 쓴다\n2) 각 라이브러리의 조기 종료 인자 이름은 설치된 버전의 공식 문서 기준으로 쓰고, 어느 문서를 따랐는지 주석으로 남긴다\n3) 표: 시험 R², MAE, 학습 시간(초), 멈춘 나무 수\n4) 점수 차이가 0.01보다 작으면 "이 데이터에서는 비슷하다"고 해석한다'},
 {t:'④ 수업용 문제 만들기',use:'이 편으로 수업할 때 확인 문제가 더 필요할 때',p:'그레이디언트 부스팅 입문 수업용 문제를 만들어 줘. 대상은 〔제조 현장 실무자, 수식에 익숙하지 않음〕이다. 예시는 콘크리트 강도로 통일한다.\n- 잔차·학습률·나무 수·조기 종료를 각각 묻는 객관식 4문제 (보기 3개, 정답 해설 포함)\n- 오해를 드러내는 문제 2개: "나무가 많을수록 늘 좋다", "분류 GBM 안에는 분류 나무가 있다"\n- 손으로 풀 수 있는 계산 문제 1개: 배합 다섯 개의 강도와 시작값(평균)으로 잔차를 구하고, 문턱 하나짜리 나무(잎 2개)의 잎 값을 잔차 평균으로 정한 뒤, 학습률 0.1로 한 번 덧댄 예측을 구하기 (소수 첫째 자리)\n정답은 맨 끝에 따로 모아 줘.'}];
var GL=[
 ['모델','GBM','Gradient Boosting Machine. 얕은 나무를 차례로 세워, 나무마다 앞까지의 잔차(손실의 기울기 반대 방향)를 배우게 하고 학습률만큼 덧대는 앙상블 모델. 그레이디언트 부스팅.'],
 ['모델','부스팅','약한 학습기를 차례로 이어, 뒤의 것이 앞의 것들이 틀린 부분을 보완하게 해 강한 모델을 만드는 방법. AdaBoost, 그레이디언트 부스팅이 있다.'],
 ['모델','앙상블','여러 모델의 답을 모아 하나의 답을 내는 방법. 나란히 평균하면 배깅·랜덤 포레스트, 차례로 더하면 부스팅.'],
 ['모델','약한 학습기','혼자서는 기준선보다 조금 나은 정도의 단순한 모델. GBM에서는 깊이 2~5의 얕은 나무.'],
 ['모델','결정 트리','"특징 ≤ 문턱?" 같은 질문을 이어 데이터를 칸으로 나누고, 칸마다 답을 적는 모델. 회귀 나무는 칸의 평균을 답한다.'],
 ['모델','잎','나무 맨 끝의 칸. 그 칸에 떨어진 데이터의 답(회귀 나무에서는 평균)을 적어 둔다.'],
 ['모델','HistGradientBoosting','scikit-learn의 히스토그램 기반 GBM. 특징 값을 최대 255칸으로 묶어 분할을 빨리 찾는다. 빈칸과 범주형 특징을 받는다.'],
 ['모델','XGBoost','2차 근사와 규제, 병렬 분할 찾기를 더한 그레이디언트 부스팅 라이브러리(Chen & Guestrin, 2016).'],
 ['모델','LightGBM','히스토그램과 잎 우선 성장, GOSS·EFB로 빠르게 만든 그레이디언트 부스팅 라이브러리(Ke 등, 2017).'],
 ['모델','CatBoost','범주형 특징 처리와 순서 부스팅에 강한 그레이디언트 부스팅 라이브러리(Prokhorenkova 등, 2018).'],
 ['학습','잔차','실제 값 − 지금까지의 예측. 제곱 오차 손실에서는 손실의 기울기를 뒤집은 것과 같다. 다음 나무의 타겟.'],
 ['학습','학습률','새 나무의 답을 예측에 얼마나 반영할지 정하는 비율 η(보통 0.1). 작을수록 차분하지만 나무가 더 필요하다. 수축(shrinkage)이라고도 한다.'],
 ['학습','덧대기','지금까지의 예측에 새 나무의 답 × 학습률을 더하는 일. F ← F + η·h. 가법 모형.'],
 ['학습','손실 함수','예측이 얼마나 틀렸는지 재는 식. 회귀는 제곱 오차(squared_error), 분류는 로그 손실(log_loss).'],
 ['학습','기울기','손실이 예측을 조금 바꿀 때 얼마나 변하는지. 그 반대 방향으로 고치면 손실이 준다. GBM은 예측값 자체를 이 방향으로 고친다.'],
 ['학습','로그 오즈','log(확률 ÷ (1 − 확률)). 분류 GBM은 이 숫자에 나무를 덧대고 시그모이드로 확률로 바꾼다.'],
 ['학습','시그모이드','어떤 숫자든 0과 1 사이 확률로 바꾸는 S자 함수 σ(z) = 1 / (1 + e^(−z)).'],
 ['학습','조기 종료','검증 점수가 정한 횟수 동안 나아지지 않으면 학습을 멈추는 일. n_iter_no_change.'],
 ['학습','검증 몫','훈련 데이터에서 떼어 근무 조건(나무 수, 학습률)을 고르는 데 쓰는 부분. 시험지와 다르다.'],
 ['학습','서브샘플','나무마다 훈련 데이터의 일부만 무작위로 쓰는 일(subsample). 확률적 그레이디언트 부스팅.'],
 ['학습','과대적합','훈련 데이터를 외워 처음 보는 데이터에서 성적이 떨어지는 일. GBM에서는 나무가 너무 많거나 깊거나 학습률이 클 때.'],
 ['학습','과소적합','모델이 너무 단순해 훈련 데이터조차 못 맞히는 일. GBM에서는 나무가 너무 적을 때.'],
 ['평가','특징 중요도','분할이 줄인 오차를 특징별로 모은 몫. 합이 1. 원인이나 방향을 말하지 않는다.'],
 ['평가','R²','결정 계수. 늘 평균만 답할 때보다 오차 제곱을 얼마나 줄였나. 1이면 완벽, 0이면 평균과 같다.'],
 ['평가','MAE','평균 절대 오차. 예측이 실제와 평균 몇 MPa 다른지.'],
 ['평가','정확도','시험 데이터 중 맞힌 비율.'],
 ['평가','혼동행렬','정답과 예측을 가로세로로 놓고 건수를 센 표. 어느 쪽으로 틀렸는지 보인다.'],
 ['평가','기준선','배우지 않고도 낼 수 있는 점수. 회귀는 늘 훈련 평균, 분류는 늘 가장 흔한 답.'],
 ['평가','합격선','지원자가 넘어야 할 점수. 회귀는 매뉴얼 경험식의 R² 0.735, 분류는 5-0 매뉴얼 씨의 정확도 76.7%.']];
var ANALOGY=[
 ['잔차를 배운다','앞 사람이 틀린 문제만 모은 오답 노트를 받아, 그 문제만 다시 공부하는 다음 사람',2,'소뇌의 오차 기반 운동 학습: 동작이 빗나가면 오르기 섬유가 오차 신호를 보내 다음 동작을 조금 고친다는 이론(Marr, 1969 등)','소뇌는 같은 회로의 시냅스를 고쳐 쓴다. GBM은 고친 내용을 새 나무로 덧붙여 쌓고, 앞의 나무는 그대로 둔다. 오차 신호가 정확히 "실제 − 예측" 숫자라는 근거도 없다'],
 ['잔차를 배운다 (초파리)','"예상보다 나빴다"는 신호를 받아 판단을 고치는 신입',2,'초파리 버섯체에서 도파민 뉴런이 보상·처벌 신호를 보내 케니언 세포와 출력 뉴런 사이 시냅스를 바꾸는 연합 학습(Aso 등, 2014)','초파리의 도파민 신호가 "예측 오차"를 얼마나 정확히 나타내는지는 아직 연구 중이다. 버섯체는 냄새와 결과를 잇는 회로이고, 나무를 덧붙이는 구조가 아니다'],
 ['학습률','한 번 틀렸다고 규칙을 통째로 바꾸지 않고 조금씩만 고치는 신중한 고참',2,'사람이 환경이 자주 바뀔 때는 빨리, 안정적일 때는 천천히 배운다는 행동·영상 연구(Behrens 등, 2007)','뇌의 학습 속도는 상황에 따라 계속 바뀐다. GBM의 η는 사람이 정한 고정값이다'],
 ['로그 오즈에 덧대기 (분류)','판정 점수를 항목마다 더하다가 문턱을 넘으면 합격 도장을 찍는 심사',2,'뉴런 활동이 증거를 쌓다가 문턱에 닿으면 결정한다는 증거 누적 모형(Gold & Shadlen, 2007)','뇌의 증거 누적은 시간에 따라 잡음 섞인 신호를 더한다. GBM은 정해진 나무 300그루의 답을 한 번에 더하는 고정 계산이다'],
 ['얕은 나무 (약한 학습기)','질문 세 개만 할 줄 아는 신입 수백 명이 차례로 앞 사람의 빈틈을 메운다',1,'','뉴런 하나하나를 "약한 학습기"로 보고 차례로 더한다는 회로는 알려지지 않았다'],
 ['조기 종료','연습 문제집 점수가 아니라 처음 보는 모의고사 점수로 공부를 멈출 때를 정한다',1,'','직접 대응하는 뇌 기능은 알려지지 않았다. 뇌에는 "검증 몫"이 따로 없다']];
var REFS=[
 'Yeh, I.-C. (1998). Modeling of strength of high-performance concrete using artificial neural networks. <i>Cement and Concrete Research</i>, 28(12), 1797–1808. <a href="https://doi.org/10.1016/S0008-8846(98)00165-3" target="_blank" rel="noopener">doi:10.1016/S0008-8846(98)00165-3</a> — 실기 과제지 원출처',
 'Yeh, I.-C. (2007). Concrete Compressive Strength [Dataset]. UCI Machine Learning Repository. <a href="https://doi.org/10.24432/C5PK67" target="_blank" rel="noopener">doi:10.24432/C5PK67</a> (CC BY 4.0) — 02장 데이터',
 'Abrams, D. A. (1918). <i>Design of Concrete Mixtures</i> (Bulletin 1). Structural Materials Research Laboratory, Lewis Institute, Chicago. · ACI Committee 209 (1992). <i>ACI 209R-92</i>. American Concrete Institute. — 매뉴얼 씨의 회귀 합격선',
 'Freund, Y., &amp; Schapire, R. E. (1997). A decision-theoretic generalization of on-line learning and an application to boosting. <i>Journal of Computer and System Sciences</i>, 55(1), 119–139. — 부스팅(AdaBoost)의 출발점',
 'Friedman, J., Hastie, T., &amp; Tibshirani, R. (2000). Additive logistic regression: A statistical view of boosting. <i>The Annals of Statistics</i>, 28(2), 337–407. — 부스팅을 덧대기 모형과 로그 오즈로 본 관점, 05장',
 'Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. <i>The Annals of Statistics</i>, 29(5), 1189–1232. — 그레이디언트 부스팅, 학습률(수축), 01·04장',
 'Friedman, J. H. (2002). Stochastic gradient boosting. <i>Computational Statistics &amp; Data Analysis</i>, 38(4), 367–378. — 서브샘플, 06장',
 'Chen, T., &amp; Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In <i>Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining</i> (pp. 785–794). <a href="https://doi.org/10.1145/2939672.2939785" target="_blank" rel="noopener">doi:10.1145/2939672.2939785</a> — 06장',
 'Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., &amp; Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. In <i>Advances in Neural Information Processing Systems 30</i>. — 06장',
 'Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., &amp; Gulin, A. (2018). CatBoost: Unbiased boosting with categorical features. In <i>Advances in Neural Information Processing Systems 31</i>. — 06장',
 'Hastie, T., Tibshirani, R., &amp; Friedman, J. (2009). <i>The Elements of Statistical Learning</i> (2nd ed.), 10장 Boosting and Additive Trees. Springer. — 01·04장',
 'Grinsztajn, L., Oyallon, E., &amp; Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on typical tabular data? In <i>Advances in Neural Information Processing Systems 35</i> (Datasets and Benchmarks Track). — 06장 "표 데이터에서 자주 1등"',
 'Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. <i>Journal of Machine Learning Research</i>, 12, 2825–2830. — A장 코드',
 'scikit-learn 문서. <a href="https://scikit-learn.org/stable/modules/ensemble.html" target="_blank" rel="noopener">Ensembles: Gradient boosting, random forests, bagging, voting, stacking</a> · <a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html" target="_blank" rel="noopener">GradientBoostingRegressor</a> — loss 이름, 조기 종료, 03·06장',
 'Marr, D. (1969). A theory of cerebellar cortex. <i>The Journal of Physiology</i>, 202(2), 437–470. — D장 소뇌의 오차 기반 학습',
 'Aso, Y. et al. (2014). The neuronal architecture of the mushroom body provides a logic for associative learning. <i>eLife</i>, 3, e04577. — D장 초파리 버섯체',
 'Behrens, T. E. J., Woolrich, M. W., Walton, M. E., &amp; Rushworth, M. F. S. (2007). Learning the value of information in an uncertain world. <i>Nature Neuroscience</i>, 10(9), 1214–1221. — D장 학습 속도',
 'Gold, J. I., &amp; Shadlen, M. N. (2007). The neural basis of decision making. <i>Annual Review of Neuroscience</i>, 30, 535–574. — D장 증거 누적'];
var TIP={who:'머신러닝을 처음 배우는 대학생·직장인·제조 현장 실무자. 5-6 결정 트리(나무 한 그루)와 5-7 랜덤 포레스트(나란히 세운 나무)를 먼저 보면 좋다. 수식은 덧대기 F ← F + η·h 하나가 중심이다',
 goal:'① "잔차를 차례로 배운다"를 지원서 1장과 01장 그림으로 설명한다 ② 학습률과 나무 수가 한 쌍이며, 둘 다 검증 몫으로 고른다는 것을 곡선으로 보인다 ③ 분류 GBM도 안쪽은 회귀 나무(로그 오즈)임을 설명한다 ④ AI가 낸 코드에서 지워진 인자 이름, 끝까지 키운 나무, 시험지로 나무 수 고르기를 찾고, 회귀·분류 합격선과 비교해 판정한다',
 time:'읽기 25분 · 실습(01·04·05장 그림) 25분 · 틀린 곳 찾기 10분 · Colab 15분',
 plans:[['강의형 50분','00장 지원서 1장을 인쇄해 1~3번 칸을 함께 읽기 8분 → 01장 그림에서 학생들이 번갈아 "덧대기"를 누르며 빨간 막대가 줄어드는 것을 보기 10분(η = 1.0으로 30그루도 해 본다) → 04장 검증 곡선으로 "언제 멈추나" 토론 12분 → 05장 덧대기 재생과 로그 오즈 그림 10분 → 면접 결과를 먼저 맞히고 발표, "10그루에서 멈추면" 다시 보기 10분.'],['실무 OJT 90분','03장 프롬프트로 각자 AI에게 코드를 받는다 20분 → 체크리스트로 서로의 코드를 바꿔 검사 20분 → Colab에서 실행하고 A장 결과와 비교 20분 → B장 ③ 프롬프트로 XGBoost·LightGBM·HistGradientBoosting을 같은 시험지에서 비교하고 "점수 차이가 작을 때 무엇으로 고르나"(속도, 설명, 유지보수) 토론 30분.'],['1:1 튜터링','01장에서 나무 한 그루를 덧댈 때마다 "이 나무가 무엇을 배웠나(아래 그림)"와 "예측이 어떻게 바뀌었나(위 그림)"를 말로 설명하게 한다. 이어 05장 로그 오즈 그림에서 배합 하나를 골라 F₀ → F → 확률 → 판정을 손으로 따라가게 한다.']],
 myths:[['나무가 많을수록 늘 좋다','훈련 성적만 늘 오른다. 검증 성적은 평평해지거나 내려간다. 학습률과 한 쌍으로, 검증 몫으로 고른다'],['GBM은 랜덤 포레스트의 빠른 판이다','나란히 평균(랜덤 포레스트)과 차례로 덧대기(GBM)는 다른 방법이다. GBM은 오히려 학습을 나눠 맡기 어렵다'],['분류 GBM 안에는 분류 나무가 있다','회귀 나무가 로그 오즈에 더할 숫자를 맞힌다. 시그모이드(또는 소프트맥스)가 확률로 바꾼다'],['GBM도 표준화해야 한다','문턱으로 일하므로 표준화가 예측을 바꾸지 않는다(02장)'],['깊은 나무를 쓰면 더 정확하다','부스팅은 얕은 나무가 기본이다. 끝까지 키우면 첫 나무가 외워 버려 시험 R²가 0.949 → 0.890으로 떨어진다'],['특징 중요도 1위가 원인이다','오차를 많이 줄인 분할에 쓰였다는 뜻일 뿐, 방향도 원인도 말하지 않는다']],
 tasks:[['과제 1 · 손으로 덧대기','배합 다섯 개(강도 20, 30, 35, 50, 60MPa)와 특징 하나로 시작값(평균) → 잔차 → 문턱 하나짜리 나무의 잎 값 → η = 0.1로 한 번 덧댄 예측을 표로 계산한다. 01장 그림과 같은 순서다.'],['과제 2 · 틀린 곳 찾기 만들기','AI에게 일부러 실수가 섞인 GBM 코드를 받고(B장 ②를 거꾸로 활용), 짝에게 찾게 한다. 실수마다 "왜 틀렸나"를 한 줄로 쓴다. 버전에 따라 지워진 인자 이름을 하나 이상 넣는다.'],['과제 3 · 학습률 × 나무 수 지도','A장 6번 칸을 고쳐 학습률(0.3, 0.1, 0.03)과 깊이(2, 3, 5)의 조합마다 검증 R² 최고점과 그 나무 수를 표로 만든다. 시험지는 마지막에 고른 조합 하나로만 연다.'],['과제 4 · 같은 집안 비교','B장 ③ 프롬프트로 XGBoost·LightGBM·HistGradientBoosting을 같은 시험지에서 돌리고, 점수·시간·멈춘 나무 수를 비교한다. 06장의 중간 정산표에 한 줄씩 더해 본다.']],
 ai:'학생이 AI에게 코드를 받는 것은 막지 말고 과정에 넣는다. GBM은 라이브러리마다 인자 이름이 달라 AI가 섞어 쓰기 쉬운 주제라 검증 연습에 특히 좋다. ① 프롬프트를 제출하게 하고 ② 체크리스트 아홉 항목을 직접 표시하게 하고 ③ 실행 결과를 기준선·합격선과 비교한 문장을 쓰게 한다. AI 답의 함수·인자는 설치된 버전의 공식 문서로 확인하게 하고, 과제에 AI 도움을 받았으면 사용 범위를 밝히게 한다.'};
var CODE={title:'GBM 면접 — 오답 노트 릴레이부터 합격 판정까지',sub:'numpy, pandas, scikit-learn, ucimlrepo · CPU 10초 안팎',file:'gbm_interview',
 desc:'이 편의 면접을 그대로 옮겼다. UCI에서 콘크리트 데이터를 받아 시험지를 봉인하고(1·2번 칸), 매뉴얼 씨의 두 합격선을 세우고(3번), 부스팅을 직접 짜서 scikit-learn과 맞춰 보고(4번), 덧대기 재생(5번), 검증 몫으로 학습률 × 나무 수 고르기(6번), 조기 종료(7번), 나무 깊이(8번), 분류의 로그 오즈(9번), 특징 중요도(10번), 빠른 구현 HistGradientBoosting(11번), 지원자 다섯 명의 중간 정산(12번), 판정(13번)까지 간다. 노트북으로 저장하면 칸마다 셀이 나뉜다.',
 cap:'실행 결과 (scikit-learn 1.9, CPU). 시간(ms)은 컴퓨터마다 다르다. 브라우저 계산과 같은 결론(회귀 R² 0.949 · 분류 93.4% 합격, 10그루면 회귀 0.640 보류, 3등급 86.4%)이 나온다. 브라우저의 콘크리트 사본은 강도가 소수 둘째 자리로 반올림되어 있어 회귀 300그루가 R² 0.950(MAE 2.72)으로 셋째 자리가 조금 다르고, 6번 칸의 검증 곡선 숫자(최고점의 나무 수 등)도 04장 그림과 조금 다르다. 곡선의 모양과 결론은 같다.'};
