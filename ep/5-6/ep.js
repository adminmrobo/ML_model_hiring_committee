var BOARD_LINE='R² 0.735 · 76.7%'; /* 회귀 합격선(매뉴얼 경험식 R²) · 분류 합격선 */
/* ══ 5-6 결정 트리 · 콘크리트 실기 과제지 (분류 + 회귀) ══════════════════════ */
/* ── TREE 엔진 시작 (이 편 전용, 브라우저·node 공용) ──
   scikit-learn의 CART(BestSplitter, 깊이 우선)를 따라 짰다.
   · 분류: 지니 불순도 1 − Σp². 회귀: 평균 제곱 오차(MSE). 둘 다 "자식 불순도의 가중 합"이 가장 작은 질문을 고른다
   · 문턱 후보: 값이 다른 이웃 두 값의 한가운데. 더 좋은 질문만 바꾼다(같으면 먼저 본 것 = 특징 번호가 작은 것, 같은 특징이면 작은 문턱)
   · 멈춤: 깊이 상한, 잎의 최소 샘플 수, 순수한 마디, 더 나눌 수 없는 마디(같은 배합만 남음)
   · 가지치기(ccp_alpha): 가장 약한 고리부터 잘라 낸다(최소 비용 복잡도 가지치기)
   scikit-learn은 같은 점수의 질문이 여럿이면 특징을 무작위 순서로 보므로 random_state에 따라 깊은 마디가 조금 다를 수 있다 */
const TREE=(function(){
  function fit(X,y,o){o=o||{};const reg=o.task==='reg',K=o.K||2,maxD=o.maxDepth==null?Infinity:o.maxDepth,minLeaf=o.minLeaf||1,minSplit=Math.max(o.minSplit||2,2*minLeaf),d=X[0].length,feats=o.feats||[...Array(d).keys()];
    const nodes=[];
    function stats(idx){if(reg){let s=0,q=0;idx.forEach(i=>{s+=y[i];q+=y[i]*y[i];});const m=s/idx.length;return {value:m,imp:Math.max(0,q/idx.length-m*m)};}
      const c=new Array(K).fill(0);idx.forEach(i=>c[y[i]]++);let sq=0;c.forEach(v=>sq+=v*v);return {value:c,imp:1-sq/(idx.length*idx.length)};}
    function best(idx){const n=idx.length;let bp=-Infinity,bf=-1,bt=0,bL=null,bR=null;
      for(const f of feats){const s=idx.slice().sort((a,b)=>X[a][f]-X[b][f]||a-b);
        if(reg){let tot=0;s.forEach(i=>tot+=y[i]);let sl=0;
          for(let p=0;p<n-1;p++){sl+=y[s[p]];const nl=p+1,nr=n-nl;if(X[s[p+1]][f]<=X[s[p]][f]+1e-7)continue;if(nl<minLeaf||nr<minLeaf)continue;
            const sr=tot-sl,pr=sl*sl/nl+sr*sr/nr;if(pr>bp){bp=pr;bf=f;bt=(X[s[p]][f]+X[s[p+1]][f])/2;bL=nl;}}}
        else{const tot=new Array(K).fill(0);s.forEach(i=>tot[y[i]]++);const cl=new Array(K).fill(0);
          for(let p=0;p<n-1;p++){cl[y[s[p]]]++;const nl=p+1,nr=n-nl;if(X[s[p+1]][f]<=X[s[p]][f]+1e-7)continue;if(nl<minLeaf||nr<minLeaf)continue;
            let ql=0,qr=0;for(let k=0;k<K;k++){ql+=cl[k]*cl[k];const r=tot[k]-cl[k];qr+=r*r;}
            const pr=-(nr*(1-qr/(nr*nr)))-(nl*(1-ql/(nl*nl)));if(pr>bp){bp=pr;bf=f;bt=(X[s[p]][f]+X[s[p+1]][f])/2;bL=nl;}}}}
      if(bf<0)return null;const L=[],R=[];idx.forEach(i=>(X[i][bf]<=bt?L:R).push(i));return {f:bf,t:bt,L,R};}
    function grow(idx,depth,parent){const st=stats(idx),id=nodes.length,nd={id,n:idx.length,depth,value:st.value,imp:st.imp,leaf:true,parent,idx};nodes.push(nd);
      if(depth<maxD&&idx.length>=minSplit&&st.imp>2.220446049250313e-16){const b=best(idx);
        if(b){nd.leaf=false;nd.f=b.f;nd.t=b.t;nd.left=grow(b.L,depth+1,id);nd.right=grow(b.R,depth+1,id);}}
      return id;}
    grow([...Array(X.length).keys()],0,-1);
    return finish({nodes,reg,K,d});}
  function finish(T){T.nodes.forEach(nd=>{nd.pred=T.reg?nd.value:argmax(nd.value);});T.depth=0;T.leaves=0;walk(T,0,nd=>{if(nd.leaf){T.leaves++;T.depth=Math.max(T.depth,nd.depth);}});return T;}
  function argmax(c){let m=0;for(let k=1;k<c.length;k++)if(c[k]>c[m])m=k;return m;}
  function walk(T,id,fn){const nd=T.nodes[id];fn(nd);if(!nd.leaf){walk(T,nd.left,fn);walk(T,nd.right,fn);}}
  function leafOf(T,x,maxDepth){let nd=T.nodes[0];while(!nd.leaf&&(maxDepth==null||nd.depth<maxDepth))nd=T.nodes[x[nd.f]<=nd.t?nd.left:nd.right];return nd;}
  function predict(T,x,maxDepth){return leafOf(T,x,maxDepth).pred;}
  /* 깊이 상한만 다른 나무는 끝까지 키운 나무를 그 깊이에서 자른 것과 같다(깊이 우선, 같은 질문 규칙). 그래서 한 번 키워 두고 잘라 쓴다 */
  function importances(T,maxDepth){const imp=new Array(T.d).fill(0);let tot=0;
    T.nodes.forEach(nd=>{if(nd.leaf||(maxDepth!=null&&nd.depth>=maxDepth))return;const L=T.nodes[nd.left],R=T.nodes[nd.right];imp[nd.f]+=nd.n*nd.imp-L.n*L.imp-R.n*R.imp;});
    imp.forEach(v=>tot+=v);return imp.map(v=>tot>0?v/tot:0);}
  /* 최소 비용 복잡도 가지치기: R(t) = (마디 샘플 비율) × 불순도. 약한 고리 g = (R(t) − R(가지)) / (잎 수 − 1)가 alpha 이하인 동안 가장 약한 곳부터 자른다 */
  function prune(T,alpha){const P={reg:T.reg,K:T.K,d:T.d,nodes:T.nodes.map(nd=>Object.assign({},nd))},N=P.nodes[0].n;
    for(;;){const rb={},nl={};(function rec(id){const nd=P.nodes[id];if(nd.leaf){rb[id]=nd.n/N*nd.imp;nl[id]=1;return;}rec(nd.left);rec(nd.right);rb[id]=rb[nd.left]+rb[nd.right];nl[id]=nl[nd.left]+nl[nd.right];})(0);
      let ga=Infinity,gi=-1;(function rec(id){const nd=P.nodes[id];if(nd.leaf)return;const g=(nd.n/N*nd.imp-rb[id])/(nl[id]-1);if(g<ga){ga=g;gi=id;}rec(nd.left);rec(nd.right);})(0);
      if(gi<0||alpha<ga)break;P.nodes[gi].leaf=true;}
    return finish(P);}
  /* 규칙 글로 옮기기 (export_text와 비슷하게) */
  function rules(T,names,fmtLeaf,maxDepth){const out=[];(function rec(id,pre){const nd=T.nodes[id];
      if(nd.leaf||(maxDepth!=null&&nd.depth>=maxDepth)){out.push(pre+'└ '+fmtLeaf(nd));return;}
      const t=+nd.t.toFixed(nd.t<10?3:1);out.push(pre+'├ '+names[nd.f]+' ≤ '+t);rec(nd.left,pre+'│  ');out.push(pre+'├ '+names[nd.f]+' > '+t);rec(nd.right,pre+'│  ');})(0,'');return out;}
  return {fit,predict,leafOf,importances,prune,rules,walk,argmax};})();
/* ── TREE 엔진 끝 ── */

/* ── 데이터 · 합격선 ── */
const ROWS=concRows(),TR=ROWS.filter(r=>!r.test),TE=ROWS.filter(r=>r.test);
const fv=r=>FEAT.map(f=>r[f]);
const A=TR.map(fv),B=TE.map(fv);
const PTR=TR.map(r=>r.pass),PTE=TE.map(r=>r.pass),YTR=TR.map(r=>r.strength),YTE=TE.map(r=>r.strength),GTR=TR.map(r=>r.grade),GTE=TE.map(r=>r.grade);
const PNAME=['불합격','합격'],FNAME=FEAT.map(f=>FEATK[f].replace('(일)',''));
function r2(y,p){const m=ML.mean(y);let a=0,b=0;for(let i=0;i<y.length;i++){a+=(y[i]-p[i])*(y[i]-p[i]);b+=(y[i]-m)*(y[i]-m);}return 1-a/b;}
function mae(y,p){let s=0;for(let i=0;i<y.length;i++)s+=Math.abs(y[i]-p[i]);return s/y.length;}
/* 회귀 합격선: 매뉴얼 씨의 경험식(5-1) 28일 강도 96.5/7^(물-결합재비) × 재령 보정 t/(4+0.85t) */
const manualMPa=(wb,t)=>96.5/Math.pow(7,wb)*t/(4+0.85*t);
const MAN=TE.map(r=>manualMPa(r.wb,r.age)),LINE_R2=r2(YTE,MAN),LINE_MAE=mae(YTE,MAN);
const BASE=ML.mean(YTR),BASE_R2=r2(YTE,YTE.map(()=>BASE)),BASE_MAE=mae(YTE,YTE.map(()=>BASE));
const BASE_K=PTR.reduce((a,v)=>a+v,0)*2>=PTR.length?1:0,BASE_ACC=PTE.filter(v=>v===BASE_K).length/PTE.length;
const G_BASE=GTE.filter(v=>v===1).length/GTE.length;
const fmtR=v=>fx(v,3),fmtM=v=>v.toFixed(2);
const accT=(T,X,Y,d)=>{let c=0;X.forEach((x,i)=>{if(TREE.predict(T,x,d)===Y[i])c++;});return c/X.length;};
const r2T=(T,X,Y,d)=>r2(Y,X.map(x=>TREE.predict(T,x,d)));
const maeT=(T,X,Y,d)=>mae(Y,X.map(x=>TREE.predict(T,x,d)));
/* 깊이 상한 d에서 실제로 남는 잎 수와 깊이 */
function sizeAt(T,d){let n=0,dep=0;(function rec(id){const nd=T.nodes[id];if(nd.leaf||(d!=null&&nd.depth>=d)){n++;dep=Math.max(dep,nd.depth);return;}rec(nd.left);rec(nd.right);})(0);return {leaves:n,depth:dep};}
function cmGrid(M,labels,colsFor){const K=labels.length;let h='<div class="cm" style="grid-template-columns:86px repeat('+K+',minmax(48px,1fr));max-width:'+(110+K*90)+'px"><div class="h">정답 ＼ 예측</div>'+labels.map(k=>'<div class="h">'+k+'</div>').join('');
  M.forEach((row,a)=>{h+='<div class="h" style="justify-content:end;text-align:right">'+labels[a]+'</div>'+row.map((v,b)=>a===b?'<div class="c'+(v?' d':'')+'" style="'+(v?'background:'+rgba(colsFor(a),.85):'')+'">'+v+'</div>':'<div class="c'+(v?' x':'')+'">'+(v||'·')+'</div>').join('');});return h+'</div>';}
function barRows(rows,max){return rows.map(r=>'<div class="row"><span style="text-align:left;font-family:var(--font)">'+r[0]+'</span><div class="tr"><i style="left:0;width:'+Math.max(0,Math.min(100,r[1]/max*100))+'%;background:'+(r[2]||'var(--blue)')+'"></i></div><span>'+r[3]+'</span></div>').join('');}
/* 기본 설정(끝까지 키운 나무)은 한 번만 키워 두고, 깊이 상한은 잘라서 쓴다 */
const MAIN={};
function mainCls(){if(!MAIN.c){const t0=performance.now();MAIN.c=TREE.fit(A,PTR,{K:2});MAIN.cms=performance.now()-t0;}return MAIN.c;}
function mainReg(){if(!MAIN.r){const t0=performance.now();MAIN.r=TREE.fit(A,YTR,{task:'reg'});MAIN.rms=performance.now()-t0;}return MAIN.r;}
function main3(){return MAIN.g||(MAIN.g=TREE.fit(A,GTR,{K:3}));}

/* ── 히어로 ── */
heroRoom($('heroCv'),{sym:'T',no:'APP-06',name:'결정 트리\n(분류 + 회귀)',role:'지원자 · 겸용팀',banner:'모델 채용 위원회 · 5-6 면접',doc:'front',asst:'smile',fly:'clip',robo:'calc',loop:10,
  beats:[{at:0.3,who:'doc',mood:'board',move:'stand',say:'질문 하나씩 볼까요'},{at:2.1,who:'asst',mood:'firm',move:'jump',say:'끝까지 적을게요!'},{at:4.0,who:'fly',mood:'point',move:'shake',say:'너무 깊으면 외워!'},
    {at:5.9,who:'robo',mood:'count',move:'pop',say:'질문값은 싸네'},{at:7.8,who:'doc',mood:'magic',move:'pop',say:'가지를 쳐 봐요'}],
  motif(ctx,t,W,H,C){const lv=REDUCE?3:Math.min(3,Math.floor((t%8)/1.5)),P=[[322,96]];ctx.lineWidth=2.5;ctx.strokeStyle=C.c_app;
    const kids=(x,y,l)=>[[x-46/l,y+30],[x+46/l,y+30]];let lvl=[[322,96]];
    for(let l=1;l<=lv;l++){const nx=[];lvl.forEach(p=>kids(p[0],p[1],l).forEach(c=>{ctx.beginPath();ctx.moveTo(p[0],p[1]);ctx.lineTo(c[0],c[1]);ctx.stroke();nx.push(c);}));lvl=nx;}
    let all=[[322,96]],cur=[[322,96]];for(let l=1;l<=lv;l++){const nx=[];cur.forEach(p=>kids(p[0],p[1],l).forEach(c=>nx.push(c)));all=all.concat(nx);cur=nx;}
    all.forEach((p,i)=>{const leaf=cur.indexOf(p)>=0&&lv>0;if(leaf)circ(ctx,p[0],p[1],5,i%3?C.c_app:C.panel,C.c_app,2);else{ctx.fillStyle=C.panel;rr(ctx,p[0]-9,p[1]-7,18,14,3);ctx.fill();ctx.stroke();}});
    ctx.fillStyle=C.muted;ctx.font=font(12,500,1);ctx.textAlign='left';ctx.fillText('깊이 '+lv,300,200);}});
$('heroLine').textContent=pct(PASSLINE)+' · R² '+fmtR(LINE_R2);
comic('comicIntro',[
  {bg:'board',cast:[{who:'doc',mood:'board',x:32,move:'nod'},{who:'asst',mood:'firm',x:76,move:'pop',flip:true}],say:{who:'doc',txt:'"재령 ≤ 42일?" 질문 하나로 772건이 둘로 나뉘어요.'},prop:'지니 0.500 → 0.416',cap:'① 질문 한 줄씩 — 가장 덜 섞이게 나누는 질문부터 고른다'},
  {bg:'board',cast:[{who:'fly',mood:'cry',x:30,move:'shake'},{who:'asst',mood:'shock',x:74,move:'tilt',flip:true}],say:{who:'fly',txt:'깊이 13에 잎이 98개? 배합 한 건짜리 잎까지 있잖아!'},prop:'훈련 99.9%',cap:'② 너무 깊은 나무 — 끝까지 물으면 훈련 배합을 외운다'},
  {bg:'board',cast:[{who:'doc',mood:'magic',x:32,move:'pop'},{who:'robo',mood:'wow',x:76,move:'jump',flip:true}],say:{who:'doc',txt:'잎을 절반으로 줄여도 시험 점수는 조금만 내려가요.'},prop:'✂ ccp_alpha',cap:'③ 가지치기 — 작고 읽을 수 있는 나무로'}]);

/* ── 01 질문 하나 고르기 (깊이 1 나무) ── */
(function(){const cv=$('splitCv'),sel=$('q1Feat'),sl=$('q1Thr'),H=380,OPT=FEAT.concat(['wb']),LAB=Object.assign({},FEATK,{wb:'물-결합재비'});
  OPT.forEach(f=>sel.add(new Option(LAB[f],f)));sel.value='age';
  const st={task:'cls',f:'age',i:0,anim:0,all:null},CACHE={};
  /* 한 특징의 모든 문턱 후보와 점수(나눈 뒤 불순도의 가중 평균) */
  function curve(task,f){const k=task+f;if(CACHE[k])return CACHE[k];const n=TR.length,ord=TR.map((r,i)=>i).sort((a,b)=>TR[a][f]-TR[b][f]||a-b),xs=ord.map(i=>TR[i][f]),ys=ord.map(i=>task==='cls'?PTR[i]:YTR[i]);
    let tot=0,tq=0,tp=0;ys.forEach(v=>{tot+=v;tq+=v*v;tp+=v;});const imp=task==='cls'?1-Math.pow(tp/n,2)-Math.pow(1-tp/n,2):tq/n-Math.pow(tot/n,2);
    const C=[];let sl=0,sq=0;for(let p=0;p<n-1;p++){sl+=ys[p];sq+=ys[p]*ys[p];if(xs[p+1]<=xs[p]+1e-7)continue;const nl=p+1,nr=n-nl;let sc,L,R;
      if(task==='cls'){const pl=sl/nl,pr=(tp-sl)/nr,gl=1-pl*pl-(1-pl)*(1-pl),gr=1-pr*pr-(1-pr)*(1-pr);sc=(nl*gl+nr*gr)/n;L={p:pl,k:sl};R={p:pr,k:tp-sl};}
      else{const ml=sl/nl,mr=(tot-sl)/nr,vl=sq/nl-ml*ml,vr=(tq-sq)/nr-mr*mr;sc=(nl*vl+nr*vr)/n;L={m:ml,v:vl};R={m:mr,v:vr};}
      C.push({t:(xs[p]+xs[p+1])/2,nl,nr,sc,L,R});}
    let b=0;C.forEach((c,i)=>{if(c.sc<C[b].sc)b=i;});return CACHE[k]={C,best:b,imp,xs,ys,lo:xs[0],hi:xs[n-1]};}
  const tf=x=>st.f==='age'?Math.log10(x):x;
  function draw(){const W=fit(cv,H),ctx=cv.getContext('2d'),C=cols(),cu=curve(st.task,st.f),c=cu.C[st.i],pl=56,pr=14,gw=W-pl-pr,top=[14,108],bot=[146,H-44];
    const x0=tf(cu.lo),x1=tf(cu.hi),pad=(x1-x0)*.03,sx=x=>pl+(tf(x)-x0+pad)/(x1-x0+2*pad)*gw;
    ctx.clearRect(0,0,W,H);ctx.font=font(12);
    /* 위: 점수 곡선 */
    const smin=Math.min(...cu.C.map(q=>q.sc)),smax=cu.imp,sy=v=>top[0]+(1-(v-smin)/(smax-smin||1))*(top[1]-top[0]);
    ctx.strokeStyle=C.rule;ctx.strokeRect(pl,top[0],gw,top[1]-top[0]);ctx.fillStyle=C.muted;ctx.textAlign='right';ctx.fillText(st.task==='cls'?fx(smax,3):smax.toFixed(0),pl-4,top[0]+10);ctx.fillText(st.task==='cls'?fx(smin,3):smin.toFixed(0),pl-4,top[1]);
    ctx.textAlign='left';ctx.fillText(W<560?(st.task==='cls'?'나눈 뒤 지니 (낮을수록 좋음)':'나눈 뒤 MSE (낮을수록 좋음)'):(st.task==='cls'?'나눈 뒤 지니(가중 평균) — 낮을수록 좋은 질문':'나눈 뒤 MSE(가중 평균, MPa²) — 낮을수록 좋은 질문'),pl+6,top[0]+12);
    ctx.strokeStyle=C.blue;ctx.lineWidth=2.2;ctx.beginPath();cu.C.forEach((q,i)=>{const X=sx(q.t),Y=sy(q.sc);i?ctx.lineTo(X,Y):ctx.moveTo(X,Y);});ctx.stroke();
    const bq=cu.C[cu.best];circ(ctx,sx(bq.t),sy(bq.sc),6,C.ok,C.panel,2);ctx.fillStyle=C.ok;ctx.font=font(12,700);ctx.textAlign=sx(bq.t)>pl+gw-90?'right':'left';ctx.fillText('가장 좋은 문턱',sx(bq.t)+(ctx.textAlign==='right'?-8:8),sy(bq.sc)-6);
    /* 아래: 점 */
    ctx.strokeStyle=C.rule;ctx.lineWidth=1;ctx.strokeRect(pl,bot[0],gw,bot[1]-bot[0]);
    ctx.fillStyle=rgba(C.blue,.07);ctx.fillRect(pl,bot[0],sx(c.t)-pl,bot[1]-bot[0]);
    const rg=RNG(3);let py;
    if(st.task==='cls'){const mid=(bot[0]+bot[1])/2;py=(v)=>v?bot[0]+(mid-bot[0])/2:mid+(bot[1]-mid)/2;ctx.fillStyle=C.muted;ctx.textAlign='right';ctx.font=font(12,700);ctx.fillStyle=C.blue;ctx.fillText('합격',pl-6,py(1)+4);ctx.fillStyle=C.text;ctx.fillText('불합격',pl-6,py(0)+4);
      cu.xs.forEach((x,i)=>{const j=(rg()-.5)*((bot[1]-bot[0])/2-14);circ(ctx,sx(x)+(rg()-.5)*2,py(cu.ys[i])+j,2.6,rgba(cu.ys[i]?C.blue:C.text,.55),null);});}
    else{const ys=v=>bot[1]-(v/85)*(bot[1]-bot[0]);ctx.fillStyle=C.muted;ctx.textAlign='right';ctx.font=font(12);[0,40,80].forEach(v=>ctx.fillText(v,pl-6,ys(v)+4));
      ctx.save();ctx.translate(12,(bot[0]+bot[1])/2);ctx.rotate(-Math.PI/2);ctx.textAlign='center';ctx.fillText('강도(MPa)',0,0);ctx.restore();
      cu.xs.forEach((x,i)=>circ(ctx,sx(x)+(rg()-.5)*2,ys(cu.ys[i]),2.6,rgba(C.text,.45),null));
      ctx.strokeStyle=C.blue;ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(pl,ys(c.L.m));ctx.lineTo(sx(c.t),ys(c.L.m));ctx.lineTo(sx(c.t),ys(c.R.m));ctx.lineTo(pl+gw,ys(c.R.m));ctx.stroke();}
    /* 축 눈금 */
    ctx.fillStyle=C.muted;ctx.font=font(12);ctx.textAlign='center';
    const ticks=st.f==='age'?[1,3,7,28,90,365]:[0,1,2,3,4,5].map(k=>cu.lo+(cu.hi-cu.lo)*k/5);ticks.forEach(v=>ctx.fillText(st.f==='age'?v:(v<5?v.toFixed(2):v.toFixed(0)),sx(v),bot[1]+16));
    ctx.fillText(LAB[st.f]+(st.f==='age'?' (로그 눈금)':''),pl+gw/2,H-6);
    /* 지금 문턱 */
    ctx.strokeStyle=C.stamp;ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(sx(c.t),top[0]);ctx.lineTo(sx(c.t),bot[1]);ctx.stroke();circ(ctx,sx(c.t),sy(c.sc),5,C.stamp,C.panel,2);
    ctx.fillStyle=C.stamp;ctx.font=font(12,700,1);ctx.textAlign=sx(c.t)>pl+gw-110?'right':'left';ctx.fillText('≤ '+tnum(c.t),sx(c.t)+(ctx.textAlign==='right'?-6:6),bot[0]+14);
    cv._sx=sx;info();}
  const tnum=t=>t<5?(+t.toFixed(3)).toString():(+t.toFixed(1)).toString();
  function info(){const cu=curve(st.task,st.f),c=cu.C[st.i],n=TR.length,lab=LAB[st.f],dec=cu.imp-c.sc,bq=cu.C[cu.best];
    $('q1ThrO').textContent='≤ '+tnum(c.t);
    let accTxt;
    if(st.task==='cls'){const al=Math.max(c.L.k,c.nl-c.L.k),ar=Math.max(c.R.k,c.nr-c.R.k),acc=(al+ar)/n;accTxt=pct(acc);
      $('q1Stat').innerHTML=statGrid([[fx(c.sc,3),'나눈 뒤 지니 (뿌리 '+fx(cu.imp,3)+')','a'],[fx(dec,3),'줄어든 불순도'],[accTxt,'이 질문 하나의 훈련 정확도'],[c.nl+' / '+c.nr,'예 / 아니요 배합 수']],2);
      $('q1Bars').innerHTML=barRows([['예 · 합격 비율',c.L.p,'var(--blue)',pct(c.L.p,0)],['아니요 · 합격 비율',c.R.p,'var(--blue)',pct(c.R.p,0)]],1)+'<p class="foot" style="margin:4px 0 0">잎의 답(다수결): 예 → <b>'+PNAME[c.L.p>.5?1:0]+'</b> · 아니요 → <b>'+PNAME[c.R.p>.5?1:0]+'</b></p>';}
    else{const R2=1-c.sc/cu.imp;accTxt='R² '+fmtR(R2);
      $('q1Stat').innerHTML=statGrid([[c.sc.toFixed(1),'나눈 뒤 MSE (뿌리 '+cu.imp.toFixed(1)+')','a'],[dec.toFixed(1),'줄어든 MSE'],[accTxt,'이 질문 하나의 훈련 R²'],[c.nl+' / '+c.nr,'예 / 아니요 배합 수']],2);
      $('q1Bars').innerHTML=barRows([['예 · 평균 강도',c.L.m,'var(--blue)',c.L.m.toFixed(1)],['아니요 · 평균 강도',c.R.m,'var(--blue)',c.R.m.toFixed(1)]],85)+'<p class="foot" style="margin:4px 0 0">잎의 답(평균): 예 → <b>'+c.L.m.toFixed(1)+'MPa</b> · 아니요 → <b>'+c.R.m.toFixed(1)+'MPa</b></p>';}
    let h='질문 <b>"'+lab+' ≤ '+tnum(c.t)+'?"</b>: 예 '+c.nl+'건, 아니요 '+c.nr+'건. '+(st.i===cu.best?'<b>이 특징에서 가장 좋은 문턱이다.</b> ':'이 특징의 가장 좋은 문턱은 '+tnum(bq.t)+'(나눈 뒤 '+(st.task==='cls'?fx(bq.sc,3):bq.sc.toFixed(1))+')다. ');
    if(st.all)h+='<br>'+st.all;
    $('q1Info').innerHTML=h;}
  function setFeat(f,keep){st.f=f;sel.value=f;const cu=curve(st.task,f);sl.min=0;sl.max=cu.C.length-1;sl.step=1;if(!keep)st.i=Math.floor(cu.C.length/2);sl.value=st.i;draw();}
  sel.addEventListener('change',()=>{cancelAnimationFrame(st.anim);st.all=null;setFeat(sel.value);});
  sl.addEventListener('input',()=>{cancelAnimationFrame(st.anim);st.i=+sl.value;draw();});
  seg('#m1 .choice button',v=>{cancelAnimationFrame(st.anim);st.task=v;st.all=null;setFeat(st.f);});
  cv.addEventListener('click',e=>{const r=cv.getBoundingClientRect(),mx=e.clientX-r.left,cu=curve(st.task,st.f);let bi=0,bd=1e9;cu.C.forEach((q,i)=>{const d=Math.abs(cv._sx(q.t)-mx);if(d<bd){bd=d;bi=i;}});st.i=bi;sl.value=bi;draw();});
  $('q1Best').addEventListener('click',()=>{cancelAnimationFrame(st.anim);const cu=curve(st.task,st.f),n=cu.C.length;if(REDUCE){st.i=cu.best;sl.value=st.i;draw();return;}let k=0;const steps=60;
    (function go(){k++;st.i=k<steps?Math.round((n-1)*k/steps):cu.best;sl.value=st.i;draw();if(k<steps+1)st.anim=requestAnimationFrame(go);else $('q1Info').innerHTML+='<br>모든 문턱 후보 <b>'+n+'개</b>를 차례로 시험하고 점수가 가장 낮은 곳에 멈췄다. 결정 트리가 마디마다 하는 일이 바로 이것이다(특징 여덟 개에 대해).';})();});
  $('q1All').addEventListener('click',()=>{cancelAnimationFrame(st.anim);const res=OPT.map(f=>{const cu=curve(st.task,f),b=cu.C[cu.best];return {f,dec:cu.imp-b.sc,t:b.t};}).sort((a,b)=>b.dec-a.dec),mx=res[0].dec;
    const raw=res.filter(r=>r.f!=='wb')[0],wi=res.findIndex(r=>r.f==='wb');
    st.all='<b>여덟 특징과 사람이 만든 특징 물-결합재비</b>, 각자의 가장 좋은 질문을 줄어든 불순도로 줄 세웠다(오른쪽). 원래 여덟 특징 중 1등은 <b>'+LAB[raw.f]+' ≤ '+tnum(raw.t)+'</b>이고, 특징 여덟 개만 받은 나무의 첫 질문이 바로 이것이다. '+(wi===0?'그런데 물-결합재비를 특징으로 넣어 주면 <b>그것이 1등</b>이 된다. 현장 지식으로 만든 특징 하나가 날것 특징 어느 것보다 배합을 잘 가른다(5-4에서도 본 일이다).':'물-결합재비는 '+(wi+1)+'등이다.');
    setFeat(raw.f,true);st.i=curve(st.task,raw.f).best;sl.value=st.i;draw();
    $('q1Bars').innerHTML=barRows(res.map((r,i)=>[(i+1)+'. '+LAB[r.f].replace('(일)',''),r.dec,i?'var(--blue)':'var(--ok)',st.task==='cls'?fx(r.dec,3):r.dec.toFixed(0)]),mx);});
  $('q1Man').addEventListener('click',()=>{cancelAnimationFrame(st.anim);st.task='cls';document.querySelectorAll('#m1 .choice button').forEach(b=>b.setAttribute('aria-pressed',b.getAttribute('data-v')==='cls'));
    const cu=curve('cls','wb');let bi=0;cu.C.forEach((q,i)=>{if(Math.abs(q.t-0.5)<Math.abs(cu.C[bi].t-0.5))bi=i;});const bq=cu.C[cu.best],mq=cu.C[bi];
    st.all='매뉴얼 씨의 첫 질문 <b>물-결합재비 ≤ 0.5</b>(그림의 문턱은 0.5에 가장 가까운 후보)는 나눈 뒤 지니 '+fx(mq.sc,3)+'. 데이터가 고른 물-결합재비 문턱은 <b>'+tnum(bq.t)+'</b>(지니 '+fx(bq.sc,3)+')다. 선배들의 경험칙이 데이터의 답과 '+(Math.abs(bq.t-0.5)<0.08?'꽤 가깝다':'조금 다르다')+'. 다만 매뉴얼 씨는 둘째 질문을 "28일 이상인가"로 정해 두었고, 나무는 그 질문도 데이터에서 다시 고른다(04장).';
    st.f='wb';sel.value='wb';sl.min=0;sl.max=cu.C.length-1;st.i=bi;sl.value=bi;draw();});
  setFeat('age');st.i=curve('cls','age').best;sl.value=st.i;draw();onRedraw(draw);})();

/* ── 02 실기 과제지 ── */
(function(){const xs=$('dxSel'),ys=$('dySel'),OPT=FEAT.concat(['wb','strength']),LAB=Object.assign({},FEATK,{wb:'물-결합재비',strength:'압축 강도(MPa)'});
  OPT.forEach(f=>{xs.add(new Option(LAB[f],f));ys.add(new Option(LAB[f],f));});xs.value='age';ys.value='strength';
  const cv=$('dataCv'),H=340;
  function draw(){const W=fit(cv,H),ctx=cv.getContext('2d'),C=cols(),fx_=xs.value,fy=ys.value,pl=52,pr=14,pt=14,pb=40,gw=W-pl-pr,gh=H-pt-pb;
    const xv=ROWS.map(r=>r[fx_]),yv=ROWS.map(r=>r[fy]),x0=Math.min(0,...xv),x1=Math.max(...xv)*1.04,y0=Math.min(0,...yv),y1=Math.max(...yv)*1.05,sx=x=>pl+(x-x0)/(x1-x0)*gw,sy=y=>pt+(1-(y-y0)/(y1-y0))*gh;
    ctx.clearRect(0,0,W,H);ctx.strokeStyle=C.rule;ctx.fillStyle=C.muted;ctx.font=font(12);
    for(let k=0;k<=5;k++){const gx=x0+(x1-x0)*k/5,gy=y0+(y1-y0)*k/5;ctx.beginPath();ctx.moveTo(sx(gx),pt);ctx.lineTo(sx(gx),pt+gh);ctx.moveTo(pl,sy(gy));ctx.lineTo(pl+gw,sy(gy));ctx.stroke();ctx.textAlign='center';ctx.fillText(gx.toFixed(gx<5?2:0),sx(gx),H-22);ctx.textAlign='right';ctx.fillText(gy.toFixed(gy<5?2:0),pl-6,sy(gy)+4);}
    ctx.textAlign='center';ctx.fillText(LAB[fx_],pl+gw/2,H-4);ctx.save();ctx.translate(12,pt+gh/2);ctx.rotate(-Math.PI/2);ctx.fillText(LAB[fy],0,0);ctx.restore();
    if(fy==='strength'){[25,35,45].forEach(v=>{ctx.setLineDash(v===35?[6,4]:[2,4]);ctx.strokeStyle=v===35?C.stamp:C.muted;ctx.beginPath();ctx.moveTo(pl,sy(v));ctx.lineTo(pl+gw,sy(v));ctx.stroke();ctx.setLineDash([]);ctx.fillStyle=v===35?C.stamp:C.muted;ctx.textAlign='right';ctx.fillText(v+'MPa'+(v===35?' 합격선':''),pl+gw-4,sy(v)-4);});}
    ROWS.forEach(r=>{const c=GRADECOL[r.grade];if(r.test)circ(ctx,sx(r[fx_]),sy(r[fy]),3.6,C.panel,c,1.6);else circ(ctx,sx(r[fx_]),sy(r[fy]),3,rgba(c,.75),null);});}
  xs.addEventListener('change',draw);ys.addEventListener('change',draw);draw();onRedraw(draw);
  const cnt=[0,0,0],tst=[0,0,0];ROWS.forEach(r=>{cnt[r.grade]++;if(r.test)tst[r.grade]++;});
  $('dataLeg').innerHTML=GRADES.map((g,k)=>'<span><i style="background:'+GRADECOL[k]+'"></i>'+g+' ('+(k===0?'25MPa 미만':k===1?'25~45':'45 이상')+') '+cnt[k]+'건 · 시험 '+tst[k]+'</span>').join('')+'<span>○ 속이 빈 점 = 봉인된 시험지</span>';
  $('concTbl').innerHTML='<table><thead><tr><th>#</th>'+FEAT.map(f=>'<th><span class="tg x">특징 X</span>'+FEATK[f]+'</th>').join('')+'<th><span class="tg t">타겟 y · B</span>압축 강도<br>(MPa)</th><th><span class="tg t">타겟 y · A</span>35MPa 합격</th><th><span class="tg t">타겟 y · C</span>등급</th><th>용도</th></tr></thead><tbody>'+
    ROWS.map(r=>'<tr><td>'+(r.i+1)+'</td>'+FEAT.map(f=>'<td>'+r[f]+'</td>').join('')+'<td class="y">'+r.strength+'</td><td class="y">'+PNAME[r.pass]+'</td><td class="y">'+GRADES[r.grade]+'</td><td>'+(r.test?'<span class="pill no">봉인</span>':'훈련')+'</td></tr>').join('')+'</tbody></table>';
  $('stdTest').addEventListener('click',async()=>{$('stdInfo').textContent='두 번 키우는 중…';await wait(20);const sc=ML.scaler(A),As=sc.t(A),Bs=sc.t(B),m0=mainCls(),m1=TREE.fit(As,PTR,{K:2});
    const a0=accT(m0,B,PTE),a1=accT(m1,Bs,PTE),r0=m0.nodes[0],r1=m1.nodes[0];
    $('stdInfo').innerHTML='그대로: 첫 질문 <b>'+FNAME[r0.f]+' ≤ '+(+r0.t.toFixed(2))+'</b> · 잎 '+m0.leaves+'개 · 시험 정확도 <b>'+pct(a0)+'</b><br>표준화: 첫 질문 <b>'+FNAME[r1.f]+' ≤ '+fx(r1.t,3)+'</b>(표준편차 단위) · 잎 '+m1.leaves+'개 · 시험 정확도 <b>'+pct(a1)+'</b><br>'+(a0===a1&&m0.leaves===m1.leaves?'<b>똑같은 나무다.</b> 문턱 숫자만 단위에 맞게 바뀌었다. '+FNAME[r1.f]+' 평균 '+sc.mu[r1.f].toFixed(1)+' + '+fx(r1.t,3)+' × 표준편차 '+sc.sd[r1.f].toFixed(1)+' = '+(sc.mu[r1.f]+r1.t*sc.sd[r1.f]).toFixed(1)+'.':'거의 같은 나무다. 표준화하면서 생긴 아주 작은 반올림 차이로 문턱 후보 한두 개가 갈렸을 수 있다.');});})();

/* ── 03 AI 비서 ── */
const MAIN_PROMPT=`너는 머신러닝 수업 조교다. Google Colab에서 바로 실행되는 파이썬 코드를 써 줘.

[환경] Colab 기본 라이브러리(numpy, pandas, scikit-learn)와 ucimlrepo만 쓴다. 첫 줄에 scikit-learn 버전을 출력한다.
[데이터] 〔ucimlrepo의 fetch_ucirepo(id=165)〕 로 콘크리트 압축 강도 데이터를 읽는다.
  - 특징 X: 〔시멘트, 고로 슬래그, 플라이 애시, 물, 고성능 감수제, 굵은 골재, 잔골재, 재령〕 8개 (열 순서대로 이름을 붙인다)
  - 분류 타겟: 〔압축 강도가 35MPa 이상이면 1, 아니면 0〕 → DecisionTreeClassifier
  - 회귀 타겟: 〔압축 강도(MPa) 그대로〕 → DecisionTreeRegressor
[규정]
  1) 강도를 〔25, 45〕MPa로 나눈 3등급으로 stratify 해서 train_test_split(test_size=0.25, random_state=42)로 먼저 나눈다.
  2) 결정 트리는 표준화하지 않는다. 나무에는 random_state=0을 준다.
  3) max_depth를 〔1, 2, 3, 4, 6, 8, None〕으로 바꿔 가며 훈련 점수와 시험 점수를 한 표로 보여 준다(분류는 정확도, 회귀는 R²와 MAE).
  4) 깊이·min_samples_leaf 같은 근무 조건을 고를 때는 훈련 데이터 안에서 GridSearchCV(cv=5)로 고른다.
[출력] 깊이 3 나무의 규칙(export_text), 특징 중요도, 혼동행렬.
[금지] 시험 데이터로 학습하거나 근무 조건을 고르지 않는다. 내 scikit-learn 버전에서 없어진 인자(criterion="mse" 등)를 쓰지 않는다. 데이터를 지어내지 않는다.
[형식] 셀 단위로 나누고, 각 셀 위에 무엇을 하는지 한국어 주석을 단다. 코드 뒤에 내가 확인해야 할 점 3가지를 적어 줘.`;
$('mainPrompt').textContent=MAIN_PROMPT;$('copyMain').addEventListener('click',()=>copyText(MAIN_PROMPT,'프롬프트를 복사했습니다'));
(function(){const LINES=[
 ['import pandas as pd'],['from ucimlrepo import fetch_ucirepo'],['from sklearn.model_selection import train_test_split'],['from sklearn.preprocessing import StandardScaler'],['from sklearn.tree import DecisionTreeClassifier'],[''],
 ['uci = fetch_ucirepo(id=165)',0,'UCI에서 콘크리트 데이터를 실제로 받았다. 괜찮다.'],
 ['X = uci.data.features',0,'특징 여덟 열이다. 괜찮다.'],['y = uci.data.targets.iloc[:, 0]   # 압축 강도(MPa)',0,'회귀 타겟으로 강도를 그대로 썼다. 이 줄 자체는 괜찮다. 문제는 이 y를 어떤 나무에 넣느냐다.'],[''],
 ['X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)',1,'<b>stratify와 random_state가 없다.</b> 돌릴 때마다 시험지가 바뀌어 점수가 달라지고, 고강도 배합이 한쪽으로 쏠릴 수 있다. 위원회 규정대로 강도 등급으로 <code>stratify</code>하고 <code>random_state=42</code>를 넣는다.'],
 ['sc = StandardScaler().fit(X_train)',0,'틀리지는 않았다. 훈련 데이터로만 fit 했으니 누수도 없다. 다만 <b>결정 트리에는 필요 없다.</b> 문턱 질문은 크기 순서만 보므로 표준화해도 같은 나무가 자란다(02장). 지워도 된다.'],
 ['X_train, X_test = sc.transform(X_train), sc.transform(X_test)',0,'윗줄과 같다. 해도 해가 없지만, 규칙을 읽을 때 문턱이 "시멘트 ≤ 0.31(표준편차 단위)"처럼 나와 오히려 읽기 어려워진다.'],[''],
 ['model = DecisionTreeClassifier(      # 강도를 맞히는 나무',2,'<b>분류기에 연속값을 넣었다.</b> y는 2.3~82.6MPa의 숫자다. 강도를 맞히려면 <code>DecisionTreeRegressor</code>를 쓴다. 분류기는 "Unknown label type: continuous" 오류를 낸다. 35MPa 합격을 맞히려는 것이었다면 y를 <code>(y >= 35).astype(int)</code>로 먼저 바꾼다.'],
 ['    criterion="mse",                  # 오차 제곱으로 자르기',3,'<b>없어진 이름.</b> <code>"mse"</code>는 scikit-learn 1.0에서 <code>"squared_error"</code>로 이름이 바뀌었고 1.2에서 없어졌다. 지금 버전에서는 오류가 난다. 기본값이 이미 squared_error이니 지워도 된다.'],
 ['    max_depth=30)                     # 깊게 키워 과대적합을 막는다',4,'<b>방향이 거꾸로다.</b> 깊이를 키우면 과대적합이 <b>심해진다.</b> 이 데이터에서 끝까지 키운 회귀 나무의 깊이는 22쯤이라 30은 "제한 없음"과 같다. 과대적합을 막으려면 깊이를 낮추거나 min_samples_leaf, ccp_alpha를 쓰고, 값은 교차 검증으로 고른다.'],
 ['model.fit(X_train, y_train)',0,'훈련 데이터로 학습했다. 괜찮다.'],[''],
 ['print("점수:", model.score(X_train, y_train))   # 0.997, 거의 완벽!',5,'<b>훈련 데이터로 채점했다.</b> 끝까지 키운 나무는 훈련 배합을 거의 다 외우므로 훈련 점수는 늘 1에 가깝다. 아무것도 말해 주지 않는 숫자다. <code>model.score(X_test, y_test)</code>로 시험지 점수를 내고, 훈련 점수와 나란히 적어 그 틈(외운 몫)을 본다.']];
  const host=$('bugCode'),found=new Set();
  host.innerHTML=LINES.map((l,i)=>'<div class="ln" role="listitem" tabindex="0" data-i="'+i+'"><span class="no">'+(i+1)+'</span><span>'+hl(l[0]||' ')+'</span></div>').join('');
  function score(){$('bugScore').textContent=found.size+' / 5';if(found.size===5){$('bugScore').className='pill ok';}}
  function pick(el){const i=+el.getAttribute('data-i'),l=LINES[i];if(!l[0])return;
    if(l[1]){found.add(l[1]);host.querySelectorAll('.ln').forEach(x=>{if(LINES[+x.getAttribute('data-i')][1]===l[1])x.classList.add('hit');});$('bugMsg').innerHTML=(i+1)+'번 줄: '+l[2]+(found.size===5?'<br><b>다섯 군데를 모두 찾았다.</b> 고친 코드는 A장의 Colab 코드와 같은 흐름이다.':'');}
    else{el.classList.add('miss');$('bugMsg').innerHTML=(i+1)+'번 줄: '+(l[2]||'이 줄은 괜찮다.');}score();}
  host.addEventListener('click',e=>{const el=e.target.closest('.ln');if(el)pick(el);});
  host.addEventListener('keydown',e=>{if((e.key==='Enter'||e.key===' ')&&e.target.classList.contains('ln')){e.preventDefault();pick(e.target);}});
  $('bugReveal').addEventListener('click',()=>{LINES.forEach((l,i)=>{if(l[1]){found.add(l[1]);host.querySelector('[data-i="'+i+'"]').classList.add('hit','reveal');}});score();$('bugMsg').innerHTML='빨간 줄 다섯 군데: ① 11번 나누기 규정 ② 15번 분류기에 연속값 ③ 16번 없어진 criterion 이름 ④ 17번 깊이의 방향 ⑤ 20번 훈련 데이터로 채점. 12~13번 표준화는 틀린 것이 아니라 "필요 없는 것"이다.';});
  $('bugReset').addEventListener('click',()=>{found.clear();host.querySelectorAll('.ln').forEach(x=>x.classList.remove('hit','miss','reveal'));$('bugScore').className='pill';score();$('bugMsg').textContent='줄을 하나 눌러 보세요.';});score();})();
(function(){const CK=[['데이터를 실제로 읽었나','AI가 iris 같은 예제 데이터로 바꾸지 않았는지, 행이 1,030개인지 본다.'],['분류에는 Classifier, 회귀에는 Regressor를 썼나','합격 여부(0/1)는 DecisionTreeClassifier, 강도(MPa)는 DecisionTreeRegressor. 섞으면 오류가 나거나 엉뚱한 일을 한다.'],['정답이 특징에 섞이지 않았나','압축 강도나 거기서 만든 열(합격, 등급)이 X에 들어가면 나무는 그 열 하나로 끝낸다(정답 누설).'],['나누기에 stratify와 random_state, 나무에 random_state를 넣었나','같은 점수의 질문이 여럿이면 나무도 난수로 고른다. 둘 다 고정해야 결과가 매번 같다.'],['훈련 점수와 시험 점수를 함께 출력했나','끝까지 키운 나무의 훈련 점수는 거의 1이다. 그 숫자만 보고 보고서를 쓰지 않는다.'],['깊이·가지치기 값은 교차 검증으로 골랐나','시험지 점수를 보며 max_depth를 고르면 시험지가 훈련 데이터가 된다.'],['인자 이름이 내 scikit-learn 버전에 있나','criterion="mse"·"mae", max_features="auto"는 없어졌다. print(sklearn.__version__)과 경고를 읽는다.'],['결과를 두 합격선·기준선과 비교했나','분류는 매뉴얼 씨 76.7%와 "늘 불합격" 49.6%, 회귀는 경험식 R² 0.735와 "늘 평균"을 넘어야 뽑을 까닭이 있다.'],['나무의 규칙을 사람이 읽어 봤나','export_text로 얕은 나무를 출력해 현장 상식("오래 굳힐수록 강하다")과 어긋나는 질문이 없는지 본다.']];
  const ul=$('checks');ul.innerHTML=CK.map((c,i)=>'<li><input type="checkbox" id="ck'+i+'"><label for="ck'+i+'"><b>'+c[0]+'</b><small>'+c[1]+'</small></label></li>').join('');
  function upd(){const n=ul.querySelectorAll('input:checked').length;ul.querySelectorAll('li').forEach(li=>li.classList.toggle('done',li.querySelector('input').checked));$('checkScore').textContent='확인 '+n+' / '+CK.length+(n===CK.length?' — 이제 돌려 본다. 결과도 확인 대상이다.':'');}
  ul.addEventListener('change',upd);upd();})();

/* ── 04 ① 나무가 자라는 과정: 특징 2개(물-결합재비 · 재령) ── */
(function(){const cv=$('growCv'),H=340,X2=TR.map(r=>[r.wb,r.age]),NM=['물-결합재비','재령'];
  const st={task:'cls',T:null,order:[],s:0,play:false,manual:false,raf:0,last:-1};
  const xs=X2.map(r=>r[0]),bx={x0:Math.min(...xs)-0.02,x1:Math.max(...xs)+0.02,y0:Math.log10(1)-0.08,y1:Math.log10(365)+0.08};
  function build(){const md=+$('grMax').value;st.T=TREE.fit(X2,st.task==='cls'?PTR:YTR,{task:st.task,K:2,maxDepth:md});
    st.order=st.T.nodes.filter(nd=>!nd.leaf).sort((a,b)=>a.depth-b.depth||a.id-b.id).map(nd=>nd.id);st.s=0;st.last=-1;draw();}
  const shown=id=>{const k=st.order.indexOf(id);return k>=0&&k<st.s;};
  function leafOf(x){let nd=st.T.nodes[0];while(shown(nd.id))nd=st.T.nodes[x[nd.f]<=nd.t?nd.left:nd.right];return nd;}
  const val=nd=>st.task==='cls'?nd.value[1]/nd.n:nd.value;
  const tn=(f,t)=>f===0?(+t.toFixed(3)).toString():(+t.toFixed(1)).toString()+'일';
  function draw(){const W=fit(cv,H),ctx=cv.getContext('2d'),C=cols(),pl=50,pr=10,pt=10,pb=38,gw=W-pl-pr,gh=H-pt-pb;
    const sx=x=>pl+(x-bx.x0)/(bx.x1-bx.x0)*gw,sy=a=>pt+(1-(Math.log10(a)-bx.y0)/(bx.y1-bx.y0))*gh;ctx.clearRect(0,0,W,H);
    /* 칸 칠하기 */
    (function rec(id,x0,x1,y0,y1){const nd=st.T.nodes[id];
      if(!shown(id)){const v=val(nd);let col;if(st.task==='cls')col=v>=.5?rgba(C.blue,.08+.32*(v-.5)*2):rgba(C.text,.04+.14*(.5-v)*2);else col=rgba(C.blue,.04+.38*clamp(v/70,0,1));
        ctx.fillStyle=col;ctx.fillRect(x0,y1,x1-x0,y0-y1);
        if(x1-x0>46&&y0-y1>22){ctx.fillStyle=C.text;ctx.font=font(11.5,700,1);ctx.textAlign='center';ctx.fillText(st.task==='cls'?pct(v,0):v.toFixed(0),(x0+x1)/2,(y0+y1)/2+4);}return;}
      if(nd.f===0){const X=sx(nd.t);rec(nd.left,x0,X,y0,y1);rec(nd.right,X,x1,y0,y1);}else{const Y=sy(nd.t);rec(nd.left,x0,x1,y0,Y);rec(nd.right,x0,x1,Y,y1);}})(0,pl,pl+gw,pt+gh,pt);
    X2.forEach((r,i)=>{const v=st.task==='cls'?PTR[i]:null;circ(ctx,sx(r[0]),sy(r[1]),2.4,st.task==='cls'?rgba(v?C.blue:C.text,.7):rgba(C.text,.45),null);});
    /* 선 */
    (function rec(id,x0,x1,y0,y1){if(!shown(id))return;const nd=st.T.nodes[id],nw=id===st.last;ctx.strokeStyle=nw?C.stamp:C.ink;ctx.lineWidth=nw?3:1.8;ctx.beginPath();
      if(nd.f===0){const X=sx(nd.t);ctx.moveTo(X,y0);ctx.lineTo(X,y1);ctx.stroke();rec(nd.left,x0,X,y0,y1);rec(nd.right,X,x1,y0,y1);}else{const Y=sy(nd.t);ctx.moveTo(x0,Y);ctx.lineTo(x1,Y);ctx.stroke();rec(nd.left,x0,x1,y0,Y);rec(nd.right,x0,x1,Y,y1);}})(0,pl,pl+gw,pt+gh,pt);
    if(st.manual){ctx.setLineDash([6,4]);ctx.strokeStyle=C.c_man;ctx.lineWidth=2.5;ctx.strokeRect(sx(bx.x0),sy(365*1.2),sx(0.5)-sx(bx.x0),sy(28)-sy(365*1.2));ctx.setLineDash([]);ctx.fillStyle=C.c_man;ctx.font=font(12,700);ctx.textAlign='left';ctx.fillText('매뉴얼 씨: 합격 칸',sx(bx.x0)+6,sy(28)-6);}
    ctx.strokeStyle=C.rule;ctx.lineWidth=1;ctx.strokeRect(pl,pt,gw,gh);ctx.fillStyle=C.muted;ctx.font=font(12);
    for(let k=0;k<=4;k++){const gx=bx.x0+(bx.x1-bx.x0)*k/4;ctx.textAlign='center';ctx.fillText(gx.toFixed(2),sx(gx),H-20);}
    [1,3,7,28,90,365].forEach(d=>{ctx.textAlign='right';ctx.fillText(d,pl-5,sy(d)+4);});
    ctx.textAlign='center';ctx.fillText('물-결합재비',pl+gw/2,H-3);ctx.save();ctx.translate(11,pt+gh/2);ctx.rotate(-Math.PI/2);ctx.fillText('재령(일, 로그 눈금)',0,0);ctx.restore();
    rules();info();}
  function rules(){const out=[];(function rec(id,pre){const nd=st.T.nodes[id];
      if(!shown(id)){out.push(esc(pre)+'└ <b>'+(st.task==='cls'?PNAME[nd.pred]+'</b> (합격 '+pct(nd.value[1]/nd.n,0)+', '+nd.n+'건)':nd.value.toFixed(1)+'MPa</b> ('+nd.n+'건)'));return;}
      const mk=id===st.last?' <span style="color:var(--stamp);font-weight:700">◀ 새 질문</span>':'';
      out.push(esc(pre)+'├ '+NM[nd.f]+' ≤ '+tn(nd.f,nd.t)+mk);rec(nd.left,pre+'│  ');out.push(esc(pre)+'├ '+NM[nd.f]+' > '+tn(nd.f,nd.t));rec(nd.right,pre+'│  ');})(0,'');
    $('growRules').innerHTML=out.join('\n');}
  function info(){let leaves=0,sc;const pred=X2.map(x=>{const nd=leafOf(x);return nd.pred;});(function rec(id){if(!shown(id)){leaves++;return;}rec(st.T.nodes[id].left);rec(st.T.nodes[id].right);})(0);
    sc=st.task==='cls'?'훈련 정확도 <b>'+pct(ML.acc(PTR,pred))+'</b>':'훈련 R² <b>'+fmtR(r2(YTR,pred))+'</b> (MAE '+fmtM(mae(YTR,pred))+'MPa)';
    let h='질문 <b>'+st.s+'</b>개 / 이 깊이까지 '+st.order.length+'개 · 잎 '+leaves+'개 · '+sc+'.';
    if(st.last>=0){const nd=st.T.nodes[st.last],L=st.T.nodes[nd.left],R=st.T.nodes[nd.right],w=(L.n*L.imp+R.n*R.imp)/nd.n;
      h+='<br>방금 나눈 마디(깊이 '+nd.depth+', '+nd.n+'건): <b>'+NM[nd.f]+' ≤ '+tn(nd.f,nd.t)+'</b> — '+(st.task==='cls'?'지니 '+fx(nd.imp,3)+' → '+fx(w,3):'MSE '+nd.imp.toFixed(1)+' → '+w.toFixed(1))+'. 예 쪽 '+L.n+'건, 아니요 쪽 '+R.n+'건.';}
    else h+='<br>아직 질문이 없다. 뿌리 하나가 곧 잎이라 모든 배합에 '+(st.task==='cls'?'"'+PNAME[st.T.nodes[0].pred]+'"(다수결)':st.T.nodes[0].value.toFixed(1)+'MPa(평균)')+'로 답한다. ▶ 자라기를 눌러 보자.';
    if(st.s===st.order.length&&st.s>0)h+='<br><b>깊이 '+$('grMax').value+'까지 다 자랐다.</b> '+(st.task==='cls'?'매뉴얼 씨는 물-결합재비 0.5와 28일, 두 질문으로 시험지 '+pct(PASSLINE)+'를 맞혔다. 나무는 같은 두 숫자로 문턱 위치와 질문 순서를 데이터에서 골랐다. "매뉴얼 씨의 칸 겹쳐 보기"로 비교해 보자. 칸이 비스듬한 경계를 계단으로 흉내 내는 것도 보인다.':'칸마다 평균 강도로 답하니 예측은 칸 안에서 평평하다. 강도가 물-결합재비를 따라 매끄럽게 변해도 나무는 계단으로만 따라간다.');
    $('growInfo').innerHTML=h;}
  function step(){if(st.s>=st.order.length)return false;st.last=st.order[st.s];st.s++;draw();return true;}
  function stop(){st.play=false;clearTimeout(st.raf);$('grPlay').textContent='▶ 자라기';}
  $('grPlay').addEventListener('click',()=>{if(st.play){stop();return;}if(st.s>=st.order.length){st.s=0;st.last=-1;}st.play=true;$('grPlay').textContent='❚❚ 멈춤';
    if(REDUCE){while(step());stop();return;}(function go(){if(!st.play)return;if(!step()){stop();return;}st.raf=setTimeout(go,st.s<4?900:450);})();});
  $('grStep').addEventListener('click',()=>{stop();step();});
  $('grReset').addEventListener('click',()=>{stop();st.s=0;st.last=-1;draw();});
  $('grMax').addEventListener('change',()=>{stop();build();});
  $('grManual').addEventListener('click',e=>{st.manual=!st.manual;e.target.setAttribute('aria-pressed',st.manual);draw();});
  seg('#grTask button',v=>{stop();st.task=v;build();});
  build();onRedraw(draw);})();

/* ── 04 ② 회귀 트리의 계단 ── */
(function(){const cv=$('stairCv'),H=320,CACHE={};
  function get(f){if(CACHE[f])return CACHE[f];const X=TR.map(r=>[r[f]]),T=TREE.fit(X,YTR,{task:'reg'}),XT=TE.map(r=>[r[f]]);return CACHE[f]={X,T,XT};}
  function draw(){const f=$('stFeat').value,d=+$('stDepth').value,g=get(f),W=fit(cv,H),ctx=cv.getContext('2d'),C=cols(),pl=46,pr=12,pt=12,pb=38,gw=W-pl-pr,gh=H-pt-pb,lg=f==='age';
    const tv=x=>lg?Math.log10(x):x,xv=g.X.map(r=>r[0]),lo=Math.min(...xv),hi=Math.max(...xv),a0=tv(lo),a1=tv(hi),pd=(a1-a0)*.03,sx=x=>pl+(tv(x)-a0+pd)/(a1-a0+2*pd)*gw,sy=v=>pt+(1-v/88)*gh;
    ctx.clearRect(0,0,W,H);ctx.strokeStyle=C.rule;ctx.lineWidth=1;ctx.strokeRect(pl,pt,gw,gh);ctx.fillStyle=C.muted;ctx.font=font(12);
    [0,20,40,60,80].forEach(v=>{ctx.textAlign='right';ctx.fillText(v,pl-5,sy(v)+4);});
    const ticks=lg?[1,3,7,28,90,365]:[0,1,2,3,4].map(k=>lo+(hi-lo)*k/4);ticks.forEach(v=>{ctx.textAlign='center';ctx.fillText(lg?v:(v<5?v.toFixed(2):v.toFixed(0)),sx(v),H-20);});
    ctx.textAlign='center';ctx.fillText({cement:'시멘트(kg/m³)',wb:'물-결합재비',age:'재령(일, 로그 눈금)'}[f],pl+gw/2,H-3);ctx.save();ctx.translate(11,pt+gh/2);ctx.rotate(-Math.PI/2);ctx.fillText('강도(MPa)',0,0);ctx.restore();
    g.X.forEach((r,i)=>circ(ctx,sx(r[0]),sy(YTR[i]),2.6,rgba(C.text,.4),null));
    if(f==='wb'){ctx.setLineDash([6,4]);ctx.strokeStyle=C.c_man;ctx.lineWidth=2;ctx.beginPath();for(let k=0;k<=80;k++){const x=lo+(hi-lo)*k/80,y=manualMPa(x,28);k?ctx.lineTo(sx(x),sy(y)):ctx.moveTo(sx(x),sy(y));}ctx.stroke();ctx.setLineDash([]);ctx.fillStyle=C.c_man;ctx.font=font(12,700);ctx.textAlign='right';ctx.fillText('매뉴얼 경험식 (28일)',pl+gw-6,sy(manualMPa(hi,28))-10);}
    ctx.strokeStyle=C.blue;ctx.lineWidth=3;ctx.beginPath();let prev=null;for(let k=0;k<=600;k++){const x=lg?Math.pow(10,a0+(a1-a0)*k/600):lo+(hi-lo)*k/600,y=TREE.predict(g.T,[x],d);if(prev===null)ctx.moveTo(sx(x),sy(y));else if(y!==prev){ctx.lineTo(sx(x),sy(prev));ctx.lineTo(sx(x),sy(y));}prev=y;}ctx.lineTo(sx(hi),sy(prev));ctx.stroke();
    const sz=sizeAt(g.T,d),trp=g.X.map(x=>TREE.predict(g.T,x,d)),tep=g.XT.map(x=>TREE.predict(g.T,x,d));
    $('stDepthO').textContent=d;
    $('stairInfo').innerHTML='깊이 <b>'+d+'</b> · 계단 <b>'+sz.leaves+'</b>칸(잎) · 훈련 R² '+fmtR(r2(YTR,trp))+' · 시험 R² <b>'+fmtR(r2(YTE,tep))+'</b> (특징 하나만 쓴 나무). 계단 한 칸의 높이는 그 칸에 든 훈련 배합들의 <b>평균 강도</b>다. 칸의 경계는 MSE를 가장 많이 줄이는 곳에 생긴다.'+
      (f==='age'?' 재령은 1·3·7·14·28·56·90일처럼 띄엄띄엄한 값뿐이라 계단 경계도 그 사이에만 생긴다.':f==='wb'?' 점선은 매뉴얼 씨의 매끄러운 곡선이다. 나무는 같은 경향을 계단으로 그린다.':'')+(d>=6?' 깊이를 올릴수록 계단이 잘아져 훈련 점수는 오르지만, 시험 점수는 따라 오르지 않는다.':'');}
  $('stFeat').addEventListener('change',draw);$('stDepth').addEventListener('input',draw);draw();onRedraw(draw);})();

/* ── 05 성과 평가 · 06 인사 카드 ── */
function renderEval(){const C=cols(),mc=mainCls(),mr=mainReg(),m3=main3();
  const pc=B.map(x=>TREE.predict(mc,x)),acc=ML.acc(PTE,pc),atr=accT(mc,A,PTR);
  $('evalCls').innerHTML=statGrid([[pct(acc),'시험 정확도 ('+Math.round(acc*258)+' / 258건)','a'],[pct(atr),'훈련 정확도 (772건)'],[pct(PASSLINE),'합격선 · 5-0 매뉴얼 씨','m'],[pct(BASE_ACC),'기준선 · 늘 '+rago('"'+PNAME[BASE_K]+'"')+'만 답할 때']],4)+
    '<p class="foot">끝까지 키운 나무: 깊이 '+mc.depth+', 잎 '+mc.leaves+'개. 잎 하나에 평균 '+(772/mc.leaves).toFixed(1)+'건이 남았다. 학습 '+Math.max(1,Math.round(MAIN.cms))+'ms(브라우저).</p>';
  const pr=B.map(x=>TREE.predict(mr,x)),R=r2(YTE,pr),M=mae(YTE,pr),Rtr=r2T(mr,A,YTR);
  $('evalReg').innerHTML=statGrid([['R² '+fmtR(R),'시험 R² (MAE '+fmtM(M)+'MPa)','a'],['R² '+fmtR(Rtr),'훈련 R²'],['R² '+fmtR(LINE_R2),'합격선 · 매뉴얼 경험식 (MAE '+fmtM(LINE_MAE)+')','m'],['R² '+fmtR(BASE_R2),'기준선 · 늘 평균 '+BASE.toFixed(1)+'MPa (MAE '+fmtM(BASE_MAE)+')']],4)+
    '<p class="foot">끝까지 키운 나무: 깊이 '+mr.depth+', 잎 '+mr.leaves+'개. 잎 하나에 평균 '+(772/mr.leaves).toFixed(1)+'건 — 거의 배합마다 잎이 하나다. 그래도 시험 R²가 합격선을 넘는 까닭은, 이 데이터에 같은 배합을 여러 재령으로 시험한 줄이 많아 "가장 비슷한 배합의 강도"가 꽤 좋은 답이기 때문이다(5-1 KNN과 같은 까닭).</p>';
  const Mc=ML.confusion(PTE,pc,2);$('cmA').innerHTML=cmGrid(Mc,PNAME,k=>k?C.blue:C.muted);
  $('cmAInfo').innerHTML='대각선이 맞힌 수다. <b style="color:var(--stamp)">강도 미달인데 "합격"이라 한 배합 '+Mc[0][1]+'건</b>이 출하 판정에서 가장 위험한 칸이다. 결정 트리는 잎의 합격 비율을 확률처럼 내지만(<code>predict_proba</code>), 끝까지 키운 나무의 잎은 대개 0% 아니면 100%라 임계값을 옮겨도 거의 바뀌지 않는다.';
  /* 예측 vs 실제 */
  const pv=$('pvCv');function drawPv(){const W=fit(pv,280),ctx=pv.getContext('2d'),C=cols(),pl=44,pR=10,pt=10,pb=34,s=Math.min(W-pl-pR,280-pt-pb),sx=v=>pl+v/85*s,sy=v=>pt+s-v/85*s;
    ctx.clearRect(0,0,W,280);ctx.strokeStyle=C.rule;ctx.strokeRect(pl,pt,s,s);ctx.setLineDash([4,4]);ctx.beginPath();ctx.moveTo(sx(0),sy(0));ctx.lineTo(sx(85),sy(85));ctx.stroke();ctx.setLineDash([]);
    ctx.fillStyle=C.muted;ctx.font=font(12);[0,40,80].forEach(v=>{ctx.textAlign='center';ctx.fillText(v,sx(v),pt+s+14);ctx.textAlign='right';ctx.fillText(v,pl-5,sy(v)+4);});
    ctx.textAlign='center';ctx.fillText('실제 강도(MPa)',pl+s/2,280-4);ctx.save();ctx.translate(11,pt+s/2);ctx.rotate(-Math.PI/2);ctx.fillText('예측 강도',0,0);ctx.restore();
    YTE.forEach((y,i)=>circ(ctx,sx(y),sy(pr[i]),3,rgba(C.blue,.6),null));ctx.fillStyle=C.blue;ctx.font=font(12,700,1);ctx.textAlign='left';ctx.fillText('R² '+fmtR(R),pl+8,pt+16);}
  drawPv();onRedraw(drawPv);
  /* 3등급 */
  const p3=B.map(x=>TREE.predict(m3,x)),a3=ML.acc(GTE,p3),a35=accT(m3,B,GTE,5);$('cm3').innerHTML=cmGrid(ML.confusion(GTE,p3,3),GRADES,k=>GRADECOL[k]);
  $('g3Stat').innerHTML=statGrid([[pct(a3),'3등급 시험 정확도 (끝까지)','a'],[pct(a35),'깊이 5로 묶으면'],[pct(G_BASE),'기준선 · 늘 "보통 강도"']],1)+'<p class="foot">등급이 셋이어도 나무는 그대로다. 지니가 세 종류의 비율로 계산되고, 잎에서 세 등급의 다수결을 낼 뿐이다. 5-4 로지스틱 회귀(직선 경계 두 개)보다 고강도를 잘 찾는지 혼동행렬의 아래 줄을 보자.</p>';
  /* 깊이 손잡이 */
  const DS=[1,2,3,4,5,6,8,10,null],DV=DS.map(d=>({d,ctr:accT(mc,A,PTR,d),cte:accT(mc,B,PTE,d),rtr:r2T(mr,A,YTR,d),rte:r2T(mr,B,YTE,d),cl:sizeAt(mc,d).leaves,rl:sizeAt(mr,d).leaves}));
  const dst={task:'cls',k:DS.length-1},dc=$('dpCv');
  $('dpChoice').innerHTML=DS.map((d,k)=>'<button data-v="'+k+'" aria-pressed="'+(k===DS.length-1)+'">'+(d==null?'끝까지':d)+'</button>').join('');
  function drawDp(){const W=fit(dc,250),ctx=dc.getContext('2d'),C=cols(),pl=46,pr=14,pt=14,pb=36,gw=W-pl-pr,gh=250-pt-pb,cls_=dst.task==='cls',y0=cls_?0.7:0.2,y1=1.0,lx=k=>pl+k/(DS.length-1)*gw,ly=v=>pt+(1-(v-y0)/(y1-y0))*gh;
    ctx.clearRect(0,0,W,250);ctx.strokeStyle=C.rule;ctx.lineWidth=1;ctx.strokeRect(pl,pt,gw,gh);ctx.fillStyle=C.muted;ctx.font=font(12);ctx.textAlign='center';DS.forEach((d,k)=>ctx.fillText(d==null?'끝':d,lx(k),250-18));ctx.fillText('깊이 (max_depth)',pl+gw/2,250-3);
    ctx.textAlign='right';(cls_?[0.7,0.8,0.9,1]:[0.2,0.4,0.6,0.8,1]).forEach(v=>ctx.fillText(cls_?pct(v,0):v.toFixed(1),pl-5,ly(v)+4));
    const line=cls_?PASSLINE:LINE_R2;ctx.setLineDash([5,4]);ctx.strokeStyle=C.c_man;ctx.beginPath();ctx.moveTo(pl,ly(line));ctx.lineTo(pl+gw,ly(line));ctx.stroke();ctx.setLineDash([]);ctx.fillStyle=C.c_man;ctx.textAlign='left';ctx.fillText('합격선 '+(cls_?pct(line):'R² '+fmtR(line)),pl+6,ly(line)-5);
    [[cls_?'ctr':'rtr',C.muted,'훈련'],[cls_?'cte':'rte',C.blue,'시험']].forEach(([key,col,lab])=>{ctx.strokeStyle=col;ctx.lineWidth=2.5;ctx.beginPath();DV.forEach((v,k)=>{const X=lx(k),Y=ly(Math.max(y0,v[key]));k?ctx.lineTo(X,Y):ctx.moveTo(X,Y);});ctx.stroke();DV.forEach((v,k)=>circ(ctx,lx(k),ly(Math.max(y0,v[key])),4,col,C.panel,1.5));
      ctx.fillStyle=col;ctx.font=font(12,700);ctx.textAlign='right';ctx.fillText(lab,lx(DS.length-1)-8,ly(DV[DS.length-1][key])+(key.endsWith('tr')?-8:16));});
    ctx.strokeStyle=C.stamp;ctx.lineWidth=1.5;ctx.beginPath();ctx.moveTo(lx(dst.k),pt);ctx.lineTo(lx(dst.k),pt+gh);ctx.stroke();}
  function dpInfo(){const v=DV[dst.k],dn=v.d==null?'끝까지':'깊이 '+v.d,cp=v.cte>PASSLINE&&v.cte>BASE_ACC,rp=v.rte>LINE_R2&&v.rte>BASE_R2;
    $('dpInfo').innerHTML='<b>'+dn+'</b> · 분류: 잎 '+v.cl+'개, 훈련 '+pct(v.ctr)+' · 시험 <b>'+pct(v.cte)+'</b> → '+(cp?'<b style="color:var(--ok)">합격선 넘음</b>':'<b style="color:var(--stamp)">합격선 아래</b>')+
      ' · 회귀: 잎 '+v.rl+'개, 훈련 R² '+fmtR(v.rtr)+' · 시험 R² <b>'+fmtR(v.rte)+'</b> → '+(rp?'<b style="color:var(--ok)">합격선 넘음</b>':'<b style="color:var(--stamp)">합격선 아래</b>')+'.<br>'+
      (v.d===1?'질문 하나("재령 ≤ 42일?")뿐인 나무다. 분류는 이것만으로 시험 '+pct(v.cte)+' — 매뉴얼 씨에 조금 못 미친다.':v.d===2?'질문 셋. 매뉴얼 씨의 표준서와 같은 크기다. 분류 '+pct(v.cte)+'로 아직 매뉴얼 씨('+pct(PASSLINE)+')에 못 미친다.':v.d===3?'질문 일곱. 사람이 읽을 수 있는 크기인데 분류는 매뉴얼 씨를 넘는다(06장 "작업 표준서로 번역").':v.d===4?'분류는 여유 있게 넘지만 회귀는 R² '+fmtR(v.rte)+'로 보류다. 강도를 숫자로 맞히려면 칸(잎)이 훨씬 많아야 한다.':v.d===5?'회귀가 합격선에 바짝 다가왔지만 아직 아래다.':v.d===6?'회귀도 합격선을 넘는다. 훈련과 시험의 틈이 벌어지기 시작한다.':v.d==null?'기본 설정이다. 훈련 점수는 거의 1(외운 답)이고, 시험 점수는 깊이 8~10과 큰 차이가 없다. 그래도 두 합격선을 모두 넘는다. 면접 결과는 이 나무로 판정한다.':'깊어질수록 훈련 점수는 1로 달려가고, 시험 점수는 오르는 폭이 줄어든다. 두 선 사이의 틈이 외운 몫이다.');}
  $('dpChoice').addEventListener('click',e=>{const b=e.target.closest('button');if(!b)return;$('dpChoice').querySelectorAll('button').forEach(x=>x.setAttribute('aria-pressed',x===b));dst.k=+b.getAttribute('data-v');drawDp();dpInfo();});
  seg('#dpTask button',v=>{dst.task=v;drawDp();});drawDp();dpInfo();onRedraw(drawDp);
  /* 가지치기 */
  const PV={leaf:[1,2,5,10,20,50],ccp:[[0,0],[0.001,0.5],[0.002,1],[0.005,2],[0.01,5],[0.02,10]]},PC={};
  async function prRun(){const kind=document.querySelector('#prKind [aria-pressed="true"]').getAttribute('data-v'),k=+$('prVal').value,key=kind+k;
    $('prValO').textContent=kind==='leaf'?PV.leaf[k]:'분류 '+PV.ccp[k][0]+' · 회귀 '+PV.ccp[k][1];
    if(!PC[key]){$('prInfo').textContent='다시 키우는 중…';await wait(15);let tc,tr_;if(kind==='leaf'){tc=TREE.fit(A,PTR,{K:2,minLeaf:PV.leaf[k]});tr_=TREE.fit(A,YTR,{task:'reg',minLeaf:PV.leaf[k]});}else{tc=TREE.prune(mc,PV.ccp[k][0]);tr_=TREE.prune(mr,PV.ccp[k][1]);}
      PC[key]={c:{l:tc.leaves,d:tc.depth,tr:accT(tc,A,PTR),te:accT(tc,B,PTE)},r:{l:tr_.leaves,d:tr_.depth,tr:r2T(tr_,A,YTR),te:r2T(tr_,B,YTE)}};}
    const p=PC[key],cp=p.c.te>PASSLINE&&p.c.te>BASE_ACC,rp=p.r.te>LINE_R2;
    $('prTbl').innerHTML='<thead><tr><th>과제</th><th>잎</th><th>깊이</th><th>훈련</th><th>시험</th><th>판정</th></tr></thead><tbody>'+
      '<tr><td>분류 (정확도)</td><td>'+p.c.l+'</td><td>'+p.c.d+'</td><td>'+pct(p.c.tr)+'</td><td><b>'+pct(p.c.te)+'</b></td><td>'+(cp?'<b style="color:var(--ok)">합격선 넘음</b>':'<b style="color:var(--stamp)">아래</b>')+'</td></tr>'+
      '<tr><td>회귀 (R²)</td><td>'+p.r.l+'</td><td>'+p.r.d+'</td><td>'+fmtR(p.r.tr)+'</td><td><b>'+fmtR(p.r.te)+'</b></td><td>'+(rp?'<b style="color:var(--ok)">합격선 넘음</b>':'<b style="color:var(--stamp)">아래</b>')+'</td></tr></tbody>';
    $('prInfo').innerHTML=(kind==='leaf'?'<b>min_samples_leaf = '+PV.leaf[k]+'</b>: 잎마다 배합이 적어도 '+PV.leaf[k]+'건 남도록 키운다. ':'<b>ccp_alpha</b> 분류 '+PV.ccp[k][0]+' · 회귀 '+PV.ccp[k][1]+'(MSE 눈금이라 값이 크다): 끝까지 키운 나무에서 "잎을 늘려 얻은 불순도 감소가 alpha보다 작은" 가지부터 잘랐다. ')+
      (k===0?'가지치기를 하지 않은 기본 설정과 같다. 훈련 점수가 거의 1이다.':'잎 수가 '+mc.leaves+'·'+mr.leaves+'개에서 '+p.c.l+'·'+p.r.l+'개로 줄었다. 훈련 점수는 내려가고(덜 외운다), 시험 점수는 '+(p.c.te>=accT(mc,B,PTE)-0.03?'분류에서 크게 떨어지지 않는다':'조금씩 떨어진다')+'. '+(k>=4?'너무 세게 자르면 회귀가 먼저 합격선 아래로 내려간다. 강도를 숫자로 맞히려면 칸이 많이 필요하기 때문이다.':'더 작고 읽기 쉬운 나무로 비슷한 일을 하는 것이 가지치기의 목적이다.'));}
  seg('#prKind button',()=>{$('prVal').value=0;prRun();});$('prVal').addEventListener('input',prRun);prRun();
  /* 06 작업 표준서로 번역 */
  const LN=FEAT.map(f=>({cement:'시멘트',slag:'슬래그',fly_ash:'플라이 애시',water:'물',superplasticizer:'감수제',coarse_agg:'굵은 골재',fine_agg:'잔골재',age:'재령'})[f]);
  function ruleShow(d){const lines=TREE.rules(mc,LN,nd=>'→ <b>'+PNAME[nd.pred]+'</b> ('+nd.n+'건 중 합격 '+nd.value[1]+')',d);
    $('ruleText').innerHTML=lines.map(l=>l.replace(/^([│ ]*[├└] )(?!→)(.*)$/,(m,a,b)=>a+esc(b))).join('\n');
    const a=accT(mc,B,PTE,d),sz=sizeAt(mc,d);let same=0;(function rec(id){const nd=mc.nodes[id];if(nd.leaf||nd.depth>=d)return;const L=mc.nodes[nd.left],R=mc.nodes[nd.right];if((L.leaf||L.depth>=d)&&(R.leaf||R.depth>=d)&&L.pred===R.pred)same++;rec(nd.left);rec(nd.right);})(0);
    $('ruleStat').innerHTML=statGrid([[pct(a),'이 규칙의 시험 정확도','a'],[pct(PASSLINE),'매뉴얼 씨 (질문 2개)','m'],[String(sz.leaves),'잎(조항) 수'],[String(sz.leaves-1),'질문 수']],2);
    $('ruleInfo').innerHTML='깊이 '+d+' 나무를 그대로 옮긴 표준서다. 문턱은 데이터가 고른 값이라 "263.5kg"처럼 어색한 숫자가 나온다. 현장에서 쓸 때는 사람이 둥근 숫자로 다듬고 다시 채점한다. '+(same?'<b>같은 답이 나란히 붙은 잎 '+same+'쌍</b>이 보인다. 지니로는 더 순수해져서 나눴지만 다수결 답은 같다. 사람이 쓸 때는 합쳐도 된다. ':'')+(a>PASSLINE?'이 크기로도 매뉴얼 씨('+pct(PASSLINE)+')를 넘는다.':'이 크기로는 아직 매뉴얼 씨('+pct(PASSLINE)+')를 넘지 못한다.');}
  seg('#ruleDepth button',v=>ruleShow(+v));ruleShow(3);
  /* 06 특징 중요도 */
  function impShow(task){const T=task==='cls'?mc:mr,im=TREE.importances(T),ord=im.map((v,j)=>[v,j]).sort((a,b)=>b[0]-a[0]),mx=ord[0][0];
    $('impBars').innerHTML=barRows(ord.map(o=>[FEATK[FEAT[o[1]]],o[0],'var(--blue)',o[0].toFixed(3)]),mx);
    $('impInfo').innerHTML='끝까지 키운 '+(task==='cls'?'분류':'회귀')+' 나무에서 가장 많이 불순도를 줄인 특징은 <b>'+FNAME[ord[0][1]]+'</b>('+ord[0][0].toFixed(3)+')와 <b>'+FNAME[ord[1][1]]+'</b>('+ord[1][0].toFixed(3)+')다. 매뉴얼 씨가 쓰던 두 숫자(물-결합재비, 재령)와 같은 뿌리다. '+
      '<span class="muted">주의: 깊은 마디의 잔질문까지 다 더한 값이라 외운 몫이 섞인다. 값이 여러 갈래(문턱 후보)인 특징이 부풀려지기도 한다. 시험지로 섞어 보는 순열 중요도(<code>permutation_importance</code>)와 함께 본다. 5-7 랜덤 포레스트는 나무 수백 그루로 평균 낸다.</span>';}
  seg('#impTask button',v=>impShow(v));impShow('cls');
  /* 06 흔들리는 나무 */
  let shakeN=0;$('shakeGo').addEventListener('click',async()=>{const b=$('shakeGo');b.disabled=true;$('shakeInfo').textContent='다섯 그루를 키우는 중…';await wait(20);const rows=[];
    const desc=(T,id)=>{const nd=T.nodes[id];return nd.leaf?'(잎)':LN[nd.f]+' ≤ '+(+nd.t.toFixed(nd.t<10?3:1));};
    for(let r=0;r<5;r++){const rg=RNG(1000+shakeN*7+r),keep=[];A.forEach((_,i)=>{if(rg()>=0.1)keep.push(i);});const T=TREE.fit(keep.map(i=>A[i]),keep.map(i=>PTR[i]),{K:2});
      rows.push([shakeN*5+r+1,keep.length,desc(T,0),desc(T,T.nodes[0].left)+' / '+desc(T,T.nodes[0].right),T.leaves,accT(T,B,PTE)]);await wait(5);}
    shakeN++;const base=['전체',772,desc(mc,0),desc(mc,mc.nodes[0].left)+' / '+desc(mc,mc.nodes[0].right),mc.leaves,accT(mc,B,PTE)];
    $('shakeTbl').innerHTML='<thead><tr><th>나무</th><th>훈련 배합</th><th>첫 질문</th><th>둘째 질문 (예 / 아니요)</th><th>잎</th><th>시험 정확도</th></tr></thead><tbody>'+[base].concat(rows).map((r,i)=>'<tr'+(i?'':' style="font-weight:700"')+'><td>'+r[0]+'</td><td>'+r[1]+'</td><td>'+r[2]+'</td><td>'+r[3]+'</td><td>'+r[4]+'</td><td>'+pct(r[5])+'</td></tr>').join('')+'</tbody>';
    const accs=rows.map(r=>r[5]),firsts=new Set(rows.map(r=>r[2])),seconds=new Set(rows.map(r=>r[3]));
    $('shakeInfo').innerHTML='배합 10%만 바꿨는데 시험 정확도가 <b>'+pct(Math.min(...accs))+' ~ '+pct(Math.max(...accs))+'</b> 사이에서 움직이고, 둘째 질문은 '+seconds.size+'가지, 첫 질문은 '+firsts.size+'가지가 나왔다. 위쪽 질문은 비교적 버티지만 아래로 갈수록 나무 모양이 달라진다. 이 흔들림을 거꾸로 이용해 여러 그루의 답을 모으는 것이 5-7 랜덤 포레스트다.';b.disabled=false;});}

/* ── 면접 결과 발표 ── */
Interview('iv6',{title:'결정 트리는 분류와 회귀, 두 업무에 모두 합격할까?',
  q:'조건: 기본 설정(끝까지 키운 나무, 특징 8개)으로 봉인된 시험지 258건에서 ① 분류 정확도가 합격선(매뉴얼 씨 '+pct(PASSLINE)+')과 기준선(늘 불합격 '+pct(BASE_ACC)+')을 넘고 ② 회귀 R²가 합격선(매뉴얼 경험식 R² '+fmtR(LINE_R2)+')을 넘으면 합격. 나무는 브라우저가 직접 키운다.',
  fly:'훈련 점수랑 시험 점수 같이 대 봐. 외운 거 빼고 남는 게 얼마야?',robo:'잎이 몇 개야? 키우는 데 몇 초?',
  async run(setP){setP(.2);await wait(30);const mc=mainCls();setP(.55);await wait(30);const mr=mainReg();setP(.9);await wait(20);
    const pc=B.map(x=>TREE.predict(mc,x)),acc=ML.acc(PTE,pc),atr=accT(mc,A,PTR),pr=B.map(x=>TREE.predict(mr,x)),R=r2(YTE,pr),M=mae(YTE,pr),Rtr=r2T(mr,A,YTR);setP(1);
    const cp=acc>PASSLINE&&acc>BASE_ACC,rp=R>LINE_R2&&R>BASE_R2,pass=cp&&rp;
    return {result:pass?'pass':'hold',flyAns:'분류 시험 '+pct(acc)+' (훈련 '+pct(atr)+') · 회귀 시험 R² '+fmtR(R)+' (훈련 '+fmtR(Rtr)+')',roboAns:'잎 '+mc.leaves+' + '+mr.leaves+'개 · 학습 '+Math.max(1,Math.round(MAIN.cms+MAIN.rms))+'ms (브라우저)',
      flyMood:pass?'arms':'coffee',flyMove:pass?'nod':'shake',roboMood:pass?'wow':'meh',roboMove:pass?'jump':'tilt',asstMood:pass?'spark':'sigh',docMood:pass?'magic':'scale',
      measure:'봉인된 시험지 결과예요. 분류는 결정 트리 <b>'+pct(acc)+'</b>, 매뉴얼 씨 '+pct(PASSLINE)+', 기준선 '+pct(BASE_ACC)+'. 회귀는 <b>R² '+fmtR(R)+'</b>(MAE '+fmtM(M)+'MPa), 경험식 R² '+fmtR(LINE_R2)+'(MAE '+fmtM(LINE_MAE)+'), 기준선 MAE '+fmtM(BASE_MAE)+'. 훈련과 시험의 틈은 분류 '+fx((atr-acc)*100,1)+'%p, 회귀 R² '+fx(Rtr-R,3)+'예요.',
      verdict:pass?'합격입니다. 겸용팀에 배치해 분류와 회귀를 함께 맡깁니다. 조건이 둘 있어요. 끝까지 키운 나무는 훈련 배합을 외우니 현장에 낼 때는 교차 검증으로 가지치기 값을 정하고, 설명용으로 깊이 3 규칙본을 함께 둡니다.':(cp?'보류입니다. 분류는 넘었지만 회귀가 합격선 아래예요.':'보류입니다. 합격선을 넘지 못했어요.'),
      key:'데이터가 그린 순서도 — 분류 '+pct(PASSLINE)+' → '+pct(acc)+' · 회귀 R² '+fmtR(LINE_R2)+' → '+fmtR(R),
      render(box){box.innerHTML='<div class="bars">'+barRows([['결정 트리 · 분류',acc,'var(--blue)',pct(acc)],['매뉴얼 씨 (5-0)',PASSLINE,'var(--c-man)',pct(PASSLINE)],['기준선 (늘 불합격)',BASE_ACC,'var(--muted)',pct(BASE_ACC)]],1)+'</div>'+
        '<div class="bars" style="margin-top:10px">'+barRows([['결정 트리 · 회귀 R²',R,'var(--blue)',fmtR(R)],['매뉴얼 경험식 R²',LINE_R2,'var(--c-man)',fmtR(LINE_R2)],['기준선 (늘 평균) R²',Math.max(0,BASE_R2),'var(--muted)',fmtR(BASE_R2)]],1)+'</div>';}};},
  alts:[
   {label:'깊이 4로 묶기 (max_depth=4)',async run(box){const mc=mainCls(),mr=mainReg(),a=accT(mc,B,PTE,4),R=r2T(mr,B,YTE,4),cp=a>PASSLINE,rp=R>LINE_R2;
     box.innerHTML=statGrid([[pct(a),'분류 시험 정확도','a'],['R² '+fmtR(R),'회귀 시험 R²','a'],[pct(PASSLINE)+' · R² '+fmtR(LINE_R2),'두 합격선','m']],3)+'<p class="foot">깊이 4(잎 많아야 16개)로 묶으면 분류는 '+(cp?/*i18n:verdict*/'합격':'보류')+', 회귀는 <b>'+(rp?/*i18n:verdict*/'합격':'보류')+'</b>다. 질문 열다섯 개로 "합격이냐"는 가를 수 있어도, "몇 MPa냐"를 맞히기에는 칸이 모자란다. 판정: <b>'+(cp&&rp?/*i18n:verdict*/'합격':'보류')+'</b>. 같은 지원자라도 근무 조건(max_depth)에 따라 맡길 수 있는 일이 달라진다.</p>';}},
   {label:'잎에 5건 이상 (min_samples_leaf=5)',async run(box){const tc=TREE.fit(A,PTR,{K:2,minLeaf:5}),tr_=TREE.fit(A,YTR,{task:'reg',minLeaf:5}),a=accT(tc,B,PTE),atr=accT(tc,A,PTR),R=r2T(tr_,B,YTE),Rtr=r2T(tr_,A,YTR);
     box.innerHTML=statGrid([[pct(a),'분류 시험 (훈련 '+pct(atr)+')','a'],['R² '+fmtR(R),'회귀 시험 (훈련 '+fmtR(Rtr)+')','a'],[tc.leaves+' + '+tr_.leaves,'잎 수 (분류 + 회귀)']],3)+'<p class="foot">한두 건짜리 잎을 막았더니 훈련과 시험의 틈이 좁아지고 잎이 크게 줄었다. 시험 점수는 조금 내려가지만 두 합격선은 '+(a>PASSLINE&&R>LINE_R2?'모두 넘는다. 판정: 합격':'다 넘지 못한다. 판정: 보류')+'. A장 8번 칸의 교차 검증이 회귀에 고른 조건이 바로 이것이다.</p>';}},
   {label:'과제 C · 강도 3등급',async run(box){const m3=main3(),a=accT(m3,B,GTE),a5=accT(m3,B,GTE,5);
     box.innerHTML=statGrid([[pct(a),'3등급 시험 정확도 (끝까지)','a'],[pct(a5),'깊이 5'],[pct(G_BASE),'기준선 · 늘 보통','m']],3)+'<p class="foot">등급이 셋이어도 같은 나무가 잎에서 세 등급의 다수결을 낸다. 기준선을 크게 넘는다. 3등급에는 위원회가 정한 합격선이 따로 없어 판정에는 넣지 않았다.</p>';}}]});

/* 평가 장은 화면에 가까워지면 계산한다 */
(function(){let done=false;function go(){if(done)return;done=true;setTimeout(renderEval,10);}
  if('IntersectionObserver' in window){const io=new IntersectionObserver(es=>{if(es.some(e=>e.isIntersecting)){go();io.disconnect();}},{rootMargin:'600px'});io.observe($('m5'));io.observe($('m6'));}else go();
  window.addEventListener('hashchange',go);if(location.hash)go();})();

/* ══ 확인 문제 ══ */
var QUIZ={
q1:[{q:'지니 불순도가 0인 마디는?',o:[['한 종류(예: 합격)만 남은 마디',1,'G = 1 − Σp². 한 종류의 비율이 1이면 1 − 1 = 0이다. 더 나눌 까닭이 없는 순수한 마디다.'],['합격과 불합격이 반반인 마디',0,'두 종류가 반반이면 1 − 0.25 − 0.25 = 0.5로 가장 섞인 상태다.'],['배합이 가장 많은 마디',0,'불순도는 수가 아니라 섞인 정도를 잰다.']]},
 {q:'결정 트리가 한 마디에서 질문을 고르는 방법은?',o:[['모든 특징의 모든 문턱 후보를 시험해, 나눈 뒤 불순도의 가중 평균이 가장 작은 질문을 고른다',1,'01장 그림이 한 특징에서 하는 일을 여덟 특징 모두에 한다. 이웃한 두 값의 한가운데가 문턱 후보다.'],['사람이 정해 준 질문 순서를 따른다',0,'그것은 매뉴얼 씨의 표준서다. 나무는 질문과 순서를 데이터에서 고른다.'],['각 특징의 평균값을 문턱으로 쓴다',0,'평균은 후보 중 하나일 뿐이다. 01장에서 재령의 가장 좋은 문턱은 훈련 평균(약 44일)이 아니라 42일이다.']]},
 {q:'회귀 트리에서 "불순도" 자리에 쓰는 것은?',o:[['마디 안 강도의 분산(그 마디의 평균으로 답할 때의 MSE)',1,'숫자가 평균 근처에 모일수록 순수하다. 나눈 두 무리의 분산 가중 평균이 가장 작은 질문을 고른다.'],['지니 불순도',0,'지니는 이름표의 비율로 계산한다. 강도처럼 연속인 숫자에는 쓰지 않는다.'],['정확도',0,'정확도는 평가 지표다. 질문을 고를 때는 불순도(분산)를 줄인다.']]}],
q2:[{q:'결정 트리에 표준화가 거의 필요 없는 까닭은?',o:[['질문이 한 특징 안의 크기 순서만 보므로, 눈금을 바꿔도 같은 배합이 같은 쪽으로 간다',1,'"시멘트 ≤ 263.5kg"이 표준화 뒤에는 "≤ −0.31"이 될 뿐 나뉘는 배합은 같다(02장 단추).'],['특징을 하나만 쓰기 때문에',0,'마디마다 다른 특징을 쓴다. 다만 한 질문은 특징 하나만 본다.'],['scikit-learn이 알아서 표준화하기 때문에',0,'그런 일은 하지 않는다. 필요가 없을 뿐이다.']]},
 {q:'과제 B(강도 MPa 맞히기)에 맞는 지원자는?',o:[['DecisionTreeRegressor',1,'타겟이 연속인 숫자면 회귀 트리. 잎에서 평균으로 답한다.'],['DecisionTreeClassifier',0,'분류기는 이름표를 맞힌다. 연속값을 넣으면 "Unknown label type: continuous" 오류가 난다.'],['둘 다 똑같다',0,'질문을 고르는 방법은 같지만 불순도와 잎의 답이 다르다.']]}],
q3:[{q:'최신 scikit-learn에서 DecisionTreeRegressor(criterion="mse")는?',o:[['오류가 난다. 1.0에서 "squared_error"로 이름이 바뀌고 1.2에서 옛 이름이 없어졌다',1,'AI가 예전 코드를 보고 배워서 자주 내는 실수다. 기본값이 squared_error이니 지워도 된다.'],['그대로 잘 돈다',0,'1.2부터는 InvalidParameterError가 난다.'],['MAE로 자른다',0,'MAE는 "absolute_error"(옛 이름 "mae")다.']]},
 {q:'끝까지 키운 나무의 훈련 정확도가 0.999로 나왔다. 가장 알맞은 해석은?',o:[['훈련 배합을 거의 외웠다는 뜻일 뿐이다. 실력은 시험지 점수로 판단한다',1,'잎에 배합이 한두 건씩 남을 때까지 나누니 훈련 점수는 늘 1에 가깝다. 훈련·시험 점수를 나란히 적어 틈을 본다.'],['이 모델은 거의 완벽하다',0,'처음 보는 배합에서의 점수는 따로 재야 한다. 이 면접에서는 90%대 초반이었다.'],['데이터에 오류가 있다',0,'오류가 아니라 끝까지 키운 나무의 정상적인 모습이다.']]}],
q4:[{q:'회귀 트리의 잎은 무엇으로 답하나?',o:[['그 잎에 모인 훈련 배합들의 평균 강도',1,'MSE를 가장 작게 만드는 상수가 평균이다. 그래서 예측선이 계단 모양이 된다.'],['그 잎의 가장 흔한 강도',0,'다수결은 분류 잎의 답이다.'],['잎 안에서 다시 직선을 맞춘 값',0,'그런 변형(모델 트리)도 있지만 CART의 회귀 트리는 평균 하나로 답한다.']]},
 {q:'결정 트리를 "탐욕 알고리즘"이라 부르는 까닭은?',o:[['지금 마디에서 가장 좋은 질문만 고르고, 뒤의 질문까지 내다보거나 되돌아가지 않기 때문에',1,'가장 좋은 나무 전체를 찾는 일은 너무 어려워서 마디마다 그 순간의 최선을 고른다.'],['잎을 되도록 많이 만들기 때문에',0,'잎 수와는 관계없는 이름이다.'],['모든 가능한 나무를 비교해 가장 좋은 것을 고르기 때문에',0,'그렇게 하지 않는다. 그래서 첫 질문이 바뀌면 나무 전체가 달라진다.']]},
 {q:'물-결합재비처럼 두 특징이 함께 만드는 비스듬한 경계를 나무는 어떻게 그리나?',o:[['축에 나란한 선을 여러 번 그어 계단으로 흉내 낸다',1,'질문 하나는 특징 하나만 본다. 그래서 사람이 물-결합재비 같은 특징을 만들어 넣어 주면 나무가 훨씬 짧아진다.'],['비스듬한 직선 하나를 긋는다',0,'그것은 5-4 로지스틱 회귀의 경계다.'],['그릴 수 없다',0,'계단을 많이 쌓으면 비슷하게 따라간다. 다만 질문이 많이 든다.']]}],
q5:[{q:'max_depth를 끝까지(None) 키우면 보통 어떻게 되나?',o:[['훈련 점수는 거의 1이 되고, 시험 점수는 덜 올라 두 점수 사이가 벌어진다',1,'그 틈이 외운 몫(과대적합)이다. 이 데이터는 칸 나누기와 잘 맞아 끝까지 키워도 시험 점수가 합격선을 넘었다.'],['훈련 점수와 시험 점수가 함께 1이 된다',0,'시험지는 처음 보는 배합이라 그렇게 되지 않는다.'],['둘 다 떨어진다',0,'훈련 점수는 깊을수록 오른다.']]},
 {q:'깊이 4 나무가 분류는 합격, 회귀는 보류였던 까닭은?',o:[['강도를 숫자로 맞히려면 잎(칸)이 훨씬 많아야 하는데, 깊이 4는 잎이 많아야 16개라서',1,'"35MPa를 넘나"는 큰 칸 몇 개로 가를 수 있지만, 2~83MPa를 맞히려면 칸마다 평균이 촘촘해야 한다.'],['회귀가 원래 분류보다 쉬워서',0,'반대다. 이 데이터에서 회귀는 칸이 더 많이 필요했다.'],['회귀 트리는 깊이 제한을 무시해서',0,'같은 깊이 제한을 따른다.']]},
 {q:'ccp_alpha를 키우면?',o:[['더 많은 가지가 잘려 잎이 줄고, 나무가 작아진다',1,'잎 하나를 늘려 얻는 불순도 감소가 alpha보다 작은 가지부터 자른다. 너무 키우면 과소적합이 된다.'],['나무가 더 깊어진다',0,'반대 방향이다.'],['학습률이 커진다',0,'결정 트리에는 학습률이 없다. 학습률은 5-9 GBM의 손잡이다.']]},
 {q:'max_depth·min_samples_leaf·ccp_alpha 값은 어떻게 고르나?',o:[['훈련 데이터 안에서 교차 검증(GridSearchCV 등)으로 고른다',1,'시험지는 마지막 채점에 한 번만 쓴다. A장 8번 칸이 그 방법이다.'],['시험지 점수가 가장 높은 값을 고른다',0,'그러면 시험지가 훈련 데이터가 되어 점수를 믿을 수 없다.'],['늘 기본값을 쓴다',0,'기본값은 끝까지 키우기라 외우기 쉽다. 데이터마다 고른다.']]}],
q6:[{q:'특징 중요도를 읽을 때 조심할 점은?',o:[['"질문에 많이 쓰여 불순도를 많이 줄였다"는 뜻이지 "원인"이라는 뜻이 아니다. 값의 갈래가 많은 특징이 부풀려지기도 한다',1,'끝까지 키운 나무의 잔질문(외운 몫)까지 더한 값이다. 순열 중요도와 함께 보고, 인과는 따로 따진다.'],['중요도가 0인 특징은 강도와 아무 관계가 없다',0,'다른 특징과 함께 움직여 질문 기회를 빼앗겼을 수 있다.'],['중요도의 합은 특징 수와 같다',0,'합은 1이다.']]},
 {q:'5-9 GBM은 분류를 할 때 속에서 어떤 나무를 쓰나?',o:[['회귀 트리 — 로그 오즈를 고칠 숫자(잔차)를 맞힌다',1,'분류도 "아직 틀린 만큼"이라는 숫자를 맞히는 일로 바꿔 회귀 트리를 차례로 덧댄다. 오늘의 MSE 회귀 트리가 그 부품이다.'],['분류 트리 — 지니로 자른다',0,'GBM의 각 나무는 숫자를 맞히는 회귀 트리다.'],['나무를 쓰지 않는다',0,'GBM(Gradient Boosting Machine)은 얕은 나무를 여러 그루 쓰는 앙상블이다.']]},
 {q:'회귀 트리에 훈련 데이터에 없던 아주 높은 시멘트량의 배합을 넣으면?',o:[['가장 바깥 잎의 평균에 머문다. 범위 밖으로 늘려 말하지 못한다',1,'잎은 평균 하나로 답하므로 예측은 훈련 강도의 범위를 넘지 못한다. 선형 모델은 직선을 따라 늘려 말한다(맞든 틀리든).'],['직선을 따라 더 큰 강도를 예측한다',0,'그것은 5-2 선형 회귀의 방식이다.'],['오류가 난다',0,'어느 잎에든 떨어지므로 답은 나온다.']]}]};

/* ══ 공통 B·C·D·E·F 데이터 ══ */
var PROMPTS=[
 {t:'① 내 공정 데이터로 결정 트리 코드 받기',use:'내 공장의 품질 데이터로 03장의 프롬프트를 맞출 때',p:MAIN_PROMPT},
 {t:'② AI 코드 검증 요청',use:'받은 코드를 다른 AI(또는 같은 AI의 새 대화)에게 검토받을 때',p:'아래 scikit-learn 결정 트리 코드를 코드 리뷰해 줘. 특히 다음을 하나씩 확인하고, 문제가 있으면 줄 번호와 고친 코드를 보여 줘.\n1) 타겟이 연속값이면 DecisionTreeRegressor, 이름표면 DecisionTreeClassifier를 썼나\n2) train_test_split에 stratify와 random_state가 있고, 나무에도 random_state가 있나\n3) 내 scikit-learn 버전(〔1.x〕)에 없는 인자(criterion="mse"/"mae", max_features="auto" 등)를 쓰지 않았나\n4) 훈련 점수만 출력하고 끝내지 않았나, 시험 점수와 나란히 보여 주나\n5) max_depth·min_samples_leaf·ccp_alpha를 시험 데이터로 고르지 않았나(교차 검증을 썼나)\n6) 정답 열(또는 정답에서 만든 열)이 특징에 섞이지 않았나\n추측으로 답하지 말고, 확실하지 않은 부분은 "공식 문서 확인 필요"라고 표시해 줘.'},
 {t:'③ 나무를 작업 표준서로 번역',use:'얕은 나무의 규칙을 현장 사람이 읽을 문서로 바꾸고 싶을 때',p:'아래는 콘크리트 35MPa 합격 판정용 결정 트리(깊이 3)의 export_text 출력이다.\n이것을 현장 작업 표준서 형식으로 바꿔 줘.\n- 조항마다 "만약 ~이고 ~이면 → 판정" 한 줄로 쓴다\n- 같은 판정이 나란히 나오는 잎은 합쳐서 조항 수를 줄인다(합쳐도 판정이 바뀌지 않는 경우만)\n- 문턱 숫자(예: 263.5kg)는 그대로 두되, 둥근 숫자로 바꾸고 싶으면 후보만 괄호로 제안한다(바꾼 뒤에는 반드시 다시 채점해야 한다고 적는다)\n- 마지막에 이 표준서로 판정할 수 없는 경우(데이터 범위 밖 배합 등)와 주의점을 3가지 적는다\n\n〔여기에 export_text 출력을 붙여 넣는다〕'},
 {t:'④ 수업용 문제 만들기',use:'이 편으로 수업할 때 확인 문제가 더 필요할 때',p:'결정 트리 입문 수업용 문제를 만들어 줘. 대상은 〔제조 현장 실무자, 수식에 익숙하지 않음〕이다. 예시는 콘크리트 강도로 통일한다.\n- 지니 불순도·회귀 트리의 잎·과대적합·가지치기를 각각 묻는 객관식 4문제 (보기 3개, 정답 해설 포함)\n- 오해를 드러내는 문제 2개: "깊게 키울수록 과대적합이 줄어든다", "특징 중요도가 높으면 그 특징이 원인이다"\n- 손으로 풀 수 있는 계산 문제 1개: 합격 6건·불합격 2건인 마디의 지니 불순도, 그리고 이 마디를 (합격 6·불합격 0)과 (합격 0·불합격 2)로 나눴을 때 줄어든 불순도\n정답은 맨 끝에 따로 모아 줘.'}];
var GL=[
 ['모델','결정 트리','예/아니요 질문을 차례로 던져 데이터를 나누고, 끝(잎)에서 다수결(분류)이나 평균(회귀)으로 답하는 모델. CART가 대표적인 방법이다.'],
 ['모델','회귀 트리','숫자를 맞히는 결정 트리. 마디 안 분산(MSE)이 가장 줄도록 나누고, 잎의 평균으로 답한다. 예측선이 계단 모양이다.'],
 ['모델','마디','나무의 한 점. 질문을 가진 마디(내부 마디)와 답을 가진 끝 마디(잎)가 있다. 맨 위 마디를 뿌리라 한다.'],
 ['모델','문턱','"시멘트 ≤ 263.5"의 263.5처럼 질문이 배합을 가르는 값. 이웃한 두 값의 한가운데에서 고른다.'],
 ['모델','탐욕 알고리즘','지금 단계에서 가장 좋아 보이는 선택만 하고 되돌아보지 않는 방법. 결정 트리는 마디마다 그 순간의 최선 질문을 고른다.'],
 ['모델','앙상블','모델 여러 개의 답을 모아 하나의 답을 내는 방법. 랜덤 포레스트(5-7)와 GBM(5-9)은 결정 트리의 앙상블이다.'],
 ['학습','불순도','한 마디가 얼마나 섞였나(분류) 또는 흩어졌나(회귀)를 재는 값. 나눈 뒤 불순도가 많이 줄수록 좋은 질문이다.'],
 ['학습','지니 불순도','1 − Σp². 무리에서 두 번 뽑았을 때 이름표가 다를 확률. 한 종류뿐이면 0, 두 종류 반반이면 0.5.'],
 ['학습','엔트로피','−Σ p log₂ p. 섞임을 비트로 잰 값. 반반이면 1, 한 종류뿐이면 0.'],
 ['학습','정보 이득','나누기 전 엔트로피에서 나눈 뒤 엔트로피(가중 평균)를 뺀 값. ID3·C4.5가 질문을 고르는 기준.'],
 ['학습','평균 제곱 오차','MSE. 오차를 제곱해 평균 낸 값. 회귀 트리는 마디의 평균으로 답할 때의 MSE(= 분산)를 불순도로 쓴다.'],
 ['학습','가지치기','나무를 작게 만드는 일. 자라기 전에 멈추는 사전 가지치기(max_depth, min_samples_leaf)와 자란 뒤 자르는 사후 가지치기(ccp_alpha)가 있다.'],
 ['학습','비용 복잡도 가지치기','잎 수에 alpha만큼 벌점을 매겨, 불순도 감소가 벌점보다 작은 가지부터 잘라 내는 사후 가지치기. scikit-learn의 ccp_alpha.'],
 ['학습','과대적합','훈련 데이터를 외워 처음 보는 데이터에서 성적이 떨어지는 일. 끝까지 키운 나무의 대표적인 약점.'],
 ['학습','과소적합','모델이 너무 단순해 훈련 데이터조차 못 맞히는 일. 깊이 1~2 나무가 그렇다.'],
 ['학습','교차 검증','훈련 데이터를 몇 조각으로 나눠 돌아가며 검증해 근무 조건을 고르는 방법. 시험지를 쓰지 않는다.'],
 ['평가','특징 중요도','나무 안에서 그 특징의 질문들이 줄인 불순도의 합(전체 합 1로 맞춤). 원인의 크기가 아니다.'],
 ['평가','정확도','맞힌 샘플의 비율. 분류 과제의 점수.'],
 ['평가','혼동행렬','정답(행)과 예측(열)을 짝지어 센 표. 어떤 이름표를 어떤 이름표로 헷갈렸는지 보인다.'],
 ['평가','R²','결정 계수. 1이면 완벽, 0이면 늘 평균만 답하는 것과 같다. 회귀 과제의 점수.'],
 ['평가','MAE','평균 절대 오차. 평균 몇 MPa 틀렸나.'],
 ['평가','기준선','배우지 않고도 낼 수 있는 점수. "늘 불합격", "늘 평균 강도" 등.']];
var ANALOGY=[
 ['질문 하나(분할)','스무고개의 질문 하나. "재령이 42일 이하인가요?"',2,'초파리는 냄새 증거가 문턱을 넘으면 행동을 바꾼다(증거 누적 모형)','초파리의 문턱은 시간에 따라 증거가 쌓이는 과정이고, 나무의 문턱은 학습이 끝나면 고정된 한 숫자다. 여러 질문을 차례로 잇는 구조와 대응한다는 근거는 없다'],
 ['지니 불순도','서류 더미에서 두 장을 뽑았을 때 서로 다른 부서 서류일 확률',1,'','뇌가 "섞임"을 이런 식으로 계산한다는 근거는 알려져 있지 않다'],
 ['가장 좋은 질문 고르기','면접관이 지원자 무리를 가장 깔끔하게 가르는 질문부터 묻는 것',1,'','사람·동물이 순서 있는 질문으로 결정한다는 연구(빠르고 간단한 휴리스틱)는 있지만, 불순도를 최소화해 질문을 고른다는 근거는 없다'],
 ['잎(분류·회귀)','결재함: 분류 잎은 "다수가 받은 도장", 회귀 잎은 "그 함 서류들의 평균 금액"',1,'','잎은 훈련 배합을 그냥 모아 둔 통이다. 뇌의 기억과 대응시킬 근거가 없다'],
 ['깊이와 과대적합','지원자마다 따로 질문을 만들어 결국 한 명씩 외워 버리는 면접',1,'','뇌도 지나치게 특수한 기억을 하지만, 나무의 깊이와 같은 손잡이가 있다는 근거는 없다'],
 ['가지치기','두꺼워진 작업 표준서에서 효과가 작은 조항을 정리하는 것',1,'뇌 발달기의 "시냅스 가지치기"와 이름이 같다','이름만 같다. 시냅스 가지치기는 쓰지 않는 연결을 없애는 발달 과정이고, 트리의 가지치기는 처음 보는 데이터에서의 성능을 위해 질문을 줄이는 계산이다']];
var REFS=[
 'Yeh, I.-C. (1998). Modeling of strength of high-performance concrete using artificial neural networks. <i>Cement and Concrete Research</i>, 28(12), 1797–1808. <a href="https://doi.org/10.1016/S0008-8846(98)00165-3" target="_blank" rel="noopener">doi:10.1016/S0008-8846(98)00165-3</a> — 실기 과제지 원출처',
 'Yeh, I.-C. (2007). Concrete Compressive Strength [Dataset]. UCI Machine Learning Repository. <a href="https://doi.org/10.24432/C5PK67" target="_blank" rel="noopener">doi:10.24432/C5PK67</a> (CC BY 4.0) — 02장 데이터',
 'Breiman, L., Friedman, J. H., Olshen, R. A., &amp; Stone, C. J. (1984). <i>Classification and Regression Trees</i>. Wadsworth. — CART: 지니 불순도, 회귀 트리, 최소 비용 복잡도 가지치기. 01·04·05장',
 'Quinlan, J. R. (1986). Induction of decision trees. <i>Machine Learning</i>, 1(1), 81–106. <a href="https://doi.org/10.1007/BF00116251" target="_blank" rel="noopener">doi:10.1007/BF00116251</a> — ID3, 엔트로피와 정보 이득. 01장',
 'Quinlan, J. R. (1993). <i>C4.5: Programs for Machine Learning</i>. Morgan Kaufmann. — 정보 이득 비, 가지치기. 01·05장',
 'Hyafil, L., &amp; Rivest, R. L. (1976). Constructing optimal binary decision trees is NP-complete. <i>Information Processing Letters</i>, 5(1), 15–17. <a href="https://doi.org/10.1016/0020-0190(76)90095-8" target="_blank" rel="noopener">doi:10.1016/0020-0190(76)90095-8</a> — 탐욕 알고리즘을 쓰는 까닭. 04장',
 'Hastie, T., Tibshirani, R., &amp; Friedman, J. (2009). <i>The Elements of Statistical Learning</i> (2nd ed.), 9.2절. Springer. — 나무 기반 방법, 불안정성. 05·06장',
 'Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. <i>Journal of Machine Learning Research</i>, 12, 2825–2830. — A장 코드',
 'scikit-learn 문서. <a href="https://scikit-learn.org/stable/modules/tree.html" target="_blank" rel="noopener">Decision Trees (사용자 안내서)</a> · <a href="https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html" target="_blank" rel="noopener">DecisionTreeRegressor</a> — criterion 이름 변경, 빈칸 지원, ccp_alpha. 03·06장',
 'DasGupta, S., Ferreira, C. H., &amp; Miesenböck, G. (2014). FoxP influences the speed and accuracy of a perceptual decision in <i>Drosophila</i>. <i>Science</i>, 344(6186), 901–904. <a href="https://doi.org/10.1126/science.1252114" target="_blank" rel="noopener">doi:10.1126/science.1252114</a> — D장 문턱 비유'];
var TIP={who:'머신러닝을 처음 배우는 대학생·직장인·제조 현장 실무자. 5-0 채용 공고(매뉴얼 씨의 순서도)를 먼저 보면 좋다. 수식은 지니 1 − Σp² 하나만 나온다',
 goal:'① 질문 하나 = 특징 하나 × 문턱 하나이고, 불순도가 가장 줄어드는 질문을 고른다는 것을 01장 그림으로 설명한다 ② 분류 잎(다수결)과 회귀 잎(평균, 계단)을 구별한다 ③ 깊이에 따른 훈련·시험 점수의 틈으로 과대적합을 읽고, 가지치기 값을 교차 검증으로 고른다 ④ 얕은 나무를 작업 표준서로 옮기고 매뉴얼 씨와 비교한다',
 time:'읽기 25분 · 실습(01·04·05장 그림) 25분 · 틀린 곳 찾기 10분 · Colab 15분',
 plans:[['강의형 50분','00장 지원서 1장을 인쇄해 나눠 주고 9칸 읽기 10분 → 01장에서 학생이 직접 문턱을 옮겨 지니 최솟값 찾기 10분 → 04장 ① 나무 자라기와 매뉴얼 씨의 칸 겹쳐 보기 10분 → 05장 깊이 손잡이로 과대적합 10분 → 면접 결과를 먼저 맞히고 발표 10분.'],['실무 OJT 90분','03장 프롬프트로 각자 코드를 받는다 20분 → 체크리스트로 서로의 코드 검사 15분 → Colab 실행, A장 결과와 비교 15분 → 깊이 3 나무를 B장 ③ 프롬프트로 "우리 공장 표준서"로 바꾸고, 기존 사내 기준과 시험지 정확도를 비교해 발표 40분.'],['1:1 튜터링','종이에 배합 8건(합격 5, 불합격 3)을 적고 손으로 지니를 계산해 첫 질문을 고르게 한다. 그다음 01장 그림에서 같은 일을 772건으로 확인한다. 06장 "흔들리는 나무"를 눌러 보며 왜 숲(5-7)이 필요한지 이야기한다.']],
 myths:[['깊게 키울수록 과대적합이 줄어든다','반대다. 깊을수록 훈련 데이터를 외운다. 과대적합을 막으려면 깊이를 낮추거나 잎 크기를 키운다'],['결정 트리도 표준화해야 한다','필요 없다. 질문은 크기 순서만 본다. 해도 해는 없지만 규칙 숫자가 읽기 어려워진다'],['훈련 정확도 99.9%면 뛰어난 모델이다','끝까지 키운 나무는 늘 그렇다. 시험 점수와 나란히 본다'],['특징 중요도 1위가 강도의 원인이다','질문에 많이 쓰였다는 뜻이다. 함께 움직이는 특징, 값의 갈래가 많은 특징에서 왜곡된다'],['결정 트리는 분류 전용이다','잎에서 평균을 내면 회귀다. GBM은 분류에서도 회귀 트리를 쓴다'],['나무는 늘 같은 모양으로 자란다','데이터가 조금만 바뀌어도, random_state만 바꿔도 깊은 마디가 달라진다(06장)']],
 tasks:[['과제 1 · 손 계산 첫 질문','배합 10건을 골라 표로 주고, 재령과 시멘트 두 특징에서 문턱 후보마다 지니를 손으로 계산해 첫 질문을 고르게 한다. 01장 그림(772건)과 결과를 비교한다.'],['과제 2 · 틀린 곳 찾기 만들기','AI에게 일부러 실수가 섞인 결정 트리 코드를 받고(B장 ②를 거꾸로 활용), 짝에게 찾게 한다. 실수마다 "왜 틀렸나"를 한 줄로 쓴다.'],['과제 3 · 우리 공장 표준서','깊이 3 나무를 B장 ③ 프롬프트로 표준서로 바꾸고, 문턱을 둥근 숫자로 다듬은 뒤 A장 코드로 다시 채점한다. 매뉴얼 씨(76.7%)와 비교해 반 쪽 보고서를 쓴다.'],['과제 4 · 가지치기 고르기','A장 8번 칸의 GridSearchCV에 ccp_alpha 후보(cost_complexity_pruning_path로 뽑은 값)를 더해 분류 나무의 근무 조건을 고르고, 교차 검증 점수·시험 점수·잎 수를 표로 정리한다.']],
 ai:'학생이 AI에게 코드를 받는 것은 막지 말고 과정에 넣는다. 결정 트리는 코드가 짧아 "돌아가면 끝"으로 여기기 쉬우므로 ① 훈련 점수와 시험 점수를 나란히 쓰게 하고 ② 체크리스트 아홉 항목을 직접 표시하게 하고 ③ 나무의 규칙을 사람 말로 옮겨 현장 상식과 맞는지 한 줄 평을 쓰게 한다. AI가 쓴 인자(criterion 이름 등)는 공식 문서와 버전으로 확인하게 하고, 과제에 AI 도움을 받았으면 사용 범위를 밝히게 한다.'};
var CODE={title:'결정 트리 면접 — 질문 하나부터 가지치기까지',sub:'numpy, pandas, scikit-learn, ucimlrepo · CPU 몇 초',file:'decision_tree_interview',
 desc:'이 편의 면접을 그대로 옮겼다. UCI에서 콘크리트 데이터를 받아 시험지를 봉인하고(1·2번 칸), 두 합격선을 세우고(3번), 지니를 직접 계산해 첫 질문을 찾고(4번), 분류 트리(5번)와 회귀 트리(6번)를 키운 뒤, 깊이(7번)와 가지치기·교차 검증(8번), 작업 표준서와 특징 중요도(9번), 3등급(10번), 판정(11번)까지 간다. 노트북으로 저장하면 칸마다 셀이 나뉜다.',
 cap:'실행 결과 (scikit-learn 1.9, CPU, random_state=0). 깊이 5까지는 브라우저 계산과 숫자가 같다. 끝까지 키운 나무는 같은 점수의 질문을 고르는 순서가 달라 브라우저(분류 93.4%, 회귀 R² 0.881)와 조금 다르지만, 두 합격선을 모두 넘는다는 결론은 같다.'};
