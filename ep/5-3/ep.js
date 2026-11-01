var BOARD_LINE='R² 0.735'; /* 회귀 합격선(매뉴얼 경험식 R²) */
/* ══ 5-3 다항 회귀와 규제 · 콘크리트 실기 과제지 ══════════════════════════ */
/* 이 편이 열렸으므로 편 목록에서 5-3을 준비된 편으로 본다 (공용 src/core.js EPISODES에도 ready:1, res:'pass'를 적어야 한다) */
(function(){const e=EPISODES[epIdx('5-3')];if(e)e.ready=1;
  /* 아직 안 열린 편으로 가는 링크는 "면접 예정"으로 바꾼다 */
  document.querySelectorAll('a[data-ep]').forEach(a=>{const k=epIdx(a.getAttribute('data-ep'));if(k>=0&&!EPISODES[k].ready){const s=document.createElement('span');s.className='soon';s.textContent=a.textContent+' (면접 예정)';a.replaceWith(s);}});})();
const FEATSH=['시멘트','슬래그','애시','물','감수제','굵은골재','잔골재','재령'];
/* ══ 다항 특징 · 릿지(닫힌 해) · 라쏘/엘라스틱넷(좌표 하강) ══ */
const PR=(function(){
/* sklearn PolynomialFeatures(include_bias=False)와 같은 순서: 차수별 combinations_with_replacement */
function combos(p,d){const out=[];for(let k=1;k<=d;k++){(function rec(start,cur){if(cur.length===k){out.push(cur.slice());return;}for(let j=start;j<p;j++){cur.push(j);rec(j,cur);cur.pop();}})(0,[]);}return out;}
function expand(Z,C){return Z.map(z=>{const r=new Float64Array(C.length);for(let c=0;c<C.length;c++){let v=1;const t=C[c];for(let k=0;k<t.length;k++)v*=z[t[k]];r[c]=v;}return r;});}
function colStats(P){const n=P.length,p=P[0].length,mu=new Float64Array(p),sd=new Float64Array(p);for(const r of P)for(let j=0;j<p;j++)mu[j]+=r[j];for(let j=0;j<p;j++)mu[j]/=n;
  for(const r of P)for(let j=0;j<p;j++){const d=r[j]-mu[j];sd[j]+=d*d;}for(let j=0;j<p;j++)sd[j]=Math.sqrt(sd[j]/n)||1;return {mu,sd};}
function stdz(P,s){return P.map(r=>{const o=new Float64Array(r.length);for(let j=0;j<r.length;j++)o[j]=(r[j]-s.mu[j])/s.sd[j];return o;});}
/* 설계: 원 특징 표준화(sc1) → 다항 d차 → 다시 표준화(sc2). 훈련 데이터로만 맞춘다 */
function design(Xtr,Xte,d,sc1){const C=combos(Xtr[0].length,d),Ptr=expand(sc1.t(Xtr),C),Pte=expand(sc1.t(Xte),C),s2=colStats(Ptr);return {C,A:stdz(Ptr,s2),B:stdz(Pte,s2),p:C.length,tf:Z=>stdz(expand(sc1.t(Z),C),s2)};}
function gram(A,y){const n=A.length,p=A[0].length,G=new Float64Array(p*p),b=new Float64Array(p);
  for(let i=0;i<n;i++){const r=A[i],yi=y[i];for(let j=0;j<p;j++){const v=r[j];if(v===0)continue;b[j]+=v*yi;const o=j*p;for(let k=j;k<p;k++)G[o+k]+=v*r[k];}}
  for(let j=0;j<p;j++)for(let k=0;k<j;k++)G[j*p+k]=G[k*p+j];return {G,b,p,n};}
/* (G + αI) w = b 를 촐레스키로 푼다 = 릿지 닫힌 해 (α=0이면 정규방정식) */
function ridgeSolve(gm,alpha){const p=gm.p,L=new Float64Array(p*p);
  for(let j=0;j<p;j++){let s=gm.G[j*p+j]+alpha;for(let k=0;k<j;k++)s-=L[j*p+k]*L[j*p+k];if(s<=0)s=1e-300;const d=Math.sqrt(s);L[j*p+j]=d;
    for(let i=j+1;i<p;i++){let t=gm.G[i*p+j];for(let k=0;k<j;k++)t-=L[i*p+k]*L[j*p+k];L[i*p+j]=t/d;}}
  const z=new Float64Array(p);for(let i=0;i<p;i++){let t=gm.b[i];for(let k=0;k<i;k++)t-=L[i*p+k]*z[k];z[i]=t/L[i*p+i];}
  const w=new Float64Array(p);for(let i=p-1;i>=0;i--){let t=z[i];for(let k=i+1;k<p;k++)t-=L[k*p+i]*w[k];w[i]=t/L[i*p+i];}return w;}
/* 좌표 하강: (1/2n)‖y−Aw‖² + α·ρ·Σ|w| + ½α(1−ρ)Σw²  (ρ=1 라쏘, 0<ρ<1 엘라스틱넷) — scikit-learn Lasso/ElasticNet과 같은 목적 */
function cdState(gm,w0){const p=gm.p,w=w0?Float64Array.from(w0):new Float64Array(p),q=new Float64Array(p);for(let j=0;j<p;j++){if(!w[j])continue;for(let k=0;k<p;k++)q[k]+=gm.G[k*p+j]*w[j];}return {w,q,sweeps:0};}
function soft(x,t){return x>t?x-t:x<-t?x+t:0;}
function cdSweep(gm,st,alpha,rho){const p=gm.p,n=gm.n,G=gm.G,w=st.w,q=st.q;let mx=0;
  for(let j=0;j<p;j++){const gjj=G[j*p+j]/n,rj=(gm.b[j]-q[j])/n+gjj*w[j],nw=soft(rj,alpha*rho)/(gjj+alpha*(1-rho)),dw=nw-w[j];
    if(dw!==0){for(let k=0;k<p;k++)q[k]+=G[k*p+j]*dw;w[j]=nw;if(Math.abs(dw)>mx)mx=Math.abs(dw);}}st.sweeps++;return mx;}
function cdFit(gm,alpha,rho,o){o=o||{};const st=o.st||cdState(gm,o.w0);const tol=o.tol||1e-6,mxs=o.max||20000;for(let s=0;s<mxs;s++){if(cdSweep(gm,st,alpha,rho)<tol)break;}return st;}
function predict(A,w,b0){return A.map(r=>{let s=b0;for(let j=0;j<w.length;j++)s+=r[j]*w[j];return s;});}
function r2(y,p){const m=y.reduce((a,v)=>a+v,0)/y.length;let ss=0,st=0;for(let i=0;i<y.length;i++){ss+=(y[i]-p[i])**2;st+=(y[i]-m)**2;}return 1-ss/st;}
function mae(y,p){let s=0;for(let i=0;i<y.length;i++)s+=Math.abs(y[i]-p[i]);return s/y.length;}
const SUP=['','','²','³','⁴'];
function name(t){const cnt={};t.forEach(j=>cnt[j]=(cnt[j]||0)+1);return Object.keys(cnt).map(j=>FEATSH[j]+SUP[cnt[j]]).join('·');}
return {combos,expand,design,gram,ridgeSolve,cdState,cdSweep,cdFit,predict,r2,mae,name,colStats,stdz};})();

/* ── 데이터 · 합격선 · 공용 계산 ── */
const ROWS=concRows(),TR=ROWS.filter(r=>!r.test),TE=ROWS.filter(r=>r.test),fv=r=>FEAT.map(f=>r[f]);
const XTR=TR.map(fv),XTE=TE.map(fv),YTR=TR.map(r=>r.strength),YTE=TE.map(r=>r.strength),YM=ML.mean(YTR),YC=YTR.map(v=>v-YM);
const SC1=ML.scaler(XTR);
/* 매뉴얼 씨의 현장 경험식: 에이브럼스 f28 = 96.5 / 7^(물-결합재비), 재령 보정 t/(4+0.85t) (ACI 209R) */
const manualPred=r=>96.5/Math.pow(7,r.wb)*r.age/(4+0.85*r.age);
const MANTE=TE.map(manualPred),PASS_R2=PR.r2(YTE,MANTE),MAN_MAE=PR.mae(YTE,MANTE);
const BASE_MAE=PR.mae(YTE,YTE.map(()=>YM)),BASE_R2=PR.r2(YTE,YTE.map(()=>YM));
const f3=v=>fx(v,3),fmtA=a=>a===0?'0':a>=10?String(Math.round(a)):a>=1?a.toFixed(a<10?2:1).replace(/\.?0+$/,''):String(+a.toPrecision(2));
const DS={};
function D(d){if(!DS[d]){const t0=performance.now(),ds=PR.design(XTR,XTE,d,SC1);ds.gm=PR.gram(ds.A,YC);ds.ms=performance.now()-t0;DS[d]=ds;}return DS[d];}
function evalW(ds,w){const ptr=PR.predict(ds.A,w,YM),pte=PR.predict(ds.B,w,YM);let z=0;for(let j=0;j<w.length;j++)if(w[j]===0)z++;return {w,tr:PR.r2(YTR,ptr),te:PR.r2(YTE,pte),mae:PR.mae(YTE,pte),pte,zeros:z,p:w.length};}
const CACHE={};
function ridge(d,a){const k='r'+d+'_'+a;if(!CACHE[k]){const t0=performance.now(),ds=D(d),w=PR.ridgeSolve(ds.gm,a);CACHE[k]=evalW(ds,w);CACHE[k].ms=performance.now()-t0+ds.ms;}return CACHE[k];}
function cdm(d,a,rho){const k='c'+d+'_'+a+'_'+rho;if(!CACHE[k]){const ds=D(d),st=PR.cdFit(ds.gm,a,rho,{max:100000});CACHE[k]=evalW(ds,st.w);CACHE[k].sweeps=st.sweeps;}return CACHE[k];}
function axisLog(ctx,C,lx,y,ticks,lab,H){ctx.fillStyle=C.muted;ctx.font=font(12);ctx.textAlign='center';ticks.forEach(t=>ctx.fillText(t,lx(t),y));if(lab)ctx.fillText(lab,(lx(ticks[0])+lx(ticks[ticks.length-1]))/2,H-3);}

/* ── 히어로 ── */
heroRoom($('heroCv'),{sym:'λ',no:'APP-03',name:'다항 회귀\n와 규제',role:'지원자 · 회귀팀',banner:'모델 채용 위원회 · 5-3 면접',doc:'front',asst:'board',fly:'clip',robo:'calc',loop:10,
  beats:[{"at": 0.4, "who": "robo", "mood": "wow", "move": "shake", "say": "몸값이 18배?"}, {"at": 2.4, "who": "doc", "mood": "magic", "move": "pop", "say": "x²과 x³을 붙여요"}, {"at": 4.4, "who": "fly", "mood": "point", "move": "tilt", "say": "외운 건 아니고?"}, {"at": 6.4, "who": "asst", "mood": "firm", "move": "nod", "say": "고삐는 λ로"}, {"at": 8.4, "who": "doc", "mood": "front", "move": "stand"}],
  motif(ctx,t,W,H,C){const a=0.5+0.5*Math.sin(t*0.7);ctx.strokeStyle=C.c_app;ctx.lineWidth=3;ctx.beginPath();
    for(let i=0;i<=48;i++){const u=i/48,x=270+u*130,y=150-40*Math.sqrt(u)-a*14*Math.sin(u*22)*(0.3+u);i?ctx.lineTo(x,y):ctx.moveTo(x,y);}ctx.stroke();
    ctx.fillStyle=C.muted;ctx.font=font(12,500,1);ctx.textAlign='left';ctx.fillText(a>0.5?'α 작음: 출렁':'α 큼: 차분',290,168);}});
$('heroLine').textContent=f3(PASS_R2);

/* ── 01 장면 ── */
comic('comicIntro',[{"bg": "lab", "prop": "x, x², x³", "cast": [{"who": "doc", "mood": "magic", "x": 30, "move": "pop"}, {"who": "asst", "mood": "spark", "x": 72}], "say": {"who": "doc", "txt": "직선만 긋는 회귀에 x²과 x³을 붙여 주면, 하는 일은 같은데 곡선이 돼요."}, "cap": "① 특징 늘리기 — 모델은 그대로, 재료만 거듭제곱"}, {"bg": "night", "prop": "4차", "cast": [{"who": "fly", "mood": "cry", "x": 30, "move": "shake"}, {"who": "robo", "mood": "wow", "x": 72, "move": "shake"}], "say": {"who": "fly", "txt": "4차는 훈련에선 0.98인데, 처음 보는 배합에서 강도가 음수로 나와!"}, "cap": "② 과대적합 — 훈련 점수와 시험 점수가 갈라진다"}, {"bg": "lab", "prop": "λ", "cast": [{"who": "asst", "mood": "firm", "x": 30, "move": "nod"}, {"who": "robo", "mood": "stack", "x": 72, "move": "bob"}], "say": {"who": "asst", "txt": "고삐를 채워요. 릿지는 가중치를 작게, 라쏘는 아예 0으로 지워요."}, "cap": "③ 규제 — 가중치에 벌점을 매긴다"}]);

/* ── 01 자기소개: 물-결합재비 하나, 28일 배합 12건으로 차수와 α 만지기 ── */
(function(){const cv=$('toyCv'),H=320;const T28=TR.filter(r=>r.age===28),E28=TE.filter(r=>r.age===28),rng=RNG(7);
  const S=T28.map((r,i)=>[rng(),i]).sort((a,b)=>a[0]-b[0]).slice(0,12).map(v=>T28[v[1]]);
  const Xa=S.map(r=>[r.wb]),ya=S.map(r=>r.strength),m=ML.mean(ya),sc=ML.scaler(Xa),Xb=E28.map(r=>[r.wb]),yb=E28.map(r=>r.strength);
  const all=T28.concat(E28).map(r=>r.wb),X0=Math.min(...all)-0.02,X1=Math.max(...all)+0.02,Y0=0,Y1=95;
  const st={d:2,la:-4,man:false},man28=x=>96.5/Math.pow(7,x)*28/(4+0.85*28),manR2=PR.r2(yb,E28.map(r=>man28(r.wb)));
  function alpha(){return st.la<=-4?0:Math.pow(10,st.la);}
  function fitToy(){const ds=PR.design(Xa,Xb,st.d,sc),gm=PR.gram(ds.A,ya.map(v=>v-m)),w=PR.ridgeSolve(gm,Math.max(alpha(),1e-9));
    return {w,ds,tr:PR.r2(ya,PR.predict(ds.A,w,m)),te:PR.r2(yb,PR.predict(ds.B,w,m)),f:x=>PR.predict(ds.tf([[x]]),w,m)[0]};}
  function draw(){const W=fit(cv,H),ctx=cv.getContext('2d'),C=cols(),pl=46,pr=12,pt=14,pb=40,gw=W-pl-pr,gh=H-pt-pb,sx=x=>pl+(x-X0)/(X1-X0)*gw,sy=v=>pt+(1-(v-Y0)/(Y1-Y0))*gh,F=fitToy();
    ctx.clearRect(0,0,W,H);ctx.strokeStyle=C.rule;ctx.fillStyle=C.muted;ctx.font=font(12);ctx.lineWidth=1;
    for(let x=0.3;x<=X1;x+=0.1){ctx.beginPath();ctx.moveTo(sx(x),pt);ctx.lineTo(sx(x),pt+gh);ctx.stroke();ctx.textAlign='center';ctx.fillText(x.toFixed(1),sx(x),H-22);}
    for(let v=0;v<=90;v+=20){ctx.beginPath();ctx.moveTo(pl,sy(v));ctx.lineTo(pl+gw,sy(v));ctx.stroke();ctx.textAlign='right';ctx.fillText(v,pl-6,sy(v)+4);}
    ctx.textAlign='center';ctx.fillText('물-결합재비 (28일 배합만)',pl+gw/2,H-4);ctx.save();ctx.translate(12,pt+gh/2);ctx.rotate(-Math.PI/2);ctx.fillText('28일 압축 강도 (MPa)',0,0);ctx.restore();
    E28.forEach(r=>circ(ctx,sx(r.wb),sy(r.strength),3.4,null,rgba(C.muted,.75),1.3));
    if(st.man){ctx.strokeStyle=C.amber;ctx.lineWidth=2.5;ctx.setLineDash([7,5]);ctx.beginPath();for(let k=0;k<=100;k++){const x=X0+(X1-X0)*k/100;k?ctx.lineTo(sx(x),sy(man28(x))):ctx.moveTo(sx(x),sy(man28(x)));}ctx.stroke();ctx.setLineDash([]);}
    ctx.save();ctx.beginPath();ctx.rect(pl,pt,gw,gh);ctx.clip();ctx.strokeStyle=C.blue;ctx.lineWidth=3.2;ctx.beginPath();
    for(let k=0;k<=300;k++){const x=X0+(X1-X0)*k/300,v=clamp(F.f(x),-500,600);k?ctx.lineTo(sx(x),sy(v)):ctx.moveTo(sx(x),sy(v));}ctx.stroke();ctx.restore();
    S.forEach(r=>circ(ctx,sx(r.wb),sy(r.strength),5.5,C.text,C.panel,1.5));
    ctx.font=font(12,700);ctx.textAlign='left';ctx.fillStyle=rgba(C.panel,.88);ctx.fillRect(pl+gw-200,pt+4,196,st.man?66:50);
    ctx.fillStyle=C.text;ctx.fillText('● 훈련 배합 12건',pl+gw-192,pt+20);ctx.fillStyle=C.muted;ctx.fillText('○ 28일 시험 배합 '+E28.length+'건',pl+gw-192,pt+36);ctx.fillStyle=C.blue;ctx.fillText('━ '+st.d+'차 다항 회귀',pl+gw-192,pt+52);
    if(st.man){ctx.fillStyle=C.amber;ctx.fillText('┅ 매뉴얼 씨 경험식',pl+gw-192,pt+66);}
    info(F);}
  function info(F){const a=alpha();let h=st.d+'차 · α = <b>'+fmtA(a)+'</b> · 특징 '+st.d+'개(x … x'+(st.d>1?['','','²','³','⁴','⁵','⁶','⁷','⁸','⁹','¹⁰'][st.d]:'')+') → 외울 숫자 '+(st.d+1)+'개. 훈련 R² (12건) <b>'+f3(F.tr)+'</b> · 시험 R² (28일 시험 배합 '+E28.length+'건) <b>'+(F.te<-99?'−'+Math.round(-F.te).toLocaleString():f3(F.te))+'</b>. ';
    if(st.d>=6&&a<0.05)h+='<br><span style="color:var(--stamp)">과대적합: 곡선이 훈련 점 사이에서 출렁이며 점 하나하나를 따라간다. 훈련 R²는 높은데 시험 R²가 무너졌다.</span> 차수는 두고 α만 올려 보자.';
    else if(st.d>=6&&a>=0.05&&F.te<0.3)h+='<br>α가 고삐를 당기기 시작했다. 출렁임이 줄었지만 아직 시험 R²가 낮다. α를 더 올려 보자.';
    else if(st.d>=6&&a>=0.05)h+='<br>차수는 '+st.d+'인데 α가 고삐를 당겨 곡선이 차분해졌다. 가중치가 작게 눌리면 높은 차수 항이 힘을 못 쓴다.';
    else if(a>=30)h+='<br>규제가 너무 세다. 가중치가 0 근처로 눌려 곡선이 평평해진다(과소적합). 훈련 R²도 떨어진다.';
    else if(st.d===1)h+='<br>1차는 직선이다. 물-결합재비가 낮은 쪽(왼쪽)에서 강도가 가파르게 오르는 휘어짐을 못 따라간다.';
    else h+='<br>2~3차는 왼쪽에서 가파르고 오른쪽에서 완만한 휘어짐을 따라간다.';
    if(st.man)h+='<br><span style="color:var(--amber)">매뉴얼 씨의 경험식(28일)은 학습 없이 같은 시험 배합에서 R² '+f3(manR2)+'다. 물-결합재비 하나로는 28일 강도의 일부만 설명된다. 나머지는 감수제·골재 같은 다른 특징이 맡는다.</span>';
    $('toyInfo').innerHTML=h;}
  function sync(){$('tyDeg').value=st.d;$('tyDegO').textContent=st.d;$('tyA').value=st.la;$('tyAO').textContent=fmtA(alpha());$('tyMan').setAttribute('aria-pressed',st.man);}
  $('tyDeg').addEventListener('input',e=>{st.d=+e.target.value;sync();draw();});
  $('tyA').addEventListener('input',e=>{st.la=+e.target.value;sync();draw();});
  $('tyMan').addEventListener('click',()=>{st.man=!st.man;sync();draw();});
  $('tyWild').addEventListener('click',()=>{st.d=10;st.la=-4;sync();draw();});
  $('tyReset').addEventListener('click',()=>{st.d=2;st.la=-4;st.man=false;sync();draw();});
  sync();draw();onRedraw(draw);})();

/* ── 02 실기 과제지 ── */
(function(){const xs=$('dxSel'),ys=$('dySel'),OPT=FEAT.concat(['wb','strength','man']),LAB=Object.assign({},FEATK,{wb:'물-결합재비',strength:'압축 강도(MPa)',man:'매뉴얼 씨의 예측(MPa)'});
  ROWS.forEach(r=>{r.man=manualPred(r);});
  OPT.forEach(f=>{xs.add(new Option(LAB[f],f));ys.add(new Option(LAB[f],f));});xs.value='age';ys.value='strength';
  const cv=$('dataCv'),H=340;
  function draw(){const W=fit(cv,H),ctx=cv.getContext('2d'),C=cols(),fx_=xs.value,fy=ys.value,pl=52,pr=14,pt=14,pb=40,gw=W-pl-pr,gh=H-pt-pb;
    const xv=ROWS.map(r=>r[fx_]),yv=ROWS.map(r=>r[fy]),x0=Math.min(0,...xv),x1=Math.max(...xv)*1.04,y0=Math.min(0,...yv),y1=Math.max(...yv)*1.05,sx=x=>pl+(x-x0)/(x1-x0)*gw,sy=y=>pt+(1-(y-y0)/(y1-y0))*gh;
    ctx.clearRect(0,0,W,H);ctx.strokeStyle=C.rule;ctx.fillStyle=C.muted;ctx.font=font(12);ctx.lineWidth=1;
    for(let k=0;k<=5;k++){const gx=x0+(x1-x0)*k/5,gy=y0+(y1-y0)*k/5;ctx.beginPath();ctx.moveTo(sx(gx),pt);ctx.lineTo(sx(gx),pt+gh);ctx.moveTo(pl,sy(gy));ctx.lineTo(pl+gw,sy(gy));ctx.stroke();ctx.textAlign='center';ctx.fillText(gx.toFixed(gx<5?2:0),sx(gx),H-22);ctx.textAlign='right';ctx.fillText(gy.toFixed(gy<5?2:0),pl-6,sy(gy)+4);}
    ctx.textAlign='center';ctx.fillText(LAB[fx_],pl+gw/2,H-4);ctx.save();ctx.translate(12,pt+gh/2);ctx.rotate(-Math.PI/2);ctx.fillText(LAB[fy],0,0);ctx.restore();
    const both=(fx_==='strength'&&fy==='man')||(fx_==='man'&&fy==='strength');
    if(both){const lo=Math.max(x0,y0),hi=Math.min(x1,y1);ctx.strokeStyle=C.c_man;ctx.setLineDash([6,4]);ctx.beginPath();ctx.moveTo(sx(lo),sy(lo));ctx.lineTo(sx(hi),sy(hi));ctx.stroke();ctx.setLineDash([]);ctx.fillStyle=C.c_man;ctx.textAlign='left';ctx.fillText('예측 = 실제',sx(hi*0.8)+6,sy(hi*0.8));}
    ROWS.forEach(r=>{const c=GRADECOL[r.grade];if(r.test)circ(ctx,sx(r[fx_]),sy(r[fy]),3.6,C.panel,c,1.6);else circ(ctx,sx(r[fx_]),sy(r[fy]),3,rgba(c,.75),null);});}
  function sel(){draw();$('dxMan').setAttribute('aria-pressed',(xs.value==='strength'&&ys.value==='man')?'true':'false');}
  xs.addEventListener('change',sel);ys.addEventListener('change',sel);draw();onRedraw(draw);
  $('dxMan').addEventListener('click',()=>{const on=$('dxMan').getAttribute('aria-pressed')==='true';if(on){xs.value='age';ys.value='strength';}else{xs.value='strength';ys.value='man';}sel();
    $('dataInfo').innerHTML=on?'재령(가로)과 강도(세로)다. 처음 28일까지 가파르게 오르고 그 뒤로 완만해지는 휘어짐이 보인다.':'가로는 실제 강도, 세로는 매뉴얼 씨의 경험식이 낸 예측이다. 점선(예측 = 실제)에 가까울수록 잘 맞힌 것이다. 시험지 258건에서 R² <b>'+f3(PASS_R2)+'</b>, MAE <b>'+fx(MAN_MAE,2)+'MPa</b>. 학습 없이 교과서 식 두 개로 이만큼 맞힌다. 이것이 회귀팀 합격선이다. 고강도 쪽에서 아래로 처지는 배합이 많다. 감수제를 많이 넣은 새 배합은 경험식이 모르는 영역이다.';});
  const cnt=[0,0,0],tst=[0,0,0];ROWS.forEach(r=>{cnt[r.grade]++;if(r.test)tst[r.grade]++;});
  $('dataLeg').innerHTML=GRADES.map((g,k)=>'<span><i style="background:'+GRADECOL[k]+'"></i>'+g+' ('+(k===0?'25MPa 미만':k===1?'25~45':'45 이상')+') '+cnt[k]+'건 · 시험 '+tst[k]+'</span>').join('')+'<span>○ 속이 빈 점 = 봉인된 시험지</span>';
  $('concTbl').innerHTML='<table><thead><tr><th>#</th>'+FEAT.map(f=>'<th><span class="tg x">특징 X</span>'+FEATK[f]+'</th>').join('')+'<th><span class="tg t">타겟 y</span>압축 강도<br>(MPa)</th><th>매뉴얼 예측<br>(MPa)</th><th>용도</th></tr></thead><tbody>'+
    ROWS.map(r=>'<tr><td>'+(r.i+1)+'</td>'+FEAT.map(f=>'<td>'+r[f]+'</td>').join('')+'<td class="y">'+r.strength+'</td><td>'+r.man.toFixed(1)+'</td><td>'+(r.test?'<span class="pill no">봉인</span>':'훈련')+'</td></tr>').join('')+'</tbody></table>';
  /* 차수별 특징 수 */
  function pf(d){const C=PR.combos(8,d),by=[0,0,0,0,0];C.forEach(t=>by[t.length]++);const p=C.length;
    let h=statGrid([[String(p),d+'차 다항 특징 수','a'],[String(p+1),'외울 숫자 (가중치 + 절편)','r'],[(772/(p+1)).toFixed(1),'숫자 하나당 훈련 배합 수',p+1>300?'s':'']]);
    h+='<div class="tw" style="margin-top:10px"><table class="tb" style="font-size:13.5px"><thead><tr><th>항의 차수</th><th>개수</th><th>예</th></tr></thead><tbody>';
    for(let k=1;k<=d;k++){const ex=C.filter(t=>t.length===k);const pick=[0,1,ex.length-1].filter((v,i,a)=>v<ex.length&&a.indexOf(v)===i).map(i=>PR.name(ex[i]));
      h+='<tr><td>'+k+'차 항</td><td>'+by[k]+'</td><td>'+pick.join(', ')+(ex.length>3?' …':'')+'</td></tr>';}
    h+='</tbody></table></div><p class="foot">'+(d===1?'1차는 원래 특징 8개 그대로다. 5-2 선형 회귀와 같다.':d===2?'2차부터 시멘트² 같은 제곱 항과 시멘트·재령 같은 상호작용 항(28개)이 붙는다.':d===3?'3차면 164개다. scikit-learn PolynomialFeatures(3)의 기본값(include_bias=True)은 상수 열 1을 하나 더 만들어 165개로 센다. 이 면접은 회귀 모델이 절편을 따로 두므로 상수 열을 빼고 쓴다.':'4차면 494개다. 훈련 배합 772건에 숫자 495개를 맞추면, 숫자 하나에 배합 1.6건꼴이다. 외우기에 딱 좋은 조건이다(05장).')+'</p>';
    $('pfInfo').innerHTML=h;}
  seg('#pfChoice button',v=>pf(+v));pf(3);})();

/* ── 03 AI 비서 ── */
const MAIN_PROMPT=`너는 머신러닝 수업 조교다. Google Colab에서 바로 실행되는 파이썬 코드를 써 줘.

[환경] Colab 기본 라이브러리(numpy, pandas, scikit-learn)와 ucimlrepo만 쓴다. 첫 줄에 scikit-learn 버전을 출력한다.
[데이터] 〔ucimlrepo의 fetch_ucirepo(id=165)〕 로 콘크리트 압축 강도 데이터를 읽는다.
  - 특징 X: 〔시멘트, 고로 슬래그, 플라이 애시, 물, 고성능 감수제, 굵은 골재, 잔골재, 재령〕 8개 (열 순서대로 이름을 붙인다)
  - 타겟 y: 〔압축 강도(MPa)〕 (회귀)
[규정]
  1) 강도를 〔25, 45〕MPa로 나눈 3등급으로 stratify 해서 train_test_split(test_size=0.25, random_state=42)로 먼저 나눈다.
  2) Pipeline으로 StandardScaler → PolynomialFeatures(degree=〔3〕, include_bias=False) → StandardScaler → 모델 순서로 묶는다. 모든 fit은 훈련 데이터로만 한다.
  3) 모델은 LinearRegression, Ridge, Lasso, ElasticNet(l1_ratio=0.5)을 각각 만든다. normalize 같은 없어진 인자는 쓰지 않는다.
  4) alpha는 훈련 데이터 안의 5겹 교차 검증(GridSearchCV 또는 RidgeCV·LassoCV)으로 고른다.
[출력] 차수 1~〔4〕별 훈련·시험 R²와 시험 MAE, 고른 alpha, 라쏘에서 0이 된 가중치 수와 남은 항 이름(get_feature_names_out).
[금지] 시험 데이터로 학습하거나 alpha·차수를 고르지 않는다. 데이터를 지어내지 않는다.
[형식] 셀 단위로 나누고, 각 셀 위에 무엇을 하는지 한국어 주석을 단다. 코드 뒤에 내가 확인해야 할 점 3가지를 적어 줘.`;
$('mainPrompt').textContent=MAIN_PROMPT;$('copyMain').addEventListener('click',()=>copyText(MAIN_PROMPT,'프롬프트를 복사했습니다'));
(function(){const LINES=[
 ['import numpy as np'],['from ucimlrepo import fetch_ucirepo'],['from sklearn.model_selection import train_test_split'],['from sklearn.preprocessing import StandardScaler, PolynomialFeatures'],['from sklearn.linear_model import Lasso, Ridge'],[''],
 ['uci = fetch_ucirepo(id=165)',0,'UCI에서 콘크리트 데이터를 실제로 받았다. 괜찮다.'],
 ['X, y = uci.data.features, uci.data.targets.iloc[:, 0]   # 강도(MPa)',0,'특징 여덟 열과 강도 숫자다. 회귀이므로 강도를 그대로 쓴다. 괜찮다.'],
 ['X = StandardScaler().fit_transform(X)   # 표준화',1,'<b>데이터 누수.</b> 나누기 전에 전체(시험지 포함)로 평균·표준편차를 구했다. 시험지 정보가 훈련에 스며든다. 먼저 나누고, 표준화와 다항 특징은 파이프라인으로 묶어 훈련 데이터로만 fit 한다.'],
 ['X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)',2,'<b>stratify와 random_state가 없다.</b> 돌릴 때마다 점수가 달라진다. 다항 회귀는 시험지에 극단적인 배합이 몇 건 들어가느냐에 따라 R²가 크게 흔들리므로 더 중요하다. 강도 등급으로 <code>stratify</code>하고 <code>random_state=42</code>를 넣는다.'],[''],
 ['poly = PolynomialFeatures(degree=10)   # 곡선을 넉넉하게',3,'<b>차수 10은 특징 폭발이다.</b> 특징 8개의 10차 다항 특징은 상수 열까지 43,758개로, 훈련 배합 772건의 57배다. 외우기만 하고 메모리도 많이 먹는다. 2·3차부터 시작해 교차 검증으로 정한다. 다항 특징 뒤에 다시 표준화하는 단계도 빠졌다.'],
 ['X_train = poly.fit_transform(X_train)',0,'훈련 데이터로 fit 했다. 괜찮다(PolynomialFeatures는 열 개수만 배운다).'],
 ['X_test = poly.transform(X_test)',0,'시험지는 transform만 했다. 괜찮다.'],[''],
 ['model = Lasso(alpha=0.1, normalize=True).fit(X_train, y_train)',4,'<b>없어진 인자.</b> <code>normalize</code>는 scikit-learn 1.2에서 없어져 최신 버전에서는 TypeError가 난다. 파이프라인에 <code>StandardScaler</code>를 넣는다.'],
 ['for a in [0.01, 0.1, 1, 10]:   # 가장 좋은 alpha 찾기',5,'<b>시험지로 α를 골랐다.</b> 시험지 점수를 보고 α를 고르면 시험지가 훈련 데이터가 된다. 마지막 점수를 믿을 수 없다. <code>RidgeCV</code>나 <code>GridSearchCV</code>로 훈련 데이터 안에서 교차 검증한다(05장).'],
 ['    print(a, Ridge(alpha=a).fit(X_train, y_train).score(X_test, y_test))',5,'<b>시험지로 α를 골랐다.</b> 바로 윗줄과 같은 문제다. 이 데이터의 4차 릿지에서는 시험지로 고르면 α = 10(R² 0.862), 교차 검증으로 고르면 α = 100(R² 0.821)이다. 0.862는 시험지를 엿본 점수다.'],
 ['print("R²:", model.score(X_test, y_test))',0,'마지막에 시험지로 한 번 채점했다. 괜찮다.']];
  const host=$('bugCode'),found=new Set();
  host.innerHTML=LINES.map((l,i)=>'<div class="ln" role="listitem" tabindex="0" data-i="'+i+'"><span class="no">'+(i+1)+'</span><span>'+hl(l[0]||' ')+'</span></div>').join('');
  function score(){$('bugScore').textContent=found.size+' / 5';if(found.size===5){$('bugScore').className='pill ok';}}
  function pick(el){const i=+el.getAttribute('data-i'),l=LINES[i];if(!l[0])return;
    if(l[1]){found.add(l[1]);host.querySelectorAll('.ln').forEach(x=>{if(LINES[+x.getAttribute('data-i')][1]===l[1])x.classList.add('hit');});$('bugMsg').innerHTML=(i+1)+'번 줄: '+l[2]+(found.size===5?'<br><b>다섯 군데를 모두 찾았다.</b> 고친 코드는 A장의 Colab 코드와 같다.':'');}
    else{el.classList.add('miss');$('bugMsg').innerHTML=(i+1)+'번 줄: '+(l[2]||'이 줄은 괜찮다.');}score();}
  host.addEventListener('click',e=>{const el=e.target.closest('.ln');if(el)pick(el);});
  host.addEventListener('keydown',e=>{if((e.key==='Enter'||e.key===' ')&&e.target.classList.contains('ln')){e.preventDefault();pick(e.target);}});
  $('bugReveal').addEventListener('click',()=>{LINES.forEach((l,i)=>{if(l[1]){found.add(l[1]);host.querySelector('[data-i="'+i+'"]').classList.add('hit','reveal');}});score();$('bugMsg').innerHTML='빨간 줄 다섯 군데: ① 9번 데이터 누수 ② 10번 나누기 규정 ③ 12번 차수 10(특징 폭발) ④ 16번 없어진 인자 normalize ⑤ 17~18번 시험지로 α 고르기.';});
  $('bugReset').addEventListener('click',()=>{found.clear();host.querySelectorAll('.ln').forEach(x=>x.classList.remove('hit','miss','reveal'));$('bugScore').className='pill';score();$('bugMsg').textContent='줄을 하나 눌러 보세요.';});score();})();
(function(){const CK=[['데이터를 실제로 읽었나','AI가 예제 데이터(make_regression, 당뇨병 데이터 등)로 바꾸지 않았는지, 행이 1,030개인지 본다.'],['타겟이 강도 숫자이고, 특징에 강도가 섞이지 않았나','강도가 특징에 들어가면 정답을 보고 푸는 셈이다(정답 누설). R²가 0.99가 넘으면 먼저 의심한다.'],['나누기를 먼저 하고, 표준화·다항 특징은 파이프라인으로 훈련 데이터에만 fit 했나','fit_transform(X) 전체가 보이면 누수다.'],['stratify와 random_state를 넣었나','등급 비율이 시험지에도 유지되고, 점수가 매번 같게 나온다.'],['차수와 특징 수를 확인했나','poly.n_output_features_를 찍어 훈련 배합 수와 비교한다. 특징이 배합 수에 가까우면 규제 없이는 위험하다.'],['다항 특징 뒤에 표준화가 있나','규제는 가중치 크기에 벌점을 준다. 열마다 단위가 다르면 벌점이 불공평하다.'],['α를 교차 검증으로 골랐나','시험지 점수를 보고 α·차수를 고르면 시험지가 훈련 데이터가 된다.'],['인자가 내 scikit-learn 버전에 있나','normalize(1.2에서 없어짐), get_feature_names(→ get_feature_names_out)를 확인한다. ConvergenceWarning은 max_iter를 늘리라는 뜻이다.'],['결과를 기준선·합격선과 비교하고, 훈련과 시험 R²를 나란히 봤나','늘 평균(R² 0, MAE 13.5MPa)과 매뉴얼 씨(R² 0.735)보다 낫지 않으면 뽑을 까닭이 없다. 훈련만 높으면 과대적합이다.']];
  const ul=$('checks');ul.innerHTML=CK.map((c,i)=>'<li><input type="checkbox" id="ck'+i+'"><label for="ck'+i+'"><b>'+c[0]+'</b><small>'+c[1]+'</small></label></li>').join('');
  function upd(){const n=ul.querySelectorAll('input:checked').length;ul.querySelectorAll('li').forEach(li=>li.classList.toggle('done',li.querySelector('input').checked));$('checkScore').textContent='확인 '+n+' / '+CK.length+(n===CK.length?' — 이제 돌려 본다. 결과도 확인 대상이다.':'');}
  ul.addEventListener('change',upd);upd();})();

/* ── 04 수습 기간: 3차 다항 특징 164개의 규제 경로와 좌표 하강 ── */
(function(){const cc=$('coefCv'),pc=$('pathCv'),HC=230,HP=220,CAP=3000,LMIN=-3,LMAX=2;
  const st={kind:'ridge',la:0,w:null,cd:null,cdA:null,path:{ridge:{},lasso:{},enet:{}},play:false,raf:0,last:null,note:''};
  const rho=()=>st.kind==='lasso'?1:0.5;
  function ds(){return D(3);}
  function setW(w){st.w=w;const r=evalW(ds(),w);st.last=r;st.path[st.kind][st.la.toFixed(1)]={tr:r.tr,te:r.te,z:r.zeros};return r;}
  function solve(){const a=Math.pow(10,st.la);if(st.kind==='ridge')return setW(PR.ridgeSolve(ds().gm,a));
    const s=PR.cdState(ds().gm,st.w&&st.cdA===st.kind?st.w:null);for(let k=0;k<CAP;k++){if(PR.cdSweep(ds().gm,s,a,rho())<1e-6)break;}st.cdA=st.kind;st.cd=null;return setW(s.w);}
  function drawCoef(){const W=fit(cc,HC),ctx=cc.getContext('2d'),C=cols(),pl=8,pr=8,pt=22,pb=26,gw=W-pl-pr,gh=HC-pt-pb,mid=pt+gh/2,M=60,w=st.w,cs=ds().C,gap=10,n=w.length,bw=(gw-2*gap)/n;
    ctx.clearRect(0,0,W,HC);ctx.strokeStyle=C.muted;ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(pl,mid);ctx.lineTo(pl+gw,mid);ctx.stroke();
    let x=pl,prev=1;const starts=[];
    for(let j=0;j<n;j++){const dg=cs[j].length;if(dg!==prev){x+=gap;prev=dg;}if(!starts[dg])starts[dg]=x;const v=w[j],h=Math.sign(v)*Math.sqrt(Math.min(1,Math.abs(v)/M))*(gh/2-2);
      if(v===0){ctx.fillStyle=rgba(C.muted,.6);ctx.fillRect(x+bw*0.2,mid-1,Math.max(1,bw*0.6),2);}else{ctx.fillStyle=v>0?C.blue:C.stamp;ctx.fillRect(x+bw*0.1,h>0?mid-h:mid,Math.max(1,bw*0.8),Math.abs(h));}x+=bw;}
    ctx.fillStyle=C.muted;ctx.font=font(12,700);ctx.textAlign='left';[[1,'1차 8'],[2,'2차 36'],[3,'3차 120']].forEach(([k,l])=>ctx.fillText(l,starts[k],pt-8));
    ctx.font=font(11);ctx.textAlign='right';ctx.fillText('막대 길이 = √|w| 눈금 (|w| = 60에서 끝)',pl+gw,HC-6);ctx.textAlign='left';ctx.fillStyle=C.blue;ctx.fillText('■ +',pl,HC-6);ctx.fillStyle=C.stamp;ctx.fillText('■ −',pl+30,HC-6);ctx.fillStyle=C.muted;ctx.fillText('▬ 0',pl+60,HC-6);}
  function drawPath(){const W=fit(pc,HP),ctx=pc.getContext('2d'),C=cols(),pl=44,pr=st.kind==='ridge'?14:40,pt=12,pb=34,gw=W-pl-pr,gh=HP-pt-pb,lx=l=>pl+(l-LMIN)/(LMAX-LMIN)*gw,Y0=0.4,ly=v=>pt+(1-(clamp(v,Y0,1)-Y0)/(1-Y0))*gh;
    ctx.clearRect(0,0,W,HP);ctx.strokeStyle=C.rule;ctx.lineWidth=1;ctx.strokeRect(pl,pt,gw,gh);ctx.fillStyle=C.muted;ctx.font=font(12);ctx.textAlign='center';
    for(let l=LMIN;l<=LMAX;l++)ctx.fillText(l<0?String(Math.pow(10,l)):String(Math.pow(10,l)),lx(l),HP-18);ctx.fillText('α (로그 눈금) — 오른쪽일수록 규제가 세다',pl+gw/2,HP-3);
    ctx.textAlign='right';[0.4,0.6,0.8,1].forEach(v=>ctx.fillText(v.toFixed(1),pl-5,ly(v)+4));
    ctx.setLineDash([5,4]);ctx.strokeStyle=C.c_man;ctx.beginPath();ctx.moveTo(pl,ly(PASS_R2));ctx.lineTo(pl+gw,ly(PASS_R2));ctx.stroke();ctx.setLineDash([]);ctx.fillStyle=C.c_man;ctx.textAlign='left';ctx.fillText('합격선 R² '+f3(PASS_R2),pl+6,ly(PASS_R2)+14);
    const P=st.path[st.kind],ks=Object.keys(P).map(Number).sort((a,b)=>a-b);
    [['tr',C.muted,'훈련 R²'],['te',C.blue,'시험 R²']].forEach(([k,col,lab])=>{ctx.strokeStyle=col;ctx.lineWidth=2.5;ctx.beginPath();ks.forEach((l,i)=>{const p=P[l.toFixed(1)];i?ctx.lineTo(lx(l),ly(p[k])):ctx.moveTo(lx(l),ly(p[k]));});ctx.stroke();
      if(ks.length){const l=ks[0],p=P[l.toFixed(1)];ctx.fillStyle=col;ctx.font=font(12,700);ctx.textAlign='left';ctx.fillText(lab,lx(l)+6,ly(p[k])+(k==='tr'?-6:14));}});
    if(st.kind!=='ridge'){const lz=z=>pt+(1-z/164)*gh;ctx.strokeStyle=C.amber;ctx.lineWidth=2;ctx.setLineDash([3,3]);ctx.beginPath();ks.forEach((l,i)=>{const p=P[l.toFixed(1)];i?ctx.lineTo(lx(l),lz(p.z)):ctx.moveTo(lx(l),lz(p.z));});ctx.stroke();ctx.setLineDash([]);
      ctx.fillStyle=C.amber;ctx.font=font(11);ctx.textAlign='left';[0,82,164].forEach(z=>ctx.fillText(z,pl+gw+4,lz(z)+4));ctx.save();ctx.translate(W-6,pt+gh/2);ctx.rotate(Math.PI/2);ctx.textAlign='center';ctx.fillText('0인 가중치 수 (점선)',0,0);ctx.restore();}
    ctx.strokeStyle=C.stamp;ctx.lineWidth=1.5;ctx.beginPath();ctx.moveTo(lx(st.la),pt);ctx.lineTo(lx(st.la),pt+gh);ctx.stroke();}
  function info(){const r=st.last,a=Math.pow(10,st.la),w=st.w;let big=0;for(let j=1;j<w.length;j++)if(Math.abs(w[j])>Math.abs(w[big]))big=j;
    const nm={ridge:'릿지',lasso:'라쏘',enet:'엘라스틱넷'}[st.kind];let s=nm+' · α = <b>'+fmtA(a)+'</b> · 훈련 R² <b>'+f3(r.tr)+'</b> · 시험 R² <b>'+f3(r.te)+'</b> · 시험 MAE '+fx(r.mae,2)+'MPa · 0인 가중치 <b>'+r.zeros+'</b> / 164 · 가장 큰 가중치 '+PR.name(ds().C[big])+' = '+fx(w[big],1);
    if(st.note)s+='<br>'+st.note;
    else if(st.kind==='ridge')s+='<br>'+(st.la>=1.5?'α가 크면 모든 막대가 고루 짧아진다. 그래도 정확히 0인 막대는 없다. 릿지는 "줄이기"만 한다.':st.la<=-1.5?'α가 작으면 규제가 거의 없다. 3차 항 막대가 길게 자라 서로 상쇄하며 훈련 배합을 쫓아간다.':'α를 오른쪽 끝(100)에서 왼쪽 끝(0.001)으로 재생해 보자. 막대가 자라면서 훈련 R²는 계속 오르고, 시험 R²는 어느 지점부터 더 오르지 않는다.');
    else s+='<br>'+(r.zeros>=150?'α가 커서 거의 모든 가중치가 0이다. 살아남은 몇 개(재령 쪽 항이 먼저 산다)만으로 예측한다.':r.zeros>=60?'회색 막대(0)가 많다. '+nm+'는 쓸모가 적은 항을 아예 지운다. 지워도 시험 R²는 크게 떨어지지 않는다.':'α가 작아 지우는 항이 줄었다. 3차 항까지 살아나며 막대가 길어진다.')+(st.kind==='enet'?' 엘라스틱넷은 L2가 섞여 같은 α에서 라쏘보다 덜 지우고, 닮은 항들을 함께 남긴다.':'');
    $('regInfo').innerHTML=s;}
  function draw(){if(!st.w)return;drawCoef();drawPath();info();}
  function sync(){$('rgA').value=st.la;$('rgAO').textContent=fmtA(Math.pow(10,st.la));$('rgStep').disabled=st.kind==='ridge';}
  function stop(){st.play=false;cancelAnimationFrame(st.raf);$('rgPlay').textContent='▶ α 경로 재생 (세게 → 약하게)';}
  function frame(){if(!st.play)return;solve();draw();if(st.la<=LMIN+1e-9){stop();st.note='재생 끝. 선 그림에서 훈련 R²(회색)는 α가 작아질수록 계속 오르지만, 시험 R²(파랑)는 α ≈ 0.1~1 근처부터 평평하다. 고삐를 더 풀어도 처음 보는 배합에서는 나아지지 않는다.';draw();return;}
    st.la=Math.round((st.la-0.1)*10)/10;sync();st.raf=requestAnimationFrame(frame);}
  $('rgPlay').addEventListener('click',()=>{if(st.play){stop();return;}st.note='';st.play=true;st.la=LMAX;st.w=null;st.cdA=null;st.path[st.kind]={};sync();$('rgPlay').textContent='❚❚ 멈춤';
    if(REDUCE){while(st.la>LMIN){solve();st.la=Math.round((st.la-0.1)*10)/10;}solve();stop();draw();return;}st.raf=requestAnimationFrame(frame);});
  $('rgStep').addEventListener('click',()=>{stop();const a=Math.pow(10,st.la);if(!st.cd){st.cd=PR.cdState(ds().gm);}const dw=PR.cdSweep(ds().gm,st.cd,a,rho());st.w=Float64Array.from(st.cd.w);const r=evalW(ds(),st.w);st.last=r;
    st.note='좌표 하강 <b>'+st.cd.sweeps+'</b>바퀴째 (모두 0에서 출발). 이번 바퀴에서 가장 크게 바뀐 가중치 변화 '+fx(dw,4)+'. '+(dw<1e-6?'<b>더 바뀌지 않는다 — 수렴.</b>':st.cd.sweeps<4?'첫 바퀴들에서 재령 항이 먼저 자라고, 남은 몫이 α보다 작은 항은 0에 머문다.':'바퀴가 거듭될수록 변화가 작아진다. 바뀌는 양이 거의 0이 되면 멈춘다.');drawCoef();info();st.cdA=null;});
  $('rgReset').addEventListener('click',()=>{stop();st.la=0;st.cd=null;st.cdA=null;st.w=null;st.note='';st.path[st.kind]={};sync();solve();draw();});
  $('rgA').addEventListener('input',e=>{stop();st.la=+e.target.value;st.cd=null;st.note='';sync();solve();draw();});
  seg('#regChoice button',v=>{stop();st.kind=v;st.cd=null;st.cdA=null;st.w=null;st.note='';sync();solve();draw();});
  /* 처음 화면: 릿지 α = 1. 가까워지면 계산한다 */
  let ready=false;function init(){if(ready)return;ready=true;sync();solve();draw();onRedraw(draw);}
  if('IntersectionObserver' in window){const io=new IntersectionObserver(es=>{if(es.some(e=>e.isIntersecting)){init();io.disconnect();}},{rootMargin:'400px'});io.observe($('m4'));}else init();
  window.addEventListener('hashchange',()=>{if(location.hash==='#m4')init();});})();

/* ── 05 성과 평가 ── */
function renderEval(){const C=cols(),m3=ridge(3,0);
  $('evalStat').innerHTML=statGrid([[f3(m3.te),'시험 R² (258건)','a'],[fx(m3.mae,2)+'MPa','시험 MAE (평균 몇 MPa 틀리나)','a'],[f3(m3.tr),'훈련 R² (772건)'],[f3(PASS_R2),'합격선 R² · 매뉴얼 씨 (MAE '+fx(MAN_MAE,2)+')','m'],[fx(BASE_MAE,2)+'MPa','기준선 MAE · 늘 평균 (R² '+f3(BASE_R2)+')']]);
  /* 차수별 */
  const DEG=[1,2,3,4].map(d=>ridge(d,0)),dc=$('degCv');
  function drawDeg(){const W=fit(dc,250),ctx=dc.getContext('2d'),C=cols(),pl=40,pr=10,pt=14,pb=32,gw=W-pl-pr,gh=250-pt-pb,Y0=-0.3,ly=v=>pt+(1-(clamp(v,Y0,1)-Y0)/(1-Y0))*gh,cw=gw/4;
    ctx.clearRect(0,0,W,250);ctx.fillStyle=C.muted;ctx.font=font(12);ctx.textAlign='right';[0,0.5,1].forEach(v=>{ctx.fillText(v.toFixed(1),pl-5,ly(v)+4);ctx.strokeStyle=C.rule;ctx.beginPath();ctx.moveTo(pl,ly(v));ctx.lineTo(pl+gw,ly(v));ctx.stroke();});
    ctx.setLineDash([5,4]);ctx.strokeStyle=C.c_man;ctx.beginPath();ctx.moveTo(pl,ly(PASS_R2));ctx.lineTo(pl+gw,ly(PASS_R2));ctx.stroke();ctx.setLineDash([]);
    DEG.forEach((r,i)=>{const x=pl+i*cw+cw*0.18,bw=cw*0.3;[[r.tr,C.muted],[r.te,C.blue]].forEach(([v,col],k)=>{const y=ly(v),y0=ly(0);ctx.fillStyle=col;ctx.fillRect(x+k*bw,Math.min(y,y0),bw-3,Math.abs(y0-y));
        ctx.font=font(11,700,1);ctx.textAlign='center';ctx.fillStyle=v<Y0?C.stamp:col;ctx.fillText(v<Y0?fx(v,1)+'↓':fx(v,2),x+k*bw+bw/2-1,v<Y0?ly(Y0)-4:y-4);});
      ctx.fillStyle=C.text;ctx.font=font(12,700);ctx.textAlign='center';fitText(ctx,(i+1)+'차 · '+r.p+'개',pl+i*cw+cw/2,250-12,cw-8);});
    ctx.font=font(11);ctx.textAlign='left';ctx.fillStyle=C.muted;ctx.fillText('■ 훈련',pl+4,pt+10);ctx.fillStyle=C.blue;ctx.fillText('■ 시험',pl+52,pt+10);ctx.fillStyle=C.c_man;ctx.fillText('┅ 합격선',pl+100,pt+10);}
  drawDeg();onRedraw(drawDeg);
  $('degInfo').innerHTML='1차 '+f3(DEG[0].te)+' → 2차 '+f3(DEG[1].te)+' → 3차 <b>'+f3(DEG[2].te)+'</b>까지 시험 R²가 오른다. 4차는 훈련 R² '+f3(DEG[3].tr)+'로 훈련 배합을 거의 외웠지만 시험 R²는 <b style="color:var(--stamp)">'+fx(DEG[3].te,1)+'</b>다. 늘 평균을 답하는 기준선(0)보다 훨씬 나쁘다. <span class="muted">Colab(scikit-learn)에서는 같은 4차가 −6.18로 나온다. 특징 494개가 서로 거의 같은 정보를 담고 있어 풀이가 아주 작은 계산 차이에도 크게 흔들린다. 숫자가 계산 방법마다 다르다는 것 자체가 폭주의 증거다.</span>';
  /* 예측 대 정답 */
  const pv=$('pvCv'),PV={man:{p:MANTE,lab:'매뉴얼 씨'},1:{p:DEG[0].pte,lab:'1차'},3:{p:DEG[2].pte,lab:'3차'},4:{p:DEG[3].pte,lab:'4차 (규제 없음)'}};let pvk='3';
  function drawPv(){const k=pvk;if(k==='4r'&&!PV['4r']){const r=ridge(4,100);PV['4r']={p:r.pte,lab:'4차 + 릿지 α = 100'};}
    const W=fit(pv,280),ctx=pv.getContext('2d'),C=cols(),s=Math.min(W-60,230),pl=Math.max(46,(W-s)/2),pt=10,V=90,sc=v=>v/V*s,P=PV[k].p;
    ctx.clearRect(0,0,W,280);ctx.strokeStyle=C.rule;ctx.strokeRect(pl,pt,s,s);ctx.setLineDash([4,4]);ctx.strokeStyle=C.muted;ctx.beginPath();ctx.moveTo(pl,pt+s);ctx.lineTo(pl+s,pt);ctx.stroke();ctx.setLineDash([]);
    ctx.fillStyle=C.muted;ctx.font=font(11);ctx.textAlign='center';[0,30,60,90].forEach(v=>{ctx.fillText(v,pl+sc(v),pt+s+13);});ctx.textAlign='right';[0,30,60,90].forEach(v=>ctx.fillText(v,pl-4,pt+s-sc(v)+4));
    ctx.textAlign='center';ctx.font=font(12);ctx.fillText('실제 강도 (MPa)',pl+s/2,pt+s+28);ctx.save();ctx.translate(pl-30,pt+s/2);ctx.rotate(-Math.PI/2);ctx.fillText('예측 (MPa)',0,0);ctx.restore();
    let out=0,neg=0;P.forEach((p,i)=>{const x=pl+sc(YTE[i]);if(p<0)neg++;if(p<0||p>V){out++;const y=p<0?pt+s-2:pt+2;ctx.fillStyle=C.stamp;ctx.beginPath();if(p<0){ctx.moveTo(x-4,y-6);ctx.lineTo(x+4,y-6);ctx.lineTo(x,y);}else{ctx.moveTo(x-4,y+6);ctx.lineTo(x+4,y+6);ctx.lineTo(x,y);}ctx.fill();}
      else circ(ctx,x,pt+s-sc(p),3,rgba(k==='man'?C.c_man:C.blue,.6),null);});
    const r2=PR.r2(YTE,P),mae=PR.mae(YTE,P);
    $('pvInfo').innerHTML=PV[k].lab+': 시험 R² <b>'+f3(r2)+'</b> · MAE <b>'+fx(mae,2)+'MPa</b>'+(out?' · <span style="color:var(--stamp)">그림 밖(0 미만 또는 90 초과)으로 튄 예측 '+out+'건'+(neg?', 그중 음수 강도 '+neg+'건':'')+'</span> (빨간 세모)':'')+'. '+
      (k==='man'?'경험식은 고강도 배합을 낮게 부르는 경향이 있다(대각선 아래).':k==='1'?'직선은 낮은 강도를 높게, 높은 강도를 낮게 부른다. 휘어짐을 못 따라가서다.':k==='3'?'대각선 둘레에 고르게 모인다.':k==='4'?'대부분은 대각선 가까이 있지만 몇 건이 엄청나게 빗나간다. R²는 제곱 오차라서 이 몇 건이 점수를 무너뜨린다.':'고삐를 매자 튀던 예측이 대각선 둘레로 돌아왔다.');}
  seg('#pvChoice button',v=>{pvk=v;drawPv();});drawPv();onRedraw(drawPv);
  /* 4차 + 릿지 */
  function d4(a){const r=ridge(4,a);let mx=0;r.w.forEach(v=>{if(Math.abs(v)>mx)mx=Math.abs(v);});
    $('d4Stat').innerHTML=statGrid([[f3(r.tr),'훈련 R²'],[r.te<-1?fx(r.te,1):f3(r.te),'시험 R²',r.te>PASS_R2?'a':'s'],[fx(r.mae,2)+'MPa','시험 MAE'],[mx>=1000?Math.round(mx).toLocaleString():fx(mx,1),'가장 큰 가중치 |w|','r']]);
    $('d4Info').innerHTML=a===0?'규제 없이 494개를 맞추면 가장 큰 가중치가 '+Math.round(mx).toLocaleString()+'까지 커진다. 큰 가중치들이 서로 상쇄하며 훈련 배합 사이를 누빈다. 훈련 데이터 밖으로 조금만 나가도 상쇄가 깨져 예측이 튄다.':
      a<=0.1?'α = '+a+'만 걸어도 가중치가 크게 줄고 시험 R²가 음수에서 올라온다.':a<=1?'시험 R²가 '+f3(r.te)+'까지 돌아왔다.':a===10?'이 시험지에서는 α = 10이 가장 좋다(R² '+f3(r.te)+'). <b>그런데 이 값을 시험지를 보고 고르면 반칙이다.</b> 아래 교차 검증은 시험지를 보지 않고 α = 100을 고른다.':a===100?'훈련 데이터 안의 교차 검증이 고르는 값이다(아래). 시험 R² '+f3(r.te)+'로 합격선을 넘는다. 시험지를 엿보지 않고 얻은 정직한 점수다.':'너무 세다. 가중치가 눌려 훈련 R²도 떨어진다(과소적합).';}
  $('d4Choice').addEventListener('click',e=>{const b=e.target.closest('button');if(!b)return;$('d4Choice').querySelectorAll('button').forEach(x=>x.setAttribute('aria-pressed',x===b));d4(+b.getAttribute('data-v'));});d4(0);
  /* 비교 표 */
  const rows=[['규제 없음 (최소제곱)',ridge(3,0)],['릿지 α = 1',ridge(3,1)],['라쏘 α = 0.1',cdm(3,0.1,1)],['엘라스틱넷 α = 0.1, ρ = 0.5',cdm(3,0.1,0.5)]];
  $('cmpTbl').innerHTML='<thead><tr><th>3차 + …</th><th>훈련 R²</th><th>시험 R²</th><th>시험 MAE</th><th>0인 가중치</th><th>외울 숫자</th></tr></thead><tbody>'+rows.map(([n,r])=>'<tr><td>'+n+'</td><td>'+f3(r.tr)+'</td><td><b>'+f3(r.te)+'</b></td><td>'+fx(r.mae,2)+'</td><td>'+r.zeros+' / 164</td><td>'+(165-r.zeros)+'</td></tr>').join('')+
    '<tr><td class="muted">참고: 매뉴얼 씨 (학습 없음)</td><td>—</td><td>'+f3(PASS_R2)+'</td><td>'+fx(MAN_MAE,2)+'</td><td>—</td><td>상수 3개</td></tr></tbody>';
  /* 인사 카드: 라쏘가 남긴 항 */
  function wb(a){const r=cdm(3,a,1),cs=D(3).C,ord=[...r.w.keys()].filter(j=>r.w[j]!==0).sort((x,y)=>Math.abs(r.w[y])-Math.abs(r.w[x])),top=ord.slice(0,12),mx=Math.max(...top.map(j=>Math.abs(r.w[j])));
    $('wBars').innerHTML=top.map(j=>{const v=r.w[j],Wd=Math.abs(v)/mx*50;return '<div class="row"><span style="text-align:left;font-family:var(--font)">'+PR.name(cs[j])+'</span><div class="tr"><span class="z"></span><i class="'+(v<0?'neg':'')+'" style="left:'+(v<0?50-Wd:50)+'%;width:'+Wd+'%"></i></div><span>'+fx(v,1)+'</span></div>';}).join('');
    $('wInfo').innerHTML='α = '+a+': 164개 중 <b>'+(164-r.zeros)+'개</b>가 살아남았다(0인 가중치 '+r.zeros+'개) · 시험 R² '+f3(r.te)+'. 맨 위에 재령·재령²·재령³이 함께 있다. 강도가 재령에 따라 휘며 오르는 곡선을 세 항이 나눠 그린다(재령²이 음수라 곡선이 꺾인다). 물은 −, 시멘트·슬래그는 +로 경험칙과 같은 방향이다. '+
      '<span class="muted">주의: 라쏘는 "이 데이터, 이 α에서" 남은 항을 보여 줄 뿐 원인을 말하지 않는다. 서로 닮은 항 가운데 하나만 남기고 나머지를 지우는 경향이 있어 α나 데이터가 조금만 바뀌어도 남는 항이 바뀐다. 그리고 물-결합재비처럼 뜻 있는 새 특징을 만들지는 않는다. 라쏘는 특징공학이 아니라 규제다.</span>';}
  seg('#wChoice button',v=>wb(+v));wb(0.1);}

/* ── 교차 검증 (05장) ── */
(function(){const AL=[0.01,0.1,1,10,100,1000],cv=$('cvCv'),RES={};let cur=null;
  async function run(d,setP){if(RES[d])return RES[d];const out=AL.map(()=>0);
    for(let f=0;f<5;f++){setP(f/5);await wait(20);const tri=[],vai=[];XTR.forEach((x,i)=>(i%5===f?vai:tri).push(i));
      const Xa=tri.map(i=>XTR[i]),Xv=vai.map(i=>XTR[i]),ya=tri.map(i=>YTR[i]),yv=vai.map(i=>YTR[i]),m=ML.mean(ya),ds=PR.design(Xa,Xv,d,ML.scaler(Xa)),gm=PR.gram(ds.A,ya.map(v=>v-m));
      AL.forEach((a,k)=>{const w=PR.ridgeSolve(gm,a);out[k]+=PR.r2(yv,PR.predict(ds.B,w,m))/5;});}
    setP(1);let b=0;out.forEach((v,k)=>{if(v>out[b])b=k;});RES[d]={cv:out,best:AL[b],test:AL.map(a=>ridge(d,a).te)};return RES[d];}
  function draw(){const W=fit(cv,240),ctx=cv.getContext('2d'),C=cols(),pl=44,pr=14,pt=14,pb=34,gw=W-pl-pr,gh=240-pt-pb,lx=a=>pl+(Math.log10(a)+2)/5*gw,Y0=0.4,ly=v=>pt+(1-(clamp(v,Y0,1)-Y0)/(1-Y0))*gh;
    ctx.clearRect(0,0,W,240);ctx.strokeStyle=C.rule;ctx.lineWidth=1;ctx.strokeRect(pl,pt,gw,gh);ctx.fillStyle=C.muted;ctx.font=font(12);ctx.textAlign='center';AL.forEach(a=>ctx.fillText(a,lx(a),240-18));ctx.fillText('α (로그 눈금)',pl+gw/2,240-3);
    ctx.textAlign='right';[0.4,0.6,0.8,1].forEach(v=>ctx.fillText(v.toFixed(1),pl-5,ly(v)+4));
    ctx.setLineDash([5,4]);ctx.strokeStyle=C.c_man;ctx.beginPath();ctx.moveTo(pl,ly(PASS_R2));ctx.lineTo(pl+gw,ly(PASS_R2));ctx.stroke();ctx.setLineDash([]);
    if(!cur||!RES[cur]){ctx.fillStyle=C.muted;ctx.textAlign='center';ctx.fillText('버튼을 누르면 교차 검증 선이 그려진다',pl+gw/2,pt+gh/2);return;}
    const R=RES[cur];ctx.strokeStyle=C.blue;ctx.lineWidth=2.5;ctx.beginPath();AL.forEach((a,k)=>{k?ctx.lineTo(lx(a),ly(R.cv[k])):ctx.moveTo(lx(a),ly(R.cv[k]));});ctx.stroke();
    AL.forEach((a,k)=>{circ(ctx,lx(a),ly(R.cv[k]),4.5,C.blue,C.panel,1.5);circ(ctx,lx(a),ly(R.test[k]),4,null,C.muted,1.5);if(R.cv[k]<Y0){ctx.fillStyle=C.blue;ctx.font=font(10,700,1);ctx.textAlign='center';ctx.fillText(fx(R.cv[k],2)+'↓',lx(a),ly(Y0)-6);}});
    ctx.strokeStyle=C.stamp;ctx.lineWidth=1.5;ctx.beginPath();ctx.moveTo(lx(R.best),pt);ctx.lineTo(lx(R.best),pt+gh);ctx.stroke();
    ctx.font=font(12,700);ctx.textAlign='left';ctx.fillStyle=C.blue;ctx.fillText('● 교차 검증 평균 R² (훈련 데이터 안)',pl+6,pt+14);ctx.fillStyle=C.muted;ctx.fillText('○ 시험 R² (참고용, 고를 때 안 봄)',pl+6,pt+30);}
  async function go(d,btn){const pg=$('cvProg'),bar=pg.querySelector('i');pg.hidden=false;btn.disabled=true;$('cvInfo').textContent=d+'차 교차 검증 중… (다섯 칸 × α 여섯 개)';const t0=performance.now();
    const R=await run(d,p=>{bar.style.width=Math.round(p*100)+'%';});pg.hidden=true;btn.disabled=false;cur=d;draw();const k=AL.indexOf(R.best),tb=AL[R.test.indexOf(Math.max(...R.test))];
    $('cvInfo').innerHTML=d+'차 릿지: 교차 검증 평균 R²가 가장 높은 α = <b>'+R.best+'</b>(검증 '+f3(R.cv[k])+'). 그 α로 훈련 772건 전체에 다시 학습하고 시험지를 한 번 연다 → 시험 R² <b>'+f3(R.test[k])+'</b>'+(R.test[k]>PASS_R2?' (합격선 위)':'')+'. '+
      (tb!==R.best?'시험지로 골랐다면 α = '+tb+'(시험 R² '+f3(Math.max(...R.test))+')였다. 그 차이가 "엿보기"의 값이다. ':'시험지로 골랐어도 같은 α였다. ')+(d===4?'4차는 α가 작으면 검증 R²가 바닥이다. 폭주는 교차 검증에서도 드러난다. ':'')+'<span class="muted">('+Math.round(performance.now()-t0)+'ms)</span>';}
  $('cv3').addEventListener('click',e=>go(3,e.currentTarget));$('cv4').addEventListener('click',e=>go(4,e.currentTarget));draw();onRedraw(draw);})();

/* ── 면접 결과 발표 ── */
Interview('iv3',{title:'다항 회귀와 규제는 강도 예측 업무에 합격할까?',
  q:'조건: 봉인된 시험지 258건에서 시험 R²가 합격선(매뉴얼 씨의 경험식 R² '+f3(PASS_R2)+')과 기준선(늘 훈련 평균, R² ≈ 0)을 모두 넘으면 합격. MAE도 함께 본다. 학습은 브라우저에서 3차 다항 특징 164개, 규제 없이 한다.',
  fly:'훈련 점수 말고 시험 점수. 처음 보는 258건에서 평균 몇 MPa 틀려?',robo:'외울 숫자 몇 개, 학습에 몇 초?',
  async run(setP){setP(.3);await wait(40);const r=ridge(3,0);setP(1);await wait(20);const pass=r.te>PASS_R2&&r.te>BASE_R2;
    return {result:pass?'pass':'hold',flyAns:'시험 R² '+f3(r.te)+' · MAE '+fx(r.mae,2)+'MPa · 훈련 R² '+f3(r.tr),roboAns:'숫자 165개 (가중치 164 + 절편) · 학습 '+Math.max(1,Math.round(r.ms))+'ms (브라우저)',flyMood:pass?'arms':'coffee',flyMove:pass?'nod':'shake',roboMood:pass?'happy':'meh',roboMove:pass?'jump':'tilt',asstMood:pass?'spark':'sigh',docMood:pass?'magic':'scale',
      measure:'봉인된 시험지 결과예요. 3차 다항 회귀 R² <b>'+f3(r.te)+'</b>(MAE '+fx(r.mae,2)+'MPa), 매뉴얼 씨 '+f3(PASS_R2)+'(MAE '+fx(MAN_MAE,2)+'MPa), 늘 평균을 답하는 기준선 '+f3(BASE_R2)+'(MAE '+fx(BASE_MAE,2)+'MPa). 훈련 R² '+f3(r.tr)+'와의 차이 '+fx(r.tr-r.te,2)+'는 따로 적어 둘게요.',
      verdict:pass?'합격입니다. 회귀팀에 배치합니다. 근무 조건이 있어요. 차수는 3까지만 쓰고, 더 올리려면 릿지를 달아 α를 교차 검증으로 정합니다. 데이터 범위를 벗어난 배합은 매뉴얼 씨의 경험식과 나란히 봅니다.':'보류입니다. 합격선을 넘지 못했어요.',
      key:'곡선을 긋는 선형 회귀 — 합격선 R² '+f3(PASS_R2)+' → '+f3(r.te)+', MAE '+fx(MAN_MAE,2)+' → '+fx(r.mae,2)+'MPa',
      render(box){box.innerHTML='<div class="bars">'+[['3차 다항 회귀',r.te,'var(--blue)'],['매뉴얼 씨 (경험식)',PASS_R2,'var(--c-man)'],['기준선 (늘 평균)',Math.max(0,BASE_R2),'var(--muted)']].map(x=>'<div class="row"><span style="text-align:left;font-family:var(--font)">'+x[0]+'</span><div class="tr"><i style="left:0;width:'+Math.max(0,x[1])*100+'%;background:'+x[2]+'"></i></div><span>'+f3(x[1])+'</span></div>').join('')+'</div><p class="foot">막대 = 시험 R² (1이면 완벽, 0이면 늘 평균과 같음).</p>';}};},
  alts:[
   {label:'차수를 4로 (규제 없음)',async run(box){const r=ridge(4,0);box.innerHTML=statGrid([[fx(r.te,1),'시험 R²','s'],[f3(r.tr),'훈련 R²'],[fx(r.mae,2)+'MPa','시험 MAE'],[f3(PASS_R2),'합격선','m']])+'<p class="foot">훈련은 거의 만점인데 시험은 늘 평균을 답하는 것보다도 못하다. 판정: <b>보류</b>. 훈련 점수만 봤다면 최우수 지원자로 뽑았을 것이다. (Colab에서는 같은 4차가 −6.18로 나온다. 풀이가 불안정해 숫자는 흔들려도 결론은 같다.)</p>';}},
   {label:'4차 + 릿지 (α = 100, 교차 검증이 고른 값)',async run(box){const r=ridge(4,100);box.innerHTML=statGrid([[f3(r.te),'시험 R²',r.te>PASS_R2?'a':'s'],[f3(r.tr),'훈련 R²'],[fx(r.mae,2)+'MPa','시험 MAE'],[f3(PASS_R2),'합격선','m']])+'<p class="foot">같은 4차인데 고삐 하나로 폭주가 멈췄다. 판정: <b>'+(r.te>PASS_R2?/*i18n:verdict*/'합격':'보류')+'</b>. 다만 3차(규제 없음)보다 낫지는 않다. 차수를 올리는 것보다 알맞은 차수를 고르는 것이 먼저다. 시험지를 보고 α = 10을 골랐다면 0.862였겠지만 그건 반칙이다.</p>';}},
   {label:'3차 + 라쏘 (α = 0.1)',async run(box){const r=cdm(3,0.1,1);box.innerHTML=statGrid([[f3(r.te),'시험 R²',r.te>PASS_R2?'a':'s'],[r.zeros+' / 164','0으로 지운 가중치','r'],[String(165-r.zeros),'외울 숫자','r'],[fx(r.mae,2)+'MPa','시험 MAE']])+'<p class="foot">절반 넘게 지우고도 시험 R²는 거의 그대로다. 판정: <b>'+(r.te>PASS_R2?/*i18n:verdict*/'합격':'보류')+'</b>. MAE는 조금 늘었다. 예산 담당은 이쪽을, 실무 면접관은 MAE가 더 작은 규제 없는 3차를 고를 것이다. 무엇을 기준으로 뽑을지는 업무가 정한다.</p>';}}]});

/* 평가 장은 화면에 가까워지면 계산한다 */
(function(){const t=$('m5');let done=false;function go(){if(done)return;done=true;setTimeout(renderEval,10);}
  if('IntersectionObserver' in window){const io=new IntersectionObserver(es=>{if(es.some(e=>e.isIntersecting)){go();io.disconnect();}},{rootMargin:'600px'});io.observe(t);io.observe($('m6'));}else go();
  window.addEventListener('hashchange',go);if(location.hash)go();})();

/* ══ 확인 문제 ══ */
var QUIZ={
q1:[{q:'다항 회귀를 "선형" 회귀라고 부르는 까닭은?',o:[['가중치에 대해 선형이기 때문이다. 바뀐 것은 특징(x, x², x³ …)뿐이다',1,'모델은 여전히 특징마다 가중치를 곱해 더한다. 그래서 최소제곱·릿지·라쏘를 그대로 쓴다.'],['그림이 늘 직선이기 때문이다',0,'특징 축 하나로 보면 곡선이다. "선형"은 그림 모양이 아니라 가중치에 대한 식의 꼴을 말한다.'],['차수가 1이기 때문이다',0,'2차, 3차여도 가중치에 대해서는 선형이다.']]},
 {q:'01장 그림에서 차수 10, α = 0일 때 일어난 일은?',o:[['훈련 R²는 높은데 시험 R²가 무너졌다(과대적합)',1,'곡선이 훈련 배합 12건 사이에서 출렁이며 점 하나하나를 따라간다. 처음 보는 배합에서는 크게 빗나간다.'],['훈련 R²와 시험 R²가 함께 올랐다',0,'훈련 R²만 올랐다. 시험 R²는 음수까지 떨어진다.'],['곡선이 평평해졌다',0,'그것은 α가 아주 클 때(과소적합)의 모습이다.']]},
 {q:'α를 0에서 크게 올리면?',o:[['가중치가 0 쪽으로 눌려 곡선이 차분해지고, 너무 크면 평평해진다',1,'α는 가중치 크기에 주는 벌점의 세기다. 알맞으면 과대적합을 막고, 지나치면 과소적합이 된다.'],['차수가 자동으로 낮아진다',0,'차수(특징 수)는 그대로다. 높은 차수 항의 가중치가 작게 눌릴 뿐이다.'],['훈련 R²가 올라간다',0,'규제는 훈련 오차를 조금 희생해 처음 보는 데이터에서의 성적을 지킨다. 훈련 R²는 보통 내려간다.']]}],
q2:[{q:'이 면접의 타겟(y)은?',o:[['압축 강도(MPa) 숫자 그 자체',1,'회귀팀의 과제다. 5-4 분류팀은 같은 강도를 35MPa 문턱에 대어 이름표로 바꿔 썼다.'],['35MPa 합격 여부',0,'그것은 분류 과제(5-4)다. 같은 데이터, 다른 질문이다.'],['물-결합재비',0,'물-결합재비는 매뉴얼 씨의 경험식이 쓰는 특징이다.']]},
 {q:'특징 8개로 3차 다항 특징(상수 열 제외)을 만들면 몇 개가 되나?',o:[['164개',1,'1차 8 + 2차 36 + 3차 120 = 164. scikit-learn 기본값(include_bias=True)은 상수 열을 더해 165개로 센다.'],['24개',0,'8 × 3은 거듭제곱만 센 수다. x₁x₂ 같은 상호작용 항이 빠졌다.'],['512개',0,'8³은 순서를 따진 수다. 곱은 순서가 상관없다.']]},
 {q:'다항 특징을 만든 뒤 한 번 더 표준화하는 까닭은?',o:[['규제는 가중치 크기에 벌점을 주므로, 열마다 단위가 다르면 벌점이 불공평해서',1,'재령³과 감수제처럼 크기가 크게 다른 열을 같은 눈금에 맞춰야 α가 모든 항에 공평하게 걸린다.'],['예측이 더 빨라져서',0,'속도와는 거의 관계없다.'],['시험 데이터에 맞추려고',0,'표준화의 평균·표준편차는 훈련 데이터로만 구한다.']]}],
q3:[{q:'AI가 준 코드에 PolynomialFeatures(degree=10)이 있다. 가장 먼저 할 일은?',o:[['poly.n_output_features_로 특징 수를 찍어 훈련 배합 수와 비교하고, 낮은 차수부터 교차 검증한다',1,'특징 8개 10차는 상수 열까지 43,758개로 배합 772건보다 훨씬 많다. 외우기만 한다.'],['곡선이 넉넉하니 그대로 쓴다',0,'곡선이 넉넉한 것이 문제다. 4차만 돼도 이 데이터에서는 폭주했다.'],['α를 0으로 둔다',0,'특징이 배합보다 많으면 규제 없이는 풀이조차 하나로 정해지지 않는다.']]},
 {q:'Lasso(alpha=0.1, normalize=True)를 최신 scikit-learn에서 돌리면?',o:[['normalize가 1.2에서 없어져 오류가 난다. 파이프라인에 StandardScaler를 넣는다',1,'AI는 예전 코드를 많이 보고 배워 없어진 인자를 쓰는 일이 잦다. 버전을 찍고 공식 문서로 확인한다.'],['경고만 뜨고 결과는 같다',0,'1.0~1.1에서는 경고였지만 1.2부터는 오류다.'],['normalize가 표준화보다 정확해서 좋다',0,'인자 자체가 없어졌다.']]}],
q4:[{q:'릿지와 라쏘의 가장 큰 차이는?',o:[['라쏘는 쓸모가 적은 가중치를 정확히 0으로 만들고, 릿지는 고루 줄이기만 한다',1,'절댓값 벌점은 0에서 뾰족해 소프트 임계값으로 0이 된다. 제곱 벌점은 0 근처에서 평평해 0까지 밀지 않는다.'],['릿지가 늘 더 정확하다',0,'데이터에 따라 다르다. 이 면접에서는 시험 R²가 비슷했다.'],['라쏘는 다항 특징에 쓸 수 없다',0,'이 면접에서 3차 다항 특징 164개에 라쏘를 썼다.']]},
 {q:'릿지는 닫힌 해가 있는데 라쏘는 좌표 하강으로 푸는 까닭은?',o:[['|w|가 0에서 미분되지 않아 식 하나로 바닥을 구할 수 없어서',1,'그래서 가중치를 하나씩 돌아가며 소프트 임계값으로 고친다. 바뀌지 않을 때까지 반복한다.'],['라쏘가 더 오래된 방법이라서',0,'라쏘(1996)가 릿지(1970)보다 나중에 나왔다.'],['특징이 많아서',0,'릿지도 같은 특징 수를 닫힌 해로 푼다.']]},
 {q:'좌표 하강에서 j번째 가중치의 "남은 몫" ρⱼ의 크기가 α보다 작으면?',o:[['그 가중치는 정확히 0이 된다',1,'S(ρ, α) = sign(ρ)·max(|ρ| − α, 0). 남은 몫이 벌점보다 작으면 0이 이득이다.'],['그 가중치가 커진다',0,'벌점보다 작은 몫은 버린다.'],['학습이 멈춘다',0,'그 가중치만 0이 되고 다른 가중치는 계속 고친다.']]}],
q5:[{q:'4차 다항 회귀가 훈련 R² 0.98, 시험 R²는 크게 음수였다. 가장 알맞은 해석은?',o:[['훈련 배합을 외웠다(과대적합). 시험 R²가 음수면 늘 평균을 답하는 것보다 못하다',1,'특징 494개가 배합 772건에 비해 많아 큰 가중치들이 서로 상쇄하며 훈련 배합을 쫓아갔다.'],['훈련 점수가 높으니 가장 좋은 지원자다',0,'면접은 봉인된 시험지로 판정한다.'],['계산 오류이니 무시한다',0,'계산 방법마다 음수의 크기가 다르게 나오는 것 자체가 풀이가 불안정하다는 신호다. 결론(폭주)은 같다.']]},
 {q:'4차 릿지에서 시험지 점수가 가장 좋은 α는 10, 교차 검증이 고른 α는 100이었다. 보고할 점수는?',o:[['교차 검증이 고른 α = 100의 시험 점수(0.821)',1,'시험지를 보고 고른 0.862는 엿본 점수다. 시험지는 마지막에 한 번만 연다.'],['α = 10의 0.862',0,'시험지로 α를 고르면 시험지가 훈련 데이터가 된다.'],['둘의 평균',0,'평균을 내도 엿본 정보가 섞인다.']]},
 {q:'R²와 MAE에 대한 설명으로 맞는 것은?',o:[['R²는 1에 가까울수록, MAE는 0에 가까울수록 좋다. 늘 평균을 답하면 R²는 0이다',1,'R² = 1 − (오차 제곱합 ÷ 평균과의 차이 제곱합). MAE는 평균 몇 MPa 틀렸는지다.'],['R²는 0과 1 사이에만 있다',0,'시험 데이터에서는 음수도 나온다. 4차가 그 예다.'],['MAE가 작으면 R²도 늘 크다',0,'대개 같이 가지만, 크게 빗나간 몇 건은 R²(제곱)를 MAE보다 훨씬 크게 무너뜨린다.']]}],
q6:[{q:'라쏘가 164개 중 98개를 0으로 지웠다. 이것은?',o:[['규제의 결과로 특징이 골라진 것이다. 남은 항이 원인이라는 뜻은 아니다',1,'라쏘는 닮은 항 중 하나만 남기는 경향이 있고, α나 데이터가 바뀌면 남는 항도 바뀐다. 뜻 있는 새 특징을 만드는 특징공학과도 다르다.'],['물-결합재비 같은 새 특징을 만든 것이다',0,'라쏘는 있는 항을 지울 뿐 새 특징을 만들지 않는다.'],['지워진 항은 강도와 아무 관계가 없다는 증명이다',0,'다른 닮은 항이 그 몫을 대신 맡았을 수 있다.']]},
 {q:'Ridge(alpha=1000)과 LogisticRegression(C=1000)에 대한 설명으로 맞는 것은?',o:[['릿지는 규제가 아주 세고, 로지스틱 회귀는 규제가 아주 약하다',1,'α는 클수록 세고, C는 규제의 역수라 클수록 약하다. C ≈ 1/α로 기억한다.'],['둘 다 규제가 아주 세다',0,'C의 방향을 α처럼 읽은 흔한 실수다(5-4 03장).'],['둘 다 규제가 아주 약하다',0,'α = 1000은 이 데이터에서 시험 R²를 크게 떨어뜨릴 만큼 세다.']]}]};

/* ══ 공통 B·C·D·E·F 데이터 ══ */
var PROMPTS=[
 {t:'① 내 공정 데이터로 다항 회귀 + 규제 코드 받기',use:'내 공장의 품질 데이터로 03장의 프롬프트를 맞출 때',p:MAIN_PROMPT},
 {t:'② AI 코드 검증 요청',use:'받은 코드를 다른 AI(또는 같은 AI의 새 대화)에게 검토받을 때',p:'아래 scikit-learn 다항 회귀·릿지·라쏘 코드를 코드 리뷰해 줘. 특히 다음을 하나씩 확인하고, 문제가 있으면 줄 번호와 고친 코드를 보여 줘.\n1) 데이터를 나누기 전에 표준화·다항 특징을 fit 해서 시험 데이터 정보가 새지 않았나 (Pipeline을 썼나)\n2) train_test_split에 stratify와 random_state가 있나\n3) PolynomialFeatures의 차수와 결과 특징 수(n_output_features_)가 훈련 샘플 수에 비해 너무 많지 않나\n4) 다항 특징 뒤에 StandardScaler가 있나\n5) 내 scikit-learn 버전(〔1.x〕)에 없는 인자(normalize, get_feature_names 등)를 쓰지 않았나\n6) alpha와 차수를 시험 데이터가 아니라 교차 검증으로 골랐나\n7) 훈련 R²와 시험 R², 기준선(DummyRegressor)을 함께 보고했나\n추측으로 답하지 말고, 확실하지 않은 부분은 "공식 문서 확인 필요"라고 표시해 줘.'},
 {t:'③ 차수와 α 함께 고르기',use:'손잡이 두 개(차수, α)를 정직하게 고르고 싶을 때',p:'콘크리트 강도 회귀에서 PolynomialFeatures의 차수(〔1~4〕)와 Ridge의 alpha(〔0.01~1000, 로그 간격 6개〕)를 함께 고르고 싶다.\n- Pipeline(StandardScaler → PolynomialFeatures(include_bias=False) → StandardScaler → Ridge)으로 묶고\n- GridSearchCV(cv=5, scoring="r2")로 훈련 데이터 안에서만 고르고\n- 결과를 차수 × alpha 표(교차 검증 평균 R²)로 보여 주고\n- 고른 조합으로 시험 데이터를 한 번만 채점하는 코드를 써 줘.\n교차 검증 점수와 시험 점수가 다를 때 어떻게 해석해야 하는지, 표준편차(cv_results_의 std_test_score)를 왜 함께 봐야 하는지도 설명해 줘.'},
 {t:'④ 수업용 문제 만들기',use:'이 편으로 수업할 때 확인 문제가 더 필요할 때',p:'다항 회귀와 규제(Ridge·Lasso·Elastic Net) 입문 수업용 문제를 만들어 줘. 대상은 〔제조 현장 실무자, 수식에 익숙하지 않음〕이다. 예시는 콘크리트 강도 예측으로 통일한다.\n- 다항 특징·과대적합·릿지·라쏘·α를 각각 묻는 객관식 5문제 (보기 3개, 정답 해설 포함)\n- 오해를 드러내는 문제 2개: "차수가 높을수록 좋다", "라쏘가 남긴 특징이 원인이다"\n- 손으로 풀 수 있는 계산 문제 1개: 특징 3개로 2차 다항 특징(상수 제외)을 만들면 몇 개인가, 소프트 임계값 S(0.3, 0.5)와 S(−0.8, 0.5)는 얼마인가\n정답은 맨 끝에 따로 모아 줘.'}];
var GL=[
 ['모델','다항 회귀','특징을 거듭제곱하고 서로 곱해 늘린 뒤 선형 회귀를 하는 방법. 가중치에 대해서는 선형이다.'],
 ['모델','다항 특징','x, x², x³, x₁x₂처럼 원래 특징을 거듭제곱하거나 곱해 만든 특징. scikit-learn PolynomialFeatures가 만든다.'],
 ['모델','상호작용 항','서로 다른 특징을 곱한 항(시멘트·재령 등). "한 특징의 효과가 다른 특징에 따라 달라진다"를 맡는다.'],
 ['모델','차수','다항 특징을 만들 때 곱하는 최대 횟수. 3차면 x³, x₁x₂x₃까지 만든다.'],
 ['모델','릿지','가중치 제곱합(L2)에 벌점을 주는 선형 회귀. 가중치를 고루 줄인다. 닫힌 해가 있다.'],
 ['모델','라쏘','가중치 절댓값 합(L1)에 벌점을 주는 선형 회귀. 쓸모가 적은 가중치를 0으로 지운다.'],
 ['모델','엘라스틱넷','L1과 L2 벌점을 l1_ratio(ρ)로 섞은 선형 회귀. 지우기도 하고 닮은 특징을 함께 남기기도 한다.'],
 ['학습','규제','가중치가 너무 커지지 않게 손실에 벌점을 더하는 일. 세기는 α(교과서의 λ)로 정한다.'],
 ['학습','최소제곱','잔차 제곱합을 가장 작게 하는 가중치를 찾는 방법. 선형 회귀의 기본 학습법.'],
 ['학습','닫힌 해','반복 없이 식 하나로 바로 구하는 답. 릿지는 w = (XᵀX + αI)⁻¹Xᵀy.'],
 ['학습','좌표 하강','가중치를 하나씩 돌아가며 고치는 학습 방법. 라쏘·엘라스틱넷을 풀 때 쓴다.'],
 ['학습','소프트 임계값','S(ρ, α) = sign(ρ)·max(|ρ| − α, 0). 크기가 α보다 작으면 0으로 만든다. 라쏘가 0을 만드는 원리.'],
 ['학습','과대적합','훈련 데이터를 외워 처음 보는 데이터에서 성적이 떨어지는 일.'],
 ['학습','과소적합','모델이 너무 단순하거나 규제가 너무 세서 훈련 데이터조차 못 맞히는 일.'],
 ['학습','조건수','풀이가 입력의 작은 변화에 얼마나 흔들리는지 나타내는 수. 클수록 불안정하다. 4차 다항 특징에서 매우 크다.'],
 ['학습','교차 검증','훈련 데이터를 몇 칸으로 나눠 한 칸씩 돌아가며 검증용으로 쓰는 방법. α·차수를 시험지 없이 고른다.'],
 ['학습','특징 선택','쓸 특징을 고르는 일. 라쏘는 규제의 결과로 특징 선택 효과를 낸다.'],
 ['학습','특징공학','사람이 지식으로 새 특징(물-결합재비, log 재령 등)을 만드는 일. 규제와는 다른 일이다.'],
 ['데이터','물-결합재비','물의 양 ÷ 결합재(시멘트 + 고로 슬래그 + 플라이 애시)의 양. 낮을수록 강도가 높아지는 경향이 있다.'],
 ['데이터','재령','콘크리트를 친 뒤 시험할 때까지 지난 날수. 강도는 처음에 빨리, 나중에 천천히 오른다.'],
 ['데이터','표준화','특징마다 평균을 빼고 표준편차로 나눠 단위를 맞추는 일. 훈련 데이터로만 맞춘다.'],
 ['데이터','데이터 누수','시험 데이터의 정보가 학습에 섞여 들어가는 일. 점수가 실제보다 좋게 나온다.'],
 ['평가','결정 계수','R². 1 − (오차 제곱합 ÷ 평균과의 차이 제곱합). 1이면 완벽, 0이면 늘 평균과 같고, 음수면 그보다 못하다.'],
 ['평가','평균 절대 오차','MAE. 예측이 평균 몇 MPa 틀렸는지. 단위가 타겟과 같아 읽기 쉽다.'],
 ['평가','잔차','정답 − 예측. 최소제곱은 잔차 제곱합을 줄인다.'],
 ['평가','기준선','배우지 않고도 낼 수 있는 점수. 회귀에서는 "늘 훈련 평균"이 기준선이다(R² 0).']];
var ANALOGY=[
 ['다항 특징','지원자에게 질문지 항목을 늘려 주는 것. 항목이 많을수록 세밀하게 답하지만, 외워서 답하기도 쉬워진다',2,'초파리 버섯체에서 냄새 신호 약 50갈래가 약 2,000개의 케니언 세포로 퍼지며 여러 입력의 조합을 만드는 것(차원 확장 뒤 선형 판독)','케니언 세포는 입력을 무작위로 섞고 문턱을 넘어야 반응한다. x²·x₁x₂처럼 정해진 곱을 빠짐없이 만드는 다항 특징과는 다르다'],
 ['과대적합','기출문제 답만 외운 지원자. 같은 문제는 만점, 새 문제는 엉뚱한 답',1,'','뇌에서 "외우기 대 일반화"를 다루는 연구는 많지만, 다항 곡선의 출렁임과 같은 계산이 일어난다는 근거는 없다'],
 ['규제 α (릿지)','"한 항목만 보고 뽑지 말라"는 채용 규정. 규정이 세면 모든 항목을 고루 조금씩 본다',2,'시냅스 항상성 조절(synaptic scaling): 뉴런이 활동을 일정하게 지키려고 시냅스 세기를 한꺼번에 비례해서 줄이거나 늘리는 것','항상성 조절은 비율로 곱해 줄이는 것에 가깝고, 오차와 벌점을 함께 줄이는 최적화라는 근거는 없다'],
 ['라쏘의 0','쓸모가 적은 서류 항목을 아예 지워 버리는 심사. 남은 항목이 적어 읽기 쉽다',2,'발달 중 쓰이지 않는 시냅스를 없애는 가지치기, 초파리 버섯체의 희소한 반응(케니언 세포 중 일부만 반응)','가지치기는 연결(가중치)을 없애지만 라쏘의 절댓값 벌점과 같은 규칙인지는 알려지지 않았다. 버섯체의 희소함은 가중치가 아니라 활동이 희소한 것이다'],
 ['교차 검증','본시험 전에 모의 면접을 다섯 번 돌려 근무 조건을 정하는 것',1,'','뇌가 학습 조건을 이런 식으로 고른다는 대응은 알려지지 않았다'],
 ['α와 C','같은 고삐인데 한쪽은 "당기는 힘", 다른 쪽은 "풀어 주는 길이"로 잰다',1,'','순전히 표기 약속의 차이다. 생물학적 대응은 없다']];
var REFS=[
 'Yeh, I.-C. (1998). Modeling of strength of high-performance concrete using artificial neural networks. <i>Cement and Concrete Research</i>, 28(12), 1797–1808. <a href="https://doi.org/10.1016/S0008-8846(98)00165-3" target="_blank" rel="noopener">doi:10.1016/S0008-8846(98)00165-3</a> — 실기 과제지 원출처',
 'Yeh, I.-C. (2007). Concrete Compressive Strength [Dataset]. UCI Machine Learning Repository. <a href="https://doi.org/10.24432/C5PK67" target="_blank" rel="noopener">doi:10.24432/C5PK67</a> (CC BY 4.0) — 02장 데이터',
 'Abrams, D. A. (1918). <i>Design of Concrete Mixtures</i> (Bulletin 1). Structural Materials Research Laboratory, Lewis Institute, Chicago. — 물-시멘트비 법칙, 매뉴얼 씨의 경험식(02장)',
 'ACI Committee 209 (1992). <i>Prediction of Creep, Shrinkage, and Temperature Effects in Concrete Structures</i> (ACI 209R-92). American Concrete Institute. — 재령 보정식 t/(4 + 0.85t), 02장',
 'Hoerl, A. E., &amp; Kennard, R. W. (1970). Ridge regression: Biased estimation for nonorthogonal problems. <i>Technometrics</i>, 12(1), 55–67. <a href="https://doi.org/10.1080/00401706.1970.10488634" target="_blank" rel="noopener">doi:10.1080/00401706.1970.10488634</a> — 릿지, 04장',
 'Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. <i>Journal of the Royal Statistical Society: Series B</i>, 58(1), 267–288. <a href="https://doi.org/10.1111/j.2517-6161.1996.tb02080.x" target="_blank" rel="noopener">doi:10.1111/j.2517-6161.1996.tb02080.x</a> — 라쏘, 04장',
 'Zou, H., &amp; Hastie, T. (2005). Regularization and variable selection via the elastic net. <i>Journal of the Royal Statistical Society: Series B</i>, 67(2), 301–320. <a href="https://doi.org/10.1111/j.1467-9868.2005.00503.x" target="_blank" rel="noopener">doi:10.1111/j.1467-9868.2005.00503.x</a> — 엘라스틱넷, 05장',
 'Friedman, J., Hastie, T., &amp; Tibshirani, R. (2010). Regularization paths for generalized linear models via coordinate descent. <i>Journal of Statistical Software</i>, 33(1), 1–22. <a href="https://doi.org/10.18637/jss.v033.i01" target="_blank" rel="noopener">doi:10.18637/jss.v033.i01</a> — 좌표 하강과 규제 경로, 04장',
 'Hastie, T., Tibshirani, R., &amp; Friedman, J. (2009). <i>The Elements of Statistical Learning</i> (2nd ed.), 3장·7장. Springer. — 축소 방법, 교차 검증, 05·06장',
 'Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. <i>Journal of Machine Learning Research</i>, 12, 2825–2830. — A장 코드',
 'scikit-learn 문서. <a href="https://scikit-learn.org/stable/modules/linear_model.html" target="_blank" rel="noopener">Linear Models</a> (Ridge, Lasso, ElasticNet의 목적 함수), <a href="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html" target="_blank" rel="noopener">PolynomialFeatures</a> — 03·06장',
 'Caron, S. J. C., Ruta, V., Abbott, L. F., &amp; Axel, R. (2013). Random convergence of olfactory inputs in the <i>Drosophila</i> mushroom body. <i>Nature</i>, 497, 113–117. <a href="https://doi.org/10.1038/nature12063" target="_blank" rel="noopener">doi:10.1038/nature12063</a> — D장 다항 특징 비유',
 'Turrigiano, G. G., Leslie, K. R., Desai, N. S., Rutherford, L. C., &amp; Nelson, S. B. (1998). Activity-dependent scaling of quantal amplitude in neocortical neurons. <i>Nature</i>, 391, 892–896. <a href="https://doi.org/10.1038/36103" target="_blank" rel="noopener">doi:10.1038/36103</a> — D장 릿지 비유',
 'Lin, A. C., Bygrave, A. M., de Calignon, A., Lee, T., &amp; Miesenböck, G. (2014). Sparse, decorrelated odor coding in the mushroom body enhances learned odor discrimination. <i>Nature Neuroscience</i>, 17(4), 559–568. <a href="https://doi.org/10.1038/nn.3660" target="_blank" rel="noopener">doi:10.1038/nn.3660</a> — D장 라쏘 비유(희소한 활동)'];
var TIP={who:'머신러닝을 처음 배우는 대학생·직장인·제조 현장 실무자. 5-2 선형 회귀(최소제곱, R²)를 먼저 보면 좋다. 수식은 릿지의 닫힌 해와 소프트 임계값 두 개만 나온다',
 goal:'① 다항 특징이 "모델은 그대로, 재료만 늘리기"임을 설명한다 ② 차수를 올릴 때 훈련·시험 R²가 갈라지는 과대적합을 그림으로 보인다 ③ 릿지·라쏘·엘라스틱넷의 차이(줄이기·지우기·섞기)를 말한다 ④ α를 시험지가 아니라 교차 검증으로 고르고, AI가 낸 코드에서 누수·특징 폭발·없어진 인자를 찾는다',
 time:'읽기 25분 · 실습(01·04·05장 그림) 25분 · 틀린 곳 찾기 10분 · Colab 15분',
 plans:[['강의형 50분','00장 지원서 1장을 인쇄해 9칸 함께 읽기 10분 → 01장에서 학생들이 차수 10·α 0을 직접 만들어 보기 10분 → 04장 α 경로 재생(릿지 → 라쏘) 10분 → 05장 4차 폭주와 릿지로 잡기 10분 → 면접 결과를 학생들이 먼저 맞히고 발표 10분.'],['실무 OJT 90분','03장 프롬프트로 각자 AI에게 코드를 받는다 20분 → 체크리스트로 서로의 코드를 바꿔 검사 20분 → Colab에서 실행하고 A장 결과와 비교(특히 4차 점수가 왜 다른가) 20분 → B장 ③ 프롬프트로 차수 × α 표를 만들어 "우리 공정에 쓸 근무 조건" 발표 30분.'],['1:1 튜터링','지원서 1장의 아홉 칸 중 막히는 칸을 고르게 하고 그 칸의 "→ 장"으로 간다. 04장에서 "좌표 하강 한 바퀴"를 몇 번 눌러 0인 가중치 수가 어떻게 바뀌는지 기록하게 한다.']],
 myths:[['차수가 높을수록 좋다','이 데이터에서는 3차가 가장 좋았고 4차는 규제 없이 폭주했다. 차수도 교차 검증으로 고른다'],['다항 회귀는 비선형 모델이다','가중치에 대해서는 선형이다. 바뀐 것은 특징이다'],['라쏘가 남긴 특징이 원인이다','규제의 결과일 뿐이다. 닮은 특징 중 하나만 남기는 경향이 있고, α가 바뀌면 남는 것도 바뀐다'],['규제는 늘 성적을 올린다','과대적합일 때 시험 성적을 지킨다. 너무 세면 과소적합이 된다(α = 1000)'],['α는 시험 점수가 가장 좋은 값으로 고른다','그러면 시험지가 훈련 데이터가 된다. 4차에서 시험지는 α = 10, 교차 검증은 α = 100을 골랐다'],['α와 C는 같은 방향이다','반대다. α는 클수록, C는 작을수록 규제가 세다']],
 tasks:[['과제 1 · 지원서 다시 쓰기','00장 지원서 3번 칸의 막대를 내 데이터의 특징 수로 다시 그리고, 2·3·4차 특징 수를 직접 계산해 적는다.'],['과제 2 · 틀린 곳 찾기 만들기','AI에게 일부러 실수가 섞인 다항 회귀 코드를 받고(B장 ②를 거꾸로 활용), 짝에게 찾게 한다. 실수마다 "왜 틀렸나"를 한 줄로 쓴다.'],['과제 3 · 차수 × α 표','B장 ③ 프롬프트로 차수 1~4 × α 6개의 교차 검증 표를 만들고, 고른 조합의 시험 점수와 매뉴얼 씨(R² 0.735)를 비교하는 반 쪽 보고서를 쓴다.'],['과제 4 · 특징공학 대 규제','A장 코드에 물-결합재비와 log(재령)을 더한 선형 회귀(외울 숫자 11개)와 3차 라쏘(외울 숫자 67개)를 나란히 돌리고, 점수·숫자 개수·설명하기 쉬움을 표로 비교한다.']],
 ai:'학생이 AI에게 코드를 받는 것은 막지 말고 과정에 넣는다. 다만 ① 프롬프트를 제출하게 하고 ② 체크리스트 아홉 항목을 직접 표시하게 하고 ③ 훈련 R²·시험 R²·합격선을 나란히 적은 문장을 쓰게 하면, "받아 적기"가 아니라 "검증하기"를 배운다. 4차 점수가 브라우저와 Colab에서 다르게 나오는 것을 토론거리로 쓰면 "불안정한 풀이"를 실감한다. AI 답의 함수·인자는 공식 문서와 버전으로 확인하게 하고, 과제에 AI 도움을 받았으면 사용 범위를 밝히게 한다.'};
var CODE={title:'다항 회귀와 규제 면접 — 곡선 긋기부터 고삐 매기까지',sub:'numpy, pandas, scikit-learn, ucimlrepo · CPU 몇 초',file:'polynomial_regularization_interview',
 desc:'이 편의 면접을 그대로 옮겼다. UCI에서 콘크리트 데이터를 받아 시험지를 봉인하고(1·2번 칸), 매뉴얼 씨의 경험식으로 합격선을 재고(3번), 다항 특징 수를 세고(4번), 차수 1~4의 다항 회귀로 과대적합을 보고(5번), 릿지·라쏘·엘라스틱넷을 붙이고(6번), α를 교차 검증으로 고른 뒤(7번), 합격 판정(8번)까지 간다. 노트북으로 저장하면 칸마다 셀이 나뉜다.',
 cap:'실행 결과 (scikit-learn 1.9, CPU). 브라우저 계산과 같은 결론(합격선 R² 0.735, 3차 0.849 · MAE 4.72MPa, 4차 + 릿지 α = 100이 0.821, 라쏘 α = 0.1이 164개 중 98개를 0으로)이 나온다. 4차(규제 없음)의 시험 R²만 계산 방법에 따라 다르다(이 출력 −6.18, 다항 뒤 한 번 더 표준화 −11.3, 브라우저 약 −23.6). 풀이가 불안정하다는 증거다.'};
