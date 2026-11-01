/* ══ 5-5 나이브 베이즈 · 콘크리트 실기 과제지 ══════════════════════════ */
const ROWS=concRows(),TR=ROWS.filter(r=>!r.test),TE=ROWS.filter(r=>r.test);
const fv=r=>FEAT.map(f=>r[f]);
const A=TR.map(fv),B=TE.map(fv);
const YTR=TR.map(r=>r.pass),YTE=TE.map(r=>r.pass),GTR=TR.map(r=>r.grade),GTE=TE.map(r=>r.grade);
const PNAME=['불합격','합격'];
const LOG2PI=Math.log(2*Math.PI);
/* 가우시안 나이브 베이즈: scikit-learn GaussianNB와 같은 계산
   클래스별 평균·분산(ddof=0) + var_smoothing(1e-9 × 가장 큰 특징 분산), 사전 확률 = 클래스 비율(또는 priors) */
function gnbFit(X,y,K,priors){const n=X.length,d=X[0].length;let vmax=0;
  for(let j=0;j<d;j++){let m=0;for(let i=0;i<n;i++)m+=X[i][j];m/=n;let v=0;for(let i=0;i<n;i++)v+=(X[i][j]-m)*(X[i][j]-m);v/=n;if(v>vmax)vmax=v;}
  const eps=1e-9*vmax,cnt=new Array(K).fill(0),mu=[],va=[];for(let k=0;k<K;k++){mu.push(new Array(d).fill(0));va.push(new Array(d).fill(0));}
  for(let i=0;i<n;i++){cnt[y[i]]++;for(let j=0;j<d;j++)mu[y[i]][j]+=X[i][j];}
  for(let k=0;k<K;k++)for(let j=0;j<d;j++)mu[k][j]/=cnt[k];
  for(let i=0;i<n;i++)for(let j=0;j<d;j++){const e=X[i][j]-mu[y[i]][j];va[y[i]][j]+=e*e;}
  for(let k=0;k<K;k++)for(let j=0;j<d;j++)va[k][j]=va[k][j]/cnt[k]+eps;
  const pri=priors||cnt.map(c=>c/n),lp=pri.map(Math.log);
  const ll=(x,k,j)=>-0.5*(LOG2PI+Math.log(va[k][j]))-(x-mu[k][j])*(x-mu[k][j])/(2*va[k][j]);
  const lik=x=>mu.map((_,k)=>{let s=0;for(let j=0;j<d;j++)s+=ll(x[j],k,j);return s;});
  const joint=x=>lik(x).map((s,k)=>s+lp[k]);
  const proba=x=>ML.softmax(joint(x));
  const predict=x=>{const z=joint(x);let m=0;for(let k=1;k<K;k++)if(z[k]>z[m])m=k;return m;};
  return {mu,va,pri,cnt,eps,ll,lik,joint,proba,predict,K,d};}
const pdf=(x,m,v)=>Math.exp(-(x-m)*(x-m)/(2*v))/Math.sqrt(2*Math.PI*v);
const ncdf=z=>{const t=1/(1+0.2316419*Math.abs(z)),d=0.3989423*Math.exp(-z*z/2),p=d*t*(0.3193815+t*(-0.3565638+t*(1.781478+t*(-1.821256+t*1.330274))));return z>0?1-p:p;};
let MB=null,MBms=0,M3=null;
function mainNB(){if(!MB){const t0=performance.now();MB=gnbFit(A,YTR,2);MBms=performance.now()-t0;}return MB;}
function main3(){return M3||(M3=gnbFit(A,GTR,3));}
const accOf=(m,X,Y)=>ML.acc(Y,X.map(m.predict));
function prf(yt,yp,K){const M=ML.confusion(yt,yp,K);return [...Array(K)].map((_,k)=>{let tp=M[k][k],col=0,row=0;for(let a=0;a<K;a++){col+=M[a][k];row+=M[k][a];}
  const P=col?tp/col:0,R=row?tp/row:0;return {k,P,R,F:(P+R)?2*P*R/(P+R):0,n:row};});}
function baseAcc(){const p=YTR.reduce((a,v)=>a+v,0)/YTR.length,maj=p>=.5?1:0;return {k:maj,acc:YTE.filter(v=>v===maj).length/YTE.length};}
function cmGrid(M,labels,colsFor){const K=labels.length;let h='<div class="cm" style="grid-template-columns:86px repeat('+K+',minmax(48px,1fr));max-width:'+(110+K*90)+'px"><div class="h">정답 ＼ 예측</div>'+labels.map(k=>'<div class="h">'+k+'</div>').join('');
  M.forEach((row,a)=>{h+='<div class="h" style="justify-content:end;text-align:right">'+labels[a]+'</div>'+row.map((v,b)=>a===b?'<div class="c'+(v?' d':'')+'" style="'+(v?'background:'+rgba(colsFor(a),.85):'')+'">'+v+'</div>':'<div class="c'+(v?' x':'')+'">'+(v||'·')+'</div>').join('');});return h+'</div>';}
/* 특징 사전: 8개 + 파생 2개 */
const XF={},XL=Object.assign({},FEATK,{wb:'물-결합재비',lage:'log 재령'});
FEAT.forEach(f=>{XF[f]=r=>r[f];});XF.wb=r=>r.wb;XF.lage=r=>Math.log10(r.age);
const SHORT={cement:'시멘트',slag:'슬래그',fly_ash:'애시',water:'물',superplasticizer:'감수제',coarse_agg:'굵은골재',fine_agg:'잔골재',age:'재령'};

/* ── 히어로 ── */
heroRoom($('heroCv'),{sym:'P',no:'APP-05',name:'나이브\n베이즈',role:'지원자 · 분류팀',banner:'모델 채용 위원회 · 5-5 면접',doc:'front',asst:'board',fly:'clip',robo:'calc',
  beats:[{at:0.5,who:'fly',mood:'point',move:'shake',say:'특징끼리 독립이라고?'},
         {at:2.3,who:'asst',mood:'shock',move:'pop',say:'물이랑 감수제는 얽혀요!'},
         {at:4.2,who:'doc',mood:'magic',move:'stand',say:'알고도 곱하는 거예요'},
         {at:6.0,who:'robo',mood:'count',move:'pop',say:'평균·분산만 세면 끝?'},
         {at:7.8,who:'fly',mood:'type',move:'lean',say:'일단 적어 둘게'}],loop:10,
  motif(ctx,t,W,H,C){[[C.muted,318,16],[C.c_app,352,20]].forEach((c,k)=>{const mu=c[1]+Math.sin(t*0.9+k*2)*5,sd=c[2];ctx.strokeStyle=c[0];ctx.lineWidth=3;ctx.beginPath();
      for(let i=0;i<=60;i++){const x=272+i*2.2,y=152-58*Math.exp(-(x-mu)*(x-mu)/(2*sd*sd));i?ctx.lineTo(x,y):ctx.moveTo(x,y);}ctx.stroke();});
    ctx.fillStyle=C.muted;ctx.font=font(12,500,1);ctx.textAlign='left';ctx.fillText('P(x | y) × P(y)',290,170);}});
$('heroLine').textContent=pct(PASSLINE);
comic('comicIntro',[
 {bg:'lab',prop:'독립?',cast:[{who:'fly',mood:'point',x:28,move:'shake'},{who:'doc',mood:'front',x:72,h:104}],say:{who:'fly',txt:'시멘트를 줄이면 슬래그를 늘리는 게 배합인데, 둘이 서로 모른다고?'},cap:'① 의심 — 배합의 특징은 서로 얽혀 있다'},
 {bg:'lab',prop:'P × P × P',cast:[{who:'doc',mood:'magic',x:28,move:'pop',h:108},{who:'asst',mood:'board',x:70}],say:{who:'doc',txt:'알면서 모르는 척하는 거예요. 그래서 "나이브". 대신 확률을 곱하기만 하면 돼요.'},cap:'② 해명 — 순진한 가정과 맞바꾼 쉬운 계산'},
 {bg:'lab',prop:'34',cast:[{who:'robo',mood:'count',x:30,move:'bob'},{who:'asst',mood:'ledger',x:72}],say:{who:'robo',txt:'평균 16개, 분산 16개, 비율 2개. 34개만 세면 수습 끝? 싸다!'},cap:'③ 예산 — 학습은 한 번 훑기'}]);

/* ── 01 특징 하나의 두 종 모양 ── */
(function(){const cv=$('bellCv'),H=300,KEYS=FEAT.concat(['wb','lage']),st={f:'cement',u:0.5};
  $('featChoice').innerHTML=KEYS.map(k=>'<button data-v="'+k+'" aria-pressed="'+(k===st.f)+'">'+XL[k]+'</button>').join('');
  function stats(f){const v=[[],[]];TR.forEach(r=>v[r.pass].push(XF[f](r)));const m=v.map(a=>ML.mean(a)),va=v.map((a,k)=>ML.mean(a.map(x=>(x-m[k])*(x-m[k]))));
    const all=v[0].concat(v[1]),mn=Math.min(...all),mx=Math.max(...all),sd=va.map(Math.sqrt);
    let lo=Math.min(mn,m[0]-2.6*sd[0],m[1]-2.6*sd[1]),hi=Math.max(mx,m[0]+2.6*sd[0],m[1]+2.6*sd[1]);return {v,m,va,sd,lo,hi,mn,mx};}
  let S=stats(st.f);
  const xOf=()=>S.lo+(S.hi-S.lo)*st.u;
  function draw(){const W=fit(cv,H),ctx=cv.getContext('2d'),C=cols(),pl=46,pr=14,pt=16,pb=40,gw=W-pl-pr,gh=H-pt-pb,NB=28,bw=(S.hi-S.lo)/NB;
    const sx=x=>pl+(x-S.lo)/(S.hi-S.lo)*gw;const hist=[0,1].map(k=>{const h=new Array(NB).fill(0);S.v[k].forEach(x=>{h[Math.min(NB-1,Math.floor((x-S.lo)/bw))]++;});return h.map(c=>c/(S.v[k].length*bw));});
    let ym=0;hist.forEach(h=>h.forEach(v=>{if(v>ym)ym=v;}));[0,1].forEach(k=>{ym=Math.max(ym,pdf(S.m[k],S.m[k],S.va[k]));});ym*=1.08;const sy=v=>pt+gh-(v/ym)*gh;
    ctx.clearRect(0,0,W,H);ctx.strokeStyle=C.rule;ctx.lineWidth=1;ctx.strokeRect(pl,pt,gw,gh);
    if(S.lo<0&&st.f!=='lage'){ctx.fillStyle=rgba(C.stamp,.08);ctx.fillRect(pl,pt,sx(0)-pl,gh);ctx.fillStyle=C.stamp;ctx.font=font(12,700);ctx.textAlign='left';ctx.fillText('음수 = 있을 수 없는 값',pl+6,pt+16);}
    const colk=[C.muted,C.blue];
    [0,1].forEach(k=>{ctx.fillStyle=rgba(colk[k],k?.28:.32);hist[k].forEach((v,b)=>{if(!v)return;const x=sx(S.lo+b*bw),w=Math.max(1,sx(S.lo+(b+1)*bw)-x-1),y=sy(v);ctx.fillRect(x+(k?w/2:0),y,w/2,pt+gh-y);});});
    [0,1].forEach(k=>{ctx.strokeStyle=k?C.blue:C.text;ctx.lineWidth=3;ctx.beginPath();for(let i=0;i<=200;i++){const x=S.lo+(S.hi-S.lo)*i/200,y=sy(pdf(x,S.m[k],S.va[k]));i?ctx.lineTo(sx(x),y):ctx.moveTo(sx(x),y);}ctx.stroke();});
    ctx.fillStyle=C.muted;ctx.font=font(12);ctx.textAlign='center';for(let k=0;k<=5;k++){const x=S.lo+(S.hi-S.lo)*k/5;ctx.fillText(Math.abs(x)<10?x.toFixed(2):x.toFixed(0),sx(x),H-22);}
    ctx.fillText(XL[st.f]+(st.f==='lage'?' (log₁₀ 일)':''),pl+gw/2,H-4);ctx.save();ctx.translate(12,pt+gh/2);ctx.rotate(-Math.PI/2);ctx.fillText('밀도 (가능도의 높이)',0,0);ctx.restore();
    const x=xOf();ctx.strokeStyle=C.stamp;ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(sx(x),pt);ctx.lineTo(sx(x),pt+gh);ctx.stroke();
    [0,1].forEach(k=>circ(ctx,sx(x),sy(pdf(x,S.m[k],S.va[k])),6,k?C.blue:C.text,C.panel,2));
    ctx.font=font(12,700);ctx.textAlign='right';ctx.fillStyle=C.text;ctx.fillText('— 불합격 '+S.v[0].length+'건',pl+gw-8,pt+16);ctx.fillStyle=C.blue;ctx.fillText('— 합격 '+S.v[1].length+'건',pl+gw-8,pt+32);
    info();}
  function info(){const x=xOf(),l0=pdf(x,S.m[0],S.va[0]),l1=pdf(x,S.m[1],S.va[1]),p0=1-mainNB().pri[1],p1=mainNB().pri[1],post=p1*l1/(p1*l1+p0*l0),ev=Math.log(l1/l0);
    $('bellXO').textContent=Math.abs(x)<10?x.toFixed(2):x.toFixed(0);
    const fmt=v=>v<1e-3?v.toExponential(2):v.toPrecision(3);
    let h='값 <b>'+(Math.abs(x)<10?x.toFixed(2):x.toFixed(0))+'</b>에서 높이: 불합격 종 <b>'+fmt(l0)+'</b> · 합격 종 <b>'+fmt(l1)+'</b> → 가능도 비 '+(l1/l0).toFixed(2)+' (증거 점수 log = <b>'+fx(ev)+'</b>, '+(ev>0?'합격 쪽':'불합격 쪽')+'). '+
      '사전 확률 '+p1.toFixed(3)+'을 곱해 비교하면 이 특징 하나만으로 본 합격 확률은 <b>'+post.toFixed(3)+'</b>이다.';
    const neg=[0,1].map(k=>ncdf((0-S.m[k])/S.sd[k]));
    if(st.f==='age')h+='<br><span style="color:var(--stamp)">재령은 3·7·28·56·90일처럼 몇 군데에 몰려 있다. 그런데 종 모양은 그 사이를 매끈하게 메우고, 0일보다 왼쪽(불합격 '+pct(neg[0])+', 합격 '+pct(neg[1])+')까지 퍼진다. "log 재령"을 눌러 비교해 보자.</span>';
    else if(st.f==='lage')h+='<br>로그를 씌우니 두 종이 막대와 훨씬 잘 맞고, 서로 더 멀리 떨어졌다. 같은 정보라도 모양을 맞춰 주면 증거가 또렷해진다.';
    else if(st.f==='slag'||st.f==='fly_ash'||st.f==='superplasticizer'){const z=TR.filter(r=>r[st.f]===0).length/TR.length;h+='<br><span style="color:var(--stamp)">배합의 '+pct(z,0)+'가 0이다. 0에 솟은 막대 하나와 나머지가 퍼진 모양은 종 하나로 그릴 수 없다.</span>';}
    else if(st.f==='wb')h+='<br>물-결합재비는 현장 기술자의 숫자다. 두 종이 꽤 떨어져 있어 특징 하나로도 증거가 강하다.';
    $('bellInfo').innerHTML=h;}
  seg('#featChoice button',v=>{st.f=v;S=stats(v);st.u=clamp((((S.m[0]+S.m[1])/2)-S.lo)/(S.hi-S.lo),0,1);$('bellX').value=st.u;draw();});
  $('bellX').addEventListener('input',e=>{st.u=+e.target.value;draw();});
  cv.addEventListener('click',e=>{const r=cv.getBoundingClientRect();st.u=clamp((e.clientX-r.left-46)/(r.width-60),0,1);$('bellX').value=st.u;draw();});
  st.u=clamp((380-S.lo)/(S.hi-S.lo),0,1);$('bellX').value=st.u;draw();onRedraw(draw);})();

/* ── 01 증거 장부 ── */
(function(){const sel=$('evPick'),m=mainNB(),pick=[];for(let i=0;i<TR.length;i+=19)pick.push(i);
  pick.forEach(i=>{const r=TR[i];sel.add(new Option('훈련 #'+(r.i+1)+' · 실제 '+r.strength.toFixed(1)+'MPa('+PNAME[r.pass]+') · 재령 '+r.age+'일',i));});
  function show(){const i=+sel.value,r=TR[i],x=A[i],ev=FEAT.map((f,j)=>m.ll(x[j],1,j)-m.ll(x[j],0,j)),pr=Math.log(m.pri[1]/m.pri[0]),tot=pr+ev.reduce((a,b)=>a+b,0),rows=[['사전 확률',pr]].concat(FEAT.map((f,j)=>[FEATK[f],ev[j]])).concat([['합계',tot]]);
    const mx=Math.max(1,...rows.map(q=>Math.abs(q[1])));
    $('evBars').innerHTML=rows.map((q,k)=>{const v=q[1],Wd=Math.min(50,Math.abs(v)/mx*50),last=k===rows.length-1;return '<div class="row"'+(last?' style="border-top:1px solid var(--rule);padding-top:6px"':'')+'><span style="text-align:left;font-family:var(--font)'+(last?';font-weight:700':'')+'">'+q[0]+'</span><div class="tr"><span class="z"></span><i class="'+(v<0?'neg':'')+'" style="left:'+(v<0?50-Wd:50)+'%;width:'+Wd+'%'+(last?'':';opacity:.8')+'"></i></div><span>'+fx(v)+'</span></div>';}).join('');
    const p=ML.sigmoid(tot),big=FEAT.map((f,j)=>[Math.abs(ev[j]),j,ev[j]]).sort((a,b)=>b[0]-a[0]).slice(0,2);
    $('evInfo').innerHTML='막대 하나가 log[P(x|합격) / P(x|불합격)]이다. 파랑(오른쪽)은 합격 쪽, 빨강(왼쪽)은 불합격 쪽 증거다. 곱셈이 로그로 덧셈이 되어 막대를 그냥 더하면 된다. 합계 <b>'+fx(tot)+'</b> → P(합격) = 1/(1+e<sup>−합계</sup>) = <b>'+p.toFixed(3)+'</b> → "'+PNAME[p>=.5?1:0]+'" 판정 (실제 '+r.strength.toFixed(1)+'MPa, '+PNAME[r.pass]+(r.pass===(p>=.5?1:0)?', 맞힘':', <b style="color:var(--stamp)">틀림</b>')+'). '+
      '가장 큰 증거 두 개: '+FEATK[FEAT[big[0][1]]]+' '+fx(big[0][2])+', '+FEATK[FEAT[big[1][1]]]+' '+fx(big[1][2])+'. <span class="muted">5-4 로지스틱 회귀도 log 오즈를 "특징별 몫의 합"으로 냈다. 몫을 정하는 방법이 다르다: 로지스틱은 정답을 보며 고치고, 나이브 베이즈는 분포의 높이에서 읽는다.</span>';}
  sel.onchange=show;const rng=RNG(55);$('evRand').addEventListener('click',()=>{sel.selectedIndex=Math.floor(rng()*pick.length);show();});
  sel.value=pick[5];show();})();

/* ── 02 실기 과제지 ── */
(function(){const xs=$('dxSel'),ys=$('dySel'),OPT=FEAT.concat(['wb','strength']),LAB=Object.assign({},FEATK,{wb:'물-결합재비',strength:'압축 강도(MPa)'});
  OPT.forEach(f=>{xs.add(new Option(LAB[f],f));ys.add(new Option(LAB[f],f));});xs.value='water';ys.value='superplasticizer';
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
  $('dataLeg').innerHTML=GRADES.map((g,k)=>'<span><i style="background:'+GRADECOL[k]+'"></i>'+g+' ('+(k===0?'25MPa 미만':k===1?'25~45':'45 이상')+') '+cnt[k]+'건 · 시험 '+tst[k]+'</span>').join('')+'<span>○ 속이 빈 점 = 봉인된 시험지</span><span>처음 축: 물 × 고성능 감수제 — 물이 많으면 감수제가 적다(얽힘)</span>';
  $('concTbl').innerHTML='<table><thead><tr><th>#</th>'+FEAT.map(f=>'<th><span class="tg x">특징 X</span>'+FEATK[f]+'</th>').join('')+'<th>압축 강도<br>(MPa)</th><th><span class="tg t">타겟 y</span>35MPa 합격</th><th><span class="tg t">타겟 y</span>등급</th><th>용도</th></tr></thead><tbody>'+
    ROWS.map(r=>'<tr><td>'+(r.i+1)+'</td>'+FEAT.map(f=>'<td>'+r[f]+'</td>').join('')+'<td>'+r.strength+'</td><td class="y">'+PNAME[r.pass]+'</td><td class="y">'+GRADES[r.grade]+'</td><td>'+(r.test?'<span class="pill no">봉인</span>':'훈련')+'</td></tr>').join('')+'</tbody></table>';
  /* 분포 모양 검사 */
  const keys=FEAT.concat(['lage']);
  $('shapeTbl').innerHTML='<thead><tr><th>특징</th><th>평균</th><th>표준편차</th><th>왜도</th><th>0인 배합</th><th>서로 다른 값</th><th>종 모양?</th></tr></thead><tbody>'+keys.map(f=>{const v=TR.map(XF[f]),m=ML.mean(v),sd=Math.sqrt(ML.mean(v.map(x=>(x-m)*(x-m)))),sk=ML.mean(v.map(x=>Math.pow((x-m)/sd,3))),z=v.filter(x=>x===0).length/v.length,u=new Set(v).size;
    const verdict=z>0.3?'<b style="color:var(--stamp)">아니오</b> · 0에 몰림':u<20?(Math.abs(sk)<0.5?'<b style="color:var(--amber)">띄엄띄엄 '+u+'값</b> · 퍼진 모양은 종에 가깝다':'<b style="color:var(--stamp)">아니오</b> · 띄엄띄엄 '+u+'값, 한쪽으로 치우침'):Math.abs(sk)>1?'<b style="color:var(--amber)">치우침</b>':'<b style="color:var(--ok)">대체로</b>';
    return '<tr><td>'+XL[f]+'</td><td>'+(Math.abs(m)<10?m.toFixed(2):m.toFixed(1))+'</td><td>'+(sd<10?sd.toFixed(2):sd.toFixed(1))+'</td><td>'+fx(sk)+'</td><td>'+pct(z,0)+'</td><td>'+u+'</td><td>'+verdict+'</td></tr>';}).join('')+'</tbody>';
  /* 표준화 무관 */
  const sc=ML.scaler(A),m0=mainNB(),ms=gnbFit(sc.t(A),YTR,2),a0=accOf(m0,B,YTE),as=accOf(ms,sc.t(B),YTE);
  $('stdStat').innerHTML=statGrid([[pct(a0),'원래 단위 (시험지)'],[pct(as),'표준화한 뒤 (시험지)']],2)+'<p class="foot">'+(Math.abs(a0-as)<1e-9?'정확히 같다. ':'거의 같다. ')+'시험지는 05장에서 처음 여는 것이 원칙이지만, 이 숫자는 "표준화가 판정을 바꾸지 않는다"는 확인용으로만 보인다.</p>';})();

/* ── 03 AI 비서 ── */
const MAIN_PROMPT=`너는 머신러닝 수업 조교다. Google Colab에서 바로 실행되는 파이썬 코드를 써 줘.

[환경] Colab 기본 라이브러리(numpy, pandas, scipy, scikit-learn)와 ucimlrepo만 쓴다. 첫 줄에 scikit-learn 버전을 출력한다.
[데이터] 〔ucimlrepo의 fetch_ucirepo(id=165)〕 로 콘크리트 압축 강도 데이터를 읽는다.
  - 특징 X: 〔시멘트, 고로 슬래그, 플라이 애시, 물, 고성능 감수제, 굵은 골재, 잔골재, 재령〕 8개. 모두 연속값(kg/m³, 일)이다.
  - 타겟 y: 〔압축 강도가 35MPa 이상이면 1, 아니면 0〕 (이진 분류)
[규정]
  1) 강도를 〔25, 45〕MPa로 나눈 3등급으로 stratify 해서 train_test_split(test_size=0.25, random_state=42)로 먼저 나눈다.
  2) 특징이 연속값이므로 GaussianNB를 쓴다. 다른 나이브 베이즈(MultinomialNB 등)를 쓰려면 까닭을 먼저 설명한다.
  3) 회귀(강도 MPa 예측)는 하지 않는다.
[출력] 사전 확률(class_prior_), 클래스별 평균(theta_)·분산(var_) 표, 훈련·시험 정확도, 혼동행렬, classification_report, 시험 데이터 앞 3개의 합격 확률(predict_proba에서 합격 열을 classes_로 확인해 고른다).
[점검] 클래스 안 특징끼리의 상관계수 상위 3쌍과, 특징별 히스토그램의 치우침(왜도)을 출력한다.
[금지] 시험 데이터로 학습하거나 하이퍼파라미터를 고르지 않는다. 데이터를 지어내지 않는다.
[형식] 셀 단위로 나누고, 각 셀 위에 무엇을 하는지 한국어 주석을 단다. 코드 뒤에 내가 확인해야 할 점 3가지를 적어 줘.`;
$('mainPrompt').textContent=MAIN_PROMPT;$('copyMain').addEventListener('click',()=>copyText(MAIN_PROMPT,'프롬프트를 복사했습니다'));
(function(){const LINES=[
 ['import pandas as pd'],['from ucimlrepo import fetch_ucirepo'],['from sklearn.model_selection import train_test_split'],['from sklearn.naive_bayes import GaussianNB, MultinomialNB'],[''],
 ['uci = fetch_ucirepo(id=165)',0,'UCI에서 콘크리트 데이터를 실제로 받았다. 괜찮다.'],
 ['X = uci.data.features',0,'특징 여덟 열이다. 괜찮다.'],['strength = uci.data.targets.iloc[:, 0]',0,'압축 강도(MPa) 열이다. 괜찮다.'],['y = (strength >= 35).astype(int)   # 35MPa 합격 = 1',0,'강도를 문턱에 대어 합격 이름표를 만들었다. 괜찮다.'],[''],
 ['X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(X, y, strength, test_size=0.25)',1,'<b>stratify와 random_state가 없다.</b> 돌릴 때마다 점수가 달라지고, 고강도 배합이 한쪽으로 쏠릴 수 있다. 위원회 규정대로 강도 등급으로 <code>stratify</code>하고 <code>random_state=42</code>를 넣는다.'],[''],
 ['model = MultinomialNB()   # 숫자 특징에 맞는 나이브 베이즈',2,'<b>종류를 잘못 골랐다.</b> MultinomialNB는 "단어가 몇 번 나왔나" 같은 횟수용이다. kg과 일(재령)을 횟수처럼 섞어 세므로 <b>단위만 바꿔도 결과가 바뀐다</b>. Colab 실행 결과(A장 6번 칸): 시험지에서 재령을 일로 적으면 77.1%, 주로 적으면 66.3%. GaussianNB는 둘 다 70.9%로 같다. 연속값에는 GaussianNB를 쓴다.'],
 ['model.fit(X_train, y_train)',0,'훈련 데이터로 학습했다. 괜찮다.'],
 ['print("정확도:", model.score(X_train, y_train))',3,'<b>훈련 데이터로 채점했다.</b> 외운 문제로 시험 본 셈이다. <code>model.score(X_test, y_test)</code>로 채점한다.'],
 ['p_pass = model.predict_proba(X_test)[:, 0]   # 합격 확률',4,'<b>열을 거꾸로 골랐다.</b> predict_proba의 열 순서는 <code>model.classes_</code> 순서([0, 1])다. 0번 열은 불합격(0)의 확률이다. 합격 확률은 <code>[:, 1]</code>이다.'],[''],
 ['reg = GaussianNB().fit(X_train, s_train.round().astype(int))   # 강도(MPa)도 예측',5,'<b>분류 모델로 회귀를 흉내 냈다.</b> 강도를 정수로 반올림하면 오류 없이 돌지만, 수십 개의 "강도 클래스" 하나에 배합이 몇 건뿐이라 평균·분산을 제대로 잴 수 없고, 훈련에 없던 강도는 영영 답할 수 없다. 나이브 베이즈는 분류 전용이다(06장). 회귀는 5-1~5-3 회귀팀에 맡긴다.'],
 ['print("예측 강도:", reg.predict(X_test[:5]))',5,'<b>분류 모델로 회귀를 흉내 냈다.</b> 바로 윗줄과 같은 문제다. 나오는 숫자는 "가장 그럴듯한 강도 클래스 번호"일 뿐, MPa를 맞힌 것이 아니다.']];
  const host=$('bugCode'),found=new Set();
  host.innerHTML=LINES.map((l,i)=>'<div class="ln" role="listitem" tabindex="0" data-i="'+i+'"><span class="no">'+(i+1)+'</span><span>'+hl(l[0]||' ')+'</span></div>').join('');
  function score(){$('bugScore').textContent=found.size+' / 5';if(found.size===5){$('bugScore').className='pill ok';}}
  function pick(el){const i=+el.getAttribute('data-i'),l=LINES[i];if(!l[0])return;
    if(l[1]){found.add(l[1]);host.querySelectorAll('.ln').forEach(x=>{if(LINES[+x.getAttribute('data-i')][1]===l[1])x.classList.add('hit');});$('bugMsg').innerHTML=(i+1)+'번 줄: '+l[2]+(found.size===5?'<br><b>다섯 군데를 모두 찾았다.</b> 고친 코드는 A장의 Colab 코드와 같다.':'');}
    else{el.classList.add('miss');$('bugMsg').innerHTML=(i+1)+'번 줄: '+(l[2]||'이 줄은 괜찮다.');}score();}
  host.addEventListener('click',e=>{const el=e.target.closest('.ln');if(el)pick(el);});
  host.addEventListener('keydown',e=>{if((e.key==='Enter'||e.key===' ')&&e.target.classList.contains('ln')){e.preventDefault();pick(e.target);}});
  $('bugReveal').addEventListener('click',()=>{LINES.forEach((l,i)=>{if(l[1]){found.add(l[1]);host.querySelector('[data-i="'+i+'"]').classList.add('hit','reveal');}});score();$('bugMsg').innerHTML='빨간 줄 다섯 군데: ① 11번 나누기 규정 ② 13번 나이브 베이즈 종류 ③ 15번 훈련 데이터로 채점 ④ 16번 predict_proba 열 순서 ⑤ 18~19번 분류 모델로 회귀.';});
  $('bugReset').addEventListener('click',()=>{found.clear();host.querySelectorAll('.ln').forEach(x=>x.classList.remove('hit','miss','reveal'));$('bugScore').className='pill';score();$('bugMsg').textContent='줄을 하나 눌러 보세요.';});score();})();
(function(){const CK=[['데이터를 실제로 읽었나','AI가 예제 데이터(iris 등)를 지어내거나 바꾸지 않았는지, 행이 1,030개인지 본다.'],['특징의 모양에 맞는 나이브 베이즈를 골랐나','연속값 → GaussianNB, 횟수 → MultinomialNB, 0/1 → BernoulliNB, 범주 → CategoricalNB. 점수가 나와도 종류가 틀리면 우연이다.'],['타겟이 이름표(클래스)인가','강도(MPa) 같은 연속값을 타겟으로 넣지 않았나. 반올림해서 넣는 것도 회귀가 아니다.'],['stratify와 random_state를 넣었나','등급 비율이 시험지에도 유지되고, 점수가 매번 같게 나온다.'],['시험지(X_test)는 마지막 채점에만 썼나','priors나 var_smoothing을 시험지 점수로 고르면 시험지가 훈련 데이터가 된다.'],['합격 확률의 열을 classes_로 확인했나','predict_proba의 열 순서는 classes_ 순서다. 합격(1)은 두 번째 열이다.'],['특징 분포를 먼저 봤나','히스토그램이 한쪽으로 치우쳤거나 몇 값에 몰렸으면 종 모양 가정이 틀린다. 로그 변환을 생각한다.'],['확률의 크기를 그대로 믿지 않았나','얽힌 특징이 많으면 확률이 0이나 1로 쏠린다. 순서는 맞아도 크기는 과장될 수 있다.'],['결과를 기준선·합격선과 비교했나','"늘 다수 쪽"(DummyClassifier)과 매뉴얼 씨(76.7%)보다 낫지 않으면 뽑을 까닭이 없다.']];
  const ul=$('checks');ul.innerHTML=CK.map((c,i)=>'<li><input type="checkbox" id="ck'+i+'"><label for="ck'+i+'"><b>'+c[0]+'</b><small>'+c[1]+'</small></label></li>').join('');
  function upd(){const n=ul.querySelectorAll('input:checked').length;ul.querySelectorAll('li').forEach(li=>li.classList.toggle('done',li.querySelector('input').checked));$('checkScore').textContent='확인 '+n+' / '+CK.length+(n===CK.length?' — 이제 돌려 본다. 결과도 확인 대상이다.':'');}
  ul.addEventListener('change',upd);upd();})();

/* ── 04 수습 기간: 한 건씩 읽으며 평균·분산 세기 (웰퍼드 방법) ── */
(function(){const cv=$('cntCv'),order=TR.map((_,i)=>i),rng=RNG(505);for(let i=order.length-1;i>0;i--){const j=Math.floor(rng()*(i+1));[order[i],order[j]]=[order[j],order[i]];}
  const RANGE=FEAT.map((f,j)=>{const v=A.map(x=>x[j]),mn=Math.min(...v),mx=Math.max(...v),p=(mx-mn)*0.12;return [f==='age'?-60:Math.max(f==='water'?mn-p:-p*0.5,mn-p),mx+p];});
  let st;function reset(){st={t:0,n:[0,0],mu:[FEAT.map(()=>0),FEAT.map(()=>0)],M:[FEAT.map(()=>0),FEAT.map(()=>0)],last:-1,ops:0,play:false};draw();}
  function read(){if(st.t>=order.length)return false;const i=order[st.t++],k=YTR[i],x=A[i];st.n[k]++;const n=st.n[k];
    for(let j=0;j<FEAT.length;j++){const d=x[j]-st.mu[k][j];st.mu[k][j]+=d/n;st.M[k][j]+=d*(x[j]-st.mu[k][j]);}st.ops+=FEAT.length*6+1;st.last=i;return true;}
  function draw(){const W0=cv.clientWidth||cv.parentElement.clientWidth||600,nc=W0>=720?4:2,nr=Math.ceil(FEAT.length/nc),ph=118,gap=10,W=fit(cv,nr*(ph+gap)-gap),ctx=cv.getContext('2d'),C=cols(),pw=(W-(nc-1)*gap)/nc;
    ctx.clearRect(0,0,W,nr*(ph+gap));
    FEAT.forEach((f,j)=>{const ox=(j%nc)*(pw+gap),oy=Math.floor(j/nc)*(ph+gap),[lo,hi]=RANGE[j],pt=18,gh=ph-36,sx=x=>ox+6+(x-lo)/(hi-lo)*(pw-12);
      ctx.strokeStyle=C.rule;ctx.lineWidth=1;ctx.strokeRect(ox+.5,oy+.5,pw-1,ph-1);ctx.fillStyle=C.text;ctx.font=font(12,700);ctx.textAlign='left';ctx.fillText(FEATK[f],ox+8,oy+14);
      if(lo<0){ctx.fillStyle=rgba(C.stamp,.07);ctx.fillRect(ox+6,oy+pt,sx(0)-ox-6,gh);}
      const base=oy+pt+gh;ctx.strokeStyle=C.muted;ctx.beginPath();ctx.moveTo(ox+6,base);ctx.lineTo(ox+pw-6,base);ctx.stroke();
      /* 읽은 배합의 눈금 (아래 줄) */
      for(let q=0;q<st.t;q++){const i=order[q],k=YTR[i];ctx.fillStyle=rgba(k?C.blue:C.text,.22);ctx.fillRect(sx(A[i][j]),base+3+(k?7:0),1.2,6);}
      let ym=0;const curves=[0,1].map(k=>{if(st.n[k]<2)return null;const v=st.M[k][j]/st.n[k]||1e-9;ym=Math.max(ym,1/Math.sqrt(2*Math.PI*v));return v;});
      [0,1].forEach(k=>{const v=curves[k];if(v==null)return;ctx.strokeStyle=k?C.blue:C.text;ctx.lineWidth=2.2;ctx.beginPath();for(let s=0;s<=80;s++){const x=lo+(hi-lo)*s/80,y=base-pdf(x,st.mu[k][j],v)/ym*(gh-4);s?ctx.lineTo(sx(x),y):ctx.moveTo(sx(x),y);}ctx.stroke();
        ctx.fillStyle=k?C.blue:C.text;ctx.fillRect(sx(st.mu[k][j])-1,base-3,2,6);});
      if(st.last>=0){ctx.strokeStyle=C.stamp;ctx.lineWidth=2;const xx=sx(A[st.last][j]);ctx.beginPath();ctx.moveTo(xx,oy+pt);ctx.lineTo(xx,base);ctx.stroke();}
      ctx.fillStyle=C.muted;ctx.font=font(10.5,500,1);ctx.textAlign='right';ctx.fillText(Math.round(lo<0&&f==='age'?0:lo)+' ~ '+Math.round(hi),ox+pw-8,oy+14);});
    info();}
  function info(){const done=st.t>=order.length;let h='읽은 배합 <b>'+st.t+'</b> / 772 · 불합격 장부 '+st.n[0]+'건 · 합격 장부 '+st.n[1]+'건 · 지금 사전 확률 P(합격) = <b>'+(st.t?(st.n[1]/st.t).toFixed(3):'—')+'</b>';
    if(st.last>=0&&!done){const r=TR[st.last];h+='<br>방금 읽은 배합: 실제 '+r.strength.toFixed(1)+'MPa → <b>'+PNAME[r.pass]+'</b> 장부만 고쳤다(빨간 세로줄이 그 배합의 값). 다른 클래스의 곡선은 그대로다.';}
    else if(st.t===0)h+='<br>장부가 비어 있다. ▶ 재생을 누르면 배합을 한 건씩 읽는다. 처음 몇 건은 곡선이 크게 흔들리고, 읽을수록 자리를 잡는다.';
    if(done){const m=mainNB();h+='<br><b>수습 끝.</b> 772건을 한 번 훑었다. 셈 '+st.ops.toLocaleString()+'번 남짓, 같은 계산을 한꺼번에 하면 브라우저에서 '+Math.max(0.1,MBms).toFixed(1)+'ms. 아래 장부의 숫자 34개가 이 지원자가 외운 전부다. 재령 그림의 합격 곡선이 0일보다 왼쪽까지 퍼진 것을 눈여겨보자(05장).';
      const err=Math.max(...[0,1].map(k=>Math.max(...FEAT.map((f,j)=>Math.abs(st.mu[k][j]-m.mu[k][j])))));if(err>1e-6)h+=' (확인: 일괄 계산과 차이 '+err.toExponential(1)+')';}
    $('cntInfo').innerHTML=h;}
  let raf=0,fr=0;
  function frame(){if(!st.play)return;fr++;const k=st.t<12?(fr%14===0?1:0):st.t<60?(fr%3===0?1:0):st.t<200?2:8;for(let i=0;i<k;i++)read();draw();if(st.t>=order.length){stop();return;}raf=requestAnimationFrame(frame);}
  function stop(){st.play=false;cancelAnimationFrame(raf);$('cntPlay').textContent='▶ 재생';}
  $('cntPlay').addEventListener('click',()=>{if(st.play){stop();return;}if(st.t>=order.length)reset();st.play=true;$('cntPlay').textContent='❚❚ 멈춤';if(REDUCE){while(read());draw();stop();return;}raf=requestAnimationFrame(frame);});
  $('cntStep').addEventListener('click',()=>{stop();read();draw();});
  $('cntAll').addEventListener('click',()=>{stop();while(read());draw();});
  $('cntReset').addEventListener('click',()=>{stop();reset();});
  reset();onRedraw(draw);
  /* 장부 표 */
  const m=mainNB();$('ledgerTbl').innerHTML='<thead><tr><th>특징</th><th>불합격 μ ± σ</th><th>합격 μ ± σ</th><th>떨어진 정도<br><small>|μ₁−μ₀| ÷ 평균 σ</small></th></tr></thead><tbody>'+FEAT.map((f,j)=>{const s0=Math.sqrt(m.va[0][j]),s1=Math.sqrt(m.va[1][j]),dd=Math.abs(m.mu[1][j]-m.mu[0][j])/((s0+s1)/2),f1=v=>v<20?v.toFixed(1):v.toFixed(0);
    return '<tr><td>'+FEATK[f]+'</td><td>'+f1(m.mu[0][j])+' ± '+f1(s0)+'</td><td>'+f1(m.mu[1][j])+' ± '+f1(s1)+'</td><td><b>'+dd.toFixed(2)+'</b></td></tr>';}).join('')+'</tbody>';
  $('ledgerFoot').innerHTML='사전 확률: 불합격 '+m.pri[0].toFixed(3)+' · 합격 '+m.pri[1].toFixed(3)+'. 평균 16개 + 분산 16개 + 사전 확률 2개 = 숫자 34개. "떨어진 정도"가 클수록 그 특징의 두 종이 멀리 떨어져 증거가 또렷하다. 재령·시멘트가 크고 골재·애시는 작다.';})();

/* ── 04 경계 그림 ── */
(function(){const cv=$('bdCv'),H=330;
  const SETS={expert:{nx:'물-결합재비',ny:'재령(일, 로그 눈금)',fx:r=>r.wb,fy:r=>Math.log10(r.age),logy:true},rawage:{nx:'물-결합재비',ny:'재령(일)',fx:r=>r.wb,fy:r=>r.age,logy:false},raw:{nx:'시멘트(kg/m³)',ny:'물(kg/m³)',fx:r=>r.cement,fy:r=>r.water,logy:false}};
  let cur='expert',D=null;
  function setup(k){cur=k;const S=SETS[k],X=TR.map(r=>[S.fx(r),S.fy(r)]),m=gnbFit(X,YTR,2),xs=X.map(p=>p[0]),ys=X.map(p=>p[1]),px=(Math.max(...xs)-Math.min(...xs))*.05,py=(Math.max(...ys)-Math.min(...ys))*.07;
    D={S,X,m,box:{x0:Math.min(...xs)-px,x1:Math.max(...xs)+px,y0:Math.min(...ys)-py-(k==='rawage'?90:0),y1:Math.max(...ys)+py},acc:accOf(m,X,YTR)};draw();}
  function draw(){const W=fit(cv,H),ctx=cv.getContext('2d'),C=cols(),pl=50,pr=10,pt=10,pb=38,gw=W-pl-pr,gh=H-pt-pb,bx=D.box,S=D.S,m=D.m;
    const sx=x=>pl+(x-bx.x0)/(bx.x1-bx.x0)*gw,sy=y=>pt+(1-(y-bx.y0)/(bx.y1-bx.y0))*gh,ix=X=>bx.x0+(X-pl)/gw*(bx.x1-bx.x0),iy=Y=>bx.y0+(1-(Y-pt)/gh)*(bx.y1-bx.y0);
    ctx.clearRect(0,0,W,H);const cs=6,nx=Math.ceil(gw/cs),ny=Math.ceil(gh/cs),G=[];
    for(let a=0;a<=nx;a++){G.push([]);for(let b=0;b<=ny;b++){const p=m.proba([ix(pl+a*cs),iy(pt+b*cs)])[1];G[a].push(p);}}
    for(let a=0;a<nx;a++)for(let b=0;b<ny;b++){const q=G[a][b];ctx.fillStyle=q>=.5?rgba(C.blue,.05+.22*(q-.5)*2):rgba(C.text,.04+.12*(.5-q)*2);ctx.fillRect(pl+a*cs,pt+b*cs,cs,cs);}
    if(cur==='rawage'){ctx.fillStyle=rgba(C.stamp,.08);ctx.fillRect(pl,sy(0),gw,pt+gh-sy(0));ctx.fillStyle=C.stamp;ctx.font=font(12,700);ctx.textAlign='left';ctx.fillText('재령 < 0 (있을 수 없는 곳)',pl+8,pt+gh-8);}
    ctx.fillStyle=C.blue;for(let a=0;a<nx;a++)for(let b=0;b<ny;b++){const q=G[a][b];if((q-.5)*(G[a+1][b]-.5)<0||(q-.5)*(G[a][b+1]-.5)<0)ctx.fillRect(pl+a*cs+cs/2-1.5,pt+b*cs+cs/2-1.5,3,3);}
    ctx.strokeStyle=C.rule;ctx.fillStyle=C.muted;ctx.font=font(12);ctx.lineWidth=1;ctx.strokeRect(pl,pt,gw,gh);
    for(let k=0;k<=4;k++){const gx=bx.x0+(bx.x1-bx.x0)*k/4;ctx.textAlign='center';ctx.fillText(gx.toFixed(gx<5?2:0),sx(gx),H-20);}
    if(S.logy){[1,3,7,28,90,365].forEach(d=>{const v=Math.log10(d);if(v<bx.y0||v>bx.y1)return;ctx.textAlign='right';ctx.fillText(d,pl-5,sy(v)+4);});}
    else for(let k=0;k<=4;k++){const gy=bx.y0+(bx.y1-bx.y0)*k/4;ctx.textAlign='right';ctx.fillText(gy.toFixed(0),pl-5,sy(gy)+4);}
    ctx.textAlign='center';ctx.fillText(S.nx,pl+gw/2,H-3);ctx.save();ctx.translate(11,pt+gh/2);ctx.rotate(-Math.PI/2);ctx.fillText(S.ny,0,0);ctx.restore();
    ctx.save();ctx.beginPath();ctx.rect(pl,pt,gw,gh);ctx.clip();
    D.X.forEach((p,i)=>circ(ctx,sx(p[0]),sy(p[1]),2.6,rgba(YTR[i]?C.blue:C.text,.6),null));
    [0,1].forEach(k=>{const mx=m.mu[k][0],my=m.mu[k][1],rx=Math.sqrt(m.va[k][0]),ry=Math.sqrt(m.va[k][1]),col=k?C.blue:C.text;ctx.strokeStyle=col;ctx.lineWidth=2.5;ctx.setLineDash([6,4]);ctx.beginPath();
      ctx.ellipse(sx(mx),sy(my),Math.abs(sx(mx+rx)-sx(mx)),Math.abs(sy(my+ry)-sy(my)),0,0,7);ctx.stroke();ctx.setLineDash([]);ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(sx(mx)-8,sy(my));ctx.lineTo(sx(mx)+8,sy(my));ctx.moveTo(sx(mx),sy(my)-8);ctx.lineTo(sx(mx),sy(my)+8);ctx.stroke();});
    ctx.restore();
    ctx.font=font(12,700);ctx.textAlign='left';const lx=pl+8;ctx.fillStyle=rgba(C.panel,.88);ctx.fillRect(lx-4,pt+4,206,50);ctx.fillStyle=C.text;ctx.fillText('● 불합격  ✛ 평균 · 점선 = 1σ 타원',lx,pt+18);ctx.fillStyle=C.blue;ctx.fillText('● 합격 35MPa↑',lx,pt+33);ctx.fillText('··· 경계 (P = 0.5)',lx,pt+48);
    const T={expert:'점선 타원이 가로·세로축에 나란하다. 이것이 "독립" 가정이다(기울어진 타원을 그릴 수 없다). 두 종의 폭이 달라 경계가 휘었다. 전문가 특징 두 개는 서로 덜 얽혀 있고 모양도 종에 가까워 이 지원자와 잘 맞는다.',
      rawage:'재령을 로그 없이 넣으면 합격 쪽 타원이 아래로 길게 늘어나 음수 재령까지 덮는다. 경계도 엉뚱한 곳에서 휜다. 같은 정보인데 모양만 바뀌어도 판정이 달라진다.',
      raw:'시멘트·물만으로는 두 무리가 크게 겹친다. 재령을 모르니 같은 배합의 7일·90일 시편을 구별할 수 없다.'};
    $('bdInfo').innerHTML='특징 2개 나이브 베이즈 · 훈련 정확도 <b>'+pct(D.acc)+'</b> (772건, 시험지는 05장에서). '+T[cur];}
  seg('#bdChoice button',v=>setup(v));setup('expert');onRedraw(draw);})();

/* ── 05 성과 평가 ── */
function renderEval(){const m=mainNB(),C=cols(),pte=B.map(m.predict),ptr=A.map(m.predict),te=ML.acc(YTE,pte),tr=ML.acc(YTR,ptr),bs=baseAcc();
  $('evalStat').innerHTML=statGrid([[pct(te),'시험 정확도 ('+Math.round(te*258)+' / 258건)','a'],[pct(tr),'훈련 정확도 (772건)'],[pct(PASSLINE),'합격선 · 5-0 매뉴얼 씨','m'],[pct(bs.acc),'기준선 · 늘 '+rago('"'+PNAME[bs.k]+'"')+'만 답할 때']],4);
  const M=ML.confusion(YTE,pte,2);$('cm2').innerHTML=cmGrid(M,PNAME,k=>k?C.blue:C.muted);
  const sc=prf(YTE,pte,2);$('prTbl').innerHTML='<thead><tr><th>이름표</th><th>시험 수</th><th>정밀도</th><th>재현율</th><th>F1</th></tr></thead><tbody>'+sc.map(s=>'<tr><td>'+PNAME[s.k]+'</td><td>'+s.n+'</td><td>'+s.P.toFixed(3)+'</td><td>'+s.R.toFixed(3)+'</td><td>'+s.F.toFixed(3)+'</td></tr>').join('')+'</tbody>';
  const fp=M[0][1],fn=M[1][0],p28=TE.filter((r,i)=>r.age===28&&r.pass===1),miss28=TE.filter((r,i)=>r.age===28&&r.pass===1&&pte[i]===0).length;
  $('hiddenTxt').innerHTML='258건 중 '+Math.round(te*258)+'건을 맞혀 정확도는 <b>'+pct(te)+'</b>, 합격선 '+pct(PASSLINE)+'에 못 미친다. 틀린 '+(fp+fn)+'건 가운데 <b>합격 배합을 불합격이라 한 것이 '+fn+'건</b>, 미달 배합을 합격이라 한 것은 '+fp+'건이다. 위험한 출하는 적지만, 멀쩡한 배합을 너무 많이 버린다. 놓친 '+fn+'건 중 <b>'+miss28+'건이 재령 28일 시편</b>이다(28일 합격 배합 '+p28.length+'건 중 '+miss28+'건). 가장 흔한 28일 시편에서 왜 헤맸을까? 아래 두 그림이 답한다.';
  /* 까닭 ① 재령 분포 */
  (function(){const cv=$('ageCv'),H=280;let mode='raw';
    function draw(){const W=fit(cv,H),ctx=cv.getContext('2d'),C=cols(),pl=46,pr=12,pt=14,pb=40,gw=W-pl-pr,gh=H-pt-pb,lg=mode==='log',f=lg?(r=>Math.log10(r.age)):(r=>r.age);
      const lo=lg?-0.4:-170,hi=lg?2.9:400,bw=lg?0.1:10,NB=Math.round((hi-lo)/bw),sx=x=>pl+(x-lo)/(hi-lo)*gw;
      const v=[[],[]];TR.forEach(r=>v[r.pass].push(f(r)));const mu=v.map(a=>ML.mean(a)),va=v.map((a,k)=>ML.mean(a.map(x=>(x-mu[k])*(x-mu[k]))));
      const hist=[0,1].map(k=>{const h=new Array(NB).fill(0);v[k].forEach(x=>{const b=Math.floor((x-lo)/bw);if(b>=0&&b<NB)h[b]++;});return h;});
      let ym=0;hist.forEach(h=>h.forEach(c=>{if(c>ym)ym=c;}));ym*=1.05;const sy=c=>pt+gh-Math.min(1,c/ym)*gh;
      ctx.clearRect(0,0,W,H);ctx.strokeStyle=C.rule;ctx.strokeRect(pl,pt,gw,gh);
      if(!lg){ctx.fillStyle=rgba(C.stamp,.08);ctx.fillRect(pl,pt,sx(0)-pl,gh);}
      [0,1].forEach(k=>{ctx.fillStyle=rgba(k?C.blue:C.muted,k?.55:.6);hist[k].forEach((c,b)=>{if(!c)return;const x=sx(lo+b*bw),w=Math.max(1.5,(sx(lo+(b+1)*bw)-x)/2-0.5),y=sy(c);ctx.fillRect(x+(k?w:0),y,w,pt+gh-y);});});
      [0,1].forEach(k=>{ctx.strokeStyle=k?C.blue:C.text;ctx.lineWidth=3;ctx.beginPath();for(let i=0;i<=240;i++){const x=lo+(hi-lo)*i/240,y=sy(v[k].length*bw*pdf(x,mu[k],va[k]));i?ctx.lineTo(sx(x),y):ctx.moveTo(sx(x),y);}ctx.stroke();});
      ctx.fillStyle=C.muted;ctx.font=font(12);ctx.textAlign='center';
      (lg?[1,3,7,28,90,365].map(d=>[Math.log10(d),d]):(W<560?[-150,0,90,180,365]:[-150,-100,-50,0,28,90,180,270,365]).map(d=>[d,d])).forEach(p=>ctx.fillText(p[1],sx(p[0]),H-22));
      ctx.fillText(lg?'재령 (일, 로그 눈금)':'재령 (일)',pl+gw/2,H-4);ctx.save();ctx.translate(12,pt+gh/2);ctx.rotate(-Math.PI/2);ctx.fillText('배합 수',0,0);ctx.restore();
      ctx.font=font(12,700);ctx.textAlign='right';ctx.fillStyle=C.text;ctx.fillText('불합격: 막대 + 곡선',pl+gw-8,pt+16);ctx.fillStyle=C.blue;ctx.fillText('합격: 막대 + 곡선',pl+gw-8,pt+32);
      if(!lg){ctx.fillStyle=C.stamp;ctx.textAlign='left';ctx.fillText('음수 재령',pl+6,pt+16);}
      const neg=[0,1].map(k=>ncdf((0-mu[k])/Math.sqrt(va[k]))),at28=[0,1].map(k=>pdf(lg?Math.log10(28):28,mu[k],va[k]));
      $('ageInfo').innerHTML=lg?'log 재령으로 바꾸면 막대가 3·7·28·90일 근처에 고르게 놓이고, 종 모양 곡선이 막대를 훨씬 잘 따라간다. 28일에서 합격 종의 높이 ÷ 불합격 종의 높이 = <b>'+(at28[1]/at28[0]).toFixed(2)+'</b> (재령 그대로일 때보다 합격 쪽으로 기운다). 이 한 가지만 바꿔도 시험 정확도가 오른다(아래 특징 바구니).':
        '막대는 3·7·28·56·90일 같은 몇 군데에 몰려 있다(실험 계획의 재령). 종 모양 곡선은 그 사이를 메우고 <b>0일보다 왼쪽까지 퍼진다</b>: 불합격 '+pct(neg[0])+', 합격 '+pct(neg[1])+'의 확률이 있을 수 없는 음수 재령에 놓였다. 합격 곡선은 폭이 넓어(오래 굳힌 배합이 섞여서) 28일에서 높이가 낮다. 28일에서 합격 종 ÷ 불합격 종 = <b>'+(at28[1]/at28[0]).toFixed(2)+'</b>. 1보다 작으니 28일이라는 사실이 <b>불합격 쪽 증거</b>로 읽힌다. 28일 합격 배합을 놓친 까닭이다.';}
    seg('#ageChoice button',v=>{mode=v;draw();});draw();onRedraw(draw);})();
  /* 까닭 ② 상관 */
  (function(){function show(k){const X=A.filter((_,i)=>YTR[i]===+k),d=FEAT.length,mu=FEAT.map((_,j)=>ML.mean(X.map(x=>x[j]))),sd=FEAT.map((_,j)=>Math.sqrt(ML.mean(X.map(x=>(x[j]-mu[j])**2)))),R=[];
      for(let a=0;a<d;a++){R.push([]);for(let b=0;b<d;b++)R[a].push(ML.mean(X.map(x=>(x[a]-mu[a])*(x[b]-mu[b])))/(sd[a]*sd[b]));}
      let h='<div style="display:grid;grid-template-columns:58px repeat('+d+',minmax(28px,1fr));gap:2px;font-size:11px;min-width:330px"><div></div>'+FEAT.map(f=>'<div style="text-align:center;color:var(--muted);writing-mode:vertical-rl;transform:rotate(180deg);height:58px;justify-self:center">'+SHORT[f]+'</div>').join('');
      for(let a=0;a<d;a++){h+='<div style="text-align:right;color:var(--muted);align-self:center;padding-right:4px">'+SHORT[FEAT[a]]+'</div>';for(let b=0;b<d;b++){const r=R[a][b],c=a===b?'var(--panel2)':r>0?rgba('#1f6fe5',Math.min(.9,Math.abs(r))):rgba('#c0392b',Math.min(.9,Math.abs(r)));
        h+='<div style="background:'+c+';border-radius:3px;text-align:center;padding:5px 0;font-family:var(--mono);color:'+(Math.abs(r)>.45&&a!==b?'#fff':'var(--text)')+'" title="'+SHORT[FEAT[a]]+' × '+SHORT[FEAT[b]]+'">'+(a===b?'·':r.toFixed(1).replace('-','−'))+'</div>';}}
      $('corrBox').innerHTML=h+'</div>';const pairs=[];for(let a=0;a<d;a++)for(let b=a+1;b<d;b++)pairs.push([Math.abs(R[a][b]),a,b,R[a][b]]);pairs.sort((p,q)=>q[0]-p[0]);
      $('corrFoot').innerHTML=PNAME[+k]+' 배합 '+X.length+'건 안에서 가장 얽힌 쌍: '+pairs.slice(0,3).map(p=>'<b>'+SHORT[FEAT[p[1]]]+'–'+SHORT[FEAT[p[2]]]+' '+fx(p[3])+'</b>').join(', ')+'. 나이브 가정은 이 칸들이 모두 0이라고 본다. 파랑 = 함께 늘고, 빨강 = 하나가 늘면 다른 하나가 준다.';}
    seg('#corrChoice button',show);show('1');})();
  /* 같은 증거 여러 번 세기 */
  (function(){const cache={};function stat(k){if(cache[k])return cache[k];const ex=x=>x.concat(new Array(k-1).fill(x[0])),Ak=A.map(ex),Bk=B.map(ex),mk=gnbFit(Ak,YTR,2),p=Bk.map(x=>mk.proba(x)[1]);
      let ll=0,xt=0;p.forEach((q,i)=>{const c=Math.min(1-1e-15,Math.max(1e-15,q));ll-=YTE[i]?Math.log(c):Math.log(1-c);if(q<.01||q>.99)xt++;});return cache[k]={acc:ML.acc(YTE,Bk.map(mk.predict)),ext:xt/p.length,ll:ll/p.length};}
    function upd(){const k=+$('dupR').value,s=stat(k),s1=stat(1);$('dupO').textContent=k+'번';
      $('dupStat').innerHTML=statGrid([[pct(s.acc),'시험 정확도','a'],[pct(s.ext,0),'0.01 미만·0.99 초과로 확신한 배합','s'],[s.ll.toFixed(3),'로그 손실 (작을수록 좋음)']],3);
      $('dupInfo').innerHTML=k===1?'지금은 시멘트를 한 번만 센다. 슬라이더를 올리면 똑같은 시멘트 열을 복사해 넣는다. 정보는 하나도 늘지 않는데, 나이브 베이즈는 서로 다른 증거로 알고 곱한다.':
        '시멘트를 '+k+'번 세면 정확도는 '+pct(s1.acc)+' → '+pct(s.acc)+'로 거의 그대로인데, 확신한 배합이 '+pct(s1.ext,0)+' → <b>'+pct(s.ext,0)+'</b>로 늘고 로그 손실이 '+s1.ll.toFixed(3)+' → <b>'+s.ll.toFixed(3)+'</b>로 나빠진다. 물과 고성능 감수제처럼 얽힌 특징이 실제 데이터에서 하는 일이 바로 이것이다. <b>정답률보다 확률이 먼저 망가진다.</b>';}
    $('dupR').addEventListener('input',upd);upd();})();
  /* 특징 바구니 */
  (function(){const BK=[['날것 8개 (기본 · 면접 판정)',fv],['재령 → log 재령',r=>fv(r).slice(0,7).concat([Math.log10(r.age)])],['8개 + 물-결합재비 + log 재령',r=>fv(r).concat([r.wb,Math.log10(r.age)])],['물-결합재비 + log 재령 둘만',r=>[r.wb,Math.log10(r.age)]]],cache={};
    $('basketChoice').innerHTML=BK.map((b,i)=>'<button data-v="'+i+'" aria-pressed="'+(i===0)+'">'+b[0]+'</button>').join('');
    function show(i){i=+i;if(!cache[i]){const f=BK[i][1],X=TR.map(f),Z=TE.map(f),mm=gnbFit(X,YTR,2);cache[i]={te:accOf(mm,Z,YTE),tr:accOf(mm,X,YTR),d:X[0].length};}const c=cache[i];
      $('basketStat').innerHTML=statGrid([[pct(c.te),'시험 정확도','a'],[pct(c.tr),'훈련 정확도'],[pct(PASSLINE),'합격선','m'],[String(c.d*4+2),'외울 숫자']],4);
      const T=['기본 설정이다. 면접 판정은 이 바구니로 한다.','재령 하나에 로그를 씌웠을 뿐인데 오른다. 종 모양에 맞게 다듬은 효과다.','매뉴얼 씨의 두 숫자를 더했다. 합격선을 넘는다. 단, 물-결합재비는 물·시멘트·슬래그·애시로 만든 값이라 원래 특징과 또 얽힌다.','특징을 오히려 줄였는데 가장 좋다. 서로 덜 얽히고 종 모양에 가까운 두 숫자만 남기면, 같은 증거를 여러 번 세는 일이 줄어든다. 5-4 로지스틱 회귀가 같은 두 특징으로 낸 87.6%와 비슷하다.'];
      $('basketInfo').innerHTML=T[i]+' <span class="muted">주의: 시험지를 보며 바구니를 고르면 반칙이다. 이 실험은 "무엇이 발목을 잡았나"를 보이는 진단이고, 실제로는 훈련 데이터 안에서 교차 검증으로 고른다.</span>';}
    seg('#basketChoice button',show);show(0);})();
  /* 과제 B 3등급 */
  const m3=main3(),p3=B.map(m3.predict),a3=ML.acc(GTE,p3),s3=prf(GTE,p3,3);
  $('cm3').innerHTML=cmGrid(ML.confusion(GTE,p3,3),GRADES,k=>GRADECOL[k]);
  $('g3Stat').innerHTML=statGrid([[pct(a3),'3등급 시험 정확도','a'],[pct(GTE.filter(v=>v===1).length/GTE.length),'기준선 (늘 보통)']],2)+'<p class="foot">보통 강도 116건 중 '+(116-Math.round(s3[1].R*116))+'건을 저강도·고강도로 흩뜨렸다. 저강도 재현율은 '+pct(s3[0].R)+'로 높지만, 그만큼 "저강도"를 남발한다. 5-4 로지스틱 회귀(소프트맥스)와 비교해 보자.</p>';
  /* 사전 확률 손잡이: 가능도 합은 그대로, 사전 확률만 바꾼다 */
  (function(){const L=B.map(x=>m.lik(x)),dp=m.pri[1];$('priorR').value=dp.toFixed(3);
    function upd(){const p=+$('priorR').value;$('priorO').textContent=p.toFixed(3);const lp1=Math.log(p),lp0=Math.log(1-p);let tp=0,fp=0,tn=0,fn=0;
      L.forEach((l,i)=>{const g=l[1]+lp1>l[0]+lp0?1:0,y=YTE[i];if(g&&y)tp++;else if(g&&!y)fp++;else if(!g&&y)fn++;else tn++;});const acc=(tp+tn)/258;
      $('priorStat').innerHTML=statGrid([[pct(acc),'시험 정확도','a'],[String(fp),'미달인데 "합격" (위험한 출하)','s'],[String(fn),'합격인데 "불합격"']],3);
      $('priorInfo').innerHTML=(Math.abs(p-dp)<0.0015?'데이터 비율('+dp.toFixed(3)+') 그대로다. ':'P(합격) = '+p.toFixed(2)+': ')+(p>0.6?'합격 쪽에 처음부터 점수를 더 주니 놓치던 합격 배합을 되찾아 정확도가 '+(acc>PASSLINE?'합격선을 넘기도 한다':'오른다')+'. 그 대가로 위험한 출하가 '+fp+'건으로 늘었다. ':p<0.4?'불합격 쪽으로 더 기울어 위험한 출하는 줄지만 멀쩡한 배합을 더 버린다. ':'')+
        '사전 확률을 바꾸는 것은 5-4의 임계값을 옮기는 것과 같은 일이다(log 오즈에 상수를 더한다). <span class="muted">시험지를 보며 고른 값은 판정에 쓰지 않는다.</span>';}
    $('priorR').addEventListener('input',upd);upd();})();}

/* ── 면접 결과 발표 ── */
const nUniq=new Set(ROWS.map(r=>r.strength.toFixed(2))).size;$('nUniq').textContent=nUniq;
Interview('iv5',{title:'나이브 베이즈는 35MPa 합격 판정 업무에 합격할까?',
  q:'조건: 봉인된 시험지 258건에서 합격선(매뉴얼 씨 '+pct(PASSLINE)+')과 기준선(늘 다수 쪽)을 모두 넘으면 합격. 학습은 브라우저에서 특징 8개 그대로, 사전 확률은 훈련 데이터 비율로 한다.',
  fly:'특징끼리 독립이라 치고 곱했다며. 처음 보는 258건에서도 그게 통해?',robo:'외울 숫자 몇 개, 학습에 몇 번 훑었어?',
  async run(setP){setP(.3);await wait(40);const m=mainNB();setP(.8);await wait(40);const pte=B.map(m.predict),te=ML.acc(YTE,pte),tr=ML.acc(YTR,A.map(m.predict)),bs=baseAcc(),M=ML.confusion(YTE,pte,2);setP(1);
    const pass=te>PASSLINE&&te>bs.acc,miss28=TE.filter((r,i)=>r.age===28&&r.pass===1&&pte[i]===0).length;
    return {result:pass?'pass':'hold',flyAns:'시험 '+pct(te)+' · 훈련 '+pct(tr)+' · 합격 배합을 "불합격"이라 한 것 '+M[1][0]+'건 (28일 시편 '+miss28+'건)',roboAns:'숫자 34개 · 훑기 1번 · 학습 '+Math.max(0.1,MBms).toFixed(1)+'ms (브라우저)',
      flyMood:pass?'arms':'coffee',roboMood:pass?'happy':'cry',asstMood:pass?'ledger':'sigh',docMood:pass?'judge':'scale',flyMove:pass?'jump':'shake',roboMove:pass?'pop':'tilt',
      measure:'봉인된 시험지 결과예요. 나이브 베이즈 <b>'+pct(te)+'</b>, 매뉴얼 씨 '+pct(PASSLINE)+', 늘 '+rago('"'+PNAME[bs.k]+'"')+'만 답하는 기준선 '+pct(bs.acc)+'. 기준선은 넘었지만 합격선에 '+((PASSLINE-te)*100).toFixed(1)+'%p 모자라요. 틀린 '+(M[0][1]+M[1][0])+'건 중 '+M[1][0]+'건이 멀쩡한 배합을 불합격이라 한 쪽이에요.',
      verdict:pass?'합격입니다.':'보류입니다. 실력이 없어서가 아니라, 이 과제지가 지원자의 두 가정과 맞지 않았어요. 특징들이 서로 얽혀 있고(독립 가정), 재령은 종 모양이 아니에요(분포 가정). 특징을 다듬어 오면 다시 봅니다. 회귀 과제는 분류 전용이라 치르지 않았어요.',
      key:'독립이라 치고 곱한다 — 숫자 34개, 훑기 1번, '+pct(te)+' (합격선 '+pct(PASSLINE)+')',
      render(box){box.innerHTML='<div class="bars">'+[['나이브 베이즈',te,'var(--blue)'],['매뉴얼 씨 (5-0)',PASSLINE,'var(--c-man)'],['기준선 (늘 '+PNAME[bs.k]+')',bs.acc,'var(--muted)']].map(r=>'<div class="row"><span style="text-align:left;font-family:var(--font)">'+r[0]+'</span><div class="tr"><i style="left:0;width:'+r[1]*100+'%;background:'+r[2]+'"></i></div><span>'+pct(r[1])+'</span></div>').join('')+'</div>';}};},
  alts:[
   {label:'물-결합재비 · log 재령 추가 (특징 10개)',async run(box){const f=r=>fv(r).concat([r.wb,Math.log10(r.age)]),m=gnbFit(TR.map(f),YTR,2),te=accOf(m,TE.map(f),YTE);
     box.innerHTML=statGrid([[pct(te),'시험 정확도','a'],[pct(PASSLINE),'합격선','m']],2)+'<p class="foot">매뉴얼 씨가 쓰던 두 숫자를 덧붙였더니 합격선을 넘는다. 판정: '+(te>PASSLINE?'<b>합격</b>':'보류')+'. 분포 모양이 맞는 특징을 사람이 만들어 주면 이 지원자도 일을 한다. (특징을 둘만 남기면 86.8%까지 오른다 — 05장 특징 바구니)</p>';}},
   {label:'강도 3등급 (과제 B)',async run(box){const m3=main3(),a3=accOf(m3,B,GTE),b3=GTE.filter(v=>v===1).length/GTE.length;
     box.innerHTML=statGrid([[pct(a3),'3등급 시험 정확도','a'],[pct(b3),'기준선 (늘 보통)']],2)+'<p class="foot">클래스가 셋이어도 방법은 같다(종 모양을 등급마다 하나씩). 기준선보다 '+((a3-b3)*100).toFixed(1)+'%p 낫지만, 보통 강도를 양쪽으로 흩뜨리는 실수가 많다. 3등급에는 따로 정한 합격선이 없어 판정하지 않는다.</p>';}},
   {label:'사전 확률 바꾸기 (P(합격) = 0.7)',async run(box){const m=gnbFit(A,YTR,2,[0.3,0.7]),g=B.map(m.predict),te=ML.acc(YTE,g),M=ML.confusion(YTE,g,2),M0=ML.confusion(YTE,B.map(mainNB().predict),2);
     box.innerHTML=statGrid([[pct(te),'시험 정확도','a'],[M0[0][1]+' → '+M[0][1],'미달인데 "합격" (건)','s'],[M0[1][0]+' → '+M[1][0],'합격인데 "불합격" (건)']],3)+'<p class="foot">정확도는 합격선을 '+(te>PASSLINE?'넘는다':'넘지 못한다')+'. 그러나 위험한 출하가 '+M0[0][1]+'건에서 '+M[0][1]+'건으로 늘었고, 0.7이라는 값은 근거 없이 시험지를 보며 고른 숫자다. 판정: <b>보류 유지</b>. 사전 확률은 현장의 실제 비율로 정하는 것이지, 점수를 올리려고 돌리는 손잡이가 아니다.</p>';}}]});

/* 평가 장은 화면에 가까워지면 계산한다 */
(function(){const t=$('m5');let done=false;function go(){if(done)return;done=true;setTimeout(renderEval,10);}
  if('IntersectionObserver' in window){const io=new IntersectionObserver(es=>{if(es.some(e=>e.isIntersecting)){go();io.disconnect();}},{rootMargin:'600px'});io.observe(t);io.observe($('m6'));}else go();
  window.addEventListener('hashchange',go);if(location.hash)go();})();
comic('comicDouble',[
 {bg:'lab',prop:'물 ↑ 감수제 ↓',cast:[{who:'fly',mood:'coffee',x:30},{who:'asst',mood:'board',x:72}],say:{who:'fly',txt:'물이 많다, 감수제가 적다… 이거 결국 같은 얘기 아냐?'},cap:'① 물과 고성능 감수제는 함께 움직인다'},
 {bg:'lab',prop:'×2',cast:[{who:'asst',mood:'shock',x:30,move:'shake'},{who:'doc',mood:'scale',x:72,h:104}],say:{who:'asst',txt:'따로따로 곱하면 같은 증거를 두 번 듣는 셈이에요!'},cap:'② 독립이라 치면 증거가 이중으로 잡힌다'},
 {bg:'lab',prop:'0.999?',cast:[{who:'robo',mood:'sigh',x:30,move:'tilt'},{who:'fly',mood:'cry',x:72}],say:{who:'robo',txt:'두 번 세면 확신만 두 배… 계산은 맞는데 장부가 부풀었네.'},cap:'③ 정답률보다 확률이 먼저 과장된다'}]);

/* ══ 확인 문제 ══ */
var QUIZ={
q1:[{q:'나이브 베이즈가 판정할 때 비교하는 값은?',o:[['클래스마다 사전 확률 × 특징별 가능도의 곱',1,'베이즈 정리의 분자다. 분모 P(x)는 두 클래스에 같아서 크기 비교만 하면 된다.'],['특징에 가중치를 곱해 더한 점수',0,'그것은 5-4 로지스틱 회귀의 선형 점수다. 나이브 베이즈는 분포의 높이를 곱한다.'],['가장 가까운 훈련 배합 k개의 다수결',0,'그것은 5-1 KNN이다.']]},
 {q:'"나이브(순진한)"라는 이름이 붙은 까닭은?',o:[['클래스를 알면 특징끼리 서로 독립이라고 가정해서',1,'실제로는 물과 고성능 감수제처럼 얽혀 있어도 따로 재서 곱한다. 그래서 계산이 쉽다.'],['훈련 데이터를 한 번만 봐서',0,'한 번 훑는 것은 가정의 결과이지 이름의 까닭이 아니다.'],['사전 확률을 쓰지 않아서',0,'사전 확률을 쓴다. 지원서 2번 칸.']]},
 {q:'어떤 값에서 합격 종의 높이가 불합격 종의 3배라면, 그 특징 하나는?',o:[['합격 쪽 증거다 (log 3 ≈ +1.1을 더한다)',1,'가능도 비가 1보다 크면 합격 쪽, 작으면 불합격 쪽이다. 로그를 씌우면 증거 장부의 막대 길이가 된다.'],['그 배합이 합격일 확률이 0.75라는 뜻이다',0,'사전 확률과 다른 특징의 증거를 모두 곱해야 사후 확률이 나온다.'],['아무 의미가 없다',0,'곱셈의 재료다.']]}],
q2:[{q:'이 면접에서 사전 확률 P(합격)은 어디서 오나?',o:[['훈련 배합 772건 중 합격 배합의 비율(377/772 ≈ 0.488)',1,'priors를 따로 주지 않으면 GaussianNB는 훈련 데이터의 클래스 비율을 쓴다.'],['시험지 258건의 합격 비율',0,'시험지는 학습에 쓰지 않는다.'],['늘 0.5',0,'priors=[0.5, 0.5]로 직접 정할 때만 그렇다.']]},
 {q:'나이브 베이즈에 표준화가 필요 없는 까닭은?',o:[['특징마다 따로 평균·분산을 재므로 단위를 바꿔도 가능도의 비율이 같아서',1,'kg을 톤으로 바꾸면 두 클래스의 종이 똑같이 줄어든다. 02장에서 표준화 전후 정확도가 같았다.'],['나이브 베이즈는 숫자 특징을 쓰지 않아서',0,'GaussianNB는 숫자 특징을 쓴다.'],['표준화하면 오류가 나서',0,'오류는 나지 않는다. 결과가 같을 뿐이다. (MultinomialNB는 음수가 생기면 오류가 난다.)']]}],
q3:[{q:'콘크리트 배합(kg, 일)처럼 연속값 특징에 맞는 것은?',o:[['GaussianNB',1,'클래스마다 정규 분포를 가정한다.'],['MultinomialNB',0,'단어 횟수용이다. 이 데이터에서 재령 단위를 일 → 주로 바꾸니 시험 정확도가 77.1% → 66.3%로 바뀌었다. 단위에 따라 바뀌는 점수는 믿을 수 없다.'],['BernoulliNB',0,'0/1 특징용이다.']]},
 {q:'model.predict_proba(X_test)[:, 0]은 무엇인가? (classes_ = [0, 1])',o:[['불합격(0)일 확률',1,'열 순서는 classes_ 순서다. 합격 확률은 [:, 1].'],['합격(1)일 확률',0,'흔한 실수다. 03장 16번 줄.'],['예측한 이름표',0,'이름표는 predict로 얻는다.']]}],
q4:[{q:'나이브 베이즈의 "학습"에서 일어나는 일은?',o:[['클래스별로 특징의 평균과 분산, 클래스 비율을 센다',1,'반복이 없다. 772건을 한 번 훑으면 끝난다.'],['손실이 줄도록 가중치를 수천 번 고친다',0,'그것은 5-4 로지스틱 회귀의 경사 하강이다.'],['훈련 배합을 모두 외워 둔다',0,'그것은 5-1 KNN이다. 나이브 베이즈는 숫자 34개만 남긴다.']]},
 {q:'GaussianNB의 경계 모양은?',o:[['두 클래스의 분산이 다르면 곡선(2차 곡선)이 된다',1,'두 종의 폭이 다르면 높이가 같아지는 곳이 휜다. 04장 경계 그림.'],['늘 직선이다',0,'두 클래스의 분산이 같을 때만 직선이 된다.'],['계단 모양이다',0,'계단 모양 경계는 다음 지원자 결정 트리(5-6)의 특징이다.']]},
 {q:'새 배합 100건이 더 들어왔다. 나이브 베이즈는?',o:[['장부(개수·평균·분산)만 고치면 된다 (partial_fit)',1,'웰퍼드 방법처럼 한 건씩 고칠 수 있다. 처음부터 다시 배울 필요가 없다.'],['처음부터 다시 수천 번 반복해야 한다',0,'반복 학습이 없는 지원자다.'],['새 배합은 쓸 수 없다',0,'쓸 수 있다.']]}],
q5:[{q:'이 면접에서 나이브 베이즈가 가장 많이 틀린 쪽은?',o:[['합격 배합을 불합격이라 한 쪽 (특히 재령 28일 시편)',1,'28일은 합격 종보다 폭이 좁은 불합격 종의 높이가 더 높아 "불합격 쪽 증거"로 읽혔다.'],['불합격 배합을 합격이라 한 쪽',0,'17건으로 적은 편이다.'],['둘이 똑같다',0,'58건 대 17건으로 크게 기울었다.']]},
 {q:'같은 시멘트 열을 다섯 번 넣었더니 정확도는 그대로인데 로그 손실이 크게 나빠졌다. 까닭은?',o:[['같은 증거를 독립된 증거 다섯 개로 알고 곱해 확률이 0이나 1로 쏠려서',1,'정보는 그대로인데 확신만 커진다. 얽힌 특징(물·감수제)이 실제로 하는 일이다.'],['특징이 많아져 과소적합해서',0,'특징 수가 아니라 중복이 문제다.'],['시멘트가 쓸모없는 특징이라서',0,'시멘트는 두 번째로 또렷한 특징이다(04장 장부).']]},
 {q:'사전 확률 P(합격)을 0.7로 올렸더니 시험 정확도가 합격선을 넘었다. 판정은?',o:[['보류 유지. 시험지를 보며 고른 값이고, 위험한 출하도 두 배로 늘었다',1,'사전 확률은 현장의 실제 비율로 정한다. 점수를 올리려고 돌리는 손잡이가 아니다.'],['합격. 숫자가 넘었으니까',0,'시험지로 고른 설정은 시험지가 훈련 데이터가 된 것과 같다.'],['불합격. 사전 확률은 바꿀 수 없다',0,'priors로 바꿀 수 있다. 다만 근거가 있어야 한다.']]}],
q6:[{q:'나이브 베이즈가 강도(MPa)를 맞히는 회귀를 하지 않는 까닭은?',o:[['클래스마다 분포를 재는 모델이라, 서로 다른 강도 값 845개를 클래스로 두면 한 클래스에 배합이 한두 건뿐이어서',1,'분산을 잴 수 없고, 훈련에 없던 강도는 고를 수도 없다. 구간으로 나누면(3등급) 분류가 된다.'],['계산이 너무 오래 걸려서',0,'계산은 아주 빠르다.'],['콘크리트 데이터가 작아서',0,'데이터 크기의 문제가 아니라 모델의 구조 문제다.']]},
 {q:'나이브 베이즈가 가장 잘 맞는 일은?',o:[['단어 횟수로 문서를 분류하는 일, 특징이 많고 데이터가 적은 일의 빠른 기준선',1,'특징이 수만 개여도 세기만 하면 된다. 스팸 분류의 고전이다.'],['특징끼리 강하게 얽힌 배합 데이터의 정밀한 확률',0,'독립 가정이 깨져 확률이 과장된다.'],['연속값 타겟을 맞히는 일',0,'분류 전용이다.']]}]};

/* ══ 공통 B·C·D·E·F 데이터 ══ */
var PROMPTS=[
 {t:'① 내 공정 데이터로 나이브 베이즈 코드 받기',use:'내 공장의 품질 데이터로 03장의 프롬프트를 맞출 때',p:MAIN_PROMPT},
 {t:'② AI 코드 검증 요청',use:'받은 코드를 다른 AI(또는 같은 AI의 새 대화)에게 검토받을 때',p:'아래 scikit-learn 나이브 베이즈 코드를 코드 리뷰해 줘. 특히 다음을 하나씩 확인하고, 문제가 있으면 줄 번호와 고친 코드를 보여 줘.\n1) 특징의 모양(연속값·횟수·0/1·범주)에 맞는 나이브 베이즈 클래스를 골랐나\n2) 타겟이 클래스 이름표인가 (연속값 타겟을 반올림해서 넣지 않았나)\n3) train_test_split에 stratify와 random_state가 있나\n4) predict_proba의 열을 classes_ 순서에 맞게 골랐나\n5) priors나 var_smoothing을 시험 데이터 점수로 고르지 않았나\n6) 평가를 시험 데이터로 했나, 기준선(DummyClassifier)과 비교했나\n추측으로 답하지 말고, 확실하지 않은 부분은 "공식 문서 확인 필요"라고 표시해 줘.'},
 {t:'③ 독립 가정 점검과 특징 다듬기',use:'나이브 베이즈 점수가 기대보다 낮을 때 까닭을 찾고 싶을 때',p:'GaussianNB로 〔콘크리트 35MPa 합격〕을 분류했더니 시험 정확도가 〔70.9%〕로 낮다. 특징은 〔시멘트, 고로 슬래그, 플라이 애시, 물, 고성능 감수제, 굵은 골재, 잔골재, 재령〕이다.\n다음을 하는 코드를 써 줘. 시험 데이터는 마지막 비교에만 쓰고, 모든 선택은 훈련 데이터 안의 5겹 교차 검증으로 한다.\n1) 클래스별로 특징 상관계수 행렬을 그리고 |r| > 0.5인 쌍을 출력\n2) 특징마다 히스토그램과 왜도, 0의 비율, 서로 다른 값의 수를 출력\n3) 치우친 특징에 log1p 또는 log를 씌운 버전, 얽힌 특징을 하나로 묶은 버전(예: 물-결합재비)을 만들어 교차 검증 점수 비교\n4) 가장 좋은 버전을 고른 뒤에만 시험 정확도를 한 번 출력\n각 단계에서 나이브 베이즈의 어떤 가정과 관계있는지 한 줄씩 설명해 줘.'},
 {t:'④ 수업용 문제 만들기',use:'이 편으로 수업할 때 확인 문제가 더 필요할 때',p:'나이브 베이즈 입문 수업용 문제를 만들어 줘. 대상은 〔제조 현장 실무자, 수식에 익숙하지 않음〕이다. 예시는 콘크리트 강도 합격 판정으로 통일한다.\n- 베이즈 정리·사전 확률·가능도·독립 가정을 각각 묻는 객관식 4문제 (보기 3개, 정답 해설 포함)\n- 오해를 드러내는 문제 2개: "나이브 베이즈로 강도(MPa)도 맞힐 수 있다", "정확도가 같으면 확률도 믿을 만하다"\n- 손으로 풀 수 있는 계산 문제 1개: 사전 확률 0.5, 두 특징의 가능도 비가 각각 3과 0.5일 때 사후 확률 (소수 둘째 자리)\n정답은 맨 끝에 따로 모아 줘.'}];
var GL=[
 ['모델','나이브 베이즈','클래스마다 특징의 분포를 재 두고, 베이즈 정리로 사전 확률 × 특징별 가능도를 곱해 가장 그럴듯한 클래스를 고르는 분류 모델. 특징끼리 독립이라고 가정한다.'],
 ['모델','베이즈 정리','P(y|x) = P(x|y)·P(y) / P(x). 증거를 본 뒤의 확률을, "그 클래스라면 이 증거가 나올 확률"로 거꾸로 구하는 식.'],
 ['모델','사전 확률','데이터를 보기 전에 이미 아는 클래스의 확률. 보통 훈련 데이터의 클래스 비율이다.'],
 ['모델','가능도','"클래스가 y라면 이 특징 값이 나올 만한가"를 나타내는 값. 가우시안 나이브 베이즈에서는 종 모양 곡선의 높이다.'],
 ['모델','사후 확률','증거(특징)를 본 뒤의 클래스 확률 P(y|x). 사전 확률 × 가능도에 비례한다.'],
 ['모델','조건부 독립','클래스를 알고 나면 특징 하나가 다른 특징에 대해 아무것도 알려 주지 않는다는 가정. 나이브 베이즈의 "나이브".'],
 ['모델','가우시안 분포','평균과 분산 두 숫자로 정해지는 종 모양 분포. 정규 분포라고도 한다.'],
 ['모델','가능도 비','두 클래스의 가능도를 나눈 값. 1보다 크면 분자 클래스 쪽 증거다. 로그를 씌우면 증거 점수가 된다.'],
 ['모델','생성 모델','"클래스마다 데이터가 어떻게 생겼나"를 먼저 그리고 베이즈 정리로 분류하는 모델. 나이브 베이즈가 대표다. 확률을 바로 맞추는 판별 모델(로지스틱 회귀)과 대비된다.'],
 ['모델','var_smoothing','GaussianNB가 모든 분산에 더하는 작은 값(가장 큰 특징 분산 × 1e-9). 분산이 0이 되어 계산이 터지는 것을 막는다.'],
 ['모델','라플라스 평활','횟수 기반 나이브 베이즈에서 모든 칸에 alpha를 더해, 한 번도 안 나온 값의 확률이 0이 되는 것을 막는 일.'],
 ['데이터','샘플','표의 한 줄. 여기서는 콘크리트 배합 하나(와 그 시편의 시험 결과).'],
 ['데이터','특징','모델이 입력으로 쓰는 열. 시멘트·고로 슬래그·플라이 애시·물·고성능 감수제·굵은 골재·잔골재·재령.'],
 ['데이터','타겟','맞히려는 정답 열. 이 면접에서는 35MPa 합격 여부와 강도 3등급.'],
 ['데이터','상관계수','두 특징이 함께 움직이는 정도(−1~1). 0이면 직선 관계가 없다. 나이브 가정은 클래스 안에서 이 값이 0이라고 본다.'],
 ['데이터','왜도','분포가 한쪽으로 치우친 정도. 종 모양이면 0 근처다.'],
 ['데이터','로그 변환','값에 로그를 씌워 한쪽으로 길게 늘어진 분포를 종 모양에 가깝게 펴는 일. 재령에 효과가 컸다.'],
 ['학습','웰퍼드 방법','데이터를 한 건씩 읽으며 평균과 분산을 고쳐 가는 계산법. 다 모아 두지 않아도 정확한 값이 나온다.'],
 ['학습','partial_fit','scikit-learn에서 새 데이터로 장부만 이어서 고치는 학습 방법. 나이브 베이즈가 지원한다.'],
 ['평가','정확도','맞힌 샘플의 비율. 틀림의 종류와 확률의 크기를 구별하지 않는다.'],
 ['평가','혼동행렬','정답(행)과 예측(열)을 짝지어 센 표. 어떤 클래스를 어떤 클래스로 헷갈렸는지 보인다.'],
 ['평가','정밀도','"이 클래스다"라고 한 것 중 실제로 맞은 비율.'],
 ['평가','재현율','실제 그 클래스 중 모델이 찾아낸 비율.'],
 ['평가','로그 손실','정답 클래스에 준 확률의 −log 평균. 확신하고 틀리면 크게 늘어난다. 확률이 믿을 만한지 보는 지표.'],
 ['평가','확률 보정','모델이 낸 확률이 실제 비율과 맞도록 고치는 일. CalibratedClassifierCV 등.'],
 ['평가','기준선','배우지 않고도 낼 수 있는 점수. "늘 다수 쪽"이라고 답하는 모델 등.']];
var ANALOGY=[
 ['사전 확률','"이 공장 배합은 원래 절반쯤 합격한다"는 경험. 시험 성적을 보기 전부터 가진 기대',2,'사람의 지각이 사전 기대와 감각 증거를 함께 써서 판단한다는 "베이즈 뇌" 가설','행동이 베이즈 계산과 맞아떨어지는 실험은 많지만, 뉴런이 사전 확률을 어떻게 담는지는 아직 논쟁 중이다'],
 ['가능도 곱하기','두 면접관이 따로 매긴 점수를 합친다. 더 믿을 만한 면접관의 말에 무게를 더 준다',2,'사람이 눈과 손으로 같은 물체 크기를 잴 때, 각 감각의 신뢰도에 따라 무게를 두어 합친다(감각 통합)','이 실험은 두 감각의 잡음이 서로 독립일 때의 최적 결합과 맞았다. 뇌가 늘 독립을 가정한다는 뜻은 아니다'],
 ['나이브(독립) 가정','이력서의 학점과 자격증을 서로 모르는 정보로 치고 둘 다 가산점을 준다. 사실 둘은 함께 움직이는데',1,'','뇌가 얽힌 단서를 어떻게 처리하는지는 연구가 진행 중이며, 나이브 베이즈와 같은 방식이라는 근거는 알려지지 않았다'],
 ['학습 = 세기','신입이 첫 달 동안 합격·불합격 배합의 평균을 장부에 적어 두는 것',1,'','초파리 버섯체의 냄새 학습은 시냅스 세기를 바꾸는 방식이다. 평균·분산을 세는 장부에 대응하는 회로는 알려지지 않았다'],
 ['같은 증거 두 번 세기','같은 소문을 두 사람에게 들었다고 두 배로 믿는 것',1,'','사람 판단에서도 비슷한 과신이 보고되지만, 신경 수준의 대응은 확인되지 않았다'],
 ['사전 확률 바꾸기','희귀병 검사: 양성이 나와도 원래 드문 병이면 진짜일 확률이 낮다',1,'','사람이 기저율을 무시하는 경향(기저율 무시)은 행동 연구에서 알려졌지만, 초파리·뇌 회로와의 대응은 이 페이지에서 다루지 않는다']];
var REFS=[
 'Yeh, I.-C. (1998). Modeling of strength of high-performance concrete using artificial neural networks. <i>Cement and Concrete Research</i>, 28(12), 1797–1808. <a href="https://doi.org/10.1016/S0008-8846(98)00165-3" target="_blank" rel="noopener">doi:10.1016/S0008-8846(98)00165-3</a> — 실기 과제지 원출처',
 'Yeh, I.-C. (2007). Concrete Compressive Strength [Dataset]. UCI Machine Learning Repository. <a href="https://doi.org/10.24432/C5PK67" target="_blank" rel="noopener">doi:10.24432/C5PK67</a> (CC BY 4.0) — 02장 데이터',
 'Bayes, T. (1763). An essay towards solving a problem in the doctrine of chances. <i>Philosophical Transactions of the Royal Society of London</i>, 53, 370–418. <a href="https://doi.org/10.1098/rstl.1763.0053" target="_blank" rel="noopener">doi:10.1098/rstl.1763.0053</a> — 베이즈 정리, 01장',
 'Domingos, P., &amp; Pazzani, M. (1997). On the optimality of the simple Bayesian classifier under zero-one loss. <i>Machine Learning</i>, 29, 103–130. <a href="https://doi.org/10.1023/A:1007413511361" target="_blank" rel="noopener">doi:10.1023/A:1007413511361</a> — 독립 가정이 깨져도 분류는 맞을 수 있는 까닭, 05장',
 'Hand, D. J., &amp; Yu, K. (2001). Idiot\'s Bayes — not so stupid after all? <i>International Statistical Review</i>, 69(3), 385–398. <a href="https://doi.org/10.1111/j.1751-5823.2001.tb00465.x" target="_blank" rel="noopener">doi:10.1111/j.1751-5823.2001.tb00465.x</a> — 나이브 베이즈의 강점과 한계, 06장',
 'Niculescu-Mizil, A., &amp; Caruana, R. (2005). Predicting good probabilities with supervised learning. <i>Proceedings of the 22nd International Conference on Machine Learning (ICML)</i>, 625–632. — 나이브 베이즈 확률이 0·1로 쏠리는 현상과 보정, 05·06장',
 'Rennie, J. D. M., Shih, L., Teevan, J., &amp; Karger, D. R. (2003). Tackling the poor assumptions of naive Bayes text classifiers. <i>Proceedings of the 20th International Conference on Machine Learning (ICML)</i>, 616–623. — ComplementNB, 06장',
 'Welford, B. P. (1962). Note on a method for calculating corrected sums of squares and products. <i>Technometrics</i>, 4(3), 419–420. <a href="https://doi.org/10.1080/00401706.1962.10490022" target="_blank" rel="noopener">doi:10.1080/00401706.1962.10490022</a> — 한 건씩 평균·분산 세기, 04장',
 'Hastie, T., Tibshirani, R., &amp; Friedman, J. (2009). <i>The Elements of Statistical Learning</i> (2nd ed.), 6.6.3절. Springer. — 나이브 베이즈와 생성 모델',
 'Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. <i>Journal of Machine Learning Research</i>, 12, 2825–2830. — A장 코드',
 'scikit-learn 문서. <a href="https://scikit-learn.org/stable/modules/naive_bayes.html" target="_blank" rel="noopener">Naive Bayes</a> — GaussianNB·MultinomialNB·ComplementNB·BernoulliNB·CategoricalNB, var_smoothing, force_alpha, 03·06장',
 'Ernst, M. O., &amp; Banks, M. S. (2002). Humans integrate visual and haptic information in a statistically optimal fashion. <i>Nature</i>, 415, 429–433. <a href="https://doi.org/10.1038/415429a" target="_blank" rel="noopener">doi:10.1038/415429a</a> — D장 감각 통합 비유',
 'Knill, D. C., &amp; Pouget, A. (2004). The Bayesian brain: the role of uncertainty in neural coding and computation. <i>Trends in Neurosciences</i>, 27(12), 712–719. <a href="https://doi.org/10.1016/j.tins.2004.10.007" target="_blank" rel="noopener">doi:10.1016/j.tins.2004.10.007</a> — D장 베이즈 뇌 비유'];
var TIP={who:'머신러닝을 처음 배우는 대학생·직장인·제조 현장 실무자. 5-0 채용 공고와 5-4 로지스틱 회귀를 먼저 보면 좋다. 수식은 베이즈 정리와 종 모양 곡선 두 개만 나온다',
 goal:'① 사전 확률 × 가능도 → 사후 확률의 흐름을 지원서 1장으로 설명한다 ② 학습이 "평균·분산 세기"임을 말한다 ③ AI가 낸 코드에서 나이브 베이즈 종류·열 순서·회귀 흉내 실수를 찾는다 ④ 독립 가정과 분포 가정이 깨지는 곳을 데이터에서 찾아 판정의 까닭을 설명한다',
 time:'읽기 25분 · 실습(01·04·05장 그림) 20분 · 틀린 곳 찾기 10분 · Colab 15분',
 plans:[['강의형 50분','00장 지원서 1장을 인쇄해 나눠 주고 9칸을 함께 읽기 10분 → 01장 종 모양 그림에서 재령·log 재령 비교 10분 → 04장 수습 재생으로 "학습 = 세기" 보기 5분 → 05장 까닭 ①·② 10분 → 면접 결과를 학생들이 먼저 맞히고 발표, 다시 보기 세 개 15분.'],['실무 OJT 90분','03장 프롬프트로 각자 AI에게 코드를 받는다 20분 → 체크리스트로 서로의 코드를 바꿔 검사(특히 나이브 베이즈 종류) 20분 → Colab에서 실행하고 A장 결과와 비교 20분 → B장 ③ 프롬프트로 "우리 공장 데이터"의 얽힌 특징과 치우친 분포를 찾아 발표 30분.'],['1:1 튜터링','01장 증거 장부에서 배합 하나를 골라 막대 여덟 개를 손으로 더해 보게 한다. 그다음 04장 장부 표의 평균·표준편차로 막대 하나를 다시 계산해 본다.']],
 myths:[['나이브 베이즈는 특징이 독립일 때만 쓸 수 있다','가정이 깨져도 분류는 꽤 맞는 경우가 많다(Domingos & Pazzani, 1997). 다만 확률의 크기는 과장된다. 이 면접에서는 분포 모양까지 어긋나 합격선을 못 넘었다'],['나이브 베이즈로 회귀도 된다','분류 전용이다. 연속값 타겟을 반올림해 넣으면 돌지만 회귀가 아니다'],['연속값이면 어느 나이브 베이즈든 괜찮다','MultinomialNB는 단위만 바꿔도 점수가 바뀐다(77.1% → 66.3%). 연속값에는 GaussianNB'],['정확도가 비슷하면 확률도 믿을 만하다','같은 시멘트를 다섯 번 세도 정확도는 그대로였지만 로그 손실은 두 배 넘게 나빠졌다'],['특징은 많을수록 좋다','물-결합재비와 log 재령 두 개만 남긴 쪽(86.8%)이 열 개(81.0%)보다 나았다'],['사전 확률은 점수가 잘 나오게 고르는 값이다','현장의 실제 비율로 정한다. 시험지를 보며 고르면 반칙이다']],
 tasks:[['과제 1 · 증거 장부 손 계산','01장에서 배합 하나를 고르고, 04장 장부 표의 μ·σ로 시멘트와 재령 두 특징의 가능도 비를 계산기로 구해 막대 길이와 비교한다.'],['과제 2 · 틀린 곳 찾기 만들기','AI에게 일부러 실수가 섞인 나이브 베이즈 코드를 받고(B장 ②를 거꾸로 활용), 짝에게 찾게 한다. 실수마다 "왜 틀렸나"를 한 줄로 쓴다.'],['과제 3 · 재면접 준비','B장 ③ 프롬프트로 특징을 다듬은 버전을 교차 검증으로 고르고, 시험 정확도를 한 번만 재서 "재면접 결과 보고서"를 반 쪽으로 쓴다.'],['과제 4 · 다른 데이터에서 다시 보기','scikit-learn의 load_breast_cancer나 문서 분류 예제(20 Newsgroups)에 GaussianNB·MultinomialNB를 써 보고, 콘크리트보다 잘 맞거나 못 맞는 까닭을 독립·분포 가정으로 설명한다.']],
 ai:'학생이 AI에게 코드를 받는 것은 막지 말고 과정에 넣는다. 다만 ① 프롬프트를 제출하게 하고 ② 체크리스트 아홉 항목을 직접 표시하게 하고 ③ 실행 결과를 기준선·합격선과 비교한 문장을 쓰게 하면, "받아 적기"가 아니라 "검증하기"를 배운다. 나이브 베이즈처럼 세 줄로 도는 모델일수록 "돌아간다 = 맞다"라는 착각이 크다. 모델 종류와 데이터 모양이 맞는지 먼저 묻게 한다.'};
var CODE={title:'나이브 베이즈 면접 — 베이즈 정리부터 보류 판정까지',sub:'numpy, pandas, scipy, scikit-learn, ucimlrepo · CPU 몇 초',file:'naive_bayes_interview',
 desc:'이 편의 면접을 그대로 옮겼다. UCI에서 콘크리트 데이터를 받아 시험지를 봉인하고(1·2번 칸), GaussianNB로 평균·분산을 세고(3번), 베이즈 정리를 손으로 다시 계산해 predict_proba와 맞춰 보고(4번), 혼동행렬과 28일 시편(5번), 표준화 무관과 MultinomialNB의 단위 문제(6번), 얽힌 특징과 같은 증거 여러 번 세기(7번), 재령 분포와 특징 다듬기(8번), 3등급(9번), 사전 확률(10번), 회귀를 못 하는 까닭(11번), 판정(12번)까지 간다. 노트북으로 저장하면 칸마다 셀이 나뉜다.',
 cap:'실행 결과 (scikit-learn 1.9, CPU). 브라우저 계산과 같은 숫자(시험 70.9% · 훈련 74.1%, 놓친 합격 배합 58건 중 28일 시편 34건, 특징 10개 81.0%, 3등급 58.1%)가 나온다. GaussianNB는 반복 없이 닫힌 식으로 계산하므로 브라우저와 소수점까지 같다.'};
