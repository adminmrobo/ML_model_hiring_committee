var BOARD_LINE='R² 0.735'; /* 회귀 합격선(매뉴얼 경험식 R²) */
/* ══ 5-2 선형 회귀 · 콘크리트 실기 과제지 ══════════════════════════════ */
/* 공용 편 목록(src/core.js)에서 5-2가 아직 "면접 예정"이면 이 페이지 안에서만 준비된 편으로 둔다 */
(function(){const i=epIdx('5-2');if(i>=0&&!EPISODES[i].ready)EPISODES[i].ready=1;})();

/* ── 계산 엔진: 선형 회귀 (정규방정식 · 경사 하강) ── */
const LR=(function(){
  /* 가우스 소거(부분 피벗)로 A·x = b를 푼다 */
  function solve(A,b){const n=b.length,M=A.map((r,i)=>r.concat([b[i]]));
    for(let c=0;c<n;c++){let p=c;for(let r=c+1;r<n;r++)if(Math.abs(M[r][c])>Math.abs(M[p][c]))p=r;[M[c],M[p]]=[M[p],M[c]];
      for(let r=c+1;r<n;r++){const f=M[r][c]/M[c][c];if(!f)continue;for(let k=c;k<=n;k++)M[r][k]-=f*M[c][k];}}
    const x=new Array(n).fill(0);for(let r=n-1;r>=0;r--){let s=M[r][n];for(let k=r+1;k<n;k++)s-=M[r][k]*x[k];x[r]=s/M[r][r];}return x;}
  function model(w,b){return {w:w,b:b,predict:x=>{let s=b;for(let j=0;j<w.length;j++)s+=w[j]*x[j];return s;}};}
  /* 정규방정식: (XᵀX)β = Xᵀy. X 맨 앞에 1(절편) 열을 붙인다 */
  function fit(X,y){const d=X[0].length+1,G=[...Array(d)].map(()=>new Array(d).fill(0)),h=new Array(d).fill(0);
    X.forEach((r,i)=>{const z=[1].concat(r);for(let a=0;a<d;a++){h[a]+=z[a]*y[i];for(let c=a;c<d;c++)G[a][c]+=z[a]*z[c];}});
    for(let a=0;a<d;a++)for(let c=0;c<a;c++)G[a][c]=G[c][a];
    const be=solve(G,h);return model(be.slice(1),be[0]);}
  /* 경사 하강 한 걸음: 손실 = 평균 제곱 오차(MSE). 기울기 = −2·평균[(y − ŷ)·x] */
  function gdStep(st,X,y,lr){const n=X.length,d=st.w.length,gw=new Array(d).fill(0);let gb=0,L=0;
    for(let i=0;i<n;i++){let p=st.b;for(let j=0;j<d;j++)p+=st.w[j]*X[i][j];const e=y[i]-p;L+=e*e;gb+=-2*e/n;for(let j=0;j<d;j++)gw[j]+=-2*e*X[i][j]/n;}
    for(let j=0;j<d;j++)st.w[j]-=lr*gw[j];st.b-=lr*gb;st.t=(st.t||0)+1;return L/n;}
  function mse(st,X,y){let L=0;for(let i=0;i<X.length;i++){let p=st.b;for(let j=0;j<st.w.length;j++)p+=st.w[j]*X[i][j];L+=(y[i]-p)*(y[i]-p);}return L/X.length;}
  function r2(yt,yp){const m=ML.mean(yt);let a=0,b=0;for(let i=0;i<yt.length;i++){a+=(yt[i]-yp[i])**2;b+=(yt[i]-m)**2;}return 1-a/b;}
  function mae(yt,yp){let s=0;for(let i=0;i<yt.length;i++)s+=Math.abs(yt[i]-yp[i]);return s/yt.length;}
  function rmse(yt,yp){let s=0;for(let i=0;i<yt.length;i++)s+=(yt[i]-yp[i])**2;return Math.sqrt(s/yt.length);}
  return {solve,fit,model,gdStep,mse,r2,mae,rmse};})();

const ROWS=concRows(),TR=ROWS.filter(r=>!r.test),TE=ROWS.filter(r=>r.test);
const fv=r=>FEAT.map(f=>r[f]);
const YTR=TR.map(r=>r.strength),YTE=TE.map(r=>r.strength),YMEAN=ML.mean(YTR);
/* 매뉴얼 씨의 현장 경험식 (학습 없음): 에이브럼스 식(물-결합재비) × ACI 209R 재령 보정 */
const manual=r=>96.5/Math.pow(7,r.wb)*r.age/(4+0.85*r.age);
const manualWC=r=>96.5/Math.pow(7,r.water/r.cement)*r.age/(4+0.85*r.age);
const MAN={te:TE.map(manual),tr:TR.map(manual)};
MAN.r2=LR.r2(YTE,MAN.te);MAN.mae=LR.mae(YTE,MAN.te);MAN.rmse=LR.rmse(YTE,MAN.te);
const BASE={te:TE.map(()=>YMEAN)};BASE.r2=LR.r2(YTE,BASE.te);BASE.mae=LR.mae(YTE,BASE.te);BASE.rmse=LR.rmse(YTE,BASE.te);
/* 특징 묶음 */
const FS={
  m8:{lab:'특징 8개 그대로',names:FEAT.map(f=>FEATK[f]),f:fv},
  wb:{lab:'8개 + 물-결합재비',names:FEAT.map(f=>FEATK[f]).concat(['물-결합재비']),f:r=>fv(r).concat([r.wb])},
  la:{lab:'8개 + log(재령)',names:FEAT.map(f=>FEATK[f]).concat(['log(재령)']),f:r=>fv(r).concat([Math.log(r.age)])},
  m10:{lab:'8개 + 물-결합재비 + log(재령)',names:FEAT.map(f=>FEATK[f]).concat(['물-결합재비','log(재령)']),f:r=>fv(r).concat([r.wb,Math.log(r.age)])},
  two:{lab:'물-결합재비 · log(재령) 둘만',names:['물-결합재비','log(재령)'],f:r=>[r.wb,Math.log(r.age)]}};
const FIT={};
function fitSet(k,raw){const key=k+(raw?'_raw':'');if(FIT[key])return FIT[key];const S=FS[k],XT=TR.map(S.f),XE=TE.map(S.f),sc=ML.scaler(XT),A=raw?XT:sc.t(XT),B=raw?XE:sc.t(XE);
  const t0=performance.now(),m=LR.fit(A,YTR),ms=performance.now()-t0,ptr=A.map(m.predict),pte=B.map(m.predict);
  return FIT[key]={k,m,sc,A,B,ms,ptr,pte,r2:LR.r2(YTE,pte),mae:LR.mae(YTE,pte),rmse:LR.rmse(YTE,pte),trR2:LR.r2(YTR,ptr),trMae:LR.mae(YTR,ptr)};}
const passOf=r2=>r2>MAN.r2&&r2>BASE.r2;
/* ── 계산 끝 ── */

const f3=v=>fx(v,3),f2=v=>fx(v,2);
$('heroLine').textContent=f3(MAN.r2);$('heroMae').textContent=f2(MAN.mae);

/* ── 히어로 ── */
heroRoom($('heroCv'),{sym:'β',no:'APP-02',name:'선형\n회귀',role:'지원자 · 회귀팀',banner:'모델 채용 위원회 · 5-2 면접',doc:'front',asst:'board',fly:'clip',robo:'calc',loop:10,
  beats:[{"at": 0.4, "who": "doc", "mood": "board", "move": "nod", "say": "직선 하나 긋기"}, {"at": 2.4, "who": "fly", "mood": "point", "move": "shake", "say": "콘크리트는 곡선이야"}, {"at": 4.4, "who": "robo", "mood": "happy", "move": "bob", "say": "숫자 9개면 끝?"}, {"at": 6.4, "who": "asst", "mood": "sigh", "move": "sway", "say": "3일 시편이 삐져요"}, {"at": 8.4, "who": "doc", "mood": "scale", "move": "stand"}],
  motif(ctx,t,W,H,C){const P=[[280,150],[292,138],[304,146],[316,128],[328,132],[340,116],[352,122],[364,104],[376,110],[388,96]];
    P.forEach(p=>circ(ctx,p[0],p[1],3.2,C.text,null));const a=-0.78+(REDUCE?0:Math.sin(t*0.9)*0.08);
    ctx.strokeStyle=C.c_app;ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(272,152);ctx.lineTo(396,152+a*124);ctx.stroke();
    ctx.fillStyle=C.muted;ctx.font=font(12,500,1);ctx.textAlign='left';ctx.fillText('ŷ = w·x + b',300,172);}});

/* ── 01 장면 ── */
comic('comicIntro',[{"bg": "board", "prop": "y = Xβ", "cast": [{"who": "doc", "mood": "board", "x": 30, "move": "nod"}, {"who": "asst", "mood": "ledger", "x": 72}], "say": {"who": "doc", "txt": "점들 사이에 직선 하나. 벗어난 만큼 제곱해 더한 값이 가장 작은 직선이에요."}, "cap": "① 최소제곱 — 잔차 제곱의 합을 가장 작게"}, {"bg": "lab", "prop": "3일 · 28일", "cast": [{"who": "fly", "mood": "point", "x": 30, "move": "shake"}, {"who": "asst", "mood": "shock", "x": 72, "move": "pop"}], "say": {"who": "fly", "txt": "3일짜리 시편까지 직선으로 잇는다고? 강도는 처음에 확 오르고 나중엔 느려."}, "cap": "② 직선의 한계 — 재령과 강도는 곡선이다"}, {"bg": "office", "prop": "9", "cast": [{"who": "robo", "mood": "happy", "x": 30, "move": "jump"}, {"who": "doc", "mood": "scale", "x": 72, "move": "tilt"}], "say": {"who": "robo", "txt": "숫자 아홉 개에 학습은 방정식 한 번 풀기. 싸긴 진짜 싸다!"}, "cap": "③ 값 — 가장 싼 직원, 대신 직선만 안다"}]);

/* ── 01 직선 직접 맞추기 ── */
(function(){const cv=$('fitCv'),H=320,LAB={wb:'물-결합재비',cement:'시멘트 (kg/m³)',age:'재령 (일)'};
  const st={k:'wb',w:0,h:YMEAN,sq:false,sel:-1,anim:0};let D=null;
  function setK(k){st.k=k;const xs=TR.map(r=>r[k]),mx=ML.mean(xs);let sxy=0,sxx=0;xs.forEach((x,i)=>{sxy+=(x-mx)*(YTR[i]-YMEAN);sxx+=(x-mx)*(x-mx);});
    const wO=sxy/sxx,lo=Math.min(...xs),hi=Math.max(...xs),pad=(hi-lo)*0.04;D={xs,mx,wO,x0:lo-pad,x1:hi+pad,step:Math.abs(wO)/200};
    const s=$('fitW');s.min=-2.5*Math.abs(wO);s.max=2.5*Math.abs(wO);s.step=D.step;st.w=0;st.h=YMEAN;st.sel=-1;sync();draw();}
  const dig=()=>Math.abs(D.wO)<0.5?3:1;
  const bOf=()=>st.h-st.w*D.mx,pred=x=>st.h+st.w*(x-D.mx);
  function stats(){let s=0,a=0,t=0;D.xs.forEach((x,i)=>{const e=YTR[i]-pred(x);s+=e*e;a+=Math.abs(e);t+=(YTR[i]-YMEAN)**2;});const n=D.xs.length;return {mse:s/n,mae:a/n,r2:1-s/t};}
  function bestMse(){let s=0;D.xs.forEach((x,i)=>{const e=YTR[i]-(YMEAN+D.wO*(x-D.mx));s+=e*e;});return s/D.xs.length;}
  function draw(){const W=fit(cv,H),ctx=cv.getContext('2d'),C=cols(),pl=50,pr=12,pt=12,pb=40,gw=W-pl-pr,gh=H-pt-pb,Y0=0,Y1=85;
    const sx=x=>pl+(x-D.x0)/(D.x1-D.x0)*gw,sy=y=>pt+(1-(y-Y0)/(Y1-Y0))*gh;cv._sx=sx;cv._sy=sy;
    ctx.clearRect(0,0,W,H);ctx.strokeStyle=C.rule;ctx.lineWidth=1;ctx.fillStyle=C.muted;ctx.font=font(12);
    for(let k=0;k<=5;k++){const gx=D.x0+(D.x1-D.x0)*k/5;ctx.beginPath();ctx.moveTo(sx(gx),pt);ctx.lineTo(sx(gx),pt+gh);ctx.stroke();ctx.textAlign='center';ctx.fillText(gx.toFixed(gx<5?2:0),sx(gx),H-22);}
    [0,20,40,60,80].forEach(v=>{ctx.beginPath();ctx.moveTo(pl,sy(v));ctx.lineTo(pl+gw,sy(v));ctx.stroke();ctx.textAlign='right';ctx.fillText(v,pl-6,sy(v)+4);});
    ctx.textAlign='center';ctx.fillText(LAB[st.k],pl+gw/2,H-4);ctx.save();ctx.translate(12,pt+gh/2);ctx.rotate(-Math.PI/2);ctx.fillText('압축 강도 (MPa)',0,0);ctx.restore();
    ctx.save();ctx.beginPath();ctx.rect(pl,pt,gw,gh);ctx.clip();
    if(st.sq){D.xs.forEach((x,i)=>{if(i%19)return;const y=YTR[i],yh=pred(x),s=Math.abs(sy(y)-sy(yh));ctx.fillStyle=rgba(C.blue,.10);ctx.strokeStyle=rgba(C.blue,.55);ctx.lineWidth=1;ctx.fillRect(sx(x),Math.min(sy(y),sy(yh)),s,s);ctx.strokeRect(sx(x),Math.min(sy(y),sy(yh)),s,s);});}
    ctx.strokeStyle=rgba(C.text,.16);ctx.lineWidth=1;ctx.beginPath();D.xs.forEach((x,i)=>{ctx.moveTo(sx(x),sy(YTR[i]));ctx.lineTo(sx(x),sy(pred(x)));});ctx.stroke();
    D.xs.forEach((x,i)=>circ(ctx,sx(x),sy(YTR[i]),2.6,rgba(C.text,.55),null));
    ctx.setLineDash([5,4]);ctx.strokeStyle=C.muted;ctx.lineWidth=1.2;ctx.beginPath();ctx.moveTo(pl,sy(YMEAN));ctx.lineTo(pl+gw,sy(YMEAN));ctx.stroke();ctx.setLineDash([]);
    ctx.strokeStyle=C.blue;ctx.lineWidth=3.5;ctx.beginPath();ctx.moveTo(sx(D.x0),sy(pred(D.x0)));ctx.lineTo(sx(D.x1),sy(pred(D.x1)));ctx.stroke();
    if(st.sel>=0){const x=D.xs[st.sel],y=YTR[st.sel];ctx.strokeStyle=C.stamp;ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(sx(x),sy(y));ctx.lineTo(sx(x),sy(pred(x)));ctx.stroke();circ(ctx,sx(x),sy(y),6,C.stamp,C.panel,2);circ(ctx,sx(x),sy(pred(x)),5,C.panel,C.blue,2.5);}
    ctx.restore();
    ctx.fillStyle=C.muted;ctx.font=font(12);ctx.textAlign='right';ctx.fillText('평균선 '+YMEAN.toFixed(1)+'MPa (기준선)',pl+gw-6,sy(YMEAN)-6);info();}
  function info(){const s=stats(),b=bOf(),best=bestMse(),d=dig();
    let h='ŷ = <b>'+fx(st.w,d)+'</b> × '+LAB[st.k].replace(/ \(.*\)/,'')+' '+(b<0?'− ':'+ ')+Math.abs(b).toFixed(1)+' &nbsp;→ 훈련 772건 MSE <b>'+s.mse.toFixed(1)+'</b> (RMSE '+Math.sqrt(s.mse).toFixed(2)+'MPa) · MAE <b>'+s.mae.toFixed(2)+'MPa</b> · R² <b>'+f3(s.r2)+'</b>. ';
    h+=Math.abs(s.mse-best)<0.05?'<b style="color:var(--ok)">이 특징 하나로 그을 수 있는 가장 좋은 직선이다(최소제곱).</b>':'최소제곱 직선의 MSE는 '+best.toFixed(1)+'이다. 아직 '+(s.mse-best).toFixed(1)+' 더 줄일 수 있다.';
    if(st.sel>=0){const r=TR[st.sel],x=D.xs[st.sel],e=YTR[st.sel]-pred(x);h+='<br>고른 배합: 실제 <b>'+r.strength+'MPa</b>, 예측 '+pred(x).toFixed(1)+'MPa → 잔차 e = '+(e>=0?'+':'−')+Math.abs(e).toFixed(1)+'MPa ('+(e>=0?'덜 예측':'더 예측')+'), 제곱 '+(e*e).toFixed(0)+'. 재령 '+r.age+'일 · 물-결합재비 '+r.wb.toFixed(2)+'.';}
    else h+='<br>그림을 누르면 가까운 배합 하나의 잔차를 잰다.';
    if(st.k==='age'&&Math.abs(s.mse-best)<0.05)h+='<br><span style="color:var(--amber)">재령은 직선이 곡선을 못 따라간다. 초기 재령과 1년 근처에서는 직선이 점보다 위, 28~90일에서는 아래로 지나간다.</span>';
    $('fitInfo').innerHTML=h;}
  function sync(){const d=dig();$('fitW').value=st.w;$('fitWO').textContent=fx(st.w,d);$('fitH').value=st.h;$('fitHO').textContent=st.h.toFixed(1);}
  $('fitW').addEventListener('input',e=>{cancelAnimationFrame(st.anim);st.w=+e.target.value;sync();draw();});
  $('fitH').addEventListener('input',e=>{cancelAnimationFrame(st.anim);st.h=+e.target.value;sync();draw();});
  $('fitSq').addEventListener('click',e=>{st.sq=!st.sq;e.target.setAttribute('aria-pressed',st.sq);draw();});
  $('fitReset').addEventListener('click',()=>{cancelAnimationFrame(st.anim);st.w=0;st.h=YMEAN;st.sel=-1;sync();draw();});
  $('fitLs').addEventListener('click',()=>{cancelAnimationFrame(st.anim);const w0=st.w,h0=st.h,N=REDUCE?1:36;let k=0;
    (function go(){k++;const t=ease(k/N);st.w=w0+(D.wO-w0)*t;st.h=h0+(YMEAN-h0)*t;sync();draw();if(k<N)st.anim=requestAnimationFrame(go);})();});
  cv.addEventListener('click',e=>{const r=cv.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top;let best=-1,bd=400;D.xs.forEach((x,i)=>{const d=(cv._sx(x)-mx)**2+(cv._sy(YTR[i])-my)**2;if(d<bd){bd=d;best=i;}});st.sel=best;draw();});
  seg('#fitFeat button',v=>{cancelAnimationFrame(st.anim);setK(v);});
  setK('wb');onRedraw(draw);})();

/* ── 02 실기 과제지 ── */
(function(){const xs=$('dxSel'),ys=$('dySel'),OPT=FEAT.concat(['wb','strength']),LAB=Object.assign({},FEATK,{wb:'물-결합재비',strength:'압축 강도(MPa)'});
  OPT.forEach(f=>{xs.add(new Option(LAB[f],f));ys.add(new Option(LAB[f],f));});xs.value='age';ys.value='strength';
  const cv=$('dataCv'),H=340;
  function draw(){const W=fit(cv,H),ctx=cv.getContext('2d'),C=cols(),fx_=xs.value,fy=ys.value,pl=52,pr=14,pt=14,pb=40,gw=W-pl-pr,gh=H-pt-pb;
    const xv=ROWS.map(r=>r[fx_]),yv=ROWS.map(r=>r[fy]),x0=Math.min(0,...xv),x1=Math.max(...xv)*1.04,y0=Math.min(0,...yv),y1=Math.max(...yv)*1.05,sx=x=>pl+(x-x0)/(x1-x0)*gw,sy=y=>pt+(1-(y-y0)/(y1-y0))*gh;
    ctx.clearRect(0,0,W,H);ctx.strokeStyle=C.rule;ctx.fillStyle=C.muted;ctx.font=font(12);
    for(let k=0;k<=5;k++){const gx=x0+(x1-x0)*k/5,gy=y0+(y1-y0)*k/5;ctx.beginPath();ctx.moveTo(sx(gx),pt);ctx.lineTo(sx(gx),pt+gh);ctx.moveTo(pl,sy(gy));ctx.lineTo(pl+gw,sy(gy));ctx.stroke();ctx.textAlign='center';ctx.fillText(gx.toFixed(gx<5?2:0),sx(gx),H-22);ctx.textAlign='right';ctx.fillText(gy.toFixed(gy<5?2:0),pl-6,sy(gy)+4);}
    ctx.textAlign='center';ctx.fillText(LAB[fx_],pl+gw/2,H-4);ctx.save();ctx.translate(12,pt+gh/2);ctx.rotate(-Math.PI/2);ctx.fillText(LAB[fy],0,0);ctx.restore();
    ROWS.forEach(r=>{const c=GRADECOL[r.grade];if(r.test)circ(ctx,sx(r[fx_]),sy(r[fy]),3.6,C.panel,c,1.6);else circ(ctx,sx(r[fx_]),sy(r[fy]),3,rgba(c,.75),null);});
    if(fx_==='age'&&fy==='strength'){ctx.fillStyle=C.stamp;ctx.font=font(12.5,700);ctx.textAlign='left';ctx.fillText('처음 한 달은 가파르게, 그 뒤로는 천천히 오른다 (곡선)',sx(40),sy(y1*0.93));}}
  xs.addEventListener('change',draw);ys.addEventListener('change',draw);draw();onRedraw(draw);
  const cnt=[0,0,0],tst=[0,0,0];ROWS.forEach(r=>{cnt[r.grade]++;if(r.test)tst[r.grade]++;});
  $('dataLeg').innerHTML=GRADES.map((g,k)=>'<span><i style="background:'+GRADECOL[k]+'"></i>'+g+' ('+(k===0?'25MPa 미만':k===1?'25~45':'45 이상')+') '+cnt[k]+'건 · 시험 '+tst[k]+'</span>').join('')+'<span>○ 속이 빈 점 = 봉인된 시험지</span>';
  $('concTbl').innerHTML='<table><thead><tr><th>#</th>'+FEAT.map(f=>'<th><span class="tg x">특징 X</span>'+FEATK[f]+'</th>').join('')+'<th>물-결합재비<br>(파생)</th><th><span class="tg t">타겟 y</span>압축 강도 (MPa)</th><th>용도</th></tr></thead><tbody>'+
    ROWS.map(r=>'<tr><td>'+(r.i+1)+'</td>'+FEAT.map(f=>'<td>'+r[f]+'</td>').join('')+'<td>'+r.wb.toFixed(3)+'</td><td class="y">'+r.strength+'</td><td>'+(r.test?'<span class="pill no">봉인</span>':'훈련')+'</td></tr>').join('')+'</tbody></table>';
  const SC=ML.scaler(TR.map(fv)),r0=TR[0],z0=SC.t([fv(r0)])[0];
  $('stdTbl').innerHTML='<thead><tr><th>특징</th><th>평균</th><th>표준편차</th><th>예: 훈련 첫 배합</th></tr></thead><tbody>'+FEAT.map((f,j)=>'<tr><td>'+FEATK[f]+'</td><td>'+SC.mu[j].toFixed(1)+'</td><td>'+SC.sd[j].toFixed(1)+'</td><td>'+r0[f]+' → <b>'+fx(z0[j])+'</b></td></tr>').join('')+'</tbody>';
  const sd=Math.sqrt(ML.mean(YTR.map(v=>(v-YMEAN)**2)));
  $('yMean').textContent=YMEAN.toFixed(1);$('ySd').textContent=sd.toFixed(1);$('baseMae').textContent=f2(BASE.mae);
  $('manR2').textContent=f3(MAN.r2);$('manMae').textContent=f2(MAN.mae);$('manWc').textContent=f3(LR.r2(YTE,TE.map(manualWC)));})();

/* ── 03 AI 비서 ── */
const MAIN_PROMPT=`너는 머신러닝 수업 조교다. Google Colab에서 바로 실행되는 파이썬 코드를 써 줘.

[환경] Colab 기본 라이브러리(numpy, pandas, scikit-learn)와 ucimlrepo만 쓴다. 첫 줄에 scikit-learn 버전을 출력한다.
[데이터] 〔ucimlrepo의 fetch_ucirepo(id=165)〕 로 콘크리트 압축 강도 데이터를 읽는다.
  - 특징 X: 〔시멘트, 고로 슬래그, 플라이 애시, 물, 고성능 감수제, 굵은 골재, 잔골재, 재령〕 8개 (열 순서대로 이름을 붙인다)
  - 타겟 y: 〔압축 강도(MPa)〕 (회귀). 타겟 열이 X에 섞이지 않았는지 assert로 확인한다.
[규정]
  1) 강도를 〔25, 45〕MPa로 나눈 3등급으로 stratify 해서 train_test_split(test_size=0.25, random_state=42)로 먼저 나눈다.
  2) StandardScaler는 훈련 데이터로만 fit 하고, 시험 데이터는 transform만 한다.
  3) LinearRegression()으로 학습한다. 내 scikit-learn 버전에서 없어진 인자(normalize, squared 등)는 쓰지 않는다.
[출력] 훈련·시험의 R², MAE, RMSE, 특징별 가중치(coef_)와 절편, 늘 평균만 답하는 DummyRegressor와의 비교, 재령 구간별 평균 잔차.
[금지] 시험 데이터로 학습하거나 특징을 고르지 않는다. 데이터를 지어내지 않는다.
[형식] 셀 단위로 나누고, 각 셀 위에 무엇을 하는지 한국어 주석을 단다. 코드 뒤에 내가 확인해야 할 점 3가지를 적어 줘.`;
$('mainPrompt').textContent=MAIN_PROMPT;$('copyMain').addEventListener('click',()=>copyText(MAIN_PROMPT,'프롬프트를 복사했습니다'));
(function(){const LINES=[
 ['import pandas as pd'],['from ucimlrepo import fetch_ucirepo'],['from sklearn.model_selection import train_test_split'],['from sklearn.preprocessing import StandardScaler'],['from sklearn.linear_model import LinearRegression'],['from sklearn.metrics import r2_score'],[''],
 ['uci = fetch_ucirepo(id=165)',0,'UCI에서 콘크리트 데이터를 실제로 받았다. 괜찮다.'],
 ['df = pd.concat([uci.data.features, uci.data.targets], axis=1)',0,'특징 여덟 열과 강도 한 열을 붙였다(9열). 괜찮다.'],
 ['X = df.iloc[:, :9]   # 특징',1,'<b>정답 누설.</b> 0~8번, 아홉 열을 다 가져와 압축 강도(8번 열)까지 특징에 들어갔다. 정답을 보고 정답을 맞히는 셈이라 R²가 1.000이 나온다. <code>df.iloc[:, :8]</code>로 고치고, R²가 이상하게 높으면 먼저 의심한다.'],
 ['y = df.iloc[:, 8]    # 압축 강도(MPa)',0,'타겟은 강도 열이다. 괜찮다.'],[''],
 ['X = StandardScaler().fit_transform(X)   # 표준화',2,'<b>데이터 누수.</b> 나누기 전에 전체(시험지 포함)로 평균·표준편차를 구했다. 먼저 나누고, 훈련 데이터로만 fit 한 뒤 시험지는 transform만 한다.'],
 ['X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)',3,'<b>stratify와 random_state가 없다.</b> 돌릴 때마다 R²가 달라지고, 고강도 배합이 한쪽으로 쏠릴 수 있다. 강도 등급으로 <code>stratify</code>하고 <code>random_state=42</code>를 넣는다.'],[''],
 ['model = LinearRegression(normalize=True)',4,'<b>없어진 인자.</b> <code>normalize</code>는 scikit-learn 1.0에서 예고된 뒤 1.2에서 없어졌다. 최신 버전에서는 오류가 난다. 지우고, 표준화는 StandardScaler로 따로 한다.'],
 ['model.fit(X_train, y_train)',0,'훈련 데이터로 학습했다. 괜찮다.'],[''],
 ['pred = model.predict(X_train)',5,'<b>훈련 데이터로 채점했다.</b> 외운 문제로 시험 본 셈이다. <code>model.predict(X_test)</code>와 <code>y_test</code>로 채점한다.'],
 ['print("R2:", r2_score(y_train, pred))',5,'<b>훈련 데이터로 채점했다.</b> 바로 윗줄과 같은 문제다. 게다가 R² 하나만 찍었다. MAE(MPa)와 기준선(늘 평균)도 함께 본다.']];
  const host=$('bugCode'),found=new Set();
  host.innerHTML=LINES.map((l,i)=>'<div class="ln" role="listitem" tabindex="0" data-i="'+i+'"><span class="no">'+(i+1)+'</span><span>'+hl(l[0]||' ')+'</span></div>').join('');
  function score(){$('bugScore').textContent=found.size+' / 5';if(found.size===5){$('bugScore').className='pill ok';}}
  function pick(el){const i=+el.getAttribute('data-i'),l=LINES[i];if(!l[0])return;
    if(l[1]){found.add(l[1]);host.querySelectorAll('.ln').forEach(x=>{if(LINES[+x.getAttribute('data-i')][1]===l[1])x.classList.add('hit');});$('bugMsg').innerHTML=(i+1)+'번 줄: '+l[2]+(found.size===5?'<br><b>다섯 군데를 모두 찾았다.</b> 고친 코드는 A장의 Colab 코드와 같다.':'');}
    else{el.classList.add('miss');$('bugMsg').innerHTML=(i+1)+'번 줄: '+(l[2]||'이 줄은 괜찮다.');}score();}
  host.addEventListener('click',e=>{const el=e.target.closest('.ln');if(el)pick(el);});
  host.addEventListener('keydown',e=>{if((e.key==='Enter'||e.key===' ')&&e.target.classList.contains('ln')){e.preventDefault();pick(e.target);}});
  $('bugReveal').addEventListener('click',()=>{LINES.forEach((l,i)=>{if(l[1]){found.add(l[1]);host.querySelector('[data-i="'+i+'"]').classList.add('hit','reveal');}});score();$('bugMsg').innerHTML='빨간 줄 다섯 군데: ① 10번 정답 누설(iloc[:, :9]) ② 13번 데이터 누수 ③ 14번 나누기 규정 ④ 16번 없어진 인자 ⑤ 19~20번 훈련 데이터로 채점.';});
  $('bugReset').addEventListener('click',()=>{found.clear();host.querySelectorAll('.ln').forEach(x=>x.classList.remove('hit','miss','reveal'));$('bugScore').className='pill';score();$('bugMsg').textContent='줄을 하나 눌러 보세요.';});score();})();
(function(){const CK=[['데이터를 실제로 읽었나','AI가 예제 데이터(옛 보스턴 집값 등)를 쓰거나 지어내지 않았는지, 행이 1,030개인지 본다.'],['정답 열이 특징에 섞이지 않았나','X의 열 수가 8인지 센다. R²가 0.99처럼 지나치게 높으면 정답 누설부터 의심한다.'],['나누기를 먼저 하고, 표준화는 훈련 데이터로만 fit 했나','fit_transform(X) 전체가 보이면 누수다.'],['stratify와 random_state를 넣었나','등급 비율이 시험지에도 유지되고, 점수가 매번 같게 나온다.'],['시험지(X_test)는 마지막 채점에만 썼나','특징을 고르려고 시험지 점수를 여러 번 보면 시험지가 훈련 데이터가 된다. 고를 때는 교차 검증을 쓴다.'],['인자가 내 scikit-learn 버전에 있나','print(sklearn.__version__). normalize(1.2에서 삭제), squared=False(1.6에서 삭제), load_boston(1.2에서 삭제)은 옛 코드의 흔적이다.'],['지표의 단위를 확인했나','MAE·RMSE는 MPa, R²는 단위가 없다. "R² 0.57"만 보고하지 않고 "평균 8.4MPa 틀린다"를 함께 쓴다.'],['기준선·합격선과 비교했나','늘 평균만 답하는 DummyRegressor(R² 0), 매뉴얼 씨의 경험식(R² 0.735)보다 낫지 않으면 뽑을 까닭이 없다.'],['잔차 그림을 그려 봤나','잔차를 재령·예측값에 대어 그렸을 때 무늬가 있으면 직선이 놓친 관계가 있다(05장).']];
  const ul=$('checks');ul.innerHTML=CK.map((c,i)=>'<li><input type="checkbox" id="ck'+i+'"><label for="ck'+i+'"><b>'+c[0]+'</b><small>'+c[1]+'</small></label></li>').join('');
  function upd(){const n=ul.querySelectorAll('input:checked').length;ul.querySelectorAll('li').forEach(li=>li.classList.toggle('done',li.querySelector('input').checked));$('checkScore').textContent='확인 '+n+' / '+CK.length+(n===CK.length?' — 이제 돌려 본다. 결과도 확인 대상이다.':'');}
  ul.addEventListener('change',upd);upd();})();

/* ── 04 수습 기간: 물-결합재비 하나로 경사 하강 vs 정규방정식 ── */
(function(){const dc=$('gdData'),bc=$('gdBowl'),lc=$('gdLoss'),RAW=TR.map(r=>r.wb),MU=ML.mean(RAW),SD=Math.sqrt(ML.mean(RAW.map(v=>(v-MU)**2)));
  const st={mode:'std',w:0,b:0,t:0,path:[[0,0]],hist:[],play:false,div:false,ne:false};let S=null;
  /* 손실 지도는 충분 통계량으로 바로 계산한다(평균 x, x², y, xy, y²). 한 걸음의 기울기도 같은 값이다 */
  function setMode(m){st.mode=m;const X=m==='std'?RAW.map(v=>(v-MU)/SD):RAW.slice(),n=X.length;let ex=0,exx=0,ey=0,exy=0,eyy=0;
    X.forEach((x,i)=>{const y=YTR[i];ex+=x;exx+=x*x;ey+=y;exy+=x*y;eyy+=y*y;});ex/=n;exx/=n;ey/=n;exy/=n;eyy/=n;
    const wO=(exy-ex*ey)/(exx-ex*ex),bO=ey-wO*ex;S={X,ex,exx,ey,exy,eyy,wO,bO};S.mO=L(wO,bO);
    const w0=Math.min(0,wO),w1=Math.max(0,wO),b0=Math.min(0,bO),b1=Math.max(0,bO),pw=(w1-w0)*0.35+1e-9,pb=(b1-b0)*0.35+1e-9;S.box={w0:w0-pw,w1:w1+pw,b0:b0-pb,b1:b1+pb};S.heat=null;reset();}
  const L=(w,b)=>S.eyy-2*w*S.exy-2*b*S.ey+w*w*S.exx+2*w*b*S.ex+b*b;
  function reset(){stop();st.w=0;st.b=0;st.t=0;st.path=[[0,0]];st.hist=[L(0,0)];st.div=false;st.ne=false;draw();}
  const lr=()=>Math.pow(10,+$('gdLr').value);
  function step(){const gw=-2*(S.exy-st.w*S.exx-st.b*S.ex),gb=-2*(S.ey-st.w*S.ex-st.b),a=lr();st.w-=a*gw;st.b-=a*gb;st.t++;const m=L(st.w,st.b);st.hist.push(m);st.path.push([st.w,st.b]);if(!isFinite(m)||m>1e7)st.div=true;}
  function draw(){drawData();drawBowl();drawLoss();info();}
  function drawData(){const W=fit(dc,280),ctx=dc.getContext('2d'),C=cols(),pl=46,pr=10,pt=10,pb=34,gw=W-pl-pr,gh=280-pt-pb,x0=0.2,x1=1.0,sx=x=>pl+(x-x0)/(x1-x0)*gw,sy=y=>pt+(1-y/85)*gh;
    ctx.clearRect(0,0,W,280);ctx.strokeStyle=C.rule;ctx.strokeRect(pl,pt,gw,gh);ctx.fillStyle=C.muted;ctx.font=font(12);ctx.textAlign='center';[0.3,0.5,0.7,0.9].forEach(v=>ctx.fillText(v,sx(v),280-18));ctx.fillText('물-결합재비',pl+gw/2,280-3);
    ctx.textAlign='right';[0,40,80].forEach(v=>ctx.fillText(v,pl-5,sy(v)+4));ctx.save();ctx.translate(11,pt+gh/2);ctx.rotate(-Math.PI/2);ctx.textAlign='center';ctx.fillText('강도 (MPa)',0,0);ctx.restore();
    RAW.forEach((x,i)=>circ(ctx,sx(x),sy(YTR[i]),2.4,rgba(C.text,.45),null));
    const pr_=x=>{const u=st.mode==='std'?(x-MU)/SD:x;return st.w*u+st.b;};
    if(isFinite(st.w)&&isFinite(st.b)){ctx.save();ctx.beginPath();ctx.rect(pl,pt,gw,gh);ctx.clip();ctx.strokeStyle=C.blue;ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(sx(x0),sy(pr_(x0)));ctx.lineTo(sx(x1),sy(pr_(x1)));ctx.stroke();ctx.restore();}}
  function drawBowl(){const W=fit(bc,230),ctx=bc.getContext('2d'),C=cols(),pl=50,pr=10,pt=8,pb=32,gw=W-pl-pr,gh=230-pt-pb,B=S.box,sx=w=>pl+(w-B.w0)/(B.w1-B.w0)*gw,sy=b=>pt+(1-(b-B.b0)/(B.b1-B.b0))*gh;
    ctx.clearRect(0,0,W,230);const nx=64,ny=40,lo=Math.log(S.mO),hi=Math.log(L(B.w0,B.b1)+L(B.w1,B.b0));
    for(let i=0;i<nx;i++)for(let j=0;j<ny;j++){const w=B.w0+(i+.5)/nx*(B.w1-B.w0),b=B.b1-(j+.5)/ny*(B.b1-B.b0),v=(Math.log(L(w,b))-lo)/(hi-lo),band=Math.floor(clamp(v,0,.999)*10);
      ctx.fillStyle=rgba(C.blue,0.42-band*0.04+((band%2)?0.03:0));ctx.fillRect(pl+i*gw/nx,pt+j*gh/ny,gw/nx+0.6,gh/ny+0.6);}
    ctx.strokeStyle=C.rule;ctx.strokeRect(pl,pt,gw,gh);ctx.fillStyle=C.muted;ctx.font=font(11.5);ctx.textAlign='center';
    for(let k=0;k<=4;k++){const w=B.w0+(B.w1-B.w0)*k/4;ctx.fillText(w.toFixed(Math.abs(B.w1-B.w0)<10?1:0),sx(w),230-18);}ctx.fillText('가중치 w'+(st.mode==='std'?' (표준화한 x 기준)':' (날것 x 기준)'),pl+gw/2,230-3);
    ctx.textAlign='right';for(let k=0;k<=3;k++){const b=B.b0+(B.b1-B.b0)*k/3;ctx.fillText(b.toFixed(0),pl-5,sy(b)+4);}ctx.save();ctx.translate(11,pt+gh/2);ctx.rotate(-Math.PI/2);ctx.textAlign='center';ctx.fillText('절편 b',0,0);ctx.restore();
    ctx.save();ctx.beginPath();ctx.rect(pl,pt,gw,gh);ctx.clip();
    const P=st.path,stp=Math.max(1,Math.floor(P.length/400));ctx.strokeStyle=C.stamp;ctx.lineWidth=1.8;ctx.beginPath();for(let i=0;i<P.length;i+=stp){const p=P[i];if(!isFinite(p[0]))break;i?ctx.lineTo(sx(p[0]),sy(p[1])):ctx.moveTo(sx(p[0]),sy(p[1]));}
    const la=P[P.length-1];if(isFinite(la[0]))ctx.lineTo(sx(la[0]),sy(la[1]));ctx.stroke();
    for(let i=0;i<Math.min(P.length,25);i++)circ(ctx,sx(P[i][0]),sy(P[i][1]),2.6,C.stamp,null);
    if(isFinite(la[0]))circ(ctx,sx(la[0]),sy(la[1]),5.5,C.stamp,C.panel,2);
    ctx.fillStyle=C.blue;ctx.strokeStyle=C.panel;ctx.lineWidth=1.5;star(ctx,sx(S.wO),sy(S.bO),9);ctx.restore();
    ctx.fillStyle=C.text;ctx.font=font(11.5,700);ctx.textAlign='left';ctx.fillText('★ 정규방정식의 답',pl+6,pt+14);ctx.fillStyle=C.stamp;ctx.fillText('● 경사 하강의 발자국',pl+6,pt+29);}
  function star(ctx,x,y,r){ctx.beginPath();for(let i=0;i<10;i++){const a=-Math.PI/2+i*Math.PI/5,rr_=i%2?r*0.45:r;ctx.lineTo(x+rr_*Math.cos(a),y+rr_*Math.sin(a));}ctx.closePath();ctx.fill();ctx.stroke();}
  function drawLoss(){const W=fit(lc,150),ctx=lc.getContext('2d'),C=cols(),pl=50,pr=10,pt=8,pb=26,gw=W-pl-pr,gh=150-pt-pb,h=st.hist,n=Math.max(60,h.length),mx=h[0]*1.05,sy=v=>pt+gh-Math.min(1,v/mx)*gh;
    ctx.clearRect(0,0,W,150);ctx.strokeStyle=C.rule;ctx.strokeRect(pl,pt,gw,gh);ctx.fillStyle=C.muted;ctx.font=font(11.5);ctx.textAlign='right';[0,Math.round(mx/2),Math.round(mx)].forEach(v=>ctx.fillText(v,pl-5,sy(v)+4));
    ctx.textAlign='center';ctx.fillText('반복 횟수 (0 ~ '+(n-1)+') · 세로: 훈련 MSE',pl+gw/2,150-6);
    ctx.setLineDash([5,4]);ctx.strokeStyle=C.blue;ctx.beginPath();ctx.moveTo(pl,sy(S.mO));ctx.lineTo(pl+gw,sy(S.mO));ctx.stroke();ctx.setLineDash([]);
    ctx.strokeStyle=C.stamp;ctx.lineWidth=2.2;ctx.beginPath();const s=Math.max(1,Math.floor(h.length/500));for(let i=0;i<h.length;i+=s){const x=pl+i/(n-1)*gw;i?ctx.lineTo(x,sy(h[i])):ctx.moveTo(x,sy(h[i]));}ctx.stroke();
    ctx.fillStyle=C.blue;ctx.font=font(11.5,700);ctx.textAlign='right';ctx.fillText('정규방정식 MSE '+S.mO.toFixed(1),pl+gw-4,sy(S.mO)-5);}
  function info(){const m=st.hist[st.hist.length-1],dw=Math.abs(st.w-S.wO),u=st.mode==='std';let h;
    if(st.div)h='<b style="color:var(--stamp)">발산!</b> 학습률 η = '+lr().toFixed(2)+'이 너무 커서 한 걸음마다 바닥을 넘어 반대편으로 더 높이 튀어 나갔다. 손실이 '+(isFinite(m)?m.toExponential(1):'∞')+'까지 커졌다. η를 줄이고 "처음으로"를 누르자.';
    else if(st.ne)h='정규방정식으로 바로 바닥에 섰다: w = <b>'+fx(S.wO,u?2:1)+'</b>, b = <b>'+S.bO.toFixed(2)+'</b>, MSE = <b>'+S.mO.toFixed(1)+'</b>. 반복 0번, 학습률도 없다.';
    else{h='반복 <b>'+st.t+'</b>번 · 훈련 MSE <b>'+m.toFixed(1)+'</b> (정규방정식 '+S.mO.toFixed(1)+') · w = '+fx(st.w,u?2:1)+', b = '+st.b.toFixed(2)+' · 바닥까지 w 차이 '+fx(dw,2);
      if(st.t===0)h+='<br>w = 0, b = 0에서 출발한다. 모든 배합을 0MPa라고 하는 직선이라 MSE가 '+m.toFixed(0)+'이다. ▶ 재생을 눌러 보자.';
      else if(dw<Math.abs(S.wO)*0.01)h+='<br><b style="color:var(--ok)">바닥 도착.</b> 경사 하강이 정규방정식과 같은 직선에 닿았다'+(u?'. 표준화하면 그릇이 동그래서 '+st.t+'번이면 충분하다.':'. 날것 그대로는 '+st.t+'번이나 걸렸다. 표준화한 쪽과 비교해 보자.');
      else if(!u&&st.t>150)h+='<br>MSE는 금방 줄었는데 바닥(★)까지는 아직 멀다. 날것 그대로의 물-결합재비는 0.27~0.9 사이에만 있어서, 손실 지도가 길쭉한 골짜기가 된다. 골짜기 바닥을 따라 기어가는 중이다. 학습률을 올려 보면? (너무 올리면 발산한다)';}
    $('gdInfo').innerHTML=h;}
  function chips(k){document.querySelectorAll('#loopChips button').forEach((b,i)=>{b.classList.toggle('on',i===k);b.classList.toggle('done',k>=0&&i<k);});}
  let raf=0,ph=0;
  function frame(){if(!st.play)return;ph=(ph+1)%8;chips(Math.floor(ph/2));const k=st.t<30?1:st.t<150?3:st.mode==='raw'?40:8;for(let i=0;i<k&&!st.div;i++)step();draw();
    if(st.div||st.t>=(st.mode==='raw'?6000:600)||Math.abs(st.w-S.wO)<Math.abs(S.wO)*1e-4){stop();return;}raf=requestAnimationFrame(frame);}
  function stop(){st.play=false;cancelAnimationFrame(raf);$('gdPlay').textContent='▶ 재생';chips(-1);}
  $('gdPlay').addEventListener('click',()=>{if(st.play){stop();return;}if(st.div||st.ne)reset();st.play=true;$('gdPlay').textContent='❚❚ 멈춤';if(REDUCE){for(let i=0;i<2000&&!st.div;i++)step();draw();stop();return;}raf=requestAnimationFrame(frame);});
  $('gdStep').addEventListener('click',async()=>{stop();if(st.div||st.ne)reset();for(let k=0;k<4;k++){chips(k);await wait(REDUCE?0:150);}step();draw();chips(-1);});
  $('gdReset').addEventListener('click',reset);
  $('gdNe').addEventListener('click',()=>{stop();st.w=S.wO;st.b=S.bO;st.path.push([S.wO,S.bO]);st.hist.push(S.mO);st.ne=true;st.div=false;draw();});
  $('gdLr').addEventListener('input',()=>{$('gdLrO').textContent=lr().toFixed(lr()<0.1?3:2);});
  seg('#gdScale button',v=>setMode(v));
  $('gdLrO').textContent=lr().toFixed(2);setMode('std');onRedraw(draw);
  /* 특징 8개 전부: 두 길 비교 */
  $('twoRun').addEventListener('click',async()=>{$('twoInfo').textContent='계산 중…';await wait(20);const F=fitSet('m8'),A=F.A,rows=[],st8={w:new Array(8).fill(0),b:0};let t0=performance.now(),L8=0;
    for(let i=1;i<=1000;i++){L8=LR.gdStep(st8,A,YTR,0.2);if(i===10||i===100||i===1000)rows.push(['경사 하강 '+i+'번',LR.mse(st8,A,YTR),st8.w.slice(),st8.b,performance.now()-t0]);}
    const t1=performance.now(),m=LR.fit(A,YTR),neMs=performance.now()-t1;rows.push(['정규방정식 (한 번)',LR.mse(m,A,YTR),m.w,m.b,neMs]);
    const ab=['시멘트','슬래그','플라이 애시','물','감수제','굵은 골재','잔골재','재령'];
    $('twoTbl').innerHTML='<thead><tr><th>방법</th><th>훈련 MSE</th>'+ab.map(a=>'<th>'+a+'</th>').join('')+'<th>절편</th><th>시간</th></tr></thead><tbody>'+rows.map((r,k)=>'<tr'+(k===3?' style="font-weight:700"':'')+'><td style="white-space:nowrap">'+r[0]+'</td><td>'+r[1].toFixed(2)+'</td>'+r[2].map(v=>'<td>'+fx(v)+'</td>').join('')+'<td>'+fx(r[3])+'</td><td>'+Math.max(0.1,r[4]).toFixed(1)+'ms</td></tr>').join('')+'</tbody>';
    $('twoInfo').innerHTML='경사 하강은 10번째에 가중치 부호부터 틀려 있다(굵은 골재·잔골재가 −). 100번째에 대강 맞고, 1,000번째에 정규방정식과 소수 둘째 자리까지 같아진다. 바닥은 하나뿐이라 어느 길로 가도 같은 가중치다. 특징 8개에서는 9×9 방정식 한 번이 1,000걸음보다 빠르다.';});})();

/* ── 05 성과 평가 ── */
function binsOf(xs,es,edges){return edges.slice(0,-1).map((lo,k)=>{const hi=edges[k+1],idx=xs.map((x,i)=>i).filter(i=>xs[i]>lo&&xs[i]<=hi);return {lo,hi,n:idx.length,m:idx.length?ML.mean(idx.map(i=>es[i])):null,x:idx.length?ML.mean(idx.map(i=>xs[i])):null};});}
const AGE_EDGES=[0,3,7,14,28,56,100,365];
function renderEval(){const F=fitSet('m8'),C=cols();
  $('evalStat').innerHTML=statGrid([[f3(F.r2),'시험 R² (특징 8개)','a'],[f2(F.mae)+'MPa','시험 MAE · 평균 오차','a'],[f2(F.rmse)+'MPa','시험 RMSE'],[f3(MAN.r2),'합격선 R² · 매뉴얼 씨 경험식','m'],[f2(MAN.mae)+'MPa','매뉴얼 씨 MAE','m'],[f3(F.trR2),'훈련 R² (772건)']],3)+
    '<p class="foot">기준선(늘 훈련 평균 '+YMEAN.toFixed(1)+'MPa)은 시험 R² '+f3(BASE.r2)+', MAE '+f2(BASE.mae)+'MPa다. 선형 회귀는 기준선은 넉넉히 넘었지만 매뉴얼 씨의 경험식에는 R²로 '+f3(MAN.r2-F.r2)+', MAE로 '+f2(F.mae-MAN.mae)+'MPa 뒤진다. 훈련 R²('+f3(F.trR2)+')도 낮으니 외운 게 아니라 <b>직선이라 못 따라가는 것</b>이다(과소적합).</p>';
  /* MAE vs RMSE */
  const es=YTE.map((y,i)=>y-F.pte[i]),big=es.map((e,i)=>[Math.abs(e),i]).sort((a,b)=>b[0]-a[0])[0],r=TE[big[1]];
  $('maeTxt').innerHTML='MAE는 잔차 크기의 평균, RMSE는 잔차 제곱 평균의 제곱근이다. 늘 RMSE ≥ MAE이고, <b>큰 실수가 많을수록 둘의 차이가 벌어진다</b>. 특징 8개 모델은 MAE '+f2(F.mae)+' · RMSE '+f2(F.rmse)+'MPa다. 가장 크게 틀린 시험 배합은 실제 '+r.strength+'MPa(재령 '+r.age+'일)를 '+F.pte[big[1]].toFixed(1)+'MPa로 불러 '+big[0].toFixed(1)+'MPa 어긋났다. 이런 큰 실수가 RMSE를 끌어올린다. 학습이 줄이는 것은 MSE(=RMSE²)라서, 선형 회귀는 이런 배합을 피하려고 직선을 그쪽으로 당긴다.';
  /* 예측 대 실제 · 잔차 */
  let cur='m8';
  function draw(){const G=fitSet(cur),pc=$('pvaCv'),rc=$('resCv'),C=cols(),H=290,ax=$('resX').value;
    {const W=fit(pc,H),ctx=pc.getContext('2d'),pl=46,pr=10,pt=10,pb=36,s=Math.min(W-pl-pr,H-pt-pb),sx=v=>pl+v/85*s,sy=v=>pt+(1-v/85)*s;ctx.clearRect(0,0,W,H);
      ctx.strokeStyle=C.rule;ctx.strokeRect(pl,pt,s,s);ctx.fillStyle=C.muted;ctx.font=font(11.5);[0,20,40,60,80].forEach(v=>{ctx.textAlign='center';ctx.fillText(v,sx(v),pt+s+14);ctx.textAlign='right';ctx.fillText(v,pl-5,sy(v)+4);});
      ctx.textAlign='center';ctx.fillText('실제 강도 (MPa)',pl+s/2,pt+s+30);ctx.save();ctx.translate(11,pt+s/2);ctx.rotate(-Math.PI/2);ctx.fillText('예측 강도 (MPa)',0,0);ctx.restore();
      ctx.setLineDash([5,4]);ctx.strokeStyle=C.muted;ctx.beginPath();ctx.moveTo(sx(0),sy(0));ctx.lineTo(sx(85),sy(85));ctx.stroke();ctx.setLineDash([]);
      TE.forEach((r,i)=>circ(ctx,sx(r.strength),sy(clamp(G.pte[i],-5,90)),3,rgba(GRADECOL[r.grade],.8),null));
      ctx.fillStyle=C.blue;ctx.font=font(12,700,1);ctx.textAlign='left';ctx.fillText('R² = '+f3(G.r2),pl+8,pt+16);ctx.fillText('MAE = '+f2(G.mae),pl+8,pt+32);}
    {const W=fit(rc,H),ctx=rc.getContext('2d'),pl=46,pr=10,pt=10,pb=36,gw=W-pl-pr,gh=H-pt-pb,es=YTE.map((y,i)=>y-G.pte[i]);
      const xv=ax==='age'?TE.map(r=>Math.log10(r.age)):ax==='pred'?G.pte.slice():TE.map(r=>r[ax]),x0=Math.min(...xv),x1=Math.max(...xv),px=(x1-x0)*0.04,X0=x0-px,X1=x1+px,sx=v=>pl+(v-X0)/(X1-X0)*gw,sy=v=>pt+(1-(v+35)/70)*gh;
      ctx.clearRect(0,0,W,H);ctx.strokeStyle=C.rule;ctx.strokeRect(pl,pt,gw,gh);ctx.fillStyle=C.muted;ctx.font=font(11.5);ctx.textAlign='right';[-30,-15,0,15,30].forEach(v=>ctx.fillText((v>0?'+':'')+v,pl-5,sy(v)+4));
      ctx.textAlign='center';if(ax==='age')[1,3,7,28,90,365].forEach(d=>ctx.fillText(d,sx(Math.log10(d)),pt+gh+14));else for(let k=0;k<=4;k++){const v=X0+(X1-X0)*k/4;ctx.fillText(v.toFixed(v<5?2:0),sx(v),pt+gh+14);}
      ctx.fillText(ax==='age'?'재령 (일, 로그 눈금)':ax==='pred'?'예측 강도 (MPa)':ax==='wb'?'물-결합재비':'시멘트 (kg/m³)',pl+gw/2,H-4);ctx.save();ctx.translate(11,pt+gh/2);ctx.rotate(-Math.PI/2);ctx.fillText('잔차 = 실제 − 예측 (MPa)',0,0);ctx.restore();
      ctx.strokeStyle=C.text;ctx.lineWidth=1.2;ctx.beginPath();ctx.moveTo(pl,sy(0));ctx.lineTo(pl+gw,sy(0));ctx.stroke();
      ctx.fillStyle=rgba(C.muted,.9);ctx.font=font(11);ctx.textAlign='left';ctx.fillText('+ 덜 예측 (실제가 더 강함)',pl+6,pt+13);ctx.fillText('− 더 예측 (실제가 더 약함)',pl+6,pt+gh-6);
      TE.forEach((r,i)=>circ(ctx,sx(xv[i]),sy(clamp(es[i],-34,34)),3,rgba(C.blue,.55),null));
      const edges=ax==='age'?AGE_EDGES.map(d=>d?Math.log10(d):-1):(()=>{const s=xv.slice().sort((a,b)=>a-b);return [-Infinity].concat([1,2,3,4,5,6,7].map(q=>s[Math.floor(q*s.length/8)])).concat([Infinity]);})();
      const bn=binsOf(xv,es,edges).filter(b=>b.n>=4);ctx.strokeStyle=C.stamp;ctx.lineWidth=3;ctx.beginPath();bn.forEach((b,k)=>{k?ctx.lineTo(sx(b.x),sy(b.m)):ctx.moveTo(sx(b.x),sy(b.m));});ctx.stroke();bn.forEach(b=>circ(ctx,sx(b.x),sy(b.m),4.5,C.stamp,C.panel,1.5));
      const ab=binsOf(TE.map(r=>r.age),es,AGE_EDGES),e1=ab[0],e5=ab.slice(4,6).filter(b=>b.n),e7=ab[6],mid=e5.length?ML.mean(e5.map(b=>b.m)):0;
      $('resInfo').innerHTML='<b>'+FS[cur].lab+'</b> · 시험 R² '+f3(G.r2)+' · MAE '+f2(G.mae)+'MPa. 빨간 선은 구간별 평균 잔차다. 무늬가 없으면 0선 근처에서 평평해야 한다. '+
        (cur==='m8'?'재령 축으로 보면 <b>3일 이하 '+fx(e1.m,1)+'MPa → 29~100일 +'+mid.toFixed(1)+'MPa → 100일 넘게 '+fx(e7.m,1)+'MPa</b>로 아래-위-아래 활 모양이다. 초기 재령은 너무 높게, 한두 달은 너무 낮게, 1년은 다시 높게 부른다. 강도가 재령에 대해 곡선으로 오르는데 직선이 그것을 못 따라간 흔적이다.':
        '재령 축의 활 모양이 거의 펴졌다(3일 이하 '+fx(e1.m,1)+'MPa, 29~100일 '+fx(mid,1)+'MPa). log(재령)이 곡선을 직선으로 펴 준 덕분이다. 남은 흩어짐은 직선 모델이 아직 모르는 상호작용 등이다(5-3).');}}
  seg('#resModel button',v=>{cur=v;draw();});$('resX').onchange=draw;draw();onRedraw(draw);
  /* 좋은 특징 실험 */
  const KS=['m8','wb','la','m10','two'],TXT={m8:'날것 특징 여덟 개. 재령이 직선으로만 들어가 곡선을 못 따라간다. 합격선 아래.',wb:'물-결합재비를 더해도 거의 그대로다. 물과 결합재가 이미 특징에 있어 선형 회귀가 비슷한 정보를 조합할 수 있었기 때문이다. 핵심은 이것이 아니다.',la:'log(재령) 하나로 크게 뛴다. 재령에 따른 곡선이 펴졌다. 이 데이터에서 직선이 진 까닭은 거의 전부 재령의 곡선이었다.',m10:'둘 다 넣은 기본 대안. 합격선을 넉넉히 넘는다. 매뉴얼 씨가 쓰던 두 숫자(물-결합재비, 재령)를 사람의 지식대로 다듬어 넣고, 가중치는 데이터에서 배웠다.',two:'매뉴얼 씨의 두 숫자만 주면 경험식을 아슬아슬하게 넘는다. 같은 정보로 상수 네 개를 교과서 대신 데이터에서 배운 셈이다.'};
  $('feChoice').innerHTML=KS.map((k,i)=>'<button data-v="'+k+'" aria-pressed="'+(i===0)+'">'+FS[k].lab+'</button>').join('');
  function bars(sel){const rows=KS.filter(k=>FIT[k]).map(k=>[FS[k].lab,FIT[k].r2,FIT[k].mae,k===sel?'var(--blue)':rgba(cols().blue,.45)]).concat([['매뉴얼 씨 경험식 (합격선)',MAN.r2,MAN.mae,'var(--c-man)'],['기준선 (늘 평균)',Math.max(0,BASE.r2),BASE.mae,'var(--muted)']]);
    $('feBars').innerHTML=rows.map(r=>'<div class="row" style="grid-template-columns:minmax(0,1.3fr) minmax(50px,2fr) auto"><span style="text-align:left;font-family:var(--font)">'+r[0]+'</span><div class="tr"><i style="left:0;width:'+Math.max(0,r[1])*100+'%;background:'+r[3]+'"></i></div><span style="white-space:nowrap">'+f3(r[1])+' · '+r[2].toFixed(2)+'</span></div>').join('')+'<p class="foot">숫자: 시험 R² · MAE(MPa). 막대 길이는 R²(0~1).</p>';
    const G=FIT[sel];$('feInfo').innerHTML='<b>'+FS[sel].lab+'</b> (외울 숫자 '+(G.m.w.length+1)+'개): 시험 R² <b>'+f3(G.r2)+'</b>, MAE '+f2(G.mae)+'MPa → '+(passOf(G.r2)?'<b style="color:var(--ok)">합격선 위</b>':'<b style="color:var(--stamp)">합격선 아래</b>')+'. '+TXT[sel];}
  $('feChoice').addEventListener('click',e=>{const b=e.target.closest('button');if(!b)return;$('feChoice').querySelectorAll('button').forEach(x=>x.setAttribute('aria-pressed',x===b));const k=b.getAttribute('data-v');fitSet(k);bars(k);});
  bars('m8');
  /* 인사 카드: 가중치 · 성적표 */
  const WV=[['특징 8개 (표준화)',()=>fitSet('m8'),'1 표준편차 늘 때 MPa'],['특징 8개 (날것 단위)',()=>fitSet('m8',true),'1kg/m³ 또는 1일 늘 때 MPa'],['10개 모델 (표준화)',()=>fitSet('m10'),'1 표준편차 늘 때 MPa']];
  $('wChoice').innerHTML=WV.map((w,k)=>'<button data-v="'+k+'" aria-pressed="'+(k===0)+'">'+w[0]+'</button>').join('');
  function wb(k){const G=WV[k][1](),w=G.m.w,names=FS[G.k].names,mx=Math.max(...w.map(Math.abs));
    $('wBars').innerHTML=names.map((nm,j)=>{const v=w[j],Wd=Math.abs(v)/mx*50;return '<div class="row" style="grid-template-columns:110px minmax(0,1fr) 64px"><span style="text-align:left;font-family:var(--font)">'+nm+'</span><div class="tr"><span class="z"></span><i class="'+(v<0?'neg':'')+'" style="left:'+(v<0?50-Wd:50)+'%;width:'+Wd+'%"></i></div><span>'+fx(v,k===1?3:2)+'</span></div>';}).join('')+'<p class="foot">단위: '+WV[k][2]+' · 절편 '+fx(G.m.b,2)+'</p>';
    $('wInfo').innerHTML=k===0?'시멘트(+'+fx(w[0])+'), 슬래그(+'+fx(w[1])+'), 재령(+'+fx(w[7])+')이 강도를 올리고 물('+fx(w[3])+')이 내린다. "물이 적고 결합재가 많고 오래 굳히면 강하다"는 현장 경험과 같은 방향이다. 표준화했으니 막대끼리 크기를 비교할 수 있다. <span class="muted">주의: 배합량은 합이 거의 일정해 서로 얽혀 있다(물을 줄이면 골재가 는다). 가중치 하나만 떼어 "원인"이라고 읽지 않는다.</span>':
      k===1?'같은 모델을 날것 단위로 학습하면 가중치가 "1kg 또는 1일당 MPa"가 된다. 고성능 감수제가 '+fx(w[4],3)+'로 가장 커 보이지만, 감수제는 0~32kg로 범위가 좁고 시멘트는 100~540kg로 넓다. 단위가 다르면 크기를 비교할 수 없다. 예측과 R²('+f3(G.r2)+')는 표준화한 모델과 똑같다.':
      '10개 모델에서는 log(재령)이 +'+fx(w[9])+'로 가장 크고, 재령(직선)은 '+fx(w[7])+'로 부호가 뒤집혔다. log(재령)이 "처음에 빨리 오름"을 맡고, 재령은 "1년 가까이 가면 눕는" 부분을 살짝 깎는 역할로 바뀐 것이다. 물-결합재비('+fx(w[8])+')가 들어오자 물('+fx(w[3])+')의 몫이 줄었다. 함께 움직이는 특징끼리는 가중치를 나눠 갖는다.';}
  seg('#wChoice button',v=>wb(+v));wb(0);
  const T=[['기준선 (늘 훈련 평균)',BASE,null],['매뉴얼 씨 경험식 (합격선)',MAN,null],['선형 회귀 · 특징 8개',fitSet('m8'),1],['선형 회귀 · + 물-결합재비·log(재령)',fitSet('m10'),1]];
  $('teamTbl').innerHTML='<thead><tr><th>지원자</th><th>시험 R²</th><th>MAE</th><th>RMSE</th><th>판정</th></tr></thead><tbody>'+T.map(t=>'<tr><td>'+t[0]+'</td><td>'+f3(t[1].r2)+'</td><td>'+f2(t[1].mae)+'</td><td>'+f2(t[1].rmse)+'</td><td>'+(t[2]?(passOf(t[1].r2)?'<b style="color:var(--ok)">합격</b>':'<b style="color:var(--amber)">보류</b>'):'—')+'</td></tr>').join('')+'</tbody>';}

/* ── 면접 결과 발표 ── */
function ivBars(rows){return '<div class="bars">'+rows.map(r=>'<div class="row" style="grid-template-columns:minmax(0,1.3fr) minmax(50px,2fr) auto"><span style="text-align:left;font-family:var(--font)">'+r[0]+'</span><div class="tr"><i style="left:0;width:'+Math.max(0,r[1])*100+'%;background:'+r[3]+'"></i></div><span style="white-space:nowrap">'+f3(r[1])+' · '+f2(r[2])+'</span></div>').join('')+'</div><p class="foot">막대 = 시험 R², 오른쪽 숫자 = 시험 R² · MAE(MPa)</p>';}
function altBox(G,txt){const p=passOf(G.r2);return statGrid([[f3(G.r2),'시험 R²','a'],[f2(G.mae)+'MPa','시험 MAE','a'],[f3(MAN.r2),'합격선 (매뉴얼 씨)','m']],3)+'<p class="foot">'+txt+' 판정: <b>'+(p?/*i18n:verdict*/'합격':'보류')+'</b>.</p>';}
Interview('iv2',{title:'선형 회귀는 강도 예측 업무에 합격할까?',
  q:'조건: 봉인된 시험지 258건에서 시험 R²가 합격선(매뉴얼 씨 경험식 '+f3(MAN.r2)+')과 기준선(늘 훈련 평균, R² '+f3(BASE.r2)+')을 모두 넘으면 합격. 학습은 브라우저에서 특징 8개 그대로, 정규방정식으로 한다.',
  fly:'처음 보는 258건에서 평균 몇 MPa 틀려? 매뉴얼 씨 공식보다 나아?',robo:'외울 숫자 몇 개, 학습에 몇 ms?',
  async run(setP){setP(.3);await wait(40);const F=fitSet('m8');setP(1);const pass=passOf(F.r2);
    return {result:pass?'pass':'hold',flyAns:'R² '+f3(F.r2)+' · MAE '+f2(F.mae)+'MPa (매뉴얼 씨 '+f3(MAN.r2)+' · '+f2(MAN.mae)+')',roboAns:'숫자 9개 · 학습 '+Math.max(0.1,F.ms).toFixed(1)+'ms (브라우저)',flyMood:pass?'arms':'coffee',flyMove:pass?'nod':'shake',roboMood:'happy',roboMove:'bob',asstMood:pass?'spark':'sigh',docMood:pass?'judge':'scale',
      measure:'봉인된 시험지 결과예요. 선형 회귀 R² <b>'+f3(F.r2)+'</b>(MAE '+f2(F.mae)+'MPa), 매뉴얼 씨 경험식 '+f3(MAN.r2)+'('+f2(MAN.mae)+'MPa), 늘 평균만 말하는 기준선 '+f3(BASE.r2)+'('+f2(BASE.mae)+'MPa). 기준선은 넘었지만 합격선에 R² '+f3(MAN.r2-F.r2)+'이 모자랍니다.',
      verdict:pass?'합격입니다. 회귀팀에 배치합니다.':'보류입니다. 특징 8개를 그대로 쓰면 재령에 따른 강도의 곡선을 직선이 따라가지 못해, 교과서 상수 네 개짜리 경험식에 집니다. 잔차 그림의 활 모양이 그 증거예요. 아래 "조건 바꿔 다시 보기"에서 특징을 다듬어 줍니다.',
      key:'직선은 곡선을 못 따라간다 — R² '+f3(F.r2)+' < 합격선 '+f3(MAN.r2),
      render(box){box.innerHTML=ivBars([['선형 회귀 (특징 8개)',F.r2,F.mae,'var(--blue)'],['매뉴얼 씨 경험식',MAN.r2,MAN.mae,'var(--c-man)'],['기준선 (늘 평균)',Math.max(0,BASE.r2),BASE.mae,'var(--muted)']]);}};},
  alts:[
   {label:'특징 추가 (+ 물-결합재비 · log 재령)',async run(box){const G=fitSet('m10');box.innerHTML=altBox(G,'특징 두 개를 더하자 R²가 '+f3(fitSet('m8').r2)+' → '+f3(G.r2)+'로 올라 경험식을 넉넉히 넘는다. 외울 숫자는 11개. 사람의 지식(좋은 특징)과 데이터(가중치 학습)가 만나면 둘 다보다 낫다. 5-4 로지스틱 회귀와 같은 교훈의 회귀판이다.');}},
   {label:'매뉴얼 씨의 두 숫자만 (물-결합재비 · log 재령)',async run(box){const G=fitSet('two');box.innerHTML=altBox(G,'경험식과 같은 두 정보만 주고, 상수는 데이터에서 배우게 했다. R² '+f3(G.r2)+'로 경험식('+f3(MAN.r2)+')을 '+f3(G.r2-MAN.r2)+' 차이로 넘는다. 아슬아슬하다.');}},
   {label:'표준화 안 함 (날것 단위)',async run(box){const G=fitSet('m8',true);box.innerHTML=altBox(G,'표준화를 빼도 R²가 '+f3(G.r2)+'로 소수 셋째 자리까지 같다. 규제 없는 선형 회귀는 특징의 단위를 바꿔도 같은 직선을 긋는다. 바뀌는 것은 가중치의 숫자(06장)와 경사 하강의 속도(04장)뿐이다.');}}]});

/* 평가 장은 화면에 가까워지면 계산한다 */
(function(){const t=$('m5');let done=false;function go(){if(done)return;done=true;setTimeout(renderEval,10);}
  if('IntersectionObserver' in window){const io=new IntersectionObserver(es=>{if(es.some(e=>e.isIntersecting)){go();io.disconnect();}},{rootMargin:'600px'});io.observe(t);io.observe($('m6'));}else go();
  window.addEventListener('hashchange',go);if(location.hash)go();})();

/* ══ 확인 문제 ══ */
var QUIZ={
q1:[{q:'선형 회귀가 "가장 작게" 만들려는 것은?',o:[['잔차(실제 − 예측)를 제곱해 더한 값',1,'최소제곱이다. 평균으로 나누면 MSE다. 제곱이라 +와 −가 지워지지 않고, 큰 잔차를 더 크게 벌한다.'],['잔차를 그냥 더한 값',0,'+와 −가 서로 지워져 엉터리 직선도 0이 될 수 있다.'],['틀린 배합의 수',0,'그것은 분류의 오류 수다. 회귀에서는 "얼마나" 틀렸는지를 잰다.']]},
 {q:'기울기 0, 높이 = 훈련 평균인 직선의 R²는?',o:[['0',1,'R²는 "평균선보다 잔차 제곱합을 얼마나 줄였나"다. 평균선 자신은 하나도 줄이지 못했으니 0이다.'],['1',0,'1은 모든 배합을 정확히 맞혔을 때다.'],['0.5',0,'평균선과 비교한 비율이라 평균선은 정확히 0이다.']]},
 {q:'물-결합재비 하나로 강도를 맞히는 직선의 기울기 부호는?',o:[['음수',1,'물-결합재비가 클수록(물이 많을수록) 강도가 낮아진다.'],['양수',0,'그러면 물을 많이 넣을수록 강해진다는 뜻이다. 현장 경험과 반대다.'],['0',0,'0이면 물-결합재비가 강도와 아무 관계가 없다는 뜻이다.']]}],
q2:[{q:'이 면접에서 타겟(y)은?',o:[['압축 강도(MPa) 그 자체',1,'크기가 있는 숫자를 맞히므로 회귀다. 5-4는 같은 강도를 35MPa 문턱에 대어 이름표로 바꿨다.'],['35MPa 합격 여부',0,'그것은 분류팀(5-4)의 타겟이다. 같은 데이터, 다른 질문이다.'],['물-결합재비',0,'물-결합재비는 특징(파생 특징)이다.']]},
 {q:'규제 없는 선형 회귀에서 표준화를 빼면 시험 R²는?',o:[['똑같다',1,'단위를 바꾸면 가중치가 그만큼 반대로 바뀌어 같은 직선이 된다. 바뀌는 것은 가중치의 숫자와 경사 하강의 속도다.'],['크게 떨어진다',0,'KNN(5-1)이나 규제 모델(5-3)은 그렇지만, 규제 없는 선형 회귀의 예측은 단위와 무관하다.'],['오류가 난다',0,'날것 그대로도 잘 돈다.']]}],
q3:[{q:'AI가 준 코드로 돌렸더니 시험 R²가 1.000이 나왔다. 가장 먼저 의심할 것은?',o:[['정답 열(압축 강도)이 특징에 섞였다',1,'정답을 보고 정답을 맞히면 R²가 1에 가깝다. X의 열 수부터 센다(iloc[:, :9] 같은 실수).'],['선형 회귀가 아주 잘 맞는 데이터다',0,'이 데이터에서 선형 회귀는 0.57 안팎이다. 너무 좋은 점수는 먼저 의심한다.'],['표준화를 잘해서',0,'규제 없는 선형 회귀는 표준화로 점수가 바뀌지 않는다.']]},
 {q:'LinearRegression(normalize=True)에서 오류가 났다. 알맞은 대응은?',o:[['normalize를 지우고, 표준화가 필요하면 StandardScaler를 따로(훈련 데이터로 fit) 쓴다',1,'normalize는 1.2에서 없어졌다. 경고·오류는 버전이 바뀌었다는 신호다.'],['scikit-learn을 옛 버전으로 내린다',0,'당장은 돌지만 다른 라이브러리와 충돌하고, 문제를 미룰 뿐이다.'],['normalize=False로 바꾼다',0,'없어진 인자는 값과 상관없이 쓸 수 없다.']]}],
q4:[{q:'정규방정식과 경사 하강에 대한 설명으로 맞는 것은?',o:[['둘은 같은 MSE의 바닥(같은 가중치)에 닿는다',1,'MSE는 그릇 모양이라 바닥이 하나다. 정규방정식은 바로 계산하고, 경사 하강은 걸어서 간다.'],['경사 하강이 늘 더 정확한 답을 준다',0,'충분히 걸으면 같은 답이다. 덜 걸으면 오히려 덜 정확하다.'],['정규방정식은 학습률을 잘 골라야 한다',0,'정규방정식에는 학습률이 없다. 학습률은 경사 하강의 걸음 크기다.']]},
 {q:'특징을 표준화하지 않고 경사 하강을 하면 흔히 생기는 일은?',o:[['손실 지도가 길쭉한 골짜기가 되어 바닥까지 아주 오래 걸린다',1,'04장 그림처럼 한 방향은 가파르고 다른 방향은 평평해, 큰 학습률은 발산하고 작은 학습률은 느리다.'],['답이 정규방정식과 달라진다',0,'바닥은 같다. 도착이 느릴 뿐이다.'],['아무 차이 없다',0,'예측은 같아도 가는 길의 속도가 크게 다르다.']]},
 {q:'학습률을 너무 크게 잡으면?',o:[['바닥을 넘어 반대편으로 튀어 손실이 커지다 발산한다',1,'한 걸음이 그릇의 폭보다 크면 매번 더 높이 튕긴다.'],['더 빨리 정확히 수렴한다',0,'적당한 범위까지만 그렇다. 넘으면 발산한다.'],['과대적합이 일어난다',0,'과대적합은 학습률이 아니라 모델의 복잡도와 데이터 양의 문제다.']]}],
q5:[{q:'MAE 8.4MPa, RMSE 11.0MPa. 이 차이가 말하는 것은?',o:[['가끔 크게 틀리는 배합이 있다',1,'RMSE는 큰 잔차를 제곱으로 키워 평균하므로 큰 실수가 많을수록 MAE보다 커진다.'],['계산이 틀렸다',0,'RMSE ≥ MAE는 늘 성립한다.'],['모델이 과대적합했다',0,'과대적합은 훈련과 시험 점수의 차이로 본다.']]},
 {q:'잔차를 재령 축에 대어 그렸더니 아래-위-아래 활 모양이 나왔다. 가장 알맞은 해석과 대응은?',o:[['강도가 재령에 대해 곡선으로 오른다. log(재령) 같은 특징을 넣는다',1,'잔차에 무늬가 있으면 모델이 놓친 관계가 있다. log(재령) 하나로 R²가 0.57 → 0.84로 올랐다.'],['데이터에 오류가 많다',0,'무늬가 규칙적이면 오류가 아니라 모델이 놓친 모양이다.'],['시험지를 더 늘린다',0,'시험지 크기는 무늬를 없애지 못한다.']]},
 {q:'특징 묶음을 고를 때 시험지 R²를 보고 가장 좋은 묶음을 고르면?',o:[['시험지가 사실상 훈련 데이터가 되어 점수를 믿을 수 없다',1,'특징 선택도 학습의 일부다. 훈련 데이터 안에서 교차 검증으로 고르고, 시험지는 마지막에 한 번만 연다. 이 면접의 묶음은 매뉴얼 씨의 지식으로 미리 정했다.'],['가장 정확한 방법이다',0,'시험지에 맞춘 선택이라 처음 보는 데이터에서는 더 낮게 나온다.'],['상관없다',0,'평가의 공정성이 무너진다.']]}],
q6:[{q:'선형 회귀가 가장 어려워하는 일은?',o:[['재령에 따른 강도처럼 휘거나 포화하는 관계를 특징 없이 맞히기',1,'예측이 특징의 가중합이라 곡선은 특징을 바꿔 펴 주거나(log), 다항 특징(5-3)을 넣어야 한다.'],['특징이 클수록 꾸준히 커지는 숫자 맞히기',0,'가장 잘하는 일이다.'],['가중치로 예측의 이유를 설명하기',0,'선형 회귀의 강점이다.']]},
 {q:'10개 모델에서 재령의 가중치가 −로 뒤집혔다. 알맞은 해석은?',o:[['log(재령)과 함께 움직이는 특징이라 몫을 나눠 가졌다. 하나만 떼어 "오래 굳히면 약해진다"고 읽지 않는다',1,'서로 닮은 특징끼리는 가중치를 나눠 갖고 부호가 뒤집히기도 한다. 재령 전체의 효과는 두 특징을 함께 봐야 한다.'],['오래 굳히면 콘크리트가 약해진다',0,'현장 경험과 반대이고, 데이터도 그렇게 말하지 않는다.'],['계산 오류다',0,'정규방정식과 Colab 결과가 같다.']]}]};

/* ══ 공통 B·C·D·E·F 데이터 ══ */
var PROMPTS=[
 {t:'① 내 공정 데이터로 선형 회귀 코드 받기',use:'내 공장의 품질 데이터로 03장의 프롬프트를 맞출 때',p:MAIN_PROMPT},
 {t:'② AI 코드 검증 요청',use:'받은 코드를 다른 AI(또는 같은 AI의 새 대화)에게 검토받을 때',p:'아래 scikit-learn 선형 회귀 코드를 코드 리뷰해 줘. 특히 다음을 하나씩 확인하고, 문제가 있으면 줄 번호와 고친 코드를 보여 줘.\n1) 타겟 열이 특징 X에 섞이지 않았나 (열 수와 열 이름을 확인)\n2) 데이터를 나누기 전에 표준화·인코딩을 해서 시험 데이터 정보가 새지 않았나\n3) train_test_split에 stratify와 random_state가 있나\n4) 내 scikit-learn 버전(〔1.x〕)에 없는 인자(normalize, squared 등)를 쓰지 않았나\n5) 평가를 시험 데이터로 했나, R²와 함께 MAE·RMSE(단위 포함)를 보고했나, DummyRegressor와 비교했나\n6) 잔차 그림(잔차 vs 예측값, 잔차 vs 주요 특징)을 그렸나\n추측으로 답하지 말고, 확실하지 않은 부분은 "공식 문서 확인 필요"라고 표시해 줘.'},
 {t:'③ 잔차 그림 읽기 상담',use:'잔차에 무늬가 보이는데 어떤 특징을 만들어야 할지 모를 때',p:'선형 회귀의 시험 데이터 잔차를 특징별 구간으로 나눠 평균 낸 표를 아래에 붙인다. 타겟은 〔콘크리트 압축 강도(MPa)〕다.\n- 어떤 특징에서 잔차에 무늬(곡선, 깔때기, 계단)가 보이는지 짚어 줘\n- 무늬마다 의심되는 관계(포화, 로그, 상호작용, 문턱)와 넣어 볼 파생 특징을 2개씩 제안해 줘\n- 각 제안을 시험 데이터가 아니라 훈련 데이터 안의 교차 검증으로 확인하는 코드를 써 줘\n- 현장 지식(〔물-결합재비가 낮을수록 강하다, 재령에 따라 처음 빨리 오른다〕)과 맞는지도 적어 줘\n\n〔여기에 구간별 평균 잔차 표를 붙여 넣는다〕'},
 {t:'④ 수업용 문제 만들기',use:'이 편으로 수업할 때 확인 문제가 더 필요할 때',p:'선형 회귀 입문 수업용 문제를 만들어 줘. 대상은 〔제조 현장 실무자, 수식에 익숙하지 않음〕이다. 예시는 콘크리트 강도 예측으로 통일한다.\n- 잔차·최소제곱·MAE와 RMSE·R²·경사 하강을 각각 묻는 객관식 5문제 (보기 3개, 정답 해설 포함)\n- 오해를 드러내는 문제 2개: "R²가 높으면 MPa 오차도 작다", "가중치가 큰 특징이 원인이다"\n- 손으로 풀 수 있는 계산 문제 1개: 배합 4개의 실제·예측 강도로 MAE, RMSE, R² 구하기 (소수 둘째 자리)\n정답은 맨 끝에 따로 모아 줘.'}];
var GL=[
 ['모델','선형 회귀','특징에 가중치를 곱해 더하고 절편을 더한 값으로 숫자를 맞히는 회귀 모델. 잔차 제곱합을 가장 작게 하는 가중치를 고른다.'],
 ['모델','회귀','크기가 있는 숫자(강도 MPa 등)를 맞히는 일. 이름표를 맞히는 분류와 짝이다.'],
 ['모델','가중치','특징마다 곱하는 숫자. 그 특징이 1단위 늘 때 예측이 얼마나 바뀌는지를 뜻한다.'],
 ['모델','절편','모든 특징이 0(표준화했다면 평균)일 때의 예측값 b.'],
 ['모델','잔차','실제 값 − 예측 값. +면 덜 예측, −면 더 예측한 것이다.'],
 ['모델','최소제곱','잔차를 제곱해 더한 값이 가장 작은 가중치를 고르는 방법.'],
 ['모델','정규방정식','최소제곱의 답을 한 번에 주는 식 (XᵀX)β = Xᵀy. 손실의 기울기가 0인 조건에서 나온다.'],
 ['데이터','샘플','표의 한 줄. 여기서는 콘크리트 배합 하나(와 그 시편의 시험 결과).'],
 ['데이터','특징','모델이 입력으로 쓰는 열. 시멘트·고로 슬래그·플라이 애시·물·고성능 감수제·굵은 골재·잔골재·재령.'],
 ['데이터','타겟','맞히려는 정답 열. 이 면접에서는 압축 강도(MPa).'],
 ['데이터','파생 특징','기존 열로 계산해 새로 만든 특징. 물-결합재비, log(재령) 등.'],
 ['데이터','물-결합재비','물의 양 ÷ 결합재(시멘트 + 고로 슬래그 + 플라이 애시)의 양. 낮을수록 강도가 높아지는 경향이 있다.'],
 ['데이터','재령','콘크리트를 친 뒤 시험할 때까지 지난 날수. 강도는 처음에 빨리, 나중에 천천히 오른다.'],
 ['데이터','표준화','특징마다 평균을 빼고 표준편차로 나눠 단위를 맞추는 일. 훈련 데이터로만 맞춘다.'],
 ['데이터','데이터 누수','시험 데이터의 정보가 학습에 섞여 들어가는 일. 점수가 실제보다 좋게 나온다.'],
 ['데이터','정답 누설','타겟(또는 타겟에서 계산한 값)이 특징에 섞여 들어가는 일. 점수가 비현실적으로 높게 나온다.'],
 ['학습','경사 하강','손실이 줄어드는 방향(기울기의 반대)으로 가중치를 조금씩 옮기는 학습 방법.'],
 ['학습','학습률','경사 하강 한 걸음의 크기. 너무 크면 발산하고, 너무 작으면 느리다.'],
 ['학습','발산','학습률이 너무 커서 손실이 줄지 않고 점점 커지는 일.'],
 ['학습','과소적합','모델이 너무 단순해 훈련 데이터조차 잘 못 맞히는 일. 특징 8개 선형 회귀의 상태.'],
 ['학습','과대적합','훈련 데이터를 외워 처음 보는 데이터에서 성적이 떨어지는 일.'],
 ['평가','MSE','평균 제곱 오차. 잔차 제곱의 평균. 선형 회귀가 학습으로 줄이는 값.'],
 ['평가','MAE','평균 절대 오차. 잔차 크기의 평균. "평균 몇 MPa 틀리나".'],
 ['평가','RMSE','MSE의 제곱근. 타겟과 같은 단위이며, 큰 실수를 MAE보다 무겁게 센다.'],
 ['평가','R²','1 − (잔차 제곱합 ÷ 평균선의 잔차 제곱합). 0이면 평균만 말하는 것과 같고, 1이면 모두 맞힌 것이다.'],
 ['평가','기준선','배우지 않고도 낼 수 있는 점수. 회귀에서는 "늘 훈련 평균"이 흔한 기준선이다.'],
 ['평가','합격선','지원자가 넘어야 할 점수. 회귀팀은 매뉴얼 씨 경험식의 시험 R²다.']];
var ANALOGY=[
 ['가중합 (선형 모델)','면접 항목별 점수에 가중치를 곱해 더한 총점',2,'뉴런이 시냅스 세기만큼 입력을 더하는 것','실제 뉴런의 합산은 시간과 위치에 따라 달라지고, 덧셈만으로 설명되지 않는다'],
 ['잔차','예상 납기와 실제 납기의 차이를 하나하나 적은 장부',2,'예상한 보상과 실제 보상의 차이를 알리는 도파민 뉴런의 예측 오류 신호','예측 오류 신호는 보상에 대한 것이고, 뇌가 그 차이를 제곱해 더한다는 근거는 없다'],
 ['최소제곱','모든 지점까지 거리의 제곱합이 가장 작은 곳에 창고를 짓는 것',1,'','최소제곱에 대응하는 신경 계산은 알려지지 않았다'],
 ['경사 하강','안개 낀 산에서 발밑의 기울기만 보고 한 걸음씩 내려가는 것',2,'초파리 버섯체에서 도파민 신호가 시냅스 세기를 조금씩 바꾸는 학습','시냅스 변화는 국소 규칙이다. 전체 손실의 기울기를 계산한다는 근거는 없다'],
 ['특징 만들기 (log 재령)','"처음 한 달이 중요하다"는 선배의 요령을 계산표에 넣는 것',1,'','뇌가 시간을 로그 눈금으로 느낀다는 연구(베버-페히너)와 연결하기도 하지만, 이 모델의 log 특징과 같은 일이라는 근거는 없다'],
 ['R²','"평균만 말하는 신입보다 얼마나 나은가"로 매긴 점수',1,'','대응하는 신경 현상이 없다. 순수하게 평가용 숫자다']];
var REFS=[
 'Yeh, I.-C. (1998). Modeling of strength of high-performance concrete using artificial neural networks. <i>Cement and Concrete Research</i>, 28(12), 1797–1808. <a href="https://doi.org/10.1016/S0008-8846(98)00165-3" target="_blank" rel="noopener">doi:10.1016/S0008-8846(98)00165-3</a> — 실기 과제지 원출처',
 'Yeh, I.-C. (2007). Concrete Compressive Strength [Dataset]. UCI Machine Learning Repository. <a href="https://doi.org/10.24432/C5PK67" target="_blank" rel="noopener">doi:10.24432/C5PK67</a> (CC BY 4.0) — 02장 데이터',
 'Abrams, D. A. (1918). <i>Design of Concrete Mixtures</i> (Bulletin 1). Structural Materials Research Laboratory, Lewis Institute, Chicago. — 물-시멘트비 법칙, 02장 매뉴얼 씨의 경험식',
 'ACI Committee 209 (1992). <i>Prediction of Creep, Shrinkage, and Temperature Effects in Concrete Structures</i> (ACI 209R-92). American Concrete Institute. — 재령 보정 t / (4 + 0.85t), 02장',
 'Legendre, A.-M. (1805). <i>Nouvelles méthodes pour la détermination des orbites des comètes</i>. Paris: Firmin Didot. — 최소제곱법의 첫 출판, 01장',
 'Galton, F. (1886). Regression towards mediocrity in hereditary stature. <i>Journal of the Anthropological Institute of Great Britain and Ireland</i>, 15, 246–263. — "회귀"라는 이름의 유래',
 'Hastie, T., Tibshirani, R., &amp; Friedman, J. (2009). <i>The Elements of Statistical Learning</i> (2nd ed.), 3장. Springer. — 선형 회귀와 최소제곱, 04·06장',
 'James, G., Witten, D., Hastie, T., &amp; Tibshirani, R. (2021). <i>An Introduction to Statistical Learning</i> (2nd ed.), 3장. Springer. — 잔차 그림과 비선형 관계, 05장',
 'Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. <i>Journal of Machine Learning Research</i>, 12, 2825–2830. — A장 코드',
 'scikit-learn 문서. <a href="https://scikit-learn.org/stable/modules/linear_model.html" target="_blank" rel="noopener">Linear Models</a> · <a href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.root_mean_squared_error.html" target="_blank" rel="noopener">root_mean_squared_error</a> — normalize·squared 인자 삭제, 03·06장',
 'Schultz, W., Dayan, P., &amp; Montague, P. R. (1997). A neural substrate of prediction and reward. <i>Science</i>, 275(5306), 1593–1599. <a href="https://doi.org/10.1126/science.275.5306.1593" target="_blank" rel="noopener">doi:10.1126/science.275.5306.1593</a> — D장 잔차 비유(예측 오류 신호)',
 'Hige, T., Aso, Y., Modi, M. N., Rubin, G. M., &amp; Turner, G. C. (2015). Heterosynaptic plasticity underlies aversive olfactory learning in <i>Drosophila</i>. <i>Neuron</i>, 88(5), 985–998. — D장 경사 하강 비유(도파민과 시냅스 변화)'];
var TIP={who:'머신러닝을 처음 배우는 대학생·직장인·제조 현장 실무자. 5-0 채용 공고를 먼저 보면 좋다. 수식은 ŷ = w·x + b, MSE, R² 세 개만 나온다',
 goal:'① 잔차와 최소제곱을 01장 그림으로 설명한다 ② MAE·RMSE·R²를 구별해 읽고, 기준선·합격선과 비교한다 ③ 정규방정식과 경사 하강이 같은 답에 닿는 까닭과 표준화·학습률의 역할을 말한다 ④ 잔차의 무늬를 보고 좋은 특징(log 재령)을 제안한다',
 time:'읽기 25분 · 실습(01·04·05장 그림) 20분 · 틀린 곳 찾기 10분 · Colab 15분',
 plans:[['강의형 50분','00장 지원서 1장을 인쇄해 나눠 주고 9칸을 함께 읽기 10분 → 01장에서 학생들이 손으로 직선을 맞춰 R²를 겨루고 "최소제곱으로 찾기"로 확인 10분 → 04장 경사 하강 재생(표준화 함/안 함, 학습률 크게) 10분 → 05장 잔차 그림에서 활 모양 찾기 10분 → 면접 결과를 먼저 맞히고 발표, "조건 바꿔 다시 보기" 10분.'],['실무 OJT 90분','03장 프롬프트로 각자 AI에게 코드를 받는다 20분 → 체크리스트로 서로의 코드를 바꿔 검사 20분 → Colab에서 실행하고 A장 결과와 비교 20분 → B장 ③ 프롬프트로 잔차 표를 읽히고, 우리 공정 데이터에서 넣어 볼 파생 특징을 발표 30분.'],['1:1 튜터링','지원서 1장의 아홉 칸 중 막히는 칸을 고르게 하고, 그 칸의 "→ 장"으로 바로 간다. 01장에서 배합 하나를 눌러 잔차와 제곱을 손으로 다시 계산하게 한다.']],
 myths:[['R²가 높으면 MPa 오차도 작다','R²는 단위가 없는 비율이다. 강도가 넓게 퍼진 데이터에서는 R²가 높아도 MAE가 클 수 있다. MAE를 함께 본다'],['R²는 0과 1 사이다','시험 데이터에서는 음수도 나온다. 평균만 말하는 것보다 못하다는 뜻이다'],['가중치가 큰 특징이 원인이다','단위에 따라 크기가 달라지고(표준화 전후), 서로 닮은 특징끼리는 가중치를 나눠 갖거나 부호가 뒤집힌다'],['선형 회귀는 곡선을 못 다룬다','모델은 가중치에 대해 선형이다. log(재령)처럼 특징을 바꾸면 곡선 관계도 맞힌다'],['경사 하강이 정규방정식보다 정확하다','같은 바닥에 닿는다. 경사 하강은 특징이 아주 많거나 데이터가 아주 클 때 쓰는 다른 길이다'],['선형 회귀도 표준화해야 점수가 오른다','규제 없는 선형 회귀는 표준화해도 예측이 같다. 표준화는 가중치 비교와 경사 하강 속도를 위해 한다']],
 tasks:[['과제 1 · 손으로 직선 맞추기 대회','01장에서 특징 하나를 골라 슬라이더만으로 R²를 최대한 올리고, 최소제곱 직선과의 MSE 차이를 적는다. 왜 사람 눈은 큰 잔차를 덜 무겁게 보는지 한 줄로 쓴다.'],['과제 2 · 틀린 곳 찾기 만들기','AI에게 일부러 실수가 섞인 선형 회귀 코드를 받고(B장 ②를 거꾸로 활용), 짝에게 찾게 한다. 실수마다 "왜 틀렸나"를 한 줄로 쓴다.'],['과제 3 · 잔차 탐정','A장 코드 6번 칸의 재령 구간별 잔차 표를 물-결합재비, 시멘트, 고성능 감수제 구간으로도 만들어 무늬를 찾는다. B장 ③ 프롬프트로 파생 특징을 받아 교차 검증으로 확인한다.'],['과제 4 · 경험식과 겨루기','매뉴얼 씨 경험식의 상수 네 개(96.5, 7, 4, 0.85) 중 하나를 바꾸면 R²가 어떻게 되는지 보고, 같은 꼴의 식을 데이터로 맞추려면 어떤 특징을 선형 회귀에 넣어야 하는지(log 강도 = log A − 물-결합재비·log B …) 적는다.']],
 ai:'학생이 AI에게 코드를 받는 것은 막지 말고 과정에 넣는다. 다만 ① 프롬프트를 제출하게 하고 ② 체크리스트 아홉 항목을 직접 표시하게 하고 ③ 실행 결과를 기준선·합격선과 비교하고 잔차 그림을 붙이게 하면, "받아 적기"가 아니라 "검증하기"를 배운다. AI 답의 함수·인자는 공식 문서와 버전으로 확인하게 하고, 과제에 AI 도움을 받았으면 사용 범위를 밝히게 한다.'};
var CODE={title:'선형 회귀 면접 — 최소제곱부터 합격 판정까지',sub:'numpy, pandas, scikit-learn, ucimlrepo · CPU 몇 초',file:'linear_regression_interview',
 desc:'이 편의 면접을 그대로 옮겼다. UCI에서 콘크리트 데이터를 받아 시험지를 봉인하고(1·2번 칸), 기준선과 매뉴얼 씨의 경험식으로 합격선을 세우고(3번), 특징 8개 선형 회귀(4번), 정규방정식과 경사 하강이 같은 답에 닿는지(5번), 재령 구간별 잔차의 무늬(6번), 좋은 특징(7번), 표준화 전후 비교(8번), 합격 판정(9번)까지 간다. 노트북으로 저장하면 칸마다 셀이 나뉜다.',
 cap:'실행 결과 (scikit-learn 1.9, CPU). Colab의 scikit-learn 버전이 1.4보다 낮으면 RMSE를 직접 계산한다. 브라우저 계산과 숫자가 같다(특징 8개 R² 0.567 · MAE 8.44, 매뉴얼 씨 0.735 · 6.72, 특징 추가 0.842 · 5.37, 판정 보류).'};
