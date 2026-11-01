/* ══ 5-4 로지스틱 회귀 · 콘크리트 실기 과제지 ══════════════════════════ */
const ROWS=concRows(),TR=ROWS.filter(r=>!r.test),TE=ROWS.filter(r=>r.test);
const fv=r=>FEAT.map(f=>r[f]);
const SC=ML.scaler(TR.map(fv)),A=SC.t(TR.map(fv)),B=SC.t(TE.map(fv));
const YTR=TR.map(r=>r.pass),YTE=TE.map(r=>r.pass),GTR=TR.map(r=>r.grade),GTE=TE.map(r=>r.grade);
const PNAME=['불합격','합격'];
const dot=(w,x)=>w.reduce((s,v,j)=>s+v*x[j],0);
function binModel(X,y,C,iters){const st=ML.binFit(X,y,{C:C,iters:iters||2000});st.p=x=>ML.sigmoid(st.b+dot(st.w,x));st.predict=(x,t)=>st.p(x)>=(t==null?.5:t)?1:0;return st;}
let MB=null,MBms=0,MG=null;
function mainBin(){if(!MB){const t0=performance.now();MB=binModel(A,YTR,1,2000);MBms=performance.now()-t0;}return MB;}
function main3(){return MG||(MG=ML.softmaxLR(A,GTR,3,{C:1,iters:1500}));}
function prf(yt,yp,K){const M=ML.confusion(yt,yp,K);return [...Array(K)].map((_,k)=>{let tp=M[k][k],col=0,row=0;for(let a=0;a<K;a++){col+=M[a][k];row+=M[k][a];}
  const P=col?tp/col:0,R=row?tp/row:0;return {k,P,R,F:(P+R)?2*P*R/(P+R):0,n:row};});}
function baseAcc(){const p=YTR.reduce((a,v)=>a+v,0)/YTR.length,maj=p>=.5?1:0;return {k:maj,acc:YTE.filter(v=>v===maj).length/YTE.length};}
const manualTest=TE.filter(r=>manualRule(r)===r.pass).length/TE.length;
function cmGrid(M,labels,colsFor){const C=cols(),K=labels.length;let h='<div class="cm" style="grid-template-columns:86px repeat('+K+',minmax(48px,1fr));max-width:'+(110+K*90)+'px"><div class="h">정답 ＼ 예측</div>'+labels.map(k=>'<div class="h">'+k+'</div>').join('');
  M.forEach((row,a)=>{h+='<div class="h" style="justify-content:end;text-align:right">'+labels[a]+'</div>'+row.map((v,b)=>a===b?'<div class="c'+(v?' d':'')+'" style="'+(v?'background:'+rgba(colsFor(a),.85):'')+'">'+v+'</div>':'<div class="c'+(v?' x':'')+'">'+(v||'·')+'</div>').join('');});return h+'</div>';}

/* ── 히어로 ── */
heroRoom($('heroCv'),{sym:'σ',no:'APP-04',name:'로지스틱\n회귀',role:'지원자 · 분류팀',banner:'모델 채용 위원회 · 5-4 면접',doc:'front',asst:'board',fly:'clip',robo:'calc',loop:10,
  beats:[{"at": 0.4, "who": "asst", "mood": "board", "move": "nod", "say": "합격선 76.7%"}, {"at": 2.4, "who": "doc", "mood": "magic", "move": "pop", "say": "점수를 확률로"}, {"at": 4.4, "who": "fly", "mood": "point", "move": "tilt", "say": "왜 합격이야?"}, {"at": 6.4, "who": "robo", "mood": "count", "move": "bob", "say": "숫자 9개!"}, {"at": 8.4, "who": "doc", "mood": "front", "move": "stand"}],
  motif(ctx,t,W,H,C){ctx.strokeStyle=C.c_app;ctx.lineWidth=3;ctx.beginPath();for(let i=0;i<=40;i++){const z=-6+12*i/40,x=272+i*3,y=150-60/(1+Math.exp(-z+Math.sin(t)*0.8));i?ctx.lineTo(x,y):ctx.moveTo(x,y);}ctx.stroke();
    ctx.fillStyle=C.muted;ctx.font=font(12,500,1);ctx.textAlign='left';ctx.fillText('p = σ(z)',300,165);}});
$('heroLine').textContent=pct(PASSLINE);

/* ── 01 장면 ── */
comic('comicIntro',[{"bg": "office", "prop": "z = w·x + b", "cast": [{"who": "doc", "mood": "board", "x": 30, "move": "nod"}, {"who": "asst", "mood": "board", "x": 72}], "say": {"who": "doc", "txt": "먼저 특징마다 가중치를 곱해 더한 점수를 내요. 여기까지는 선형 회귀와 같아요."}, "cap": "① 점수 — 선형 결합"}, {"bg": "board", "prop": "σ(z)", "cast": [{"who": "doc", "mood": "magic", "x": 30, "move": "pop"}, {"who": "fly", "mood": "point", "x": 72, "move": "tilt", "flip": true}], "say": {"who": "fly", "txt": "점수가 37이면 합격 확률 37%야? 100을 넘으면 어쩔 건데?"}, "cap": "② 시그모이드 — 점수를 0~1 확률로 누른다"}, {"bg": "site", "prop": "0.5", "cast": [{"who": "robo", "mood": "count", "x": 30, "move": "bob"}, {"who": "asst", "mood": "spark", "x": 72, "move": "jump"}], "say": {"who": "robo", "txt": "문턱을 0.5보다 올리면 미달 배합을 합격으로 내보내는 일이 줄어. 레미콘 회수비 절약!"}, "cap": "③ 문턱 — 확률을 판정으로 바꾼다"}]);

/* ── 01 시그모이드 직접 만지기: 물-결합재비 하나로 합격 확률 ── */
(function(){const cv=$('sigCv'),L=TR.map(r=>r.wb),Y=TR.map(r=>r.pass);
  const st={mid:0.65,w:6,lin:false,probe:0.45,anim:0};const H=300,X0=0.22,X1=0.95;
  const lin=(function(){const n=L.length,mx=ML.mean(L),my=ML.mean(Y);let sxy=0,sxx=0;for(let i=0;i<n;i++){sxy+=(L[i]-mx)*(Y[i]-my);sxx+=(L[i]-mx)*(L[i]-mx);}const a=sxy/sxx;return {a,b:my-a*mx};})();
  const p=x=>ML.sigmoid(-st.w*(x-st.mid));
  function loss(){let s=0;for(let i=0;i<L.length;i++){const q=p(L[i]);s-=Y[i]?Math.log(Math.max(q,1e-12)):Math.log(Math.max(1-q,1e-12));}return s/L.length;}
  function acc(){let c=0;for(let i=0;i<L.length;i++)if((p(L[i])>=.5?1:0)===Y[i])c++;return c/L.length;}
  function draw(){const W=fit(cv,H),ctx=cv.getContext('2d'),C=cols(),pl=46,pr=14,pt=18,pb=40,gw=W-pl-pr,gh=H-pt-pb;const sx=x=>pl+(x-X0)/(X1-X0)*gw,sy=v=>pt+(1-v)*gh;
    ctx.clearRect(0,0,W,H);ctx.font=font(12);ctx.fillStyle=C.muted;ctx.strokeStyle=C.rule;ctx.lineWidth=1;
    for(let x=0.3;x<=0.91;x+=0.1){ctx.beginPath();ctx.moveTo(sx(x),pt);ctx.lineTo(sx(x),pt+gh);ctx.stroke();ctx.textAlign='center';ctx.fillText(x.toFixed(1),sx(x),H-22);}
    [0,0.5,1].forEach(v=>{ctx.textAlign='right';ctx.fillText(v,pl-8,sy(v)+4);});ctx.textAlign='center';ctx.fillText('물-결합재비 (물 ÷ 결합재)',pl+gw/2,H-4);
    ctx.save();ctx.translate(12,pt+gh/2);ctx.rotate(-Math.PI/2);ctx.fillText('35MPa 합격 확률 p',0,0);ctx.restore();
    ctx.setLineDash([5,4]);ctx.strokeStyle=C.muted;ctx.beginPath();ctx.moveTo(pl,sy(0.5));ctx.lineTo(pl+gw,sy(0.5));ctx.moveTo(sx(st.mid),pt);ctx.lineTo(sx(st.mid),pt+gh);ctx.stroke();ctx.setLineDash([]);
    ctx.fillStyle=rgba(C.blue,.06);ctx.fillRect(pl,pt,sx(st.mid)-pl,gh);
    ctx.fillStyle=C.blue;ctx.font=font(12,700);ctx.textAlign='right';ctx.fillText('← 합격으로 판정',Math.max(sx(st.mid)-8,pl+100),sy(0.66));ctx.textAlign='left';ctx.fillStyle=C.muted;ctx.fillText('불합격으로 판정 →',Math.min(sx(st.mid)+8,pl+gw-110),sy(0.34));
    if(st.lin){ctx.strokeStyle=C.amber;ctx.lineWidth=2.5;ctx.setLineDash([7,5]);ctx.beginPath();ctx.moveTo(sx(X0),sy(lin.a*X0+lin.b));ctx.lineTo(sx(X1),sy(lin.a*X1+lin.b));ctx.stroke();ctx.setLineDash([]);
      ctx.fillStyle=C.amber;ctx.font=font(12,700);ctx.textAlign='left';ctx.fillText('선형 회귀(점선): 0.9에서 '+fx(lin.a*0.9+lin.b),sx(0.66),sy(lin.a*0.9+lin.b)+4);}
    ctx.strokeStyle=C.blue;ctx.lineWidth=3.5;ctx.beginPath();for(let k=0;k<=200;k++){const x=X0+(X1-X0)*k/200;k?ctx.lineTo(sx(x),sy(p(x))):ctx.moveTo(sx(x),sy(p(x)));}ctx.stroke();
    const rng=RNG(5);for(let i=0;i<L.length;i++){const j=(rng()-.5)*22;circ(ctx,sx(L[i]),sy(Y[i])+(Y[i]?j+12:j-12),3,rgba(Y[i]?C.blue:C.text,.55),null);}
    const q=p(st.probe);ctx.strokeStyle=C.stamp;ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(sx(st.probe),pt+gh);ctx.lineTo(sx(st.probe),sy(q));ctx.lineTo(pl,sy(q));ctx.stroke();circ(ctx,sx(st.probe),sy(q),6,C.stamp,C.panel,2);
    ctx.fillStyle=C.stamp;ctx.font=font(12,700,1);ctx.textAlign='left';ctx.fillText('p = '+q.toFixed(2),sx(st.probe)+8,sy(q)-8);info();}
  function info(){const q=p(st.probe),z=-st.w*(st.probe-st.mid);
    $('sigInfo').innerHTML='물-결합재비 <b>'+st.probe.toFixed(2)+'</b>인 배합: 점수 z = −'+st.w.toFixed(1)+' × ('+st.probe.toFixed(2)+' − '+st.mid.toFixed(2)+') = <b>'+fx(z)+'</b> → 합격 확률 σ(z) = <b>'+q.toFixed(3)+'</b> → '+(q>=0.5?'<b>합격</b>':'<b>불합격</b>')+'으로 판정. '+
      '훈련 배합 772건의 평균 손실(교차 엔트로피) <b>'+loss().toFixed(3)+'</b> · 정확도 <b>'+pct(acc())+'</b>. 그림을 누르면 그 물-결합재비의 배합을 재 본다.'+
      (st.lin?'<br><span style="color:var(--amber)">점선(선형 회귀)은 물-결합재비 0.25에서 '+fx(lin.a*0.25+lin.b)+', 0.9에서 '+fx(lin.a*0.9+lin.b)+'을 낸다. 1보다 크거나 0보다 작은 "확률"은 없다.</span>':'');}
  function sync(){$('sgMid').value=st.mid;$('sgMidO').textContent=st.mid.toFixed(2);$('sgW').value=st.w;$('sgWO').textContent=st.w.toFixed(1);}
  $('sgMid').addEventListener('input',e=>{st.mid=+e.target.value;sync();draw();});
  $('sgW').addEventListener('input',e=>{st.w=+e.target.value;sync();draw();});
  $('sgLin').addEventListener('click',e=>{st.lin=!st.lin;e.target.setAttribute('aria-pressed',st.lin);draw();});
  $('sgReset').addEventListener('click',()=>{cancelAnimationFrame(st.anim);st.mid=0.65;st.w=6;st.probe=0.45;sync();draw();});
  cv.addEventListener('click',e=>{const r=cv.getBoundingClientRect(),x=X0+(e.clientX-r.left-46)/(r.width-60)*(X1-X0);st.probe=clamp(x,X0,X1);draw();});
  /* 학습: z = a·u + c (u = 표준화한 물-결합재비), 규제 C=1. 결과를 경계·가파름으로 바꿔 보여 준다 */
  $('sgFit').addEventListener('click',()=>{cancelAnimationFrame(st.anim);const mu=ML.mean(L),sd=Math.sqrt(ML.mean(L.map(v=>(v-mu)*(v-mu))));let a=-st.w*sd,c=a*(mu-st.mid)/sd,it=0;const n=L.length,lam=1/n;
    (function go(){for(let s=0;s<(REDUCE?600:8);s++){let ga=0,gc=0;for(let i=0;i<n;i++){const u=(L[i]-mu)/sd,q=ML.sigmoid(a*u+c);ga+=(q-Y[i])*u/n;gc+=(q-Y[i])/n;}ga+=lam*a;a-=1.0*ga;c-=1.0*gc;it++;}
      st.w=clamp(-a/sd,1,40);st.mid=clamp(mu-c*sd/a,0.25,0.85);sync();draw();if(it<600)st.anim=requestAnimationFrame(go);else $('sigInfo').innerHTML+='<br><b>학습 완료</b>: 경계 물-결합재비 '+st.mid.toFixed(2)+', 가파름 '+st.w.toFixed(1)+'. 0편 매뉴얼 씨의 문턱 0.5와 비교해 보자. 재령을 모르니 정확도는 70% 안팎에서 멈춘다.';})();});
  sync();draw();onRedraw(draw);})();

/* ── 02 실기 과제지 ── */
(function(){const xs=$('dxSel'),ys=$('dySel'),OPT=FEAT.concat(['wb','strength']),LAB=Object.assign({},FEATK,{wb:'물-결합재비',strength:'압축 강도(MPa)'});
  OPT.forEach(f=>{xs.add(new Option(LAB[f],f));ys.add(new Option(LAB[f],f));});xs.value='wb';ys.value='strength';
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
  $('concTbl').innerHTML='<table><thead><tr><th>#</th>'+FEAT.map(f=>'<th><span class="tg x">특징 X</span>'+FEATK[f]+'</th>').join('')+'<th>압축 강도<br>(MPa)</th><th><span class="tg t">타겟 y</span>35MPa 합격</th><th><span class="tg t">타겟 y</span>등급</th><th>용도</th></tr></thead><tbody>'+
    ROWS.map(r=>'<tr><td>'+(r.i+1)+'</td>'+FEAT.map(f=>'<td>'+r[f]+'</td>').join('')+'<td>'+r.strength+'</td><td class="y">'+PNAME[r.pass]+'</td><td class="y">'+GRADES[r.grade]+'</td><td>'+(r.test?'<span class="pill no">봉인</span>':'훈련')+'</td></tr>').join('')+'</tbody></table>';
  const r0=TR[0],z0=SC.t([fv(r0)])[0];
  $('stdTbl').innerHTML='<thead><tr><th>특징</th><th>평균</th><th>표준편차</th><th>예: 훈련 첫 배합</th></tr></thead><tbody>'+FEAT.map((f,j)=>'<tr><td>'+FEATK[f]+'</td><td>'+SC.mu[j].toFixed(1)+'</td><td>'+SC.sd[j].toFixed(1)+'</td><td>'+r0[f]+' → <b>'+fx(z0[j])+'</b></td></tr>').join('')+'</tbody>';})();

/* ── 03 AI 비서 ── */
const MAIN_PROMPT=`너는 머신러닝 수업 조교다. Google Colab에서 바로 실행되는 파이썬 코드를 써 줘.

[환경] Colab 기본 라이브러리(numpy, pandas, scikit-learn)와 ucimlrepo만 쓴다. 첫 줄에 scikit-learn 버전을 출력한다.
[데이터] 〔ucimlrepo의 fetch_ucirepo(id=165)〕 로 콘크리트 압축 강도 데이터를 읽는다.
  - 특징 X: 〔시멘트, 고로 슬래그, 플라이 애시, 물, 고성능 감수제, 굵은 골재, 잔골재, 재령〕 8개 (열 순서대로 이름을 붙인다)
  - 타겟 y: 〔압축 강도가 35MPa 이상이면 1, 아니면 0〕 (이진 분류)
[규정]
  1) 강도를 〔25, 45〕MPa로 나눈 3등급으로 stratify 해서 train_test_split(test_size=0.25, random_state=42)로 먼저 나눈다.
  2) StandardScaler는 훈련 데이터로만 fit 하고, 시험 데이터는 transform만 한다.
  3) LogisticRegression(C=〔1〕, max_iter=1000)으로 학습한다. 내 scikit-learn 버전에서 없어진 인자(multi_class 등)는 쓰지 않는다.
[출력] 훈련·시험 정확도, 혼동행렬, classification_report, 특징별 가중치(coef_), 시험 데이터 앞 3개의 합격 확률(predict_proba).
[금지] 시험 데이터로 학습하거나 하이퍼파라미터를 고르지 않는다. 데이터를 지어내지 않는다.
[형식] 셀 단위로 나누고, 각 셀 위에 무엇을 하는지 한국어 주석을 단다. 코드 뒤에 내가 확인해야 할 점 3가지를 적어 줘.`;
$('mainPrompt').textContent=MAIN_PROMPT;$('copyMain').addEventListener('click',()=>copyText(MAIN_PROMPT,'프롬프트를 복사했습니다'));
(function(){const LINES=[
 ['import pandas as pd'],['from ucimlrepo import fetch_ucirepo'],['from sklearn.model_selection import train_test_split'],['from sklearn.preprocessing import StandardScaler'],['from sklearn.linear_model import LogisticRegression'],[''],
 ['uci = fetch_ucirepo(id=165)',0,'UCI에서 콘크리트 데이터를 실제로 받았다. 괜찮다.'],
 ['X = uci.data.features',0,'특징 여덟 열이다. 괜찮다.'],['y = (uci.data.targets.iloc[:, 0] >= 35).astype(int)   # 35MPa 합격 = 1',0,'강도를 문턱에 대어 합격 이름표를 만들었다. 괜찮다.'],[''],
 ['X = StandardScaler().fit_transform(X)   # 표준화',1,'<b>데이터 누수.</b> 나누기 전에 전체(시험지 포함)로 평균·표준편차를 구했다. 시험지 정보가 훈련에 스며든다. 먼저 나누고, 훈련 데이터로만 fit 한 뒤 시험지는 transform만 한다.'],
 ['X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)',2,'<b>stratify와 random_state가 없다.</b> 돌릴 때마다 점수가 달라지고, 고강도 배합이 한쪽으로 쏠릴 수 있다. 위원회 규정대로 강도 등급으로 <code>stratify</code>하고 <code>random_state=42</code>를 넣는다.'],[''],
 ['model = LogisticRegression(multi_class="multinomial",   # 분류 설정',3,'<b>없어진 인자.</b> <code>multi_class</code>는 scikit-learn 1.5에서 예고된 뒤 1.8에서 없어졌다. 최신 버전에서는 오류가 난다. 지운다(클래스가 셋 이상이면 알아서 소프트맥스로 학습한다).'],
 ['                           C=0.001)   # 과대적합을 막으려고 규제를 약하게',4,'<b>C의 방향이 거꾸로다.</b> C는 규제의 역수라서 0.001은 규제를 "아주 세게" 거는 값이다. 주석과 반대다. 이 데이터에서는 정확도가 합격선 근처까지 떨어진다(05장 C 그래프).'],
 ['model.fit(X_train, y_train)',0,'훈련 데이터로 학습했다. 괜찮다.'],[''],
 ['pred = model.predict(X_train)',5,'<b>훈련 데이터로 채점했다.</b> 외운 문제로 시험 본 셈이다. <code>model.predict(X_test)</code>와 <code>y_test</code>로 채점한다.'],
 ['print("정확도:", (pred == y_train).mean())',5,'<b>훈련 데이터로 채점했다.</b> 바로 윗줄과 같은 문제다. 시험지로 채점해야 처음 보는 배합에서의 실력이 나온다.']];
  const host=$('bugCode'),found=new Set();
  host.innerHTML=LINES.map((l,i)=>'<div class="ln" role="listitem" tabindex="0" data-i="'+i+'"><span class="no">'+(i+1)+'</span><span>'+hl(l[0]||' ')+'</span></div>').join('');
  function score(){$('bugScore').textContent=found.size+' / 5';if(found.size===5){$('bugScore').className='pill ok';}}
  function pick(el){const i=+el.getAttribute('data-i'),l=LINES[i];if(!l[0])return;
    if(l[1]){found.add(l[1]);host.querySelectorAll('.ln').forEach(x=>{if(LINES[+x.getAttribute('data-i')][1]===l[1])x.classList.add('hit');});$('bugMsg').innerHTML=(i+1)+'번 줄: '+l[2]+(found.size===5?'<br><b>다섯 군데를 모두 찾았다.</b> 고친 코드는 A장의 Colab 코드와 같다.':'');}
    else{el.classList.add('miss');$('bugMsg').innerHTML=(i+1)+'번 줄: '+(l[2]||'이 줄은 괜찮다.');}score();}
  host.addEventListener('click',e=>{const el=e.target.closest('.ln');if(el)pick(el);});
  host.addEventListener('keydown',e=>{if((e.key==='Enter'||e.key===' ')&&e.target.classList.contains('ln')){e.preventDefault();pick(e.target);}});
  $('bugReveal').addEventListener('click',()=>{LINES.forEach((l,i)=>{if(l[1]){found.add(l[1]);host.querySelector('[data-i="'+i+'"]').classList.add('hit','reveal');}});score();$('bugMsg').innerHTML='빨간 줄 다섯 군데: ① 11번 데이터 누수 ② 12번 나누기 규정 ③ 14번 없어진 인자 ④ 15번 C의 방향 ⑤ 18~19번 훈련 데이터로 채점.';});
  $('bugReset').addEventListener('click',()=>{found.clear();host.querySelectorAll('.ln').forEach(x=>x.classList.remove('hit','miss','reveal'));$('bugScore').className='pill';score();$('bugMsg').textContent='줄을 하나 눌러 보세요.';});score();})();
(function(){const CK=[['데이터를 실제로 읽었나','AI가 예제 데이터를 지어내거나 다른 데이터로 바꾸지 않았는지, 행이 1,030개인지 본다.'],['특징 열과 타겟 열이 내가 말한 것과 같나','압축 강도(정답)가 특징에 섞여 들어가면 정답을 보고 푸는 셈이다(정답 누설).'],['나누기를 먼저 하고, 표준화는 훈련 데이터로만 fit 했나','fit_transform(X) 전체가 보이면 누수다.'],['stratify와 random_state를 넣었나','등급 비율이 시험지에도 유지되고, 점수가 매번 같게 나온다.'],['시험지(X_test)는 마지막 채점에만 썼나','C를 고르려고 시험지를 여러 번 보면 시험지가 훈련 데이터가 된다. 고를 때는 교차 검증을 쓴다.'],['인자가 내 scikit-learn 버전에 있나','print(sklearn.__version__), 경고(FutureWarning)를 읽는다. multi_class, penalty는 최신 버전에서 바뀌었다.'],['C의 방향을 맞게 썼나','C가 클수록 규제가 약하다. 주석과 값이 맞는지 본다.'],['어느 쪽이 1인지 확인했나','model.classes_ 순서가 predict_proba 열 순서다. 합격(1)의 확률은 두 번째 열이다.'],['결과를 기준선·합격선과 비교했나','"늘 불합격"(DummyClassifier)과 매뉴얼 씨(76.7%)보다 낫지 않으면 뽑을 까닭이 없다.']];
  const ul=$('checks');ul.innerHTML=CK.map((c,i)=>'<li><input type="checkbox" id="ck'+i+'"><label for="ck'+i+'"><b>'+c[0]+'</b><small>'+c[1]+'</small></label></li>').join('');
  function upd(){const n=ul.querySelectorAll('input:checked').length;ul.querySelectorAll('li').forEach(li=>li.classList.toggle('done',li.querySelector('input').checked));$('checkScore').textContent='확인 '+n+' / '+CK.length+(n===CK.length?' — 이제 돌려 본다. 결과도 확인 대상이다.':'');}
  ul.addEventListener('change',upd);upd();})();

/* ── 04 수습 기간: 특징 2개로 35MPa 합격을 가르는 경사 하강 ── */
(function(){const cv=$('trainCv'),lc=$('lossCv'),H=320;
  const SETS={expert:{nx:'물-결합재비',ny:'재령(일, 로그 눈금)',fx:r=>r.wb,fy:r=>Math.log10(r.age),logy:true},raw:{nx:'시멘트(kg/m³)',ny:'물(kg/m³)',fx:r=>r.cement,fy:r=>r.water,logy:false}};
  const st={set:'expert',w:[0,0],b:0,t:0,hist:[],play:false,sel:-1,data:null};
  function setSet(k){st.set=k;const S=SETS[k],raw=TR.map(r=>[S.fx(r),S.fy(r)]),sc=ML.scaler(raw);
    st.data={X:sc.t(raw),y:YTR.slice(),raw,sc};const xs=raw.map(r=>r[0]),ys=raw.map(r=>r[1]),px=(Math.max(...xs)-Math.min(...xs))*.04,py=(Math.max(...ys)-Math.min(...ys))*.06;
    st.box={x0:Math.min(...xs)-px,x1:Math.max(...xs)+px,y0:Math.min(...ys)-py,y1:Math.max(...ys)+py};reset();}
  function lam(){const c=$('trC').value;return c==='inf'?0:1/(+c*st.data.X.length);}
  function lr(){return Math.pow(10,+$('trLr').value);}
  function reset(){st.w=[0,0];st.b=0;st.t=0;st.sel=-1;st.hist=[ML.binLoss({w:st.w,b:st.b},st.data.X,st.data.y,lam())];draw();}
  const zOf=(s1,s2)=>st.w[0]*s1+st.w[1]*s2+st.b;
  const toS=(u,v)=>{const sc=st.data.sc;return [(u-sc.mu[0])/sc.sd[0],(v-sc.mu[1])/sc.sd[1]];};
  function step(){ML.binStep(st,st.data.X,st.data.y,lr(),lam());st.hist.push(ML.binLoss(st,st.data.X,st.data.y,lam()));}
  function draw(){const W=fit(cv,H),ctx=cv.getContext('2d'),C=cols(),pl=50,pr=10,pt=10,pb=38,gw=W-pl-pr,gh=H-pt-pb,bx=st.box,S=SETS[st.set];
    const sx=x=>pl+(x-bx.x0)/(bx.x1-bx.x0)*gw,sy=y=>pt+(1-(y-bx.y0)/(bx.y1-bx.y0))*gh,ix=X=>bx.x0+(X-pl)/gw*(bx.x1-bx.x0),iy=Y=>bx.y0+(1-(Y-pt)/gh)*(bx.y1-bx.y0);
    ctx.clearRect(0,0,W,H);const cs=8;
    for(let X=pl;X<pl+gw;X+=cs)for(let Y=pt;Y<pt+gh;Y+=cs){const s=toS(ix(X+cs/2),iy(Y+cs/2)),q=ML.sigmoid(zOf(s[0],s[1]));ctx.fillStyle=q>=.5?rgba(C.blue,.05+.22*(q-.5)*2):rgba(C.text,.04+.12*(.5-q)*2);ctx.fillRect(X,Y,cs,cs);}
    ctx.strokeStyle=C.rule;ctx.fillStyle=C.muted;ctx.font=font(12);ctx.lineWidth=1;ctx.strokeRect(pl,pt,gw,gh);
    for(let k=0;k<=4;k++){const gx=bx.x0+(bx.x1-bx.x0)*k/4;ctx.textAlign='center';ctx.fillText(gx.toFixed(gx<5?2:0),sx(gx),H-20);}
    if(S.logy){[1,3,7,28,90,365].forEach(d=>{const v=Math.log10(d);if(v<bx.y0||v>bx.y1)return;ctx.textAlign='right';ctx.fillText(d,pl-5,sy(v)+4);ctx.strokeStyle=rgba(C.muted,.25);ctx.beginPath();ctx.moveTo(pl,sy(v));ctx.lineTo(pl+gw,sy(v));ctx.stroke();});}
    else for(let k=0;k<=4;k++){const gy=bx.y0+(bx.y1-bx.y0)*k/4;ctx.textAlign='right';ctx.fillText(gy.toFixed(0),pl-5,sy(gy)+4);}
    ctx.textAlign='center';ctx.fillText(S.nx,pl+gw/2,H-3);ctx.save();ctx.translate(11,pt+gh/2);ctx.rotate(-Math.PI/2);ctx.fillText(S.ny,0,0);ctx.restore();
    if(Math.abs(st.w[0])+Math.abs(st.w[1])>1e-9&&Math.abs(st.w[1])>1e-9){const sc=st.data.sc,pts=[];for(let k=0;k<=60;k++){const u=bx.x0+(bx.x1-bx.x0)*k/60,s1=(u-sc.mu[0])/sc.sd[0],s2=-(st.w[0]*s1+st.b)/st.w[1];pts.push([sx(u),sy(s2*sc.sd[1]+sc.mu[1])]);}
      ctx.save();ctx.beginPath();ctx.rect(pl,pt,gw,gh);ctx.clip();ctx.strokeStyle=C.blue;ctx.lineWidth=3;ctx.beginPath();pts.forEach((p,i)=>i?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]));ctx.stroke();ctx.restore();}
    st.data.raw.forEach((r,i)=>{const s=st.data.X[i],q=ML.sigmoid(zOf(s[0],s[1])),yy=st.data.y[i],l=yy?-Math.log(Math.max(q,1e-9)):-Math.log(Math.max(1-q,1e-9));
      circ(ctx,sx(r[0]),sy(r[1]),3,yy?C.blue:C.text,null);if(l>0.69){ctx.strokeStyle=rgba(C.stamp,.75);ctx.lineWidth=Math.min(3.5,(l-0.69)*2+0.8);ctx.beginPath();ctx.arc(sx(r[0]),sy(r[1]),5.5,0,7);ctx.stroke();}
      if(i===st.sel){ctx.strokeStyle=C.stamp;ctx.lineWidth=2;ctx.beginPath();ctx.arc(sx(r[0]),sy(r[1]),12,0,7);ctx.stroke();}});
    ctx.font=font(12,700);ctx.textAlign='left';const lx=pl+8;ctx.fillStyle=rgba(C.panel,.85);ctx.fillRect(lx-4,pt+4,170,50);ctx.fillStyle=C.text;ctx.fillText('● 불합격 (클래스 0)',lx,pt+18);ctx.fillStyle=C.blue;ctx.fillText('● 합격 35MPa↑ (클래스 1)',lx,pt+33);ctx.fillStyle=C.stamp;ctx.fillText('○ 빨간 테두리 = 틀린 배합',lx,pt+48);
    drawLoss();info();cv._sx=sx;cv._sy=sy;}
  function drawLoss(){const W=fit(lc,240),ctx=lc.getContext('2d'),C=cols(),pl=44,pr=10,pt=12,pb=30,gw=W-pl-pr,gh=240-pt-pb,h=st.hist,n=Math.max(50,h.length),mx=Math.max(0.75,...h);
    ctx.clearRect(0,0,W,240);ctx.strokeStyle=C.rule;ctx.strokeRect(pl,pt,gw,gh);ctx.fillStyle=C.muted;ctx.font=font(12);ctx.textAlign='right';[0,mx/2,mx].forEach(v=>ctx.fillText(v.toFixed(2),pl-5,pt+gh-(v/mx)*gh+4));
    ctx.textAlign='center';ctx.fillText('반복 횟수 (0 ~ '+(n-1)+')',pl+gw/2,240-8);ctx.save();ctx.translate(11,pt+gh/2);ctx.rotate(-Math.PI/2);ctx.fillText('평균 손실',0,0);ctx.restore();
    ctx.strokeStyle=C.stamp;ctx.lineWidth=2.5;ctx.beginPath();h.forEach((v,i)=>{const x=pl+i/(n-1)*gw,y=pt+gh-Math.min(1,v/mx)*gh;i?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.stroke();
    ctx.fillStyle=C.stamp;ctx.font=font(12,700,1);ctx.textAlign='right';ctx.fillText('L = '+h[h.length-1].toFixed(3),pl+gw-6,pt+16);}
  function info(){const X=st.data.X,y=st.data.y,S=SETS[st.set];let c=0;X.forEach((s,i)=>{if((ML.sigmoid(zOf(s[0],s[1]))>=.5?1:0)===y[i])c++;});
    let h='반복 <b>'+st.t+'</b>번 · 평균 손실 <b>'+st.hist[st.hist.length-1].toFixed(3)+'</b> · 훈련 정확도 <b>'+pct(c/X.length)+'</b> (772건) · w₁('+S.nx.replace(/\(.*\)/,'')+') = '+fx(st.w[0])+', w₂('+S.ny.replace(/\(.*\)/,'')+') = '+fx(st.w[1])+', b = '+fx(st.b);
    if(st.sel>=0){const s=X[st.sel],q=ML.sigmoid(zOf(s[0],s[1])),yy=y[st.sel],r=TR[st.sel];h+='<br>고른 배합: 실제 강도 <b>'+r.strength+'MPa</b>('+PNAME[yy]+'), 물-결합재비 '+r.wb.toFixed(2)+'·재령 '+r.age+'일·시멘트 '+r.cement+'kg → z = '+fx(zOf(s[0],s[1]))+', P(합격) = <b>'+q.toFixed(3)+'</b>, 손실 = −log('+(yy?'p':'1−p')+') = <b>'+(yy?-Math.log(Math.max(q,1e-12)):-Math.log(Math.max(1-q,1e-12))).toFixed(3)+'</b>';}
    else if(st.t===0)h+='<br>처음에는 w와 b가 모두 0이라 모든 배합의 확률이 0.5다. 손실은 log 2 ≈ 0.693에서 시작한다. ▶ 재생을 눌러 보자.';
    else if(st.t>200)h+=st.set==='expert'?'<br>전문가 특징 두 개로 훈련 정확도가 85% 근처까지 간다. 경계선이 "물-결합재비가 낮고 오래 굳힌" 왼쪽 위를 합격으로 가른다. 매뉴얼 씨의 직각 규칙을 비스듬한 직선 하나로 바꾼 셈이다.':'<br>시멘트와 물만으로는 70% 근처에서 멈춘다. 재령(얼마나 굳혔나)을 모르면 같은 배합도 7일째와 90일째 강도가 달라 겹칠 수밖에 없다. 손실이 0으로 가지 않는 것이 그 증거다.';
    $('trainInfo').innerHTML=h;}
  function chips(k){document.querySelectorAll('#loopChips button').forEach((b,i)=>{b.classList.toggle('on',i===k);b.classList.toggle('done',k>=0&&i<k);});}
  let raf=0,ph=0;
  function frame(){if(!st.play)return;ph=(ph+1)%8;chips(Math.floor(ph/2));const k=st.t<40?1:st.t<200?3:8;for(let i=0;i<k;i++)step();draw();if(st.t>=1500){stop();return;}raf=requestAnimationFrame(frame);}
  function stop(){st.play=false;cancelAnimationFrame(raf);$('trPlay').textContent='▶ 재생';chips(-1);}
  $('trPlay').addEventListener('click',()=>{if(st.play){stop();return;}st.play=true;$('trPlay').textContent='❚❚ 멈춤';if(REDUCE){for(let i=0;i<400;i++)step();draw();stop();return;}raf=requestAnimationFrame(frame);});
  $('trStep').addEventListener('click',async()=>{stop();for(let k=0;k<4;k++){chips(k);await wait(REDUCE?0:160);}step();draw();chips(-1);});
  $('trReset').addEventListener('click',()=>{stop();reset();});
  $('trLr').addEventListener('input',()=>{$('trLrO').textContent=lr().toFixed(2);});
  $('trC').addEventListener('change',()=>{st.hist.push(ML.binLoss(st,st.data.X,st.data.y,lam()));draw();});
  seg('#m4 .choice button',v=>{stop();setSet(v);});
  cv.addEventListener('click',e=>{const r=cv.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top;let best=-1,bd=300;st.data.raw.forEach((p,i)=>{const d=(cv._sx(p[0])-mx)**2+(cv._sy(p[1])-my)**2;if(d<bd){bd=d;best=i;}});st.sel=best;draw();});
  $('trLrO').textContent=lr().toFixed(2);setSet('expert');onRedraw(draw);})();

/* ── 05 성과 평가 ── */
function renderEval(){const m=mainBin(),pte=B.map(x=>m.predict(x)),ptr=A.map(x=>m.predict(x)),te=ML.acc(YTE,pte),tr=ML.acc(YTR,ptr),bs=baseAcc(),C=cols();
  $('evalStat').innerHTML=statGrid([[pct(te),'시험 정확도 ('+Math.round(te*258)+' / 258건)','a'],[pct(tr),'훈련 정확도 (772건)'],[pct(PASSLINE),'합격선 · 5-0 매뉴얼 씨','m'],[pct(bs.acc),'기준선 · 늘 '+rago('"'+PNAME[bs.k]+'"')+'만 답할 때']],4);
  const M=ML.confusion(YTE,pte,2);$('cm7').innerHTML=cmGrid(M,PNAME,k=>k?C.blue:C.muted);
  const sc=prf(YTE,pte,2);$('prTbl').innerHTML='<thead><tr><th>이름표</th><th>시험 수</th><th>정밀도</th><th>재현율</th><th>F1</th></tr></thead><tbody>'+sc.map(s=>'<tr><td>'+PNAME[s.k]+'</td><td>'+s.n+'</td><td>'+s.P.toFixed(3)+'</td><td>'+s.R.toFixed(3)+'</td><td>'+s.F.toFixed(3)+'</td></tr>').join('')+'</tbody>';
  const fp=M[0][1],fn=M[1][0];
  $('hiddenTxt').innerHTML='258건 중 '+Math.round(te*258)+'건을 맞혀 정확도는 <b>'+pct(te)+'</b>다. 틀린 '+(fp+fn)+'건은 두 종류다. <b style="color:var(--stamp)">불합격 배합을 합격이라 한 '+fp+'건</b>은 강도가 모자란 콘크리트가 현장으로 나가는 실수다. 구조물의 안전과 회수 비용이 걸린다. <b>합격 배합을 불합격이라 한 '+fn+'건</b>은 멀쩡한 배합을 버리거나 다시 시험하는 비용이다. 두 실수의 값이 다르면 정확도 하나로 판정하면 안 된다. 아래 임계값 실험에서 "합격"이라고 말하는 문턱을 올려 보자.';
  /* 과제 B: 3등급 */
  const m3=main3(),p3=B.map(m3.predict),a3=ML.acc(GTE,p3),s3=prf(GTE,p3,3);
  $('cm3').innerHTML=cmGrid(ML.confusion(GTE,p3,3),GRADES,k=>GRADECOL[k]);
  $('g3Stat').innerHTML=statGrid([[pct(a3),'3등급 시험 정확도','a'],[pct(s3[2].R,0),'고강도 재현율 (실제 고강도 중 찾은 비율)','s']],2)+'<p class="foot">기준선(늘 "보통 강도")은 '+pct(GTE.filter(v=>v===1).length/GTE.length)+'다. 등급이 셋이 되면 직선 경계 두 개로 가르는 셈이라 어려워진다. 고강도를 보통으로 낮춰 부르는 실수가 가장 많다.</p>';
  const sel=$('mixPick');sel.innerHTML='';TE.forEach((r,i)=>sel.add(new Option('시험 '+(i+1)+' · 실제 '+r.strength+'MPa('+GRADES[r.grade]+') · 재령 '+r.age+'일'+(p3[i]!==GTE[i]?' · 틀림':''),i)));
  const firstMiss=p3.findIndex((v,i)=>v!==GTE[i]);sel.value=firstMiss>=0?firstMiss:0;
  function bars(){const i=+sel.value,p=m3.proba(B[i]),top=p.indexOf(Math.max(...p));$('probBars').innerHTML=p.map((v,k)=>'<div class="row"><span style="text-align:left;font-family:var(--font)">'+GRADES[k]+(k===GTE[i]?' <b style="color:var(--ok)">정답</b>':'')+'</span><div class="tr"><i style="left:0;width:'+(v*100)+'%;background:'+GRADECOL[k]+';'+(k===top?'':'opacity:.45')+'"></i></div><span>'+v.toFixed(3)+'</span></div>').join('')+'<p class="foot">세 확률의 합은 1이다(소프트맥스). 가장 큰 막대가 예측이다'+(top!==GTE[i]?' — 이 배합은 틀렸다. 정답 등급의 확률이 몇이었는지 보자.':'.')+'</p>';}
  sel.onchange=bars;bars();
  /* 임계값 */
  const pr=B.map(m.p),R=ML.roc(YTE,pr),rc=$('rocCv');
  function th(){const t=+$('thR').value;$('thO').textContent=t.toFixed(2);let tp=0,fp=0,tn=0,fn=0;pr.forEach((p,i)=>{const g=p>=t?1:0,y=YTE[i];if(g&&y)tp++;else if(g&&!y)fp++;else if(!g&&y)fn++;else tn++;});
    const acc=(tp+tn)/YTE.length,prec=(tp+fp)?tp/(tp+fp):0,rec=(tp+fn)?tp/(tp+fn):0;
    $('th2x2').innerHTML=cmGrid([[tn,fp],[fn,tp]],PNAME,k=>k?cols().blue:cols().muted);
    $('thStat').innerHTML=statGrid([[pct(acc),'정확도'],[(tp+fp)?pct(prec):'—','정밀도 ("합격" 중 진짜)'],[String(fp),'강도 미달 출하 (건)','s']],3);
    drawRoc((fp+tn)?fp/(fp+tn):0,rec);
    $('thInfo').innerHTML=t>=0.65?'임계값 '+t.toFixed(2)+': "합격"이라고 말하기가 까다로워졌다. 강도 미달 출하가 <b>'+fp+'건</b>으로 줄고 정밀도가 '+pct(prec)+'로 오른다. 대신 진짜 합격 배합 '+fn+'건을 불합격으로 돌려 재시험 비용이 든다(재현율 '+pct(rec)+'). 안전이 걸린 출하 판정이라면 이쪽이 맞다.':
      t<=0.35?'임계값 '+t.toFixed(2)+': 합격 배합은 거의 놓치지 않지만(재현율 '+pct(rec)+'), 강도 미달 배합 <b>'+fp+'건</b>이 합격 도장을 받는다. 정확도가 0.5일 때와 비슷해 보여도 틀림의 종류가 위험한 쪽으로 옮겨 갔다.':
      '임계값 '+t.toFixed(2)+': 정확도 '+pct(acc)+', 강도 미달 출하 '+fp+'건, 재현율 '+pct(rec)+'. 임계값은 모델이 아니라 <b>업무</b>가 정한다. ROC 곡선 아래 넓이(AUC) '+R.auc.toFixed(3)+'는 임계값과 상관없는 "순서 매기기 실력"이다. 0.5면 동전 던지기, 1이면 완벽이다.';}
  function drawRoc(fx_,ty){const W=fit(rc,260),ctx=rc.getContext('2d'),C=cols(),s=Math.min(W-60,210),pl=44,pt=10;ctx.clearRect(0,0,W,260);
    ctx.strokeStyle=C.rule;ctx.strokeRect(pl,pt,s,s);ctx.setLineDash([4,4]);ctx.beginPath();ctx.moveTo(pl,pt+s);ctx.lineTo(pl+s,pt);ctx.stroke();ctx.setLineDash([]);
    ctx.strokeStyle=C.blue;ctx.lineWidth=2.5;ctx.beginPath();R.pts.forEach((p,i)=>{const x=pl+p[0]*s,y=pt+s-p[1]*s;i?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.stroke();
    circ(ctx,pl+fx_*s,pt+s-ty*s,7,C.stamp,C.panel,2);ctx.fillStyle=C.muted;ctx.font=font(12);ctx.textAlign='center';ctx.fillText('거짓 양성 비율 (미달을 합격이라 함)',pl+s/2,pt+s+30);
    ctx.save();ctx.translate(12,pt+s/2);ctx.rotate(-Math.PI/2);ctx.fillText('재현율',0,0);ctx.restore();ctx.fillStyle=C.blue;ctx.font=font(13,700,1);ctx.textAlign='left';ctx.fillText('AUC = '+R.auc.toFixed(3),pl+s*0.4,pt+s*0.8);
    ctx.fillStyle=C.muted;ctx.font=font(11);ctx.fillText('0',pl-10,pt+s+12);ctx.fillText('1',pl+s-4,pt+s+12);ctx.fillText('1',pl-12,pt+8);}
  $('thR').oninput=th;th();onRedraw(th);
  /* 규제 C */
  const CS=[0.001,0.01,0.1,1,10,100,1000],RES={1:{tr,te,w:m.w.reduce((a,v)=>a+Math.abs(v),0)/8}};
  $('cChoice').innerHTML=CS.map(c=>'<button data-v="'+c+'" aria-pressed="'+(c===1)+'">C = '+c+'</button>').join('');
  const cc=$('cCv');
  function drawC(cur){const W=fit(cc,240),ctx=cc.getContext('2d'),C=cols(),pl=44,pr=14,pt=14,pb=34,gw=W-pl-pr,gh=240-pt-pb,lx=c=>pl+(Math.log10(c)+3)/6*gw,ly=v=>pt+(1-(v-0.7)/0.25)*gh;
    ctx.clearRect(0,0,W,240);ctx.strokeStyle=C.rule;ctx.strokeRect(pl,pt,gw,gh);ctx.fillStyle=C.muted;ctx.font=font(12);ctx.textAlign='center';CS.forEach(c=>ctx.fillText(c,lx(c),240-16));ctx.fillText('C (로그 눈금) — 왼쪽일수록 규제가 세다',pl+gw/2,240-2);
    ctx.textAlign='right';[0.7,0.8,0.9].forEach(v=>{ctx.fillText(pct(v,0),pl-5,ly(v)+4);});
    ctx.setLineDash([5,4]);ctx.strokeStyle=rgba(C.c_man,.9);ctx.beginPath();ctx.moveTo(pl,ly(PASSLINE));ctx.lineTo(pl+gw,ly(PASSLINE));ctx.stroke();ctx.setLineDash([]);ctx.fillStyle=C.c_man;ctx.textAlign='left';ctx.fillText('합격선 '+pct(PASSLINE),pl+gw-110,ly(PASSLINE)+14);
    [['tr',C.muted,'훈련'],['te',C.blue,'시험']].forEach(([k,col,lab])=>{const pts=CS.filter(c=>RES[c]).map(c=>[lx(c),ly(RES[c][k])]);ctx.strokeStyle=col;ctx.lineWidth=2.5;ctx.beginPath();pts.forEach((p,i)=>i?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]));ctx.stroke();pts.forEach(p=>circ(ctx,p[0],p[1],4.5,col,C.panel,1.5));
      if(pts.length){const l=pts[0];ctx.fillStyle=col;ctx.font=font(12,700);ctx.textAlign='left';ctx.fillText(lab,l[0]+8,l[1]+(k==='tr'?14:-8));}});
    if(cur!=null){ctx.strokeStyle=C.stamp;ctx.lineWidth=1.5;ctx.beginPath();ctx.moveTo(lx(cur),pt);ctx.lineTo(lx(cur),pt+gh);ctx.stroke();}}
  async function pickC(c){if(!RES[c]){$('cInfo').textContent='C = '+c+'로 학습 중…';await wait(20);const mm=binModel(A,YTR,c,c<=0.01?1000:2000);RES[c]={tr:ML.acc(YTR,A.map(x=>mm.predict(x))),te:ML.acc(YTE,B.map(x=>mm.predict(x))),w:mm.w.reduce((a,v)=>a+Math.abs(v),0)/8};}
    const r=RES[c];drawC(c);$('cInfo').innerHTML='C = <b>'+c+'</b>: 훈련 '+pct(r.tr)+' · 시험 <b>'+pct(r.te)+'</b> · 가중치 크기 평균 '+r.w.toFixed(2)+'. '+(c<=0.001?'규제가 너무 세서 가중치가 0 근처에 눌렸다. 시험 정확도가 합격선 바로 위까지 내려간다(과소적합).':c<=0.01?'규제가 세다. 가중치가 줄면서 정확도도 조금 떨어진다.':c>=10?'규제를 더 약하게 해도 거의 변하지 않는다. 특징 8개에 배합 772건이라 직선 하나로는 외울 여지가 적기 때문이다. 다른 데이터에서는 C가 크면 과대적합이 보이기도 한다.':c===1?'기본값이다. 단, C를 시험지 점수를 보고 고르면 반칙이다. 실제로는 훈련 데이터 안에서 교차 검증으로 고른다.':'C를 하나씩 눌러 두 선의 모양을 완성해 보자.');}
  $('cChoice').addEventListener('click',e=>{const b=e.target.closest('button');if(!b)return;$('cChoice').querySelectorAll('button').forEach(x=>x.setAttribute('aria-pressed',x===b));pickC(+b.getAttribute('data-v'));});
  pickC(1);onRedraw(()=>{const b=$('cChoice').querySelector('[aria-pressed="true"]');drawC(b?+b.getAttribute('data-v'):1);});
  /* 인사 카드: 가중치 */
  const WS=[['합격 (과제 A)',m.w]].concat(GRADES.map((g,k)=>[g+' (과제 B)',m3.W[k]]));
  $('wChoice').innerHTML=WS.map((w,k)=>'<button data-v="'+k+'" aria-pressed="'+(k===0)+'">'+w[0]+'</button>').join('');
  function wb(k){const w=WS[k][1],mx=Math.max(...w.map(Math.abs));$('wBars').innerHTML=FEAT.map((f,j)=>{const v=w[j],Wd=Math.abs(v)/mx*50;return '<div class="row"><span style="text-align:left;font-family:var(--font)">'+FEATK[f]+'</span><div class="tr"><span class="z"></span><i class="'+(v<0?'neg':'')+'" style="left:'+(v<0?50-Wd:50)+'%;width:'+Wd+'%"></i></div><span>'+fx(v)+'</span></div>';}).join('');
    const ord=w.map((v,j)=>[v,j]).sort((a,b)=>Math.abs(b[0])-Math.abs(a[0])),lab=WS[k][0].split(' (')[0];
    $('wInfo').innerHTML='<b>'+lab+'</b> 쪽으로 점수 z를 가장 크게 미는 특징은 '+FEATK[FEAT[ord[0][1]]]+'('+(ord[0][0]>0?'+':'−')+')와 '+FEATK[FEAT[ord[1][1]]]+'('+(ord[1][0]>0?'+':'−')+')다. 파란 막대는 그 특징이 클수록 '+lab+' 확률을 올리고, 빨간 막대는 내린다. '+(k===0?'재령과 시멘트가 +, 물이 −인 것은 현장 경험칙("물이 적고 오래 굳히면 강하다")과 같은 방향이다. ':'')+
      '<span class="muted">주의: 서로 함께 움직이는 특징(물과 고성능 감수제, 골재 두 가지)은 가중치를 나눠 갖거나 부호가 뒤집힐 수 있다. 가중치 하나만 떼어 "원인"이라고 읽지 않는다.</span>';}
  seg('#wChoice button',v=>wb(+v));wb(0);}

/* ── 면접 결과 발표 ── */
Interview('iv4',{title:'로지스틱 회귀는 35MPa 합격 판정 업무에 합격할까?',
  q:'조건: 봉인된 시험지 258건에서 합격선(매뉴얼 씨 '+pct(PASSLINE)+')과 기준선(늘 다수 쪽)을 모두 넘으면 합격. 학습은 브라우저에서 특징 8개, C = 1로 한다.',
  fly:'처음 보는 258건에서도 되나? 미달 배합을 합격이라 한 건 몇 건이야?',robo:'외울 숫자 몇 개, 학습에 몇 초?',
  async run(setP){setP(.3);await wait(40);const m=mainBin();setP(.8);await wait(40);const pte=B.map(x=>m.predict(x)),te=ML.acc(YTE,pte),tr=ML.acc(YTR,A.map(x=>m.predict(x))),bs=baseAcc(),M=ML.confusion(YTE,pte,2);setP(1);
    const pass=te>PASSLINE&&te>bs.acc;
    return {result:pass?'pass':'hold',flyAns:'시험 '+pct(te)+' · 훈련 '+pct(tr)+' · 미달인데 "합격" '+M[0][1]+'건',roboAns:'숫자 9개 · 학습 '+Math.max(1,Math.round(MBms))+'ms (브라우저)',flyMood:pass?'arms':'coffee',flyMove:pass?'nod':'shake',roboMood:'happy',roboMove:'jump',asstMood:pass?'spark':'sigh',docMood:pass?'judge':'scale',
      measure:'봉인된 시험지 결과예요. 로지스틱 회귀 <b>'+pct(te)+'</b>, 매뉴얼 씨 '+pct(PASSLINE)+', 늘 '+rago('"'+PNAME[bs.k]+'"')+'만 답하는 기준선 '+pct(bs.acc)+'. 틀린 '+(M[0][1]+M[1][0])+'건 중 '+M[0][1]+'건이 미달 배합을 합격이라 한 쪽이에요.',
      verdict:pass?'합격입니다. 분류팀에 배치합니다. 조건이 하나 있어요. 출하 판정에 쓸 때는 임계값을 0.5보다 올려 강도 미달 출하를 줄이고, 그 기준을 품질 담당과 함께 정합니다.':'보류입니다. 합격선을 넘지 못했어요.',
      key:'확률로 답하는 직선 — 숫자 9개로 합격선 '+pct(PASSLINE)+' → '+pct(te),
      render(box){box.innerHTML='<div class="bars">'+[['로지스틱 회귀',te,'var(--blue)'],['매뉴얼 씨 (5-0)',PASSLINE,'var(--c-man)'],['기준선 (늘 '+PNAME[bs.k]+')',bs.acc,'var(--muted)']].map(r=>'<div class="row"><span style="text-align:left;font-family:var(--font)">'+r[0]+'</span><div class="tr"><i style="left:0;width:'+r[1]*100+'%;background:'+r[2]+'"></i></div><span>'+pct(r[1])+'</span></div>').join('')+'</div>';}};},
  alts:[
   {label:'전문가 특징 둘만 (물-결합재비 · log 재령)',async run(box){const f=r=>[r.wb,Math.log10(r.age)],sc=ML.scaler(TR.map(f)),A2=sc.t(TR.map(f)),B2=sc.t(TE.map(f)),m=binModel(A2,YTR,1,2000),te=ML.acc(YTE,B2.map(x=>m.predict(x)));
     box.innerHTML=statGrid([[pct(te),'시험 정확도','a'],[pct(PASSLINE),'합격선','m']],2)+'<p class="foot">특징 여덟 개 대신 매뉴얼 씨가 쓰던 두 숫자만 줬더니 오히려 더 잘한다. 사람의 지식(좋은 특징)과 데이터(가중치 학습)가 만나면 둘 다보다 낫다. 판정: '+(te>PASSLINE?/*i18n:verdict*/'합격':'보류')+'.</p>';}},
   {label:'규제 아주 세게 (C = 0.001)',async run(box){const m=binModel(A,YTR,0.001,1000),te=ML.acc(YTE,B.map(x=>m.predict(x)));box.innerHTML=statGrid([[pct(te),'시험 정확도','a'],[pct(PASSLINE),'합격선','m']],2)+'<p class="foot">가중치를 세게 누르면 직선이 제자리를 못 찾아 합격선 바로 위까지 내려온다. 판정: '+(te>PASSLINE?'아슬아슬하게 합격':'보류')+'. 근무 조건(C)이 지원자의 실력을 바꾼다.</p>';}},
   {label:'출하 기준 강화 (임계값 0.7)',async run(box){const m=mainBin(),g=B.map(x=>m.predict(x,0.7)),M=ML.confusion(YTE,g,2),te=ML.acc(YTE,g),prec=M[1][1]/(M[1][1]+M[0][1]);
     box.innerHTML=statGrid([[pct(te),'시험 정확도','a'],[String(M[0][1]),'강도 미달 출하 (건)','s'],[pct(prec),'"합격" 중 진짜 합격']],3)+'<p class="foot">정확도는 내려가도 위험한 실수가 줄어든다. 정확도만 보면 이 근무 방식이 손해처럼 보이지만, 출하 판정에서는 이쪽이 낫다. 판정: '+(te>PASSLINE?'합격선도 넘는다.':'정확도로는 합격선 아래 — 무엇을 기준으로 뽑을지는 업무가 정한다.')+'</p>';}}]});

/* 평가 장은 화면에 가까워지면 계산한다 */
(function(){const t=$('m5');let done=false;function go(){if(done)return;done=true;setTimeout(renderEval,10);}
  if('IntersectionObserver' in window){const io=new IntersectionObserver(es=>{if(es.some(e=>e.isIntersecting)){go();io.disconnect();}},{rootMargin:'600px'});io.observe(t);io.observe($('m6'));}else go();
  window.addEventListener('hashchange',go);if(location.hash)go();})();

/* ══ 확인 문제 ══ */
var QUIZ={
q1:[{q:'로지스틱 회귀가 내놓는 σ(z)의 뜻으로 가장 알맞은 것은?',o:[['클래스 1(35MPa 합격)일 확률 P(y=1 | x)',1,'시그모이드는 어떤 점수든 0과 1 사이로 바꾼다. 그 값을 클래스 1일 확률로 읽는다.'],['콘크리트의 예상 강도(MPa)',0,'강도처럼 크기에 뜻이 있는 숫자를 맞히는 일은 회귀다(5-1~5-3). 로지스틱 회귀는 이름과 달리 분류 모델이다.'],['맞힌 배합의 수',0,'그것은 평가 지표(정확도)를 셀 때 쓰는 값이다.']]},
 {q:'선형 점수 z = 0인 배합의 합격 확률은?',o:[['0.5',1,'σ(0) = 1 / (1 + e⁰) = 1/2. 그래서 z = 0인 곳이 곧 결정 경계(임계값 0.5)다.'],['0',0,'z가 −∞로 갈 때 0에 가까워진다.'],['1',0,'z가 +∞로 갈 때 1에 가까워진다.']]},
 {q:'물-결합재비 하나로 합격 확률을 낼 때 가중치 w의 부호는?',o:[['음수',1,'물-결합재비가 클수록(물이 많을수록) 약해지므로, 커질수록 합격 확률이 내려가야 한다.'],['양수',0,'그러면 물을 많이 넣을수록 합격 확률이 올라간다. 현장 경험과 반대다.'],['0',0,'0이면 물-결합재비가 확률에 아무 영향도 주지 않는다.']]}],
q2:[{q:'이 면접(과제 A)에서 타겟(y)은?',o:[['압축 강도가 35MPa 이상인지(합격/불합격)',1,'압축 강도를 문턱에 대어 만든 이름표다. 시멘트·물·재령 등 여덟 열이 특징(X)이다.'],['압축 강도(MPa) 그 자체',0,'그것은 회귀팀(5-1~5-3)의 타겟이다. 같은 데이터, 다른 질문이다.'],['재령(일)',0,'이 면접에서 재령은 특징이다.']]},
 {q:'표준화의 평균과 표준편차는 어디서 구해야 하나?',o:[['훈련 데이터에서만',1,'시험지는 처음 보는 데이터 역할이다. 시험지의 통계를 미리 쓰면 데이터 누수다.'],['훈련 + 시험 전체에서',0,'시험지 정보가 훈련에 스며든다. 03장 틀린 곳 찾기의 첫 번째 실수다.'],['시험 데이터에서만',0,'학습할 때 쓸 수 없는 정보다.']]}],
q3:[{q:'LogisticRegression(C=0.001)에 대한 설명으로 맞는 것은?',o:[['규제가 아주 세다',1,'C는 규제 세기의 역수다. 작을수록 가중치를 세게 누른다. 릿지의 alpha와 방향이 반대다.'],['규제가 아주 약하다',0,'흔한 착각이다. 릿지·라쏘의 alpha처럼 읽으면 틀린다.'],['규제가 없다',0,'규제를 없애려면 C를 아주 크게(1.8 이상에서는 C=np.inf) 둔다.']]},
 {q:'AI가 준 코드에 FutureWarning이 떴다. 가장 알맞은 대응은?',o:[['scikit-learn 버전을 확인하고 경고가 가리키는 인자를 새 방식으로 고친다',1,'라이브러리는 버전마다 인자가 바뀐다. 경고는 다음 버전에서 오류가 될 수 있다는 예고다.'],['경고는 무시해도 결과는 같다',0,'지금은 같아도 다음 버전에서 오류가 나거나 다르게 동작할 수 있다.'],['AI에게 같은 질문을 다시 해서 다른 코드를 받는다',0,'다시 물어도 같은 실수를 할 수 있다. 공식 문서와 버전으로 확인한다.']]}],
q4:[{q:'경사 하강 한 걸음에서 가중치를 고치는 크기를 정하는 것은?',o:[['배합마다 "틀린 만큼(p − y)"에 특징 x를 곱해 평균 낸 값과 학습률',1,'로지스틱 회귀의 기울기는 평균[(p − y)·x]다. 학습률이 너무 크면 손실이 출렁인다.'],['배합의 수',0,'수는 평균을 낼 때만 쓴다.'],['임계값',0,'임계값은 학습이 끝난 뒤 확률을 이름표로 바꿀 때 쓴다.']]},
 {q:'정답이 합격(y=1)인 배합에게 p = 0.01이라고 했다. 손실은?',o:[['약 4.6 (−log 0.01)',1,'확신하고 틀리면 손실이 크다. p = 0.99였다면 약 0.01이다.'],['0.01',0,'손실은 확률이 아니라 −log(확률)이다.'],['0',0,'맞혔을 때 손실이 0에 가까워진다.']]},
 {q:'시멘트·물 두 특징으로는 손실이 0.5 근처에서 더 내려가지 않는다. 가장 큰 까닭은?',o:[['재령(얼마나 굳혔나)을 몰라 같은 배합도 강도가 달라서',1,'정보가 없으면 아무리 학습해도 겹침이 남는다. 재령을 넣은 전문가 특징은 훨씬 낮은 손실에 닿는다.'],['학습률이 너무 작아서',0,'학습률을 키워도 바닥은 그대로다. 바닥 자체가 높은 것이다.'],['규제가 없어서',0,'규제는 바닥을 조금 올릴 뿐, 겹침을 없애지 못한다.']]}],
q5:[{q:'콘크리트 출하 판정에서 더 위험한 틀림은?',o:[['불합격(강도 미달) 배합을 합격이라 하는 것',1,'강도가 모자란 콘크리트가 구조물에 쓰인다. 그래서 출하 판정은 "합격"이라고 말하는 임계값을 올린다.'],['합격 배합을 불합격이라 하는 것',0,'재시험·폐기 비용이 들지만 안전 문제는 아니다.'],['둘 다 똑같다',0,'정확도는 둘을 똑같이 센다. 그래서 정확도만으로 판정하면 안 된다.']]},
 {q:'임계값을 0.5에서 0.7로 올리면?',o:[['"합격"이라고 하는 수가 줄어 정밀도는 오르고 재현율은 내린다',1,'강도 미달 출하가 줄고, 대신 진짜 합격 배합을 더 많이 불합격으로 돌린다.'],['정확도가 늘 오른다',0,'이 시험지에서는 정확도가 오히려 내려간다. 그래도 업무에 따라 더 나은 선택일 수 있다.'],['모델의 가중치가 바뀐다',0,'임계값은 학습이 끝난 뒤 확률을 이름표로 바꾸는 문턱이다. 가중치는 그대로다.']]},
 {q:'C를 고를 때 시험지 점수를 보고 가장 좋은 값을 고르면?',o:[['시험지가 사실상 훈련 데이터가 되어 점수를 믿을 수 없다',1,'하이퍼파라미터는 훈련 데이터 안에서 교차 검증으로 고르고, 시험지는 마지막에 한 번만 연다.'],['가장 정확한 방법이다',0,'시험지에 맞춘 값이라 처음 보는 데이터에서는 더 낮게 나온다.'],['상관없다',0,'평가의 공정성이 무너진다.']]}],
q6:[{q:'로지스틱 회귀가 가장 어려워하는 데이터는?',o:[['"너무 낮거나 너무 높으면 불량"처럼 경계가 굽거나 두 갈래인 데이터',1,'경계가 특징 공간에서 직선(평면)이기 때문이다. 특징을 만들어 넣거나 다른 지원자(커널 SVM, 트리)를 부른다.'],['특징이 클수록 합격 확률이 꾸준히 오르는 데이터',0,'가장 잘하는 일이다.'],['특징이 여덟 개인 표 데이터',0,'특징이 수백 개여도 잘 돈다. 규제가 도와준다.']]},
 {q:'scikit-learn 1.8 이상에서 L1 규제 로지스틱 회귀를 쓰는 방법은?',o:[['LogisticRegression(l1_ratio=1, solver="saga")',1,'1.8부터 penalty는 사라지는 중이고 l1_ratio로 정한다. L1은 saga나 liblinear가 필요하다.'],['LogisticRegression(multi_class="l1")',0,'multi_class는 규제와 관계없고, 이미 없어진 인자다.'],['Lasso(classifier=True)',0,'그런 인자는 없다. Lasso는 회귀 모델이다.']]}]};

/* ══ 공통 B·C·D·E·F 데이터 ══ */
var PROMPTS=[
 {t:'① 내 공정 데이터로 로지스틱 회귀 코드 받기',use:'내 공장의 품질 데이터로 03장의 프롬프트를 맞출 때',p:MAIN_PROMPT},
 {t:'② AI 코드 검증 요청',use:'받은 코드를 다른 AI(또는 같은 AI의 새 대화)에게 검토받을 때',p:'아래 scikit-learn 로지스틱 회귀 코드를 코드 리뷰해 줘. 특히 다음을 하나씩 확인하고, 문제가 있으면 줄 번호와 고친 코드를 보여 줘.\n1) 데이터를 나누기 전에 표준화·인코딩을 해서 시험 데이터 정보가 새지 않았나\n2) train_test_split에 stratify와 random_state가 있나\n3) 내 scikit-learn 버전(〔1.x〕)에 없는 인자(multi_class, penalty 등)를 쓰지 않았나\n4) C의 방향(클수록 규제 약함)과 주석이 맞나\n5) 시험 데이터로 하이퍼파라미터나 임계값을 고르지 않았나\n6) 평가를 시험 데이터로 했나, 기준선(DummyClassifier)과 비교했나\n추측으로 답하지 말고, 확실하지 않은 부분은 "공식 문서 확인 필요"라고 표시해 줘.'},
 {t:'③ 임계값 정하기 상담',use:'틀림의 비용이 다를 때 임계값을 업무 기준으로 정하고 싶을 때',p:'콘크리트 35MPa 합격 판정 모델의 시험 결과다. 임계값 0.3·0.5·0.7에서의 혼동행렬을 아래에 붙인다.\n- 강도 미달 배합을 합격으로 출하하면 1건당 〔회수·보강 비용 500만 원〕\n- 합격 배합을 불합격으로 돌리면 1건당 〔재시험 비용 30만 원〕\n이 비용을 기준으로 임계값별 총비용을 계산하고, 어떤 임계값을 추천하는지 근거와 함께 설명해 줘. 계산 과정을 표로 보여 주고, 시험 데이터로 임계값을 고르면 생기는 문제와 대안(검증 데이터, 교차 검증)도 짚어 줘.\n\n〔여기에 혼동행렬을 붙여 넣는다〕'},
 {t:'④ 수업용 문제 만들기',use:'이 편으로 수업할 때 확인 문제가 더 필요할 때',p:'로지스틱 회귀 입문 수업용 문제를 만들어 줘. 대상은 〔제조 현장 실무자, 수식에 익숙하지 않음〕이다. 예시는 콘크리트 강도 합격 판정으로 통일한다.\n- 시그모이드·임계값·교차 엔트로피·규제 C를 각각 묻는 객관식 4문제 (보기 3개, 정답 해설 포함)\n- 오해를 드러내는 문제 2개: "C가 클수록 규제가 세다", "정확도가 높으면 위험한 실수도 적다"\n- 손으로 풀 수 있는 계산 문제 1개: z = 2일 때 σ(z)와 y=1일 때 손실 (소수 둘째 자리)\n정답은 맨 끝에 따로 모아 줘.'}];
var GL=[
 ['모델','로지스틱 회귀','선형 점수에 시그모이드를 씌워 클래스 확률을 내는 분류 모델. 이름은 회귀지만 일은 분류다.'],
 ['모델','선형 점수','특징에 가중치를 곱해 더하고 절편을 더한 값 z = w·x + b. 로지스틱 회귀의 첫 단계다.'],
 ['모델','시그모이드','σ(z) = 1/(1+e^−z). 어떤 수든 0과 1 사이로 바꾸는 S자 함수. 로지스틱 함수라고도 한다.'],
 ['모델','로짓','확률 p의 로그 오즈 log(p/(1−p)). 로지스틱 회귀에서는 로짓이 곧 선형 점수 z다.'],
 ['모델','오즈','p/(1−p). 일어날 확률이 일어나지 않을 확률의 몇 배인지.'],
 ['모델','소프트맥스','점수 여러 개를 합이 1인 확률들로 바꾸는 함수. 다중 분류에서 시그모이드의 역할을 한다.'],
 ['모델','결정 경계','두 클래스의 확률이 같아지는 곳. 로지스틱 회귀에서는 z = 0인 직선(평면)이다.'],
 ['모델','임계값','확률을 클래스 이름표로 바꾸는 문턱. 기본은 0.5지만 업무에 따라 옮긴다.'],
 ['데이터','샘플','표의 한 줄. 여기서는 콘크리트 배합 하나(와 그 시편의 시험 결과).'],
 ['데이터','특징','모델이 입력으로 쓰는 열. 시멘트·고로 슬래그·플라이 애시·물·고성능 감수제·굵은 골재·잔골재·재령.'],
 ['데이터','타겟','맞히려는 정답 열. 이 면접에서는 35MPa 합격 여부와 강도 3등급.'],
 ['데이터','물-결합재비','물의 양 ÷ 결합재(시멘트 + 고로 슬래그 + 플라이 애시)의 양. 낮을수록 강도가 높아지는 경향이 있다.'],
 ['데이터','재령','콘크리트를 친 뒤 시험할 때까지 지난 날수. 강도는 처음에 빨리, 나중에 천천히 오른다.'],
 ['데이터','표준화','특징마다 평균을 빼고 표준편차로 나눠 단위를 맞추는 일. 훈련 데이터로만 맞춘다.'],
 ['데이터','데이터 누수','시험 데이터의 정보가 학습에 섞여 들어가는 일. 점수가 실제보다 좋게 나온다.'],
 ['학습','교차 엔트로피','정답 클래스에 준 확률의 −log. 확신하고 틀리면 크게 벌한다. 로지스틱 회귀의 손실 함수.'],
 ['학습','경사 하강','손실이 줄어드는 방향(기울기의 반대)으로 가중치를 조금씩 옮기는 학습 방법.'],
 ['학습','학습률','경사 하강 한 걸음의 크기. 너무 크면 출렁이고, 너무 작으면 느리다.'],
 ['학습','규제','가중치가 너무 커지지 않게 손실에 벌점을 더하는 일. L2(릿지형)와 L1(라쏘형)이 있다.'],
 ['학습','과대적합','훈련 데이터를 외워 처음 보는 데이터에서 성적이 떨어지는 일.'],
 ['학습','과소적합','모델이 너무 단순하거나 규제가 너무 세서 훈련 데이터조차 못 맞히는 일.'],
 ['평가','정확도','맞힌 샘플의 비율. 틀림의 종류를 구별하지 않는다.'],
 ['평가','혼동행렬','정답(행)과 예측(열)을 짝지어 센 표. 어떤 클래스를 어떤 클래스로 헷갈렸는지 보인다.'],
 ['평가','정밀도','"이 클래스다"라고 한 것 중 실제로 맞은 비율. 출하 판정에서는 "합격" 도장의 믿음직함.'],
 ['평가','재현율','실제 그 클래스 중 모델이 찾아낸 비율.'],
 ['평가','ROC 곡선','임계값을 바꿔 가며 거짓 양성 비율(가로)과 재현율(세로)을 이은 곡선. 아래 넓이가 AUC.'],
 ['평가','기준선','배우지 않고도 낼 수 있는 점수. "늘 다수 쪽"이라고 답하는 모델 등.']];
var ANALOGY=[
 ['선형 점수','면접관마다 항목별 점수에 가중치를 곱해 더한 총점',2,'뉴런이 시냅스 세기만큼 입력을 더하는 것','실제 뉴런의 합산은 시간과 위치에 따라 달라지고, 덧셈만으로 설명되지 않는다'],
 ['시그모이드','총점이 높을수록 합격 가능성이 오르지만 100%를 넘지 않는다',2,'뉴런의 입력 세기에 따른 발화율이 포화하는 S자 모양','발화율 곡선은 뉴런마다 모양이 다르고, 시그모이드처럼 깔끔한 식이 아니다'],
 ['임계값','출하 도장을 찍는 기준. 안전이 걸리면 높이고, 놓치면 비싼 일이면 낮춘다',2,'초파리가 냄새 증거를 모아 일정 수준을 넘으면 결정하는 것(증거 누적 모형)','초파리의 결정은 시간에 따라 증거가 쌓이는 과정이다. 로지스틱 회귀의 임계값은 한 번에 비교한다'],
 ['교차 엔트로피','확신에 차서 틀린 답변일수록 감점이 크다',2,'예상과 결과가 어긋날 때 커지는 예측 오류 신호(도파민 뉴런)','예측 오류 신호는 보상의 차이를 나타낸다. 로그 손실과 같은 식이라는 근거는 없다'],
 ['규제 C','"한 항목만 보고 뽑지 말라"는 채용 규정. 규정이 세면(C 작음) 다 비슷하게 본다',1,'','시냅스 항상성 같은 현상과 연결하기도 하지만 대응이 확인되지 않았다'],
 ['소프트맥스','세 등급 심사위원이 각자 점수를 내면, 점수 차이를 "각 등급일 확률"로 나누는 것',2,'뉴런들이 서로 억제해 가장 센 신호만 두드러지게 하는 정규화','뇌의 정규화는 나눗셈 꼴이 많지만 지수 함수를 쓴다는 근거는 없다']];
var REFS=[
 'Yeh, I.-C. (1998). Modeling of strength of high-performance concrete using artificial neural networks. <i>Cement and Concrete Research</i>, 28(12), 1797–1808. <a href="https://doi.org/10.1016/S0008-8846(98)00165-3" target="_blank" rel="noopener">doi:10.1016/S0008-8846(98)00165-3</a> — 실기 과제지 원출처',
 'Yeh, I.-C. (2007). Concrete Compressive Strength [Dataset]. UCI Machine Learning Repository. <a href="https://doi.org/10.24432/C5PK67" target="_blank" rel="noopener">doi:10.24432/C5PK67</a> (CC BY 4.0) — 02장 데이터',
 'Berkson, J. (1944). Application of the logistic function to bio-assay. <i>Journal of the American Statistical Association</i>, 39(227), 357–365. <a href="https://doi.org/10.1080/01621459.1944.10500699" target="_blank" rel="noopener">doi:10.1080/01621459.1944.10500699</a> — "로짓"이라는 이름, 01장',
 'Cox, D. R. (1958). The regression analysis of binary sequences. <i>Journal of the Royal Statistical Society: Series B</i>, 20(2), 215–242. <a href="https://doi.org/10.1111/j.2517-6161.1958.tb00292.x" target="_blank" rel="noopener">doi:10.1111/j.2517-6161.1958.tb00292.x</a> — 이진 결과의 로지스틱 회귀, 01장',
 'Hosmer, D. W., Lemeshow, S., &amp; Sturdivant, R. X. (2013). <i>Applied Logistic Regression</i> (3rd ed.). Wiley. — 해석과 진단, 06장',
 'Hastie, T., Tibshirani, R., &amp; Friedman, J. (2009). <i>The Elements of Statistical Learning</i> (2nd ed.), 4장. Springer. — 다항 로지스틱 회귀와 규제, 04·06장',
 'Fawcett, T. (2006). An introduction to ROC analysis. <i>Pattern Recognition Letters</i>, 27(8), 861–874. <a href="https://doi.org/10.1016/j.patrec.2005.10.010" target="_blank" rel="noopener">doi:10.1016/j.patrec.2005.10.010</a> — 05장 임계값과 ROC',
 'Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. <i>Journal of Machine Learning Research</i>, 12, 2825–2830. — A장 코드',
 'scikit-learn 문서. <a href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html" target="_blank" rel="noopener">LogisticRegression</a> — 1.8의 penalty → l1_ratio 변경, C의 뜻, 03·06장',
 'DasGupta, S., Ferreira, C. H., &amp; Miesenböck, G. (2014). FoxP influences the speed and accuracy of a perceptual decision in <i>Drosophila</i>. <i>Science</i>, 344(6186), 901–904. <a href="https://doi.org/10.1126/science.1252114" target="_blank" rel="noopener">doi:10.1126/science.1252114</a> — D장 임계값 비유',
 'Carandini, M., &amp; Heeger, D. J. (2012). Normalization as a canonical neural computation. <i>Nature Reviews Neuroscience</i>, 13, 51–62. <a href="https://doi.org/10.1038/nrn3136" target="_blank" rel="noopener">doi:10.1038/nrn3136</a> — D장 소프트맥스 비유'];
var TIP={who:'머신러닝을 처음 배우는 대학생·직장인·제조 현장 실무자. 5-0 채용 공고를 먼저 보면 좋다. 수식은 σ(z)와 −log(p) 두 개만 나온다',
 goal:'① 선형 점수 → 시그모이드 → 확률 → 임계값의 흐름을 지원서 1장으로 설명한다 ② 경사 하강 한 걸음이 무엇을 고치는지 말한다 ③ AI가 낸 코드에서 누수·버전·C 방향 실수를 찾는다 ④ 틀림의 비용을 따져 임계값을 정하고 판정한다',
 time:'읽기 25분 · 실습(01·04·05장 그림) 20분 · 틀린 곳 찾기 10분 · Colab 15분',
 plans:[['강의형 50분','00장 지원서 1장을 인쇄해 나눠 주고 9칸을 함께 읽기 10분 → 01장 슬라이더로 시그모이드 감 잡기 10분 → 04장 수습 재생(전문가 특징·날것 특징) 10분 → 05장 임계값 실험 10분 → 면접 결과를 학생들이 먼저 맞히고 발표 10분.'],['실무 OJT 90분','03장 프롬프트로 각자 AI에게 코드를 받는다 20분 → 체크리스트로 서로의 코드를 바꿔 검사 20분 → Colab에서 실행하고 A장 결과와 비교 20분 → B장 ③ 프롬프트로 "우리 공장 기준" 틀림 비용을 넣어 임계값을 정하고 발표 30분.'],['1:1 튜터링','지원서 1장의 아홉 칸 중 막히는 칸을 고르게 하고, 그 칸의 "→ 장"으로 바로 간다. 04장에서 배합 하나를 눌러 확률과 손실을 손으로 다시 계산해 보게 한다.']],
 myths:[['로지스틱 회귀는 회귀 모델이다','분류 모델이다. 회귀하는 것은 확률의 로그 오즈다'],['C가 클수록 규제가 세다','반대다. C는 규제의 역수다. 릿지의 alpha와 헷갈리기 쉽다'],['임계값은 늘 0.5다','기본값일 뿐이다. 위험한 실수를 줄여야 하면 옮긴다'],['정확도가 높으면 위험한 실수도 적다','정확도는 두 종류의 틀림을 똑같이 센다. 혼동행렬에서 위험한 칸을 따로 본다'],['특징은 많을수록 좋다','이 면접에서는 전문가 특징 두 개(물-결합재비, log 재령)가 날것 특징 여덟 개보다 나았다'],['AI가 준 코드는 돌아가면 맞다','돌아가도 틀릴 수 있다(누수, 훈련 데이터로 채점). 체크리스트로 읽고 결과를 기준선과 비교한다']],
 tasks:[['과제 1 · 지원서 다시 쓰기','00장 지원서의 예시 값 0.82 대신, 04장에서 고른 배합 하나의 실제 z와 확률로 4번·7번 칸을 다시 채운다.'],['과제 2 · 틀린 곳 찾기 만들기','AI에게 일부러 실수가 섞인 로지스틱 회귀 코드를 받고(B장 ②를 거꾸로 활용), 짝에게 찾게 한다. 실수마다 "왜 틀렸나"를 한 줄로 쓴다.'],['과제 3 · 출하 기준서','B장 ③ 프롬프트로 틀림 비용을 넣어 임계값을 고르고, "우리 공장 출하 판정 기준서"를 반 쪽으로 쓴다. 시험지로 임계값을 고르면 왜 안 되는지도 적는다.'],['과제 4 · 좋은 특징 찾기','A장 코드에 물-결합재비와 log(재령)를 특징으로 더해 다시 돌리고, 정확도와 가중치가 어떻게 바뀌는지 표로 정리한다.']],
 ai:'학생이 AI에게 코드를 받는 것은 막지 말고 과정에 넣는다. 다만 ① 프롬프트를 제출하게 하고 ② 체크리스트 아홉 항목을 직접 표시하게 하고 ③ 실행 결과를 기준선·합격선과 비교한 문장을 쓰게 하면, "받아 적기"가 아니라 "검증하기"를 배운다. AI 답의 함수·인자는 공식 문서와 버전으로 확인하게 하고, 과제에 AI 도움을 받았으면 사용 범위를 밝히게 한다.'};
var CODE={title:'로지스틱 회귀 면접 — 시그모이드부터 합격 판정까지',sub:'numpy, pandas, scikit-learn, ucimlrepo · CPU 몇 초',file:'logistic_regression_interview',
 desc:'이 편의 면접을 그대로 옮겼다. UCI에서 콘크리트 데이터를 받아 시험지를 봉인하고(1·2번 칸), 35MPa 합격을 시그모이드로 직접 확인하고(3번), 혼동행렬과 틀림의 종류(4번), 임계값(5번), 강도 3등급을 소프트맥스로(6번), 전문가 특징 두 개(7번), 규제 C(8번), 버전에 맞는 L1 규제(9번), 합격 판정(10번)까지 간다. 노트북으로 저장하면 칸마다 셀이 나뉜다.',
 cap:'실행 결과 (scikit-learn 1.9, CPU). Colab의 scikit-learn 버전이 1.8보다 낮으면 9번 칸이 penalty="l1"로 자동으로 바뀐다. 브라우저 계산과 같은 결론(시험 86.4%, 미달 출하 13건, 전문가 특징 두 개 87.6%)이 나온다.'};
