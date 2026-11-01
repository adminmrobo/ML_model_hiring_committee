/* ══ 5-0 채용 공고 · 판단을 데이터로 (콘크리트) ══════════════════════ */
const ROWS=concRows(),TR=ROWS.filter(r=>!r.test),TE=ROWS.filter(r=>r.test);
const accOf=(rows,f)=>rows.reduce((a,r)=>a+(f(r)===r.pass?1:0),0)/rows.length;
const pctN=(x)=>pct(x);
const AGES=[1,3,7,14,28,56,90,180,365];
const MAJ=(()=>{const p=TR.filter(r=>r.pass).length;return p>TR.length/2?1:0;})();
const BASE=accOf(TE,()=>MAJ);
const MAN_TR=accOf(TR,manualRule),MAN_TE=accOf(TE,manualRule);

/* ── 히어로 ── */
heroRoom($('heroCv'),{sym:'§',man:1,no:'EMP-0000',name:'매뉴얼 씨',role:'기존 직원',banner:'모델 채용 위원회 · 채용 공고',doc:'front',asst:'board',fly:'clip',robo:'calc',loop:10,
  beats:[{"at": 0.4, "who": "doc", "mood": "front", "move": "pop", "say": "공고를 냅시다"}, {"at": 2.4, "who": "asst", "mood": "ledger", "move": "nod", "say": "합격선은 매뉴얼 씨가!"}, {"at": 4.4, "who": "fly", "mood": "point", "move": "tilt", "say": "규칙 두 줄로?"}, {"at": 6.4, "who": "robo", "mood": "coin", "move": "bob", "say": "연봉 0원이야"}, {"at": 8.4, "who": "doc", "mood": "board", "move": "stand"}],
  motif(ctx,t,W,H,C){/* 사원증 옆 공시체 세 개 */for(let k=0;k<3;k++){const x=286+k*40,y=232,h=50;ctx.fillStyle=k===1?rgba(C.blue,.55):'#9aa3b5';ctx.fillRect(x,y-h,26,h);ctx.beginPath();ctx.ellipse(x+13,y-h,13,5,0,0,7);ctx.fillStyle='#d5dbe6';ctx.fill();ctx.strokeStyle=C.ink;ctx.lineWidth=1.5;ctx.stroke();ctx.strokeRect(x,y-h,26,h);}
    ctx.fillStyle=C.muted;ctx.font=font(12,500,1);ctx.textAlign='left';ctx.fillText('공시체 · 28일',292,252);}});

/* ── 01 장면 ── */
comic('comicIntro',[{"bg": "office", "prop": "§1 · §2", "cast": [{"who": "doc", "mood": "board", "x": 30, "move": "nod"}, {"who": "fly", "mood": "point", "x": 74, "move": "tilt", "flip": true}], "say": {"who": "fly", "txt": "두 줄짜리 규칙으로 여태 버텼다고? 처음 보는 배합에도 통하는지 볼 거야."}, "cap": "① 규칙 — 사람이 쓴 판단 기준(룰 베이스)"}, {"bg": "site", "prop": "772 · 258", "cast": [{"who": "asst", "mood": "ledger", "x": 28}, {"who": "robo", "mood": "calc", "x": 72, "move": "bob"}], "say": {"who": "asst", "txt": "배합 1,030건 중 258건은 봉투에 봉인해요. 마지막에 한 번만 엽니다."}, "cap": "② 시험지 — 학습에 쓰지 않는 데이터"}, {"bg": "office", "prop": "76.7%", "cast": [{"who": "doc", "mood": "verdict", "x": 30, "move": "pop"}, {"who": "robo", "mood": "coin", "x": 72, "move": "jump"}], "say": {"who": "robo", "txt": "연봉 0원짜리가 76.7%라. 새 직원은 이걸 넘어야 월급 줄게."}, "cap": "③ 합격선 — 매뉴얼 씨의 시험지 점수"}]);

/* ── 01 판단 기계 ── */
(function(){function upd(){const wb=+$('jWb').value,age=+$('jAge').value,A=wb<0.5,B=age>=28,ok=A&&B;$('jWbO').textContent=wb.toFixed(2);
  const box=(t,on,c)=>'<div style="border:2px solid '+(on?'var(--blue)':'var(--rule)')+';background:'+(on?'var(--blue-soft)':'var(--panel2)')+';border-radius:10px;padding:8px 12px">'+t+'</div>';
  $('jFlow').innerHTML=box('<b>입력</b> 물-결합재비 '+wb.toFixed(2)+' · 재령 '+age+'일',true)+'<div style="text-align:center;color:var(--muted)">↓ 작업 표준서(규칙)</div>'+
    box('제1조 물-결합재비 &lt; 0.5 → <b>'+(A?'예':'아니오')+'</b><br>제1조 재령 ≥ 28일 → <b>'+(B?'예':'아니오')+'</b>',A&&B)+'<div style="text-align:center;color:var(--muted)">↓ 결정</div>'+
    '<div style="border:3px '+(ok?'solid var(--blue)':'dashed var(--muted)')+';border-radius:10px;padding:10px 12px;text-align:center;font-size:20px;font-weight:700;color:'+(ok?'var(--blue)':'var(--text)')+'">'+(ok?'합격 (35MPa 이상으로 판정)':'불합격')+'</div>';}
  $('jWb').addEventListener('input',upd);$('jAge').addEventListener('change',upd);upd();})();

/* ── 02 IF · 논리 · 순서도 ── */
(function(){const picks=[TR.find(r=>r.wb<0.45&&r.age===28&&r.pass),TR.find(r=>r.wb<0.45&&r.age===7),TR.find(r=>r.wb>0.6&&r.age>=28),TR.find(r=>r.wb<0.5&&r.age>=28&&!r.pass),TR.find(r=>r.wb>=0.5&&r.age>=56&&r.pass)].filter(Boolean);
  const sel=$('ifPick');picks.forEach((r,i)=>sel.add(new Option('#'+(r.i+1)+' · 물결합재비 '+r.wb.toFixed(2)+' · '+r.age+'일',i)));
  function upd(){const r=picks[+sel.value],A=r.wb<0.5,B=r.age>=28,ok=A&&B;
    $('ifEq').innerHTML='=IF(AND('+r.wb.toFixed(2)+'&lt;0.5, '+r.age+'&gt;=28), "합격", "불합격")\n= IF(AND('+(A?'TRUE':'FALSE')+', '+(B?'TRUE':'FALSE')+'), …)\n= <b>"'+(ok?'합격':'불합격')+'"</b>';
    const right=ok===(r.pass===1);$('ifInfo').innerHTML='매뉴얼의 답은 <b>'+(ok?'합격':'불합격')+'</b>, 실제 28일 강도 시험 결과는 <b>'+r.strength.toFixed(1)+'MPa</b>('+(r.pass?'35MPa 이상':'35MPa 미만')+'). '+(right?'규칙이 맞혔다.':'<span style="color:var(--stamp);font-weight:700">규칙이 틀렸다.</span> '+(ok?'조건은 다 맞췄는데 강도가 안 나왔다. 골재나 혼화재 같은, 규칙이 보지 않는 재료가 원인일 수 있다.':'규칙은 불합격이라 했지만 강도가 나왔다. 규칙이 모르는 길(감수제, 긴 재령)로 강해진 배합이다.'));
    const on=(id,v)=>{const e=$(id);if(e){e.style.stroke=v?'var(--blue)':'var(--muted)';e.style.strokeWidth=v?3.5:2;}};
    on('fe0',1);on('fd1',1);on('fe1',A);on('fd2',A);on('fe2',A&&B);on('fy',ok);on('fe3',!A);on('fe4',A&&!B);on('fn',!ok);}
  sel.addEventListener('change',upd);upd();
  const st={A:true,B:false};
  function lg(){const rows=[[0,0],[0,1],[1,0],[1,1]];const f={AND:(a,b)=>a&&b,OR:(a,b)=>a||b,'NOT A':(a)=>!a,NOR:(a,b)=>!(a||b)};
    $('lgTable').innerHTML='<thead><tr><th>A</th><th>B</th>'+Object.keys(f).map(k=>'<th>'+k+'</th>').join('')+'</tr></thead><tbody>'+rows.map(([a,b])=>{const cur=(!!a===st.A&&!!b===st.B);return '<tr style="'+(cur?'outline:2px solid var(--blue);outline-offset:-2px':'')+'"><td>'+a+'</td><td>'+b+'</td>'+Object.values(f).map(fn=>{const v=fn(a,b)?1:0;return '<td class="'+(v?'one':'')+'">'+v+'</td>';}).join('')+'</tr>';}).join('')+'</tbody>';
    $('lgInfo').innerHTML='지금 A = '+(st.A?1:0)+', B = '+(st.B?1:0)+' → AND = <b>'+(st.A&&st.B?1:0)+'</b>(매뉴얼의 합격 조건), OR = '+(st.A||st.B?1:0)+', NOR = '+(!(st.A||st.B)?1:0)+'. NOR가 1이면 "물-결합재비도 높고 재령도 짧다", 즉 두 조건 모두 어긋난 배합이다.';
    $('lgA').setAttribute('aria-pressed',st.A);$('lgB').setAttribute('aria-pressed',st.B);}
  $('lgA').addEventListener('click',()=>{st.A=!st.A;lg();});$('lgB').addEventListener('click',()=>{st.B=!st.B;lg();});lg();})();

/* ── 03 규칙 편집기 ── */
const RULE={mode:'expert',cut:0.5,age:28,sp:false,opened:false};
function ratioOf(r,mode){return mode==='novice'?r.water/r.cement:r.wb;}
function ruleFn(o){return r=>{const x=ratioOf(r,o.mode);if(x<o.cut&&r.age>=o.age)return 1;if(o.sp&&r.superplasticizer>=10&&x<o.cut&&r.age>=7)return 1;return 0;};}
/* 중첩 IF 3등급 (02장) */
function nestIf(r){return (r.wb<0.4&&r.age>=28)?2:(r.wb<0.6&&r.age>=7)?1:0;}
$('nestAcc').textContent=pct(TR.reduce((a,r)=>a+(nestIf(r)===r.grade?1:0),0)/TR.length)+' (3등급, 훈련 772건)';
(function(){const cv=$('ruleCv'),H=330;
  function draw(){const W=fit(cv,H),ctx=cv.getContext('2d'),C=cols(),pl=50,pr=12,pt=12,pb=38,gw=W-pl-pr,gh=H-pt-pb,x0=0.2,x1=RULE.mode==='novice'?1.9:0.95;
    const sx=x=>pl+(clamp(x,x0,x1)-x0)/(x1-x0)*gw,ly=a=>Math.log(a),sy=a=>pt+(1-(ly(a)-ly(1))/(ly(365)-ly(1)))*gh;
    ctx.clearRect(0,0,W,H);ctx.fillStyle=rgba(C.blue,.1);ctx.fillRect(pl,sy(365),sx(RULE.cut)-pl,sy(RULE.age)-sy(365));
    if(RULE.sp){ctx.fillStyle=rgba(C.blue,.05);ctx.fillRect(pl,sy(RULE.age),sx(RULE.cut)-pl,sy(7)-sy(RULE.age));}
    ctx.strokeStyle=C.rule;ctx.lineWidth=1;ctx.strokeRect(pl,pt,gw,gh);ctx.fillStyle=C.muted;ctx.font=font(12);
    [1,3,7,28,90,365].forEach(a=>{ctx.textAlign='right';ctx.fillText(a,pl-6,sy(a)+4);ctx.beginPath();ctx.moveTo(pl,sy(a));ctx.lineTo(pl+gw,sy(a));ctx.strokeStyle=rgba(C.rule,.6);ctx.stroke();});
    for(let k=0;k<=5;k++){const x=x0+(x1-x0)*k/5;ctx.textAlign='center';ctx.fillText(x.toFixed(2),sx(x),H-20);}
    ctx.textAlign='center';ctx.fillText(RULE.mode==='novice'?'물-시멘트비 (물 ÷ 시멘트)':'물-결합재비 (물 ÷ (시멘트+슬래그+플라이 애시))',pl+gw/2,H-4);
    ctx.save();ctx.translate(12,pt+gh/2);ctx.rotate(-Math.PI/2);ctx.fillText('재령 (일, 로그 눈금)',0,0);ctx.restore();
    const f=ruleFn(RULE),R=RULE.opened?ROWS:TR,rng=RNG(7);
    R.forEach(r=>{const x=sx(ratioOf(r,RULE.mode)),y=sy(r.age)+(rng()-.5)*9,c=r.pass?C.blue:C.text;if(r.test)circ(ctx,x,y,3.6,C.panel,c,1.6);else circ(ctx,x,y,3,rgba(c,.7),null);if(f(r)!==r.pass){ctx.strokeStyle=rgba(C.stamp,.8);ctx.lineWidth=1.3;ctx.beginPath();ctx.arc(x,y,5.5,0,7);ctx.stroke();}});
    ctx.strokeStyle=C.blue;ctx.lineWidth=2.5;ctx.beginPath();ctx.moveTo(sx(RULE.cut),pt);ctx.lineTo(sx(RULE.cut),sy(RULE.age));ctx.lineTo(pl,sy(RULE.age));ctx.stroke();}
  function stats(){const f=ruleFn(RULE),tr=accOf(TR,f),te=accOf(TE,f);
    const items=[[pct(tr),'훈련 정확도 (772건)','m'],[RULE.opened?pct(te):'봉인','시험지 정확도 (258건)','s'],[pct(BASE),'기준선 · 늘 "'+(MAJ?'합격':'불합격')+'"'],[(RULE.sp?3:2)+'개','조항 수 (그 밖 제외)']];
    $('ruleStat').innerHTML=statGrid(items,2);
    const groups=[['혼화재 없는 배합 (시멘트만)',r=>r.slag===0&&r.fly_ash===0&&r.superplasticizer===0],['혼화재를 섞은 배합',r=>!(r.slag===0&&r.fly_ash===0&&r.superplasticizer===0)],['재령 28일 미만',r=>r.age<28],['재령 28일 이상',r=>r.age>=28]];
    $('subTb').innerHTML='<thead><tr><th>배합 무리 (훈련)</th><th>건수</th><th>이 매뉴얼</th><th>신입 매뉴얼</th><th>전문가 매뉴얼</th></tr></thead><tbody>'+groups.map(([n,g])=>{const R=TR.filter(g);return '<tr><td>'+n+'</td><td>'+R.length+'</td><td><b>'+pct(accOf(R,f))+'</b></td><td>'+pct(accOf(R,ruleFn({mode:'novice',cut:0.6,age:28})))+'</td><td>'+pct(accOf(R,manualRule))+'</td></tr>';}).join('')+'</tbody>';
    $('ruleInfo').innerHTML=(RULE.mode==='novice'?'<b>신입 매뉴얼</b>은 물과 시멘트만 본다. 슬래그·플라이 애시를 넣은 배합은 물-시멘트비가 높게 나와 강한데도 불합격으로 잘린다(빨간 테두리가 오른쪽에 몰린다). ':'<b>전문가 매뉴얼</b>은 슬래그·플라이 애시도 시멘트처럼 굳는다는 지식으로 셋을 "결합재"로 묶는다. ')+
      '지금 문턱: '+(RULE.mode==='novice'?'물-시멘트비':'물-결합재비')+' &lt; '+RULE.cut.toFixed(2)+', 재령 ≥ '+RULE.age+'일'+(RULE.sp?', 감수제 조항 켬':'')+' → 훈련 772건 중 '+Math.round(tr*772)+'건을 맞혔다. '+(RULE.opened?'시험지까지 열었다. 이제 문턱을 고치면 시험지를 보고 푼 셈이 된다.':'파란 영역이 "합격"이라고 하는 곳이다.');}
  function upd(){$('rCutO').textContent=RULE.cut.toFixed(2);$('ratioName').textContent=RULE.mode==='novice'?'물-시멘트비':'물-결합재비';draw();stats();}
  function preset(m){RULE.mode=m;RULE.cut=m==='novice'?0.6:0.5;RULE.age=28;RULE.sp=false;$('rCut').max=m==='novice'?1.8:0.9;$('rCut').value=RULE.cut;$('rAge').value='28';$('rSp').checked=false;upd();}
  seg('#manPick button',v=>preset(v));
  $('rCut').addEventListener('input',e=>{RULE.cut=+e.target.value;upd();});$('rAge').addEventListener('change',e=>{RULE.age=+e.target.value;upd();});$('rSp').addEventListener('change',e=>{RULE.sp=e.target.checked;upd();});
  $('ruleReset').addEventListener('click',()=>preset(RULE.mode));
  $('ruleTest').addEventListener('click',()=>{RULE.opened=true;$('ruleTest').disabled=true;$('ruleTest').textContent='시험지 열림';upd();});
  window.ruleRedraw=upd;upd();onRedraw(draw);})();

/* ── 04 기계에게 문턱 찾기 · 로지스틱 회귀 미리 보기 ── */
const CUTS=[];for(let c=0.25;c<=0.805;c+=0.01)CUTS.push(Math.round(c*100)/100);const GAGES=[1,3,7,14,28,56,90];
let GRID=null;
function gridSearch(){if(GRID)return GRID;const M=GAGES.map(a=>CUTS.map(c=>accOf(TR,ruleFn({mode:'expert',cut:c,age:a}))));let b={a:-1};
  GAGES.forEach((ag,i)=>CUTS.forEach((c,j)=>{if(M[i][j]>b.a+1e-12)b={a:M[i][j],cut:c,age:ag};}));GRID={M,best:b};return GRID;}
(function(){const cv=$('gridCv'),H=230;let shown=-1;
  function draw(){const W=fit(cv,H),ctx=cv.getContext('2d'),C=cols(),pl=40,pr=8,pt=8,pb=34,gw=W-pl-pr,gh=H-pt-pb,cw=gw/CUTS.length,ch=gh/GAGES.length;ctx.clearRect(0,0,W,H);
    const G=gridSearch();GAGES.forEach((ag,i)=>CUTS.forEach((c,j)=>{const lim=shown<0?-1:shown;const k=i*CUTS.length+j;if(lim>=0&&k>lim)return;const v=G.M[i][j];ctx.fillStyle=rgba(C.blue,clamp((v-0.5)/0.3,0.03,1));ctx.fillRect(pl+j*cw,pt+(GAGES.length-1-i)*ch,cw+0.5,ch-1);}));
    ctx.fillStyle=C.muted;ctx.font=font(11);ctx.textAlign='right';GAGES.forEach((a,i)=>ctx.fillText(a+'일',pl-4,pt+(GAGES.length-1-i)*ch+ch/2+4));ctx.textAlign='center';[0.3,0.4,0.5,0.6,0.7,0.8].forEach(c=>ctx.fillText(c.toFixed(1),pl+(CUTS.indexOf(c)+.5)*cw,H-18));ctx.fillText('물-결합재비 문턱',pl+gw/2,H-3);
    const mark=(cut,age,col,lab,dy)=>{dy=dy||0;const j=CUTS.indexOf(cut),i=GAGES.indexOf(age);if(j<0||i<0)return;const x=pl+j*cw,y=pt+(GAGES.length-1-i)*ch;ctx.strokeStyle=col;ctx.lineWidth=2.5;ctx.strokeRect(x-2,y-1,cw+4,ch+1);ctx.fillStyle=col;ctx.font=font(11,700);ctx.textAlign=j>CUTS.length/2?'right':'left';ctx.fillText(lab,j>CUTS.length/2?x-6:x+cw+6,y+ch/2+4+dy);};
    mark(0.5,28,C.c_man,'전문가');if(shown>=CUTS.length*GAGES.length-1||shown<0&&GRID&&GRID.done)mark(G.best.cut,G.best.age,C.stamp,'기계',-ch*0.6);}
  $('autoBtn').addEventListener('click',()=>{const G=gridSearch(),N=CUTS.length*GAGES.length;$('autoBtn').disabled=true;shown=0;
    (function go(){shown=Math.min(N-1,shown+(REDUCE?N:12));draw();if(shown<N-1){requestAnimationFrame(go);return;}GRID.done=true;
      const f=ruleFn({mode:'expert',cut:G.best.cut,age:G.best.age});
      $('autoInfo').innerHTML='기계가 '+N+'가지 문턱 조합을 모두 시험했다. 고른 문턱: 물-결합재비 &lt; <b>'+G.best.cut.toFixed(2)+'</b>, 재령 ≥ <b>'+G.best.age+'일</b> → 훈련 <b>'+pct(G.best.a)+'</b> · 시험지 <b>'+pct(accOf(TE,f))+'</b>. 전문가의 문턱(0.50, 28일, 시험지 '+pct(MAN_TE)+')과 거의 같다. 전문가의 감은 좋았다. 다만 <b>모양</b>이 직사각형 하나로 묶여 있는 한, 숫자를 아무리 잘 골라도 이 근처에서 멈춘다.';draw();})();});
  draw();onRedraw(draw);})();
let LRRES=null;
function trainLR(){if(LRRES)return LRRES;const fit2=(fx)=>{const Xtr=TR.map(fx),Xte=TE.map(fx),sc=ML.scaler(Xtr),A=sc.t(Xtr),B=sc.t(Xte),ytr=TR.map(r=>r.pass);
    const st=ML.binFit(A,ytr,{C:1,iters:3000,lr:0.5});const pr=x=>ML.sigmoid(st.b+st.w.reduce((s,w,j)=>s+w*x[j],0))>=0.5?1:0;
    return {tr:A.reduce((a,x,i)=>a+(pr(x)===TR[i].pass?1:0),0)/A.length,te:B.reduce((a,x,i)=>a+(pr(x)===TE[i].pass?1:0),0)/B.length,n:fx(TR[0]).length+1};};
  LRRES={raw:fit2(r=>FEAT.map(f=>r[f])),exp:fit2(r=>[r.wb,Math.log(r.age)])};return LRRES;}
$('lrBtn').addEventListener('click',async()=>{$('lrBtn').disabled=true;$('lrInfo').textContent='학습 중…';await wait(30);const R=trainLR();
  $('lrStat').innerHTML=statGrid([[pct(MAN_TE),'매뉴얼 씨 (규칙 2조)','m'],[pct(R.raw.te),'로지스틱 회귀 · 날것 특징 8개','a'],[pct(R.exp.te),'로지스틱 회귀 · 전문가 특징 2개','a'],[pct(BASE),'기준선']],2);
  $('lrInfo').innerHTML='시험지 258건 기준. 날것 특징 8개로 배운 모델은 '+pct(R.raw.te)+'(훈련 '+pct(R.raw.tr)+'), 물-결합재비와 log 재령 두 특징으로 배운 모델은 '+pct(R.exp.te)+'(훈련 '+pct(R.exp.tr)+')다. 둘 다 매뉴얼 씨('+pct(MAN_TE)+')를 넘는다. 자세한 면접은 <a href="5-4_logistic_regression.html">5-4 로지스틱 회귀</a>에서 한다. 브라우저에서 경사 하강으로 풀어 Colab(scikit-learn)과 소수점 아래가 조금 다를 수 있다.';});

/* ── 05 카드 ── */
(function(){const CARDS=[
  ['이 배합의 28일 압축 강도는 몇 MPa?','강도 42.6MPa처럼 답한다','reg','강도는 크기와 차이에 뜻이 있는 숫자다. 40과 42의 차이는 2MPa라는 실제 의미가 있다. 회귀다(5-1~5-3).'],
  ['이 배합은 35MPa 기준을 넘나?','합격 / 불합격','cls','답이 두 이름표 중 하나다. 이진 분류다(5-4). 같은 데이터에 질문만 바꾼 것이다.'],
  ['이 배합은 저·보통·고 중 어느 등급?','세 이름표 중 하나','cls','이름표가 셋인 다중 분류다. 등급에 순서가 있지만, 답은 이름표 하나다. 클래스 수는 2개가 아니어도 된다.'],
  ['슬럼프 등급 코드 1~5','현장 기록표의 숫자 코드','edge','숫자로 적혀 있지만 "3 − 1 = 2"에 뜻이 없다면 이름표다(분류). 다만 순서가 있고 간격이 대체로 고르면 회귀로 다루기도 한다. 이런 순서형 타겟은 업무에 맞춰 고른다. 경계 사례다.'],
  ['이 공시체를 며칠째 깼을까? (재령 추정)','1, 3, 7, 28…일','edge','재령은 숫자고 크기에 뜻이 있으니 회귀로 볼 수 있다. 하지만 실제로는 1·3·7·28·56·90… 몇 가지 값만 쓰이니 "어느 시험일인가"를 고르는 분류로 볼 수도 있다. 질문을 어떻게 쓰느냐에 달렸다. 경계 사례다.'],
  ['이번 달 레미콘 출하량은 몇 m³?','숫자 하나','reg','양이 크고 작음에 뜻이 있다. 회귀다.'],
  ['이 배합을 만든 공장은 A·B·C 중 어디?','공장 코드','cls','공장 코드는 이름표다. 분류다. 정답 기록이 없다면 비지도(06장)로 비슷한 배합끼리 묶어 볼 수는 있다.'],
  ['내일 출하가 목표를 넘을까?','넘음 / 못 넘음','cls','"얼마나?"를 물으면 회귀, "넘을까?"를 물으면 분류다. 데이터는 같다.']];
  const host=$('cards'),done={};let cur=-1;
  host.innerHTML=CARDS.map((c,i)=>'<button data-i="'+i+'" aria-pressed="false">'+c[0]+'<small>'+c[1]+'</small></button>').join('');
  host.addEventListener('click',e=>{const b=e.target.closest('button');if(!b)return;cur=+b.getAttribute('data-i');host.querySelectorAll('button').forEach(x=>x.setAttribute('aria-pressed',x===b));$('cardInfo').textContent='"'+CARDS[cur][0]+'" — 아래에서 회귀·분류·경계 사례 중 하나를 고르세요.';});
  document.querySelectorAll('#m5 [data-slot]').forEach(s=>s.addEventListener('click',()=>{if(cur<0){$('cardInfo').textContent='먼저 카드를 하나 고르세요.';return;}const c=CARDS[cur],ok=s.getAttribute('data-slot')===c[2];done[cur]=ok;
    const b=host.querySelector('[data-i="'+cur+'"]');b.classList.toggle('ok',ok);b.classList.toggle('no',!ok);
    $('cardInfo').innerHTML=(ok?'<b>맞다.</b> ':'<b>다시 보자.</b> 정답은 '+({reg:'회귀',cls:'분류',edge:'경계 사례'})[c[2]]+'. ')+c[3];
    const n=Object.values(done).filter(Boolean).length;$('cardScore').textContent='맞힌 카드 '+n+' / '+CARDS.length;}));
  $('cardReset').addEventListener('click',()=>{cur=-1;for(const k in done)delete done[k];host.querySelectorAll('button').forEach(x=>{x.classList.remove('ok','no');x.setAttribute('aria-pressed','false');});$('cardScore').textContent='';$('cardInfo').textContent='카드를 하나 고르세요.';});})();

/* ── 06 지도 vs 비지도 ── */
(function(){const cv=$('svCv'),H=320;let mode='sup';
  function draw(){const W=fit(cv,H),ctx=cv.getContext('2d'),C=cols(),pl=50,pr=12,pt=12,pb=38,gw=W-pl-pr,gh=H-pt-pb,sx=x=>pl+(x-90)/(560-90)*gw,sy=y=>pt+(1-y/85)*gh;ctx.clearRect(0,0,W,H);
    ctx.strokeStyle=C.rule;ctx.strokeRect(pl,pt,gw,gh);ctx.fillStyle=C.muted;ctx.font=font(12);[100,200,300,400,500].forEach(x=>{ctx.textAlign='center';ctx.fillText(x,sx(x),H-20);});[0,20,40,60,80].forEach(y=>{ctx.textAlign='right';ctx.fillText(y,pl-6,sy(y)+4);});
    ctx.textAlign='center';ctx.fillText('시멘트 (kg/m³)',pl+gw/2,H-4);ctx.save();ctx.translate(12,pt+gh/2);ctx.rotate(-Math.PI/2);ctx.fillText(mode==='sup'?'압축 강도 (MPa)':'?',0,0);ctx.restore();
    if(mode==='sup'){ctx.setLineDash([5,4]);ctx.strokeStyle=C.muted;[25,45].forEach(y=>{ctx.beginPath();ctx.moveTo(pl,sy(y));ctx.lineTo(pl+gw,sy(y));ctx.stroke();});ctx.setLineDash([]);}
    ROWS.forEach(r=>{const y=mode==='sup'?r.strength:(r.age/365*80);circ(ctx,sx(r.cement),sy(mode==='sup'?r.strength:r.strength),3,mode==='sup'?rgba(GRADECOL[r.grade],.75):rgba(C.muted,.55),null);});}
  function info(){$('svLeg').innerHTML=mode==='sup'?GRADES.map((g,k)=>'<span><i style="background:'+GRADECOL[k]+'"></i>'+g+(k===0?' (&lt;25MPa)':k===1?' (25~45)':' (≥45)')+'</span>').join(''):'<span><i style="background:var(--muted)"></i>정답(강도 등급) 없음</span>';
    $('svInfo').innerHTML=mode==='sup'?'정답이 보이면 할 일이 분명하다. "이 배합의 등급을 맞혀라." 지도 본부의 일이다. 시멘트 양 하나로는 등급이 꽤 겹친다는 것도 보인다.':'정답을 가리면 같은 점들이 모두 회색이다. 시험 기록이 없으니 "맞혀라"는 질문이 성립하지 않는다. 대신 "비슷한 배합끼리 몇 무리로 나뉘나?"를 물을 수 있다. 비지도 본부의 일이다. 콘크리트는 무리가 뚜렷하지 않아서, 비지도 본부는 무리가 또렷한 손글씨 숫자로 면접한다.';}
  seg('#m6 .choice button',v=>{mode=v;draw();info();});draw();info();onRedraw(draw);})();

/* ── 07 체크리스트 ── */
(function(){const CK=[['데이터를 실제로 읽었나','AI가 예제 데이터를 지어내거나 다른 데이터로 바꾸지 않았는지 본다.'],['특징과 타겟이 내가 말한 것과 같나','강도(타겟)가 특징에 섞여 들어가면 정답을 보고 푸는 셈이다.'],['나누기를 먼저 하고, 표준화는 훈련 데이터로만 맞췄나','전체 데이터로 fit_transform 하면 시험지 정보가 샌다(데이터 누수).'],['random_state와 stratify를 넣었나','점수가 매번 같게 나오고, 등급 비율이 훈련·시험에 고르게 들어간다.'],['시험지는 마지막에 한 번만 썼나','문턱이나 하이퍼파라미터를 시험지 점수로 고르지 않는다.'],['라이브러리 버전을 확인했나','print(sklearn.__version__). 버전에 따라 인자 이름이 바뀐다.'],['결과를 합격선과 비교했나','매뉴얼 씨 76.7%, 기준선 49.6%를 넘지 못하면 배운 것이 없다.']];
  const ul=$('checks0');ul.innerHTML=CK.map((c,i)=>'<li><input type="checkbox" id="c0'+i+'"><label for="c0'+i+'"><b>'+c[0]+'</b><small>'+c[1]+'</small></label></li>').join('');
  function upd(){const n=ul.querySelectorAll('input:checked').length;ul.querySelectorAll('li').forEach(li=>li.classList.toggle('done',li.querySelector('input').checked));$('checks0Score').textContent='확인 '+n+' / '+CK.length;}ul.addEventListener('change',upd);upd();})();

/* ── 면접 결과: 매뉴얼 씨 ── */
Interview('iv0',{title:'기존 직원 매뉴얼 씨는 새 판정 업무를 계속 맡을 수 있을까?',
  q:'조건: 봉인된 시험지 258건에서 "35MPa 이상인가"를 맞힌 비율. 작업 표준서(물-결합재비 &lt; 0.5이고 재령 ≥ 28일)를 그대로 쓴다. 이 점수가 지원자들의 첫 합격선이 된다.',
  fly:'처음 보는 배합에서도 되나? 혼화재 넣은 배합은?',robo:'연봉 0원인데, 숨은 비용은 없어?',
  async run(setP){setP(.5);await wait(60);const te=MAN_TE,tr=MAN_TR,mix=TE.filter(r=>!(r.slag===0&&r.fly_ash===0&&r.superplasticizer===0)),plain=TE.filter(r=>r.slag===0&&r.fly_ash===0&&r.superplasticizer===0);setP(1);
    return {result:'hold',flyAns:'시험 '+pct(te)+' · 훈련 '+pct(tr)+' · 혼화재 배합 '+pct(accOf(mix,manualRule))+' / 시멘트만 '+pct(accOf(plain,manualRule)),roboAns:'학습비 0원 · 규칙 2조 · 대신 전문가의 경험이 든다',flyMood:'coffee',flyMove:'tilt',roboMood:'meh',roboMove:'sway',asstMood:'firm',docMood:'scale',
      measure:'봉인된 시험지 258건 중 '+Math.round(te*258)+'건을 맞혔어요. <b>'+pct(te)+'</b>예요. 늘 "'+(MAJ?'합격':'불합격')+'"이라고만 답하는 기준선 '+pct(BASE)+'보다는 한참 낫지만, 등급 판정은 할 수 없고 혼화재를 섞은 배합에서 더 자주 틀려요.',
      verdict:'보류입니다. 매뉴얼 씨는 지금처럼 "35MPa 합격" 1차 판정을 계속 맡아요. 다만 등급 판정과 새 배합은 새 직원을 뽑습니다. 그리고 매뉴얼 씨의 점수 '+pct(te)+'를 <b>첫 합격선</b>으로 정합니다.',
      key:'첫 합격선 = '+pct(te)+' (시험지 258건, 35MPa 이상인가)',
      render(box){box.innerHTML=statGrid([[pct(te),'매뉴얼 씨 · 시험지','m'],[pct(BASE),'기준선 · 늘 "'+(MAJ?'합격':'불합격')+'"'],[pct(accOf(mix,manualRule)),'혼화재 배합 ('+mix.length+'건)'],[pct(accOf(plain,manualRule)),'시멘트만 쓴 배합 ('+plain.length+'건)']],2);}};},
  alts:[{label:'신입 매뉴얼(물-시멘트비 &lt; 0.6)이었다면',async run(box){const f=ruleFn({mode:'novice',cut:0.6,age:28});box.innerHTML=statGrid([[pct(accOf(TE,f)),'신입 매뉴얼 · 시험지','m'],[pct(MAN_TE),'전문가 매뉴얼 · 시험지','m']],2)+'<p class="foot">결합재라는 지식 하나가 차이를 만든다. 지식이 없는 규칙은 데이터가 아니라 사람의 한계에서 멈춘다.</p>';}},
        {label:'기계가 고른 문턱으로',async run(box){const G=gridSearch(),f=ruleFn({mode:'expert',cut:G.best.cut,age:G.best.age});box.innerHTML=statGrid([[G.best.cut.toFixed(2)+' / '+G.best.age+'일','기계가 고른 문턱'],[pct(accOf(TE,f)),'시험지 정확도','a']],2)+'<p class="foot">숫자만 데이터로 바꿔도 조금 나아진다. 하지만 모양(문턱 두 개의 직사각형)은 그대로라 크게 달라지지 않는다. 합격선은 사람이 쓴 원래 규칙의 점수로 정한다.</p>';}}]});

/* ── 08 채용 계획 ── */
(function(){const HOOK={'5-1':'가장 가까운 배합 k개에게 묻는다. 평균이면 회귀, 다수결이면 분류','5-2':'강도를 직선 하나로. 가중치가 곧 재료의 영향','5-3':'곡선을 그리면 과대적합, 그래서 규제(릿지·라쏘)','5-4':'점수를 확률로 바꿔 35MPa 합격을 판정','5-5':'특징마다 확률을 곱한다. 특징이 서로 얽히면?','5-6':'스무고개로 판정. 매뉴얼 씨와 가장 닮은 지원자','5-7':'나무 여러 그루의 다수결','5-8':'가장 넓은 길을 내는 경계, 커널로 곡선까지','5-9':'앞 나무의 오답을 다음 나무가 고친다','5-10':'정답 없이 숫자 손글씨를 k개 무리로','5-11':'무리의 계보도를 그린다','5-12':'빽빽한 곳만 무리로, 외톨이는 잡음','5-13':'무리를 종 모양 분포로 본다','5-14':'64차원을 몇 개 축으로 줄인다','5-15':'이웃을 지키며 2차원 지도로 펼친다'};
  $('planTb').innerHTML='<thead><tr><th>편</th><th>지원자</th><th>본부 › 팀</th><th>실기 과제지</th><th>한 줄 소개</th></tr></thead><tbody>'+EPISODES.slice(1).map(e=>{const uns=['군집','차원축소'].indexOf(e.team)>=0;return '<tr><td>'+(e.ready?'<a href="'+e.f+'">'+e.n+'</a>':e.n)+'</td><td><b>'+e.t+'</b></td><td>'+(uns?'비지도':'지도')+' › '+e.team+'팀</td><td>'+(uns?'손글씨 숫자':'콘크리트')+'</td><td>'+(HOOK[e.n]||'')+'</td></tr>';}).join('')+'</tbody>';})();

/* ══ 확인 문제 ══ */
var QUIZ={
q1:[{q:'룰 베이스와 머신러닝을 가르는 기준으로 가장 알맞은 것은?',o:[['판단 규칙을 사람이 쓰나, 데이터로 만드나',1,'둘 다 판단하는 프로그램(인공지능)이다. 규칙의 출처가 다르다.'],['컴퓨터를 쓰나 안 쓰나',0,'둘 다 컴퓨터로 돈다.'],['정확도가 높은가 낮은가',0,'03장처럼 지식이 좋으면 규칙도 꽤 정확하다. 정확도는 기준이 아니다.']]},
 {q:'"룰 베이스는 인공지능이 아니다"라는 말은?',o:[['틀리다. 전문가 시스템처럼 규칙으로 판단하는 프로그램도 인공지능이라 불러 왔다',1,'머신러닝은 인공지능을 만드는 여러 방법 가운데 하나다.'],['맞다. 인공지능은 데이터로 배워야만 한다',0,'흔한 오해다. 데이터로 배우는 것은 머신러닝의 특징이다.']]}],
q2:[{q:'=IF(AND(0.45&lt;0.5, 14&gt;=28), "합격", "불합격")의 결과는?',o:[['"불합격"',1,'AND는 둘 다 참이어야 참이다. 재령 14일이 28일 조건을 못 맞춘다.'],['"합격"',0,'물-결합재비 조건만 맞았다. AND는 둘 다 필요하다.']]},
 {q:'NOR(A, B)가 1(참)인 경우는?',o:[['A도 B도 거짓일 때',1,'NOR = NOT(A OR B). 둘 다 아닐 때만 참이다.'],['A와 B가 모두 참일 때',0,'그것은 AND다.'],['둘 중 하나만 참일 때',0,'그것은 XOR다.']]}],
q3:[{q:'신입 매뉴얼(물-시멘트비)이 전문가 매뉴얼(물-결합재비)보다 혼화재 배합에서 더 틀리는 까닭은?',o:[['슬래그·플라이 애시도 시멘트처럼 굳는다는 지식이 없어서',1,'전문가는 결합재로 묶어 본다. 그 지식이 규칙의 정확도를 바꾼다.'],['조항 수가 적어서',0,'두 매뉴얼은 조항 수가 같다. 다른 것은 지식이다.'],['데이터가 틀려서',0,'같은 데이터다.']]},
 {q:'시험지를 연 뒤에 문턱을 고쳐 점수를 올렸다. 이 점수는?',o:[['더 이상 처음 보는 데이터의 점수가 아니다',1,'시험지를 보고 푼 셈이다. 시험지는 마지막에 한 번만 연다.'],['더 정확한 점수다',0,'시험지에만 맞춘 점수라 새 배합에서는 더 낮게 나온다.']]}],
q4:[{q:'"기계에게 문턱 찾기 맡기기"에서 기계가 한 일은?',o:[['규칙의 모양은 그대로 두고 두 숫자(문턱)를 훈련 데이터로 골랐다',1,'가장 단순한 머신러닝이다. 모양까지 배우는 것은 5-4 이후의 지원자들이다.'],['새 조항을 만들어 냈다',0,'모양(조항)은 사람이 정한 그대로다.'],['시험지로 문턱을 골랐다',0,'훈련 772건만 썼다. 시험지는 채점에만 쓴다.']]},
 {q:'전문가 특징 2개를 받은 로지스틱 회귀가 날것 특징 8개를 받은 모델 이상으로 잘한 까닭은?',o:[['물-결합재비와 log 재령에 강도를 가르는 지식이 이미 담겨 있어서',1,'좋은 특징은 지식의 압축이다. 머신러닝도 지식을 버리지 않는다.'],['특징이 적을수록 늘 좋아서',0,'늘 그렇지는 않다. 어떤 특징이냐가 중요하다.']]}],
q5:[{q:'"이 배합의 28일 강도는 몇 MPa?"는?',o:[['회귀',1,'크기와 차이에 뜻이 있는 숫자를 맞힌다.'],['분류',0,'답이 이름표가 아니라 연속된 숫자다.']]},
 {q:'숫자로 적힌 공장 코드(1, 2, 3)를 맞히는 문제는?',o:[['분류',1,'숫자지만 크기·차이에 뜻이 없는 이름표다.'],['회귀',0,'"공장 3 − 공장 1 = 2"에는 뜻이 없다.']]}],
q6:[{q:'시험 기록(강도) 없이 배합만 1,000건 있다. 할 수 있는 일은?',o:[['비슷한 배합끼리 묶어 보는 비지도 학습',1,'정답이 없으니 맞히기는 못 하고, 구조를 찾는다.'],['35MPa 합격 분류',0,'합격 여부를 배우려면 정답(강도)이 있어야 한다.']]}],
q7:[{q:'합격선을 매뉴얼 씨의 "훈련" 점수가 아니라 "시험지" 점수로 정하는 까닭은?',o:[['지원자도 시험지로 채점하니 같은 시험지로 비교해야 공정하다',1,'같은 시험지, 같은 규정이 면접의 첫 조건이다.'],['시험지 점수가 늘 더 높아서',0,'이번에는 조금 높았지만 우연이다.']]},
 {q:'"늘 많은 쪽만 답하는" 기준선을 함께 보는 까닭은?',o:[['배우지 않고도 낼 수 있는 점수보다 나아야 배운 것이기 때문',1,'이 데이터는 합격·불합격이 반반이라 기준선이 약 50%다. 한쪽이 드물면 기준선이 높아진다.'],['기준선은 볼 필요가 없다',0,'불균형 데이터에서는 기준선이 90%가 넘기도 한다.']]}]};

/* ══ 공통 B·C·D·E·F 데이터 ══ */
var PROMPTS=[
 {t:'① 우리 공장의 작업 표준서를 IF로',use:'현장 규칙을 엑셀 수식과 순서도로 옮길 때',p:'너는 품질 관리 실무 교육 조교다. 아래 작업 표준서 조항을 엑셀 IF 함수(필요하면 AND/OR/중첩 IF)와 순서도(텍스트 그림)로 옮겨 줘. 각 조항이 어떤 입력 열을 쓰는지 표로 정리하고, 조항끼리 부딪히는 경우(한 배합이 두 조항에 걸리는 경우)가 있으면 찾아 줘.\n\n〔작업 표준서: 예) 1. 물-결합재비 0.5 미만이고 재령 28일 이상이면 합격 2. 고성능 감수제 10kg/m³ 이상이면 재령 7일부터 합격 3. 그 밖에는 불합격〕'},
 {t:'② 규칙 vs 데이터 비교 코드',use:'내 데이터에서 매뉴얼 규칙과 머신러닝을 같은 시험지로 비교할 때',p:'Google Colab에서 돌아가는 파이썬 코드를 써 줘. scikit-learn만 쓴다.\n- 데이터: 〔ucimlrepo의 fetch_ucirepo(id=165), 특징 8개, 타겟 = 압축 강도〕\n- 타겟을 "〔35〕MPa 이상이면 1"로 바꾼다.\n- train_test_split(test_size=0.25, stratify=강도를 25·45MPa로 나눈 3등급, random_state=42)\n- 규칙: 〔물/(시멘트+슬래그+플라이 애시) < 0.5 이고 재령 ≥ 28〕의 훈련·시험 정확도\n- 같은 규칙 모양에서 두 문턱을 훈련 데이터로 격자 탐색\n- LogisticRegression(표준화는 훈련 데이터로만 fit)의 시험 정확도\n- 기준선(DummyClassifier)\n결과를 한 표로 출력하고, 시험 데이터는 마지막 채점에만 쓰게 해 줘.'},
 {t:'③ 회귀일까 분류일까 판정',use:'내 업무 질문이 회귀·분류·비지도 중 무엇인지 헷갈릴 때',p:'내 업무 질문 다섯 개를 보고 각각 회귀/이진 분류/다중 분류/비지도(군집) 중 무엇인지 판정해 줘. 판정 기준은 "답의 크기와 차이에 의미가 있는가", "정답 기록이 있는가" 두 가지다. 경계 사례면 두 가지 방식을 모두 설명하고 어느 쪽이 업무에 맞는지 근거를 들어 줘.\n\n〔질문: 1) … 2) … 3) … 4) … 5) …〕'},
 {t:'④ 수업용 카드 만들기',use:'05장 같은 회귀·분류 카드 연습 문제가 더 필요할 때',p:'제조 현장(〔콘크리트·사출·반도체 등〕) 상황으로 "회귀일까 분류일까" 카드 12장을 만들어 줘. 각 카드에 질문, 답의 예, 정답(회귀/분류/경계 사례), 한 줄 해설을 붙인다. 경계 사례(숫자 코드, 순서 있는 등급, 몇 가지 값만 쓰는 숫자)를 3장 넣어 줘.'}];
var GL=[
 ['큰 그림','인공지능','사람의 지적인 판단을 컴퓨터가 하도록 만드는 기술 전체. 룰 베이스와 머신러닝을 모두 포함한다.'],
 ['큰 그림','룰 베이스','사람이 쓴 if 규칙으로 판단하는 방식. 전문가 시스템이 대표적이다.'],
 ['큰 그림','전문가 시스템','전문가의 지식을 규칙으로 옮겨 판단을 대신하게 한 프로그램. 1980년대에 널리 쓰였다.'],
 ['큰 그림','머신러닝','판단 규칙을 사람이 다 쓰지 않고 데이터에서 만드는 방법.'],
 ['큰 그림','딥러닝','층을 깊게 쌓은 신경망으로 특징까지 배우는 머신러닝의 한 갈래.'],
 ['규칙','IF 함수','조건이 참이면 앞의 값, 거짓이면 뒤의 값을 내는 엑셀 함수.'],
 ['규칙','순서도','판단 단계를 마름모(질문)와 화살표로 그린 그림.'],
 ['규칙','NOR','두 조건이 모두 거짓일 때만 참. NOR만으로 모든 논리 회로를 만들 수 있다.'],
 ['콘크리트','물-결합재비','물 ÷ (시멘트 + 고로 슬래그 + 플라이 애시). 낮을수록 대체로 강하다.'],
 ['콘크리트','결합재','물과 반응해 굳는 가루 재료. 시멘트와 슬래그·플라이 애시 같은 혼화재를 함께 부른다.'],
 ['콘크리트','재령','콘크리트를 친 뒤 지난 날수. 같은 배합이라도 재령이 길수록 강하다.'],
 ['콘크리트','압축 강도','공시체를 눌러 부서질 때의 힘을 넓이로 나눈 값(MPa).'],
 ['데이터','샘플','표의 한 줄. 여기서는 배합 하나.'],
 ['데이터','특징','판단에 쓰는 입력 열. 재료 일곱 가지와 재령.'],
 ['데이터','타겟','맞히려는 정답 열. 강도(회귀) 또는 합격 여부·등급(분류).'],
 ['데이터','시험지','학습에 쓰지 않고 마지막 채점에만 쓰는 데이터(테스트 데이터).'],
 ['문제','회귀','크기와 차이에 뜻이 있는 숫자를 맞히는 문제.'],
 ['문제','분류','정해진 이름표 가운데 하나를 고르는 문제. 둘이면 이진, 여럿이면 다중 분류.'],
 ['문제','지도 학습','정답이 달린 데이터로 배우는 방식.'],
 ['문제','비지도 학습','정답 없이 데이터의 구조(무리, 축)를 찾는 방식.'],
 ['평가','정확도','맞힌 샘플의 비율.'],
 ['평가','기준선','배우지 않고도 낼 수 있는 점수. 늘 많은 쪽만 답하는 모델 등.'],
 ['평가','합격선','이 시리즈에서 지원자가 넘어야 하는 점수. 매뉴얼 씨의 시험지 점수.']];
var ANALOGY=[
 ['판단','면접관이 지원서를 보고 합격·불합격을 정하는 일',2,'초파리가 냄새 증거를 모아 피할지 다가갈지 고르는 지각 결정','사람과 초파리의 판단에는 목적과 상황 이해가 들어간다. 모델은 입력→결정의 대응만 배운다'],
 ['룰 베이스','업무 매뉴얼·작업 표준서',2,'타고난 반사(정해진 자극에 정해진 반응)','반사는 진화가 만든 규칙이고, 매뉴얼은 사람이 쓴 규칙이다. 둘 다 새 상황에 약하다는 점만 닮았다'],
 ['머신러닝','선배의 판정 기록 수천 장을 보고 감을 익히는 신입',2,'경험으로 냄새와 결과를 짝지어 배우는 연상 학습','신입은 이유를 물을 수 있지만, 모델은 기록에 없는 이유를 모른다'],
 ['시험지 봉인','기출문제 대신 처음 보는 문제로 치는 시험',1,'','뇌에는 "봉인된 시험지"가 따로 없다. 평가 절차의 비유다'],
 ['합격선','작년 직원의 성적보다 나아야 새로 뽑는다',1,'','현실의 채용은 점수 하나로 정하지 않는다. 이 시리즈의 약속일 뿐이다']];
var REFS=[
 'Yeh, I.-C. (1998). Modeling of strength of high-performance concrete using artificial neural networks. <i>Cement and Concrete Research</i>, 28(12), 1797–1808. <a href="https://doi.org/10.1016/S0008-8846(98)00165-3" target="_blank" rel="noopener">doi:10.1016/S0008-8846(98)00165-3</a> — 실기 과제지',
 'Yeh, I.-C. (2007). Concrete Compressive Strength [Dataset]. UCI Machine Learning Repository. <a href="https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength" target="_blank" rel="noopener">archive.ics.uci.edu/dataset/165</a> (CC BY 4.0) — 02~04장',
 'Abrams, D. A. (1918). <i>Design of Concrete Mixtures</i>. Structural Materials Research Laboratory, Lewis Institute, Bulletin 1. — 물-시멘트비가 낮을수록 강도가 높다는 경험칙(에이브럼스 법칙), 03장',
 'Samuel, A. L. (1959). Some studies in machine learning using the game of checkers. <i>IBM Journal of Research and Development</i>, 3(3), 210–229. <a href="https://doi.org/10.1147/rd.33.0210" target="_blank" rel="noopener">doi:10.1147/rd.33.0210</a> — "명시적으로 프로그래밍하지 않고 배우는" 머신러닝, 04장',
 'Russell, S., &amp; Norvig, P. (2021). <i>Artificial Intelligence: A Modern Approach</i> (4th ed.). Pearson. — 룰 베이스(지식 기반) 시스템과 학습, 01장',
 'Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. <i>Journal of Machine Learning Research</i>, 12, 2825–2830. — A장 코드',
 'DasGupta, S., Ferreira, C. H., &amp; Miesenböck, G. (2014). FoxP influences the speed and accuracy of a perceptual decision in <i>Drosophila</i>. <i>Science</i>, 344(6186), 901–904. <a href="https://doi.org/10.1126/science.1252114" target="_blank" rel="noopener">doi:10.1126/science.1252114</a> — D장 판단 비유'];
var TIP={who:'머신러닝을 처음 배우는 대학생·직장인·고등학생, 특히 제조·품질 현장 실무자. 수식이 거의 없고 엑셀 IF 함수를 알면 충분하다',
 goal:'① 룰 베이스와 머신러닝의 차이를 "규칙을 누가 쓰나"로 설명한다 ② 규칙이 무너지는 세 가지 모습(지식 부족, 규칙 충돌, 모양의 한계)을 예로 든다 ③ 업무 질문을 회귀·분류·비지도로 나눈다 ④ 시험지 봉인과 합격선의 뜻을 말한다',
 time:'읽기 20분 · 실습(02·03·04장) 20분 · 카드 10분 · Colab 10분',
 plans:[['강의형 50분','00장 공고 1장을 인쇄해 나눠 주기 5분 → 02장 IF 함수와 논리 연산 10분 → 03장 매뉴얼 편집기로 학생마다 문턱을 정해 훈련 점수 경쟁, 마지막에 시험지를 함께 열기 15분 → 04장 기계에게 맡기기 10분 → 05장 카드 10분.'],['현장 OJT 60분','참가자 각자 자기 공정의 작업 표준서 한 조항을 IF로 써 오기(B장 ①) → 그 조항이 무너지는 사례를 03장 표처럼 무리별로 찾기 → 06장 지도/비지도로 우리 데이터가 어디에 속하는지 토론.'],['1:1 튜터링','03장에서 신입 매뉴얼과 전문가 매뉴얼을 번갈아 눌러 보게 하고, "어떤 지식이 차이를 만들었나"를 말로 설명하게 한다.']],
 myths:[['룰 베이스는 인공지능이 아니다','규칙으로 판단하는 프로그램도 인공지능이다. 머신러닝은 그 규칙을 데이터로 만드는 방법이다'],['머신러닝을 쓰면 현장 지식은 필요 없다','04장처럼 좋은 특징(물-결합재비)이 모델을 살린다. 지식은 특징으로 들어간다'],['숫자로 된 답이면 회귀다','공장 코드·슬럼프 등급 코드처럼 숫자라도 이름표면 분류다'],['훈련 점수가 높으면 좋은 규칙이다','문턱을 계속 고치면 훈련 점수는 오른다. 처음 보는 시험지 점수로 판단한다'],['시험지는 여러 번 봐도 된다','본 순간 시험지가 아니다. 고르는 일은 훈련 데이터 안에서 한다']],
 tasks:[['과제 1 · 우리 공장 매뉴얼','자기 업무의 판단 규칙 하나를 IF 함수와 순서도로 쓰고, 그 규칙이 틀리는 사례를 세 가지 적는다.'],['과제 2 · 문턱 경쟁','03장 매뉴얼 편집기에서 훈련 점수를 최대로 만든 뒤 시험지를 연다. 훈련과 시험 점수 차이를 기록하고 이유를 쓴다.'],['과제 3 · AI와 함께 규칙 vs 데이터','B장 ② 프롬프트로 코드를 받아 07장 체크리스트로 검사한 뒤 Colab에서 돌리고, 결과표를 이 페이지의 숫자와 비교한다.']],
 ai:'생성형 AI에게 "작업 표준서를 IF로 바꿔 줘", "규칙과 모델을 비교하는 코드를 써 줘"를 시키는 것은 좋은 출발점이다. 다만 AI가 낸 규칙의 문턱 숫자와 코드의 시험지 처리는 반드시 사람이 확인하게 한다. 07장 체크리스트를 제출물에 붙이게 하면 "받아 적기"가 "검증하기"가 된다.'};
var CODE={title:'채용 공고 — 매뉴얼 씨 대 데이터',sub:'numpy, pandas, scikit-learn, ucimlrepo · CPU 몇 초',file:'job_posting_rules_vs_data',
 desc:'이 편을 그대로 옮겼다. UCI에서 콘크리트 데이터를 받아 시험지를 봉인하고(1·2번 칸), 매뉴얼 씨의 IF 규칙과 신입 매뉴얼을 채점하고(3번), 같은 규칙 모양의 문턱을 훈련 데이터로 찾고(4번), 깊이 1 결정 트리가 문턱 하나를 배우게 한 뒤(5번), 로지스틱 회귀로 "모양까지" 배우게 한다(6번). 노트북으로 저장하면 칸마다 셀이 나뉜다.',
 cap:'실행 결과 (scikit-learn 1.9, CPU). 브라우저 계산과 같은 시험지(258건)를 쓴다. 로지스틱 회귀는 브라우저가 경사 하강으로 풀어 소수점 아래가 조금 다를 수 있다.'};
