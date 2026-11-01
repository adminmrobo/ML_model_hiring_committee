var BOARD_LINE='R² 0.735 · 76.7%'; /* 회귀 합격선(매뉴얼 경험식 R²) · 분류 합격선 */
/* ══ 5-1 KNN · 콘크리트 실기 과제지 ══════════════════════════════════ */
const ROWS=concRows(),TR=ROWS.filter(r=>!r.test),TE=ROWS.filter(r=>r.test);
const fv=r=>FEAT.map(f=>r[f]);
const Ar=TR.map(fv),Br=TE.map(fv),SC=ML.scaler(Ar),A=SC.t(Ar),B=SC.t(Br);
const YTR=TR.map(r=>r.strength),YTE=TE.map(r=>r.strength),PTR=TR.map(r=>r.pass),PTE=TE.map(r=>r.pass);
const PNAME=['불합격','합격'];

/* ── KNN 엔진 (이 편 전용, 브라우저·node 공용) ──
   거리 제곱을 모두 재고, 가까운 k개를 고른다. 같은 거리면 앞 번호(행 번호가 작은 배합)를 먼저 고른다.
   거리 가중은 scikit-learn과 같다: 무게 = 1/거리, 거리 0인 이웃이 있으면 그 이웃들의 평균만 쓴다 */
const KNN=(function(){
  function d2(a,b){let s=0;for(let j=0;j<a.length;j++){const t=a[j]-b[j];s+=t*t;}return s;}
  function nearest(X,x,k,skip){k=Math.min(k,X.length-(skip==null?0:1));const id=new Int32Array(k),dd=new Float64Array(k);let n=0;
    for(let i=0;i<X.length;i++){if(i===skip)continue;const v=d2(X[i],x);let p;
      if(n<k)p=n++;else if(v<dd[k-1])p=k-1;else continue;
      while(p>0&&dd[p-1]>v){dd[p]=dd[p-1];id[p]=id[p-1];p--;}dd[p]=v;id[p]=i;}
    return {id:Array.from(id.subarray(0,n)),d2:Array.from(dd.subarray(0,n))};}
  function agg(nb,y,k,w){const id=nb.id.slice(0,k),dd=nb.d2.slice(0,k);
    if(w){const z=id.filter((_,j)=>dd[j]===0);if(z.length)return z.reduce((s,i)=>s+y[i],0)/z.length;let sw=0,sv=0;id.forEach((i,j)=>{const ww=1/Math.sqrt(dd[j]);sw+=ww;sv+=ww*y[i];});return sv/sw;}
    return id.reduce((s,i)=>s+y[i],0)/id.length;}
  function predict(X,y,x,k,w,skip){return agg(nearest(X,x,k,skip),y,k,w);}
  /* 다수결: 동점이면 작은 이름표(0)를 고른다(scikit-learn과 같다) */
  function vote(X,y,x,k){const nb=nearest(X,x,k);let s=0;nb.id.forEach(i=>s+=y[i]);return s*2>nb.id.length?1:0;}
  return {d2,nearest,agg,predict,vote};})();
function r2(y,p){const m=ML.mean(y);let a=0,b=0;for(let i=0;i<y.length;i++){a+=(y[i]-p[i])*(y[i]-p[i]);b+=(y[i]-m)*(y[i]-m);}return 1-a/b;}
function mae(y,p){let s=0;for(let i=0;i<y.length;i++)s+=Math.abs(y[i]-p[i]);return s/y.length;}
/* 매뉴얼 씨의 경험식: 28일 강도 96.5/7^(물-결합재비) × 재령 보정 t/(4+0.85t) */
const manualMPa=(wb,t)=>96.5/Math.pow(7,wb)*t/(4+0.85*t);
const MAN=TE.map(r=>manualMPa(r.wb,r.age)),LINE_R2=r2(YTE,MAN),LINE_MAE=mae(YTE,MAN);
const MAN_WC=r2(YTE,TE.map(r=>manualMPa(r.water/r.cement,r.age)));
const BASE=ML.mean(YTR),BASE_R2=r2(YTE,YTE.map(()=>BASE)),BASE_MAE=mae(YTE,YTE.map(()=>BASE));
const manualAcc=TE.filter(r=>manualRule(r)===r.pass).length/TE.length;
/* 시험지 예측은 한 번씩만 계산해 둔다 */
const CACHE={};
function testPred(kind){if(CACHE[kind])return CACHE[kind];const t0=performance.now();let p;
  if(kind==='std')p=B.map(x=>KNN.predict(A,YTR,x,5,false));
  else if(kind==='raw')p=Br.map(x=>KNN.predict(Ar,YTR,x,5,false));
  else if(kind==='dist')p=B.map(x=>KNN.predict(A,YTR,x,5,true));
  else if(kind==='man')p=MAN;
  CACHE[kind]=p;CACHE[kind+'_ms']=performance.now()-t0;return p;}
const fmtR=v=>fx(v,3),fmtM=v=>v.toFixed(2);
function cmGrid(M,labels,colsFor){const K=labels.length;let h='<div class="cm" style="grid-template-columns:86px repeat('+K+',minmax(48px,1fr));max-width:'+(110+K*90)+'px"><div class="h">정답 ＼ 예측</div>'+labels.map(k=>'<div class="h">'+k+'</div>').join('');
  M.forEach((row,a)=>{h+='<div class="h" style="justify-content:end;text-align:right">'+labels[a]+'</div>'+row.map((v,b)=>a===b?'<div class="c'+(v?' d':'')+'" style="'+(v?'background:'+rgba(colsFor(a),.85):'')+'">'+v+'</div>':'<div class="c'+(v?' x':'')+'">'+(v||'·')+'</div>').join('');});return h+'</div>';}
function barRows(rows,max){return rows.map(r=>'<div class="row"><span style="text-align:left;font-family:var(--font)">'+r[0]+'</span><div class="tr"><i style="left:0;width:'+Math.max(0,Math.min(100,r[1]/max*100))+'%;background:'+(r[2]||'var(--blue)')+'"></i></div><span>'+r[3]+'</span></div>').join('');}

/* ── 히어로 ── */
heroRoom($('heroCv'),{sym:'k',no:'APP-01',name:'KNN\n(k-최근접 이웃)',role:'지원자 · 입문팀',banner:'모델 채용 위원회 · 5-1 면접',doc:'front',asst:'board',fly:'clip',robo:'calc',loop:10,
  beats:[{"at": 0.4, "who": "fly", "mood": "point", "move": "tilt", "say": "비슷하다는 게 뭔데?"}, {"at": 2.4, "who": "doc", "mood": "board", "move": "nod", "say": "가까운 k개에게 물어요"}, {"at": 4.4, "who": "robo", "mood": "calc", "move": "shake", "say": "매번 772건을 뒤져?"}, {"at": 6.4, "who": "asst", "mood": "spark", "move": "jump", "say": "표준화하면 돼요!"}, {"at": 8.4, "who": "doc", "mood": "front", "move": "stand"}],
  motif:(function(){const rg=RNG(11),P=[];for(let i=0;i<24;i++)P.push([268+rg()*140,92+rg()*80]);
    return function(ctx,t,W,H,C){const qx=338+Math.cos(t*0.6)*38,qy=132+Math.sin(t*0.9)*24,o=P.map((p,i)=>[(p[0]-qx)**2+(p[1]-qy)**2,i]).sort((a,b)=>a[0]-b[0]).slice(0,5);
      ctx.strokeStyle=C.c_app;ctx.lineWidth=2;o.forEach(e=>{const p=P[e[1]];ctx.beginPath();ctx.moveTo(qx,qy);ctx.lineTo(p[0],p[1]);ctx.stroke();});
      P.forEach((p,i)=>circ(ctx,p[0],p[1],3.4,o.some(e=>e[1]===i)?C.c_app:rgba(C.text,.55),null));
      circ(ctx,qx,qy,6,C.panel,C.stamp,2.5);ctx.fillStyle=C.muted;ctx.font=font(12,500,1);ctx.textAlign='left';ctx.fillText('k = 5',300,190);};})()});
$('heroLine').textContent='R² '+fmtR(LINE_R2);

/* ── 01 장면 ── */
comic('comicIntro',[{"bg": "site", "prop": "?", "cast": [{"who": "fly", "mood": "point", "x": 30, "move": "tilt"}, {"who": "asst", "mood": "board", "x": 72}], "say": {"who": "fly", "txt": "새 배합이 들어왔어. 이 배합 강도는 몇이야?"}, "cap": "① 처음 보는 배합 — 정답을 모른다"}, {"bg": "site", "prop": "k = 5", "cast": [{"who": "doc", "mood": "board", "x": 30, "move": "nod"}, {"who": "asst", "mood": "spark", "x": 72, "move": "pop"}], "say": {"who": "doc", "txt": "장부에서 가장 비슷한 배합 다섯을 찾아요. 강도는 평균, 합격은 다수결."}, "cap": "② 이웃 다섯에게 묻기 — 평균이면 회귀, 다수결이면 분류"}, {"bg": "night", "prop": "kg ≠ 일", "cast": [{"who": "robo", "mood": "calc", "x": 30, "move": "shake"}, {"who": "fly", "mood": "arms", "x": 72, "move": "nod", "flip": true}], "say": {"who": "fly", "txt": "kg과 일을 그대로 재면 골재 무게가 거리를 다 먹어. 단위부터 맞춰."}, "cap": "③ 표준화 — 단위가 다른 특징의 키를 맞춘다"}]);

/* ── 01 자기소개: 특징 2개로 이웃 찾기 ── */
const F2=r=>[r.wb,Math.log10(r.age)];
const R2tr=TR.map(F2),SC2=ML.scaler(R2tr),A2=SC2.t(R2tr);
(function(){const cv=$('knnCv'),H=340,st={wb:0.45,age:28,k:5,w:false,real:null,pick:0};
  const xs=R2tr.map(r=>r[0]),X0=Math.min(...xs)-0.02,X1=Math.max(...xs)+0.02,Y0=-0.08,Y1=Math.log10(365)+0.08;
  const order=TE.map((r,i)=>i).sort((a,b)=>((a*37)%258)-((b*37)%258));
  let geo=null;
  function compute(){const s=SC2.t([[st.wb,Math.log10(st.age)]])[0],nb=KNN.nearest(A2,s,st.k);return {s,nb,mean:KNN.agg(nb,YTR,st.k,st.w),votes:nb.id.reduce((a,i)=>a+PTR[i],0)};}
  function draw(){const W=fit(cv,H),ctx=cv.getContext('2d'),C=cols(),pl=50,pr=14,pt=14,pb=40,gw=W-pl-pr,gh=H-pt-pb;
    const sx=x=>pl+(x-X0)/(X1-X0)*gw,sy=y=>pt+(1-(y-Y0)/(Y1-Y0))*gh;geo={sx,sy,pl,pt,gw,gh};
    ctx.clearRect(0,0,W,H);ctx.font=font(12);ctx.fillStyle=C.muted;ctx.strokeStyle=C.rule;ctx.lineWidth=1;
    for(let x=0.3;x<=X1;x+=0.1){ctx.beginPath();ctx.moveTo(sx(x),pt);ctx.lineTo(sx(x),pt+gh);ctx.stroke();ctx.textAlign='center';ctx.fillText(x.toFixed(1),sx(x),H-22);}
    [1,3,7,28,90,365].forEach(d=>{const v=Math.log10(d);ctx.beginPath();ctx.moveTo(pl,sy(v));ctx.lineTo(pl+gw,sy(v));ctx.stroke();ctx.textAlign='right';ctx.fillText(d+'일',pl-6,sy(v)+4);});
    ctx.textAlign='center';ctx.fillText('물-결합재비 (물 ÷ 결합재) — 왼쪽일수록 물이 적다',pl+gw/2,H-4);
    ctx.save();ctx.translate(11,pt+gh/2);ctx.rotate(-Math.PI/2);ctx.fillText('재령 (로그 눈금)',0,0);ctx.restore();
    const c=compute(),inN=new Set(c.nb.id);
    R2tr.forEach((r,i)=>{if(!inN.has(i))circ(ctx,sx(r[0]),sy(r[1]),2.8,rgba(PTR[i]?C.blue:C.text,.42),null);});
    const qx=sx(st.wb),qy=sy(Math.log10(st.age)),rk=Math.sqrt(c.nb.d2[c.nb.d2.length-1]);
    const rx=rk*SC2.sd[0]*gw/(X1-X0),ry=rk*SC2.sd[1]*gh/(Y1-Y0);
    ctx.save();ctx.beginPath();ctx.rect(pl,pt,gw,gh);ctx.clip();ctx.setLineDash([6,4]);ctx.strokeStyle=C.blue;ctx.lineWidth=1.6;ctx.fillStyle=rgba(C.blue,.07);
    ctx.beginPath();ctx.ellipse(qx,qy,Math.max(rx,4),Math.max(ry,4),0,0,7);ctx.fill();ctx.stroke();ctx.setLineDash([]);
    c.nb.id.forEach((i,j)=>{const r=R2tr[i];ctx.strokeStyle=rgba(C.blue,.8);ctx.lineWidth=st.w?Math.min(5,0.6+1.2/Math.max(Math.sqrt(c.nb.d2[j]),0.25)):1.6;ctx.beginPath();ctx.moveTo(qx,qy);ctx.lineTo(sx(r[0]),sy(r[1]));ctx.stroke();});
    c.nb.id.forEach(i=>{const r=R2tr[i];circ(ctx,sx(r[0]),sy(r[1]),4.6,PTR[i]?C.blue:C.text,C.panel,1.5);});ctx.restore();
    ctx.fillStyle=C.stamp;ctx.beginPath();for(let a=0;a<10;a++){const rr_=a%2?4:10,an=-Math.PI/2+a*Math.PI/5;ctx.lineTo(qx+rr_*Math.cos(an),qy+rr_*Math.sin(an));}ctx.closePath();ctx.fill();ctx.strokeStyle=C.panel;ctx.lineWidth=1.5;ctx.stroke();
    ctx.font=font(12,700);ctx.textAlign='left';ctx.fillStyle=rgba(C.panel,.88);ctx.fillRect(pl+6,pt+4,186,50);ctx.fillStyle=C.blue;ctx.fillText('● 35MPa 이상 (훈련 배합)',pl+12,pt+19);ctx.fillStyle=C.text;ctx.fillText('● 35MPa 미만',pl+12,pt+34);ctx.fillStyle=C.stamp;ctx.fillText('★ 새 배합 (누르면 옮겨 간다)',pl+12,pt+49);
    info(c);}
  function info(c){const ys=c.nb.id.map(i=>YTR[i]),shown=ys.slice(0,10).map(v=>v.toFixed(1)).join(', ')+(ys.length>10?' … 외 '+(ys.length-10)+'건':''),pass=c.votes*2>st.k?1:0;
    let h='새 배합: 물-결합재비 <b>'+st.wb.toFixed(2)+'</b> · 재령 <b>'+Math.round(st.age)+'일</b>. 가장 가까운 훈련 배합 <b>'+st.k+'건</b>의 강도: '+shown+' MPa.<br>'+
      '회귀 → '+(st.w?'거리 가중 평균':'평균')+' <b>'+c.mean.toFixed(1)+'MPa</b> · 분류 → 합격 '+c.votes+'표 : 불합격 '+(st.k-c.votes)+'표 → <b>'+PNAME[pass]+'</b>'+(c.votes*2===st.k?' (동점이라 작은 이름표 "불합격")':'')+
      '. 매뉴얼 경험식은 '+manualMPa(st.wb,st.age).toFixed(1)+'MPa.';
    if(st.real!=null)h+='<br>이 배합은 시험지에서 꺼낸 것이다. 실제 강도 <b>'+st.real.toFixed(1)+'MPa</b>('+PNAME[st.real>=PASS_MPA?1:0]+'), KNN의 오차 '+fx(c.mean-st.real,1)+'MPa.';
    else if(st.k===1)h+='<br>k = 1이면 가장 가까운 한 건의 답을 그대로 베낀다. 그 한 건이 이상한 시편이면 그대로 틀린다.';
    else if(st.k>=30)h+='<br>k가 크면 멀리 있는 배합까지 투표에 끼어 답이 전체 평균 쪽으로 끌려간다. 타원이 얼마나 커졌는지 보자.';
    $('knnInfo').innerHTML=h;}
  function sync(){$('kK').value=st.k;$('kKO').textContent=st.k;$('kW').setAttribute('aria-pressed',st.w);}
  $('kK').addEventListener('input',e=>{st.k=+e.target.value;sync();draw();});
  $('kW').addEventListener('click',()=>{st.w=!st.w;sync();draw();});
  $('kRand').addEventListener('click',()=>{const r=TE[order[st.pick++%order.length]];st.wb=r.wb;st.age=r.age;st.real=r.strength;draw();});
  $('kReset').addEventListener('click',()=>{st.wb=0.45;st.age=28;st.k=5;st.w=false;st.real=null;sync();draw();});
  cv.addEventListener('click',e=>{if(!geo)return;const r=cv.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top;
    st.wb=clamp(X0+(mx-geo.pl)/geo.gw*(X1-X0),X0,X1);st.age=clamp(Math.pow(10,Y0+(1-(my-geo.pt)/geo.gh)*(Y1-Y0)),1,365);st.real=null;draw();});
  sync();draw();onRedraw(draw);})();

/* ── 02 실기 과제지 ── */
$('lineR2').textContent='R² '+fmtR(LINE_R2);$('wbR2').textContent=fmtR(LINE_R2);$('wcR2').textContent=fmtR(MAN_WC);
/* 매뉴얼 경험식: 계산기 + 예측 vs 실제 */
(function(){const cv=$('manCv'),H=280;
  function draw(){const W=fit(cv,H),ctx=cv.getContext('2d'),C=cols(),pl=44,pr=12,pt=12,pb=36,s=Math.min(W-pl-pr,H-pt-pb),m=90,sx=v=>pl+v/m*s,sy=v=>pt+s-v/m*s;
    ctx.clearRect(0,0,W,H);ctx.strokeStyle=C.rule;ctx.strokeRect(pl,pt,s,s);ctx.fillStyle=C.muted;ctx.font=font(12);
    [0,30,60,90].forEach(v=>{ctx.textAlign='center';ctx.fillText(v,sx(v),pt+s+15);ctx.textAlign='right';ctx.fillText(v,pl-5,sy(v)+4);});
    ctx.textAlign='center';ctx.fillText('실제 강도 (MPa)',pl+s/2,pt+s+31);ctx.save();ctx.translate(11,pt+s/2);ctx.rotate(-Math.PI/2);ctx.fillText('경험식 (MPa)',0,0);ctx.restore();
    ctx.setLineDash([4,4]);ctx.strokeStyle=C.muted;ctx.beginPath();ctx.moveTo(sx(0),sy(0));ctx.lineTo(sx(m),sy(m));ctx.stroke();ctx.setLineDash([]);
    TE.forEach((r,i)=>circ(ctx,sx(r.strength),sy(Math.min(m,MAN[i])),3,rgba(C.c_man,.7),null));
    ctx.fillStyle=C.c_man;ctx.font=font(12,700,1);ctx.textAlign='left';ctx.fillText('R² '+fmtR(LINE_R2)+' · MAE '+fmtM(LINE_MAE),pl+8,pt+16);}
  function calc(){const wb=+$('mWb').value,t=+$('mAge').value,f28=96.5/Math.pow(7,wb),ft=manualMPa(wb,t);$('mWbO').textContent=wb.toFixed(2);
    $('manInfo').innerHTML='f<sub>28</sub> = 96.5 / 7<sup>'+wb.toFixed(2)+'</sup> = <b>'+f28.toFixed(1)+'MPa</b> → 재령 '+t+'일: × '+t+' / (4 + 0.85×'+t+') = <b>'+ft.toFixed(1)+'MPa</b>'+(ft>=PASS_MPA?' (35MPa 이상)':'')+'. 상수 96.5·7·4·0.85는 교과서 값 그대로이고 데이터로 고치지 않았다.';}
  $('mWb').addEventListener('input',calc);$('mAge').addEventListener('change',calc);calc();draw();onRedraw(draw);})();
/* 핵심 장면: 표준화 전후의 이웃 */
(function(){const sel=$('utPick'),SHOW=['cement','slag','water','superplasticizer','coarse_agg','age'];
  const info=TE.map((r,i)=>{const nr=KNN.nearest(Ar,Br[i],5),ns=KNN.nearest(A,B[i],5);return {nr,ns,ov:nr.id.filter(j=>ns.id.indexOf(j)>=0).length};});
  TE.forEach((r,i)=>sel.add(new Option('시험 #'+(r.i+1)+' · 재령 '+r.age+'일 · 실제 '+r.strength.toFixed(1)+'MPa · 겹치는 이웃 '+info[i].ov+'/5',i)));
  const def=TE.findIndex(r=>r.i===399);sel.value=def>=0?def:0;
  const SH={cement:'시멘트',slag:'슬래그',water:'물',superplasticizer:'감수제',coarse_agg:'굵은골재',age:'재령'};
  function tbl(nb,q,isStd){return '<thead><tr><th>#</th>'+SHOW.map(f=>'<th>'+SH[f]+'</th>').join('')+'<th>강도</th><th>거리</th></tr></thead><tbody>'+
    nb.id.map((j,n)=>{const r=TR[j];return '<tr><td>'+(r.i+1)+'</td>'+SHOW.map(f=>{const diff=f==='age'?r.age!==q.age:false;return '<td'+(diff?' style="color:var(--stamp);font-weight:700"':'')+'>'+(+r[f].toFixed(1))+'</td>';}).join('')+'<td>'+r.strength.toFixed(1)+'</td><td>'+Math.sqrt(nb.d2[n]).toFixed(isStd?2:1)+'</td></tr>';}).join('')+'</tbody>';}
  function share(a,b,sd){const t=a.map((v,j)=>{const d=(v-b[j])/(sd?sd[j]:1);return d*d;}),s=t.reduce((x,y)=>x+y,0)||1;return t.map(v=>v/s);}
  function bars(sh,lab){const ord=FEAT.map((f,j)=>[sh[j],j]);return '<div style="font-size:13px;font-weight:700;margin-bottom:4px">'+lab+'</div>'+ord.map(o=>'<div class="row"><span style="text-align:left;font-family:var(--font)">'+FEATK[FEAT[o[1]]]+'</span><div class="tr"><i style="left:0;width:'+(o[0]*100)+'%;background:'+(FEAT[o[1]]==='age'||FEAT[o[1]]==='water'?'var(--stamp)':'var(--blue)')+'"></i></div><span>'+pct(o[0],0)+'</span></div>').join('');}
  function show(){const i=+sel.value,q=TE[i],I=info[i],pr=ML.mean(I.nr.id.map(j=>YTR[j])),ps=ML.mean(I.ns.id.map(j=>YTR[j]));
    $('utHead').innerHTML='<div class="tw"><table class="tb" style="font-size:13px;margin-bottom:6px"><thead><tr><th>새 배합</th>'+SHOW.map(f=>'<th>'+SH[f]+'</th>').join('')+'<th>실제 강도</th></tr></thead><tbody><tr><td><b>#'+(q.i+1)+'</b></td>'+SHOW.map(f=>'<td><b>'+(+q[f].toFixed(1))+'</b></td>').join('')+'<td><b>'+q.strength.toFixed(1)+'</b></td></tr></tbody></table></div>';
    $('utRaw').innerHTML=tbl(I.nr,q,false);$('utStd').innerHTML=tbl(I.ns,q,true);
    $('utBarRaw').innerHTML=bars(share(Br[i],Ar[I.nr.id[0]]),'표준화 안 함 · 1위 이웃 #'+(TR[I.nr.id[0]].i+1));
    $('utBarStd').innerHTML=bars(share(Br[i],Ar[I.ns.id[0]],SC.sd),'표준화 · 1위 이웃 #'+(TR[I.ns.id[0]].i+1));
    const dAgeR=I.nr.id.filter(j=>TR[j].age!==q.age).length,dAgeS=I.ns.id.filter(j=>TR[j].age!==q.age).length;
    $('utInfo').innerHTML='표준화 안 함: 이웃 평균 <b>'+pr.toFixed(1)+'MPa</b>(오차 '+fx(pr-q.strength,1)+') · 표준화: <b>'+ps.toFixed(1)+'MPa</b>(오차 '+fx(ps-q.strength,1)+') · 실제 '+q.strength.toFixed(1)+'MPa. 두 목록에 함께 든 이웃은 '+I.ov+'건이다. '+
      '재령이 다른 이웃(빨간 칸)이 표준화 전 '+dAgeR+'건, 표준화 후 '+dAgeS+'건. '+(i===def?'표준화하지 않으면 28일 배합에게 3일·7일 배합이 "가장 비슷하다"고 뽑힌다. kg 단위로 몇십씩 나는 시멘트·슬래그 차이가 재령 21일 차이와 같은 무게로 더해지기 때문이다. 아래 막대에서 빨간 막대(물, 재령)의 몫을 비교해 보자.':'다른 배합도 골라 보자. 겹치는 이웃이 0~2건인 배합에서 차이가 크다.');}
  sel.addEventListener('change',show);show();
  const v=SC.sd.map(s=>s*s),tot=v.reduce((a,b)=>a+b,0);
  $('stdTbl').innerHTML='<thead><tr><th>특징</th><th>평균</th><th>표준편차</th><th>거리에서의 몫 (표준화 안 함)</th></tr></thead><tbody>'+FEAT.map((f,j)=>'<tr><td>'+FEATK[f]+'</td><td>'+SC.mu[j].toFixed(1)+'</td><td>'+SC.sd[j].toFixed(1)+'</td><td'+(v[j]/tot<0.02?' style="color:var(--stamp);font-weight:700"':'')+'>'+pct(v[j]/tot)+'</td></tr>').join('')+'</tbody>';})();
/* 산점도와 표 */
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
  $('concTbl').innerHTML='<table><thead><tr><th>#</th>'+FEAT.map(f=>'<th><span class="tg x">특징 X</span>'+FEATK[f]+'</th>').join('')+'<th><span class="tg t">타겟 y</span>압축 강도(MPa)</th><th><span class="tg t">과제 B</span>35MPa 합격</th><th>용도</th></tr></thead><tbody>'+
    ROWS.map(r=>'<tr><td>'+(r.i+1)+'</td>'+FEAT.map(f=>'<td>'+r[f]+'</td>').join('')+'<td class="y">'+r.strength+'</td><td>'+PNAME[r.pass]+'</td><td>'+(r.test?'<span class="pill no">봉인</span>':'훈련')+'</td></tr>').join('')+'</tbody></table>';})();

/* ── 03 AI 비서 ── */
const MAIN_PROMPT=`너는 머신러닝 수업 조교다. Google Colab에서 바로 실행되는 파이썬 코드를 써 줘.

[환경] Colab 기본 라이브러리(numpy, pandas, scikit-learn)와 ucimlrepo만 쓴다. 첫 줄에 scikit-learn 버전을 출력한다.
[데이터] 〔ucimlrepo의 fetch_ucirepo(id=165)〕 로 콘크리트 압축 강도 데이터를 읽는다.
  - 특징 X: 〔시멘트, 고로 슬래그, 플라이 애시, 물, 고성능 감수제, 굵은 골재, 잔골재, 재령〕 8개 (열 순서대로 이름을 붙인다)
  - 타겟 y: 〔압축 강도(MPa)〕 — 연속된 숫자이므로 회귀다
[규정]
  1) 강도를 〔25, 45〕MPa로 나눈 3등급으로 stratify 해서 train_test_split(test_size=0.25, random_state=42)로 먼저 나눈다.
  2) StandardScaler는 훈련 데이터로만 fit 하고, 시험 데이터는 transform만 한다.
  3) KNeighborsRegressor로 학습한다. n_neighbors 말고는 인자 이름을 붙여 쓴다(weights="distance" 등).
  4) k는 훈련 데이터 안에서 교차 검증으로 고른다. 시험 데이터는 마지막 채점에 한 번만 쓴다.
[출력] 고른 k, 시험 R²와 MAE, 늘 평균만 답하는 기준선(DummyRegressor)의 R²와 MAE, 표준화를 뺐을 때의 시험 R².
[금지] 시험 데이터로 학습하거나 k를 고르지 않는다. 데이터를 지어내지 않는다.
[형식] 셀 단위로 나누고, 각 셀 위에 무엇을 하는지 한국어 주석을 단다. 코드 뒤에 내가 확인해야 할 점 3가지를 적어 줘.`;
$('mainPrompt').textContent=MAIN_PROMPT;$('copyMain').addEventListener('click',()=>copyText(MAIN_PROMPT,'프롬프트를 복사했습니다'));
(function(){const LINES=[
 ['import pandas as pd'],['from ucimlrepo import fetch_ucirepo'],['from sklearn.model_selection import train_test_split'],['from sklearn.preprocessing import StandardScaler'],
 ['from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor',0,'두 지원자를 다 불러온 것 자체는 괜찮다. 어디에 무엇을 쓰는지가 문제다.'],['from sklearn.metrics import r2_score',0,'회귀 점수 R²를 쓴다. 괜찮다.'],[''],
 ['uci = fetch_ucirepo(id=165)',0,'UCI에서 콘크리트 데이터를 실제로 받았다. 괜찮다.'],['X = uci.data.features',0,'특징 여덟 열이다. 괜찮다.'],['y = uci.data.targets.iloc[:, 0]   # 압축 강도(MPa)',0,'타겟은 강도 숫자다. 회귀 과제다. 괜찮다.'],[''],
 ['X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)',1,'<b>stratify와 random_state가 없다.</b> 돌릴 때마다 점수가 달라지고, 고강도 배합이 한쪽으로 쏠릴 수 있다. 위원회 규정대로 강도 등급으로 <code>stratify</code>하고 <code>random_state=42</code>를 넣는다.'],
 ['sc = StandardScaler()',0,'표준화 도구를 만들었다. 괜찮다.'],['X_train = sc.fit_transform(X_train)',0,'훈련 데이터로 평균·표준편차를 구해 바꿨다. 괜찮다.'],
 ['X_test = sc.fit_transform(X_test)   # 시험지도 표준화',2,'<b>시험지로 다시 fit 했다.</b> 시험지는 시험지의 평균·표준편차로 바뀌어 훈련 데이터와 눈금이 달라진다. 거리가 엉뚱해지고, 시험지 정보도 새어 들어간다. <code>sc.transform(X_test)</code>만 쓴다.'],[''],
 ['best_k, best = 1, -1',0,'가장 좋은 k를 기억할 자리다. 이 줄만으로는 괜찮다.'],
 ['for k in range(1, 31):   # 가장 좋은 k 찾기',0,'k를 1부터 30까지 재 보는 것은 좋은 생각이다. 무엇으로 재는지가 문제다.'],
 ['    m = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)',3,'<b>회귀 과제에 분류기를 썼다.</b> 강도(MPa)는 연속된 숫자라 <code>ValueError: Unknown label type: continuous</code>가 난다. 혹시 정수로 반올림해 넣으면 오류 없이 "가장 흔한 강도 값"을 다수결로 고르는 엉뚱한 모델이 된다. <code>KNeighborsRegressor</code>를 쓴다.'],
 ['    s = r2_score(y_test, m.predict(X_test))',4,'<b>시험지로 k를 골랐다.</b> 시험지 점수가 가장 좋은 k를 고르면, 그 점수는 시험지에 맞춘 점수다. k는 훈련 데이터 안의 교차 검증(<code>cross_val_score</code>, 04장의 LOO)으로 고른다.'],
 ['    if s > best: best_k, best = k, s',4,'<b>시험지로 k를 골랐다.</b> 바로 윗줄과 같은 문제다. 여기서 기억한 best가 그대로 보고된다.'],[''],
 ['model = KNeighborsRegressor(best_k, "distance").fit(X_train, y_train)',5,'<b>키워드 전용 인자를 위치로 넘겼다.</b> 지금 버전에서는 n_neighbors 말고는 이름을 붙여야 해서 <code>TypeError</code>가 난다. <code>KNeighborsRegressor(n_neighbors=best_k, weights="distance")</code>로 쓴다.'],
 ['print("R²:", best)',4,'<b>시험지로 고른 점수를 그대로 보고했다.</b> 고른 뒤의 모델로, 시험지를 한 번만 열어 채점한 점수를 보고한다.']];
  const host=$('bugCode'),found=new Set();
  host.innerHTML=LINES.map((l,i)=>'<div class="ln" role="listitem" tabindex="0" data-i="'+i+'"><span class="no">'+(i+1)+'</span><span>'+hl(l[0]||' ')+'</span></div>').join('');
  function score(){$('bugScore').textContent=found.size+' / 5';if(found.size===5){$('bugScore').className='pill ok';}}
  function pick(el){const i=+el.getAttribute('data-i'),l=LINES[i];if(!l[0])return;
    if(l[1]){found.add(l[1]);host.querySelectorAll('.ln').forEach(x=>{if(LINES[+x.getAttribute('data-i')][1]===l[1])x.classList.add('hit');});$('bugMsg').innerHTML=(i+1)+'번 줄: '+l[2]+(found.size===5?'<br><b>다섯 군데를 모두 찾았다.</b> 고친 코드는 A장의 Colab 코드와 같은 순서(나누기 → 훈련으로만 표준화 → 훈련 안에서 k 고르기 → 시험지 한 번)를 따른다.':'');}
    else{el.classList.add('miss');$('bugMsg').innerHTML=(i+1)+'번 줄: '+(l[2]||'이 줄은 괜찮다.');}score();}
  host.addEventListener('click',e=>{const el=e.target.closest('.ln');if(el)pick(el);});
  host.addEventListener('keydown',e=>{if((e.key==='Enter'||e.key===' ')&&e.target.classList.contains('ln')){e.preventDefault();pick(e.target);}});
  $('bugReveal').addEventListener('click',()=>{LINES.forEach((l,i)=>{if(l[1]){found.add(l[1]);host.querySelector('[data-i="'+i+'"]').classList.add('hit','reveal');}});score();$('bugMsg').innerHTML='빨간 줄 다섯 군데: ① 12번 나누기 규정 ② 15번 시험지로 다시 fit ③ 19번 회귀에 분류기 ④ 20·21·24번 시험지로 k 고르고 그 점수 보고 ⑤ 23번 키워드 전용 인자.';});
  $('bugReset').addEventListener('click',()=>{found.clear();host.querySelectorAll('.ln').forEach(x=>x.classList.remove('hit','miss','reveal'));$('bugScore').className='pill';score();$('bugMsg').textContent='줄을 하나 눌러 보세요.';});score();})();
(function(){const CK=[['데이터를 실제로 읽었나','AI가 예제 데이터를 지어내거나 다른 데이터로 바꾸지 않았는지, 행이 1,030개인지 본다.'],['타겟이 숫자인지 이름표인지에 맞는 지원자를 불렀나','강도(MPa)는 KNeighborsRegressor, 35MPa 합격 여부는 KNeighborsClassifier다.'],['나누기를 먼저 하고, 표준화는 훈련 데이터로만 fit 했나','시험지에 fit_transform이 보이면 눈금이 어긋나고 누수다.'],['표준화를 뺀 것은 아닌가','KNN은 단위에 가장 예민하다. 이 데이터에서 표준화를 빼면 R² 0.767 → 0.716이다.'],['stratify와 random_state를 넣었나','등급 비율이 시험지에도 유지되고, 점수가 매번 같게 나온다.'],['k를 훈련 데이터 안에서 골랐나','시험지로 k를 고르면 점수가 부풀려진다. 교차 검증(cross_val_score, LOO)을 쓴다.'],['인자가 내 scikit-learn 버전에 맞나','print(sklearn.__version__). weights, metric은 이름을 붙여 넘긴다.'],['훈련 점수를 성적으로 보고하지 않았나','KNN의 훈련 점수는 자기 자신이 이웃이라 k = 1에서 거의 완벽하다. 의미가 없다.'],['결과를 기준선·합격선과 비교했나','늘 평균만 답하는 기준선(R² 0, MAE 13.5MPa)과 매뉴얼 경험식(R² 0.735)보다 낫지 않으면 뽑을 까닭이 없다.']];
  const ul=$('checks');ul.innerHTML=CK.map((c,i)=>'<li><input type="checkbox" id="ck'+i+'"><label for="ck'+i+'"><b>'+c[0]+'</b><small>'+c[1]+'</small></label></li>').join('');
  function upd(){const n=ul.querySelectorAll('input:checked').length;ul.querySelectorAll('li').forEach(li=>li.classList.toggle('done',li.querySelector('input').checked));$('checkScore').textContent='확인 '+n+' / '+CK.length+(n===CK.length?' — 이제 돌려 본다. 결과도 확인 대상이다.':'');}
  ul.addEventListener('change',upd);upd();})();

/* ── 04 수습 기간 ① 예측 지도 (특징 2개) ── */
(function(){const cv=$('mapCv'),H=330,st={k:1,w:false,play:0};
  const xs=R2tr.map(r=>r[0]),X0=Math.min(...xs)-0.02,X1=Math.max(...xs)+0.02,Y0=-0.08,Y1=Math.log10(365)+0.08;
  const LOO2={};
  function loo2(k,w){const key=k+(w?'w':'');if(LOO2[key]!=null)return LOO2[key];const p=A2.map((x,i)=>KNN.predict(A2,YTR,x,k,w,i));return (LOO2[key]=r2(YTR,p));}
  function draw(){const W=fit(cv,H),ctx=cv.getContext('2d'),C=cols(),pl=50,pr=60,pt=10,pb=38,gw=W-pl-pr,gh=H-pt-pb,cs=W<600?7:8;
    const sx=x=>pl+(x-X0)/(X1-X0)*gw,sy=y=>pt+(1-(y-Y0)/(Y1-Y0))*gh,ix=X=>X0+(X-pl)/gw*(X1-X0),iy=Y=>Y0+(1-(Y-pt)/gh)*(Y1-Y0);
    ctx.clearRect(0,0,W,H);
    for(let X=pl;X<pl+gw;X+=cs)for(let Y=pt;Y<pt+gh;Y+=cs){const s=SC2.t([[ix(X+cs/2),iy(Y+cs/2)]])[0],v=KNN.predict(A2,YTR,s,st.k,st.w);ctx.fillStyle=rgba(C.blue,0.04+0.86*clamp(v/80,0,1));ctx.fillRect(X,Y,cs,cs);}
    ctx.strokeStyle=C.rule;ctx.lineWidth=1;ctx.strokeRect(pl,pt,gw,gh);ctx.fillStyle=C.muted;ctx.font=font(12);
    for(let x=0.3;x<=X1;x+=0.1){ctx.textAlign='center';ctx.fillText(x.toFixed(1),sx(x),H-20);}
    [1,3,7,28,90,365].forEach(d=>{ctx.textAlign='right';ctx.fillText(d+'일',pl-5,sy(Math.log10(d))+4);});
    ctx.textAlign='center';ctx.fillText('물-결합재비',pl+gw/2,H-3);ctx.save();ctx.translate(10,pt+gh/2);ctx.rotate(-Math.PI/2);ctx.fillText('재령',0,0);ctx.restore();
    R2tr.forEach(r=>circ(ctx,sx(r[0]),sy(r[1]),1.7,rgba(C.text,.5),null));
    /* 색 막대 */
    const bx=pl+gw+16;for(let i=0;i<=40;i++){ctx.fillStyle=rgba(C.blue,0.04+0.86*i/40);ctx.fillRect(bx,pt+gh-(i+1)*gh/41,14,gh/41+1);}
    ctx.fillStyle=C.muted;ctx.textAlign='left';[0,20,40,60,80].forEach(v=>ctx.fillText(v,bx+18,pt+gh-v/80*gh+4));ctx.fillText('MPa',bx-2,pt+gh+16);
    info();}
  function info(){const k=st.k,tr=r2(YTR,A2.map(x=>KNN.predict(A2,YTR,x,k,st.w))),lo=loo2(k,st.w);
    $('mapInfo').innerHTML='k = <b>'+k+'</b>'+(st.w?' (거리 가중)':'')+' · 훈련 R² <b>'+fmtR(tr)+'</b>(자기 자신 포함) · 하나 빼고 맞히기(LOO) R² <b>'+fmtR(lo)+'</b>. '+
      (k<=2?'k가 작으면 지도가 얼룩덜룩하다. 시편 하나하나의 우연(같은 배합인데 강도가 다른 시편)까지 그대로 따라간다. 훈련 R²는 높지만 LOO는 낮다 — 과대적합이다.':
       k<=20?'얼룩이 사라지고 "물이 적고 오래 굳힐수록 진하다"는 큰 무늬가 드러난다. 훈련 R²와 LOO R²의 차이가 줄었다.':
       'k가 너무 크면 먼 배합까지 평균에 끼어 지도가 흐려진다. 특히 가장자리(아주 낮은 물-결합재비, 1일·365일)가 안쪽 값으로 끌려간다 — 과소적합이다.')+
      (st.w?' 거리 가중을 켜면 훈련 R²는 k와 상관없이 거의 1이다(자기 자신의 거리가 0이라 자기 답만 쓴다). 그래서 훈련 점수로는 아무것도 고를 수 없다.':'');}
  function sync(){$('mapK').value=st.k;$('mapKO').textContent=st.k;$('mapW').setAttribute('aria-pressed',st.w);}
  $('mapK').addEventListener('input',e=>{st.k=+e.target.value;sync();draw();});
  $('mapW').addEventListener('click',()=>{st.w=!st.w;sync();draw();});
  const SEQ=[1,2,3,5,8,12,20,30,50,75,100];
  $('mapPlay').addEventListener('click',async()=>{if(st.play)return;st.play=1;$('mapPlay').disabled=true;for(const k of SEQ){st.k=k;sync();draw();await wait(REDUCE?60:650);}st.k=5;sync();draw();st.play=0;$('mapPlay').disabled=false;});
  sync();draw();onRedraw(draw);})();

/* ── 04 수습 기간 ② LOO로 k 고르기 (특징 8개) ── */
const LOO=(function(){let N=null;
  function lists(){if(N)return N;N={std:[],raw:[],self:[]};for(let i=0;i<A.length;i++){N.std.push(KNN.nearest(A,A[i],30,i));N.raw.push(KNN.nearest(Ar,Ar[i],30,i));N.self.push(KNN.nearest(A,A[i],30));}return N;}
  function score(kind,k){const L=lists()[kind];return r2(YTR,L.map(nb=>KNN.agg(nb,YTR,k,false)));}
  return {lists,score};})();
(function(){const cv=$('looCv'),H=280,RES={},KS=[...Array(30)].map((_,i)=>i+1);let shown=0,busy=false;
  function draw(){const W=fit(cv,H),ctx=cv.getContext('2d'),C=cols(),pl=46,pr=14,pt=12,pb=36,gw=W-pl-pr,gh=H-pt-pb,lx=k=>pl+(k-1)/29*gw,y0=0.5,y1=1,ly=v=>pt+(1-(clamp(v,y0,y1)-y0)/(y1-y0))*gh;
    ctx.clearRect(0,0,W,H);ctx.strokeStyle=C.rule;ctx.strokeRect(pl,pt,gw,gh);ctx.fillStyle=C.muted;ctx.font=font(12);
    [1,5,10,15,20,25,30].forEach(k=>{ctx.textAlign='center';ctx.fillText(k,lx(k),H-20);});ctx.fillText('k (이웃 수)',pl+gw/2,H-4);
    [0.5,0.6,0.7,0.8,0.9,1].forEach(v=>{ctx.textAlign='right';ctx.fillText(v.toFixed(1),pl-5,ly(v)+4);ctx.strokeStyle=rgba(C.muted,.18);ctx.beginPath();ctx.moveTo(pl,ly(v));ctx.lineTo(pl+gw,ly(v));ctx.stroke();});
    ctx.save();ctx.translate(11,pt+gh/2);ctx.rotate(-Math.PI/2);ctx.textAlign='center';ctx.fillText('R² (훈련 772건)',0,0);ctx.restore();
    const S=[['self',C.muted,'훈련 R² (자기 자신 포함)'],['std',C.blue,'LOO · 표준화'],['raw',C.amber,'LOO · 표준화 안 함']];
    S.forEach(([key,col,lab],si)=>{const pts=KS.slice(0,shown).map(k=>[lx(k),ly(RES[k][key])]);ctx.strokeStyle=col;ctx.lineWidth=2.4;ctx.setLineDash(key==='self'?[5,4]:[]);ctx.beginPath();pts.forEach((p,i)=>i?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]));ctx.stroke();ctx.setLineDash([]);pts.forEach(p=>circ(ctx,p[0],p[1],2.8,col,null));
      ctx.fillStyle=col;ctx.font=font(12,700);ctx.textAlign='right';ctx.fillText(lab,pl+gw-8,pt+16+si*16);});
    if(shown){let bk=1;KS.slice(0,shown).forEach(k=>{if(RES[k].std>RES[bk].std)bk=k;});circ(ctx,lx(bk),ly(RES[bk].std),7,null,C.stamp,2.5);ctx.fillStyle=C.stamp;ctx.font=font(12,700);ctx.textAlign='left';ctx.fillText('최고 k = '+bk,lx(bk)-8,ly(RES[bk].std)-12);}}
  function chips(i){document.querySelectorAll('#looChips button').forEach((b,j)=>b.setAttribute('aria-pressed',j===i?'true':'false'));}
  function calc(k){if(!RES[k])RES[k]={self:LOO.score('self',k),std:LOO.score('std',k),raw:LOO.score('raw',k)};}
  function info(){if(!shown)return;let bk=1;KS.slice(0,shown).forEach(k=>{if(RES[k].std>RES[bk].std)bk=k;});const r5=RES[Math.min(5,shown)];
    $('looInfo').innerHTML='k = '+shown+'까지 쟀다. 표준화한 LOO R²가 가장 높은 k는 <b>'+bk+'</b>('+fmtR(RES[bk].std)+')'+(shown>=5?', 기본값 k = 5는 '+fmtR(RES[5].std)+'. 그래서 05장은 k = 5로 시험지를 연다':'')+'. '+
      (shown>=5?'같은 k = 5에서 표준화 안 함은 '+fmtR(RES[5].raw)+'. 모든 k에서 주황 선이 파란 선 아래다. ':'')+'훈련 R²(회색 점선)는 k = 1에서 '+fmtR(RES[1].self)+'로 거의 1이지만, 자기 자신이 이웃이라 생긴 착시다. 이 772건은 모두 훈련 데이터다. 시험지는 아직 봉인되어 있다.';}
  async function play(){if(busy)return;busy=true;$('looPlay').disabled=true;LOO.lists();for(let k=shown+1;k<=30;k++){calc(k);shown=k;if(!REDUCE){chips(k%4);draw();await wait(k<=6?260:90);}}chips(-1);draw();info();busy=false;$('looPlay').disabled=false;}
  $('looPlay').addEventListener('click',play);
  $('looReset').addEventListener('click',()=>{if(busy)return;shown=0;chips(-1);draw();$('looInfo').textContent='▶를 누르면 k마다 772번씩 "하나 빼고 맞히기"를 해서 점을 찍는다.';});
  seg('#looChips button',v=>{const T=['훈련 772건 중 하나를 잠시 빼서 "처음 보는 배합" 역할을 맡긴다.','남은 771건과 거리를 재서 가까운 k건을 고른다. 자기 자신은 이웃이 될 수 없다.','이웃 k건의 강도를 평균 내어 뺀 배합의 강도를 맞혀 본다.','772건 모두를 한 번씩 빼 보고, 맞힌 값과 실제 강도로 R²를 낸다. 이것을 k마다 반복한다.'];$('looInfo').textContent=T[+v];});
  draw();onRedraw(draw);})();

/* ── 05 성과 평가 ── */
const MODELS=[['std','KNN · 표준화 (k = 5)','var(--blue)'],['raw','KNN · 표준화 안 함','var(--amber)'],['dist','KNN · 표준화 + 거리 가중','var(--ok)'],['man','매뉴얼 경험식 (합격선)','var(--c-man)']];
function renderEval(){const p=testPred('std'),R=r2(YTE,p),M=mae(YTE,p);
  $('evalStat').innerHTML=statGrid([['R² '+fmtR(R),'시험 R² (MAE '+fmtM(M)+'MPa)','a'],['R² '+fmtR(LINE_R2),'회귀 합격선 · 매뉴얼 경험식 (MAE '+fmtM(LINE_MAE)+')','m'],['R² '+fmtR(BASE_R2),'기준선 · 늘 훈련 평균 '+BASE.toFixed(1)+'MPa (MAE '+fmtM(BASE_MAE)+')'],[(R>=LINE_R2?'+':'')+fx(R-LINE_R2,3),'합격선과의 차이 (R²)','s']],4);
  $('rTbl').innerHTML='<thead><tr><th>모델 (시험지 258건)</th><th>R²</th><th>MAE (MPa)</th><th>판정</th></tr></thead><tbody>'+
    MODELS.map(m=>{const q=testPred(m[0]),rr=r2(YTE,q);return '<tr><td><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:'+m[2]+';margin-right:6px"></span>'+m[1]+'</td><td>'+fmtR(rr)+'</td><td>'+fmtM(mae(YTE,q))+'</td><td>'+(m[0]==='man'?'합격선':(rr>LINE_R2&&rr>BASE_R2?'<b style="color:var(--ok)">합격</b>':'<b style="color:var(--stamp)">보류</b>'))+'</td></tr>';}).join('')+
    '<tr><td>기준선 · 늘 훈련 평균</td><td>'+fmtR(BASE_R2)+'</td><td>'+fmtM(BASE_MAE)+'</td><td>—</td></tr></tbody>';
  $('pvChoice').innerHTML=MODELS.map((m,i)=>'<button data-v="'+m[0]+'" aria-pressed="'+(i===0)+'">'+m[1]+'</button>').join('');
  const cv=$('pvCv');let cur='std';
  function draw(){const W=fit(cv,300),ctx=cv.getContext('2d'),C=cols(),pl=44,pt=10,s=Math.min(W-60,250),m=90,sx=v=>pl+v/m*s,sy=v=>pt+s-v/m*s,q=testPred(cur),col=getComputedStyle(document.documentElement).getPropertyValue(MODELS.find(x=>x[0]===cur)[2].slice(4,-1)).trim()||C.blue;
    ctx.clearRect(0,0,W,300);ctx.strokeStyle=C.rule;ctx.strokeRect(pl,pt,s,s);ctx.fillStyle=C.muted;ctx.font=font(12);
    [0,30,60,90].forEach(v=>{ctx.textAlign='center';ctx.fillText(v,sx(v),pt+s+15);ctx.textAlign='right';ctx.fillText(v,pl-5,sy(v)+4);});
    ctx.textAlign='center';ctx.fillText('실제 강도 (MPa)',pl+s/2,pt+s+31);ctx.save();ctx.translate(11,pt+s/2);ctx.rotate(-Math.PI/2);ctx.fillText('예측 강도 (MPa)',0,0);ctx.restore();
    ctx.setLineDash([4,4]);ctx.strokeStyle=C.muted;ctx.beginPath();ctx.moveTo(sx(0),sy(0));ctx.lineTo(sx(m),sy(m));ctx.stroke();ctx.strokeStyle=rgba(C.stamp,.6);ctx.beginPath();ctx.moveTo(sx(35),pt);ctx.lineTo(sx(35),pt+s);ctx.moveTo(pl,sy(35));ctx.lineTo(pl+s,sy(35));ctx.stroke();ctx.setLineDash([]);
    YTE.forEach((y,i)=>circ(ctx,sx(y),sy(Math.min(m,q[i])),3.2,rgba(col,.7),null));
    ctx.fillStyle=col;ctx.font=font(12,700,1);ctx.textAlign='left';ctx.fillText('R² '+fmtR(r2(YTE,q))+' · MAE '+fmtM(mae(YTE,q)),pl+8,pt+16);
    const mx=Math.max(...YTE),hi=YTE.map((y,i)=>[y,q[i]]).filter(v=>v[0]>=70),under=hi.filter(v=>v[1]<v[0]).length;
    $('pvInfo').innerHTML=(cur==='man'?'매뉴얼 경험식은 학습 없이 교과서 상수만으로 대각선 근처에 모인다. 사람의 지식이 얼마나 센지 보여 주는 합격선이다.':
      cur==='raw'?'표준화하지 않으면 재령이 다른 배합을 이웃으로 데려와 아래위로 크게 흩어진다(02장의 #400처럼). R²가 합격선 '+fmtR(LINE_R2)+' 아래다.':
      cur==='dist'?'거리 가중을 켜면 똑같은 배합(거리 0)이 있을 때 그 답을 그대로 쓰고, 가까운 이웃의 말을 크게 듣는다. 이 데이터는 같은 배합을 여러 재령으로 시험한 줄이 많아 효과가 크다.':
      '대각선 근처에 모였지만 오른쪽 위 끝을 보자.')+' 실제 70MPa 이상인 시험 배합 '+hi.length+'건 중 '+under+'건을 낮게 예측했다. KNN은 이웃의 평균이라 훈련 데이터의 가장 센 값(최고 '+Math.max(...YTR).toFixed(1)+'MPa)을 넘는 답을 낼 수 없고, 끝으로 갈수록 안쪽으로 끌려간다.';}
  seg('#pvChoice button',v=>{cur=v;draw();});draw();onRedraw(draw);
  /* 과제 B */
  const vs=B.map(x=>KNN.vote(A,PTR,x,5)),vr=Br.map(x=>KNN.vote(Ar,PTR,x,5)),as=ML.acc(PTE,vs),ar=ML.acc(PTE,vr),Mc=ML.confusion(PTE,vs,2);
  $('cmB').innerHTML=cmGrid(Mc,PNAME,k=>k?cols().blue:cols().muted);
  $('clsStat').innerHTML=statGrid([[pct(as),'시험 정확도 · 표준화','a'],[pct(ar),'표준화 안 함'],[pct(PASSLINE),'분류 합격선 · 5-0 매뉴얼 씨','m']],3)+'<p class="foot">강도 미달인데 "합격"이라 한 배합 '+Mc[0][1]+'건. 이웃을 찾는 과정은 회귀와 똑같고 마지막 줄만 "평균" 대신 "다수결"이다. 5-4 로지스틱 회귀(86.4%)와 거의 같은 성적이다.</p>';
  cost();}
function cost(){delete CACHE.std;const p=testPred('std'),ms=CACHE.std_ms;
  $('costStat').innerHTML=statGrid([['0 ms','fit = 외우기 (저장만)'],[ms.toFixed(1)+' ms','시험 258건 predict (이 브라우저)','a'],[(258*772).toLocaleString(),'거리 계산 횟수 (258 × 772)'],[(258*772*8).toLocaleString(),'뺄셈·제곱 횟수 (× 특징 8)','s']],2);
  $('costInfo').innerHTML='예측 한 건마다 훈련 772건 전부와 거리를 잰다. 훈련 데이터가 10배가 되면 예측 시간도 약 10배가 된다(정렬 대신 이웃 k개만 고르게 짜도 거리는 모두 재야 한다). scikit-learn은 kd-tree 같은 색인으로 이 수를 줄이지만, 특징이 많아지면 색인의 효과가 떨어진다(06장).';return p;}
$('costRun').addEventListener('click',()=>cost());

/* ── 면접 결과 발표 ── */
Interview('iv1',{title:'KNN은 콘크리트 강도 예측 업무(회귀)에 합격할까?',
  q:'조건: 봉인된 시험지 258건에서 시험 R²가 회귀 합격선(매뉴얼 경험식 R² '+fmtR(LINE_R2)+')과 기준선(늘 훈련 평균, R² '+fmtR(BASE_R2)+')을 모두 넘으면 합격. 근무 조건은 04장에서 고른 k = 5, 표준화. 거리 계산은 브라우저가 직접 한다.',
  fly:'처음 보는 258건에서도 되나? 표준화를 빼면 어떻게 되는데?',robo:'외울 숫자 몇 개? 예측 한 번에 계산은 몇 번?',
  async run(setP){setP(.25);await wait(40);const p=testPred('std');setP(.6);await wait(40);const pr=testPred('raw');setP(1);
    const R=r2(YTE,p),M=mae(YTE,p),Rr=r2(YTE,pr),pass=R>LINE_R2&&R>BASE_R2;
    return {result:pass?'pass':'hold',flyAns:'시험 R² '+fmtR(R)+' · MAE '+fmtM(M)+'MPa · 표준화를 빼면 R² '+fmtR(Rr),roboAns:'외울 숫자 6,176개 · 예측 1건에 거리 772번 · 258건 '+CACHE.std_ms.toFixed(0)+'ms',flyMood:pass?'arms':'coffee',flyMove:pass?'nod':'shake',roboMood:pass?'sigh':'meh',roboMove:'sway',asstMood:pass?'spark':'sigh',docMood:pass?'judge':'scale',
      measure:'봉인된 시험지 결과예요. KNN <b>R² '+fmtR(R)+'</b>(MAE '+fmtM(M)+'MPa), 매뉴얼 경험식 R² '+fmtR(LINE_R2)+'(MAE '+fmtM(LINE_MAE)+'), 기준선 R² '+fmtR(BASE_R2)+'(MAE '+fmtM(BASE_MAE)+'). 합격선과의 차이는 '+fx(R-LINE_R2,3)+'로 아슬아슬해요. 같은 지원자가 표준화를 빼면 R² '+fmtR(Rr)+', 합격선 아래예요.',
      verdict:pass?'합격입니다. 입문팀에 배치합니다. 조건이 둘 있어요. 표준화를 절대 빼지 않을 것, 데이터가 크게 늘면 예측 비용을 다시 심사할 것.':'보류입니다. 회귀 합격선을 넘지 못했어요.',
      key:'비슷한 배합 다섯 — 회귀 합격선 R² '+fmtR(LINE_R2)+' → '+fmtR(R),
      render(box){box.innerHTML='<div class="bars">'+barRows([['KNN (k = 5, 표준화)',R,'var(--blue)','R² '+fmtR(R)],['매뉴얼 경험식',LINE_R2,'var(--c-man)','R² '+fmtR(LINE_R2)],['KNN (표준화 안 함)',Rr,'var(--amber)','R² '+fmtR(Rr)],['기준선 (늘 평균)',BASE_R2,'var(--muted)','R² '+fmtR(BASE_R2)]],1)+'</div>';}};},
  alts:[
   {label:'표준화 안 함 (kg과 일을 그대로)',async run(box){const p=testPred('raw'),R=r2(YTE,p);
     box.innerHTML=statGrid([['R² '+fmtR(R),'시험 R² (MAE '+fmtM(mae(YTE,p))+')','a'],['R² '+fmtR(LINE_R2),'회귀 합격선','m']],2)+'<p class="foot">같은 지원자, 같은 k인데 거리 눈금만 바꿨더니 합격선 아래로 내려간다. 판정: '+(R>LINE_R2?/*i18n:verdict*/'합격':'보류')+'. 물(1%)과 고성능 감수제(0.1%)가 거리에서 거의 무시되고, 시멘트·슬래그·골재의 kg 차이가 이웃을 고른다(02장).</p>';}},
   {label:'거리 가중 (weights="distance")',async run(box){const p=testPred('dist'),R=r2(YTE,p);
     box.innerHTML=statGrid([['R² '+fmtR(R),'시험 R² (MAE '+fmtM(mae(YTE,p))+')','a'],['R² '+fmtR(LINE_R2),'회귀 합격선','m']],2)+'<p class="foot">가까운 이웃의 말을 크게 들었더니 여유 있게 넘는다. 판정: '+(R>LINE_R2?/*i18n:verdict*/'합격':'보류')+'. 같은 배합을 여러 재령으로 시험한 줄이 많은 이 데이터에서는 "가장 비슷한 배합"의 답이 특히 믿을 만하기 때문이다. 이 조건도 04장처럼 훈련 데이터 안에서 고른다(A장 7번 칸의 방식으로).</p>';}},
   {label:'같은 이웃에게 분류를 맡기면 (35MPa 다수결)',async run(box){const v=B.map(x=>KNN.vote(A,PTR,x,5)),a=ML.acc(PTE,v),M=ML.confusion(PTE,v,2);
     box.innerHTML=statGrid([[pct(a),'시험 정확도','a'],[pct(PASSLINE),'분류 합격선 (5-0)','m'],[String(M[0][1]),'미달인데 "합격" (건)','s']],3)+'<p class="foot">이웃은 똑같고 마지막에 평균 대신 표를 셌다. 분류 업무도 합격선을 넘는다. 판정: '+(a>PASSLINE?/*i18n:verdict*/'합격':'보류')+'. 한 지원자가 두 업무를 다 해내는 것이 입문팀 KNN의 장점이다.</p>';}}]});

/* 평가 장은 화면에 가까워지면 계산한다 */
(function(){const t=$('m5');let done=false;function go(){if(done)return;done=true;setTimeout(renderEval,10);}
  if('IntersectionObserver' in window){const io=new IntersectionObserver(es=>{if(es.some(e=>e.isIntersecting)){go();io.disconnect();}},{rootMargin:'600px'});io.observe(t);}else go();
  window.addEventListener('hashchange',go);if(location.hash)go();})();

/* ── 06 인사 카드: 차원의 저주 · 외울 숫자 ── */
(function(){const DIMS=[1,2,4,8,16,32,64,128],N=500,Q=12,rg=RNG(7),RC={},HIST={};
  DIMS.forEach(d=>{const P=[];for(let i=0;i<N;i++){const p=new Float64Array(d);for(let j=0;j<d;j++)p[j]=rg();P.push(p);}let s=0;
    for(let q=0;q<Q;q++){const x=new Float64Array(d);for(let j=0;j<d;j++)x[j]=rg();let mn=1e9,mx=0;const ds=[];P.forEach(p=>{const v=Math.sqrt(KNN.d2(p,x));ds.push(v);if(v<mn)mn=v;if(v>mx)mx=v;});s+=(mx-mn)/mn;if(q===0)HIST[d]=ds;}RC[d]=s/Q;});
  const cv=$('cursCv'),hc=$('cursHist');
  function draw(){const W=fit(cv,250),ctx=cv.getContext('2d'),C=cols(),pl=46,pr=12,pt=12,pb=36,gw=W-pl-pr,gh=250-pt-pb,lx=d=>pl+Math.log2(d)/7*gw,ymax=Math.log10(RC[1])+0.2,ymin=-1.2,ly=v=>pt+(1-(Math.log10(v)-ymin)/(ymax-ymin))*gh;
    ctx.clearRect(0,0,W,250);ctx.strokeStyle=C.rule;ctx.strokeRect(pl,pt,gw,gh);ctx.fillStyle=C.muted;ctx.font=font(12);
    DIMS.forEach(d=>{ctx.textAlign='center';ctx.fillText(d,lx(d),250-20);});ctx.fillText('차원 수 (특징 개수, 로그 눈금)',pl+gw/2,250-4);
    [0.1,1,10,100,1000].forEach(v=>{if(Math.log10(v)<ymin||Math.log10(v)>ymax)return;ctx.textAlign='right';ctx.fillText(v,pl-5,ly(v)+4);});
    ctx.save();ctx.translate(11,pt+gh/2);ctx.rotate(-Math.PI/2);ctx.textAlign='center';ctx.fillText('(최장 − 최단) ÷ 최단',0,0);ctx.restore();
    ctx.strokeStyle=C.blue;ctx.lineWidth=2.5;ctx.beginPath();DIMS.forEach((d,i)=>{const x=lx(d),y=ly(RC[d]);i?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.stroke();DIMS.forEach(d=>circ(ctx,lx(d),ly(RC[d]),4,d===8?C.stamp:C.blue,C.panel,1.5));
    ctx.fillStyle=C.stamp;ctx.font=font(12,700);ctx.textAlign='left';ctx.fillText('콘크리트 (8)',lx(8)+8,ly(RC[8])-8);}
  function hist(){const d=DIMS[+$('cursD').value];$('cursDO').textContent=d;const W=fit(hc,190),ctx=hc.getContext('2d'),C=cols(),ds=HIST[d],m=ML.mean(ds),nb=30,bins=new Array(nb).fill(0);
    ds.forEach(v=>{const b=Math.min(nb-1,Math.floor(v/m/2*nb));bins[b]++;});const mx=Math.max(...bins),pl=10,pr=10,pt=8,gh=150,gw=W-pl-pr;
    ctx.clearRect(0,0,W,190);bins.forEach((c,i)=>{const h=c/mx*gh;ctx.fillStyle=rgba(C.blue,.7);ctx.fillRect(pl+i*gw/nb+1,pt+gh-h,gw/nb-2,h);});
    ctx.strokeStyle=C.rule;ctx.beginPath();ctx.moveTo(pl,pt+gh);ctx.lineTo(pl+gw,pt+gh);ctx.stroke();ctx.fillStyle=C.muted;ctx.font=font(12);ctx.textAlign='center';['0','평균 거리','평균의 2배'].forEach((t,i)=>ctx.fillText(t,pl+i*gw/2+(i===0?6:i===2?-24:0),pt+gh+16));
    const mn=Math.min(...ds),mxd=Math.max(...ds);
    $('cursInfo').innerHTML='무작위 점 500개, 질문 점 하나. <b>'+d+'차원</b>에서 가장 가까운 점은 평균 거리의 '+pct(mn/m,0)+', 가장 먼 점은 '+pct(mxd/m,0)+' 거리에 있다. 왼쪽 그림의 값(최장 − 최단) ÷ 최단은 '+RC[d].toFixed(2)+'. '+
      (d<=8?'아직 "가까운 이웃"과 "먼 남"이 뚜렷이 구별된다. 콘크리트는 8차원이라 KNN이 일할 수 있다.':'차원이 커지면 모든 점이 비슷한 거리에 몰려 "가장 가까운 이웃"이 별 뜻이 없어진다. 쓸모없는 특징을 더하는 것은 차원을 늘리는 일이다.')+' (실제 데이터는 특징끼리 얽혀 있어 이 무작위 예보다 덜 나쁠 수 있다.)';}
  $('cursD').addEventListener('input',hist);draw();hist();onRedraw(()=>{draw();hist();});
  $('memBars').innerHTML=barRows([['KNN · 외울 숫자',6176,'var(--blue)','6,176'],['로지스틱 회귀 (5-4)',9,'var(--muted)','9'],['매뉴얼 경험식 · 상수',4,'var(--c-man)','4'],['KNN · 예측 1건 연산',772*8,'var(--amber)','6,176'],['로지스틱 회귀 · 예측 1건',8,'var(--muted)','8']],6176);})();

/* ══ 확인 문제 ══ */
var QUIZ={
q1:[{q:'KNN이 새 배합의 강도를 예측하는 방법은?',o:[['가장 가까운 훈련 배합 k개의 강도를 평균 낸다',1,'회귀(KNeighborsRegressor)는 이웃 값의 평균, 분류(KNeighborsClassifier)는 이웃 이름표의 다수결이다.'],['훈련 데이터로 가중치를 학습해 식을 만든다',0,'그것은 선형 회귀·로지스틱 회귀의 방식이다. KNN은 식을 만들지 않는다.'],['강도의 전체 평균을 답한다',0,'그것은 기준선이다. k를 훈련 데이터 전체 수로 두면 KNN도 이렇게 된다.']]},
 {q:'KNN의 fit()이 실제로 하는 일은?',o:[['훈련 데이터를 저장한다(외운다)',1,'계산은 질문이 올 때 한다. 그래서 게으른 학습이라 부른다. scikit-learn은 빨리 찾기 위한 색인(kd-tree 등)을 만들기도 한다.'],['가중치를 경사 하강으로 고친다',0,'KNN에는 고칠 가중치가 없다.'],['가장 좋은 k를 찾는다',0,'k는 사람이 정하거나 교차 검증으로 고른다. fit은 k를 고르지 않는다.']]},
 {q:'35MPa 합격 여부(두 클래스)를 다수결로 정할 때 k를 홀수로 두는 까닭은?',o:[['표가 동점이 되지 않게',1,'k = 4면 2 : 2가 나올 수 있다. 홀수면 두 클래스 사이에 동점이 없다.'],['계산이 빨라서',0,'홀짝과 속도는 상관없다.'],['scikit-learn이 짝수를 허용하지 않아서',0,'짝수도 된다. 동점이면 정해진 규칙(작은 이름표 등)으로 고른다.']]}],
q2:[{q:'표준화하지 않으면 콘크리트 데이터의 거리에서 가장 무시되는 특징은?',o:[['고성능 감수제와 물',1,'표준편차가 6kg, 21kg으로 작아 거리에서 차지하는 몫이 0.1%, 1%뿐이다. 강도에는 큰 영향을 주는데도 그렇다.'],['시멘트',0,'표준편차가 104kg으로 가장 커서 오히려 거리를 독차지한다.'],['굵은 골재',0,'표준편차가 78kg이라 거리에서 16% 가까이 차지한다.']]},
 {q:'표준화의 평균과 표준편차는 어디서 구해야 하나?',o:[['훈련 데이터에서만',1,'시험지는 처음 보는 데이터 역할이다. 시험지는 훈련 데이터로 구한 값으로 transform만 한다.'],['훈련과 시험을 따로따로',0,'03장 틀린 곳 찾기의 실수다. 눈금이 서로 달라져 거리가 엉뚱해진다.'],['훈련 + 시험 전체에서',0,'시험지 정보가 훈련에 스며드는 데이터 누수다.']]},
 {q:'매뉴얼 씨의 경험식에서 물-시멘트비 대신 물-결합재비를 쓴 효과는?',o:[['슬래그·플라이 애시를 섞은 배합까지 맞혀 R²가 0.058에서 0.735로 올랐다',1,'혼화재도 강도를 내는 결합재라는 사람의 지식이다. 이 지식이 회귀 합격선을 세웠다.'],['계산이 간단해졌다',0,'계산 양은 거의 같다. 차이는 지식이다.'],['아무 차이가 없다',0,'시험지 R²가 열 배 넘게 달라진다.']]}],
q3:[{q:'KNeighborsRegressor(5, "distance")를 실행하면?',o:[['TypeError가 난다. weights="distance"처럼 이름을 붙여야 한다',1,'n_neighbors 말고는 키워드 전용 인자다. 옛 코드를 보고 배운 AI가 자주 내는 실수다.'],['k = 5, 거리 가중으로 잘 돈다',0,'옛 버전에서는 됐지만 지금은 오류다.'],['"distance"가 metric으로 들어간다',0,'위치 인자로 넘길 수 없어 오류가 난다.']]},
 {q:'시험지 R²가 가장 높은 k를 골라 그 점수를 보고했다. 문제는?',o:[['시험지에 맞춘 점수라 처음 보는 데이터에서는 더 낮게 나온다',1,'k는 훈련 데이터 안의 교차 검증으로 고르고, 시험지는 마지막에 한 번만 연다.'],['문제없다. 가장 정확한 방법이다',0,'시험지가 사실상 훈련 데이터가 된다.'],['k가 너무 커진다',0,'k의 크기와는 상관없는 문제다.']]}],
q4:[{q:'KNN의 훈련 R²가 k = 1에서 거의 1인 까닭은?',o:[['훈련 배합의 가장 가까운 이웃이 자기 자신(거리 0)이라서',1,'외운 답을 읽는 셈이다. 그래서 자기 자신을 빼는 LOO로 잰다.'],['k = 1이 가장 좋은 모델이라서',0,'LOO와 시험 점수에서는 k = 1이 k = 4~5보다 낮다.'],['계산 오류다',0,'오류가 아니라 KNN의 성질이다.']]},
 {q:'k를 아주 크게(예: 100) 하면 예측 지도는?',o:[['먼 배합까지 평균에 끼어 흐려진다(과소적합)',1,'k가 훈련 데이터 수와 같아지면 늘 전체 평균, 즉 기준선이 된다.'],['더 얼룩덜룩해진다',0,'얼룩은 k가 작을 때 생긴다.'],['변하지 않는다',0,'k는 KNN의 가장 중요한 근무 조건이다.']]},
 {q:'KNN에서 LOO(하나 빼고 맞히기)가 특히 싼 까닭은?',o:[['다시 학습할 것이 없어 거리표에서 자기 자신만 빼면 되므로',1,'다른 모델은 772번 다시 학습해야 하지만 KNN은 이웃 목록에서 한 줄을 지우면 끝이다.'],['데이터가 작아서',0,'데이터가 작은 것도 돕지만, 핵심은 학습이 없다는 점이다.'],['시험지를 쓰기 때문에',0,'LOO는 훈련 데이터만 쓴다.']]}],
q5:[{q:'이 면접에서 표준화를 빼면 회귀 판정은?',o:[['R²가 합격선 아래로 내려가 보류가 된다',1,'0.767 → 0.716. 같은 지원자도 근무 조건 하나로 판정이 바뀐다.'],['더 좋아진다',0,'단위가 큰 특징이 이웃을 고르게 되어 나빠진다.'],['변하지 않는다',0,'KNN은 거리로 일하므로 눈금이 바뀌면 답이 바뀐다.']]},
 {q:'KNN이 실제 80MPa인 시험 배합을 예측하기 어려운 까닭은?',o:[['이웃의 평균이라 훈련 데이터의 범위 끝에서 안쪽으로 끌려가고, 범위 밖은 낼 수 없어서',1,'가장 센 이웃들의 평균도 가장 센 값보다 작다. 외삽을 못 하는 것이 KNN의 약점이다.'],['k가 홀수라서',0,'홀짝과는 상관없다.'],['표준화를 해서',0,'표준화는 특징의 눈금일 뿐 타겟 범위와는 상관없다.']]},
 {q:'R²가 0이면?',o:[['늘 평균만 답하는 기준선과 같은 실력',1,'R²는 "평균만 답하기"보다 오차 제곱을 얼마나 줄였나다. 음수면 평균보다 못하다.'],['예측이 완벽하다',0,'완벽하면 1이다.'],['오차가 0MPa',0,'그것은 MAE = 0이다.']]}],
q6:[{q:'특징을 수백 개로 늘리면 KNN에게 생기는 일로 맞는 것은?',o:[['모든 점의 거리가 비슷해져 "가장 가까운 이웃"의 뜻이 흐려진다',1,'차원의 저주다. 쓸모없는 특징을 빼거나 차원을 줄여(5-14 PCA) 쓴다.'],['늘 정확도가 오른다',0,'쓸모없는 특징은 거리를 흐린다. 이 면접에서도 특징 2개 KNN이 8개보다 나았다.'],['예측이 빨라진다',0,'거리 계산이 특징 수만큼 늘어 느려진다.']]},
 {q:'KNN이 가장 잘 맞는 업무는?',o:[['특징이 적고 데이터가 수만 건 이하이며, 비슷한 과거 사례를 근거로 보여 줘야 하는 판정',1,'이웃 목록이 곧 설명이고, 새 사례는 장부에 더하기만 하면 된다.'],['초당 수만 건을 빠르게 판정하는 실시간 업무',0,'질문마다 전체와 거리를 재야 해 비싸다. 근사 검색 같은 도구가 필요하다.'],['지금까지 없던 더 센 배합의 강도를 예측하는 일',0,'KNN은 훈련 범위 밖을 예측하지 못한다.']]}]};

/* ══ 공통 B·C·D·E·F 데이터 ══ */
var PROMPTS=[
 {t:'① 내 공정 데이터로 KNN 회귀 코드 받기',use:'내 공장의 품질 데이터로 03장의 프롬프트를 맞출 때',p:MAIN_PROMPT},
 {t:'② AI 코드 검증 요청',use:'받은 코드를 다른 AI(또는 같은 AI의 새 대화)에게 검토받을 때',p:'아래 scikit-learn KNN 코드를 코드 리뷰해 줘. 특히 다음을 하나씩 확인하고, 문제가 있으면 줄 번호와 고친 코드를 보여 줘.\n1) 타겟이 연속값인데 KNeighborsClassifier를 쓰지 않았나 (또는 그 반대)\n2) 데이터를 나눈 뒤 StandardScaler를 훈련 데이터로만 fit 했나, 시험 데이터에 fit_transform을 쓰지 않았나\n3) train_test_split에 stratify와 random_state가 있나\n4) k(n_neighbors)를 시험 데이터 점수로 고르지 않았나. 교차 검증을 썼나\n5) weights, metric 같은 인자를 이름 없이 위치로 넘기지 않았나 (내 버전: 〔1.x〕)\n6) 결과를 기준선(DummyRegressor)·기존 방식과 비교했나\n추측으로 답하지 말고, 확실하지 않은 부분은 "공식 문서 확인 필요"라고 표시해 줘.'},
 {t:'③ 비슷한 과거 사례 찾기 도구로 바꾸기',use:'예측값보다 "가장 비슷한 과거 배합 목록"이 필요할 때',p:'콘크리트 배합 데이터(특징: 〔시멘트, 고로 슬래그, 플라이 애시, 물, 고성능 감수제, 굵은 골재, 잔골재, 재령〕, 결과: 〔압축 강도〕)가 있다. 새 배합 하나를 넣으면 가장 비슷한 과거 배합 〔5〕건을 표로 보여 주는 Colab 코드를 써 줘.\n- StandardScaler를 과거 데이터로 fit 하고 NearestNeighbors로 찾는다\n- 표에는 각 이웃의 특징, 강도, 거리, 그리고 새 배합과 가장 많이 다른 특징 두 개를 적는다\n- 거리 계산에 쓸 특징을 고르는 옵션을 넣는다 (예: 재령이 같은 배합만 비교)\n- 표준화를 하지 않았을 때 이웃이 어떻게 바뀌는지도 나란히 보여 준다'},
 {t:'④ 수업용 문제 만들기',use:'이 편으로 수업할 때 확인 문제가 더 필요할 때',p:'KNN 입문 수업용 문제를 만들어 줘. 대상은 〔제조 현장 실무자, 수식에 익숙하지 않음〕이다. 예시는 콘크리트 강도로 통일한다.\n- 거리·표준화·k·거리 가중을 각각 묻는 객관식 4문제 (보기 3개, 정답 해설 포함)\n- 오해를 드러내는 문제 2개: "KNN도 학습할 때 가중치를 배운다", "훈련 점수가 높으면 k가 좋다"\n- 손으로 풀 수 있는 계산 문제 1개: 두 배합의 표준화한 특징 3개로 유클리드 거리 구하기, 이웃 강도 5개로 평균과 거리 가중 평균 구하기 (소수 첫째 자리)\n정답은 맨 끝에 따로 모아 줘.'}];
var GL=[
 ['모델','KNN','k-최근접 이웃. 새 데이터와 가장 가까운 훈련 데이터 k개의 답을 평균(회귀) 또는 다수결(분류)로 모으는 모델.'],
 ['모델','유클리드 거리','특징마다 차이를 제곱해 더하고 제곱근을 씌운 거리. 자로 잰 직선 거리다.'],
 ['모델','맨해튼 거리','특징마다 차이의 절댓값을 더한 거리. 바둑판 길을 따라 걷는 거리다.'],
 ['모델','다수결','이웃 k개의 이름표 중 가장 많은 것을 답하는 방법. 분류에서 쓴다.'],
 ['모델','거리 가중','가까운 이웃일수록 큰 무게(거리의 역수)를 주어 평균 내는 방법.'],
 ['모델','게으른 학습','미리 식을 만들지 않고 데이터를 저장해 두었다가 질문이 올 때 계산하는 방식. 사례 기반 학습.'],
 ['모델','차원의 저주','특징(차원)이 많아질수록 점들이 서로 비슷하게 멀어져 "가까움"이 뜻을 잃는 현상.'],
 ['모델','근사 이웃 검색','가장 가까운 이웃을 정확히 찾는 대신 거의 가장 가까운 것을 아주 빨리 찾는 방법. 큰 데이터, 벡터 검색에서 쓴다.'],
 ['데이터','샘플','표의 한 줄. 여기서는 콘크리트 배합 하나(와 그 시편의 시험 결과).'],
 ['데이터','특징','모델이 입력으로 쓰는 열. 시멘트·고로 슬래그·플라이 애시·물·고성능 감수제·굵은 골재·잔골재·재령.'],
 ['데이터','타겟','맞히려는 정답 열. 이 면접에서는 압축 강도(MPa)와 35MPa 합격 여부.'],
 ['데이터','물-결합재비','물의 양 ÷ 결합재(시멘트 + 고로 슬래그 + 플라이 애시)의 양. 낮을수록 강도가 높아지는 경향이 있다.'],
 ['데이터','재령','콘크리트를 친 뒤 시험할 때까지 지난 날수. 강도는 처음에 빨리, 나중에 천천히 오른다.'],
 ['데이터','표준화','특징마다 평균을 빼고 표준편차로 나눠 눈금을 맞추는 일. 훈련 데이터로만 맞춘다.'],
 ['데이터','데이터 누수','시험 데이터의 정보가 학습에 섞여 들어가는 일. 점수가 실제보다 좋게 나온다.'],
 ['데이터','에이브럼스 식','콘크리트 강도가 물-시멘트비가 커질수록 지수적으로 줄어든다는 경험식(1918). 매뉴얼 씨의 경험식의 뿌리.'],
 ['학습','교차 검증','훈련 데이터를 나눠 일부로 맞혀 보며 근무 조건을 고르는 방법. 시험지를 열지 않는다.'],
 ['학습','LOO','leave-one-out. 훈련 데이터를 하나씩 빼고 나머지로 맞혀 보는 교차 검증. KNN에서는 특히 싸다.'],
 ['학습','과대적합','훈련 데이터를 외워 처음 보는 데이터에서 성적이 떨어지는 일. KNN에서는 k가 너무 작을 때.'],
 ['학습','과소적합','모델이 너무 단순해 훈련 데이터조차 못 맞히는 일. KNN에서는 k가 너무 클 때.'],
 ['평가','R²','결정 계수. 늘 평균만 답할 때보다 오차 제곱을 얼마나 줄였나. 1이면 완벽, 0이면 평균과 같다.'],
 ['평가','MAE','평균 절대 오차. 예측이 실제와 평균 몇 MPa 다른지.'],
 ['평가','기준선','배우지 않고도 낼 수 있는 점수. 회귀에서는 "늘 훈련 평균을 답하기".'],
 ['평가','합격선','지원자가 넘어야 할 점수. 회귀는 매뉴얼 경험식의 R², 분류는 5-0 매뉴얼 씨의 정확도.']];
var ANALOGY=[
 ['이웃에게 묻기','처음 보는 배합이 오면 "예전에 이와 비슷했던 배합은 어땠지?" 하고 작업 일지를 뒤지는 고참',2,'초파리의 냄새 회로가 비슷한 냄새에 겹치는 뉴런 무리를 할당해, 비슷한 것을 빨리 찾는 근사 이웃 검색처럼 일한다는 제안(Dasgupta 등, 2017)','초파리 회로는 무작위 투영과 승자 독식으로 "대충 비슷한 것"을 찾는다. KNN처럼 모든 기억과 정확한 거리를 재지 않는다. 이 제안은 계산 모형이며 행동으로 다 확인된 것은 아니다'],
 ['거리','두 배합의 차이를 항목별로 재서 "다른 정도" 점수 하나로 합친 것',1,'','뇌가 기억 사이의 유클리드 거리를 계산한다는 근거는 없다'],
 ['표준화','킬로그램, 일, 퍼센트를 한 저울에 올리기 전에 눈금부터 맞추는 일',2,'감각 뉴런이 주변 자극의 세기에 맞춰 반응 범위를 옮기는 적응·정규화','뇌의 적응은 시간에 따라 계속 바뀐다. 표준화는 훈련 데이터로 한 번 정한 고정 눈금이다'],
 ['k와 다수결','혼자 판단하지 않고 비슷한 경험이 있는 선배 다섯에게 물어 다수 의견을 따른다',1,'','뉴런 집단의 활동을 "투표"로 보는 설명은 있지만, 기억 k개를 골라 세는 회로는 알려지지 않았다'],
 ['게으른 학습','규칙집을 만들지 않고 사례철을 통째로 들고 다니는 직원',2,'해마가 개별 사건(에피소드)을 저장하고, 단서가 오면 비슷한 기억을 꺼내는 일','해마는 기억을 압축·재구성하고, 잠자는 동안 대뇌 피질로 옮겨 정리한다. KNN은 아무것도 정리하지 않는다'],
 ['거리 가중','가장 비슷한 사례의 말을 가장 크게 듣는다',1,'','직접 대응하는 뇌 기능은 알려지지 않았다']];
var REFS=[
 'Yeh, I.-C. (1998). Modeling of strength of high-performance concrete using artificial neural networks. <i>Cement and Concrete Research</i>, 28(12), 1797–1808. <a href="https://doi.org/10.1016/S0008-8846(98)00165-3" target="_blank" rel="noopener">doi:10.1016/S0008-8846(98)00165-3</a> — 실기 과제지 원출처',
 'Yeh, I.-C. (2007). Concrete Compressive Strength [Dataset]. UCI Machine Learning Repository. <a href="https://doi.org/10.24432/C5PK67" target="_blank" rel="noopener">doi:10.24432/C5PK67</a> (CC BY 4.0) — 02장 데이터',
 'Abrams, D. A. (1918). <i>Design of Concrete Mixtures</i> (Bulletin 1). Structural Materials Research Laboratory, Lewis Institute, Chicago. — 물-시멘트비 법칙, 매뉴얼 씨의 경험식(02장)',
 'ACI Committee 209 (1992). <i>Prediction of Creep, Shrinkage, and Temperature Effects in Concrete Structures</i> (ACI 209R-92). American Concrete Institute. — 재령에 따른 강도 보정식 t/(4 + 0.85t), 02장',
 'Fix, E., &amp; Hodges, J. L. (1951). <i>Discriminatory Analysis. Nonparametric Discrimination: Consistency Properties</i> (Report No. 4). USAF School of Aviation Medicine, Randolph Field, Texas. — 최근접 이웃 규칙의 출발점, 01장',
 'Cover, T. M., &amp; Hart, P. E. (1967). Nearest neighbor pattern classification. <i>IEEE Transactions on Information Theory</i>, 13(1), 21–27. <a href="https://doi.org/10.1109/TIT.1967.1053964" target="_blank" rel="noopener">doi:10.1109/TIT.1967.1053964</a> — 최근접 이웃 분류의 오류 한계, 01장',
 'Beyer, K., Goldstein, J., Ramakrishnan, R., &amp; Shaft, U. (1999). When is "nearest neighbor" meaningful? In <i>Database Theory — ICDT\'99</i> (LNCS 1540, pp. 217–235). Springer. — 차원이 늘면 거리 차이가 사라지는 현상, 06장',
 'Hastie, T., Tibshirani, R., &amp; Friedman, J. (2009). <i>The Elements of Statistical Learning</i> (2nd ed.), 2장·13장. Springer. — 최근접 이웃, k와 편향-분산, 차원의 저주, 04·06장',
 'Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. <i>Journal of Machine Learning Research</i>, 12, 2825–2830. — A장 코드',
 'scikit-learn 문서. <a href="https://scikit-learn.org/stable/modules/neighbors.html" target="_blank" rel="noopener">Nearest Neighbors</a> · <a href="https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html" target="_blank" rel="noopener">KNeighborsRegressor</a> — weights, metric, algorithm, 03·06장',
 'Dasgupta, S., Stevens, C. F., &amp; Navlakha, S. (2017). A neural algorithm for a fundamental computing problem. <i>Science</i>, 358(6364), 793–796. <a href="https://doi.org/10.1126/science.aam9868" target="_blank" rel="noopener">doi:10.1126/science.aam9868</a> — D장 초파리 냄새 회로와 비슷한 것 찾기'];
var TIP={who:'머신러닝을 처음 배우는 대학생·직장인·제조 현장 실무자. 5-0 채용 공고를 먼저 보면 좋다. 수식은 거리 d와 평균 두 개만 나온다',
 goal:'① 거리 → k개 고르기 → 평균/다수결의 흐름을 지원서 1장으로 설명한다 ② 표준화가 이웃을 어떻게 바꾸는지 예로 보인다 ③ k를 시험지 없이 교차 검증으로 고른다 ④ AI가 낸 코드에서 분류기 오용·누수·시험지로 k 고르기를 찾고, R²·MAE로 합격선과 비교해 판정한다',
 time:'읽기 25분 · 실습(01·02·04장 그림) 20분 · 틀린 곳 찾기 10분 · Colab 15분',
 plans:[['강의형 50분','00장 지원서 1장을 인쇄해 나눠 주고 9칸을 함께 읽기 10분 → 01장 그림에 새 배합을 놓고 k를 바꿔 보기 10분 → 02장 #400 배합으로 표준화 전후 이웃 비교 10분 → 04장 예측 지도와 LOO 곡선 10분 → 면접 결과를 학생들이 먼저 맞히고 발표, "표준화 안 함" 다시 보기 10분.'],['실무 OJT 90분','03장 프롬프트로 각자 AI에게 코드를 받는다 20분 → 체크리스트로 서로의 코드를 바꿔 검사 20분 → Colab에서 실행하고 A장 결과와 비교 20분 → B장 ③ 프롬프트로 "우리 공장 유사 배합 찾기" 도구를 만들고 이웃 목록이 현장 감각과 맞는지 토론 30분.'],['1:1 튜터링','02장에서 학습자가 아는 배합 하나를 고르게 하고 이웃 다섯을 함께 읽는다. "왜 이 배합이 이웃인가"를 막대(거리 몫)로 설명하게 한 뒤, 표준화를 껐다 켜며 답이 어떻게 바뀌는지 말로 정리하게 한다.']],
 myths:[['KNN도 학습할 때 무언가를 배운다','fit은 저장만 한다. 고르는 것(k, 거리, 가중)은 사람이나 교차 검증이 한다'],['훈련 점수가 높은 k가 좋은 k다','k = 1의 훈련 점수는 자기 자신을 읽은 것이다. LOO나 교차 검증으로 잰다'],['표준화는 해도 그만 안 해도 그만이다','KNN에서는 판정을 바꾼다(R² 0.767 ↔ 0.716). 물·감수제가 거리에서 사라진다'],['특징은 많을수록 좋다','쓸모없는 특징은 거리를 흐린다. 특징 2개(물-결합재비, log 재령) KNN이 8개보다 나았다'],['KNN은 분류 모델이다','이웃 평균으로 회귀도 한다. 같은 이웃, 다른 질문이다'],['학습이 공짜니 KNN은 싸다','예측마다 데이터 전체와 거리를 잰다. 비용이 예측 쪽으로 옮겨 갔을 뿐이다']],
 tasks:[['과제 1 · 지원서 다시 쓰기','00장 지원서 6번 칸의 예시(46.2, 46.2, 30.4, 23.7, 40.9)를 01장 그림에서 고른 배합의 실제 이웃 강도로 바꿔 쓰고, 8번 칸의 거리 가중 평균도 손으로 계산한다.'],['과제 2 · 틀린 곳 찾기 만들기','AI에게 일부러 실수가 섞인 KNN 코드를 받고(B장 ②를 거꾸로 활용), 짝에게 찾게 한다. 실수마다 "왜 틀렸나"를 한 줄로 쓴다.'],['과제 3 · 거리 바꾸기','A장 코드에 metric="manhattan"(p=1)과 MinMaxScaler·RobustScaler를 더해 시험 R²가 어떻게 바뀌는지 표로 정리한다. k는 훈련 데이터 안에서 고른다.'],['과제 4 · 특징 고르기','A장 8번 칸처럼 특징 조합(예: 물-결합재비·log 재령·고성능 감수제)을 바꿔 LOO R²로 비교하고, 차원의 저주와 연결해 결과를 설명한다.']],
 ai:'학생이 AI에게 코드를 받는 것은 막지 말고 과정에 넣는다. 다만 ① 프롬프트를 제출하게 하고 ② 체크리스트 아홉 항목을 직접 표시하게 하고 ③ 실행 결과를 기준선·합격선과 비교한 문장을 쓰게 하면, "받아 적기"가 아니라 "검증하기"를 배운다. AI 답의 함수·인자는 공식 문서와 버전으로 확인하게 하고, 과제에 AI 도움을 받았으면 사용 범위를 밝히게 한다.'};
var CODE={title:'KNN 면접 — 이웃에게 묻기부터 합격 판정까지',sub:'numpy, pandas, scikit-learn, ucimlrepo · CPU 몇 초',file:'knn_interview',
 desc:'이 편의 면접을 그대로 옮겼다. UCI에서 콘크리트 데이터를 받아 시험지를 봉인하고(1·2번 칸), 매뉴얼 씨의 경험식으로 회귀 합격선을 세우고(3번), KNN을 직접 짜서 scikit-learn과 맞춰 보고(4번), 표준화 전후의 이웃(5번), 근무 조건별 성적(6번), 훈련 데이터 안에서 k 고르기(7번), 좋은 특징 두 개(8번), 같은 이웃의 분류(9번), 비용(10번), 판정(11번)까지 간다. 노트북으로 저장하면 칸마다 셀이 나뉜다.',
 cap:'실행 결과 (scikit-learn 1.9, CPU). 시간(ms)은 컴퓨터마다 다르다. 브라우저 계산과 같은 결론(표준화 R² 0.767 합격, 표준화 안 함 0.716 보류, 거리 가중 0.814, 분류 85.7%)이 나온다. 7번 칸의 훈련·LOO 값은 거리가 같은 이웃을 고르는 순서 때문에 04장 그림과 소수점 셋째 자리에서 조금 다를 수 있다.'};
