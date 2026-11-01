var BOARD_LINE='R² 0.735 · 76.7%'; /* 회귀 합격선(매뉴얼 경험식 R²) · 분류 합격선 */
/* ══ 5-8 SVM (SVC · SVR) · 콘크리트 실기 과제지 ══════════════════════════ */
/*SVM-ENGINE*/
/* ── SVM 엔진 (이 편 전용, 브라우저·node 공용) ──
   libsvm(= scikit-learn SVC·SVR의 속)과 같은 쌍대 문제를 같은 방법(SMO, 2차 정보로 두 α 고르기, 허용 오차 0.001)으로 푼다.
   다른 점: 줄이기(shrinking)와 캐시를 쓰지 않고 커널 행렬을 미리 다 만든다(훈련 772건이라 가능). */
const SVM=(function(){
  const TAU=1e-12;
  function gram(X,gamma,kind){const n=X.length,d=X[0].length,K=new Float64Array(n*n);
    for(let i=0;i<n;i++)for(let j=i;j<n;j++){let v;if(kind==='linear'){v=0;for(let k=0;k<d;k++)v+=X[i][k]*X[j][k];}else{let s=0;for(let k=0;k<d;k++){const t=X[i][k]-X[j][k];s+=t*t;}v=Math.exp(-gamma*s);}K[i*n+j]=v;K[j*n+i]=v;}return K;}
  function kern(a,b,gamma,kind){if(kind==='linear'){let v=0;for(let k=0;k<a.length;k++)v+=a[k]*b[k];return v;}let s=0;for(let k=0;k<a.length;k++){const t=a[k]-b[k];s+=t*t;}return Math.exp(-gamma*s);}
  /* 쌍대 문제: ½αᵀQα + pᵀα 최소화, 0 ≤ α ≤ C, Σ yα = 0. Q(t,u) = sg[t]·sg[u]·K(idx[t], idx[u]) */
  function solver(K,n,idx,sg,y,p,C,eps){const L=idx.length,a=new Float64Array(L),G=Float64Array.from(p),QD=new Float64Array(L);
    for(let t=0;t<L;t++)QD[t]=K[idx[t]*n+idx[t]];
    const up=t=>a[t]>=C,lo=t=>a[t]<=0;let it=0,done=false,gap=Infinity;
    function q(t,u){return sg[t]*sg[u]*K[idx[t]*n+idx[u]];}
    function select(){let Gmax=-Infinity,i=-1;for(let t=0;t<L;t++){if(y[t]===1){if(!up(t)&&-G[t]>=Gmax){Gmax=-G[t];i=t;}}else{if(!lo(t)&&G[t]>=Gmax){Gmax=G[t];i=t;}}}
      let Gmax2=-Infinity,j=-1,omin=Infinity;if(i<0){gap=0;return null;}const ri=idx[i]*n,si=sg[i];
      for(let t=0;t<L;t++){const Qit=si*sg[t]*K[ri+idx[t]];if(y[t]===1){if(!lo(t)){const gd=Gmax+G[t];if(G[t]>=Gmax2)Gmax2=G[t];if(gd>0){const qd=QD[i]+QD[t]-2*y[i]*Qit,od=-(gd*gd)/(qd>0?qd:TAU);if(od<=omin){j=t;omin=od;}}}}
        else{if(!up(t)){const gd=Gmax-G[t];if(-G[t]>=Gmax2)Gmax2=-G[t];if(gd>0){const qd=QD[i]+QD[t]+2*y[i]*Qit,od=-(gd*gd)/(qd>0?qd:TAU);if(od<=omin){j=t;omin=od;}}}}}
      gap=Gmax+Gmax2;if(gap<eps||j<0)return null;return [i,j];}
    function update(i,j){const Qij=q(i,j),oi=a[i],oj=a[j];let ai=oi,aj=oj;
      if(y[i]!==y[j]){let qd=QD[i]+QD[j]+2*Qij;if(qd<=0)qd=TAU;const dl=(-G[i]-G[j])/qd,df=ai-aj;ai+=dl;aj+=dl;
        if(df>0){if(aj<0){aj=0;ai=df;}}else{if(ai<0){ai=0;aj=-df;}}
        if(df>0){if(ai>C){ai=C;aj=C-df;}}else{if(aj>C){aj=C;ai=C+df;}}}
      else{let qd=QD[i]+QD[j]-2*Qij;if(qd<=0)qd=TAU;const dl=(G[i]-G[j])/qd,sm=ai+aj;ai-=dl;aj+=dl;
        if(sm>C){if(ai>C){ai=C;aj=sm-C;}}else{if(aj<0){aj=0;ai=sm;}}
        if(sm>C){if(aj>C){aj=C;ai=sm-C;}}else{if(ai<0){ai=0;aj=sm;}}}
      a[i]=ai;a[j]=aj;const di=ai-oi,dj=aj-oj,ri=idx[i]*n,rj=idx[j]*n,si=sg[i],sj=sg[j];
      for(let t=0;t<L;t++){const kt=idx[t];G[t]+=sg[t]*(si*K[ri+kt]*di+sj*K[rj+kt]*dj);}}
    function rho(){let ub=Infinity,lb=-Infinity,sf=0,nf=0;for(let t=0;t<L;t++){const yG=y[t]*G[t];
        if(up(t)){if(y[t]===-1)ub=Math.min(ub,yG);else lb=Math.max(lb,yG);}else if(lo(t)){if(y[t]===1)ub=Math.min(ub,yG);else lb=Math.max(lb,yG);}else{nf++;sf+=yG;}}
      return nf>0?sf/nf:(ub+lb)/2;}
    function obj(){let s=0;for(let t=0;t<L;t++)s+=a[t]*(G[t]+p[t]);return s/2;}
    function step(m){for(let k=0;k<m&&!done;k++){const w=select();if(!w){done=true;break;}update(w[0],w[1]);it++;if(it>=1e7)done=true;}return done;}
    return {a,G,step,rho,obj,get it(){return it},get done(){return done},get gap(){return gap}};}
  /* 모델: coef(샘플별 계수), rho, 서포트 벡터 목록. f(x) = Σ coef·K(xᵢ, x) − rho */
  function model(X,coef,rho,gamma,kind){const sv=[];coef.forEach((c,i)=>{if(c!==0)sv.push(i);});
    function f(x){let s=-rho;for(const i of sv)s+=coef[i]*kern(X[i],x,gamma,kind);return s;}
    let w=null;if(kind==='linear'){w=new Array(X[0].length).fill(0);sv.forEach(i=>X[i].forEach((v,k)=>w[k]+=coef[i]*v));}
    return {f,sv,coef,rho,gamma,kind,w,b:-rho};}
  /* SVC: y는 0/1, 1을 +1 쪽으로. 한 번에 풀거나(fit), 조금씩 풀며 보여 준다(begin → step) */
  function svcBegin(X,y01,o){const n=X.length,K=o.K||gram(X,o.gamma,o.kind),idx=[],sg=[],yy=[],p=[];for(let i=0;i<n;i++){const s=y01[i]?1:-1;idx.push(i);sg.push(s);yy.push(s);p.push(-1);}
    const S=solver(K,n,idx,sg,yy,p,o.C,o.eps||1e-3);
    S.model=()=>model(X,Array.from(S.a,(v,i)=>v*sg[i]),S.rho(),o.gamma,o.kind);return S;}
  function svc(X,y01,o){const S=svcBegin(X,y01,o);while(!S.step(5000));const m=S.model();m.iters=S.it;m.predict=x=>m.f(x)>0?1:0;return m;}
  function svr(X,yv,o){const n=X.length,K=o.K||gram(X,o.gamma,o.kind),idx=[],sg=[],yy=[],p=[],e=o.epsilon==null?0.1:o.epsilon;
    for(let i=0;i<n;i++){idx.push(i);sg.push(1);yy.push(1);p.push(e-yv[i]);}for(let i=0;i<n;i++){idx.push(i);sg.push(-1);yy.push(-1);p.push(e+yv[i]);}
    const S=solver(K,n,idx,sg,yy,p,o.C,o.eps||1e-3);while(!S.step(5000));const coef=[];for(let i=0;i<n;i++)coef.push(S.a[i]-S.a[i+n]);
    const m=model(X,coef,S.rho(),o.gamma,o.kind);m.iters=S.it;m.predict=m.f;return m;}
  /* 다중 분류: 일대일(OvO) 투표. 동점이면 작은 번호 등급 */
  function ovo(X,y,K_,o){const ms=[];for(let a=0;a<K_;a++)for(let b=a+1;b<K_;b++){const id=[];y.forEach((v,i)=>{if(v===a||v===b)id.push(i);});ms.push([a,b,svc(id.map(i=>X[i]),id.map(i=>y[i]===a?1:0),o)]);}
    return {predict(x){const v=new Array(K_).fill(0);ms.forEach(([a,b,m])=>{v[m.f(x)>0?a:b]++;});let k=0;for(let c=1;c<K_;c++)if(v[c]>v[k])k=c;return k;}};}
  /* gamma="scale" = 1 / (특징 수 × 훈련 행렬 전체 분산) */
  function gScale(X){let s=0,s2=0,N=0;X.forEach(r=>r.forEach(v=>{s+=v;s2+=v*v;N++;}));const m=s/N;return 1/(X[0].length*(s2/N-m*m));}
  return {gram,kern,solver,svcBegin,svc,svr,ovo,gScale};})();
/*END-SVM-ENGINE*/

/* ── 데이터 ── */
const ROWS=concRows(),TR=ROWS.filter(r=>!r.test),TE=ROWS.filter(r=>r.test);
const fv=r=>FEAT.map(f=>r[f]);
const Ar=TR.map(fv),Br=TE.map(fv),SC=ML.scaler(Ar),A=SC.t(Ar),B=SC.t(Br);
const YTR=TR.map(r=>r.pass),YTE=TE.map(r=>r.pass),STR=TR.map(r=>r.strength),STE=TE.map(r=>r.strength),GTR=TR.map(r=>r.grade),GTE=TE.map(r=>r.grade);
const PNAME=['불합격','합격'];
function r2(y,p){const m=ML.mean(y);let a=0,b=0;for(let i=0;i<y.length;i++){a+=(y[i]-p[i])*(y[i]-p[i]);b+=(y[i]-m)*(y[i]-m);}return 1-a/b;}
function mae(y,p){let s=0;for(let i=0;i<y.length;i++)s+=Math.abs(y[i]-p[i]);return s/y.length;}
const fmtR=v=>v.toFixed(3),fmtM=v=>v.toFixed(2);
/* 매뉴얼 씨의 경험식(5-1과 같다): 28일 강도 96.5/7^(물-결합재비) × 재령 보정 t/(4+0.85t) */
const manualMPa=(wb,t)=>96.5/Math.pow(7,wb)*t/(4+0.85*t);
const MAN=TE.map(r=>manualMPa(r.wb,r.age)),LINE_R2=r2(STE,MAN),LINE_MAE=mae(STE,MAN);
const BASE=ML.mean(STR),BASE_R2=r2(STE,STE.map(()=>BASE)),BASE_MAE=mae(STE,STE.map(()=>BASE));
function baseAcc(){const p=ML.mean(YTR),maj=p>=.5?1:0;return {k:maj,acc:YTE.filter(v=>v===maj).length/YTE.length};}
/* 특징 8개 · RBF · γ = "scale" (표준화했으니 1/8). 커널 표는 한 번만 만든다 */
const GAM=SVM.gScale(A);let KA=null;const kA=()=>KA||(KA=SVM.gram(A,GAM,'rbf'));
const SVCS={},SVRS={};
function getSVC(C){if(!SVCS[C]){const t0=performance.now(),m=SVM.svc(A,YTR,{C,gamma:GAM,K:kA()});m.ms=performance.now()-t0;m.te=ML.acc(YTE,B.map(m.predict));m.tr=ML.acc(YTR,A.map(m.predict));SVCS[C]=m;}return SVCS[C];}
function getSVR(C){if(!SVRS[C]){const t0=performance.now(),m=SVM.svr(A,STR,{C,gamma:GAM,K:kA()});m.ms=performance.now()-t0;m.p=B.map(m.predict);m.R=r2(STE,m.p);m.M=mae(STE,m.p);m.trR=r2(STR,A.map(m.predict));SVRS[C]=m;}return SVRS[C];}
const C_CLS=10,C_REG=100; /* 05장 교차 검증이 고른 값 (A장 코드와 같다) */
function cmGrid(M,labels,colsFor){const K=labels.length;let h='<div class="cm" style="grid-template-columns:86px repeat('+K+',minmax(48px,1fr));max-width:'+(110+K*90)+'px"><div class="h">정답 ＼ 예측</div>'+labels.map(k=>'<div class="h">'+k+'</div>').join('');
  M.forEach((row,a)=>{h+='<div class="h" style="justify-content:end;text-align:right">'+labels[a]+'</div>'+row.map((v,b)=>a===b?'<div class="c'+(v?' d':'')+'" style="'+(v?'background:'+rgba(colsFor(a),.85):'')+'">'+v+'</div>':'<div class="c'+(v?' x':'')+'">'+(v||'·')+'</div>').join('');});return h+'</div>';}
function barRows(rows,max){return rows.map(r=>'<div class="row"><span style="text-align:left;font-family:var(--font)">'+r[0]+'</span><div class="tr"><i style="left:0;width:'+Math.max(0,r[1]/max*100)+'%;background:'+r[2]+'"></i></div><span>'+r[3]+'</span></div>').join('');}
/* 특징 2개(물-결합재비 · log 재령): 01장·04장 */
const F2=r=>[r.wb,Math.log10(r.age)],R2tr=TR.map(F2),SC2=ML.scaler(R2tr),A2=SC2.t(R2tr);
const BOX2=(function(){const xs=R2tr.map(r=>r[0]);return {x0:Math.min(...xs)-0.02,x1:Math.max(...xs)+0.02,y0:-0.1,y1:Math.log10(365)+0.1};})();
const toS2=(u,v)=>[(u-SC2.mu[0])/SC2.sd[0],(v-SC2.mu[1])/SC2.sd[1]];
function axes2(ctx,C,W,H,pl,pt,gw,gh,sx,sy){ctx.strokeStyle=C.rule;ctx.lineWidth=1;ctx.strokeRect(pl,pt,gw,gh);ctx.fillStyle=C.muted;ctx.font=font(12);
  for(let x=0.3;x<=0.91;x+=0.1){ctx.textAlign='center';ctx.fillText(x.toFixed(1),sx(x),H-20);}
  [1,3,7,28,90,365].forEach(d=>{const v=Math.log10(d);ctx.textAlign='right';ctx.fillText(d,pl-5,sy(v)+4);ctx.strokeStyle=rgba(C.muted,.22);ctx.beginPath();ctx.moveTo(pl,sy(v));ctx.lineTo(pl+gw,sy(v));ctx.stroke();});
  ctx.textAlign='center';ctx.fillText('물-결합재비',pl+gw/2,H-3);ctx.save();ctx.translate(11,pt+gh/2);ctx.rotate(-Math.PI/2);ctx.fillText('재령(일, 로그 눈금)',0,0);ctx.restore();}
/* 등고선(마칭 스퀘어): 격자 값 v에서 level인 곳을 선분으로 */
function contour(ctx,v,nx,ny,X,Y,level){ctx.beginPath();for(let j=0;j<ny;j++)for(let i=0;i<nx;i++){const a=v[j*(nx+1)+i]-level,b=v[j*(nx+1)+i+1]-level,c=v[(j+1)*(nx+1)+i+1]-level,d=v[(j+1)*(nx+1)+i]-level,p=[];
    if(a*b<0)p.push([X(i+a/(a-b)),Y(j)]);if(b*c<0)p.push([X(i+1),Y(j+b/(b-c))]);if(d*c<0)p.push([X(i+d/(d-c)),Y(j+1)]);if(a*d<0)p.push([X(i),Y(j+a/(a-d))]);
    for(let k=0;k+1<p.length;k+=2){ctx.moveTo(p[k][0],p[k][1]);ctx.lineTo(p[k+1][0],p[k+1][1]);}}ctx.stroke();}

/* ── 히어로 ── */
heroRoom($('heroCv'),{sym:'SV',no:'APP-08',name:'SVM\n서포트 벡터',role:'지원자 · 겸용팀',banner:'모델 채용 위원회 · 5-8 면접',doc:'board',asst:'board',fly:'clip',robo:'calc',loop:10,
  beats:[{at:0.4,who:'doc',mood:'magic',move:'pop',say:'공간을 휘어 볼까요?'},{at:2.6,who:'asst',mood:'shock',move:'shake',say:'네? 공간을요?'},{at:5.0,who:'fly',mood:'point',move:'lean',say:'길은 누가 정해?'},{at:7.2,who:'robo',mood:'count',move:'jump',say:'길가만 외운다고?'},{at:8.8,who:'doc',mood:'board',move:'stand'}],
  motif(ctx,t,W,H,C){const x0=270,x1=410,bend=REDUCE?1:0.5+0.5*Math.sin(t*0.9),cy=x=>122+bend*20*Math.sin((x-x0)/(x1-x0)*Math.PI*1.3-0.5)-(x-x0)*0.12;
    ctx.fillStyle=rgba(C.blue,.13);ctx.beginPath();for(let x=x0;x<=x1;x+=4)ctx.lineTo(x,cy(x)-11);for(let x=x1;x>=x0;x-=4)ctx.lineTo(x,cy(x)+11);ctx.fill();
    ctx.strokeStyle=C.c_app;ctx.lineWidth=1.4;ctx.setLineDash([5,4]);[-11,11].forEach(o=>{ctx.beginPath();for(let x=x0;x<=x1;x+=4)x===x0?ctx.moveTo(x,cy(x)+o):ctx.lineTo(x,cy(x)+o);ctx.stroke();});ctx.setLineDash([]);
    ctx.lineWidth=3;ctx.beginPath();for(let x=x0;x<=x1;x+=4)x===x0?ctx.moveTo(x,cy(x)):ctx.lineTo(x,cy(x));ctx.stroke();
    [[282,-30],[300,-44],[322,-26],[350,-38],[372,-24],[392,-40],[338,-11]].forEach(p=>circ(ctx,p[0],cy(p[0])+p[1],4,C.text,null));
    [[286,30],[308,44],[330,26],[356,40],[380,28],[400,42],[366,11]].forEach(p=>circ(ctx,p[0],cy(p[0])+p[1],4,C.blue,null));
    [[338,-11],[366,11]].forEach(p=>circ(ctx,p[0],cy(p[0])+p[1],8,null,C.stamp,2));
    ctx.fillStyle=C.muted;ctx.font=font(12,500,1);ctx.textAlign='left';ctx.fillText('K(x, x′)',300,190);}});
$('heroLine').textContent='R² '+fmtR(LINE_R2)+' · '+pct(PASSLINE);

/* ── 01 장면 ── */
comic('comicIntro',[
 {bg:'night',cast:[{who:'doc',mood:'board',x:30,move:'nod'},{who:'fly',mood:'point',x:76,move:'tilt',flip:true}],say:{who:'doc',txt:'두 무리 사이에 길을 낸다면, 양쪽에서 가장 멀게. 그래야 새 배합이 와도 덜 흔들려요.'},cap:'① 넓은 길(마진): 양쪽 배합에서 가장 멀리 떨어진 경계',prop:'2/‖w‖'},
 {bg:'night',cast:[{who:'asst',mood:'ledger',x:28},{who:'robo',mood:'count',x:74,move:'bob'}],say:{who:'robo',txt:'길가에 선 배합 몇 개만 외우면 된다고? 나머지는 공짜네.'},cap:'② 서포트 벡터: 길을 정하는 것은 길가의 몇 배합뿐',prop:'α > 0'},
 {bg:'night',cast:[{who:'doc',mood:'magic',x:30,move:'pop'},{who:'asst',mood:'shock',x:74,move:'shake'}],say:{who:'asst',txt:'종이를 휘었더니 굽은 경계가 곧은 길이 됐어요!'},cap:'③ 커널 마술: 곧은 길이 안 나면 공간을 휜다',prop:'K(x, x′)'}]);

/* ── 01 곧은 길 내기: 직선 SVM, 특징 2개 ── */
(function(){const cv=$('mgCv'),H=340,MODELS={};let KL=null;
  const st={C:1,mode:'all',sel:-1,sub:null,m:null};
  function base(C){if(!MODELS[C]){KL=KL||SVM.gram(A2,0,'linear');MODELS[C]=SVM.svc(A2,YTR,{C,kind:'linear',K:KL});}return MODELS[C];}
  function role(m,i){const c=Math.abs(m.coef[i]);return c===0?0:c<st.C-1e-9?1:2;} /* 0 아님 · 1 길 가장자리 · 2 길 안/반대편 */
  function solve(){const m0=base(st.C);st.sub=null;st.m=m0;
    if(st.mode==='sv'){const id=m0.sv.slice();st.sub=id;const m=SVM.svc(id.map(i=>A2[i]),id.map(i=>YTR[i]),{C:st.C,kind:'linear'});st.m=m;}
    else if(st.mode==='drop'){const free=m0.sv.slice().sort((a,b)=>Math.abs((YTR[a]?1:-1)*m0.f(A2[a])-1)-Math.abs((YTR[b]?1:-1)*m0.f(A2[b])-1)).slice(0,20),drop=new Set(free);const id=A2.map((_,i)=>i).filter(i=>!drop.has(i));st.sub=id;st.dropped=free;const m=SVM.svc(id.map(i=>A2[i]),id.map(i=>YTR[i]),{C:st.C,kind:'linear'});st.m=m;}
    draw();}
  function draw(){const W=fit(cv,H),ctx=cv.getContext('2d'),C=cols(),pl=50,pr=10,pt=10,pb=38,gw=W-pl-pr,gh=H-pt-pb,bx=BOX2;
    const sx=x=>pl+(x-bx.x0)/(bx.x1-bx.x0)*gw,sy=y=>pt+(1-(y-bx.y0)/(bx.y1-bx.y0))*gh;cv._sx=sx;cv._sy=sy;
    ctx.clearRect(0,0,W,H);axes2(ctx,C,W,H,pl,pt,gw,gh,sx,sy);
    const m0=base(st.C),m=st.m;
    function lineAt(mm,k){const pts=[];for(let q=0;q<=40;q++){const u=bx.x0+(bx.x1-bx.x0)*q/40,s1=(u-SC2.mu[0])/SC2.sd[0],s2=(k-mm.w[0]*s1-mm.b)/mm.w[1];pts.push([sx(u),sy(s2*SC2.sd[1]+SC2.mu[1])]);}return pts;}
    ctx.save();ctx.beginPath();ctx.rect(pl,pt,gw,gh);ctx.clip();
    const up=lineAt(m,1),dn=lineAt(m,-1);ctx.fillStyle=rgba(C.blue,.12);ctx.beginPath();up.forEach((p,i)=>i?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]));dn.slice().reverse().forEach(p=>ctx.lineTo(p[0],p[1]));ctx.fill();
    function poly(pts,col,lw,dash){ctx.strokeStyle=col;ctx.lineWidth=lw;ctx.setLineDash(dash||[]);ctx.beginPath();pts.forEach((p,i)=>i?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]));ctx.stroke();ctx.setLineDash([]);}
    if(st.mode!=='all')poly(lineAt(m0,0),rgba(C.muted,.9),2,[3,4]);
    poly(up,C.blue,1.5,[6,4]);poly(dn,C.blue,1.5,[6,4]);poly(lineAt(m,0),C.blue,3);ctx.restore();
    const inSub=st.sub?new Set(st.sub):null;
    R2tr.forEach((r,i)=>{const on=!inSub||inSub.has(i);circ(ctx,sx(r[0]),sy(r[1]),2.8,rgba(YTR[i]?C.blue:C.text,on?.7:.12),null);});
    if(st.mode==='drop')st.dropped.forEach(i=>{const r=R2tr[i];ctx.strokeStyle=C.amber;ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(sx(r[0])-4,sy(r[1])-4);ctx.lineTo(sx(r[0])+4,sy(r[1])+4);ctx.moveTo(sx(r[0])+4,sy(r[1])-4);ctx.lineTo(sx(r[0])-4,sy(r[1])+4);ctx.stroke();});
    else m0.sv.forEach(i=>{const r=R2tr[i],ro=role(m0,i);ctx.strokeStyle=rgba(C.stamp,ro===1?.95:.45);ctx.lineWidth=ro===1?2.2:1;ctx.beginPath();ctx.arc(sx(r[0]),sy(r[1]),ro===1?6:5,0,7);ctx.stroke();});
    if(st.sel>=0){const r=R2tr[st.sel];ctx.strokeStyle=C.amber;ctx.lineWidth=2.5;ctx.beginPath();ctx.arc(sx(r[0]),sy(r[1]),11,0,7);ctx.stroke();}
    ctx.font=font(12,700);ctx.textAlign='left';const lx=pl+gw-206;ctx.fillStyle=rgba(C.panel,.88);ctx.fillRect(lx-4,pt+4,204,52);ctx.fillStyle=C.blue;ctx.fillText('● 35MPa 합격  ',lx,pt+18);ctx.fillStyle=C.text;ctx.fillText('● 불합격',lx+100,pt+18);
    ctx.fillStyle=C.stamp;ctx.fillText('◯ 굵은 고리 = 길 가장자리',lx,pt+34);ctx.fillStyle=rgba(C.stamp,.7);ctx.fillText('○ 가는 고리 = 길 안·반대편',lx,pt+50);
    info();}
  function info(){const m0=base(st.C),m=st.m,ro=[0,0,0];m0.sv.forEach(i=>ro[role(m0,i)]++);const tr=ML.acc(YTR,A2.map(x=>m.f(x)>0?1:0)),wd=2/Math.hypot(m.w[0],m.w[1]);
    let h='C = <b>'+st.C+'</b> · 경계 f = '+fx(m.w[0],3)+'×(물-결합재비) + '+fx(m.w[1],3)+'×(log 재령) '+(m.b<0?'− ':'+ ')+Math.abs(m.b).toFixed(3)+' (표준화 단위) · 길의 폭 2/‖w‖ = <b>'+wd.toFixed(3)+'</b> · 서포트 벡터 <b>'+m0.sv.length+'</b>개 / 772 (가장자리 '+ro[1]+' · 길 안이나 반대편 '+ro[2]+') · 훈련 정확도 '+pct(tr);
    if(st.mode==='sv')h+='<br><b>서포트 벡터 '+st.sub.length+'개만 남기고 다시 풀었다.</b> 나머지 '+(772-st.sub.length)+'개를 지웠는데 경계가 그대로다(회색 점선 = 처음 경계와 겹친다). w 차이 '+Math.max(Math.abs(m.w[0]-m0.w[0]),Math.abs(m.w[1]-m0.w[1])).toFixed(3)+'. 답이 서포트 벡터에만 기대고 있다는 뜻이다.';
    else if(st.mode==='drop')h+='<br><b>길 가장자리에 가장 가까운 서포트 벡터 '+st.dropped.length+'개(주황 ×)를 지우고 다시 풀었다.</b> 경계가 움직였다(회색 점선 = 처음 경계). 길의 폭 '+(2/Math.hypot(m0.w[0],m0.w[1])).toFixed(3)+' → '+wd.toFixed(3)+'. 길가의 배합을 바꾸면 답이 바뀐다. 반대로 멀리 떨어진 배합은 몇 개를 지워도 그대로다.';
    else h+='<br>'+(st.C<=0.01?'C가 아주 작으면 침범 벌점이 가벼워 길이 넓어진다. 길 안에 든 배합이 많아 서포트 벡터도 많다(규제가 세다).':st.C>=100?'C가 크면 침범을 엄하게 벌해 길이 좁아진다. 그래도 이 두 특징으로는 겹침이 남아 길 안·반대편 배합이 없어지지 않는다.':'C를 바꿔 길의 폭과 서포트 벡터 수를 비교해 보자. 경계는 0편 매뉴얼 씨처럼 "물-결합재비는 낮게, 재령은 길게" 쪽을 합격으로 가른다.');
    if(st.sel>=0){const i=st.sel,r=TR[i],f=m0.f(A2[i]),c=m0.coef[i],rr=role(m0,i);h+='<br>고른 배합: 실제 '+r.strength+'MPa('+PNAME[YTR[i]]+'), 물-결합재비 '+r.wb.toFixed(2)+', 재령 '+r.age+'일 → f(x) = <b>'+fx(f)+'</b>, y·f(x) = '+fx((YTR[i]?1:-1)*f)+', α = '+Math.abs(c).toFixed(3)+' → '+(rr===0?'서포트 벡터가 아니다(지워도 경계 그대로)':rr===1?'<b>길 가장자리의 서포트 벡터</b>':'<b>길 안이나 반대편에 든 서포트 벡터</b>(α = C, 벌점을 받는 중)');}
    $('mgInfo').innerHTML=h;}
  seg('#mgC button',v=>{st.C=+v;st.mode='all';$('mgSV').setAttribute('aria-pressed','false');$('mgDrop').setAttribute('aria-pressed','false');$('mgInfo').textContent='C = '+v+'로 학습 중…';setTimeout(solve,20);});
  $('mgSV').addEventListener('click',e=>{st.mode=st.mode==='sv'?'all':'sv';e.currentTarget.setAttribute('aria-pressed',st.mode==='sv');$('mgDrop').setAttribute('aria-pressed','false');solve();});
  $('mgDrop').addEventListener('click',e=>{st.mode=st.mode==='drop'?'all':'drop';e.currentTarget.setAttribute('aria-pressed',st.mode==='drop');$('mgSV').setAttribute('aria-pressed','false');solve();});
  $('mgReset').addEventListener('click',()=>{st.mode='all';st.sel=-1;st.C=1;document.querySelectorAll('#mgC button').forEach(b=>b.setAttribute('aria-pressed',b.getAttribute('data-v')==='1'));$('mgSV').setAttribute('aria-pressed','false');$('mgDrop').setAttribute('aria-pressed','false');solve();});
  cv.addEventListener('click',e=>{if(!cv._sx)return;const r=cv.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top;let best=-1,bd=300;R2tr.forEach((p,i)=>{const d=(cv._sx(p[0])-mx)**2+(cv._sy(p[1])-my)**2;if(d<bd){bd=d;best=i;}});st.sel=best;draw();});
  let started=false;function start(){if(started)return;started=true;solve();}
  if('IntersectionObserver' in window){const io=new IntersectionObserver(es=>{if(es.some(e=>e.isIntersecting)){start();io.disconnect();}},{rootMargin:'400px'});io.observe(cv);}else start();
  onRedraw(()=>{if(st.m)draw();});})();

/* ── 01 커널 마술 (장난감 예) ── */
(function(){const cv=$('kCv'),H=280,P=[[-2.7,0],[-2.4,0],[-2.1,0],[-1.8,0],[1.75,0],[2.05,0],[2.35,0],[2.65,0],[-1.15,1],[-0.75,1],[-0.35,1],[0.1,1],[0.5,1],[0.85,1],[1.2,1]];
  const st={u:0,raf:0};
  function draw(){const W=fit(cv,H),ctx=cv.getContext('2d'),C=cols(),pl=20,pr=20,pt=16,pb=34,gw=W-pl-pr,gh=H-pt-pb,th=+$('kTh').value;
    const sx=x=>pl+(x+3)/6*gw,sy=y=>pt+gh-(y/8)*gh;ctx.clearRect(0,0,W,H);
    ctx.strokeStyle=C.text;ctx.lineWidth=1.6;ctx.beginPath();ctx.moveTo(pl,sy(0));ctx.lineTo(pl+gw,sy(0));ctx.stroke();ctx.fillStyle=C.muted;ctx.font=font(12);ctx.textAlign='center';[-3,-2,-1,0,1,2,3].forEach(v=>ctx.fillText(v,sx(v),H-16));ctx.fillText('x (원래 특징 하나)',pl+gw/2,H-2);
    if(st.u>0){ctx.strokeStyle=rgba(C.blue,.35*st.u);ctx.lineWidth=1.5;ctx.beginPath();for(let k=0;k<=60;k++){const x=-3+6*k/60,y=Math.min(8,st.u*x*x);k?ctx.lineTo(sx(x),sy(y)):ctx.moveTo(sx(x),sy(y));}ctx.stroke();
      ctx.fillStyle=rgba(C.muted,st.u);ctx.textAlign='left';ctx.fillText('높이 = x² (새 특징)',sx(-2.95),sy(7.4));}
    if(st.u>=1){ctx.strokeStyle=C.blue;ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(pl,sy(th));ctx.lineTo(pl+gw,sy(th));ctx.stroke();ctx.fillStyle=C.blue;ctx.font=font(12,700);ctx.textAlign='right';ctx.fillText('휜 공간의 곧은 경계: x² = '+th.toFixed(1),pl+gw,sy(th)-6);
      const r=Math.sqrt(th);ctx.strokeStyle=C.stamp;ctx.lineWidth=2;ctx.setLineDash([4,3]);[-r,r].forEach(x=>{ctx.beginPath();ctx.moveTo(sx(x),sy(th));ctx.lineTo(sx(x),sy(0)+8);ctx.stroke();});ctx.setLineDash([]);
      ctx.fillStyle=C.stamp;ctx.textAlign='center';[-r,r].forEach(x=>ctx.fillText((x<0?'−':'')+r.toFixed(2),sx(x),sy(0)+22));}
    P.forEach(p=>{const y=st.u*p[0]*p[0];circ(ctx,sx(p[0]),sy(y),6,p[1]?C.blue:C.text,C.panel,1.5);});info();}
  function info(){const th=+$('kTh').value;let bad=0;P.forEach(p=>{if((p[0]*p[0]<th?1:0)!==p[1])bad++;});
    $('kInfo').innerHTML=st.u<1?'수직선 위에서는 파란 무리가 가운데, 검은 무리가 양쪽에 있다. 문턱 하나(점 하나)로는 어디를 잘라도 갈리지 않는다. "공간 휘기"를 눌러 각 점을 x²만큼 들어 올려 보자.':
      '점을 (x, x²)로 올리니 <b>가로 직선 하나</b>로 갈린다(틀린 점 <b>'+bad+'</b>개). 이 직선을 원래 수직선으로 내려 보면 경계가 x = ±'+Math.sqrt(th).toFixed(2)+' <b>두 군데</b>다. 휜 공간의 곧은 길이 원래 공간에서는 굽은(여기서는 두 갈래) 경계가 된다. 슬라이더로 높이를 바꿔 틀린 점이 0이 되는 범위를 찾아보자.';}
  function anim(to){cancelAnimationFrame(st.raf);if(REDUCE){st.u=to;draw();return;}const from=st.u,t0=performance.now();(function go(now){const k=Math.min(1,(now-t0)/1100);st.u=from+(to-from)*ease(k);draw();if(k<1)st.raf=requestAnimationFrame(go);})(t0);}
  $('kGo').addEventListener('click',()=>anim(1));$('kBack').addEventListener('click',()=>anim(0));
  $('kTh').addEventListener('input',()=>{$('kThO').textContent=(+$('kTh').value).toFixed(1);draw();});draw();onRedraw(draw);})();

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
  $('concTbl').innerHTML='<table><thead><tr><th>#</th>'+FEAT.map(f=>'<th><span class="tg x">특징 X</span>'+FEATK[f]+'</th>').join('')+'<th><span class="tg t">타겟 y (회귀)</span>압축 강도<br>(MPa)</th><th><span class="tg t">타겟 y (분류)</span>35MPa 합격</th><th><span class="tg t">타겟 y</span>등급</th><th>용도</th></tr></thead><tbody>'+
    ROWS.map(r=>'<tr><td>'+(r.i+1)+'</td>'+FEAT.map(f=>'<td>'+r[f]+'</td>').join('')+'<td class="y">'+r.strength+'</td><td class="y">'+PNAME[r.pass]+'</td><td class="y">'+GRADES[r.grade]+'</td><td>'+(r.test?'<span class="pill no">봉인</span>':'훈련')+'</td></tr>').join('')+'</tbody></table>';
  const r0=TR[0],z0=SC.t([fv(r0)])[0];
  $('stdTbl').innerHTML='<thead><tr><th>특징</th><th>평균</th><th>표준편차</th><th>예: 훈련 첫 배합</th></tr></thead><tbody>'+FEAT.map((f,j)=>'<tr><td>'+FEATK[f]+'</td><td>'+SC.mu[j].toFixed(1)+'</td><td>'+SC.sd[j].toFixed(1)+'</td><td>'+r0[f]+' → <b>'+fx(z0[j])+'</b></td></tr>').join('')+'</tbody>';
  /* 표준화 전 거리 제곱에서 특징마다의 몫 = 분산 몫 */
  const v=SC.sd.map(s=>s*s),tot=v.reduce((a,b)=>a+b,0),ord=FEAT.map((f,j)=>[j,v[j]/tot]).sort((a,b)=>b[1]-a[1]),mx=ord[0][1];
  $('varBars').innerHTML=ord.map(([j,s])=>{const key=FEAT[j]==='water'||FEAT[j]==='superplasticizer';return '<div class="row"><span style="text-align:left;font-family:var(--font)'+(key?';color:var(--stamp);font-weight:700':'')+'">'+FEATK[FEAT[j]]+'</span><div class="tr"><i style="left:0;width:'+(s/mx*100)+'%;background:'+(key?'var(--stamp)':'var(--muted)')+'"></i><i style="left:'+(0.125/mx*100)+'%;width:2px;background:var(--blue)"></i></div><span>'+(s*100).toFixed(s<0.02?1:0)+'%</span></div>';}).join('')+'<p class="foot" style="margin-top:6px"><span style="color:var(--blue);font-weight:700">파란 눈금</span> = 표준화한 뒤의 몫 12.5%</p>';})();

/* ── 03 AI 비서 ── */
const MAIN_PROMPT=`너는 머신러닝 수업 조교다. Google Colab에서 바로 실행되는 파이썬 코드를 써 줘.

[환경] Colab 기본 라이브러리(numpy, pandas, scikit-learn)와 ucimlrepo만 쓴다. 첫 줄에 scikit-learn 버전을 출력한다.
[데이터] 〔ucimlrepo의 fetch_ucirepo(id=165)〕 로 콘크리트 압축 강도 데이터를 읽는다.
  - 특징 X: 〔시멘트, 고로 슬래그, 플라이 애시, 물, 고성능 감수제, 굵은 골재, 잔골재, 재령〕 8개 (열 순서대로 이름을 붙인다)
  - 타겟 1 (분류): 〔압축 강도가 35MPa 이상이면 1, 아니면 0〕 → SVC
  - 타겟 2 (회귀): 〔압축 강도(MPa)〕 → SVR
[규정]
  1) 강도를 〔25, 45〕MPa로 나눈 3등급으로 stratify 해서 train_test_split(test_size=0.25, random_state=42)로 먼저 나눈다.
  2) StandardScaler는 훈련 데이터로만 fit 하고, 시험 데이터는 transform만 한다.
  3) 커널은 RBF, gamma="scale". C는 〔0.1, 1, 10, 100〕 중에서 훈련 데이터 안의 5겹 교차 검증(분류는 StratifiedKFold, 회귀는 KFold, shuffle=True, random_state=42)으로 고른다.
[출력] 교차 검증 점수표, 고른 C, 시험 정확도와 혼동행렬(분류), 시험 R²와 MAE(회귀), 서포트 벡터 수.
[금지] 시험 데이터로 C·gamma를 고르지 않는다. 데이터를 지어내지 않는다. r2_score는 (정답, 예측) 순서로 쓴다.
[형식] 셀 단위로 나누고, 각 셀 위에 무엇을 하는지 한국어 주석을 단다. 코드 뒤에 내가 확인해야 할 점 3가지를 적어 줘.`;
$('mainPrompt').textContent=MAIN_PROMPT;$('copyMain').addEventListener('click',()=>copyText(MAIN_PROMPT,'프롬프트를 복사했습니다'));
(function(){const LINES=[
 ['import pandas as pd'],['from ucimlrepo import fetch_ucirepo'],['from sklearn.model_selection import train_test_split, GridSearchCV'],['from sklearn.svm import SVC, SVR'],['from sklearn.metrics import accuracy_score, r2_score'],[''],
 ['uci = fetch_ucirepo(id=165)',0,'UCI에서 콘크리트 데이터를 실제로 받았다. 괜찮다.'],
 ['X = uci.data.features',0,'특징 여덟 열이다. 괜찮다.'],['strength = uci.data.targets.iloc[:, 0]',0,'회귀 타겟(MPa)이다. 괜찮다.'],['y = (strength >= 35).astype(int)   # 35MPa 합격 = 1',0,'분류 이름표를 만들었다. 괜찮다.'],[''],
 ['X_tr, X_te, y_tr, y_te, s_tr, s_te = train_test_split(X, y, strength, test_size=0.25)',1,'<b>stratify와 random_state가 없다.</b> 돌릴 때마다 점수가 달라지고, 고강도 배합이 한쪽으로 쏠릴 수 있다. 강도 등급으로 <code>stratify</code>하고 <code>random_state=42</code>를 넣는다.'],
 ['# SVM은 단위와 상관없으니 표준화는 생략한다',2,'<b>표준화를 뺐다.</b> RBF 커널은 거리로 닮음을 잰다. 표준화하지 않으면 시멘트·골재(kg 단위로 수백)가 거리를 차지하고 물·감수제가 사라진다(02장). 이 데이터에서 회귀 R²가 0.73 근처로 떨어져 합격선 아래다. <code>StandardScaler</code>를 훈련 데이터로 fit 한다.'],[''],
 ['grid = GridSearchCV(SVC(), {"C": [0.1, 1, 10, 100]}, cv=5)',0,'C 후보를 교차 검증으로 비교하는 도구다. 여기까지는 괜찮다.'],
 ['grid.fit(X_te, y_te)   # 가장 좋은 C 찾기',3,'<b>시험 데이터로 C를 골랐다.</b> 봉인된 시험지를 근무 조건 고르기에 썼으니, 마지막 시험 점수를 믿을 수 없다. <code>grid.fit(X_tr, y_tr)</code>처럼 훈련 데이터 안에서 고른다.'],
 ['clf = SVC(C=grid.best_params_["C"], gamma=50)   # gamma가 클수록 경계가 부드럽다',4,'<b>γ의 방향이 거꾸로다.</b> γ가 클수록 한 배합이 닿는 거리가 짧아져 경계가 구불구불해지고(외우기), 작을수록 부드럽다. 50은 표준화한 8특징에서 기본값(0.125)의 400배다. 특별한 까닭이 없으면 <code>gamma="scale"</code>로 두고, 바꾸려면 C와 함께 교차 검증한다.'],
 ['clf.fit(X_tr, y_tr)',0,'훈련 데이터로 학습했다. 괜찮다.'],['print("정확도:", accuracy_score(y_te, clf.predict(X_te)))',0,'시험 데이터로, (정답, 예측) 순서로 채점했다. 괜찮다.'],[''],
 ['reg = SVR(C=100).fit(X_tr, s_tr)',0,'회귀에는 SVR을 썼고 훈련 데이터로 학습했다. 이 줄 자체는 괜찮다(표준화가 빠진 문제는 13번 줄).'],
 ['print("R²:", r2_score(reg.predict(X_te), s_te))',5,'<b>r2_score의 인자 순서가 거꾸로다.</b> <code>r2_score(y_true, y_pred)</code>다. R²는 정답의 흩어짐을 기준으로 나누므로 순서를 바꾸면 다른 숫자가 나온다. <code>r2_score(s_te, reg.predict(X_te))</code>로 고친다.']];
  const host=$('bugCode'),found=new Set();
  host.innerHTML=LINES.map((l,i)=>'<div class="ln" role="listitem" tabindex="0" data-i="'+i+'"><span class="no">'+(i+1)+'</span><span>'+hl(l[0]||' ')+'</span></div>').join('');
  function score(){$('bugScore').textContent=found.size+' / 5';if(found.size===5){$('bugScore').className='pill ok';}}
  function pick(el){const i=+el.getAttribute('data-i'),l=LINES[i];if(!l[0])return;
    if(l[1]){found.add(l[1]);host.querySelectorAll('.ln').forEach(x=>{if(LINES[+x.getAttribute('data-i')][1]===l[1])x.classList.add('hit');});$('bugMsg').innerHTML=(i+1)+'번 줄: '+l[2]+(found.size===5?'<br><b>다섯 군데를 모두 찾았다.</b> 고친 코드는 A장의 Colab 코드와 같은 일을 한다.':'');}
    else{el.classList.add('miss');$('bugMsg').innerHTML=(i+1)+'번 줄: '+(l[2]||'이 줄은 괜찮다.');}score();}
  host.addEventListener('click',e=>{const el=e.target.closest('.ln');if(el)pick(el);});
  host.addEventListener('keydown',e=>{if((e.key==='Enter'||e.key===' ')&&e.target.classList.contains('ln')){e.preventDefault();pick(e.target);}});
  const NAMES=['','나누기 규정','표준화 생략','시험 데이터로 C 고르기','γ의 방향','r2_score 인자 순서'];
  $('bugReveal').addEventListener('click',()=>{LINES.forEach((l,i)=>{if(l[1]){found.add(l[1]);host.querySelector('[data-i="'+i+'"]').classList.add('hit','reveal');}});score();
    $('bugMsg').innerHTML='빨간 줄 다섯 군데: '+LINES.map((l,i)=>l[1]?'<b>'+(i+1)+'번</b> '+NAMES[l[1]]:'').filter(Boolean).join(' · ')+'.';});
  $('bugReset').addEventListener('click',()=>{found.clear();host.querySelectorAll('.ln').forEach(x=>x.classList.remove('hit','miss','reveal'));$('bugScore').className='pill';score();$('bugMsg').textContent='줄을 하나 눌러 보세요.';});score();})();
(function(){const CK=[['데이터를 실제로 읽었나','AI가 예제 데이터(iris 등)를 지어내거나 바꾸지 않았는지, 행이 1,030개인지 본다.'],['특징 열과 두 타겟이 내가 말한 것과 같나','압축 강도(정답)가 특징에 섞여 들어가면 정답을 보고 푸는 셈이다.'],['나누기를 먼저 하고 stratify·random_state를 넣었나','등급 비율이 시험지에도 유지되고, 점수가 매번 같게 나온다.'],['표준화가 있고, 훈련 데이터로만 fit 했나','SVM은 표준화가 사실상 필수다. 전체로 fit 하면 누수다. Pipeline으로 묶으면 교차 검증 안에서도 안전하다.'],['C·γ를 훈련 데이터 안의 교차 검증으로 골랐나','GridSearchCV.fit(X_test, …)가 보이면 시험지를 훈련에 쓴 것이다.'],['C·γ의 방향을 맞게 읽었나','C가 크면 규제가 약하다(5-4와 같은 방향). γ가 크면 경계가 구불구불하다. 주석과 값이 맞는지 본다.'],['분류에는 SVC, 회귀에는 SVR을 썼나','강도(MPa)를 SVC에 넣으면 "연속값 이름표" 오류가 나거나, 이름표를 반올림해 엉뚱한 분류가 된다.'],['지표 함수의 인자 순서가 (정답, 예측)인가','r2_score, mean_absolute_error, confusion_matrix 모두 정답이 먼저다.'],['결과를 기준선·합격선과 비교했나','분류는 매뉴얼 씨 76.7%와 늘 다수 쪽, 회귀는 매뉴얼 경험식 R² 0.735와 늘 평균. 넘지 못하면 뽑을 까닭이 없다.']];
  const ul=$('checks');ul.innerHTML=CK.map((c,i)=>'<li><input type="checkbox" id="ck'+i+'"><label for="ck'+i+'"><b>'+c[0]+'</b><small>'+c[1]+'</small></label></li>').join('');
  function upd(){const n=ul.querySelectorAll('input:checked').length;ul.querySelectorAll('li').forEach(li=>li.classList.toggle('done',li.querySelector('input').checked));$('checkScore').textContent='확인 '+n+' / '+CK.length+(n===CK.length?' — 이제 돌려 본다. 결과도 확인 대상이다.':'');}
  ul.addEventListener('change',upd);upd();})();

/* ── 04 수습 기간 ①: RBF SVC를 SMO로 (특징 2개) ── */
(function(){const cv=$('trainCv'),oc=$('objCv'),H=320,KS={};
  const st={C:10,g:0.5,S:null,hist:[],play:false,sel:-1,m:null};
  function K(g){return KS[g]||(KS[g]=SVM.gram(A2,g,'rbf'));}
  function reset(){stop();st.S=SVM.svcBegin(A2,YTR,{C:st.C,gamma:st.g,K:K(st.g)});st.hist=[];st.m=st.S.model();st.sel=-1;draw();}
  function adv(n){st.S.step(n);st.m=st.S.model();if(isFinite(st.S.gap))st.hist.push([st.S.it,st.S.gap,st.m.sv.length]);}
  function draw(){const W=fit(cv,H),ctx=cv.getContext('2d'),C=cols(),pl=50,pr=10,pt=10,pb=38,gw=W-pl-pr,gh=H-pt-pb,bx=BOX2,m=st.m;
    const sx=x=>pl+(x-bx.x0)/(bx.x1-bx.x0)*gw,sy=y=>pt+(1-(y-bx.y0)/(bx.y1-bx.y0))*gh;cv._sx=sx;cv._sy=sy;ctx.clearRect(0,0,W,H);
    const cs=8,nx=Math.ceil(gw/cs),ny=Math.ceil(gh/cs),V=new Float64Array((nx+1)*(ny+1)),X=i=>pl+Math.min(gw,i*cs),Y=j=>pt+Math.min(gh,j*cs);
    const ux=X=>bx.x0+(X-pl)/gw*(bx.x1-bx.x0),uy=Y=>bx.y0+(1-(Y-pt)/gh)*(bx.y1-bx.y0);
    if(m.sv.length)for(let j=0;j<=ny;j++)for(let i=0;i<=nx;i++)V[j*(nx+1)+i]=m.f(toS2(ux(X(i)),uy(Y(j))));
    for(let j=0;j<ny;j++)for(let i=0;i<nx;i++){const f=(V[j*(nx+1)+i]+V[j*(nx+1)+i+1]+V[(j+1)*(nx+1)+i]+V[(j+1)*(nx+1)+i+1])/4;ctx.fillStyle=f>=0?rgba(C.blue,.06+.2*Math.min(1,f)):rgba(C.text,.04+.1*Math.min(1,-f));ctx.fillRect(X(i),Y(j),X(i+1)-X(i),Y(j+1)-Y(j));}
    axes2(ctx,C,W,H,pl,pt,gw,gh,sx,sy);
    ctx.strokeStyle=rgba(C.c_man,.8);ctx.lineWidth=1.5;ctx.setLineDash([2,4]);ctx.beginPath();ctx.moveTo(sx(0.5),pt);ctx.lineTo(sx(0.5),pt+gh);ctx.moveTo(pl,sy(Math.log10(28)));ctx.lineTo(pl+gw,sy(Math.log10(28)));ctx.stroke();ctx.setLineDash([]);
    if(m.sv.length){ctx.save();ctx.beginPath();ctx.rect(pl,pt,gw,gh);ctx.clip();ctx.strokeStyle=C.blue;ctx.lineWidth=1.4;ctx.setLineDash([5,4]);contour(ctx,V,nx,ny,X_=>pl+X_*cs,Y_=>pt+Y_*cs,1);contour(ctx,V,nx,ny,X_=>pl+X_*cs,Y_=>pt+Y_*cs,-1);ctx.setLineDash([]);ctx.lineWidth=3;contour(ctx,V,nx,ny,X_=>pl+X_*cs,Y_=>pt+Y_*cs,0);ctx.restore();}
    R2tr.forEach((r,i)=>circ(ctx,sx(r[0]),sy(r[1]),2.6,rgba(YTR[i]?C.blue:C.text,.75),null));
    m.sv.forEach(i=>{const r=R2tr[i];ctx.strokeStyle=rgba(C.stamp,.75);ctx.lineWidth=1.3;ctx.beginPath();ctx.arc(sx(r[0]),sy(r[1]),5,0,7);ctx.stroke();});
    if(st.sel>=0){const r=R2tr[st.sel];ctx.strokeStyle=C.amber;ctx.lineWidth=2.5;ctx.beginPath();ctx.arc(sx(r[0]),sy(r[1]),11,0,7);ctx.stroke();}
    ctx.font=font(12,700);ctx.textAlign='left';const lx=pl+8;ctx.fillStyle=rgba(C.panel,.85);ctx.fillRect(lx-4,pt+4,196,50);ctx.fillStyle=C.blue;ctx.fillText('● 합격  ',lx,pt+18);ctx.fillStyle=C.text;ctx.fillText('● 불합격',lx+56,pt+18);ctx.fillStyle=C.stamp;ctx.fillText('○ 서포트 벡터 (α > 0)',lx,pt+33);ctx.fillStyle=C.c_man;ctx.fillText('┆ 매뉴얼 씨의 직각 규칙',lx,pt+48);
    drawObj();info();}
  function drawObj(){const W=fit(oc,240),ctx=oc.getContext('2d'),C=cols(),pl=48,pr=10,pt=12,pb=30,gw=W-pl-pr,gh=240-pt-pb,h=st.hist,n=Math.max(200,h.length?h[h.length-1][0]:0);
    ctx.clearRect(0,0,W,240);ctx.strokeStyle=C.rule;ctx.strokeRect(pl,pt,gw,gh);const L0=-3.5,L1=1.5,ly=v=>pt+(1-(Math.log10(Math.max(v,1e-4))-L0)/(L1-L0))*gh;
    ctx.fillStyle=C.muted;ctx.font=font(12);ctx.textAlign='right';[0.001,0.01,0.1,1,10].forEach(v=>ctx.fillText(v,pl-5,ly(v)+4));
    ctx.setLineDash([4,4]);ctx.strokeStyle=C.ok||C.blue;ctx.beginPath();ctx.moveTo(pl,ly(1e-3));ctx.lineTo(pl+gw,ly(1e-3));ctx.stroke();ctx.setLineDash([]);ctx.fillStyle=C.ok||C.blue;ctx.textAlign='left';ctx.fillText('멈춤 0.001',pl+6,ly(1e-3)-5);
    ctx.fillStyle=C.muted;ctx.textAlign='center';ctx.fillText('반복 횟수 (0 ~ '+n+')',pl+gw/2,240-8);ctx.save();ctx.translate(11,pt+gh/2);ctx.rotate(-Math.PI/2);ctx.fillText('어긋남 (로그)',0,0);ctx.restore();
    const smax=Math.max(1,...h.map(p=>p[2]));ctx.fillStyle=rgba(C.stamp,.16);h.forEach(p=>{const x=pl+p[0]/n*gw,hh=p[2]/772*gh*0.45;ctx.fillRect(x-1,pt+gh-hh,2,hh);});
    ctx.strokeStyle=C.blue;ctx.lineWidth=2.2;ctx.beginPath();h.forEach((p,i)=>{const x=pl+p[0]/n*gw,y=Math.max(pt,ly(p[1]));i?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.stroke();
    if(h.length){const last=h[h.length-1];ctx.fillStyle=C.blue;ctx.font=font(12,700,1);ctx.textAlign='right';ctx.fillText('어긋남 '+last[1].toFixed(4),pl+gw-6,pt+16);ctx.fillStyle=C.stamp;ctx.fillText('서포트 벡터 '+last[2]+'개 (막대)',pl+gw-6,pt+32);}}
  function info(){const m=st.m,S=st.S,tr=m.sv.length?ML.acc(YTR,A2.map(x=>m.f(x)>0?1:0)):null;
    let h='C = <b>'+st.C+'</b> · γ = <b>'+st.g+'</b> · 반복 <b>'+S.it+'</b>번 · 서포트 벡터 <b>'+m.sv.length+'</b>개 / 772'+(tr!=null?' · 훈련 정확도 <b>'+pct(tr)+'</b>':'')+(S.done?' · <b style="color:var(--ok)">학습 완료</b> (가장 어긋난 두 배합의 어긋남 '+S.gap.toPrecision(2)+' &lt; 0.001)':'');
    if(st.sel>=0){const i=st.sel,r=TR[i],a=S.a[i],f=m.f(A2[i]);h+='<br>고른 배합: 실제 '+r.strength+'MPa('+PNAME[YTR[i]]+'), 물-결합재비 '+r.wb.toFixed(2)+', 재령 '+r.age+'일 → α = <b>'+a.toFixed(3)+'</b>'+(a>=st.C-1e-9?' (= C, 길 안이나 반대편)':a>0?' (0과 C 사이, 길 가장자리)':' (서포트 벡터 아님)')+', f(x) = <b>'+fx(f)+'</b>';}
    else if(S.it===0)h+='<br>처음에는 모든 α가 0이라 경계가 없다. ▶ 재생을 누르면 가장 어긋난 두 배합부터 α를 고쳐 나간다.';
    else if(S.done)h+='<br>'+(st.g>=50?'γ = 50: 배합 하나하나를 작은 섬처럼 감싼다. 훈련 정확도는 오르지만 서포트 벡터가 늘고, 처음 보는 배합에서는 믿기 어렵다(외우기).':st.g<=0.1?'γ = 0.1: 한 배합이 멀리까지 닿아 경계가 거의 직선이다. 01장의 직선 SVM과 비슷하다.':st.C<=0.1?'C = 0.1: 침범 벌점이 가벼워 길이 넓고, 서포트 벡터가 많다.':'경계가 "물-결합재비 낮고 오래 굳힌" 왼쪽 위를 합격으로 감싸며, 매뉴얼 씨의 직각 규칙(점선)을 부드럽게 휜 모양이 된다. γ와 C를 바꿔 보자.');
    $('trainInfo').innerHTML=h;}
  function chips(k){document.querySelectorAll('#loopChips button').forEach((b,i)=>{b.classList.toggle('on',i===k);b.classList.toggle('done',k>=0&&i<k);});}
  let raf=0,ph=0;
  function frame(){if(!st.play)return;ph=(ph+1)%9;chips(Math.floor(ph/3));const k=st.S.it<40?1:st.S.it<300?6:st.S.it<2000?40:200;adv(k);draw();if(st.S.done){stop();return;}raf=requestAnimationFrame(frame);}
  function stop(){st.play=false;cancelAnimationFrame(raf);$('trPlay').textContent='▶ 재생';chips(-1);}
  $('trPlay').addEventListener('click',()=>{if(st.play){stop();return;}if(st.S.done)reset();st.play=true;$('trPlay').textContent='❚❚ 멈춤';if(REDUCE){while(!st.S.done)adv(500);draw();stop();return;}raf=requestAnimationFrame(frame);});
  $('trStep').addEventListener('click',async()=>{stop();if(st.S.done)return;for(let k=0;k<3;k++){chips(k);await wait(REDUCE?0:170);}adv(1);draw();chips(-1);});
  $('trEnd').addEventListener('click',()=>{stop();while(!st.S.done)adv(500);draw();});
  $('trReset').addEventListener('click',reset);
  seg('#trC button',v=>{st.C=+v;reset();});seg('#trG button',v=>{st.g=+v;reset();});
  cv.addEventListener('click',e=>{if(!cv._sx)return;const r=cv.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top;let best=-1,bd=300;R2tr.forEach((p,i)=>{const d=(cv._sx(p[0])-mx)**2+(cv._sy(p[1])-my)**2;if(d<bd){bd=d;best=i;}});st.sel=best;draw();});
  let started=false;function start(){if(started)return;started=true;reset();}
  if('IntersectionObserver' in window){const io=new IntersectionObserver(es=>{if(es.some(e=>e.isIntersecting)){start();io.disconnect();}},{rootMargin:'400px'});io.observe(cv);}else start();
  window.addEventListener('hashchange',()=>{if(location.hash==='#m4')start();});
  onRedraw(()=>{if(st.S)draw();});})();

/* ── 04 수습 기간 ②: SVR의 ε 띠 (28일 시편, 특징 1개) ── */
(function(){const cv=$('epsCv'),H=300,T28=TR.filter(r=>r.age===28),raw=T28.map(r=>[r.wb]),sc=ML.scaler(raw),X1=sc.t(raw),Y1=T28.map(r=>r.strength),Kx=SVM.gram(X1,1,'rbf'),MS={};let e=5;
  function model(eps){if(!MS[eps]){const m=SVM.svr(X1,Y1,{C:100,gamma:1,epsilon:eps,K:Kx});m.p=X1.map(m.predict);MS[eps]=m;}return MS[eps];}
  const xs=raw.map(r=>r[0]),x0=Math.min(...xs)-0.02,x1=Math.max(...xs)+0.02,y0=0,y1=Math.max(...Y1)*1.05;
  function draw(){const W=fit(cv,H),ctx=cv.getContext('2d'),C=cols(),pl=46,pr=12,pt=12,pb=38,gw=W-pl-pr,gh=H-pt-pb,m=model(e),sx=x=>pl+(x-x0)/(x1-x0)*gw,sy=y=>pt+(1-(y-y0)/(y1-y0))*gh;
    ctx.clearRect(0,0,W,H);ctx.strokeStyle=C.rule;ctx.strokeRect(pl,pt,gw,gh);ctx.fillStyle=C.muted;ctx.font=font(12);
    for(let x=0.3;x<=x1;x+=0.1){ctx.textAlign='center';ctx.fillText(x.toFixed(1),sx(x),H-20);}[0,20,40,60,80].forEach(v=>{if(v>y1)return;ctx.textAlign='right';ctx.fillText(v,pl-5,sy(v)+4);});
    ctx.textAlign='center';ctx.fillText('물-결합재비 (28일 시편만)',pl+gw/2,H-3);ctx.save();ctx.translate(11,pt+gh/2);ctx.rotate(-Math.PI/2);ctx.fillText('28일 압축 강도 (MPa)',0,0);ctx.restore();
    const cur=[];for(let k=0;k<=120;k++){const u=x0+(x1-x0)*k/120;cur.push([sx(u),m.f([(u-sc.mu[0])/sc.sd[0]])]);}
    ctx.save();ctx.beginPath();ctx.rect(pl,pt,gw,gh);ctx.clip();ctx.fillStyle=rgba(C.blue,.14);ctx.beginPath();cur.forEach((p,i)=>i?ctx.lineTo(p[0],sy(p[1]+e)):ctx.moveTo(p[0],sy(p[1]+e)));cur.slice().reverse().forEach(p=>ctx.lineTo(p[0],sy(p[1]-e)));ctx.fill();
    ctx.strokeStyle=C.blue;ctx.lineWidth=1.3;ctx.setLineDash([5,4]);[e,-e].forEach(o=>{ctx.beginPath();cur.forEach((p,i)=>i?ctx.lineTo(p[0],sy(p[1]+o)):ctx.moveTo(p[0],sy(p[1]+o)));ctx.stroke();});ctx.setLineDash([]);
    ctx.lineWidth=3;ctx.beginPath();cur.forEach((p,i)=>i?ctx.lineTo(p[0],sy(p[1])):ctx.moveTo(p[0],sy(p[1])));ctx.stroke();ctx.restore();
    const sv=new Set(m.sv);raw.forEach((r,i)=>{circ(ctx,sx(r[0]),sy(Y1[i]),3,rgba(C.text,sv.has(i)?.8:.35),null);if(sv.has(i)){ctx.strokeStyle=rgba(C.stamp,.8);ctx.lineWidth=1.3;ctx.beginPath();ctx.arc(sx(r[0]),sy(Y1[i]),5.5,0,7);ctx.stroke();}});
    ctx.font=font(12,700);ctx.textAlign='right';ctx.fillStyle=C.stamp;ctx.fillText('○ 서포트 벡터 '+m.sv.length+'개 / '+X1.length,pl+gw-6,pt+16);ctx.fillStyle=C.blue;ctx.fillText('띠 = 예측 ± '+e+'MPa',pl+gw-6,pt+32);
    const inside=m.p.filter((p,i)=>Math.abs(p-Y1[i])<e-1e-6).length;
    $('epsInfo').innerHTML='ε = <b>'+e+'MPa</b> · 서포트 벡터 <b>'+m.sv.length+'</b>개 / '+X1.length+'건 · 띠 안 '+inside+'건(α = 0, 지워도 곡선 그대로) · 훈련 R² '+fmtR(r2(Y1,m.p))+' · MAE '+fmtM(mae(Y1,m.p))+'MPa. '+
      (e<=0.5?'띠가 아주 좁으면 거의 모든 배합이 띠 밖이라 거의 다 서포트 벡터다. 05장의 8특징 SVR(ε = 0.1)이 이렇다.':e>=10?'띠가 넓으면 서포트 벡터가 크게 줄어 가벼워진다. 다만 ±10MPa 안이면 "맞음"으로 치는 셈이라, 현장 기준에 비해 너무 너그럽다.':'ε을 바꾸면 곡선은 조금만 변하지만 서포트 벡터 수는 크게 바뀐다. 같은 물-결합재비에서도 강도가 아래위로 흩어지는 것은 시멘트 대신 슬래그·플라이 애시를 쓴 배합의 차이다(28일에는 아직 덜 굳었다).');}
  seg('#epsC button',v=>{e=+v;draw();});
  let started=false;function start(){if(started)return;started=true;draw();onRedraw(draw);}
  if('IntersectionObserver' in window){const io=new IntersectionObserver(es=>{if(es.some(x=>x.isIntersecting)){start();io.disconnect();}},{rootMargin:'400px'});io.observe(cv);}else start();})();

/* ── 05 성과 평가 ── */
let EVAL=null;
function renderEval(){if(EVAL)return;const m=getSVC(C_CLS),bs=baseAcc(),C=cols(),pte=B.map(m.predict);EVAL={m};
  $('evalStat').innerHTML=statGrid([[pct(m.te),'시험 정확도 ('+Math.round(m.te*258)+' / 258건)','a'],[pct(m.tr),'훈련 정확도 (772건)'],[pct(PASSLINE),'합격선 · 5-0 매뉴얼 씨','m'],[pct(bs.acc),'기준선 · 늘 '+rago('"'+PNAME[bs.k]+'"')+'만 답할 때']],4)+
    '<p class="foot">서포트 벡터 '+m.sv.length+'개 / 772 · SMO 반복 '+m.iters+'번 · 브라우저 학습 '+Math.max(1,Math.round(m.ms))+'ms · γ = "scale" = '+GAM.toFixed(3)+'</p>';
  const M=ML.confusion(YTE,pte,2);$('cm7').innerHTML=cmGrid(M,PNAME,k=>k?C.blue:C.muted);
  const t0=performance.now(),o=SVM.ovo(A,GTR,3,{C:C_CLS,gamma:GAM}),p3=B.map(o.predict),a3=ML.acc(GTE,p3);EVAL.ms3=performance.now()-t0;
  $('cm3').innerHTML=cmGrid(ML.confusion(GTE,p3,3),GRADES,k=>GRADECOL[k]);
  $('g3Stat').innerHTML=statGrid([[pct(a3),'3등급 시험 정확도 (일대일 투표, C = 10)','a'],[pct(GTE.filter(v=>v===1).length/GTE.length),'기준선 · 늘 "보통 강도"']],2)+'<p class="foot">이진 SVC 셋(저↔보통, 저↔고, 보통↔고)이 투표한다. 5-4 로지스틱 회귀(소프트맥스)는 이 과제에서 크게 못 미쳤다.</p>';
  /* 길에서의 거리 */
  const F=B.map(m.f);let ni=0,ci=0,no=0,co=0;F.forEach((f,i)=>{const ok=(f>0?1:0)===YTE[i];if(Math.abs(f)<1){ni++;if(ok)ci++;}else{no++;if(ok)co++;}});
  $('distTxt').innerHTML='SVC의 기본 출력은 f(x), 곧 "경계에서 어느 쪽으로 얼마나 떨어졌나"다. 시험 배합 258건 중 <b>길 안(|f| &lt; 1)에 든 '+ni+'건</b>의 정확도는 <b>'+pct(ci/ni)+'</b>, 길 밖에 확실히 자리 잡은 '+no+'건은 <b>'+pct(co/no)+'</b>다. 틀린 '+(M[0][1]+M[1][0])+'건 가운데 강도 미달 배합을 합격이라 한 것이 <b style="color:var(--stamp)">'+M[0][1]+'건</b>이다. 출하 판정에 쓴다면 "길 안에 든 배합은 재시험"처럼 거리로 규칙을 정할 수 있다. 확률이 꼭 필요하면 <code>probability=True</code>(Platt 보정)나 5-4 로지스틱 회귀를 쓴다.';
  /* 회귀 */
  const R=getSVR(C_REG);
  $('regStat').innerHTML=statGrid([['R² '+fmtR(R.R),'시험 R² (MAE '+fmtM(R.M)+'MPa)','a'],['R² '+fmtR(R.trR),'훈련 R² (772건)'],['R² '+fmtR(LINE_R2),'회귀 합격선 · 매뉴얼 경험식 (MAE '+fmtM(LINE_MAE)+')','m'],['MAE '+fmtM(BASE_MAE),'기준선 · 늘 훈련 평균 '+BASE.toFixed(1)+'MPa']],4)+
    '<p class="foot">서포트 벡터 '+R.sv.length+'개 / 772 (ε = 0.1MPa가 좁아 거의 다) · SMO 반복 '+R.iters+'번 · 브라우저 학습 '+Math.max(1,Math.round(R.ms))+'ms</p>';
  const rc=$('regCv');let cur='100';
  function drawReg(){const W=fit(rc,320),ctx=rc.getContext('2d'),C=cols(),pl=46,pr=12,pt=12,pb=38,gw=W-pl-pr,gh=320-pt-pb,mx=85,sx=v=>pl+v/mx*gw,sy=v=>pt+(1-v/mx)*gh;
    const p=cur==='man'?MAN:getSVR(+cur).p,col=cur==='man'?C.c_man:cur==='100'?C.blue:cur==='10'?C.info||C.blue:C.amber;
    ctx.clearRect(0,0,W,320);ctx.strokeStyle=C.rule;ctx.strokeRect(pl,pt,gw,gh);ctx.fillStyle=C.muted;ctx.font=font(12);[0,20,40,60,80].forEach(v=>{ctx.textAlign='center';ctx.fillText(v,sx(v),320-20);ctx.textAlign='right';ctx.fillText(v,pl-5,sy(v)+4);});
    ctx.textAlign='center';ctx.fillText('실제 강도 (MPa)',pl+gw/2,320-3);ctx.save();ctx.translate(11,pt+gh/2);ctx.rotate(-Math.PI/2);ctx.fillText('예측 강도 (MPa)',0,0);ctx.restore();
    ctx.strokeStyle=C.muted;ctx.setLineDash([4,4]);ctx.beginPath();ctx.moveTo(sx(0),sy(0));ctx.lineTo(sx(mx),sy(mx));ctx.stroke();ctx.setLineDash([]);
    STE.forEach((y,i)=>circ(ctx,sx(y),sy(clamp(p[i],0,mx)),3.2,rgba(col,.7),null));
    const rr=r2(STE,p),mm=mae(STE,p);ctx.fillStyle=col;ctx.font=font(12,700,1);ctx.textAlign='left';ctx.fillText('R² '+fmtR(rr)+' · MAE '+fmtM(mm),pl+8,pt+16);
    $('regInfo').innerHTML=(cur==='man'?'매뉴얼 경험식: 물-결합재비와 재령 두 숫자로 교과서 식을 쓴다. 슬래그·플라이 애시·감수제를 모른다.':cur==='1'?'C = 1(기본값): R² '+fmtR(rr)+'로 <b>회귀 합격선 아래</b>다. 벌점이 가벼워 예측이 평균 쪽으로 눌려 있다(높은 강도를 낮게, 낮은 강도를 높게). 분류에서는 기본값으로도 합격했지만 회귀에서는 근무 조건을 꼭 골라야 한다.':cur==='10'?'C = 10: R² '+fmtR(rr)+'. 합격선은 넘지만 교차 검증은 C = 100을 더 좋게 봤다.':'C = 100(교차 검증으로 고름): R² '+fmtR(rr)+', 평균 '+fmtM(mm)+'MPa 빗나간다. 매뉴얼 경험식(MAE '+fmtM(LINE_MAE)+')의 절반 남짓이다.');}
  seg('#regPick button',v=>{cur=v;$('regInfo').textContent='계산 중…';setTimeout(drawReg,15);});drawReg();onRedraw(drawReg);
  renderCost();}
function renderCost(){const m=getSVC(C_CLS),R=getSVR(C_REG);
  $('costTbl').innerHTML='<thead><tr><th>지원자</th><th>외울 숫자</th><th>예측 한 번</th><th>브라우저 학습</th></tr></thead><tbody>'+
    '<tr><td>5-4 로지스틱 회귀</td><td>9개 (가중치 8 + 절편)</td><td>곱하기 8번</td><td>—</td></tr>'+
    '<tr><td>SVC (RBF, C = 10)</td><td>'+(m.sv.length*9+1).toLocaleString()+'개 (서포트 벡터 '+m.sv.length+' × (특징 8 + α) + b)</td><td>커널 '+m.sv.length+'번</td><td>'+Math.max(1,Math.round(m.ms))+'ms</td></tr>'+
    '<tr><td>SVR (RBF, C = 100)</td><td>'+(R.sv.length*9+1).toLocaleString()+'개 (서포트 벡터 '+R.sv.length+' × 9 + b)</td><td>커널 '+R.sv.length+'번</td><td>'+Math.max(1,Math.round(R.ms))+'ms</td></tr></tbody>';}
/* 교차 검증: 접기 번호(data.js)대로 브라우저가 직접 학습 */
(function(){const CS=[0.1,1,10,100],CR=[1,10,100];let running=false;
  function tables(rc,rr){const bc=rc?CS.reduce((a,c)=>rc[c]>rc[a]?c:a,CS[0]):null,br=rr?CR.reduce((a,c)=>rr[c]>rr[a]?c:a,CR[0]):null;
    $('cvTblC').innerHTML='<thead><tr><th>분류 C</th><th>교차 검증 정확도</th><th class="muted">시험 (참고)</th></tr></thead><tbody>'+CS.map(c=>'<tr'+(c===bc?' style="background:var(--ok-soft)"':'')+'><td>'+c+(c===bc?' ✓':'')+'</td><td>'+(rc?'<b>'+pct(rc[c])+'</b>':'—')+'</td><td class="muted">'+(rc?pct(getSVC(c).te):'봉인')+'</td></tr>').join('')+'</tbody>';
    $('cvTblR').innerHTML='<thead><tr><th>회귀 C</th><th>교차 검증 R²</th><th class="muted">시험 R² (참고)</th></tr></thead><tbody>'+CR.map(c=>'<tr'+(c===br?' style="background:var(--ok-soft)"':'')+'><td>'+c+(c===br?' ✓':'')+'</td><td>'+(rr?'<b>'+fmtR(rr[c])+'</b>':'—')+'</td><td class="muted">'+(rr?fmtR(getSVR(c).R):'봉인')+'</td></tr>').join('')+'</tbody>';
    return [bc,br];}
  tables(null,null);
  $('cvGo').addEventListener('click',async()=>{if(running)return;running=true;const btn=$('cvGo'),pg=$('cvProg'),bar=pg.querySelector('i');btn.disabled=true;pg.hidden=false;const tot=CS.length*5+CR.length*5;let done=0;const rc={},rr={};
    function folds(F,k){const tr=[],va=[];F.forEach((f,i)=>(f===k?va:tr).push(i));return [tr,va];}
    for(const c of CS){let s=0;for(let k=0;k<5;k++){const [tr,va]=folds(FOLD_C,k),Xt=tr.map(i=>A[i]),m=SVM.svc(Xt,tr.map(i=>YTR[i]),{C:c,gamma:SVM.gScale(Xt)});s+=ML.acc(va.map(i=>YTR[i]),va.map(i=>m.predict(A[i])));done++;bar.style.width=(done/tot*100)+'%';$('cvInfo').textContent='분류 C = '+c+', '+(k+1)+'번째 묶음 채점 중…';await wait(0);}rc[c]=s/5;tables(rc,null);}
    for(const c of CR){let s=0;for(let k=0;k<5;k++){const [tr,va]=folds(FOLD_R,k),Xt=tr.map(i=>A[i]),m=SVM.svr(Xt,tr.map(i=>STR[i]),{C:c,gamma:SVM.gScale(Xt)});s+=r2(va.map(i=>STR[i]),va.map(i=>m.predict(A[i])));done++;bar.style.width=(done/tot*100)+'%';$('cvInfo').textContent='회귀 C = '+c+', '+(k+1)+'번째 묶음 채점 중…';await wait(0);}rr[c]=s/5;}
    const [bc,br]=tables(rc,rr);pg.hidden=true;btn.textContent='교차 검증 완료';
    $('cvInfo').innerHTML='교차 검증이 고른 근무 조건: <b>분류 C = '+bc+'</b>, <b>회귀 C = '+br+'</b> (A장 Colab 결과와 같다). 회색 "시험" 열은 고른 뒤에 확인용으로만 연다. 분류는 시험지만 보면 C = 100('+pct(getSVC(100).te)+')이 더 좋아 보이지만, 그걸 보고 고르면 시험지를 훈련에 쓴 셈이다. 회귀의 기본값 C = 1은 교차 검증에서도 시험에서도 합격선 아래다.';running=false;});})();

/* ── 면접 결과 발표 ── */
function judge(acc,R){const bs=baseAcc();return {cls:acc>PASSLINE&&acc>bs.acc,reg:R>LINE_R2&&R>BASE_R2};}
function altBox(box,a,atr,R,Rtr,note){const j=judge(a,R),ok=j.cls&&j.reg;
  box.innerHTML=statGrid([[pct(a),'분류 시험 정확도 (훈련 '+pct(atr)+')',j.cls?'a':'s'],['R² '+fmtR(R),'회귀 시험 R² (훈련 '+fmtR(Rtr)+')',j.reg?'a':'s'],[ok?/*i18n:verdict*/'합격':'보류','겸용 판정 (분류 '+(j.cls?/*i18n:verdict*/'합격':'보류')+' · 회귀 '+(j.reg?/*i18n:verdict*/'합격':'보류')+')',ok?'a':'s']],3)+'<p class="foot">'+note+'</p>';}
Interview('iv8',{title:'SVM은 콘크리트 강도 업무(분류 + 회귀)에 합격할까?',
  q:'조건: 봉인된 시험지 258건에서 분류 정확도가 합격선(매뉴얼 씨 '+pct(PASSLINE)+')과 기준선(늘 다수 쪽)을, 회귀 R²가 합격선(매뉴얼 경험식 '+fmtR(LINE_R2)+')과 기준선(늘 평균)을 모두 넘으면 합격. 겸용팀이라 두 과목 다 본다. 근무 조건은 05장 교차 검증으로 고른 C(분류 10, 회귀 100), RBF 커널, γ = "scale", 표준화. 브라우저가 SMO로 직접 푼다.',
  fly:'처음 보는 배합에서도 길이 맞아? 강도 숫자까지 맞혀?',robo:'서포트 벡터 몇 개 외워? 예측 한 번에 계산 몇 번이야?',
  async run(setP){setP(.2);await wait(30);const m=getSVC(C_CLS);setP(.5);await wait(30);const R=getSVR(C_REG);setP(1);await wait(20);
    const bs=baseAcc(),j=judge(m.te,R.R),pass=j.cls&&j.reg,M=ML.confusion(YTE,B.map(m.predict),2);
    return {result:pass?'pass':'hold',flyAns:'분류 '+pct(m.te)+' (훈련 '+pct(m.tr)+') · 회귀 R² '+fmtR(R.R)+' (MAE '+fmtM(R.M)+'MPa)',roboAns:'서포트 벡터 '+m.sv.length+'개 + '+R.sv.length+'개 · 예측 한 번에 커널 '+m.sv.length+'번 · 학습 '+Math.max(1,Math.round(m.ms+R.ms))+'ms',
      flyMood:pass?'arms':'coffee',flyMove:pass?'nod':'shake',roboMood:pass?'wow':'meh',roboMove:pass?'pop':'tilt',asstMood:pass?'spark':'sigh',docMood:pass?'magic':'scale',
      measure:'봉인된 시험지 결과예요. 분류는 SVM <b>'+pct(m.te)+'</b>, 매뉴얼 씨 '+pct(PASSLINE)+', 기준선 '+pct(bs.acc)+'. 강도 미달을 합격이라 한 배합은 '+M[0][1]+'건이에요. 회귀는 SVM <b>R² '+fmtR(R.R)+'</b>(MAE '+fmtM(R.M)+'MPa), 매뉴얼 경험식 R² '+fmtR(LINE_R2)+'(MAE '+fmtM(LINE_MAE)+'), 늘 평균 MAE '+fmtM(BASE_MAE)+'.',
      verdict:pass?'합격입니다. 겸용팀에 배치합니다. 조건이 있어요. 표준화한 데이터만 받고, C와 γ는 훈련 데이터 안의 교차 검증으로 정합니다. 기본값 그대로 회귀를 맡기면 합격선 아래예요.':'보류입니다. 두 과목 중 하나가 합격선을 넘지 못했어요.',
      key:'가장 넓은 길 — 분류 '+pct(PASSLINE)+' → '+pct(m.te)+', 회귀 R² '+fmtR(LINE_R2)+' → '+fmtR(R.R),
      render(box){box.innerHTML='<div class="bars">'+barRows([['SVC 분류',m.te,'var(--blue)',pct(m.te)],['매뉴얼 씨 (분류)',PASSLINE,'var(--c-man)',pct(PASSLINE)],['기준선 (늘 '+PNAME[bs.k]+')',bs.acc,'var(--muted)',pct(bs.acc)]],1)+'</div><div class="bars" style="margin-top:10px">'+barRows([['SVR 회귀',R.R,'var(--blue)','R² '+fmtR(R.R)],['매뉴얼 경험식',LINE_R2,'var(--c-man)','R² '+fmtR(LINE_R2)],['기준선 (늘 평균)',Math.max(0,BASE_R2),'var(--muted)','R² '+fx(BASE_R2,3)]],1)+'</div>';}};},
  alts:[
   {label:'기본 설정 그대로 (C = 1)',async run(box){const m=getSVC(1),R=getSVR(1);altBox(box,m.te,m.tr,R.R,R.trR,'근무 조건을 고르지 않고 기본값 C = 1로 두면 분류는 합격선을 넘지만 회귀는 R² '+fmtR(R.R)+'로 합격선 '+fmtR(LINE_R2)+' 아래다. 강도 숫자는 범위가 넓어(2~83MPa) 벌점이 가벼우면 예측이 평균 쪽으로 눌린다. 같은 지원자라도 근무 조건이 실력을 바꾼다.');}},
   {label:'표준화를 빼면',async run(box){const g=SVM.gScale(Ar),t0=performance.now(),m=SVM.svc(Ar,YTR,{C:C_CLS,gamma:g}),R=SVM.svr(Ar,STR,{C:C_REG,gamma:g}),p=Br.map(R.predict);
     altBox(box,ML.acc(YTE,Br.map(m.predict)),ML.acc(YTR,Ar.map(m.predict)),r2(STE,p),r2(STR,Ar.map(R.predict)),'kg와 일 단위 그대로 넣었다(γ = "scale"이 단위에 맞춰 '+g.toExponential(1)+'로 바뀐다). 분류는 버티지만 회귀가 합격선 바로 아래로 내려온다. 거리를 시멘트·슬래그·골재가 차지하고 물과 감수제가 사라졌기 때문이다(02장). 학습 '+Math.round(performance.now()-t0)+'ms.');}},
   {label:'γ를 크게 (γ = 5)',async run(box){const t0=performance.now(),m=SVM.svc(A,YTR,{C:C_CLS,gamma:5}),R=SVM.svr(A,STR,{C:C_REG,gamma:5});
     altBox(box,ML.acc(YTE,B.map(m.predict)),ML.acc(YTR,A.map(m.predict)),r2(STE,B.map(R.predict)),r2(STR,A.map(R.predict)),'γ를 기본값(0.125)의 40배로 키우면 한 배합이 바로 옆까지만 닿는다. 훈련 점수는 거의 만점인데 시험 점수가 떨어진다. 외우기(과대적합)다. 서포트 벡터 '+m.sv.length+'개(분류) — 기본값의 두 배가 넘는다. 학습 '+Math.round(performance.now()-t0)+'ms.');}}]});

/* 평가 장은 화면에 가까워지면 계산한다 */
(function(){const t=$('m5');let done=false;function go(){if(done)return;done=true;setTimeout(renderEval,10);}
  if('IntersectionObserver' in window){const io=new IntersectionObserver(es=>{if(es.some(e=>e.isIntersecting)){go();io.disconnect();}},{rootMargin:'600px'});io.observe(t);io.observe($('m6'));}else go();
  window.addEventListener('hashchange',go);if(location.hash)go();})();

/* ══ 확인 문제 ══ */
var QUIZ={
q1:[{q:'SVM이 여러 경계 후보 중 하나를 고르는 기준은?',o:[['양쪽 클래스에서 가장 멀리 떨어진 경계(가장 넓은 길)',1,'마진(길의 폭)을 가장 크게 하는 경계다. 새 배합이 조금 흔들려도 반대편으로 덜 넘어간다.'],['훈련 정확도가 가장 높은 경계',0,'훈련 정확도가 같은 경계가 여럿일 때 SVM은 그중 길이 가장 넓은 것을 고른다. 정확도만 보면 경계가 한쪽에 바싹 붙을 수 있다.'],['모든 점까지의 거리 합이 가장 작은 경계',0,'그것은 회귀선을 고르는 방식에 가깝다. SVM은 가장 가까운 점들(서포트 벡터)과의 거리만 본다.']]},
 {q:'서포트 벡터가 아닌 훈련 배합을 모두 지우고 다시 학습하면?',o:[['경계가 그대로다',1,'SVM의 답은 α > 0인 배합, 곧 서포트 벡터에만 기댄다. 01장 그림에서 직접 확인했다.'],['경계가 크게 바뀐다',0,'서포트 벡터를 지울 때 바뀐다. 멀리 떨어진 배합은 답에 영향이 없다.'],['학습이 안 된다',0,'서포트 벡터만으로도 학습이 된다. 바로 그 점이 SVM의 특징이다.']]},
 {q:'커널 트릭에 대한 설명으로 맞는 것은?',o:[['휜 공간의 좌표를 만들지 않고, 두 배합의 닮음(내적)을 K(x, x′)로 바로 계산한다',1,'SVM은 배합끼리의 내적만 쓰므로, 내적을 커널 함수로 바꿔 끼우면 된다. RBF 커널은 무한 차원의 내적과 같다.'],['특징 열을 실제로 수백 개 만들어 표에 붙인다',0,'그렇게 해도 되지만 비싸다. 커널 트릭은 그 일을 하지 않고 같은 결과를 낸다.'],['데이터를 표준화하는 다른 이름이다',0,'표준화는 단위를 맞추는 준비 작업이다. 커널과는 다른 일이다.']]}],
q2:[{q:'SVM(RBF 커널)에 표준화가 사실상 필수인 까닭은?',o:[['커널이 거리로 닮음을 재서, 단위가 큰 특징이 거리를 독차지하기 때문',1,'표준화 전에는 물의 몫이 1.2%, 고성능 감수제는 0.1%뿐이었다(02장). 강도에 중요한 특징이 거리에서 사라진다.'],['표준화하지 않으면 코드가 오류를 낸다',0,'오류 없이 돌아간다. 그래서 더 위험하다. 결과만 조용히 나빠진다.'],['서포트 벡터 수를 줄이려고',0,'서포트 벡터 수는 바뀔 수 있지만 목적이 아니다. 거리의 공정함이 목적이다.']]},
 {q:'이 면접에서 SVR의 타겟(y)은?',o:[['압축 강도(MPa) 그 자체',1,'SVR은 숫자를 맞히는 회귀다. 같은 배합표에서 SVC는 35MPa 합격 이름표를 타겟으로 쓴다.'],['35MPa 합격 여부',0,'그것은 SVC(분류)의 타겟이다.'],['서포트 벡터의 수',0,'그것은 학습 결과로 나오는 값이다.']]}],
q3:[{q:'AI가 준 코드에 grid.fit(X_te, y_te)가 있다. 무엇이 문제인가?',o:[['시험 데이터로 C를 골라 시험 점수를 믿을 수 없게 된다',1,'근무 조건은 훈련 데이터 안의 교차 검증으로 고르고, 시험지는 마지막에 한 번만 연다.'],['GridSearchCV는 SVM에 쓸 수 없다',0,'쓸 수 있다. 다만 훈련 데이터에 fit 해야 한다.'],['문제없다. 더 정확한 C를 찾는다',0,'시험지에 맞춘 C라 처음 보는 데이터에서는 점수가 더 낮게 나온다.']]},
 {q:'r2_score(reg.predict(X_te), s_te)처럼 인자 순서를 바꾸면?',o:[['다른 값이 나온다. R²는 정답의 흩어짐으로 나누므로 순서가 중요하다',1,'r2_score(정답, 예측)이 맞는 순서다. MAE는 순서를 바꿔도 같지만 R²는 다르다.'],['같은 값이 나온다',0,'R²는 대칭이 아니다. 분모에 첫 번째 인자의 분산이 들어간다.'],['오류가 난다',0,'오류 없이 다른 숫자를 낸다. 그래서 찾기 어렵다.']]}],
q4:[{q:'SMO가 α를 한 번에 두 개씩 고치는 까닭은?',o:[['Σ αᵢyᵢ = 0을 지키려면 하나가 오른 만큼 다른 하나가 움직여야 해서',1,'둘을 함께 움직이면 합이 유지되고, 그 둘의 가장 좋은 값이 식 한 줄로 나온다.'],['계산을 빨리 끝내려고 아무 둘이나 고른다',0,'아무 둘이 아니라 "가장 어긋난 두 배합"을 고른다(libsvm의 2차 정보 선택).'],['서포트 벡터가 늘 두 개라서',0,'서포트 벡터는 수십~수백 개다.']]},
 {q:'RBF 커널의 γ를 아주 크게 하면?',o:[['한 배합이 바로 옆까지만 닿아 경계가 구불구불해지고 외우기 쉽다',1,'04장 γ = 50 그림과 면접 결과 "γ를 크게"에서 훈련 점수는 오르고 시험 점수는 떨어졌다.'],['경계가 직선에 가까워진다',0,'그것은 γ가 아주 작을 때다.'],['아무 변화가 없다',0,'γ는 C와 함께 가장 중요한 손잡이다.']]},
 {q:'SVR에서 ε 띠 안에 든 훈련 배합의 α는?',o:[['0이다. 지워도 예측 곡선이 그대로다',1,'띠 안의 오차는 벌하지 않으므로 그 배합은 서포트 벡터가 아니다.'],['C다',0,'α = C는 띠 밖으로 나가 벌점을 받는 배합이다.'],['가장 크다',0,'띠 안의 배합은 답에 영향을 주지 않는다.']]}],
q5:[{q:'교차 검증은 분류 C = 10을 골랐는데, 시험지에서는 C = 100이 더 높았다. 어떻게 해야 하나?',o:[['C = 10을 쓴다. 시험지 점수를 보고 바꾸면 시험지를 훈련에 쓴 셈이다',1,'시험지는 "처음 보는 데이터"의 대역이다. 보고 고른 순간 그 역할을 잃는다.'],['C = 100으로 바꾼다. 점수가 높으니까',0,'그 점수는 더 이상 공정한 시험 점수가 아니다.'],['둘의 평균 C = 55를 쓴다',0,'근거가 없다. 고르기는 교차 검증으로 한다.']]},
 {q:'SVR을 기본값 C = 1로 쓰면 이 데이터에서?',o:[['R²가 회귀 합격선 아래다',1,'C = 1에서 R² 0.672. 벌점이 가벼워 예측이 평균 쪽으로 눌린다. C = 100에서 0.907로 합격한다.'],['분류보다 정확하다',0,'회귀와 분류는 지표가 달라 바로 비교할 수 없다.'],['C = 100과 같다',0,'면접 결과 "조건 바꿔 보기"에서 크게 달랐다.']]},
 {q:'SVC의 f(x) 값이 0.2인 시험 배합에 대한 설명으로 맞는 것은?',o:[['합격 쪽이지만 길 안(|f| < 1)에 들어 있어 덜 확실하다',1,'05장에서 길 안에 든 배합의 정확도는 길 밖보다 훨씬 낮았다.'],['합격 확률이 20%다',0,'f(x)는 확률이 아니라 경계로부터의 (눈금 매긴) 거리다.'],['서포트 벡터다',0,'서포트 벡터는 훈련 배합 가운데 α > 0인 것이다. 시험 배합은 서포트 벡터가 아니다.']]}],
q6:[{q:'SVM이 가장 어울리지 않는 일은?',o:[['배합 수십만 건으로 매일 다시 학습해야 하는 일',1,'커널 SVM의 학습은 배합 수의 제곱 이상으로 늘어난다. LinearSVC, 커널 근사, 트리 앙상블을 먼저 본다.'],['배합 수백~수만 건에서 굽은 경계가 필요한 판정',0,'가장 잘하는 일이다.'],['특징이 많고 샘플이 적은 데이터',0,'길을 넓게 하는 것이 규제라 이런 데이터에서도 잘 버틴다.']]},
 {q:'SVC의 C와 로지스틱 회귀(5-4)의 C는?',o:[['같은 방향이다. 클수록 규제가 약하다',1,'둘 다 "데이터 손실 쪽 무게"다. 5-3 릿지의 alpha와는 반대 방향이다.'],['반대 방향이다',0,'릿지·라쏘의 alpha와 헷갈린 것이다.'],['SVM에는 규제가 없다',0,'길을 넓게(‖w‖를 작게) 하는 것 자체가 규제다.']]}]};

/* ══ 공통 B·C·D·E·F 데이터 ══ */
var PROMPTS=[
 {t:'① 내 공정 데이터로 SVM 코드 받기',use:'내 공장의 품질 데이터로 03장의 프롬프트를 맞출 때',p:MAIN_PROMPT},
 {t:'② AI 코드 검증 요청',use:'받은 코드를 다른 AI(또는 같은 AI의 새 대화)에게 검토받을 때',p:'아래 scikit-learn SVM(SVC·SVR) 코드를 코드 리뷰해 줘. 다음을 하나씩 확인하고, 문제가 있으면 줄 번호와 고친 코드를 보여 줘.\n1) 데이터를 나누기 전에 표준화해서 시험 데이터 정보가 새지 않았나, 표준화가 빠지지 않았나\n2) train_test_split에 stratify와 random_state가 있나\n3) C·gamma·epsilon을 훈련 데이터 안의 교차 검증으로 골랐나(GridSearchCV를 시험 데이터에 fit 하지 않았나)\n4) gamma와 C의 방향을 주석이 맞게 설명하나\n5) 분류에는 SVC, 회귀에는 SVR을 썼나, r2_score 등 지표 함수의 인자가 (정답, 예측) 순서인가\n6) 내 scikit-learn 버전(〔1.x〕)에서 기본값이 바뀐 인자(gamma, LinearSVC의 dual)를 옛 방식으로 쓰지 않았나\n추측으로 답하지 말고, 확실하지 않은 부분은 "공식 문서 확인 필요"라고 표시해 줘.'},
 {t:'③ 큰 데이터로 옮기기 상담',use:'내 데이터가 수십만 건이라 SVC가 너무 느릴 때',p:'콘크리트 품질 데이터가 〔30만 건, 특징 12개〕다. scikit-learn SVC(RBF)로 학습하니 너무 느리다. 다음 세 가지 방법을 같은 데이터에서 비교하는 Colab 코드를 써 줘.\n1) LinearSVC (+ 표준화)\n2) Nystroem 또는 RBFSampler로 RBF 커널을 흉내 낸 특징 + LinearSVC (Pipeline)\n3) HistGradientBoostingClassifier\n각각 학습 시간, 교차 검증 정확도, 시험 정확도를 표로 출력하고, 커널 근사의 n_components를 바꾸면 정확도와 시간이 어떻게 변하는지 설명해 줘. 시험 데이터로 하이퍼파라미터를 고르지 않는다.'},
 {t:'④ 수업용 문제 만들기',use:'이 편으로 수업할 때 확인 문제가 더 필요할 때',p:'SVM 입문 수업용 문제를 만들어 줘. 대상은 〔제조 현장 실무자, 수식에 익숙하지 않음〕이다. 예시는 콘크리트 강도 합격 판정으로 통일한다.\n- 마진·서포트 벡터·C·γ·ε 띠를 각각 묻는 객관식 5문제 (보기 3개, 정답 해설 포함)\n- 오해를 드러내는 문제 2개: "γ가 클수록 경계가 부드럽다", "SVM은 단위와 상관없다"\n- 손으로 풀 수 있는 문제 1개: 직선 경계 f(x) = 2x₁ − x₂ + 1에서 길의 폭 2/‖w‖와, 점 (1, 1)의 y·f(x) (y = +1)\n정답은 맨 끝에 따로 모아 줘.'}];
var GL=[
 ['모델','서포트 벡터 머신','두 클래스 사이에 가장 넓은 길(마진)을 내는 경계를 찾는 모델. 분류는 SVC, 회귀는 SVR.'],
 ['모델','마진','경계와 가장 가까운 훈련 샘플 사이의 거리. 길의 폭(2/‖w‖)으로 그린다. SVM은 이것을 가장 크게 한다.'],
 ['모델','서포트 벡터','길 가장자리나 길 안·반대편에 놓여 경계를 정하는 훈련 샘플. 쌍대 문제에서 α > 0인 샘플이다.'],
 ['모델','초평면','특징 공간을 둘로 가르는 평면(특징 2개면 직선). w·x + b = 0.'],
 ['모델','소프트 마진','겹치는 데이터를 위해 길 침범을 허용하되, 침범한 만큼(슬랙 ξ) 벌점을 C의 무게로 더하는 방식.'],
 ['모델','힌지 손실','max(0, 1 − y·f(x)). 길 밖 제자리에 있으면 0, 가장자리를 넘은 만큼 직선으로 커진다.'],
 ['모델','커널 트릭','특징을 휜 공간으로 옮기지 않고, 두 샘플의 닮음(휜 공간의 내적)을 커널 함수 K(x, x′)로 바로 계산하는 방법.'],
 ['모델','RBF 커널','K(x, x′) = exp(−γ‖x − x′‖²). 가까울수록 1, 멀수록 0. γ가 크면 바로 옆만 닮았다고 본다.'],
 ['모델','SVR','서포트 벡터 회귀. 예측선 둘레에 폭 ±ε의 띠를 두르고, 띠 밖으로 나간 만큼만 벌한다.'],
 ['모델','ε 띠','SVR에서 벌점을 주지 않는 오차 범위. 타겟 단위(여기서는 MPa)다.'],
 ['모델','일대일','다중 분류를 클래스 두 개씩의 이진 분류 여러 개로 나눠 투표하는 방식(OvO). SVC의 기본이다.'],
 ['학습','쌍대 문제','가중치 w 대신 샘플마다 무게 α를 찾도록 바꿔 쓴 같은 문제. 샘플끼리는 내적으로만 만나 커널을 끼울 수 있다.'],
 ['학습','SMO','쌍대 문제를 "가장 어긋난 두 α만 고치기"의 반복으로 푸는 방법(Platt, 1998). libsvm이 쓴다.'],
 ['학습','규제','모델이 너무 복잡해지지 않게 벌점을 두는 일. SVM에서는 길을 넓게(‖w‖를 작게) 하는 것이 규제다.'],
 ['학습','과대적합','훈련 데이터를 외워 처음 보는 데이터에서 성적이 떨어지는 일. γ나 C가 너무 크면 생긴다.'],
 ['학습','교차 검증','훈련 데이터를 몇 묶음으로 나눠 번갈아 채점하며 근무 조건(하이퍼파라미터)을 고르는 방법. 시험지는 쓰지 않는다.'],
 ['데이터','표준화','특징마다 평균을 빼고 표준편차로 나눠 단위를 맞추는 일. 훈련 데이터로만 맞춘다. SVM에는 사실상 필수다.'],
 ['데이터','데이터 누수','시험 데이터의 정보가 학습이나 근무 조건 고르기에 섞여 들어가는 일. 점수가 실제보다 좋게 나온다.'],
 ['평가','정확도','맞힌 샘플의 비율. 틀림의 종류를 구별하지 않는다.'],
 ['평가','혼동행렬','정답(행)과 예측(열)을 짝지어 센 표.'],
 ['평가','결정 계수','R². 1이면 완벽, 0이면 늘 평균을 답하는 것과 같다. 회귀 과목의 판정 지표다.'],
 ['평가','평균 절대 오차','MAE. 예측이 평균 몇 MPa 빗나가는지.'],
 ['평가','기준선','배우지 않고도 낼 수 있는 점수. "늘 다수 쪽", "늘 평균" 등.'],
 ['도구','커널 근사','RBF 커널을 흉내 내는 특징을 몇백 개 만들어 직선 모델에 넣는 방법(Nystroem, RBFSampler). 큰 데이터용.']];
var ANALOGY=[
 ['최대 마진','두 마을 사이에 도로를 낼 때 양쪽 집에서 가장 멀게 한가운데로 낸다',1,'','뇌가 마진을 최대로 하는 경계를 계산한다는 근거는 알려져 있지 않다'],
 ['서포트 벡터','커트라인 근처의 애매한 지원자 몇 명의 서류만 다시 보면 커트라인이 정해진다',1,'','커트라인 근처의 사례를 더 기억한다는 생각과 연결하기도 하지만 SVM과의 대응은 확인되지 않았다'],
 ['소프트 마진 C','규정 위반 몇 건은 눈감아 주되 한 건마다 벌점을 매긴다. C가 크면 깐깐한 감독관',1,'','비유일 뿐이다. 실제 벌점은 침범한 거리(ξ)에 비례한다'],
 ['커널 트릭','종이 위에 섞여 찍힌 점들을, 종이를 휘어 접어 올리면 칼질 한 번으로 가를 수 있다',2,'초파리 버섯체는 투사 뉴런 약 50종의 냄새 신호를 약 2,000개의 케니언 세포로 무작위로 퍼뜨려 차원을 크게 늘린다. 늘린 공간에서는 냄새가 직선 하나로 잘 갈린다','커널은 차원을 실제로 늘리지 않고 닮음만 계산한다. 버섯체는 실제 세포를 늘리고, 학습도 SVM의 최적화가 아니라 도파민에 따른 시냅스 세기 변화로 한다'],
 ['RBF 커널과 γ','가까운 배합일수록 "닮았다"고 보는 범위. γ가 크면 아주 가까운 것만 닮았다고 본다',2,'뉴런의 반응이 특정 자극 근처에서 가장 크고 멀어질수록 줄어드는 종 모양 조율 곡선(장소 세포 등)','종 모양은 닮았지만 실제 조율 곡선은 모양과 폭이 제각각이고, 뇌가 이것을 커널로 합산한다는 근거는 없다'],
 ['ε 띠','도면의 공차: ±5MPa 안이면 맞은 것으로 친다',1,'','공차는 사람이 정하는 규칙이다. SVR의 ε도 사람이 정하는 근무 조건이다'],
 ['SMO','줄 선 사람들을 두 명씩 짝지어 자리를 바꿔 가며 정리한다',1,'','비유일 뿐이다. 실제로는 "가장 어긋난 두 명"을 계산으로 고른다']];
var REFS=[
 'Yeh, I.-C. (1998). Modeling of strength of high-performance concrete using artificial neural networks. <i>Cement and Concrete Research</i>, 28(12), 1797–1808. <a href="https://doi.org/10.1016/S0008-8846(98)00165-3" target="_blank" rel="noopener">doi:10.1016/S0008-8846(98)00165-3</a> — 실기 과제지 원출처',
 'Yeh, I.-C. (2007). Concrete Compressive Strength [Dataset]. UCI Machine Learning Repository. <a href="https://doi.org/10.24432/C5PK67" target="_blank" rel="noopener">doi:10.24432/C5PK67</a> (CC BY 4.0) — 02장 데이터',
 'Boser, B. E., Guyon, I. M., &amp; Vapnik, V. N. (1992). A training algorithm for optimal margin classifiers. <i>Proceedings of the 5th Annual Workshop on Computational Learning Theory (COLT)</i>, 144–152. <a href="https://doi.org/10.1145/130385.130401" target="_blank" rel="noopener">doi:10.1145/130385.130401</a> — 최대 마진과 커널, 01장',
 'Cortes, C., &amp; Vapnik, V. (1995). Support-vector networks. <i>Machine Learning</i>, 20(3), 273–297. <a href="https://doi.org/10.1007/BF00994018" target="_blank" rel="noopener">doi:10.1007/BF00994018</a> — 소프트 마진, 04장',
 'Smola, A. J., &amp; Schölkopf, B. (2004). A tutorial on support vector regression. <i>Statistics and Computing</i>, 14(3), 199–222. <a href="https://doi.org/10.1023/B:STCO.0000035301.49549.88" target="_blank" rel="noopener">doi:10.1023/B:STCO.0000035301.49549.88</a> — SVR과 ε 띠, 04·05장',
 'Platt, J. C. (1998). <i>Sequential minimal optimization: A fast algorithm for training support vector machines</i> (Technical Report MSR-TR-98-14). Microsoft Research. — SMO, 04장',
 'Fan, R.-E., Chen, P.-H., &amp; Lin, C.-J. (2005). Working set selection using second order information for training support vector machines. <i>Journal of Machine Learning Research</i>, 6, 1889–1918. — 두 α 고르기(이 페이지의 계산), 04장',
 'Chang, C.-C., &amp; Lin, C.-J. (2011). LIBSVM: A library for support vector machines. <i>ACM Transactions on Intelligent Systems and Technology</i>, 2(3), 27. <a href="https://doi.org/10.1145/1961189.1961199" target="_blank" rel="noopener">doi:10.1145/1961189.1961199</a> — scikit-learn SVC·SVR의 속, 04장',
 'Hsu, C.-W., Chang, C.-C., &amp; Lin, C.-J. (2003). <i>A practical guide to support vector classification</i>. National Taiwan University. — 표준화와 C·γ 격자 탐색, 02·05장',
 'Rahimi, A., &amp; Recht, B. (2007). Random features for large-scale kernel machines. <i>Advances in Neural Information Processing Systems</i>, 20. — 커널 근사, 06장',
 'Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. <i>Journal of Machine Learning Research</i>, 12, 2825–2830. — A장 코드',
 'scikit-learn 문서. <a href="https://scikit-learn.org/stable/modules/svm.html" target="_blank" rel="noopener">Support Vector Machines</a> — gamma="scale", 일대일 다중 분류, probability, 03·06장',
 'Caron, S. J. C., Ruta, V., Abbott, L. F., &amp; Axel, R. (2013). Random convergence of olfactory inputs in the <i>Drosophila</i> mushroom body. <i>Nature</i>, 497, 113–117. <a href="https://doi.org/10.1038/nature12063" target="_blank" rel="noopener">doi:10.1038/nature12063</a> — D장 커널 비유',
 'Litwin-Kumar, A., Harris, K. D., Axel, R., Sompolinsky, H., &amp; Abbott, L. F. (2017). Optimal degrees of synaptic connectivity. <i>Neuron</i>, 93(5), 1153–1164. <a href="https://doi.org/10.1016/j.neuron.2017.01.030" target="_blank" rel="noopener">doi:10.1016/j.neuron.2017.01.030</a> — D장 커널 비유(차원 확장과 직선 분리)'];
var TIP={who:'머신러닝을 처음 배우는 대학생·직장인·제조 현장 실무자. 5-4 로지스틱 회귀(직선 경계, C)와 5-1 KNN(거리, 표준화)을 먼저 보면 좋다. 수식은 2/‖w‖, max(0, 1 − y·f), exp(−γ‖x − x′‖²) 세 개만 나온다',
 goal:'① 가장 넓은 길·서포트 벡터·소프트 마진을 지원서 1장으로 설명한다 ② 커널 트릭이 "휘지 않고 닮음만 잰다"는 것을 말한다 ③ C·γ·ε의 방향을 그림으로 예측한다 ④ AI가 낸 코드에서 표준화 누락·시험 데이터로 고르기·지표 인자 순서 실수를 찾는다 ⑤ 교차 검증으로 근무 조건을 고르고 두 과목을 판정한다',
 time:'읽기 25분 · 실습(01·04·05장 그림) 25분 · 틀린 곳 찾기 10분 · Colab 15분',
 plans:[['강의형 50분','00장 지원서 1장을 인쇄해 나눠 주고 1·2·5번 칸 읽기 10분 → 01장 C 버튼과 "서포트 벡터만 남기기" 10분 → 커널 마술(장난감) 5분 → 04장 SMO 재생에서 γ = 0.1과 50 비교 10분 → 05장 교차 검증 돌리기 5분 → 면접 결과를 먼저 맞히고 발표, "조건 바꿔 보기" 세 개 10분.'],['실무 OJT 90분','03장 프롬프트로 각자 AI에게 코드를 받는다 20분 → 체크리스트로 서로의 코드를 바꿔 검사 20분 → Colab에서 실행해 A장 결과와 비교 20분 → B장 ③ 프롬프트로 "우리 데이터가 30만 건이면" 대안을 비교하고 발표 30분.'],['1:1 튜터링','01장에서 점 하나를 눌러 α와 y·f(x)를 읽게 하고, 그 배합이 서포트 벡터인지 왜 그런지 말하게 한다. 04장 ε 띠에서 ε을 바꿀 때 서포트 벡터 수를 먼저 예상하게 한 뒤 눌러 본다.']],
 myths:[['SVM은 단위와 상관없다','RBF 커널은 거리로 일한다. 표준화하지 않으면 회귀 R²가 합격선 아래로 떨어졌다'],['γ가 클수록 경계가 부드럽다','반대다. γ가 크면 한 배합이 바로 옆만 닮았다고 봐 경계가 구불구불해진다'],['C가 클수록 규제가 세다','반대다. C는 침범 벌점의 무게라 클수록 규제가 약하다(5-4 로지스틱과 같은 방향)'],['서포트 벡터는 몇 개뿐이다','깔끔하게 갈리는 데이터에서나 그렇다. 겹치는 데이터, 좁은 ε의 SVR에서는 훈련 샘플 대부분이 서포트 벡터일 수 있다'],['커널은 데이터를 고차원으로 실제로 옮긴다','옮기지 않는다. 고차원에서의 내적과 같은 값을 바로 계산할 뿐이다'],['기본값으로 돌려도 충분하다','분류는 기본값 C = 1로도 합격했지만 회귀는 R² 0.672로 보류였다. 교차 검증으로 고른다']],
 tasks:[['과제 1 · 지원서 다시 쓰기','00장 2번 칸의 그림을 01장에서 C = 1일 때 실제 경계식(w, b)과 길의 폭으로 다시 그리고, 서포트 벡터 수를 적는다.'],['과제 2 · 틀린 곳 찾기 만들기','AI에게 일부러 실수가 섞인 SVR 코드를 받고(B장 ②를 거꾸로 활용), 짝에게 찾게 한다. 실수마다 "왜 틀렸나"를 한 줄로 쓴다.'],['과제 3 · C와 γ 지도','A장 코드에 GridSearchCV로 C(0.1~1000)와 γ(0.01~10)의 격자를 돌려 교차 검증 점수 표를 만들고, 외우기·뭉개기 구역을 표시한다. 시험지는 마지막에 한 번만 연다.'],['과제 4 · 큰 데이터 대비','B장 ③ 프롬프트로 LinearSVC·커널 근사·트리 앙상블을 비교하고, "우리 공장 데이터 크기면 누구를 뽑을지" 반 쪽으로 쓴다.']],
 ai:'학생이 AI에게 코드를 받는 것은 막지 말고 과정에 넣는다. 다만 ① 프롬프트를 제출하게 하고 ② 체크리스트 아홉 항목을 직접 표시하게 하고 ③ 실행 결과를 기준선·합격선(분류 76.7%, 회귀 R² 0.735)과 비교한 문장을 쓰게 하면, "받아 적기"가 아니라 "검증하기"를 배운다. SVM은 AI가 표준화를 빼거나 시험 데이터로 C를 고르는 실수가 특히 잦으니, 그 두 줄을 먼저 찾게 한다. 과제에 AI 도움을 받았으면 사용 범위를 밝히게 한다.'};
var CODE={title:'SVM 면접 — 가장 넓은 길부터 겸용 판정까지',sub:'numpy, pandas, scikit-learn, ucimlrepo · CPU 몇 초',file:'svm_interview',
 desc:'이 편의 면접을 그대로 옮겼다. 데이터를 받아 시험지를 봉인하고(1·2번 칸), 합격선을 다시 재고(3번), 특징 2개 직선 SVM으로 길의 폭과 서포트 벡터를 확인하고(4번), γ를 바꿔 보고(5번), 분류 C를 교차 검증으로 골라 시험지를 연다(6·7번). 회귀는 SVR로 같은 일을 하고(8번), 기본값·표준화 빼기·큰 γ를 비교한 뒤(9번) 겸용 판정(10번)까지 간다. 노트북으로 저장하면 칸마다 셀이 나뉜다.',
 cap:'실행 결과 (scikit-learn 1.9, CPU). 브라우저 계산과 같은 결론이 나온다: 교차 검증이 분류 C = 10, 회귀 C = 100을 고르고, 분류 88.0%, 회귀 R² 0.907로 두 과목 모두 합격. 기본값 C = 1이면 회귀 0.672, 표준화를 빼면 0.729로 보류.'};
