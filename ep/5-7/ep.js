var BOARD_LINE='R² 0.735 · 76.7%'; /* 회귀 합격선(매뉴얼 경험식 R²) · 분류 합격선 */
/* ══ 5-7 랜덤 포레스트 · 콘크리트 실기 과제지 ══════════════════════════ */
const ROWS=concRows(),TR=ROWS.filter(r=>!r.test),TE=ROWS.filter(r=>r.test);
const fv=r=>FEAT.map(f=>r[f]);
const A=TR.map(fv),B=TE.map(fv);                 /* 나무는 표준화하지 않는다: 원래 단위 그대로 */
const PTR=TR.map(r=>r.pass),PTE=TE.map(r=>r.pass),YTR=TR.map(r=>r.strength),YTE=TE.map(r=>r.strength),GTR=TR.map(r=>r.grade),GTE=TE.map(r=>r.grade);
const PNAME=['불합격','합격'];
const SHORT={cement:'시멘트',slag:'슬래그',fly_ash:'애시',water:'물',superplasticizer:'감수제',coarse_agg:'자갈',fine_agg:'모래',age:'재령'};

/* RF-ENGINE-START */
/* ── 나무와 숲 엔진 (이 편 전용, 브라우저·node 공용) ──
   CART: 분류는 지니 불순도, 회귀는 제곱 오차(MSE)로 자른다. 문턱은 이웃한 두 값의 가운데(scikit-learn과 같다).
   부트스트랩은 뽑힌 횟수를 무게로 쓴다(scikit-learn과 같은 방식). 끝까지 키운다(잎이 순수하거나 배합이 하나 남을 때까지).
   갈림마다 특징을 무작위 순서로 mtry개 본다(모두 값이 같은 특징뿐이면 더 본다).
   Extra Trees는 특징마다 문턱을 하나만 무작위로 뽑고, 부트스트랩을 하지 않는다 */
const FOREST=(function(){
  function grow(X,y,o,rng){
    const n=X.length,d=X[0].length,K=o.K,w=new Float64Array(n);
    if(o.boot){for(let t=0;t<n;t++)w[Math.floor(rng()*n)]++;}else w.fill(1);
    const ids=[];for(let i=0;i<n;i++)if(w[i]>0)ids.push(i);
    const T={feat:[],thr:[],L:[],R:[],val:[],nw:[],imp:new Float64Array(d),w,depth:0,leaves:0};
    function stats(list){let W=0;if(K){const c=new Float64Array(K);for(const i of list){c[y[i]]+=w[i];W+=w[i];}let g=1;for(let k=0;k<K;k++){const p=c[k]/W;g-=p*p;}return {W,imp:g,val:Array.from(c,v=>v/W)};}
      let s=0,s2=0;for(const i of list){s+=w[i]*y[i];s2+=w[i]*y[i]*y[i];W+=w[i];}const m=s/W;return {W,imp:Math.max(0,s2/W-m*m),val:m};}
    function node(list,depth){const id=T.feat.length,st=stats(list);T.feat.push(-1);T.thr.push(0);T.L.push(-1);T.R.push(-1);T.val.push(st.val);T.nw.push(st.W);
      if(depth>T.depth)T.depth=depth;
      if(list.length<2||st.imp<=1e-12||(o.maxDepth&&depth>=o.maxDepth)){T.leaves++;return id;}
      const order=[...Array(d).keys()];for(let k=d-1;k>0;k--){const r=Math.floor(rng()*(k+1)),t=order[k];order[k]=order[r];order[r]=t;}
      let best=null,visited=0,found=0;
      for(const j of order){if(visited>=o.mtry&&found>0)break;visited++;
        let lo=Infinity,hi=-Infinity;for(const i of list){const v=X[i][j];if(v<lo)lo=v;if(v>hi)hi=v;}if(hi<=lo)continue;found++;
        if(o.extra){let thr=lo+rng()*(hi-lo);if(thr>=hi)thr=lo;const l=[],r=[];for(const i of list)(X[i][j]<=thr?l:r).push(i);
          const sl=stats(l),sr=stats(r),q=(sl.W*sl.imp+sr.W*sr.imp)/st.W;if(!best||q<best.q)best={q,j,thr,l,r};continue;}
        const s=list.slice().sort((a,b)=>X[a][j]-X[b][j]);
        if(K){const cl=new Float64Array(K),ct=new Float64Array(K);let Wl=0;for(const i of s)ct[y[i]]+=w[i];
          for(let t=0;t<s.length-1;t++){const i=s[t];cl[y[i]]+=w[i];Wl+=w[i];const v=X[i][j],v2=X[s[t+1]][j];if(v2<=v)continue;
            const Wr=st.W-Wl;let gl=Wl,gr=Wr;for(let k=0;k<K;k++){gl-=cl[k]*cl[k]/Wl;gr-=(ct[k]-cl[k])*(ct[k]-cl[k])/Wr;}const q=(gl+gr)/st.W;
            if(!best||q<best.q-1e-12)best={q,j,thr:(v+v2)/2,t,s};}}
        else{let St=0;for(const i of s)St+=w[i]*y[i];let Sl=0,Wl=0;
          for(let t=0;t<s.length-1;t++){const i=s[t];Sl+=w[i]*y[i];Wl+=w[i];const v=X[i][j],v2=X[s[t+1]][j];if(v2<=v)continue;
            const Wr=st.W-Wl,q=-(Sl*Sl/Wl+(St-Sl)*(St-Sl)/Wr);if(!best||q<best.q-1e-9)best={q,j,thr:(v+v2)/2,t,s};}}}
      if(!best){T.leaves++;return id;}
      let l,r;if(best.l){l=best.l;r=best.r;}else{l=best.s.slice(0,best.t+1);r=best.s.slice(best.t+1);}
      if(!l.length||!r.length){T.leaves++;return id;}
      const sl=stats(l),sr=stats(r);T.imp[best.j]+=st.W*st.imp-sl.W*sl.imp-sr.W*sr.imp;
      T.feat[id]=best.j;T.thr[id]=best.thr;const a=node(l,depth+1);T.L[id]=a;const b=node(r,depth+1);T.R[id]=b;return id;}
    node(ids,0);return T;}
  function leaf(T,x){let k=0;while(T.feat[k]>=0)k=x[T.feat[k]]<=T.thr[k]?T.L[k]:T.R[k];return T.val[k];}
  /* 숲: 분류는 나무들의 확률을 평균한다(scikit-learn과 같은 "부드러운 투표"), 회귀는 평균 */
  function make(X,y,o){o=Object.assign({mtry:X[0].length,boot:true,extra:false},o);const rng=RNG(o.seed||1),trees=[];
    const F={o,trees,X,y,add(m){for(let t=0;t<(m||1);t++)trees.push(grow(X,y,o,rng));return F;},
      proba(x,upto){const n=Math.min(upto||trees.length,trees.length);if(o.K){const p=new Array(o.K).fill(0);for(let t=0;t<n;t++){const v=leaf(trees[t],x);for(let k=0;k<o.K;k++)p[k]+=v[k]/n;}return p;}
        let s=0;for(let t=0;t<n;t++)s+=leaf(trees[t],x);return s/n;},
      predict(x,upto){const p=F.proba(x,upto);if(!o.K)return p;let m=0;for(let k=1;k<o.K;k++)if(p[k]>p[m])m=k;return m;},
      /* OOB: 그 배합을 뽑지 않은 나무들에게만 묻는다 */
      oob(upto){const n=X.length,nt=Math.min(upto||trees.length,trees.length),acc=[],cnt=new Float64Array(n);for(let i=0;i<n;i++)acc.push(o.K?new Array(o.K).fill(0):0);
        for(let t=0;t<nt;t++){const T=trees[t];for(let i=0;i<n;i++)if(T.w[i]===0){const v=leaf(T,X[i]);cnt[i]++;if(o.K)for(let k=0;k<o.K;k++)acc[i][k]+=v[k];else acc[i]+=v;}}
        const ok=[];for(let i=0;i<n;i++)if(cnt[i]>0)ok.push(i);
        if(o.K){let c=0;ok.forEach(i=>{const p=acc[i];let m=0;for(let k=1;k<o.K;k++)if(p[k]>p[m])m=k;if(m===y[i])c++;});return {score:c/ok.length,n:ok.length};}
        return {score:r2(ok.map(i=>y[i]),ok.map(i=>acc[i]/cnt[i])),n:ok.length};},
      importance(upto){const d=X[0].length,nt=Math.min(upto||trees.length,trees.length),s=new Array(d).fill(0);for(let t=0;t<nt;t++){const T=trees[t];let z=0;for(let j=0;j<d;j++)z+=T.imp[j];if(z>0)for(let j=0;j<d;j++)s[j]+=T.imp[j]/z;}const z=s.reduce((a,b)=>a+b,0)||1;return s.map(v=>v/z);},
      nodes(upto){let s=0;for(let t=0;t<Math.min(upto||trees.length,trees.length);t++)s+=trees[t].feat.length;return s;}};
    return F;}
  return {grow,leaf,make};})();
function r2(y,p){let m=0;for(const v of y)m+=v;m/=y.length;let a=0,b=0;for(let i=0;i<y.length;i++){a+=(y[i]-p[i])*(y[i]-p[i]);b+=(y[i]-m)*(y[i]-m);}return 1-a/b;}
function mae(y,p){let s=0;for(let i=0;i<y.length;i++)s+=Math.abs(y[i]-p[i]);return s/y.length;}
/* RF-ENGINE-END */

/* 두 합격선과 기준선 */
const manualMPa=(wb,t)=>96.5/Math.pow(7,wb)*t/(4+0.85*t); /* 매뉴얼 씨의 경험식: 에이브럼스 식(물-결합재비) × ACI 209R 재령 보정 */
const MAN=TE.map(r=>manualMPa(r.wb,r.age)),LINE_R2=r2(YTE,MAN),LINE_MAE=mae(YTE,MAN);
const BASE=ML.mean(YTR),BASE_R2=r2(YTE,YTE.map(()=>BASE)),BASE_MAE=mae(YTE,YTE.map(()=>BASE));
const BASE_ACC=(function(){const p=PTR.reduce((a,v)=>a+v,0)/PTR.length,maj=p>=.5?1:0;return {k:maj,acc:PTE.filter(v=>v===maj).length/PTE.length};})();
const fmtR=v=>fx(v,3),fmtM=v=>v.toFixed(2);
const accOf=(F,X,y,n)=>{let c=0;for(let i=0;i<X.length;i++)if(F.predict(X[i],n)===y[i])c++;return c/X.length;};

/* ── 숲 창고: 같은 난수(seed)로 심으면 같은 숲이 나온다. 04장 재생 · 05장 · 면접 · 06장이 같은 숲을 함께 쓴다 ──
   seed는 이 편의 기본 숲을 고정하려고 고른 값이다(Colab의 random_state=0과 같은 역할) */
const KIND={cls:{X:A,y:PTR,o:{K:2,mtry:2,seed:4},name:'과제 A · 35MPa 분류'},reg:{X:A,y:YTR,o:{K:0,mtry:8,seed:1},name:'과제 B · 강도 회귀'},
  etc:{X:A,y:PTR,o:{K:2,mtry:2,seed:1,boot:false,extra:true}},etr:{X:A,y:YTR,o:{K:0,mtry:8,seed:1,boot:false,extra:true}},g3:{X:A,y:GTR,o:{K:3,mtry:2,seed:1}}};
const POOL={};
function pool(k){if(!POOL[k]){const c=KIND[k];POOL[k]={F:FOREST.make(c.X,c.y,c.o),ms:0};}return POOL[k];}
function growSync(k,n){const P=pool(k);if(P.F.trees.length<n){const t0=performance.now();P.F.add(n-P.F.trees.length);P.ms+=performance.now()-t0;}return P.F;}
async function ensure(k,n,setP){const P=pool(k);while(P.F.trees.length<n){growSync(k,Math.min(n,P.F.trees.length+4));if(setP)setP(P.F.trees.length/n);await wait(0);}return P.F;}

/* ── 히어로 ── */
heroRoom($('heroCv'),{sym:'RF',no:'APP-07',name:'랜덤\n포레스트',role:'지원자 · 겸용팀',banner:'모델 채용 위원회 · 5-7 면접',doc:'front',asst:'board',fly:'clip',robo:'calc',loop:10,
  beats:[{at:0.4,who:'fly',mood:'point',move:'shake',say:'한 그루는 흔들려'},{at:2.2,who:'robo',mood:'stack',move:'pop',say:'나무 100그루 값이면?'},
    {at:4.2,who:'doc',mood:'magic',move:'pop',say:'서로 다르게 틀리면요'},{at:6.2,who:'asst',mood:'spark',move:'jump',say:'표를 모아 볼게요'},{at:8.0,who:'fly',mood:'arms',move:'stand',say:'모으니 버티네'}],
  motif(ctx,t,W,H,C){const base=226,xs=[276,302,328,354],votes=[1,1,0,1];
    xs.forEach((x,i)=>{const sw=REDUCE?0:Math.sin(t*1.8+i*1.3)*2.5;ctx.fillStyle=C.c_app;ctx.beginPath();ctx.moveTo(x+sw,base-44);ctx.lineTo(x-12,base-12);ctx.lineTo(x+12,base-12);ctx.closePath();ctx.fill();
      ctx.beginPath();ctx.moveTo(x+sw*0.6,base-30);ctx.lineTo(x-14,base-4);ctx.lineTo(x+14,base-4);ctx.closePath();ctx.fill();ctx.fillStyle=C.muted;ctx.fillRect(x-2,base-4,4,8);
      const on=REDUCE||((t*0.8+i*0.37)%2)<1.4,v=votes[(i+Math.floor(t/2.5))%4];if(on){ctx.fillStyle=v?C.blue:C.text;ctx.font=font(10.5,700);ctx.textAlign='center';ctx.fillText(v?'합격':'불합격',x,base-50);}});
    ctx.fillStyle=C.muted;ctx.font=font(11,500,1);ctx.textAlign='center';ctx.fillText('다수결 → 합격',315,base+20);}});

/* ── 01 첫머리 장면 ── */
comic('comicIntro',[
 {bg:'party',cast:[{who:'fly',mood:'point',x:30,move:'shake'},{who:'asst',mood:'shock',x:74,move:'tilt'}],say:{who:'fly',txt:'이 나무, 훈련 배합은 다 맞히는데 처음 보는 배합엔 자꾸 틀려!'},prop:'나무 1그루',cap:'① 한 그루의 실수 — 끝까지 키운 나무는 훈련 데이터를 외우고, 데이터가 조금만 바뀌어도 질문이 통째로 바뀐다.'},
 {bg:'party',cast:[{who:'doc',mood:'magic',x:32,move:'pop',h:104},{who:'robo',mood:'count',x:76,move:'bob'}],say:{who:'doc',txt:'과제지도 질문 항목도 무작위로 나눠 주면 나무마다 다른 곳에서 틀려요.'},prop:'부트스트랩 + 특징 √8',cap:'② 무작위로 다르게 — 나무마다 다시 뽑은 배합, 갈림마다 고른 특징 몇 개.'},
 {bg:'party',cast:[{who:'robo',mood:'stack',x:28,move:'jump'},{who:'fly',mood:'arms',x:72,move:'nod'}],say:{who:'robo',txt:'백 그루가 투표하니 실수가 지워지네. 월급도 백 배지만!'},prop:'다수결',cap:'③ 다수결 — 서로 다르게 틀린 표는 흩어지고, 맞는 쪽 표가 남는다.'}]);

/* ── 01 직접 만져 보기: 특징 2개(물-결합재비 · log 재령)로 심은 시연용 숲 ── */
(function(){const cv=$('forestCv'),H=300,F2=r=>[r.wb,Math.log10(r.age)],X2=TR.map(F2),GX=110,GY=64;
  const X0=0.21,X1=0.93,Y0=-0.08,Y1=Math.log10(365)+0.08;
  const st={seed:11,n:1,pts:true,F:null,grid:[],prev:null,busy:false,draws:0};
  const cellX=gx=>X0+(gx+.5)/GX*(X1-X0),cellY=gy=>Y1-(gy+.5)/GY*(Y1-Y0);
  /* 배합 점 근처(가로·세로 2칸 안)의 칸: 빈 땅이 아니라 실제 배합이 있는 곳에서 판정이 얼마나 바뀌는지 잰다 */
  const NEAR=new Uint8Array(GX*GY);let NN=0;X2.forEach(p=>{const gx=Math.floor((p[0]-X0)/(X1-X0)*GX),gy=Math.floor((Y1-p[1])/(Y1-Y0)*GY);for(let a=-2;a<=2;a++)for(let b=-2;b<=2;b++){const x=gx+a,y=gy+b;if(x>=0&&x<GX&&y>=0&&y<GY)NEAR[y*GX+x]=1;}});NEAR.forEach(v=>NN+=v);
  async function plant(){st.busy=true;$('forestInfo').textContent='숲을 심는 중…';const F=FOREST.make(X2,PTR,{K:2,mtry:1,seed:st.seed}),g=[];
    for(let t=0;t<100;t+=10){F.add(10);for(let k=t;k<t+10;k++){const a=new Float32Array(GX*GY),T=F.trees[k];for(let gy=0;gy<GY;gy++)for(let gx=0;gx<GX;gx++)a[gy*GX+gx]=FOREST.leaf(T,[cellX(gx),cellY(gy)])[1];g.push(a);}await wait(0);}
    const old={t1:st.grid[0],f:st.grid.length?vote(100):null};st.F=F;st.grid=g;st.busy=false;
    if(old.t1){const nf=vote(100);let c1=0,cf=0;for(let i=0;i<GX*GY;i++){if(!NEAR[i])continue;if((old.t1[i]>=.5)!==(g[0][i]>=.5))c1++;if((old.f[i]>=.5)!==(nf[i]>=.5))cf++;}st.prev={c1:c1/NN,cf:cf/NN};}
    draw();minis();}
  function vote(n){const s=new Float32Array(GX*GY);for(let t=0;t<n;t++){const a=st.grid[t];for(let i=0;i<s.length;i++)s[i]+=a[i];}for(let i=0;i<s.length;i++)s[i]/=n;return s;}
  function paint(ctx,s,pl,pt,gw,gh,C,soft){const cw=gw/GX,ch=gh/GY;for(let gy=0;gy<GY;gy++)for(let gx=0;gx<GX;gx++){const q=s[gy*GX+gx];
      ctx.fillStyle=q>=.5?rgba(C.blue,soft?.10+.30*(q-.5)*2:.32):rgba(C.text,soft?.04+.14*(.5-q)*2:.14);ctx.fillRect(pl+gx*cw,pt+gy*ch,cw+.6,ch+.6);}}
  function draw(){const W=fit(cv,H),ctx=cv.getContext('2d'),C=cols(),pl=46,pr=10,pt=10,pb=38,gw=W-pl-pr,gh=H-pt-pb;const sx=x=>pl+(x-X0)/(X1-X0)*gw,sy=y=>pt+(1-(y-Y0)/(Y1-Y0))*gh;
    ctx.clearRect(0,0,W,H);if(!st.grid.length)return;const s=vote(st.n);paint(ctx,s,pl,pt,gw,gh,C,true);
    ctx.strokeStyle=C.rule;ctx.lineWidth=1;ctx.strokeRect(pl,pt,gw,gh);ctx.fillStyle=C.muted;ctx.font=font(12);
    for(let x=0.3;x<=0.91;x+=0.1){ctx.textAlign='center';ctx.fillText(x.toFixed(1),sx(x),H-22);}
    [1,3,7,28,90,365].forEach(d=>{ctx.textAlign='right';ctx.fillText(d,pl-5,sy(Math.log10(d))+4);});
    ctx.textAlign='center';ctx.fillText('물-결합재비',pl+gw/2,H-4);ctx.save();ctx.translate(11,pt+gh/2);ctx.rotate(-Math.PI/2);ctx.fillText('재령(일, 로그 눈금)',0,0);ctx.restore();
    if(st.pts){const rg=RNG(3);X2.forEach((p,i)=>{const j=(rg()-.5)*0.03;circ(ctx,sx(p[0]),sy(p[1]+j),2.4,rgba(PTR[i]?C.blue:C.text,.7),null);});}
    ctx.font=font(12,700);ctx.textAlign='left';ctx.fillStyle=rgba(C.panel,.88);ctx.fillRect(pl+6,pt+6,196,40);ctx.fillStyle=C.blue;ctx.fillText('■ 합격으로 판정 (진할수록 몰표)',pl+12,pt+21);ctx.fillStyle=C.muted;ctx.fillText('■ 불합격으로 판정',pl+12,pt+38);
    info(s);}
  function info(s){let c=0;X2.forEach((p,i)=>{if((st.F.predict(p,st.n))===PTR[i])c++;});let split=0;for(let i=0;i<s.length;i++)if(s[i]>0.2&&s[i]<0.8)split++;
    let h='나무 <b>'+st.n+'</b>그루의 투표 · 훈련 772건 정확도 <b>'+pct(c/772)+'</b>'+(st.n>1?' · 표가 20~80%로 갈린 칸 '+pct(split/s.length,0)+' (흐린 칸)':' · 한 그루는 늘 0 아니면 1이라 계단 모양 경계가 그대로 보인다')+'.';
    if(st.prev)h+='<br>방금 다시 뽑기: 배합이 있는 곳 근처의 칸 가운데 나무 ① 혼자의 판정은 <b>'+pct(st.prev.c1)+'</b>가 뒤집혔고, 100그루 숲의 판정은 <b>'+pct(st.prev.cf)+'</b>만 뒤집혔다. 같은 지원자를 다시 뽑아도 숲은 거의 흔들리지 않는다. 몇 번 더 눌러 보자.';
    else h+='<br>슬라이더를 1 → 100으로 올려 보자. 그다음 "다시 뽑기"를 눌러 나무 한 그루와 숲 중 누가 더 흔들리는지 보자.';
    $('forestInfo').innerHTML=h;}
  function minis(){const host=$('miniTrees');if(!host.children.length)host.innerHTML=[0,1,2].map(k=>'<div><canvas class="fig" id="mini'+k+'" aria-label="숲의 나무 '+(k+1)+'번 혼자 그린 판정 지도"></canvas><div class="foot" id="miniL'+k+'" style="margin-top:4px;font-size:12.5px"></div></div>').join('');
    [0,1,2].forEach(k=>{const c=$('mini'+k),W=fit(c,96),ctx=c.getContext('2d'),C=cols();ctx.clearRect(0,0,W,96);paint(ctx,st.grid[k],0,0,W,96,C,false);ctx.strokeStyle=C.rule;ctx.strokeRect(.5,.5,W-1,95);
      const T=st.F.trees[k],j=T.feat[0],th=T.thr[0];$('miniL'+k).innerHTML='나무 '+'①②③'[k]+' 뿌리 질문: '+(j===0?'물-결합재비 ≤ '+th.toFixed(2):'재령 ≤ '+Math.pow(10,th).toFixed(1)+'일')+' · 잎 '+T.leaves+'개';});}
  $('fN').addEventListener('input',e=>{st.n=+e.target.value;$('fNO').textContent=st.n;draw();});
  $('fRedraw').addEventListener('click',()=>{if(st.busy)return;st.seed+=7;st.draws++;plant();});
  $('fPts').addEventListener('click',e=>{st.pts=!st.pts;e.target.setAttribute('aria-pressed',st.pts);draw();});
  onRedraw(()=>{if(st.grid.length){draw();minis();}});
  let started=false;const go=()=>{if(!started){started=true;plant();}};
  if('IntersectionObserver' in window){const io=new IntersectionObserver(es=>{if(es.some(e=>e.isIntersecting)){go();io.disconnect();}},{rootMargin:'400px'});io.observe(cv);}else go();})();

/* ── 02 실기 과제지 ── */
$('lineR2').textContent='R² '+fmtR(LINE_R2);$('lineMAE').textContent=fmtM(LINE_MAE);$('baseMAE').textContent=fmtM(BASE_MAE);$('heroLine').textContent='R² '+fmtR(LINE_R2)+' · '+pct(PASSLINE);
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
  $('concTbl').innerHTML='<table><thead><tr><th>#</th>'+FEAT.map(f=>'<th><span class="tg x">특징 X</span>'+FEATK[f]+'</th>').join('')+'<th><span class="tg t">타겟 y · 회귀</span>압축 강도<br>(MPa)</th><th><span class="tg t">타겟 y · 분류</span>35MPa 합격</th><th>용도</th></tr></thead><tbody>'+
    ROWS.map(r=>'<tr><td>'+(r.i+1)+'</td>'+FEAT.map(f=>'<td>'+r[f]+'</td>').join('')+'<td class="y">'+r.strength+'</td><td class="y">'+PNAME[r.pass]+'</td><td>'+(r.test?'<span class="pill no">봉인</span>':'훈련')+'</td></tr>').join('')+'</tbody></table>';
  /* 표준화 전후의 뿌리 질문: 특징 하나로 지니가 가장 많이 주는 문턱을 찾는다 */
  const SC=ML.scaler(A),AS=SC.t(A);
  function best1(j,X){const s=[...X.keys()].sort((a,b)=>X[a][j]-X[b][j]),n=s.length,P1=PTR.reduce((a,v)=>a+v,0);let l1=0,bq=-1,bt=0,bn=0;
    for(let t=0;t<n-1;t++){l1+=PTR[s[t]];const v=X[s[t]][j],v2=X[s[t+1]][j];if(v2<=v)continue;const nl=t+1,nr=n-nl,r1=P1-l1,gl=1-(l1/nl)**2-((nl-l1)/nl)**2,gr=1-(r1/nr)**2-((nr-r1)/nr)**2,g0=1-(P1/n)**2-((n-P1)/n)**2,q=g0-(nl*gl+nr*gr)/n;if(q>bq+1e-12){bq=q;bt=(v+v2)/2;bn=nl;}}
    return {thr:bt,n:bn,q:bq};}
  $('rootTbl').innerHTML='<thead><tr><th>특징 하나로 자르면</th><th>문턱 (원래 단위)</th><th>문턱 (표준화)</th><th>왼쪽 배합 수</th><th>지니 감소</th></tr></thead><tbody>'+
    ['age','cement','water','superplasticizer'].map(f=>{const j=FEAT.indexOf(f),a=best1(j,A),b=best1(j,AS);return '<tr><td>'+FEATK[f]+'</td><td>≤ '+(+a.thr.toFixed(2))+' '+FEATU[f]+'</td><td>≤ '+fx(b.thr,3)+'</td><td>'+a.n+' / '+b.n+(a.n===b.n?' <b style="color:var(--ok)">같음</b>':'')+'</td><td>'+a.q.toFixed(3)+'</td></tr>';}).join('')+'</tbody>';})();

/* ── 03 AI 비서 ── */
const MAIN_PROMPT=`너는 머신러닝 수업 조교다. Google Colab에서 바로 실행되는 파이썬 코드를 써 줘.

[환경] Colab 기본 라이브러리(numpy, pandas, scikit-learn)와 ucimlrepo만 쓴다. 첫 줄에 scikit-learn 버전을 출력한다.
[데이터] 〔ucimlrepo의 fetch_ucirepo(id=165)〕 로 콘크리트 압축 강도 데이터를 읽는다.
  - 특징 X: 〔시멘트, 고로 슬래그, 플라이 애시, 물, 고성능 감수제, 굵은 골재, 잔골재, 재령〕 8개 (열 순서대로 이름을 붙인다)
  - 회귀 타겟: 〔압축 강도(MPa)〕 · 분류 타겟: 〔압축 강도가 35MPa 이상이면 1, 아니면 0〕
[규정]
  1) 강도를 〔25, 45〕MPa로 나눈 3등급으로 stratify 해서 train_test_split(test_size=0.25, random_state=42)로 먼저 나눈다.
  2) 나무 모델이므로 표준화는 하지 않는다.
  3) 회귀는 RandomForestRegressor, 분류는 RandomForestClassifier. 둘 다 n_estimators=〔100〕, random_state=〔0〕, oob_score=True. 내 scikit-learn 버전에서 없어진 값(max_features="auto" 등)은 쓰지 않는다.
[출력] 회귀: 시험 R²·MAE·OOB R². 분류: 시험 정확도·OOB 정확도·혼동행렬. 두 숲의 feature_importances_를 큰 순서로. 나무 1·10·100그루일 때의 시험 점수.
[금지] 시험 데이터로 학습하거나 하이퍼파라미터를 고르지 않는다. 훈련 점수를 성과로 보고하지 않는다. 데이터를 지어내지 않는다.
[형식] 셀 단위로 나누고, 각 셀 위에 무엇을 하는지 한국어 주석을 단다. 코드 뒤에 내가 확인해야 할 점 3가지를 적어 줘.`;
$('mainPrompt').textContent=MAIN_PROMPT;$('copyMain').addEventListener('click',()=>copyText(MAIN_PROMPT,'프롬프트를 복사했습니다'));
(function(){const LINES=[
 ['import pandas as pd'],['from ucimlrepo import fetch_ucirepo'],['from sklearn.model_selection import train_test_split'],['from sklearn.ensemble import RandomForestClassifier'],[''],
 ['uci = fetch_ucirepo(id=165)',0,'UCI에서 콘크리트 데이터를 실제로 받았다. 괜찮다.'],
 ['X = uci.data.features',0,'특징 여덟 열이다. 나무 모델이라 표준화하지 않은 것도 괜찮다.'],['y = uci.data.targets.iloc[:, 0]          # 압축 강도(MPa)',0,'강도 숫자를 그대로 가져왔다. 괜찮다.'],
 ['X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)',1,'<b>stratify와 random_state가 없다.</b> 돌릴 때마다 시험지가 바뀌어 점수가 달라지고, 고강도 배합이 한쪽으로 쏠릴 수 있다. 위원회 규정대로 강도 등급으로 <code>stratify</code>하고 <code>random_state=42</code>를 넣는다.'],[''],
 ['reg = RandomForestClassifier(n_estimators=100, random_state=0)   # 강도 예측',2,'<b>회귀에 분류기를 썼다.</b> 강도(79.99MPa 같은 연속값)를 클래스로 보려다 "Unknown label type: continuous" 오류가 난다. 강도를 맞히는 일은 <code>RandomForestRegressor</code>다.'],
 ['reg.fit(X_train, y_train)',2,'<b>회귀에 분류기를 썼다.</b> 바로 윗줄의 분류기로 강도 숫자를 학습하려는 줄이라 여기서 오류가 난다.'],[''],
 ['clf = RandomForestClassifier(n_estimators=100, max_features="auto",',3,'<b>없어진 값.</b> <code>max_features="auto"</code>는 scikit-learn 1.1에서 예고된 뒤 1.3에서 없어졌다. 지금은 오류가 난다. 분류 기본값 <code>"sqrt"</code>를 쓰거나 아예 지운다.'],
 ['                             bootstrap=False, oob_score=True)',4,'<b>부트스트랩 없이 OOB를 달라고 했다.</b> 다시 뽑지 않으면 모든 나무가 모든 배합을 보니 "가방 밖" 배합이 없다. scikit-learn은 "Out of bag estimation only available if bootstrap=True" 오류를 낸다. <code>bootstrap=True</code>(기본값)로 둔다.'],
 ['clf.fit(X_train, y_train >= 35)',0,'강도를 35MPa 문턱에 대어 합격 이름표를 만들어 학습했다. 괜찮다.'],
 ['print("OOB:", clf.oob_score_)',0,'위의 bootstrap 문제를 고치면 이 줄은 가방 밖 정확도를 잘 출력한다.'],
 ['print("정확도:", clf.score(X_train, y_train >= 35))',5,'<b>훈련 데이터로 채점했다.</b> 끝까지 키운 나무들의 숲은 훈련 배합을 거의 외워 99.9%가 나온다. 실력이 아니다. <code>clf.score(X_test, y_test >= 35)</code>로 시험지를 채점하고, 시험지 없이 보고 싶으면 OOB를 본다.'],
 ['print(sorted(zip(clf.feature_importances_, X.columns), reverse=True))',0,'특징 중요도를 큰 순서로 늘어놓았다. 괜찮다. 다만 이 숫자를 원인의 크기로 읽지 않는다(06장).']];
  const host=$('bugCode'),found=new Set();
  host.innerHTML=LINES.map((l,i)=>'<div class="ln" role="listitem" tabindex="0" data-i="'+i+'"><span class="no">'+(i+1)+'</span><span>'+hl(l[0]||' ')+'</span></div>').join('');
  function score(){$('bugScore').textContent=found.size+' / 5';$('bugScore').className=found.size===5?'pill ok':'pill';}
  function pick(el){const i=+el.getAttribute('data-i'),l=LINES[i];if(!l[0])return;
    if(l[1]){found.add(l[1]);host.querySelectorAll('.ln').forEach(x=>{if(LINES[+x.getAttribute('data-i')][1]===l[1])x.classList.add('hit');});$('bugMsg').innerHTML=(i+1)+'번 줄: '+l[2]+(found.size===5?'<br><b>다섯 군데를 모두 찾았다.</b> 고친 코드는 A장의 Colab 코드와 같은 일을 한다.':'');}
    else{el.classList.add('miss');$('bugMsg').innerHTML=(i+1)+'번 줄: '+(l[2]||'이 줄은 괜찮다.');}score();}
  host.addEventListener('click',e=>{const el=e.target.closest('.ln');if(el)pick(el);});
  host.addEventListener('keydown',e=>{if((e.key==='Enter'||e.key===' ')&&e.target.classList.contains('ln')){e.preventDefault();pick(e.target);}});
  $('bugReveal').addEventListener('click',()=>{LINES.forEach((l,i)=>{if(l[1]){found.add(l[1]);host.querySelector('[data-i="'+i+'"]').classList.add('hit','reveal');}});score();$('bugMsg').innerHTML='빨간 줄 다섯 군데: ① 9번 나누기 규정 ② 11~12번 회귀에 분류기 ③ 14번 없어진 값 "auto" ④ 15번 부트스트랩 없이 OOB ⑤ 18번 훈련 데이터로 채점.';});
  $('bugReset').addEventListener('click',()=>{found.clear();host.querySelectorAll('.ln').forEach(x=>x.classList.remove('hit','miss','reveal'));score();$('bugMsg').textContent='줄을 하나 눌러 보세요.';});score();})();
(function(){const CK=[['데이터를 실제로 읽었나','AI가 예제 데이터를 지어내거나 다른 데이터로 바꾸지 않았는지, 행이 1,030개인지 본다.'],['회귀에는 Regressor, 분류에는 Classifier를 썼나','강도(MPa)를 맞히는 줄에 Classifier가 보이면 틀렸다. 이름이 비슷해 가장 흔한 실수다.'],['나누기 규정(stratify, random_state=42)을 지켰나','등급 비율이 시험지에도 유지되고, 시험지가 매번 같다.'],['숲에도 random_state를 고정했나','숲은 무작위로 심는다. 고정하지 않으면 돌릴 때마다 점수가 조금씩 다르다(05장 다시 뽑기).'],['인자가 내 scikit-learn 버전에 있나','max_features="auto"는 1.3에서 없어졌다. print(sklearn.__version__)과 경고를 읽는다.'],['회귀 숲의 max_features 기본값을 알고 썼나','RandomForestRegressor 기본값 1.0은 특징 모두 = 배깅이다. 일부러 고른 값인지 확인한다.'],['OOB를 쓰려면 bootstrap=True인가','부트스트랩을 끄면 가방 밖 배합이 없어 OOB를 낼 수 없다.'],['훈련 점수를 성과로 보고하지 않았나','끝까지 키운 숲의 훈련 정확도는 99.9% 근처다. 시험지 점수나 OOB 점수를 본다.'],['결과를 두 합격선·기준선과 비교하고, 중요도를 원인으로 읽지 않았나','분류 76.7%, 회귀 R² 0.735, 늘 다수·늘 평균 기준선. 특징 중요도는 "많이 물어본 것"이다.']];
  const ul=$('checks');ul.innerHTML=CK.map((c,i)=>'<li><input type="checkbox" id="ck'+i+'"><label for="ck'+i+'"><b>'+c[0]+'</b><small>'+c[1]+'</small></label></li>').join('');
  function upd(){const n=ul.querySelectorAll('input:checked').length;ul.querySelectorAll('li').forEach(li=>li.classList.toggle('done',li.querySelector('input').checked));$('checkScore').textContent='확인 '+n+' / '+CK.length+(n===CK.length?' — 이제 돌려 본다. 결과도 확인 대상이다.':'');}
  ul.addEventListener('change',upd);upd();})();

/* ── 04 수습 기간: 한 그루씩 심으며 OOB 점수 재기 ── */
const GROW=(function(){const tc=$('treeCv'),bc=$('bootCv'),oc=$('oobCv');
  const st={kind:'cls',shown:0,play:false,trk:{}};
  function trk(){const k=st.kind;if(!st.trk[k]){const n=A.length,c=KIND[k].o.K;st.trk[k]={done:0,acc:Array.from({length:n},()=>c?[0,0]:0),cnt:new Float64Array(n),curve:[],roots:new Array(8).fill(0)};}return st.trk[k];}
  /* k번째 나무까지 OOB 누적: 나무를 하나씩 더할 때 그 나무가 안 본 배합에만 표를 더한다 */
  function upto(n){const k=st.kind,F=growSync(k,n),T0=trk(),cls=!!KIND[k].o.K,y=KIND[k].y;
    while(T0.done<n){const T=F.trees[T0.done];for(let i=0;i<A.length;i++)if(T.w[i]===0){const v=FOREST.leaf(T,A[i]);T0.cnt[i]++;if(cls){T0.acc[i][0]+=v[0];T0.acc[i][1]+=v[1];}else T0.acc[i]+=v;}
      T0.roots[T.feat[0]]++;let s;const ok=[];for(let i=0;i<A.length;i++)if(T0.cnt[i]>0)ok.push(i);
      if(cls){let c=0;ok.forEach(i=>{if((T0.acc[i][1]>T0.acc[i][0]?1:0)===y[i])c++;});s=c/ok.length;}else s=r2(ok.map(i=>y[i]),ok.map(i=>T0.acc[i]/T0.cnt[i]));
      T0.curve.push({s,n:ok.length});T0.done++;}}
  const thrTxt=(j,t)=>{const f=FEAT[j];return SHORT[f]+' ≤ '+(f==='age'?(+t.toFixed(1))+'일':f==='superplasticizer'?t.toFixed(1):Math.round(t*10)/10);};
  function drawTree(){const W=fit(tc,196),ctx=tc.getContext('2d'),C=cols();ctx.clearRect(0,0,W,196);
    if(!st.shown){ctx.fillStyle=C.muted;ctx.font=font(13);ctx.textAlign='center';ctx.fillText('▶ 재생이나 "한 그루 더"를 누르면 첫 나무를 심는다',W/2,98);return;}
    const T=pool(st.kind).F.trees[st.shown-1],cls=!!KIND[st.kind].o.K;
    function box(id,x,y,w,lv){const leaf=T.feat[id]<0;let txt,col=C.blue;
      if(leaf){const v=T.val[id];if(cls){txt=(v[1]>=.5?'합격':'불합격')+' '+Math.round(Math.max(v[0],v[1])*100)+'%';col=v[1]>=.5?C.blue:C.text;}else{txt=v.toFixed(1)+'MPa';col=C.text;}}
      else txt=thrTxt(T.feat[id],T.thr[id]);
      ctx.font=font(lv===2?10.5:12,700);const tw=Math.min(w,ctx.measureText(txt).width+12);ctx.fillStyle=leaf?rgba(col,.12):rgba(C.blue,.10);ctx.strokeStyle=col;ctx.lineWidth=1.6;rr(ctx,x-tw/2,y-11,tw,22,5);ctx.fill();ctx.stroke();
      ctx.fillStyle=C.text;ctx.textAlign='center';ctx.fillText(txt,x,y+4,w-4);}
    const lv=[[0,W/2]],ys=[22,82,142];
    ctx.strokeStyle=rgba(C.blue,.6);ctx.lineWidth=1.5;
    function walk(id,x,d,span){if(d>2)return;box(id,x,ys[d],span-6,d);if(T.feat[id]<0||d===2){if(T.feat[id]>=0&&d===2){ctx.fillStyle=C.muted;ctx.font=font(11);ctx.textAlign='center';ctx.fillText('⋮',x,ys[d]+26);}return;}
      const xl=x-span/4,xr=x+span/4;ctx.strokeStyle=rgba(C.blue,.6);ctx.beginPath();ctx.moveTo(x,ys[d]+11);ctx.lineTo(xl,ys[d+1]-11);ctx.moveTo(x,ys[d]+11);ctx.lineTo(xr,ys[d+1]-11);ctx.stroke();
      ctx.fillStyle=C.muted;ctx.font=font(10.5);ctx.textAlign='center';ctx.fillText('예',(x+xl)/2-6,(ys[d]+ys[d+1])/2);ctx.fillText('아니오',(x+xr)/2+12,(ys[d]+ys[d+1])/2);
      walk(T.L[id],xl,d+1,span/2);walk(T.R[id],xr,d+1,span/2);}
    walk(0,W/2,0,W);ctx.fillStyle=C.muted;ctx.font=font(11.5);ctx.textAlign='left';ctx.fillText('나무 #'+st.shown+' · 깊이 '+T.depth+' · 잎 '+T.leaves+'개 · 위쪽 세 층만',4,190);}
  function drawBoot(){const W=fit(bc,92),ctx=bc.getContext('2d'),C=cols(),n=A.length,colsN=Math.ceil(n/7),cw=(W-2)/colsN,ch=9;ctx.clearRect(0,0,W,92);
    if(!st.shown){return;}const T=pool(st.kind).F.trees[st.shown-1];
    for(let i=0;i<n;i++){const x=1+(i%colsN)*cw,y=2+Math.floor(i/colsN)*(ch+1),c=T.w[i];if(c===0){ctx.strokeStyle=rgba(C.stamp,.8);ctx.lineWidth=1;ctx.strokeRect(x+.5,y+.5,Math.max(1,cw-1.2),ch-1);}
      else{ctx.fillStyle=c===1?rgba(C.blue,.35):c===2?rgba(C.blue,.75):C.ink;ctx.fillRect(x,y,Math.max(1,cw-.6),ch);}}
    let u=0,o=0;for(let i=0;i<n;i++){if(T.w[i]>0)u++;else o++;}ctx.fillStyle=C.muted;ctx.font=font(11.5);ctx.textAlign='left';ctx.fillText('부트스트랩: 뽑힘 '+u+'건('+pct(u/n,0)+') · 가방 밖 '+o+'건(빨간 테두리)',2,86);}
  function drawOob(){const W=fit(oc,226),ctx=oc.getContext('2d'),C=cols(),pl=44,pr=10,pt=12,pb=32,gw=W-pl-pr,gh=226-pt-pb,cls=!!KIND[st.kind].o.K,y0=cls?0.75:0.6,y1=cls?0.95:1.0;
    const sx=k=>pl+(k-1)/99*gw,sy=v=>pt+(1-(clamp(v,y0,y1)-y0)/(y1-y0))*gh;ctx.clearRect(0,0,W,226);ctx.strokeStyle=C.rule;ctx.lineWidth=1;ctx.strokeRect(pl,pt,gw,gh);
    ctx.fillStyle=C.muted;ctx.font=font(12);ctx.textAlign='right';for(let k=0;k<=4;k++){const v=y0+(y1-y0)*k/4;ctx.fillText(cls?pct(v,0):v.toFixed(2),pl-5,sy(v)+4);}
    ctx.textAlign='center';[1,25,50,75,100].forEach(k=>ctx.fillText(k,sx(k),pt+gh+14));ctx.fillText('심은 나무 수',pl+gw/2,226-3);
    const line=cls?PASSLINE:LINE_R2;ctx.setLineDash([5,4]);ctx.strokeStyle=rgba(C.c_man,.9);ctx.beginPath();ctx.moveTo(pl,sy(line));ctx.lineTo(pl+gw,sy(line));ctx.stroke();ctx.setLineDash([]);
    ctx.fillStyle=C.c_man;ctx.textAlign='right';ctx.font=font(11.5);ctx.fillText('참고: 합격선 '+(cls?pct(line):'R² '+fmtR(line)),pl+gw-6,sy(line)+14);
    const cv=trk().curve.slice(0,st.shown);if(cv.length){ctx.strokeStyle=C.blue;ctx.lineWidth=2.5;ctx.beginPath();cv.forEach((p,i)=>{i?ctx.lineTo(sx(i+1),sy(p.s)):ctx.moveTo(sx(i+1),sy(p.s));});ctx.stroke();
      const last=cv[cv.length-1];circ(ctx,sx(cv.length),sy(last.s),5,C.blue,C.panel,2);ctx.fillStyle=C.blue;ctx.font=font(12.5,700,1);ctx.textAlign='left';ctx.fillText('OOB '+(cls?pct(last.s):'R² '+fmtR(last.s)),pl+8,pt+16);}}
  function drawRoots(){const R=trk().roots,mx=Math.max(1,...R),tot=R.reduce((a,b)=>a+b,0);
    $('rootBars').innerHTML='<p class="foot" style="margin:0 0 4px">뿌리 질문으로 뽑힌 특징 ('+tot+'그루)</p>'+FEAT.map((f,j)=>'<div class="row"><span style="text-align:left;font-family:var(--font)">'+FEATK[f]+'</span><div class="tr"><i style="left:0;width:'+(R[j]/mx*100)+'%"></i></div><span>'+R[j]+'</span></div>').join('');}
  function info(){const cls=!!KIND[st.kind].o.K,cv=trk().curve;let h;
    if(!st.shown)h='아직 한 그루도 심지 않았다. '+(cls?'과제 A: 갈림마다 특징 여덟 개 중 두 개만 무작위로 받아 지니 불순도를 가장 많이 줄이는 질문을 고른다.':'과제 B: 회귀 숲의 기본값대로 갈림마다 특징 여덟 개를 모두 보고 제곱 오차를 가장 많이 줄이는 질문을 고른다. 나무끼리의 차이는 부트스트랩에서만 나온다(배깅).');
    else{const T=pool(st.kind).F.trees[st.shown-1],c=cv[st.shown-1],first=cv[0];
      h='<b>'+st.shown+'</b>그루째 · 이 나무의 뿌리 질문은 <b>'+thrTxt(T.feat[0],T.thr[0])+'</b> · OOB '+(cls?'정확도':'R²')+' <b>'+(cls?pct(c.s):fmtR(c.s))+'</b> (가방 밖 표를 받은 배합 '+c.n+'건)';
      if(st.shown===1)h+='<br>한 그루일 때는 772건 중 이 나무가 안 본 약 37%만 채점된다. 끝까지 키운 나무 하나의 실력이다.';
      else if(st.shown<20)h+='<br>나무가 늘수록 표를 받는 배합이 772건 전부로 늘고, 서로 다른 나무의 실수가 지워지며 점수가 오른다. 뿌리 질문이 나무마다 다르다는 것도 보자.';
      else h+='<br>1그루 '+(cls?pct(first.s):fmtR(first.s))+' → '+st.shown+'그루 '+(cls?pct(c.s):fmtR(c.s))+'. 곡선이 평평해지면 더 심어도 점수는 거의 그대로이고 시간과 메모리만 든다. 시험지는 아직 열지 않았다.'+(cls?' 뿌리는 재령·시멘트·물 쪽이 많다. 특징 두 개만 받으니 가끔 약한 특징이 뿌리에 오기도 한다.':' 회귀 숲은 특징을 모두 보기 때문에 뿌리 질문이 거의 늘 같은 특징이다. 나무끼리 더 닮는다는 뜻이다.');}
    $('growInfo').innerHTML=h;$('gCount').textContent=st.shown+'그루';}
  function draw(){upto(st.shown);drawTree();drawBoot();drawOob();drawRoots();info();}
  let raf=0,ph=0;
  function frame(){if(!st.play)return;ph++;const every=st.shown<8?18:st.shown<30?5:2;if(ph%every===0){st.shown++;draw();}if(st.shown>=100){stop();return;}raf=requestAnimationFrame(frame);}
  function stop(){st.play=false;cancelAnimationFrame(raf);$('gPlay').textContent='▶ 재생';}
  $('gPlay').addEventListener('click',()=>{if(st.play){stop();return;}if(st.shown>=100){st.shown=0;delete st.trk[st.kind];}st.play=true;$('gPlay').textContent='❚❚ 멈춤';if(REDUCE){st.shown=100;draw();stop();return;}raf=requestAnimationFrame(frame);});
  $('gStep').addEventListener('click',()=>{stop();if(st.shown<100){st.shown++;draw();}});
  $('gReset').addEventListener('click',()=>{stop();st.shown=0;delete st.trk[st.kind];draw();});
  seg('#growChoice button',v=>{stop();st.kind=v;st.shown=0;delete st.trk[v];draw();});
  draw();onRedraw(()=>{drawTree();drawBoot();drawOob();});
  return st;})();

/* ── 05 성과 평가 ── */
function cmGrid(M,labels,colsFor){const K=labels.length;let h='<div class="cm" style="grid-template-columns:86px repeat('+K+',minmax(48px,1fr));max-width:'+(110+K*90)+'px"><div class="h">정답 ＼ 예측</div>'+labels.map(k=>'<div class="h">'+k+'</div>').join('');
  M.forEach((row,a)=>{h+='<div class="h" style="justify-content:end;text-align:right">'+labels[a]+'</div>'+row.map((v,b)=>a===b?'<div class="c'+(v?' d':'')+'" style="'+(v?'background:'+rgba(colsFor(a),.85):'')+'">'+v+'</div>':'<div class="c'+(v?' x':'')+'">'+(v||'·')+'</div>').join('');});return h+'</div>';}
function barRows(rows,max){return rows.map(r=>'<div class="row"><span style="text-align:left;font-family:var(--font)">'+r[0]+'</span><div class="tr"><i style="left:0;width:'+Math.max(0,Math.min(100,r[1]/max*100))+'%;background:'+(r[2]||'var(--blue)')+'"></i></div><span>'+r[3]+'</span></div>').join('');}
const RES={};
/* 시험지 점수는 한 번만 계산해 둔다 */
async function results(setP){if(RES.done)return RES;
  await ensure('cls',100,p=>setP&&setP(p*.35));await ensure('reg',100,p=>setP&&setP(.35+p*.6));
  const Fc=pool('cls').F,Fr=pool('reg').F;let t0=performance.now();const pc=B.map(x=>Fc.predict(x,100));RES.predMs=performance.now()-t0;t0=performance.now();const pr=B.map(x=>Fr.predict(x,100));RES.predMsR=performance.now()-t0;
  Object.assign(RES,{pc,pr,acc:ML.acc(PTE,pc),tr:accOf(Fc,A,PTR,100),one:accOf(Fc,B,PTE,1),ten:accOf(Fc,B,PTE,10),oobC:Fc.oob(100).score,
    R:r2(YTE,pr),M:mae(YTE,pr),trR:r2(YTR,A.map(x=>Fr.predict(x,100))),p1:B.map(x=>Fr.predict(x,1)),oobR:Fr.oob(100).score});
  RES.R1=r2(YTE,RES.p1);RES.M1=mae(YTE,RES.p1);RES.done=true;if(setP)setP(1);return RES;}
async function renderEval(){const S=await results();
  $('evalCls').innerHTML=statGrid([[pct(S.acc),'시험 정확도 ('+Math.round(S.acc*258)+' / 258건)','a'],[pct(S.tr),'훈련 정확도 (772건) — 외운 몫'],[pct(PASSLINE),'분류 합격선 · 5-0 매뉴얼 씨','m'],[pct(BASE_ACC.acc),'기준선 · 늘 '+rago('"'+PNAME[BASE_ACC.k]+'"')+'만 답할 때']],4)+
    '<p class="foot">시험지를 열기 전 04장의 OOB 정확도는 '+pct(S.oobC)+'였다. 앞쪽 나무만 쓰면 1그루 '+pct(S.one)+' → 10그루 '+pct(S.ten)+' → 100그루 '+pct(S.acc)+'. A장 Colab(random_state=0)은 84.9% → 92.2% → 93.4%, 300그루도 93.4%다. 훈련 정확도가 100%가 아닌 것은 같은 배합인데 합격·불합격이 갈린 줄이 있기 때문이다.</p>';
  $('evalReg').innerHTML=statGrid([['R² '+fmtR(S.R),'시험 R² (MAE '+fmtM(S.M)+'MPa)','a'],['R² '+fmtR(S.trR),'훈련 R² — 외운 몫'],['R² '+fmtR(LINE_R2),'회귀 합격선 · 매뉴얼 경험식 (MAE '+fmtM(LINE_MAE)+')','m'],['R² '+fmtR(BASE_R2),'기준선 · 늘 훈련 평균 '+BASE.toFixed(1)+'MPa (MAE '+fmtM(BASE_MAE)+')']],4)+
    '<p class="foot">OOB R² '+fmtR(S.oobR)+'. 숲의 첫 나무 하나만 쓰면 R² '+fmtR(S.R1)+'(MAE '+fmtM(S.M1)+'MPa)다. A장 Colab은 100그루 R² 0.927(MAE 3.37), 1그루 0.788, 300그루 0.929다. 시험 R²가 OOB R²보다 높게 나온 것은 이 시험지 258건의 사정이다. 둘 중 어느 쪽도 정확한 미래 점수는 아니다.</p>';
  const M=ML.confusion(PTE,S.pc,2);$('cmA').innerHTML=cmGrid(M,PNAME,k=>k?cols().blue:cols().muted);
  $('cmAInfo').innerHTML='강도 미달인데 "합격"이라 한 배합 <b style="color:var(--stamp)">'+M[0][1]+'건</b>, 합격인데 "불합격"이라 한 배합 '+M[1][0]+'건. 5-4 로지스틱 회귀는 미달인데 합격이 13건이었다. 출하 판정에 쓸 때는 <code>predict_proba</code>의 문턱을 0.5보다 올리는 방법(5-4 05장)이 숲에도 그대로 통한다.';
  /* 비교표 */
  $('cmpTbl').innerHTML='<thead><tr><th>지원자 (시험지 258건)</th><th>분류 · 35MPa 정확도</th><th>회귀 · R² (MAE)</th></tr></thead><tbody>'+
    [['매뉴얼 씨 (5-0, 합격선)',pct(PASSLINE),'R² '+fmtR(LINE_R2)+' ('+fmtM(LINE_MAE)+')'],['5-1 KNN (k = 5, 표준화)','85.7%','R² 0.767'],['5-2 선형 회귀','—','R² 0.567'],['5-3 다항 회귀 3차','—','R² 0.849'],['5-4 로지스틱 회귀','86.4%','—'],
     ['결정 트리 1그루, 끝까지 (A장)','92.6%','R² 0.891'],['<b>랜덤 포레스트 100그루 (이 페이지)</b>','<b>'+pct(S.acc)+'</b> (93.4%)','<b>R² '+fmtR(S.R)+'</b> ('+fmtM(S.M)+') (0.927)'],['Extra Trees 100그루 (A장)','91.9%','R² 0.948 (2.60)']]
    .map(r=>'<tr><td>'+r[0]+'</td><td>'+r[1]+'</td><td>'+r[2]+'</td></tr>').join('')+'</tbody>';
  pvInit(S);spreadInit(S);render06(S);}

/* 예측 vs 실제 */
function pvInit(S){const MODELS=[['rf','숲 100그루',S.pr,'blue'],['t1','나무 1그루',S.p1,'amber'],['man','매뉴얼 경험식',MAN,'c-man']];
  $('pvChoice').innerHTML=MODELS.map((m,i)=>'<button data-v="'+m[0]+'" aria-pressed="'+(i===0)+'">'+m[1]+'</button>').join('');let cur='rf';const cv=$('pvCv');
  function draw(){const W=fit(cv,300),ctx=cv.getContext('2d'),C=cols(),pl=44,pt=10,s=Math.min(W-60,250),m=90,sx=v=>pl+v/m*s,sy=v=>pt+s-v/m*s,md=MODELS.find(x=>x[0]===cur),q=md[2],col=C[md[3].replace('-','_')]||C.blue;
    ctx.clearRect(0,0,W,300);ctx.strokeStyle=C.rule;ctx.strokeRect(pl,pt,s,s);ctx.fillStyle=C.muted;ctx.font=font(12);
    [0,30,60,90].forEach(v=>{ctx.textAlign='center';ctx.fillText(v,sx(v),pt+s+15);ctx.textAlign='right';ctx.fillText(v,pl-5,sy(v)+4);});
    ctx.textAlign='center';ctx.fillText('실제 강도 (MPa)',pl+s/2,pt+s+31);ctx.save();ctx.translate(11,pt+s/2);ctx.rotate(-Math.PI/2);ctx.fillText('예측 강도 (MPa)',0,0);ctx.restore();
    ctx.setLineDash([4,4]);ctx.strokeStyle=C.muted;ctx.beginPath();ctx.moveTo(sx(0),sy(0));ctx.lineTo(sx(m),sy(m));ctx.stroke();ctx.strokeStyle=rgba(C.stamp,.6);ctx.beginPath();ctx.moveTo(sx(35),pt);ctx.lineTo(sx(35),pt+s);ctx.moveTo(pl,sy(35));ctx.lineTo(pl+s,sy(35));ctx.stroke();ctx.setLineDash([]);
    YTE.forEach((y,i)=>circ(ctx,sx(y),sy(Math.min(m,q[i])),3.2,rgba(col,.7),null));
    ctx.fillStyle=col;ctx.font=font(12,700,1);ctx.textAlign='left';ctx.fillText('R² '+fmtR(r2(YTE,q))+' · MAE '+fmtM(mae(YTE,q)),pl+8,pt+16);
    const hi=YTE.map((y,i)=>[y,q[i]]).filter(v=>v[0]>=70),under=hi.filter(v=>v[1]<v[0]).length;
    $('pvInfo').innerHTML=(cur==='man'?'매뉴얼 경험식은 학습 없이 교과서 상수만으로 대각선 근처에 모인다. 회귀 합격선이다.':cur==='t1'?'숲의 첫 나무 하나다. 잎마다 배합 한두 건의 강도를 그대로 답하니 대각선 근처에 있다가도 크게 빗나가는 점이 섞인다.':'100그루의 평균이라 한 그루보다 대각선에 바짝 붙는다.')+
      ' 실제 70MPa 이상인 시험 배합 '+hi.length+'건 중 '+under+'건을 낮게 예측했다. 나무의 잎은 훈련 배합 강도의 평균이라 훈련 데이터의 최고값(82.6MPa)을 넘는 답을 낼 수 없고, 평균을 내는 숲은 양 끝에서 안쪽으로 끌려간다.';}
  seg('#pvChoice button',v=>{cur=v;draw();});draw();onRedraw(draw);}

/* 흔들림: 다시 뽑기 */
function spreadInit(S){const D=[{a1:S.one,a10:S.ten,a100:S.acc}],cv=$('spreadCv');let seed=1000,busy=false;
  function draw(){const W=fit(cv,250),ctx=cv.getContext('2d'),C=cols(),pl=50,pr=12,pt=12,pb=30,gw=W-pl-pr,gh=250-pt-pb,y0=0.74,y1=0.96,sy=v=>pt+(1-(clamp(v,y0,y1)-y0)/(y1-y0))*gh,cx=k=>pl+gw*(k+.5)/3;
    ctx.clearRect(0,0,W,250);ctx.strokeStyle=C.rule;ctx.strokeRect(pl,pt,gw,gh);ctx.fillStyle=C.muted;ctx.font=font(12);ctx.textAlign='right';[0.75,0.8,0.85,0.9,0.95].forEach(v=>{ctx.fillText(pct(v,0),pl-5,sy(v)+4);ctx.strokeStyle=rgba(C.rule,.6);ctx.beginPath();ctx.moveTo(pl,sy(v));ctx.lineTo(pl+gw,sy(v));ctx.stroke();});
    ctx.setLineDash([5,4]);ctx.strokeStyle=rgba(C.c_man,.9);ctx.beginPath();ctx.moveTo(pl,sy(PASSLINE));ctx.lineTo(pl+gw,sy(PASSLINE));ctx.stroke();ctx.setLineDash([]);ctx.fillStyle=C.c_man;ctx.textAlign='left';ctx.fillText('합격선 '+pct(PASSLINE),pl+6,sy(PASSLINE)-5);
    ['a1','a10','a100'].forEach((k,c)=>{const v=D.map(d=>d[k]),mn=Math.min(...v),mx=Math.max(...v),mean=ML.mean(v);ctx.fillStyle=C.text;ctx.font=font(12.5,700);ctx.textAlign='center';ctx.fillText(['나무 1그루','10그루','100그루'][c],cx(c),250-10);
      ctx.strokeStyle=rgba(C.blue,.35);ctx.lineWidth=10;ctx.beginPath();ctx.moveTo(cx(c),sy(mn));ctx.lineTo(cx(c),sy(mx)+.01);ctx.stroke();ctx.lineWidth=1;
      const rg=RNG(c+1);v.forEach((a,i)=>circ(ctx,cx(c)+(rg()-.5)*Math.min(70,gw/5),sy(a),4,i===0?C.stamp:rgba(C.blue,.8),C.panel,1));
      ctx.strokeStyle=C.ink;ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(cx(c)-26,sy(mean));ctx.lineTo(cx(c)+26,sy(mean));ctx.stroke();});
    const sd=k=>{const v=D.map(d=>d[k]),m=ML.mean(v);return Math.sqrt(ML.mean(v.map(x=>(x-m)*(x-m))))*100;};
    $('spStat').textContent='심은 숲 '+D.length+'개'+(D.length>1?' · 표준편차 1그루 '+sd('a1').toFixed(1)+'%p · 10그루 '+sd('a10').toFixed(1)+'%p · 100그루 '+sd('a100').toFixed(1)+'%p':'');
    if(D.length>=4){const r=k=>{const v=D.map(d=>d[k]);return pct(Math.min(...v))+' ~ '+pct(Math.max(...v));};
      $('spInfo').innerHTML='나무 1그루는 '+r('a1')+'를 오가고, 100그루는 '+r('a100')+' 안에 모인다. 평균은 오르고(약한 나무 여럿이 강한 숲 하나) 흔들림은 준다(분산 감소). 빨간 점은 05장 위쪽의 기본 숲이다. A장 Colab에서 20번 다시 뽑은 결과도 같다: 1그루 81.0~88.4%, 10그루 89.1~93.8%, 100그루 92.2~94.6%.';}}
  async function more(m){if(busy)return;busy=true;$('spOne').disabled=$('spFive').disabled=true;
    for(let r=0;r<m;r++){seed+=1;$('spInfo').textContent='숲 '+(D.length+1)+'번째를 심는 중… (100그루)';const F=FOREST.make(A,PTR,{K:2,mtry:2,seed});for(let t=0;t<100;t+=10){F.add(10);await wait(0);}
      D.push({a1:accOf(F,B,PTE,1),a10:accOf(F,B,PTE,10),a100:accOf(F,B,PTE,100)});draw();if(D.length<4)$('spInfo').textContent='점이 '+D.length+'개. 몇 번 더 뽑아 보자.';}
    busy=false;$('spOne').disabled=$('spFive').disabled=false;}
  $('spOne').onclick=()=>more(1);$('spFive').onclick=()=>more(5);draw();onRedraw(draw);}

/* ── 06 인사 카드: 특징 중요도 · 예산 · Extra Trees ── */
function render06(S){const IMP={reg:pool('reg').F.importance(100),cls:pool('cls').F.importance(100)};
  function bars(k){const w=IMP[k],ord=w.map((v,j)=>[v,j]).sort((a,b)=>b[0]-a[0]),mx=ord[0][0];
    $('impBars').innerHTML=ord.map(([v,j])=>'<div class="row"><span style="text-align:left;font-family:var(--font)">'+FEATK[FEAT[j]]+'</span><div class="tr"><i style="left:0;width:'+(v/mx*100)+'%;'+(j===0||j===3||j===7?'':'opacity:.55')+'"></i></div><span>'+v.toFixed(3)+'</span></div>').join('');
    const top=ord.slice(0,3).map(o=>FEATK[FEAT[o[1]]]).join(' · ');
    $('impInfo').innerHTML=(k==='reg'?'회귀 숲':'분류 숲')+'이 가장 많이 물어본 특징은 <b>'+top+'</b>다. 진한 막대(시멘트·물·재령)는 매뉴얼 씨의 두 숫자, 물-결합재비(물 ÷ 시멘트 등)와 재령을 이루는 재료다. 숲은 물-결합재비를 따로 받지 않았지만 시멘트와 물로 자르고 재령으로 다시 자르며 같은 지식을 찾아냈다. A장 Colab의 회귀 숲도 시멘트 0.344 · 재령 0.330 · 물 0.103 순이다.'+
      (k==='cls'?' 분류 숲은 모래·자갈 같은 골재가 회귀 숲보다 높게 나온다. 특징 두 개만 받는 갈림에서 약한 특징끼리 경쟁해 뽑히는 일이 생기고, 값이 많은 연속 특징은 쓸데없는 갈림에도 쓰이기 쉽기 때문이다.':'')+
      '<br><span class="muted">확인 방법: 시험지에서 특징 하나를 섞어 R²가 얼마나 떨어지는지 보는 순열 중요도(A장 10번 칸)는 재령 0.69 · 시멘트 0.58 · 물 0.19 · 슬래그 0.17 순으로, 1·2위는 같고 골재 둘은 0.03 아래로 내려간다. 어느 쪽이든 "많이 의지했다"는 뜻이지 "그 재료를 늘리면 강도가 그만큼 오른다"는 원인이 아니다.</span>';}
  seg('#impChoice button',v=>bars(v));bars('reg');
  const Pc=pool('cls'),Pr=pool('reg'),nc=Pc.F.nodes(100),nr=Pr.F.nodes(100);
  $('costStat').innerHTML=statGrid([['200','나무 수 (분류 100 + 회귀 100)'],[(nc+nr).toLocaleString(),'외울 마디 수 (분류 '+nc.toLocaleString()+' + 회귀 '+nr.toLocaleString()+')','a'],[Math.round(Pc.ms+Pr.ms).toLocaleString()+' ms','학습 (이 브라우저, 한 줄로)'],[(S.predMs+S.predMsR).toFixed(0)+' ms','시험 258건 예측 (두 숲)','s']],2);
  $('costInfo').innerHTML='5-4 로지스틱 회귀는 숫자 9개, 5-1 KNN은 훈련 데이터 6,176개를 외웠다. 숲은 마디마다 특징 번호·문턱·답을 들고 있으니 그보다 훨씬 무겁다. 회귀 숲이 분류 숲보다 마디가 많은 것은 잎이 "순수"해지려면 강도 값이 같아질 때까지 갈라야 해서다. 예측 한 건은 나무마다 뿌리에서 잎까지 평균 '+(Pc.F.trees.reduce((a,T)=>a+T.depth,0)/100).toFixed(0)+'번 안팎의 질문을 지나니 100그루면 천 번쯤이다. 이 브라우저는 나무를 한 줄로 심었지만, scikit-learn은 <code>n_jobs=-1</code>로 CPU 여러 개에 나눠 심는다.';}
async function etRun(){const box=$('etStat');box.innerHTML='<p class="muted">Extra Trees 200그루를 심는 중…</p>';await ensure('etc',100);await ensure('etr',100);const S=await results(),Fc=pool('etc').F,Fr=pool('etr').F,pr=B.map(x=>Fr.predict(x)),acc=accOf(Fc,B,PTE,100),R=r2(YTE,pr),M=mae(YTE,pr);
  box.innerHTML=statGrid([[pct(acc),'분류 시험 정확도 (숲 '+pct(S.acc)+')','a'],['R² '+fmtR(R),'회귀 시험 R², MAE '+fmtM(M)+' (숲 '+fmtR(S.R)+')','a'],[Math.round(pool('etc').ms+pool('etr').ms)+' ms','학습 (숲 '+Math.round(pool('cls').ms+pool('reg').ms)+' ms)']],3);
  $('etInfo').innerHTML='이 과제지에서 Extra Trees는 회귀에서 숲보다 낫고(A장 Colab R² 0.948 vs 0.927), 분류에서는 비슷하거나 조금 낮다(91.9% vs 93.4%). 문턱을 재지 않으니 대개 더 빠르다. 어느 쪽이 나은지는 데이터마다 다르므로 교차 검증으로 고른다. 근무 조건(인자)은 랜덤 포레스트와 같고, 기본값이 <code>bootstrap=False</code>라는 점만 다르다.';return {acc,R,M};}
$('etRun').addEventListener('click',etRun);

/* ── 면접 결과 발표 ── */
Interview('iv7',{title:'랜덤 포레스트는 35MPa 합격 판정과 강도 예측, 두 업무에 모두 합격할까?',
  q:'조건: 봉인된 시험지 258건에서 분류 정확도가 매뉴얼 씨('+pct(PASSLINE)+')와 기준선('+pct(BASE_ACC.acc)+')을, 회귀 R²가 매뉴얼 경험식(R² '+fmtR(LINE_R2)+')과 기준선(R² 0)을 모두 넘으면 합격. 근무 조건은 기본값(나무 100그루). 숲은 브라우저가 직접 심는다.',
  fly:'한 그루는 흔들린다며. 다시 뽑아도 버텨? 시험지 없이 미리 알 수는 있어?',robo:'나무 200그루 몸값은? 마디 몇 개, 학습 몇 초?',
  async run(setP){const S=await results(setP);const pa=S.acc>PASSLINE&&S.acc>BASE_ACC.acc,pr=S.R>LINE_R2&&S.R>BASE_R2,pass=pa&&pr,nodes=pool('cls').F.nodes(100)+pool('reg').F.nodes(100);
    return {result:pass?'pass':'hold',flyAns:'분류 '+pct(S.acc)+' · 회귀 R² '+fmtR(S.R)+' · 시험지 없이 OOB '+pct(S.oobC)+' / R² '+fmtR(S.oobR),roboAns:'마디 '+nodes.toLocaleString()+'개 · 학습 '+Math.round(pool('cls').ms+pool('reg').ms).toLocaleString()+'ms (브라우저)',
      flyMood:pass?'arms':'coffee',flyMove:pass?'jump':'shake',roboMood:pass?'wow':'sigh',roboMove:pass?'jump':'tilt',asstMood:pass?'spark':'sigh',docMood:pass?'judge':'scale',
      measure:'봉인된 시험지 결과예요. 분류: 랜덤 포레스트 <b>'+pct(S.acc)+'</b>, 매뉴얼 씨 '+pct(PASSLINE)+', 기준선 '+pct(BASE_ACC.acc)+'. 회귀: <b>R² '+fmtR(S.R)+'</b>(MAE '+fmtM(S.M)+'MPa), 경험식 R² '+fmtR(LINE_R2)+'(MAE '+fmtM(LINE_MAE)+'), 기준선 R² '+fmtR(BASE_R2)+'. 지금까지 본 지원자 중 두 과제 모두 1위예요. 같은 숲의 첫 나무 하나만 쓰면 '+pct(S.one)+' · R² '+fmtR(S.R1)+'로 내려가요.',
      verdict:pass?'합격입니다. 겸용팀에 배치합니다. 조건이 둘 있어요. 70MPa를 넘는 고강도 배합은 낮춰 부르는 버릇이 있으니 그 구간은 따로 확인할 것, 응답 시간이 빠듯한 장비에 올릴 때는 나무 수와 깊이를 다시 심사할 것.':'보류입니다. 두 합격선 중 하나를 넘지 못했어요.',
      key:'다르게 틀리는 나무 백 그루 — '+pct(PASSLINE)+' → '+pct(S.acc)+' · R² '+fmtR(LINE_R2)+' → '+fmtR(S.R),
      render(box){box.innerHTML='<div class="bars">'+barRows([['숲 100그루 · 분류',S.acc,'var(--blue)',pct(S.acc)],['나무 1그루 · 분류',S.one,'var(--amber)',pct(S.one)],['매뉴얼 씨 (5-0)',PASSLINE,'var(--c-man)',pct(PASSLINE)],['기준선 (늘 '+PNAME[BASE_ACC.k]+')',BASE_ACC.acc,'var(--muted)',pct(BASE_ACC.acc)]],1)+'</div>'+
        '<div class="bars" style="margin-top:12px">'+barRows([['숲 100그루 · 회귀',S.R,'var(--blue)','R² '+fmtR(S.R)],['나무 1그루 · 회귀',S.R1,'var(--amber)','R² '+fmtR(S.R1)],['매뉴얼 경험식',LINE_R2,'var(--c-man)','R² '+fmtR(LINE_R2)],['기준선 (늘 평균)',Math.max(0,BASE_R2),'var(--muted)','R² '+fmtR(BASE_R2)]],1)+'</div>';}};},
  alts:[
   {label:'나무 1그루만 (숲의 첫 나무)',async run(box){const S=await results();const pa=S.one>PASSLINE,pr=S.R1>LINE_R2;
     box.innerHTML=statGrid([[pct(S.one),'분류 시험 정확도','a'],['R² '+fmtR(S.R1),'회귀 시험 R² (MAE '+fmtM(S.M1)+')','a'],[pct(PASSLINE)+' · R² '+fmtR(LINE_R2),'두 합격선','m']],3)+'<p class="foot">부트스트랩으로 뽑은 과제지 한 장, 끝까지 키운 나무 한 그루다. 판정: '+(pa&&pr?'두 합격선은 넘지만':'합격선 하나 이상을 못 넘고')+' 숲보다 분류 '+((S.acc-S.one)*100).toFixed(1)+'%p, R² '+fx(S.R-S.R1,3)+' 낮다. 05장 "다시 뽑기"에서 보듯 이 한 그루는 다시 뽑으면 크게 흔들린다. 같은 직원을 내일 다시 뽑으면 성적이 바뀌는 셈이다.</p>';}},
   {label:'Extra Trees (100그루)',async run(box){const r=await etRun();const pass=r.acc>PASSLINE&&r.R>LINE_R2;
     box.innerHTML=statGrid([[pct(r.acc),'분류 시험 정확도','a'],['R² '+fmtR(r.R),'회귀 시험 R² (MAE '+fmtM(r.M)+')','a'],[pct(PASSLINE)+' · R² '+fmtR(LINE_R2),'두 합격선','m']],3)+'<p class="foot">문턱까지 무작위로 뽑는 동생도 두 업무 모두 합격선을 넘는다. 판정: '+(pass?/*i18n:verdict*/'합격':'보류')+'. 회귀는 숲보다 낫고 분류는 비슷하다. 06장에 자세한 비교가 있다.</p>';}},
   {label:'3등급 분류 (저 · 보통 · 고강도)',async run(box){box.innerHTML='<p class="muted">3등급 숲 100그루를 심는 중…</p>';const F=await ensure('g3',100),p=B.map(x=>F.predict(x,100)),a=ML.acc(GTE,p),base=GTE.filter(v=>v===1).length/GTE.length,M=ML.confusion(GTE,p,3);
     box.innerHTML=statGrid([[pct(a),'3등급 시험 정확도','a'],[pct(base),'기준선 (늘 "보통")'],['85.3%','A장 Colab (random_state=0)']],3)+'<div class="tw" style="margin-top:10px">'+cmGrid(M,GRADES,k=>GRADECOL[k])+'</div><p class="foot">클래스가 셋이어도 숲은 그대로다. 잎이 세 등급의 비율을 들고 있고, 나무들의 비율을 평균 내어 가장 큰 등급을 고른다. 틀린 배합은 거의 다 이웃 등급(저↔보통, 보통↔고)으로 헷갈린 것이다.</p>';}}]});

/* 평가 장은 화면에 가까워지면 계산한다 */
(function(){let done=false;function go(){if(done)return;done=true;setTimeout(renderEval,10);}
  if('IntersectionObserver' in window){const io=new IntersectionObserver(es=>{if(es.some(e=>e.isIntersecting)){go();io.disconnect();}},{rootMargin:'600px'});io.observe($('m5'));io.observe($('m6'));}else go();
  window.addEventListener('hashchange',go);if(location.hash)go();})();

/* ══ 확인 문제 ══ */
var QUIZ={
q1:[{q:'랜덤 포레스트가 나무끼리 "덜 닮게" 만드는 두 장치는?',o:[['부트스트랩(다시 뽑은 과제지)과 갈림마다 특징 무작위',1,'나무마다 다른 배합을 보고, 갈림마다 다른 특징 후보를 받으니 서로 다른 곳에서 틀린다.'],['가지치기와 표준화',0,'숲의 나무는 가지치기 없이 끝까지 키우고, 나무는 표준화가 필요 없다.'],['학습률과 규제 C',0,'경사 하강으로 배우는 모델(5-4)의 손잡이다. 숲에는 없다.']]},
 {q:'나무를 끝없이 늘려도 남는 흔들림(분산)은?',o:[['나무끼리 닮은 몫 ρσ²',1,'(1 − ρ)σ²/B는 B가 커지면 0으로 가지만 ρσ²은 남는다. 그래서 나무를 덜 닮게 하는 장치가 중요하다.'],['0이 된다',0,'나무들이 완전히 독립이라면 그렇지만, 같은 데이터에서 자란 나무들은 서로 닮았다.'],['나무 수에 비례해 커진다',0,'나무를 늘리면 흔들림은 줄거나 그대로다. 늘지 않는다.']]},
 {q:'숲의 나무 한 그루는 어떤 나무인가?',o:[['가지치기 없이 끝까지 키운 결정 트리',1,'한 그루는 과대적합되어 있다. 편향이 작고 분산이 큰 나무를 여럿 모아 분산을 줄인다.'],['깊이 1짜리 나무(문턱 하나)',0,'얕은 나무를 차례로 쌓는 것은 5-9 GBM의 방식이다.'],['로지스틱 회귀',0,'숲의 재료는 결정 트리다.']]}],
q2:[{q:'이 면접에서 표준화를 하지 않는 이유는?',o:[['나무는 "문턱보다 작은가"만 물어 단위를 바꿔도 같은 배합끼리 갈리므로',1,'02장 표처럼 문턱의 숫자만 바뀌고 갈리는 배합 수는 같다.'],['표준화를 하면 데이터 누수가 생겨서',0,'훈련 데이터로만 맞추면 누수는 없다. 할 필요가 없을 뿐이다.'],['랜덤 포레스트는 표준화한 데이터를 넣으면 오류가 나서',0,'오류는 나지 않는다. 결과가 거의 같을 뿐이다.']]},
 {q:'과제 B(회귀)의 합격 조건은?',o:[['시험 R²가 매뉴얼 경험식의 R² 0.735와 기준선(R² 0)을 넘는 것',1,'회귀 합격선은 5-1부터 쓰는 매뉴얼 씨의 경험식이다.'],['시험 정확도가 76.7%를 넘는 것',0,'그것은 과제 A(분류)의 합격선이다.'],['훈련 R²가 0.9를 넘는 것',0,'훈련 점수는 외운 몫이 섞여 있어 판정에 쓰지 않는다.']]}],
q3:[{q:'scikit-learn 1.9에서 RandomForest<wbr>Classifier(<wbr>max_features="auto")를 쓰면?',o:[['오류가 난다. 1.3에서 없어졌으니 "sqrt"(기본값)를 쓴다',1,'1.1에서 예고된 뒤 1.3에서 없어졌다. 옛 블로그 코드에 많이 남아 있다.'],['기본값과 같게 돈다',0,'예전에는 그랬지만 지금은 오류다.'],['특징을 모두 쓴다',0,'예전 회귀 숲의 "auto"가 그런 뜻이었다. 지금은 1.0으로 쓴다.']]},
 {q:'RandomForest<wbr>Classifier(<wbr>bootstrap=False, oob_score=True)는?',o:[['오류가 난다. 다시 뽑지 않으면 가방 밖 배합이 없다',1,'OOB는 부트스트랩에서 빠진 배합으로 채점하는 방법이다.'],['OOB 점수가 1.0이 된다',0,'계산 자체를 할 수 없다.'],['경고만 내고 훈련 점수를 OOB로 쓴다',0,'scikit-learn은 오류를 낸다.']]},
 {q:'강도(MPa)를 맞히려고 RandomForest<wbr>Classifier에 강도 숫자를 그대로 넣으면?',o:[['연속값 타겟이라 오류가 난다. RandomForest<wbr>Regressor를 써야 한다',1,'이름이 비슷한 두 클래스를 바꿔 쓰는 실수가 흔하다.'],['자동으로 회귀로 바뀐다',0,'바뀌지 않는다.'],['정확도가 오른다',0,'오류가 나서 학습이 되지 않는다.']]}],
q4:[{q:'772건에서 772번 복원 추출하면, 한 번도 안 뽑히는 배합은 대략 몇 %인가?',o:[['약 37% (1/e)',1,'한 배합이 772번 모두 빠질 확률은 (1 − 1/772)⁷⁷² ≈ 1/e ≈ 0.368이다. 04장 띠의 빨간 테두리가 그만큼이다.'],['0%',0,'복원 추출이라 같은 배합이 여러 번 뽑히고, 그만큼 빠지는 배합이 생긴다.'],['50%',0,'n이 커질수록 1/e ≈ 36.8%에 다가간다.']]},
 {q:'OOB 점수를 수습 기간(시험지를 열기 전)에 봐도 되는 까닭은?',o:[['배합마다 그 배합을 보지 않은 나무들에게만 물어 채점하므로',1,'훈련 데이터 안에서 "처음 보는 배합" 역할을 만든다. 시험지는 쓰지 않는다.'],['OOB는 훈련 점수라서',0,'훈련 점수는 외운 몫이 섞여 99.9%가 나온다. OOB는 그렇지 않다.'],['OOB는 시험지의 일부라서',0,'OOB는 훈련 772건 안에서만 계산한다.']]},
 {q:'나무를 100그루에서 300그루로 늘리면?',o:[['점수는 거의 그대로이고 과대적합은 늘지 않는다. 시간과 메모리만 늘어난다',1,'A장 Colab: 분류 93.4% → 93.4%, 회귀 R² 0.927 → 0.929.'],['과대적합이 심해진다',0,'나무를 더 심는 것은 평균을 더 많이 내는 것일 뿐이다. 과대적합을 키우지 않는다.'],['늘 크게 오른다',0,'OOB 곡선이 평평해진 뒤에는 거의 오르지 않는다.']]}],
q5:[{q:'"다시 뽑기"에서 100그루 점들이 1그루 점들보다 좁게 모이는 까닭은?',o:[['여러 나무를 평균 내면 흔들림(분산)이 줄어서',1,'지원서 6번 칸의 식 그대로다. 평균도 올라간다.'],['100그루의 나무가 더 깊어서',0,'깊이는 비슷하다. 숫자가 많을 뿐이다.'],['다시 뽑을 때 시험지가 바뀌어서',0,'시험지는 그대로다. 숲의 난수만 바뀐다.']]},
 {q:'랜덤 포레스트 회귀가 훈련 데이터의 최고 강도(82.6MPa)보다 높은 값을 내지 못하는 까닭은?',o:[['잎의 답이 훈련 배합 강도의 평균이라 훈련 범위를 넘을 수 없어서',1,'5-1 KNN과 같은 한계다. 범위 밖을 내다봐야 하는 일에는 맞지 않는다.'],['규제가 걸려 있어서',0,'숲에는 규제 항이 없다.'],['표준화를 하지 않아서',0,'표준화와는 상관없다.']]},
 {q:'훈련 정확도 99.9%, 시험 정확도 93% 안팎이다. 어떻게 읽나?',o:[['끝까지 키운 나무들이 훈련 배합을 외웠다. 실력은 시험지나 OOB로 본다',1,'숲에서는 훈련 점수가 늘 높게 나온다. 그래서 OOB가 유용하다.'],['데이터 누수가 있다',0,'시험지를 학습에 쓰지 않았다. 외운 몫일 뿐이다.'],['코드에 버그가 있다',0,'끝까지 키운 나무라면 정상이다.']]}],
q6:[{q:'회귀 숲의 특징 중요도 "시멘트 0.34"의 뜻으로 맞는 것은?',o:[['숲이 시멘트로 자를 때 줄어든 제곱 오차가 전체 감소량의 34%라는 뜻',1,'"많이 의지했다"는 뜻이다. 원인의 크기나 방향은 말해 주지 않는다.'],['시멘트를 1kg 늘리면 강도가 0.34MPa 오른다',0,'그것은 선형 회귀의 가중치처럼 읽은 것이다. 중요도에는 방향도 단위도 없다.'],['시멘트를 빼면 정확도가 34% 떨어진다',0,'그것을 직접 재는 것이 순열 중요도이고, 값이 다르다.']]},
 {q:'Extra Trees가 랜덤 포레스트와 다른 점은?',o:[['문턱을 무작위로 뽑고, 기본값으로 부트스트랩을 하지 않는다',1,'나무가 더 제각각이 되어 분산이 더 줄고, 문턱을 재지 않으니 빠르다.'],['나무를 차례로 심어 앞 나무가 틀린 만큼을 배운다',0,'그것은 부스팅(5-9 GBM)이다.'],['깊이 1짜리 나무만 쓴다',0,'Extra Trees도 기본값은 끝까지 키운다.']]},
 {q:'랜덤 포레스트가 가장 맞지 않는 일은?',o:[['훈련 범위 밖을 내다보는 예측, 판정 근거를 규칙 한 장으로 보여 줘야 하는 업무',1,'잎의 평균으로 답하고, 나무 백 그루의 순서도는 한 장에 들어가지 않는다.'],['단위가 제각각인 공정 표 데이터',0,'가장 잘하는 일이다. 표준화도 필요 없다.'],['분류와 회귀를 함께 해야 하는 일',0,'겸용팀 지원자다. 둘 다 잘한다.']]}]};

/* ══ 공통 B·C·D·E·F 데이터 ══ */
var PROMPTS=[
 {t:'① 내 공정 데이터로 랜덤 포레스트 코드 받기',use:'내 공장의 품질 데이터로 03장의 프롬프트를 맞출 때',p:MAIN_PROMPT},
 {t:'② AI 코드 검증 요청',use:'받은 코드를 다른 AI(또는 같은 AI의 새 대화)에게 검토받을 때',p:'아래 scikit-learn 랜덤 포레스트 코드를 코드 리뷰해 줘. 특히 다음을 하나씩 확인하고, 문제가 있으면 줄 번호와 고친 코드를 보여 줘.\n1) 회귀 타겟에 Regressor, 분류 타겟에 Classifier를 썼나\n2) train_test_split에 stratify와 random_state가 있고, 숲에도 random_state가 있나\n3) 내 scikit-learn 버전(〔1.x〕)에 없는 값(max_features="auto" 등)을 쓰지 않았나\n4) oob_score=True인데 bootstrap=False로 두지 않았나\n5) 성과를 훈련 점수가 아니라 시험 점수나 OOB 점수로 보고했나, 기준선과 비교했나\n6) 시험 데이터로 n_estimators·max_features 등을 고르지 않았나\n추측으로 답하지 말고, 확실하지 않은 부분은 "공식 문서 확인 필요"라고 표시해 줘.'},
 {t:'③ 특징 중요도 해석 상담',use:'feature_importances_ 순위를 보고서에 쓰기 전에',p:'랜덤 포레스트 회귀 모델의 특징 중요도(feature_importances_)가 아래와 같다. 이 순위를 품질 보고서에 쓰려고 한다.\n〔시멘트 0.34, 재령 0.33, 물 0.10, 고로 슬래그 0.08, 고성능 감수제 0.06, 잔골재 0.04, 굵은 골재 0.03, 플라이 애시 0.02〕\n1) 불순도 기반 중요도가 무엇을 재는지, 원인(인과)으로 읽으면 안 되는 까닭을 비전문가용으로 설명해 줘.\n2) 서로 얽힌 특징(물과 고성능 감수제, 골재 두 가지)이 있을 때 생기는 왜곡을 설명해 줘.\n3) sklearn.inspection.permutation_importance로 시험 데이터에서 다시 재는 Colab 코드와, 두 결과를 나란히 그리는 막대그래프 코드를 써 줘.\n4) 보고서에 쓸 수 있는 안전한 문장 예시 3개를 제안해 줘.'},
 {t:'④ 다음 지원자와 비교 실험 설계',use:'랜덤 포레스트를 GBM 계열(5-9)과 공정하게 비교하고 싶을 때',p:'콘크리트 강도 데이터(UCI id=165)로 RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor를 공정하게 비교하는 Colab 코드를 써 줘. Colab에 xgboost나 lightgbm이 설치되어 있으면 함께 비교하고, 없으면 건너뛰게 해 줘.\n- 시험 데이터는 train_test_split(test_size=0.25, stratify=강도 3등급(25·45MPa), random_state=42)로 한 번만 떼어 둔다\n- 하이퍼파라미터는 훈련 데이터 안에서 5겹 교차 검증(RandomizedSearchCV, 시도 〔20〕번)으로 고른다\n- 출력: 모델별 교차 검증 R², 시험 R²·MAE, 학습 시간, 예측 시간을 한 표로\n- random_state를 모두 고정하고, 시험 데이터는 마지막에 한 번만 쓴다\n결과 표를 읽는 법(점수 차이가 난수 흔들림보다 큰지 보는 법)도 짧게 설명해 줘.'}];
var GL=[
 ['모델','랜덤 포레스트','부트스트랩과 특징 무작위로 서로 다르게 키운 결정 트리 여럿의 답을 모아(분류는 다수결·확률 평균, 회귀는 평균) 내는 앙상블 모델.'],
 ['모델','앙상블','모델 여럿의 답을 모아 하나의 답을 내는 방법. 나란히 모으는 배깅과 차례로 고쳐 가는 부스팅이 대표적이다.'],
 ['모델','배깅','Bootstrap AGGregatING. 부트스트랩한 데이터로 모델을 여럿 학습해 평균 내는 방법. 랜덤 포레스트는 배깅에 특징 무작위를 더한 것이다.'],
 ['모델','부트스트랩','데이터 n건에서 n번 복원 추출해 새 데이터를 만드는 일. 같은 줄이 여러 번 뽑히고, 약 37%는 한 번도 안 뽑힌다.'],
 ['모델','복원 추출','뽑은 것을 다시 넣고 또 뽑는 방식. 같은 것이 여러 번 나올 수 있다.'],
 ['모델','OOB','Out-of-Bag(가방 밖). 부트스트랩에서 빠진 배합. 그 배합을 안 본 나무들에게만 물어 낸 점수가 OOB 점수다.'],
 ['모델','특징 무작위','가지를 나눌 때마다 특징 몇 개만 무작위로 후보로 주는 일. scikit-learn의 max_features. 나무끼리 덜 닮게 한다.'],
 ['모델','결정 트리','"특징 ≤ 문턱?" 질문을 이어 데이터를 가르는 모델. 잎에서 답한다(5-6).'],
 ['모델','지니 불순도','한 마디 안에서 무작위로 두 배합을 골랐을 때 이름표가 다를 확률. 1 − Σp². 0이면 순수하다.'],
 ['모델','다수결','나무들의 답 중 가장 많은 쪽을 고르는 일. scikit-learn은 나무들의 확률을 평균 내어 고른다(부드러운 투표).'],
 ['모델','Extra Trees','Extremely Randomized Trees. 문턱까지 무작위로 뽑고 부트스트랩 없이 키운 나무들의 앙상블.'],
 ['모델','특징 중요도','숲이 그 특징으로 자를 때 줄어든 불순도를 모든 나무에서 더해 합이 1이 되게 나눈 값. 원인의 크기가 아니다.'],
 ['모델','순열 중요도','시험 데이터에서 특징 하나의 값을 무작위로 섞었을 때 점수가 얼마나 떨어지는지로 잰 중요도.'],
 ['학습','분산','데이터를 조금 바꿔 다시 학습했을 때 예측이 흔들리는 정도. 끝까지 키운 나무 한 그루는 분산이 크다.'],
 ['학습','편향','모델이 너무 단순해 생기는 한쪽으로 치우친 오차. 끝까지 키운 나무는 편향이 작다.'],
 ['학습','과대적합','훈련 데이터를 외워 처음 보는 데이터에서 성적이 떨어지는 일.'],
 ['학습','하이퍼파라미터','학습 전에 사람이 정하는 근무 조건. n_estimators, max_features, min_samples_leaf 등.'],
 ['데이터','특징','모델이 입력으로 쓰는 열. 시멘트·고로 슬래그·플라이 애시·물·고성능 감수제·굵은 골재·잔골재·재령.'],
 ['데이터','타겟','맞히려는 정답 열. 이 면접에서는 강도(회귀)와 35MPa 합격 여부(분류).'],
 ['데이터','표준화','특징마다 평균을 빼고 표준편차로 나눠 단위를 맞추는 일. 나무 모델에는 필요 없다.'],
 ['평가','정확도','맞힌 샘플의 비율.'],
 ['평가','혼동행렬','정답(행)과 예측(열)을 짝지어 센 표.'],
 ['평가','R²','결정계수. 1이면 완벽, 0이면 늘 평균을 답하는 것과 같다.'],
 ['평가','MAE','평균 절대 오차. 예측이 실제에서 평균 몇 MPa 벗어났나.'],
 ['평가','기준선','배우지 않고도 낼 수 있는 점수. 늘 다수 쪽(분류), 늘 평균(회귀).']];
var ANALOGY=[
 ['부트스트랩','면접관마다 서류 더미에서 무작위로 뽑은(겹칠 수 있는) 지원서 묶음만 읽고 기준을 세운다',1,'','실제 위원회는 같은 서류를 다 함께 본다. 일부러 덜 보게 해서 의견을 갈라놓는 것은 통계적 장치다'],
 ['특징 무작위','면접관마다 질문할 수 있는 항목을 제비뽑기로 몇 개만 준다',2,'초파리 버섯체의 케년 세포는 냄새 입력(투사 뉴런) 50여 종 중 평균 6~7개를 무작위에 가깝게 골라 받는다','케년 세포 하나는 나무가 아니라 입력을 더해 문턱을 넘는지 보는 단순한 장치다. 학습은 그 뒤 출력 쪽 시냅스에서 일어난다'],
 ['다수결 · 평균','황소 무게 맞히기 대회에서 787명의 추측을 모았더니 가운데 값이 실제 무게와 1% 안쪽이었다(골턴, 1907)',2,'여러 뉴런의 반응을 모아 읽으면 하나보다 정확하다(집단 부호화)','뉴런들은 서로 상관되어 있어 모아도 이득에 한계가 있다. 이 점은 숲의 ρ와 오히려 닮았다. 다만 뇌가 표를 세는 것은 아니다'],
 ['OOB 점수','면접관마다 자기가 읽지 않은 지원서로 자기 기준을 미리 시험해 본다',1,'','사람은 한 번 본 것을 잊지 못하지만, 나무는 정말로 그 배합을 본 적이 없다'],
 ['특징 중요도','회의록에서 어떤 질문이 판정을 가장 많이 갈랐는지 센다',1,'','많이 물어본 질문이 판정의 원인이라는 보장은 없다. 비슷한 질문이 둘이면 공을 나눠 갖는다'],
 ['Extra Trees','질문의 문턱까지 주사위로 정하고, 여럿의 판정을 모은다',2,'초파리 후각 회로의 무작위 연결이 비슷한 냄새끼리 비슷한 꼬리표를 붙이는 데 쓰인다는 해석(무작위 투영 해싱)','초파리 해싱은 정답 없이 비슷한 것을 찾는 일이다. Extra Trees는 정답을 보고 무작위 문턱 중 나은 것을 고른다']];
var REFS=[
 'Yeh, I.-C. (1998). Modeling of strength of high-performance concrete using artificial neural networks. <i>Cement and Concrete Research</i>, 28(12), 1797–1808. <a href="https://doi.org/10.1016/S0008-8846(98)00165-3" target="_blank" rel="noopener">doi:10.1016/S0008-8846(98)00165-3</a> — 실기 과제지 원출처',
 'Yeh, I.-C. (2007). Concrete Compressive Strength [Dataset]. UCI Machine Learning Repository. <a href="https://doi.org/10.24432/C5PK67" target="_blank" rel="noopener">doi:10.24432/C5PK67</a> (CC BY 4.0) — 02장 데이터',
 'Breiman, L. (2001). Random forests. <i>Machine Learning</i>, 45(1), 5–32. <a href="https://doi.org/10.1023/A:1010933404324" target="_blank" rel="noopener">doi:10.1023/A:1010933404324</a> — 랜덤 포레스트, OOB, 특징 중요도, 01·04·06장',
 'Breiman, L. (1996). Bagging predictors. <i>Machine Learning</i>, 24(2), 123–140. <a href="https://doi.org/10.1007/BF00058655" target="_blank" rel="noopener">doi:10.1007/BF00058655</a> — 배깅, 지원서 2번 칸',
 'Ho, T. K. (1998). The random subspace method for constructing decision forests. <i>IEEE Transactions on Pattern Analysis and Machine Intelligence</i>, 20(8), 832–844. <a href="https://doi.org/10.1109/34.709601" target="_blank" rel="noopener">doi:10.1109/34.709601</a> — 특징 무작위, 지원서 3번 칸',
 'Geurts, P., Ernst, D., &amp; Wehenkel, L. (2006). Extremely randomized trees. <i>Machine Learning</i>, 63(1), 3–42. <a href="https://doi.org/10.1007/s10994-006-6226-1" target="_blank" rel="noopener">doi:10.1007/s10994-006-6226-1</a> — Extra Trees, 06장',
 'Breiman, L., Friedman, J. H., Olshen, R. A., &amp; Stone, C. J. (1984). <i>Classification and Regression Trees</i>. Wadsworth. — CART, 지니 불순도, 04장',
 'Hastie, T., Tibshirani, R., &amp; Friedman, J. (2009). <i>The Elements of Statistical Learning</i> (2nd ed.), 15장 Random Forests. Springer. — 분산 식 ρσ² + (1 − ρ)σ²/B, 01장',
 'Strobl, C., Boulesteix, A.-L., Zeileis, A., &amp; Hothorn, T. (2007). Bias in random forest variable importance measures: Illustrations, sources and a solution. <i>BMC Bioinformatics</i>, 8, 25. <a href="https://doi.org/10.1186/1471-2105-8-25" target="_blank" rel="noopener">doi:10.1186/1471-2105-8-25</a> — 중요도의 치우침, 06장',
 'Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. <i>Journal of Machine Learning Research</i>, 12, 2825–2830. — A장 코드',
 'scikit-learn 문서. <a href="https://scikit-learn.org/stable/modules/ensemble.html" target="_blank" rel="noopener">Ensembles: Gradient boosting, random forests, bagging, voting, stacking</a> · <a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html" target="_blank" rel="noopener">RandomForestClassifier</a> — max_features 기본값과 "auto" 삭제, 03·06장',
 'Galton, F. (1907). Vox populi. <i>Nature</i>, 75, 450–451. <a href="https://doi.org/10.1038/075450a0" target="_blank" rel="noopener">doi:10.1038/075450a0</a> — D장 다수결 비유',
 'Caron, S. J. C., Ruta, V., Abbott, L. F., &amp; Axel, R. (2013). Random convergence of olfactory inputs in the <i>Drosophila</i> mushroom body. <i>Nature</i>, 497, 113–117. <a href="https://doi.org/10.1038/nature12063" target="_blank" rel="noopener">doi:10.1038/nature12063</a> — D장 특징 무작위 비유',
 'Dasgupta, S., Stevens, C. F., &amp; Navlakha, S. (2017). A neural algorithm for a fundamental computing problem. <i>Science</i>, 358(6364), 793–796. <a href="https://doi.org/10.1126/science.aam9868" target="_blank" rel="noopener">doi:10.1126/science.aam9868</a> — D장 Extra Trees 비유'];
var TIP={who:'머신러닝을 처음 배우는 대학생·직장인·제조 현장 실무자. 5-6 결정 트리를 먼저 보면 좋다. 수식은 분산 식 ρσ² + (1 − ρ)σ²/B와 1/e 두 개만 나온다',
 goal:'① 부트스트랩과 특징 무작위가 왜 필요한지 "서로 다르게 틀리기"로 설명한다 ② 나무 수에 따라 점수와 흔들림이 어떻게 바뀌는지 그림으로 말한다 ③ OOB 점수와 시험 점수, 훈련 점수를 구별한다 ④ AI가 낸 코드에서 Regressor/Classifier 혼동, 없어진 인자, 훈련 데이터 채점을 찾는다 ⑤ 특징 중요도를 원인으로 읽지 않는다',
 time:'읽기 25분 · 실습(01·04·05장 그림) 25분 · 틀린 곳 찾기 10분 · Colab 15분',
 plans:[['강의형 50분','00장 지원서를 인쇄해 나눠 주고 9칸을 함께 읽기 10분 → 01장 슬라이더 1 → 100과 다시 뽑기 10분 → 04장 숲 키우기 재생(분류·회귀 비교) 10분 → 05장 다시 뽑기 5번 10분 → 면접 결과를 먼저 맞히고 발표 10분.'],['실무 OJT 90분','03장 프롬프트로 각자 AI에게 코드를 받는다 20분 → 체크리스트로 서로의 코드를 바꿔 검사 20분 → Colab에서 실행하고 A장 결과와 비교 20분 → B장 ③ 프롬프트로 특징 중요도를 순열 중요도와 비교하고, "보고서에 쓸 수 있는 문장"을 정해 발표 30분.'],['1:1 튜터링','01장에서 나무 수를 1로 두고 다시 뽑기를 세 번 눌러 보게 한 뒤 100으로 같은 일을 해 보게 한다. "왜 숲은 덜 흔들리나"를 학습자가 자기 말로 설명할 때까지 지원서 5·6번 칸을 오간다.']],
 myths:[['나무를 많이 심을수록 과대적합된다','반대다. 나무를 늘리면 평균을 더 많이 낼 뿐이라 점수는 평평해진다. 과대적합은 나무 하나하나의 깊이와 관계있다'],['랜덤 포레스트는 결정 트리를 가지치기해서 모은다','기본값은 가지치기 없이 끝까지 키운다. 숲이 분산을 줄여 주기 때문이다'],['특징 중요도가 높으면 그 재료가 강도를 올린다','많이 물어봤다는 뜻일 뿐이다. 방향도 인과도 말해 주지 않는다. 얽힌 특징끼리는 공을 나눠 갖는다'],['훈련 정확도 99.9%면 훌륭한 모델이다','끝까지 키운 나무는 훈련 데이터를 외운다. 시험 점수나 OOB 점수를 본다'],['랜덤 포레스트도 표준화해야 한다','나무는 순서만 본다. 표준화해도 결과가 거의 같다'],['랜덤 포레스트 회귀도 특징을 무작위로 고른다','scikit-learn 회귀 숲의 기본값은 특징 모두(1.0)다. 기본 설정에서는 배깅이다']],
 tasks:[['과제 1 · 다시 뽑기 실험','A장 5번 칸을 나무 1·3·10·30·100그루로 넓혀 다시 뽑기 20번의 표준편차를 표와 그래프로 정리하고, 지원서 6번 칸의 식과 비교한다.'],['과제 2 · max_features 탐색','회귀 숲에서 max_features를 0.2~1.0으로 바꿔 OOB R²를 그리고, 시험지를 열지 않고 고른 값이 시험 R²에서도 좋은지 확인한다.'],['과제 3 · 중요도 보고서','B장 ③ 프롬프트로 불순도 중요도와 순열 중요도를 나란히 그리고, "우리 공장 품질 보고서"에 쓸 수 있는 문장 세 개를 쓴다. 원인처럼 들리는 표현을 고친다.'],['과제 4 · 틀린 곳 찾기 만들기','AI에게 일부러 실수가 섞인 랜덤 포레스트 코드를 받고(B장 ②를 거꾸로 활용), 짝에게 찾게 한다. 실수마다 "왜 틀렸나"를 한 줄로 쓴다.']],
 ai:'학생이 AI에게 코드를 받는 것은 막지 말고 과정에 넣는다. 다만 ① 프롬프트를 제출하게 하고 ② 체크리스트 아홉 항목을 직접 표시하게 하고 ③ 훈련·OOB·시험 점수를 나란히 쓰고 기준선·합격선과 비교한 문장을 쓰게 하면, "받아 적기"가 아니라 "검증하기"를 배운다. 특히 숲은 짧은 코드로 높은 점수가 나와 의심 없이 넘어가기 쉽다. "그 점수는 무엇으로 쟀나"를 매번 묻게 한다. AI 답의 함수·인자는 공식 문서와 버전으로 확인하게 하고, 과제에 AI 도움을 받았으면 사용 범위를 밝히게 한다.'};
var CODE={title:'랜덤 포레스트 면접 — 나무 한 그루에서 숲까지',sub:'numpy, pandas, scikit-learn, ucimlrepo · CPU 1분 안팎',file:'random_forest_interview',
 desc:'이 편의 면접을 그대로 옮겼다. UCI에서 콘크리트 데이터를 받아 시험지를 봉인하고(1·2번 칸), 두 합격선을 다시 재고(3번), 나무 한 그루와 숲 1·10·100·300그루(4번), 다시 뽑기 20번의 흔들림(5번), 부트스트랩 63%와 OOB(6번), 특징 무작위와 나무끼리의 상관(7번), 회귀 숲(8번), 3등급(9번), 특징 중요도와 순열 중요도(10번), Extra Trees(11번), 예산(12번), 판정(13번)까지 간다. 노트북으로 저장하면 칸마다 셀이 나뉜다.',
 cap:'실행 결과 (scikit-learn 1.9, CPU). 숲의 random_state=0. 시간(ms)은 컴퓨터마다 다르다. 브라우저의 숲(01·04·05·06장)은 난수가 달라 나무가 다르지만 같은 결론(분류 93% 안팎, 회귀 R² 0.93 안팎, 두 업무 모두 합격)이 나온다.'};
