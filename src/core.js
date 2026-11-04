/* ══ 공용 도구 ══════════════════════════════════════════════════ */
var $=function(id){return document.getElementById(id)};
var REDUCE=!!(window.matchMedia&&window.matchMedia('(prefers-reduced-motion: reduce)').matches);
var FF='"IBM Plex Sans KR","Apple SD Gothic Neo","Malgun Gothic",system-ui,sans-serif';
var MF='"IBM Plex Mono","IBM Plex Sans KR",monospace';
function font(s,w,m){return (w||400)+' '+s+'px '+(m?MF:FF)}
/* 칸 너비(maxW)를 넘는 글자는 글자 크기를 줄여 맞춘다 — 번역판에서 라벨이 길어질 때 */
function fitText(ctx,txt,x,y,maxW){var f=ctx.font,m=/(\d+(?:\.\d+)?)px/.exec(f),s=m?+m[1]:12,w=ctx.measureText(txt).width;if(w>maxW&&m){ctx.font=f.replace(m[0],Math.max(8,s*maxW/w).toFixed(1)+'px');ctx.fillText(txt,x,y);ctx.font=f;}else ctx.fillText(txt,x,y);}
function cols(){var cs=getComputedStyle(document.documentElement),o={};
  ['bg','panel','panel2','rule','text','muted','soft','ink','blue','stamp','ok','amber','info','c-doc','c-asst','c-fly','c-robo','c-app','c-man'].forEach(function(k){o[k.replace('c-','c_')]=cs.getPropertyValue('--'+k).trim();});return o;}
function rgba(c,a){if(!c)return 'rgba(0,0,0,'+a+')';if(c[0]==='#'){var h=c.length===4?c.replace(/#(.)(.)(.)/,'#$1$1$2$2$3$3'):c;return 'rgba('+parseInt(h.slice(1,3),16)+','+parseInt(h.slice(3,5),16)+','+parseInt(h.slice(5,7),16)+','+a+')';}return c;}
function fx(x,d){d=d==null?2:d;var m=Math.pow(10,d),r=Math.round(x*m)/m;if(Math.abs(r)<0.5/m)r=0;return (r<0?'−':'')+Math.abs(r).toFixed(d);}
function pct(x,d){d=d==null?1:d;return (x*100).toFixed(d)+'%'}
function fit(c,h){var dpr=window.devicePixelRatio||1,w=c.clientWidth||c.parentElement.clientWidth||600;
  if(c.width!==Math.round(w*dpr)||c.height!==Math.round(h*dpr)){c.width=Math.round(w*dpr);c.height=Math.round(h*dpr);c.style.height=h+'px';}
  var x=c.getContext('2d');x.setTransform(dpr,0,0,dpr,0,0);return w;}
function rr(ctx,x,y,w,h,r){ctx.beginPath();if(ctx.roundRect)ctx.roundRect(x,y,w,h,r);else ctx.rect(x,y,w,h);}
function circ(ctx,x,y,r,fill,stroke,lw){ctx.beginPath();ctx.arc(x,y,r,0,7);if(fill){ctx.fillStyle=fill;ctx.fill();}if(stroke){ctx.strokeStyle=stroke;ctx.lineWidth=lw||1.5;ctx.stroke();}}
function ease(t){return t<.5?2*t*t:1-Math.pow(-2*t+2,2)/2}
function clamp(x,a,b){return Math.max(a,Math.min(b,x))}
var REDRAW=[];function onRedraw(fn){REDRAW.push(fn);}
function redrawAll(){REDRAW.forEach(function(f){try{f();}catch(e){}});}
if(window.matchMedia){var mq=window.matchMedia('(prefers-color-scheme: dark)');(mq.addEventListener?mq.addEventListener('change',redrawAll):mq.addListener&&mq.addListener(redrawAll));}
new MutationObserver(redrawAll).observe(document.documentElement,{attributes:true,attributeFilter:['data-theme']});
var rzT;window.addEventListener('resize',function(){clearTimeout(rzT);rzT=setTimeout(redrawAll,90);});
var TICK=[],VIS=new Set(),IO=('IntersectionObserver' in window)?new IntersectionObserver(function(es){es.forEach(function(e){e.isIntersecting?VIS.add(e.target):VIS.delete(e.target);});},{rootMargin:'60px'}):null,lastTs=0,ticking=false;
function onTick(el,fn){TICK.push({el:el,fn:fn});if(IO)IO.observe(el);else VIS.add(el);if(!ticking){ticking=true;requestAnimationFrame(loop);}}
function loop(ts){var dt=lastTs?Math.min(0.05,(ts-lastTs)/1000):0.016;lastTs=ts;TICK.forEach(function(m){if(VIS.has(m.el))m.fn(dt);});requestAnimationFrame(loop);}
function wait(ms){return new Promise(function(r){setTimeout(r,ms)})}
function toast(msg){var t=$('toast');if(!t)return;t.textContent=msg;t.classList.add('on');clearTimeout(toast.t);toast.t=setTimeout(function(){t.classList.remove('on')},1800);}
function seg(sel,fn){var bs=document.querySelectorAll(sel);bs.forEach(function(b){b.addEventListener('click',function(){bs.forEach(function(x){x.setAttribute('aria-pressed',x===b?'true':'false')});fn(b.getAttribute('data-v'),b);});});}
function RNG(a){return function(){a|=0;a=a+0x6D2B79F5|0;var t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296}}
function rago(w){if(typeof LANG!=='undefined'&&LANG!=='ko')return w;/* 한국어 조사 '(이)라고' — 다른 언어에서는 이름만 */var t=w.replace(/["'”」』)\s]+$/,''),c=t.charCodeAt(t.length-1);return w+((c>=0xAC00&&c<=0xD7A3&&(c-0xAC00)%28)?'이라고':'라고');}
function esc(s){return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')}

/* ══ 콘크리트 데이터 (지도 본부 공통 실기 과제지) ═════════════════════ */
var FEAT=['cement','slag','fly_ash','water','superplasticizer','coarse_agg','fine_agg','age'];
var FEATK={cement:'시멘트',slag:'고로 슬래그',fly_ash:'플라이 애시',water:'물',superplasticizer:'고성능 감수제',coarse_agg:'굵은 골재',fine_agg:'잔골재',age:'재령(일)'};
var FEATU={cement:'kg/m³',slag:'kg/m³',fly_ash:'kg/m³',water:'kg/m³',superplasticizer:'kg/m³',coarse_agg:'kg/m³',fine_agg:'kg/m³',age:'일'};
var GRADES=['저강도','보통 강도','고강도'],GRADE_CUT=[25,45],GRADECOL=['#8f7e64','#1f6fe5','#c0392b'];
var PASS_MPA=35; /* 이진 분류 질문: 압축 강도 35MPa 이상인가 */
function concRows(){return CONC.map(function(r,i){var o={i:i,strength:r[8],test:CONC_TEST.indexOf(i)>=0};FEAT.forEach(function(f,j){o[f]=r[j];});
  o.binder=o.cement+o.slag+o.fly_ash;o.wb=o.water/o.binder;o.grade=o.strength<GRADE_CUT[0]?0:o.strength<GRADE_CUT[1]?1:2;o.pass=o.strength>=PASS_MPA?1:0;return o;});}
/* 매뉴얼 씨의 작업 표준서: 물-결합재비 0.5 미만이고 28일 이상 굳혔으면 35MPa 합격 */
function manualRule(r){return (r.wb<0.5&&r.age>=28)?1:0;}

/* ══ 손글씨 숫자 (비지도 본부 공통 실기 과제지, meta.json에 "data":"digits"가 있는 편만 싣는다) ══ */
var CLU_LINE=0.108; /* 군집팀 합격선: 매뉴얼 씨 규칙 묶기의 ARI */
var DR_LINE=0.181;  /* 차원축소팀 합격선: 매뉴얼 씨 2D 지도(잉크 양·위쪽 비율)에서 5-NN 정확도(5겹) */
function digitRows(){if(typeof DIG_PIX==='undefined')return [];var n=DIG_Y.length,o=[];for(var i=0;i<n;i++){var px=new Array(64);for(var j=0;j<64;j++)px[j]=parseInt(DIG_PIX[i*64+j],17);o.push({i:i,px:px,y:DIG_Y[i]});}return o;}
function digitFeat(px){var ink=0,top=0,c=0;for(var j=0;j<64;j++){ink+=px[j];if(j<32)top+=px[j];}c=px[27]+px[28]+px[35]+px[36];return {ink:ink,top:top/(ink||1),center:c};}
/* 매뉴얼 씨의 정리 규칙: 가운데가 비었나 · 잉크가 많나 · 위가 무거운가 → 8묶음 */
function manualDigit(px){var f=digitFeat(px);return (f.center<12?4:0)+(f.ink>300?2:0)+(f.top>0.5?1:0);}
function drawDigit(ctx,px,x,y,s,ink){var c=s/8;for(var j=0;j<64;j++){var v=px[j]/16;if(v<=0)continue;ctx.fillStyle=rgba(ink||'#111827',v);ctx.fillRect(x+(j%8)*c,y+Math.floor(j/8)*c,Math.ceil(c),Math.ceil(c));}}
var DIGCOL=['#1f6fe5','#c0392b','#2f9d63','#d0782b','#8e5bd6','#17a2b8','#c23b6e','#7a6a12','#4a5568','#0f766e'];

/* ══ 등장인물 ══════════════════════════════════════════════════════ */
var CAST_SHORT={doc:'박사',asst:'조수',fly:'초파리',robo:'주판봇'}; /* 네 컷 말풍선 이름표 */
var CAST={doc:'위원장 박사',asst:'인사 담당 조수',fly:'실무 면접관 초파리',robo:'예산 담당 주판봇'};
var DEFAULT={doc:'doc_front',asst:'asst_smile',fly:'fly_clip',robo:'robo_smile'};
function sprite(who,mood){var k=who+'_'+mood;return SPR[k]||SPR[DEFAULT[who]]||'';}
function hydrate(root){root=root||document;
  root.querySelectorAll('img[data-who]').forEach(function(i){i.src=sprite(i.getAttribute('data-who'),i.getAttribute('data-mood'));});
  root.querySelectorAll('img[data-scene]').forEach(function(i){i.src=SPR[i.getAttribute('data-scene')]||'';});}
function say(who,mood,txt,extra){return '<div class="say '+who+'"'+(extra||'')+'><img src="'+sprite(who,mood)+'" alt="" aria-hidden="true"><div class="bub"><span class="who">'+CAST[who]+'</span>'+txt+'</div></div>';}
/* 지원자·매뉴얼 씨는 그림 대신 사원증 기호 */
function sayCard(kind,sym,name,txt){return '<div class="say '+kind+'"><span class="sym" aria-hidden="true">'+sym+'</span><div class="bub"><span class="who">'+name+'</span>'+txt+'</div></div>';}

/* ══ 편 목록 · 부서 배치도 · 채용 현황판 ═════════════════════════════ */
/* res: 그 편 면접 결과(pass 합격 · hold 보류 · fail 불합격). 아직 안 만든 편은 비워 둔다 */
var EPISODES=[
 {n:'5-0',f:'5-0_job_posting.html',t:'채용 공고',sym:'§',team:'공고',ready:1,res:'hold'},
 {n:'5-1',f:'5-1_knn.html',t:'KNN',sym:'k',team:'입문',ready:1,res:'pass'},
 {n:'5-2',f:'5-2_linear_regression.html',t:'선형 회귀',sym:'β',team:'회귀',ready:1,res:'hold'},
 {n:'5-3',f:'5-3_polynomial_regularization.html',t:'다항 회귀와 규제',sym:'λ',team:'회귀',ready:1,res:'pass'},
 {n:'5-4',f:'5-4_logistic_regression.html',t:'로지스틱 회귀',sym:'σ',team:'분류',ready:1,res:'pass'},
 {n:'5-5',f:'5-5_naive_bayes.html',t:'나이브 베이즈',sym:'P',team:'분류',ready:1,res:'hold'},
 {n:'5-6',f:'5-6_decision_tree.html',t:'결정 트리',sym:'T',team:'겸용',ready:1,res:'pass'},
 {n:'5-7',f:'5-7_random_forest.html',t:'랜덤 포레스트',sym:'RF',team:'겸용',ready:1,res:'pass'},
 {n:'5-8',f:'5-8_svm.html',t:'SVM',sym:'SV',team:'겸용',ready:1,res:'pass'},
 {n:'5-9',f:'5-9_gbm.html',t:'GBM',sym:'GB',team:'겸용',ready:1,res:'pass'},
 {n:'5-10',f:'5-10_kmeans.html',t:'K-Means',sym:'μ',team:'군집',ready:1,res:'pass'},
 {n:'5-11',f:'5-11_hierarchical.html',t:'계층적 군집',sym:'H',team:'군집',ready:1,res:'pass'},
 {n:'5-12',f:'5-12_dbscan.html',t:'DBSCAN',sym:'ε',team:'군집',ready:1,res:'hold'},
 {n:'5-13',f:'5-13_gmm.html',t:'GMM',sym:'N',team:'군집',ready:1,res:'pass'},
 {n:'5-14',f:'5-14_pca.html',t:'PCA',sym:'PC',team:'차원축소',ready:1,res:'pass'},
 {n:'5-15',f:'5-15_tsne_umap.html',t:'t-SNE·UMAP',sym:'tS',team:'차원축소',ready:1,res:'pass'}];
var HQ=[['지도 본부 — 정답이 있는 업무',[['입문팀','분류와 회귀를 함께','입문'],['회귀팀','숫자를 맞힌다','회귀'],['분류팀','이름표를 붙인다','분류'],['겸용팀','둘 다 한다','겸용']]],
        ['비지도 본부 — 정답이 없는 업무',[['군집팀','비슷한 것끼리 묶는다','군집'],['차원축소팀','축을 줄여 펼친다','차원축소']]]];
var PASSLINE=198/258; /* 5-0에서 매뉴얼 씨가 봉인된 시험지(258건)의 "35MPa 이상인가"에서 낸 정확도 = 첫 합격선 */
var HIRE={mine:null};
function epIdx(n){for(var i=0;i<EPISODES.length;i++)if(EPISODES[i].n===n)return i;return -1;}
function renderSeries(){var el=$('series');if(!el)return;var cur=epIdx(EP.n);
  el.innerHTML='<a class="home" href="'+(EP.n==='5-0'?'#m8':'5-0_job_posting.html#m8')+'">⌂ 전체 지도 (5-0)</a>'+EPISODES.map(function(e,i){var lab=e.n+' '+e.t;
    if(i===cur)return '<a class="cur" href="#top" aria-current="page">'+lab+'</a>';
    return e.ready?'<a href="'+e.f+'">'+lab+'</a>':'<span class="s" title="면접 예정">'+lab+'</span>';}).join('');}
function stampOf(i){var e=EPISODES[i],cur=epIdx(EP.n);if(i===cur)return HIRE.mine;return e.ready?e.res:null;}
function renderDeptMap(){var el=$('dmap');if(!el)return;var cur=epIdx(EP.n);
  function seat(e,i){var st=stampOf(i),cls='seat'+(e.ready?'':' soon')+(i===cur?' cur':''),
      inner='<span class="sy">'+e.sym+'</span><span><span class="n">'+e.n+(e.ready?'':' · 면접 예정')+'</span>'+e.t+'</span>'+(st?'<span class="st'+(st==='pass'?'':' hold')+'">'+(st==='pass'?/*i18n:verdict*/'합격':st==='hold'?'보류':/*i18n:verdict*/'불합격')+'</span>':'');
    return (e.ready&&i!==cur)?'<a class="'+cls+'" href="'+e.f+'">'+inner+'</a>':'<div class="'+cls+'"'+(i===cur?' aria-current="page"':'')+'>'+inner+'</div>';}
  var h='<div class="hq"><h4>채용 공고</h4><div class="top0">'+seat(EPISODES[0],0)+'<span class="muted" style="font-size:13.5px">기존 직원 매뉴얼 씨(룰 베이스)의 점수가 첫 합격선이 된다</span></div></div>';
  HQ.forEach(function(q){h+='<div class="hq"><h4>'+q[0]+'</h4><div class="teams">'+q[1].map(function(t){return '<div class="team"><b>'+t[0]+' · '+t[1]+'</b>'+EPISODES.map(function(e,i){return e.team===t[2]?seat(e,i):'';}).join('')+'</div>';}).join('')+'</div></div>';});
  el.innerHTML='<div class="dmap">'+h+'</div><p class="foot" style="margin-top:10px">흐린 자리는 아직 면접 전이다. 도장은 각 편에서 면접 결과를 발표하면 찍힌다. 번호는 읽는 순서(쉬운 지원자부터)다.</p>';}
/* 편 끝 이동: 늘 5-0 전체 지도로 돌아가고, 읽는 순서상 앞뒤에서 준비된 편으로 잇는다 */
function renderNext(){var el=$('nextNav');if(!el)return;var cur=epIdx(EP.n),h='';
  if(EP.n!=='5-0')h+='<a href="5-0_job_posting.html#m8">⌂ 전체 지도로 돌아가기 (5-0)</a>';
  for(var i=cur-1;i>=0;i--)if(EPISODES[i].ready&&i>0){h+='<a href="'+EPISODES[i].f+'">← '+EPISODES[i].n+' '+EPISODES[i].t+'</a>';break;}
  var nx=null;for(var j=cur+1;j<EPISODES.length;j++)if(EPISODES[j].ready){nx=EPISODES[j];break;}
  var plan=EPISODES[cur+1];
  if(nx)h+='<a href="'+nx.f+'">다음 면접: '+nx.n+' '+nx.t+' →</a>';
  if(plan&&(!nx||plan.n!==nx.n))h+='<span>읽는 순서상 다음: '+plan.n+' '+plan.t+' (면접 예정)</span>';
  el.innerHTML=h;}
function updateBoard(){var b=$('board');if(!b)return;var n=0,cur=epIdx(EP.n);EPISODES.forEach(function(e,i){var s=stampOf(i);if(s==='pass')n++;});
  $('bP').textContent=n;$('bL').textContent=window.BOARD_LINE||pct(PASSLINE);b.classList.remove('bump');void b.offsetWidth;if(!REDUCE&&HIRE.mine)b.classList.add('bump');renderDeptMap();}

/* ══ 면접 결과 발표 ═══════════════════════════════════════════════ */
function Interview(id,o){var el=$(id);
  el.innerHTML='<div class="iv-head"><span class="lab">면접 결과 발표 · '+EP.n+'</span><h3>'+o.title+'</h3></div><p class="iv-q">'+o.q+'</p>'+
   '<div class="iv-panel"><div class="ask fly"><img data-who="fly" data-mood="point" alt="" aria-hidden="true"><span class="nm">실무 면접관 · 초파리</span><q>'+o.fly+'</q><span class="ans" hidden></span></div>'+
   '<div class="ask robo"><img data-who="robo" data-mood="calc" alt="" aria-hidden="true"><span class="nm">예산 담당 · 주판봇</span><q>'+o.robo+'</q><span class="ans" hidden></span></div></div>'+
   '<div class="iv-go"><button class="btn go">면접 결과 발표 ▶</button><span class="muted" style="font-size:13.5px">합격일지 먼저 맞혀 보세요.</span><div class="prog" hidden><i></i></div></div><div class="result" hidden aria-live="polite"></div>';
  hydrate(el);
  var btn=el.querySelector('.go'),prog=el.querySelector('.prog'),bar=prog.querySelector('i'),res=el.querySelector('.result'),done=false;
  function setP(p){bar.style.width=Math.round(p*100)+'%';}
  async function open(){if(done)return;done=true;btn.disabled=true;btn.textContent='채점 중…';prog.hidden=false;await wait(30);
    var out=await o.run(setP);prog.hidden=true;btn.textContent='발표 완료';btn.parentNode.querySelector('.muted').textContent='';
    var w=out.result,lab=w==='pass'?/*i18n:verdict*/'합격':w==='hold'?'보류':/*i18n:verdict*/'불합격';
    var fa=el.querySelector('.fly .ans'),ra=el.querySelector('.robo .ans');fa.innerHTML=out.flyAns;ra.innerHTML=out.roboAns;fa.hidden=ra.hidden=false;
    el.querySelector('.fly img').src=sprite('fly',out.flyMood||'arms');el.querySelector('.robo img').src=sprite('robo',out.roboMood||'happy');react(el.querySelector('.fly img'),out.flyMove||(w==='pass'?'jump':'shake'));react(el.querySelector('.robo img'),out.roboMove||(w==='pass'?'pop':'tilt'));
    res.innerHTML='<div class="hstamp'+(w==='pass'?'':' hold')+'" aria-hidden="true">'+lab+'</div><div class="rbox"></div>'+say('asst',out.asstMood||(w==='pass'?'ledger':'sigh'),out.measure,' style="margin-top:12px"')+
      '<div class="verdict"><img class="mv-nod go" src="'+sprite('doc',out.docMood||(w==='pass'?'judge':'scale'))+'" alt="" aria-hidden="true"><div class="bub"><span class="who">위원장 · 판정</span>'+out.verdict+'<span class="key">'+out.key+'</span></div></div>'+
      (o.alts?'<div class="alt"><span class="lab">조건 바꿔 다시 보기</span>'+o.alts.map(function(a,i){return '<button class="btn" data-alt="'+i+'" aria-pressed="false">'+a.label+'</button>';}).join('')+'</div><div class="altout"></div>':'');
    res.hidden=false;out.render(res.querySelector('.rbox'));
    var st=res.querySelector('.hstamp');requestAnimationFrame(function(){st.classList.add('on');});
    HIRE.mine=w;updateBoard();
    if(o.alts)res.querySelector('.alt').addEventListener('click',async function(e){var b=e.target.closest('[data-alt]');if(!b)return;var a=o.alts[+b.getAttribute('data-alt')],box=res.querySelector('.altout');
      res.querySelectorAll('[data-alt]').forEach(function(x){x.setAttribute('aria-pressed',x===b?'true':'false');});box.innerHTML='<p class="muted">계산 중…</p>';await wait(20);await a.run(box);});}
  btn.addEventListener('click',open);el._open=open;}
function statGrid(items,n){return '<div class="stat"'+(n?' style="grid-template-columns:repeat('+n+',minmax(0,1fr))"':'')+'>'+items.map(function(s){return '<div class="'+(s[2]||'')+'"><b>'+s[0]+'</b><span>'+s[1]+'</span></div>';}).join('')+'</div>';}

/* ══ 확인 문제 ════════════════════════════════════════════════════ */
function renderQuiz(){document.querySelectorAll('[data-quiz]').forEach(function(box){var qs=QUIZ[box.getAttribute('data-quiz')];if(!qs)return;
  box.innerHTML=qs.map(function(q,i){return '<div class="qz"><div class="q">Q'+(i+1)+'. '+q.q+'</div><div class="opts">'+q.o.map(function(o,k){return '<button class="o" data-ok="'+(o[1]?1:0)+'">'+o[0]+'<span class="why" hidden><b>'+(o[1]?'정답.':'오답.')+'</b>'+o[2]+'</span></button>';}).join('')+'</div></div>';}).join('');
  box.addEventListener('click',function(e){var b=e.target.closest('button.o');if(!b)return;var qz=b.closest('.qz');
    qz.querySelectorAll('button.o').forEach(function(x){var good=x.getAttribute('data-ok')==='1';x.classList.toggle('ok',good);x.classList.toggle('no',!good&&x===b);x.querySelector('.why').hidden=false;});});});}

/* ══ 복사·내려받기 ════════════════════════════════════════════════ */
function copyText(txt,msg){function fb(){var ta=document.createElement('textarea');ta.value=txt;ta.style.position='fixed';ta.style.opacity='0';document.body.appendChild(ta);ta.select();try{document.execCommand('copy');toast(msg||'복사했습니다');}catch(err){toast('직접 선택해 복사해 주세요');}ta.remove();}
  try{navigator.clipboard.writeText(txt).then(function(){toast(msg||'복사했습니다')},fb);}catch(err){fb();}}
function download(name,txt,mime){try{var bl=new Blob([txt],{type:mime||'text/plain'}),a=document.createElement('a');a.href=URL.createObjectURL(bl);a.download=name;document.body.appendChild(a);a.click();setTimeout(function(){URL.revokeObjectURL(a.href);a.remove();},400);toast(name+' 저장');}catch(err){toast('저장하지 못했습니다. 복사를 써 주세요');}}

/* ══ 공통 A: Colab 코드 ════════════════════════════════════════════ */
var KW=/\b(import|from|as|def|return|for|in|if|else|elif|lambda|with|while|True|False|None|and|or|not|print|range|try|except)\b/g;
function hl(code){return code.split('\n').map(function(line){var out='',i=0,q=null,start=0;
  for(;i<line.length;i++){var ch=line[i];
    if(q){if(ch==='\\'){i++;continue;}if(ch===q){out+='<span class="s">'+esc(line.slice(start,i+1))+'</span>';q=null;start=i+1;}continue;}
    if(ch==='"'||ch==="'"){out+=esc(line.slice(start,i)).replace(KW,'<span class="k">$1</span>');q=ch;start=i;continue;}
    if(ch==='#'){out+=esc(line.slice(start,i)).replace(KW,'<span class="k">$1</span>')+'<span class="c">'+esc(line.slice(i))+'</span>';start=line.length;break;}}
  if(start<line.length)out+=q?'<span class="s">'+esc(line.slice(start))+'</span>':esc(line.slice(start)).replace(KW,'<span class="k">$1</span>');
  return out;}).join('\n');}
function notebook(c){var cells=[{cell_type:'markdown',metadata:{},source:['# '+c.title+'\n','\n','모델 채용 위원회 '+EP.n+' · 글 안상선. 위에서부터 차례로 실행한다(Shift + Enter).']}];
  var chunks=[],cur=[];c.text.split('\n').forEach(function(l){if(/^# -{6,}/.test(l)&&cur.length){chunks.push(cur);cur=[];}cur.push(l);});if(cur.length)chunks.push(cur);
  chunks.forEach(function(ch){while(ch.length&&ch[ch.length-1].trim()==='')ch.pop();cells.push({cell_type:'code',execution_count:null,metadata:{},outputs:[],source:ch.map(function(l,i){return l+(i<ch.length-1?'\n':'')})});});
  return JSON.stringify({cells:cells,metadata:{kernelspec:{display_name:'Python 3',language:'python',name:'python3'},language_info:{name:'python'}},nbformat:4,nbformat_minor:5},null,1);}
function renderCode(){var box=$('codebox');if(!box||!window.CODE)return;var c=CODE,src=$('src-code').textContent.replace(/^\n/,'');c.text=src;
  box.innerHTML='<div class="codepanel"><div class="codehead"><div><b>'+c.title+'</b><span>'+c.sub+'</span></div><div class="codebtns"><button class="btn" data-a="copy">복사</button><button class="btn" data-a="py">.py 저장</button><button class="btn" data-a="nb">노트북(.ipynb) 저장</button></div></div>'+
   '<p style="margin:10px 0 0;font-size:15px">'+c.desc+'</p><pre class="code" tabindex="0"><code>'+hl(src)+'</code></pre>'+
   '<figure class="runshot"><pre>'+esc(CODE_OUT)+'</pre><figcaption>'+c.cap+'</figcaption></figure></div>';
  box.addEventListener('click',function(e){var b=e.target.closest('[data-a]');if(!b)return;var a=b.getAttribute('data-a'),name=EP.n+'_'+c.file;
    if(a==='copy')copyText(src,'코드를 복사했습니다');else if(a==='py')download(name+'.py',src,'text/x-python');else download(name+'.ipynb',notebook(c),'application/x-ipynb+json');});}

/* ══ 공통 B: 프롬프트 ══════════════════════════════════════════════ */
function renderPrompts(){var el=$('prompts');if(!el)return;
  el.innerHTML=PROMPTS.map(function(o,i){return '<article class="pcard"><h4>'+o.t+'</h4><span class="muted" style="font-size:13.5px">'+o.use+'</span><pre>'+esc(o.p)+'</pre><div class="bts"><button class="btn" data-c="'+i+'">복사</button><button class="btn" data-cc="'+i+'">프롬프트 + 코드 복사</button></div></article>';}).join('');
  el.addEventListener('click',function(e){var b=e.target.closest('button');if(!b)return;var c=b.getAttribute('data-c'),cc=b.getAttribute('data-cc');
    if(c!=null)copyText(PROMPTS[+c].p);if(cc!=null)copyText(PROMPTS[+cc].p+'\n\n```python\n'+$('src-code').textContent.replace(/^\n/,'')+'\n```','프롬프트와 코드를 복사했습니다');});}

/* ══ 콘크리트 전문용어 (모든 편 공통 용어 사전) ════════════════════════
   편마다 GL에 같은 용어가 있어도 이 정의가 우선한다(편끼리 뜻이 어긋나지 않게). */
var DOMAIN_GL=[
 ['콘크리트','재령','콘크리트를 부어 만든 뒤 강도를 잴 때까지 지난 날수. 쉽게 말해 "며칠 굳혔나". 처음 며칠은 빠르게, 28일 뒤로는 천천히 강해진다. 이 데이터에는 1~365일이 있다.'],
 ['콘크리트','28일 강도','재령 28일에 잰 압축 강도. 현장에서 콘크리트의 합격 여부를 판정하는 기준으로 가장 흔히 쓴다.'],
 ['콘크리트','압축 강도','콘크리트 시편을 눌러 부서질 때까지 버틴 힘을 면적으로 나눈 값. 단위는 MPa. 클수록 단단하다. 이 시리즈의 회귀 타겟이다.'],
 ['콘크리트','MPa','메가파스칼. 압력·강도의 단위로 1MPa는 1㎟에 약 0.1kg의 무게가 누르는 정도다. 일반 건물 콘크리트는 대개 20~40MPa다.'],
 ['콘크리트','결합재','물과 반응해 굳으면서 재료를 붙여 주는 가루. 이 데이터에서는 시멘트 + 고로 슬래그 + 플라이 애시.'],
 ['콘크리트','물-결합재비','물의 양 ÷ 결합재의 양. 낮을수록(물이 적을수록) 강해진다. 현장 기술자가 강도를 가늠할 때 가장 먼저 보는 숫자다.'],
 ['콘크리트','물-시멘트비','물의 양 ÷ 시멘트의 양. 물-결합재비의 옛 형태로, 슬래그·플라이 애시를 섞은 배합에서는 강도를 잘 설명하지 못한다.'],
 ['콘크리트','시멘트','물과 반응해(수화) 굳는 회색 가루. 콘크리트를 단단하게 만드는 주재료다.'],
 ['콘크리트','고로 슬래그','제철소 용광로(고로)에서 쇠를 뽑고 남은 찌꺼기를 곱게 간 가루. 시멘트 일부를 대신하며, 초기 강도는 느리게 나오지만 오래 두면 강해진다.'],
 ['콘크리트','플라이 애시','화력 발전소에서 석탄을 태우고 굴뚝으로 날아가는 고운 재를 모은 것. 시멘트 일부를 대신하며, 고로 슬래그처럼 강도가 늦게 오른다.'],
 ['콘크리트','혼화재','시멘트 일부를 대신하거나 성질을 바꾸려고 섞는 가루 재료. 고로 슬래그와 플라이 애시가 대표적이다.'],
 ['콘크리트','고성능 감수제','적은 양으로 물을 덜 넣어도 콘크리트가 잘 흐르게 해 주는 약제. 물을 줄일 수 있으니 결과적으로 강도를 높인다.'],
 ['콘크리트','굵은 골재','콘크리트 속의 자갈. 대략 5mm보다 큰 돌로, 부피의 큰 몫을 차지해 뼈대 역할을 한다.'],
 ['콘크리트','잔골재','콘크리트 속의 모래. 대략 5mm보다 작은 알갱이로 자갈 사이 빈틈을 메운다.'],
 ['콘크리트','배합','시멘트·물·골재·혼화재를 몇 kg씩 섞을지 정한 조합. 이 데이터의 한 줄이 배합 하나(와 그 시편의 강도)다.'],
 ['콘크리트','시편','강도를 재려고 같은 배합으로 따로 만든 작은 원기둥 시험체. 정해진 재령에 눌러 부순다.'],
 ['콘크리트','양생','부은 콘크리트가 잘 굳도록 온도와 물기를 지켜 주는 일. 양생 기간이 곧 재령과 이어진다.'],
 ['콘크리트','수화','시멘트가 물과 만나 천천히 굳는 화학 반응. 재령이 길수록 반응이 더 진행돼 강해진다.'],
 ['콘크리트','레미콘','공장에서 배합해 트럭으로 현장까지 실어 나르는 굳지 않은 콘크리트(ready-mixed concrete).'],
 ['콘크리트','에이브럼스 식','"물-시멘트비가 낮을수록 강하다"를 식으로 쓴 경험식(1918). 강도 = A ÷ B^(물-시멘트비). 매뉴얼 씨는 물-결합재비로 넓혀 쓴다.']];
function glossAll(){var names={};DOMAIN_GL.forEach(function(d){names[d[1]]=1;});return DOMAIN_GL.concat((window.GL||[]).filter(function(g){return !names[g[1]];}));}

/* ══ 공통 C: 용어 + 말풍선 ═════════════════════════════════════════ */
function renderGloss(){if(!$('glossary'))return;var GL=glossAll(),DOMN={},PAGEUSED={};DOMAIN_GL.forEach(function(d){DOMN[d[1]]=1;});
  $('glossary').innerHTML='<thead><tr><th>용어</th><th>묶음</th><th>뜻</th></tr></thead><tbody>'+GL.map(function(g){return '<tr><td>'+g[1]+'</td><td style="white-space:nowrap">'+g[0]+'</td><td>'+g[2]+'</td></tr>';}).join('')+'</tbody>';
  var MAP={};GL.forEach(function(g){MAP[g[1]]=g;});var terms=Object.keys(MAP).sort(function(a,b){return b.length-a.length});
  var RX=new RegExp('('+terms.map(function(t){return t.replace(/[-\/\\^$*+?.()|[\]{}]/g,'\\$&')}).join('|')+')');
  document.querySelectorAll('section.part:not(.common) p, .lede, .pair p, .explain li').forEach(function(p){if(p.closest('.interview,.quiz,.say,.verdict,.sheet'))return;var used=Object.assign({},PAGEUSED),w=document.createTreeWalker(p,NodeFilter.SHOW_TEXT,null),list=[],n;while((n=w.nextNode()))if(RX.test(n.nodeValue)&&!n.parentElement.closest('a,b,.str,.jt,button,code'))list.push(n);
    list.forEach(function(t){var parts=t.nodeValue.split(RX),frag=document.createDocumentFragment(),ch=false;parts.forEach(function(s,i){if(i%2&&!used[s]){used[s]=1;if(DOMN[s])PAGEUSED[s]=1;ch=true;var sp=document.createElement('span');sp.className='jt';sp.tabIndex=0;sp.setAttribute('data-k',s);sp.textContent=s;frag.appendChild(sp);}else if(s)frag.appendChild(document.createTextNode(s));});if(ch)t.parentNode.replaceChild(frag,t);});});
  var tip=document.createElement('div');tip.id='tip';tip.setAttribute('role','tooltip');document.body.appendChild(tip);var cur=null;
  function show(el){var g=MAP[el.getAttribute('data-k')];if(!g)return;cur=el;tip.innerHTML='<b>'+g[1]+'</b><small>'+g[0]+'</small>'+g[2];tip.classList.add('on');var r=el.getBoundingClientRect(),tw=tip.offsetWidth,th=tip.offsetHeight,x=clamp(r.left+r.width/2-tw/2,10,window.innerWidth-tw-10),y=r.top-th-10;if(y<70)y=r.bottom+10;tip.style.left=x+'px';tip.style.top=y+'px';}
  function hide(){tip.classList.remove('on');cur=null;}
  document.addEventListener('mouseover',function(e){var el=e.target.closest&&e.target.closest('.jt');if(el)show(el);else if(cur)hide();});
  document.addEventListener('focusin',function(e){var el=e.target.closest&&e.target.closest('.jt');if(el)show(el);});
  document.addEventListener('focusout',function(e){if(e.target.closest&&e.target.closest('.jt'))hide();});
  document.addEventListener('click',function(e){var el=e.target.closest&&e.target.closest('.jt');if(el){cur===el?hide():show(el);}});
  window.addEventListener('scroll',function(){if(cur)hide();},{passive:true});document.addEventListener('keydown',function(e){if(e.key==='Escape')hide();});}

/* ══ 공통 D·E·F: 비유 · 레퍼런스 · 교육자 TIP ══════════════════════ */
var STRN={3:['직접','s3'],2:['부분','s2'],1:['없음','s1']};
function renderAppendix(){
  if($('analogy'))$('analogy').innerHTML='<thead><tr><th>개념</th><th>면접·일상 비유</th><th>초파리·뇌 비유</th><th>비유가 틀리는 지점</th></tr></thead><tbody>'+ANALOGY.map(function(a){var s=STRN[a[2]];return '<tr><td><b style="color:var(--text)">'+a[0]+'</b></td><td>'+a[1]+'</td><td><span class="str '+s[1]+'">'+s[0]+'</span> '+(a[3]||'')+'</td><td>'+a[4]+'</td></tr>';}).join('')+'</tbody>';
  if($('refs'))$('refs').innerHTML=REFS.map(function(r){return '<li>'+r+'</li>';}).join('');
  if($('tipbox')){var T=TIP;$('tipbox').innerHTML='<div class="card tw" style="padding:6px 12px"><table class="tb"><tbody><tr><th style="width:120px">대상</th><td>'+T.who+'</td></tr><tr><th>학습 목표</th><td>'+T.goal+'</td></tr><tr><th>예상 시간</th><td>'+T.time+'</td></tr></tbody></table></div>'+
    '<h3 style="margin:22px 0 10px;font-size:17px">수업 운영안</h3><div class="tipgrid">'+T.plans.map(function(p){return '<div><h4>'+p[0]+'</h4><p>'+p[1]+'</p></div>';}).join('')+'</div>'+
    '<h3 style="margin:22px 0 10px;font-size:17px">흔한 오해</h3><div class="card tw" style="padding:6px 12px"><table class="tb"><thead><tr><th>오해</th><th>바른 설명</th></tr></thead><tbody>'+T.myths.map(function(m){return '<tr><td>'+m[0]+'</td><td>'+m[1]+'</td></tr>';}).join('')+'</tbody></table></div>'+
    (T.tasks?'<h3 style="margin:22px 0 10px;font-size:17px">과제 · 생성형 AI 활용 수업</h3><ul class="tasks">'+T.tasks.map(function(t){return '<li><b>'+t[0]+'</b>'+t[1]+'</li>';}).join('')+'</ul>':'')+
    (T.ai?'<div class="explain" style="margin-top:16px"><h4>수업에서 생성형 AI를 쓸 때</h4><p>'+T.ai+'</p></div>':'');}}

/* ══ 지원서 1장: 인쇄 · 칸 이동 ═══════════════════════════════════ */
function sheetTools(){var b=$('sheetPrint');if(b)b.addEventListener('click',function(){document.body.classList.add('print-sheet');setTimeout(function(){window.print();setTimeout(function(){document.body.classList.remove('print-sheet');},400);},50);});}

/* ══ 언어 선택 (한국어 원본 + 번역판). 번역이 준비된 언어만 고를 수 있다 ══ */
var LANGS=[['ko','한국어'],['en','English'],['zh','中文'],['ja','日本語'],['uz',"O'zbekcha"],['ru','Русский']];
function langHref(to){var f=location.pathname.split('/').pop()||'index.html';return (LANG==='ko'?'':'../')+(to==='ko'?'':to+'/')+f+location.hash;}
(function(){var ok=LANG_READY.slice(),q=null;try{q=new URLSearchParams(location.search).get('lang');}catch(e){}
  if(q&&ok.indexOf(q)>=0){try{localStorage.setItem('mhc-lang',q);}catch(e){}if(q!==LANG){location.replace(langHref(q));return;}}
  var saved=null;try{saved=localStorage.getItem('mhc-lang');}catch(e){}
  if(saved&&!q&&LANG==='ko'&&saved!=='ko'&&ok.indexOf(saved)>=0){location.replace(langHref(saved));return;} /* 기본 주소에 다시 오면 전에 고른 언어로 */
  if(!saved&&!q&&LANG==='ko'&&location.protocol!=='file:'){ /* 기본(한국어) 주소로 들어왔을 때만 브라우저 언어로 보낸다. 언어 폴더 주소를 직접 열면 그 언어를 존중 */var nav=(navigator.languages||[navigator.language||'']).map(function(s){return (s||'').slice(0,2).toLowerCase()});
    for(var i=0;i<nav.length;i++){if(ok.indexOf(nav[i])>=0){if(nav[i]!==LANG){try{localStorage.setItem('mhc-lang',nav[i]);}catch(e){}location.replace(langHref(nav[i]));return;}break;}}}
  function mount(){var host=document.querySelector('nav.top .wrap');if(!host||$('langSel'))return;var s=document.createElement('select');s.id='langSel';s.className='langsel';s.setAttribute('aria-label','Language / 언어');
    LANGS.forEach(function(l){var o=document.createElement('option');o.value=l[0];var rd=ok.indexOf(l[0])>=0;o.textContent=l[1]+(rd?'':' (준비 중)');o.disabled=!rd;if(l[0]===LANG)o.selected=true;s.appendChild(o);});
    s.addEventListener('change',function(){try{localStorage.setItem('mhc-lang',s.value);}catch(e){}if(s.value!==LANG)location.href=langHref(s.value);});host.appendChild(s);}
  if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',mount);else mount();})();

/* ══ 히어로: 면접실 (코드로 그린 장면 + 기존 캐릭터 그림) ═════════════ */
function heroRoom(cv,opt){var ctx=cv.getContext('2d'),t=0,IM={},W=1100,H=330;
  /* opt.beats: [{at:초, who:'fly', mood:'point', move:'jump|shake|lean|stand|pop', say:'짧은 말'}] — opt.loop초마다 되풀이. 등장(0~1.4초)에는 위원이 차례로 튀어나오고 사원증이 미끄러져 들어온다 */
  var seats=[['doc',opt.doc||'front','위원장'],['asst',opt.asst||'board','인사'],['fly',opt.fly||'clip','실무'],['robo',opt.robo||'calc','예산']];
  var BE=(opt.beats||[]).slice().sort(function(a,b){return a.at-b.at}),LOOP=opt.loop||12,cur={};
  function load(who,mood){var k=who+'_'+mood;if(IM[k]!==undefined)return;IM[k]=null;var i=new Image();i.onload=function(){IM[k]=i;draw();};i.src=sprite(who,mood);}
  seats.forEach(function(s){load(s[0],s[1]);});BE.forEach(function(e){load(e.who,e.mood||DEFAULT[e.who].split('_')[1]);});
  function state(who,base){var tt=REDUCE?(BE.length?BE[0].at+.01:0):(t<1.4?0:(t-1.4)%LOOP),st={mood:base,move:null,since:0,say:null};
    BE.forEach(function(e){if(e.who===who&&e.at<=tt){st.mood=e.mood||st.mood;st.move=e.move||null;st.since=tt-e.at;st.say=e.say&&tt-e.at<3.2?e.say:null;}});return st;}
  function moveOff(m,s){if(!m||REDUCE)return {x:0,y:0,r:0};if(m==='jump')return {x:0,y:s<.7?-Math.abs(Math.sin(s*9))*16:0,r:0};if(m==='shake')return {x:s<.8?Math.sin(s*40)*4:0,y:0,r:0};
    if(m==='lean')return {x:0,y:0,r:Math.min(1,s*4)*-.12};if(m==='stand')return {x:0,y:-Math.min(1,s*5)*16,r:0};if(m==='pop')return {x:0,y:0,r:0,sc:s<.4?1+Math.sin(s/.4*Math.PI)*.15:1};if(m==='bob')return {x:0,y:Math.sin(s*5)*5,r:0};if(m==='sway')return {x:0,y:0,r:Math.sin(s*4)*.08};if(m==='nod')return {x:0,y:s<1?Math.abs(Math.sin(s*8))*5:0,r:0};if(m==='tilt')return {x:0,y:0,r:Math.min(1,s*4)*.14};return {x:0,y:0,r:0};}
  function draw(){var C=cols(),dark=getComputedStyle(document.documentElement).colorScheme==='dark';ctx.setTransform(1,0,0,1,0,0);
    /* 벽 · 창 · 바닥 */
    var g=ctx.createLinearGradient(0,0,0,H);g.addColorStop(0,dark?'#1a2232':'#e9e2d2');g.addColorStop(1,dark?'#121826':'#ddd3bf');ctx.fillStyle=g;ctx.fillRect(0,0,W,H);
    ctx.fillStyle=dark?'#0d1320':'#cfd9e6';rr(ctx,44,34,190,130,6);ctx.fill();ctx.strokeStyle=dark?'#33415a':'#9aa8bb';ctx.lineWidth=4;ctx.stroke();
    ctx.strokeStyle=dark?'rgba(150,170,210,.18)':'rgba(255,255,255,.75)';ctx.lineWidth=3;for(var y=46;y<160;y+=12){ctx.beginPath();ctx.moveTo(50,y);ctx.lineTo(228,y+((y/12)%2?1:0));ctx.stroke();}
    if(dark){for(var k=0;k<14;k++){ctx.fillStyle='rgba(255,230,150,'+(0.25+0.25*Math.sin(t*1.3+k))+')';ctx.fillRect(60+((k*37)%160),60+((k*53)%90),3,3);}}
    ctx.fillStyle=dark?'#0f1522':'#cbbfa6';ctx.fillRect(0,H-64,W,64);
    /* 현수막 */
    ctx.fillStyle=C.ink;rr(ctx,330,26,520,46,8);ctx.fill();ctx.fillStyle=C.bg;ctx.font=font(22,700);ctx.textAlign='center';ctx.fillText(opt.banner||'모델 채용 위원회',590,57);
    /* 시계 */
    circ(ctx,960,62,26,dark?'#1e2736':'#fffdf8',C.ink,3);var a=t*0.4;ctx.strokeStyle=C.ink;ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(960,62);ctx.lineTo(960+14*Math.cos(a-1.5),62+14*Math.sin(a-1.5));ctx.moveTo(960,62);ctx.lineTo(960+20*Math.cos(a*12-1.5),62+20*Math.sin(a*12-1.5));ctx.stroke();
    /* 위원회 탁자 뒤 네 명 */
    var x0=470,dx=150,ty=236;
    var bubbles=[];
    seats.forEach(function(s,i){var st=state(s[0],s[1]),im=IM[s[0]+'_'+st.mood]||IM[s[0]+'_'+s[1]],cx=x0+i*dx;if(!im)return;var h=s[0]==='doc'?118:132,w=im.width*h/im.height,bob=REDUCE?0:Math.sin(t*1.6+i)*1.6,mo=moveOff(st.move,st.since);
      var ent=REDUCE?1:clamp((t-i*.22)/.5,0,1),sc=(mo.sc||1)*(0.55+0.45*ease(ent));ctx.save();ctx.globalAlpha=ent;ctx.translate(cx+mo.x,ty+18+bob+mo.y);ctx.rotate(mo.r);ctx.scale(sc,sc);ctx.imageSmoothingEnabled=true;ctx.drawImage(im,-w/2,-h,w,h);ctx.restore();
      if(st.say)bubbles.push([cx,ty-h+8+mo.y,st.say,s[0]]);});
    ctx.fillStyle=dark?'#3a2d22':'#7a5a3a';rr(ctx,x0-90,ty,dx*4+40,22,4);ctx.fill();ctx.fillStyle=dark?'#2a2019':'#5f452c';ctx.fillRect(x0-80,ty+22,dx*4+20,50);
    seats.forEach(function(s,i){var cx=x0+i*dx;ctx.fillStyle=dark?'#e8ecf3':'#fffdf8';rr(ctx,cx-38,ty-12,76,22,3);ctx.fill();ctx.fillStyle=C['c_'+s[0]]||C.ink;ctx.font=font(13,700);ctx.textAlign='center';ctx.fillText(s[2],cx,ty+4);});
    /* 지원자 자리: 의자 + 큰 사원증 */
    var ax=190-(REDUCE?0:(1-ease(clamp(t/1.1,0,1)))*260),ay=212;ctx.fillStyle=dark?'#2e3a4e':'#8f7e64';rr(ctx,ax-46,ay+34,92,14,4);ctx.fill();ctx.fillRect(ax-40,ay+48,8,46);ctx.fillRect(ax+32,ay+48,8,46);rr(ctx,ax-42,ay-40,84,76,8);ctx.fill();
    var by=ay-70+(REDUCE?0:Math.sin(t*2)*2);ctx.strokeStyle=C.c_app;ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(ax-18,by-30);ctx.lineTo(ax,by);ctx.lineTo(ax+18,by-30);ctx.stroke();
    ctx.fillStyle=dark?'#161d29':'#fffdf8';rr(ctx,ax-62,by,124,96,10);ctx.fill();ctx.lineWidth=3;ctx.stroke();
    ctx.fillStyle=opt.man?C.c_man:C.c_app;rr(ctx,ax-50,by+12,44,44,8);ctx.fill();ctx.fillStyle='#fff';ctx.font=font(opt.sym&&opt.sym.length>1?18:26,700,1);ctx.textAlign='center';ctx.fillText(opt.sym||'σ',ax-28,by+43);
    ctx.fillStyle=C.muted;ctx.font=font(11,500,1);ctx.textAlign='left';ctx.fillText(opt.no||'',ax,by+26);ctx.fillStyle=C.text;ctx.font=font(13,700);
    (opt.name||'').split('\n').forEach(function(l,i){ctx.fillText(l,ax,by+44+i*16);});
    ctx.fillStyle=opt.man?C.c_man:C.c_app;ctx.font=font(12,700);ctx.textAlign='center';ctx.fillText(opt.role||'지원자',ax,by+84);
    if(opt.motif){ctx.save();opt.motif(ctx,t,W,H,C);ctx.restore();}
    bubbles.forEach(function(bb){ctx.font=font(13,700);var tw=ctx.measureText(bb[2]).width+18,bx=clamp(bb[0]-tw/2,8,W-tw-8),by2=Math.max(6,bb[1]-34);ctx.fillStyle=dark?'#1e2736':'#fffdf8';ctx.strokeStyle=C['c_'+bb[3]]||C.ink;ctx.lineWidth=2;rr(ctx,bx,by2,tw,26,8);ctx.fill();ctx.stroke();
      ctx.beginPath();ctx.moveTo(bb[0]-6,by2+26);ctx.lineTo(bb[0],by2+34);ctx.lineTo(bb[0]+6,by2+26);ctx.fill();ctx.fillStyle=C.text;ctx.textAlign='left';ctx.fillText(bb[2],bx+9,by2+18);});}
  draw();onRedraw(draw);if(!REDUCE)onTick(cv,function(dt){t+=dt;draw();});}

/* ══ 연출: 말풍선 등장 · 그림 동작 · 네 컷 장면 ═════════════════════════
   say 이미지에 data-move="jump|shake|tilt|nod|pop|sway|bob"을 달면 화면에 들어올 때 그 동작을 한다.
   comic('host',[{bg:'office|lab|site|night|party|board', cast:[{who,mood,x:0~100,move,flip,h}], say:{who,txt}, cap:'설명', prop:'소품 글자'}]) */
var MOVES=['jump','shake','tilt','nod','pop','sway','bob'];
function revealMotion(root){root=root||document;var els=root.querySelectorAll('.say,.comic .pnl');if(REDUCE||!('IntersectionObserver' in window)){els.forEach(function(e){e.classList.add('in');});return;}
  var io=new IntersectionObserver(function(es){es.forEach(function(e){if(e.isIntersecting){e.target.classList.add('in');io.unobserve(e.target);}});},{rootMargin:'0px 0px -8% 0px'});els.forEach(function(e){if(!e.classList.contains('in')){e.classList.add('pre');io.observe(e);}});}
var SCENEBG={office:['#efe7d6','#d9ccb3','#1d2533','#141a26'],lab:['#e3edf5','#c7d6e6','#152030','#0e1622'],site:['#efe2cf','#cdb48f','#262019','#18130e'],night:['#dfe3ee','#b9c1d6','#141826','#0b0e18'],party:['#f6e7ef','#e9c9d8','#2a1b26','#1a1018'],board:['#eef2ea','#cfd9c6','#1b241d','#111812']};
function comic(id,panels){var el=typeof id==='string'?$(id):id;if(!el)return;el.classList.add('comic');
  el.innerHTML=panels.map(function(p,i){var b=SCENEBG[p.bg||'office'];
    return '<figure class="pnl" style="--p1:'+b[0]+';--p2:'+b[1]+';--d1:'+b[2]+';--d2:'+b[3]+';--dl:'+(i*0.18)+'s"><div class="stage">'+(p.prop?'<span class="prop">'+p.prop+'</span>':'')+
      (p.cast||[]).map(function(c){return '<img data-who="'+c.who+'" data-mood="'+c.mood+'" alt="" aria-hidden="true" class="'+(c.move?'mv-'+c.move:'')+(c.flip?' flip':'')+'" style="left:'+c.x+'%;height:'+(c.h||118)+'px">';}).join('')+
      (p.say?'<div class="cbub '+p.say.who+'"><b>'+CAST_SHORT[p.say.who]+'</b> '+p.say.txt+'</div>':'')+
      '</div>'+(p.cap?'<figcaption>'+p.cap+'</figcaption>':'')+'</figure>';}).join('');
  hydrate(el);el.setAttribute('role','group');}
function react(img,move){if(!img||REDUCE)return;MOVES.forEach(function(m){img.classList.remove('mv-'+m);});void img.offsetWidth;img.classList.add('mv-'+move,'go');}

/* ══ 시작 ═════════════════════════════════════════════════════════ */
function boot(){renderSeries();renderNext();updateBoard();renderQuiz();renderCode();renderPrompts();renderAppendix();renderGloss();hydrate(document);sheetTools();revealMotion(document);
  if(location.hash){var id=decodeURIComponent(location.hash.slice(1));setTimeout(function(){var e=$(id);if(e)e.scrollIntoView();},80);}}
