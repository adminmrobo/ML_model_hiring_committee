#!/usr/bin/env python3
"""모델 채용 위원회 5-x 빌드: 공용 조각 + 편별 조각 → 한 장짜리 HTML.
   편마다 그 편이 쓰는 그림만 base64로 넣고, Colab 코드를 실제로 실행해 결과를 페이지에 싣는다.
   사용: python3 build.py            (모든 편)
         python3 build.py 5-4        (한 편만)
         python3 build.py --lang en  (그 언어만, 한국어 빌드 생략)
   번역판: i18n/<언어>/<편>.json 과 i18n/<언어>/common.json 의 번역(tr)을 한국어 원본에 끼워 out/<언어>/에 만든다.
   번역이 없는 조각은 한국어로 남고, 빌드 끝에 언어별 번역률을 보여 준다. (i18n/README.md)"""
import base64, json, os, re, subprocess, sys, html, hashlib
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'i18n'))
import i18n

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC, EPD, ASSETS, OUT, DATA = (os.path.join(ROOT, d) for d in ('src', 'ep', 'assets', 'out', 'data'))
rd = lambda p: open(p, encoding='utf-8').read()
LANGS = ['ko', 'en', 'zh', 'ja', 'uz', 'ru']

CSS = rd(f'{SRC}/base.css') + rd(f'{SRC}/extra.css')
CORE, MLJS, DATAJS, DIGITSJS = rd(f'{SRC}/core.js'), rd(f'{SRC}/ml.js'), rd(f'{SRC}/concrete.js'), rd(f'{SRC}/digits.js')
CORE_SPRITES = {'fly_point', 'fly_arms', 'fly_coffee', 'fly_clip', 'robo_calc', 'robo_happy', 'robo_meh', 'robo_smile',
                'asst_ledger', 'asst_sigh', 'asst_smile', 'asst_board', 'doc_judge', 'doc_scale', 'doc_front'}
AUTHOR = rd(f'{SRC}/author.html')
NOTES_COMMON = [
    ('지원서 1장', '각 편 00장의 9칸 지원서는 이 시리즈를 위해 새로 그렸다. 로지스틱 회귀 편은 널리 쓰이는 9칸 요약 인포그래픽의 구성(이진 분류 → 선형 점수 → 시그모이드 → 확률 → 경계 → 임계값 → 손실 → 학습 → 다중 분류)을 따랐고, 원본 그림과 로고는 옮기지 않았다.'),
    ('그림', '등장인물 그림은 「인공지능 알고리즘 패밀리 조직도」에서 다시 썼다. 면접실 장면, 사원증, 도표는 이 페이지가 코드로 직접 그린다.'),
    ('"채용 위원회" 컨셉', '모델을 지원자에, 학습과 평가를 면접에 빗댄 은유다. 실제 채용 절차나 특정 회사를 나타내지 않는다. 불합격은 "이 일과 맞지 않는다"는 뜻일 뿐 모델의 우열이 아니다.'),
    ('등장인물', 'Money-Robo는 저자가 운영하는 서비스의 마스코트 캐릭터다. 박사·조수·초파리 연구원·매뉴얼 씨·지원자는 실존 인물을 나타내지 않는다. 대사 속 돈·예산 표현은 계산 비용을 빗댄 말버릇이다.'),
    ('데이터', '지도 본부의 실기 과제지는 UCI Machine Learning Repository의 Concrete Compressive Strength 데이터(Yeh, 1998, 1,030건, CC BY 4.0)다. 시험지 나누기는 전 편 공통으로 test_size=0.25, stratify=강도 3등급, random_state=42(훈련 772 · 시험 258)다. 강도 등급(25·45MPa)과 합격 기준(35MPa)은 이 시리즈가 수업용으로 정한 문턱이며 특정 설계 기준을 대신하지 않는다.'),
    ('언어', '이 시리즈는 한국어로 쓴다. 번역판(English·中文·日本語·Oʻzbekcha·Русский)은 한국어판 내용이 확정된 뒤 같은 파일 이름으로 더한다. Colab 코드의 주석과 실행 결과는 한국어 원문 그대로 둔다.'),
]


def notes_section(items, sid='notes'):
    li = ''.join(f'<li><b>{k}.</b> {v}</li>' for k, v in items)
    return (f'<section class="part common notes" id="{sid}">\n<div class="wrap">\n'
            f'  <div class="part-head"><div class="part-no">G</div><div class="en">Notes</div><h2>설정 및 유의사항</h2>'
            f'<p class="lin">이 페이지의 데이터, 그림, 컨셉에 대한 안내</p></div>\n'
            f'  <div class="card"><ul class="reflist">{li}</ul></div>\n</div>\n</section>')


RUNNER = '''import sys, types, pandas as pd
# Colab에서는 ucimlrepo가 UCI에서 데이터를 받는다. 빌드에서는 같은 내용의 로컬 사본을 돌려준다.
_df = pd.read_csv({local!r})
_m = types.ModuleType("ucimlrepo")
def fetch_ucirepo(id=None, **k):
    assert id == 165, id
    ns = types.SimpleNamespace
    return ns(data=ns(features=_df.iloc[:, :8].copy(), targets=_df.iloc[:, 8:].copy()))
_m.fetch_ucirepo = fetch_ucirepo
sys.modules["ucimlrepo"] = _m
sys.argv = [{path!r}]
exec(compile(open({path!r}, encoding="utf-8").read(), {path!r}, "exec"), {{"__name__": "__main__"}})
'''


def run_code(path):
    """Colab 코드를 그대로 실행한다. UCI에서 받는 부분만 같은 내용의 로컬 사본으로 바꾼다.
       코드가 바뀌지 않았으면 지난 실행 결과(out/.code_cache)를 다시 쓴다."""
    h = hashlib.sha1((rd(path) + rd(f'{DATA}/concrete.csv')[:2000]).encode()).hexdigest()
    cp = f'{OUT}/.code_cache/{os.path.basename(os.path.dirname(path))}.json'
    if os.path.exists(cp):
        c = json.loads(rd(cp))
        if c['h'] == h: return c['out']
    o = _run_code(path)
    os.makedirs(os.path.dirname(cp), exist_ok=True)
    open(cp, 'w', encoding='utf-8').write(json.dumps({'h': h, 'out': o}, ensure_ascii=False))
    return o


def _run_code(path):
    env = dict(os.environ, MPLBACKEND='Agg', PYTHONHASHSEED='0', PYTHONWARNINGS='default')
    prog = RUNNER.format(local=f'{DATA}/concrete.csv', path=path)
    r = subprocess.run([sys.executable, '-c', prog], capture_output=True, text=True, timeout=600, env=env, cwd=os.path.dirname(path))
    if r.returncode:
        raise SystemExit(f'코드 실행 실패: {path}\n{r.stderr}')
    return r.stdout.rstrip()


def need_sprites(text):
    s = set(CORE_SPRITES)
    for who, mood in re.findall(r'data-who="(\w+)"\s+data-mood="(\w+)"', text): s.add(f'{who}_{mood}')
    for who, mood in re.findall(r"(?:sprite|say)\('(\w+)','(\w+)'", text): s.add(f'{who}_{mood}')
    for k in re.findall(r'data-scene="(\w+)"', text): s.add(k)
    for who, mood in re.findall(r"who:'(\w+)',\s*mood:'(\w+)'", text): s.add(f'{who}_{mood}')
    for who, mood in re.findall(r"\['(doc|asst|fly|robo)','(\w+)'", text): s.add(f'{who}_{mood}')
    for who in ('doc', 'asst', 'fly', 'robo'):
        for k in re.findall(who + r":'(\w+)'", text): s.add(f'{who}_{k}')
    return s


def b64(key):
    p = f'{ASSETS}/{key}.webp'
    if not os.path.exists(p): return None
    return 'data:image/webp;base64,' + base64.b64encode(open(p, 'rb').read()).decode()


def common_sections():
    return f'''
<!-- ═══════════ 부서 배치도 ═══════════ -->
<section class="part" id="map">
<div class="wrap">
  <div class="part-head"><div class="part-no">▦</div><div class="en">Seating chart</div><h2>부서 배치도 — 열다섯 자리</h2><p class="lin">지도 본부와 비지도 본부, 여섯 팀. 번호는 쉬운 지원자부터 읽는 순서다</p></div>
  <div id="dmap"></div>
</div>
</section>

<!-- ═══════════ 공통 A: Colab 코드 ═══════════ -->
<section class="part common" id="code">
<div class="wrap">
  <div class="part-head"><div class="part-no">A</div><div class="en">Colab code</div><h2>Colab 코드 — 이 면접을 직접 돌려 보기</h2><p class="lin">Colab에 기본으로 설치된 라이브러리만 쓴다. 아래 결과는 이 코드를 실제로 실행한 것이다</p></div>
  <div id="codebox"></div>
</div>
</section>

<!-- ═══════════ 공통 B: 프롬프트 ═══════════ -->
<section class="part common" id="prompt">
<div class="wrap">
  <div class="part-head"><div class="part-no">B</div><div class="en">Prompts</div><h2>AI 응용 프롬프트 — 이 코드를 출발점으로</h2><p class="lin">〔 〕 안을 내 상황으로 바꿔 쓴다</p></div>
  <div class="prompts" id="prompts"></div>
  <p class="muted" style="font-size:14px;margin-top:12px">AI의 답은 출발점일 뿐이다. 제안된 함수·인자·수치는 공식 문서와 실행 결과로 확인한다. 공모전·논문·과제에 AI 도움을 받았다면 규정에 따라 사용 사실과 범위를 밝힌다.</p>
  <div class="toast" id="toast" role="status" aria-live="polite"></div>
</div>
</section>

<!-- ═══════════ 공통 C: 용어 ═══════════ -->
<section class="part common" id="glossary-sec">
<div class="wrap">
  <div class="part-head"><div class="part-no">C</div><div class="en">Glossary</div><h2>용어 정리</h2><p class="lin">본문의 파란 점선 밑줄과 같은 사전</p></div>
  <div class="card tw" style="padding:6px 12px"><table class="tb" id="glossary"></table></div>
</div>
</section>

<!-- ═══════════ 공통 D: 비유 ═══════════ -->
<section class="part common" id="analogy-sec">
<div class="wrap">
  <div class="part-head"><div class="part-no">D</div><div class="en">Analogies</div><h2>비유 모음 — 그리고 비유가 틀리는 지점</h2><p class="lin">초파리·뇌 비유에는 강도를 표시했다: <span class="str s3">직접</span> 같은 일을 한다는 근거가 있음 · <span class="str s2">부분</span> 일부만 닮음 · <span class="str s1">없음</span> 대응이 알려지지 않음</p></div>
  <div class="card tw" style="padding:6px 12px"><table class="tb" id="analogy"></table></div>
</div>
</section>

<!-- ═══════════ 공통 E: 저자와 레퍼런스 ═══════════ -->
<section class="part common" id="refs-sec">
<div class="wrap">
  <div class="part-head"><div class="part-no">E</div><div class="en">References</div><h2>저자 소개와 레퍼런스</h2><p class="lin">E-1 저자 · E-2 레퍼런스</p></div>
  {AUTHOR}
  <div class="card"><h3>E-2 레퍼런스</h3><ol class="reflist" id="refs"></ol></div>
</div>
</section>

<!-- ═══════════ 공통 F: 교육자 TIP ═══════════ -->
<section class="part common" id="teach">
<div class="wrap">
  <div class="part-head"><div class="part-no">F</div><div class="en">For educators</div><h2>교육자를 위한 TIP</h2><p class="lin">시간은 제작자의 추정이다. 수업에 맞춰 조정한다</p></div>
  <div id="tipbox"></div>
</div>
</section>
'''


def src_for(ep, lang, name):
    return f'{EPD}/{ep}/{name}'   # 원본은 한국어 하나. 번역은 i18n/ 카탈로그로 끼운다


I18N = os.path.join(ROOT, 'i18n')
# 언어별 글꼴: 구글 글꼴 이름, CSS 이름. 한국어 글꼴(IBM Plex Sans KR)은 뒤에 그대로 남겨 한글 대비로 둔다
FONTS = {'en': ('IBM+Plex+Sans:wght@400;500;700', '"IBM Plex Sans"'), 'uz': ('IBM+Plex+Sans:wght@400;500;700', '"IBM Plex Sans"'),
         'ru': ('IBM+Plex+Sans:wght@400;500;700', '"IBM Plex Sans"'), 'zh': ('Noto+Sans+SC:wght@400;500;700', '"Noto Sans SC"'),
         'ja': ('Noto+Sans+JP:wght@400;500;700', '"Noto Sans JP"')}


def font_head(lang):
    if lang not in FONTS: return '', ''
    g, n = FONTS[lang]
    disp = n if lang in ('ru', 'zh', 'ja') else '"Black Han Sans"'
    return (f'<link href="https://fonts.googleapis.com/css2?family={g}&display=swap" rel="stylesheet">\n'
            f'<style>:root{{--font:{n},"IBM Plex Sans KR",system-ui,sans-serif;--stampf:{disp},{n},sans-serif}}'
            + 'table.tb td,table.tb th{overflow-wrap:anywhere;hyphens:auto}'
            + ('*{word-break:normal!important;line-break:strict}p,li,small,td,th,label,.bub,figcaption{overflow-wrap:anywhere}' if lang in ('zh', 'ja') else '') + '</style>',
            f"FF='{n},'+FF;")
_TR = {}


def load_tr(lang):
    """i18n/<lang>/*.json 의 tr 값을 {키: 번역문} 하나로 모은다"""
    if lang not in _TR:
        T, d = {}, f'{I18N}/{lang}'
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                if f.endswith('.json'):
                    for k, e in json.loads(rd(f'{d}/{f}')).items():
                        if isinstance(e, dict) and e.get('tr'): T[k] = '' if e['tr'] == '∅' else e['tr']   # ∅ = 일부러 비운 번역
        _TR[lang] = T
    return _TR[lang]


def langs_ready(ep, need=0.9):
    """그 편(i18n/<lang>/<편>.json)과 공용(common.json) 조각의 90% 이상이 번역된 언어만 언어 상자에 연다"""
    ok = ['ko']
    for l in LANGS[1:]:
        n = d = 0
        for f in (f'{I18N}/{l}/{ep}.json', f'{I18N}/{l}/common.json'):
            if os.path.exists(f):
                c = json.loads(rd(f)); d += len(c); n += sum(1 for e in c.values() if isinstance(e, dict) and e.get('tr'))
        if d and n / d >= need: ok.append(l)
    return ok


def build(ep, lang='ko', out_cache={}, T=None, write=True):
    """T: 번역 사전({키: 번역문}) 또는 i18n.Collector. lang이 ko가 아니면 load_tr(lang)."""
    d = f'{EPD}/{ep}'
    if T is None and lang != 'ko':
        T = load_tr(lang)
        ov = f'{I18N}/{lang}/_overrides.json'   # 공용 조각을 이 편에서만 다르게 번역할 때 {편: {키: 번역}}
        if os.path.exists(ov):
            o = json.loads(rd(ov)).get(ep, {})
            if o: T = dict(T, **{k: (v['tr'] if isinstance(v, dict) else v) for k, v in o.items()})
    miss = []
    trh = (lambda s, part='html': i18n.apply(s, 'html', T, miss, part)) if T is not None else (lambda s, part='html': s)
    trj = (lambda s, part='js': i18n.apply(s, 'js', T, miss, part)) if T is not None else (lambda s, part='js': s)
    m = json.loads(rd(src_for(ep, lang, 'meta.json')))
    body, outro, js = rd(src_for(ep, lang, 'body.html')), rd(src_for(ep, lang, 'outro.html')), rd(src_for(ep, lang, 'ep.js'))
    code = rd(f'{d}/code.py')
    data_js = rd(f'{d}/data.js') if os.path.exists(f'{d}/data.js') else ''
    assert '</script' not in code.lower(), 'code.py에 </script 금지'
    if ep not in out_cache: out_cache[ep] = run_code(f'{d}/code.py')
    out = out_cache[ep]
    keys = need_sprites(body + outro + js + data_js) | set(m.get('sprites', []))
    spr, missing = {}, []
    for k in sorted(keys):
        v = b64(k)
        if v: spr[k] = v
        else: missing.append(k)
    toc = ''.join(f'<a href="#{a}">{b}</a>' for a, b in m['toc'] + [['notes', '유의사항']])
    ep_meta = json.dumps({'n': m['n'], 'title': (T.get(i18n.key(m['title'])) if T is not None and not hasattr(T, 'see') else None) or m['title']}, ensure_ascii=False)
    if T is not None and hasattr(T, 'see'): T.see(m['title'], 'meta.json: 편 제목 (EP.title)', 'meta')
    page = f'''<!DOCTYPE html>
<html lang="{lang}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>{m['n']} {m['title']} | 모델 채용 위원회</title>
<meta name="description" content="{html.escape(m['desc'])}">
<meta name="author" content="안상선">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+KR:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500;700&family=Black+Han+Sans&display=swap" rel="stylesheet">
<style>{CSS}</style>
{font_head(lang)[0]}
</head>
<body>
<nav class="top" aria-label="이 페이지의 목차">
  <div class="wrap">
    <div class="toc">{toc}</div>
    <div class="board" id="board" aria-live="polite" title="채용 현황판: 지금까지 합격한 지원자 수와 첫 합격선(5-0 매뉴얼 씨의 시험지 정확도)">
      <span class="lab">채용 현황</span>합격 <b id="bP">0</b>/15<span class="bar">· 합격선 <span id="bL">-</span></span>
    </div>
  </div>
</nav>
{body}
{common_sections()}
{outro}
{notes_section([('데이터와 계산', m['foot'])] + NOTES_COMMON)}
<footer>
  <div class="wrap">
    <p><b style="color:var(--text)">모델 채용 위원회 · {m['n']} {m['title']}</b> · <a href="5-0_job_posting.html#m8">전체 지도 (5-0)</a> — 글 안상선 ((주)M-Robo 대표)</p>
    <p>이 저작물은 <a href="https://creativecommons.org/licenses/by-nc/4.0/deed.ko" target="_blank" rel="license noopener">CC BY-NC 4.0</a>으로 공개한다. 비상업적 목적에 한해 자유롭게 공유하고 고쳐 쓸 수 있으며, 출처는 반드시 표기해야 한다. 출처 표기 예: 안상선, 「모델 채용 위원회 · {m['n']} {m['title']}」, AI Model Review.</p>
  </div>
</footer>
<script type="text/plain" id="src-code">
{code}</script>
<script>
var SPR={json.dumps(spr)};
var EP={ep_meta};
var LANG={json.dumps(lang)};
var LANG_READY={json.dumps(langs_ready(ep))};
var CODE_OUT={json.dumps(out, ensure_ascii=False)};
{DIGITSJS if m.get('data') == 'digits' else ''}
{DATAJS}
{trj(CORE, 'core.js')}
{font_head(lang)[1]}
{MLJS}
{trj(data_js, 'data.js')}
{trj(js, 'ep.js')}
boot();
</script>
</body>
</html>
'''
    if T is not None:
        cut = page.index('<script type="text/plain" id="src-code">')
        page = trh(page[:cut], 'html') + page[cut:]
    if not write: return miss
    od = OUT if lang == 'ko' else f'{OUT}/{lang}'
    os.makedirs(od, exist_ok=True)
    path = f"{od}/{m['file']}"
    open(path, 'w', encoding='utf-8').write(page)
    if ep == '5-0':  # 첫 화면(index.html)은 전체 지도가 있는 채용 공고와 같다
        open(f"{od}/index.html", 'w', encoding='utf-8').write(page)
    img = sum(len(v) for v in spr.values())
    cov = '' if T is None else f'  · 번역 안 된 조각 {len(miss)}개'
    print(f"{lang}/{m['file']}: {len(page)//1024} KB (그림 {img//1024} KB, {len(spr)}장)" + cov + (f'  ⚠ 없는 그림: {missing}' if missing else ''))
    return miss


if __name__ == '__main__':
    args = sys.argv[1:]
    only = None
    if '--lang' in args:
        i = args.index('--lang'); only = args[i + 1]; del args[i:i + 2]
    eps = args or sorted([e for e in os.listdir(EPD) if re.fullmatch(r'5-\d+', e)], key=lambda s: [int(x) for x in s.split('-')])
    for e in eps:
        for l in ([only] if only else langs_ready(e)):
            if only and only != 'ko' and not os.path.isdir(f'{I18N}/{only}'): raise SystemExit(f'i18n/{only}/ 가 없다')
            build(e, l)
