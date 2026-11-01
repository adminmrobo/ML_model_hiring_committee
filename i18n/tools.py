#!/usr/bin/env python3
"""번역 작업 도구
   python3 i18n/tools.py init en        i18n/source/*.json 을 복사해 i18n/en/ 을 만든다 (있던 번역은 유지)
   python3 i18n/tools.py status         언어별 · 파일별 번역률
   python3 i18n/tools.py check en       번역문 검사 → i18n/en/_report.md (태그·링크·class·남은 한글·용어집)
   python3 i18n/tools.py pack en 5-4    번역가(사람·Genspark 등)에게 넘길 파일 묶음 → i18n/handoff/en_5-4.json
   python3 i18n/tools.py merge en <파일> 돌려받은 파일의 tr 값을 i18n/en/ 에 합친다 (키가 맞는 것만)
"""
import os, sys, json, glob, shutil
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import i18n

LANGS = ['en', 'zh', 'ja', 'uz', 'ru']
NAMES = {'en': 'English', 'zh': '简体中文', 'ja': '日本語', 'uz': "Oʻzbekcha (lotin)", 'ru': 'Русский'}
J = lambda p: json.load(open(p, encoding='utf-8'))
W = lambda p, o: json.dump(o, open(p, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
order = lambda f: (0, 0) if f == 'common' else (1, int(f.split('-')[1]))


def files(d): return sorted([os.path.basename(p)[:-5] for p in glob.glob(f'{d}/*.json') if not os.path.basename(p).startswith('_')], key=order)


def init(lang):
    d = f'{HERE}/{lang}'; os.makedirs(d, exist_ok=True)
    for f in files(f'{HERE}/source'):
        src = J(f'{HERE}/source/{f}.json'); p = f'{d}/{f}.json'
        old = J(p) if os.path.exists(p) else {}
        W(p, {k: dict(v, tr=old.get(k, {}).get('tr', '')) for k, v in src.items()})
    print(f'{lang}: {d} 준비됨')


def status():
    src = files(f'{HERE}/source')
    print('파일      조각   ' + '  '.join(f'{l:>6s}' for l in LANGS))
    for f in src:
        c = J(f'{HERE}/source/{f}.json'); row = []
        for l in LANGS:
            p = f'{HERE}/{l}/{f}.json'
            if not os.path.exists(p): row.append('     -'); continue
            t = J(p); n = sum(1 for k in c if t.get(k, {}).get('tr'))
            row.append(f'{100 * n / max(1, len(c)):5.0f}%')
        print(f'{f:7s} {len(c):6d}   ' + '  '.join(row))


def kind_of(v): return 'js' if '문자열' in v.get('where', '') else 'html'


def check(lang):
    d = f'{HERE}/{lang}'; gl = {}
    gp = f'{HERE}/glossary/{lang}.json'
    if os.path.exists(gp):
        g = J(gp)
        gl = {k: v['t'] for k, v in g.items() if v.get('t') and len(k) >= 2}
    lines, nerr, nwarn, done, total = [f'# {NAMES.get(lang, lang)} 번역 검사\n'], 0, 0, 0, 0
    for f in files(d):
        errs = []
        for k, v in J(f'{d}/{f}.json').items():
            total += 1
            tr = v.get('tr', '')
            if not tr: continue
            done += 1
            probs = i18n.check(v['ko'], tr, kind_of(v))
            hard = [p for p in probs if not p.startswith('앞뒤 공백')]
            soft = [p for p in probs if p.startswith('앞뒤 공백')] if lang not in ('zh', 'ja') else []
            for term, t in gl.items():   # 용어집: 원문에 용어가 있는데 번역에 그 번역어가 없으면 경고
                if term in v['ko'] and t.lower() not in tr.lower() and t.split(' (')[0].lower() not in tr.lower():
                    soft.append(f'용어집: "{term}" → "{t}" 가 안 보임')
            if hard or soft:
                nerr += bool(hard); nwarn += len(soft)
                errs.append(f'- `{k}` {"**오류** " + "; ".join(hard) if hard else ""} {"경고: " + "; ".join(soft[:3]) if soft else ""}\n  - ko: {v["ko"][:160]}\n  - tr: {tr[:160]}')
        if errs: lines.append(f'\n## {f}\n' + '\n'.join(errs))
    lines.insert(1, f'번역 {done:,d} / {total:,d}조각 · 오류 {nerr} · 경고 {nwarn}\n')
    open(f'{d}/_report.md', 'w', encoding='utf-8').write('\n'.join(lines))
    print(f'{lang}: 번역 {done}/{total} · 오류 {nerr} · 경고 {nwarn} → {d}/_report.md')
    return nerr


def pack(lang, which):
    """번역가용 파일: 편 카탈로그 + 그 편에 나오는 용어집만 + 규칙 요약"""
    os.makedirs(f'{HERE}/handoff', exist_ok=True)
    d = f'{HERE}/{lang}'
    if not os.path.isdir(d): init(lang)
    cat = J(f'{d}/{which}.json')
    gl = J(f'{HERE}/glossary/{lang}.json') if os.path.exists(f'{HERE}/glossary/{lang}.json') else {}
    text = ' '.join(v['ko'] for v in cat.values())
    terms = {k: v for k, v in gl.items() if k in text}
    out = {'_rules': open(f'{HERE}/RULES_{lang}.md', encoding='utf-8').read() if os.path.exists(f'{HERE}/RULES_{lang}.md') else open(f'{HERE}/README.md', encoding='utf-8').read().split('## 번역 규칙')[1].split('\n## ')[0],
           '_target_language': NAMES[lang], '_glossary': terms,
           'items': {k: {'ko': v['ko'], 'ctx': v['ctx'], 'tr': v.get('tr', '')} for k, v in cat.items()}}
    p = f'{HERE}/handoff/{lang}_{which}.json'; W(p, out)
    print(f'{p} · 조각 {len(cat)} · 용어 {len(terms)}')


def merge(lang, path):
    got = J(path); got = got.get('items', got)
    d = f'{HERE}/{lang}'; n = 0
    for f in files(d):
        c = J(f'{d}/{f}.json'); ch = False
        for k, v in got.items():
            tr = v.get('tr') if isinstance(v, dict) else v
            if k in c and tr: c[k]['tr'] = tr; ch = True; n += 1
        if ch: W(f'{d}/{f}.json', c)
    print(f'{lang}: {n}개 합침')


if __name__ == '__main__':
    a = sys.argv[1:]
    if not a: print(__doc__); sys.exit()
    {'init': lambda: init(a[1]), 'status': status, 'check': lambda: sys.exit(1 if check(a[1]) else 0),
     'pack': lambda: pack(a[1], a[2]), 'merge': lambda: merge(a[1], a[2])}[a[0]]()
