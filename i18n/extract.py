#!/usr/bin/env python3
"""한국어 원본에서 번역할 조각을 뽑아 i18n/source/*.json 카탈로그를 만든다.
   사용: python3 i18n/extract.py
   - 두 편 이상에 나오거나 core.js(공용)에 있는 조각 → source/common.json
   - 그 밖의 조각 → source/<편>.json
   각 항목: {"키": {"ko": 원문, "ctx": 앞뒤 문맥(⟦ ⟧가 번역할 곳), "where": 출처, "tr": ""}}
   이미 있는 번역 폴더(i18n/<언어>/)가 있으면 sync()로 새 조각을 더하고 사라진 조각은 _stale.json으로 옮긴다."""
import os, sys, json, re
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
import i18n
sys.argv = [sys.argv[0]]
import build

EPS = sorted([e for e in os.listdir(build.EPD) if re.fullmatch(r'5-\d+', e)], key=lambda s: [int(x) for x in s.split('-')])
KIND = {'core.js': '공용 스크립트(core.js) 문자열', 'ep.js': '편 스크립트(ep.js) 문자열', 'data.js': '편 데이터(data.js) 문자열',
        'html': '페이지 HTML', 'meta': '편 제목'}


def collect():
    per = {}
    for e in EPS:
        C = i18n.Collector()
        build.build(e, 'ko', T=C, write=False)
        per[e] = C.seen
    return per


def main():
    per = collect()
    eps_of, core, first = {}, set(), {}
    for e, seen in per.items():
        for text, ctx, part, k in seen:
            eps_of.setdefault(k, set()).add(e)
            if part == 'core.js': core.add(k)
            first.setdefault(k, (text, ctx, part, e))
    cats = {'common': {}}
    for e in EPS: cats[e] = {}
    for e in EPS:   # 편 순서대로, 페이지에 나오는 순서대로
        for text, ctx, part, k in per[e]:
            scope = 'common' if (k in core or len(eps_of[k]) > 1) else e
            if k in cats[scope]: continue
            where = KIND.get(part, part) + ('' if scope == 'common' else f' · {e}')
            if scope == 'common': where += ' · 나오는 편: ' + ', '.join(sorted(eps_of[k], key=lambda s: [int(x) for x in s.split('-')]))
            cats[scope][k] = {'ko': text, 'ctx': ctx if ctx != text else '', 'where': where, 'tr': ''}
    od = f'{HERE}/source'; os.makedirs(od, exist_ok=True)
    tot = 0
    for name, c in cats.items():
        json.dump(c, open(f'{od}/{name}.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        n = sum(len(v['ko']) for v in c.values()); tot += n
        print(f'{name:7s} 조각 {len(c):5d}개 · 글자 {n:7,d}')
    print(f'합계 글자 {tot:,d}')
    for l in build.LANGS[1:]:
        if os.path.isdir(f'{HERE}/{l}'): sync(l, cats)


def sync(lang, cats):
    """번역 폴더를 새 카탈로그에 맞춘다. 있던 번역은 그대로 두고, 원문이 바뀌어 키가 사라진 항목은 _stale.json에 모은다."""
    d = f'{HERE}/{lang}'; old = {}
    for f in os.listdir(d):
        if f.endswith('.json') and not f.startswith('_'):
            for k, v in json.load(open(f'{d}/{f}', encoding='utf-8')).items(): old[k] = v
    used = set()
    for name, c in cats.items():
        out = {}
        for k, v in c.items():
            o = old.get(k, {})
            out[k] = dict(v, tr=o.get('tr', ''))
            if o.get('tr'): used.add(k)
        json.dump(out, open(f'{d}/{name}.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    stale = {k: v for k, v in old.items() if v.get('tr') and k not in used and not any(k in c for c in cats.values())}
    if stale: json.dump(stale, open(f'{d}/_stale.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f'{lang}: 번역 {len(used):,d}개 유지 · 원문이 바뀐 옛 번역 {len(stale)}개 → _stale.json')


if __name__ == '__main__':
    main()
