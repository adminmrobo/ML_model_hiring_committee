#!/usr/bin/env python3
"""공용 조각 검토 도구 — 여러 편이 함께 쓰는 문장 조각(common.json)이 편마다 앞뒤 조각과 맞게 읽히는지 본다.
   python3 qa/shared_review.py collect LANG           → qa/review/LANG_ctx.json (편 스크립트에서 쓰이는 공용 조각과 편별 문맥)
                                                        + 키를 반으로 나눈 LANG_keys_a.json / LANG_keys_b.json
   python3 qa/shared_review.py render LANG [제안.json|-] 키 [키...]
                                                      → 편마다 한국어 문맥과, 지금 번역(+제안)을 끼운 결과를 나란히 출력
   제안 파일 형식: {"common": {키: 번역}, "overrides": {편: {키: 번역}}, "ep": {편: {키: 번역}}}
   반영: python3 i18n/apply_props.py LANG 제안a.json 제안b.json"""
import json, os, re, sys, glob
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, R); sys.path.insert(0, f'{R}/i18n')
import i18n
RD = f'{R}/qa/review'; os.makedirs(RD, exist_ok=True)


def collect(lang):
    argv = sys.argv; sys.argv = [argv[0]]
    import build
    sys.argv = argv
    c = json.load(open(f'{R}/i18n/{lang}/common.json', encoding='utf-8'))
    eps = sorted([e for e in os.listdir(build.EPD) if re.fullmatch(r'5-\d+', e)], key=lambda s: int(s.split('-')[1]))
    ctx = {}
    for e in eps:
        C = i18n.Collector(); build.build(e, 'ko', T=C, write=False)
        for text, cx, part, k in C.seen:
            if k in c and part in ('ep.js', 'data.js'): ctx.setdefault(k, {}).setdefault(e, set()).add(cx[:260])
    out = {k: {'ko': c[k]['ko'], 'tr': c[k]['tr'], 'ctx': {e: sorted(v) for e, v in d.items()}} for k, d in ctx.items()}
    json.dump(out, open(f'{RD}/{lang}_ctx.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    ks = list(out); h = len(ks) // 2
    json.dump(ks[:h], open(f'{RD}/{lang}_keys_a.json', 'w')); json.dump(ks[h:], open(f'{RD}/{lang}_keys_b.json', 'w'))
    print(f'{lang}: 공용 조각 {len(out)}개 · 편별 쓰임 {sum(len(d) for d in out.values())}곳 → {RD}/{lang}_ctx.json, _keys_a/_keys_b.json')


def table(lang, ep, prop):
    D = f'{R}/i18n/{lang}'; T = {}
    for f in glob.glob(D + '/*.json'):
        if os.path.basename(f).startswith('_'): continue
        for k, v in json.load(open(f, encoding='utf-8')).items():
            if v.get('tr'): T[k] = '' if v['tr'] == '∅' else v['tr']
    ov = json.load(open(D + '/_overrides.json', encoding='utf-8')) if os.path.exists(D + '/_overrides.json') else {}
    T.update(ov.get(ep, {}))
    if prop:
        for sec in (prop.get('common', {}), prop.get('ep', {}).get(ep, {}), prop.get('overrides', {}).get(ep, {})):
            T.update({k: ('' if v == '∅' else v) for k, v in sec.items()})
    return T


def render(lang, propf, keys):
    SH = json.load(open(f'{RD}/{lang}_ctx.json', encoding='utf-8'))
    prop = None if propf == '-' else json.load(open(propf, encoding='utf-8'))
    for key in keys:
        e = SH[key]; print(f'=== {key} ko={e["ko"]!r} tr={e["tr"]!r}')
        for ep, cs in e['ctx'].items():
            T = table(lang, ep, prop)
            for c in cs:
                src = c.replace('⟦', '').replace('⟧', '')
                try: tr = i18n.apply(src, 'js', T)
                except Exception as x: tr = f'(문맥이 잘려 해석 못 함: ep/{ep}/ep.js에서 직접 확인) ' + src
                print(f'  [{ep}] KO: {c[:200]}\n        {lang.upper()}: {tr[:240]}')


if __name__ == '__main__':
    a = sys.argv[1:]
    if not a: print(__doc__); sys.exit()
    if a[0] == 'collect': collect(a[1])
    elif a[0] == 'render': render(a[1], a[2], a[3:])
