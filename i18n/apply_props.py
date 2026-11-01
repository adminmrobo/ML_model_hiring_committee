#!/usr/bin/env python3
"""공용 조각 손질 제안(prop_*.json)을 번역 폴더에 반영한다.
   사용: python3 i18n/apply_props.py zh prop_a.json prop_b.json
   제안 형식: {"common": {키: 번역}, "overrides": {편: {키: 번역}}, "ep": {편: {키: 번역}}} — 뒤 파일이 앞 파일을 덮는다."""
import json, sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
lang, props = sys.argv[1], [json.load(open(p, encoding='utf-8')) for p in sys.argv[2:]]
D = f'{HERE}/{lang}/'
J = lambda p: json.load(open(p, encoding='utf-8'))
W = lambda p, o: json.dump(o, open(p, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
seen = {}
for i, P in enumerate(props):
    for sec in ('common',):
        for k in P.get(sec, {}):
            if k in seen: print(f'충돌 common {k}: 제안 {seen[k]} → {i} (뒤의 것 사용)')
            seen[k] = i
cm = J(D + 'common.json'); n = 0
for P in props:
    for k, v in P.get('common', {}).items():
        if k in cm: cm[k]['tr'] = v; n += 1
W(D + 'common.json', cm)
ovp = D + '_overrides.json'; ov = J(ovp) if os.path.exists(ovp) else {}
m = e = 0
for P in props:
    for ep, d in P.get('overrides', {}).items():
        ov.setdefault(ep, {}).update(d); m += len(d)
    for ep, d in P.get('ep', {}).items():
        f = J(D + ep + '.json')
        for k, v in d.items():
            if k in f: f[k]['tr'] = v; e += 1
            else: print('없는 키', ep, k)
        W(D + ep + '.json', f)
W(ovp, {k: v for k, v in ov.items() if v})
print(f'{lang}: common {n} · overrides {m} · ep {e}')
