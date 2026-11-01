#!/usr/bin/env python3
"""가짜 번역 왕복 시험 — 번역 장치(조각 뽑기 → 끼우기)가 페이지를 깨지 않는지 본다.
   사용: python3 qa/pseudo_build.py [편 ...]    (기본 5-0 5-4 5-12 5-15)
   모든 조각의 한글을 'ẍ'로 바꾼 '가짜 언어'로 빌드해 qa/pseudo/xx/에 쓴다.
   그다음 python3 qa/check_pages.py 대신 아래 출력의 '남은 조각 0'과,
   브라우저에서 qa/pseudo/xx/*.html을 열어 화면에 한글이 없는지(코드 영역 제외) 확인한다."""
import glob, json, os, re, sys
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, R); sys.path.insert(0, f'{R}/i18n')
eps = sys.argv[1:] or ['5-0', '5-4', '5-12', '5-15']
sys.argv = [sys.argv[0]]
import build
H = re.compile(r'[가-힣]+')
T = {}
for f in glob.glob(f'{R}/i18n/source/*.json'):
    for k, v in json.load(open(f, encoding='utf-8')).items(): T[k] = H.sub(lambda m: 'ẍ' * len(m.group(0)), v['ko'])
build.OUT = f'{R}/qa/pseudo'
for e in eps:
    miss = build.build(e, 'xx', T=T)
    print(e, '남은 조각', len(miss), [m[:30] for m in miss[:3]])
