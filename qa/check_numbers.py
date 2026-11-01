#!/usr/bin/env python3
"""기준 숫자 점검 — 각 편의 Colab 코드를 실제로 돌린 출력(빌드 캐시 out/.code_cache)에 기준값이 그대로 나오는지 본다.
   사용: python3 qa/check_numbers.py          (코드가 바뀐 편은 build.py가 다시 실행해 캐시를 갱신한다)
   기준값이 바뀌었다면 의도한 변경인지 확인하고, 페이지 본문·5-15 결산표(FINAL)·README의 숫자도 함께 고친다."""
import json, os, sys
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, R); sys.argv = [sys.argv[0]]
import build
EXPECT = {  # 편: 출력에 반드시 나와야 하는 값 (정확도·R²·ARI는 소수 셋째 자리)
 '5-0': ['0.767'], '5-1': ['0.857', '0.767'], '5-2': ['0.567', '0.842', '0.735'], '5-3': ['0.849'], '5-4': ['0.864'],
 '5-5': ['0.709', '0.810'], '5-6': ['0.926', '0.891'], '5-7': ['0.934', '0.927'], '5-8': ['0.880', '0.907'], '5-9': ['0.934', '0.949'],
 '5-10': ['0.666', '0.183', '0.108'], '5-11': ['0.794'], '5-12': ['0.001', '25.16', '0.483'], '5-13': ['0.836'],
 '5-14': ['0.603', '0.285', '0.181'], '5-15': ['0.976']}
bad = 0
for ep, vals in EXPECT.items():
    out = build.run_code(f'{build.EPD}/{ep}/code.py')
    miss = [v for v in vals if v not in out]
    bad += bool(miss)
    print(('  ' if not miss else '✗ ') + ep, '·'.join(vals), ('없음: ' + ', '.join(miss)) if miss else 'OK')
sys.exit(1 if bad else 0)
