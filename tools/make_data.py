#!/usr/bin/env python3
"""데이터 파일을 원자료에서 다시 만든다.
   python3 tools/make_data.py
   1) data/raw/Concrete_Data.xls (UCI id=165 원본) → data/concrete.csv (열 이름을 짧게, 1,030행)
   2) data/concrete.csv → src/concrete.js : CONC(값 9열) + CONC_TEST(시험지 258행 번호)
      시험지 규정: train_test_split(test_size=0.25, stratify=강도 3등급(<25 · 25~45 · ≥45 MPa), random_state=42)
   3) scikit-learn load_digits → src/digits.js : DIG_PIX(한 장 64글자, 0~9·a~g = 밝기 0~16) + DIG_Y(정답)
   --check 를 붙이면 파일을 쓰지 않고 지금 파일과 같은지만 확인한다."""
import json, os, sys
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECK = '--check' in sys.argv
COLS = ['cement', 'slag', 'fly_ash', 'water', 'superplasticizer', 'coarse_agg', 'fine_agg', 'age', 'strength']


def out(path, text):
    p = f'{R}/{path}'
    if CHECK:
        same = os.path.exists(p) and open(p, encoding='utf-8').read() == text
        print(('같음  ' if same else '다름! ') + path); return same
    open(p, 'w', encoding='utf-8').write(text); print('작성', path); return True


ok = True
# 1) xls → csv
raw = f'{R}/data/raw/Concrete_Data.xls'
if os.path.exists(raw):
    df = pd.read_excel(raw); df.columns = COLS
    ok &= out('data/concrete.csv', df.to_csv(index=False, lineterminator='\n'))
df = pd.read_csv(f'{R}/data/concrete.csv')
assert len(df) == 1030 and list(df.columns) == COLS
# 2) concrete.js
grade = np.digitize(df['strength'], [25, 45])
idx = np.arange(len(df))
_, te = train_test_split(idx, test_size=0.25, stratify=grade, random_state=42)
te = sorted(int(i) for i in te); assert len(te) == 258
rows = ','.join('[' + ','.join(repr(round(float(v), 2)) for v in r) + ']' for r in df.values)
js = ('/* 콘크리트 압축 강도 데이터 1,030건 (Yeh 1998, UCI Machine Learning Repository, CC BY 4.0).\n'
      '   열: 시멘트, 고로 슬래그, 플라이 애시, 물, 고성능 감수제, 굵은 골재, 잔골재 (kg/m³), 재령(일), 압축 강도(MPa)\n'
      '   CONC_TEST: 시험지 행 번호 = train_test_split(test_size=0.25, stratify=강도 3등급, random_state=42) */\n'
      f'var CONC=[{rows}];\nvar CONC_TEST=[{", ".join(map(str, te))}];\n')
ok &= out('src/concrete.js', js)
# 3) digits.js
d = load_digits(); H = '0123456789abcdefg'
pix = ''.join(H[int(v)] for v in d.data.ravel())
js = ('/* 손글씨 숫자 1,797장 (scikit-learn load_digits = UCI Optical Recognition of Handwritten Digits, CC BY 4.0의 테스트 세트). 8×8 픽셀, 밝기 0~16.\n'
      '   DIG_PIX: 한 장에 64글자, 글자 하나가 픽셀 하나(0~9, a=10 … g=16). DIG_Y: 숨긴 정답(채점에만 쓴다) */\n'
      f'var DIG_PIX="{pix}";\nvar DIG_Y=[{",".join(map(str, d.target))}];\n')
ok &= out('src/digits.js', js)
sys.exit(0 if ok else 1)
