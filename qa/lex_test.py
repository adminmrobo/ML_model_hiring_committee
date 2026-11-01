#!/usr/bin/env python3
"""자바스크립트 조각 뽑기 시험 — 모든 .js에서 한글 문자열을 'Ⓧ'로 바꿔도 문법이 깨지지 않는지(node --check),
   주석 밖에 한글이 남지 않는지 본다. 소스에 새 문법(템플릿 ${} 안 한글, 한글 정규식 등)을 넣었을 때 돌린다.
   사용: python3 qa/lex_test.py   (모두 'syntax OK · left 0'이어야 한다)"""
import glob, os, re, subprocess, sys, tempfile
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f'{R}/i18n'); import i18n
class P(dict):
    def get(self, k, d=None): return 'Ⓧ'
bad = 0
for f in sorted(glob.glob(f'{R}/src/*.js') + glob.glob(f'{R}/ep/*/ep.js') + glob.glob(f'{R}/ep/*/data.js')):
    s = open(f, encoding='utf-8').read()
    try: out = i18n.apply(s, 'js', P())
    except Exception as e: print('✗', f, e); bad += 1; continue
    t = tempfile.NamedTemporaryFile('w', suffix='.js', delete=False, encoding='utf-8'); t.write(out); t.close()
    r = subprocess.run(['node', '--check', t.name], capture_output=True, text=True); os.unlink(t.name)
    nc = re.sub(r'/\*.*?\*/', '', out, flags=re.S); nc = re.sub(r'(?m)(^|[^:\\])//.*$', r'\1', nc)
    left = re.findall(r'[가-힣]+', nc)
    ok = r.returncode == 0 and not left; bad += not ok
    print(('  ' if ok else '✗ ') + os.path.relpath(f, R), 'syntax', 'OK' if r.returncode == 0 else 'FAIL', '· left', len(left))
sys.exit(1 if bad else 0)
