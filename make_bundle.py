#!/usr/bin/env python3
"""전체 결과물을 ZIP 하나로 묶는다 → dist/model_hiring_committee.zip
   사용: python3 make_bundle.py        (먼저 python3 build.py · python3 make_site.py 로 site/를 최신으로)
   들어가는 것: README.md(6개 언어 한 파일) · LICENSE.md · site/ · docs/ · local/ · 소스(build.py, src/, ep/, assets/, data/)
               · i18n/ · qa/ · tools/ · test/ · out/.code_cache(Colab 실행 결과 캐시 — 다시 빌드가 빨라진다)
   빼는 것  : out/의 페이지(빌드하면 다시 생김), 검사 산출물(qa/shots · review · pseudo · work), __pycache__,
               옛 스크린샷(shots/), 검토 후 쓰지 않은 데이터(AI4I 2020, 물고기)와 그 시험(test/ml_check*.js)"""
import os, shutil, subprocess, sys, zipfile, re
R = os.path.dirname(os.path.abspath(__file__))
NAME = 'model_hiring_committee'
DIST = f'{R}/dist'; STAGE = f'{DIST}/{NAME}'
INCLUDE = ['README.md', 'LICENSE.md', 'build.py', 'make_site.py', 'make_readme.py', 'make_bundle.py',
           'site', 'docs', 'local', 'src', 'ep', 'assets', 'data', 'i18n', 'qa', 'tools', 'test', 'out/.code_cache']
SKIP_DIR = {'__pycache__', 'shots', 'review', 'pseudo', 'work'}
SKIP_FILE = re.compile(r'(^|/)(ai4i2020\.csv|fish\.csv|ml_check2?\.js|_report\.md)$')

# 1) 맨 위 README.md · LICENSE.md (6개 언어, 링크는 site/ 아래)
subprocess.run([sys.executable, f'{R}/make_readme.py', '--bundle', f'{R}/README.md'], check=True)
# 2) site/ 확인
n_html = sum(f.endswith('.html') for _, _, fs in os.walk(f'{R}/site') for f in fs)
assert n_html == 102 and os.path.exists(f'{R}/site/.nojekyll') and os.path.exists(f'{R}/site/README.md'), '먼저 python3 make_site.py'
# 3) README의 상대 링크가 모두 실제 파일인지
links = re.findall(r'\]\(([^)#]+)\)', open(f'{R}/README.md', encoding='utf-8').read())
bad = [l for l in links if not l.startswith('http') and not os.path.exists(os.path.join(R, l))]
assert not bad, f'README 링크 깨짐: {bad[:5]}'
# 4) 모으기
shutil.rmtree(STAGE, ignore_errors=True); os.makedirs(STAGE)
for item in INCLUDE:
    src = f'{R}/{item}'
    if os.path.isdir(src):
        for root, dirs, files in os.walk(src):
            dirs[:] = [d for d in dirs if d not in SKIP_DIR]
            for f in files:
                rel = os.path.relpath(os.path.join(root, f), R)
                if SKIP_FILE.search(rel): continue
                os.makedirs(os.path.dirname(f'{STAGE}/{rel}'), exist_ok=True)
                shutil.copy2(os.path.join(root, f), f'{STAGE}/{rel}')
    elif os.path.exists(src):
        shutil.copy2(src, f'{STAGE}/{item}')
# 5) 압축
zp = f'{DIST}/{NAME}.zip'
if os.path.exists(zp): os.remove(zp)
with zipfile.ZipFile(zp, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for root, _, files in os.walk(STAGE):
        for f in files:
            p = os.path.join(root, f); z.write(p, os.path.relpath(p, DIST))
n = sum(len(fs) for _, _, fs in os.walk(STAGE))
print(f'{zp} · 파일 {n}개 · {os.path.getsize(zp) / 1e6:.1f} MB (README 링크 {len(links)}개 확인)')
