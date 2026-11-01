#!/usr/bin/env python3
"""GitHub Pages에 올릴 폴더(site/)를 만든다: out/의 한국어(맨 위)와 번역판(en/ zh/ ja/ uz/ ru/)만 복사.
   사용: python3 make_site.py   →  site/ 를 저장소 루트(또는 docs/)에 그대로 올리면 된다."""
import os, shutil, json
R = os.path.dirname(os.path.abspath(__file__)); OUT, SITE = f'{R}/out', f'{R}/site'
shutil.rmtree(SITE, ignore_errors=True); os.makedirs(SITE)
n = 0
for root, dirs, files in os.walk(OUT):
    dirs[:] = [d for d in dirs if not d.startswith('.')]
    for f in files:
        if f.endswith('.html'):
            rel = os.path.relpath(os.path.join(root, f), OUT)
            os.makedirs(os.path.dirname(f'{SITE}/{rel}'), exist_ok=True)
            shutil.copy2(os.path.join(root, f), f'{SITE}/{rel}'); n += 1
open(f'{SITE}/.nojekyll', 'w').close()
import subprocess, sys; subprocess.run([sys.executable, f'{R}/make_readme.py', SITE], check=True)
langs = sorted(d for d in os.listdir(SITE) if os.path.isdir(f'{SITE}/{d}'))
size = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(SITE) for f in fs)
print(f'site/: HTML {n}개 · 언어 폴더 {langs} · {size/1e6:.1f} MB')
