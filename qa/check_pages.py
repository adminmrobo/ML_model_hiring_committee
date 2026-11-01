#!/usr/bin/env python3
"""완성 페이지 점검 (Playwright · 내장 Chromium)
   사용: python3 qa/check_pages.py            → 한국어 + 모든 번역 언어
         python3 qa/check_pages.py en zh       → 고른 언어만 (ko = 한국어)
   페이지마다 휴대폰 폭(390px)으로 열고 면접 버튼을 모두 누른 뒤 확인한다.
     sw   : 가로 스크롤 폭. 390이어야 한다(넘치면 휴대폰에서 옆으로 밀림)
     err  : 실행 오류(pageerror) 수 — 0이어야 한다
     ko   : (번역판만) Colab 코드 영역 밖에 남은 한글 조각 수 — 0이어야 한다
     img  : 깨진 캐릭터 그림 수 — 0이어야 한다
   문제가 있는 페이지만 자세히 출력하고, 끝에 언어별 합계를 낸다. 문제가 하나라도 있으면 종료 코드 1."""
import asyncio, glob, os, re, sys
from playwright.async_api import async_playwright
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LANGS = sys.argv[1:] or ['ko'] + sorted(d for d in os.listdir(f'{R}/out') if os.path.isdir(f'{R}/out/{d}') and not d.startswith('.'))
H = re.compile(r'.{0,25}[가-힣]+.{0,15}')

async def one(b, lang, f):
    pg = await b.new_page(viewport={'width': 390, 'height': 900})
    errs = []; pg.on('pageerror', lambda e: errs.append(str(e)[:120]))
    await pg.goto('file://' + f); await pg.wait_for_timeout(3500)
    await pg.evaluate("document.querySelectorAll('.interview button').forEach(b=>{try{b.click()}catch(e){}})")
    await pg.wait_for_timeout(2000)
    sw = await pg.evaluate('document.documentElement.scrollWidth')
    img = await pg.evaluate("[...document.querySelectorAll('img[data-who],img[data-scene]')].filter(i=>!i.complete||i.naturalWidth===0).length")
    left = []
    if lang != 'ko':
        txt = await pg.evaluate("(()=>{const c=document.body.cloneNode(true);c.querySelectorAll('pre.code,script,#codebox,.runshot,#langSel').forEach(e=>e.remove());return c.innerText})()")
        left = sorted(set(H.findall(txt)))
    await pg.close()
    return os.path.basename(f), sw, errs, left, img

async def main():
    bad = 0
    async with async_playwright() as p:
        b = await p.chromium.launch()
        for lang in LANGS:
            d = f'{R}/out' if lang == 'ko' else f'{R}/out/{lang}'
            files = sorted(glob.glob(f'{d}/5-*.html'), key=lambda s: int(os.path.basename(s).split('_')[0].split('-')[1]))
            res = []
            for i in range(0, len(files), 4): res += await asyncio.gather(*[one(b, lang, f) for f in files[i:i + 4]])
            nb = 0
            for n, sw, errs, left, img in res:
                if sw > 392 or errs or left or img:
                    nb += 1; print(f'  ✗ {lang}/{n}: sw={sw} err={len(errs)} ko={len(left)} img={img}', errs[:2], left[:4])
            print(f'{lang}: {len(res)}쪽 점검 · 문제 {nb}쪽'); bad += nb
        await b.close()
    sys.exit(1 if bad else 0)

asyncio.run(main())
