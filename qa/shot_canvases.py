#!/usr/bin/env python3
"""그래프(캔버스) 모아 찍기 — 번역 라벨이 그림 밖으로 넘치거나 겹치는지 눈으로 확인한다.
   사용: python3 qa/shot_canvases.py en 5-3 5-9     → qa/shots/en_5-3.png, en_5-9.png
         python3 qa/shot_canvases.py zh               → 그 언어 16편 모두
   1100px 폭으로 열어 높이 60px 이상인 캔버스를 모두 찍고, 두 줄씩 한 장(콘택트 시트)으로 붙인다."""
import asyncio, glob, os, sys
from playwright.async_api import async_playwright
from PIL import Image
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTD = f'{R}/qa/shots'; os.makedirs(OUTD, exist_ok=True)
lang, eps = sys.argv[1], sys.argv[2:]
d = f'{R}/out' if lang == 'ko' else f'{R}/out/{lang}'
files = sorted(glob.glob(f'{d}/5-*.html'), key=lambda s: int(os.path.basename(s).split('_')[0].split('-')[1]))
if eps: files = [f for f in files if os.path.basename(f).split('_')[0] in eps]

async def one(b, f):
    pg = await b.new_page(viewport={'width': 1100, 'height': 900})
    await pg.goto('file://' + f); await pg.wait_for_timeout(3500)
    ims = []
    for i, c in enumerate(await pg.query_selector_all('section canvas')):
        try:
            await c.scroll_into_view_if_needed(); await pg.wait_for_timeout(200)
            bb = await c.bounding_box()
            if bb and bb['height'] >= 60:
                p = f'{OUTD}/_tmp_{i}.png'; await c.screenshot(path=p); ims.append(Image.open(p).copy()); os.remove(p)
        except Exception: pass
    await pg.close()
    if not ims: return
    w = 550; th = [im.resize((w, int(im.height * w / im.width))) for im in ims]
    rows = [th[i:i + 2] for i in range(0, len(th), 2)]
    sheet = Image.new('RGB', (1100, sum(max(t.height for t in r) for r in rows)), 'white'); y = 0
    for r in rows:
        x = 0
        for t in r: sheet.paste(t, (x, y)); x += w
        y += max(t.height for t in r)
    name = f"{OUTD}/{lang}_{os.path.basename(f).split('_')[0]}.png"; sheet.save(name); print(name, len(ims), '장')

async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch()
        for f in files: await one(b, f)
        await b.close()
asyncio.run(main())
