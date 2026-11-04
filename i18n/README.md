# 번역 (i18n)

한국어 원본은 `ep/`와 `src/`의 파일 하나뿐이다. 번역판은 원본을 복사하지 않고, 한글이 든 문장 조각을 JSON 카탈로그로 뽑아 번역한 뒤 빌드할 때 끼워 넣는다. 한국어를 고쳐도 번역판이 따로 놀지 않는다.

## 흐름

| 단계 | 명령 | 결과 |
|---|---|---|
| 1. 조각 뽑기 | `python3 i18n/extract.py` | `i18n/source/common.json`, `5-0.json` … `5-15.json` |
| 2. 언어 폴더 만들기 | `python3 i18n/tools.py init en` | `i18n/en/*.json` (tr 칸이 빈 카탈로그) |
| 3. 번역 | 직접 채우거나 `pack` → 외부(Genspark 등) → `merge` | `tr` 칸이 채워짐 |
| 4. 검사 | `python3 i18n/tools.py check en` | `i18n/en/_report.md` |
| 5. 빌드 | `python3 build.py --lang en` | `out/en/*.html` |
| 진행률 | `python3 i18n/tools.py status` | 언어 × 파일 번역률 표 |

- **키**: 한국어 원문의 sha1 앞 12자리. 같은 문장은 어느 편에 나와도 한 번만 번역한다.
- **common.json**: 두 편 이상에 나오거나 공용 스크립트(core.js)에 있는 조각. 먼저 번역하면 모든 편이 함께 좋아진다.
- **원문을 고치면**: `extract.py`를 다시 돌린다. 있던 번역은 유지되고, 원문이 바뀐 조각은 `tr`이 비며 옛 번역은 `_stale.json`에 남는다(참고용).
- **번역이 없는 조각**은 한국어로 보인다. 빌드 로그에 "번역 안 된 조각 N개"로 나온다.
- 그 편과 common.json 조각의 90% 이상이 번역되면 그 언어가 언어 선택 상자에 열린다. 그 전에도 `build.py --lang en`으로 미리 볼 수 있다(한국어판의 언어 상자에는 "준비 중"으로 남음).
- **번역하지 않는 것**: Colab 코드와 그 실행 결과(공통 A장), 데이터, 그림. 코드 주석은 한국어 원문 그대로 둔다(컨셉 문서 §7).
- **글꼴**: en·uz·ru는 IBM Plex Sans, zh는 Noto Sans SC, ja는 Noto Sans JP를 더 불러온다. zh·ja는 줄바꿈 규칙(keep-all)을 풀었다.
- **한국어 조사 함수** `rago()`는 다른 언어에서 이름만 돌려준다.

## 같은 한국어, 다른 뜻

- **소스에 꼬리표 달기**: 같은 한국어가 뜻이 다르면 소스의 문자열 바로 앞에 `/*i18n:태그*/` 주석을 단다. 그 조각은 따로 키를 받는다. 지금은 면접 판정 `/*i18n:verdict*/'합격'`(Hired)과 콘크리트 판정 `'합격'`(Pass)을 나눈다. 한국어 화면은 그대로다.
- **편에서만 다르게**: 공용 조각을 한 편에서만 다르게 번역해야 하면 `i18n/<언어>/_overrides.json`에 `{"5-10": {"키": "번역"}}`으로 적는다. 빌드할 때 그 편에만 적용된다.
- **일부러 비우기**: 번역이 빈 문자열이어야 하면 `∅` 한 글자를 쓴다(빈 칸은 "아직 번역 안 함").
- **캔버스 라벨**: 칸 너비가 정해진 라벨은 `fitText(ctx, 글자, x, y, 최대너비)`로 그리면 길어진 번역이 자동으로 작아진다.

## 조각의 두 종류

- **페이지 HTML** (`where`가 "페이지 HTML"): 문단·제목·말풍선 하나가 조각 하나. 안에 `<b>`, `<a href>`, `<span class="who">` 같은 인라인 태그가 그대로 들어 있다.
- **스크립트 문자열** (`where`에 "문자열"): 자바스크립트가 숫자와 이어 붙여 화면에 쓰는 문장 조각. 앞뒤가 잘린 조각이 많아서 `ctx`(앞뒤 코드, 번역할 곳은 ⟦ ⟧)를 보고 번역해야 한다.

## 번역 규칙

These rules are for whoever translates (a person or an AI such as Genspark). Source language is Korean.

1. **Translate only the `tr` field.** Never change keys, `ko` or `ctx`. Leave `tr` empty if unsure; it falls back to Korean.
2. **Keep every HTML tag and attribute exactly**: same tags, same order where possible, same `href`, `class`, `id`, `data-*`, `style`. Translate only human-readable text, including `alt`, `title`, `aria-label` values that appear inside the fragment.
3. **Script fragments** (`where` contains "문자열"): the text is glued to numbers or other fragments at run time. Read `ctx`; the part between ⟦ ⟧ is this fragment. Keep leading/trailing spaces and punctuation that connect to the neighbours (for zh/ja, drop spaces between CJK words but keep a space before a number if the Korean has one). Korean units/counters glued after a number (`'개'`, `'장'`, `'번'`, `'건'`, `'묶음'`) become the natural unit in the target language (" items", " images", " times", " cases", " clusters"), or empty if none is needed. Korean particle fragments such as `'는'`, `'은'`, `'이라고'`, `'라고'` become empty: write the single character `∅` (an empty `tr` means "not translated yet" and falls back to Korean).
4. **Placeholders and code stay as-is**: numbers, `%`, units (write them the target language's standard way only in prose, e.g. Russian МПа; never inside code), `R²`, variable and function names (`fit`, `predict`, `random_state=42`, `n_neighbors` …), file names, URLs, `\n` line breaks, math (`z = w·x + b`, `σ(z)`), circled numbers ①②③.
5. **Use the glossary** (`_glossary`, from `i18n/glossary/<lang>.json`). Character names and series labels are fixed: the Chair (Doctor), the HR Assistant, the Fruit Fly, AbaBot, Mr. Manual; verdicts Hired / On hold; "hiring bar".
6. **Voice**: explanations are plain, short, declarative textbook sentences for beginners (high-school to university). Dialogue keeps each character's voice: the Doctor is calm and polite; the Assistant is brisk and polite; the Fruit Fly is blunt, informal, skeptical; AbaBot is informal and talks about money/cost. Mr. Manual speaks in stiff regulation style ("Article 1. …").
7. **Do not add or drop facts**, numbers, or caveats. Do not "improve" the content. Keep sentence count roughly the same.
8. **Korean-specific references**: keep Korean proper nouns of sources in romanization with the original in parentheses only if needed. Links to Korean pages stay; do not invent translated URLs.
9. **Length**: UI labels (buttons, chips, table headers, short `aria-label`s) should stay about as short as the Korean.
10. **Uzbek** uses Latin script with ʻ (U+02BB) in oʻ, gʻ. **Chinese** is Simplified (mainland). **Russian** uses «ёлочки» quotes.

## 검사 항목 (`tools.py check`)

오류(빌드 전에 고친다): 태그 구성이 다름, `href`·`class`가 바뀜, 한글이 남음, 빈 번역. 경고: 앞뒤 공백이 다름(zh·ja는 제외), 용어집의 번역어가 안 보임.

## 외부 번역(Genspark 등)에 넘길 때

```
python3 i18n/tools.py init en
python3 i18n/tools.py pack en common      # → i18n/handoff/en_common.json
python3 i18n/tools.py pack en 5-4         # 편마다 하나씩
```

넘기는 파일에는 규칙(`_rules`), 그 편에 나오는 용어(`_glossary`), 조각(`items`)이 들어 있다. 요청 문구 예시:

> 첨부한 JSON의 `items` 안 각 항목에서 `tr`만 채워 같은 구조의 JSON으로 돌려줘. 대상 언어는 `_target_language`, 규칙은 `_rules`, 용어는 `_glossary`를 따른다. 키·ko·ctx는 바꾸지 않는다.

돌려받으면:

```
python3 i18n/tools.py merge en 받은파일.json
python3 i18n/tools.py check en
python3 build.py --lang en
```

파일 하나가 크면(편마다 조각 450~710개, 원문 2.4만~4.3만 자) 외부 도구의 한도에 맞춰 나눠 보내도 된다. 키가 맞는 것만 합쳐지므로 순서와 일부 누락은 상관없다.
