# Brief for an episode-translation agent (language LANG)

You translate Korean strings of an educational ML HTML series ("The Model Hiring Committee": 15 ML models interview for jobs at a concrete factory; supervised 5-1..5-9 use the UCI concrete compressive-strength data (35 MPa pass/fail, strength regression); unsupervised 5-10..5-15 use sklearn digits, 1,797 8×8 handwritten images whose labels are sealed in an "answer envelope" used only for grading; 5-12 DBSCAN, 5-2 linear regression, 5-5 naive Bayes are ON HOLD, all others hired; 5-15 is the finale: 12 hired / 3 on hold of 15).

Files: each assigned file /home/claude/series5/i18n/LANG/<name>.json is an object keyed by id; values have "ko" (source), "ctx" (surrounding source; the fragment is between ⟦ ⟧), "where" ("페이지 HTML" = HTML fragment; contains "문자열" = JavaScript string fragment glued to numbers/other strings at runtime), "tr" (empty — fill it).

Read FIRST and follow exactly:
1. "## 번역 규칙" (rules 1–10) in /home/claude/series5/i18n/README.md
2. /home/claude/series5/i18n/LANG/_conventions.md (fixed decisions for this language)
3. Glossary /home/claude/series5/i18n/glossary/LANG.json
Reference: the reviewed English of the same key is in /home/claude/series5/i18n/en/<same file>.json ("tr") — use it for meaning and to see how fragments combine; translate from the Korean. Verdict keys (ko 합격/불합격 whose English is Hired/Rejected) differ from concrete Pass/Fail keys. In common.json, leave the language-menu entry '한국어' empty.

Essentials: keep every HTML tag/attribute identical (href, class, id, data-*, style); translate alt/title/aria-label; JS fragments: read ctx so the assembled sentence reads naturally; Korean counters after numbers → the target's unit or ∅; particle-only fragments → ∅ (∅ = intentionally empty); never leave Hangul; keep numbers, identifiers, math, URLs, \n. Plain textbook style; keep each character's voice; don't add or drop facts. Shared fragments in common.json are handled by another agent unless common.json is in your list.

Method: a Python script; process entries in batches of ~60 (print id/ko/ctx + English tr, write translations to a dict, save immediately with json.dump(ensure_ascii=False, indent=1)). Keep scripts in your own scratch folder (given in your prompt; e.g. qa/work/LANG_<group>/). Other agents work in parallel on other files: never touch other files; never run extract.py, init or merge.

Finish: cd /home/claude/series5 && python3 i18n/tools.py check LANG ; open i18n/LANG/_report.md and fix every **오류** under your files' headings; fix genuine glossary misses among warnings (ignore false matches); re-run until 0 errors in your sections. Then python3 build.py --lang LANG <your episodes> must run without exceptions.
Report back briefly: counts per file, errors remaining, ≤5 judgement calls worth reviewing.
