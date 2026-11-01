"""모델 채용 위원회 — 번역 도구 (문자열 분리 · 적용 · 검사)

원본은 한국어 소스 하나다. 한글이 든 문장 조각(세그먼트)을 뽑아
키 = sha1(한국어 원문)[:12] 로 카탈로그에 담고, 빌드할 때 같은 방법으로
조각을 찾아 번역문으로 바꿔 끼운다. 번역이 없는 조각은 한국어 그대로 남는다.

  HTML: 블록 경계(인라인이 아닌 태그) 사이의 글 덩어리 하나가 조각 하나.
        <b>, <a>, <span> 같은 인라인 태그는 조각 안에 그대로 들어간다.
        블록 태그의 alt · title · aria-label · placeholder · content 값도 조각.
  JS:   한글이 든 문자열 리터럴 하나가 조각 하나 (주석은 건너뜀).
"""
import re, hashlib, json, os

HANGUL = re.compile(r'[\uac00-\ud7a3\u3131-\u318e]')
def has_ko(s): return bool(HANGUL.search(s))
def key(s): return hashlib.sha1(s.encode('utf-8')).hexdigest()[:12]

# ───────────────────────── HTML ─────────────────────────
INLINE = set('a abbr b bdi bdo br cite code data dfn em i kbd mark q s samp small span strong sub sup time u var wbr sym'.split())
ATTRS = ('alt', 'title', 'aria-label', 'placeholder', 'content', 'data-tip', 'label')
TAG = re.compile(r'<!--.*?-->|<(/?)([a-zA-Z][\w:-]*)((?:[^>"\']|"[^"]*"|\'[^\']*\')*)>', re.S)
ATTR = re.compile(r'([\w:-]+)\s*=\s*("([^"]*)"|\'([^\']*)\')')
RAW = ('script', 'style', 'textarea', 'pre')   # 안쪽은 건드리지 않는다 (pre = 코드)


def html_segments(src):
    """[(start, end, kind)] — kind 'text' 또는 'attr'. 위치는 src 기준."""
    segs = []
    run_s = None   # 현재 글 덩어리 시작

    def close(end):
        nonlocal run_s
        if run_s is None: return
        s, e = run_s, end
        while s < e and src[s].isspace(): s += 1
        while e > s and src[e - 1].isspace(): e -= 1
        if e > s and has_ko(src[s:e]): segs.append((s, e, 'text'))
        run_s = None

    pos = 0
    for m in TAG.finditer(src):
        if m.start() < pos: continue
        # 태그 앞의 글
        if m.start() > pos and src[pos:m.start()].strip():
            if run_s is None: run_s = pos
        tok = m.group(0)
        if tok.startswith('<!--'):
            close(m.start()); pos = m.end(); continue
        closing, name = m.group(1), m.group(2).lower()
        if name in INLINE:
            if run_s is None: run_s = m.start()
            pos = m.end(); continue
        close(m.start())
        if not closing:
            for a in ATTR.finditer(m.group(3)):
                if a.group(1).lower() in ATTRS:
                    val = a.group(3) if a.group(3) is not None else a.group(4)
                    if has_ko(val):
                        vs = m.start(3) + a.start(2) + 1
                        segs.append((vs, vs + len(val), 'attr'))
            if name in RAW and not (name == 'pre' and 'code' not in m.group(3)):   # 코드가 아닌 <pre>(수식 예시 등)는 번역한다
                j = src.lower().find('</' + name, m.end())
                pos = len(src) if j < 0 else j
                continue
        pos = m.end()
    if pos < len(src) and src[pos:].strip() and run_s is None: run_s = pos
    close(len(src))
    return segs


# ───────────────────────── JS ─────────────────────────
REGEX_BEFORE = set('(,=:[!&|?{};+-*%<>~^}')
REGEX_WORDS = {'return', 'typeof', 'case', 'in', 'of', 'new', 'delete', 'void', 'throw', 'else', 'do', 'yield', 'await'}


class JSLexError(Exception): pass


def js_strings(src):
    """[(start, end, quote, tag)] — 한글이 든 문자열 리터럴 (따옴표 포함 위치). 템플릿의 ${}는 지원하지 않는다.
       바로 앞에 /*i18n:태그*/ 주석이 있으면 tag가 붙는다: 같은 한국어라도 뜻이 다르면 키를 나누는 데 쓴다
       (예: 면접 판정 /*i18n:verdict*/'합격' 과 콘크리트 판정 '합격')."""
    out, i, n, last, word, tag = [], 0, len(src), '', '', None
    while i < n:
        c = src[i]
        if c.isspace(): i += 1; continue
        if src.startswith('//', i):
            j = src.find('\n', i); i = n if j < 0 else j; continue
        if src.startswith('/*', i):
            j = src.find('*/', i + 2); e = n if j < 0 else j + 2
            m = re.fullmatch(r'/\*i18n:(\w+)\*/', src[i:e]); tag = m.group(1) if m else tag
            i = e; continue
        if c in '\'"`':
            j = i + 1
            while j < n and src[j] != c:
                if src[j] == '\\': j += 2; continue
                if c == '`' and src.startswith('${', j):
                    if has_ko(src[i:src.find('`', j)]): raise JSLexError('템플릿 ${} 안 한글: ' + src[i:i + 60])
                    d = 0
                    while j < n:
                        if src[j] == '{': d += 1
                        elif src[j] == '}':
                            d -= 1
                            if d == 0: break
                        j += 1
                if c != '`' and src[j] == '\n': raise JSLexError('닫히지 않은 문자열: ' + src[i:i + 60])
                j += 1
            if has_ko(src[i + 1:j]): out.append((i, j + 1, c, tag))
            i = j + 1; last, word, tag = 'a', '', None; continue
        if c == '/':
            if last == '' or last in REGEX_BEFORE or word in REGEX_WORDS:
                j, cls = i + 1, False
                while j < n:
                    if src[j] == '\\': j += 2; continue
                    if src[j] == '[': cls = True
                    elif src[j] == ']': cls = False
                    elif src[j] == '/' and not cls: break
                    elif src[j] == '\n': raise JSLexError('정규식 해석 실패: ' + src[i - 30:i + 40])
                    j += 1
                j += 1
                while j < n and src[j].isalpha(): j += 1
                if has_ko(src[i:j]): raise JSLexError('정규식 안 한글: ' + src[i:j])
                i = j; last, word, tag = 'a', '', None; continue
            i += 1; last, word, tag = '/', '', None; continue
        if c.isalnum() or c in '_$.' or ord(c) > 127:
            j = i
            while j < n and (src[j].isalnum() or src[j] in '_$' or (ord(src[j]) > 127 and not src[j].isspace())): j += 1
            if j == i: j = i + 1
            w = src[i:j]; last, word, tag = 'a', w, None; i = j; continue
        last, word, tag = c, '', None; i += 1
    return out


ESC = {'n': '\n', 't': '\t', 'r': '\r', '\\': '\\', "'": "'", '"': '"', '`': '`', '0': '\0', 'b': '\b', 'f': '\f', 'v': '\v', '/': '/'}


def js_decode(raw):
    o, i = [], 0
    while i < len(raw):
        c = raw[i]
        if c == '\\' and i + 1 < len(raw):
            d = raw[i + 1]
            if d == 'u':
                if raw[i + 2] == '{':
                    j = raw.index('}', i); o.append(chr(int(raw[i + 3:j], 16))); i = j + 1; continue
                o.append(chr(int(raw[i + 2:i + 6], 16))); i += 6; continue
            if d == 'x': o.append(chr(int(raw[i + 2:i + 4], 16))); i += 4; continue
            if d == '\n': i += 2; continue
            o.append(ESC.get(d, d)); i += 2; continue
        o.append(c); i += 1
    return ''.join(o)


def js_encode(s, q):
    s = s.replace('\\', '\\\\')
    if q == '`': return s.replace('`', '\\`').replace('${', '\\${')
    s = s.replace(q, '\\' + q).replace('\n', '\\n').replace('\r', '\\r').replace('\u2028', '\\u2028')
    return s.replace('</script', '<\\/script')


# ───────────────────────── 조각 목록 · 적용 ─────────────────────────
def segments(src, kind):
    """[(start, end, text, wrap, 키)] text = 번역 단위 원문, wrap = 다시 넣을 때 필요한 정보"""
    out = []
    if kind == 'html':
        for s, e, k in html_segments(src): out.append((s, e, src[s:e], ('html', k), key(src[s:e])))
    else:
        for s, e, q, tag in js_strings(src):
            text = js_decode(src[s + 1:e - 1])
            out.append((s + 1, e - 1, text, ('js', q), key(tag + '|' + text) if tag else key(text)))
    return out


class Collector(dict):
    """빌드를 '뽑기 모드'로 돌릴 때 쓰는 사전. 찾는 조각을 모두 기록하고 번역은 돌려주지 않는다."""
    def __init__(self): super().__init__(); self.seen = []
    def see(self, text, ctx, part, k=None): self.seen.append((text, ctx, part, k or key(text)))


def apply(src, kind, T, miss=None, part=None):
    segs = segments(src, kind)
    parts, pos = [], 0
    see = getattr(T, 'see', None)
    for s, e, text, (k, q), kk in segs:
        parts.append(src[pos:s])
        if see: see(text, context(src, s, e), part or kind, kk)
        t = T.get(kk)
        if t is None:
            if miss is not None: miss.append(text)
            parts.append(src[s:e])
        elif k == 'js':
            parts.append(js_encode(t, q))
        elif q == 'attr':
            parts.append(t.replace('"', '&quot;'))
        else:
            parts.append(t)
        pos = e
    parts.append(src[pos:])
    return ''.join(parts)


def context(src, s, e, w=90):
    a = src.rfind('\n', 0, s); a = max(a + 1, s - w)
    b = src.find('\n', e); b = min(len(src) if b < 0 else b, e + w)
    return (src[a:s] + '⟦' + src[s:e] + '⟧' + src[e:b]).strip()


# ───────────────────────── 검사 ─────────────────────────
TAGNAME = re.compile(r'</?([a-zA-Z][\w-]*)')
PH = re.compile(r'(&[a-z#0-9]+;|https?://\S+?(?=["\s<]|$)|\{\d+\}|\b[\w.]+\(\)|\\n)')


def check(ko, tr, kind):
    """번역문 문제 목록 (빈 목록이면 통과)"""
    p = []
    if kind in ('html', 'text'):
        a, b = sorted(TAGNAME.findall(ko)), sorted(TAGNAME.findall(tr))
        if a != b: p.append(f'태그가 다름: {a} → {b}')
        for h in re.findall(r'href="([^"]*)"', ko):
            if f'href="{h}"' not in tr: p.append(f'링크 바뀜: {h}')
        for c in re.findall(r'class="([^"]*)"', ko):
            if f'class="{c}"' not in tr: p.append(f'class 바뀜: {c}')
    if tr == '∅': return p   # 일부러 비움
    if ko.strip() and not tr.strip() and len(ko.strip()) > 2: p.append('빈 번역')
    if has_ko(tr): p.append('한글이 남음')
    if ko[:1].isspace() != tr[:1].isspace() or ko[-1:].isspace() != tr[-1:].isspace(): p.append('앞뒤 공백이 원문과 다름')
    return p
