# ===============================================================
# 모델 채용 위원회 5-0 · 채용 공고 — 매뉴얼 씨 대 데이터
#  - 라이브러리: numpy, pandas, scikit-learn (Colab 기본), ucimlrepo (없으면 자동 설치)
#  - 하는 일: 콘크리트 배합 1,030건으로 "35MPa 이상인가"를
#             ① 사람이 쓴 IF 규칙 ② 데이터로 고른 문턱 ③ 데이터로 배운 모델로 판정해 비교한다
# ===============================================================
import subprocess, sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier

# ---------------------------------------------------------------
# 1) 실기 과제지 받기: UCI 콘크리트 압축 강도 (id=165)
#    [내 데이터로 바꾸려면] 이 칸에서 df만 만들어 주면 된다 (특징 열 + 타겟 열)
try:
    from ucimlrepo import fetch_ucirepo
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ucimlrepo"], check=True)
    from ucimlrepo import fetch_ucirepo

FEATURES = ["cement", "slag", "fly_ash", "water", "superplasticizer", "coarse_agg", "fine_agg", "age"]
uci = fetch_ucirepo(id=165)
df = pd.concat([uci.data.features, uci.data.targets], axis=1)
df.columns = FEATURES + ["strength"]          # 열 순서대로 짧은 이름을 붙인다
assert df.shape == (1030, 9)
df["wb"] = df.water / (df.cement + df.slag + df.fly_ash)   # 물-결합재비
df["wc"] = df.water / df.cement                            # 물-시멘트비
df["pass35"] = (df.strength >= 35).astype(int)             # 질문: 35MPa 이상인가
df["grade"] = np.digitize(df.strength, [25, 45])           # 0 저강도 · 1 보통 · 2 고강도
print(df[FEATURES + ["strength"]].describe().T[["mean", "min", "max"]].round(1))

# ---------------------------------------------------------------
# 2) 시험지 봉인: 위원회 공통 규정 (25%를 시험지로, 강도 3등급 비율 유지, 난수 고정)
tr, te = train_test_split(df, test_size=0.25, stratify=df.grade, random_state=42)
print(f"훈련 {len(tr)}건 · 시험 {len(te)}건 · 시험지 합격 비율 {te.pass35.mean():.3f}")

def acc(rows, pred):
    return (np.asarray(pred) == rows.pass35.to_numpy()).mean()

# ---------------------------------------------------------------
# 3) 매뉴얼 씨의 작업 표준서 (엑셀이라면 =IF(AND(물결합재비<0.5, 재령>=28), "합격", "불합격"))
def manual(rows, cut=0.5, age=28, ratio="wb"):
    return ((rows[ratio] < cut) & (rows.age >= age)).astype(int)

print(f"[전문가 매뉴얼] 물-결합재비<0.5 & 재령>=28 : 훈련 {acc(tr, manual(tr)):.3f} · 시험 {acc(te, manual(te)):.3f}  ← 첫 합격선")
for cut in (0.5, 0.6):
    print(f"[신입 매뉴얼]   물-시멘트비<{cut} & 재령>=28  : 훈련 {acc(tr, manual(tr, cut, ratio='wc')):.3f} · 시험 {acc(te, manual(te, cut, ratio='wc')):.3f}")
plain = (te.slag == 0) & (te.fly_ash == 0) & (te.superplasticizer == 0)
print(f"  시험지 무리별 (전문가): 시멘트만 {acc(te[plain], manual(te[plain])):.3f} ({plain.sum()}건) · 혼화재 섞음 {acc(te[~plain], manual(te[~plain])):.3f} ({(~plain).sum()}건)")
dummy = DummyClassifier(strategy="most_frequent").fit(tr[FEATURES], tr.pass35)
print(f"[기준선] 늘 많은 쪽만 답함 : 시험 {dummy.score(te[FEATURES], te.pass35):.3f}")

# 등급까지 매기는 중첩 IF (사람이 감으로 고른 문턱)
nest = np.where((tr.wb < 0.4) & (tr.age >= 28), 2, np.where((tr.wb < 0.6) & (tr.age >= 7), 1, 0))
print(f"[중첩 IF] 3등급 훈련 정확도 {(nest == tr.grade).mean():.3f}")

# ---------------------------------------------------------------
# 4) 규칙의 모양은 그대로, 두 문턱만 데이터로 고르기 (훈련 데이터로만!)
best = (0, None, None)
for cut in np.round(np.arange(0.25, 0.805, 0.01), 2):
    for age in (1, 3, 7, 14, 28, 56, 90):
        a = acc(tr, manual(tr, cut, age))
        if a > best[0] + 1e-12:
            best = (a, cut, age)
print(f"[기계가 고른 문턱] 물-결합재비<{best[1]} & 재령>={best[2]} : 훈련 {best[0]:.3f} · 시험 {acc(te, manual(te, best[1], best[2])):.3f}")

# ---------------------------------------------------------------
# 5) 깊이 1 결정 트리 = 데이터가 고른 문턱 하나짜리 규칙
stump = DecisionTreeClassifier(max_depth=1).fit(tr[["wb"]], tr.pass35)
print(f"[결정 트리 깊이 1] 물-결합재비 <= {stump.tree_.threshold[0]:.3f} 이면 합격 → 시험 {stump.score(te[['wb']], te.pass35):.3f}")

# ---------------------------------------------------------------
# 6) 모양까지 데이터로: 로지스틱 회귀 미리 보기 (5-4에서 면접)
def logreg(cols):
    X_tr, X_te = tr[cols].to_numpy(), te[cols].to_numpy()
    ss = StandardScaler().fit(X_tr)               # 표준화는 훈련 데이터로만 맞춘다
    m = LogisticRegression(C=1, max_iter=5000).fit(ss.transform(X_tr), tr.pass35)
    return m.score(ss.transform(X_tr), tr.pass35), m.score(ss.transform(X_te), te.pass35)

tr["logage"], te["logage"] = np.log(tr.age), np.log(te.age)
for name, cols in [("날것 특징 8개", FEATURES), ("전문가 특징 2개 (물-결합재비, log 재령)", ["wb", "logage"])]:
    a, b = logreg(cols)
    print(f"[로지스틱 회귀 · {name}] 훈련 {a:.3f} · 시험 {b:.3f}")
