# ===============================================================
# 모델 채용 위원회 5-4 · 로지스틱 회귀 면접
#  - 라이브러리: numpy, pandas, scikit-learn (Colab 기본) + ucimlrepo (자동 설치)
#  - 하는 일: 콘크리트 배합 1,030건으로 "35MPa 합격인가"를 학습하고
#             시그모이드·임계값·소프트맥스·규제를 직접 확인한 뒤 합격선과 비교한다
# ===============================================================
import subprocess, sys
import numpy as np
import pandas as pd
import sklearn
from scipy.special import expit, softmax
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, roc_auc_score
try:
    from ucimlrepo import fetch_ucirepo
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ucimlrepo"], check=True)
    from ucimlrepo import fetch_ucirepo

np.set_printoptions(precision=3, suppress=True)
print("scikit-learn", sklearn.__version__)

# ---------------------------------------------------------------
# 1) 실기 과제지 읽기: 콘크리트 배합 1,030건 (UCI id=165, CC BY 4.0)
#    [내 데이터로 바꾸려면] 이 칸의 데이터 읽기, FEATURES, PASS_MPA만 바꾼다
FEATURES = ["cement", "slag", "fly_ash", "water", "superplasticizer", "coarse_agg", "fine_agg", "age"]
uci = fetch_ucirepo(id=165)
df = pd.concat([uci.data.features, uci.data.targets], axis=1)
df.columns = FEATURES + ["strength"]          # 열 순서대로 짧은 이름을 붙인다
assert df.shape == (1030, 9)
PASS_MPA = 35                                  # 수업용 합격 기준 (설계 기준이 아니다)
X = df[FEATURES].to_numpy()
y = (df["strength"] >= PASS_MPA).astype(int).to_numpy()      # 합격 1, 불합격 0
grade = np.digitize(df["strength"], [25, 45])                 # 0 저강도 · 1 보통 · 2 고강도
print(f"배합 {len(df)}건 · 합격 비율 {y.mean():.3f}")

# ---------------------------------------------------------------
# 2) 시험지 봉인: 위원회 공통 규정 (25%를 시험지로, 강도 등급 비율 유지, 난수 고정)
#    표준화는 훈련 데이터로만 맞춘다(fit). 시험지는 transform만 한다
idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=0.25, stratify=grade, random_state=42)
ss = StandardScaler().fit(X[idx_tr])
A, B = ss.transform(X[idx_tr]), ss.transform(X[idx_te])
y_tr, y_te, g_tr, g_te = y[idx_tr], y[idx_te], grade[idx_tr], grade[idx_te]
print(f"훈련 {len(A)}건 · 시험 {len(B)}건")

# ---------------------------------------------------------------
# 3) 과제 A · 이진 분류: 35MPa 합격인가
#    선형 점수 z = w·x + b → 시그모이드 σ(z) = 1 / (1 + e^(−z)) = 합격 확률
lr = LogisticRegression(C=1, max_iter=1000).fit(A, y_tr)
print("클래스 순서:", lr.classes_, "→ predict_proba의 두 번째 열이 합격(1) 확률")
print(pd.Series(lr.coef_[0], index=FEATURES).round(2).to_dict(), "절편", lr.intercept_.round(2))
z = lr.decision_function(B[:4])
print("z      :", z)
print("σ(z)   :", expit(z))
print("proba  :", lr.predict_proba(B[:4])[:, 1])        # 같아야 한다
print(f"[과제 A] 훈련 정확도 {lr.score(A, y_tr):.3f} · 시험 정확도 {lr.score(B, y_te):.3f}")

# ---------------------------------------------------------------
# 4) 성과 평가: 혼동행렬 — 두 가지 틀림은 값이 다르다
pred = lr.predict(B)
cm = confusion_matrix(y_te, pred)
print(pd.DataFrame(cm, index=["실제 불합격", "실제 합격"], columns=["예측 불합격", "예측 합격"]))
print(f"강도 미달인데 '합격' (위험한 출하): {cm[0, 1]}건 · 합격인데 '불합격' (재시험 비용): {cm[1, 0]}건")
print(classification_report(y_te, pred, target_names=["불합격", "합격"], digits=3))

# ---------------------------------------------------------------
# 5) 임계값 바꾸기: "합격"이라고 말하는 문턱을 올리면 위험한 출하가 줄어든다
p = lr.predict_proba(B)[:, 1]
print(f"ROC-AUC {roc_auc_score(y_te, p):.3f}")
for th in (0.3, 0.5, 0.7):
    guess = (p >= th).astype(int)
    fp = int(((guess == 1) & (y_te == 0)).sum())
    print(f"  임계값 {th}: 정확도 {(guess == y_te).mean():.3f} · 정밀도 {precision_score(y_te, guess):.3f} · 재현율 {recall_score(y_te, guess):.3f} · 미달 출하 {fp}건")

# ---------------------------------------------------------------
# 6) 과제 B · 다중 분류: 강도 3등급 (소프트맥스)
lr3 = LogisticRegression(C=1, max_iter=1000).fit(A, g_tr)
print(f"[과제 B] 훈련 {lr3.score(A, g_tr):.3f} · 시험 {lr3.score(B, g_te):.3f} · 기준선(늘 보통) {np.mean(g_te == 1):.3f}")
print(pd.DataFrame(lr3.predict_proba(B[:3]), columns=["저강도", "보통", "고강도"]).round(3))
print("소프트맥스 직접 계산과 같은가?", np.allclose(softmax(lr3.decision_function(B[:3]), axis=1), lr3.predict_proba(B[:3])))
print(pd.DataFrame(confusion_matrix(g_te, lr3.predict(B)), index=["저", "보통", "고"], columns=["저", "보통", "고"]))

# ---------------------------------------------------------------
# 7) 좋은 특징: 매뉴얼 씨의 두 숫자(물-결합재비, log 재령)만 주면?
wb = (df["water"] / (df["cement"] + df["slag"] + df["fly_ash"])).to_numpy()
Z = np.column_stack([wb, np.log10(df["age"].to_numpy())])
s2 = StandardScaler().fit(Z[idx_tr])
lr2 = LogisticRegression(C=1).fit(s2.transform(Z[idx_tr]), y_tr)
print(f"[전문가 특징 2개] 시험 정확도 {lr2.score(s2.transform(Z[idx_te]), y_te):.3f}  (특징 8개: {lr.score(B, y_te):.3f})")

# ---------------------------------------------------------------
# 8) 규제 손잡이 C: 작을수록 규제가 세다 (Ridge의 alpha와 방향이 반대)
for C in (0.001, 0.01, 0.1, 1, 10, 1000):
    m = LogisticRegression(C=C, max_iter=5000).fit(A, y_tr)
    print(f"  C={C:<6g} 훈련 {m.score(A, y_tr):.3f} · 시험 {m.score(B, y_te):.3f} · 가중치 크기 {np.abs(m.coef_).mean():.2f}")

# ---------------------------------------------------------------
# 9) L1 규제(라쏘와 같은 벌점): scikit-learn 버전에 따라 쓰는 법이 다르다
#    1.8부터 penalty 인자는 사라지는 중이고 l1_ratio로 정한다 (l1_ratio=1 → L1)
ver = tuple(int(v) for v in sklearn.__version__.split(".")[:2])
if ver >= (1, 8):
    l1 = LogisticRegression(l1_ratio=1, C=0.05, solver="saga", max_iter=20000)
else:
    l1 = LogisticRegression(penalty="l1", C=0.05, solver="saga", max_iter=20000)
l1.fit(A, y_tr)
print(f"[L1, C=0.05] 시험 정확도 {l1.score(B, y_te):.3f} · 0이 된 가중치:", [f for f, w in zip(FEATURES, l1.coef_[0]) if w == 0])

# ---------------------------------------------------------------
# 10) 합격선과 비교: 매뉴얼 씨(5-0, 물-결합재비 < 0.5 이고 재령 ≥ 28일) · 늘 다수 쪽만 답하는 기준선
rule = ((wb < 0.5) & (df["age"].to_numpy() >= 28)).astype(int)
manual = (rule[idx_te] == y_te).mean()
dummy = DummyClassifier(strategy="most_frequent").fit(A, y_tr)
score = lr.score(B, y_te)
print(f"기준선 {dummy.score(B, y_te):.3f} · 매뉴얼 씨 {manual:.3f} · 로지스틱 회귀 {score:.3f}")
print("판정:", "합격" if score > manual and score > dummy.score(B, y_te) else "보류")
