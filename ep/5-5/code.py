# ===============================================================
# 모델 채용 위원회 5-5 · 나이브 베이즈 면접
#  - 라이브러리: numpy, pandas, scipy, scikit-learn (Colab 기본) + ucimlrepo (자동 설치)
#  - 하는 일: 콘크리트 배합 1,030건으로 "35MPa 합격인가"를 가우시안 나이브 베이즈로 학습하고
#             베이즈 정리를 손으로 다시 계산한 뒤, 독립 가정이 깨지는 곳을 찾아 합격선과 비교한다
# ===============================================================
import subprocess, sys
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import norm
from scipy.special import softmax
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix, classification_report, log_loss
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
#    나이브 베이즈는 특징마다 따로 평균·분산을 재므로 표준화가 필요 없다 (6번 칸에서 확인)
idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=0.25, stratify=grade, random_state=42)
A, B = X[idx_tr], X[idx_te]
y_tr, y_te, g_tr, g_te = y[idx_tr], y[idx_te], grade[idx_tr], grade[idx_te]
print(f"훈련 {len(A)}건 · 시험 {len(B)}건")

# ---------------------------------------------------------------
# 3) 과제 A · 이진 분류: 35MPa 합격인가
#    학습 = 클래스마다 특징의 평균과 분산을 세는 일 (반복 없이 한 번 훑기)
nb = GaussianNB().fit(A, y_tr)
print("사전 확률 P(불합격), P(합격):", nb.class_prior_.round(3))
print(pd.DataFrame({"불합격 평균": nb.theta_[0], "불합격 표준편차": np.sqrt(nb.var_[0]),
                    "합격 평균": nb.theta_[1], "합격 표준편차": np.sqrt(nb.var_[1])}, index=FEATURES).round(1))
print("외운 숫자:", nb.theta_.size + nb.var_.size + nb.class_prior_.size, "개 (평균 16 + 분산 16 + 사전 확률 2)")
print(f"[과제 A] 훈련 정확도 {nb.score(A, y_tr):.3f} · 시험 정확도 {nb.score(B, y_te):.3f}")

# ---------------------------------------------------------------
# 4) 베이즈 정리를 손으로: 사후 ∝ 사전 확률 × Π 특징별 가능도
#    곱하면 숫자가 너무 작아지므로 로그를 씌워 더한다
x0 = B[0]
sd = np.sqrt(nb.var_)
log_joint = np.log(nb.class_prior_) + norm.logpdf(x0, nb.theta_, sd).sum(axis=1)
print("시험 첫 배합:", dict(zip(FEATURES, x0.tolist())), "· 실제", round(float(df["strength"].iloc[idx_te[0]]), 1), "MPa")
print("특징별 증거 log[P(x|합격)/P(x|불합격)]:", (norm.logpdf(x0, nb.theta_[1], sd[1]) - norm.logpdf(x0, nb.theta_[0], sd[0])).round(2))
print("손 계산 사후 확률 :", softmax(log_joint))
print("predict_proba     :", nb.predict_proba(B[:1])[0])          # 같아야 한다

# ---------------------------------------------------------------
# 5) 성과 평가: 혼동행렬 — 어디서 틀렸나
pred = nb.predict(B)
cm = confusion_matrix(y_te, pred)
print(pd.DataFrame(cm, index=["실제 불합격", "실제 합격"], columns=["예측 불합격", "예측 합격"]))
print(classification_report(y_te, pred, target_names=["불합격", "합격"], digits=3))
age_te = B[:, 7]
k28 = (age_te == 28) & (y_te == 1)
print(f"재령 28일 합격 배합 {k28.sum()}건 중 '불합격'으로 놓친 배합 {(k28 & (pred == 0)).sum()}건")

# ---------------------------------------------------------------
# 6) 표준화 무관: 단위를 바꿔도 두 클래스의 가능도가 같은 비율로 바뀐다
ss = StandardScaler().fit(A)
print(f"표준화한 뒤 시험 정확도 {GaussianNB().fit(ss.transform(A), y_tr).score(ss.transform(B), y_te):.3f}")
#    종류를 잘못 고르면: MultinomialNB는 kg·일을 "단어 횟수"처럼 센다 → 단위만 바꿔도 점수가 바뀐다
for unit, k in (("일", 1), ("주", 7)):
    A_u, B_u = A.copy(), B.copy()
    A_u[:, 7] /= k
    B_u[:, 7] /= k
    print(f"  MultinomialNB, 재령 단위 {unit}: 시험 {MultinomialNB().fit(A_u, y_tr).score(B_u, y_te):.3f}"
          f" · GaussianNB: {GaussianNB().fit(A_u, y_tr).score(B_u, y_te):.3f}")

# ---------------------------------------------------------------
# 7) 독립 가정 점검 ①: 같은 클래스 안에서도 특징끼리 얽혀 있다
for c, name in [(0, "불합격"), (1, "합격")]:
    R = pd.DataFrame(A[y_tr == c], columns=FEATURES).corr()
    pairs = R.where(np.triu(np.ones_like(R, dtype=bool), 1)).stack().sort_values(key=abs, ascending=False)
    print(f"{name} 안에서 가장 얽힌 쌍:", {f"{a}-{b}": round(v, 2) for (a, b), v in pairs.head(3).items()})
#    같은 증거를 여러 번 세면: 정확도는 비슷한데 확률이 0이나 1로 쏠린다 (과신)
for k in (1, 3, 5):
    Ak = np.column_stack([A] + [A[:, [0]]] * (k - 1))
    Bk = np.column_stack([B] + [B[:, [0]]] * (k - 1))
    m = GaussianNB().fit(Ak, y_tr)
    p = m.predict_proba(Bk)[:, 1]
    print(f"  시멘트를 {k}번 세기: 정확도 {m.score(Bk, y_te):.3f} · 확률 0.01 미만·0.99 초과 {np.mean((p < .01) | (p > .99)):.3f} · 로그 손실 {log_loss(y_te, p):.3f}")

# ---------------------------------------------------------------
# 8) 독립 가정 점검 ②: 재령은 종 모양이 아니다 (1·3·7·14·28·56·90…일로 띄엄띄엄)
print("재령 값과 배합 수:", df["age"].value_counts().sort_index().to_dict())
print("종 모양으로 맞추면 재령 < 0일에 놓이는 확률 (불합격, 합격):", norm.cdf(0, nb.theta_[:, 7], sd[:, 7]).round(3))
wb = (df["water"] / (df["cement"] + df["slag"] + df["fly_ash"])).to_numpy()
la = np.log10(df["age"].to_numpy())
sets = {"날것 8개": X,
        "재령 → log 재령": np.column_stack([X[:, :7], la]),
        "8개 + 물-결합재비 + log 재령": np.column_stack([X, wb, la]),
        "물-결합재비 + log 재령 둘만": np.column_stack([wb, la])}
for name, Z in sets.items():
    m = GaussianNB().fit(Z[idx_tr], y_tr)
    print(f"  {name:<22} 훈련 {m.score(Z[idx_tr], y_tr):.3f} · 시험 {m.score(Z[idx_te], y_te):.3f}")

# ---------------------------------------------------------------
# 9) 과제 B · 다중 분류: 강도 3등급 (클래스가 셋이어도 같은 방법)
nb3 = GaussianNB().fit(A, g_tr)
print(f"[과제 B] 훈련 {nb3.score(A, g_tr):.3f} · 시험 {nb3.score(B, g_te):.3f} · 기준선(늘 보통) {np.mean(g_te == 1):.3f}")
print(pd.DataFrame(confusion_matrix(g_te, nb3.predict(B)), index=["저", "보통", "고"], columns=["저", "보통", "고"]))

# ---------------------------------------------------------------
# 10) 사전 확률 바꾸기: 데이터 비율 대신 정해 준 값 (임계값을 옮기는 것과 같다)
for pr in ([0.5, 0.5], [0.3, 0.7], [0.7, 0.3]):
    m = GaussianNB(priors=pr).fit(A, y_tr)
    c = confusion_matrix(y_te, m.predict(B))
    print(f"  priors={pr}: 시험 {m.score(B, y_te):.3f} · 미달인데 '합격' {c[0, 1]}건 · 합격인데 '불합격' {c[1, 0]}건")

# ---------------------------------------------------------------
# 11) 회귀는? 강도(MPa) 자체를 클래스로 두면 클래스 하나에 배합이 한두 건뿐이다
n_cls = df["strength"].round(2).nunique()
print(f"서로 다른 강도 값 {n_cls}개 → 클래스마다 평균·분산을 잴 수 없다. 구간으로 나누면(3등급) 분류가 된다")

# ---------------------------------------------------------------
# 12) 합격선과 비교: 매뉴얼 씨(5-0) · 늘 다수 쪽만 답하는 기준선
rule = ((wb < 0.5) & (df["age"].to_numpy() >= 28)).astype(int)
manual = (rule[idx_te] == y_te).mean()
dummy = DummyClassifier(strategy="most_frequent").fit(A, y_tr).score(B, y_te)
score = nb.score(B, y_te)
print(f"기준선 {dummy:.3f} · 매뉴얼 씨 {manual:.3f} · 나이브 베이즈 {score:.3f}")
print("판정:", "합격" if score > manual and score > dummy else "보류")
