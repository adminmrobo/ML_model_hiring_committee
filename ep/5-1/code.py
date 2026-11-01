# ===============================================================
# 모델 채용 위원회 5-1 · KNN(k-최근접 이웃) 면접
#  - 라이브러리: numpy, pandas, scikit-learn (Colab 기본) + ucimlrepo (자동 설치)
#  - 하는 일: 콘크리트 배합 1,030건으로 "강도가 몇 MPa인가"(회귀)와 "35MPa 합격인가"(분류)를
#             같은 이웃에게 묻고, 표준화·k·거리 가중을 바꿔 가며 회귀 합격선과 비교한다
# ===============================================================
import subprocess, sys, time
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.dummy import DummyRegressor
from sklearn.metrics import r2_score, mean_absolute_error, confusion_matrix
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
y = df["strength"].to_numpy()                  # 회귀 타겟: 압축 강도(MPa)
passed = (y >= PASS_MPA).astype(int)           # 분류 타겟: 35MPa 합격 1, 불합격 0
grade = np.digitize(y, [25, 45])               # 0 저강도 · 1 보통 · 2 고강도 (나누기 기준)
print(f"배합 {len(df)}건 · 강도 평균 {y.mean():.1f}MPa · 합격 비율 {passed.mean():.3f}")

# ---------------------------------------------------------------
# 2) 시험지 봉인: 위원회 공통 규정 (25%를 시험지로, 강도 등급 비율 유지, 난수 고정)
#    표준화는 훈련 데이터로만 맞춘다(fit). 시험지는 transform만 한다
idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=0.25, stratify=grade, random_state=42)
ss = StandardScaler().fit(X[idx_tr])
A, B = ss.transform(X[idx_tr]), ss.transform(X[idx_te])
y_tr, y_te = y[idx_tr], y[idx_te]
print(f"훈련 {len(A)}건 · 시험 {len(B)}건")

# ---------------------------------------------------------------
# 3) 회귀 합격선: 매뉴얼 씨의 현장 경험식 (학습 없음, 사람의 지식)
#    28일 강도 = 96.5 / 7^(물-결합재비)  (에이브럼스 식, 물-시멘트비를 물-결합재비로 넓힘)
#    재령 보정 = 28일 강도 × t / (4 + 0.85t)  (ACI 209R, 습윤 양생 보통 시멘트)
wb = df["water"] / (df["cement"] + df["slag"] + df["fly_ash"])
age = df["age"].to_numpy()
manual = (96.5 / 7 ** wb * age / (4 + 0.85 * age)).to_numpy()
LINE_R2 = r2_score(y_te, manual[idx_te])
dummy = DummyRegressor().fit(A, y_tr)          # 기준선: 늘 훈련 평균을 답한다
print(f"매뉴얼 경험식  R² {LINE_R2:.3f} · MAE {mean_absolute_error(y_te, manual[idx_te]):.2f}MPa  ← 회귀 합격선")
print(f"기준선(평균)   R² {dummy.score(B, y_te):.3f} · MAE {mean_absolute_error(y_te, dummy.predict(B)):.2f}MPa")

# ---------------------------------------------------------------
# 4) KNN을 직접 짜 보기: 거리를 모두 재고, 가까운 k개의 강도를 평균 낸다. 학습(fit)은 "외우기"뿐이다
def knn_predict(Z_train, y_train, Z_query, k, skip_self=False):
    d2 = ((Z_query[:, None, :] - Z_train[None, :, :]) ** 2).sum(axis=2)   # 거리 제곱 (질문 수 × 훈련 수)
    if skip_self:
        np.fill_diagonal(d2, np.inf)                                     # 자기 자신은 이웃에서 뺀다
    nb = np.argsort(d2, axis=1, kind="stable")[:, :k]                    # 가까운 k개의 번호
    return y_train[nb].mean(axis=1), nb

mine, _ = knn_predict(A, y_tr, B, 5)
knr = KNeighborsRegressor(n_neighbors=5).fit(A, y_tr)
same = np.isclose(mine, knr.predict(B)).sum()
print(f"직접 짠 KNN R² {r2_score(y_te, mine):.3f} · scikit-learn R² {knr.score(B, y_te):.3f} · 같은 답 {same}/{len(B)}건")
#    다른 답이 몇 건 나올 수 있다. 똑같은 배합을 여러 번 시험한 줄이 있어 k번째 자리에 거리가 같은 이웃이 둘 이상이면
#    누구를 고를지가 구현마다 다르기 때문이다. 점수는 소수 셋째 자리까지 거의 같다

# ---------------------------------------------------------------
# 5) 표준화 전후로 이웃이 바뀐다: 시험 배합 하나 (표의 400번째 줄 = 행 번호 399, 28일, 실제 39.3MPa)
q = int(np.where(idx_te == 399)[0][0])
show = ["cement", "slag", "water", "superplasticizer", "coarse_agg", "age", "strength"]
print(df.loc[[399], show].round(1).to_string())
for name, Ztr, Zq in [("표준화 안 함", X[idx_tr], X[idx_te]), ("표준화", A, B)]:
    p, nb = knn_predict(Ztr, y_tr, Zq[[q]], 5)
    print(f"[{name}] 이웃 5개 → 평균 {p[0]:.1f}MPa")
    print(df.loc[idx_tr[nb[0]], show].round(1).to_string())

# ---------------------------------------------------------------
# 6) 성과 평가: 봉인된 시험지 258건, k = 5
for name, model, Ztr, Zte in [("표준화 안 함", KNeighborsRegressor(n_neighbors=5), X[idx_tr], X[idx_te]),
                              ("표준화", KNeighborsRegressor(n_neighbors=5), A, B),
                              ("표준화 + 거리 가중", KNeighborsRegressor(n_neighbors=5, weights="distance"), A, B)]:
    pr = model.fit(Ztr, y_tr).predict(Zte)
    r2 = r2_score(y_te, pr)
    print(f"  {name:<10} R² {r2:.3f} · MAE {mean_absolute_error(y_te, pr):.2f}MPa → {'합격' if r2 > LINE_R2 else '보류'}")

# ---------------------------------------------------------------
# 7) 수습 기간 = k 고르기. 시험지는 보지 않고 훈련 데이터 안에서 "하나 빼고 맞히기"(LOO)로 고른다
#    훈련 R²는 자기 자신이 이웃에 들어가므로 k = 1이면 거의 1이다(외운 답)
print("   k  훈련R²  LOO R²  (표준화 안 함 LOO)")
for k in (1, 3, 4, 5, 7, 10, 15, 30):
    tr_r2 = r2_score(y_tr, knn_predict(A, y_tr, A, k)[0])
    loo = r2_score(y_tr, knn_predict(A, y_tr, A, k, skip_self=True)[0])
    loo_raw = r2_score(y_tr, knn_predict(X[idx_tr], y_tr, X[idx_tr], k, skip_self=True)[0])
    print(f"  {k:>2}  {tr_r2:.3f}   {loo:.3f}   ({loo_raw:.3f})")

# ---------------------------------------------------------------
# 8) 좋은 특징: 매뉴얼 씨의 두 숫자(물-결합재비, log 재령)만으로 이웃을 찾으면?
Z = np.column_stack([wb, np.log10(age)])
s2 = StandardScaler().fit(Z[idx_tr])
for k in (1, 5, 50):
    m2 = KNeighborsRegressor(n_neighbors=k).fit(s2.transform(Z[idx_tr]), y_tr)
    print(f"  [특징 2개] k={k:<2} 훈련 R² {m2.score(s2.transform(Z[idx_tr]), y_tr):.3f} · 시험 R² {m2.score(s2.transform(Z[idx_te]), y_te):.3f}")

# ---------------------------------------------------------------
# 9) 같은 이웃, 다른 질문: 다수결이면 분류(35MPa 합격인가)
p_tr, p_te = passed[idx_tr], passed[idx_te]
rule = ((wb < 0.5) & (age >= 28)).astype(int).to_numpy()       # 5-0 매뉴얼 씨의 작업 표준서
line_acc = (rule[idx_te] == p_te).mean()
for name, Ztr, Zte in [("표준화 안 함", X[idx_tr], X[idx_te]), ("표준화", A, B)]:
    clf = KNeighborsClassifier(n_neighbors=5).fit(Ztr, p_tr)
    print(f"  [분류 · {name}] 시험 정확도 {clf.score(Zte, p_te):.3f}  (분류 합격선 {line_acc:.3f})")
print(confusion_matrix(p_te, KNeighborsClassifier(n_neighbors=5).fit(A, p_tr).predict(B)))

# ---------------------------------------------------------------
# 10) 예산: 학습은 공짜, 예측이 비싸다
t0 = time.perf_counter(); knr.fit(A, y_tr); t_fit = time.perf_counter() - t0
t0 = time.perf_counter(); knr.predict(B); t_pred = time.perf_counter() - t0
print(f"외우는 숫자 {A.size}개(훈련 {A.shape[0]}건 × 특징 {A.shape[1]}개) · 예측 한 건마다 거리 {A.shape[0]}번")
print(f"fit {t_fit * 1000:.1f}ms · 시험 {len(B)}건 predict {t_pred * 1000:.1f}ms  (컴퓨터마다 다르다)")

# ---------------------------------------------------------------
# 11) 판정: 시험 R²가 회귀 합격선(매뉴얼 경험식)과 기준선을 모두 넘으면 합격
score = knr.score(B, y_te)
print(f"기준선 {dummy.score(B, y_te):.3f} · 매뉴얼 경험식 {LINE_R2:.3f} · KNN(k=5, 표준화) {score:.3f}")
print("판정:", "합격" if score > LINE_R2 and score > dummy.score(B, y_te) else "보류")
