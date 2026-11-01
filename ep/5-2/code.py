# ===============================================================
# 모델 채용 위원회 5-2 · 선형 회귀 면접
#  - 라이브러리: numpy, pandas, scikit-learn (Colab 기본) + ucimlrepo (자동 설치)
#  - 하는 일: 콘크리트 배합 1,030건으로 압축 강도(MPa)를 맞히는 선형 회귀를 학습하고
#             잔차·MAE·RMSE·R², 정규방정식과 경사 하강, 좋은 특징을 확인한 뒤 합격선과 비교한다
# ===============================================================
import subprocess, sys
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
try:
    from sklearn.metrics import root_mean_squared_error          # 1.4부터 있다
except ImportError:
    def root_mean_squared_error(y_true, y_pred):                 # 1.3 이하용
        return np.sqrt(mean_squared_error(y_true, y_pred))
try:
    from ucimlrepo import fetch_ucirepo
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ucimlrepo"], check=True)
    from ucimlrepo import fetch_ucirepo

np.set_printoptions(precision=3, suppress=True)
print("scikit-learn", sklearn.__version__)

# ---------------------------------------------------------------
# 1) 실기 과제지 읽기: 콘크리트 배합 1,030건 (UCI id=165, CC BY 4.0)
#    [내 데이터로 바꾸려면] 이 칸의 데이터 읽기, FEATURES, 타겟 열만 바꾼다
FEATURES = ["cement", "slag", "fly_ash", "water", "superplasticizer", "coarse_agg", "fine_agg", "age"]
uci = fetch_ucirepo(id=165)
df = pd.concat([uci.data.features, uci.data.targets], axis=1)
df.columns = FEATURES + ["strength"]          # 열 순서대로 짧은 이름을 붙인다
assert df.shape == (1030, 9)
X = df[FEATURES].to_numpy()
y = df["strength"].to_numpy()                 # 타겟: 압축 강도(MPa) 그 자체 → 회귀
grade = np.digitize(y, [25, 45])              # 나누기 규정에만 쓰는 강도 3등급
print(f"배합 {len(df)}건 · 강도 평균 {y.mean():.2f}MPa · 표준편차 {y.std():.2f}MPa")

# ---------------------------------------------------------------
# 2) 시험지 봉인: 위원회 공통 규정 (25%를 시험지로, 강도 등급 비율 유지, 난수 고정)
#    표준화는 훈련 데이터로만 맞춘다(fit). 시험지는 transform만 한다
idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=0.25, stratify=grade, random_state=42)
ss = StandardScaler().fit(X[idx_tr])
A, B = ss.transform(X[idx_tr]), ss.transform(X[idx_te])
y_tr, y_te = y[idx_tr], y[idx_te]
print(f"훈련 {len(A)}건 · 시험 {len(B)}건")

def report(name, pred_te, pred_tr=None):
    s = f"{name:<28} 시험 R² {r2_score(y_te, pred_te):6.3f} · MAE {mean_absolute_error(y_te, pred_te):5.2f} · RMSE {root_mean_squared_error(y_te, pred_te):5.2f}"
    if pred_tr is not None:
        s += f" | 훈련 R² {r2_score(y_tr, pred_tr):.3f}"
    print(s)

# ---------------------------------------------------------------
# 3) 합격선: 기준선(늘 훈련 평균) · 매뉴얼 씨의 현장 경험식 (학습 없음)
#    28일 강도 f28 = 96.5 / 7^(물-결합재비)  (에이브럼스 식, 물-시멘트비 대신 물-결합재비)
#    재령 보정 f(t) = f28 · t / (4 + 0.85·t)  (ACI 209R, 습윤 양생 보통 시멘트)
dummy = DummyRegressor(strategy="mean").fit(A, y_tr)
report("기준선 (늘 훈련 평균)", dummy.predict(B), dummy.predict(A))
wb = (df["water"] / (df["cement"] + df["slag"] + df["fly_ash"])).to_numpy()
age = df["age"].to_numpy()
manual = 96.5 / 7 ** wb * age / (4 + 0.85 * age)
report("매뉴얼 씨 경험식", manual[idx_te], manual[idx_tr])
wc = (df["water"] / df["cement"]).to_numpy()
report("  (물-시멘트비로 쓰면)", (96.5 / 7 ** wc * age / (4 + 0.85 * age))[idx_te])

# ---------------------------------------------------------------
# 4) 지원자: 선형 회귀, 특징 8개 그대로 (표준화)
#    ŷ = w₁x₁ + … + w₈x₈ + b, 잔차 제곱합 Σ(y − ŷ)²이 가장 작은 w, b를 고른다(최소제곱)
lin = LinearRegression().fit(A, y_tr)
print(pd.Series(lin.coef_, index=FEATURES).round(2).to_dict(), "절편", round(lin.intercept_, 2))
report("선형 회귀 (특징 8개)", lin.predict(B), lin.predict(A))
print("시험 앞 3건  실제:", y_te[:3], " 예측:", lin.predict(B[:3]), " 잔차:", y_te[:3] - lin.predict(B[:3]))

# ---------------------------------------------------------------
# 5) 같은 답을 찾는 두 길: 정규방정식(한 번에) vs 경사 하강(조금씩)
A1 = np.column_stack([np.ones(len(A)), A])                       # 맨 앞에 1(절편) 열
beta = np.linalg.solve(A1.T @ A1, A1.T @ y_tr)                   # (XᵀX)β = Xᵀy
print("정규방정식 절편·가중치:", beta.round(2))
w, b, lr_ = np.zeros(8), 0.0, 0.2
for t in range(1, 1001):
    e = y_tr - (A @ w + b)                                        # 잔차
    w -= lr_ * (-2 * A.T @ e / len(A))                            # MSE의 기울기
    b -= lr_ * (-2 * e.mean())
    if t in (10, 100, 1000):
        print(f"  경사 하강 {t:4d}번: 훈련 MSE {np.mean(e ** 2):7.2f} · 가중치 {w.round(2)}")
print("scikit-learn과 같은가?", np.allclose(beta[1:], lin.coef_, atol=1e-6), np.allclose(w, lin.coef_, atol=1e-3))

# ---------------------------------------------------------------
# 6) 잔차에 무늬가 있나: 재령 구간별 평균 잔차 (훈련 데이터)
res = y_tr - lin.predict(A)
bins = pd.cut(age[idx_tr], [0, 3, 7, 14, 28, 56, 100, 365], labels=["1~3", "4~7", "8~14", "15~28", "29~56", "57~100", "101~365"])
print(pd.DataFrame({"평균 잔차(MPa)": res, "건수": 1}).groupby(bins, observed=True).agg({"평균 잔차(MPa)": "mean", "건수": "sum"}).round(2).T)
print("→ 초기 재령은 과대 예측(−), 28~100일은 과소 예측(+), 1년은 다시 과대 예측. 직선이 곡선을 못 따라간다")

# ---------------------------------------------------------------
# 7) 좋은 특징: 물-결합재비와 log(재령)을 더하면?
def with_features(cols, name):
    Z = np.column_stack(cols)
    s = StandardScaler().fit(Z[idx_tr])
    m = LinearRegression().fit(s.transform(Z[idx_tr]), y_tr)
    report(name, m.predict(s.transform(Z[idx_te])), m.predict(s.transform(Z[idx_tr])))
    return m
with_features([X, wb], "8개 + 물-결합재비")
with_features([X, np.log(age)], "8개 + log(재령)")
m10 = with_features([X, wb, np.log(age)], "8개 + 물-결합재비 + log(재령)")
with_features([wb, np.log(age)], "물-결합재비 · log(재령) 둘만")
print("10개 모델 가중치:", {k: round(float(v), 2) for k, v in zip(FEATURES + ["wb", "log_age"], m10.coef_)})

# ---------------------------------------------------------------
# 8) 표준화는 점수를 바꾸나? (규제 없는 선형 회귀는 단위를 바꿔도 예측이 같다)
raw = LinearRegression().fit(X[idx_tr], y_tr)
report("특징 8개, 표준화 안 함", raw.predict(X[idx_te]))
print("날것 단위 가중치(MPa / 1kg 또는 1일):", pd.Series(raw.coef_, index=FEATURES).round(3).to_dict())

# ---------------------------------------------------------------
# 9) 판정: 시험 R²가 합격선(매뉴얼 씨 경험식)과 기준선을 모두 넘으면 합격
line = r2_score(y_te, manual[idx_te])
base = r2_score(y_te, dummy.predict(B))
score = r2_score(y_te, lin.predict(B))
print(f"기준선 R² {base:.3f} · 매뉴얼 씨 R² {line:.3f} · 선형 회귀 R² {score:.3f}")
print("판정:", "합격" if score > line and score > base else "보류")
