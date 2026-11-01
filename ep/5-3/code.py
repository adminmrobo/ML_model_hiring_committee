# ===============================================================
# 모델 채용 위원회 5-3 · 다항 회귀와 규제 면접 (Ridge · Lasso · Elastic Net)
#  - 라이브러리: numpy, pandas, scikit-learn (Colab 기본) + ucimlrepo (자동 설치)
#  - 하는 일: 콘크리트 배합 1,030건으로 압축 강도(MPa)를 맞힌다.
#             특징을 2·3·4차로 늘려 곡선을 그리고, 과대적합을 규제(α)로 잡은 뒤
#             매뉴얼 씨의 현장 경험식(합격선 R² 0.735)과 비교한다
# ===============================================================
import subprocess, sys, warnings
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split, PredefinedSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.dummy import DummyRegressor
from sklearn.metrics import r2_score, mean_absolute_error
try:
    from ucimlrepo import fetch_ucirepo
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ucimlrepo"], check=True)
    from ucimlrepo import fetch_ucirepo

np.set_printoptions(precision=3, suppress=True)
print("scikit-learn", sklearn.__version__)

# ---------------------------------------------------------------
# 1) 실기 과제지 읽기: 콘크리트 배합 1,030건 (UCI id=165, CC BY 4.0)
#    [내 데이터로 바꾸려면] 이 칸의 데이터 읽기와 FEATURES만 바꾼다
FEATURES = ["cement", "slag", "fly_ash", "water", "superplasticizer", "coarse_agg", "fine_agg", "age"]
uci = fetch_ucirepo(id=165)
df = pd.concat([uci.data.features, uci.data.targets], axis=1)
df.columns = FEATURES + ["strength"]          # 열 순서대로 짧은 이름을 붙인다
assert df.shape == (1030, 9)
X = df[FEATURES].to_numpy()
y = df["strength"].to_numpy()                  # 타겟: 압축 강도(MPa) 그 자체 (회귀)
grade = np.digitize(y, [25, 45])               # 나누기 규정에만 쓰는 강도 3등급
print(f"배합 {len(df)}건 · 강도 평균 {y.mean():.1f}MPa · 범위 {y.min():.1f}~{y.max():.1f}MPa")

# ---------------------------------------------------------------
# 2) 시험지 봉인: 위원회 공통 규정 (25%를 시험지로, 강도 등급 비율 유지, 난수 고정)
#    행 순서를 원래 순서로 정렬해 둔다(브라우저와 같은 교차 검증 칸을 쓰려고)
idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=0.25, stratify=grade, random_state=42)
idx_tr, idx_te = np.sort(idx_tr), np.sort(idx_te)
X_tr, X_te, y_tr, y_te = X[idx_tr], X[idx_te], y[idx_tr], y[idx_te]
print(f"훈련 {len(X_tr)}건 · 시험 {len(X_te)}건")

def report(name, model, Xa=X_tr, Xb=X_te):
    """훈련 R² · 시험 R² · 시험 MAE를 한 줄로"""
    tr, te = r2_score(y_tr, model.predict(Xa)), r2_score(y_te, model.predict(Xb))
    mae = mean_absolute_error(y_te, model.predict(Xb))
    print(f"  {name:<26} 훈련 R² {tr:6.3f} · 시험 R² {te:7.3f} · 시험 MAE {mae:5.2f}MPa")
    return te

# ---------------------------------------------------------------
# 3) 합격선: 매뉴얼 씨의 현장 경험식 (학습 없음, 사람의 지식)
#    28일 강도 f28 = 96.5 / 7^(물-결합재비) MPa (에이브럼스 식, A=14,000psi≈96.5MPa, B=7)
#    재령 보정 f(t) = f28 · t / (4 + 0.85t)  (ACI 209R, 습윤 양생 보통 시멘트)
def manual(Xm):
    wb = Xm[:, 3] / (Xm[:, 0] + Xm[:, 1] + Xm[:, 2])
    t = Xm[:, 7]
    return 96.5 / 7 ** wb * t / (4 + 0.85 * t)
PASS_R2 = r2_score(y_te, manual(X_te))
dummy = DummyRegressor().fit(X_tr, y_tr)        # 기준선: 늘 훈련 평균
print(f"매뉴얼 경험식  시험 R² {PASS_R2:.3f} · MAE {mean_absolute_error(y_te, manual(X_te)):.2f}MPa  ← 회귀팀 합격선")
print(f"기준선(평균)   시험 R² {r2_score(y_te, dummy.predict(X_te)):.3f} · MAE {mean_absolute_error(y_te, dummy.predict(X_te)):.2f}MPa")

# ---------------------------------------------------------------
# 4) 다항 특징: 특징 8개를 거듭제곱·곱으로 늘린다 (상수 열 1은 빼고 센다)
for d in (1, 2, 3, 4):
    pf = PolynomialFeatures(d, include_bias=False).fit(X_tr)
    print(f"  {d}차 → 특징 {pf.n_output_features_}개")
names = PolynomialFeatures(3, include_bias=False).fit(X_tr).get_feature_names_out(FEATURES)
print("3차 특징 예:", list(names[[0, 7, 8, 43, 44, 163]]))

# ---------------------------------------------------------------
# 5) 차수 올리기: 표준화 → 다항 특징 → 선형 회귀 (규제 없음)
#    2차·3차는 곡선을 잡고, 4차는 훈련 데이터를 외우다 시험지에서 무너진다
print("[다항 회귀, 규제 없음]")
poly = {}
for d in (1, 2, 3, 4):
    poly[d] = make_pipeline(StandardScaler(), PolynomialFeatures(d, include_bias=False), LinearRegression()).fit(X_tr, y_tr)
    report(f"{d}차", poly[d])
# 4차는 풀이가 불안정하다: 같은 모델을 계산 순서만 바꿔 풀어도 시험 점수가 크게 달라진다
p4 = make_pipeline(StandardScaler(), PolynomialFeatures(4, include_bias=False), StandardScaler(), LinearRegression()).fit(X_tr, y_tr)
report("4차 (다항 뒤 한 번 더 표준화)", p4)
Z4 = p4[:3].transform(X_tr)
s = np.linalg.svd(Z4 - Z4.mean(0), compute_uv=False)
print(f"  4차 특징 행렬의 조건수 {s[0] / s[-1]:.1e} (클수록 풀이가 흔들린다)")

# ---------------------------------------------------------------
# 6) 규제 붙이기: 다항 특징을 한 번 더 표준화한 뒤 Ridge · Lasso · Elastic Net
#    Ridge  : 오차² 합 + α·Σw²          → 가중치를 고루 줄인다
#    Lasso  : 오차² 평균/2 + α·Σ|w|      → 쓸모없는 가중치를 0으로 지운다
#    ElasticNet: 둘을 l1_ratio로 섞는다
def reg(d, model):
    return make_pipeline(StandardScaler(), PolynomialFeatures(d, include_bias=False), StandardScaler(), model).fit(X_tr, y_tr)
print("[3차 + Ridge]")
for a in (0.1, 1, 10, 100):
    report(f"α={a}", reg(3, Ridge(alpha=a)))
print("[4차 + Ridge] 폭주가 잡히나")
for a in (0.1, 1, 10, 100):
    report(f"α={a}", reg(4, Ridge(alpha=a)))
print("[3차 + Lasso] 0이 된 가중치를 센다 (164개 중)")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")          # 아주 작은 α에서 수렴 경고가 날 수 있다
    for a in (0.01, 0.1, 1):
        m = reg(3, Lasso(alpha=a, max_iter=100000))
        report(f"α={a} · 0 {int((m[-1].coef_ == 0).sum())}개", m)
lasso = reg(3, Lasso(alpha=0.1, max_iter=100000))
keep = np.argsort(-np.abs(lasso[-1].coef_))[:6]
print("  α=0.1 라쏘가 크게 남긴 특징:", [(str(names[k]), round(float(lasso[-1].coef_[k]), 2)) for k in keep])
print("[3차 + Elastic Net, l1_ratio=0.5]")
for a in (0.01, 0.1):
    m = reg(3, ElasticNet(alpha=a, l1_ratio=0.5, max_iter=100000))
    report(f"α={a} · 0 {int((m[-1].coef_ == 0).sum())}개", m)

# ---------------------------------------------------------------
# 7) α 고르기: 시험지가 아니라 훈련 데이터 안의 5겹 교차 검증으로
#    칸 나누기: 훈련 배합을 번갈아 0,1,2,3,4,0,1,… 칸에 넣는다 (브라우저와 같은 칸)
folds = PredefinedSplit(np.arange(len(idx_tr)) % 5)
grid = {"ridge__alpha": [0.01, 0.1, 1, 10, 100, 1000]}
best = {}
for d in (3, 4):
    pipe = make_pipeline(StandardScaler(), PolynomialFeatures(d, include_bias=False), StandardScaler(), Ridge())
    gs = GridSearchCV(pipe, grid, cv=folds, scoring="r2").fit(X_tr, y_tr)
    cvs = {a: round(float(v), 3) for a, v in zip(grid["ridge__alpha"], gs.cv_results_["mean_test_score"])}
    print(f"  {d}차 Ridge 교차 검증 R²: {cvs} → α={gs.best_params_['ridge__alpha']}")
    best[d] = report(f"{d}차 Ridge α={gs.best_params_['ridge__alpha']} (CV로 고름)", gs.best_estimator_)

# ---------------------------------------------------------------
# 8) 면접 판정: 시험 R²가 합격선(매뉴얼 경험식)과 기준선을 모두 넘으면 합격
score = r2_score(y_te, poly[3].predict(X_te))
base = r2_score(y_te, dummy.predict(X_te))
print(f"기준선 R² {base:.3f} · 매뉴얼 경험식 {PASS_R2:.3f} · 3차 다항 회귀 {score:.3f} · 4차 {r2_score(y_te, poly[4].predict(X_te)):.2f}")
print("판정(3차):", "합격" if score > PASS_R2 and score > base else "보류")
print("판정(4차 + Ridge, CV):", "합격" if best[4] > PASS_R2 else "보류")
