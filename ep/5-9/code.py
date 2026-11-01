# ===============================================================
# 모델 채용 위원회 5-9 · GBM(그레이디언트 부스팅) 면접
#  - 라이브러리: numpy, pandas, scikit-learn (Colab 기본) + ucimlrepo (자동 설치)
#  - 하는 일: 콘크리트 배합 1,030건으로 강도(MPa, 회귀)와 35MPa 합격(분류)을
#             "얕은 나무가 앞 나무의 잔차를 차례로 배운다"로 풀고, 학습률·나무 수·조기 종료를 확인한 뒤
#             합격선과 비교한다. 끝에 5-5~5-9 지원자 다섯 명의 중간 정산표를 낸다
#  - XGBoost·LightGBM·CatBoost는 같은 원리의 다른 구현이다. 이 코드는 scikit-learn만 쓴다(11번 칸 주석 참고)
# ===============================================================
import subprocess, sys, time
import numpy as np
import pandas as pd
import sklearn
from scipy.special import expit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (GradientBoostingRegressor, GradientBoostingClassifier,
                              HistGradientBoostingRegressor, HistGradientBoostingClassifier,
                              RandomForestRegressor, RandomForestClassifier)
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR
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
grade = np.digitize(y, [25, 45])               # 0 저강도 · 1 보통 · 2 고강도 (나누기 기준, 3등급 과제)
print(f"배합 {len(df)}건 · 강도 평균 {y.mean():.1f}MPa · 합격 비율 {passed.mean():.3f}")

# ---------------------------------------------------------------
# 2) 시험지 봉인: 위원회 공통 규정 (25%를 시험지로, 강도 등급 비율 유지, 난수 고정)
#    나무는 "문턱보다 큰가 작은가"만 묻기 때문에 표준화가 필요 없다. 그래서 X를 그대로 쓴다
idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=0.25, stratify=grade, random_state=42)
X_tr, X_te, y_tr, y_te = X[idx_tr], X[idx_te], y[idx_tr], y[idx_te]
p_tr, p_te, g_tr, g_te = passed[idx_tr], passed[idx_te], grade[idx_tr], grade[idx_te]
print(f"훈련 {len(X_tr)}건 · 시험 {len(X_te)}건")

# ---------------------------------------------------------------
# 3) 합격선: 매뉴얼 씨 (학습 없음, 사람의 지식)
#    분류: 작업 표준서 "물-결합재비 0.5 미만이고 28일 이상이면 합격"
#    회귀: 28일 강도 = 96.5 / 7^(물-결합재비) (에이브럼스 식) × 재령 보정 t / (4 + 0.85t) (ACI 209R)
wb = (df["water"] / (df["cement"] + df["slag"] + df["fly_ash"])).to_numpy()
age = df["age"].to_numpy()
rule = ((wb < 0.5) & (age >= 28)).astype(int)
LINE_ACC = (rule[idx_te] == p_te).mean()
manual = 96.5 / 7 ** wb * age / (4 + 0.85 * age)
LINE_R2 = r2_score(y_te, manual[idx_te])
dummy = DummyRegressor().fit(X_tr, y_tr)       # 기준선: 늘 훈련 평균을 답한다
print(f"분류 합격선 {LINE_ACC:.3f} · 기준선(늘 불합격) {(p_te == 0).mean():.3f}")
print(f"회귀 합격선 R² {LINE_R2:.3f} (MAE {mean_absolute_error(y_te, manual[idx_te]):.2f}) · "
      f"기준선 R² {dummy.score(X_te, y_te):.3f} (MAE {mean_absolute_error(y_te, dummy.predict(X_te)):.2f})")

# ---------------------------------------------------------------
# 4) 부스팅을 직접 짜 보기: 시작은 평균 → 잔차(아직 틀린 만큼)에 얕은 나무를 맞춘다 → 학습률만큼 덧댄다 → 반복
def my_boost(Xa, ya, Xb, n_trees, lr=0.1, depth=3):
    F_a = np.full(len(Xa), ya.mean())          # 시작값 F0 = 훈련 평균
    F_b = np.full(len(Xb), ya.mean())
    for t in range(n_trees):
        residual = ya - F_a                    # 오답 노트: 실제 − 지금까지의 예측
        h = DecisionTreeRegressor(max_depth=depth, random_state=0).fit(Xa, residual)
        F_a += lr * h.predict(Xa)              # 잔차를 배운 나무를 조금(lr)만 덧댄다
        F_b += lr * h.predict(Xb)
    return F_b

gbr = GradientBoostingRegressor(n_estimators=300, learning_rate=0.1, max_depth=3, random_state=0).fit(X_tr, y_tr)
for n in (10, 100, 300):
    mine = my_boost(X_tr, y_tr, X_te, n)
    lib = list(gbr.staged_predict(X_te))[n - 1]
    print(f"  나무 {n:>3}그루: 직접 짠 부스팅 R² {r2_score(y_te, mine):.3f} · scikit-learn R² {r2_score(y_te, lib):.3f} · "
          f"예측 차이 최대 {np.abs(mine - lib).max():.4f}MPa")
#    차이가 조금 나는 그루 수가 있다. 점수가 똑같은 분할이 여럿이면 특징을 살펴보는 순서(난수)에 따라 다른 쪽을 고르기 때문이다

# ---------------------------------------------------------------
# 5) 덧대기 재생: 나무가 늘면 시험 R²가 오른다 (이 칸은 성과 평가용으로 시험지를 연다)
print("  나무  시험R²  MAE    훈련R²")
stage_te = list(gbr.staged_predict(X_te))
stage_tr = list(gbr.staged_predict(X_tr))
for n in (1, 3, 10, 30, 50, 100, 200, 300):
    print(f"  {n:>4}  {r2_score(y_te, stage_te[n - 1]):.3f}   {mean_absolute_error(y_te, stage_te[n - 1]):.2f}   {r2_score(y_tr, stage_tr[n - 1]):.3f}")

# ---------------------------------------------------------------
# 6) 수습 기간 = 학습률 × 나무 수 고르기. 시험지는 보지 않는다
#    훈련 772건을 행 번호 순으로 세워 다섯 줄마다 한 줄(155건)을 검증 몫으로 떼고, 나머지 617건으로 학습한다
tr_sorted = np.sort(idx_tr)
va, fi = tr_sorted[::5], np.setdiff1d(tr_sorted, tr_sorted[::5])
print(f"  검증 몫 {len(va)}건 · 학습 몫 {len(fi)}건")
print("  학습률  최고 검증R²(나무 수)  최고-0.005 안에 처음 든 나무 수  300그루 검증R²  300그루 훈련R²")
for lr in (1.0, 0.3, 0.1, 0.03):
    m = GradientBoostingRegressor(n_estimators=600, learning_rate=lr, random_state=0).fit(X[fi], y[fi])
    s = np.array([r2_score(y[va], q) for q in m.staged_predict(X[va])])
    first = int(np.argmax(s >= s.max() - 0.005)) + 1
    tr300 = r2_score(y[fi], list(m.staged_predict(X[fi]))[299])
    print(f"  {lr:<6}  {s.max():.3f} ({s.argmax() + 1:>3})              {first:>3}                          {s[299]:.3f}          {tr300:.3f}")

# ---------------------------------------------------------------
# 7) 조기 종료를 scikit-learn에 맡기면: 훈련 데이터의 10%를 검증 몫으로 떼어, 10그루 동안 나아지지 않으면 멈춘다
es = GradientBoostingRegressor(n_estimators=1000, n_iter_no_change=10, validation_fraction=0.1, random_state=42).fit(X_tr, y_tr)
print(f"  조기 종료: {es.n_estimators_}그루에서 멈춤 · 시험 R² {es.score(X_te, y_te):.3f}")

# ---------------------------------------------------------------
# 8) 나무 깊이: 부스팅은 "얕은" 나무를 쓴다. 끝까지 키운 나무는 첫 그루가 훈련 데이터를 외워 다음 나무가 배울 잔차가 사라진다
for d in (1, 2, 3, 5, None):
    m = GradientBoostingRegressor(n_estimators=300, max_depth=d, random_state=0).fit(X_tr, y_tr)
    print(f"  깊이 {str(d):>4}: 훈련 R² {m.score(X_tr, y_tr):.3f} · 시험 R² {m.score(X_te, y_te):.3f}")

# ---------------------------------------------------------------
# 9) 분류도 속은 회귀 나무: 로그 오즈(log(p/(1−p)))에 나무를 덧대고, 시그모이드로 확률을 낸다
gbc = GradientBoostingClassifier(n_estimators=300, learning_rate=0.1, max_depth=3, random_state=0).fit(X_tr, p_tr)
print("  안쪽 나무:", type(gbc.estimators_[0, 0]).__name__)       # DecisionTreeRegressor
f0 = np.log(p_tr.mean() / (1 - p_tr.mean()))
print(f"  시작 로그 오즈 F0 = log({p_tr.mean():.3f} / {1 - p_tr.mean():.3f}) = {f0:.3f}")
z = gbc.decision_function(X_te[:3])
print("  로그 오즈:", z, "→ 시그모이드:", expit(z), "= predict_proba:", gbc.predict_proba(X_te[:3])[:, 1])
stage_acc = [(q.ravel() > 0).astype(int) for q in gbc.staged_decision_function(X_te)]   # 로그 오즈 > 0 ⇔ 확률 > 0.5
for n in (10, 50, 100, 300):
    print(f"  [35MPa 분류] 나무 {n:>3}그루 시험 정확도 {(stage_acc[n - 1] == p_te).mean():.3f}")
print(f"  훈련 정확도 {gbc.score(X_tr, p_tr):.3f} · 시험 정확도 {gbc.score(X_te, p_te):.3f}")
print(confusion_matrix(p_te, gbc.predict(X_te)))
gbc3 = GradientBoostingClassifier(n_estimators=100, random_state=0).fit(X_tr, g_tr)
print(f"  [3등급 분류] 단계마다 나무 3그루(등급마다 1그루) × 100단계 · 시험 정확도 {gbc3.score(X_te, g_te):.3f} (기준선 {(g_te == 1).mean():.3f})")

# ---------------------------------------------------------------
# 10) 특징 중요도: 분할이 줄인 제곱 오차를 특징별로 더한 몫 (GBM 300그루 vs 랜덤 포레스트 100그루)
rf = RandomForestRegressor(n_estimators=100, random_state=0).fit(X_tr, y_tr)
imp = pd.DataFrame({"GBM": gbr.feature_importances_, "랜덤 포레스트": rf.feature_importances_}, index=FEATURES)
print(imp.sort_values("GBM", ascending=False).round(3).to_string())

# ---------------------------------------------------------------
# 11) 같은 원리의 빠른 구현: HistGradientBoosting (특징 값을 최대 255칸으로 묶어 분할을 빨리 찾는다)
#     XGBoost(xgboost.XGBRegressor), LightGBM(lightgbm.LGBMRegressor), CatBoost(catboost.CatBoostRegressor)도
#     "잔차(기울기)에 나무를 덧댄다"는 같은 원리다. Colab에서 쓰려면 import만 바꾸고 인자 이름을 문서로 확인한다
for name, model in [("GradientBoosting 300", GradientBoostingRegressor(n_estimators=300, random_state=0)),
                    ("HistGradientBoosting", HistGradientBoostingRegressor(random_state=0))]:
    t0 = time.perf_counter(); model.fit(X_tr, y_tr); t = time.perf_counter() - t0
    pr = model.predict(X_te)
    print(f"  [회귀] {name:<21} R² {r2_score(y_te, pr):.3f} · MAE {mean_absolute_error(y_te, pr):.2f} · 학습 {t * 1000:.0f}ms")
hgc = HistGradientBoostingClassifier(random_state=0).fit(X_tr, p_tr)
print(f"  [분류] HistGradientBoosting 시험 정확도 {hgc.score(X_te, p_te):.3f}")

# ---------------------------------------------------------------
# 12) 중간 정산: 지원자 다섯 명(5-5 ~ 5-9)을 같은 시험지로
sv_c = make_pipeline(StandardScaler(), SVC(C=10)).fit(X_tr, p_tr)
sv_r = make_pipeline(StandardScaler(), SVR(C=100)).fit(X_tr, y_tr)
rows = [("5-5 나이브 베이즈", GaussianNB().fit(X_tr, p_tr).score(X_te, p_te), None),
        ("5-6 결정 트리 (끝까지)", DecisionTreeClassifier(random_state=0).fit(X_tr, p_tr).score(X_te, p_te),
         DecisionTreeRegressor(random_state=0).fit(X_tr, y_tr).score(X_te, y_te)),
        ("5-7 랜덤 포레스트 100", RandomForestClassifier(random_state=0).fit(X_tr, p_tr).score(X_te, p_te), rf.score(X_te, y_te)),
        ("5-8 SVM (RBF, C=10 · SVR C=100)", sv_c.score(X_te, p_te), sv_r.score(X_te, y_te)),
        ("5-9 GBM 300", gbc.score(X_te, p_te), gbr.score(X_te, y_te))]
for name, acc, r2 in rows:
    print(f"  {name:<30} 분류 {acc:.3f}  회귀 R² {'—' if r2 is None else f'{r2:.3f}'}")

# ---------------------------------------------------------------
# 13) 판정: 회귀 R²가 합격선(매뉴얼 경험식)을, 분류 정확도가 합격선(매뉴얼 씨)과 기준선을 넘으면 합격
r2_final, acc_final = gbr.score(X_te, y_te), gbc.score(X_te, p_te)
print(f"회귀 R² {r2_final:.3f} (합격선 {LINE_R2:.3f}) · 분류 {acc_final:.3f} (합격선 {LINE_ACC:.3f})")
print("판정:", "합격" if r2_final > LINE_R2 and acc_final > LINE_ACC and acc_final > (p_te == 0).mean() else "보류")
