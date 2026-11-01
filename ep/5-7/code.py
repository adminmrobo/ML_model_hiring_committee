# ===============================================================
# 모델 채용 위원회 5-7 · 랜덤 포레스트 면접 (+ Extra Trees)
#  - 라이브러리: numpy, pandas, scikit-learn (Colab 기본) + ucimlrepo (자동 설치)
#  - 하는 일: 콘크리트 배합 1,030건으로 "35MPa 합격인가"(분류)와 "강도가 몇 MPa인가"(회귀)를
#             나무 한 그루와 숲(부트스트랩 + 특징 무작위)으로 풀고, OOB 점수·특징 중요도·Extra Trees까지 본 뒤
#             두 합격선과 비교한다
# ===============================================================
import subprocess, sys, time
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.inspection import permutation_importance
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
#    나무는 "문턱보다 큰가 작은가"만 묻기 때문에 표준화가 필요 없다. 원래 단위(kg, 일) 그대로 넣는다
idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=0.25, stratify=grade, random_state=42)
A, B = X[idx_tr], X[idx_te]
p_tr, p_te, y_tr, y_te, g_tr, g_te = passed[idx_tr], passed[idx_te], y[idx_tr], y[idx_te], grade[idx_tr], grade[idx_te]
print(f"훈련 {len(A)}건 · 시험 {len(B)}건")

# ---------------------------------------------------------------
# 3) 두 합격선: 매뉴얼 씨(5-0)의 작업 표준서(분류)와 현장 경험식(회귀) · 배우지 않는 기준선
wb = (df["water"] / (df["cement"] + df["slag"] + df["fly_ash"])).to_numpy()
age = df["age"].to_numpy()
rule = ((wb < 0.5) & (age >= 28)).astype(int)
LINE_ACC = (rule[idx_te] == p_te).mean()
manual = 96.5 / 7 ** wb * age / (4 + 0.85 * age)        # 에이브럼스 식(물-결합재비) × ACI 209R 재령 보정
LINE_R2 = r2_score(y_te, manual[idx_te])
base_acc = DummyClassifier(strategy="most_frequent").fit(A, p_tr).score(B, p_te)
base_r2 = round(DummyRegressor().fit(A, y_tr).score(B, y_te), 6) + 0.0   # 늘 평균 → R²는 0 근처
print(f"분류 합격선 {LINE_ACC:.3f} (기준선 {base_acc:.3f}) · 회귀 합격선 R² {LINE_R2:.3f} (기준선 {base_r2:.3f})")

# ---------------------------------------------------------------
# 4) 나무 한 그루 vs 숲: 끝까지 키운 결정 트리는 훈련 데이터를 거의 외운다. 숲은 그런 나무를 여럿 모은다
#    RandomForestClassifier 기본값: 나무 100그루, 부트스트랩, 갈림마다 특징 √8 ≈ 2개만 후보로 본다
tree = DecisionTreeClassifier(random_state=0).fit(A, p_tr)
print(f"[결정 트리 1그루] 훈련 {tree.score(A, p_tr):.3f} · 시험 {tree.score(B, p_te):.3f} · 깊이 {tree.get_depth()} · 잎 {tree.get_n_leaves()}")
for n in (1, 10, 100, 300):
    rf = RandomForestClassifier(n_estimators=n, random_state=0).fit(A, p_tr)
    print(f"[숲 {n:>3}그루] 훈련 {rf.score(A, p_tr):.3f} · 시험 {rf.score(B, p_te):.3f}")

# ---------------------------------------------------------------
# 5) 흔들림: 난수(random_state)만 바꿔 20번 다시 뽑는다. 나무가 많을수록 점수가 덜 흔들린다
for n in (1, 10, 100):
    s = np.array([RandomForestClassifier(n_estimators=n, random_state=r).fit(A, p_tr).score(B, p_te) for r in range(20)])
    print(f"  {n:>3}그루 × 20번: 최저 {s.min():.3f} · 평균 {s.mean():.3f} · 최고 {s.max():.3f} · 표준편차 {s.std():.3f}")

# ---------------------------------------------------------------
# 6) 부트스트랩과 OOB: 772건에서 772번 "복원 추출"하면 약 63%만 뽑히고 37%는 남는다(가방 밖, out-of-bag)
#    나무마다 자기가 못 본 배합으로 채점하면 시험지를 열지 않고도 실력을 가늠할 수 있다
rng = np.random.default_rng(0)
pick = rng.integers(0, len(A), len(A))
print(f"부트스트랩 한 번: 서로 다른 배합 {len(np.unique(pick))}건 ({len(np.unique(pick)) / len(A):.3f}) · 이론값 1 − 1/e = {1 - np.exp(-1):.3f}")
rf = RandomForestClassifier(n_estimators=100, random_state=0, oob_score=True).fit(A, p_tr)
print(f"OOB 정확도 {rf.oob_score_:.3f} (시험지 없이) · 시험 정확도 {rf.score(B, p_te):.3f}")
print(pd.DataFrame(confusion_matrix(p_te, rf.predict(B)), index=["실제 불합격", "실제 합격"], columns=["예측 불합격", "예측 합격"]))

# ---------------------------------------------------------------
# 7) 특징 무작위가 하는 일: 나무끼리 덜 닮게 만든다 (닮은 나무끼리는 평균을 내도 실수가 줄지 않는다)
def tree_corr(forest):
    P = np.array([t.predict_proba(B)[:, 1] for t in forest.estimators_])
    c = np.corrcoef(P)
    return c[np.triu_indices_from(c, 1)].mean()
for mf, name in [(None, "특징 8개 모두 (배깅)"), ("sqrt", "특징 √8 ≈ 2개 (기본값)"), (1, "특징 1개")]:
    m = RandomForestClassifier(n_estimators=100, max_features=mf, random_state=0).fit(A, p_tr)
    print(f"  {name:<16} 시험 {m.score(B, p_te):.3f} · 나무끼리 상관 {tree_corr(m):.2f}")

# ---------------------------------------------------------------
# 8) 회귀: 같은 숲이 평균을 내면 강도(MPa) 예측
#    RandomForestRegressor의 기본값은 max_features=1.0 (갈림마다 특징 8개를 모두 본다 = 배깅)
for n in (1, 10, 100, 300):
    reg = RandomForestRegressor(n_estimators=n, random_state=0).fit(A, y_tr)
    pr = reg.predict(B)
    print(f"[회귀 숲 {n:>3}그루] 시험 R² {r2_score(y_te, pr):.3f} · MAE {mean_absolute_error(y_te, pr):.2f}MPa")
reg = RandomForestRegressor(n_estimators=100, random_state=0, oob_score=True).fit(A, y_tr)
print(f"OOB R² {reg.oob_score_:.3f} · 시험 R² {reg.score(B, y_te):.3f} · 훈련 R² {reg.score(A, y_tr):.3f}")
print(f"[결정 트리 1그루 회귀] 시험 R² {DecisionTreeRegressor(random_state=0).fit(A, y_tr).score(B, y_te):.3f}")

# ---------------------------------------------------------------
# 9) 3등급 분류: 저강도 · 보통 · 고강도
rf3 = RandomForestClassifier(n_estimators=100, random_state=0).fit(A, g_tr)
print(f"[3등급] 시험 {rf3.score(B, g_te):.3f} · 기준선(늘 보통) {np.mean(g_te == 1):.3f}")

# ---------------------------------------------------------------
# 10) 특징 중요도: 그 특징으로 자를 때 불순도(지니·MSE)가 얼마나 줄었나를 모든 나무에서 더한 값(합 1)
#     매뉴얼 씨가 쓰던 두 숫자는 물-결합재비와 재령이다. 숲은 무엇을 많이 물어봤을까
imp = pd.DataFrame({"회귀 숲": reg.feature_importances_, "분류 숲": rf.feature_importances_}, index=FEATURES)
print(imp.sort_values("회귀 숲", ascending=False).round(3))
#     불순도 기반 중요도는 값이 많은 연속 특징을 부풀리는 버릇이 있다. 시험지에서 특징을 섞어 보는 방법으로 확인한다
pi = permutation_importance(reg, B, y_te, n_repeats=10, random_state=0)
print("섞으면 떨어지는 R²:", pd.Series(pi.importances_mean, index=FEATURES).sort_values(ascending=False).round(3).to_dict())

# ---------------------------------------------------------------
# 11) Extra Trees: 문턱까지 무작위로 뽑고, 부트스트랩 없이 훈련 데이터 전체로 키운다
etc = ExtraTreesClassifier(n_estimators=100, random_state=0).fit(A, p_tr)
etr = ExtraTreesRegressor(n_estimators=100, random_state=0).fit(A, y_tr)
pe = etr.predict(B)
print(f"[Extra Trees] 분류 시험 {etc.score(B, p_te):.3f} · 회귀 시험 R² {r2_score(y_te, pe):.3f} · MAE {mean_absolute_error(y_te, pe):.2f}MPa")

# ---------------------------------------------------------------
# 12) 예산: 나무 100그루 = 외울 마디 수와 학습·예측 시간 (시간은 컴퓨터마다 다르다)
for name, m in [("분류 숲", RandomForestClassifier(n_estimators=100, random_state=0)), ("회귀 숲", RandomForestRegressor(n_estimators=100, random_state=0)),
                ("분류 숲 n_jobs=-1", RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1))]:
    t0 = time.perf_counter(); m.fit(A, p_tr if "분류" in name else y_tr); t_fit = time.perf_counter() - t0
    t0 = time.perf_counter(); m.predict(B); t_pred = time.perf_counter() - t0
    nodes = sum(t.tree_.node_count for t in m.estimators_)
    print(f"  {name:<16} 마디 {nodes:,}개 · 평균 깊이 {np.mean([t.get_depth() for t in m.estimators_]):.1f} · fit {t_fit * 1000:.0f}ms · predict {t_pred * 1000:.0f}ms")

# ---------------------------------------------------------------
# 13) 판정: 분류는 매뉴얼 씨(76.7%)와 기준선, 회귀는 매뉴얼 경험식(R² 0.735)과 기준선을 모두 넘으면 합격
acc, r2 = rf.score(B, p_te), reg.score(B, y_te)
print(f"분류: 랜덤 포레스트 {acc:.3f} · 매뉴얼 씨 {LINE_ACC:.3f} · 기준선 {base_acc:.3f}")
print(f"회귀: 랜덤 포레스트 R² {r2:.3f} · 매뉴얼 경험식 {LINE_R2:.3f} · 기준선 {base_r2:.3f}")
ok = acc > LINE_ACC and acc > base_acc and r2 > LINE_R2 and r2 > base_r2
print("판정:", "합격 (분류·회귀 모두)" if ok else "보류")
