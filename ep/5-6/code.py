# ===============================================================
# 모델 채용 위원회 5-6 · 결정 트리 면접 (분류 + 회귀)
#  - 라이브러리: numpy, pandas, scikit-learn (Colab 기본) + ucimlrepo (자동 설치)
#  - 하는 일: 콘크리트 배합 1,030건으로 "35MPa 합격인가"(분류)와 "강도가 몇 MPa인가"(회귀)를
#             예/아니요 질문의 나무로 배우고, 깊이·가지치기를 바꿔 가며 두 합격선과 비교한다
# ===============================================================
import subprocess, sys
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text
from sklearn.dummy import DummyClassifier, DummyRegressor
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
grade = np.digitize(y, [25, 45])               # 0 저강도 · 1 보통 · 2 고강도
print(f"배합 {len(df)}건 · 합격 비율 {passed.mean():.3f}")

# ---------------------------------------------------------------
# 2) 시험지 봉인: 위원회 공통 규정 (25%를 시험지로, 강도 등급 비율 유지, 난수 고정)
#    결정 트리는 표준화가 필요 없다. 문턱 질문은 단위를 바꿔도 같은 배합을 같은 쪽으로 보낸다
idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=0.25, stratify=grade, random_state=42)
A, B = X[idx_tr], X[idx_te]
p_tr, p_te, y_tr, y_te, g_tr, g_te = passed[idx_tr], passed[idx_te], y[idx_tr], y[idx_te], grade[idx_tr], grade[idx_te]
print(f"훈련 {len(A)}건 · 시험 {len(B)}건")

# ---------------------------------------------------------------
# 3) 두 합격선: 매뉴얼 씨 (5-0 작업 표준서 · 5-1 경험식)
wb = (df["water"] / (df["cement"] + df["slag"] + df["fly_ash"])).to_numpy()
age = df["age"].to_numpy()
rule = ((wb < 0.5) & (age >= 28)).astype(int)            # "물-결합재비 0.5 미만이고 28일 이상이면 합격"
LINE_ACC = (rule[idx_te] == p_te).mean()
manual = 96.5 / 7 ** wb * age / (4 + 0.85 * age)          # 에이브럼스 식 × ACI 209R 재령 보정
LINE_R2 = r2_score(y_te, manual[idx_te])
base_acc = DummyClassifier(strategy="most_frequent").fit(A, p_tr).score(B, p_te)
base_reg = DummyRegressor().fit(A, y_tr)
print(f"분류 합격선(매뉴얼 씨) {LINE_ACC:.3f} · 기준선(늘 불합격) {base_acc:.3f}")
print(f"회귀 합격선(경험식) R² {LINE_R2:.3f} · MAE {mean_absolute_error(y_te, manual[idx_te]):.2f} · 기준선 MAE {mean_absolute_error(y_te, base_reg.predict(B)):.2f}")

# ---------------------------------------------------------------
# 4) 질문 하나 = 깊이 1 나무. 지니 불순도를 직접 계산해 가장 좋은 문턱을 찾는다
def gini(lab):
    if len(lab) == 0:
        return 0.0
    p = np.bincount(lab, minlength=2) / len(lab)
    return 1 - (p ** 2).sum()

print(f"뿌리의 지니 불순도 {gini(p_tr):.3f}  (반반이면 0.5, 한쪽뿐이면 0)")
best = (np.inf, None, None)
for j, f in enumerate(FEATURES):
    v = np.unique(A[:, j])
    for t in (v[:-1] + v[1:]) / 2:                       # 문턱 후보 = 이웃한 두 값의 한가운데
        L, R = p_tr[A[:, j] <= t], p_tr[A[:, j] > t]
        w = (len(L) * gini(L) + len(R) * gini(R)) / len(p_tr)
        if w < best[0]:
            best = (w, f, t)
print(f"직접 찾은 첫 질문: {best[1]} ≤ {best[2]:.1f} → 자식 지니(가중 평균) {best[0]:.3f}")
stump = DecisionTreeClassifier(max_depth=1, random_state=0).fit(A, p_tr)
print(f"scikit-learn 깊이 1: {FEATURES[stump.tree_.feature[0]]} ≤ {stump.tree_.threshold[0]:.1f} · 시험 정확도 {stump.score(B, p_te):.3f}")

# ---------------------------------------------------------------
# 5) 분류 트리 · 기본 설정(끝까지 키움). 같은 점수의 질문이 여럿이면 random_state가 고른다
clf = DecisionTreeClassifier(random_state=0).fit(A, p_tr)
print(f"[분류] 깊이 {clf.get_depth()} · 잎 {clf.get_n_leaves()}개 · 훈련 {clf.score(A, p_tr):.3f} · 시험 {clf.score(B, p_te):.3f}")
cm = confusion_matrix(p_te, clf.predict(B))
print(pd.DataFrame(cm, index=["실제 불합격", "실제 합격"], columns=["예측 불합격", "예측 합격"]))
ss = StandardScaler().fit(A)
same = DecisionTreeClassifier(random_state=0).fit(ss.transform(A), p_tr).score(ss.transform(B), p_te)
print(f"표준화한 뒤 같은 나무: 시험 {same:.3f}  (같아야 한다)")
print("random_state만 바꾸면:", [round(DecisionTreeClassifier(random_state=s).fit(A, p_tr).score(B, p_te), 3) for s in range(5)])

# ---------------------------------------------------------------
# 6) 회귀 트리 · 잎의 답 = 그 잎에 모인 훈련 배합의 평균 강도. 질문은 MSE(오차 제곱 평균)가 가장 줄도록 고른다
reg = DecisionTreeRegressor(random_state=0).fit(A, y_tr)
pr = reg.predict(B)
print(f"[회귀] 깊이 {reg.get_depth()} · 잎 {reg.get_n_leaves()}개 · 훈련 R² {reg.score(A, y_tr):.3f} · 시험 R² {r2_score(y_te, pr):.3f} · MAE {mean_absolute_error(y_te, pr):.2f}")
r2d = DecisionTreeRegressor(max_depth=2, random_state=0).fit(A, y_tr)
print(export_text(r2d, feature_names=FEATURES, decimals=1))
leaf = r2d.apply(A)
print("잎 네 개의 훈련 배합 평균 강도:", [round(float(y_tr[leaf == k].mean()), 2) for k in np.unique(leaf)])

# ---------------------------------------------------------------
# 7) 깊이 손잡이: 얕으면 과소적합, 끝까지 키우면 훈련을 외운다 (과대적합)
print(" 깊이   분류 훈련  분류 시험 | 회귀 훈련R²  회귀 시험R²")
for d in (1, 2, 3, 4, 5, 6, 8, None):
    c = DecisionTreeClassifier(max_depth=d, random_state=0).fit(A, p_tr)
    r = DecisionTreeRegressor(max_depth=d, random_state=0).fit(A, y_tr)
    print(f" {str(d):>4}    {c.score(A, p_tr):.3f}     {c.score(B, p_te):.3f}   |   {r.score(A, y_tr):.3f}       {r.score(B, y_te):.3f}")

# ---------------------------------------------------------------
# 8) 가지치기: min_samples_leaf(잎의 최소 배합 수)와 ccp_alpha(비용 복잡도)
for m in (1, 5, 20):
    c = DecisionTreeClassifier(min_samples_leaf=m, random_state=0).fit(A, p_tr)
    r = DecisionTreeRegressor(min_samples_leaf=m, random_state=0).fit(A, y_tr)
    print(f"  min_samples_leaf={m:<2} 분류 시험 {c.score(B, p_te):.3f} (잎 {c.get_n_leaves()}) · 회귀 시험 R² {r.score(B, y_te):.3f} (잎 {r.get_n_leaves()})")
for a in (0.002, 0.005, 0.01):
    c = DecisionTreeClassifier(ccp_alpha=a, random_state=0).fit(A, p_tr)
    print(f"  ccp_alpha={a:<5} 분류 시험 {c.score(B, p_te):.3f} · 잎 {c.get_n_leaves()}개")
#    근무 조건은 시험지가 아니라 훈련 데이터 안의 교차 검증으로 고른다
gs = GridSearchCV(DecisionTreeRegressor(random_state=0), {"max_depth": [4, 6, 8, 10, None], "min_samples_leaf": [1, 2, 5]}, cv=5).fit(A, y_tr)
print(f"교차 검증으로 고른 회귀 조건 {gs.best_params_} · 검증 R² {gs.best_score_:.3f} · 시험 R² {gs.score(B, y_te):.3f}")

# ---------------------------------------------------------------
# 9) 나무를 작업 표준서로: 깊이 3 나무의 규칙과 특징 중요도
c3 = DecisionTreeClassifier(max_depth=3, random_state=0).fit(A, p_tr)
print(export_text(c3, feature_names=FEATURES, decimals=1, class_names=["불합격", "합격"]))
print(f"깊이 3 규칙의 시험 정확도 {c3.score(B, p_te):.3f}  (매뉴얼 씨 {LINE_ACC:.3f})")
print("특징 중요도(분류, 끝까지):", " · ".join(f"{k} {v:.3f}" for k, v in pd.Series(clf.feature_importances_, index=FEATURES).sort_values(ascending=False).items()))
print("특징 중요도(회귀, 끝까지):", " · ".join(f"{k} {v:.3f}" for k, v in pd.Series(reg.feature_importances_, index=FEATURES).sort_values(ascending=False).items()))

# ---------------------------------------------------------------
# 10) 과제 C · 강도 3등급 (같은 나무, 잎마다 다수결)
for d in (5, None):
    c = DecisionTreeClassifier(max_depth=d, random_state=0).fit(A, g_tr)
    print(f"[3등급] 깊이 {d}: 시험 {c.score(B, g_te):.3f}  (기준선 늘 보통 {np.mean(g_te == 1):.3f})")

# ---------------------------------------------------------------
# 11) 판정: 기본 설정(끝까지 키운 나무)이 분류·회귀 두 합격선을 모두 넘으면 합격
acc, r2 = clf.score(B, p_te), r2_score(y_te, pr)
print(f"분류 {acc:.3f} (합격선 {LINE_ACC:.3f}) · 회귀 R² {r2:.3f} (합격선 {LINE_R2:.3f})")
print("판정:", "합격" if acc > LINE_ACC and acc > base_acc and r2 > LINE_R2 else "보류")
