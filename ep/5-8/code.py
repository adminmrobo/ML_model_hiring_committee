# ===============================================================
# 모델 채용 위원회 5-8 · SVM 면접 (SVC 분류 + SVR 회귀)
#  - 라이브러리: numpy, pandas, scikit-learn (Colab 기본) + ucimlrepo (자동 설치)
#  - 하는 일: 콘크리트 배합 1,030건으로 "35MPa 합격인가"(분류)와 "강도 몇 MPa인가"(회귀)를
#             가장 넓은 길(마진)·서포트 벡터·소프트 마진 C·RBF 커널·ε 띠로 풀고 합격선과 비교한다
# ===============================================================
import subprocess, sys
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix, classification_report, r2_score, mean_absolute_error
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
strength = df["strength"].to_numpy()
y = (strength >= PASS_MPA).astype(int)                        # 분류: 합격 1, 불합격 0
grade = np.digitize(strength, [25, 45])                       # 0 저강도 · 1 보통 · 2 고강도
wb = df["water"].to_numpy() / (df["cement"] + df["slag"] + df["fly_ash"]).to_numpy()   # 물-결합재비
age = df["age"].to_numpy()

# ---------------------------------------------------------------
# 2) 시험지 봉인: 위원회 공통 규정 + 표준화(훈련 데이터로만 fit)
#    SVM은 거리(커널)와 ‖w‖로 일하므로 표준화가 사실상 필수다
idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=0.25, stratify=grade, random_state=42)
ss = StandardScaler().fit(X[idx_tr])
A, B = ss.transform(X[idx_tr]), ss.transform(X[idx_te])
y_tr, y_te, g_tr, g_te = y[idx_tr], y[idx_te], grade[idx_tr], grade[idx_te]
s_tr, s_te = strength[idx_tr], strength[idx_te]
print(f"훈련 {len(A)}건 · 시험 {len(B)}건")

# ---------------------------------------------------------------
# 3) 합격선: 매뉴얼 씨(5-0) — 분류는 작업 표준서 한 줄, 회귀는 경험식(에이브럼스 식 + ACI 209R 재령 보정)
rule = ((wb < 0.5) & (age >= 28)).astype(int)
manual_acc = (rule[idx_te] == y_te).mean()
manual_mpa = 96.5 / 7 ** wb * age / (4 + 0.85 * age)
LINE_R2 = r2_score(s_te, manual_mpa[idx_te])
dummy = DummyClassifier(strategy="most_frequent").fit(A, y_tr).score(B, y_te)
print(f"분류 합격선 {manual_acc:.3f} · 기준선(늘 다수 쪽) {dummy:.3f}")
print(f"회귀 합격선 R² {LINE_R2:.3f} (MAE {mean_absolute_error(s_te, manual_mpa[idx_te]):.2f}) · "
      f"기준선(늘 평균) MAE {mean_absolute_error(s_te, np.full(len(s_te), s_tr.mean())):.2f}")

# ---------------------------------------------------------------
# 4) 가장 넓은 길: 특징 2개(물-결합재비, log 재령)로 직선 SVM
#    마진(길의 폭) = 2 / ‖w‖, 서포트 벡터 = 길 위·길 안·반대편에 놓인 배합
Z = np.column_stack([wb, np.log10(age)])
s2 = StandardScaler().fit(Z[idx_tr])
Z_tr, Z_te = s2.transform(Z[idx_tr]), s2.transform(Z[idx_te])
for C in (0.01, 1, 100):
    m = SVC(kernel="linear", C=C).fit(Z_tr, y_tr)
    w = m.coef_[0]
    print(f"  직선 C={C:<5g} w={w.round(3)} b={m.intercept_[0]:.3f} · 길의 폭 {2 / np.linalg.norm(w):.3f} · "
          f"서포트 벡터 {m.n_support_.sum()}개 · 시험 {m.score(Z_te, y_te):.3f}")
m = SVC(kernel="linear", C=1).fit(Z_tr, y_tr)
sv_only = SVC(kernel="linear", C=1).fit(Z_tr[m.support_], y_tr[m.support_])   # 서포트 벡터만 남기고 다시 학습
print("  서포트 벡터만으로 다시 풀어도 같은 길인가?", np.allclose(m.coef_, sv_only.coef_, atol=1e-2), sv_only.coef_[0].round(3))

# ---------------------------------------------------------------
# 5) 커널 트릭: RBF K(x, x') = exp(−γ‖x − x'‖²). γ가 크면 경계가 구불구불해진다 (특징 2개, C = 10)
for g in (0.5, 5, 50):
    m = SVC(C=10, gamma=g).fit(Z_tr, y_tr)
    print(f"  γ={g:<4g} 훈련 {m.score(Z_tr, y_tr):.3f} · 서포트 벡터 {m.n_support_.sum()}개")

# ---------------------------------------------------------------
# 6) 과제 A · 분류(SVC, RBF, 특징 8개): C는 훈련 데이터 안의 5겹 교차 검증으로 고른다
cv_c = StratifiedKFold(5, shuffle=True, random_state=42)
cv_scores = {C: cross_val_score(SVC(C=C), A, y_tr, cv=cv_c).mean() for C in (0.1, 1, 10, 100)}
for C, sc in cv_scores.items():
    print(f"  SVC C={C:<5g} 교차 검증 {sc:.3f}")
C_cls = max(cv_scores, key=cv_scores.get)
svc = SVC(C=C_cls).fit(A, y_tr)
acc = svc.score(B, y_te)
print(f"[과제 A] C={C_cls} · γ='scale'={1 / (A.shape[1] * A.var()):.3f} · 훈련 {svc.score(A, y_tr):.3f} · 시험 {acc:.3f} · "
      f"서포트 벡터 {svc.n_support_.sum()}개 / {len(A)}")
cm = confusion_matrix(y_te, svc.predict(B))
print(pd.DataFrame(cm, index=["실제 불합격", "실제 합격"], columns=["예측 불합격", "예측 합격"]))
print(classification_report(y_te, svc.predict(B), target_names=["불합격", "합격"], digits=3))

# ---------------------------------------------------------------
# 7) 과제 A' · 강도 3등급: SVC는 일대일(OvO) 이진 분류 세 개의 투표로 다중 분류를 한다
svc3 = SVC(C=C_cls).fit(A, g_tr)
print(f"[3등급] 시험 {svc3.score(B, g_te):.3f} · 기준선(늘 보통) {np.mean(g_te == 1):.3f}")

# ---------------------------------------------------------------
# 8) 과제 B · 회귀(SVR, RBF): ε 띠 안의 오차는 벌하지 않는다. C는 5겹 교차 검증으로 고른다
cv_r = KFold(5, shuffle=True, random_state=42)
cv_r2 = {C: cross_val_score(SVR(C=C), A, s_tr, cv=cv_r).mean() for C in (1, 10, 100)}
for C, sc in cv_r2.items():
    print(f"  SVR C={C:<4g} 교차 검증 R² {sc:.3f}")
C_reg = max(cv_r2, key=cv_r2.get)
svr = SVR(C=C_reg).fit(A, s_tr)
p = svr.predict(B)
r2 = r2_score(s_te, p)
print(f"[과제 B] C={C_reg} · ε=0.1 · 시험 R² {r2:.3f} · MAE {mean_absolute_error(s_te, p):.2f}MPa · 서포트 벡터 {len(svr.support_)}개")
for eps in (1, 5):
    m = SVR(C=C_reg, epsilon=eps).fit(A, s_tr)
    print(f"  ε={eps}: 시험 R² {r2_score(s_te, m.predict(B)):.3f} · 서포트 벡터 {len(m.support_)}개")

# ---------------------------------------------------------------
# 9) 근무 조건 실험: 기본값 C=1, 표준화 빼기, γ 크게(5)
print(f"  기본 C=1      분류 {SVC().fit(A, y_tr).score(B, y_te):.3f} · 회귀 R² {SVR().fit(A, s_tr).score(B, s_te):.3f}")
print(f"  표준화 없음   분류 {SVC(C=C_cls).fit(X[idx_tr], y_tr).score(X[idx_te], y_te):.3f} · "
      f"회귀 R² {SVR(C=C_reg).fit(X[idx_tr], s_tr).score(X[idx_te], s_te):.3f}")
print(f"  γ=5           분류 {SVC(C=C_cls, gamma=5).fit(A, y_tr).score(B, y_te):.3f} (훈련 {SVC(C=C_cls, gamma=5).fit(A, y_tr).score(A, y_tr):.3f}) · "
      f"회귀 R² {SVR(C=C_reg, gamma=5).fit(A, s_tr).score(B, s_te):.3f}")
for kernel in ("linear", "poly"):
    print(f"  커널 {kernel:<6} C={C_cls}: 분류 시험 {SVC(kernel=kernel, C=C_cls).fit(A, y_tr).score(B, y_te):.3f}")

# ---------------------------------------------------------------
# 10) 판정: 겸용팀이라 분류와 회귀 둘 다 합격선을 넘어야 한다
ok_cls = acc > manual_acc and acc > dummy
ok_reg = r2 > LINE_R2
print(f"분류 {acc:.3f} (합격선 {manual_acc:.3f}) · 회귀 R² {r2:.3f} (합격선 {LINE_R2:.3f})")
print("판정:", "합격" if ok_cls and ok_reg else "보류")
