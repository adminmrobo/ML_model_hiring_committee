# ===============================================================
# 모델 채용 위원회 5-13 · GMM(가우시안 혼합 모델) 면접
#  - 라이브러리: numpy, scikit-learn (Colab 기본). 인터넷이 없어도 된다(load_digits는 scikit-learn 안에 들어 있다)
#  - 하는 일: 손글씨 숫자 1,797장을 정답 없이 "종 모양 열 개의 섞임"으로 묶는다.
#             64칸을 PCA로 30축까지 줄이고 → EM으로 종 모양을 맞추고 → 공분산 모양과 묶음 수(BIC)를 고른 뒤
#             채점 봉투(숨긴 정답)를 열어 ARI로 매뉴얼 씨의 합격선과 비교한다
#  - 정답 y는 학습(1~7번 칸)에 한 번도 쓰지 않는다. 8번 칸 "채점 봉투 열기"부터만 쓴다
# ===============================================================
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")   # 작은 행렬 계산에서 스레드끼리 다투지 않게 한다(결과는 같고 더 빠르다)
import time
import numpy as np
import sklearn
from scipy.stats import multivariate_normal
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.utils import check_random_state
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

np.set_printoptions(precision=3, suppress=True)
print("scikit-learn", sklearn.__version__)

# ---------------------------------------------------------------
# 1) 실기 과제지: 손글씨 숫자 1,797장 (8×8 픽셀, 밝기 0~16)
#    [내 데이터로 바꾸려면] X만 바꾼다. 한 줄이 글씨(샘플) 하나, 열이 특징이다
X, y_envelope = load_digits(return_X_y=True)
#    y_envelope = 채점 봉투(숨긴 정답 0~9). 8번 칸 전까지는 열지 않는다
print("과제지", X.shape, "· 밝기", X.min(), "~", X.max(), "· 늘 0인 칸", int((X.std(0) == 0).sum()), "개")

# ---------------------------------------------------------------
# 2) 매뉴얼 씨의 정리 규칙 (합격선): 가운데가 비었나 · 잉크가 많나 · 위가 무거운가 → 8묶음
ink = X.sum(1)
top = X[:, :32].sum(1) / np.maximum(ink, 1)
center = X[:, [27, 28, 35, 36]].sum(1)
manual = (center < 12) * 4 + (ink > 300) * 2 + (top > 0.5) * 1
print("매뉴얼 씨 묶음 크기:", np.bincount(manual))

# ---------------------------------------------------------------
# 3) 먼저 줄인다: 64칸 → PCA 30축 (가장 많이 퍼진 방향 30개, 5-14 PCA 편)
#    64칸 그대로 종 모양을 맞추면 묶음 하나에 공분산 숫자가 64×65/2 = 2,080개. 1,797장으로는 모자란다
pca = PCA(n_components=30, random_state=0)
Z = pca.fit_transform(X)
print(f"30축이 담은 분산 {pca.explained_variance_ratio_.sum():.3f}")
for d in (64, 30):
    n_par = 10 * d * (d + 1) // 2 + 10 * d + 9      # 공분산 + 평균 + 비중(합이 1이라 9개)
    print(f"  {d}차원 full GMM 10묶음이 외울 숫자(모수): {n_par:,}개  ← 글씨는 1,797장")

# ---------------------------------------------------------------
# 4) EM을 직접 짜 보기: 시작 묶음(K-Means) → [E] 소속 확률 → [M] 비중·평균·공분산 → 반복
def my_em(Z, start_labels, K, tol=1e-3, reg=1e-6, max_iter=100):
    n, d = Z.shape
    R = np.eye(K)[start_labels]                     # 처음엔 K-Means 묶음을 확률 1로 놓는다
    prev = -np.inf
    for it in range(max_iter + 1):
        nk = R.sum(0) + 10 * np.finfo(float).eps    # [M] 묶음마다 "나눠 가진 글씨 수"
        w, mu = nk / n, R.T @ Z / nk[:, None]       # 비중, 평균
        logp = np.empty((n, K))
        for k in range(K):
            D = Z - mu[k]
            C = (R[:, k, None] * D).T @ D / nk[k] + reg * np.eye(d)   # 공분산 (+ 아주 작은 값으로 안정화)
            L = np.linalg.cholesky(C)
            q = np.linalg.solve(L, D.T)
            logp[:, k] = np.log(w[k]) - 0.5 * (d * np.log(2 * np.pi) + (q ** 2).sum(0)) - np.log(np.diag(L)).sum()
        m = logp.max(1, keepdims=True)
        lse = (m + np.log(np.exp(logp - m).sum(1, keepdims=True)))[:, 0]
        R = np.exp(logp - lse[:, None])             # [E] 소속 확률: 묶음마다 합이 1
        ll = lse.mean()                             # 평균 로그 가능도: 커질수록 종 모양이 글씨를 잘 덮는다
        if abs(ll - prev) < tol:
            break
        prev = ll
    return R, ll, it

start = KMeans(10, n_init=1, random_state=check_random_state(0)).fit(Z).labels_   # GaussianMixture(random_state=0)와 같은 시작점
t0 = time.perf_counter()
R_mine, ll_mine, it_mine = my_em(Z, start, 10)
t_mine = time.perf_counter() - t0
gmm = GaussianMixture(n_components=10, covariance_type="full", random_state=0).fit(Z)
labels = gmm.predict(Z)
print(f"  직접 짠 EM: {it_mine}번 · 평균 로그 가능도 {ll_mine:.4f} · {t_mine * 1000:.0f}ms")
print(f"  scikit-learn: {gmm.n_iter_}번 · 평균 로그 가능도 {gmm.lower_bound_:.4f} · 같은 묶음 비율 {(R_mine.argmax(1) == labels).mean():.4f}")

# ---------------------------------------------------------------
# 5) 소속 확률: 글씨마다 "묶음별 확률"을 나눠 갖는다. 30차원에서는 거의 0 아니면 1로 날카롭다
P = gmm.predict_proba(Z)
mx = P.max(1)
print("  묶음 크기:", np.bincount(labels), "· 비중:", gmm.weights_.round(3))
print(f"  [30차원] 가장 큰 확률이 0.9 미만인 글씨 {(mx < 0.9).sum()}장 · 0.99 미만 {(mx < 0.99).sum()}장")
Z2 = Z[:, :2]
gmm2 = GaussianMixture(10, random_state=0).fit(Z2)
mx2 = gmm2.predict_proba(Z2).max(1)
print(f"  [2차원 PC1·PC2] 가장 큰 확률이 0.9 미만인 글씨 {(mx2 < 0.9).sum()}장 — 종이 넓게 겹친다")

# ---------------------------------------------------------------
# 6) 공분산 모양 네 가지: 정답 없이 BIC(작을수록 좋다)로 비교한다
models = {}
for ct in ("full", "tied", "diag", "spherical"):
    m = GaussianMixture(10, covariance_type=ct, random_state=0).fit(Z)
    models[ct] = m
    print(f"  {ct:<9} 모수 {m._n_parameters():>5,}개 · BIC {m.bic(Z):>9,.0f} · 반복 {m.n_iter_}번")

# ---------------------------------------------------------------
# 7) 묶음 수 고르기 (BIC)와 시작점 운
print("  k    BIC")
bic_k = {}
for k in (4, 6, 7, 8, 9, 10, 11, 12, 14, 16, 20):
    m = GaussianMixture(k, random_state=0).fit(Z)
    bic_k[k] = m
    print(f"  {k:>2}  {m.bic(Z):,.0f}")
best_k = min(bic_k, key=lambda k: bic_k[k].bic(Z))
print(f"  BIC가 고른 k = {best_k}  (숫자가 열 종류라는 것은 채점 봉투 속 정보다)")
seeds = {}
for s in range(6):
    m = GaussianMixture(10, random_state=s).fit(Z)
    seeds[s] = m
    print(f"  random_state={s}: BIC {m.bic(Z):,.0f}")
m10 = GaussianMixture(10, random_state=0, n_init=10).fit(Z)
print(f"  n_init=10 (열 번 시작해 가능도가 가장 높은 것): BIC {m10.bic(Z):,.0f}")

# ---------------------------------------------------------------
# 8) 채점 봉투 열기: 여기서부터 정답(y_envelope)을 쓴다. 묶음 번호는 이름이 아니므로 ARI·NMI로 잰다
ari, nmi = adjusted_rand_score, normalized_mutual_info_score
rng = np.random.default_rng(0)
print(f"  무작위 10묶음        ARI {ari(y_envelope, rng.integers(0, 10, len(X))):.3f}")
print(f"  매뉴얼 씨 (합격선)   ARI {ari(y_envelope, manual):.3f} · NMI {nmi(y_envelope, manual):.3f}")
print(f"  GMM full k=10       ARI {ari(y_envelope, labels):.3f} · NMI {nmi(y_envelope, labels):.3f} · 실루엣(64칸) {silhouette_score(X, labels):.3f}")
for ct, m in models.items():
    print(f"    공분산 {ct:<9} ARI {ari(y_envelope, m.predict(Z)):.3f}")
print(f"    BIC가 고른 k={best_k}     ARI {ari(y_envelope, bic_k[best_k].predict(Z)):.3f}")
for s, m in seeds.items():
    print(f"    random_state={s}    ARI {ari(y_envelope, m.predict(Z)):.3f}")
print(f"    n_init=10          ARI {ari(y_envelope, m10.predict(Z)):.3f}")
raw = GaussianMixture(10, random_state=0).fit(X)
print(f"    PCA 없이 64칸       ARI {ari(y_envelope, raw.predict(X)):.3f}")
Zs = PCA(30, random_state=0).fit_transform(StandardScaler().fit_transform(X))
print(f"    표준화 후 PCA 30    ARI {ari(y_envelope, GaussianMixture(10, random_state=0).fit(Zs).predict(Zs)):.3f}")

# ---------------------------------------------------------------
# 9) 업무 점수: 묶음마다 가장 전형적인 한 장(그 종의 밀도가 가장 높은 글씨)에만 사람이 이름을 붙인다
dens = np.column_stack([multivariate_normal(gmm.means_[k], gmm.covariances_[k]).logpdf(Z) for k in range(10)])
reps = [int(np.flatnonzero(labels == k)[dens[labels == k, k].argmax()]) for k in range(10)]
name = {k: y_envelope[reps[k]] for k in range(10)}          # 사람이 대표 10장만 보고 붙인 이름
named = np.array([name[c] for c in labels])
print("  대표 이름:", [int(name[k]) for k in range(10)])
print(f"  대표 10장만 이름 붙이기 → 전체 {(named == y_envelope).mean():.3f} 맞음")
score = gmm.score_samples(Z)                                 # 글씨마다 로그 가능도: 낮으면 어느 종에도 잘 안 맞는 글씨
for frac in (0.05, 0.10, 0.20):
    low = score <= np.quantile(score, frac)
    fixed = np.where(low, y_envelope, named)                 # 가능도가 낮은 글씨만 사람이 다시 본다
    print(f"  가능도 하위 {frac:.0%}({low.sum()}장)를 사람이 다시 보면 → {(fixed == y_envelope).mean():.3f} 맞음")

# ---------------------------------------------------------------
# 10) GMM은 새 글씨도 그린다 (생성 모델): 종 모양에서 점을 뽑아 PCA를 거꾸로 돌린다
Znew, comp = gmm.sample(10)
new = np.clip(pca.inverse_transform(Znew), 0, 16)
k0 = int(comp[0])
print(f"  묶음 {k0}(대표 이름 {name[k0]})에서 뽑은 새 글씨:")
for r in new[0].reshape(8, 8):
    print("   ", "".join(" .:-=+*#%@"[min(9, int(v / 16 * 9.99))] * 2 for v in r))

# ---------------------------------------------------------------
# 11) 군집팀 비교 (같은 1,797장, random_state=0)
team = [("5-10 K-Means k=10 (원래 픽셀)", KMeans(10, n_init=10, random_state=0).fit_predict(X)),
        ("5-11 계층적 ward", AgglomerativeClustering(10, linkage="ward").fit_predict(X)),
        ("5-13 GMM full k=10 (PCA 30)", labels)]
for nm_, lab in team:
    print(f"  {nm_:<30} ARI {ari(y_envelope, lab):.3f}")

# ---------------------------------------------------------------
# 12) 판정: ARI가 매뉴얼 씨의 합격선과 무작위(≈0)를 모두 넘으면 합격
final = ari(y_envelope, labels)
line = ari(y_envelope, manual)
print(f"GMM ARI {final:.3f} (합격선 {line:.3f})")
print("판정:", "합격" if final > line and final > 0.01 else "보류")
