# ===============================================================
# 모델 채용 위원회 5-14 · PCA(주성분 분석) 면접
#  - 라이브러리: numpy, scikit-learn (Colab 기본). 인터넷이 없어도 된다(load_digits는 scikit-learn에 들어 있다)
#  - 하는 일: 손글씨 숫자 1,797장(8×8 = 64칸)을 정답 없이 "가장 퍼진 방향"의 축 몇 개로 돌려 세우고,
#             설명된 분산, 성분 그림, 복원으로 무엇을 버렸는지 본 뒤,
#             2차원 지도를 채점 봉투(숨긴 정답)로 채점해 차원축소팀 합격선과 비교한다
#  - 규칙: 정답 y는 학습(PCA 맞추기)에 쓰지 않는다. "채점" 표시가 붙은 칸에서만 봉투를 연다
# ===============================================================
import time
import numpy as np
import sklearn
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, confusion_matrix

np.set_printoptions(precision=3, suppress=True, linewidth=120)
print("scikit-learn", sklearn.__version__)

# ---------------------------------------------------------------
# 1) 실기 과제지: 손글씨 숫자 1,797장. 한 장 = 8×8 픽셀 = 밝기(0~16) 64개
#    [내 데이터로 바꾸려면] X만 바꾼다. 한 줄에 샘플 하나, 열은 같은 단위의 숫자들
X, y_sealed = load_digits(return_X_y=True)     # y_sealed: 채점 봉투(숨긴 정답). 학습에는 쓰지 않는다
assert X.shape == (1797, 64)
print(f"글씨 {X.shape[0]}장 · 특징 {X.shape[1]}칸 · 밝기 {X.min():.0f}~{X.max():.0f}")
print("첫 장 (8×8):")
print(X[0].reshape(8, 8).astype(int))

# 5-NN 지도 채점 도구: 지도 위 가까운 5장의 정답으로 맞히기, 5겹 교차 검증 (채점 칸에서만 부른다)
def map_score(Z):
    return cross_val_score(KNeighborsClassifier(n_neighbors=5), Z, y_sealed, cv=5).mean()

# ---------------------------------------------------------------
# 2) 합격선 [채점]: 매뉴얼 씨의 2D 지도 = (잉크 양, 위쪽 비율). 사람이 정한 두 숫자로 펼친 지도
ink = X.sum(axis=1)
top = X[:, :32].sum(axis=1) / np.maximum(ink, 1)
manual_map = np.c_[ink, top]
DR_LINE = map_score(manual_map)
print(f"차원축소팀 합격선 · 매뉴얼 씨 지도 5-NN {DR_LINE:.3f}")
print(f"  (같은 지도를 표준화하면 {map_score(StandardScaler().fit_transform(manual_map)):.3f}. 원래 단위에서는 잉크 양이 거리를 독차지한다)")
rng = np.random.RandomState(0)
print(f"기준선 · 가장 흔한 숫자 비율 {np.bincount(y_sealed).max() / len(y_sealed):.3f} · 무작위 지도 {map_score(rng.randn(len(X), 2)):.3f}")

# ---------------------------------------------------------------
# 3) PCA를 직접 짜 보기: 평균 빼기 → 공분산 행렬(64×64) → 고유값 분해 → 큰 고유값 순서로 축 세우기
mu = X.mean(axis=0)
Xc = X - mu
cov = Xc.T @ Xc / (len(X) - 1)
eigval, eigvec = np.linalg.eigh(cov)                  # 대칭 행렬용. 작은 값부터 나온다
order = np.argsort(eigval)[::-1]
eigval, eigvec = eigval[order], eigvec[:, order]
pca = PCA().fit(X)                                    # scikit-learn (정답 y를 넣지 않는다)
print("직접 계산한 설명된 분산 비율(1~5):", (eigval / eigval.sum())[:5])
print("scikit-learn explained_variance_ratio_:", pca.explained_variance_ratio_[:5])
same = [abs(abs(eigvec[:, k] @ pca.components_[k]) - 1) < 1e-6 for k in range(10)]
print("성분 1~10 방향 일치(부호 무시):", all(same), "· 성분끼리 직교:", np.allclose(pca.components_ @ pca.components_.T, np.eye(64)))

# ---------------------------------------------------------------
# 4) 설명된 분산: 축 k개가 전체 퍼짐(분산)의 몇 %를 담나
cum = np.cumsum(pca.explained_variance_ratio_)
for k in (1, 2, 3, 5, 10, 20, 30, 40):
    print(f"  {k:2d}성분: {cum[k - 1]:.3f}")
print(f"80%에 {np.argmax(cum >= 0.8) + 1}성분 · 90%에 {np.argmax(cum >= 0.9) + 1}성분 · 95%에 {np.argmax(cum >= 0.95) + 1}성분")
print("PCA(n_components=0.9)가 고른 성분 수:", PCA(n_components=0.9).fit(X).n_components_)

# ---------------------------------------------------------------
# 5) 성분 그림: 성분 하나 = 64칸짜리 "밝기 바꾸는 방향". 8×8로 접으면 그림이 된다(×100, +는 밝게 −는 어둡게)
print("평균 글씨 (8×8):")
print(mu.reshape(8, 8).round(1))
for k in range(2):
    print(f"성분 {k + 1} (×100):")
    print((pca.components_[k].reshape(8, 8) * 100).round(0).astype(int))

# ---------------------------------------------------------------
# 6) 복원: 평균 + Σ(점수 × 성분). 성분을 적게 쓰면 흐린 글씨가 된다. 버린 정보 = 복원 오차
for k in (1, 2, 5, 10, 21, 40, 64):
    p = PCA(n_components=k).fit(X)
    R = p.inverse_transform(p.transform(X))
    mse = ((X - R) ** 2).mean()
    print(f"  {k:2d}성분 복원: 픽셀당 제곱 오차 {mse:6.2f} · 남은 분산 비율 {((X - R) ** 2).sum() / (Xc ** 2).sum():.3f}")
p2 = PCA(n_components=2).fit(X)
R2 = p2.inverse_transform(p2.transform(X[:1]))
print("첫 장을 2성분으로 복원 (8×8, 반올림):")
print(R2.reshape(8, 8).round(0).astype(int))

# ---------------------------------------------------------------
# 7) 성과 평가 [채점]: 2D 지도(성분 1·2)를 채점 봉투로 채점. 5-NN 5겹 교차 검증
Z2 = p2.transform(X)
print("지도 좌표 처음 3장:", Z2[:3].round(2).tolist())
acc2 = map_score(Z2)
print(f"PCA 2D 지도 5-NN {acc2:.3f} (합격선 {DR_LINE:.3f})")
for k in (1, 3, 5, 10, 30):
    print(f"  {k:2d}성분 공간 5-NN {map_score(PCA(n_components=k).fit_transform(X)):.3f}")
print(f"  원래 64칸 5-NN {map_score(X):.3f}")
print(f"표준화한 뒤 PCA 2D 5-NN {map_score(PCA(n_components=2).fit_transform(StandardScaler().fit_transform(X))):.3f}")
pred = cross_val_predict(KNeighborsClassifier(n_neighbors=5), Z2, y_sealed, cv=5)
print("숫자별 정확도 0~9:", np.array([(pred[y_sealed == d] == d).mean() for d in range(10)]))
cm = confusion_matrix(y_sealed, pred)
pairs = sorted(((cm[a, b] + cm[b, a], a, b) for a in range(10) for b in range(a + 1, 10)), reverse=True)[:4]
print("가장 많이 겹친 숫자 쌍:", ", ".join(f"{a}↔{b} {n}장" for n, a, b in pairs))

# ---------------------------------------------------------------
# 8) 업무 점수 [채점]: 지도 위에서 K-Means로 10묶음 → 묶음마다 중심에 가장 가까운 한 장에만 사람이 이름을 붙이고
#    그 이름을 묶음 전체에 퍼뜨리면, 1,797장 중 몇 %가 맞게 이름 붙나 (사람 손: 10장)
def label_by_reps(Z):
    km = KMeans(n_clusters=10, n_init=10, random_state=0).fit(Z)
    out = np.empty_like(y_sealed)
    for c in range(10):
        m = np.where(km.labels_ == c)[0]
        rep = m[((Z[m] - km.cluster_centers_[c]) ** 2).sum(axis=1).argmin()]
        out[m] = y_sealed[rep]                         # 사람이 대표 한 장만 보고 이름을 붙인다
    return (out == y_sealed).mean(), adjusted_rand_score(y_sealed, km.labels_)
for name, Z in [("PCA 2D 지도", Z2), ("PCA 10성분", PCA(n_components=10).fit_transform(X)), ("원래 64칸", X)]:
    a, r = label_by_reps(Z)
    print(f"  {name}: 대표 10장으로 이름 붙이기 {a:.3f} · K-Means ARI {r:.3f}")

# ---------------------------------------------------------------
# 9) 다른 지원자의 전처리: GMM(5-13)은 64칸 그대로는 불안정해서 PCA 30성분 위에서 일한다
t0 = time.time()
Z30 = PCA(n_components=30, random_state=0).fit_transform(X)
gmm = GaussianMixture(n_components=10, covariance_type="full", random_state=0).fit(Z30)
print(f"PCA 30성분 + GMM(full) ARI {adjusted_rand_score(y_sealed, gmm.predict(Z30)):.3f} · {time.time() - t0:.1f}초")

# ---------------------------------------------------------------
# 10) 다음 지원자 미리 보기 [채점]: 이웃 관계를 지키며 펼치는 t-SNE (5-15). 같은 5-NN 채점
t0 = time.time()
Zt = TSNE(n_components=2, random_state=0).fit_transform(X)
print(f"t-SNE 2D 지도 5-NN {map_score(Zt):.3f} · {time.time() - t0:.1f}초 (PCA 2D는 {acc2:.3f})")

# ---------------------------------------------------------------
# 11) 판정: 2D 지도 5-NN이 합격선(매뉴얼 씨 지도)과 기준선(가장 흔한 숫자)을 모두 넘으면 합격
base = np.bincount(y_sealed).max() / len(y_sealed)
print("판정:", "합격" if acc2 > DR_LINE and acc2 > base else "보류",
      f"(5-NN {acc2:.3f} > 합격선 {DR_LINE:.3f}, 기준선 {base:.3f}) · 설명된 분산 {cum[1]:.3f}")
