# ===============================================================
# 모델 채용 위원회 5-15 · t-SNE · UMAP 면접 (비지도 본부 › 차원축소팀, 마지막 면접)
#  - 라이브러리: numpy, scipy, scikit-learn (Colab 기본) + umap-learn (없으면 자동 설치)
#  - 하는 일: 정답 없는 손글씨 숫자 1,797장(8×8 = 64칸)을 "이웃을 지키며" 2차원 지도로 펼치고,
#             매뉴얼 씨의 지도 · PCA 지도와 같은 잣대(2D 지도 위 5-NN 정확도)로 채점해 합격선과 비교한다
#  - 규칙: 정답(y)은 지도를 만드는 데 한 번도 쓰지 않는다. y는 "채점 봉투"다. 봉투를 여는 칸에는 [채점]이라고 적었다
#  - 인터넷이 필요 없다(load_digits는 scikit-learn 안에 들어 있다). umap-learn 설치만 인터넷을 쓴다
# ===============================================================
import subprocess, sys, time, warnings
import numpy as np
import sklearn
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, pairwise_distances_argmin
try:
    import umap
except ImportError:
    # PyPI의 "umap"은 다른 패키지다. 이름이 umap-learn이어야 한다
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "umap-learn"], check=True)
    import umap
warnings.filterwarnings("ignore", message=".*n_jobs value.*")   # random_state를 주면 UMAP이 한 줄(1코어)로 돈다는 안내

np.set_printoptions(precision=3, suppress=True)
print("scikit-learn", sklearn.__version__, "· umap-learn", umap.__version__)

# ---------------------------------------------------------------
# 1) 실기 과제지: 손글씨 숫자 1,797장. 한 장 = 8×8 픽셀 = 밝기(0~16) 64개
#    [내 데이터로 바꾸려면] X만 바꾼다. 한 줄에 한 장(또는 한 건), 열은 숫자여야 한다
digits = load_digits()
X = digits.data                    # (1797, 64) — 지원자가 보는 것은 이것뿐이다
y = digits.target                  # 채점 봉투: 정답 0~9. 지도를 만드는 칸에서는 쓰지 않는다
assert X.shape == (1797, 64)
print(f"글씨 {X.shape[0]}장 · 칸 {X.shape[1]}개 · 밝기 {X.min():.0f}~{X.max():.0f}")

# 2D 지도의 잣대 [채점]: 지도 위 좌표만 보고 가까운 이웃 5장의 정답으로 맞히기(5겹 교차 검증)
#   이웃이 지도 위에서도 같은 숫자끼리 모여 있으면 높게 나온다. 지도가 얼마나 "읽을 만한가"를 잰다
def map_score(Z):
    return cross_val_score(KNeighborsClassifier(n_neighbors=5), Z, y, cv=5).mean()

# ---------------------------------------------------------------
# 2) 합격선: 매뉴얼 씨의 지도 (사람이 정한 두 숫자: 잉크 양, 위쪽 절반의 잉크 비율)
ink = X.sum(axis=1)
top = X[:, :32].sum(axis=1) / np.maximum(ink, 1)
Z_man = np.c_[ink, top]
LINE = map_score(Z_man)
BASE = cross_val_score(DummyClassifier(), Z_man, y, cv=5).mean()   # 늘 가장 흔한 숫자만 답하기
print(f"[채점] 합격선 · 매뉴얼 씨 지도 5-NN {LINE:.3f} · 기준선(가장 흔한 숫자) {BASE:.3f}")

# ---------------------------------------------------------------
# 3) 앞 지원자 5-14 PCA: 가장 퍼진 두 방향으로 그림자를 비춘다 (직선 투영)
t0 = time.perf_counter()
Z_pca = PCA(n_components=2, random_state=0).fit_transform(X)
print(f"PCA 2D {time.perf_counter() - t0:.2f}초")

# ---------------------------------------------------------------
# 4) t-SNE: 64칸 공간의 이웃 확률(P)과 지도 위 이웃 확률(Q)이 같아지도록 점을 옮긴다
#    perplexity ≈ "한 장이 챙기는 이웃 수". 기본값 30. 처음 250번은 끌림을 12배로 키운다(early_exaggeration)
t0 = time.perf_counter()
tsne = TSNE(n_components=2, perplexity=30, random_state=0)
Z_tsne = tsne.fit_transform(X)            # fit_transform만 있다. 새 글씨를 놓는 transform은 없다
T_TSNE = time.perf_counter() - t0
print(f"t-SNE {T_TSNE:.1f}초 · 반복 {tsne.n_iter_ + 1}번 · KL 발산 {tsne.kl_divergence_:.3f} · transform 있음? {hasattr(tsne, 'transform')}")

# ---------------------------------------------------------------
# 5) UMAP: 이웃 그래프(가까운 15장)를 만들고, 그 그래프를 2D에서 다시 그린다
#    n_neighbors ≈ "얼마나 넓게 보나", min_dist = 지도 위에서 점끼리 얼마나 붙을 수 있나
t0 = time.perf_counter()
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=0)
Z_umap = reducer.fit_transform(X)         # y를 넣지 않는다! fit_transform(X, y=y)는 정답으로 지도를 그리는 "지도학습 UMAP"이다
T_UMAP = time.perf_counter() - t0
print(f"UMAP {T_UMAP:.1f}초 (처음 실행은 numba 컴파일 시간이 섞인다)")

# ---------------------------------------------------------------
# 6) 채점 봉투 열기 [채점]: 같은 잣대로 지도 다섯 장 비교
#    + 정답 없이 재는 점수: trustworthiness(지도 위 이웃이 원래 64칸에서도 이웃이었나, 1이 완벽)
rows = [("매뉴얼 씨 (잉크·위쪽)", Z_man), ("PCA 2D (5-14)", Z_pca), ("t-SNE", Z_tsne), ("UMAP", Z_umap)]
print("  지도                    5-NN 정확도   trustworthiness(정답 없이)")
for name, Z in rows:
    print(f"  {name:<20}   {map_score(Z):.3f}         {trustworthiness(X, Z, n_neighbors=5):.3f}")
print(f"  {'(참고) 원래 64칸':<20}   {map_score(X):.3f}         —")

# ---------------------------------------------------------------
# 7) 근무 조건 바꿔 보기: t-SNE perplexity, UMAP n_neighbors · min_dist [채점은 5-NN 열만]
for p in (5, 100):
    Z = TSNE(n_components=2, perplexity=p, random_state=0).fit_transform(X)
    print(f"  t-SNE perplexity {p:>3}: 5-NN {map_score(Z):.3f} · trustworthiness {trustworthiness(X, Z, n_neighbors=5):.3f}")
for n in (5, 50, 200):
    Z = umap.UMAP(n_neighbors=n, random_state=0).fit_transform(X)
    print(f"  UMAP n_neighbors {n:>3}: 5-NN {map_score(Z):.3f} · trustworthiness {trustworthiness(X, Z, n_neighbors=5):.3f}")
for md in (0.0, 0.99):
    Z = umap.UMAP(min_dist=md, random_state=0).fit_transform(X)
    print(f"  UMAP min_dist {md:<4}: 5-NN {map_score(Z):.3f}")

# ---------------------------------------------------------------
# 8) 거리와 크기는 믿지 말 것 [채점]: 숫자별 무리의 중심 거리 · 무리 크기를 64칸과 지도에서 비교
#    중심 거리 순위 상관(1이면 64칸의 "먼 무리·가까운 무리" 순서를 그대로 지킴), 크기 비 = 가장 큰 무리 반지름 ÷ 가장 작은 무리
def geometry(Z):
    c = np.array([Z[y == d].mean(axis=0) for d in range(10)])
    r = np.array([np.linalg.norm(Z[y == d] - c[d], axis=1).mean() for d in range(10)])
    return c, r
c64, r64 = geometry(X)
print(f"  64칸 원래 공간: 크기 비 {r64.max() / r64.min():.2f}")
for name, Z in rows[1:]:
    c, r = geometry(Z)
    print(f"  {name:<14} 중심 거리 순위 상관 {spearmanr(pdist(c64), pdist(c))[0]:.3f} · 크기 비 {r.max() / r.min():.2f}")

# ---------------------------------------------------------------
# 9) 업무 점수 [채점]: 지도 위에서 K-Means로 묶고, 묶음마다 중심에 가장 가까운 "대표 한 장"에만 사람이 이름을 붙인다
#    → 같은 묶음 전체에 그 이름을 퍼뜨리면 1,797장 중 몇 %가 맞나 (= 라벨링 비용 절감)
#    주의: t-SNE·UMAP 지도는 무리를 실제보다 또렷하게 그리므로, 그 위의 군집 점수는 후하게 나온다
def propagate(Z, k):
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(Z)
    rep = pairwise_distances_argmin(km.cluster_centers_, Z)       # 묶음마다 대표 한 장
    named = y[rep][km.labels_]                                   # 대표의 이름을 묶음 전체에
    return (named == y).mean(), adjusted_rand_score(y, km.labels_)
for name, Z in rows + [("(참고) 원래 64칸", X)]:
    a10, ari = propagate(Z, 10)
    a30, _ = propagate(Z, 30)
    print(f"  {name:<20} 대표 10장 → {a10:.3f} · 대표 30장 → {a30:.3f} · K-Means ARI {ari:.3f}")

# ---------------------------------------------------------------
# 10) 새 글씨가 들어오면: UMAP은 transform으로 기존 지도 위에 놓는다. t-SNE는 처음부터 다시 그려야 한다
#     1,797장을 1,347장(지도 만들기)과 450장(나중에 들어온 글씨)으로 나눈다. 나누기에도 정답은 쓰지 않는다
i_old, i_new = train_test_split(np.arange(len(X)), test_size=0.25, random_state=0)
u = umap.UMAP(random_state=0).fit(X[i_old])
Z_new = u.transform(X[i_new])
acc_new = KNeighborsClassifier(n_neighbors=5).fit(u.embedding_, y[i_old]).score(Z_new, y[i_new])   # [채점]
print(f"  UMAP transform: 새 글씨 {len(i_new)}장을 기존 지도에 놓고 5-NN으로 읽기 {acc_new:.3f}")

# ---------------------------------------------------------------
# 11) 판정: 2D 지도 5-NN 정확도가 합격선(매뉴얼 씨 지도)과 기준선을 모두 넘으면 합격
s_t, s_u = map_score(Z_tsne), map_score(Z_umap)
print(f"t-SNE {s_t:.3f} · UMAP {s_u:.3f} (합격선 {LINE:.3f}, 기준선 {BASE:.3f})")
print("판정:", "합격" if min(s_t, s_u) > LINE and min(s_t, s_u) > BASE else "보류")
