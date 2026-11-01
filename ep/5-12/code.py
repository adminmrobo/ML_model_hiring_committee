# ===============================================================
# 모델 채용 위원회 5-12 · DBSCAN 면접 (비지도 본부 › 군집팀)
#  - 라이브러리: numpy, scikit-learn (Colab 기본). 인터넷이 필요 없다(load_digits는 scikit-learn 안에 들어 있다)
#  - 하는 일: 손글씨 숫자 1,797장을 정답 없이 "촘촘한 곳 = 묶음, 외톨이 = 잡음"으로 나누고,
#             eps를 k-거리 그래프로 고른 뒤(정답 없이), 마지막에만 채점 봉투(정답)를 열어 합격선과 비교한다
#  - 규칙: 정답 y는 학습과 근무 조건 고르기에 쓰지 않는다. y를 쓰는 줄에는 모두 "# [채점]"을 붙였다
# ===============================================================
import time, warnings
import numpy as np
import sklearn
from sklearn.datasets import load_digits
from sklearn.cluster import DBSCAN, HDBSCAN, KMeans, AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score, pairwise_distances
warnings.filterwarnings("ignore", category=FutureWarning)
np.set_printoptions(precision=3, suppress=True)
print("scikit-learn", sklearn.__version__)

# ---------------------------------------------------------------
# 1) 실기 과제지: 손글씨 숫자 1,797장 (8×8 픽셀, 밝기 0~16 → 특징 64개)
#    정답 y(0~9)는 "채점 봉투"에 봉인한다. 비지도 학습이라 시험지를 따로 떼지 않고 1,797장 전체로 묶는다
#    [내 데이터로 바꾸려면] X만 바꾼다. 정답이 없으면 [채점] 줄은 지운다
digits = load_digits()
X = digits.data                                # (1797, 64) — 묶을 때는 이것만 쓴다
y_sealed = digits.target                       # [채점] 봉투 속 정답. 5·8~12번 칸의 채점에만 연다
assert X.shape == (1797, 64)
print(f"글씨 {X.shape[0]}장 · 특징 {X.shape[1]}개(픽셀) · 밝기 {X.min():.0f}~{X.max():.0f}")

def label_by_representative(X, labels, y):
    """업무 점수: 묶음마다 평균 글씨에 가장 가까운 한 장(대표)에만 사람이 이름을 붙이고 묶음 전체에 퍼뜨린다.
       잡음(-1)은 이름 없이 남으므로 틀린 것으로 센다. 반환: 맞게 이름 붙은 비율"""
    ok = 0
    for c in sorted(set(labels) - {-1}):
        m = np.where(labels == c)[0]
        rep = m[np.argmin(((X[m] - X[m].mean(0)) ** 2).sum(1))]   # 대표 고르기에는 정답을 쓰지 않는다
        ok += (y[m] == y[rep]).sum()                              # [채점] 대표의 이름이 맞게 퍼졌나
    return ok / len(labels)

def report(name, labels):
    k, noise = labels.max() + 1, (labels == -1).sum()
    ari = adjusted_rand_score(y_sealed, labels)                   # [채점]
    nmi = normalized_mutual_info_score(y_sealed, labels)          # [채점]
    biz = label_by_representative(X, labels, y_sealed)            # [채점]
    print(f"{name} → 묶음 {k}개 · 잡음 {noise}장({noise / len(labels):.1%}) · "
          f"ARI {ari:.3f} · NMI {nmi:.3f} · 업무 점수 {biz:.1%}")
    return ari, biz

# ---------------------------------------------------------------
# 2) 합격선: 매뉴얼 씨의 정리 규칙 (가운데가 비었나 · 잉크가 많나 · 위가 무거운가 → 8묶음, 학습 없음)
ink = X.sum(1)
top = X[:, :32].sum(1) / np.maximum(ink, 1)
center = X[:, [27, 28, 35, 36]].sum(1)
manual = (center < 12) * 4 + (ink > 300) * 2 + (top > 0.5) * 1
LINE_ARI, LINE_BIZ = report("매뉴얼 씨 8묶음 (합격선)", manual)

# ---------------------------------------------------------------
# 3) DBSCAN 기본값 그대로: eps=0.5, min_samples=5
#    밝기 0~16짜리 64칸에서 가장 가까운 이웃도 10 넘게 떨어져 있다. 반지름 0.5 안에는 자기 자신뿐 → 전부 잡음
labels = DBSCAN().fit_predict(X)
print("기본값 eps=0.5:", "묶음", labels.max() + 1, "· 잡음", (labels == -1).sum(), "장")

# ---------------------------------------------------------------
# 4) eps 고르기 (정답 없이): k-거리 그래프
#    점마다 min_samples번째로 가까운 점(자기 자신 포함)까지의 거리를 재서 작은 것부터 늘어놓는다.
#    곡선이 확 꺾이는 "무릎"을 eps로 삼는다. 무릎 = 양 끝을 잇는 직선에서 가장 멀리 떨어진 점
def knee_eps(X, min_samples):
    d, _ = NearestNeighbors(n_neighbors=min_samples).fit(X).kneighbors(X)   # 첫 이웃은 자기 자신(거리 0)
    k = np.sort(d[:, min_samples - 1])
    x = np.arange(len(k)) / (len(k) - 1)
    yy = (k - k[0]) / (k[-1] - k[0])
    i = np.argmax(np.abs(yy - x))
    return round(float(k[i]), 2), i, k

for ms in [3, 5, 10, 20]:
    eps_k, i, k = knee_eps(X, ms)
    print(f"min_samples={ms:>2}: k-거리 중앙값 {np.median(k):.2f} · 무릎 {i + 1}번째(위쪽 {1 - i / len(k):.0%}) → eps {eps_k}")
EPS, MS = knee_eps(X, 5)[0], 5                 # 기준 모델: scikit-learn 기본 min_samples=5 + 무릎 eps

# ---------------------------------------------------------------
# 5) 기준 모델 학습 → 정답 없이 재는 점수 → 채점 봉투 열기
t0 = time.time()
db = DBSCAN(eps=EPS, min_samples=MS).fit(X)
labels = db.labels_
print(f"DBSCAN(eps={EPS}, min_samples={MS}) 학습 {time.time() - t0:.2f}초 · 핵심점 {len(db.core_sample_indices_)}장")
inside = labels != -1
if labels.max() + 1 >= 2:
    print(f"실루엣(잡음 뺀 {inside.sum()}장, 정답 없이) {silhouette_score(X[inside], labels[inside]):.3f}")
for c in range(labels.max() + 1):
    print(f"  묶음 {c}: {np.sum(labels == c)}장 · 봉투 속 숫자 분포 {np.bincount(y_sealed[labels == c], minlength=10)}")  # [채점]
ARI, BIZ = report(f"DBSCAN eps={EPS} (기준 모델)", labels)
rng = np.random.default_rng(0)
print(f"무작위 기준선(같은 크기 묶음을 무작위로) ARI {adjusted_rand_score(y_sealed, rng.permutation(labels)):.3f}")  # [채점]

# ---------------------------------------------------------------
# 6) 민감도: eps × min_samples에 따라 ARI(위)와 잡음 비율(아래)이 어떻게 바뀌나
EPS_GRID = list(range(16, 29))
print("ARI      eps:" + "".join(f"{e:>6}" for e in EPS_GRID))
noise_rows = []
for ms in [3, 5, 10, 20]:
    row_a, row_n = [], []
    for e in EPS_GRID:
        lb = DBSCAN(eps=e, min_samples=ms).fit_predict(X)
        row_a.append(adjusted_rand_score(y_sealed, lb))      # [채점]
        row_n.append((lb == -1).mean())
    print(f"min_samples={ms:>2}  " + "".join(f"{v:6.2f}" for v in row_a))
    noise_rows.append((ms, row_n))
for ms, row_n in noise_rows:
    print(f"잡음 ms={ms:>2}     " + "".join(f"{v:6.0%}" for v in row_n))

# ---------------------------------------------------------------
# 7) 차원의 저주: 64차원에서는 "가장 가까운 이웃"과 "가장 먼 점"의 거리가 비슷해진다
D = pairwise_distances(X)
np.fill_diagonal(D, np.nan)
print(f"손글씨 64차원: 가장 가까운 ÷ 가장 먼 거리(중앙값) {np.nanmedian(np.nanmin(D, 1) / np.nanmax(D, 1)):.3f}")
for dim in [2, 8, 16, 64]:
    U = np.random.default_rng(0).uniform(0, 16, (400, dim))   # 구조 없이 고르게 흩뿌린 점 400개
    Du = pairwise_distances(U)
    np.fill_diagonal(Du, np.nan)
    print(f"고른 무작위 {dim:>2}차원: 가장 가까운 ÷ 가장 먼 거리(중앙값) {np.nanmedian(np.nanmin(Du, 1) / np.nanmax(Du, 1)):.3f}")

# ---------------------------------------------------------------
# 8) 근무 조건을 바꿔 보기 ① min_samples=3 (eps는 같은 방법, 정답 없이 k-거리 무릎으로)
eps3 = knee_eps(X, 3)[0]
report(f"DBSCAN ms=3, eps={eps3}", DBSCAN(eps=eps3, min_samples=3).fit_predict(X))

# ---------------------------------------------------------------
# 9) 근무 조건을 바꿔 보기 ② 정답을 엿보고 eps 고르기 — [반칙] 실제 업무에서는 봉투가 없다
best = max((adjusted_rand_score(y_sealed, DBSCAN(eps=e, min_samples=5).fit_predict(X)), e)    # [채점] 반칙
           for e in np.arange(16, 28.01, 0.5))
report(f"[반칙] 엿본 eps={best[1]:.1f}", DBSCAN(eps=best[1], min_samples=5).fit_predict(X))

# ---------------------------------------------------------------
# 10) eps 없이: HDBSCAN (밀도 문턱을 여러 높이에서 보고 오래 버티는 묶음을 고른다)
report("HDBSCAN min_cluster_size=5", HDBSCAN(min_cluster_size=5, copy=True).fit_predict(X))

# ---------------------------------------------------------------
# 11) 비교용: 군집팀 다른 지원자(같은 1,797장)와 그림용 2D 지도
report("K-Means k=10 (5-10)", KMeans(10, n_init=10, random_state=0).fit_predict(X))
report("계층적 ward 10묶음 (5-11)", AgglomerativeClustering(10).fit_predict(X))
t0 = time.time()
xy = TSNE(n_components=2, random_state=0).fit_transform(X)    # 위치를 보여 주는 그림용. DBSCAN은 64차원에서 돈다
print(f"t-SNE 2D 지도 {time.time() - t0:.1f}초 · 첫 장 좌표 {xy[0].round(1)}")

# ---------------------------------------------------------------
# 12) 판정: ARI가 합격선(매뉴얼 씨)과 무작위(≈0)를 넘고, 업무 점수도 매뉴얼 씨보다 높아야 합격
passed = ARI > LINE_ARI and ARI > 0.0 and BIZ > LINE_BIZ
print(f"기준 모델 ARI {ARI:.3f} (합격선 {LINE_ARI:.3f}) · 업무 점수 {BIZ:.1%} (매뉴얼 씨 {LINE_BIZ:.1%})")
print("판정:", "합격" if passed else "보류 — k-거리 무릎으로 고른 eps가 64차원에서 너무 커서 거의 한 묶음이 됐다")
