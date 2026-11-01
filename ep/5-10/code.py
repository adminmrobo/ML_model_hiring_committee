# ===============================================================
# 모델 채용 위원회 5-10 · K-Means 면접 (+ K-Medoids)
#  - 라이브러리: numpy, scikit-learn (Colab 기본). 인터넷이 없어도 된다(load_digits는 scikit-learn 안에 들어 있다)
#  - 하는 일: 손글씨 숫자 1,797장을 정답 없이 k개 묶음으로 나누고, 중심 그림("평균 글씨")을 보고,
#             관성(엘보)·실루엣으로 k를 고른 뒤, 마지막에 채점 봉투(숨긴 정답)를 열어 ARI로 채점한다
#  - 규칙: 정답 y는 7번 칸(채점 봉투 열기)부터만 쓴다. 학습과 k 고르기에는 쓰지 않는다
# ===============================================================
import time
import numpy as np
import sklearn
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans, kmeans_plusplus
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             silhouette_score, pairwise_distances)

np.set_printoptions(precision=3, suppress=True)
print("scikit-learn", sklearn.__version__)

# ---------------------------------------------------------------
# 1) 실기 과제지: 손글씨 숫자 1,797장 (8×8 픽셀, 밝기 0~16)
#    [내 데이터로 바꾸려면] X만 바꾼다. 한 줄 = 한 장, 열 = 특징(픽셀). 정답 열은 없어도 된다
digits = load_digits()
X = digits.data                       # (1797, 64): 학습에 쓰는 것은 이것뿐
envelope = digits.target              # 채점 봉투: 숨긴 정답 0~9. 7번 칸 전에는 열지 않는다
print(f"글씨 {X.shape[0]}장 · 특징 {X.shape[1]}칸 · 밝기 {X.min():.0f}~{X.max():.0f}")
SHADE = " .:-=+*#%@"
def show(img64):
    """8×8 글씨를 글자 그림으로 (밝을수록 진한 글자)"""
    return ["".join(SHADE[min(9, int(v / 16 * 9.99))] * 2 for v in row) for row in np.asarray(img64).reshape(8, 8)]
print("\n".join(show(X[0])))

# ---------------------------------------------------------------
# 2) 매뉴얼 씨의 정리 규칙 (사람이 정한 세 질문 → 8묶음). 정답은 쓰지 않는다
ink = X.sum(1)                                         # 잉크 양
top = X[:, :32].sum(1) / np.maximum(ink, 1)            # 위쪽 절반에 잉크가 몇 %인가
center = X[:, [27, 28, 35, 36]].sum(1)                 # 가운데 네 칸의 밝기
manual = (center < 12) * 4 + (ink > 300) * 2 + (top > 0.5) * 1
print("매뉴얼 씨 8묶음 크기:", np.bincount(manual, minlength=8))

# ---------------------------------------------------------------
# 3) K-Means 학습: k = 10, k-means++ 시작 10번 중 관성이 가장 작은 것
#    n_init 기본값은 1.4부터 "auto"(k-means++면 1번)다. 버전과 무관하게 같게 하려고 10을 적는다
t0 = time.perf_counter()
km = KMeans(n_clusters=10, n_init=10, random_state=0).fit(X)
print(f"관성(inertia) {km.inertia_:,.1f} · 반복 {km.n_iter_}번 · {1000*(time.perf_counter()-t0):.0f}ms")
print("묶음 크기:", np.bincount(km.labels_))
print("중심 10개 = 묶음마다 '평균 글씨' (64칸 평균을 다시 8×8로)")
pics = [show(c) for c in km.cluster_centers_]
for r in range(8):
    print(" ".join(p[r] for p in pics[:5]))
print()
for r in range(8):
    print(" ".join(p[r] for p in pics[5:]))

# ---------------------------------------------------------------
# 4) 수습 기간: 배정 → 평균으로 이동을 직접 짜서 scikit-learn과 맞춰 보기
#    scikit-learn이 쓴 시작(kmeans_plusplus를 같은 난수 순서로 10번)을 다시 만들고, 관성이 가장 작은 시작에서 출발한다
rs = np.random.RandomState(0)
starts = [kmeans_plusplus(X, 10, random_state=rs)[1] for _ in range(10)]
def lloyd(X, init_idx, max_iter=300, tol=1e-4):
    C = X[init_idx].astype(float)
    tol = tol * X.var(0).mean()                        # scikit-learn과 같은 멈춤 기준
    old = None
    for it in range(1, max_iter + 1):
        d = ((X[:, None, :] - C[None]) ** 2).sum(2)    # 글씨 × 중심 거리 제곱
        lab = d.argmin(1)                              # ① 가장 가까운 중심에 배정
        inertia = d.min(1).sum()
        C_new = np.array([X[lab == c].mean(0) if (lab == c).any() else C[c] for c in range(len(C))])  # ② 평균으로 이동
        shift = ((C_new - C) ** 2).sum()
        C = C_new
        changed = len(X) if old is None else int((lab != old).sum())
        yield it, inertia, changed, lab, C
        if changed == 0 or shift <= tol:
            break
        old = lab
inertias = [KMeans(10, init=X[s], n_init=1).fit(X).inertia_ for s in starts]
best = int(np.argmin(inertias))
for it, inertia, changed, lab, C in lloyd(X, starts[best]):
    if it <= 5 or changed == 0 or it % 4 == 0:
        print(f"  반복 {it:2d} · 관성 {inertia:12,.1f} · 묶음이 바뀐 글씨 {changed:4d}장")
d = ((X[:, None, :] - C[None]) ** 2).sum(2); lab = d.argmin(1)
print("직접 짠 결과 = scikit-learn 결과:", np.array_equal(lab, km.labels_), f"(관성 {d.min(1).sum():,.1f})")

# ---------------------------------------------------------------
# 5) 초기값 운: 시작 10번의 관성 (정답 없이 고른다: 관성이 가장 작은 시작)
for i, v in enumerate(inertias):
    print(f"  시작 {i+1:2d}: 관성 {v:12,.1f}" + ("  ← n_init='auto'면 이것 하나만" if i == 0 else "") + ("  ← 가장 작음 = 고른 시작" if i == best else ""))

# ---------------------------------------------------------------
# 6) k 고르기 (정답 없이): 관성(엘보)과 실루엣 점수
models = {}
print("  k   관성          실루엣")
for k in range(2, 16):
    models[k] = KMeans(k, n_init=10, random_state=0).fit(X)
    print(f"  {k:2d}  {models[k].inertia_:12,.1f}  {silhouette_score(X, models[k].labels_):.3f}")
print("  → 관성은 k가 늘면 늘 준다(엘보가 뚜렷하지 않다). 실루엣은 9·14에서 가장 높고 10은 0.183으로 비슷하다")

# ---------------------------------------------------------------
# 7) 채점 봉투 열기: 숨긴 정답과 겹치는 정도 (여기서 처음으로 y를 쓴다)
y = envelope
rng = np.random.RandomState(0)
rand_ari = np.mean([adjusted_rand_score(y, rng.randint(0, 10, len(y))) for _ in range(20)])
ink10 = np.digitize(ink, np.linspace(ink.min(), ink.max(), 11)[1:-1])   # 잉크 양만으로 10구간
def score(name, lab):
    print(f"  {name:22s} ARI {adjusted_rand_score(y, lab):.3f} · NMI {normalized_mutual_info_score(y, lab):.3f}")
print(f"  {'무작위 10묶음 (20번 평균)':22s} ARI {rand_ari:.3f}")
score("잉크 양 10구간", ink10)
score("매뉴얼 씨 8묶음 (합격선)", manual)
score("K-Means k=10", km.labels_)
print("묶음 × 정답 표 (행 = 묶음, 열 = 숫자 0~9)")
print("         " + " ".join(f"{d:3d}" for d in range(10)))
for c in range(10):
    row = np.bincount(y[km.labels_ == c], minlength=10)
    print(f"  묶음 {c}: " + " ".join(f"{v:3d}" for v in row) + f"  → 가장 많은 숫자 {row.argmax()} ({row.max()/row.sum():.0%})")

# ---------------------------------------------------------------
# 8) 업무 점수: 묶음마다 대표 한 장(중심에 가장 가까운 글씨)에만 사람이 이름을 붙이면
def label_by_representative(model):
    pred = np.empty(len(X), int)
    for c in range(model.n_clusters):
        m = np.where(model.labels_ == c)[0]
        rep = m[np.argmin(((X[m] - model.cluster_centers_[c]) ** 2).sum(1))]
        pred[m] = y[rep]                              # 사람이 대표 한 장만 보고 이름을 붙인다
    return (pred == y).mean()
for k in [10, 20, 30, 50]:
    model = models[k] if k in models else KMeans(k, n_init=10, random_state=0).fit(X)
    print(f"  k={k:2d}: 이름표 {k:2d}장 → 1,797장 중 {label_by_representative(model):.1%} 맞게 이름 붙음 · ARI {adjusted_rand_score(y, model.labels_):.3f}")
D = pairwise_distances(X)
acc = []
for _ in range(100):                                   # 비교: 무작위 10장에 이름을 붙이고 가장 가까운 이름표를 따른다
    s = rng.choice(len(X), 10, replace=False)
    acc.append((y[s[D[:, s].argmin(1)]] == y).mean())
print(f"  비교: 무작위 10장에 이름표 → {np.mean(acc):.1%} (100번 평균)")

# ---------------------------------------------------------------
# 9) 표준화는 약일까: KNN(5-1)에서는 약이었지만 여기서는?
Xs = StandardScaler().fit_transform(X)
km_s = KMeans(10, n_init=10, random_state=0).fit(Xs)
score("K-Means k=10 (표준화)", km_s.labels_)
sd = X.std(0)
print(f"  픽셀 표준편차: 가장 작은(0이 아닌) 칸 {sd[sd > 0].min():.3f} · 가장 큰 칸 {sd.max():.2f}")
print("  → 가장자리 칸은 거의 늘 0이라 표준편차가 작다. 표준화하면 그 칸에 가끔 묻은 잉크가 수십 배로 커져 거리를 좌우한다")

# ---------------------------------------------------------------
# 10) K-Medoids: 대표가 '실제 글씨 한 장'인 K-Means의 사촌
#     (scikit-learn-extra의 KMedoids는 Colab 기본이 아니라 직접 짠다: BUILD 시작 + 배정·교체 반복)
def k_medoids(D, k, max_iter=100):
    med = [int(D.sum(1).argmin())]                      # BUILD: 거리 합이 가장 작은 글씨부터
    dmin = D[med[0]].copy()
    for _ in range(1, k):
        gain = np.maximum(dmin[None, :] - D, 0).sum(1); gain[med] = -1
        med.append(int(gain.argmax())); dmin = np.minimum(dmin, D[med[-1]])
    for it in range(max_iter):
        lab = D[:, med].argmin(1)                       # ① 가장 가까운 대표에 배정
        new = [int(np.where(lab == c)[0][D[np.ix_(lab == c, lab == c)].sum(1).argmin()]) for c in range(k)]  # ② 묶음 안 거리 합이 가장 작은 글씨로 교체
        if new == med: break
        med = new
    return np.array(med), D[:, med].argmin(1)
t0 = time.perf_counter()
med, lab_m = k_medoids(D, 10)
print(f"  K-Medoids k=10: {1000*(time.perf_counter()-t0):.0f}ms · 대표 글씨 번호 {med.tolist()}")
score("K-Medoids k=10", lab_m)
print(f"  대표 10장의 정답: {y[med].tolist()} · 대표에게만 이름표 → {(y[med][lab_m] == y).mean():.1%}")

# ---------------------------------------------------------------
# 11) 판정: 숨긴 정답과의 ARI가 합격선(매뉴얼 씨)과 무작위 기준선을 모두 넘으면 합격
LINE = adjusted_rand_score(y, manual)
A = adjusted_rand_score(y, km.labels_)
print(f"K-Means ARI {A:.3f} · 합격선 {LINE:.3f} · 무작위 {rand_ari:.3f} → {'합격' if A > LINE and A > rand_ari else '보류'}")
