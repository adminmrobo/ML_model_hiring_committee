# ===============================================================
# 모델 채용 위원회 5-11 · 계층적 군집(Hierarchical Clustering) 면접
#  - 라이브러리: numpy, scipy, scikit-learn, matplotlib (모두 Colab 기본). 인터넷이 없어도 돈다
#  - 하는 일: 손글씨 숫자 1,797장을 정답 없이 "가장 가까운 둘부터 합치기"로 묶어 가계도(덴드로그램)를 만들고,
#             연결 방식 네 가지와 자르는 높이를 비교한 뒤, 채점 봉투(숨긴 정답)를 열어 합격선과 비교한다
#  - 규칙: 정답 y는 학습(합치기)에 쓰지 않는다. y는 "채점"이라고 적힌 칸에서만 쓴다
# ===============================================================
import time
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, cophenet
from scipy.spatial.distance import pdist
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

np.set_printoptions(precision=3, suppress=True)
print("scikit-learn", sklearn.__version__)

# ---------------------------------------------------------------
# 1) 실기 과제지: 손글씨 숫자 1,797장 (8×8 = 64칸, 밝기 0~16). scikit-learn에 들어 있어 내려받지 않는다
#    [내 데이터로 바꾸려면] X만 바꾼다. 한 줄에 글씨(문서) 하나, 열은 같은 길이의 숫자들
X, y = load_digits(return_X_y=True)
print(f"글씨 {X.shape[0]}장 · 특징(픽셀) {X.shape[1]}개 · 밝기 {X.min():.0f}~{X.max():.0f}")
#    y(정답 0~9)는 "채점 봉투"에 넣어 둔다. 아래 합치기는 X만 쓴다
n = len(X)
print(f"서로 다른 두 장의 쌍: {n * (n - 1) // 2:,}개 (모두 거리를 잰다)")

# ---------------------------------------------------------------
# 2) 합격선: 매뉴얼 씨의 정리 규칙 (가운데가 비었나 · 잉크가 많나 · 위가 무거운가 → 8묶음) [채점]
ink = X.sum(axis=1)
top = X[:, :32].sum(axis=1) / np.maximum(ink, 1)
center = X[:, [27, 28, 35, 36]].sum(axis=1)
manual = (center < 12) * 4 + (ink > 300) * 2 + (top > 0.5) * 1
LINE = adjusted_rand_score(y, manual)
rng = np.random.RandomState(0)
rand = np.mean([adjusted_rand_score(y, rng.randint(0, 10, n)) for _ in range(20)])
print(f"매뉴얼 씨 8묶음 ARI {LINE:.3f} (NMI {normalized_mutual_info_score(y, manual):.3f}) = 합격선 · 무작위 10묶음 ARI {rand:.3f}")

# ---------------------------------------------------------------
# 3) 직접 짜 보기: 가장 가까운 두 묶음을 찾아 합치고, 새 묶음과 나머지의 거리를 고쳐 쓴다 (ward 연결)
#    느린 O(n³) 판이라 앞 100장만 쓴다. scikit-learn과 같은 묶음이 나오는지 본다
def my_ward(Xs, k):
    groups = [[i] for i in range(len(Xs))]
    while len(groups) > k:
        best = None
        for a in range(len(groups)):
            for b in range(a + 1, len(groups)):
                A, B = Xs[groups[a]], Xs[groups[b]]
                # ward: 둘을 합치면 묶음 안 제곱 거리 합(관성)이 얼마나 늘어나나
                cost = len(A) * len(B) / (len(A) + len(B)) * ((A.mean(0) - B.mean(0)) ** 2).sum()
                if best is None or cost < best[0]:
                    best = (cost, a, b)
        _, a, b = best
        groups[a] += groups[b]
        del groups[b]
    lab = np.zeros(len(Xs), int)
    for g, mem in enumerate(groups):
        lab[mem] = g
    return lab

mine = my_ward(X[:100], 10)
lib = AgglomerativeClustering(n_clusters=10, linkage="ward").fit_predict(X[:100])
print(f"앞 100장, 10묶음: 직접 짠 것과 scikit-learn의 일치도(ARI) {adjusted_rand_score(lib, mine):.3f}  (1이면 같은 묶음)")

# ---------------------------------------------------------------
# 4) 본 면접: 1,797장 전체, ward 연결, 10묶음 (숫자는 0~9 열 가지라는 업무 지식으로 10을 정했다)
t0 = time.perf_counter()
hc = AgglomerativeClustering(n_clusters=10, linkage="ward").fit(X)
print(f"학습 {1000 * (time.perf_counter() - t0):.0f}ms · 묶음 크기 {sorted(np.bincount(hc.labels_).tolist(), reverse=True)}")
print("predict가 있나?", hasattr(hc, "predict"), "· fit_predict가 있나?", hasattr(hc, "fit_predict"))

# ---------------------------------------------------------------
# 5) 연결 방식 네 가지: 묶음과 묶음 사이 거리를 어떻게 재나 (ARI·NMI는 채점, 실루엣은 정답 없이)
rows = []
for how in ["ward", "average", "complete", "single"]:
    lab = AgglomerativeClustering(n_clusters=10, linkage=how).fit_predict(X)
    rows.append([how, adjusted_rand_score(y, lab), normalized_mutual_info_score(y, lab),
                 silhouette_score(X, lab), np.bincount(lab).max()])
print(pd.DataFrame(rows, columns=["연결", "ARI", "NMI", "실루엣", "가장 큰 묶음"]).round(3).to_string(index=False))
#    single은 "가장 가까운 한 쌍"만 보므로 사슬처럼 이어져 거의 전부가 한 묶음이 된다(사슬 효과)

# ---------------------------------------------------------------
# 6) 덴드로그램(가계도): scipy linkage로 합친 기록 1,796줄을 만든다. 마지막 30갈래만 그린다
Z = linkage(X, method="ward")
print("마지막 다섯 번 합친 높이:", Z[-5:, 2])
print(f"코페네틱 상관 {cophenet(Z, pdist(X))[0]:.3f} (가계도의 높이가 원래 거리를 얼마나 닮았나)")
plt.figure(figsize=(10, 4))
dendrogram(Z, truncate_mode="lastp", p=30, no_labels=False)
plt.axhline(Z[-10, 2] + 1, color="red", ls="--")   # 이 높이에서 자르면 10묶음
plt.title("ward 덴드로그램 (마지막 30갈래)")
plt.show()

# ---------------------------------------------------------------
# 7) 자르는 높이 = 묶음 수. 정답 없이 재는 실루엣과, 채점 봉투로 재는 ARI를 나란히 [ARI는 채점]
print("  묶음  실루엣  ARI")
for k in (2, 5, 8, 9, 10, 12, 15, 20, 30):
    lab = AgglomerativeClustering(n_clusters=k).fit_predict(X)
    print(f"  {k:>4}  {silhouette_score(X, lab):.3f}   {adjusted_rand_score(y, lab):.3f}")
for th in (300, 400):
    m = AgglomerativeClustering(n_clusters=None, distance_threshold=th).fit(X)
    print(f"  높이 {th}에서 자르기(distance_threshold) → {m.n_clusters_}묶음")

# ---------------------------------------------------------------
# 8) 채점 봉투 열기: 묶음 × 정답 표 [채점]
table = pd.crosstab(hc.labels_, y, rownames=["묶음"], colnames=["정답"])
print(table.to_string())

# ---------------------------------------------------------------
# 9) 업무 점수: 묶음마다 평균에 가장 가까운 한 장(대표)에만 사람이 이름을 붙이고, 그 이름을 묶음 전체에 붙인다 [채점]
def propagate(lab):
    ok = 0
    for c in np.unique(lab):
        idx = np.where(lab == c)[0]
        rep = idx[np.argmin(((X[idx] - X[idx].mean(0)) ** 2).sum(1))]   # 대표 한 장
        ok += (y[idx] == y[rep]).sum()                                  # 대표의 이름을 묶음 전체에
    return ok / n

print(f"  매뉴얼 씨 8묶음: 이름 8장 → 맞게 붙은 비율 {propagate(manual):.3f}")
for k in (10, 12, 20, 50):
    print(f"  ward {k:>2}묶음: 이름 {k:>2}장 → 맞게 붙은 비율 {propagate(AgglomerativeClustering(n_clusters=k).fit_predict(X)):.3f}")

# ---------------------------------------------------------------
# 10) 예측이 없다: 새 글씨가 오면? 앞 1,500장으로 묶고, 뒤 297장은 "가장 가까운 묶음 평균"으로 배정해 본다
part = AgglomerativeClustering(n_clusters=10).fit_predict(X[:1500])
cent = np.array([X[:1500][part == c].mean(0) for c in range(10)])
new = np.argmin(((X[1500:, None, :] - cent[None]) ** 2).sum(2), axis=1)
print(f"  앞 1,500장 묶음 ARI {adjusted_rand_score(y[:1500], part):.3f} · 새 297장 배정 ARI {adjusted_rand_score(y[1500:], new):.3f} [채점]")
#    이 배정은 계층적 군집이 아니라 우리가 덧붙인 규칙(K-Means식)이다. 새 글씨를 넣어 다시 합치면 가계도가 바뀔 수 있다

# ---------------------------------------------------------------
# 11) 비교: 표준화하면? 그리고 5-10 K-Means(k=10)와 [채점]
ward_std = AgglomerativeClustering(n_clusters=10).fit_predict(StandardScaler().fit_transform(X))
km = KMeans(n_clusters=10, n_init=10, random_state=0).fit(X)
print(f"  ward 표준화 ARI {adjusted_rand_score(y, ward_std):.3f} (원래 픽셀 {adjusted_rand_score(y, hc.labels_):.3f})")
print(f"  K-Means k=10 ARI {adjusted_rand_score(y, km.labels_):.3f} · NMI {normalized_mutual_info_score(y, km.labels_):.3f} · "
      f"실루엣 {silhouette_score(X, km.labels_):.3f} · 업무 점수 {propagate(km.labels_):.3f}")

# ---------------------------------------------------------------
# 12) 판정: ARI가 합격선(매뉴얼 씨)과 무작위 기준선을 모두 넘으면 합격 [채점]
score = adjusted_rand_score(y, hc.labels_)
print(f"ward 10묶음 ARI {score:.3f} (합격선 {LINE:.3f}, 무작위 {rand:.3f})")
print("판정:", "합격" if score > LINE and score > rand else "보류")
