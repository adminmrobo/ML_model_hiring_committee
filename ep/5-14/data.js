/* ══ 5-14 미리 계산한 값 (Python · scikit-learn 1.9.1, 손글씨 1,797장 전체) ══════════
   이 페이지가 브라우저에서 직접 계산하지 않는 숫자만 여기 둔다. A장 Colab 코드의 해당 칸이 같은 값을 낸다.
   - team: 차원축소팀 2D 지도 5-NN(5겹). manual = 합격선(A장 2번 칸), tsne = A장 10번 칸, umap = 작업지시서 확인 값(5-15에서 다시 잰다)
   - pre: 군집팀이 PCA 공간 위에서 낸 ARI. gmm30 = A장 9번 칸, kmRaw·kmRep = A장 8번 칸(원래 64칸 K-Means)
   - manualStd: 매뉴얼 씨 지도를 표준화했을 때(A장 2번 칸) */
var PRE={
  team:{manual:0.181,manualStd:0.270,tsne:0.976,umap:0.976,raw:0.963},
  pre:{gmm30:0.836,km2:0.393,km10:0.652,kmRaw:0.666,kmRepRaw:0.792,kmRep10:0.782,kmRep2:0.494,kmPur2:0.607},
  colab:{acc2:0.603,accStd:0.514}
};
