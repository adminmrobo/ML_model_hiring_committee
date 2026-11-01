/* ══ 5-11 미리 계산한 값 (Python · scikit-learn 1.9.1, 같은 1,797장) ══════════
   이 페이지가 브라우저에서 직접 계산하지 않는 숫자만 여기 둔다. A장 Colab 코드 10·11번 칸이 같은 값을 낸다.
   - km: 5-10 K-Means(k=10, n_init=10, random_state=0, 원래 픽셀). 업무 점수는 이 편과 같은 규칙(묶음 평균에 가장 가까운 한 장)
   - wardStd: 표준화한 뒤 ward 10묶음의 ARI
   - split: 앞 1,500장으로 ward 10묶음을 만들고, 뒤 297장을 가장 가까운 묶음 평균에 배정했을 때의 ARI */
var PRE={
  km:{ari:0.666,nmi:0.742,sil:0.183,prop:0.792},
  wardStd:0.664,
  split:{fit:0.767,assigned:0.637}
};
