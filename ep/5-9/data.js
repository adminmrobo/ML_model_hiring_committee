/* ══ 5-9 미리 계산한 값 (Python · scikit-learn 1.9.1, 같은 시험지 258건) ══════════
   이 페이지가 브라우저에서 직접 계산하지 않는 숫자만 여기 둔다. A장 Colab 코드 11·12번 칸이 같은 값을 낸다.
   - hgb: HistGradientBoostingRegressor/Classifier(random_state=0), 기본 설정
   - ledger: 5-5 ~ 5-8 지원자의 확정 숫자 (작업지시서 표 = A장 12번 칸, random_state=0)
   - rfImp: 5-7 랜덤 포레스트 회귀(100그루, random_state=0)의 특징 중요도 */
var PRE={
  hgb:{r2:0.938,mae:2.92,acc:0.938,iters:100},
  gbr100:{r2:0.927,mae:3.50},gbc100:{acc:0.919},
  rfImp:{cement:0.344,slag:0.081,fly_ash:0.016,water:0.103,superplasticizer:0.063,coarse_agg:0.028,fine_agg:0.035,age:0.330},
  /* [편, 지원자, 기준 모델, 분류 35MPa, 3등급, 회귀 R², 회귀 MAE, 판정] — null은 하지 않은 과제 */
  ledger:[
    ['5-5','나이브 베이즈','GaussianNB (특징 8개)',0.709,0.581,null,null,'hold'],
    ['5-6','결정 트리','끝까지 키운 나무 (기본 설정)',0.926,null,0.891,3.95,'pass'],
    ['5-7','랜덤 포레스트','나무 100그루',0.934,0.853,0.927,3.37,'pass'],
    ['5-8','SVM','SVC RBF C=10 · SVR RBF C=100 (표준화)',0.880,0.864,0.907,3.78,'pass']],
  /* 앞선 지원자 (참고 줄) */
  earlier:[
    ['5-1','KNN','k = 5, 표준화',0.857,null,0.767,null,'pass'],
    ['5-2','선형 회귀','특징 8개',null,null,0.567,null,'hold'],
    ['5-3','다항 회귀와 규제','3차',null,null,0.849,null,'pass'],
    ['5-4','로지스틱 회귀','C = 1, 표준화',0.864,null,null,null,'pass']]
};
