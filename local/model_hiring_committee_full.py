#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
 모델 채용 위원회 — 머신러닝 열다섯 명의 입사 면접 · 로컬 풀버전 코드
===============================================================================
 이 파일 하나로 시리즈 16편(5-0 ~ 5-15)의 "면접 판정 숫자"를 내 컴퓨터에서
 그대로 다시 얻는다. 각 편 페이지의 Colab 코드(A장)에서 판정에 쓰인 설정만
 모아, 같은 데이터 · 같은 시험지 나누기 · 같은 난수로 돌린다.

 ■ 결과로 나와야 하는 값 (시리즈 최종본과 같아야 한다)
   합격선(기존 직원 매뉴얼 씨) : 분류 76.7% · 회귀 R² 0.735 · 군집 ARI 0.108 · 2D 지도 5-NN 18.1%
   5-1 KNN 85.7% / R² 0.767      5-2 선형 회귀 R² 0.567(보류)   5-3 3차 다항 R² 0.849
   5-4 로지스틱 86.4%            5-5 나이브 베이즈 70.9%(보류)   5-6 결정 트리 92.6% / R² 0.891
   5-7 랜덤 포레스트 93.4% / 0.927   5-8 SVM 88.0% / 0.907        5-9 GBM 93.4% / 0.949
   5-10 K-Means ARI 0.666        5-11 계층(ward) 0.794          5-12 DBSCAN 0.001(보류)
   5-13 GMM 0.836                5-14 PCA 60.3%                 5-15 t-SNE 97.6% · UMAP 97.6%
   → 지원자 15명 중 합격 12 · 보류 3

 ■ 준비 (한 번만)
   pip install numpy pandas scikit-learn umap-learn ucimlrepo
   - 확인한 버전: Python 3.13 · scikit-learn 1.9.1 · numpy 2.5.3 · pandas 3.0.5 · umap-learn 0.5.12
   - scikit-learn 1.3 이상이면 돌아간다(HDBSCAN이 1.3에 들어왔다). 버전이 다르면
     소수 셋째 자리가 조금 달라질 수 있다(특히 t-SNE · UMAP · GMM).
   - umap-learn이 없으면 5-15의 UMAP만 건너뛴다(t-SNE로 판정).

 ■ 데이터
   지도 본부(5-1 ~ 5-9) : UCI Concrete Compressive Strength (id=165, Yeh 1998, CC BY 4.0), 1,030건
     아래 순서로 찾는다. 하나만 있으면 된다.
       ① 이 파일과 같은 폴더(또는 ../data/)의 concrete.csv  (키트에 들어 있는 사본)
       ② 같은 폴더의 Concrete_Data.xls  (UCI에서 내려받은 원본, pip install xlrd 필요)
       ③ 인터넷으로 ucimlrepo.fetch_ucirepo(id=165)
   비지도 본부(5-10 ~ 5-15) : scikit-learn 내장 손글씨 숫자 load_digits (1,797장, 인터넷 필요 없음)

 ■ 실행
   python3 model_hiring_committee_full.py            전체 (보통 1분 안팎. t-SNE · UMAP이 가장 오래 걸리고, UMAP 첫 실행은 컴파일 시간이 더해진다)
   python3 model_hiring_committee_full.py --quick    t-SNE · UMAP을 건너뛴다(5-15는 "건너뜀")
   python3 model_hiring_committee_full.py --only 5-4 5-10    고른 편만 (합격선은 늘 먼저 계산한다)

 ■ 위원회 규칙 (모든 편 공통)
   1) 시험지 봉인: train_test_split(test_size=0.25, stratify=강도 3등급, random_state=42)
      → 훈련 772건 · 시험 258건. 시험지는 마지막 채점에 한 번만 쓴다.
   2) 표준화가 필요한 모델은 훈련 데이터로만 맞추고(fit), 시험지는 바꾸기만(transform) 한다.
   3) 비지도 본부는 정답을 "채점 봉투"에 봉인한다. 학습·설정 고르기에는 정답을 쓰지 않고, 채점에만 쓴다.
   4) 판정: 점수가 합격선(매뉴얼 씨)과 기준선(아무 생각 없이 답하기)을 모두 넘으면 합격, 아니면 보류.
      겸용팀(5-6 ~ 5-9)은 분류와 회귀를 둘 다 넘어야 합격이다.

 글 · 안상선 ((주)M-Robo 대표) · CC BY-NC 4.0 · 「모델 채용 위원회」 시리즈의 부록 코드
===============================================================================
"""
import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import r2_score, mean_absolute_error, adjusted_rand_score, silhouette_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, NearestNeighbors
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                              GradientBoostingClassifier, GradientBoostingRegressor)
from sklearn.svm import SVC, SVR
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, HDBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

warnings.filterwarnings("ignore", category=FutureWarning)   # 버전 안내 경고는 결과와 무관하므로 숨긴다
warnings.filterwarnings("ignore", message=".*n_jobs value.*")   # UMAP: 난수를 고정하면 병렬을 끈다는 안내

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = []          # 결산표에 쓸 줄: (편, 지원자, 팀, 핵심 숫자, 판정)


# =============================================================================
# 0. 작은 도구들
# =============================================================================
def title(text):
    """편 제목 줄을 찍는다."""
    print("\n" + "=" * 78 + f"\n {text}\n" + "=" * 78)


def verdict(ok):
    """판정 글자: 합격선과 기준선을 모두 넘었으면 '합격', 아니면 '보류'."""
    return "합격" if ok else "보류"


def record(ep, who, team, score, ok, note=""):
    """결산표에 한 줄을 남기고, 판정을 화면에 찍는다."""
    RESULTS.append((ep, who, team, score, verdict(ok), note))
    print(f"→ 판정: {verdict(ok)}  ({score}){'  · ' + note if note else ''}")


# =============================================================================
# 1. 지도 본부 데이터: 콘크리트 배합 1,030건
# =============================================================================
FEATURES = ["cement", "slag", "fly_ash", "water", "superplasticizer", "coarse_agg", "fine_agg", "age"]
#            시멘트 · 고로 슬래그 · 플라이 애시 · 물 · 고성능 감수제 · 굵은 골재 · 잔골재 (kg/m³) · 재령(일)
PASS_MPA = 35        # 수업용 합격 기준: 압축 강도 35MPa 이상이면 "합격 배합" (설계 기준이 아니다)


def load_concrete():
    """콘크리트 데이터를 찾아 읽는다. 열 이름을 짧게 바꾸고 (1030, 9)인지 확인한다."""
    names = FEATURES + ["strength"]
    for p in [os.path.join(HERE, "concrete.csv"), os.path.join(HERE, "..", "data", "concrete.csv")]:
        if os.path.exists(p):
            print("데이터: 로컬 사본", os.path.relpath(p))
            df = pd.read_csv(p)
            break
    else:
        xls = os.path.join(HERE, "Concrete_Data.xls")
        if os.path.exists(xls):
            print("데이터: UCI 원본 엑셀", xls)
            df = pd.read_excel(xls)
        else:
            print("데이터: 인터넷 ucimlrepo (id=165)")
            from ucimlrepo import fetch_ucirepo
            uci = fetch_ucirepo(id=165)
            df = pd.concat([uci.data.features, uci.data.targets], axis=1)
    df.columns = names                      # 원본의 긴 열 이름을 열 순서대로 짧은 이름으로
    assert df.shape == (1030, 9), df.shape
    return df


def prepare_supervised():
    """데이터를 읽고, 위원회 규정대로 시험지를 봉인하고, 매뉴얼 씨의 합격선 두 개를 계산한다."""
    title("공통 준비 · 지도 본부 실기 과제지 (콘크리트 1,030건)")
    df = load_concrete()
    X = df[FEATURES].to_numpy()
    y = df["strength"].to_numpy()                       # 회귀 타겟: 압축 강도(MPa)
    passed = (y >= PASS_MPA).astype(int)                # 분류 타겟: 35MPa 이상 1, 미만 0
    grade = np.digitize(y, [25, 45])                    # 강도 3등급: 0 저강도(<25) · 1 보통 · 2 고강도(≥45)

    # --- 시험지 봉인: 25%를 시험지로. 3등급 비율을 훈련/시험에 똑같이 나눈다(stratify). 난수 42로 고정
    idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=0.25, stratify=grade, random_state=42)
    print(f"배합 {len(df)}건 → 훈련 {len(idx_tr)}건 · 시험지 {len(idx_te)}건 (봉인)")

    # --- 사람이 아는 두 숫자: 물-결합재비(물 ÷ 결합재)와 재령
    #     결합재 = 시멘트 + 고로 슬래그 + 플라이 애시 (물과 반응해 굳는 가루)
    wb = (df["water"] / (df["cement"] + df["slag"] + df["fly_ash"])).to_numpy()
    age = df["age"].to_numpy()

    # --- 매뉴얼 씨의 분류 규칙 (5-0 작업 표준서 제1조): 물-결합재비 < 0.5 이고 재령 ≥ 28일이면 합격
    rule = ((wb < 0.5) & (age >= 28)).astype(int)
    line_acc = (rule[idx_te] == passed[idx_te]).mean()

    # --- 매뉴얼 씨의 회귀 경험식 (학습 없음, 교과서 숫자 네 개)
    #     28일 강도 = 96.5 / 7^(물-결합재비) MPa            (에이브럼스 식, Abrams 1918: 원식 14,000 psi ÷ 7^x)
    #     재령 t일 강도 = 28일 강도 × t / (4.0 + 0.85t)     (ACI 209R-92 표 2.2.1, 습윤 양생 1종 시멘트)
    manual_mpa = 96.5 / 7 ** wb * age / (4 + 0.85 * age)
    line_r2 = r2_score(y[idx_te], manual_mpa[idx_te])

    print(f"분류 합격선 · 매뉴얼 씨 규칙  {(rule[idx_te] == passed[idx_te]).sum()}/{len(idx_te)} = {line_acc:.3f}")
    print(f"회귀 합격선 · 매뉴얼 경험식   R² {line_r2:.3f} (MAE {mean_absolute_error(y[idx_te], manual_mpa[idx_te]):.2f}MPa)")

    d = dict(df=df, X=X, y=y, passed=passed, grade=grade, idx_tr=idx_tr, idx_te=idx_te,
             wb=wb, age=age, rule=rule, LINE_ACC=line_acc, LINE_R2=line_r2)
    # 자주 쓰는 조각을 미리 잘라 둔다
    d["X_tr"], d["X_te"] = X[idx_tr], X[idx_te]
    d["y_tr"], d["y_te"] = y[idx_tr], y[idx_te]
    d["p_tr"], d["p_te"] = passed[idx_tr], passed[idx_te]
    # 표준화(훈련 데이터로만 맞춤) — 거리 · 경사 하강 · 커널을 쓰는 모델에 필요하다
    ss = StandardScaler().fit(d["X_tr"])
    d["A"], d["B"] = ss.transform(d["X_tr"]), ss.transform(d["X_te"])
    # 기준선: 분류는 "늘 다수 쪽(불합격)", 회귀는 "늘 훈련 평균"
    d["BASE_ACC"] = DummyClassifier(strategy="most_frequent").fit(d["A"], d["p_tr"]).score(d["B"], d["p_te"])
    d["BASE_R2"] = DummyRegressor().fit(d["A"], d["y_tr"]).score(d["B"], d["y_te"])
    print(f"기준선 · 분류(늘 불합격) {d['BASE_ACC']:.3f} · 회귀(늘 평균) R² {d['BASE_R2']:.3f}")
    return d


def ep_5_0(d):
    title("5-0 채용 공고 · 기존 직원 매뉴얼 씨 (룰 베이스)")
    # 매뉴얼 씨는 지원자가 아니라 기준이다. 그의 시험지 점수가 그대로 첫 합격선이 된다.
    # 1차 판정(35MPa)은 계속 맡지만, 등급 판정과 새 배합(혼화재 · 짧은 재령)은 맡기 어려워 '보류'.
    tr_acc = (d["rule"][d["idx_tr"]] == d["p_tr"]).mean()
    print(f"훈련 {tr_acc:.3f} · 시험지 {d['LINE_ACC']:.3f}  (규칙이 두 줄뿐이라 외울 틈이 없다 → 두 점수가 비슷)")
    RESULTS.append(("5-0", "매뉴얼 씨 (룰 베이스)", "기존 직원",
                    f"분류 {d['LINE_ACC']:.1%} · 회귀 R² {d['LINE_R2']:.3f}", "보류", "합격선의 출처"))
    print("→ 판정: 보류 (합격선의 출처)")


# =============================================================================
# 2. 지도 본부 지원자 (5-1 ~ 5-9)
# =============================================================================
def ep_5_1(d):
    title("5-1 KNN · 입문팀 (분류 + 회귀) — 가까운 배합 k개에게 묻는다")
    # 단위가 다른 특징(kg vs 일)을 그대로 재면 큰 숫자가 거리를 독차지한다 → 표준화한 A, B를 쓴다.
    knr = KNeighborsRegressor(n_neighbors=5).fit(d["A"], d["y_tr"])      # 회귀: 이웃 5개 강도의 평균
    r2 = knr.score(d["B"], d["y_te"])
    raw = KNeighborsRegressor(n_neighbors=5).fit(d["X_tr"], d["y_tr"]).score(d["X_te"], d["y_te"])
    knc = KNeighborsClassifier(n_neighbors=5).fit(d["A"], d["p_tr"])     # 분류: 이웃 5개의 다수결
    acc = knc.score(d["B"], d["p_te"])
    print(f"회귀 R² {r2:.3f} (표준화 안 하면 {raw:.3f}) · 분류 {acc:.3f}")
    # 입문팀의 정식 과제는 회귀(강도 맞히기). 분류도 합격선을 넘는지 함께 본다.
    ok = r2 > d["LINE_R2"] and r2 > d["BASE_R2"]
    record("5-1", "KNN", "입문", f"분류 {acc:.1%} · 회귀 R² {r2:.3f}", ok, f"표준화를 빼면 R² {raw:.3f}")


def ep_5_2(d):
    title("5-2 선형 회귀 · 회귀팀 — 오차 제곱의 합이 가장 작은 직선")
    lin = LinearRegression().fit(d["A"], d["y_tr"])        # 특징 8개 그대로(표준화). 규칙 없는 최소제곱
    r2 = r2_score(d["y_te"], lin.predict(d["B"]))
    # 직선은 "처음 확 오르고 나중엔 느린" 재령 곡선을 못 따라간다. 사람이 아는 특징 두 개를 더하면?
    Z = np.column_stack([d["X"], d["wb"], np.log(d["age"])])  # 8개 + 물-결합재비 + log(재령)
    s = StandardScaler().fit(Z[d["idx_tr"]])
    r2_good = LinearRegression().fit(s.transform(Z[d["idx_tr"]]), d["y_tr"]).score(s.transform(Z[d["idx_te"]]), d["y_te"])
    print(f"특징 8개 R² {r2:.3f} · 좋은 특징 2개를 더하면 R² {r2_good:.3f}  (합격선 {d['LINE_R2']:.3f})")
    ok = r2 > d["LINE_R2"] and r2 > d["BASE_R2"]
    record("5-2", "선형 회귀", "회귀", f"R² {r2:.3f}", ok, f"좋은 특징을 넣으면 {r2_good:.3f}")


def ep_5_3(d):
    title("5-3 다항 회귀와 규제 · 회귀팀 — x², x³을 붙여 곡선을 긋는다")
    # 표준화 → 3차 다항 특징(164개) → 선형 회귀. 모델은 그대로, 재료만 거듭제곱한다.
    poly = {}
    for deg in (1, 2, 3, 4):
        poly[deg] = make_pipeline(StandardScaler(), PolynomialFeatures(deg, include_bias=False),
                                  LinearRegression()).fit(d["X_tr"], d["y_tr"])
        tr = poly[deg].score(d["X_tr"], d["y_tr"])
        te = poly[deg].score(d["X_te"], d["y_te"])
        print(f"  {deg}차: 훈련 R² {tr:.3f} · 시험 R² {te:8.3f}{'   ← 과대적합(훈련은 최고, 시험지는 폭주)' if deg == 4 else ''}")
    r2 = poly[3].score(d["X_te"], d["y_te"])
    ok = r2 > d["LINE_R2"] and r2 > d["BASE_R2"]
    record("5-3", "다항 회귀와 규제", "회귀", f"3차 R² {r2:.3f}", ok, "4차는 규제(릿지·라쏘) 없이는 보류")


def ep_5_4(d):
    title("5-4 로지스틱 회귀 · 분류팀 — 선형 점수에 시그모이드를 씌운 확률")
    lr = LogisticRegression(C=1, max_iter=1000).fit(d["A"], d["p_tr"])
    acc = lr.score(d["B"], d["p_te"])
    print(f"시험 정확도 {acc:.3f} (합격선 {d['LINE_ACC']:.3f} · 기준선 {d['BASE_ACC']:.3f}) · 외울 숫자 {lr.coef_.size + 1}개")
    ok = acc > d["LINE_ACC"] and acc > d["BASE_ACC"]
    record("5-4", "로지스틱 회귀", "분류", f"{acc:.1%}", ok)


def ep_5_5(d):
    title("5-5 나이브 베이즈 · 분류팀 — 특징끼리 독립이라고 가정하고 확률을 곱한다")
    # 특징마다 따로 평균 · 분산을 재므로 표준화가 필요 없다 → 원래 단위 X_tr를 쓴다.
    nb = GaussianNB().fit(d["X_tr"], d["p_tr"])
    acc = nb.score(d["X_te"], d["p_te"])
    # 물과 감수제처럼 서로 얽힌 특징을 '독립'으로 본 대가. 사람이 아는 두 특징을 더하면?
    Z = np.column_stack([d["X"], d["wb"], np.log10(d["age"])])
    acc_good = GaussianNB().fit(Z[d["idx_tr"]], d["p_tr"]).score(Z[d["idx_te"]], d["p_te"])
    print(f"시험 정확도 {acc:.3f} · 좋은 특징을 넣으면 {acc_good:.3f}  (합격선 {d['LINE_ACC']:.3f})")
    ok = acc > d["LINE_ACC"] and acc > d["BASE_ACC"]
    record("5-5", "나이브 베이즈", "분류", f"{acc:.1%}", ok, f"좋은 특징을 넣으면 {acc_good:.1%}")


def ep_5_6(d):
    title("5-6 결정 트리 · 겸용팀 — 질문 한 줄씩 스무고개 (분류 + 회귀)")
    # 나무는 "문턱보다 큰가"만 물으므로 표준화가 필요 없다. 기본 설정 = 끝까지 키운 나무.
    clf = DecisionTreeClassifier(random_state=0).fit(d["X_tr"], d["p_tr"])
    reg = DecisionTreeRegressor(random_state=0).fit(d["X_tr"], d["y_tr"])
    acc = clf.score(d["X_te"], d["p_te"])
    r2 = r2_score(d["y_te"], reg.predict(d["X_te"]))
    print(f"분류 {acc:.3f} (잎 {clf.get_n_leaves()}개) · 회귀 R² {r2:.3f} (깊이 {reg.get_depth()}, 잎 {reg.get_n_leaves()}개)")
    ok = acc > d["LINE_ACC"] and acc > d["BASE_ACC"] and r2 > d["LINE_R2"]
    record("5-6", "결정 트리", "겸용", f"{acc:.1%} · R² {r2:.3f}", ok, "브라우저 계산은 93.4%(같은 점수 질문 고르기 차이)")


def ep_5_7(d):
    title("5-7 랜덤 포레스트 · 겸용팀 — 무작위로 다르게 키운 나무 100그루의 다수결")
    rf = RandomForestClassifier(n_estimators=100, random_state=0, oob_score=True).fit(d["X_tr"], d["p_tr"])
    reg = RandomForestRegressor(n_estimators=100, random_state=0, oob_score=True).fit(d["X_tr"], d["y_tr"])
    acc, r2 = rf.score(d["X_te"], d["p_te"]), reg.score(d["X_te"], d["y_te"])
    print(f"분류 {acc:.3f} (OOB {rf.oob_score_:.3f}, 시험지 없이 미리 본 점수) · 회귀 R² {r2:.3f} (OOB {reg.oob_score_:.3f})")
    ok = acc > d["LINE_ACC"] and acc > d["BASE_ACC"] and r2 > d["LINE_R2"] and r2 > d["BASE_R2"]
    record("5-7", "랜덤 포레스트", "겸용", f"{acc:.1%} · R² {r2:.3f}", ok)


def ep_5_8(d):
    title("5-8 SVM · 겸용팀 — 가장 넓은 길, 그리고 커널로 공간 휘기 (SVC + SVR)")
    # C(규제 손잡이)는 시험지가 아니라 훈련 데이터 안의 5겹 교차 검증으로 고른다(규칙 1).
    cv_c = StratifiedKFold(5, shuffle=True, random_state=42)
    cv = {C: cross_val_score(SVC(C=C), d["A"], d["p_tr"], cv=cv_c).mean() for C in (0.1, 1, 10, 100)}
    C_cls = max(cv, key=cv.get)
    acc = SVC(C=C_cls).fit(d["A"], d["p_tr"]).score(d["B"], d["p_te"])
    cv_r = KFold(5, shuffle=True, random_state=42)
    cvr = {C: cross_val_score(SVR(C=C), d["A"], d["y_tr"], cv=cv_r).mean() for C in (1, 10, 100)}
    C_reg = max(cvr, key=cvr.get)
    r2 = r2_score(d["y_te"], SVR(C=C_reg).fit(d["A"], d["y_tr"]).predict(d["B"]))
    print(f"교차 검증으로 고른 C: 분류 {C_cls} · 회귀 {C_reg}  →  분류 {acc:.3f} · 회귀 R² {r2:.3f}")
    ok = acc > d["LINE_ACC"] and acc > d["BASE_ACC"] and r2 > d["LINE_R2"]
    record("5-8", "SVM", "겸용", f"{acc:.1%} · R² {r2:.3f}", ok, f"SVC RBF C={C_cls} · SVR C={C_reg}")


def ep_5_9(d):
    title("5-9 GBM · 겸용팀 — 앞 나무의 오답(잔차)을 다음 나무가 고친다")
    gbr = GradientBoostingRegressor(n_estimators=300, learning_rate=0.1, max_depth=3, random_state=0).fit(d["X_tr"], d["y_tr"])
    gbc = GradientBoostingClassifier(n_estimators=300, learning_rate=0.1, max_depth=3, random_state=0).fit(d["X_tr"], d["p_tr"])
    r2, acc = gbr.score(d["X_te"], d["y_te"]), gbc.score(d["X_te"], d["p_te"])
    print(f"나무 300그루 · 학습률 0.1 · 깊이 3 → 분류 {acc:.3f} · 회귀 R² {r2:.3f}")
    ok = r2 > d["LINE_R2"] and acc > d["LINE_ACC"] and acc > d["BASE_ACC"]
    record("5-9", "GBM", "겸용", f"{acc:.1%} · R² {r2:.3f}", ok)


# =============================================================================
# 3. 비지도 본부 데이터: 손글씨 숫자 1,797장
# =============================================================================
def prepare_unsupervised():
    title("공통 준비 · 비지도 본부 실기 과제지 (손글씨 숫자 1,797장, 8×8 픽셀)")
    X, envelope = load_digits(return_X_y=True)   # envelope = 채점 봉투(숨긴 정답 0~9). 채점에만 쓴다
    print(f"글씨 {X.shape[0]}장 · 특징 {X.shape[1]}칸(픽셀, 밝기 0~16) · 시험지를 떼지 않고 전체로 학습")

    # --- 매뉴얼 씨의 정리 규칙: 사람이 정한 세 질문 → 2×2×2 = 8묶음 (정답을 쓰지 않는다)
    ink = X.sum(1)                                   # 잉크 양(밝기 합)
    top = X[:, :32].sum(1) / np.maximum(ink, 1)      # 위쪽 절반(앞 네 줄)에 잉크가 몇 %인가
    center = X[:, [27, 28, 35, 36]].sum(1)           # 가운데 네 칸의 밝기 (0처럼 가운데가 빈 숫자를 잡는다)
    manual = (center < 12) * 4 + (ink > 300) * 2 + (top > 0.5) * 1
    clu_line = adjusted_rand_score(envelope, manual)

    # --- 매뉴얼 씨의 2D 지도: 위의 두 숫자(잉크 양, 위쪽 비율)를 그대로 좌표로 쓴다
    def map_score(Z):
        """지도 채점: 지도 위에서 가까운 5장의 정답으로 맞히기(5-NN), 5겹 교차 검증. [채점 봉투 사용]"""
        return cross_val_score(KNeighborsClassifier(n_neighbors=5), Z, envelope, cv=5).mean()
    dr_line = map_score(np.c_[ink, top])
    base_map = np.bincount(envelope).max() / len(envelope)   # 기준선: 늘 가장 흔한 숫자만 답하기

    rng = np.random.RandomState(0)                    # 무작위 기준선: 아무렇게나 10묶음
    rand_ari = np.mean([adjusted_rand_score(envelope, rng.randint(0, 10, len(X))) for _ in range(20)])
    print(f"군집 합격선 · 매뉴얼 씨 8묶음 ARI {clu_line:.3f} (무작위 {rand_ari:.3f})")
    print(f"차원축소 합격선 · 매뉴얼 씨 2D 지도 5-NN {dr_line:.3f} (기준선 {base_map:.3f})")
    return dict(X=X, y=envelope, manual=manual, CLU_LINE=clu_line, DR_LINE=dr_line,
                RAND=rand_ari, BASE_MAP=base_map, map_score=map_score)


def ep_5_10(u):
    title("5-10 K-Means · 군집팀 — 중심 10개를 찍고, 가까운 쪽으로 배정, 평균으로 이동")
    # n_init 기본값은 버전마다 달라서(1.4부터 'auto') 10을 적어 둔다. 원래 픽셀 그대로(표준화 안 함).
    km = KMeans(n_clusters=10, n_init=10, random_state=0).fit(u["X"])
    sil = silhouette_score(u["X"], km.labels_)                  # 정답 없이 재는 점수
    ari = adjusted_rand_score(u["y"], km.labels_)               # [채점] 숨긴 정답과 겹치는 정도
    print(f"관성 {km.inertia_:,.0f} · 실루엣 {sil:.3f} · ARI {ari:.3f}")
    ok = ari > u["CLU_LINE"] and ari > u["RAND"]
    record("5-10", "K-Means", "군집", f"ARI {ari:.3f} · 실루엣 {sil:.3f}", ok)


def ep_5_11(u):
    title("5-11 계층적 군집 · 군집팀 — 가장 가까운 둘부터 합쳐 가계도(덴드로그램)를 만든다")
    scores = {}
    for how in ("ward", "average", "complete", "single"):
        lab = AgglomerativeClustering(n_clusters=10, linkage=how).fit_predict(u["X"])
        scores[how] = adjusted_rand_score(u["y"], lab)
        print(f"  {how:<8} 10묶음 ARI {scores[how]:.3f}{'   ← 사슬 효과: 거의 한 묶음' if how == 'single' else ''}")
    ari = scores["ward"]
    ok = ari > u["CLU_LINE"] and ari > u["RAND"]
    record("5-11", "계층적 군집", "군집", f"ARI {ari:.3f}", ok, f"single 연결이면 {scores['single']:.3f}")


def knee_eps(X, min_samples):
    """정답 없이 eps 고르기: 점마다 min_samples번째 이웃(자기 자신 포함)까지 거리를 재 작은 순으로 늘어놓고,
       양 끝을 잇는 직선에서 가장 멀리 떨어진 점(= 곡선의 '무릎')의 거리를 eps로 쓴다."""
    dist, _ = NearestNeighbors(n_neighbors=min_samples).fit(X).kneighbors(X)
    k = np.sort(dist[:, min_samples - 1])
    x = np.arange(len(k)) / (len(k) - 1)
    yy = (k - k[0]) / (k[-1] - k[0])
    return round(float(k[np.argmax(np.abs(yy - x))]), 2)


def ep_5_12(u):
    title("5-12 DBSCAN · 군집팀 — 촘촘한 동네만 묶고, 외톨이는 잡음")
    MS = 5                                           # scikit-learn 기본 min_samples
    EPS = knee_eps(u["X"], MS)                       # 정답을 보지 않고 고른 eps (k-거리 그래프의 무릎)
    lab = DBSCAN(eps=EPS, min_samples=MS).fit_predict(u["X"])
    n_cl, noise = lab.max() + 1, (lab == -1).sum()
    biggest = np.bincount(lab[lab >= 0]).max() / len(lab) if n_cl > 0 else 0
    ari = adjusted_rand_score(u["y"], lab)
    hdb = adjusted_rand_score(u["y"], HDBSCAN(min_cluster_size=5, copy=True).fit_predict(u["X"]))
    print(f"eps {EPS} (무릎) · 묶음 {n_cl}개 · 잡음 {noise}장 · 가장 큰 묶음 {biggest:.0%} · ARI {ari:.3f}")
    print(f"참고: 같은 밀도 집안 HDBSCAN(min_cluster_size=5) ARI {hdb:.3f}")
    # 64칸 공간은 밀도가 고르게 퍼져 있어, 정답 없이 고른 eps가 거의 모든 글씨를 한 묶음으로 잇는다.
    ok = ari > u["CLU_LINE"] and ari > u["RAND"]
    record("5-12", "DBSCAN", "군집", f"ARI {ari:.3f} · {biggest:.0%}가 한 묶음", ok,
           f"eps는 k-거리 무릎({EPS}). HDBSCAN은 ARI {hdb:.3f}")


def ep_5_13(u):
    title("5-13 GMM · 군집팀 — 종 모양(가우시안) 10개를 섞어 확률로 나눠 갖는다")
    # 64칸 그대로면 묶음마다 공분산 숫자가 2,080개라 1,797장으로 모자란다 → 먼저 PCA 30축으로 줄인다.
    Z = PCA(n_components=30, random_state=0).fit_transform(u["X"])
    gmm = GaussianMixture(n_components=10, covariance_type="full", random_state=0).fit(Z)
    lab = gmm.predict(Z)
    ari = adjusted_rand_score(u["y"], lab)
    print(f"PCA 30축 위 full 공분산 GMM · EM {gmm.n_iter_}번 · ARI {ari:.3f}")
    ok = ari > u["CLU_LINE"] and ari > 0.01
    record("5-13", "GMM", "군집", f"ARI {ari:.3f}", ok, "군집팀 1위")


def ep_5_14(u):
    title("5-14 PCA · 차원축소팀 — 가장 많이 퍼진 두 방향으로 그림자를 비춘다")
    p2 = PCA(n_components=2).fit(u["X"])             # 정답을 넣지 않는다
    acc = u["map_score"](p2.transform(u["X"]))       # [채점] 2D 지도 5-NN
    ev = p2.explained_variance_ratio_.sum()
    print(f"2D 지도 5-NN {acc:.3f} · 두 축이 담은 분산 {ev:.3f}  (합격선 {u['DR_LINE']:.3f}, 기준선 {u['BASE_MAP']:.3f})")
    ok = acc > u["DR_LINE"] and acc > u["BASE_MAP"]
    record("5-14", "PCA", "차원축소", f"5-NN {acc:.1%} · 설명된 분산 {ev:.1%}", ok)


def ep_5_15(u, quick):
    title("5-15 t-SNE · UMAP · 차원축소팀 — 이웃을 지키며 2D 지도로 펼친다")
    if quick:
        print("--quick: t-SNE · UMAP을 건너뛴다 (보통 1~3분 걸린다)")
        RESULTS.append(("5-15", "t-SNE · UMAP", "차원축소", "건너뜀", "-", "--quick"))
        return
    t0 = time.perf_counter()
    Zt = TSNE(n_components=2, perplexity=30, random_state=0).fit_transform(u["X"])   # 새 글씨를 놓는 transform은 없다
    s_t = u["map_score"](Zt)
    print(f"t-SNE (perplexity 30) 5-NN {s_t:.3f} · {time.perf_counter() - t0:.0f}초")
    try:
        import umap                                   # pip install umap-learn
        t0 = time.perf_counter()
        Zu = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=0).fit_transform(u["X"])   # y를 넣지 않는다
        s_u = u["map_score"](Zu)
        print(f"UMAP (n_neighbors 15) 5-NN {s_u:.3f} · {time.perf_counter() - t0:.0f}초 (첫 실행은 numba 컴파일 시간 포함)")
    except ImportError:
        s_u = None
        print("umap-learn이 없어 UMAP을 건너뛴다 → pip install umap-learn")
    best = min(s_t, s_u) if s_u is not None else s_t   # 둘 다 넘어야 합격 (UMAP이 없으면 t-SNE만)
    ok = best > u["DR_LINE"] and best > u["BASE_MAP"]
    score = f"5-NN {s_t:.1%}" + (f" · {s_u:.1%}" if s_u is not None else "")
    record("5-15", "t-SNE · UMAP", "차원축소", score, ok, "거리와 묶음 크기는 믿지 말 것")


# =============================================================================
# 4. 결산
# =============================================================================
def final_table(d, u):
    title("결산 · 지원자 15명 (매뉴얼 씨는 기준)")
    print(f"{'편':<6}{'지원자':<14}{'팀':<8}{'판정':<6}핵심 숫자")
    for ep, who, team, score, res, note in RESULTS:
        print(f"{ep:<6}{who:<14}{team:<8}{res:<6}{score}" + (f"   ({note})" if note else ""))
    cand = [r for r in RESULTS if r[0] != "5-0" and r[4] in ("합격", "보류")]
    n_pass = sum(r[4] == "합격" for r in cand)
    print(f"\n합격 {n_pass} · 보류 {len(cand) - n_pass}  (판정한 지원자 {len(cand)}명)")
    if d:
        print(f"합격선 · 분류 {d['LINE_ACC']:.1%} · 회귀 R² {d['LINE_R2']:.3f}", end="")
    if u:
        print(f" · 군집 ARI {u['CLU_LINE']:.3f} · 2D 지도 5-NN {u['DR_LINE']:.1%}", end="")
    print()


def main():
    ap = argparse.ArgumentParser(description="모델 채용 위원회 · 전체 모델 면접을 로컬에서 다시 돌린다")
    ap.add_argument("--quick", action="store_true", help="t-SNE · UMAP(5-15)을 건너뛴다")
    ap.add_argument("--only", nargs="*", help="고른 편만, 예: --only 5-4 5-10")
    a = ap.parse_args()
    want = set(a.only) if a.only else None
    run = lambda ep: want is None or ep in want
    print(f"Python {sys.version.split()[0]} · scikit-learn {sklearn.__version__} · numpy {np.__version__} · pandas {pd.__version__}")
    t_all = time.perf_counter()

    sup = [("5-1", ep_5_1), ("5-2", ep_5_2), ("5-3", ep_5_3), ("5-4", ep_5_4), ("5-5", ep_5_5),
           ("5-6", ep_5_6), ("5-7", ep_5_7), ("5-8", ep_5_8), ("5-9", ep_5_9)]
    uns = [("5-10", ep_5_10), ("5-11", ep_5_11), ("5-12", ep_5_12), ("5-13", ep_5_13), ("5-14", ep_5_14)]
    d = u = None
    if want is None or want & ({"5-0"} | {e for e, _ in sup}):
        d = prepare_supervised()
        if run("5-0"): ep_5_0(d)
        for ep, f in sup:
            if run(ep): f(d)
    if want is None or want & ({e for e, _ in uns} | {"5-15"}):
        u = prepare_unsupervised()
        for ep, f in uns:
            if run(ep): f(u)
        if run("5-15"): ep_5_15(u, a.quick)
    final_table(d, u)
    print(f"\n전체 {time.perf_counter() - t_all:.0f}초")


if __name__ == "__main__":
    main()
