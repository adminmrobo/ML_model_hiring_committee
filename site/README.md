# 모델 채용 위원회 — 머신러닝 열다섯 명의 입사 면접

콘크리트 공장이 품질 판단을 맡길 새 직원을 뽑습니다. 지원자는 KNN부터 t-SNE · UMAP까지 머신러닝 모델 열다섯 명이고, 합격선은 사람이 쓴 규칙으로 일해 온 기존 직원 매뉴얼 씨의 점수입니다. 위원장 박사, 인사 담당 조수, 실무 면접관 초파리, 예산 담당 Money-Robo가 면접을 보고, 합격과 보류는 페이지가 브라우저에서 직접 계산한 숫자가 정합니다.

- 글: 안상선 ((주)M-Robo 대표)
- 라이선스: CC BY-NC 4.0 (비상업적 이용, 출처 표기 필수) — [LICENSE.md](LICENSE.md)
- 시작: [index.html](index.html)

## 언어 / Languages

모든 페이지를 영어·중국어·일본어·우즈베크어·러시아어로도 볼 수 있습니다. 위쪽 메뉴의 언어 선택 상자로 바꾸고, 처음 방문하면 브라우저 언어에 맞는 판으로 자동으로 이동합니다(고른 언어는 기억). 주소 뒤에 ?lang=en 처럼 붙여 직접 지정할 수도 있습니다. Colab 코드의 주석과 실행 결과는 한국어 원문 그대로입니다.

| 언어 | 폴더 |
|---|---|
| 한국어 (기본) | / |
| English | en/ |
| 中文 (简体) | zh/ |
| 日本語 | ja/ |
| Oʻzbekcha | uz/ |
| Русский | ru/ |

## 편 목록

5-0 채용 공고에서 전체 지도를 보고, 각 편이 끝나면 다시 5-0으로 돌아옵니다. 쉬운 지원자부터 번호를 매겼지만 각 편은 따로 읽어도 됩니다.

| 편 | 파일 | 제목 | 팀 | 판정 |
|---|---|---|---|---|
| 5-0 | [5-0_job_posting.html](5-0_job_posting.html) | 채용 공고 — 판단을 데이터로 | 채용 공고 | 보류 |
| 5-1 | [5-1_knn.html](5-1_knn.html) | KNN | 입문팀 | 합격 |
| 5-2 | [5-2_linear_regression.html](5-2_linear_regression.html) | 선형 회귀 | 회귀팀 | 보류 |
| 5-3 | [5-3_polynomial_regularization.html](5-3_polynomial_regularization.html) | 다항 회귀와 규제 | 회귀팀 | 합격 |
| 5-4 | [5-4_logistic_regression.html](5-4_logistic_regression.html) | 로지스틱 회귀 | 분류팀 | 합격 |
| 5-5 | [5-5_naive_bayes.html](5-5_naive_bayes.html) | 나이브 베이즈 | 분류팀 | 보류 |
| 5-6 | [5-6_decision_tree.html](5-6_decision_tree.html) | 결정 트리 | 겸용팀 | 합격 |
| 5-7 | [5-7_random_forest.html](5-7_random_forest.html) | 랜덤 포레스트 | 겸용팀 | 합격 |
| 5-8 | [5-8_svm.html](5-8_svm.html) | SVM | 겸용팀 | 합격 |
| 5-9 | [5-9_gbm.html](5-9_gbm.html) | GBM | 겸용팀 | 합격 |
| 5-10 | [5-10_kmeans.html](5-10_kmeans.html) | K-Means | 군집팀 | 합격 |
| 5-11 | [5-11_hierarchical.html](5-11_hierarchical.html) | 계층적 군집 | 군집팀 | 합격 |
| 5-12 | [5-12_dbscan.html](5-12_dbscan.html) | DBSCAN | 군집팀 | 보류 |
| 5-13 | [5-13_gmm.html](5-13_gmm.html) | GMM | 군집팀 | 합격 |
| 5-14 | [5-14_pca.html](5-14_pca.html) | PCA | 차원축소팀 | 합격 |
| 5-15 | [5-15_tsne_umap.html](5-15_tsne_umap.html) | t-SNE · UMAP | 차원축소팀 | 합격 |

## 각 편에 들어 있는 것

- 지원서 1장 — 9칸 한 장 요약(OVERVIEW)
- 면접 장면, 직접 조작하는 실습과 움직이는 그림, 확인 문제
- Colab 코드 — 복사, .py · .ipynb 내려받기. 페이지에 실제 실행 결과를 실었습니다
- AI 응용 프롬프트, 용어 정리(콘크리트 용어 포함), 비유 모음, 레퍼런스
- 교육자를 위한 TIP

## 데이터

- 지도 본부(5-1 ~ 5-9): UCI Machine Learning Repository, Concrete Compressive Strength (Yeh, 1998, 1,030건, CC BY 4.0)
- 비지도 본부(5-10 ~ 5-15): scikit-learn 내장 손글씨 숫자 load_digits (1,797장)

각 HTML은 그림·데이터·스크립트를 파일 안에 담고 있어 빌드 없이 브라우저로 열면 됩니다(글꼴만 Google Fonts에서 불러옵니다). 편끼리의 링크는 같은 폴더 구조에서 작동하므로 파일 이름을 바꾸지 마세요.

---

# The Model Hiring Committee — Fifteen Machine-Learning Candidates

A concrete plant is hiring a new employee to make its quality calls. The candidates are fifteen machine-learning models, from KNN to t-SNE · UMAP, and the hiring bar is the score of Mr. Manual, the current employee who works from rules written by people. The Chair (Doctor), the HR Assistant, the Fruit Fly and Money-Robo run the interviews, and every Hired or On hold verdict is decided by numbers the page computes in your browser.

- Author: Ahn Sangsun (CEO, M-Robo Co., Ltd.)
- License: CC BY-NC 4.0 (non-commercial use, attribution required) — [LICENSE.md](LICENSE.md)
- Start: [en/index.html](en/index.html)

## Episodes

Start with the full map in 5-0 Job Posting; every episode leads back to it. Candidates are numbered from easiest to hardest, but each episode stands on its own.

| Ep. | File | Title | Team | Verdict |
|---|---|---|---|---|
| 5-0 | [en/5-0_job_posting.html](en/5-0_job_posting.html) | Job Posting — Decisions from Data | Job posting | On hold |
| 5-1 | [en/5-1_knn.html](en/5-1_knn.html) | KNN | Starter Team | Hired |
| 5-2 | [en/5-2_linear_regression.html](en/5-2_linear_regression.html) | Linear regression | Regression Team | On hold |
| 5-3 | [en/5-3_polynomial_regularization.html](en/5-3_polynomial_regularization.html) | Polynomial regression and regularization | Regression Team | Hired |
| 5-4 | [en/5-4_logistic_regression.html](en/5-4_logistic_regression.html) | Logistic regression | Classification Team | Hired |
| 5-5 | [en/5-5_naive_bayes.html](en/5-5_naive_bayes.html) | Naive Bayes | Classification Team | On hold |
| 5-6 | [en/5-6_decision_tree.html](en/5-6_decision_tree.html) | Decision tree | Dual-role Team | Hired |
| 5-7 | [en/5-7_random_forest.html](en/5-7_random_forest.html) | Random forest | Dual-role Team | Hired |
| 5-8 | [en/5-8_svm.html](en/5-8_svm.html) | SVM | Dual-role Team | Hired |
| 5-9 | [en/5-9_gbm.html](en/5-9_gbm.html) | GBM | Dual-role Team | Hired |
| 5-10 | [en/5-10_kmeans.html](en/5-10_kmeans.html) | K-Means | Clustering Team | Hired |
| 5-11 | [en/5-11_hierarchical.html](en/5-11_hierarchical.html) | Hierarchical clustering | Clustering Team | Hired |
| 5-12 | [en/5-12_dbscan.html](en/5-12_dbscan.html) | DBSCAN | Clustering Team | On hold |
| 5-13 | [en/5-13_gmm.html](en/5-13_gmm.html) | GMM | Clustering Team | Hired |
| 5-14 | [en/5-14_pca.html](en/5-14_pca.html) | PCA | Dimensionality Reduction Team | Hired |
| 5-15 | [en/5-15_tsne_umap.html](en/5-15_tsne_umap.html) | T-SNE · UMAP | Dimensionality Reduction Team | Hired |

## In every episode

- Application, page 1 — a nine-box one-page overview
- Interview scenes, hands-on labs and animated diagrams, check questions
- Colab code — copy, download .py / .ipynb; real run output is shown on the page
- AI prompts, glossary (including concrete terms), analogies, references
- Tips for educators

## Data

- Supervised Division (5-1 – 5-9): UCI Machine Learning Repository, Concrete Compressive Strength (Yeh, 1998, 1,030 mixes, CC BY 4.0)
- Unsupervised Division (5-10 – 5-15): scikit-learn built-in handwritten digits, load_digits (1,797 images)

Each HTML file contains its images, data and scripts, so you just open it in a browser (only fonts load from Google Fonts). Links between episodes work in the same folder layout, so keep the file names.

---

# 模型招聘委员会 — 十五位机器学习应聘者的入职面试

一家混凝土工厂要招一名新员工来负责质量判定。应聘者是从 KNN 到 t-SNE · UMAP 的十五个机器学习模型，录用线是靠人写的规则工作的现任员工“手册先生”的得分。主席博士、人事助理、技术面试官果蝇和预算官 Money-Robo 主持面试，录用还是待定，由页面在浏览器中直接计算的数字决定。

- 作者：Ahn Sangsun（M-Robo 公司代表）
- 许可：CC BY-NC 4.0（非商业使用，须注明出处） — [LICENSE.md](LICENSE.md)
- 开始: [zh/index.html](zh/index.html)

## 篇章目录

先在 5-0 招聘启事看全图，每一篇结束后都会回到 5-0。编号从易到难，但每篇也可单独阅读。

| 篇 | 文件 | 标题 | 小组 | 判定 |
|---|---|---|---|---|
| 5-0 | [zh/5-0_job_posting.html](zh/5-0_job_posting.html) | 招聘启事 — 用数据做判断 | 招聘启事 | 待定 |
| 5-1 | [zh/5-1_knn.html](zh/5-1_knn.html) | KNN | 入门组 | 录用 |
| 5-2 | [zh/5-2_linear_regression.html](zh/5-2_linear_regression.html) | 线性回归 | 回归组 | 待定 |
| 5-3 | [zh/5-3_polynomial_regularization.html](zh/5-3_polynomial_regularization.html) | 多项式回归与正则化 | 回归组 | 录用 |
| 5-4 | [zh/5-4_logistic_regression.html](zh/5-4_logistic_regression.html) | 逻辑回归 | 分类组 | 录用 |
| 5-5 | [zh/5-5_naive_bayes.html](zh/5-5_naive_bayes.html) | 朴素贝叶斯 | 分类组 | 待定 |
| 5-6 | [zh/5-6_decision_tree.html](zh/5-6_decision_tree.html) | 决策树 | 两用组 | 录用 |
| 5-7 | [zh/5-7_random_forest.html](zh/5-7_random_forest.html) | 随机森林 | 两用组 | 录用 |
| 5-8 | [zh/5-8_svm.html](zh/5-8_svm.html) | SVM | 两用组 | 录用 |
| 5-9 | [zh/5-9_gbm.html](zh/5-9_gbm.html) | GBM | 两用组 | 录用 |
| 5-10 | [zh/5-10_kmeans.html](zh/5-10_kmeans.html) | K-Means | 聚类组 | 录用 |
| 5-11 | [zh/5-11_hierarchical.html](zh/5-11_hierarchical.html) | 层次聚类 | 聚类组 | 录用 |
| 5-12 | [zh/5-12_dbscan.html](zh/5-12_dbscan.html) | DBSCAN | 聚类组 | 待定 |
| 5-13 | [zh/5-13_gmm.html](zh/5-13_gmm.html) | GMM | 聚类组 | 录用 |
| 5-14 | [zh/5-14_pca.html](zh/5-14_pca.html) | PCA | 降维组 | 录用 |
| 5-15 | [zh/5-15_tsne_umap.html](zh/5-15_tsne_umap.html) | T-SNE · UMAP | 降维组 | 录用 |

## 每篇包含

- 申请表·第1页 — 九格一页总览
- 面试场景、可亲手操作的实验与动态图示、确认题
- Colab 代码 — 可复制、下载 .py / .ipynb，页面附有真实运行结果
- AI 应用提示词、术语表（含混凝土术语）、比喻集、参考文献
- 给教育者的 TIP

## 数据

- 监督学习本部（5-1 ~ 5-9）：UCI Machine Learning Repository，Concrete Compressive Strength（Yeh, 1998，1,030 组，CC BY 4.0）
- 无监督学习本部（5-10 ~ 5-15）：scikit-learn 内置手写数字 load_digits（1,797 张）

每个 HTML 文件都内含图片、数据和脚本，用浏览器直接打开即可（只有字体从 Google Fonts 加载）。篇与篇之间的链接依赖相同的文件夹结构，请勿更改文件名。

---

# モデル採用委員会 — 機械学習モデル15名の入社面接

コンクリート工場が品質判定を任せる新しい社員を採用します。応募者は KNN から t-SNE・UMAP までの機械学習モデル15名、採用ラインは人が書いたルールで働いてきた既存社員マニュアルさんのスコアです。委員長の博士、人事担当の助手、実務面接官のショウジョウバエ、予算担当 Money-Robo が面接し、採用か保留かはページがブラウザで直接計算した数字が決めます。

- 著者：アン・サンソン（株式会社 M-Robo 代表）
- ライセンス：CC BY-NC 4.0（非営利利用、出典表示必須） — [LICENSE.md](LICENSE.md)
- スタート: [ja/index.html](ja/index.html)

## 編の一覧

まず 5-0 求人票で全体マップを見てください。各編の最後は 5-0 に戻ります。やさしい応募者から番号を振っていますが、各編は単独でも読めます。

| 編 | ファイル | タイトル | チーム | 判定 |
|---|---|---|---|---|
| 5-0 | [ja/5-0_job_posting.html](ja/5-0_job_posting.html) | 求人票 — 判断をデータで | 求人票 | 保留 |
| 5-1 | [ja/5-1_knn.html](ja/5-1_knn.html) | KNN | 入門チーム | 採用 |
| 5-2 | [ja/5-2_linear_regression.html](ja/5-2_linear_regression.html) | 線形回帰 | 回帰チーム | 保留 |
| 5-3 | [ja/5-3_polynomial_regularization.html](ja/5-3_polynomial_regularization.html) | 多項式回帰と正則化 | 回帰チーム | 採用 |
| 5-4 | [ja/5-4_logistic_regression.html](ja/5-4_logistic_regression.html) | ロジスティック回帰 | 分類チーム | 採用 |
| 5-5 | [ja/5-5_naive_bayes.html](ja/5-5_naive_bayes.html) | ナイーブベイズ | 分類チーム | 保留 |
| 5-6 | [ja/5-6_decision_tree.html](ja/5-6_decision_tree.html) | 決定木 | 兼用チーム | 採用 |
| 5-7 | [ja/5-7_random_forest.html](ja/5-7_random_forest.html) | ランダムフォレスト | 兼用チーム | 採用 |
| 5-8 | [ja/5-8_svm.html](ja/5-8_svm.html) | SVM | 兼用チーム | 採用 |
| 5-9 | [ja/5-9_gbm.html](ja/5-9_gbm.html) | GBM | 兼用チーム | 採用 |
| 5-10 | [ja/5-10_kmeans.html](ja/5-10_kmeans.html) | K-Means | クラスタリングチーム | 採用 |
| 5-11 | [ja/5-11_hierarchical.html](ja/5-11_hierarchical.html) | 階層的クラスタリング | クラスタリングチーム | 採用 |
| 5-12 | [ja/5-12_dbscan.html](ja/5-12_dbscan.html) | DBSCAN | クラスタリングチーム | 保留 |
| 5-13 | [ja/5-13_gmm.html](ja/5-13_gmm.html) | GMM | クラスタリングチーム | 採用 |
| 5-14 | [ja/5-14_pca.html](ja/5-14_pca.html) | PCA | 次元削減チーム | 採用 |
| 5-15 | [ja/5-15_tsne_umap.html](ja/5-15_tsne_umap.html) | T-SNE · UMAP | 次元削減チーム | 採用 |

## 各編の内容

- 応募書類・1枚目 — 9マスの1枚まとめ
- 面接シーン、手で動かす実習と動く図、確認問題
- Colab コード — コピー、.py / .ipynb のダウンロード。実際の実行結果をページに掲載
- AI 活用プロンプト、用語集（コンクリート用語を含む）、たとえ集、参考文献
- 教育者向け TIP

## データ

- 教師あり本部（5-1 〜 5-9）：UCI Machine Learning Repository, Concrete Compressive Strength（Yeh, 1998, 1,030件, CC BY 4.0）
- 教師なし本部（5-10 〜 5-15）：scikit-learn 内蔵の手書き数字 load_digits（1,797枚）

各 HTML は画像・データ・スクリプトをファイル内に持っているので、ビルドなしでブラウザで開くだけです（フォントのみ Google Fonts から読み込みます）。編どうしのリンクは同じフォルダ構成で動くので、ファイル名は変えないでください。

---

# Modellarni ishga olish qoʻmitasi — oʻn besh mashinali oʻqitish nomzodining suhbati

Beton zavodi sifat boʻyicha qarorlarni topshirish uchun yangi xodim qidirmoqda. Nomzodlar — KNN dan t-SNE · UMAP gacha boʻlgan oʻn beshta mashinali oʻqitish modeli, ishga olish chegarasi esa odamlar yozgan qoidalar bilan ishlaydigan amaldagi xodim Manual janobning balli. Suhbatni Rais (Doktor), Kadrlar yordamchisi, Meva pashshasi va Money-Robo oʻtkazadi; «Ishga olindi» yoki «Kutishda» qarorini sahifa brauzerda oʻzi hisoblagan sonlar belgilaydi.

- Muallif: Ahn Sangsun (M-Robo kompaniyasi rahbari)
- Litsenziya: CC BY-NC 4.0 (notijorat foydalanish, manbani koʻrsatish majburiy) — [LICENSE.md](LICENSE.md)
- Boshlash: [uz/index.html](uz/index.html)

## Boblar

Avval 5-0 vakansiya eʼlonida umumiy xaritani koʻring; har bir bob oxirida yana 5-0 ga qaytasiz. Nomzodlar osonidan qiyiniga qarab raqamlangan, lekin har bir bobni alohida oʻqish mumkin.

| Bob | Fayl | Sarlavha | Jamoa | Qaror |
|---|---|---|---|---|
| 5-0 | [uz/5-0_job_posting.html](uz/5-0_job_posting.html) | Vakansiya eʼloni — qarorlar maʼlumotlardan | Vakansiya eʼloni | Kutishda |
| 5-1 | [uz/5-1_knn.html](uz/5-1_knn.html) | KNN | Boshlangʻich jamoa | Ishga olindi |
| 5-2 | [uz/5-2_linear_regression.html](uz/5-2_linear_regression.html) | Chiziqli regressiya | Regressiya jamoasi | Kutishda |
| 5-3 | [uz/5-3_polynomial_regularization.html](uz/5-3_polynomial_regularization.html) | Polinomial regressiya va regulyarizatsiya | Regressiya jamoasi | Ishga olindi |
| 5-4 | [uz/5-4_logistic_regression.html](uz/5-4_logistic_regression.html) | Logistik regressiya | Klassifikatsiya jamoasi | Ishga olindi |
| 5-5 | [uz/5-5_naive_bayes.html](uz/5-5_naive_bayes.html) | Naiv Bayes | Klassifikatsiya jamoasi | Kutishda |
| 5-6 | [uz/5-6_decision_tree.html](uz/5-6_decision_tree.html) | Qaror daraxti | Ikki vazifali jamoa | Ishga olindi |
| 5-7 | [uz/5-7_random_forest.html](uz/5-7_random_forest.html) | Tasodifiy oʻrmon | Ikki vazifali jamoa | Ishga olindi |
| 5-8 | [uz/5-8_svm.html](uz/5-8_svm.html) | SVM | Ikki vazifali jamoa | Ishga olindi |
| 5-9 | [uz/5-9_gbm.html](uz/5-9_gbm.html) | GBM | Ikki vazifali jamoa | Ishga olindi |
| 5-10 | [uz/5-10_kmeans.html](uz/5-10_kmeans.html) | K-Means | Klasterlash jamoasi | Ishga olindi |
| 5-11 | [uz/5-11_hierarchical.html](uz/5-11_hierarchical.html) | Ierarxik klasterlash | Klasterlash jamoasi | Ishga olindi |
| 5-12 | [uz/5-12_dbscan.html](uz/5-12_dbscan.html) | DBSCAN | Klasterlash jamoasi | Kutishda |
| 5-13 | [uz/5-13_gmm.html](uz/5-13_gmm.html) | GMM | Klasterlash jamoasi | Ishga olindi |
| 5-14 | [uz/5-14_pca.html](uz/5-14_pca.html) | PCA | Oʻlchamni kamaytirish jamoasi | Ishga olindi |
| 5-15 | [uz/5-15_tsne_umap.html](uz/5-15_tsne_umap.html) | T-SNE · UMAP | Oʻlchamni kamaytirish jamoasi | Ishga olindi |

## Har bir bobda

- Ariza, 1-sahifa — toʻqqiz katakli bir sahifalik xulosa
- Suhbat sahnalari, qoʻlda boshqariladigan mashqlar va harakatlanuvchi rasmlar, tekshiruv savollari
- Colab kodi — nusxa olish, .py / .ipynb yuklab olish; sahifada haqiqiy natija koʻrsatilgan
- AI uchun promptlar, atamalar lugʻati (beton atamalari bilan), oʻxshatishlar, manbalar
- Oʻqituvchilar uchun maslahatlar

## Maʼlumotlar

- Nazoratli boʻlim (5-1 – 5-9): UCI Machine Learning Repository, Concrete Compressive Strength (Yeh, 1998, 1,030 ta tarkib, CC BY 4.0)
- Nazoratsiz boʻlim (5-10 – 5-15): scikit-learn ichidagi qoʻlyozma raqamlar load_digits (1,797 ta rasm)

Har bir HTML fayl rasmlar, maʼlumotlar va skriptlarni oʻz ichida saqlaydi — brauzerda ochish kifoya (faqat shriftlar Google Fonts dan yuklanadi). Boblar orasidagi havolalar bir xil papka tuzilmasida ishlaydi, shuning uchun fayl nomlarini oʻzgartirmang.

---

# Комиссия по найму моделей — собеседование пятнадцати кандидатов машинного обучения

Бетонный завод нанимает нового сотрудника, которому доверит решения о качестве. Кандидаты — пятнадцать моделей машинного обучения, от KNN до t-SNE · UMAP, а планка найма — результат действующего сотрудника, Мистера Инструкции, который работает по правилам, написанным людьми. Собеседование ведут Председатель (Доктор), Ассистент по кадрам, Дрозофила и Money-Robo, а решение «Принят» или «Отложено» выносят числа, которые страница считает прямо в браузере.

- Автор: Ан Сансон (генеральный директор M-Robo Co., Ltd.)
- Лицензия: CC BY-NC 4.0 (некоммерческое использование, обязательное указание авторства) — [LICENSE.md](LICENSE.md)
- Начать: [ru/index.html](ru/index.html)

## Главы

Начните с общей карты в 5-0 «Вакансия»; каждая глава возвращает к ней. Кандидаты пронумерованы от простого к сложному, но каждую главу можно читать отдельно.

| Глава | Файл | Название | Группа | Решение |
|---|---|---|---|---|
| 5-0 | [ru/5-0_job_posting.html](ru/5-0_job_posting.html) | Вакансия — решения на основе данных | Вакансия | Отложено |
| 5-1 | [ru/5-1_knn.html](ru/5-1_knn.html) | KNN | Вводная группа | Принят |
| 5-2 | [ru/5-2_linear_regression.html](ru/5-2_linear_regression.html) | Линейная регрессия | Группа регрессии | Отложено |
| 5-3 | [ru/5-3_polynomial_regularization.html](ru/5-3_polynomial_regularization.html) | Полиномиальная регрессия и регуляризация | Группа регрессии | Принят |
| 5-4 | [ru/5-4_logistic_regression.html](ru/5-4_logistic_regression.html) | Логистическая регрессия | Группа классификации | Принят |
| 5-5 | [ru/5-5_naive_bayes.html](ru/5-5_naive_bayes.html) | Наивный Байес | Группа классификации | Отложено |
| 5-6 | [ru/5-6_decision_tree.html](ru/5-6_decision_tree.html) | Дерево решений | Универсальная группа | Принят |
| 5-7 | [ru/5-7_random_forest.html](ru/5-7_random_forest.html) | Случайный лес | Универсальная группа | Принят |
| 5-8 | [ru/5-8_svm.html](ru/5-8_svm.html) | SVM | Универсальная группа | Принят |
| 5-9 | [ru/5-9_gbm.html](ru/5-9_gbm.html) | GBM | Универсальная группа | Принят |
| 5-10 | [ru/5-10_kmeans.html](ru/5-10_kmeans.html) | K-Means | Группа кластеризации | Принят |
| 5-11 | [ru/5-11_hierarchical.html](ru/5-11_hierarchical.html) | Иерархическая кластеризация | Группа кластеризации | Принят |
| 5-12 | [ru/5-12_dbscan.html](ru/5-12_dbscan.html) | DBSCAN | Группа кластеризации | Отложено |
| 5-13 | [ru/5-13_gmm.html](ru/5-13_gmm.html) | GMM | Группа кластеризации | Принят |
| 5-14 | [ru/5-14_pca.html](ru/5-14_pca.html) | PCA | Группа снижения размерности | Принят |
| 5-15 | [ru/5-15_tsne_umap.html](ru/5-15_tsne_umap.html) | T-SNE · UMAP | Группа снижения размерности | Принят |

## В каждой главе

- Анкета, стр. 1 — сводка из девяти клеток на одной странице
- Сцены собеседования, интерактивные практикумы и анимированные схемы, проверочные вопросы
- Код для Colab — копирование, скачивание .py / .ipynb; на странице показан реальный результат запуска
- Промпты для ИИ, глоссарий (включая термины по бетону), аналогии, источники
- Советы для преподавателей

## Данные

- Отдел обучения с учителем (5-1 – 5-9): UCI Machine Learning Repository, Concrete Compressive Strength (Yeh, 1998, 1 030 составов, CC BY 4.0)
- Отдел обучения без учителя (5-10 – 5-15): встроенные в scikit-learn рукописные цифры load_digits (1 797 изображений)

Каждый HTML-файл содержит изображения, данные и скрипты внутри — достаточно открыть его в браузере (только шрифты загружаются из Google Fonts). Ссылки между главами работают при той же структуре папок, поэтому не переименовывайте файлы.

---
