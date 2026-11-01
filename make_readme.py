#!/usr/bin/env python3
"""site/README.md (6개 언어 소개)와 site/LICENSE.md를 만든다. make_site.py가 부른다."""
import json, os, sys
R = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, f'{R}/i18n'); import i18n
EPS = sorted([json.load(open(f'{R}/ep/{e}/meta.json', encoding='utf-8')) for e in os.listdir(f'{R}/ep') if e.startswith('5-')], key=lambda m: int(m['n'].split('-')[1]))
TEAM = {'5-0': 'job', '5-1': 'start', '5-2': 'reg', '5-3': 'reg', '5-4': 'cls', '5-5': 'cls', '5-6': 'dual', '5-7': 'dual', '5-8': 'dual', '5-9': 'dual',
        '5-10': 'clu', '5-11': 'clu', '5-12': 'clu', '5-13': 'clu', '5-14': 'dr', '5-15': 'dr'}
RES = {'5-0': 'hold', '5-2': 'hold', '5-5': 'hold', '5-12': 'hold'}
URL = os.environ.get('SITE_URL', '')   # 예: https://adminmrobo.github.io/ML_05_Model-hiring-committee/

L = {
 'ko': dict(name='모델 채용 위원회 — 머신러닝 열다섯 명의 입사 면접',
  intro='콘크리트 공장이 품질 판단을 맡길 새 직원을 뽑습니다. 지원자는 KNN부터 t-SNE · UMAP까지 머신러닝 모델 열다섯 명이고, 합격선은 사람이 쓴 규칙으로 일해 온 기존 직원 매뉴얼 씨의 점수입니다. 위원장 박사, 인사 담당 조수, 실무 면접관 초파리, 예산 담당 Money-Robo가 면접을 보고, 합격과 보류는 페이지가 브라우저에서 직접 계산한 숫자가 정합니다.',
  author='글: 안상선 ((주)M-Robo 대표)', lic='라이선스: CC BY-NC 4.0 (비상업적 이용, 출처 표기 필수) — [LICENSE.md](LICENSE.md)', start='시작',
  langs='언어 / Languages', langtxt='모든 페이지를 영어·중국어·일본어·우즈베크어·러시아어로도 볼 수 있습니다. 위쪽 메뉴의 언어 선택 상자로 바꾸고, 처음 방문하면 브라우저 언어에 맞는 판으로 자동으로 이동합니다(고른 언어는 기억). 주소 뒤에 ?lang=en 처럼 붙여 직접 지정할 수도 있습니다. Colab 코드의 주석과 실행 결과는 한국어 원문 그대로입니다.',
  langcol=('언어', '폴더'), default='한국어 (기본)',
  eps='편 목록', cols=('편', '파일', '제목', '팀', '판정'), order='5-0 채용 공고에서 전체 지도를 보고, 각 편이 끝나면 다시 5-0으로 돌아옵니다. 쉬운 지원자부터 번호를 매겼지만 각 편은 따로 읽어도 됩니다.',
  team=dict(job='채용 공고', start='입문팀', reg='회귀팀', cls='분류팀', dual='겸용팀', clu='군집팀', dr='차원축소팀'), res={'pass': '합격', 'hold': '보류'},
  inside='각 편에 들어 있는 것',
  items=['지원서 1장 — 9칸 한 장 요약(OVERVIEW)', '면접 장면, 직접 조작하는 실습과 움직이는 그림, 확인 문제', 'Colab 코드 — 복사, .py · .ipynb 내려받기. 페이지에 실제 실행 결과를 실었습니다', 'AI 응용 프롬프트, 용어 정리(콘크리트 용어 포함), 비유 모음, 레퍼런스', '교육자를 위한 TIP'],
  data='데이터', datatxt=['지도 본부(5-1 ~ 5-9): UCI Machine Learning Repository, Concrete Compressive Strength (Yeh, 1998, 1,030건, CC BY 4.0)', '비지도 본부(5-10 ~ 5-15): scikit-learn 내장 손글씨 숫자 load_digits (1,797장)'],
  files='각 HTML은 그림·데이터·스크립트를 파일 안에 담고 있어 빌드 없이 브라우저로 열면 됩니다(글꼴만 Google Fonts에서 불러옵니다). 편끼리의 링크는 같은 폴더 구조에서 작동하므로 파일 이름을 바꾸지 마세요.'),
 'en': dict(name='The Model Hiring Committee — Fifteen Machine-Learning Candidates',
  intro='A concrete plant is hiring a new employee to make its quality calls. The candidates are fifteen machine-learning models, from KNN to t-SNE · UMAP, and the hiring bar is the score of Mr. Manual, the current employee who works from rules written by people. The Chair (Doctor), the HR Assistant, the Fruit Fly and Money-Robo run the interviews, and every Hired or On hold verdict is decided by numbers the page computes in your browser.',
  author='Author: Ahn Sangsun (CEO, M-Robo Co., Ltd.)', lic='License: CC BY-NC 4.0 (non-commercial use, attribution required) — [LICENSE.md](LICENSE.md)', start='Start',
  eps='Episodes', cols=('Ep.', 'File', 'Title', 'Team', 'Verdict'), order='Start with the full map in 5-0 Job Posting; every episode leads back to it. Candidates are numbered from easiest to hardest, but each episode stands on its own.',
  team=dict(job='Job posting', start='Starter Team', reg='Regression Team', cls='Classification Team', dual='Dual-role Team', clu='Clustering Team', dr='Dimensionality Reduction Team'), res={'pass': 'Hired', 'hold': 'On hold'},
  inside='In every episode', items=['Application, page 1 — a nine-box one-page overview', 'Interview scenes, hands-on labs and animated diagrams, check questions', 'Colab code — copy, download .py / .ipynb; real run output is shown on the page', 'AI prompts, glossary (including concrete terms), analogies, references', 'Tips for educators'],
  data='Data', datatxt=['Supervised Division (5-1 – 5-9): UCI Machine Learning Repository, Concrete Compressive Strength (Yeh, 1998, 1,030 mixes, CC BY 4.0)', 'Unsupervised Division (5-10 – 5-15): scikit-learn built-in handwritten digits, load_digits (1,797 images)'],
  files='Each HTML file contains its images, data and scripts, so you just open it in a browser (only fonts load from Google Fonts). Links between episodes work in the same folder layout, so keep the file names.'),
 'zh': dict(name='模型招聘委员会 — 十五位机器学习应聘者的入职面试',
  intro='一家混凝土工厂要招一名新员工来负责质量判定。应聘者是从 KNN 到 t-SNE · UMAP 的十五个机器学习模型，录用线是靠人写的规则工作的现任员工“手册先生”的得分。主席博士、人事助理、技术面试官果蝇和预算官 Money-Robo 主持面试，录用还是待定，由页面在浏览器中直接计算的数字决定。',
  author='作者：Ahn Sangsun（M-Robo 公司代表）', lic='许可：CC BY-NC 4.0（非商业使用，须注明出处） — [LICENSE.md](LICENSE.md)', start='开始',
  eps='篇章目录', cols=('篇', '文件', '标题', '小组', '判定'), order='先在 5-0 招聘启事看全图，每一篇结束后都会回到 5-0。编号从易到难，但每篇也可单独阅读。',
  team=dict(job='招聘启事', start='入门组', reg='回归组', cls='分类组', dual='两用组', clu='聚类组', dr='降维组'), res={'pass': '录用', 'hold': '待定'},
  inside='每篇包含', items=['申请表·第1页 — 九格一页总览', '面试场景、可亲手操作的实验与动态图示、确认题', 'Colab 代码 — 可复制、下载 .py / .ipynb，页面附有真实运行结果', 'AI 应用提示词、术语表（含混凝土术语）、比喻集、参考文献', '给教育者的 TIP'],
  data='数据', datatxt=['监督学习本部（5-1 ~ 5-9）：UCI Machine Learning Repository，Concrete Compressive Strength（Yeh, 1998，1,030 组，CC BY 4.0）', '无监督学习本部（5-10 ~ 5-15）：scikit-learn 内置手写数字 load_digits（1,797 张）'],
  files='每个 HTML 文件都内含图片、数据和脚本，用浏览器直接打开即可（只有字体从 Google Fonts 加载）。篇与篇之间的链接依赖相同的文件夹结构，请勿更改文件名。'),
 'ja': dict(name='モデル採用委員会 — 機械学習モデル15名の入社面接',
  intro='コンクリート工場が品質判定を任せる新しい社員を採用します。応募者は KNN から t-SNE・UMAP までの機械学習モデル15名、採用ラインは人が書いたルールで働いてきた既存社員マニュアルさんのスコアです。委員長の博士、人事担当の助手、実務面接官のショウジョウバエ、予算担当 Money-Robo が面接し、採用か保留かはページがブラウザで直接計算した数字が決めます。',
  author='著者：アン・サンソン（株式会社 M-Robo 代表）', lic='ライセンス：CC BY-NC 4.0（非営利利用、出典表示必須） — [LICENSE.md](LICENSE.md)', start='スタート',
  eps='編の一覧', cols=('編', 'ファイル', 'タイトル', 'チーム', '判定'), order='まず 5-0 求人票で全体マップを見てください。各編の最後は 5-0 に戻ります。やさしい応募者から番号を振っていますが、各編は単独でも読めます。',
  team=dict(job='求人票', start='入門チーム', reg='回帰チーム', cls='分類チーム', dual='兼用チーム', clu='クラスタリングチーム', dr='次元削減チーム'), res={'pass': '採用', 'hold': '保留'},
  inside='各編の内容', items=['応募書類・1枚目 — 9マスの1枚まとめ', '面接シーン、手で動かす実習と動く図、確認問題', 'Colab コード — コピー、.py / .ipynb のダウンロード。実際の実行結果をページに掲載', 'AI 活用プロンプト、用語集（コンクリート用語を含む）、たとえ集、参考文献', '教育者向け TIP'],
  data='データ', datatxt=['教師あり本部（5-1 〜 5-9）：UCI Machine Learning Repository, Concrete Compressive Strength（Yeh, 1998, 1,030件, CC BY 4.0）', '教師なし本部（5-10 〜 5-15）：scikit-learn 内蔵の手書き数字 load_digits（1,797枚）'],
  files='各 HTML は画像・データ・スクリプトをファイル内に持っているので、ビルドなしでブラウザで開くだけです（フォントのみ Google Fonts から読み込みます）。編どうしのリンクは同じフォルダ構成で動くので、ファイル名は変えないでください。'),
 'uz': dict(name='Modellarni ishga olish qoʻmitasi — oʻn besh mashinali oʻqitish nomzodining suhbati',
  intro='Beton zavodi sifat boʻyicha qarorlarni topshirish uchun yangi xodim qidirmoqda. Nomzodlar — KNN dan t-SNE · UMAP gacha boʻlgan oʻn beshta mashinali oʻqitish modeli, ishga olish chegarasi esa odamlar yozgan qoidalar bilan ishlaydigan amaldagi xodim Manual janobning balli. Suhbatni Rais (Doktor), Kadrlar yordamchisi, Meva pashshasi va Money-Robo oʻtkazadi; «Ishga olindi» yoki «Kutishda» qarorini sahifa brauzerda oʻzi hisoblagan sonlar belgilaydi.',
  author='Muallif: Ahn Sangsun (M-Robo kompaniyasi rahbari)', lic='Litsenziya: CC BY-NC 4.0 (notijorat foydalanish, manbani koʻrsatish majburiy) — [LICENSE.md](LICENSE.md)', start='Boshlash',
  eps='Boblar', cols=('Bob', 'Fayl', 'Sarlavha', 'Jamoa', 'Qaror'), order='Avval 5-0 vakansiya eʼlonida umumiy xaritani koʻring; har bir bob oxirida yana 5-0 ga qaytasiz. Nomzodlar osonidan qiyiniga qarab raqamlangan, lekin har bir bobni alohida oʻqish mumkin.',
  team=dict(job='Vakansiya eʼloni', start='Boshlangʻich jamoa', reg='Regressiya jamoasi', cls='Klassifikatsiya jamoasi', dual='Ikki vazifali jamoa', clu='Klasterlash jamoasi', dr='Oʻlchamni kamaytirish jamoasi'), res={'pass': 'Ishga olindi', 'hold': 'Kutishda'},
  inside='Har bir bobda', items=['Ariza, 1-sahifa — toʻqqiz katakli bir sahifalik xulosa', 'Suhbat sahnalari, qoʻlda boshqariladigan mashqlar va harakatlanuvchi rasmlar, tekshiruv savollari', 'Colab kodi — nusxa olish, .py / .ipynb yuklab olish; sahifada haqiqiy natija koʻrsatilgan', 'AI uchun promptlar, atamalar lugʻati (beton atamalari bilan), oʻxshatishlar, manbalar', 'Oʻqituvchilar uchun maslahatlar'],
  data='Maʼlumotlar', datatxt=['Nazoratli boʻlim (5-1 – 5-9): UCI Machine Learning Repository, Concrete Compressive Strength (Yeh, 1998, 1,030 ta tarkib, CC BY 4.0)', 'Nazoratsiz boʻlim (5-10 – 5-15): scikit-learn ichidagi qoʻlyozma raqamlar load_digits (1,797 ta rasm)'],
  files='Har bir HTML fayl rasmlar, maʼlumotlar va skriptlarni oʻz ichida saqlaydi — brauzerda ochish kifoya (faqat shriftlar Google Fonts dan yuklanadi). Boblar orasidagi havolalar bir xil papka tuzilmasida ishlaydi, shuning uchun fayl nomlarini oʻzgartirmang.'),
 'ru': dict(name='Комиссия по найму моделей — собеседование пятнадцати кандидатов машинного обучения',
  intro='Бетонный завод нанимает нового сотрудника, которому доверит решения о качестве. Кандидаты — пятнадцать моделей машинного обучения, от KNN до t-SNE · UMAP, а планка найма — результат действующего сотрудника, Мистера Инструкции, который работает по правилам, написанным людьми. Собеседование ведут Председатель (Доктор), Ассистент по кадрам, Дрозофила и Money-Robo, а решение «Принят» или «Отложено» выносят числа, которые страница считает прямо в браузере.',
  author='Автор: Ан Сансон (генеральный директор M-Robo Co., Ltd.)', lic='Лицензия: CC BY-NC 4.0 (некоммерческое использование, обязательное указание авторства) — [LICENSE.md](LICENSE.md)', start='Начать',
  eps='Главы', cols=('Глава', 'Файл', 'Название', 'Группа', 'Решение'), order='Начните с общей карты в 5-0 «Вакансия»; каждая глава возвращает к ней. Кандидаты пронумерованы от простого к сложному, но каждую главу можно читать отдельно.',
  team=dict(job='Вакансия', start='Вводная группа', reg='Группа регрессии', cls='Группа классификации', dual='Универсальная группа', clu='Группа кластеризации', dr='Группа снижения размерности'), res={'pass': 'Принят', 'hold': 'Отложено'},
  inside='В каждой главе', items=['Анкета, стр. 1 — сводка из девяти клеток на одной странице', 'Сцены собеседования, интерактивные практикумы и анимированные схемы, проверочные вопросы', 'Код для Colab — копирование, скачивание .py / .ipynb; на странице показан реальный результат запуска', 'Промпты для ИИ, глоссарий (включая термины по бетону), аналогии, источники', 'Советы для преподавателей'],
  data='Данные', datatxt=['Отдел обучения с учителем (5-1 – 5-9): UCI Machine Learning Repository, Concrete Compressive Strength (Yeh, 1998, 1 030 составов, CC BY 4.0)', 'Отдел обучения без учителя (5-10 – 5-15): встроенные в scikit-learn рукописные цифры load_digits (1 797 изображений)'],
  files='Каждый HTML-файл содержит изображения, данные и скрипты внутри — достаточно открыть его в браузере (только шрифты загружаются из Google Fonts). Ссылки между главами работают при той же структуре папок, поэтому не переименовывайте файлы.'),
}


BUNDLE = {
 'ko': dict(h1='이 묶음에 든 것', intro='완성된 사이트, 설명서, 재현 코드, 그리고 처음부터 다시 만드는 데 필요한 소스와 도구가 모두 들어 있습니다.', cols=('폴더 · 파일', '내용'),
  rows=[('site/', '배포용 완성 사이트 — 6개 언어 × 17쪽(HTML 102개), 사이트용 README · LICENSE, .nojekyll. 이 폴더의 내용물을 GitHub Pages 저장소 맨 위에 올립니다'),
        ('docs/제작_재현_매뉴얼.md', '최종본을 처음부터 같은 결과로 다시 만드는 단계별 절차와 체크리스트(한국어)'),
        ('docs/모델채용위원회_컨셉_설정.md', '설계의 원본: 컨셉, 등장인물, 말투, 편 구성, 연출, 데이터, 다국어 방식(한국어)'),
        ('docs/작업지시_*.md', '편 제작을 나눠 맡길 때 쓴 작업 지시서 3종(한국어)'),
        ('local/', '로컬 풀버전 코드 — 전 모델의 판정 숫자를 한 번에 다시 얻는다(1분 안팎) + 콘크리트 데이터 사본'),
        ('build.py · src/ · ep/ · assets/ · data/', '사이트를 만드는 소스와 빌더(편마다 Colab 코드를 실제로 실행해 결과를 싣는다)'),
        ('i18n/', '번역 장치와 6개 언어 번역 카탈로그 · 용어집 · 언어별 규칙'),
        ('qa/ · tools/ · test/', '점검 도구(화면 · 숫자 · 번역), 데이터 재생성, 브라우저 ML 엔진 대조 시험')],
  h2='쓰는 법', uses=[
   ('사이트 보기 · 올리기', 'site/index.html을 브라우저로 엽니다. 배포는 site/ 안의 내용물(.nojekyll 포함)을 저장소 맨 위에 올리면 됩니다.', 'cd site && python3 -m http.server 8000    # http://localhost:8000 — 언어 자동 전환까지 확인'),
   ('판정 숫자 재현', '각 편 페이지 없이 15명 면접의 판정 숫자만 다시 얻습니다. 끝에 결산표(합격 12 · 보류 3)가 나옵니다.', 'pip install numpy pandas scikit-learn umap-learn\npython3 local/model_hiring_committee_full.py'),
   ('처음부터 다시 만들기', 'docs/제작_재현_매뉴얼.md의 단계를 따릅니다. 핵심 명령은 아래와 같습니다.', 'python3 build.py              # out/ 에 6개 언어 페이지\npython3 qa/check_numbers.py   # 기준 숫자 확인\npython3 qa/check_pages.py     # 화면 점검 (pip install playwright)\npython3 make_site.py          # site/ 다시 만들기')]),
 'en': dict(h1="What's in this package", intro='The finished site, the documentation, the reproduction code, and every source file and tool needed to rebuild it from scratch.', cols=('Folder · file', 'Contents'),
  rows=[('site/', 'The finished site to publish — 6 languages × 17 pages (102 HTML files), site README · LICENSE, .nojekyll. Upload the contents of this folder to the root of a GitHub Pages repository'),
        ('docs/제작_재현_매뉴얼.md', 'Production manual: step-by-step procedure and checklists to rebuild the final edition exactly (Korean)'),
        ('docs/모델채용위원회_컨셉_설정.md', 'Design source: concept, characters, voice, episode plan, staging, data, translation method (Korean)'),
        ('docs/작업지시_*.md', 'Three work orders used to split episode production (Korean)'),
        ('local/', 'Full local code — reproduces every verdict number in one run (about a minute) + a copy of the concrete data'),
        ('build.py · src/ · ep/ · assets/ · data/', 'Site sources and builder (it really runs each episode\'s Colab code and embeds the output)'),
        ('i18n/', 'Translation machinery, catalogs for 6 languages, glossaries, per-language rules'),
        ('qa/ · tools/ · test/', 'Checks (pages, numbers, translations), data regeneration, browser ML engine tests')],
  h2='How to use it', uses=[
   ('View or publish the site', 'Open site/index.html in a browser. To publish, upload the contents of site/ (including .nojekyll) to the root of your repository.', 'cd site && python3 -m http.server 8000    # http://localhost:8000 — also tests automatic language switching'),
   ('Reproduce the numbers', 'Get the verdict numbers of all 15 interviews without the pages. A final tally (12 hired · 3 on hold) is printed at the end.', 'pip install numpy pandas scikit-learn umap-learn\npython3 local/model_hiring_committee_full.py'),
   ('Rebuild from scratch', 'Follow docs/제작_재현_매뉴얼.md. The core commands are:', 'python3 build.py              # pages for 6 languages in out/\npython3 qa/check_numbers.py   # reference numbers\npython3 qa/check_pages.py     # page checks (pip install playwright)\npython3 make_site.py          # rebuild site/')]),
 'zh': dict(h1='本资料包内容', intro='包含完成的网站、说明文档、复现代码，以及从零重建所需的全部源文件和工具。', cols=('文件夹 · 文件', '内容'),
  rows=[('site/', '用于发布的完整网站 — 6 种语言 × 17 页（HTML 102 个）、网站用 README · LICENSE、.nojekyll。把此文件夹里的内容上传到 GitHub Pages 仓库根目录'),
        ('docs/제작_재현_매뉴얼.md', '制作复现手册：按步骤以相同结果重建最终版的流程与检查清单（韩语）'),
        ('docs/모델채용위원회_컨셉_설정.md', '设计原稿：概念、人物、口吻、篇章结构、演出、数据、多语言方式（韩语）'),
        ('docs/작업지시_*.md', '分工制作篇章时使用的 3 份作业指示（韩语）'),
        ('local/', '本地完整代码 — 一次运行复现全部模型的判定数字（约 1 分钟）+ 混凝土数据副本'),
        ('build.py · src/ · ep/ · assets/ · data/', '网站源文件与构建器（会实际运行每篇的 Colab 代码并嵌入结果）'),
        ('i18n/', '翻译机制、6 种语言的翻译目录、术语表、各语言规则'),
        ('qa/ · tools/ · test/', '检查工具（页面 · 数字 · 翻译）、数据重新生成、浏览器 ML 引擎对照测试')],
  h2='使用方法', uses=[
   ('查看 · 发布网站', '用浏览器打开 site/index.html。发布时，把 site/ 里的内容（包括 .nojekyll）上传到仓库根目录即可。', 'cd site && python3 -m http.server 8000    # http://localhost:8000 — 可一并确认语言自动切换'),
   ('复现判定数字', '不打开各篇页面，直接重新得到 15 位应聘者的判定数字。最后会输出汇总表（录用 12 · 待定 3）。', 'pip install numpy pandas scikit-learn umap-learn\npython3 local/model_hiring_committee_full.py'),
   ('从零重建', '按照 docs/제작_재현_매뉴얼.md 的步骤进行。核心命令如下：', 'python3 build.py              # 在 out/ 生成 6 种语言页面\npython3 qa/check_numbers.py   # 确认基准数字\npython3 qa/check_pages.py     # 页面检查（pip install playwright）\npython3 make_site.py          # 重新生成 site/')]),
 'ja': dict(h1='このパッケージの中身', intro='完成したサイト、説明書、再現コード、そしてゼロから作り直すためのソースとツールがすべて入っています。', cols=('フォルダ · ファイル', '内容'),
  rows=[('site/', '公開用の完成サイト — 6言語 × 17ページ（HTML 102個）、サイト用 README · LICENSE、.nojekyll。このフォルダの中身を GitHub Pages リポジトリの最上位に置きます'),
        ('docs/제작_재현_매뉴얼.md', '制作再現マニュアル：最終版を同じ結果で作り直す手順とチェックリスト（韓国語）'),
        ('docs/모델채용위원회_컨셉_설정.md', '設計の原本：コンセプト、登場人物、口調、編の構成、演出、データ、多言語の方式（韓国語）'),
        ('docs/작업지시_*.md', '編の制作を分担したときの作業指示書 3種（韓国語）'),
        ('local/', 'ローカル完全版コード — 全モデルの判定数値を一度に再現（約1分）＋ コンクリートデータの写し'),
        ('build.py · src/ · ep/ · assets/ · data/', 'サイトのソースとビルダー（各編の Colab コードを実際に実行して結果を載せる）'),
        ('i18n/', '翻訳の仕組み、6言語の翻訳カタログ、用語集、言語別ルール'),
        ('qa/ · tools/ · test/', '点検ツール（画面 · 数値 · 翻訳）、データ再生成、ブラウザ ML エンジンの照合テスト')],
  h2='使い方', uses=[
   ('サイトを見る · 公開する', 'site/index.html をブラウザで開きます。公開は site/ の中身（.nojekyll を含む）をリポジトリの最上位に置くだけです。', 'cd site && python3 -m http.server 8000    # http://localhost:8000 — 言語の自動切り替えも確認できる'),
   ('判定数値の再現', '各編のページなしで、15名の面接の判定数値だけを得ます。最後に集計表（採用 12 · 保留 3）が出ます。', 'pip install numpy pandas scikit-learn umap-learn\npython3 local/model_hiring_committee_full.py'),
   ('ゼロから作り直す', 'docs/제작_재현_매뉴얼.md の手順に従います。主なコマンドは次のとおりです。', 'python3 build.py              # out/ に6言語のページ\npython3 qa/check_numbers.py   # 基準数値の確認\npython3 qa/check_pages.py     # 画面点検（pip install playwright）\npython3 make_site.py          # site/ を作り直す')]),
 'uz': dict(h1='Toʻplam tarkibi', intro='Tayyor sayt, qoʻllanmalar, qayta hosil qilish kodi hamda saytni noldan qayta qurish uchun barcha manbalar va vositalar.', cols=('Papka · fayl', 'Tarkibi'),
  rows=[('site/', 'Eʼlon qilish uchun tayyor sayt — 6 til × 17 sahifa (102 ta HTML), sayt uchun README · LICENSE, .nojekyll. Shu papka ichidagilarni GitHub Pages omborining ildiziga yuklang'),
        ('docs/제작_재현_매뉴얼.md', 'Ishlab chiqarish qoʻllanmasi: yakuniy nashrni aynan shunday qayta qurish bosqichlari va nazorat roʻyxatlari (koreyscha)'),
        ('docs/모델채용위원회_컨셉_설정.md', 'Loyiha asosi: konsepsiya, qahramonlar, nutq uslubi, boblar tuzilishi, sahnalashtirish, maʼlumotlar, koʻp tillilik (koreyscha)'),
        ('docs/작업지시_*.md', 'Boblarni boʻlib ishlashda ishlatilgan 3 ta ish topshirigʻi (koreyscha)'),
        ('local/', 'Toʻliq lokal kod — barcha modellarning qaror sonlarini bir marta ishga tushirishda qayta hosil qiladi (taxminan 1 daqiqa) + beton maʼlumotlari nusxasi'),
        ('build.py · src/ · ep/ · assets/ · data/', 'Sayt manbalari va quruvchi (har bir bobning Colab kodini haqiqatan ishga tushirib, natijani joylaydi)'),
        ('i18n/', 'Tarjima mexanizmi, 6 til uchun tarjima kataloglari, lugʻatlar, tillar boʻyicha qoidalar'),
        ('qa/ · tools/ · test/', 'Tekshiruv vositalari (sahifalar · sonlar · tarjimalar), maʼlumotlarni qayta yaratish, brauzer ML dvigatelini solishtirish testlari')],
  h2='Qanday foydalaniladi', uses=[
   ('Saytni koʻrish · eʼlon qilish', 'site/index.html faylini brauzerda oching. Eʼlon qilish uchun site/ ichidagilarni (.nojekyll bilan) omboringiz ildiziga yuklang.', 'cd site && python3 -m http.server 8000    # http://localhost:8000 — tilni avtomatik almashtirishni ham tekshiradi'),
   ('Sonlarni qayta hosil qilish', 'Sahifalarsiz 15 ta suhbatning qaror sonlarini oling. Oxirida yakuniy jadval chiqadi (12 ta ishga olindi · 3 ta kutishda).', 'pip install numpy pandas scikit-learn umap-learn\npython3 local/model_hiring_committee_full.py'),
   ('Noldan qayta qurish', 'docs/제작_재현_매뉴얼.md dagi bosqichlarga amal qiling. Asosiy buyruqlar:', 'python3 build.py              # out/ da 6 tildagi sahifalar\npython3 qa/check_numbers.py   # tayanch sonlarni tekshirish\npython3 qa/check_pages.py     # sahifalarni tekshirish (pip install playwright)\npython3 make_site.py          # site/ ni qayta yaratish')]),
 'ru': dict(h1='Что в этом пакете', intro='Готовый сайт, документация, код для воспроизведения и все исходные файлы и инструменты, нужные, чтобы собрать всё заново.', cols=('Папка · файл', 'Содержимое'),
  rows=[('site/', 'Готовый сайт для публикации — 6 языков × 17 страниц (102 HTML-файла), README · LICENSE сайта, .nojekyll. Загрузите содержимое этой папки в корень репозитория GitHub Pages'),
        ('docs/제작_재현_매뉴얼.md', 'Руководство по производству: пошаговая процедура и чек-листы, чтобы собрать финальную версию с тем же результатом (на корейском)'),
        ('docs/모델채용위원회_컨셉_설정.md', 'Исходник замысла: концепция, персонажи, манера речи, структура глав, постановка, данные, многоязычность (на корейском)'),
        ('docs/작업지시_*.md', 'Три рабочих задания, по которым главы делались параллельно (на корейском)'),
        ('local/', 'Полный локальный код — за один запуск воспроизводит все числа решений (около минуты) + копия данных о бетоне'),
        ('build.py · src/ · ep/ · assets/ · data/', 'Исходники сайта и сборщик (он действительно запускает код Colab каждой главы и встраивает результат)'),
        ('i18n/', 'Механизм перевода, каталоги переводов для 6 языков, глоссарии, правила для каждого языка'),
        ('qa/ · tools/ · test/', 'Проверки (страницы · числа · переводы), пересоздание данных, сверка браузерного ML-движка')],
  h2='Как пользоваться', uses=[
   ('Посмотреть или опубликовать сайт', 'Откройте site/index.html в браузере. Для публикации загрузите содержимое site/ (вместе с .nojekyll) в корень репозитория.', 'cd site && python3 -m http.server 8000    # http://localhost:8000 — заодно проверяется автоматический выбор языка'),
   ('Воспроизвести числа', 'Получите числа решений всех 15 собеседований без страниц. В конце печатается итоговая таблица (принято 12 · отложено 3).', 'pip install numpy pandas scikit-learn umap-learn\npython3 local/model_hiring_committee_full.py'),
   ('Собрать заново с нуля', 'Следуйте шагам docs/제작_재현_매뉴얼.md. Основные команды:', 'python3 build.py              # страницы на 6 языках в out/\npython3 qa/check_numbers.py   # проверка эталонных чисел\npython3 qa/check_pages.py     # проверка страниц (pip install playwright)\npython3 make_site.py          # пересобрать site/')]),
}


def title(m, lang, T):
    if lang == 'ko': return m['title']
    t = T.get(i18n.key(m['title'])) or m['title']
    return t[:1].upper() + t[1:]


def tables():
    out = {}
    for lang in L:
        T = {}
        if lang != 'ko':
            d = f'{R}/i18n/{lang}'
            for f in os.listdir(d):
                if f.endswith('.json') and not f.startswith('_'):
                    for k, v in json.load(open(f'{d}/{f}', encoding='utf-8')).items():
                        if v.get('tr'): T[k] = v['tr']
        out[lang] = T
    return out


def build(prefix='', bundle=False):
    """prefix: 페이지 링크 앞에 붙일 경로(묶음 README에서는 'site/'). bundle: 묶음 안내(구성 · 쓰는 법)를 더한다"""
    TT = tables(); parts = []
    for lang, s in L.items():
        p = prefix + ('' if lang == 'ko' else f'{lang}/')
        cols = s['cols']
        lines = [f"# {s['name']}", '', s['intro'], '', f"- {s['author']}", f"- {s['lic']}",
                 f"- {s['start']}: [{p}index.html]({p}index.html)" + (f" · {URL}{p}" if URL else ''), '']
        if lang == 'ko':
            lines += [f"## {s['langs']}", '', s['langtxt'], '', f"| {s['langcol'][0]} | {s['langcol'][1]} |", '|---|---|', f"| {s['default']} | {prefix or '/'} |",
                      f'| English | {prefix}en/ |', f'| 中文 (简体) | {prefix}zh/ |', f'| 日本語 | {prefix}ja/ |', f'| Oʻzbekcha | {prefix}uz/ |', f'| Русский | {prefix}ru/ |', '']
        if bundle:
            b = BUNDLE[lang]
            lines += [f"## {b['h1']}", '', b['intro'], '', f"| {b['cols'][0]} | {b['cols'][1]} |", '|---|---|'] + [f"| `{k}` | {v} |" for k, v in b['rows']] + ['']
            lines += [f"## {b['h2']}", '']
            for head, txt, cmd in b['uses']:
                lines += [f"**{head}** — {txt}", '', '```', cmd, '```', '']
        lines += [f"## {s['eps']}", '', s['order'], '', '| ' + ' | '.join(cols) + ' |', '|---|---|---|---|---|']
        for m in EPS:
            n = m['n']; r = s['res'][RES.get(n, 'pass')]
            lines.append(f"| {n} | [{p}{m['file']}]({p}{m['file']}) | {title(m, lang, TT[lang])} | {s['team'][TEAM[n]]} | {r} |")
        lines += ['', f"## {s['inside']}", ''] + [f'- {i}' for i in s['items']] + ['', f"## {s['data']}", ''] + [f'- {i}' for i in s['datatxt']] + ['', s['files'], '', '---', '']
        parts.append('\n'.join(lines))
    return '\n'.join(parts)


LICENSE = '''# 라이선스 / License

© 안상선 ((주)M-Robo 대표) · Ahn Sangsun (M-Robo Co., Ltd.)

이 저장소의 글, 그림, 페이지는 **크리에이티브 커먼즈 저작자표시-비영리 4.0 국제 라이선스(CC BY-NC 4.0)**를 따릅니다.

- 출처를 밝히면 복제·배포·수정할 수 있습니다.
- 상업적 목적으로는 쓸 수 없습니다.

The text, images and pages in this repository are licensed under **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)**.

전문 / Full text: https://creativecommons.org/licenses/by-nc/4.0/legalcode

데이터 / Data: UCI Concrete Compressive Strength (Yeh, 1998) — CC BY 4.0 · scikit-learn load_digits — BSD-3-Clause.

Money-Robo는 (주)M-Robo 서비스의 마스코트 캐릭터입니다. / Money-Robo is the mascot character of M-Robo's service.
'''

if __name__ == '__main__':
    if len(sys.argv) > 2 and sys.argv[1] == '--bundle':          # 묶음 맨 위 README: 링크는 site/ 아래로
        open(sys.argv[2], 'w', encoding='utf-8').write(build(prefix='site/', bundle=True))
        open(os.path.join(os.path.dirname(sys.argv[2]), 'LICENSE.md'), 'w', encoding='utf-8').write(LICENSE)
        print('묶음 README.md · LICENSE.md 작성'); sys.exit()
    site = sys.argv[1] if len(sys.argv) > 1 else f'{R}/site'
    open(f'{site}/README.md', 'w', encoding='utf-8').write(build())
    open(f'{site}/LICENSE.md', 'w', encoding='utf-8').write(LICENSE)
    print('README.md · LICENSE.md 작성')
