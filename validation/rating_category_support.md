# 5段階評定で観測されないカテゴリと未採点の違い

## 受験者一人の評定パターン

同じ4人の評価者が同じ課題を採点し、Aが `3, 3, 3, 3`、Bが `1, 2, 4, 5` を得たとする。Aでは1・2・4・5が、Bでは3が現れていないが、どちらも4件の採点が完了している。尺度のカテゴリは1〜5のまま扱う。両者の合計は12点。固定したRSM／PCMの単位識別力モデルでは、能力に関わる尤度の部分は、観測された採点枠とその合計点で決まる。同じ採点枠、同じ合計点、同じ能力事前分布を使うと、能力の尤度の形と事後分布が一致する。異なる課題や評価者から得た同じ合計点では、この等価性は保証されない。[Masters（1982）](https://doi.org/10.1007/BF02296272)

実行例は、120人の背景データと3人の確認用受験者を含む。背景には2課題を4評価者全員が採点した960評定があり、各課題×評価者の組合せに1〜5の観測がある。確認用の3人は課題I1だけを受けた。AとBは4人全員が採点し、Cは `3, 3, 3, 未採点`。合計971件の評定を使う。mfrmpyの公開APIと、mfrmr、TAM、eRm、rasch、RaschPy、ConQuestで、AとBの能力推定値が各方式内で一致し、Cが残り3評定で推定されることを確認した。

| パッケージと設定 | AとBの能力推定値（logit） | Cの能力推定値（logit） |
|---|---:|---:|
| mfrmpy 開発ツリー、RSM MML・EAP | −0.2810 | −0.3972 |
| mfrmr 0.2.4、RSM MML・EAP | −0.2810 | −0.3972 |
| TAM 4.3.25、同じRSMの設計配列・EAP | −0.2810 | −0.3972 |
| ConQuest 5.47.5、RSM MML・WLE | −0.3420 | −0.5123 |
| eRm 1.0-10、`LRSM` の設計行列・受験者ML | −0.3452 | −0.5166 |
| Rのrasch 1.12.1、`rasch_mfrm` のPCML・受験者WLE | −0.3534 | −0.5283 |
| RaschPy 1.4.0、`cp-full-rcl`・受験者ML | −0.3566 | −0.5316 |

能力集計の方式は表に記した。ConQuestのEAPもAとBで一致した。mfrmpyではA・Bの事後SDはどちらも0.441、欠測1件のCは0.491だった。公開版0.2.1を作業ツリーから切り離して実行しても、971評定が残り、同じ能力推定値と事後SDを得た。適合はAとBで違い、OutfitはAが0.141、Bが3.549だった。四つの評定しかない例での記述値であり、全員一致や不一致の理由を判定するには、個々の期待評点と残差も見る。[受験者別の出力](evidence/rating_category_people.csv)

## 採点が欠ける行と、校正で使われないカテゴリ

採点が欠けた場合は、その採点枠の尤度寄与を除く。mfrmpyでCの未採点を空欄の行として渡した場合と、その行を省いた場合の能力・ステップ推定値は一致した。TAMとeRmではワイド行列の `NA`、RaschPyではワイド行列の欠損、raschでは長形式の欠損コード、ConQuestの今回の長形式では行の省略で表した。TAM、eRm、RaschPy、rasch向けに行った1〜5から0〜4への平行移動は、すべてのカテゴリの間隔と順序を保つ。設計上割り当てていない採点と、割り当てたのに得られなかった採点は、評定だけの入力では同じように消える。欠測の偏りを調べるには、予定の割付と実施状況を保持する。[TAMの入力仕様](https://alexanderrobitzsch.github.io/TAM/reference/tam.mml.html)・[eRmの仕様](https://pmair78.r-universe.dev/eRm/doc/manual.html)・[raschの仕様](https://drjoshmcgrane.github.io/rasch/reference/rasch_mfrm.html)・[ConQuestの欠測コード](https://conquestmanual.acer.org/s2-00.html)

ベースデータからR4がつけた3だけを2または4に置き換えた例では、R4の評定数を保ったまま、R4に限り3の観測数がゼロになった。

- 共通ステップのRSMでは、他の評価者の評定が3の情報を含む。mfrmpyではカテゴリの支持はあり、求積31点で区間が数値診断により保留されたが、61点ではR4 − R1の区間が返った。mfrmrでは局所的なカテゴリ支持の警告が出た。TAM・eRm・rasch・RaschPyでも元の5段階を使ってこの集計を適合できた。
- 課題別ステップを持つPCMでは両課題で3が観測される。mfrmpyで支持があり、求積61点で区間が返った。評価者別ステップを持つPCMではR4のステップに必要な3が観測されない。mfrmpyは構造・カテゴリ支持で区間を保留し、mfrmrは5段階保持の設定で適合前に停止した。

次に全データの3を2または4に置き換えた。元の5段階を維持する無制約RSMでは、3へのステップを大きくし、3から4へのステップを同じだけ小さくすると、他カテゴリの分子を保ったまま3の確率を小さくできる。3の実測がゼロであれば、この方向に尤度が改善し続けるため、二つの隣接ステップの有限な無制約最尤推定値は得られない。共通RSMでは全体が、PCMでは独立した各ステップの適用範囲が、この判定の単位となる。

| ソフト | 全体で3が一度も現れない場合の実行結果 |
|---|---|
| mfrmpy | 1〜5を保ち、カテゴリ支持と構造境界で区間を保留した。 |
| mfrmr | `preserve` で適合前に停止した。`collapse` では1・2・4・5を1・2・3・4へ写像して適合した。 |
| TAM・ConQuest | 5カテゴリの設計を明示した今回のMMLでは、どちらも1,000反復の上限まで到達した。 |
| eRm | 明示した5カテゴリの設計行列で `LRSM` は結果を返したが、`NaNs produced` の警告が出て、7つの係数SEのうち有限だったのは5つだった。 |
| Rのrasch | `rasch_mfrm` は内部の空きカテゴリを詰め、4カテゴリとしてPCMLを適合した。 |
| RaschPy | 5カテゴリを保持し、3の両側のステップに追加の平滑化をかけた。警告はそのステップを個別に解釈する根拠が乏しいことを明示した。 |

これらは同じ一組の合成データによる挙動確認。校正全体のカテゴリが空の場合、元の尺度を維持した推定、隣接カテゴリの統合・再符号化、平滑化は異なる推定条件になる。DFFで集団ごとにステップを校正する場合も、同じ採点尺度を比較していることを確認する。[適合状況と警告](evidence/rating_category_support.csv)・[mfrmrのカテゴリ方針](https://ryuya-dot-com.github.io/mfrmr/reference/mfrmr_workflow_methods.html)・[RaschPy 1.4.0の推定法](https://pypi.org/project/raschpy/1.4.0/)

## 共通作品が尺度の上限・下限にそろう場合

開発版の `check_rater_contrast` は、尺度端点を明示すると、全評定が上端または下端にそろった作品を除いた割付でも、指定した対比を支えられるかを返す。作品の判定には、そのIDに属する採点済みの全評定を使う。`5, 5` の作品だけでR1とR2が直接つながる例では、除外後に支持が消えた。R1–R3とR3–R2が別々の得点のある作品を採点していれば、直接経路を除いても間接経路で支持が残った。加法RSMで合計10点を与える評定の組は `(5,5)` のみで、合計9点には `(4,5)` と `(5,4)` がある。各合計を与える組を列挙して評価者別得点の条件付き情報を計算し、両配置の判定を独立に確認した。[公開APIの検証例](../tests/test_python_library.py)・[戻り値の仕様](../docs/python_library.md#which-ratings-support-a-rater-comparison-development-version)

同じ機能で、指定した尺度に対する全体と評価者別・群別のカテゴリ件数を確認できる。欠測評定は件数に含めず、同一セルに複数の実測評定があればカテゴリ件数には各評定を含める。群間DFFを保存する場合も、評価者別・群別の使用件数と、共通経路上の上限・下限作品数が出力される。この検証が扱うのは割付とCMLの情報の有無。有限解や区間の被覆率には別の検証が必要になる。

## 公開作文データで観測された評定パターン

Rのsirt 4.2.133に収録された `data.ratings1` は、2009年のオーストリアの教育水準調査で得た第8学年のドイツ語作文評定である。135人、実際に登場する評価者7人、観点5つで、尺度は0〜3。本節の数は収録された1,370件の観点別評定から集計した。[sirtのデータ仕様](https://alexanderrobitzsch.github.io/sirt/reference/data.ratings1.html)

| 観測したパターン | 収録データでの件数 | mfrmpyで点検する単位 |
|---|---|---|
| 複数の評価者がつけた評点がすべて上限の3 | 評価者が2人以上の46人のうち1人。評価者1人の作品も含めると5/135人 | 作品の全観測。CMLで極端合計の作品がどの対比を支えているかを確認する |
| 同じ観点で全評価者が同じ中間評点をつけた | 複数評価者がいる230人×観点セルのうち60セル | 一人・一観点の評定列。評価者の尺度使用とは別の集計 |
| 同じ観点の評定に、観測された上下の間のカテゴリがない | 同じ230セルで1件。評点は1と3で、2が現れなかった | 評点が得られた二つの枠と、尺度全体のカテゴリ支持 |
| 上限の3を一度も使わなかった評価者×観点 | 35組のうち3組（各37〜41評定）。評価者7人はそれぞれ観点を合わせると0〜3をすべて使った | ステップを独立に推定する範囲。観点別・評価者別・評価者×観点別で必要な観測が変わる |

89人を評価者1人、27人を2人、2人を6人、17人を7人が採点した。評価者対の共通作品は17〜21件だった。各作品に観測された全評点が上端または下端にそろった作品を除くと、直接の共通作品が減る評価者対は1組だった。作品・評価者・観点を指定した開発版の `check_rater_contrast` では、除外後も全21対が観測から支持された。このデータの支持は一部の多重採点に支えられており、1人しか採点しない作品が多いという割合だけでは判断できない。逆に、各対がつながるという結果から区間精度や採点予定の欠測機構は判断できない。

この作文データは4段階・単一時点である。5段階で特定の受験者に1・2・4・5だけが現れるケースや、一人の評価者が数十件すべてを4にするケースの頻度は未確認。評価者全体ではカテゴリがそろっても、特定の観点に絞るとカテゴリが空くことは実際に起きている。上限得点にそろった実例は大学の口頭試験にもあり、同じ試験では「受験を試みない」を表す名目上の評点1が使われず、通常の採点とは異なる扱いだった。採点を試みた結果の空カテゴリ、未割当・未採点のセル、受験拒否などの運用コードを区別する。[Svien（2022）表6・図12・ルーブリック](https://teval.jalt.org/sites/default/files/26_1_20_Svien_speaking_exam.pdf#page=11)

集計は次のRコードで再現できる。作品端点の判定には全観点を、中間の空きには同じ観点で複数評価者がいるセルを使う。

```r
data(data.ratings1, package = "sirt")
d <- transform(data.ratings1, rater = droplevels(rater))
a <- reshape(d, direction = "long", varying = list(paste0("k", 1:5)),
             v.names = "score", timevar = "criterion")
stopifnot(!anyNA(a$score), !anyDuplicated(a[c("idstud", "rater", "criterion")]))
by_work <- split(a$score, a$idstud)
maximum <- vapply(by_work, function(s) all(s == 3), logical(1))
minimum <- vapply(by_work, function(s) all(s == 0), logical(1))
stopifnot(!any(minimum))
table(table(d$idstud), maximum)
by_criterion <- split(a$score, interaction(a$idstud, a$criterion, drop = TRUE))
multiple <- by_criterion[lengths(by_criterion) > 1]
sum(vapply(multiple, function(s) length(unique(s)) == 1 && s[1] %in% 1:2, logical(1)))
unname(Filter(function(s) any(!seq(min(s), max(s)) %in% s), multiple))
sum(xtabs(~ rater + criterion + score, a)[, , "3"] == 0)
table(a$rater, a$score)
crossprod(table(d$idstud, d$rater) > 0)
inside <- d[!d$idstud %in% names(maximum)[minimum | maximum], ]
crossprod(table(inside$idstud, inside$rater) > 0)
```

## 再現

```sh
python -m validation.rating_category_support /tmp/rating-check generate
python -m validation.rating_category_support /tmp/rating-check mfrmpy
python -m validation.rating_category_support /tmp/rating-check raschpy
Rscript validation/rating_category_support.R /tmp/rating-check
```

ConQuestの入力は `base.csv` または `global_gap.csv` から未採点の行を省き、`Score` を1引いて `cq-ratings.csv` とした。実行フォルダの `mean.txt` に `1 1 0.0`、`covariance.txt` に `1 1 1.0` を指定する。ConQuest 5.47.5のコマンドを使い、EAP・WLE・適合履歴を読んだ。[ConQuest入力コマンド](rating_category_support_conquest.cqc)
