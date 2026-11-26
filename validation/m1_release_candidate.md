# mfrmpy 0.2.2のM1公開記録

配布候補は `check_rater_contrast` と、同じ関数の予定・実測照合。観測から指定対比を識別できるか、どの共通作品が経路を支えるか、どの未取得セルが単独で識別を回復するかを返す。能力や評価者差の推定、区間の被覆や実用精度の判断は、それぞれ該当する分析で扱う。既存APIと保存形式に対する追加なので、0.2.xの互換性方針に従い版番号を0.2.2とした。[M1の目的と検証](../ROADMAP.md)

候補の点検では、各校で作品 `1` と `2` を別々に1人採点した4評定で、学校内番号を渡すと前の候補が `observed` を返した。更新後の既定動作は、実測・予定表・両者の照合で同じIDが複数群に出た場合に入力エラーを返す。同じ番号の別作品に `A:1` と `B:1` のような固有IDを付けると `common_location_only` を返す。実際に同じ作品を別条件で採点し、作品ごとの水準を共有する場合は `shared_work_locations=True` を指定できる。ガイドには単位重みと、CMLで情報が残るための条件も明記した。

## 候補のソースと配布物

| 用途 | ローカルブランチ | 差分の基点 | 候補コミット |
|---|---|---|---|
| Pythonパッケージの正本 | `release/mfrmpy-0.2.2-m1` | `ad3748f3e6d6e1bcb6cdedddac3c916a82a4a9d6` | `2d19c74c40b468ecc68bf83e1265ab80e29ccb29` |
| 共有アプリ側の公開候補 | `release/mfrmpy-0.2.2-m1` | `f2e1ef93bfb50486b58e93056e2e568b7a63cf2e` | `1bd0fab8456cfedbd29d64260a3bb95260616ed9` |

Python側に割付判定の実装、公開API、ガイド、変更履歴、テストを収めた。M4の表変更、`category_design`、群間DFFの推定・保存、年度差の検証例はそれぞれの公開判断に残す。差分はPython側の基点コミットに適用できる。共有アプリ側には、Pythonリポジトリ専用の `README.md` を除く同じ差分を適用した。[候補の差分](evidence/m1_release_candidate.patch)

両候補をそれぞれビルドし、sdistから生成したwheelとsdistがリポジトリ間でバイト単位で一致することを確認した。配布メタデータは両形式とも `twine check` に合格した。次の計算値はこのローカル候補の配布物に対応する。

| ファイル | SHA-256 |
|---|---|
| `mfrmpy-0.2.2-py3-none-any.whl` | `f2430c6a8bdaf5e3c98876081e2fbb3041df2081d8c49fdbfd8999a016d89bed` |
| `mfrmpy-0.2.2.tar.gz` | `a714d285c5d9adfe1f979d20815df8c62ea4baa2f7cf8a9a8b9671a4d9ae0450` |

## インストール後の確認

ソースツリー外の仮想環境でwheelを導入し、同じsdistのガイドからM1の3つのコード例を実行した。同梱の採点例ではR1・R3の直接経路と共通受験者30人を確認。片校の共通採点が欠ける例では `common_location_only`、B校に観測上の経路なし、予定セル `b1 / B / R2` を補うと観測上の識別が回復することを確認した。例の実行中は推定エンジンを呼ぶと失敗するようにした。

| 環境 | 導入経路 | 確認結果 |
|---|---|---|
| macOS arm64、Python 3.12.13 | `figures` なしで更新後の候補wheelを導入 | 依存関係検査とガイドの3例に合格 |
| macOS arm64、Python 3.12.13 | `figures` ありで更新後の候補wheelを導入 | 割付判定・予定照合・インポートのテスト12件に合格 |
| macOS arm64、Python 3.11.14 | 公開記録とSHA-256が一致する0.2.1から更新 | 対象テスト12件に合格。0.2.1で作った `.mfrm` を再適合せず開き、分析IDと内容指紋が一致 |

開発版でも、割付判定・群間DFF・カテゴリ設計の関連テスト18件が通り、既存のDFF `.mfrm` の利用例2件を再読込できた。変更前の候補ではPython 3.11・3.12で配布物の全49テストが通過済み。今回は変更した割付判定とその利用経路を再検証した。Python 3.11では宣言された下限のNumPy 1.26.0、pandas 3.0.0、SciPy 1.10.0、PyArrow 16.0.0と、Matplotlib 3.8.0を使用。依存パッケージの導入にはローカルのwheelキャッシュを使用した。

## 公開と導入の検証

2026年11月21日、両候補を各リポジトリの `main` に反映した。パッケージCIは両方で、Linux・macOS・WindowsのPython 3.11と3.12、およびLinux・Python 3.11の依存下限構成の計7環境に合格した。両CIのwheelとsdistは、上表のローカル配布物とバイト単位で一致した。[PythonパッケージのCI](https://github.com/Ryuya-dot-com/mfrmpy/actions/runs/44856475183)・[共有アプリ側のパッケージCI](https://github.com/Ryuya-dot-com/MFRM_Python_Application/actions/runs/44856472415)

TestPyPIとPyPIには、共有アプリ側の同じCI実行（`validated_run_id=44856472415`）の成果物を公開した。両インデックスからwheelとsdistを取得し、上表のSHA-256とファイル内容の一致を確認した。HomepageとRepositoryはPythonパッケージの正本、Issuesは同リポジトリの課題トラッカーを指し、分類子は `Development Status :: 4 - Beta` だった。[TestPyPI公開](https://github.com/Ryuya-dot-com/MFRM_Python_Application/actions/runs/44856998618)・[PyPI公開](https://github.com/Ryuya-dot-com/MFRM_Python_Application/actions/runs/44857450235)

| インデックス | macOS arm64・Python 3.12.13での導入経路 | 確認結果 |
|---|---|---|
| [TestPyPI](https://test.pypi.org/project/mfrmpy/0.2.2/) | 依存パッケージを事前導入した環境に0.2.2を指定して導入 | 依存関係検査、ガイドの3例、作品IDの既定の検査と共有の明示指定に合格 |
| [PyPI](https://pypi.org/project/mfrmpy/0.2.2/) | 空の環境から版指定なしで依存パッケージを含めて導入 | 0.2.2が選ばれ、依存関係検査、ガイドの3例、作品IDの検査に合格 |
| [PyPI](https://pypi.org/project/mfrmpy/0.2.2/) | 0.2.1の導入済み環境を版指定なしで更新 | 0.2.2が選ばれ、新規導入と同じ検証に合格。0.2.1で保存した `.mfrm` を再適合せず開き、分析IDと内容指紋が一致 |

同じ共有アプリコミットのWindows CIとGitHub Pagesの配信も成功した。本番パッケージの導入確認時点で、共有アプリのStreamlit CIは独立Pythonテストを実行中だった。[Windows CI](https://github.com/Ryuya-dot-com/MFRM_Python_Application/actions/runs/44856472452)・[GitHub Pages](https://github.com/Ryuya-dot-com/MFRM_Python_Application/actions/runs/44856472460)・[Streamlit CI](https://github.com/Ryuya-dot-com/MFRM_Python_Application/actions/runs/44856472430)
