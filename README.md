# 米国各州 RAG アプリ（卒業試験提出用）

GitHubリポジトリ: [yichi77-oss/America_RAG](https://github.com/yichi77-oss/America_RAG)

Streamlit + LangChain + FAISS + OpenAI を使って、`knowledge/*.md` の州データを参照しながら回答するRAGアプリです。  
ユーザーは州とトピックを選ぶだけで、質問文が自動生成され、参照資料ベースで回答が返ります。

## プロジェクト概要

- **目的**: アメリカ50州 + ワシントンD.C.の基礎情報を、Markdownコーパスを根拠に回答する学習用RAGアプリ
- **技術方針**: ローカル実行中心、ベクトル検索 + LLM回答、回答根拠として参照ファイルを表示
- **特徴**:
  - 入力UIは「州選択 + トピック選択」で簡単操作
  - コーパスは `knowledge` 直下のMarkdownのみ
  - 回答は日本語固定、資料に無い情報の推測を抑制

## デモ

### デモURL

- 未設定（ローカル実行版）
- 公開する場合は Streamlit Community Cloud / AWS / Render などにデプロイし、URLをここに記載してください

### デモアカウント

- `ID: demo_user`
- `Password: demo_pass_123`
- 環境変数で上書き可能:
  - `DEMO_USERNAME`
  - `DEMO_PASSWORD`

## アプリの画面（スクリーンショット/GIF）

### 画面キャプチャ

- 既存画像: `19-1127177-0292500B-STD-00.png`
- README内に表示したい場合は、`assets/` 配下へ整理してリンクしてください
![画面キャプチャ](19-1127177-0292500B-STD-00.png)

### 動作GIF

- `assets/app_demo.gif` を追加済み
- アプリ内で「この質問で回答を取得」実行中のローディング表示として使用
![ローディングGIF](assets/app_demo.gif)

## 要件定義（本アプリで満たす範囲）

- Markdownで管理された州情報をナレッジとして使う
- ユーザーは州・トピック選択だけで質問できる
- 回答は参照資料に基づく（ハルシネーション抑制）
- 参照したMarkdownファイルをUI上で確認できる
- コーパスの再読み込み（再インデックス）ができる

## 機能一覧

- 州選択（50州 + ワシントンD.C.）
- トピック選択（総合 / 地理 / 気候 / 歴史 / 観光 / 食事 / 日本からの直行便）
- ログイン認証（デモID/パスワード）
- 質問文の自動生成
- 類似検索（FAISS）
- LLM回答生成（OpenAI）
- 参照元ファイル一覧表示
- 州のランダム選択
- コーパス再読み込み

## 主要エンドポイントと機能説明

このプロジェクトは **Web APIサーバーを持たない Streamlit アプリ** です。  
そのためRESTエンドポイントはありません。画面操作の主機能は以下です。

- **回答生成実行**: 「この質問で回答を取得」ボタン
  - 選択値から質問生成
  - ベクトル検索
  - LLMにコンテキスト付きで問い合わせ
- **再読み込み実行**: 「読み込みを更新」ボタン
  - キャッシュクリア後にコーパス再インデックス

## 使用技術（フレームワーク / ライブラリ）

- Python 3.12
- Streamlit
- LangChain
  - `langchain-core`
  - `langchain-openai`
  - `langchain-community`
  - `langchain-text-splitters`
- `faiss-cpu`

依存関係は `requirements.txt` を参照してください。

## 外部API情報

- **OpenAI API**
  - 用途: 埋め込み生成（`text-embedding-3-small`）、回答生成（`gpt-4o-mini`）
  - 必須環境変数: `OPENAI_API_KEY`
  - 注意: 利用料金が発生するため、APIキー管理に注意

## ディレクトリ構成（主要部）

- `us_states_rag_app.py`: Streamlit本体
- `knowledge/*.md`: RAG参照コーパス（州ごとMarkdown）
- `scripts/gen_us_states_corpus.py`: コーパス生成スクリプト
- `requirements.txt`: Python依存関係
- `.env.example`: 環境変数テンプレート

## 前提条件（必要ツール / バージョン）

- Python 3.12.x
- `pip` 最新推奨
- OpenAI APIキー（課金可能なプロジェクト）
- macOS / Linux / Windows（WSL可）

## 環境変数

`.env.example` をコピーして `.env` を作成してください。

| 変数名 | 必須 | 説明 |
|---|---|---|
| `OPENAI_API_KEY` | 必須 | OpenAI APIキー |
| `DEMO_USERNAME` | 任意 | ログイン用ユーザーID（未設定時: `demo_user`） |
| `DEMO_PASSWORD` | 任意 | ログイン用パスワード（未設定時: `demo_pass_123`） |

## 環境構築方法（ローカル）

```bash
python3.12 -m venv .venv312
source .venv312/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

`.env` に `OPENAI_API_KEY` を設定後、必要ならコーパスを再生成します。

```bash
python scripts/gen_us_states_corpus.py
```

## セットアップ手順

### ローカル実行

```bash
source .venv312/bin/activate
streamlit run us_states_rag_app.py
```

または:

```bash
./run_streamlit.sh
```

### 本番実行（現状）

- 未構築（ローカル中心）
- 本番運用時は、後述のAWS手順を参考にインフラ構築が必要

## AWSデプロイ手順（ECS/Fargate 例）

> 現在は未実装。卒業試験要件のため、標準的な手順を明示します。

1. **コンテナ化**
   - `Dockerfile` を作成し、`streamlit run us_states_rag_app.py --server.port 8501` で起動
2. **ECRにイメージ登録**
   - リポジトリ作成、Docker build/push
3. **ECSクラスター作成（Fargate）**
   - タスク定義でCPU/メモリ、環境変数 `OPENAI_API_KEY` をSecrets Manager経由で設定
4. **ALB + ECS Service**
   - 8501ポートをALB配下で公開
5. **ログ監視**
   - CloudWatch Logsを有効化
6. **ドメイン設定（任意）**
   - Route53 + ACMでHTTPS化

### Lambdaについて

- 本アプリは常時稼働UI（Streamlit）のため、Lambda単体よりECS/Fargateのほうが適しています
- もしLambdaを使う場合は、バッチ（コーパス生成）用途に分離する設計が現実的です

## 実装予定の機能（今後の拡張）

- デモURLの公開（クラウドデプロイ）
- 動作GIFの追加
- UIスクリーンショットを `assets/` 配下に整理
- 認証機能（必要なら）
- 回答評価ログ（質問/参照元/応答）保存
- コーパス更新の差分反映

## 提出要件チェック

- [x] マークダウン記法を使って書いている
- [ ] デモURLがある（未設定）
- [x] デモアカウントの認証情報がある
- [x] 要件定義の内容が書かれている
- [x] 機能が一覧で書かれている
- [x] 使っているフレームワークやライブラリの情報が書かれている
- [x] 使っている外部APIの情報が書かれている
- [x] 環境構築方法が書かれている
- [x] 実装の予定の機能が書かれている
- [x] アプリの動作を表すGIF画像がある
- [x] アプリの画面キャプチャがある（既存画像あり）
- [x] 前提条件（必要なツール、バージョン）が記載されている
- [x] プロジェクト概要が記載されている
- [x] 環境変数説明あり
- [x] AWSデプロイ手順（ECS/Fargate, Lambdaなど）が明示
- [x] 主要エンドポイントと機能説明
- [x] セットアップ手順（ローカル・本番）が記載されている

## 難しい/未完了の項目

- **デモURL**: デプロイ先未確定のため未設定

上記はこのREADMEで「未対応」として明示しています。
