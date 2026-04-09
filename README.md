# Semi-Auto Trading OS Mobile

スマホのSafari / Chromeで使えるように整えた、Streamlitベースの半自動トレード監査ツールです。

## 同梱ファイル
- `app.py` : アプリ本体
- `requirements.txt` : Community Cloud用の依存関係
- `.streamlit/config.toml` : 基本設定
- `README.md` : 使い方

## ローカル起動
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Community Cloud に公開する手順
1. このフォルダ一式を GitHub リポジトリへアップロード
2. Streamlit Community Cloud に GitHub アカウントを接続
3. `Deploy an app` を選ぶ
4. リポジトリを指定し、エントリーポイントを `app.py` にする
5. 必要なら Advanced settings で Python バージョンを選ぶ
6. Deploy

## メモ
- SQLite の `trade_journal.db` は実行時に自動生成されます
- Community Cloud 上では永続保存に制約があるため、長期保管は CSV ダウンロード併用が無難です
- 次の拡張候補は、マーケットスキャン、通知、複数時間足、自動候補抽出です
