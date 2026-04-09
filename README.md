# Semi-Auto Trading OS Mobile + Scanner

スマホ向けの半自動トレード監査ツールに、ウォッチリスト・スキャナーと Discord Webhook 通知を追加した版です。

## 追加したもの
- yfinance を使ったウォッチリスト・スキャン
- 条件に合う候補をプランへワンタップ反映
- Discord Webhook 通知
- スキャナー候補のチャート表示
- 既存の監査 / ジャーナル / 規律曲線

## ローカル起動
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Community Cloud
- 既存の GitHub リポジトリをこの版で上書きするか、新しいリポジトリへアップロード
- Main file path は `app.py`
- 依存関係は `requirements.txt` に含めています

## 注意
- この版のスキャナーは「市場全体」ではなく「ウォッチリスト」走査です
- Community Cloud 上の保存は永続保証が弱いので、ジャーナルは CSV ダウンロード併用が安全です
- Discord Webhook URL は secrets 未使用のため、必要時に都度入力する設計です
