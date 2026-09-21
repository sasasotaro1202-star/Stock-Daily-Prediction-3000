# PayPay Securities Daily Prediction

PayPay証券の公式取扱リストを基準に、**固定3,000銘柄ではなく、その時点で取引可能な対象を全件動的に取得**する日次予測・研究基盤です。

## 株価予測Universe

対象はPayPay証券の公式「日本株」「米国株」の取扱リストから取得します。日本株ページは個別株・国内ETF・REIT、米国株ページは個別株・米国ETFを掲載しています。件数上限は設定しません。

毎回のUniverseは公式ページから再生成し、取得時刻・ソースハッシュ・スナップショットを保存します。追加・取扱終了を次回更新へ反映し、過去Universeは残してsurvivorship biasを監査できるようにします。

投資信託・日本株CFD・10倍CFD・iDeCoもPayPay証券の商品ラインナップには存在しますが、株式のclose-to-close予測へ混在させません。これらは将来、NAV/CFDなど商品別の予測パイプラインとして同じMaster Catalogから分離処理します。

## 予測

本番Gateを通過した対象について、
- 翌営業日の上昇確率
- 翌営業日の期待リターン（調整後系列ベース）
- 期待終値（実価格ベース）
- q10/q50/q90の条件付き予測レンジ
- クロスセクショナル順位
- モデル間不確実性

を出力します。

## モデル

Logistic Regression / ExtraTrees / HistGradientBoostingをCPU・無料枠の基礎候補とし、OOSで比較します。LightGBM / XGBoost / CatBoost / PatchTST等はchallengerとして追加可能ですが、複雑化だけを理由に昇格させません。

市場状態別routingもOOS結果から決め、同じモデルを全局面へ固定しません。

## 安全性

PIT / available_at、causal feature、chronological Walk-forward OOS、calibration、独立leakage audit、frozen holdout、reproducibility manifest、release gateを必須にします。重要データが不足した場合は推測値を作らずDEFERRED/FAILにします。

## 自動化

GitHub ActionsでUniverse更新、4分割の差分価格取得、品質チェック、OOS研究、校正、Gate、承認済み時だけ予測を定期実行します。

Research system only; not investment advice or a profit guarantee.
