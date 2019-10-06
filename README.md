# 課題２ 物件価格予測回帰問題

## データセット作成

#### データの前処理
* 目的変数
priceのヒストグラムは対数正規のような形になっています．このような分布のデータを学習に取り入れると，大きい値の誤差を小さくするように学習が進み，典型的なpriceの分布に対して精度が出ない問題が発生することがあります．
そこで，priceの値に対して対数をとり，データの分布を正規分布に近い形へ変形させます．

<div align="center">
<img src='https://user-images.githubusercontent.com/53504951/66251823-58f06580-e78f-11e9-89ac-b6350d4f6b82.png' width='430'> <img src='https://user-images.githubusercontent.com/53504951/66251537-1711f000-e78c-11e9-8f0a-36752d22adc0.png' width='430'>
</div>

#### 外れ値の処理
データの分布が大きく外れた値が学習データに含まれていると，典型的なデータに対して精度が落ちてしまう問題が発生するため，外れ値を削除する処理を加えます．外れ値の定義は以下に示します．
```math
|x - median| > 1.5 * IQR
```
```math
IQR : 四分位範囲(InterQuartile Range)
```

#### 説明変数の作成
既存の説明変数から新たな説明変数を作成します．既存の説明変数は以下の変数を使用します．  
sqft_living : リビングの面積  
sqft_living15 : 最近傍15件のリビング面積の平均  
sqft_lot : 宅地面積  
sqft_lot15 : 最近傍15件の宅地面積の平均  
yr_built : 建築完成年  
yr_renovation : リノベーションした年  

これらから，以下の特徴量を生成します．  
diviation_sqft_living : sqft_living15におけるsqft_livingの偏差値  
diviation_sqft_lot : sqft_lot15におけるsqft_lotの偏差値  
renovaiton_class : renovationしているかを分類しました．また，renovationの施工していた場合，経過している年数を考慮し，以下ルールに従ってカテゴリ分類しました．

<div align="center">
<img src='https://user-images.githubusercontent.com/53504951/66273786-29864980-e8b2-11e9-8715-b95e21d60f51.png' width='430'> 
</div>


### trainデータ・testデータ・validationデータ

#### データの割合
trainデータを80%，testデータを10%，validationデータを10%に全体のデータを分割しました．trianデータ，testデータ，validationデータは母集団の分布と同じになるように分割しました．trainデータに対して，外れ値を削除する処理を行いました．  
学習モデルの評価を行うために5fold-crossvalidationを行いました．

### 説明変数の選択
説明変数の選択の手法として，Feature Importanceによる選択，相関係数による選択，相関係数からF値を求め，p値を指標とした選択を行った，3つの実験と全てのデータを入力した実験を行いました．

#### Feature Importance による選択
RnadomForest(RF)，LASSO回帰，再帰的特徴消去(RFE)，線形回帰におけるRidge回帰(Ridge)，線形回帰(LinReg)，XGBoost(XGB), LightGBM(lgbm), Catboost(CB)によるfeature importanceを求め，[0,1]に正規化しました．
その結果を以下に示します．これらの平均(mean)を求め，上位15位(grade ~ view)までの特徴量を採用しました．

<div align="center">
<img src='https://user-images.githubusercontent.com/53504951/66258298-903a3300-e7de-11e9-8ca0-25a44595304c.png' width='500'>
</div>


#### 相関係数による選択
* Correlation
各特徴量の相関関係を以下に示します．Priceとの相関係数の絶対値が0.5以上の特徴量を採用しました．Priceとの相関係数の相関係数が0.5以上となったのは，Bathrooms，sqft_living，sqft_above，sqft_living15です．

<img src='https://user-images.githubusercontent.com/53504951/66251965-2f383e00-e791-11e9-897d-8453a78d5e9f.png' width='430'> <img src='https://user-images.githubusercontent.com/53504951/66251998-70c8e900-e791-11e9-8875-d1ea9198f2e5.png' width='430'>

* KBest(f-Regression)
各特徴量と目的変数から相関係数を求め，Fを計算しp値による特徴量抽出を行います．上位10種の特徴量を採用しました．

---
## モデル構築
XGBoost, LightGBM, Catboostで求めたHouse PriceをLinearRegressionの入力としてHouse Priceを求めるスタッキンを行いました．optunaによるパラメータチューニングをXGBoost, LightGBM, CatBoostに対して行いました．

#### Stacking Model
勾配ブースティングを用いたスタッキングモデルを提案します．勾配ブースティングにはXGBoost，LightGBM, CatBoostを用います．これらの勾配ブースティングはKaggleのコンペティションにおいて勝利に貢献してきた実績を考慮し今回使用します．CatBoostは学習時間を他の手法と比べ要しますが，カテゴリ変数に対して定評があります．データの中にはcondition, grade, viewなどのcategory変数で示すべき特徴が存在し，自作の特徴量であるrenovation_classもcatgory変数です．そこでCatBoostを用いました．
XGBoost, LightGBM, CatBoostが予測したHouse Priceを入力とし，LinearRegressionを用いてHouse Priceを予測するモデルを提案します．

<div align="center">
<img src='https://user-images.githubusercontent.com/53504951/66265072-f6f53600-e84a-11e9-8019-285d3f283e94.png' width='800'>
</div>


---
 ## テスト結果
結果を以下に示します．
各手法のFeatureImportanceの平均から抽出した説明変数を用いた実験が最も良い結果でした．

<div align="center">
<img src='https://user-images.githubusercontent.com/53504951/66276394-90652c00-e8cd-11e9-9555-3ea85ce1d731.png' width='800'>
</div>

---

### コードの実行手順
mian.pyは以下のpathに存在しています．
\\src\main.py

`python main.py --ave_importances True --n_traials 150`
```