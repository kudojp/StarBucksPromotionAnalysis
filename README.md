# Starbucks Promotion Analysis
(後半に日本語の説明あり)

This is my final project in Udacity Data Scinentist Nanodegree. In this project, I will create a predictive model to predict whether a customer would complete a coupon **when they first view it**. (It is tricky that even if a customer does not view and displat the coupon in the shop, the coupon can be automatically used once he spends the amount needed.)

Data set used in this project is a simplified version of the real Starbucks app.


## 1. Data Overview

This data records 17k customers' behaviors, including receiving offers, opening offers, and making purchases.
As a simplification, there are no explicit products to track. Only the amounts of each transaction or offer are recorded.  

There are 3 types of offers that can be sent. Offers can be delivered via multiple channels.

```
 * buy-one-get-one (BOGO)
 (a user needs to spend a certain amount at a time to get a reward equal to that threshold amount)
* discount
 (a user gains a reward equal to a fraction of the amount spent)
* informational.
 (there is no reward, but neither is there a requisite amount that the user is expected to spend.)
```


## 2. Dataset Detail

1. portfolio.json : Coupons detail (10 types of coupons x 6 fields)
```
id (string) - offer id
offer_type (string) - type of offer ie BOGO, discount, informational
difficulty (int) - minimum required spend to complete an offer
reward (int) - reward given for completing an offer
duration (int) - time for offer to be open, in days
channels (list of strings)
```

2. profile.json  :  Rewards program users (17000 users x 5 attributes)  
```
gender: (categorical) M, F, O, or null   
age: (numeric) missing value encoded as 118   
id: (string/hash)   
became_member_on: (date) format YYYYMMDD   
income: (numeric)   
```
3. transcript.json  :  Event log (306648 events x 7 fields)
```
person: (string/hash)
event: (string) offer received, offer viewed, transaction, offer completed
value: (dictionary) different values depending on event type
offer id: (string/hash) not associated with any "transaction"
amount: (numeric) money spent in "transaction"
reward: (numeric) money gained from "offer completed"
time: (numeric) hours after start of test
```



## 3. Project Goal

The goal of this project is to create a machine learning model to predict whether a customer would complete the offer *WHEN THEY VIEW THEIR OFFER FIRST*. (Even though it is suggested that　coupons can be completed without being viewed in this scenario.)

In simple words, I made a model to answer this question.

```
When a person view a coupon, would the person accomplish it?
```

**[NOTE]** This model'S purpose is only to predict whether a customer would complete the coupon or not. Whether issuing coupon to the customer would be beneficial to the company or not is not discussed here. For example, If a coupon which makes $1000 rewards is issued, almost all customers would accomplish it. It has to be decided what kind of coupon would attract a customer and also still beneficial to the company based on the behavior of customers without any coupon (which is not given in this dataset).

## 4. Project Overview

In this project, I took these steps below.

(In 190315StarbucksPromotionAnalysis)
```
1. Exploratory analysis of the dataset
2. Make a dataframe whose each row represents each coupon sent in this survey period.
```

(In 190403StarbucksPromotionAnalysis)
```
3. Make a dataframe whose each row represents the last coupons which were sent to each customers.  
(this dataframe contains only customers to whom at least 2 coupons had been sent before the last one.)
4. create a predictive model from dataframe of (3)
5. interpret the model
```

## 5. Detail of Predictive Model

#### Model variables

 When customer A receive a coupon X, my predictive model uses 3 types of variables as X.
 ```
 1. Type of coupon X (BOGO or Discount, how much more did he had to spent when he saw the coupon)
 2. Attributes of a customer A (age, income, gender...)
 3. Behaviors taken by customer A when he got coupons before this coupon X
 4. Situation when the customer viewed coupon X (how many hours are left etc.)
```

 And this model predicts y as 1 (will complete) or 0 (will not complete).


#### Dataset size used for creating the model

 Since I decided to predict the customer's latest reaction to the coupon,
 ```
 2844 customers who received 2 coupons before the latest one
 873  customers who received 3 coupons before the latest one
 111  customers who received 4 coupons before the latest one
 ```
 These 3828 customer's action to the latest coupon is my concern.

 I took 10% testing data and create a model from 90% dataset. I used Random Forest Classifier and tuned hyperparameters by using Grid Search.

 The final model tuned predicted testing data with f1 score ******. This model is pickled in a file "data/model.pkl".

#### Model variables

'age', 'income', 'gender_F', 'gender_M',
       'gender_O', 'became_year', 'became_month_sin', 'became_month_cos',
       'became_day_sin', 'became_day_cos', 'became_dow_sin',
       'became_dow_cos',
       'offer_bogo', 'offer_disc', 'difficulty',
       'duration', 'reward', 'email', 'mobile', 'social', 'web',
       't_received', 't_viewed', 'amt_till_viewed',
        '0.0', '6.0', '12.0', '18.0', '24.0', '30.0',
       '36.0', '42.0', '48.0', '54.0', '60.0', '66.0', '72.0', '78.0',
       '84.0', '90.0', '96.0', '102.0', '108.0', '114.0',
       'past_completion_rate',
       'short_little_comp_rate',
       'short_lot_comp_rate',
       'long_little_comp_rate',
       'long_lot_comp_rate',
        't_left_when_viewed','amt_needed_when_viewed'

* variables ['short_little_comp_rate', 'short_lot_comp_rate', 'long_little_comp_rate', 'long_lot_comp_rate'] is inputed by *********



## 7. Conclusion (interpretation)





## 8. Repo Structure
```
.
├── 190315StarbucksPromotionAnalysis.ipynb
├── 190403StarbucksPromotionAnalysis.ipynb
├── README.md
└── data
    ├── firstly_viewed_offers.csv
    ├── interaction.csv
    ├── merged_df2.csv
    ├── portfolio.json
    ├── portfolio_clean.csv
    ├── profile.json
    ├── profile_clean.csv
    ├── transcript.json
    ├── transcript_clean.csv
    └── model.pkl

```




# 以下日本語
## 概要
本プロジェクトではスターバックスのオンラインクーポンに対する顧客の反応について分析した。あるクーポンを見た時（まず見る前提で考える）に、その「顧客の属性」、「送られたクーポンの属性」、「その顧客の過去の購買行動」をもとにそのクーポンを達成されるかを予測する予測モデルを作った。


クーポンは３種類ある。
1. BOGO(ある会計において規定額に達したら一定額の割引がなされる)
2. Discount(クーポンが送られてからの累計使用金額が規定額に達したら一定額の割引がなされる)
3. Information(ある一定期間閲覧可能なインフォページが送られる)

注意点として
1. クーポンの有効期間はクーポンが送られた瞬間からであり、見られた瞬間からではない
2. そのためクーポンを閲覧できる時間はクーポンにより異なる
3. クーポンは見られたかに関わらず規定額に達したら割引がなされる

## 工夫した点  
各個人の過去のクーポンに対する行動を、その人間の新たな「属性」として落とし込む点。
過去に送られたクーポンの総数、各クーポンの種類の数、クーポンを見た時のクーポンの残り有効時間は各顧客によってばらばらである。これらを各顧客に有限個の変数に落とし込むことに苦戦した。

これは最終的に、
1. 過去にその顧客に送られたクーポンが開封されるのにかかった時間の分布
2. 過去にその顧客がクーポンを開封した際のその後の達成率
3. 各顧客がクーポンを受け取った時の状況を「有効期間までの残り時間が5日8時間以内 or 5日8時間より多い」✖️「達成までの残り必要使用額が$10未満 or $10以上」の４パターンに分け、それぞれの状況での過去の達成率(0~1)

という３つの観点から変数に落とし込んだ。なお、今回はデータ数の制限から3.で4つの状況に分類したが、これに加え、「達成時の割引額の大きさ」に応じて状況をさらに２分割し合計８パターンの状況に分割することでより精度が改善されるが期待される。
```
過去に顧客のクーポンを見た瞬間の状況を４つのセグメントに分けて考えるというアイデアは以下のポストを参考にした。

https://note.mu/hik0107/n/n854ff66b2621
```

## どう役に立つのか
「顧客に見せるクーポンを顧客がクーポンを開封した瞬間に決定できる」という前提を置くとする。  

この時、表示するクーポンの有効残期間、どれだけの達成基準額、どれだけのクーポン額によって顧客がクーポンを達成できるかどうかが推測できる。これをもとに達成が予想されるクーポンを表示させることができる。（なお今回はモデルの解釈可能性を優先して、ランダムフォレスト分類器を用いたが、ロジスティック回帰などによって達成確率を０〜１まで数字として出すことによって、クーポンを各個人に応じてより細かく最適化することが可能になる）

なおこの方法では問題点が２点ある。
1. 有効残額
現在の仕様ではクーポンの消滅期間はクーポンの中身に依存しているため
2.
顧客をスターバックスに来店させるということが目的になっており最終的なクーポンによる損得は加味されていない。




なお顧客のクーポンを見た瞬間の状況を出来るだけシンプルに４つのセグメントに分けて考えるという点に関しては
以下のポストを参考にした。まずはシンプルにぶつ切りにしてみて問題があれば、あるいはより正確にする必要があればセグメント切りを改善していくというコンセプトのポストである。
https://note.mu/hik0107/n/n854ff66b2621
