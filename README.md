# Starbucks Promotion Analysis

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

The goal of this project is to create a machine learning model to predict whether a customer would complete the offer *WHEN THEY VIEW THEIR OFFER FIRST*.(Even though it is suggested that　coupones can be completed without being viewed in this scenario, I think it is not practical)

```
When a person view a coupon, would the person accomplish it?
```


I made the predictive model to answer this question.


## 4. Project Overview

In this project, I took the steps below.

(In 190315StarbucksPromotionAnalysis)
```
1. Exploratory analysis of the dataset
2. Make a dataframe whose each row represents each coupon sent in this survey period.
```

(In 190403StarbucksPromotionAnalysis)
```
3. Make a dataframe whose each row represents the last coupons which were sent to each customers.  
(this dataframe contains only customers to whom at least 2 coupons have been sent)
4. create a predictive model from dataframe of step(3)
5. interprete the model of step(4)
6.
```

## 5. Detail of Predictive Model

#### Model variables

 When customer A receive a coupon B, my predictive model uses 3 types of variables as X.
 ```
 1. attributes of a customer A (age, income, gender...)
 2. type of coupon B (BOGO or Discount, how much more did he had to spent when he saw the coupon)
 3. past behaviors of customer A when he got coupons (when )
```

 And this model predicts y as 1 (will complete) or 0 (will not complete).


#### 6. Model dataset size
Since I decided to predict the customer's last reaction to the coupon,
 ```
 2844 customers received 2 coupons before the latest one
 873  customers received 3 coupons before the latest one
 111  customers received 4 coupons before the latest one
 ```
 Totally 3828 customer's action to the latest coupon is my concern.

 I took 10% testing data and create a model from 90% dataset by using Grid Search.

 #### Model parameter



## 7. Conclusion (interpretation含む)



## 8. Repo Structure
```
.
├── 190315StarbucksPromotionAnalysis.ipynb
├── 190403StarbucksPromotionAnalysis.ipynb
├── README.md
└── data
    ├── firstly_viewed_offers.csv
    ├── hour_vs_viewed.csv
    ├── hour_vs_viewed_modified
    ├── interaction.csv
    ├── merged_df.csv
    ├── merged_df3.csv
    ├── merged_df4.csv
    ├── portfolio.json
    ├── portfolio.xlsx
    ├── portfolio_clean.csv
    ├── portfolio_clean.xlsx
    ├── profile.json
    ├── profile.xlsx
    ├── profile_clean.csv
    ├── profile_clean.xlsx
    ├── transcript.json
    ├── transcript.xlsx
    ├── transcript_clean.csv
    └── transcript_clean.xlsx
```




## 9. The points where I put a lot of efforts (工夫した点・学んだ点)

本プロジェクトではスターバックスのオンラインクーポンに対する顧客の反応について分析した。

クーポンは３種類あり
BOGO(ある会計において規定額に達したら一定額の割引がなされる)
Discount(クーポンが送られてからの累計使用金額が規定額に達したら一定額の割引がなされる)
Information(ある一定期間閲覧可能なインフォページが送られる)

注意点として
クーポンの有効期間はクーポンが送られた瞬間からであり、見られた瞬間からではない
そのためクーポンを閲覧できる時間はクーポンにより異なる
クーポンは見られたかに関わらず規定額に達したら割引がなされる

本プロジェクトではあるクーポンを見た時（まず見る前提で考える）に、その「顧客の属性」、「送られたクーポンの属性」、「その顧客の過去の購買行動」をもとにそのクーポンを達成されるかを予測する予測モデルを作った。




各個人の過去のクーポンに対する行動を、その人間の新たな「属性」として落とし込む点。
過去に送られたクーポンの総数、各クーポンの種類の数、クーポンを見た時のクーポンの残り有効時間は各顧客によってばらばらである。
それを各顧客に有限個の変数に落とし込むことに苦戦した。

最終的には、各顧客がクーポンを受け取った時の状況を  
「有効期間までの残り時間が5日8時間以内 or 5日8時間より多い」
「達成までの残り必要使用額が$10未満 or $10以上」
の４パターンに分け、過去のそれぞれの状況での達成率(0~1)を４つの変数に落とし込んだ。

もっとデータの数が多ければこれに加えて
達成時の割引額の大きさに応じて状況をさらに２分割し合計８パターンに分割することが理想である。

今回のデータでは過去のクーポンが２〜４回と少ないためこの方法ではnull値が多くなってしまう。
補完の方法としては....




なおこのモデルを活かせば、よりデータを収集することによって
「顧客に見せるクーポンが顧客がクーポンを見た時に決定できる」という前提を置くと、どのクーポンを置いた時に達成率が最大になるかを自動的に計算することによって最も達成率が高いクーポンを表示させるという方法を達成できる。

なおこの方法では顧客をスターバックスに来店させるということが目的になっており最終的なクーポンによる損得は加味されていない。

これに関しては
「顧客が日常的に使用する使用額」
「クーポンによっていくら使うことになるかの期待値を求めるモデル」
を作ることで計算できる。

例えば顧客が平均月$30




なお顧客のクーポンを見た瞬間の状況を出来るだけシンプルに４つのセグメントに分けて考えるという点に関しては
以下のポストを参考にした。まずはシンプルにぶつ切りにしてみて問題があれば、あるいはより正確にする必要があればセグメント切りを改善していくというコンセプトのポストである。
https://note.mu/hik0107/n/n854ff66b2621
