# bigdata-final-project
- [**Korean-hate-speech 데이터셋**](https://github.com/kocohub/korean-hate-speech)을 이용해 lstm 기반 혐오 댓글 분석   
  : 본 프로젝트는 혐오 댓글 데이터에서 키워드 추출 및 시각화를 진행하고, LSTM을 이용해 혐오 댓글을 분석함.
<br></br>

- 데이터 정제 및 분포 시각화   
1. bias 컬럼 분포 시각화  
![화면 캡처 2022-08-08 173419](https://user-images.githubusercontent.com/49390382/183381898-68d88cc5-1527-4db6-9df0-1c94ae41e96c.png)   

2. bias별 hate 컬럼 분포 시각화   
![화면 캡처 2022-08-08 173454](https://user-images.githubusercontent.com/49390382/183382332-a908e2c5-d0d1-4813-98c5-d6480249806b.png)   

3. 혐오성 댓글에서 자주 등장하는 단어 상위 50개 시각화   
![화면 캡처 2022-08-08 173602](https://user-images.githubusercontent.com/49390382/183382593-e4b7b7e4-b638-42a8-8906-6b8223227f23.png)

4. 워드클라우드 - 댓글 데이터에서 자주 등장하는 단어 300개 시각화   
![화면 캡처 2022-08-08 173639](https://user-images.githubusercontent.com/49390382/183384673-2a32088b-53f1-43f2-a95b-591f342115e6.png)
<br></br>
<br></br>
- LSTM을 이용한 혐오 댓글 분석  
1. 데이터 정제   
1.1. train/test 데이터 분리 - train_test_split()   
1.2. hate 컬럼에 대해 원핫인코딩 - get_dummies()   
1.3. 데이터 길이를 통일시켜 모델의 입력으로 사용할 수 있도록 변환 - Tokenizer()   

2. train 데이터로 모델 훈련
3. 모델 정확도와 loss값 시각화   
![그림1](https://user-images.githubusercontent.com/49390382/183385765-d3c4b993-3817-4cf7-a972-5e2dd45ce634.png)
![그림2](https://user-images.githubusercontent.com/49390382/183385841-100f8f52-6a04-42b8-9af1-60711f25c9f7.png)

4. test 데이터로 predict   
5. 모델 평가지표 나타내기(precision, recall, f1-score 등)   
![그림3](https://user-images.githubusercontent.com/49390382/183386275-f3308f48-c269-47ad-8273-464178f39ef6.png)
