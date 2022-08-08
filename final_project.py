""" 한국어 혐오 표현 
    댓글에서 혐오 표현 탐지 
    - comments: 댓글[string]
    - contain_gender_bias: 성 차별 포함 여부(True/False)[bool]
    - bias: 차별의 종류(none/gender/others)[string]
    - hate: 혐오 표현 여부(none/hate/offensive)[string] """

import pandas as pd

import re
from konlpy.tag import Okt
from collections import Counter
import nltk

from wordcloud import WordCloud
import matplotlib
from matplotlib import font_manager, rc

train = pd.read_csv('labeled/train.tsv', delimiter='\t')
train
train.isnull().sum()   # null값 확인

# test에는 comments 컬럼만 있음
test = pd.read_csv('test.no_label.tsv', delimiter='\t')
test
test.isnull().sum()


# 기사 제목과 댓글을 합쳐 새 dataframe 만들기
#train_title = pd.read_csv('news_title/train.news_title.txt', header=None)

train_title = pd.read_csv('news_title/dev.news_title.txt', header=None)
train_title.rename(columns = {0:'title'}, inplace=True)
train_df = pd.concat([train_title, train], axis=1)

test_title = pd.read_csv('news_title/test.news_title.txt', header=None)
test_title.rename(columns = {0: 'title'}, inplace=True)
test_df = pd.concat([test_title, test], axis=1)

### 데이터 분포 확인
import seaborn as sns
import matplotlib.pyplot as plt

# 한글 깨짐 방지
font_path = "c:/Windows/fonts/malgun.ttf"
font_name = font_manager.FontProperties(fname = font_path).get_name()
matplotlib.rc('font', family=font_name)

# bias 컬럼의 분포
g = sns.countplot(train_df['bias'])
g.set_title('Bias의 분포')
g.set_xticklabels(g.get_xticklabels())

# bias 컬럼을 기준으로 hate 분포 확인 - gender의 경우 hate가 제일 많음
plt.figure(figsize=(12, 8))
g = sns.countplot(x = "bias", hue = "hate" , data = train_df)
g.set_title('Bias별 Hate 분포 확인')


### 워드클라우드 - 혐오표현(hate)이 포함된 댓글 키워드 시각화 
# 불용어
stop_words = [')','?','1','"(', '_', ')/','\n','.',',', '<','!','(','(', '??','..', '4', '|', '>', '?(', '"…', '#', '&', '・', "']",'.',' ','/',"'",'’','”','“','·', '[','!','\n','·','‘','"','\n ',']',':','…',')','(','-', 'nan','가','요','답변','...','을','수','에','질문','제','를','이','도',
                      '좋','1','는','로','으로','2','것','은','다',',','니다','대','들',
                      '들','데','..','의','때','겠','고','게','네요','한','일','할',
                      '10','?','하는','06','주','려고','인데','거','좀','는데','~','ㅎㅎ',
                      '하나','이상','20','뭐','까','있는','잘','습니다','다면','했','주려',
                      '지','있','못','후','중','줄','6','과','어떤','기본','!!',
                      '단어','라고','합','가요','....','보이','네','무지','했습니다',
              '이다','대해','에게','입니다','있다','사람','대한','3','합니다','및','장','에서','하고','검','한다','만',
             '적', '성', '삼', '등', '전', '인', '그', '했다', '와', '위', '해', '권', '된', '서', '말', '분',
             '것', '그', '이', '수', '최고', '우리', '생각', '자신', '이야기', '점', '현실', '더', '보고', '존재', '모습', 
                       '속', '말', '장면', '일', '대한', '뿐',  '가장', '때', '정말', '지금', '나', '상황', '정도' '면', '습', '게', '자', '끝', '볼', '건', '못', 
                       '마치', '기도', '보', '곳', '그', '이상', '원래', '일이', '전', '사람', '도', '막', '를', '다른', '부터', '자기', '시대','평',
                        '뭐', '더', '막상', '전혀', '내', '살', '현재', '지금', '이제',  '사', '인', '법',  '꼭', '간','향후', '당신', '손', 
                       '저', '경우', '전', '얼마', '일단', '걸', '안', '바로', '그냥', '위해', '때문', '은', '앞',  '볼', '자기', '처럼', '순간', '앞', '감정', 
                       '관련', '일', '가야', '살', '보','요', '보고', '수', '제', '두', '몇', '제', '죽', '때', '해', '이', '중', '내내', '후',   '감',
                       '여러','대한', '것', '시작', '래야', '진짜','또', '수도', '오히려', '니', '여기', '꼭', '과연', '나라', '자', '과거', '최후', '무엇',
                       '누가', '뒤', '얘기', '방식', '알', '그것', '탓', '계속', '방법', '대해', '마지막', '악', '처음', '기분', '의미', '놈', 
                       '역사'
             ]

# 분석할 데이터 추출 - 명사 추출하기
message = ''
for comments in train_df['comments'][train_df['hate']=='hate']:
    message = message + (re.sub(r'[^\w]', ' ', comments)) +''  # 특수문자 제거
        
message #출력하여 내용 확인

# ### 1-3. 품사 태깅 : 명사 추출
t = Okt()
tokens_ko = t.morphs(message)  # 형태소
tokens_ko = t.nouns(message)  # 명사 저장
tokens_ko

ko = nltk.Text(tokens_ko)   
print(len(ko.tokens))          # 토큰 전체 갯수
print(len(set(ko.tokens)))     # 토큰 unique 갯수
ko.vocab().most_common(100)   # 가장 많이 등장한 상위 100개 단어

tokens_ko = [each_word for each_word in tokens_ko
             if each_word not in stop_words]

ko = nltk.Text(tokens_ko)
ko.vocab().most_common(50)

plt.figure(figsize=(15,6))
koplot = ko.plot(50)   # 상위 50개 단어를 시각화
koplot.set_title('혐오성 댓글에서 상위 50개 단어')
plt.show()


# 혐오성 댓글에서 상위 300개 단어
data = ko.vocab().most_common(300)
data

# list tuple을 딕셔너리로
def todict(list_tuple):    
    todict = {}
    for i in range(0,len(list_tuple)):
        todict[data[i][0]] = data[i][1]
    return todict

# 워드클라우드 - 혐오성 댓글에서 가장 많이 언급되는 단어 시각화
from PIL import Image
import numpy as np

icon = Image.open('mask4.png')    # 마스크가 될 이미지 불러오기 

mask = Image.new("RGB", icon.size, (255,255,255))
mask.paste(icon,icon)
mask = np.array(mask)

wordcloud = WordCloud(font_path='c:/Windows/Fonts/malgun.ttf',
                      relative_scaling = 0.2,
                      #stopwords=STOPWORDS,
                      background_color='white',
                      mask=mask,
                      width=1000, height=850
                      ).generate_from_frequencies(todict(data))

plt.figure(figsize=(16,8))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()


# 데이터 정제 - train에서 comments를 제외하고 모두 int로
# contain_gender_bias: True는 1, Fakse는 0
#train_df['contain_gender_bias'] = train_df['contain_gender_bias'].replace({'True':1, 'False':0}).astype(int)
train_df = train_df.drop(columns=['title', 'contain_gender_bias', 'bias'])

# bias: none은 0, gender는 1, others는 2
#train_df['bias'] = train_df['bias'].replace({'none':0, 'gender':1, 'others':2}).astype(int)

# hate: none은 0, hate은 1, offensive는 2
train_df['hate'] = train_df['hate'].replace({'none':0, 'hate':1, 'offensive':2}).astype(int)

### Word2Vec으로 워드 임베딩
# Word2Vec의 sentences로 넣어 줄 문장
t = Okt()
words = []
for comments in train_df['comments']:
    words.append(comments)
     
words #출력하여 내용 확인

#  명사 추출 - 리스트 컴프리헨션
word_sentences = [t.nouns(i) for i in words if t.nouns(i) not in stop_words]  # 문장별로 명사 토콘화

word_sentences

from gensim.models import Word2Vec
word2vec = Word2Vec(sentences=word_sentences, vector_size = 100 , window = 5, min_count=5 , workers = 4 , sg = 0)
word2vec.wv.most_similar('중국')

# train/test 데이터 분리
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(word_sentences, pd.DataFrame(train_df['hate']), test_size=0.2, random_state=0)

# 혐오 댓글 여부(y) 원핫인코딩
y_train = pd.get_dummies(y_train['hate'], prefix='hate')
y_train.rename(columns={'hate_0':'none', 'hate_1':'hate', 'hate_2':'offensive'}, inplace=True)

y_test = pd.get_dummies(y_test['hate'], prefix='hate')
y_test.rename(columns={'hate_0':'none', 'hate_1':'hate', 'hate_2':'offensive'}, inplace=True)

from keras.preprocessing import text, sequence

max_features = 10000 # feature: 예측에 영향을 미치는 변수, 여기서는 텍스트
maxlen = 20

# 각 단어를 tokenizer의 인덱스로 매핑
tokenizer = text.Tokenizer(num_words=max_features)  # keras의 text.Tokenizer
tokenizer.fit_on_texts(X_train)  # x_train으로 훈련
X_train = tokenizer.texts_to_sequences(X_train)
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)

X_test = tokenizer.texts_to_sequences(X_test)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)


import keras
from keras.models import Sequential
from keras.layers import Dense,Embedding,LSTM,Dropout
from keras.callbacks import ReduceLROnPlateau
import tensorflow as tf
# 딥러닝 모델 설정
batch_size = 10  # 한 번의 훈련 당 batch_size
epochs = 10   # 훈련 횟수
embed_size = 100
# keras.callback의 ReduceLROnPlateau()
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience = 2, verbose=1,factor=0.5, min_lr=0.00001)

# 딥러닝 모델
model = Sequential()
#Non-trainable embeddidng layer
#model.add(Embedding(max_features, output_dim=embed_size, input_length=maxlen, trainable=False))
model.add(Embedding(max_features, output_dim=embed_size, input_length=maxlen))

#LSTM 
#model.add(LSTM(units=128 , return_sequences = True , recurrent_dropout = 0.1 , dropout = 0.1))
model.add(LSTM(units=64, recurrent_dropout=0.25, dropout=0.25))
model.add(Dense(3, activation='softmax'))
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

#model.compile(optimizer=tf.keras.optimizers.Adam(lr = 0.01), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# 모델 훈련
history = model.fit(X_train, y_train, batch_size = batch_size , validation_data=(X_test, y_test), 
                    epochs = epochs , callbacks = [learning_rate_reduction])

# 모델 검증
# - 모델의 정확도
print("Accuracy of the model on Training Data is - " , model.evaluate(X_train,y_train)[1]*100 , "%")
print("Accuracy of the model on Testing Data is - " , model.evaluate(X_test,y_test)[1]*100 , "%")

# - train/test 데이터의 정확도, loss를 시각화
# 6 훈련 과정 시각화 (정확도)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# 7 훈련 과정 시각화 (손실)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# - 샘플로 5개 예측값
pred = model.predict(X_test)
pred = pred.round()
#original = np.argmax(y_test, axis=1)
pred[:5]

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
# - 모델 평가지표 한번에 보기
print(classification_report(y_test, pred, target_names=['none', 'hate', 'offensive']))
