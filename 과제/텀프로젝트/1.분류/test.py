import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from xgboost import plot_importance
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
# 한글 지원
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname='./font/malgun.ttf').get_name()
rc('font', family=font_name)

freeze = pd.read_csv("./data/data.csv", encoding='ms949')

# 칼럼명 공백 -> "_"
freeze.columns = freeze.columns.str.replace(" ","_")
print(freeze.info())

print("\n")

# y변수 빈도수
print(freeze['동파유무'].value_counts())

print("\n")

freeze_train, freeze_test = train_test_split(freeze, test_size=0.4)

print("freeze_train :", freeze_train.shape)
print("freeze_test :", freeze_test.shape)

print("\n")

cols = list(freeze.columns)
col_x = cols[1:]
col_y = cols[0]

model = XGBClassifier(n_estimators=500)
model.fit(freeze_train[col_x], freeze_train[col_y])
print("model :", model)

print("\n")

fscore = model.get_booster().get_fscore()
print("fscore :", fscore)
len(fscore)
sorted(fscore)

print("\n")

# 내림차순 정렬 key
names = sorted(fscore, key=fscore.get, reverse=True)
print("name :", names)

print("\n")

# fscore 내림차순 정렬
score = [fscore[key] for key in names]
print("score :", score)

print("\n")

fscore_df = pd.DataFrame({'names':names, 'fscore':score})
print(fscore_df.info())

# 중요변수 top10 시각화
new_fscore_df = fscore_df.set_index('names')
new_fscore_df.iloc[:10, :].plot(kind='bar')

#plot_importance(model)

print("\n")

# model 평가
y_pred = model.predict(freeze_test[col_x])
y_true = freeze_test[col_y]

report = classification_report(y_true, y_pred)
print("모델 평가 리포트 :\n", report)

print("\n")

acc = accuracy_score(y_true, y_pred)
print("accuracy =", acc)
plt.show()
