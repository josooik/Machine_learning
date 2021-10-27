from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

print("VADER – 영화 감상평 감성 분석 \n")

analyzer = SentimentIntensityAnalyzer()
scores = analyzer.polarity_scores(df['review'][0])
print("scores :", scores)

def vader_polarity(review, threshold=0.1):
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(review)

    agg_score = scores['compound']
    return 1 if agg_score >= threshold else 0

df['vader_pred'] = df['review'].apply(lambda x: vader_polarity(x, 0.1))
y_target = df['sentiment'].values
vader_pred = df['vader_pred'].values

print("confusion_matrix(y_target, preds) :\n", confusion_matrix(y_target, vader_pred))
print("accuracy_score(y_target, preds) :", accuracy_score(y_target, vader_pred))
print("precision_score(y_target, preds) :", precision_score(y_target, vader_pred))
print("recall_score(y_target, preds) :", recall_score(y_target, vader_pred))