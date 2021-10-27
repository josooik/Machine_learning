from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import numpy as np

items = ['TV', '냉장고', '전자레인지', '컴퓨터', '선풍기', '선풍기', '믹서', '믹서']

encoder = LabelEncoder()
encoder.fit(items)
print(encoder.transform(items))
print(encoder.classes_)
print(encoder.inverse_transform([4, 5, 2, 0, 1, 1, 3, 3]))

labels = encoder.transform(items)
print(labels)
labels = labels.reshape(-1, 1)
print(labels)

oh_encoder = OneHotEncoder()
oh_encoder.fit(labels)
oh_labels = oh_encoder.transform(labels)
print(oh_labels)
print(oh_labels.toarray())
print(oh_labels.shape)