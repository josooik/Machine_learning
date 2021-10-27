from sklearn.preprocessing import LabelEncoder

items = ['TV', '냉장고', '전자레인지', '컴퓨터', '선풍기', '선풍기', '믹서', '믹서']

encoder = LabelEncoder()
encoder.fit(items)
print(encoder.transform(items))
print(encoder.classes_)
print(encoder.inverse_transform([4, 5, 2, 0, 1, 1, 3, 3]))