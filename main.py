from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

texts = [
    "I love this product",
    "This is amazing",
    "I hate this",
    "This is terrible"
]

labels = [
    "positive",
    "positive",
    "negative",
    "negative"
]

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(texts)

model = MultinomialNB()

model.fit(X, labels)


sentence = input("Write a sentence: ")

result = model.predict(
    vectorizer.transform([sentence])
)

print("Sentiment:", result[0])