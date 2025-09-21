import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# 1. Load Dataset
df = pd.read_csv("dataset.csv")

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['sentiment'], test_size=0.3, random_state=42
)

# 3. Convert text to TF-IDF features
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 4. Train Model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# 5. Evaluate Model
y_pred = model.predict(X_test_vec)
print("Model Performance:\n")
print(classification_report(y_test, y_pred))

# 6. Try Custom Sentences
custom_sentences = [
    "I really enjoyed this product",
    "This is the worst experience ever",
    "It was fine, nothing special"
]

custom_vec = vectorizer.transform(custom_sentences)
predictions = model.predict(custom_vec)

print("\nCustom Predictions:")
for text, sentiment in zip(custom_sentences, predictions):
    print(f"{text} -> {sentiment}")

# 7. Visualization
sentiment_counts = df['sentiment'].value_counts()
sentiment_counts.plot(kind='bar', color=['green', 'red', 'blue'])
plt.title("Sentiment Distribution in Dataset")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()
