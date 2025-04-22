import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd
from utils import mask_pii, clean_text


# Load the dataset
df = pd.read_csv("data/email_dataset.csv")
df.dropna(inplace=True)

df["masked_email"] = df["email"].apply(mask_pii)
df["cleaned_email"] = df["masked_email"].apply(clean_text)


# Vectorize the cleaned emails
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df['cleaned_email'])

# Target labels
y = df['type']

# Train the model
model = LogisticRegression()
model.fit(X, y)

# Save model and vectorizer
joblib.dump(model, 'model.joblib')
joblib.dump(vectorizer, 'vectorizer.joblib')

print("✅ Model and vectorizer saved successfully!")

