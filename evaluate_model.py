import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# 1. Carregar dataset
df = pd.read_csv("features.csv")

X = df.drop("label", axis=1)
y = df["label"]

# 2. Dividir treino / teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 3. Treinar modelo
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 4. Previsões
y_pred = model.predict(X_test)

# 5. Resultados
print("\n=== CONFUSION MATRIX ===")
print(confusion_matrix(y_test, y_pred))

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred))

