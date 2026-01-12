import os
from feature_extractor import compute_features
from risk_scoring import compute_risk_score

DATASET = {
    "real": ("data/real", 0),
    "ai": ("data/ai", 1)
}

THRESHOLD = 50  # >=50 => AI

y_true = []
y_pred = []

print("\n📊 AVALIAÇÃO DO MODELO\n")

for label_name, (folder, label) in DATASET.items():
    for file in os.listdir(folder):
        if not file.endswith(".mp4"):
            continue

        path = os.path.join(folder, file)

        features = compute_features(path)
        risk, _, reasons = compute_risk_score(features, mode="normal")

        prediction = 1 if risk >= THRESHOLD else 0

        y_true.append(label)
        y_pred.append(prediction)

        print(
            file.ljust(20),
            "| real =", label,
            "| risk =", risk,
            "| pred =", prediction,
            "| reasons =", ", ".join(reasons)
        )

# -----------------------------
# MÉTRICAS
# -----------------------------
TP = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp == 1)
TN = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp == 0)
FP = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
FN = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

accuracy = (TP + TN) / len(y_true)
precision = TP / (TP + FP + 1e-6)
recall = TP / (TP + FN + 1e-6)
f1 = 2 * precision * recall / (precision + recall + 1e-6)

print("\n📈 RESULTADOS GLOBAIS")
print("Accuracy :", round(accuracy, 2))
print("Precision:", round(precision, 2))
print("Recall   :", round(recall, 2))
print("F1-score :", round(f1, 2))

print("\n🧠 MATRIZ DE CONFUSÃO")
print("TP (AI bem detetado)   :", TP)
print("TN (Real bem detetado):", TN)
print("FP (Real → AI)        :", FP)
print("FN (AI → Real)        :", FN)


