import os
import csv
from feature_extractor import compute_features

OUTPUT = "features.csv"

with open(OUTPUT, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "face_variance",
        "face_entropy",
        "face_temporal",
        "label"
    ])

    for label, folder in [(0, "data/real"), (1, "data/ai")]:
        for file in os.listdir(folder):
            if not file.lower().endswith(".mp4"):
                continue

            path = os.path.join(folder, file)
            feats = compute_features(path)

            writer.writerow([
                feats["face_variance"],
                feats["face_entropy"],
                feats["face_temporal"],
                label
            ])

            print(f"Processado: {path}")

