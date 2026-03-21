import os
import pandas as pd

from image_features import compute_image_features
from feature_extractor import compute_features

DATA_DIR = "data"
OUTPUT_CSV = "features.csv"

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")
VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv")


def process_folder(folder_path, label, media_type):
    rows = []

    if not os.path.exists(folder_path):
        print(f"Pasta nao encontrada: {folder_path}")
        return rows

    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)

        if not os.path.isfile(filepath):
            continue

        try:
            if media_type == "image":
                if not filename.lower().endswith(IMAGE_EXTS):
                    continue
                feats = compute_image_features(filepath)

            elif media_type == "video":
                if not filename.lower().endswith(VIDEO_EXTS):
                    continue
                feats = compute_features(filepath)

            else:
                continue

            feats["type"] = media_type
            feats["label"] = label
            feats["file"] = filename

            rows.append(feats)
            print(f"Processado: {filepath}")

        except Exception as e:
            print(f"Erro em {filepath}: {e}")

    return rows


def main():
    all_rows = []

    all_rows += process_folder(os.path.join(DATA_DIR, "image_real"), 0, "image")
    all_rows += process_folder(os.path.join(DATA_DIR, "image_ai"), 1, "image")
    all_rows += process_folder(os.path.join(DATA_DIR, "video_real"), 0, "video")
    all_rows += process_folder(os.path.join(DATA_DIR, "video_ai"), 1, "video")

    if not all_rows:
        print("Nenhum ficheiro processado.")
        return

    df = pd.DataFrame(all_rows)

    # preencher colunas em falta, porque imagens e videos têm features diferentes
    df = df.fillna(0)

    # guardar
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nDataset guardado em {OUTPUT_CSV}")
    print(f"Total de amostras: {len(df)}")
    print("Colunas:")
    print(df.columns.tolist())
    print("\nPreview:")
    print(df.head())


if __name__ == "__main__":
    main()
