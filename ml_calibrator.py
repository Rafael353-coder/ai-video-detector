import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from xgboost import XGBClassifier


CSV_PATH = "features.csv"


def train_one_model(df: pd.DataFrame, model_path: str, model_name: str, is_video: bool):
    df = df.copy()

    df = df.drop(columns=["file"], errors="ignore")

    if "type" in df.columns:
        df = df.drop(columns=["type"], errors="ignore")

    if "label" not in df.columns:
        raise ValueError("A coluna 'label' nao existe.")

    X = df.drop("label", axis=1)
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"\n===== {model_name} =====")
    print(f"Amostras: {len(df)}")
    print(f"Features: {list(X.columns)}")

    if is_video:
        model = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss"
        )
    else:
        model = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss"
        )

    print("A treinar modelo...")
    model.fit(X_train, y_train)

    print("A avaliar modelo...")
    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred, zero_division=0)
    rec = recall_score(y_test, pred, zero_division=0)
    f1 = f1_score(y_test, pred, zero_division=0)
    cm = confusion_matrix(y_test, pred)

    print(f"Accuracy : {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall   : {rec:.3f}")
    print(f"F1-score : {f1:.3f}")
    print("Matriz de confusao:")
    print(cm)

    joblib.dump(model, model_path)
    print(f"Modelo guardado em {model_path}")


def main():
    print("A ler dataset...")
    df = pd.read_csv(CSV_PATH)

    if "type" not in df.columns:
        raise ValueError("A coluna 'type' nao existe em features.csv")

    df_image = df[df["type"] == "image"].copy()
    df_video = df[df["type"] == "video"].copy()

    if len(df_image) == 0:
        raise ValueError("Nao existem amostras de imagem no dataset.")
    if len(df_video) == 0:
        raise ValueError("Nao existem amostras de video no dataset.")

    train_one_model(df_image, "model_image.pkl", "MODELO DE IMAGEM", is_video=False)
    train_one_model(df_video, "model_video.pkl", "MODELO DE VIDEO", is_video=True)


if __name__ == "__main__":
    main()
