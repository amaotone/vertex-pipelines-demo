from pathlib import Path
import fire
import pandas as pd
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
import category_encoders as ce

TARGET_NAME = "Survived"


def train(
    processed_train_data_path: str,
    model_path: str,
) -> None:
    train = pd.read_csv(processed_train_data_path, index_col=0)
    X_train = train.drop(TARGET_NAME, axis=1)
    y_train = train[TARGET_NAME]

    # train
    model = make_pipeline(ce.OrdinalEncoder(), HistGradientBoostingClassifier())
    model.fit(X_train, y_train)

    # dump
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)


if __name__ == "__main__":
    fire.Fire(train)
