import fire
import pandas as pd
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import HistGradientBoostingClassifier

TARGET_NAME = "Survived"


def train(
    processed_train_data_path: str,
    model_path: str,
) -> None:
    train = pd.read_csv(processed_train_data_path, index_col=0)
    X_train = train.drop(TARGET_NAME, axis=1)
    y_train = train[TARGET_NAME]

    # train
    model = make_pipeline(LabelEncoder(), HistGradientBoostingClassifier())
    model.fit(X_train, y_train)

    # dump
    joblib.dump(model, model_path)


if __name__ == "__main__":
    fire.Fire(train)
