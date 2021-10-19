import fire
import pandas as pd
from pathlib import Path


def preprocessing(
    train_data_path: str,
    valid_data_path: str,
    processed_train_data_path: str,
    processed_valid_data_path: str,
) -> None:
    train = pd.read_csv(train_data_path, index_col=0)
    valid = pd.read_csv(valid_data_path, index_col=0)

    # new feature
    train["FamilySize"] = train["SibSp"] + train["Parch"] + 1
    valid["FamilySize"] = valid["SibSp"] + valid["Parch"] + 1

    # drop columns
    drop_cols = ["Name", "Ticket", "Cabin"]
    train.drop(drop_cols, axis=1, inplace=True)
    valid.drop(drop_cols, axis=1, inplace=True)

    # dump
    Path(processed_train_data_path).parent.mkdir(parents=True, exist_ok=True)
    train.to_csv(processed_train_data_path)
    Path(processed_valid_data_path).parent.mkdir(parents=True, exist_ok=True)
    valid.to_csv(processed_valid_data_path)


if __name__ == "__main__":
    fire.Fire(preprocessing)
