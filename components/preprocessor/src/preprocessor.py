import fire
import pandas as pd


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
    drop_cols = ["Name", "Cabin", "Cabin"]
    train.drop(drop_cols, axis=1, inplace=True)
    valid.drop(drop_cols, axis=1, inplace=True)

    # dump
    train.to_csv(processed_train_data_path)
    valid.to_csv(processed_valid_data_path)


if __name__ == "__main__":
    fire.Fire(preprocessing)
