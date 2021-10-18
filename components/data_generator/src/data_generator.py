import fire
import pandas as pd
from sklearn.model_selection import train_test_split

TITANIC_URL = (
    "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
)


def fetch_dataset(train_data_path: str, valid_data_path: str) -> None:
    df = pd.read_csv(TITANIC_URL, index_col=0)
    train, valid = train_test_split(df, test_size=0.2)
    train.to_csv(train_data_path)
    valid.to_csv(valid_data_path)


if __name__ == "__main__":
    fire.Fire(fetch_dataset)
