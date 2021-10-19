# Vertex Pipelines Demo

## Usage

```
$ poetry install 
$ poetry run inv build-all compile run pipeline.json
```

## Components

## data_generator
- input
    - None
- output
    - train_data_path: str
    - valid_data_path: str

## preprocessor

- input
    - train_data_path: str
    - valid_data_path: str
- output
    - processed_train_data_path: str
    - processed_valid_data_path: str

## trainer

- input
    - processed_train_data_path: str
- output
    - model_path: str

## evaluator

- input
    - model_path: str
    - processed_valid_data_path: str
- output
    - accuracy
    - confusion matrix
    - classification report
