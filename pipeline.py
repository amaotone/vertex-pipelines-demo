import os
from string import Template
from pathlib import Path
import kfp
from kfp.v2.compiler import Compiler

NAME = "titanic-pipeline-demo"
GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
ARTIFACT_REGISTRY_ENDPOINT = os.environ.get("ARTIFACT_REGISTRY_ENDPOINT")
ARTIFACT_REGISTRY_REPOSITORY = os.environ.get("ARTIFACT_REGISTRY_REPOSITORY")
GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME")

ROOT_DIR = Path(__file__).parent


def load_spec(name: str) -> str:
    path = ROOT_DIR / "components" / name / f"{name}.yml"
    txt = Template(path.read_text())
    image = f"{ARTIFACT_REGISTRY_ENDPOINT}/{GCP_PROJECT_ID}/{ARTIFACT_REGISTRY_REPOSITORY}/{name}"
    return txt.substitute(image=image)


def get_data_generator_op() -> kfp.dsl.ContainerOp:
    txt = load_spec("data_generator")
    op = kfp.components.load_component_from_text(txt)
    return op()


def get_preprocessor_op(
    train_data_path: str, valid_data_path: str
) -> kfp.dsl.ContainerOp:
    txt = load_spec("preprocessor")
    op = kfp.components.load_component_from_text(txt)
    return op(train_data_path=train_data_path, valid_data_path=valid_data_path)


def get_trainer_op(processed_train_data_path: str) -> kfp.dsl.ContainerOp:
    txt = load_spec("trainer")
    op = kfp.components.load_component_from_text(txt)
    return op(processed_train_data_path=processed_train_data_path)


def get_evaluator_op(
    model_path: str, processed_valid_data_path: str
) -> kfp.dsl.ContainerOp:
    txt = load_spec("evaluator")
    op = kfp.components.load_component_from_text(txt)
    return op(
        model_path=model_path, processed_valid_data_path=processed_valid_data_path
    )


@kfp.dsl.pipeline(name=NAME, pipeline_root=f"gs://{GCS_BUCKET_NAME}")
def titanic_pipeline():
    data_generator = get_data_generator_op().set_cpu_limit("1")
    preprocessor = get_preprocessor_op(
        train_data_path=data_generator.outputs["train_data_path"],
        valid_data_path=data_generator.outputs["valid_data_path"],
    )
    trainer = get_trainer_op(
        processed_train_data_path=preprocessor.outputs["processed_train_data_path"]
    )
    evaluator = get_evaluator_op(
        model_path=trainer.outputs["model_path"],
        processed_valid_data_path=preprocessor.outputs["processed_valid_data_path"],
    )


Compiler().compile(titanic_pipeline, "pipeline.json")
