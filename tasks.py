from invoke import task
from pathlib import Path
from threading import Thread
import os
from kfp.v2.google.client import AIPlatformClient


DOCKER_IMAGE_PREFIX = "titanic"
REGION = os.environ.get("REGION")
GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
ARTIFACT_REGISTRY_ENDPOINT = os.environ.get("ARTIFACT_REGISTRY_ENDPOINT")
ARTIFACT_REGISTRY_REPOSITORY = os.environ.get("ARTIFACT_REGISTRY_REPOSITORY")


@task
def build(c, name):
    root_dir = Path(__file__).parent
    target_dir = root_dir / "components" / name
    if not target_dir.exists():
        raise FileNotFoundError(f"{target_dir} not found.")
    image_name = "_".join((DOCKER_IMAGE_PREFIX, name))
    tag = f"{ARTIFACT_REGISTRY_ENDPOINT}/{GCP_PROJECT_ID}/{ARTIFACT_REGISTRY_REPOSITORY}/{name}"
    c.run(f"docker build {target_dir} -t {image_name}")
    c.run(f"docker tag {image_name} {tag}")
    c.run(f"docker push {tag}")


@task
def build_all(c, parallel=False):
    names = ["data_generator", "preprocessor", "trainer", "evaluator"]
    if parallel:
        threads = [Thread(target=build, args=[c, name]) for name in names]
        [x.start() for x in threads]
        [x.join() for x in threads]
    else:
        for name in names:
            build(c, name)


@task
def compile(c):
    c.run("dsl-compile --py pipeline.py --output pipeline.yml")


@task
def run(c, spec_path):
    client = AIPlatformClient(project_id=GCP_PROJECT_ID, region=REGION)
    run_name = client.create_run_from_job_spec(spec_path, enable_caching=True)
