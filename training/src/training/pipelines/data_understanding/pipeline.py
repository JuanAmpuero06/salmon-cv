from kedro.pipeline import Pipeline, node, pipeline
from .nodes import create_coco_eda_samples


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=create_coco_eda_samples,
                inputs=[
                    "params:coco_json_relpath",
                    "params:frames_dirname",
                    "params:n_samples",
                    "params:seed",
                    "params:output_dir_relpath",
                ],
                outputs="eda_manifest",
                name="create_coco_eda_samples_node",
            ),
        ]
    )