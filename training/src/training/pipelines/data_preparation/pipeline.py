from kedro.pipeline import Pipeline, node, pipeline

from .nodes import prepare_yolo_dataset


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=prepare_yolo_dataset,
                inputs=[
                    "params:annotated_data_root_relpath",
                    "params:output_yolo_root_relpath",
                    "params:target_classes",
                    "params:class_to_idx",
                    "params:split",
                    "params:seed",
                    "params:frames_dirname",
                    "params:copy_images",
                ],
                outputs="yolo_dataset_manifest",
                name="prepare_yolo_dataset_node",
            ),
        ]
    )