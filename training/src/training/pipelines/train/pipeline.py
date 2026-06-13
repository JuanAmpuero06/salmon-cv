from kedro.pipeline import Pipeline, node, pipeline
from .nodes import train_yolo_detector


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=train_yolo_detector,
                inputs=[
                    "params:data_yaml_relpath",
                    "params:model_name",
                    "params:imgsz",
                    "params:epochs",
                    "params:batch",
                    "params:workers",
                    "params:patience",
                    "params:device",
                    "params:pretrained",
                    "params:project_relpath",
                    "params:run_name",
                    "params:save",
                    "params:save_period",
                    "params:plots",
                    "params:verbose",
                    "params:resume",
                    "params:resume_checkpoint_relpath",
                ],
                outputs="train_run_manifest",
                name="train_yolo_detector_node",
            ),
        ]
    )