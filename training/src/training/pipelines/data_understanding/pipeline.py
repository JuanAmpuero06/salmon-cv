from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    create_coco_eda_samples,
    summarize_coco_clip,
    summarize_dataset_clips,
)


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
            node(
                func=summarize_coco_clip,
                inputs=[
                    "params:coco_json_relpath",
                    "params:frames_dirname",
                ],
                outputs="clip_summary",
                name="summarize_coco_clip_node",
            ),
            node(
                func=summarize_dataset_clips,
                inputs=[
                    "params:annotated_data_root_relpath",
                    "params:frames_dirname",
                ],
                outputs="dataset_clip_inventory",
                name="summarize_dataset_clips_node",
            ),
        ]
    )