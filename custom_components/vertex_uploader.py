"""
Custom TFX component for importing a model into Vertex AI.
Author: Sayak Paul
Reference: https://github.com/GoogleCloudPlatform/mlops-with-vertex-ai/blob/31bf8a43783a8aa57e823b6441c4822d5bf80fa1/src/tfx_pipelines/components.py#L74
"""

import os
import tensorflow as tf

from tfx.dsl.component.experimental.decorators import component
from tfx.dsl.component.experimental.annotations import Parameter
from tfx.types.standard_artifacts import String
from google.cloud import aiplatform as vertex_ai
from tfx import v1 as tfx
from absl import logging


@component
def VertexUploader(
    project: Parameter[str],
    region: Parameter[str],
    model_display_name: Parameter[str],
    pushed_model_location: Parameter[str],
    serving_image_uri: Parameter[str],
    uploaded_model: tfx.dsl.components.OutputArtifact[String],
):

    vertex_ai.init(project=project, location=region)

    pushed_model_dir = os.path.join(
        pushed_model_location, tf.io.gfile.listdir(pushed_model_location)[-1]
    )

    logging.info(f"Model registry location: {pushed_model_dir}")

    vertex_model = vertex_ai.Model.upload(
        display_name=model_display_name,
        artifact_uri=pushed_model_dir,
        serving_container_image_uri=serving_image_uri,
        parameters_schema_uri=None,
        instance_schema_uri=None,
        explanation_metadata=None,
        explanation_parameters=None,
    )

    uploaded_model.set_string_custom_property(
        "model_resource_name", str(vertex_model.resource_name)
    )
    logging.info(f"Model resource: {str(vertex_model.resource_name)}")
