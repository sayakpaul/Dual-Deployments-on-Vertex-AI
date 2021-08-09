"""
Custom TFX component for deploying a model to a Vertex AI Endpoint.
Author: Sayak Paul
Reference: https://github.com/GoogleCloudPlatform/mlops-with-vertex-ai/blob/31bf8a43783a8aa57e823b6441c4822d5bf80fa1/build/utils.py#L97
"""

from tfx.dsl.component.experimental.decorators import component
from tfx.dsl.component.experimental.annotations import Parameter
from tfx.types.standard_artifacts import String
from google.cloud import aiplatform as vertex_ai
from tfx import v1 as tfx
from absl import logging


@component
def VertexDeployer(
    project: Parameter[str],
    region: Parameter[str],
    model_display_name: Parameter[str],
    deployed_model_display_name: Parameter[str],
):

    logging.info(f"Endpoint display: {deployed_model_display_name}")
    vertex_ai.init(project=project, location=region)

    endpoints = vertex_ai.Endpoint.list(
        filter=f"display_name={deployed_model_display_name}", order_by="update_time"
    )

    if len(endpoints) > 0:
        logging.info(f"Endpoint {deployed_model_display_name} already exists.")
        endpoint = endpoints[-1]
    else:
        endpoint = vertex_ai.Endpoint.create(deployed_model_display_name)

    model = vertex_ai.Model.list(
        filter=f"display_name={model_display_name}", order_by="update_time"
    )[-1]

    endpoint = vertex_ai.Endpoint.list(
        filter=f"display_name={deployed_model_display_name}", order_by="update_time"
    )[-1]

    deployed_model = endpoint.deploy(
        model=model,
        # Syntax from here: https://git.io/JBQDP
        traffic_split={"0": 100},
        machine_type="n1-standard-4",
        min_replica_count=1,
        max_replica_count=1,
    )

    logging.info(f"Model deployed to: {deployed_model}")
