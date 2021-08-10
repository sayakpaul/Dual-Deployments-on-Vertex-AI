"""
Custom TFX component for Firebase upload.
Author: Chansung Park
"""

from tfx import types
from tfx.dsl.component.experimental.decorators import component
from tfx.dsl.component.experimental.annotations import Parameter
from tfx import v1 as tfx
from absl import logging

import firebase_admin
from firebase_admin import ml
from firebase_admin import storage
from firebase_admin import credentials
from google.cloud import storage as gcs_storage


@component
def FirebasePublisher(
    pushed_model: tfx.dsl.components.InputArtifact[
        tfx.types.standard_artifacts.PushedModel
    ],
    credential_uri: Parameter[str],
    firebase_dest_gcs_bucket: Parameter[str],
    model_display_name: Parameter[str],
    model_tag: Parameter[str],
) -> tfx.dsl.components.OutputDict(result=str):
    """
    publish trained tflite model to Firebase ML, this component assumes that 
    trained model and Firebase credential files are stored in GCS locations.
    
    Args:   
        pushed_model: The URI of pushed model obtained from previous component (i.e. Pusher)
        credential_uri: The URI of Firebase credential. In order to get one, go to Firebase dashboard 
            and on the Settings page, create a service account and download the service account key file. 
            Keep this file safe, since it grants administrator access to your project.
        firebase_dest_gcs_bucket: GCS bucket where the model is going to be temporarily stored.
            In order to create one, go to Firebase dashboard and on the Storage page, enable Cloud Storage. 
            Take note of your bucket name.
        model_display_name: The name to be appeared on Firebase ML dashboard
        model_tag: The tage name to be appeared on Firebase ML dashboard
    """
    
    model_uri = f"{pushed_model.uri}/model.tflite"
    
    assert model_uri.split("://")[0] == "gs"
    assert credential_uri.split("://")[0] == "gs"

    # create gcs client instance
    gcs_client = gcs_storage.Client()

    # get credential for firebase
    credential_gcs_bucket = credential_uri.split("//")[1].split("/")[0]
    credential_blob_path = "/".join(credential_uri.split("//")[1].split("/")[1:])

    bucket = gcs_client.bucket(credential_gcs_bucket)
    blob = bucket.blob(credential_blob_path)
    blob.download_to_filename("credential.json")
    logging.info(f"download credential.json from {credential_uri} is completed")

    # get tflite model file
    tflite_gcs_bucket = model_uri.split("//")[1].split("/")[0]
    tflite_blob_path = "/".join(model_uri.split("//")[1].split("/")[1:])

    bucket = gcs_client.bucket(tflite_gcs_bucket)
    blob = bucket.blob(tflite_blob_path)
    blob.download_to_filename("model.tflite")
    logging.info(f"download model.tflite from {model_uri} is completed")

    firebase_admin.initialize_app(
        credentials.Certificate("credential.json"),
        options={"storageBucket": firebase_dest_gcs_bucket},
    )
    logging.info("firebase_admin initialize app is completed")

    model_list = ml.list_models(list_filter=f"display_name={model_display_name}")
    # update
    if len(model_list.models) > 0:
        # get the first match model
        model = model_list.models[0]
        source = ml.TFLiteGCSModelSource.from_tflite_model_file("model.tflite")
        model.model_format = ml.TFLiteFormat(model_source=source)

        updated_model = ml.update_model(model)
        ml.publish_model(updated_model.model_id)

        logging.info("model exists, so update it in FireBase ML")
        return {"result": "model updated"}
    # create
    else:
        # load a tflite file and upload it to Cloud Storage
        source = ml.TFLiteGCSModelSource.from_tflite_model_file("model.tflite")

        # create the model object
        tflite_format = ml.TFLiteFormat(model_source=source)
        model = ml.Model(
            display_name=model_display_name,
            tags=[model_tag],
            model_format=tflite_format,
        )

        # Add the model to your Firebase project and publish it
        new_model = ml.create_model(model)
        ml.publish_model(new_model.model_id)

        logging.info("model doesn exists, so create one in FireBase ML")
        return {"result": "model created"}
