We expose our notebooks that are fully runnable on [Google Colab](https://colab.research.google.com/) to facilitate easier learning. Below
you can find short descriptions of the individual notebooks present here.

## Notebooks

| <h4>Notebook</h4> | <h4>Colab</h4>     | <h4>Description</h4>                       |
| :-------- | :------- | :-------------------------------- |
| `Dataset_Prep.ipynb` | <a href="https://colab.research.google.com/github/sayakpaul/Dual-Deployments-on-Vertex-AI/blob/main/notebooks/Dataset_Prep.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | Prepares dataset for AutoML. |
| `Dual_Deployments_With_AutoML.ipynb` | <a href="https://colab.research.google.com/github/sayakpaul/Dual-Deployments-on-Vertex-AI/blob/main/notebooks/Dual_Deployments_With_AutoML.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | Shows how to build a Kubeflow Pipeline <br> to train and deploy two different models <br> using AutoML and Vertex AI. |
| `Model_Tests.ipynb` | <a href="https://colab.research.google.com/github/sayakpaul/Dual-Deployments-on-Vertex-AI/blob/main/notebooks/Model_Tests.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | Shows how to test the models trained <br> using the notebook above in a <br> standalone manner. |
| `Custom_Model_TFX.ipynb` | <a href="https://colab.research.google.com/github/sayakpaul/Dual-Deployments-on-Vertex-AI/blob/main/notebooks/Custom_Model_TFX.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | Shows how to build a TFX pipeline using <br> custom components to train and deploy <br> two different models and run them using Vertex AI. |

## References

* [Kubeflow Pipelines](https://www.kubeflow.org/docs/components/pipelines/overview/pipelines-overview/)
* [AutoML SDKs from Kubeflow](https://google-cloud-pipeline-components.readthedocs.io/en/latest/google_cloud_pipeline_components.aiplatform.html#module-google_cloud_pipeline_components.aiplatform)
* [Vertex AI Pipelines](https://cloud.google.com/vertex-ai/docs/pipelines)
* [TensorFlow Extended (TFX)](https://www.tensorflow.org/tfx)
