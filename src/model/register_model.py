
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import json
import mlflow
import mlflow.sklearn
import os
import dagshub
from src.logger import logging

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")


# FIX: Only call dagshub.init() — do NOT call mlflow.set_tracking_uri() separately.
# dagshub.init() sets the tracking URI AND configures the artifact store correctly.
# -------------------------------------------------------------------------------------
dagshub.init(repo_owner='chinmayeeM220', repo_name='Capstone-Project', mlflow=True)
# -------------------------------------------------------------------------------------


def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)

        if 'run_id' not in model_info or 'model_path' not in model_info:
            raise KeyError("experiment_info.json must contain 'run_id' and 'model_path' keys")

        logging.debug('Model info loaded from %s', file_path)
        return model_info
    except FileNotFoundError:
        logging.error('File not found: %s', file_path)
        raise
    except KeyError as e:
        logging.error('Missing key in model info: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the model info: %s', e)
        raise


def validate_run_has_model(run_id: str, model_path: str) -> None:
    """
    Verify the MLflow run exists, is FINISHED, and has the model artifact
    before attempting registration.
    """
    client = mlflow.tracking.MlflowClient()

    try:
        run = client.get_run(run_id)
    except Exception:
        raise RuntimeError(
            f"Run ID '{run_id}' not found on the MLflow tracking server. "
            "Re-run model_evaluation.py to generate a fresh run."
        )

    run_status = run.info.status
    if run_status != "FINISHED":
        raise RuntimeError(
            f"Run ID '{run_id}' has status '{run_status}' (expected FINISHED). "
            "Re-run model_evaluation.py and ensure it completes without errors."
        )

    artifacts = client.list_artifacts(run_id, path=model_path)
    if not artifacts:
        raise RuntimeError(
            f"No artifacts found at path '{model_path}' in run '{run_id}'. "
            "Re-run model_evaluation.py to generate a valid run."
        )

    logging.info(
        "Run '%s' validated: status=FINISHED, artifact path='%s' exists.",
        run_id, model_path
    )


def register_model(model_name: str, model_info: dict):
    """Register the model to the MLflow Model Registry."""
    try:
        run_id = model_info['run_id']
        model_path = model_info['model_path']

        validate_run_has_model(run_id, model_path)

        model_uri = f"runs:/{run_id}/{model_path}"

        model_version = mlflow.register_model(model_uri, model_name)
        logging.info(
            "Model '%s' version %s registered successfully.",
            model_name, model_version.version
        )

        client = mlflow.tracking.MlflowClient()

        # Use alias instead of deprecated transition_model_version_stage (MLflow 2.x+)
        client.set_registered_model_alias(
            name=model_name,
            alias="staging",
            version=model_version.version
        )

        logging.info(
            "Model '%s' version %s aliased as 'staging'.",
            model_name, model_version.version
        )

    except RuntimeError:
        raise
    except Exception as e:
        logging.error('Error during model registration: %s', e)
        raise


def main():
    try:
        model_info_path = 'reports/experiment_info.json'
        model_info = load_model_info(model_info_path)

        logging.info(
            "Attempting to register run_id='%s' with model_path='%s'",
            model_info['run_id'], model_info['model_path']
        )

        model_name = "my_model"
        register_model(model_name, model_info)

        logging.info('Model registration pipeline completed successfully.')

    except Exception as e:
        logging.error('Failed to complete the model registration process: %s', e)
        raise


if __name__ == '__main__':
    main()