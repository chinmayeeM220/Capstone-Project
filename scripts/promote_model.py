# promote model

import os
import mlflow
import dagshub


def promote_model():
    # FIX: use dagshub.init() with correct repo
    # removed wrong repo vikashdas770/YT-Capstone-Project
    dagshub.init(repo_owner='chinmayeeM220', repo_name='Capstone-Project', mlflow=True)

    client = mlflow.MlflowClient()
    model_name = "my_model"

    # FIX: get_latest_versions with stages is deprecated and removed in MLflow 2.9+
    # Use get_model_version_by_alias instead

    # Get the latest version in staging via alias
    try:
        staging_version = client.get_model_version_by_alias(model_name, "staging")
        latest_version_staging = staging_version.version
    except Exception as e:
        raise RuntimeError(
            f"No model found with alias 'staging' for '{model_name}'. "
            "Make sure model_registration ran successfully."
        ) from e

    # Archive current production model if exists
    try:
        prod_version = client.get_model_version_by_alias(model_name, "production")
        # Remove production alias from old version
        client.delete_registered_model_alias(model_name, "production")
        print(f"Removed 'production' alias from version {prod_version.version}")
    except Exception:
        # No production model exists yet — that's fine for first run
        print("No existing production model found, skipping archive step.")

    # Promote staging model to production via alias
    client.set_registered_model_alias(
        name=model_name,
        alias="production",
        version=latest_version_staging
    )
    print(f"Model version {latest_version_staging} promoted to production.")


if __name__ == "__main__":
    promote_model()