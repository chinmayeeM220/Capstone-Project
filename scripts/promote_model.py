import os
import mlflow


def promote_model():
    # Use env vars directly — dagshub.init() causes browser auth prompt in CI
    dagshub_token = os.getenv("CAPSTONE_TEST")
    if not dagshub_token:
        raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
    mlflow.set_tracking_uri("https://dagshub.com/chinmayeeM220/Capstone-Project.mlflow")

    client = mlflow.MlflowClient()
    model_name = "my_model"

    try:
        staging_version = client.get_model_version_by_alias(model_name, "staging")
        latest_version_staging = staging_version.version
    except Exception as e:
        raise RuntimeError(
            f"No model found with alias 'staging' for '{model_name}'. "
            "Make sure model_registration ran successfully."
        ) from e

    try:
        prod_version = client.get_model_version_by_alias(model_name, "production")
        client.delete_registered_model_alias(model_name, "production")
        print(f"Removed 'production' alias from version {prod_version.version}")
    except Exception:
        print("No existing production model found, skipping archive step.")

    client.set_registered_model_alias(
        name=model_name,
        alias="production",
        version=latest_version_staging
    )
    print(f"Model version {latest_version_staging} promoted to production.")


if __name__ == "__main__":
    promote_model()
