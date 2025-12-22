import json
import numpy as np
import pandas as pd
from zenml import step
from zenml.integrations.mlflow.services import MLFlowDeploymentService


@step(enable_cache=False)
def predictor(
        service: MLFlowDeploymentService,
        input_data: str,
) -> np.ndarray:
    """Run an inference request against a prediction service.

    Args:
        service (MLFlowDeploymentService): The deployed MLFlow service for prediction.
        input_data (str): The input data as a JSON string.

    Returns:
        np.ndarray: The model's prediction.
    """
    # Start the service
    service.start(timeout=10)

    # Parse the JSON string
    data = json.loads(input_data)

    # Extract data from split-oriented JSON
    if "data" in data and "columns" in data:
        df = pd.DataFrame(data["data"], columns=data["columns"])
    else:
        raise ValueError(f"Expected JSON with 'data' and 'columns' keys, got: {data.keys()}")

    print(f"Input DataFrame shape: {df.shape}")
    print(f"Input DataFrame columns: {df.columns.tolist()}")

    # MLflow typically expects data in records format or as a numpy array
    # Try converting to a list of dictionaries (records format)
    prediction_data = df.to_dict(orient="records")

    print(f"Sending {len(prediction_data)} records for prediction")

    try:
        # Try records format first (most common for MLflow)
        prediction = service.predict(prediction_data)
    except Exception as e:
        print(f"Records format failed: {e}")
        print("Trying numpy array format...")
        # Fallback to numpy array
        prediction = service.predict(df.values)

    print(f"Prediction result: {prediction}")

    return np.array(prediction)