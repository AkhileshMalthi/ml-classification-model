"""
Script to compare experiment runs and promote the best model to Production stage.
"""

import os
import mlflow
from mlflow import MlflowClient

def promote_best_model(
    model_name="ClassificationModel",
    metric_name="f1_score",
    mlflow_tracking_uri=None
):
    """
    Compare all model versions and promote the best one to Production stage.
    
    Args:
        model_name: Name of the registered model
        metric_name: Metric to use for comparison (f1_score, accuracy, etc.)
        mlflow_tracking_uri: MLflow tracking server URI
    """
    if mlflow_tracking_uri is None:
        mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    client = MlflowClient()
    
    print(f"Analyzing model: {model_name}")
    print(f"Selecting best model based on: {metric_name}\n")
    
    try:
        # Get all versions of the registered model
        model_versions = client.search_model_versions(f"name='{model_name}'")
        
        if not model_versions:
            print(f"No versions found for model '{model_name}'")
            return
        
        print(f"Found {len(model_versions)} model version(s)")
        print("="*80)
        
        best_version = None
        best_metric_value = -float('inf')
        
        # Evaluate each version
        for version in model_versions:
            version_num = version.version
            run_id = version.run_id
            
            # Get the run to access metrics
            run = client.get_run(run_id)
            metrics = run.data.metrics
            
            if metric_name in metrics:
                metric_value = metrics[metric_name]
                print(f"Version {version_num}: {metric_name}={metric_value:.4f} | Stage: {version.current_stage}")
                
                if metric_value > best_metric_value:
                    best_metric_value = metric_value
                    best_version = version
            else:
                print(f"Version {version_num}: {metric_name} not found | Stage: {version.current_stage}")
        
        print("="*80)
        
        if best_version is None:
            print(f"Could not find any version with metric '{metric_name}'")
            return
        
        print(f"\nBest model: Version {best_version.version} with {metric_name}={best_metric_value:.4f}")
        
        # Archive current Production models
        for version in model_versions:
            if version.current_stage == "Production":
                print(f"Archiving current Production version {version.version}...")
                client.transition_model_version_stage(
                    name=model_name,
                    version=version.version,
                    stage="Archived"
                )
        
        # Promote best model to Production
        print(f"Promoting version {best_version.version} to Production stage...")
        client.transition_model_version_stage(
            name=model_name,
            version=best_version.version,
            stage="Production",
            archive_existing_versions=True
        )
        
        print(f"\n✓ Successfully promoted version {best_version.version} to Production!")
        print(f"Model URI: models:/{model_name}/Production")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Promote best model to Production stage")
    parser.add_argument(
        "--model-name",
        default="ClassificationModel",
        help="Name of the registered model (default: ClassificationModel)"
    )
    parser.add_argument(
        "--metric",
        default="f1_score",
        help="Metric to use for comparison (default: f1_score)"
    )
    parser.add_argument(
        "--mlflow-uri",
        default=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
        help="MLflow tracking URI (default: http://localhost:5000)"
    )
    
    args = parser.parse_args()
    
    promote_best_model(
        model_name=args.model_name,
        metric_name=args.metric,
        mlflow_tracking_uri=args.mlflow_uri
    )
