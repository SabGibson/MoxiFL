import mlflow


def create_or_retrieve_experiment(
    exp_name: str, artifact_location: str, tags: dict[str, any]
) -> str:
    """create or retrieve experiement"""
    try:
        experiment_id = mlflow.create_experiment(
            name=exp_name, artifact_location=artifact_location, tags=tags
        )
    except:
        print(f"Experiment {exp_name} already exists!")
        print("Attempting retreival!")
        experiment_id = mlflow.get_experiment_by_name(name=exp_name)

    return experiment_id
