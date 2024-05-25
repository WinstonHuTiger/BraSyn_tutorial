"""MLCube handler file"""
import typer
from generate_missing_modality import infer as infer_mri
app = typer.Typer()
import yaml


@app.command("infer")
def infer(
    data_path: str = typer.Option(..., "--data_path"),
    parameters_file: str = typer.Option(..., "--parameters_file"),
    output_path: str = typer.Option(..., "--output_path"),
    
    # # Provide additional parameters as described in the mlcube.yaml file
    # # e.g. model weights:
    weights: str = typer.Option(..., "--weights"),
   
):
    # Modify the infer command as needed
    
    with open(parameters_file) as f:
        parameters = yaml.safe_load(f)
    
    infer_mri(data_path, output_path, parameters, weights)
    


@app.command("hotfix")
def hotfix():
    # NOOP command for typer to behave correctly. DO NOT REMOVE OR MODIFY
    pass


if __name__ == "__main__":
    app()
