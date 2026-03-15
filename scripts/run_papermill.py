import papermill as pm
import os
import sys

def main():
    notebooks_dir = "notebooks"
    notebooks_to_run = [
        "01_eda_preprocessing.ipynb",
        "02_mining_clustering.ipynb",
        "03_modeling_classification.ipynb",
        "04_modeling_regression_timeseries.ipynb",
        "05_evaluation_report.ipynb"
    ]

    for nb in notebooks_to_run:
        input_path = os.path.join(notebooks_dir, nb)
        # Avoid overwriting the raw notebook unless specified. Here we execute in place or create a copy.
        # However, for simplicity, we execute and save it back to itself or to an executed version.
        # Standard practice is to add _executed to the filename, but since the assignment might expect
        # the notebooks themselves to contain the outputs, we can overwrite or create new ones.
        output_path = os.path.join(notebooks_dir, nb.replace(".ipynb", "_executed.ipynb"))
        
        print(f"Running {nb}...")
        try:
            pm.execute_notebook(
                input_path,
                output_path,
                kernel_name='python3',
                cwd=notebooks_dir
                # parameters=dict(seed=42) # Can pass parameters if parameterized via papermill
            )
            print(f"Successfully finished {nb}.")
        except Exception as e:
            print(f"Error executing {nb}: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()
