import pandas as pd
import yaml
import os

def generate_cqs_yml(input_excel, output_folder):
    """
    Generate a cqs.yml file from an Excel file.

    :param input_excel: Path to the input Excel file.
    :param output_folder: Path to the folder where the cqs.yml file will be saved.
    """
    # Load the Excel file
    df = pd.read_excel(input_excel)

    # Check if required columns exist
    if 'Competency Questions' not in df.columns or 'Ontology_file_name' not in df.columns:
        raise ValueError("The Excel file must contain 'Competency Questions' and 'Ontology_file_name' columns.")

    # Group data by ontology file name
    grouped = df.groupby('Ontology_file_name')

    for ontology_file, group in grouped:
        # Extract the ontology name (remove .owl extension)
        ontology_name = os.path.splitext(ontology_file)[0]

        # Prepare the data for the YAML file
        cqs_data = {
            'ontology': {
                'name': ontology_name,
                'cqs': []
            }
        }

        # Add questions to the YAML structure
        for idx, row in group.iterrows():
            question_id = f"{ontology_name}_{idx + 1}"
            question_text = row['Competency Questions']
            cqs_data['ontology']['cqs'].append({
                'ID': question_id,
                'question': question_text
            })

        # Create the output folder if it doesn't exist
        output_path = os.path.join(output_folder, ontology_name, 'cqs')
        os.makedirs(output_path, exist_ok=True)

        # Write the YAML file
        output_file = os.path.join(output_path, 'cqs.yml')
        with open(output_file, 'w') as f:
            yaml.dump(cqs_data, f, default_flow_style=False, sort_keys=False)

        print(f"Generated: {output_file}")


if __name__ == "__main__":
    # Input Excel file path
    input_excel = "CQs.xlsx"

    # Output folder path
    output_folder = "./data"

    # Generate cqs.yml files
    generate_cqs_yml(input_excel, output_folder)