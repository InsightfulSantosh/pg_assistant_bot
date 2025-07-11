import pandas as pd
import os

def load_and_lowercase_csv(input_path, output_folder="data/formated_data"):
    # Load the CSV
    df = pd.read_csv(input_path)
    
    # Convert column names to lowercase
    df.columns = df.columns.str.lower()
    
    # Convert all string values to lowercase
    df = df.applymap(lambda x: x.lower() if isinstance(x, str) else x)

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Define the output file name
    file_name = os.path.basename(input_path)
    output_path = os.path.join(output_folder, file_name)

    # Save the modified DataFrame
    df.to_csv(output_path, index=False)

    print(f"File saved to: {output_path}")
    return df

df = load_and_lowercase_csv("data/raw/professionals_in_pg.csv")
