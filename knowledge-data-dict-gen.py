import pandas as pd
import json

def load_jsonl_to_df(file_path):
    """
    Load a JSONL file and flatten the 'metadata' column if it exists.
    """
    # Read the file using pandas read_json with lines=True
    df = pd.read_json(file_path, lines=True)
    
    # If there's a "metadata" column, normalize its contents into separate columns.
    if 'metadata' in df.columns:
        # Normalize the metadata column
        metadata_df = pd.json_normalize(df['metadata'])
        # Combine with the rest of the DataFrame (dropping the original metadata column)
        df = pd.concat([df.drop(columns=['metadata']), metadata_df], axis=1)
    
    return df

# Load each file into its own DataFrame.
# Adjust the file paths as needed.
df_attributes_flat = load_jsonl_to_df("scripts/process_jsonl/attributes_flat.jsonl")
df_attributes_grouped = load_jsonl_to_df("scripts/process_jsonl/attributes_grouped.jsonl")
df_courses_term_1 = load_jsonl_to_df("scripts/process_jsonl/courses_term.jsonl")
df_courses_term_2 = load_jsonl_to_df("scripts/process_jsonl/courses.jsonl")   # e.g., the file with "GLBS" in the id
df_programs = load_jsonl_to_df("scripts/process_jsonl/programs.jsonl")

# Print the first few rows of each DataFrame for inspection.
print("Attributes Flat DataFrame:")
print(df_attributes_flat.head(), "\n")

print("Attributes Grouped DataFrame:")
print(df_attributes_grouped.head(), "\n")

print("Courses Term DataFrame (AAMW):")
print(df_courses_term_1.head(), "\n")

print("Courses Term DataFrame (GLBS):")
print(df_courses_term_2.head(), "\n")

print("Programs DataFrame:")
print(df_programs.head())