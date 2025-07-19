import pandas as pd
import os
import sys
import shutil

def validate_and_merge(folder_path, merged_filename):
    # List all .xlsx files in the folder
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.xlsx')]
    if not files:
        print("No Excel files found in folder.")
        return

    validation_results = []
    valid_dfs = []

    print(f"Found {len(files)} Excel files. Validating...")

    for file in files:
        try:
            df = pd.read_excel(file)
            if df.empty:
                validation_results.append((os.path.basename(file), "⚠️ Empty file"))
            else:
                validation_results.append((os.path.basename(file), f"✅ {df.shape[0]} rows, {df.shape[1]} columns"))
                df['source_file'] = os.path.basename(file)
                valid_dfs.append(df)
        except Exception as e:
            validation_results.append((os.path.basename(file), f"❌ Error reading: {e}"))

    print("\n📊 Validation Results:")
    for fname, status in validation_results:
        print(f"- {fname}: {status}")

    if not valid_dfs:
        print("No valid files to merge. Exiting.")
        return

    merged_df = pd.concat(valid_dfs, ignore_index=True)
    merged_path = os.path.join(folder_path, merged_filename)

    merged_df.to_excel(merged_path, index=False)
    print(f"\n✅ Merged file saved as: {merged_path}")

    # Move original files to archive
    archive_folder = os.path.join(folder_path, "archive")
    os.makedirs(archive_folder, exist_ok=True)

    for file in files:
        shutil.move(file, os.path.join(archive_folder, os.path.basename(file)))

    print(f"📁 All original files moved to: {archive_folder}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python validate_merge_archive.py <folder_path> <merged_filename.xlsx>")
        sys.exit(1)

    folder_path = sys.argv[1]
    merged_filename = sys.argv[2]

    validate_and_merge(folder_path, merged_filename)
