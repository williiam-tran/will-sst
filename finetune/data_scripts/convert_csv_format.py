#!/usr/bin/env python3
import csv
import os

def convert_csv_to_pipe_format(input_path, output_path):
    """Convert comma-separated CSV to pipe-delimited format expected by filter_data.py"""
    converted_lines = []

    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header row

        for row in reader:
            if len(row) < 2:
                continue

            filename = row[0].strip()
            text = row[1].strip()

            # Remove surrounding quotes if present
            if text.startswith('"') and text.endswith('"'):
                text = text[1:-1]

            # Replace newlines with spaces (multi-line CSV fields)
            text = ' '.join(text.split())

            # Ensure text ends with proper punctuation
            if text and text[-1] not in '.,?!':
                text += '.'

            converted_lines.append(f"{filename}|{text}\n")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(converted_lines)

    print(f"✅ Converted {len(converted_lines)} lines")
    print(f"📝 Output saved to: {output_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    dataset_dir = os.path.join(project_root, "finetune", "dataset")

    input_file = os.path.join(dataset_dir, "metadata.csv")
    output_file = os.path.join(dataset_dir, "metadata.csv")
    backup_file = os.path.join(dataset_dir, "metadata.csv.backup")

    # Backup original
    if os.path.exists(input_file):
        import shutil
        shutil.copy2(input_file, backup_file)
        print(f"📦 Backed up original to: {backup_file}")

    convert_csv_to_pipe_format(input_file, output_file)
