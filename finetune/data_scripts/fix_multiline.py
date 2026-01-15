#!/usr/bin/env python3
import os

def fix_multiline_metadata(input_path):
    """Fix multi-line text in pipe-delimited metadata"""
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split by the filename pattern to identify each entry
    lines = content.strip().split('\n')

    # Assuming only one audio file entry that spans multiple lines
    # Find the line with the pipe delimiter (filename|text)
    output_lines = []
    current_line = []

    for line in lines:
        if line.strip():  # Skip empty lines
            current_line.append(line.strip())

    # Join all non-empty lines into one, replacing newlines with spaces
    if current_line:
        # The first part before | is the filename, rest is text
        full_line = ' '.join(current_line)
        output_lines.append(full_line)

    # Write back
    with open(input_path, 'w', encoding='utf-8') as f:
        for line in output_lines:
            f.write(line + '\n')

    print(f"✅ Fixed multi-line metadata")
    print(f"📝 Output saved to: {input_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    dataset_dir = os.path.join(project_root, "finetune", "dataset")

    metadata_file = os.path.join(dataset_dir, "metadata.csv")
    fix_multiline_metadata(metadata_file)
