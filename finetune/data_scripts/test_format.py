#!/usr/bin/env python3
import os
import re

ACRONYM = re.compile(r"(?:[a-zA-Z]\.){2,}")
ACRONYM_NO_PERIOD = re.compile(r"(?:[A-Z]){2,}")

def text_filter(text: str) -> bool:
    if not text:
        return False
    if re.search(r"\d", text):
        return False
    if ACRONYM.search(text) or ACRONYM_NO_PERIOD.search(text):
        return False
    if text[-1] not in ".,?!":
        return False
    return True

# Test the parsing
metadata_path = "finetune/dataset/metadata.csv"

with open(metadata_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

print(f"📄 Found {len(lines)} lines in metadata.csv\n")

for i, line in enumerate(lines, 1):
    parts = line.strip().split('|')
    print(f"Line {i}:")
    print(f"  Parts: {len(parts)}")

    if len(parts) >= 2:
        filename = parts[0]
        text = parts[1]
        print(f"  Filename: {filename}")
        print(f"  Text length: {len(text)} chars")
        print(f"  Text preview: {text[:100]}...")
        print(f"  Text ends with: '{text[-1]}'")
        print(f"  Passes text_filter: {text_filter(text)}")

        if not text_filter(text):
            print(f"  ❌ Filter reasons:")
            if not text:
                print(f"     - Empty text")
            if re.search(r"\d", text):
                print(f"     - Contains digits")
            if ACRONYM.search(text) or ACRONYM_NO_PERIOD.search(text):
                print(f"     - Contains acronyms")
            if text[-1] not in ".,?!":
                print(f"     - Doesn't end with punctuation")
    else:
        print(f"  ⚠️  Invalid format (expected 2+ parts)")
    print()
