#!/usr/bin/env python3
import re

ACRONYM = re.compile(r"(?:[a-zA-Z]\.){2,}")
ACRONYM_NO_PERIOD = re.compile(r"(?:[A-Z]){2,}")

metadata_path = "finetune/dataset/metadata.csv"

with open(metadata_path, 'r', encoding='utf-8') as f:
    line = f.readline()
    parts = line.strip().split('|')
    text = parts[1] if len(parts) >= 2 else ""

# Check for digits
digit_matches = re.findall(r"\d+", text)
print(f"🔢 Digit matches: {digit_matches}")
print(f"   Total: {len(digit_matches)}\n")

# Check for acronyms with periods
acronym_matches = ACRONYM.findall(text)
print(f"📝 Acronym matches (with periods): {acronym_matches}")
print(f"   Total: {len(acronym_matches)}\n")

# Check for acronyms without periods (consecutive capitals)
acronym_no_period_matches = ACRONYM_NO_PERIOD.findall(text)
print(f"🔠 Acronym matches (no periods, 2+ capitals): {acronym_no_period_matches}")
print(f"   Total: {len(acronym_no_period_matches)}")
