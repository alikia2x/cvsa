import random

# File paths
input_file = 'data/filter/test.jsonl'
output_file = 'data/filter/test_filtered.jsonl'
removed_lines_file = 'data/filter/removed_lines.jsonl'

# Read all lines from the input file
with open(input_file, 'r') as file:
    lines = file.readlines()

# Identify lines that match `"label": 0`
matching_lines = [line for line in lines if '"label": 0' in line]

LINES = 50

# Randomly select 200 lines to remove
if len(matching_lines) >= LINES:
    lines_to_remove = random.sample(matching_lines, LINES)
else:
    lines_to_remove = matching_lines  # If fewer than 200 lines are available, remove all

# Remove the selected lines from the original list
filtered_lines = [line for line in lines if line not in lines_to_remove]

# Write the filtered lines back to the original file
with open(output_file, 'w') as file:
    file.writelines(filtered_lines)

# Save the removed lines to another file
with open(removed_lines_file, 'w') as file:
    file.writelines(lines_to_remove)

print(f"Removed {len(lines_to_remove)} lines and saved them to {removed_lines_file}")
