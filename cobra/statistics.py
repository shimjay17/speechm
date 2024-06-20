import pandas as pd
from jiwer import wer
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Load the Excel file
file_path = '/mnt/hdd/hsyoon/workspace/samsung/cobra/evaluation_results_test.xlsx'
results_df = pd.read_excel(file_path)

# Initialize counters and data structures
generated_length_zero_count = 0
input_length_count = {}
all_references = []
all_predictions = []

# Iterate over the rows to perform the required analysis
for index, row in results_df.iterrows():
    generated_length = row['generated_length']
    input_length = row['input_length']
    
    if generated_length == 0:
        generated_length_zero_count += 1
        if input_length not in input_length_count:
            input_length_count[input_length] = 0
        input_length_count[input_length] += 1
    else:
        all_references.append(row['input'][0].upper())  # Convert input to uppercase
        all_predictions.append(row['generated'][0].upper())  # Convert generated to uppercase

# Calculate word error rate if there are valid predictions
if all_references and all_predictions:
    word_error_rate = wer(all_references, all_predictions)
else:
    word_error_rate = None

# Print the results
print(f"Number of generated_length = 0: {generated_length_zero_count}")
print("Counts of input_length when generated_length = 0:")
for input_length, count in input_length_count.items():
    print(f"  Input length {input_length}: {count} instances")
print(f"Word Error Rate (WER): {word_error_rate}")

with open('/mnt/hdd/hsyoon/workspace/samsung/cobra/evaluation_results_test.txt', 'w') as f:
    f.write(f"Number of generated_length = 0: {generated_length_zero_count}\n")
    f.write("Counts of input_length when generated_length = 0:\n")
    for input_length, count in input_length_count.items():
        f.write(f"  Input length {input_length}: {count} instances\n")
    f.write(f"Valid Word Error Rate (WER): {word_error_rate}\n")