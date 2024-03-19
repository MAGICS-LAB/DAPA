import pandas as pd
import os
import sys
import argparse

def count_results_in_csv(args):
    count_1 = 0
    count_0 = 0
    count_err = 0
    for filename in os.listdir(args.directory_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(args.directory_path, filename)
            try:
                df = pd.read_csv(file_path)
                if 'Success' in df.columns:
                    if df['Success'].sum() == 0:
                        count_0 += 1
                    elif df['Success'].sum() == 1:
                        count_1 += 1
                    else:
                        count_err += 1
            except Exception as e:
                print(f"Error reading {filename}: {e}")

    return count_1, count_0, count_err

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analysis the safe rate of the model response.')
    parser.add_argument('directory_path', default='Results2/meta-llama/Llama-2-7b-chat-hf/GPTFuzzer',
                        help='Result directory path to analysis the safe rate of the response (default: Results2/meta-llama/Llama-2-7b-chat-hf/GPTFuzzer)')
    args = parser.parse_args()

    count_1, count_0, count_err = count_results_in_csv(args)
    total_responses = count_0 + count_1 + count_err
    percentage_safe = (count_0 / total_responses * 100) if total_responses > 0 else 0
    print(f"Total Responses: {total_responses}")
    print(f"Count of Safe Responses(0): {count_1}")
    print(f"Count of Not Safe Responses(1): {count_0}")
    print(f"Percentage of Safe Responses: {percentage_safe:.2f}%")