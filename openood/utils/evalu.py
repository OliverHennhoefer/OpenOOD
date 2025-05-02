import numpy as np
import pandas as pd
import io  # Used to treat string as file
import sys  # Used for printing errors and exiting


def read_multiline_input(prompt):
    """
    Reads multi-line input from the user until an empty line is entered
    after at least one line of content has been provided.
    """
    lines = []
    print(prompt)
    print("Paste your table below. Press Enter on an empty line when finished:")
    while True:
        try:
            line = input()
            # If the line is empty AND we already have some lines, stop reading
            if not line.strip() and lines:
                break
            # If the line is not empty, add it
            elif line.strip():
                lines.append(line)
            # If the line is empty and we have no lines yet, prompt again (or just continue loop)
            elif not line.strip() and not lines:
                # print("Waiting for data...") # Optional feedback
                continue
        except EOFError:
            # Handles cases where input stream ends (e.g., piping from file)
            if not lines:
                print("Error: No input received before end of stream.", file=sys.stderr)
                sys.exit(1)
            break  # End reading if EOF is detected after getting some lines

    if not lines:
        print("Error: No data was entered.", file=sys.stderr)
        sys.exit(1)

    return "\n".join(lines)


def parse_table_string(table_string, input_num):
    """
    Parses a multi-line string containing a space-delimited table into a pandas DataFrame.
    The first column is expected to be the index.
    """
    try:
        # Use io.StringIO to treat the string as a file
        # Use sep='\s+' to handle one or more spaces as delimiters
        # Use index_col=0 to set the first column as the DataFrame index
        df = pd.read_csv(io.StringIO(table_string), sep='\s+', index_col=0)

        # Basic validation: Check if DataFrame is empty
        if df.empty:
            raise ValueError("Parsed table is empty.")

        # Check if all data columns are numeric (attempt conversion, report error if fails)
        for col in df.columns:
            # pd.to_numeric will raise ValueError if conversion fails for any element
            pd.to_numeric(df[col], errors='raise')

        return df

    except ValueError as e:
        # Catch errors from pd.read_csv (e.g., inconsistent columns) or pd.to_numeric
        print(f"Error parsing table for input {input_num}: {e}", file=sys.stderr)
        print("Please ensure the table is correctly formatted with a header, an index column (dataset name),",
              file=sys.stderr)
        print("and numeric values separated by spaces.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        # Catch any other unexpected parsing errors
        print(f"An unexpected error occurred during parsing of input {input_num}: {e}", file=sys.stderr)
        sys.exit(1)


# --- Configuration ---
NUM_TABLE_INPUTS = 3  # Number of tables to average
DECIMAL_PLACES = 2  # Number of decimal places for Mean ± SD output

# --- Data Collection ---
dataframes = []
expected_index = None
expected_columns = None

print(f"This script will ask for {NUM_TABLE_INPUTS} multi-line table inputs.")

for i in range(NUM_TABLE_INPUTS):
    input_prompt = f"\n--- Please provide Table Input {i + 1} ---"
    table_str = read_multiline_input(input_prompt)

    # Parse the string into a DataFrame
    current_df = parse_table_string(table_str, i + 1)

    # --- Validation: Check consistency across tables ---
    if i == 0:
        # Store the index and columns from the first table as the reference
        expected_index = current_df.index
        expected_columns = current_df.columns
        print(f"Successfully parsed Input {i + 1}. Expecting subsequent tables to have:")
        print(f" - Datasets (rows): {list(expected_index)}")
        print(f" - Metrics (columns): {list(expected_columns)}")
    else:
        # Compare index and columns with the first table
        if not current_df.index.equals(expected_index):
            print(f"Error: Input {i + 1} has different datasets (rows) than the first input.", file=sys.stderr)
            print(f"Expected: {list(expected_index)}", file=sys.stderr)
            print(f"Got:      {list(current_df.index)}", file=sys.stderr)
            sys.exit(1)
        if not current_df.columns.equals(expected_columns):
            print(f"Error: Input {i + 1} has different metrics (columns) than the first input.", file=sys.stderr)
            print(f"Expected: {list(expected_columns)}", file=sys.stderr)
            print(f"Got:      {list(current_df.columns)}", file=sys.stderr)
            sys.exit(1)
        print(f"Successfully parsed Input {i + 1}. Structure matches the first table.")

    dataframes.append(current_df)

# --- Calculations ---
# Ensure we got the expected number of DataFrames (should be guaranteed by loop and exits)
if len(dataframes) != NUM_TABLE_INPUTS:
    print("Error: Did not collect the expected number of tables. Exiting.", file=sys.stderr)
    sys.exit(1)

# Stack the numerical values of the DataFrames into a 3D NumPy array
# Shape will be (num_tables, num_datasets, num_metrics)
try:
    # Ensure all data is numeric before stacking (already checked in parse function, but belt-and-suspenders)
    numeric_dataframes = [df.apply(pd.to_numeric) for df in dataframes]
    data_3d = np.array([df.values for df in numeric_dataframes])
except Exception as e:
    print(f"Error converting table data to numeric for calculation: {e}", file=sys.stderr)
    sys.exit(1)

# Calculate mean and standard deviation along the first axis (axis=0), which represents the different tables
# Use ddof=1 for the sample standard deviation
means_2d = np.mean(data_3d, axis=0)
stds_2d = np.std(data_3d, axis=0, ddof=1)

# --- Formatting Results ---
# Create a DataFrame to store the formatted "Mean ± SD" strings
# Use the index and columns from the first DataFrame
results_df = pd.DataFrame(index=expected_index, columns=expected_columns, dtype=object)

# Format string for "Mean ± SD"
format_string = f"{{:.{DECIMAL_PLACES}f}} ± {{:.{DECIMAL_PLACES}f}}"

# Iterate through the datasets (rows) and metrics (columns) to format results
for i, dataset_name in enumerate(expected_index):
    for j, metric_name in enumerate(expected_columns):
        mean = means_2d[i, j]
        std = stds_2d[i, j]

        # Handle potential NaN in standard deviation (e.g., if NUM_TABLE_INPUTS < 2)
        if np.isnan(std):
            results_df.loc[dataset_name, metric_name] = f"{mean:.{DECIMAL_PLACES}f} ± N/A"
        else:
            results_df.loc[dataset_name, metric_name] = format_string.format(mean, std)

# --- Display Result ---
print("\n--- Aggregated Results (Mean ± Sample StDev) ---")
# Add index=True (default) or index=True, header=True explicitly if needed. to_string includes them by default.
print(results_df.to_string())