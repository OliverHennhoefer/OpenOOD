import sys # Used for printing warnings to stderr

# --- Configuration: Mapping column names to their expected start index ---
# This mapping assumes a consistent structure in the data rows where:
# - Index 0 is the dataset name.
# - Each metric occupies 3 slots: value, '±', std_dev.
# Adjust these indices if your table structure differs significantly.
COLUMN_TO_BASE_INDEX = {
    "FPR@95": 1,
    "AUROC": 4,
    "AUPR_IN": 7,
    "AUPR_OUT": 10,
    "ACC": 13,
}

def format_metric_from_parts(val_part, pm_part, std_part):
    """
    Formats value, pm, std parts into LaTeX 'value{\tiny$\pm$std}',
    checking if the middle part is indeed '±'.

    Args:
        val_part (str): The part containing the mean value.
        pm_part (str): The part expected to be '±'.
        std_part (str): The part containing the standard deviation.

    Returns:
        str: The formatted LaTeX string.
    """
    # Check if the plus-minus symbol is correct
    if pm_part != '±':
        print(
            f"Warning: Expected '±' between '{val_part}' and '{std_part}', "
            f"but found '{pm_part}'. Formatting might be incorrect.",
            file=sys.stderr
        )
        # Proceed with formatting anyway

    # Escape backslashes for LaTeX commands within the f-string
    return f"{val_part}{{\\tiny$\\pm${std_part}}}"

def generate_latex_table_rows(output_cols):
    """
    Prompts the user to paste table data, then parses each data row
    to generate and print a LaTeX-formatted string including the dataset name
    and the metrics specified in `output_cols`.

    Args:
        output_cols (list): A list of strings representing the column names
                            (e.g., ["AUROC", "AUPR_OUT"]) to include in the
                            LaTeX output, in the desired order.
    """
    print("Paste your table data below (including the header line).")
    print("Indicate the end of your input by either:")
    print("  1. Pressing Enter on an empty line.")
    print("  2. Using Ctrl+D (on Unix-like systems) or Ctrl+Z then Enter (on Windows).")
    print("-" * 40)

    lines = []
    while True:
        try:
            line = input()
            if not line.strip():
                if lines: # If we have collected lines, empty line signifies end.
                    break
                else: # If we haven't collected lines yet, ignore empty lines.
                    continue
            lines.append(line)
        except EOFError:
            break # End of input signal

    print("-" * 40)
    print(f"Generating LaTeX rows for columns: {output_cols}")
    print("-" * 40)

    if not lines:
        print("No input received.")
        return

    # --- Determine which indices to extract based on output_cols ---
    indices_to_format = []
    valid_output_cols = []
    for col_name in output_cols:
        if col_name in COLUMN_TO_BASE_INDEX:
            indices_to_format.append(COLUMN_TO_BASE_INDEX[col_name])
            valid_output_cols.append(col_name)
        else:
            print(
                f"Warning: Requested column '{col_name}' not found in known columns "
                f"{list(COLUMN_TO_BASE_INDEX.keys())}. It will be skipped.",
                file=sys.stderr
            )

    if not indices_to_format:
        print("Error: None of the requested columns were found or valid. No output generated.", file=sys.stderr)
        return

    # --- Process Data Lines ---
    is_first_line = True
    for line in lines:
        # Skip the first line (assumed header)
        if is_first_line:
            is_first_line = False
            # Optional: print(f"Skipping header: {line.strip()}", file=sys.stderr)
            continue

        line = line.strip()
        if not line:
            continue # Skip empty lines within the data

        parts = line.split()

        # Check if the line has enough parts for the *last* requested column index
        # The maximum index needed is base_index + 2
        max_required_index = 0
        if indices_to_format:
             max_required_index = max(indices_to_format) + 2

        # We need at least 1 part for the name + enough parts for the metrics
        min_expected_parts = 1 + max_required_index # Indexing starts from 0

        if len(parts) <= max_required_index: # Use <= because index is 0-based
             print(
                f"Warning: Skipping line. Expected at least {max_required_index + 1} parts "
                f"to extract columns up to '{valid_output_cols[-1]}', "
                f"but found {len(parts)}: '{line}'",
                file=sys.stderr
            )
             continue

        dataset_name = parts[0]
        formatted_metrics = []

        try:
            for i, base_index in enumerate(indices_to_format):
                # Extract parts for the current metric
                # Indices are base_index, base_index + 1, base_index + 2
                val_part = parts[base_index]
                pm_part = parts[base_index + 1]
                std_part = parts[base_index + 2]

                # Format the metric
                fmt_metric = format_metric_from_parts(val_part, pm_part, std_part)
                formatted_metrics.append(fmt_metric)

            # Join the formatted metrics with the LaTeX table cell separator '&'
            metrics_string = " & ".join(formatted_metrics)

            # Print the final formatted LaTeX row
            print(f"{dataset_name} & {metrics_string}")

        except IndexError:
            # This might happen if the line *looked* long enough initially but had
            # unexpected spacing causing split() to produce fewer parts than anticipated
            # for a specific required index. The earlier check handles most cases.
            print(
                f"Warning: Skipping line due to unexpected structure "
                f"(IndexError accessing parts for requested columns): '{line}'",
                file=sys.stderr
            )
        except Exception as e:
            # Catch any other potential errors during formatting
            print(
                f"Warning: Skipping line due to an unexpected error '{e}': '{line}'",
                file=sys.stderr
            )

# --- Main execution block ---
if __name__ == "__main__":
    desired_columns = ["FPR@95", "AUPR_IN"]
    #desired_columns = ["FPR@95", "AUROC", "AUPR_IN"]

    print(f"Requesting columns: {desired_columns}")
    generate_latex_table_rows(output_cols=desired_columns)

    print("\n--- Example with different columns ---")
    desired_columns_2 = ["FPR@95", "ACC"]
    print(f"Requesting columns: {desired_columns_2}")
    generate_latex_table_rows(output_cols=desired_columns_2)