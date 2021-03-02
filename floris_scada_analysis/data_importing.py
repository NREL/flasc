import re

def fix_csv_contents(csv_contents, line_format_str):
    """Check the contents of the raw database .csv file and ensure each row
       fits a predefined formatting. This can pick out irregularities in rows,
       such as a missing or deformed time entry in a row.

    Args:
        csv_contents ([str]): Contents of the preprocessed .csv file

    Returns:
        csv_contents ([str]): Contents of the postprocessed .csv file
    """

    # Specify required row format in each csv file
    line_format = re.compile(line_format_str)

    # Split large string into separate line strings
    csv_contents = csv_contents.split("\n")

    # Remove last line if empty
    if len(csv_contents[-1]) == 0:
        csv_contents = csv_contents[0:-1]  # Skip last line

    # Check and fix formatting of each line
    pop_ids = []  # List with to-be-removed row ids
    for i in range(1, len(csv_contents)):  # Skip first line
        csv_line = csv_contents[i]
        if line_format.match(csv_line) is None:
            print(
                "    Ignoring this row due to incorrect format: '"
                + csv_line
                + "'"
            )
            pop_ids.append(i)

    for i in pop_ids[::-1]:  # Back to front to avoid data shifts
        csv_contents.pop(i)  # Remove lines

    csv_contents = "\n".join(csv_contents) + "\n"
    return csv_contents