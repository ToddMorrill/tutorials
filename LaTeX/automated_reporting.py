import os

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report


def generate_dummy_report():
    # dummy data
    ground_truth = np.random.choice(a=[False, True], size=(1000, 1))
    predictions = np.random.choice(a=[False, True], size=(1000, 1))
    return classification_report(ground_truth, predictions, output_dict=True)


def prepare_report_df(report: dict) -> pd.DataFrame:
    """Convert sklearn classification report to a dataframe. 

    Args:
        report (dict): sklearn classification report.

    Returns:
        pd.DataFrame: Formatted dataframe.
    """
    accuracy = report.pop('accuracy')
    df = pd.DataFrame(report).T
    df.index.name = 'Class'
    df = df.reset_index()
    df.columns = [x.title() for x in df.columns]
    df['Class'] = df['Class'].apply(lambda x: x.title())
    # remove column name
    df = df.rename(columns={'Class': ''})
    df['Support'] = df['Support'].astype(int)
    return df


def generate_table(df: pd.DataFrame,
                   index: bool = False,
                   column_format: str = None,
                   caption: str = None,
                   float_format: str = '%.2f') -> str:
    """Generate a LaTeX table based on the passed dataframe.

    Args:
        df (pd.DataFrame): Dataframe.
        index (bool, optional): If True, include the dataframe index in the LaTeX table. Defaults to False.
        column_format (str, optional): LaTeX column format (e.g. 'c | c | c'). Defaults to None.
        caption (str, optional): Table caption. Defaults to None.
        float_format (str, optional): Formatting for floats. Defaults to '%.2f'.

    Returns:
        str: [description]
    """
    # column_format='c | c | c',
    # caption=('test caption', 'test'),
    # label='tab:test'
    table_string = df.to_latex(index=index,
                               column_format=column_format,
                               caption=caption,
                               float_format=float_format,
                               bold_rows=True)

    if caption:
        # if you add a caption, it will enclose everything in table environment
        table_split = table_string.split('\n')
        table_split[0] = table_split[0] + '[ht]'  # inline with text
        table_string = '\n'.join(table_split)

    # TODO: remove \toprule, \midrule, \bottomrule, add preferred borders
    # TODO: bold column headers and row labels
    return table_string


def save_table(table_string: str, file_path: str) -> None:
    """Save the passed LaTeX table.

    Args:
        table_string (str): LaTeX table.
        file_path (str): File destination.
    """
    with open(file_path, 'w') as f:
        f.write(table_string)


def latex_table(report: dict, file_path: str) -> None:
    """Convert an sklearn style classification report into a LaTeX table and 
    save the result.

    Args:
        report (dict): sklearn style classification report.
        file_path (str): File destination.
    """
    df = prepare_report_df(report)
    table_string = generate_table(df)
    save_table(table_string, file_path)


def main():
    report = generate_dummy_report()

    # save to the tables folder
    directory = 'tables'
    os.makedirs(directory, exist_ok=True)
    save_file_path = os.path.join(directory, 'dummy_table.tex')
    latex_table(report, save_file_path)


if __name__ == '__main__':
    main()
