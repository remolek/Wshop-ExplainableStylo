"""
This module provides shared variables for the other modules:
'preprocess_spacy.py', 'feature_extraction.py', 'text_classify.py', and 'text_visualise.py'.

It includes:
- Spacy language model
...

Author: Jeremi K. Ochab
Date: June 27, 2023
"""

import spacy
import os
# import numpy as np
try:
    from tqdm.notebook import tqdm as tqdm_notebook
except ImportError:
    from tqdm import tqdm

tqdm_display = True

nlp = None
def _add_file_extension(filename,defult_ext):
    file, ext = os.path.splitext(filename)
    if ext:
        return filename
    elif defult_ext:
        filename = "{}{}".format(file, defult_ext)
        return filename
    else:
        raise ValueError("No file extension provided in neither filename: {} nor default extension {}.".format(filename,defult_ext))

def _generate_unique_filename(filename, extension):
    base_filename = filename
    counter = 1
    while os.path.exists(f"{base_filename}_{counter}{extension}"):
        counter += 1
    return f"{base_filename}_{counter}{extension}"

def _prepare_filename(filename, extension, overwrite):
    base_filename = filename
    filename = _add_file_extension(base_filename, extension)
    if os.path.exists(filename) and not overwrite:
        filename = _generate_unique_filename(base_filename, extension)
    return filename

def _check_filename(filename, extension, verbose):
    if not isinstance(filename, str):
        if verbose<2:
            print("Warning: Invalid filename format. Please provide a string.")
        return False
    if extension:
        filename = _add_file_extension(filename, extension)
    if os.path.isfile(filename):
        return filename
    else:
        if verbose<2:
            print("Warning: File '{}' does not exist. Exiting...".format(filename))
        return False

def _check_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Default folder '{folder_path}' created.")
    # else:
    #     print(f"Folder '{folder_path}' already exists.")






