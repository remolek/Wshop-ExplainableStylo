"""
This module provides wrapper functions for text preprocessing using spaCy.
Combine it with 'feature_extraction.py' module, to extract interesting textual features,
with 'text_classify.py', to classify texts using those features, and 'text_visualise.py', to plot results and their SHAP explanations.

It includes functions to load text files from a directory, add custom labels to spaCy documents,
save spaCy documents along with metadata to disk, load spaCy documents and associated metadata from disk,
and preprocess raw texts using spaCy.

Author: Jeremi K. Ochab
Date: June 27, 2023
"""
# TO DO: update docstrings

import os
import pandas as pd
import spacy
# Get information about available language models
# models_info = spacy.info() # list(models_info["pipelines"].keys())
models_info = spacy.util.get_installed_models()
# python -m spacy download pl_core_news_sm

from spacy.tokens import Doc, DocBin
import json

def is_running_in_jupyter():
    try:
        # Check if the 'get_ipython' function exists
        get_ipython()
        return True
    except NameError:
        return False

if is_running_in_jupyter():
    from tqdm.notebook import tqdm_notebook as tqdm_ui
else:
    from tqdm import tqdm as tqdm_ui

import glob

from . import shared_module
tqdm_display = shared_module.tqdm_display


import subprocess

def _check_spacy_model(model_name,
                       verbose = 1):
    if verbose<1:
            print("Checking if the model is available...")
    try:
        # Run the shell command
        command = f"python -m spacy info {model_name} --url"
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True)

        # Check if the command was successful
        if result.returncode == 0:
            # Extract and return the URL from the command output
            url = result.stdout.strip()

            if url:
                if verbose<1:
                    print(f"The {model_name} is available from Spacy models.")
                return True
            else:
                if verbose<2:
                    print(f"The {model_name} is NOT available from Spacy models.")
                return False
                        
        else:
            # Handle any errors
            if verbose<2:
                print(f"Error: {result.stderr.strip()}")
                print(f"The {model_name} is NOT available from Spacy models.")
            return False
    except Exception as e:
        if verbose<2:
            print(f"An error occurred: {e}")
        return None
        
def _choose_spacy_model(model,
                       verbose = 0):
    model_set = ''
    if(model in models_info):
        shared_module.nlp = spacy.load(model)
        model_set = model
        if verbose<1:
            print("Provided model name for Spacy: '{}'.".format(model))
    elif _check_spacy_model(model,verbose = verbose):
        while True:
            choice = input("Spacy model not installed.\n Would you like to install it now: [y/n]").lower()
            if choice == 'y':
                spacy.cli.download(model)
                # In order for it to be found after installation, you will need to restart or reload your Python process.
                shared_module.nlp = spacy.load(model)
                model_set = model
                break
            elif choice == 'n':
                break
            else:
                 print("Invalid response. Please enter 'y' or 'n'.")
    else:
        while True:
            choice = input("Is it a spacy_sentence_bert model: [y/n]").lower()
            if choice == 'y':
                if verbose<1:
                    print("Provided model name for spacy_sentence_bert: '{}'.".format(model))
                import spacy_sentence_bert
                shared_module.nlp = spacy_sentence_bert.load_model(model)
                model_set = model
                break
            elif choice == 'n':
                break
            else:
                 print("Invalid response. Please enter 'y' or 'n'.")
    if not model_set:
        while True:
            print('Available Spacy models:')
            for modelindex in range(len(models_info)):
                print(f'[{modelindex}]: {models_info[modelindex]}')
            choice1 = int(input("Choose the model number:\n"))
            if 0<=choice1 and choice1 < len(models_info):
                print(f"Choosing the available model: {models_info[choice1]}.")
                shared_module.nlp = spacy.load(models_info[choice1])
                model_set = models_info[choice1]
                break
            else:
                print(f'Invalid reponse. Choose an integer between 0 and {len(models_info)-1}.')
    return model_set



def load_texts(metadata,
               from_variable = None,
               verbose = 0,
               precheck = False):
    """
    """
    if not precheck:
        if not isinstance(metadata, dict):
            if verbose<2:
                print("Warning: Wrong format of metadata. Please provide a dictionary. Exiting...")
            return
        if 'labels' not in metadata or 'files' not in metadata:
            if verbose<2:
                print("Warning: Wrong format of metadata. Minimally, please provide {'experiment_name':name,'labels': array_of_labels, 'files': {'filename': array_of_filenames, 'class': array_of_classes}}. Exiting...")
            return
        if 'filename' not in metadata['files'] or 'class' not in metadata['files']:
            if verbose<2:
                print("Warning: Wrong format of metadata. Minimally, please provide {'experiment_name':name,'labels': array_of_labels, 'files': {'filename': array_of_filenames, 'class': array_of_classes}}. Exiting...")
            return
        if not all(isinstance(f, str) for f in metadata['files']['filename']):
            if verbose<2:
                print("Warning: Invalid filename format. Please provide an array of strings.")
            return
        if not all(isinstance(f, str) for f in metadata['files']['class']):
            if verbose<2:
                print("Warning: Invalid class name format. Please provide an array of strings.")
            return
            
    labels = metadata['labels']
    # labels = metadata['files'].keys() # TO DO: ewentualnie można się pozbyć 'labels'?
    if verbose<1:
        print("List of labels used: '{}'.".format(labels))

    if 'filename' in labels:
        position = labels.index('filename')
    else:
        if verbose<1:
            print(f"'filename' not found in the 'metadata['labels']'. Nothing to load.")
        return
    
    texts = []
    if from_variable is None:
        for i in tqdm_ui(zip(*(metadata['files'][l] for l in labels))):
            filename = i[position]
            filename = shared_module._check_filename(filename, '',verbose)
            if filename:
                with open(filename, 'r', encoding="utf8", errors='ignore') as in_file: 
                    file_meta = {c:name for c, name in zip(labels,i)}
            # TO DO: zamiast replace() dać jakąś ogólną funkcję preprocesującą? Albo ustawić inne parametry już w spacym?
                    text = (in_file.read().replace('\n', ' '), file_meta)
                    texts.append(text)
            else:
                return
    else:
        for index, i in tqdm_ui(enumerate(zip(*(metadata['files'][l] for l in labels)))):
            file_meta = {c:name for c, name in zip(labels,i)}
            text = (from_variable[index].replace('\n', ' '), file_meta)
            texts.append(text)
    return texts

def _legacy_load_dir(dirpath,
             labels = ['filename','subdir'],
             label_sep = '_',
             verbose = 0):
    """
    Load text files from a directory and extract their content along with optional metadata.

    Parameters
    ----------
    dirpath : str or list
        Path to the corpus directory containing the text files. If a string is provided,
        it represents the path to a single directory. If a list of strings is provided,
        each represents the path to a subdirectory.
    labels : str or list, optional
        List of labels for each text file. If a single string is provided, it represents the text class.
        If a list of strings is provided, the first string represents the text class.
        (Default is ['subdir','filename']).
    label_sep : str, optional
        Separator used in the filenames to separate labels (default is '_').
    verbose : int, optional
        Verbosity level (0: all messages, 1: warnings, 2: no messages) (default is 0).

    Returns
    -------
    list
        A list of tuples, where each tuple has the format (text, metadata).
        'text' is a string with the content of the file.
        'metadata' is a dictionary containing the extracted labels.

    Raises
    ------
    OSError
        If the specified directory or subdirectories do not exist.
    ValueError
        If the input labels are not in the correct format.

    Notes
    --------
    The output list of tuples can be fed directly into Spacy's nlp.pipe(list, as_tuples=True).
    

    Examples
    --------
    # Load text files from a single directory
    >>> texts = load_dir('/path/to/corpus', labels=['author', 'title'])

    # Load text files from multiple subdirectories
    >>> texts = load_dir(['/path/to/corpus/subdir1', '/path/to/corpus/subdir2'], labels='label')
    """
    
    if isinstance(dirpath, str):
        if not os.path.exists(dirpath):
            if verbose<2:
                print("Warning: Directory '{}' does not exist.".format(dirpath))
            return
        dirs = glob.glob(f''+dirpath+'/*')
        if verbose<1:
            print("Found subdirectories '{}'.".format(str(dirs).replace(',',',\n')))
    elif not isinstance(dirpath, list) or not all(isinstance(d, str) for d in dirpath):
        if verbose<2:
            print("Warning: Invalid corpus directory path format. Please provide a single string or a list of strings.")
        return
    else:
        for d in dirpath:
            if not os.path.exists(d):
                if verbose<2:
                    print("Warning: Subdirectory '{}' does not exist.".format(d))
                return
        dirs = dirpath
        if verbose<1:
            print("Found subdirectories '{}'.".format(str(dirs).replace(',','\n')))
    
    if isinstance(labels, str):
        labels = [labels] # Make a list if only one int is given.
    elif not isinstance(labels, list) or not all(isinstance(f, str) for f in labels):
        if verbose<2:
            print("Warning: Invalid label name format. Please provide a single string or a list of strings.")
        return

    if not isinstance(label_sep, str):
        label_sep = '_'
        if verbose<2:
            print("Warning: Invalid label separator format. Please provide a single string. Proceeding with the default separator '_'.")
    # if len(labels)<1:
    #     labels.extend(['subdir'])
    # if len(labels)<2:
    #     labels.extend(['filename'])
    labels[:0] = ['filename','subdir']
    if verbose<1:
        print("List of labels used: '{}'.".format(labels))

#     TO DO: uporządkować filename, filename0
    texts = []
    for d in dirs:
        if verbose<1:
            print("Reading files from directory '{}'.".format(d))        
        filenames = glob.glob(f''+d+'\\*.txt')
        for filename in tqdm_ui(filenames):
            with open(filename, 'r', encoding="utf8", errors='ignore') as in_file: 
                dirname =  os.path.split(os.path.dirname(filename))[-1]
                filename0 = os.path.splitext(os.path.basename(filename))[0]
                if len(filename0.split('_')) != len(labels)-2:
                    if verbose<2:
                        print("Warning: Filename '{}' in directory '{}' has a different number of labels than provided label list {}. Current label separataor in filenames is '{}'.".format(filename0,dirname,labels[1:], label_sep))
                    return                    
                metadata = {c:name for c, name in zip(labels,[filename,dirname]+filename0.split('_'))}
# TO DO: zamiast replace() dać jakąś ogólną funkcję preprocesującą? Albo ustawić inne parametry już w spacym?
                text = (in_file.read().replace('\n', ' '), metadata)
                texts.append(text)
    
    return texts
    
def spacy_add_doc_labels(labels = [],
                        verbose = 0,
                       precheck = False):
    """
    Enable adding custom labels to SpaCy documents (available via 'doc._.label').

    Parameters
    ----------
    labels : str or list, optional
        The labels to be added to the documents. Can be a single label as a string
        or a list of labels as strings (default is []).
    verbose : int, optional
        Verbosity level (0: all messages, 1: warnings, 2: no messages) (default is 0).
    precheck : bool, optional
        If True, skips the label format validation step (default is False).
        Used together with other preprocess_spacy functions.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the label format is not valid.

    Examples
    --------
    # Add a single label to the documents
    >>> spacy_add_doc_labels('positive')

    # Add multiple labels to the documents
    >>> spacy_add_doc_labels(['positive', 'negative', 'neutral'])
    """
    
    if not precheck:
        if isinstance(labels, str):
            labels = [labels] # Make a list if only one int is given.
        elif not isinstance(labels, list) or not all(isinstance(f, str) for f in labels):
            if verbose<2:
                print("Warning: Invalid label format. Please provide a single string or a list of strings.")
            return
    
    for l in labels:
        if(Doc.has_extension(l)==False):
            Doc.set_extension(l, default=None)
    if verbose<1:
        print("Labels {} will be now accessible in documents via 'document._.label'.".format(labels))
    
    return


def _legacy_spacy_save_docs(docs,
                    metadata,
                    verbose = 0,
                    precheck = False):
    """
    Save SpaCy documents along with metadata to disk.

    Parameters
    ----------
    docs : spacy.tokens.doc.Doc or list
        A single SpaCy document or a list of SpaCy documents to be saved.
    metadata : dict
        Metadata associated with the documents. The metadata dictionary should have the following keys:
        - 'filename': The filename to be used for saving the documents and metadata. File extensions are removed.
        - 'model': The name of the model associated with the documents.
        - 'labels': A list of label names associated with the documents (default is ['subdir','filename']). A list shorter than default is extended.
    verbose : int, optional
        Verbosity level (0: all messages, 1: warnings, 2: no messages) (default is 0).

    Returns
    -------
    filename : str
        Output filename (no extension).

    Raises
    ------
    ValueError
        If the input documents or metadata are not in the correct format.

    Examples
    --------
    # Save a single SpaCy document with metadata
    >>> metadata = {'filename': 'document1', 'model': 'en_core_web_sm', 'labels': ['category1', 'category2']}
    >>> spacy_save_docs(doc, metadata)

    # Save a list of SpaCy documents with metadata
    >>> metadata = {'filename': 'documents', 'model': 'en_core_web_md', 'labels': ['category1', 'category2', 'category3']}
    >>> spacy_save_docs(docs, metadata)
    """
    
    default_labels = ['subdir','filename']
    if not precheck:
        if isinstance(docs, spacy.tokens.doc.Doc) or isinstance(docs,spacy.tokens.span.Span):
            docs = [docs] # Make a list if only one Doc is given.
        elif not isinstance(docs, list) or not all(isinstance(doc, spacy.tokens.doc.Doc) or isinstance(doc, spacy.tokens.span.Span) for doc in docs):
            if verbose<2:
                print("Warning: Invalid document format. Please provide a single spacy.tokens.doc.Doc, spacy.tokens.span.Span or a list of such objects.")
            return

        if not isinstance(metadata, dict):
            if verbose<2:
                print("Warning: Wrong format of metadata. Please provide '{'filename':filename, 'model':model, 'labels':labels}'. Exiting...")
            return
    
        if 'filename' not in metadata:
            if verbose<2:
                print("Warning: No filename provided. Exiting...")
            return

        if 'model' not in metadata:
            if verbose<2:
                print("Warning: No model name provided. Exiting...")
            return
        
        if 'labels' not in metadata:
            labels = []
        else:
            labels = metadata['labels']
            if len(labels)<1:
                labels.extend([default_labels[0]])
                if verbose<1:
                    print("Provided label list was shorter than default and has been extended to '{}'.".format())
            if len(labels)<2:
                labels.extend([default_labels[1]])
                if verbose<1:
                    print("Provided label list was shorter than default and has been extended to '{}'.".format())
            elif isinstance(labels, str):
                labels = [labels] # Make a list if only one int is given.    
            elif not isinstance(labels, list) or not all(isinstance(f, str) for f in labels):
                if verbose<2:
                    print("Warning: Invalid metadata name format. Please provide a single string or a list of strings.")
                return
        metadata['labels'] = labels  
        
        if not all(isinstance(f, str) for f in [metadata['filename'], metadata['model']]):
            if verbose<2:
                print("Warning: Invalid filename or model name format. Please provide a string.")
            return

        if verbose<1:
            print("Number of documents provided: {}.".format(len(docs)))


    filename = metadata['filename']
    model = metadata['model']
  
    # # Making sure we have the name of the model for later loading.
    # if (not model) and not (model_sep in filename):
    #     if verbose<2:
    #         print("Warning: No model name provided. Please provide a filename in the format 'filename{}model' or provide separate 'filename' and 'model' arguments.".format(model_sep))
    #     return
    # if filename.count(model_sep)>0:
    #     filename = filename.replace(model_sep,"")    
    #     if verbose<2:#"Warning: model name is ambigous. Please provide a filename in the format 'filename{}model' or provide separate 'filename' and 'model' arguments or change the separator."
    #         print("Warning: filename contains restricted separator '{}'. Changing filename to '{}'.".format(model_sep, filename))
    # if model:
    #     filename = filename +model_sep+ model

    filename = os.path.splitext(filename)[0]
    with open(filename+'.json', 'w') as outfile:
        json.dump(metadata, outfile)
    if verbose<1:
            print("Metadata saved as {}.".format(filename+".json"))

    # Create and save a collection of docs
    docbin = DocBin(docs=docs, store_user_data=True)
    docbin.to_disk(filename+".spacy")
    if verbose<1:
            print("Documents saved as {}.".format(filename+".spacy"))
    return filename

def spacy_save_docs(docs,
                    filename,
                    overwrite = False,
                    verbose = 0,
                    precheck = False):
    """
    Save SpaCy documents along with metadata to disk.

    Parameters
    ----------
    docs : spacy.tokens.doc.Doc or list
        A single SpaCy document or a list of SpaCy documents to be saved.
    metadata : dict
        Metadata associated with the documents. The metadata dictionary should have the following keys:
        - 'filename': The filename to be used for saving the documents and metadata. File extensions are removed.
        - 'model': The name of the model associated with the documents.
        - 'labels': A list of label names associated with the documents (default is ['subdir','filename']). A list shorter than default is extended.
    verbose : int, optional
        Verbosity level (0: all messages, 1: warnings, 2: no messages) (default is 0).

    Returns
    -------
    filename : str
        Output filename (no extension).

    Raises
    ------
    ValueError
        If the input documents or metadata are not in the correct format.

    Examples
    --------
    # Save a single SpaCy document with metadata
    >>> metadata = {'filename': 'document1', 'model': 'en_core_web_sm', 'labels': ['category1', 'category2']}
    >>> spacy_save_docs(doc, metadata)

    # Save a list of SpaCy documents with metadata
    >>> metadata = {'filename': 'documents', 'model': 'en_core_web_md', 'labels': ['category1', 'category2', 'category3']}
    >>> spacy_save_docs(docs, metadata)
    """
    
    default_labels = ['subdir','filename']
    if not precheck:
        if isinstance(docs, spacy.tokens.doc.Doc) or isinstance(docs,spacy.tokens.span.Span):
            docs = [docs] # Make a list if only one Doc is given.
        elif not isinstance(docs, list) or not all(isinstance(doc, spacy.tokens.doc.Doc) or isinstance(doc, spacy.tokens.span.Span) for doc in docs):
            if verbose<2:
                print("Warning: Invalid document format. Please provide a single spacy.tokens.doc.Doc, spacy.tokens.span.Span or a list of such objects.")
            return

        if not isinstance(filename, str):
            if verbose<2:
                print("Warning: Invalid filename name format. Please provide a string.")
            return

        if verbose<1:
            print("Number of documents provided: {}.".format(len(docs)))

    filename = shared_module._prepare_filename(filename, '.spacy', overwrite)
    # Create and save a collection of docs
    docbin = DocBin(docs=docs, store_user_data=True)
    docbin.to_disk(filename)
    if verbose<1:
            print("Documents saved as {}.".format(filename))
    return filename


def _legacy_spacy_load_docs(filename, 
                    verbose = 0):
    """
    Load SpaCy documents and associated metadata from disk.

    Parameters
    ----------
    filename : str
        The filename for the documents (.spacy) or metadata (.json) to be loaded. File extensions are ignored.
    verbose : int, optional
        Verbosity level (0: all messages, 1: warnings, 2: no messages) (default is 0).

    Returns
    -------
    tuple
        A tuple containing the loaded SpaCy documents and the metadata dictionary.

    Raises
    ------
    ValueError
        If the filename or metadata is not in the correct format.
    FileNotFoundError
        If the metadata or data file does not exist.

    Examples
    --------
    # Load SpaCy documents with metadata
    >>> docs, metadata = spacy_load_docs('documents')
    
    """
    precheck = True
    filename = shared_module._check_filename(filename, '.json', verbose)
    # if not isinstance(filename, str):
    #     if verbose<2:
    #         print("Warning: Invalid filename format. Please provide a string.")
    # filename = os.path.splitext(filename)[0]
    # if os.path.isfile(filename+'.json'):
    if filename:
        with open(filename+'.json', 'r') as infile:
            metadata = json.load(infile)
    else:
        # if verbose<2:
        #     print("Warning: Metadata file '{}.json' does not exist. Exiting...".format(filename))
        return
        
    # if not os.path.isfile(filename+'.spacy'):
    #     if verbose<2:
    #         print("Warning: Data file '{}.spacy' does not exist. Exiting...".format(filename))
    #     return

    if not isinstance(metadata, dict):
        if verbose<2:
            print("Warning: Wrong format of metadata. Please provide '{'filename':filename, 'model':model, 'labels':labels}'. Exiting...")
        return

    
    if 'filename' not in metadata:
        if verbose<2:
            print("Warning: No filename provided. Exiting...")
        return

    if 'model' not in metadata:
        if verbose<2:
            print("Warning: No model name provided. Exiting...")
        return
        
    if 'labels' not in metadata:
        if verbose<2:
            print("Warning: No document labels provided. Exiting...")
        return
    
    filename = metadata['filename']
    model = metadata['model']
    labels = metadata['labels']
    if labels == []:
        if verbose<1:
            print("Document labels are empty.")
    elif isinstance(labels, str):
        metadata['labels'] = [labels] # Make a list if only one int is given.    
    elif not isinstance(labels, list) or not all(isinstance(f, str) for f in labels):
        if verbose<2:
            print("Warning: Invalid metadata name format. Please provide a single string or a list of strings.")
        return
    
    if not all(isinstance(f, str) for f in [filename,model]):
        if verbose<2:
            print("Warning: Invalid filename or model name format. Please provide a string.")
        return        
    # if filename.count(model_sep)!=1 and not model:
    #     if verbose<2:
    #         print("Warning: unable to retrieve model name. Please provide a filename in the format 'filename{}model' or provide separate 'filename' and 'model' arguments or change the separator.".format(model_sep, filename))
    #         return
    # elif filename.count(model_sep) ==1 and not model:
    #     model = filename.split(model_sep)[1]

    if(model == "pl_core_news_lg"):
        shared_module.nlp = spacy.load(model)
        if verbose<1:
            print("Provided model name for Spacy: '{}'.".format(model))
    else:
        if verbose<1:
            print("Provided model name for spacy_sentence_bert: '{}'.".format(model))
        import spacy_sentence_bert
        shared_module.nlp = spacy_sentence_bert.load_model(model)
    shared_module.nlp.max_length = 1600000 # default 1000 000 - takes ca. 1GB RAM per 100 000 words.
    
    spacy_add_doc_labels(labels, precheck = precheck)    

    filename = shared_module._check_filename(filename, '.spacy', verbose)
    if filename:
        # Load a collection of docs
        docbin = DocBin().from_disk(filename+'.spacy')
        # Deserialize
        docs = list(docbin.get_docs(shared_module.nlp.vocab))
        if verbose<1:
            print("Loaded {} documents.".format(len(docs)))
        # return model_set as a feedback when invoked from base.py to change its class parameters
        return docs, metadata
    else:
        return

def spacy_load_docs(filelist,
                    model,
                    labels,
                    verbose = 0,
                    precheck = False):
    """
    Load SpaCy documents and associated metadata from disk.

    Parameters
    ----------
    filelist : str or list of strings
        Contains the filename(s) for the documents (.spacy) or metadata (.json) to be loaded. File extensions are ignored.
    verbose : int, optional
        Verbosity level (0: all messages, 1: warnings, 2: no messages) (default is 0).

    Returns
    -------
    tuple
        A tuple containing the loaded SpaCy documents and the metadata dictionary.

    Raises
    ------
    ValueError
        If the filename or metadata is not in the correct format.
    FileNotFoundError
        If the metadata or data file does not exist.

    Examples
    --------
    # Load SpaCy documents with metadata
    >>> docs, metadata = spacy_load_docs('documents')
    
    """

    model_set = _choose_spacy_model(model,verbose)
    if model_set:
        shared_module.nlp.max_length = 1600000 # default 1000 000 - takes ca. 1GB RAM per 100 000 words.
    
    spacy_add_doc_labels(labels, precheck = precheck)    

    if isinstance(filelist, str):
        filelist = [filelist]
    
    docbin = DocBin()
    for i in range(len(filelist)):
        filename = filelist.pop(0)
        filename = shared_module._check_filename(filename, '.spacy', verbose)
        if filename:
            # Load a collection of docs
            d = DocBin().from_disk(filename)
            docbin.merge(d)
        else:
            return

    # Deserialize
    docs = list(docbin.get_docs(shared_module.nlp.vocab))
    if verbose<1:
        print("Loaded {} documents.".format(len(docs)))
    return docs, model_set


# Sentence BERT models available (https://github.com/MartinoMensio/spacy-sentence-bert):
#     xx_quora_distilbert_multilingual
#     xx_paraphrase_xlm_r_multilingual_v1
#     xx_distiluse_base_multilingual_cased_v2
#     xx_stsb_xlm_r_multilingual
# Polish spaCy models available (https://spacy.io/models/pl):
#     pl_core_news_sm
#     pl_core_news_md
#     pl_core_news_lg

def spacy_preproc(raw_texts,
                  model,
                  labels = [],
                  verbose = 0):
    """
    Preprocess raw texts using spaCy.

    Parameters
    ----------
    raw_texts : list
        A list of tuples (str, dict) containing (text, metadata).
    model : str
        The name of the spaCy model to use.
    filename : str, optional
        The filename to save the processed documents (default is '').
    labels : list, optional
        A list of labels to assign to the documents (default is to read the labels from the first text's metadata).
    save_to_file : bool, optional
        Whether to save the processed documents to a file (default is True if filename is provided).
    verbose : int, optional
        Verbosity level. Set 0 for no output, 1 for minimal output, and 2 for detailed output. Defaults to 0.

    Returns
    -------
    tuple
        A tuple containing the processed documents (list of spaCy Doc objects) and the metadata (dict).
        
    The processed documents are preprocessed versions of the input raw texts using the specified spaCy model. Each document is represented as a spaCy Doc object.

    If `save_to_file` is True, the processed documents are saved to a file with the specified filename. The saved files contain the preprocessed documents in a spaCy binary format (.spacy) and the metadata in a JSON file (.json). 


    Raises
    ------
    ValueError
        If the input text format is invalid.
    ValueError
        If the model name format is invalid.
    ValueError
        If the filename format is invalid.

    Notes
    -----
    The metadata dictionary includes the following information:
    - 'filename': The filename provided (or an empty string if not provided) without extension.
    - 'model': The name of the spaCy model used for preprocessing.
    - 'labels': The list of labels assigned to the documents.

    Examples
    --------
    # Preprocess raw texts using spaCy model 'en_core_web_sm'
    texts = [('This is a sample text.', {'label': 'sample'})]
    processed_docs, metadata = spacy_preproc(texts, 'en_core_web_sm')

    # Preprocess raw texts and save the processed documents to a file
    texts = [('This is another sample text.', {'label': 'sample'})]
    processed_docs, metadata = spacy_preproc(texts, 'en_core_web_sm', filename='processed', save_to_file=True)
    """
    if not isinstance(raw_texts, list):
        if verbose<2:
            print("Warning: Invalid input text format. Please provide a list of tuples (str, dict) containing (text, metadata).")
        return
    #TO DO: można złagodzić niektóre z tych warunków, nadając własne numery tekstom
    elif not all(isinstance(t, tuple) for t in raw_texts):
        if verbose<2:
            print("Warning: Invalid input text format. Please provide a list of tuples (str, dict) containing (text, metadata).")
        return
    elif not all(isinstance(t[0], str) for t in raw_texts):
        if verbose<2:
            print("Warning: Invalid input text format. Please provide a list of tuples (str, dict) containing (text, metadata).")
        return
    elif not all(isinstance(t[1], dict) for t in raw_texts):
        if verbose<2:
            print("Warning: Invalid input text format. Please provide a list of tuples (str, dict) containing (text, metadata).")
        return
  
    if not isinstance(model, str):
        if verbose<2:
            print("Warning: Invalid model name format. Please provide a string.")
        return
    
    if labels == []:
        labels = list(raw_texts[0][1].keys())
        precheck = True
        if verbose<1:
            print("Reading labels from provided texts.")
    else:
        # Additional check done in spacy_save_docs() and spacy_add_doc_labels()
        precheck = False

    spacy_add_doc_labels(labels, precheck = precheck)    

    model_set = _choose_spacy_model(model,verbose)
    if model_set:
        shared_module.nlp.max_length = 1600000 # default 1000 000 - takes ca. 1GB RAM per 100 000 words.

    if verbose<1:
        print("Number of documents provided: {}.".format(len(raw_texts)))

    docs = []
    for doc, context in tqdm_ui(shared_module.nlp.pipe(raw_texts, as_tuples=True)):
        for l in labels:
            doc._.set(l,context[l])
        # doc._.year = context["year"]
        docs.append(doc)
    # LEGACY
    # metadata['doc_dataframe'] = {l:[d.doc._.get(l) for d in docs] for l in labels}
    # return model_set as a feedback when invoked from base.py to change its class parameters
    return docs, model_set