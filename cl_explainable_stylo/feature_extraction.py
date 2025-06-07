"""
This module provides wrapper functions for textual feature extraction using spaCy.
Combine it with 'preprocess_spacy.py' module, to first load and preprocess the texts,
and later with 'text_classify.py', to classify texts using those features, and 'text_visualise.py', to plot results and their SHAP explanations.

It includes functions to ...

Author: Jeremi K. Ochab
Date: June 27, 2023
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from . import preprocess_spacy, shared_module
nlp = shared_module.nlp
# Global control of tqdm progress bar instead of providing it in each function separately.
tqdm_notebook = shared_module.tqdm_notebook
tqdm_display = shared_module.tqdm_display

import spacy
import numpy as np


def choose_features(doc,feature=10,
                    verbose=0,
                    string=True,
                    tqdm_display = False):
    """
    Extracts selected type of features from Doc.

    Parameters
    ----------
    doc : spacy.tokens.doc.Doc or spacy.tokens.span.Span
        The input document preprocessed by Spacy.
    feature : int
        The numeral for selecting the feature type.

    Returns
    -------
    str
        A string containing the selected space-separated features.

    Raises
    ------
    None

    Notes
    -----
    The function supports counting different types of features based on the provided feature IDs.
    Available feature IDs and their corresponding types:
    - Tokens:
        - 10: Select all words.
        - 11: Select all lemmas.
        - 12: Select non-NER words, replacing named entities with their entity type.
        - 13: Select non-NER lemmas.
    - Token N-grams, dependency-based:
        - 20: Select all words and punctuation (excluding numerals) in dependency-based bigrams.
        - 21: Select all lemmas in dependency-based bigrams.
        - 22: Select non-NER words in dependency-based bigrams, replacing named entities with their entity type.
        - 23: Select non-NER lemmas in dependency-based bigrams, replacing named entities with their entity type.
    - Part-of-speech tags:
        - 30: Select all parts of speech.
        - 31: Select parts of speech without punctuation.
    - Dependency-based tags:
        - 40: Select dependency labels without punctuation.
    - Morphology annotation:
        - 50: Select morphology annotations with punctuation.
        - 51: Select morphology annotations without punctuation.
        - 52: Select non-NER morphology annotations, replacing named entities with their entity type.
    - Named entities:
        - 60: Select NER types.
        - 61: Select all named entities.

    """
    valid_features = [10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 40, 50, 51, 52, 60, 61]
    if not ( isinstance(doc, spacy.tokens.doc.Doc) or isinstance(doc, spacy.tokens.span.Span)):
        if verbose<2:
            print("Warning: Invalid document format. Please provide a single spacy.tokens.doc.Doc or spacy.tokens.span.Span.\nSee Notes in help(choose_features).")
        return
    if not isinstance(feature, int):
        if verbose<2:
            print("Warning: Invalid features format. Please provide a single integer.\nSee Notes in help(choose_features).")
        return
    elif feature not in valid_features:
        if verbose<2:
            print("Warning: Invalid feature provided: {} \nSee Notes in `help(choose_features)' for available options.".format(feature))
        return
    # TO DO: nie wszystkie dane mają tytuły tekstów! Zastąpić czymś ogólniejszym.
    # if verbose<1:
    #         print("From {}, {}:".format(doc.doc._.author,doc.doc._.title))

    
    # VERSION 1:
    # ---------all words
    if(feature==10):
        if verbose<1:
            print("-- Extracting all words.")
        words = [token.lower_ for token in tqdm_notebook(doc,display = tqdm_display) if not token.is_punct] # tokens that arent punctuations; wystarczy mniej cech
    # noun tokens that arent stop words or punctuations
    # if (not token.is_stop and not token.is_punct and token.pos_ == "NOUN")

    # ---------all lemmas
    elif(feature==11):
        if verbose<1:
            print("-- Extracting all lemmas.")
        words = [token.lemma_ for token in tqdm_notebook(doc,display = tqdm_display) if not token.is_punct]    
    # ---------non-NER words
        # tokens that arent punctuations
        # replaces a token with its named entity type if it is a part of an entity ("San Francisco" -> "placeName placeName")    
    elif(feature==12):
        if verbose<1:
            print("-- Extracting non-NER words (replacing named entities with their entity type).")
        words = [token.text.lower() if token.ent_type_ == '' 
                 else token.ent_type_ for token in tqdm_notebook(doc,display = tqdm_display) if not token.is_punct]
    # ---------non-NER lemmas
    elif(feature==13):
        if verbose<1:
            print("-- Extracting non-NER lemmas (replacing named entities with their entity type).")
        words = [token.lemma_ if token.ent_type_ == '' 
                 else token.ent_type_ for token in tqdm_notebook(doc,display = tqdm_display) if not token.is_punct]
    
    # VERSION 2: dependency-based bigrams: ancestor_child
    # tokeny rozdzielam '_', żeby było to jednoznaczne dla CountVectorizera (nie whitespace, bo zostanie usunięte; nie znaki wewnątrzwyrazowe)
    # i jednocześnie, żeby przełknął to LGBM (nazwy cech mogą mieć ograniczony zakres znaków niealfanumerycznych)
    # ---------all words and punctuation, no numerals
    elif(feature==20):
        if verbose<1:
            print("-- Extracting dependency-based word bigrams (including punctuation, excluding numerals).")
        words = ['_'.join([token.lower_, f.lower_]) for token in tqdm_notebook(doc,display = tqdm_display)
         for f in token.children if not f.like_num]
        # Stara wersja; niejednoznaczna 
        # words = ['_'.join([f.lower_ 
        #                    for f in token.subtree
        #                    if not (f.is_punct or f.like_num)]) for token in tqdm_notebook(doc,display = tqdm_display)
        #  if sum([not (f.is_punct or f.like_num) for f in token.subtree])>1 
        #  and sum([not (f.is_punct or f.like_num) for f in token.subtree])<3]
    # ---------all lemmas
    elif(feature==21):
        if verbose<1:
            print("-- Extracting dependency-based lemma bigrams (including punctuation, excluding numerals).")
        words = ['_'.join([token.lemma_, f.lemma_]) for token in tqdm_notebook(doc,display = tqdm_display)
         for f in token.children if not f.like_num]
    # ---------non-NER words
    elif(feature==22):
        if verbose<1:
            print("-- Extracting dependency-based non-NER word bigrams (including punctuation, excluding numerals, replacing named entities with their entity type).")
        words = ['_'.join([token.lower_ if token.ent_type_ == '' else token.ent_type_,
                           f.lower_  if f.ent_type_ == '' else f.ent_type_])
                 for token in tqdm_notebook(doc,display = tqdm_display) for f in token.children if not f.like_num]
    # ---------non-NER lemmas
    elif(feature==23):
        if verbose<1:
            print("-- Extracting dependency-based non-NER lemma bigrams (including punctuation, excluding numerals, replacing named entities with their entity type).")
        words = ['_'.join([token.lemma_ if token.ent_type_ == '' else token.ent_type_,
                           f.lemma_  if f.ent_type_ == '' else f.ent_type_])
                 for token in tqdm_notebook(doc,display = tqdm_display) for f in token.children if not f.like_num]    
    
    # VERSION 3: POS
    elif(feature==30):
        if verbose<1:
            print("-- Extracting all parts of speech.")
        words = [token.pos_ for token in tqdm_notebook(doc,display = tqdm_display)]
    # ---------without punctuation
    elif(feature==31):
        if verbose<1:
            print("-- Extracting all parts of speech (no punctuation).")
        words = [token.pos_ for token in tqdm_notebook(doc,display = tqdm_display) if not token.is_punct]
    elif(feature==32):
        if verbose<1:
            print("-- Extracting all parts of speech (no punctuation).")
        words = [token.pos_ for token in tqdm_notebook(doc,display = tqdm_display) if not token.pos_ == 'SPACE']
    
    # VERSION 4: dependency without punctuation
    elif(feature==40):
        if verbose<1:
            print("-- Extracting dependency labels without punctuation.")
        words = [token.dep_ for token in tqdm_notebook(doc,display = tqdm_display) if not token.is_punct]
        
    # VERSION 5: morphology annotation with punctuation
    elif(feature==50):
        if verbose<1:
            print("-- Extracting morphology annotations with punctuation.")
        words = [str(token.morph) for token in tqdm_notebook(doc,display = tqdm_display)]
#         niektóre słowa, np. 'a', 'aby', 'albo', 'ale', itp. ('ADV', 'CCONJ', 'INTJ', 'PART', 'SCONJ', 'SYM')
#         mają pustą morfologię '' i nie pojawią się jako cecha
    # ---------without punctuation
    elif(feature==51):
        if verbose<1:
            print("-- Extracting morphology annotations without punctuation.")
        words = [str(token.morph) for token in tqdm_notebook(doc,display = tqdm_display) if not token.is_punct] 
        
    # ---------non-NER morphology
    elif(feature==52):
        if verbose<1:
            print("-- Extracting morphology annotations with punctuation (replacing named entities with their entity type).")
        words = [str(token.morph) if token.ent_type_ == '' else token.ent_type_ for token in tqdm_notebook(doc,display = tqdm_display)]

    # VERSION 6: same NER-y
    # ---------NER type
    elif(feature==60):
        if verbose<1:
            print("-- Extracting all named-entity types.")
        words = [token.ent_type_ for token in doc]
    # ---------all NERs
    elif(feature==61):
        if verbose<1:
            print("-- Extracting all named entities.")
        words = [token.text for token in tqdm_notebook(doc.ents,display = tqdm_display)]

    if string:
        words = ' '.join(words)
        
    return words

def _legacy_subsample_docs(docs,
                   metadata,
                   sample_type='t',
                   sample_length = 3200,
                   save_samples = False,
                   verbose = 0,
                   precheck = False):
                   # tqdm_display = False):
    """
    Subsample documents based on the specified sampling type and length.

    Parameters
    ----------
    docs : spacy.tokens.doc.Doc or list
        A single SpaCy document or a list of SpaCy documents to be subsampled.
    sample_type : str, optional
        The type of subsampling to perform (default is 't').
        - 't': Divide by tokens.
        - 's': Divide by sentences.
        - 'ts': Divide by tokens, rounded to sentences.
        - 'none': No subsampling (return the input documents unchanged).
    sample_length : int, optional
        The length of each subsampled document (default is 3200).
    verbose : int, optional
        Verbosity level (0: all messages, 1: warnings, 2: no messages) (default is 0).
    precheck : bool, optional
        If True, skip the input type validation step (default is False).
        Used together with other feature_extraction functions.

    Returns
    -------
    list
        A list of subsampled SpaCy documents.

    Notes
    -----
    - For 't' (tokens) sampling, each document is divided into segments of length `sample_length`.
    - For 's' (sentences) sampling, each document is divided into segments of length `sample_length` sentences.
    - For 'ts' (tokens rounded to sentences) sampling, each document is divided into segments of length `sample_length` tokens, rounded to the nearest sentence boundaries.
    - For 'none' sampling, the input documents are returned unchanged.
    - Other bag-of-words subsampling types ('random-t' and 'random-s') are not implemented yet. They preclude full feature extraction.

    Raises
    ------
    ValueError
        If the input documents are not in the correct format.
    ValueError
        If the sampling type is not one of the supported types.
    ValueError
        If the sample length is not a positive integer.

    Examples
    --------
    # Subsample documents by tokens rounded to sentences with a sample length of 3200
    >>> subsampled_docs = subsample_docs(docs, sample_type='ts', sample_length=3200)

    # No subsampling, return the input documents unchanged
    >>> subsampled_docs = subsample_docs(docs, sample_type='none')
    """    
    default_sample_length = {'t':3200, 'random-t':3200, 's':200, 'random-s': 200,'ts':3200,'none':0}
    #---- START Check input types
    if not precheck:
        if isinstance(docs, spacy.tokens.doc.Doc):
            docs = [docs] # Make a list if only one Doc is given.
        elif not isinstance(docs, list) or not all(isinstance(doc, spacy.tokens.doc.Doc) for doc in docs):
            if verbose<2:
                print("Warning: Invalid document format. Please provide a single spacy.tokens.doc.Doc or a list of spacy.tokens.doc.Doc.")
            return
        if verbose<1:
            print("Number of documents provided: {}.".format(len(docs)))
    
    if not sample_type in list(default_sample_length.keys()):
        if verbose<2:
            print("Warning: Invalid sampling type format. Please provide one of the strings: {}.".format(list(default_sample_length.keys())))
        return
    
    if not isinstance(sample_length, int):            
        if verbose<2:
            print("Warning: Invalid sample length provided. Please provide a single integer.")
        return
    if not sample_length > 0 and not sample_type == 'none':
        if verbose<2:
            print("Warning: Invalid sample length provided.")
        sample_length = default_sample_length[sample_type]
        if verbose<1:
            print("Proceeding with default: {}.".format(default_sample_length[sample_type]))
    #---- END Check input types    
    
    sdocs =[]
    sample_name = 'subsamples_{}_{}'.format(sample_type,str(sample_length))
    for doc in docs:
        doc.spans[sample_name]=[]
    # ---------no subsampling
    if(sample_type=='none'):
        sdocs = docs
    else:
        # ---------divide by tokens
        if(sample_type=='t'):
            for doc in tqdm_notebook(docs,display = tqdm_display):
                for sample in range(int(np.floor(len(doc)/sample_length))):
                    sdoc = doc[sample*sample_length:np.min([sample*sample_length+sample_length,len(doc)])]
                    # sdocs.append(sdoc)
                    doc.spans[sample_name].append(sdoc)

        # ---------divide by sentences
        elif(sample_type=='s'):
            for doc in tqdm_notebook(docs,display = tqdm_display):
                sents = [sent for sent in doc.sents]
                for sample in range(int(np.floor(len(sents)/sample_length))):
                    sdoc = doc[sents[sample].start:sents[sample+sample_length].end]
                    # sdocs.append(sdoc)
                    doc.spans[sample_name].append(sdoc)

        # ---------divide by words, but round to sentences
        # sample_length musi być większe od długości zdania, żeby zdania się nie powtórzyły w sąsiadujących próbkach.
        elif(sample_type=='ts'):
            for doc in tqdm_notebook(docs,display = tqdm_display):
                for sample in range(int(np.floor(len(doc)/sample_length))):
                    sdoc = doc[doc[sample*sample_length].sent.start:doc[(sample+1)*sample_length].sent.start]
                    # sdocs.append(sdoc)
                    doc.spans[sample_name].append(sdoc)

    # TO DO: po losowaniu słów traci się możliwość analizy DEP, n-gramów itp. Trzeba by to zapisywać w jakichś metadanych i wyłączać te możliwości.
        # ---------bag of words
        elif(sample_type=='random-t'):
            if verbose<1:
                print("Sampling type not implemented yet.")
            sdocs = docs
        # ---------bag of sentences
        elif(sample_type=='random-s'):
            if verbose<1:
                print("Sampling type not implemented yet.")
            sdocs = docs
        
        for doc in docs:
            sdocs.extend(doc.spans[sample_name])
            
        if verbose<1:
            print("Number of text samples produced: {}.".format(len(sdocs)))

        # Dataframe for storing sample metadata
        # TO DO: Add sampling scheme
        if save_samples:
            metadata[sample_name] = {l:[d.doc._.get(l) for d in sdocs] for l in metadata['labels']}
            preprocess_spacy.spacy_save_docs(docs,metadata)
            
    return sdocs

def subsample_docs(docs,
                   metadata,
                   subsample_scheme = {'sample_type':'none','sample_length':800},
                   verbose = 0,
                   precheck = False):
                   # tqdm_display = False):
    """
    Subsample documents based on the specified sampling type and length.

    Parameters
    ----------
    docs : spacy.tokens.doc.Doc or list
        A single SpaCy document or a list of SpaCy documents to be subsampled.
    sample_type : str, optional
        The type of subsampling to perform (default is 't').
        - 't': Divide by tokens.
        - 's': Divide by sentences.
        - 'ts': Divide by tokens, rounded to sentences.
        - 'none': No subsampling (return the input documents unchanged).
    sample_length : int, optional
        The length of each subsampled document (default is 3200).
    verbose : int, optional
        Verbosity level (0: all messages, 1: warnings, 2: no messages) (default is 0).
    precheck : bool, optional
        If True, skip the input type validation step (default is False).
        Used together with other feature_extraction functions.

    Returns
    -------
    list
        A list of subsampled SpaCy documents.

    Notes
    -----
    - For 't' (tokens) sampling, each document is divided into segments of length `sample_length`.
    - For 's' (sentences) sampling, each document is divided into segments of length `sample_length` sentences.
    - For 'ts' (tokens rounded to sentences) sampling, each document is divided into segments of length `sample_length` tokens, rounded to the nearest sentence boundaries.
    - For 'none' sampling, the input documents are returned unchanged.
    - Other bag-of-words subsampling types ('random-t' and 'random-s') are not implemented yet. They preclude full feature extraction.

    Raises
    ------
    ValueError
        If the input documents are not in the correct format.
    ValueError
        If the sampling type is not one of the supported types.
    ValueError
        If the sample length is not a positive integer.

    Examples
    --------
    # Subsample documents by tokens rounded to sentences with a sample length of 3200
    >>> subsampled_docs = subsample_docs(docs, sample_type='ts', sample_length=3200)

    # No subsampling, return the input documents unchanged
    >>> subsampled_docs = subsample_docs(docs, sample_type='none')
    """    
    default_sample_length = {'t':3200, 'random-t':3200, 's':200, 'random-s': 200,'ts':3200,'none':0}
    valid_keys = ['sample_type','sample_length']
    #---- START Check input types
    if not precheck:
        if isinstance(docs, spacy.tokens.doc.Doc):
            docs = [docs] # Make a list if only one Doc is given.
        elif not isinstance(docs, list) or not all(isinstance(doc, spacy.tokens.doc.Doc) for doc in docs):
            if verbose<2:
                print("Warning: Invalid document format. Please provide a single spacy.tokens.doc.Doc or a list of spacy.tokens.doc.Doc.")
            return
        if verbose<1:
            print("Number of documents provided: {}.".format(len(docs)))

        if not isinstance(subsample_scheme, dict):
            if verbose<2:
                print("Warning: Invalid sampling scheme format. Please provide a dict.")
            return
        elif not all([k in subsample_scheme.keys() for k in valid_keys]):
            if verbose<2:
                print("Warning: Invalid keys in sampling scheme. Please provide all {}.".format(valid_keys))
            return
        
    sample_type = subsample_scheme['sample_type']
    sample_length = subsample_scheme['sample_length']
    
    if not precheck:    
        if not sample_type in list(default_sample_length.keys()):
            if verbose<2:
                print("Warning: Invalid sampling type format. Please provide one of the strings: {}.".format(list(default_sample_length.keys())))
            return
        
        if not isinstance(sample_length, int):            
            if verbose<2:
                print("Warning: Invalid sample length provided. Please provide a single integer.")
            return
        if not sample_length > 0 and not sample_type == 'none':
            if verbose<2:
                print("Warning: Invalid sample length provided.")
            sample_length = default_sample_length[sample_type]
            if verbose<1:
                print("Proceeding with default: {}.".format(default_sample_length[sample_type]))
    #---- END Check input types    
    
    sdocs =[]
    labels = metadata['labels']
    sample_name = 'subsamples_{}_{}'.format(sample_type,str(sample_length))
    for doc in docs:
        doc.spans[sample_name]=[]
    # ---------no subsampling
    if(sample_type=='none'):
        sdocs = docs
    else:
        # ---------divide by tokens
        if(sample_type=='t'):
            for doc in tqdm_notebook(docs,display = tqdm_display):
                for sample in range(int(np.floor(len(doc)/sample_length))):
                    sdoc = doc[sample*sample_length:np.min([sample*sample_length+sample_length,len(doc)])]
                    doc.spans[sample_name].append(sdoc)

                    sdoc = sdoc.as_doc()
                    for l in labels:
                        sdoc._.set(l,doc._.get(l))
                    sdocs.append(sdoc)

        # ---------divide by sentences
        elif(sample_type=='s'):
            for doc in tqdm_notebook(docs,display = tqdm_display):
                sents = [sent for sent in doc.sents]
                for sample in range(int(np.floor(len(sents)/sample_length))):
                    sdoc = doc[sents[sample].start:sents[sample+sample_length].end]
                    doc.spans[sample_name].append(sdoc)
                    sdoc = sdoc.as_doc()
                    for l in labels:
                        sdoc._.set(l,doc._.get(l))
                    sdocs.append(sdoc)

        # ---------divide by words, but round to sentences
        # sample_length musi być większe od długości zdania, żeby zdania się nie powtórzyły w sąsiadujących próbkach.
        elif(sample_type=='ts'):
            for doc in tqdm_notebook(docs,display = tqdm_display):
                for sample in range(int(np.floor(len(doc)/sample_length))):
                    sdoc = doc[doc[sample*sample_length].sent.start:doc[(sample+1)*sample_length].sent.start]
                    doc.spans[sample_name].append(sdoc)
                    sdoc = sdoc.as_doc()
                    for l in labels:
                        sdoc._.set(l,doc._.get(l))
                    sdocs.append(sdoc)

    # TO DO: po losowaniu słów traci się możliwość analizy DEP, n-gramów itp. Trzeba by to zapisywać w jakichś metadanych i wyłączać te możliwości.
        # ---------bag of words
        elif(sample_type=='random-t'):
            if verbose<1:
                print("Sampling type not implemented yet.")
            sdocs = docs
        # ---------bag of sentences
        elif(sample_type=='random-s'):
            if verbose<1:
                print("Sampling type not implemented yet.")
            sdocs = docs
        
        # for doc in docs:
        #     sdocs.extend(doc.spans[sample_name])
            
        if verbose<1:
            print("Number of text samples produced: {}.".format(len(sdocs)))

        smeta = {'sample_name':sample_name,
                 'sample_type':sample_type,
                 'sample_length':sample_length,
                 'files':{l:[d.doc._.get(l) for d in sdocs] for l in labels}}
            
    return sdocs, smeta



def count_features(docs,
                   feature_scheme = {'features':[13,23,30,52],
                                     'max_features':1000,
                                     'n_grams_word':(1,3),
                                     'n_grams_pos':(1,3),
                                     'n_grams_dep':(1,3),
                                     'n_grams_morph':(1,1),
                                     'min_cull_word':0., # ignore terms that have a document frequency strictly lower than the given threshold
                                     'max_cull_word':1., # ignore terms that have a document frequency strictly higher than the given threshold
                                     'min_cull_d2':0.,
                                     'max_cull_d2':1.,
                                     'remove_duplicates':True},
                   verbose = 0,
                   tqdm_propagate = False):
    """
    Count the features in the given documents.

    Parameters
    ----------
    docs : spacy.tokens.doc.Doc or spacy.tokens.span.Span or list
        A single `spacy.tokens.doc.Doc` object or `spacy.tokens.span.Span` or a list of such objects representing the documents.
    features : int or list
        A single integer or a list of integers representing the features to count.
    max_features : int, optional
        The maximum number of features to consider (default is 1000).
    n_grams_word : tuple, optional
        The n-gram range for word features (default is (1, 3)).
    n_grams_pos : tuple, optional
        The n-gram range for POS features (default is (1, 3)).
    n_grams_dep : tuple, optional
        The n-gram range for dependency features (default is (1, 3)).
    n_grams_morph : tuple, optional
        The n-gram range for morphological features (default is (1, 1)).
    min_cull_word : int, optional
        The minimum document frequency threshold for token (word/lemma) features (default is 0). Ignores features that have a document frequency strictly lower than the given threshold.
    max_cull_word : int, optional
        The maximum document frequency threshold for token (word/lemma) features (default is 1). Ignores features that have a document frequency strictly higher than the given threshold.
    min_cull_d2 : int, optional
        The minimum document frequency threshold for dependency features (default is 0).
    max_cull_d2 : int, optional
        The maximum document frequency threshold for dependency features (default is 1).
    remove_duplicates : bool, optional
        Remove feature duplicates resulting from different feature types (default is True).
    verbose : int, optional
        Verbosity level (0: all messages, 1: warnings, 2: no messages) (default is 0).
    tqdm_propagate : bool, optional
        Display tqdm progress bars for functions called from within count_features (default is False).

    Returns
    -------
    tuple
        A tuple containing the feature names and the feature matrix.
        - feature_names : list
            A list of feature names formatted as 'FeatureName_FeatureTypeID'.
        - dfX : pandas.DataFrame
            A DataFrame representing the feature matrix.

    Notes
    -----
    The function supports counting the same types of features as `choose_features()`.
    Available feature IDs and their corresponding types:
    - Tokens:
        - 10: Select all words.
        - 11: Select all lemmas.
        - 12: Select non-NER words, replacing named entities with their entity type.
        - 13: Select non-NER lemmas.
    - Token N-grams, dependency-based:
        - 20: Select all words and punctuation (excluding numerals) in dependency-based bigrams.
        - 21: Select all lemmas in dependency-based bigrams.
        - 22: Select non-NER words in dependency-based bigrams, replacing named entities with their entity type.
        - 23: Select non-NER lemmas in dependency-based bigrams, replacing named entities with their entity type.
    - Part-of-speech tags:
        - 30: Select all parts of speech.
        - 31: Select parts of speech without punctuation.
        - 32: Select parts of speech without 'SPACE'.
    - Dependency-based tags:
        - 40: Select dependency labels without punctuation.
    - Morphology annotation:
        - 50: Select morphology annotations with punctuation.
        - 52: Select non-NER morphology annotations, replacing named entities with their entity type.
    - Named entities:
        - 60: Select NER types.
        - 61: Select all named entities.
    - Invalid feature IDs will be skipped, and a warning message will be displayed if `verbose` is set to 0 or 1.
    
    Example situation where removing duplicates might be needed:
    - Suppose that in a given corpus the feature 'PronType=Prs|Reflex=Yes_50' is always realised with 'się_10' (word) and the corresponding 'się_11' (lemma). These three features will be reduced to one.
    - A specific POS 5-gram has only one corresponding word 5-gram realisation.
    
    """
    #---- START Check input types
    if isinstance(docs, spacy.tokens.doc.Doc) or isinstance(docs,spacy.tokens.span.Span):
        docs = [docs] # Make a list if only one Doc is given.
    elif not isinstance(docs, list) or not all(isinstance(doc, spacy.tokens.doc.Doc) or isinstance(doc, spacy.tokens.span.Span) for doc in docs):
        if verbose<2:
            print("Warning: Invalid document format. Please provide a single spacy.tokens.doc.Doc, spacy.tokens.span.Span or a list of such objects.")
        return
    if verbose<1:
        print("Number of documents provided: {}.".format(len(docs)))
    

    features = feature_scheme['features']
    max_features = feature_scheme['max_features']
    n_grams_word = feature_scheme['n_grams_word']
    n_grams_pos = feature_scheme['n_grams_pos']
    n_grams_dep = feature_scheme['n_grams_dep']
    n_grams_morph = feature_scheme['n_grams_morph']
    min_cull_word = feature_scheme['min_cull_word']
    max_cull_word = feature_scheme['max_cull_word']
    min_cull_d2 = feature_scheme['min_cull_d2']
    max_cull_d2 = feature_scheme['max_cull_d2']
    remove_duplicates = feature_scheme['remove_duplicates']
    
    valid_features = [10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 40, 50, 52, 60, 61]
    if isinstance(features, int):
        features = [features] # Make a list if only one int is given.
    elif not isinstance(features, list) or not all(isinstance(f, int) for f in features):
        if verbose<2:
            print("Warning: Invalid features format. Please provide a single integer or a list of integers.")
        return
    
    invalid_features = [f for f in features if f not in valid_features]
    features = [f for f in features if f in valid_features]
    
    if invalid_features:
        if verbose<2:
            print("Warning: Invalid features were provided and will be dropped: {} \nSee Notes in `help(choose_features)' for available options.".format(", ".join(str(f) for f in invalid_features)))

    if verbose<1:
            print("Features to be extracted: {}.".format(features))
    
    precheck = True
    global tqdm_display
    
#     TO DO: add metadata checking from preprocess_spacy.spacy_save_docs()
    
    #---- END Check input types
     

    feature_params = {
        range(0, 20): {
            'max_features': max_features,
            'ngram_range': n_grams_word,
            'min_df': min_cull_word,
            'max_df': max_cull_word
        },
        range(20, 30): {# dla dep-bigramów nie ustawiać n-gramów
            'max_features': max_features,
            'token_pattern': r'(?u)\b\w+_\w+\b',
            'min_df': min_cull_d2,
            'max_df': max_cull_d2
        },
        range(30, 40): {# dla .pos_ # max_df nie używane, bo pewnie wszystko zniknie
            'max_features': max_features,
            'ngram_range': n_grams_pos,
            'lowercase': False
        },
        range(40, 50): {#,token_pattern=r'(?u)\b\w+_\w+\b')# dla .dep_ # max_df nie używane, bo pewnie wszystko zniknie
            'max_features': max_features,
            'ngram_range': n_grams_dep,
            'token_pattern': r'(?u)\b\w[:\w]+\b'
        },
        range(50, 60): {# dla .morph
            'max_features': max_features,
            'ngram_range': n_grams_morph,
            'token_pattern': r'(?u)\b[\w=\|]+\b',
            'lowercase': False
        },
        range(60, 70): {# dla NER-ów (nie używać n-gramów, bo CountVectorizer nie widzi odstępów)
            'max_features': max_features,
            'lowercase': False
        }
    }
    

    
    #---- START Counting features depending on their type
    feature_names = None
    for lt in tqdm_notebook(features,display = tqdm_display,miniters=1):
        params = next((params for rng, params in feature_params.items() if lt in rng), None)
        if params is None:
            if verbose < 2:
                print("Warning: Invalid feature {}. Skipping...".format(lt))
            continue

        count_vect = CountVectorizer(**params)
        tqdm_display = tqdm_propagate
        texts = [choose_features(doc,feature = lt,verbose = 1 if i > 0 else verbose) for i, doc in enumerate(docs)]
        tqdm_display = shared_module.tqdm_display
        X0 = count_vect.fit_transform(texts)
        X0 = X0.astype('float32')

        # if lt<30 && lt>=20: # może też dla .dep_
        # Muszę tu zmienić na łącznik
        feature_names0 = [f.replace(':','_') for f in list(count_vect.get_feature_names_out())]
        # else:
        # feature_names0 = list(count_vect.get_feature_names_out())

        # UWAGA na cechy z różnych features, które nazywają się tak samo
        feature_names0[:] = [s + '_'+ str(lt) for s in feature_names0] 

        dfX0 = pd.DataFrame(X0.toarray(),columns=feature_names0)

        # Append to the feature table
        if feature_names == None: #if X == None: 
            feature_names = feature_names0
            dfX = dfX0.copy()
        else:
            feature_names.extend(feature_names0)
            dfX = dfX.join(dfX0)
            
    #---- END Counting features depending on their type
    
    if remove_duplicates == True:
        dfX = remove_duplicate_features(docs, dfX, features, n_grams_word)
        # feature_names = dfX.columns.to_list()
        
    return dfX


# SLOW! Need to use count vectorizer again?
# No feature validation implemented
from collections import defaultdict, Counter
from itertools import islice
import pandas as pd

def generate_ngrams(tokens, n):
    """Generate n-grams from a list of tokens."""
    return zip(*(islice(tokens, i, None) for i in range(n)))


def count_features_list(docs,
                        feature_list,
                        verbose=0,
                        tqdm_propagate = False):
    """
    Count the given features in each document, supporting varying n-gram sizes,
    efficient n-gram generation, and integer counts.

    Parameters
    ----------
    docs : list of spacy Doc or Span
        The input documents for feature extraction.
    feature_list : list of str
        Features in the form "<feature_string>_<feature_type_id>".
    verbose : int
        Verbosity level; prints progress if > 0.

    Returns
    -------
    pd.DataFrame
        DataFrame of shape (len(docs), len(feature_list)) with integer counts.
    """
    #---- START Check input types
    if isinstance(docs, spacy.tokens.doc.Doc) or isinstance(docs,spacy.tokens.span.Span):
        docs = [docs] # Make a list if only one Doc is given.
    elif not isinstance(docs, list) or not all(isinstance(doc, spacy.tokens.doc.Doc) or isinstance(doc, spacy.tokens.span.Span) for doc in docs):
        if verbose<2:
            print("Warning: Invalid document format. Please provide a single spacy.tokens.doc.Doc, spacy.tokens.span.Span or a list of such objects.")
        return
    if verbose<1:
        print("Number of documents provided: {}.".format(len(docs)))


    # Map each feature to its column index
    feature_positions = {feat: idx for idx, feat in enumerate(feature_list)}

    # Organize features by suffix and n-gram size
    suffix_map = defaultdict(lambda: defaultdict(list))  # {suffix: {n: [tuple_tokens]}}
    for feat in feature_list:
        suffix = feat[-2:]
        feat_text = feat[:-3]
        tokens = tuple(feat_text.split())
        suffix_map[suffix][len(tokens)].append(tokens)

    # Prepare result storage
    results = [[0] * len(feature_list) for _ in docs]

    # Process each document
    for i, doc in tqdm_notebook(enumerate(docs),display = tqdm_display,miniters=1):
        if verbose:
            print(f"Processing document {i+1}/{len(docs)}")

        # For each feature type suffix
        for suffix, ngram_groups in suffix_map.items():
            feature_type = int(suffix)

            # Assume choose_features returns a list of strings for this doc and feature
            tokens = choose_features(doc, feature=feature_type, string = False, verbose=2, tqdm_display=False)

            # Generate and count n-grams per size
            counters = {}
            for n, feature_tuples in ngram_groups.items():
                counters[n] = Counter(generate_ngrams(tokens, n))

            # Fill counts for each feature in this suffix
            for n, feature_tuples in ngram_groups.items():
                counter_n = counters[n]
                for feat_tuple in feature_tuples:
                    count = counter_n.get(feat_tuple, 0)
                    feat_name = f"{' '.join(feat_tuple)}_{suffix}"
                    col_idx = feature_positions[feat_name]
                    results[i][col_idx] = count

    # Build DataFrame with integer dtype
    df = pd.DataFrame(results, columns=feature_list, dtype=float)
    
    # SIMPLE OLD VERSION
    # df = pd.DataFrame(None,columns= feature_list)
    # features = set([f[-2:] for f in feature_list])
    # for i, doc in tqdm_notebook(enumerate(docs),display = tqdm_display,miniters=1):
    #     counts = []
    #     for lt in features:
    #         text = choose_features(doc,feature = int(lt),verbose = 1 if i > 0 else verbose)
    #         counts += [text.count(f[:-3]) for f in feature_list if f[-2:]==lt]
    #     df.loc[-1] = np.array(counts,dtype='float')
    #     df.index = df.index + 1
    #     df = df.sort_index()
    
    return df

    
def _legacy_count_features(docs,metadata,
                   feature_scheme = {'features':[13,23,30,52],
                                     'max_features':1000,
                                     'n_grams_word':(1,3),
                                     'n_grams_pos':(1,3),
                                     'n_grams_dep':(1,3),
                                     'n_grams_morph':(1,1),
                                     'min_cull_word':0., # ignore terms that have a document frequency strictly lower than the given threshold
                                     'max_cull_word':1., # ignore terms that have a document frequency strictly higher than the given threshold
                                     'min_cull_d2':0,
                                     'max_cull_d2':1,
                                     'remove_duplicates':True},
                   subsample_scheme = {'sample_type':'none','sample_length':0,'save_samples':False},
                   verbose = 0,
                   tqdm_propagate = False):
    """
    Count the features in the given documents.

    Parameters
    ----------
    docs : spacy.tokens.doc.Doc or spacy.tokens.span.Span or list
        A single `spacy.tokens.doc.Doc` object or `spacy.tokens.span.Span` or a list of such objects representing the documents.
    features : int or list
        A single integer or a list of integers representing the features to count.
    max_features : int, optional
        The maximum number of features to consider (default is 1000).
    n_grams_word : tuple, optional
        The n-gram range for word features (default is (1, 3)).
    n_grams_pos : tuple, optional
        The n-gram range for POS features (default is (1, 3)).
    n_grams_dep : tuple, optional
        The n-gram range for dependency features (default is (1, 3)).
    n_grams_morph : tuple, optional
        The n-gram range for morphological features (default is (1, 1)).
    min_cull_word : int, optional
        The minimum document frequency threshold for token (word/lemma) features (default is 0). Ignores features that have a document frequency strictly lower than the given threshold.
    max_cull_word : int, optional
        The maximum document frequency threshold for token (word/lemma) features (default is 1). Ignores features that have a document frequency strictly higher than the given threshold.
    min_cull_d2 : int, optional
        The minimum document frequency threshold for dependency features (default is 0).
    max_cull_d2 : int, optional
        The maximum document frequency threshold for dependency features (default is 1).
    subsample_scheme : dict, optional
        Parameters provided to `subsample_docs()`. The format is `{'sample_type': str,'sample_length': int}` (default is {'sample_type':'none','sample_length':0}).
    remove_duplicates : bool, optional
        Remove feature duplicates resulting from different feature types (default is True).
    verbose : int, optional
        Verbosity level (0: all messages, 1: warnings, 2: no messages) (default is 0).
    tqdm_propagate : bool, optional
        Display tqdm progress bars for functions called from within count_features (default is False).

    Returns
    -------
    tuple
        A tuple containing the feature names and the feature matrix.
        - feature_names : list
            A list of feature names formatted as 'FeatureName_FeatureTypeID'.
        - dfX : pandas.DataFrame
            A DataFrame representing the feature matrix.

    Notes
    -----
    The function supports counting the same types of features as `choose_features()`.
    Available feature IDs and their corresponding types:
    - Tokens:
        - 10: Select all words.
        - 11: Select all lemmas.
        - 12: Select non-NER words, replacing named entities with their entity type.
        - 13: Select non-NER lemmas.
    - Token N-grams, dependency-based:
        - 20: Select all words and punctuation (excluding numerals) in dependency-based bigrams.
        - 21: Select all lemmas in dependency-based bigrams.
        - 22: Select non-NER words in dependency-based bigrams, replacing named entities with their entity type.
        - 23: Select non-NER lemmas in dependency-based bigrams, replacing named entities with their entity type.
    - Part-of-speech tags:
        - 30: Select all parts of speech.
        - 31: Select parts of speech without punctuation.
        - 32: Select parts of speech without 'SPACE'.
    - Dependency-based tags:
        - 40: Select dependency labels without punctuation.
    - Morphology annotation:
        - 50: Select morphology annotations with punctuation.
        - 52: Select non-NER morphology annotations, replacing named entities with their entity type.
    - Named entities:
        - 60: Select NER types.
        - 61: Select all named entities.
    - Invalid feature IDs will be skipped, and a warning message will be displayed if `verbose` is set to 0 or 1.
    
    Example situation where removing duplicates might be needed:
    - Suppose that in a given corpus the feature 'PronType=Prs|Reflex=Yes_50' is always realised with 'się_10' (word) and the corresponding 'się_11' (lemma). These three features will be reduced to one.
    - A specific POS 5-gram has only one corresponding word 5-gram realisation.
    
    """
    #---- START Check input types
    if isinstance(docs, spacy.tokens.doc.Doc) or isinstance(docs,spacy.tokens.span.Span):
        docs = [docs] # Make a list if only one Doc is given.
    elif not isinstance(docs, list) or not all(isinstance(doc, spacy.tokens.doc.Doc) or isinstance(doc, spacy.tokens.span.Span) for doc in docs):
        if verbose<2:
            print("Warning: Invalid document format. Please provide a single spacy.tokens.doc.Doc, spacy.tokens.span.Span or a list of such objects.")
        return
    if verbose<1:
        print("Number of documents provided: {}.".format(len(docs)))
    

    features = feature_scheme['features']
    max_features = feature_scheme['max_features']
    n_grams_word = feature_scheme['n_grams_word']
    n_grams_pos = feature_scheme['n_grams_pos']
    n_grams_dep = feature_scheme['n_grams_dep']
    n_grams_morph = feature_scheme['n_grams_morph']
    min_cull_word = feature_scheme['min_cull_word']
    max_cull_word = feature_scheme['max_cull_word']
    min_cull_d2 = feature_scheme['min_cull_d2']
    max_cull_d2 = feature_scheme['max_cull_d2']
    remove_duplicates = feature_scheme['remove_duplicates']
    
    valid_features = [10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 40, 50, 52, 60, 61]
    if isinstance(features, int):
        features = [features] # Make a list if only one int is given.
    elif not isinstance(features, list) or not all(isinstance(f, int) for f in features):
        if verbose<2:
            print("Warning: Invalid features format. Please provide a single integer or a list of integers.")
        return
    
    invalid_features = [f for f in features if f not in valid_features]
    features = [f for f in features if f in valid_features]
    
    if invalid_features:
        if verbose<2:
            print("Warning: Invalid features were provided and will be dropped: {} \nSee Notes in `help(choose_features)' for available options.".format(", ".join(str(f) for f in invalid_features)))
    
    precheck = True
    global tqdm_display
    
#     TO DO: add metadata checking from preprocess_spacy.spacy_save_docs()
    
    #---- END Check input types
     

    feature_params = {
        range(0, 20): {
            'max_features': max_features,
            'ngram_range': n_grams_word,
            'min_df': min_cull_word,
            'max_df': max_cull_word
        },
        range(20, 30): {# dla dep-bigramów nie ustawiać n-gramów
            'max_features': max_features,
            'token_pattern': r'(?u)\b\w+_\w+\b',
            'min_df': min_cull_d2,
            'max_df': max_cull_d2
        },
        range(30, 40): {# dla .pos_ # max_df nie używane, bo pewnie wszystko zniknie
            'max_features': max_features,
            'ngram_range': n_grams_pos,
            'lowercase': False
        },
        range(40, 50): {#,token_pattern=r'(?u)\b\w+_\w+\b')# dla .dep_ # max_df nie używane, bo pewnie wszystko zniknie
            'max_features': max_features,
            'ngram_range': n_grams_dep,
            'token_pattern': r'(?u)\b\w[:\w]+\b'
        },
        range(50, 60): {# dla .morph
            'max_features': max_features,
            'ngram_range': n_grams_morph,
            'token_pattern': r'(?u)\b[\w=\|]+\b',
            'lowercase': False
        },
        range(60, 70): {# dla NER-ów (nie używać n-gramów, bo CountVectorizer nie widzi odstępów)
            'max_features': max_features,
            'lowercase': False
        }
    }
    
    sample_type=subsample_scheme['sample_type']
    sample_length=subsample_scheme['sample_length']
    save_samples = subsample_scheme['save_samples']
    
    # TO DO: zrobić sprawdzanie, czy w metadata jest już ten subsample_scheme, żeby wiedzieć, czy są policzone sdocs?
    tqdm_display = tqdm_propagate
    sdocs = subsample_docs(docs,metadata,sample_type=sample_type,sample_length=sample_length,precheck = precheck,save_samples = save_samples)
    tqdm_display = shared_module.tqdm_display
    
    #---- START Counting features depending on their type
    feature_names = None
    for lt in tqdm_notebook(features,display = tqdm_display,miniters=1):
        params = next((params for rng, params in feature_params.items() if lt in rng), None)
        if params is None:
            if verbose < 2:
                print("Warning: Invalid feature {}. Skipping...".format(lt))
            continue

        count_vect = CountVectorizer(**params)
        tqdm_display = tqdm_propagate
        texts = [choose_features(doc,feature = lt,verbose = verbose) for doc in sdocs]
        tqdm_display = shared_module.tqdm_display
        X0 = count_vect.fit_transform(texts)
        X0 = X0.astype('float32')

        # if lt<30 && lt>=20: # może też dla .dep_
        # Muszę tu zmienić na łącznik
        feature_names0 = [f.replace(':','_') for f in list(count_vect.get_feature_names_out())]
        # else:
        # feature_names0 = list(count_vect.get_feature_names_out())

        # UWAGA na cechy z różnych features, które nazywają się tak samo
        feature_names0[:] = [s + '_'+ str(lt) for s in feature_names0] 

        dfX0 = pd.DataFrame(X0.toarray(),columns=feature_names0)

        # Append to the feature table
        if feature_names == None: #if X == None: 
            feature_names = feature_names0
            dfX = dfX0.copy()
        else:
            feature_names.extend(feature_names0)
            dfX = dfX.join(dfX0)
            
    #---- END Counting features depending on their type
    
    if remove_duplicates == True:
        dfX = remove_duplicate_features(sdocs, dfX, features, n_grams_word)
        # feature_names = dfX.columns.to_list()
        
    return dfX

# TO DO: change this name later
dic_to_lextypes = {10: 'LOWER',    11: 'LEMMA',    12: 'LOWER',    13: 'LEMMA',    
                   20: 'D2-LOWER',    21: 'D2-LEMMA',    22: 'D2-LOWER',    23: 'D2-LEMMA',    
                   30: 'POS',    31: 'POS',    32: 'POS',
                   40: 'DEP',
                   50: 'MORPH',    52: 'MORPH',
                   60: 'ENT_TYPE',    61: 'ORTH'}

from spacy.matcher import Matcher #PhraseMatcher
from spacy.tokens import Span
from spacy.matcher import DependencyMatcher # https://spacy.io/api/dependencymatcher

# TO DO: include all possible overlaps
def remove_duplicate_features(docs, dfX, features, n_grams_word):
    if (50 in features): feature = 50
    elif (52 in features): feature = 52
    else: return
    
    if (1 not in n_grams_word): return
    
    if 10 in features or 12 in features:
        token_type = 'word'
    elif 11 in features or 13 in features:
        token_type = 'lemma'
    else: return
    
    feature_type = dic_to_lextypes[feature]
#---     This part might take a while
    matcher = Matcher(nlp.vocab)
    for feature in dfX.filter(regex=(".*_"+str(feature))).columns:
        patterns = [[{feature_type:f} for f in feature.split('_')[0].split()]]
        matcher.add(feature, patterns)
    matches = []
    for doc in docs:
        matches = matches + matcher(doc,as_spans = True)
# ---

    dfMorph = pd.DataFrame({'feature_names':[m.label_ for m in matches],
                       token_type:[m.text for m in matches]
                            if token_type == 'word' 
                            else [m.lemma_ for m in matches]})
    dfMorph = dfMorph[~dfMorph.duplicated(subset=['feature_names',token_type])]
    dfMorph = dfMorph[~dfMorph.duplicated('feature_names',keep=False)]
    dfMorph = dfMorph[~dfMorph.duplicated(token_type)]
    
    return dfX[dfX.columns.difference(dfMorph.feature_names)]