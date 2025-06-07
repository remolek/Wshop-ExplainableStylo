"""
This module defines a base class 'explain_style' providing a pipeline for preprocessing texts (currently by Spacy), extracting their features, classifying them with LGBM, explaining the classifiers with SHAP and visualising the results.

To initialize the class you need a .json file with metadata of the form, minimally:
{"experiment_name":"...",
"labels": ["filename", "class"],
"files":
	{"filename": ["path_to_file1.txt","path_to_file2.txt"],
	"class": ["file1_class", "file2_class"]}
 }

Author: Jeremi K. Ochab
Date: August 14, 2023
"""

from . import shared_module, preprocess_spacy, feature_extraction, text_classify, text_visualise
import shap
import re
import numpy as np
import pandas as pd
import json, pickle
import os

# def get_config(): 
# def set_config():

# Użytkownikowi do interakcji można udostępnić:
# - cały feature_scheme
# - subsample_scheme (może chcieć albo nie, może wybrać długości próbek tekstu)
# ? preproc_scheme (może chcieć inny język spróbować?)
# - w classify() argumenty: class_category = 'class', class_regrouping = [], group_category
# - które wykresy, wizualizacje chce generować


class explain_style:
    default_classifier_scheme = {'objective': 'binary', 'learning_rate': 0.5, 'metric': ['auc','binary',"xentropy"],
                       'nthread': 12, 'boosting':'dart', # https://neptune.ai/blog/lightgbm-parameters-guide
                        # 'max_bin':,
                        'num_leaves': 5, 'num_iterations': 100,
                        # 'min_data_in_leaf': 10,
                        # 'min_sum_hessian_in_leaf':,
                        'max_depth': 5, 'bagging_freq': 3, 'bagging_fraction':0.8,
                        # 'feature_fraction':,
                        'verbose': -1}
    default_cv_scheme = {'cv_method':'StratifiedKFold',
                         'n_repeats':10, # liczba powtórzeń
                         'n_splits':10, # liczba podziałów wewnątrz *KFold i *ShuffleSplit
                         'shuffle':True,# KFold, StratifiedKFold,StratifiedGroupKFold
                         'n_groups': 2, #LeavePGroupsOut
                         'p': 2, #LeavePOut
                         'test_fold': [1, 0, 1, 1, 0], #PredefinedSplit
                         'test_size': 0.2,#GroupShuffleSplit, ShuffleSplit
                         'train_size': 0.2,# ShuffleSplit
                         'random_state':None,#KFold,RepeatedKFold, RepeatedStratifiedKFold,ShuffleSplit
                         'val_fraction':0.25, #train_test_split
                         'scoring':{'acc':text_classify.dic_to_score['acc'],'f1':text_classify.dic_to_score['f1']}} # ułamek danych treningowych do walidacji
    default_preproc_scheme = {'model':'pl_core_news_lg'}
    default_subsample_scheme = {'sample_type':'none','sample_length':800}
    default_feature_scheme = {'features':[13,23,32,52,61],
                       'max_features':1000,
                       'n_grams_word':(1,3),
                       'n_grams_pos':(1,3),
                       'n_grams_dep':(1,3),
                       'n_grams_morph':(1,1),
                       'min_cull_word':0., # ignore terms that have a document frequency strictly lower than the given threshold
                       'max_cull_word':1., # ignore terms that have a document frequency strictly higher than the given threshold
                       'min_cull_d2':0.,
                       'max_cull_d2':1.,
                       'remove_duplicates':False}
    
    def __init__(self, metadata_json, manual = True):
        self.verbose = 0
        # Paramaters for each processing step are duplicated (self.variable_scheme and self.metadata['variable_scheme'])
        # for more flexible loading and saving config
        self.preproc_scheme = explain_style.default_preproc_scheme
        self.subsample_scheme = explain_style.default_subsample_scheme
        self.feature_scheme = explain_style.default_feature_scheme
        self.cv_scheme = explain_style.default_cv_scheme
        self.classifier_scheme = explain_style.default_classifier_scheme
        self._meta_loaded = False
        self.load_init(metadata_json)
        if manual:
            if self.verbose<1:
                print("You are in manual mode. The next steps would be to run in sequence '.texts_load()', '.texts_preprocess()', '.texts_subsample()', '.extract_features()', '.classify()', '.explain()', and then any of the '.plot_...()' methods. If you would like to change the parameters provided in metadata or set by default  ('preproc_scheme', 'subsample_scheme', 'feature_scheme', 'cv_scheme', 'classifier_scheme'), please, load them via '.load_parameter_name()' to avoid incosistent file naming and config saving.")
        else:
            self.texts_load()
            self.texts_preprocess()
            self.texts_subsample()
            self.extract_features()
            self.classify()
            self.explain()
            
    # TO DO: szczegółowy config ładowany/zapisywany do jednego jsona, czy do kilku osobnych?
    # Lepiej do jednego, żeby zapisać np. metadane typu data, nazwa eksperymentu, jakaś notatka? itp.
    def load_init(self,filename):
        filename = shared_module._check_filename(filename, '.json', self.verbose)
        if filename:
            with open(filename, 'r', encoding="utf-8") as outfile:
                self.metadata = json.load(outfile)
            if not isinstance(self.metadata, dict):
                if self.verbose<2:
                    print("Warning: Wrong format of metadata. Please provide a dictionary. Exiting...")
                return
            if 'labels' not in self.metadata or 'files' not in self.metadata:
                if self.verbose<2:
                    print("Warning: Wrong format of metadata. Minimally, please provide {'experiment_name':name,'labels': array_of_labels, 'files': {'filename': array_of_filenames, 'class': array_of_classes}}. Exiting...")
                return
            if 'filename' not in self.metadata['files'] or 'class' not in self.metadata['files']:
                if self.verbose<2:
                    print("Warning: Wrong format of metadata. Minimally, please provide {'experiment_name':name,'labels': array_of_labels, 'files': {'filename': array_of_filenames, 'class': array_of_classes}}. Exiting...")
                return
            if not all(isinstance(f, str) for f in self.metadata['files']['filename']):
                if self.verbose<2:
                    print("Warning: Invalid filename format. Please provide an array of strings.")
                return
            if not all(isinstance(f, str) for f in self.metadata['files']['class']):
                if self.verbose<2:
                    print("Warning: Invalid class name format. Please provide an array of strings.")
                return
            
            if self.verbose<1:
                    print("Initialisation metadata loaded from {}.".format(filename))
                    print("Available text labels {}.".format(self.metadata['labels']))
                    print("Predefined text classes {}.".format(np.unique(self.metadata['files']['class']).tolist()))
            if 'preproc_scheme' in self.metadata.keys() and self.metadata['preproc_scheme']:
                self.preproc_scheme = self.metadata['preproc_scheme']
                if self.verbose<1:
                    print("Text preprocessing parameters loaded. Later, you can use load_params_preproc(filename) to set these parameters.")
            else:
                self.metadata['preproc_scheme'] = self.preproc_scheme
                if self.verbose<1 and self._meta_loaded: # If load_init() used later, the other attributes might already be set differently.
                    print("Using default text preprocessing parameters.")
            if 'subsample_scheme' in self.metadata.keys() and self.metadata['subsample_scheme']:
                self.subsample_scheme = self.metadata['subsample_scheme']
                if self.verbose<1:
                    print("Text subsampling parameters loaded. Later, you can use load_params_subsample(filename) to set these parameters.")
            else:
                self.metadata['subsample_scheme'] = self.subsample_scheme
                if self.verbose<1 and self._meta_loaded:
                    print("Using default text subsampling parameters.")
            if 'feature_scheme' in self.metadata.keys() and self.metadata['feature_scheme']:
                self.feature_scheme = self.metadata['feature_scheme']
                if self.verbose<1:
                    print("Feature extraction parameters loaded. Later, you can use load_params_feature(filename) to set these parameters.")
            else:
                self.metadata['feature_scheme'] = self.feature_scheme
                if self.verbose<1 and self._meta_loaded:
                    print("Using default feature extraction parameters.")
            if 'cv_scheme' in self.metadata.keys() and self.metadata['cv_scheme']:
                self.cv_scheme = self.metadata['cv_scheme']
                if self.verbose<1:
                    print("Cross-validation parameters loaded. Later, you can use load_params_cv(filename) to set these parameters.")
            else:
                self.metadata['cv_scheme'] = self.cv_scheme
                if self.verbose<1 and self._meta_loaded:
                    print("Using default cross-validation parameters.")
            if 'classifier_scheme' in self.metadata.keys() and self.metadata['classifier_scheme']:
                self.classifier_scheme = self.metadata['classifier_scheme']
                if self.verbose<1:
                    print("Classifier parameters loaded.")
            else:
                self.metadata['classifier_scheme'] = self.classifier_scheme
                if self.verbose<1 and self._meta_loaded:
                    print("Using default classifier parameters. Later, you can use load_params_classifier(filename) to set these parameters.")
            self._meta_loaded = True
        return 
    
    #----- Helper functions for loading and saving all config files
    def _check_attribute(self,attribute):
        assert isinstance(attribute, str), f"Warning: Invalid attribute '{attribute}' format. Please provide a string."
        assert hasattr(self, attribute), f"The '{attribute}' attribute has not been defined yet."
        return getattr(self, attribute)

    def _default_filename(self, filename, attribute=''):
        if filename is None:
            self._check_attribute('metadata')
            self._check_attribute('subsample_scheme')
            folder_name = 'explain_'+self.metadata['experiment_name']
            subfolder = 'subsamples_{}_{}'.format(self.subsample_scheme['sample_type'],self.subsample_scheme['sample_length'])
            folder_name = os.path.join(folder_name,subfolder)
            shared_module._check_folder(folder_name)
            filename = attribute
            filename = os.path.join(folder_name, filename)
        return filename

    def _check_filename_postfix(self,postfix = None):
        if postfix is None:
            if hasattr(self, 'filename_postfix'):
                postfix = '_'+self.filename_postfix
            else:
                postfix =''
        return postfix
        
    def _load_params(self, attribute, message, filename = None):
        filename = self._default_filename(filename,'config_'+attribute)
        
        if attribute == 'cv_scheme':
            filename = shared_module._check_filename(filename, '.pkl',self.verbose)
            if filename:
                with open(filename, 'rb') as outfile:
                    setattr(self, attribute, pickle.load(outfile))
                    self.metadata[attribute] = getattr(self, attribute)
                if self.verbose < 1:
                    print(f"'{message}' loaded from '{filename}'.")
        else:
            filename = shared_module._check_filename(filename, '.json',self.verbose)
            if filename:
                with open(filename, 'r', encoding="utf-8") as outfile:
                    setattr(self, attribute, json.load(outfile))
                    self.metadata[attribute] = getattr(self, attribute)
                if self.verbose < 1:
                    print(f"'{message}' loaded from '{filename}'.")
    
    def load_params_cv(self,filename = None):
        self._load_params('cv_scheme', 'Cross-validation parameters', filename)
    def load_params_classifier(self,filename = None):
        self._load_params('classifier_scheme', 'Classifier parameters', filename)
    def load_params_preproc(self,filename = None):
        self._load_params('preproc_scheme', 'Preprocessing parameters', filename)
    def load_params_subsample(self,filename = None):
        self._load_params('subsample_scheme', 'Text subsampling parameters', filename)
        self._load_params('metadata_subsampled', 'Text subsampling metadata', filename)
    def load_params_feature(self,filename = None):
        self._load_params('feature_scheme', 'Feature extraction parameters', filename)

    # TO DO: ustalić prawa użytkownika (samodzielne ustawianie overwrite, filename)
    def _save_params(self, attribute, message, filename = None):
        self._check_attribute(attribute)
        filename = self._default_filename(filename,'config_'+attribute)
        if attribute == 'cv_scheme':
            filename = shared_module._prepare_filename(filename, '.pkl', overwrite = False)
            with open(filename, 'wb') as outfile:
                pickle.dump(getattr(self, attribute), outfile)
        else:
            filename = shared_module._prepare_filename(filename, '.json', overwrite = False)
            with open(filename, 'w', encoding="utf-8") as outfile:
                json.dump(getattr(self, attribute), outfile)
        if self.verbose < 1:
            print(f"{message} saved as '{filename}'. It can be reloaded with '.load_params_{attribute.split('_')[0]}(filename)'.")

    # TO DO: zapisyanie scoring (funkcje pythona) jest niemożliwe do jsona
    def save_params_cv(self,filename = None):
        self._save_params('cv_scheme', 'Cross-validation parameters', filename)
    def save_params_classifier(self,filename = None):
        self._save_params('classifier_scheme', 'Classifier parameters', filename)
    def save_params_preproc(self,filename = None):
        self._save_params('preproc_scheme', 'Preprocessing parameters', filename)
    def save_params_subsample(self,filename = None):
        self._save_params('subsample_scheme', 'Text subsampling parameters', filename)
        self._save_params('metadata_subsampled', 'Text subsampling metadata', filename)
    def save_params_feature(self,filename = None):
        self._save_params('feature_scheme', 'Feature extraction parameters', filename)
    def save_config(self,filename = None):
        filename = self._default_filename(filename,'config')
        filename = shared_module._prepare_filename(filename, '.json', overwrite = False)
        with open(filename, 'w', encoding="utf-8") as outfile:
            json.dump(self.metadata, outfile)
        if self.verbose < 1:
            print(f"All experiment metadata and parameters saved as '{filename}'. They can be loaded with '.load_init(filename)'.")
    #----- END
        
    #----- Saving&loading important class instance attributes 
    # TO DO: niejednoznaczne ładowanie dokumentów po subsamplingu. Jeśli się załaduje, to do .docs, czy .docs_subsampled? Wtedy czy .subsample_scheme powinno być zmienione? Potem jeśli docs_subsampled istnieją, to one są klasyfikowane.
    def docs_load(self,filename = None):
        self._check_attribute('metadata')
        filename = self._default_filename(filename,attribute = 'docs')
        self._check_attribute('preproc_scheme')
        self.docs, model_set = preprocess_spacy.spacy_load_docs(filename,self.preproc_scheme['model'],self.metadata['labels'],
                                    verbose = self.verbose, precheck = True)
        # The model preproc_scheme['model'] loads only during preprocess_spacy.spacy_load_docs or preprocess_spacy.spacy_load_docs or preprocess_spacy.spacy_preproc
        # If the model was not available, another one will be chosen. The model's name is updated.
        if model_set != self.preproc_scheme['model']:
            if self.verbose < 2:
                print(f"The model has changed from {self.preproc_scheme['model']} to {model_set}.")
            self.preproc_scheme['model'] = model_set
    def docs_save(self,docs = None, filename = None, precheck = False, overwrite = False):
        if docs is None: # Left in case you want to save subsampled_docs
            docs = self._check_attribute('docs')
        filename = self._default_filename(filename,attribute='docs')
        preprocess_spacy.spacy_save_docs(docs, filename, overwrite = overwrite, verbose = self.verbose, precheck = precheck) 
        if self.verbose < 1:
            print(f"They can be reloaded with '.docs_load(filename)'.")

    def features_load(self,filename = None):
        filename = self._default_filename(filename,attribute='features')
        filename = shared_module._check_filename(filename, '.csv',self.verbose)
        if hasattr(self, 'feature_dataframe'):
            if self.verbose < 1:
                print(f"Current features will be overwritten.")            
        self.feature_dataframe = pd.read_csv(filename)
        self.feature_names = self.feature_dataframe.columns.to_list()
        self.feature_names_waterfall = [f.replace('=',':') for f in self.feature_names]
        self.feature_names_waterfall = [re.sub('_[\d]+','',f) for f in self.feature_names_waterfall]
        if self.verbose < 1:
            print(f"Features loaded from '{filename}' to '.feature_dataframe' attribute.")
    def features_save(self,filename = None,overwrite = False):
        feature_dataframe = self._check_attribute('feature_dataframe')
        filename = self._default_filename(filename,attribute='features')
        filename = shared_module._prepare_filename(filename, '.csv', overwrite)
        feature_dataframe.to_csv(filename,index = False)
        if self.verbose < 1:
            print(f"'.feature_dataframe' saved as '{filename}'. It can be reloaded with '.features_load(filename)'.")
    
    def scores_load(self,filename = None,postfix = None):
        postfix = self._check_filename_postfix(postfix)
        self.filename_postfix = postfix
        filename = self._default_filename(filename,attribute='scores'+postfix)
        self.scores = text_classify.load_scores(filename,self.verbose)
    def scores_save(self,filename = None,postfix = None, overwrite = False):
        postfix = self._check_filename_postfix(postfix)
        scores = self._check_attribute('scores')
        filename = self._default_filename(filename,attribute='scores'+postfix)
        scores.save(filename,overwrite)
        # self.scores.save(filename,overwrite)
    
    def classifiers_load(self,filename = None,postfix = None):
        postfix = self._check_filename_postfix(postfix)
        filename = self._default_filename(filename,attribute='classifiers'+postfix)
        self.classifiers = text_classify.load_models(filename,self.verbose)
        # TO DO: brzydkie zarządzanie kiedy jest lub nie ma subsamplingu tekstów
        class_category = str.split(postfix,'_')[1]
        class_category = str.split(class_category,'-')[1]
        #group_category = str.split(postfix,'_')[2]
        #group_category = str.split(group_category,'-')[1]
        #class_regrouping = [] # Using default for now; TO DO store/restore class_regrouping
        #if self.docs_subsampled:
        #    sample_classes = self.metadata_subsampled['files']
        #else:
        #    sample_classes = self.metadata['files']
        if class_category not in self.metadata['labels']:
            if self.verbose < 2:
                print("Warning: Class name '{}' not available in provided labels: {}.".format(class_category,self.metadata['labels']))
            self.classifiers = None
            return
        # Klasy: to są kategorie, które chcemy umieć rozpoznawać    
        #y, yD, classD = text_classify.target_classes(sample_classes[class_category],class_regrouping=class_regrouping)
        #self.classes_used = yD
        self.classes_used = self.classifiers[0]['classes_used']
        
    def classifiers_save(self,filename = None,postfix = None,overwrite = False):
        postfix = self._check_filename_postfix(postfix)
        classifiers = self._check_attribute('classifiers')
        filename = self._default_filename(filename,attribute='classifiers'+postfix)
        text_classify.save_models(classifiers,filename,overwrite)
        
    def explanations_load(self,filename = None,postfix = None):
        postfix = self._check_filename_postfix(postfix)
        filename = self._default_filename(filename,attribute='explanations'+postfix)
        self.explanations = text_classify.load_exps(filename,self.verbose)
    def explanations_save(self,filename = None,postfix = None,overwrite = False):
        postfix = self._check_filename_postfix(postfix)
        explanations = self._check_attribute('explanations')
        filename = self._default_filename(filename,attribute='explanations'+postfix)
        explanations.save(filename,overwrite)

    #----- Workhorse methods
    def texts_load(self, from_variable = None, metadata = None):
        if metadata is None:
                metadata = self._check_attribute('metadata')
        self.texts = preprocess_spacy.load_texts(metadata,from_variable = from_variable, verbose = self.verbose, precheck = self._meta_loaded)
    def texts_preprocess(self,texts = None,model = None,filename = None,labels = None,save_to_file = True):
        if texts is None:
            texts = self.texts
        if model is None:
            model = self.preproc_scheme['model']
        filename = self._default_filename(filename,attribute='docs')
        if labels is None:
            labels = self.metadata['labels']
        self.docs, model_set = preprocess_spacy.spacy_preproc(texts,model,labels = labels, verbose = self.verbose)
        # The model preproc_scheme['model'] loads only during preprocess_spacy.spacy_load_docs or preprocess_spacy.spacy_load_docs or preprocess_spacy.spacy_preproc
        # If the model was not available, another one will be chosen. The model's name is updated.
        if model_set != model:
            if self.verbose < 2:
                print(f"The model has changed from {self.preproc_scheme['model']} to {model_set}.")
            self.preproc_scheme['model'] = model_set
        if save_to_file == True:
            self.docs_save(self.docs,filename, precheck = True)
            self.save_params_preproc()

    def texts_subsample(self, subsample_scheme = None, save_to_file = False):
        if subsample_scheme is None:
            subsample_scheme = self.subsample_scheme
        if subsample_scheme['sample_type'] == 'none':
            save_to_file = False
            self.docs_subsampled = self._check_attribute('docs')
            self.metadata_subsampled = self._check_attribute('metadata')
            if self.verbose < 1:
                print("No text subsampling with current setting: 'sample_type = 'none'.")
        else:
            self.subsample_scheme = subsample_scheme
            if self.verbose < 1:
                print("'subsample_scheme' changed to '{}'.".format(self.subsample_scheme))
            self.docs_subsampled, self.metadata_subsampled = feature_extraction.subsample_docs(self.docs,self.metadata,subsample_scheme=subsample_scheme,
                                                            verbose=self.verbose,precheck = True)
        if save_to_file:
    # TO DO: problem z zapisywaniem subsamplowanych doców??
            self.docs_save(docs = self.docs_subsampled, precheck = True)
            self.save_params_subsample()
            self.save_params_preproc()

    def extract_features(self, feature_scheme = None, feature_list = None, save_to_file = False):
        if feature_list:
            # To use this, first put your list of feature names into self.feature_names
            self.feature_names = feature_list
            self.feature_dataframe = feature_extraction.count_features_list(self.docs_subsampled if self.docs_subsampled else self.docs,
                                                                       feature_list=self.feature_names,verbose = self.verbose, tqdm_propagate = False)            
            self.feature_names_waterfall = [f.replace('=',':') for f in self.feature_names]
            self.feature_names_waterfall = [re.sub('_[\d]+','',f) for f in self.feature_names_waterfall]
            if save_to_file:
                if self.verbose < 2:
                    print("Warning: Using feature list, so no 'feature_scheme' will be saved")
                self.features_save()
        else:
            if feature_scheme is None:
                feature_scheme = self.feature_scheme
            self.feature_dataframe = feature_extraction.count_features(self.docs_subsampled if self.docs_subsampled else self.docs,
                                                                       feature_scheme=feature_scheme,verbose = self.verbose, tqdm_propagate = False)
            self.feature_names = self.feature_dataframe.columns.to_list()
            self.feature_names_waterfall = [f.replace('=',':') for f in self.feature_names]
            self.feature_names_waterfall = [re.sub('_[\d]+','',f) for f in self.feature_names_waterfall]
            if save_to_file:
                self.save_params_feature()
                self.features_save()

    # def extract_features(self, feature_scheme = None, save_to_file = False):
    #     if feature_scheme is None:
    #         feature_scheme = self.feature_scheme
    #     self.feature_dataframe = feature_extraction.count_features(self.docs_subsampled if self.docs_subsampled else self.docs,
    #                                                                feature_scheme=feature_scheme,verbose = self.verbose, tqdm_propagate = False)
    #     self.feature_names = self.feature_dataframe.columns.to_list()
    #     self.feature_names_waterfall = [f.replace('=',':') for f in self.feature_names]
    #     self.feature_names_waterfall = [re.sub('_[\d]+','',f) for f in self.feature_names_waterfall]
    #     if save_to_file:
    #         self.save_params_feature()
    #         self.features_save()

# Invoke some default, to have self.classes_used even before you run classify()
        # if self.docs_subsampled:
        #     sample_classes = self.metadata_subsampled['files']
        # else:
        #     sample_classes = self.metadata['files']
        # if class_category not in self.metadata['labels']:
        #     if self.verbose < 2:
        #         print("Warning: Class name '{}' not available in provided labels: {}.".format(class_category,self.metadata['labels']))
        #     self.classifiers = None
        #     return
        # # Klasy: to są kategorie, które chcemy umieć rozpoznawać    
        # y, yD, classD = text_classify.target_classes(sample_classes[class_category],class_regrouping=class_regrouping)
        # self.classes_used = yD
    
    def classify(self, class_category = 'class', class_regrouping = [], group_category=None, save_to_file = False):
    # TO DO: brzydkie zarządzanie kiedy jest lub nie ma subsamplingu tekstów
        if self.docs_subsampled:
            sample_classes = self.metadata_subsampled['files']
        else:
            sample_classes = self.metadata['files']
        if class_category not in self.metadata['labels']:
            if self.verbose < 2:
                print("Warning: Class name '{}' not available in provided labels: {}.".format(class_category,self.metadata['labels']))
            self.classifiers = None
            return
        # Klasy: to są kategorie, które chcemy umieć rozpoznawać    
        y, yD, classD = text_classify.target_classes(sample_classes[class_category],class_regrouping=class_regrouping)
        self.classes_used = yD
            
        if group_category:
            if group_category not in self.metadata['labels']:
                if self.verbose < 2:
                    print("Warning: Group name '{}' not available in provided labels: {}.".format(group_category,self.metadata['labels'])) 
                self.classifiers = None
                return
            # Grupy: to są grupy danych wewnątrz klas, których nie chcemy (nawet implycytnie) się uczyć rozpoznawać
            # Przykład:
                    # klasyfikacja tekstów na naukowe i literackie; grupami mogą być autorzy;
                    # lepiej, żeby klasyfikator umiał nawet wyekstrachować styl z jednego autora, a potem rozpoznać ten styl w jeszcze nieznanym autorze
            groups, groupD, group_namesD = text_classify.target_classes(sample_classes[group_category],no_classes=0)
            if not self.cv_scheme['cv_method'] in ['GroupKFold','StratifiedGroupKFold','GroupShuffleSplit']:
                self.cv_scheme['cv_method'] = 'StratifiedGroupKFold'
            # if self.verbose < 1:
            #     print("Proceeding groups: {}.".format(list(group_namesD.keys())))
        else:
            groups = None
        if self.verbose < 1:
            print("Proceeding with {} cross-validation.".format(self.cv_scheme['cv_method']))

        self.classifiers = text_classify.cv_classifier(self.feature_dataframe,class_labels=y, group_labels=groups,
                                                    classifier='LGBM',
                                                     cv_scheme=self.cv_scheme,classifier_scheme = self.classifier_scheme,
                                                     verbose=self.verbose)
        for c in range(len(self.classifiers)):
            self.classifiers[c]['classes_used'] = self.classes_used
        self.scores = text_classify.collect_scores(self.classifiers,self.feature_dataframe,
                                                   scoring = self.cv_scheme['scoring'],baseline_strategy='most_frequent')
        self.scores.print_scores()
        if save_to_file:
            self.filename_postfix = f"class-{class_category}{'_group-'+group_category if group_category else ''}"
            self.scores_save()
            self.classifiers_save()
            self.save_params_classifier()
            self.save_params_cv()
# Make classify, classify_cv, classifiers_cv and classifiers and make classify_cv also compute the final classify
    # def classify(self, class_category='class', class_regrouping=[], group_category=None, save_to_file=False):
    #     # Determine the appropriate sample classes based on subsampling
    #     sample_classes = self.metadata_subsampled['files'] if self.docs_subsampled else self.metadata['files']
    
    #     # Check if the specified class category exists
    #     if class_category not in self.metadata['labels']:
    #         if self.verbose < 2:
    #             print(f"Warning: Class name '{class_category}' not available in provided labels: {self.metadata['labels']}.")
    #         self.classifiers = None
    #         return
    
    #     # Generate target classes
    #     y, yD, classD = text_classify.target_classes(sample_classes[class_category], class_regrouping=class_regrouping)
    #     self.classes_used = yD
    
    #     # Train the model on the full dataset
    #     model = text_classify.train_full_lgbm(
    #         feature_dataframe=self.feature_dataframe,
    #         class_labels=y,
    #         classifier_scheme=self.classifier_scheme,
    #         verbose=self.verbose
    #     )
    
    #     # Store the trained model
    #     self.classifiers = [{'trained_model': model, 'classes_used': self.classes_used}]
    
    #     # Optionally, compute and display scores
    #     if hasattr(text_classify, 'collect_scores') and self.cv_scheme.get('scoring'):
    #         self.scores = text_classify.collect_scores(
    #             self.classifiers,
    #             self.feature_dataframe,
    #             scoring=self.cv_scheme['scoring'],
    #             baseline_strategy='most_frequent'
    #         )
    #         self.scores.print_scores()
    
    #     # Save results if specified
    #     if save_to_file:
    #         self.filename_postfix = f"class-{class_category}{'_group-' + group_category if group_category else ''}"
    #         self.scores_save()
    #         self.classifiers_save()
    #         self.save_params_classifier()
    #         self.save_params_cv()

    
    def explain(self, models = None, minimize_data_use=True, save_to_file = False):
        self._check_attribute('feature_dataframe')
        self._check_attribute('classes_used')
        if models is None:
            models = self._check_attribute('classifiers')
        self.explanations = text_classify.collect_shap(models,self.feature_dataframe,
                                          feature_names = self.feature_names, output_names=[i[0] for i in self.classes_used.values() if len(i)],minimize_data_use=minimize_data_use)
        self.explanations.compute_average()
        if save_to_file:
            self.explanations_save()        

    #----- END

   
    # Visualisation and analysis methods
    
    def _default_title(self,title):
        if not title:
                if hasattr(self,'docs_subsampled'):
                    title ='Kropka = jeden tekst (wynik {}x walidacji)'.format(self.cv_scheme['n_splits'])
                else:
                    title ='Kropka = próbka tekstu ok. {} słów (wynik {}x walidacji)'.format(subsample_scheme['sample_length'],self.cv_scheme['n_splits'])
        return title

    def _check_subsampled(self):
        if hasattr(self,'docs_subsampled'):
            sample_classes = self.metadata_subsampled['files']
        else:
            sample_classes = self._check_attribute('metadata')['files']
        return sample_classes

    
    def plot_summary(self,
                     group_by = None,
                     show = True, max_display = 20,
                     filename = 'SHAP_wszystkie',
                     title ='',
                    file_ext = '.png'):

        feature_dataframe = self._check_attribute('feature_dataframe')
        input = self._check_attribute('explanations')

        postfix = self._check_filename_postfix()
        filename = self._default_filename(None,attribute=filename+postfix)
        filename = shared_module._prepare_filename(filename,file_ext,overwrite = False)

        if group_by:
            # Uśrednij po wszystkich podgrupach danej grupy (miało sens dla styl naukowy/literacki, podgrupy autorskie)
            # TO DO: sprawdzić na powyższych danych
            sample_classes = self._check_subsampled()
            if group_by not in self.metadata['labels']:
                if self.verbose < 2:
                    print("Warning: Class name '{}' not available in provided labels: {}.".format(group_by,self.metadata['labels']))
                return
            groups, groupsD, group_namesD = text_classify.target_classes(sample_classes[group_by],no_classes=0)
            input.compute_group(groups)
            text_visualise.plot_summary(input.shap_group, feature_dataframe, show = show, max_display = max_display,
                                    filename = filename, title = self._default_title(title))
        else:
            # Bez uśredniania
            text_visualise.plot_summary(input.shap_average, feature_dataframe, show = show, max_display = max_display,
                                    filename = filename, title = self._default_title(title))

    def plot_group_summary(self,
                           group_by, max_display=20, show = True, color_legend=True,
                           filename = 'SHAP_group_all',
                           title ='',
                           file_ext = '.png'):
        sample_classes = self._check_subsampled()
        if group_by not in self.metadata['labels']:
            if self.verbose < 2:
                print("Warning: Class name '{}' not available in provided labels: {}.".format(group_by,self.metadata['labels']))
            return
        input = self._check_attribute('explanations')
        
        postfix = self._check_filename_postfix()
        filename = self._default_filename(None,attribute=filename+postfix)
        filename = shared_module._prepare_filename(filename,file_ext,overwrite = False)
        text_visualise.plot_group_summary(input.shap_average,self.feature_names_waterfall, sample_classes,group_by,
                                          max_display=max_display,show = show, color_legend=color_legend,
                                          filename = filename, title = self._default_title(title))

    # TO DO: czy 'author' ma się pojawić
    # TO DO: tytuł ma podawać też numer próbki
    def plot_text(self,
                  index, show = True, max_display = 20,
                  filename = None,
                  title = None,
                  file_ext = '.png'):
        sample_classes = self._check_subsampled()
        #if group_by not in self.metadata['labels']:
        #    if self.verbose < 2:
        #        print("Warning: Class name '{}' not available in provided labels: {}.".format(group_by,self.metadata['labels']))
        #    return
        # TO DO: ewentualnie do zmiany domyślna nazwa pliku
        if filename is None:
            filename = 'SHAP_{}_sample{}'.format(sample_classes['class'][index],index)
        if title is None:
            title = sample_classes['class'][index]
        filename = shared_module._prepare_filename(filename,file_ext,overwrite = False)
        text_visualise.plot_text(self.explanations.compute_text(index),
                                 show = show, max_display = max_display, filename = filename, title = title)

    # TO DO: Czy włączać jako wariant poprzedniej metody?
    def plot_group_texts(self,
                         group_by, group = None, show = True, max_display = 20,
                         filename = 'SHAP_group_',
                         title = '',
                         file_ext = '.png'):
        '''group = None - plot all groups (e.g., authors)
        group = 'GroupName', group = 0 - plot a selected group
        '''
        
        sample_classes = self._check_subsampled()
        if group_by not in self.metadata['labels']:
            if self.verbose < 2:
                print("Warning: Class name '{}' not available in provided labels: {}.".format(group_by,self.metadata['labels']))
            return
            
        groups, groupsD, group_namesD = text_classify.target_classes(sample_classes[group_by],no_classes=0)
        self.explanations.compute_group(groups) # provides previous computation or computes new (and overwrites) if groups are different from before
        
        #- Choose groups by name or int index
        if group is not None:
            if isinstance(group,str):
                index = group_namesD[group]
            else:
                index = group
            iterator = [index]
        else:
            iterator = groupsD.keys()
        #- 

    # TO DO: czy ten fragment powinien być dostępny całej instancji?
        metadata_df = pd.DataFrame(sample_classes)
        scores = self._check_attribute('scores')
        metadata_df['errors'] = scores.df_preds['errors']
    #-
        feature_dataframe = self._check_attribute('feature_dataframe')
        for i in iterator:
            if title:
                title = "{}\n{}".format(title,groupsD[i][0])
            else:
                title_errors = metadata_df[metadata_df[group_by] == groupsD[i][0]]['errors'].mean()
                title = "Uśredniony wkład cech dla {}=={},\n błędów klasyfikacji: {:.2f}/{}".format(group_by,groupsD[i][0],title_errors,self.cv_scheme['n_repeats'])
            temp = self.explanations.compute_text_group(i,feature_dataframe)
            temp.feature_names = self.feature_names_waterfall
            # shap.plots.waterfall(temp, show = False, max_display = liczba_cech)
            postfix = self._check_filename_postfix()
            newfilename = self._default_filename(None,attribute=filename+groupsD[i][0]+postfix)
            newfilename = shared_module._prepare_filename(newfilename,file_ext,overwrite = False)
            text_visualise.plot_text(temp, max_display = max_display, filename = newfilename, title = title)      

#----- END

