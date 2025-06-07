"""
This module provides wrapper functions for presenting and plotting classification results and their explanations.
The modules you might need to run first are 'preprocess_spacy.py' module, to first load and preprocess the texts,
'feature_extraction.py' module, to extract interesting textual features, and 'text_classify.py', to classify texts using these features.

It includes functions to ...

Author: Jeremi K. Ochab
Date: June 27, 2023
"""

import numpy as np

# For SHAP plotting
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import shap
from shap.plots._labels import labels
from shap.plots import colors

# For matcher
from . import feature_extraction, shared_module

import re
from spacy.matcher import Matcher #PhraseMatcher
from spacy.tokens import Span
from spacy.matcher import DependencyMatcher # https://spacy.io/api/dependencymatcher

# For text highlighting
import spacy
from spacy import displacy
# https://spacy.io/api/top-level#displacy.render
# https://spacy.io/usage/visualizers
import matplotlib as mpl
# https://stackoverflow.com/questions/68765137/displacy-custom-colors-for-custom-entities-using-displacy
from matplotlib.colors import to_hex
from matplotlib import colormaps

def plot_summary(explanations, feature_dataframe, show = True, max_display = 20, filename = '',title =''):
    shap.summary_plot(explanations, feature_dataframe, show = False, max_display = max_display)
    if title:
        plt.title(title)
    if filename:
        plt.savefig(filename, bbox_inches="tight")
    if show:
        plt.show()

def plot_text(explanations, show = True, max_display = 20, filename = '',title =''):
    shap.plots.waterfall(explanations,show = False, max_display = max_display)
    if title:
        plt.title(title)
    if filename:
        plt.savefig(filename, bbox_inches="tight")
    if show:
        plt.show()             

def plot_group_summary(shap_values,feature_names, metadata,group_by,
                   max_display=10,color=None,
             axis_color="#333333", show=True,
             color_legend=True, color_legend_label=None,
                  filename='',title = ''):
    """Create a SHAP beeswarm plot, colored by feature values when they are provided.
    Parameters
    ----------
    shap_values : Explanation
        This is an :class:`.Explanation` object containing a matrix of SHAP values
        (# samples x # features).
    max_display : int
        How many top features to include in the plot (default is 10, or 7 for
        interaction plots).
    show : bool
        Whether ``matplotlib.pyplot.show()`` is called before returning.
        Setting this to ``False`` allows the plot
        to be customized further after it has been created.
    color_legend : bool
        Whether to draw the color legend.
    Notes
    --------
    Abdriged and modified from <https://github.com/slundberg/shap/blob/master/shap/plots/_beeswarm.py>.
    """

    features = np.repeat([metadata[group_by]],len(feature_names),axis=0).astype('object').T
    
    row_height = 0.4
    axis_color="#333333"
    num_features = shap_values.shape[1]
    feature_order = np.argsort(np.sum(np.abs(shap_values), axis=0))
    feature_order = feature_order[-min(max_display, len(feature_order)):]
    # shaps = shap_values[:, i]
    for pos, i in enumerate(feature_order):
        plt.axhline(y=pos, color="#cccccc", lw=0.5, dashes=(1, 5), zorder=-1)
        shaps = shap_values[:, i]
        values = None if features is None else features[:, i]
        inds = np.arange(len(shaps))
        np.random.shuffle(inds)
        if values is not None:
            values = values[inds]
        shaps = shaps[inds]

        num_values = len(np.unique(values))
        range_values = np.unique(values)
        dict_values = values.copy()
        for j in range(num_values):
            dict_values[values == range_values[j]] = j
                
        N = len(shaps)
        nbins = 100
        quant = np.round(nbins * (shaps - np.min(shaps)) / (np.max(shaps) - np.min(shaps) + 1e-8))
        inds = np.argsort(quant + np.random.randn(N) * 1e-6)
        layer = 0
        last_bin = -1
        ys = np.zeros(N)
        for ind in inds:
            if quant[ind] != last_bin:
                layer = 0
            ys[ind] = np.ceil(layer / 2) * ((layer % 2) * 2 - 1)
            layer += 1
            last_bin = quant[ind]
        ys *= 0.9 * (row_height / np.max(ys + 1))
        
        cvals = dict_values
        if num_values <= 10:
            cmap = 'Set1'#'tab10'
        elif num_values <= 39:
            # Handpicked color map
            mycolors = np.vstack((plt.cm.tab10.colors, plt.cm.Accent.colors, plt.cm.Set1.colors, plt.cm.Set3.colors))
            cmap = mcolors.ListedColormap(mycolors)
        else:
            cvals = cvals/np.max(cvals)
            cmap = 'gist_rainbow'

        # plot the non-nan values colored by the trimmed feature value
        scatter = plt.scatter(shaps, pos + ys,
                   cmap=cmap, s=16, c=cvals, alpha=1, linewidth=0,
                   zorder=3, rasterized=len(shaps) > 500)

    plt.gca().xaxis.set_ticks_position('bottom')
    plt.gca().yaxis.set_ticks_position('none')
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().tick_params(color=axis_color, labelcolor=axis_color)
    plt.yticks(range(len(feature_order)), [feature_names[i] for i in feature_order], fontsize=13)
    plt.gca().tick_params('y', length=20, width=0.5, which='major')
    plt.gca().tick_params('x', labelsize=11)
    plt.ylim(-1, len(feature_order))
    plt.xlabel(labels['VALUE'], fontsize=13)
    if not color_legend_label:
        color_legend_label = np.unique(metadata[group_by])
    plt.legend(scatter.legend_elements()[0],color_legend_label,loc="lower left", title=group_by)
    plt.tight_layout()
    if title:
        plt.title(title)
    if filename:
        plt.savefig(filename, bbox_inches="tight")
    if show:
        plt.show()
        

import spacy
from spacy import displacy
# https://spacy.io/api/top-level#displacy.render
# https://spacy.io/usage/visualizers
from matplotlib import pyplot as plt
import matplotlib as mpl
# https://stackoverflow.com/questions/68765137/displacy-custom-colors-for-custom-entities-using-displacy
from matplotlib.colors import to_hex
from matplotlib import colormaps

def match_lextype(doc,feature_values,lextype = 10,option='feature_type'):
    feature_type = feature_extraction.dic_to_lextypes[lextype]
    if option == 'feature_type':
        match_name = str(lextype)+'-'+feature_type
        span_name = str(lextype)
        sh_condition = feature_values[feature_values['lextypes'] == lextype]['shap'].unique()
    elif option == 'feature':
        match_name = feature_values['feature_names'].split('_')[0]
        span_name = match_name
        sh_condition = [feature_values['shap']]
    else:
        print('"option" must be either "feature_type" or "feature".')
        return
        
    if feature_type in ['LEMMA','POS','MORPH','LOWER','ORTH']:
        #     N-grams are split on whitespace
        matcher = Matcher(shared_module.nlp.vocab)
        
        for sh in sh_condition:
            if option == 'feature_type':
                patterns = [[{feature_type:f} for f in feature.split('_')[0].split()] for feature in feature_values[(feature_values['feature_types'] == feature_type) & (feature_values['shap']== sh)]['feature_names']]
            # patterns = [[{feature_type:f} for f in feature.split()] for feature in feature_values[feature_values['shap']== sh]['feature_names']]
            elif option == 'feature':
                patterns = [[{feature_type:f} for f in feature_values['feature_names'].split('_')[0].split()]]
            
            if feature_type == 'POS':
                matcher.add(match_name+' '+str(sh), patterns)
            else:
                matcher.add(match_name+' '+str(sh), patterns)

        matches = matcher(doc)
        doc.spans[span_name] = list()
        for match_id, start, end in matches:               
            span = Span(doc, start, end, label=match_id)
            if (lextype == 61) & (len(span.ents)==0):
                None
            else:
                doc.spans[span_name].append(span)
            
    elif feature_type in ['D2-LOWER','D2-LEMMA']:
        matcher = DependencyMatcher(shared_module.nlp.vocab)
        for sh in sh_condition:
            patterns = [[
              # anchor token: parent
              {
                "RIGHT_ID": feature.split('_')[0],
                "RIGHT_ATTRS": {feature_type.split('-')[1]: feature.split('_')[0]}
              },
              # child
              {
                "LEFT_ID": feature.split('_')[0],
                "REL_OP": op,
                "RIGHT_ID": feature.split('_')[1],
                "RIGHT_ATTRS": {feature_type.split('-')[1]: feature.split('_')[1]}
              }
            ]
            # szukanie prawych >++ i lewych >-- dzieci
            # https://spacy.io/usage/rule-based-matching#dependencymatcher-operators          
            for op in ['>++','>--'] for feature in feature_values[(feature_values['lextypes'] == lextype) & (feature_values['shap']== sh)]['feature_names']]
            matcher.add(match_name+' '+str(sh), patterns)

        matches = matcher(doc)
        doc.spans[span_name] = list()
        for match_id, position in matches:
            span = Span(doc.doc, np.min(position), np.max(position)+1, label=match_id)
            doc.spans[span_name].append(span)

    return None

# # TO DO: POPRAW ZAKRES WARTOŚCI, PODAJ ETYKIETY KLAS NA GRANICACH
# def plot_colormap(pallete,plot = True):
#     if plot == True:
#         fig, ax = plt.subplots(figsize=(10,0.5))
#         col_map = plt.get_cmap(pallete)
#         mpl.colorbar.ColorbarBase(ax, cmap=col_map, orientation = 'horizontal')
#         return plt.show(fig)
#     else:
#         return None

# def choose_color(option,features):
#     colors = {}
#     if option == 'shap':
#         # POKOLORUJ WEDŁUG PERCENTYLI
#         # pallete = shap.plots.colors.red_blue # używana w SHAP beeswarm plot
#         pallete = 'PiYG' # zielone: SHAP>0, fuksja: SHAP<0, jasne: bliskie zera
#         plot_colormap(pallete,plot = True)

#         for sh in features['shap'].unique():
#             c = features[features['shap']==sh]['shap_pct'].mean()
#             for _, row in features[features['shap']==sh].iterrows():
#                 colors[str(row['lextypes'])+'-'+feature_extraction.dic_to_lextypes[row['lextypes']]+' '+str(sh)] = to_hex(colormaps[pallete](c))
#     elif option == 'feature_types':
#         # POKOLORUJ WEDŁUG TYPU CECHY
#         pallete = 'Set3'#'tab10'
#         plot_colormap(pallete,plot = False)
#         # for lt in ['LEMMA','POS','MORPH','LOWER','D2-LOWER','D2-LEMMA']:
#         for i, lt in enumerate(features['lextypes']):
#             for _, row in features[features['lextypes']==lt].iterrows():    
#                 colors[str(row['lextypes'])+'-'+feature_extraction.dic_to_lextypes[row['lextypes']]+' '+str(row['shap'])] = to_hex(colormaps[pallete](i))
#     elif option == 'feature_names':
#         # POKOLORUJ WEDŁUG NAZWY CECHY
#         pallete = 'Set3'#'tab10'
#         plot_colormap(pallete,plot = False)
#         for i, row in features.iterrows():
#             colors[row['feature_names'].split('_')[0]+' '+str(row['shap'])] = to_hex(colormaps[pallete](i))
#     else:
#         print('"option" must be one of: "shap", "feature_types" or "feature_names".')
#         return
#     return colors