import pandas as pd
import spacy
import numpy as np
from transform_data_by_response import transform_data_by_response
from fluency import calc_fluency
from flexibility_elaboration import calc_flexibility_and_elaboration, calc_flexibility_and_elaboration_multi_target
from originality import calc_originality


def z_score(array):
    return (array - np.nanmean(array)) / np.std(array)


def mean_by_subject(by_subject_df, by_response_df, variable):
    return by_subject_df.apply(
        lambda row:
            np.nanmean(by_response_df.loc[by_response_df.ID == row.ID, variable]),
        axis=1
    )


def calc_all_creativity(data_by_response, target_word, nlp=None, output_prefix='', multi_target=False):
    if nlp is None:
        nlp = spacy.load('en_vectors_web_lg')
    # make sure prefix ends in _
    if not (output_prefix.endswith('_') | (output_prefix == '')):
        output_prefix = output_prefix + '_'

    results_df = pd.DataFrame({'responseID': data_by_response.responseID, 'ID': data_by_response.ID})
    results_df['fluency'] = calc_fluency(data_by_response, nlp)
    results_df[['clean_response', 'elaboration', 'flexibility']] = \
        calc_flexibility_and_elaboration(list(data_by_response.response), target_word, nlp) if not multi_target else \
        calc_flexibility_and_elaboration_multi_target(list(data_by_response.response),
                                                      list(data_by_response.target_word),
                                                      nlp)
    results_df['originality'] = calc_originality(data_by_response.response)

    results_df['z_elaboration'] = z_score(results_df.elaboration)
    results_df['z_flexibility'] = z_score(results_df.flexibility)
    results_df['z_originality'] = z_score(results_df.originality)

    results_by_subject = pd.DataFrame({'ID': results_df.ID.unique()})
    elab_out_name, flex_out_name, orig_out_name, raw_flue_out_name, flue_out_name, creativity_out_name = [
        output_prefix + n for n in ['elaboration', 'flexibility', 'originality', 'raw_fluency', 'fluency',
                                    'creativity_score']]
    results_by_subject[elab_out_name] = mean_by_subject(results_by_subject, results_df, 'z_elaboration')
    results_by_subject[flex_out_name] = mean_by_subject(results_by_subject, results_df, 'z_flexibility')
    results_by_subject[orig_out_name] = mean_by_subject(results_by_subject, results_df, 'z_originality')

    results_by_subject[raw_flue_out_name] = mean_by_subject(results_by_subject, results_df, 'fluency')
    results_by_subject[flue_out_name] = z_score(results_by_subject[output_prefix + 'raw_fluency'])

    results_by_subject[creativity_out_name] = results_by_subject[[elab_out_name, flex_out_name,
                                                                  orig_out_name, flue_out_name]].mean(axis=1)

    return {'results_by_subject': results_by_subject, 'results_by_response': results_df}

