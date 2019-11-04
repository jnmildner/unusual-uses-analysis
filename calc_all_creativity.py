import pandas as pd
import spacy
import numpy as np
from transform_data_by_response import transform_data_by_response
from fluency import calc_fluency
from flexibility_elaboration import calc_flexibility_and_elaboration, calc_flexibility_and_elaboration_multi_target
from originality import calc_originality


def z_score(array):
    """Convert array of numbers to Z scores"""
    return (array - np.nanmean(array)) / np.std(array)


def mean_by_subject(by_subject_df, by_response_df, variable):
    """Calculate the mean score by subject (variable 'ID' in both by_subject_df and by_response_df

    Arguments
    ---------
    by_subject_df: pandas dataframe
        output dataframe (one row per subject, column 'ID' has subject IDs)
    by_response_df: pandas dataframe
        input dataframe (one row per response, multiple responses per subject, column 'ID' has subject IDs)
    variable: str
        column name of data to average in by_response_df

    Returns
    -------
    pandas series
        new column to add to by_subject_df containing the mean of 'variable' by subject
    """
    return by_subject_df.apply(
        lambda row:
            np.nanmean(by_response_df.loc[by_response_df.ID == row.ID, variable]),
        axis=1
    )


def calc_all_creativity(data_by_response, target_word=None, nlp=None, output_prefix='', multi_target=False):
    """ Calculate fluency, flexibility, elaboration, and originality. Then Z score and calculate creativity score

    This function calls fluency.py, flexibility_elaboration.py, and originality.py to calculate the four
    divergent thinking metrics. Flexibility, elaboration, and originality are calculated for each response, then
    Z scored and averaged within each participant. Fluency is calculated for each participant and then Z scored.
    Each participant's Z scored flexibility, elaboration, originality, and fluency are then averaged into a single
    score (labeled 'creativity_score').

    Arguments
    ---------
    data_by_response: pandas dataframe
        dataframe as created by transform_data_by_response.py. One row per response with columns 'responseID', 'ID',
        and 'response'. If multi_target is True, there should also be a 'target_word' column
    target_word: str, optional
        task's target word, used to calculate flexibility in flexibility_elaboration.py.
        Should be string if multi_target is False, and list if multi_target is True
    nlp: Spacy model, optional
        output from spacy.load(). If not provided, will load 'en_vectors_web_lg'.
    output_prefix: str, optional
        prefix to use for column names in output. (default no prefix)
    multi_target: bool, optional
        True if responses use different target words, False if all responses use the same target word. (default False)

    Returns
    -------
    {
        results_by_subject: pandas dataframe
            dataframe with one row per subject containing each subject's average z scored originality, flexibility,
            and elaboration, raw and z-scored fluency, and creativity_score
        results_by_response: pandas dataframe
            dataframe with one row per response containing cleaned responses, and both raw and z scored originality,
            flexibility, and originality
    }
    """
    if (target_word is None) & (not multi_target):
        raise TypeError(
            "If the task has a single target word, provide a target_word argument. If there are multiple target words, "
            + "set the multi_target argument to True and make sure data_by_response has a 'target_word' column")

    if nlp is None:
        nlp = spacy.load('en_vectors_web_lg')
    # make sure prefix ends in _ if there is one
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
    elab_out_name, flex_out_name, orig_out_name, flue_out_name, creativity_out_name = [
        output_prefix + n for n in ['elaboration', 'flexibility', 'originality', 'fluency',
                                    'creativity_score']]
    results_by_subject[elab_out_name + '_z'] = mean_by_subject(results_by_subject, results_df, 'z_elaboration')
    results_by_subject[elab_out_name + '_raw'] = mean_by_subject(results_by_subject, results_df, 'elaboration')
    results_by_subject[flex_out_name + '_z'] = mean_by_subject(results_by_subject, results_df, 'z_flexibility')
    results_by_subject[flex_out_name + '_raw'] = mean_by_subject(results_by_subject, results_df, 'flexibility')
    results_by_subject[orig_out_name + '_z'] = mean_by_subject(results_by_subject, results_df, 'z_originality')
    results_by_subject[orig_out_name + '_raw'] = mean_by_subject(results_by_subject, results_df, 'originality')

    results_by_subject[flue_out_name + '_raw'] = mean_by_subject(results_by_subject, results_df, 'fluency')
    results_by_subject[flue_out_name + '_z'] = z_score(results_by_subject[flue_out_name + '_raw'])

    results_by_subject[creativity_out_name] = results_by_subject[[elab_out_name + '_z',
                                                                  flex_out_name + '_z',
                                                                  orig_out_name + '_z',
                                                                  flue_out_name + '_z']].mean(axis=1)

    return {'results_by_subject': results_by_subject, 'results_by_response': results_df}

