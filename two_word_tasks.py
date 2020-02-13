import pandas as pd
import spacy
import datetime
from transform_data_by_response import transform_data_by_response


def prep_data(data, delimiter='/', response_column='response'):
    """Put data in correct format (one row per trial, columns are ID, response, target words
    Arguments:
    ----------
        data: pandas.DataFrame with columns ID, trial, response, word1, word2
        delimiter: string
        response_column: string
    Returns:
    --------
        pandas.DataFrame
    """
    data['trialID'] = data.ID + '_' + data.trial.map(str)
    transformed_data = transform_data_by_response(data,
                                                  delimiter=delimiter,
                                                  id_column='trialID',
                                                  response_column=response_column)
    transformed_data['target_words'] = transformed_data.apply(
        lambda row: data.loc[data.trialID == row.ID, ['word1', 'word2']].values[0],
        axis=1
    )
    return transformed_data




nlp = spacy.load('en_vectors_web_lg')
fall17_sc_E = pd.read_csv('data/20200212_fall17_creativeE_long.csv')
transformed_data = prep_data(fall17_sc_E)
