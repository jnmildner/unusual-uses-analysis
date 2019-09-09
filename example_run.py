import pandas as pd
import numpy as np
import spacy
from transform_data_by_response import transform_data_by_response
from fluency import calc_fluency
from flexibility_elaboration import calc_flexibility_and_elaboration
from originality import calc_originality


raw_data = pd.read_csv('data/raw_example_data.csv')
data_by_response = transform_data_by_response(raw_data, delimiter='/', id_column='ID',
                                              response_column='Alternative Uses')
creativity = pd.DataFrame({'responseID': data_by_response.responseID,
                           'ID': data_by_response.ID})

# load spacy vectors here, so we only have to do it once
# first time using this, run `python -m spacy download en_vectors_web_lg` and `python -m spacy download en_core_web_md`
nlp = spacy.load('en_vectors_web_lg')
creativity['fluency'] = calc_fluency(data_by_response, nlp, id_column='ID', response_column='response')
creativity[['clean_response', 'elaboration', 'flexibility']] = \
    calc_flexibility_and_elaboration(list(data_by_response.response), u'pen', nlp)
creativity['originality'] = calc_originality(data_by_response.response)


def z_score(array):
    return (array - np.nanmean(array)) / np.std(array)


# elaboration, flexibility, and originality are measured at the response level, so Z score at that level
creativity['z_elaboration'] = z_score(creativity.elaboration)
creativity['z_flexibility'] = z_score(creativity.flexibility)
creativity['z_originality'] = z_score(creativity.originality)

creativity_by_participant = pd.DataFrame({'ID': creativity.ID.unique()})
creativity_by_participant['z_elaboration'] = creativity_by_participant.apply(
    lambda row:
        np.nanmean(creativity.loc[creativity.ID == row.ID, 'z_elaboration']),
    axis=1)
creativity_by_participant['z_flexibility'] = creativity_by_participant.apply(
    lambda row:
        np.nanmean(creativity.loc[creativity.ID == row.ID, 'z_flexibility']),
    axis=1)
creativity_by_participant['z_originality'] = creativity_by_participant.apply(
    lambda row:
        np.nanmean(creativity.loc[creativity.ID == row.ID, 'z_originality']),
    axis=1)
creativity_by_participant['fluency'] = creativity_by_participant.apply(
    lambda row:
        np.nanmean(creativity.loc[creativity.ID == row.ID, 'fluency']),
    axis=1)
# fluency is calculated on the participant level, so z score on the participant level
creativity_by_participant['z_fluency'] = z_score(creativity_by_participant['fluency'])

creativity_by_participant['creativity_score'] = creativity_by_participant[
    ['z_elaboration', 'z_flexibility', 'z_originality', 'z_fluency']].mean(axis=1)
