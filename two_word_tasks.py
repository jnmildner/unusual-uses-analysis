import pandas as pd
import spacy
import datetime
from transform_data_by_response import transform_data_by_response
from flexibility_elaboration import calc_flexibility_and_elaboration_two_target
from fluency import calc_fluency
from originality import calc_originality
from calc_all_creativity import z_score, mean_by_subject


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


def calc_all_creativity(data, nlp, output_prefix):
    results_df = pd.DataFrame({"responseID": data.responseID, "ID": data.ID})
    results_df["fluency"] = calc_fluency(data, nlp)
    results_df[["clean_response", "elaboration", "flexibility"]] = \
        calc_flexibility_and_elaboration_two_target(data.response, data.target_words, nlp)
    results_df["originality"] = calc_originality(data.response)

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


today_prefix = datetime.date.today().strftime('%Y%m%d')
output_location = 'output/' + today_prefix + '_'
output_location_response = 'output/data_by_response/' + today_prefix + '_'

nlp = spacy.load('en_vectors_web_lg')

fall17_sc_E_raw = pd.read_csv('data/20200212_fall17_creativeE_long.csv')
fall17_sc_E_out = calc_all_creativity(
    prep_data(fall17_sc_E_raw, delimiter='\n'),
    nlp, 'sc_E_')
fall17_sc_E_out['results_by_subject'].to_csv(
    output_location + 'fall17_socialCreativity_E.csv'
)
fall17_sc_E_out['results_by_response'].to_csv(
    output_location_response + 'fall17_socialCreativity_E_by_response.csv',
    index=False)

spring18_sc_E_raw = pd.read_csv('data/20200212_spring18_creativeE_long.csv')
spring18_sc_E_out = calc_all_creativity(
    prep_data(spring18_sc_E_raw, delimiter='\n'),
    nlp, 'sc_E_')
spring18_sc_E_out['results_by_subject'].to_csv(
    output_location + 'spring18_socialCreativity_E.csv'
)
spring18_sc_E_out['results_by_response'].to_csv(
    output_location_response + 'spring18_socialCreativity_E_by_response.csv',
    index=False)

fall17_sc_G_raw = pd.read_csv('data/20200212_fall17_creativeG_long.csv')
fall17_sc_G_out = calc_all_creativity(
    prep_data(fall17_sc_G_raw, delimiter='\n'),
    nlp, 'sc_G_')
fall17_sc_G_out['results_by_subject'].to_csv(
    output_location + 'fall17_socialCreativity_G.csv'
)
fall17_sc_G_out['results_by_response'].to_csv(
    output_location_response + 'fall17_socialCreativity_G_by_response.csv',
    index=False)

spring18_sc_G_raw = pd.read_csv('data/20200212_spring18_creativeG_long.csv')
spring18_sc_G_out = calc_all_creativity(
    prep_data(spring18_sc_G_raw, delimiter='\n'),
    nlp, 'sc_G_')
spring18_sc_G_out['results_by_subject'].to_csv(
    output_location + 'spring18_socialCreativity_G.csv'
)
spring18_sc_G_out['results_by_response'].to_csv(
    output_location_response + 'spring18_socialCreativity_G_by_response.csv',
    index=False)

fall17_sc_E_raw = pd.read_csv('data/20200212_fall17_creativeE_long.csv')
fall17_sc_E_out = calc_all_creativity(
    prep_data(fall17_sc_E_raw, delimiter='\n'),
    nlp, 'sc_E_')
fall17_sc_E_out['results_by_subject'].to_csv(
    output_location + 'fall17_socialCreativity_E.csv'
)
fall17_sc_E_out['results_by_response'].to_csv(
    output_location_response + 'fall17_socialCreativity_E_by_response.csv',
    index=False)

spring18_sc_H_raw = pd.read_csv('data/20200212_spring18_creativeH_long.csv')
spring18_sc_H_out = calc_all_creativity(
    prep_data(spring18_sc_H_raw, delimiter='\n'),
    nlp, 'sc_H_')
spring18_sc_H_out['results_by_subject'].to_csv(
    output_location + 'spring18_socialCreativity_H.csv'
)
spring18_sc_H_out['results_by_response'].to_csv(
    output_location_response + 'spring18_socialCreativity_H_by_response.csv',
    index=False)
