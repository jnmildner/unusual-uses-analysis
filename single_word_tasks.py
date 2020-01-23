import pandas as pd
import spacy
import datetime
from transform_data_by_response import transform_data_by_response
from calc_all_creativity import calc_all_creativity

nlp = spacy.load('en_vectors_web_lg')


def transform_and_calc(data, nlp=nlp,  delimiter='/', id_column='ID',
                       response_column='response', output_prefix='',
                       target_word=None, multi_target=False):
    # append trial number to subject IDs to differentiate data from multiple
    # trials by the same subject
    if multi_target:
        data['trialID'] = data.ID + '_' + data.trial.map(str)
        id_column = 'trialID'
    transformed_data = transform_data_by_response(data,
                                                  delimiter=delimiter,
                                                  id_column=id_column,
                                                  response_column=response_column)
    if multi_target:
        transformed_data['target_word'] = transformed_data.apply(
            lambda row: list(data.loc[data.trialID == row.ID, 'word'])[0],
            axis=1
        )
    return calc_all_creativity(transformed_data,
                               nlp=nlp,
                               target_word=target_word,
                               output_prefix=output_prefix,
                               multi_target=multi_target)


today_prefix = datetime.date.today().strftime('%Y%m%d')
output_location = 'output/' + today_prefix + '_'
output_location_response = 'output/data_by_response/' + today_prefix + '_'

target_1 = u'pen'
target_2 = u'megaphone'
fall17_uu_raw = pd.read_csv('data/20190919_fall17_unusualUses.csv')
fall17_uu_pen_out = transform_and_calc(fall17_uu_raw,
                                       delimiter='/n',
                                       id_column='ID',
                                       response_column='unusualUses_1',
                                       output_prefix='uu_1_',
                                       target_word=target_1)
fall17_uu_meg_out = transform_and_calc(fall17_uu_raw,
                                       delimiter='/n',
                                       id_column='ID',
                                       response_column='unusualUses_2',
                                       output_prefix='uu_2_',
                                       target_word=target_2)
fall17_uu_results = fall17_uu_pen_out['results_by_subject'].merge(
    fall17_uu_meg_out['results_by_subject'])
fall17_uu_results.to_csv(output_location + 'fall17_unusualUses.csv',
                         index=False)
# also save the results by response, for debugging or later use
fall17_uu_pen_out['results_by_response'].to_csv(
    output_location_response + 'fall17_unusualUses_pen_by_response.csv',
    index=False)
fall17_uu_meg_out['results_by_response'].to_csv(
    output_location_response + 'fall17_unusualUses_meg_by_response.csv',
    index=False)


spring18_uu_raw = pd.read_csv('data/20190919_spring18_unusualUses.csv')
spring18_uu_pen_out = transform_and_calc(spring18_uu_raw,
                                         delimiter='/n',
                                         id_column='ID',
                                         response_column='unusualUses_1',
                                         output_prefix='uu_1_',
                                         target_word=target_1)
spring18_uu_meg_out = transform_and_calc(spring18_uu_raw,
                                         delimiter='/n',
                                         id_column='ID',
                                         response_column='unusualUses_2',
                                         output_prefix='uu_2_',
                                         target_word=target_2)
spring18_uu_results = spring18_uu_pen_out['results_by_subject'].merge(
    spring18_uu_meg_out['results_by_subject'])
spring18_uu_results.to_csv(output_location + 'spring18_unusualUses.csv',
                           index=False)
# also save the results by response, for debugging or later use
spring18_uu_pen_out['results_by_response'].to_csv(
    output_location_response + 'spring18_unusualUses_pen_by_response.csv',
    index=False)
spring18_uu_meg_out['results_by_response'].to_csv(
    output_location_response + 'spring18_unusualUses_meg_by_response.csv',
    index=False)

# social creativity task B
fall17_sc_B_raw = pd.read_csv('data/20190919_fall17_creativeB_long.csv')
fall17_sc_B_out = transform_and_calc(fall17_sc_B_raw, delimiter='/n',
                                     output_prefix='sc_B', multi_target=True)
fall17_sc_B_out['results_by_subject'].to_csv(
    output_location + 'fall17_socialCreativity_B.csv', index=False)
fall17_sc_B_out['results_by_response'].to_csv(
    output_location_response + 'fall17_socialCreativity_B_by_response.csv',
    index=False)

spring18_sc_B_raw = pd.read_csv('data/20190919_spring18_creativeB_long.csv')
spring18_sc_B_out = transform_and_calc(spring18_sc_B_raw, delimiter='/n',
                                     output_prefix='sc_B', multi_target=True)
spring18_sc_B_out['results_by_subject'].to_csv(
    output_location + 'spring18_socialCreativity_B.csv', index=False)
spring18_sc_B_out['results_by_response'].to_csv(
    output_location_response + 'spring18_socialCreativity_B_by_response.csv',
    index=False)

# social creativity task C
fall17_sc_C_raw = pd.read_csv('data/20190919_fall17_creativeC_long.csv')
fall17_sc_C_out = transform_and_calc(fall17_sc_C_raw, delimiter='/n',
                                     output_prefix='sc_C', multi_target=True)
fall17_sc_C_out['results_by_subject'].to_csv(
    output_location + 'fall17_socialCreativity_C.csv', index=False)
fall17_sc_C_out['results_by_response'].to_csv(
    output_location_response + 'fall17_socialCreativity_C_by_response.csv',
    index=False)

spring18_sc_C_raw = pd.read_csv('data/20190919_spring18_creativeC_long.csv')
spring18_sc_C_out = transform_and_calc(spring18_sc_C_raw, delimiter='/n',
                                     output_prefix='sc_C', multi_target=True)
spring18_sc_C_out['results_by_subject'].to_csv(
    output_location + 'spring18_socialCreativity_C.csv', index=False)
spring18_sc_C_out['results_by_response'].to_csv(
    output_location_response + 'spring18_socialCreativity_C_by_response.csv',
    index=False)

# social creativity task D
fall17_sc_D_raw = pd.read_csv('data/20190919_fall17_creativeD_long.csv')
fall17_sc_D_out = transform_and_calc(fall17_sc_D_raw, delimiter='/n',
                                     output_prefix='sc_D', multi_target=True)
fall17_sc_D_out['results_by_subject'].to_csv(
    output_location + 'fall17_socialCreativity_D.csv', index=False)
fall17_sc_D_out['results_by_response'].to_csv(
    output_location_response + 'fall17_socialCreativity_D_by_response.csv',
    index=False)

spring18_sc_D_raw = pd.read_csv('data/20190919_spring18_creativeD_long.csv')
spring18_sc_D_out = transform_and_calc(spring18_sc_D_raw, delimiter='/n',
                                     output_prefix='sc_D', multi_target=True)
spring18_sc_D_out['results_by_subject'].to_csv(
    output_location + 'spring18_socialCreativity_D.csv', index=False)
spring18_sc_D_out['results_by_response'].to_csv(
    output_location_response + 'spring18_socialCreativity_D_by_response.csv',
    index=False)

# social creativity task F
fall17_sc_F_raw = pd.read_csv('data/20190919_fall17_creativeF_long.csv')
fall17_sc_F_out = transform_and_calc(fall17_sc_F_raw, delimiter='/n',
                                     output_prefix='sc_F', multi_target=True)
fall17_sc_F_out['results_by_subject'].to_csv(
    output_location + 'fall17_socialCreativity_F.csv', index=False)
fall17_sc_F_out['results_by_response'].to_csv(
    output_location_response + 'fall17_socialCreativity_F_by_response.csv',
    index=False)

spring18_sc_F_raw = pd.read_csv('data/20190919_spring18_creativeF_long.csv')
spring18_sc_F_out = transform_and_calc(spring18_sc_F_raw, delimiter='/n',
                                     output_prefix='sc_F', multi_target=True)
spring18_sc_F_out['results_by_subject'].to_csv(
    output_location + 'spring18_socialCreativity_F.csv', index=False)
spring18_sc_F_out['results_by_response'].to_csv(
    output_location_response + 'spring18_socialCreativity_F_by_response.csv',
    index=False)








