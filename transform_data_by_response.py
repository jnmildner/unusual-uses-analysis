import pandas as pd
import numpy as np
import re


def parse_responses(text: str, default_delimiter='\n', force_default=False) -> list:
    """Find most common punctuation in string of responses,
    delimit on that one (greedy)"""

    if pd.isnull(text):
        return []
    possible_delimiters = ['/', '\\', '\n', ',', '|']

    # remove spaces around possible delimiters (to remove empty responses)
    regex = r'\s*([' + ''.join(possible_delimiters) + r'])\s*'
    clean_text = re.sub(regex, r'\1', text)

    # find most common possible delimiter
    default_delimiter_count = len(re.findall(default_delimiter, clean_text))
    max_count = 0
    max_delimiter = ''
    for d in possible_delimiters:
        if d == default_delimiter:
            continue
        d_count = len(re.findall(re.escape(d), clean_text))
        if d_count > max_count:
            max_count = d_count
            max_delimiter = d

    text_delimiter = (default_delimiter if force_default | (default_delimiter_count >= max_count)
                      else max_delimiter)

    # split on text delimiter found
    split_pattern = r'[^' + text_delimiter + r']+'
    responses = re.findall(split_pattern, clean_text)

    return responses


def transform_data_by_response(raw_data, delimiter='/', id_column='ID',
                               response_column='response'):
    subject_ids = raw_data[id_column]
    raw_data = raw_data.set_index(id_column, drop=False)
    responses_by_subject = {}
    response_data = pd.DataFrame(columns=['ID', 'response_num', 'response'])
    for subject in subject_ids:
        response = raw_data.at[subject, response_column]
        if not isinstance(response, str):
            continue
        responses_by_subject[subject] = parse_responses(response,
                                                        default_delimiter=delimiter)
        nrow = len(responses_by_subject[subject])
        sub_df = pd.DataFrame(data={'ID': [subject] * nrow,
                                    'response_num': range(1, nrow + 1),
                                    'response': responses_by_subject[subject]})
        response_data = response_data.append(sub_df, ignore_index=True)

    response_data['responseID'] = np.random.choice(range(1, len(response_data) + 1),
                                                   size=len(response_data),
                                                   replace=False)
    return response_data.set_index('responseID', drop=False)
