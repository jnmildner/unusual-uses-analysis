import pandas as pd
import numpy as np


def calculate_corrected_fluency(response_list, nlp):
    threshold = .8
    response_vecs = [nlp(response) for response in response_list]
    clean_response_vecs = []
    for response in response_vecs:
        no_punct_list = [token for token in response
                         if not token.is_punct]
        no_num_list = [token for token in no_punct_list
                       if not token.like_num]
        no_stops_list = [token.text for token in no_num_list
                         if not token.is_stop]
        no_stops = nlp(' '.join(no_stops_list))
        if np.count_nonzero(no_stops.vector) == 0:
            continue
        clean_response_vecs.append(no_stops)
    merged = []
    unique_resp_list = []
    # loop thru responses except the last one (redundant for pairwise sim)
    for index, resp in enumerate(clean_response_vecs[:-1]):
        # skip if we already merged this response with another one
        if index in merged:
            continue
        # also skip if there is no vector found for this response
        # (i.e. it does not contain any words recognized by spacy)
        if np.count_nonzero(resp.vector) == 0:
            continue
        similarities = [resp.similarity(r) for r in
                        clean_response_vecs[index + 1:]]
        # for each response, check if most similar exceeds threshold.
        max_similarity = max(similarities)
        # keep merging until most similar item is no longer above threshold.
        # first, copy the response vector so we can modify it in the loop
        # without messing with the original
        response_vecs_copy = clean_response_vecs.copy()
        new_resp = resp
        while max_similarity >= threshold:
            # get index of most similar item
            merge_index = similarities.index(max_similarity) + index + 1
            # use the index to get the most similar response
            similar_resp = response_vecs_copy[merge_index]
            # combine current response and similar response into one
            new_resp = nlp(new_resp.text + '. ' + similar_resp.text)
            # put the new response at the current index
            response_vecs_copy[index] = new_resp
            # remove the merged response from the array
            response_vecs_copy.pop(merge_index)
            # find the index of the merged item in the original array
            # add this to merged to skip it when the outer loop gets there
            merged.append(clean_response_vecs.index(similar_resp))
            # regenerate the similarity array with the modified responses
            similarities = [new_resp.similarity(r) for r in
                            response_vecs_copy[index + 1:]]
            # re-calculate the max similarity.
            # Loop will end if it's below threshold
            max_similarity = (max(similarities) if len(similarities) >= 1
                              else 0)
        unique_resp_list.append(response_vecs_copy[index])
    return len(unique_resp_list)


def calc_fluency(response_df, nlp, id_column='ID', response_column='response'):
    responses_by_id = {ID: list(response_df.loc[response_df[id_column] == ID, response_column]) for
                       ID in response_df[id_column].unique()}
    data = pd.DataFrame({id_column: response_df[id_column].unique()})
    data['fluency'] = data.apply(lambda row: calculate_corrected_fluency(responses_by_id[row[id_column]], nlp), axis=1)
    full_data = response_df.merge(data, on=id_column, sort=False)
    return list(full_data['fluency'])
