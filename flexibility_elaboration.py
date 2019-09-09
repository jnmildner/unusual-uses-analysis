import numpy as np
import pandas as pd
import spacy
import random
from clean_text import clean_text


def calc_flexibility_and_elaboration(responses, target_word, nlp):
    data = pd.DataFrame({'clean_response': clean_text(response, nlp)} for response in responses)
    data['elaboration'] = data.apply(lambda row: len(row.clean_response.split()) if
                                     row.clean_response is not None else 0, axis=1)
    # to control for effects of response length (elaboration) on semantic similarity, calculate similarity expected by
    # chance for all given response lengths to subtract from response similarity
    # (Forthmann et al, 2018 https://doi.org/10.1002/jocb.240)
    nlp_smaller = spacy.load('en_core_web_md')
    word_counts = data.elaboration.unique()
    bootstrapped_sims = {}
    print('correcting flexibility for word count...')
    for sample_size in word_counts:
        sample_sims = []
        print('bootstrapping at word count ' + str(sample_size))
        if sample_size == 0:
            bootstrapped_sims[sample_size] = 0
            continue
        for i in list(range(0, 10000)):
            sampled_keys = random.sample(nlp_smaller.vocab.vectors.keys(), sample_size)
            while any([nlp_smaller.vocab[key].is_stop for key in sampled_keys]):
                sampled_keys = random.sample(nlp_smaller.vocab.vectors.keys(), sample_size)
            sampled_text = ' '.join([nlp_smaller.vocab[key].text for key in sampled_keys])
            sim_to_target = nlp_smaller(sampled_text).similarity(nlp_smaller(target_word))
            sample_sims.append(sim_to_target)
        bootstrapped_sims[sample_size] = np.mean(sample_sims)

    target_vec = nlp(target_word)
    data['raw_similarity'] = data.apply(lambda row: None if row.clean_response is None else
                                        (nlp(row.clean_response.lower()).similarity(target_vec) if
                                         np.count_nonzero(nlp(row.clean_response.lower()).vector) != 0 else None),
                                        axis=1)
    data['corrected_similarity'] = data.apply(lambda row: row.raw_similarity - bootstrapped_sims[row.elaboration],
                                              axis=1)
    # flexibility is dissimilarity score, so invert the similarity score to get flexibility
    data['flexibility'] = (1 - abs(data['corrected_similarity']))
    return data[['clean_response', 'elaboration', 'flexibility']]
