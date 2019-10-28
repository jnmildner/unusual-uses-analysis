import numpy as np
import pandas as pd
import spacy
import random
from clean_text import clean_text
import glob
import pickle


def calc_similarity(clean_response, target, nlp):
    if clean_response is None:
        return None
    else:
        response_vec = nlp(clean_response.lower())
        target_vec = nlp(target)
        return response_vec.similarity(target_vec) if np.count_nonzero(response_vec) != 0 else None


def bootstrap_similarity(word_counts, target):
    nlp_smaller = spacy.load('en_core_web_md')
    # check if this word has been corrected before
    boot_filename = 'bootstraps/' + target + '.pkl'
    stored_bootstraps = glob.glob(boot_filename)
    print('correcting flexibility for word count...')
    if len(stored_bootstraps) == 0:
        bootstrapped_sims = {}
    else:
        print('Reading bootstrap data from file ' + stored_bootstraps[0])
        boot_file = open(stored_bootstraps[0], 'rb')
        bootstrapped_sims = pickle.load(boot_file)
        boot_file.close()
    for sample_size in word_counts:
        sample_sims = []
        if sample_size == 0:
            bootstrapped_sims[sample_size] = 0
            continue
        elif sample_size in bootstrapped_sims:
            continue
        print('bootstrapping at word count ' + str(sample_size))
        for i in list(range(0, 10000)):
            sampled_keys = random.sample(nlp_smaller.vocab.vectors.keys(), sample_size)
            while any([nlp_smaller.vocab[key].is_stop for key in sampled_keys]):
                sampled_keys = random.sample(nlp_smaller.vocab.vectors.keys(), sample_size)
            sampled_text = ' '.join([nlp_smaller.vocab[key].text for key in sampled_keys])
            sim_to_target = nlp_smaller(sampled_text).similarity(nlp_smaller(target))
            sample_sims.append(sim_to_target)
        bootstrapped_sims[sample_size] = np.mean(sample_sims)
    boot_file = open(boot_filename, 'wb')
    pickle.dump(bootstrapped_sims, boot_file, -1)
    boot_file.close()

    return bootstrapped_sims


def calc_flexibility_and_elaboration(responses, target_word, nlp):
    data = pd.DataFrame({'clean_response': clean_text(response, nlp)} for response in responses)
    data['elaboration'] = data.apply(lambda row: len(row.clean_response.split()) if
                                     row.clean_response is not None else 0, axis=1)
    # to control for effects of response length (elaboration) on semantic similarity, calculate similarity expected by
    # chance for all given response lengths to subtract from response similarity
    # (Forthmann et al, 2018 https://doi.org/10.1002/jocb.240)
    word_counts = data.elaboration.unique()
    bootstrapped_sims = bootstrap_similarity(word_counts, target_word)

    target_vec = nlp(target_word)
    data['raw_similarity'] = data.apply(lambda row: calc_similarity(row.clean_response, target_word, nlp),
                                        axis=1)
    data['corrected_similarity'] = data.apply(lambda row: row.raw_similarity - bootstrapped_sims[row.elaboration],
                                              axis=1)
    # flexibility is dissimilarity score, so invert the similarity score to get flexibility
    data['flexibility'] = (1 - abs(data['corrected_similarity']))
    return data[['clean_response', 'elaboration', 'flexibility']]


def calc_flexibility_and_elaboration_multi_target(responses, target_words, nlp):
    data = pd.DataFrame({'response': responses, 'target_word': target_words})
    data['clean_response'] = [clean_text(response, nlp) for response in data.response]
    data['elaboration'] = data.apply(lambda row: len(row.clean_response.split()) if
                                     row.clean_response is not None else 0, axis=1)
    # to control for effects of response length (elaboration) on semantic similarity, calculate similarity expected by
    # chance for all given response lengths to subtract from response similarity
    # (Forthmann et al, 2018 https://doi.org/10.1002/jocb.240)
    most_common_target = list(data.target_word.mode())[0]
    word_counts = data.elaboration.unique()
    bootstrapped_sims = bootstrap_similarity(word_counts, most_common_target)

    data['raw_similarity'] = data.apply(lambda row: calc_similarity(row.clean_response, row.target_word, nlp),
                                        axis=1)
    data['corrected_similarity'] = data.apply(lambda row: row.raw_similarity - bootstrapped_sims[row.elaboration],
                                              axis=1)
    # flexibility is dissimilarity score, so invert the similarity score to get flexibility
    data['flexibility'] = (1 - abs(data['corrected_similarity']))
    return data[['clean_response', 'elaboration', 'flexibility']]
