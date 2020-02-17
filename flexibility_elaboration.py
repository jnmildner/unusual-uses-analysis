import numpy as np
import pandas as pd
import spacy
import random
from clean_text import clean_text
import glob
import pickle

msg_prefix = '[FLEX] '


def calc_similarity(clean_response, target, nlp):
    """Calculate the similarity between response and target using Spacy

    Arguments
    ---------
    clean_response: string
    target: string
    nlp: Spacy model

    Returns
    -------
    similarity
    """
    if clean_response is None:
        return None
    else:
        response_vec = nlp(clean_response.lower())
        target_vec = nlp(target)
        return response_vec.similarity(target_vec) if np.count_nonzero(response_vec) != 0 else None


def bootstrap_similarity(word_counts, target):
    """Calculate the average similarity for random words of each response length.

    Longer responses have higher similarity. To control for this, draw random words 10,000 times for each response
    length. Calculate the average similarity for each response length. To save time, store bootstrap data for each
    target word in a pickle.

    Arguments
    ---------
    word_counts: list
        List of unique elaboration scores
    target: string
        Target word

    Returns
    -------
    bootstrapped similarities: object
        keys for each word count with values for average similarity for that response length
    """
    nlp_smaller = spacy.load('en_core_web_md')
    # check if this word has been corrected before
    boot_filename = 'bootstraps/' + target + '.pkl'
    stored_bootstraps = glob.glob(boot_filename)
    print(msg_prefix + 'Correcting flexibility for word count for target word: ' + target)
    if len(stored_bootstraps) == 0:
        bootstrapped_sims = {}
    else:
        print(msg_prefix + 'Reading bootstrap data from file ' + stored_bootstraps[0])
        boot_file = open(stored_bootstraps[0], 'rb')
        bootstrapped_sims = pickle.load(boot_file)
        boot_file.close()
    for sample_size in word_counts:
        sample_sims = []
        if (sample_size == 0) | (sample_size is None) | (np.isnan(sample_size)):
            bootstrapped_sims[sample_size] = 0
            continue
        elif sample_size in bootstrapped_sims:
            continue
        else:
            sample_size = sample_size.astype(int)
        print(msg_prefix + 'Bootstrapping at word count ' + str(sample_size))
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
    """Calculate flexibility (spacy similarity corrected for chance similarity) and
    elaboration (number of words).

    Arguments
    ---------
        responses: list
        target_word: string
        nlp: Spacy model

    Returns
    -------
        pandas dataframe with 3 columns: clean_response, elaboration, flexibility
    """
    data = pd.DataFrame({'clean_response': clean_text(response, nlp)} for response in responses)
    data['elaboration'] = data.apply(lambda row: int(len(row.clean_response.split())) if
    row.clean_response is not None else None, axis=1)
    # to control for effects of response length (elaboration) on semantic similarity, calculate similarity expected by
    # chance for all given response lengths to subtract from response similarity
    # (Forthmann et al, 2018 https://doi.org/10.1002/jocb.240)
    word_counts = data.elaboration.unique()
    bootstrapped_sims = bootstrap_similarity(word_counts, target_word)

    data['raw_similarity'] = data.apply(lambda row: calc_similarity(row.clean_response, target_word, nlp),
                                        axis=1)
    data['corrected_similarity'] = data.apply(
        lambda row:
        row.raw_similarity - bootstrapped_sims[row.elaboration] if not np.isnan(
            row.elaboration) else row.raw_similarity,
        axis=1)
    # flexibility is dissimilarity score, so invert the similarity score to get flexibility
    data['flexibility'] = (1 - abs(data['corrected_similarity']))
    return data[['clean_response', 'elaboration', 'flexibility']]


def calc_flexibility_and_elaboration_multi_target(responses, target_words, nlp):
    """"""
    data = pd.DataFrame({'response': responses, 'target_word': target_words})
    data['clean_response'] = [clean_text(response, nlp) for response in data.response]
    data['elaboration'] = data.apply(lambda row: len(row.clean_response.split()) if
    row.clean_response is not None else None, axis=1)
    # to control for effects of response length (elaboration) on semantic similarity, calculate similarity expected by
    # chance for all given response lengths to subtract from response similarity
    # (Forthmann et al, 2018 https://doi.org/10.1002/jocb.240)
    bootstrapped_sims = {}
    for target in data.target_word.unique():
        word_counts = data.elaboration.loc[data.target_word == target].unique()
        bootstrapped_sims[target] = bootstrap_similarity(word_counts, target)

    data['raw_similarity'] = data.apply(lambda row: calc_similarity(row.clean_response, row.target_word, nlp),
                                        axis=1)
    data['corrected_similarity'] = data.apply(
        lambda row:
        row.raw_similarity - bootstrapped_sims[row.target_word][row.elaboration] if not
        np.isnan(row.elaboration) else row.raw_similarity,
        axis=1)
    # flexibility is dissimilarity score, so invert the similarity score to get flexibility
    data['flexibility'] = (1 - abs(data['corrected_similarity']))
    return data[['clean_response', 'elaboration', 'flexibility']]


def calc_flexibility_and_elaboration_two_target(responses, target_words, nlp):
    """version of calc_flexibility_and_elaboration_multi_target for tasks that
    use two target words at once (i.e each trial has a set of two target words).

    Arguments:
    ----------
        responses: list
        target_words: list of lists
        nlp: Spacy model
    """
    data = pd.DataFrame({'response': responses, 'target_words': target_words})
    data['clean_response'] = [clean_text(response, nlp) for response in data.response]
    data['elaboration'] = data.apply(lambda row: len(row.clean_response.split()) if
    row.clean_response is not None else None, axis=1)
    # flatten list of target words (from list of lists to single list)
    all_target_words = [item for sublist in data.target_words for item in sublist]
    bootstrapped_sims = {}
    for target in set(all_target_words):
        word_counts = data.elaboration.loc[
            data.target_words.apply(lambda x: target in x)
        ].unique()
        bootstrapped_sims[target] = bootstrap_similarity(word_counts, target)

    data['raw_similarity1'] = data.apply(
        lambda row: calc_similarity(row.clean_response, row.target_words[0], nlp),
        axis=1)
    data['raw_similarity2'] = data.apply(
        lambda row: calc_similarity(row.clean_response, row.target_words[1], nlp),
        axis=1)
    data['corrected_similarity1'] = data.apply(
        lambda row:
        row.raw_similarity1 - bootstrapped_sims[row.target_words[0]][row.elaboration] if not
        np.isnan(row.elaboration) else row.raw_similarity1,
        axis=1)
    data['corrected_similarity2'] = data.apply(
        lambda row:
        row.raw_similarity2 - bootstrapped_sims[row.target_words[1]][row.elaboration] if not
        np.isnan(row.elaboration) else row.raw_similarity2,
        axis=1)
    data['highest_similarity'] = data.apply(
        lambda row: max(abs(row.corrected_similarity1), abs(row.corrected_similarity2)),
        axis=1
    )
    data['flexibility'] = (1 - abs(data['highest_similarity']))
    return data[['clean_response', 'elaboration', 'flexibility']]

