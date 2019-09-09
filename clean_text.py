import numpy as np


def clean_text(text, nlp):
    vec = nlp(text)
    removed_punct = [token for token in vec if not token.is_punct]
    removed_nums = [token for token in removed_punct if not token.like_num]
    removed_stops = [token.text for token in removed_nums if not token.is_stop]
    cleaned_text = ' '.join(removed_stops)
    return cleaned_text if np.count_nonzero(nlp(cleaned_text).vector) != 0 else None
