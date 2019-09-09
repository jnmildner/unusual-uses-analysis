from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np


def calc_originality(responses, clusters=8):
    responses = [' ' if pd.isnull(r) else r for r in responses]
    vectorizer = TfidfVectorizer()
    responses_tfidf = vectorizer.fit_transform(responses)
    cluster_model = KMeans(n_clusters=clusters)
    responses_cluster_distances = cluster_model.fit_transform(responses_tfidf)
    nearest_cluster_distances = np.min(responses_cluster_distances, axis=1)
    return nearest_cluster_distances


