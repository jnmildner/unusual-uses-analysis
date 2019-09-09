This repo contains 4 measures of creativity from unusual uses/alternative uses tasks.

1. Fluency:
    number of responses given by a participant. Responses that are highly similar are collapsed into 1 before counting
2. Elaboration:
    number of words per response (after removing stopwords)
3. Flexibility:
    dissimilarity between response and target word, corrected for elaboration
4. Originality:
    distance from nearest response cluster
    
To get a single creativity score per participant, Z score each of these and average them.

The file `example_run.py` contains an example analysis.