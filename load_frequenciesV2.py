import pandas as pd
import string
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess

data = {'sentid': [], 'pairid': [], 'sentence': [], 'roi': []}
numset = ['0,', '1', '2', '3', '4', '5', '6', '7', '8', '9']

def main():
    freq_dict ={}
    freq_df = pd.read_csv('freqs_coca.tsv', sep='\t')
    words_list = freq_df['word'].tolist()
    count_list = freq_df['count'].tolist()
    for i in range(len(words_list)):
        freq_dict[words_list[i]] = count_list[i]
    #print(freq_dict)

    evaluate_df = pd.read_csv('results/homonym_minimal_pairs_byword.tsv', sep='\t')
    eval_words = evaluate_df['word_mod'].tolist()
    eval_freqs = []
    # count was used to check how many unfound words there were. Only 4 so we can ignore
    # count = 0

    for word in eval_words:
        # https://www.geeksforgeeks.org/python/string-punctuation-in-python/
        word = word.lower().strip().translate(str.maketrans('', '', string.punctuation))
        if word in freq_dict:
            eval_freqs.append(freq_dict[word])
        else:
            # count += 1
            eval_freqs.append(0)
    # print(count)

    evaluate_df['freqs'] = eval_freqs

    # MODIFICATION: Add log frequency to avoid regression domination by high-frequency words 
    evaluate_df['log_freq'] = np.log1p(evaluate_df['freqs'])

    evaluate_df.to_csv('results/homonym_minimal_pairs_byword_adjusted.tsv',
                       sep='\t', index=False)

    # MODIFICATION: Use surprisal instead of raw probabilities
    # GPT-like probabilities are highly nonlinear; surprisal linearizes them
    surprisal = np.array(evaluate_df['surp']).reshape((-1, 1))
    log_freqs = np.array(evaluate_df['log_freq']).reshape((-1, 1))


    # LOWESS docs: https://www.statsmodels.org/stable/generated/statsmodels.nonparametric.smoothers_lowess.lowess.html
    loess_fit = lowess(
        evaluate_df['surp'].values,
        evaluate_df['log_freq'].values,
        frac=0.25,   # smoothing parameter
        it=3,
        return_sorted=False
    )

    # Expected surprisal from smoothed frequency–surprisal curve
    evaluate_df['expected_surp_loess'] = loess_fit

    evaluate_df['adjusted_surp'] = (
        evaluate_df['surp'] - evaluate_df['expected_surp_loess']
    )

    evaluate_df['adjusted_prob'] = np.exp(-evaluate_df['adjusted_surp'])

    evaluate_df.to_csv('results/homonym_minimal_pairs_byword_adjusted.tsv',
                       sep='\t', index=False)

if __name__ == "__main__":
    main()
