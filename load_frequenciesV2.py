import pandas as pd
import string
import numpy as np
from sklearn.linear_model import LinearRegression

# Reference: GAM workflow inspiration from Statistical Learning with Python (GAM chapter).
# https://mibur1.github.io/psy112/book/2_models/3_GAM.html

from gam_utils import fit_regularized_gam


data = {'sentid': [], 'pairid': [], 'sentence': [], 'roi': []}
numset = ['0,', '1', '2', '3', '4', '5', '6', '7', '8', '9']

def main():
    freq_dict = {}
    freq_df = pd.read_csv('freqs_coca.tsv', sep='\t')
    words_list = freq_df['word'].tolist()
    count_list = freq_df['count'].tolist()
    for i in range(len(words_list)):
        freq_dict[words_list[i]] = count_list[i]

    evaluate_df = pd.read_csv('results/large_homonym_minimal_pairs_byword.tsv', sep='\t')
    eval_words = evaluate_df['word_mod'].tolist()
    eval_freqs = []

    for word in eval_words:
        # https://www.geeksforgeeks.org/python/string-punctuation-in-python/
        word = word.lower().strip().translate(str.maketrans('', '', string.punctuation))
        if word in freq_dict:
            eval_freqs.append(freq_dict[word])
        else:
            eval_freqs.append(0)

    evaluate_df['freqs'] = eval_freqs

    # log frequency
    evaluate_df['log_freq'] = np.log1p(evaluate_df['freqs'])

    evaluate_df.to_csv('results/large_homonym_minimal_pairs_byword_adjusted.tsv', sep='\t', index=False)

    surprisal = np.array(evaluate_df['surp']).reshape((-1, 1))
    log_freqs = np.array(evaluate_df['log_freq']).reshape((-1, 1))
   
    
    linreg = LinearRegression()
    linreg.fit(log_freqs, surprisal)

    
    linear_expected = linreg.predict(log_freqs)

    evaluate_df['expected_surp_linear'] = linear_expected.flatten()

    evaluate_df['adjusted_surp_linear'] = (evaluate_df['surp'] - evaluate_df['expected_surp_linear'])

    evaluate_df['adjusted_prob_linear'] = np.exp(-evaluate_df['adjusted_surp_linear'])
   
    gam, best_lam, _ = fit_regularized_gam(
        log_freqs,
        surprisal,
        lam_values=np.logspace(-3, 2, 8),
        n_splines=10,
        n_splits=5,
        test_size=0.2,
        random_state=42,
        verbose=True,
    )
    print(f"GAM lam selected for adjustment: {best_lam:.4f}")

    expected = gam.predict(log_freqs)
    evaluate_df['expected_surp_gam'] = expected

    evaluate_df['adjusted_surp'] = (evaluate_df['surp'] - evaluate_df['expected_surp_gam'])

    evaluate_df['adjusted_prob'] = np.exp(-evaluate_df['adjusted_surp'])
    #filtering out tokens like ')' and ',' which corrupted the GAM regression slightly
    evaluate_df = evaluate_df[evaluate_df['word_mod'].str.contains('[a-zA-Z]', regex=True, na=False)]
    evaluate_df = evaluate_df[evaluate_df['freqs'] > 0]

    evaluate_df.to_csv('results/large_homonym_minimal_pairs_byword_adjusted_GAM.tsv', sep='\t', index=False)

if __name__ == "__main__":
    main()
