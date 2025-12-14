import pandas as pd
import string
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.utils import resample

# Reference: plotting patterns adapted from pyGAM "Tour" notebook examples.
# https://pygam.readthedocs.io/en/latest/notebooks/tour_of_pygam.html

from gam_utils import fit_regularized_gam


def plot_gam_with_ci(df, X, gam):
    XX = np.linspace(X.min(), X.max(), 300)
    preds = gam.predict(XX)
    conf_intervals = gam.confidence_intervals(XX)

    plt.figure(figsize=(8,5))
    plt.scatter(df["log_freq"], df["surp"], alpha=0.3, label="Data")
    plt.plot(XX, preds, color='red', label="GAM fit")
    plt.fill_between(XX, conf_intervals[:,0], conf_intervals[:,1],
                     color='red', alpha=0.2, label="95% CI")
    plt.xlabel("log frequency")
    plt.ylabel("surprisal")
    plt.title("GAM Regression with 95% Confidence Interval")
    plt.legend()
    plt.tight_layout()
    plt.savefig("graphs/large_gam_regression_ci.png")
    plt.close()


def plot_linear_with_ci(df, X, linreg):
    # Bootstrapped CI for linear regression
    XX = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
    preds = linreg.predict(XX)

    boot_preds = []
    for _ in range(300):
        sample = resample(df)
        X_sample = sample["log_freq"].values.reshape(-1, 1)
        y_sample = sample["surp"].values
        lr = LinearRegression().fit(X_sample, y_sample)
        boot_preds.append(lr.predict(XX))

    boot_preds = np.array(boot_preds)
    lower = np.percentile(boot_preds, 2.5, axis=0)
    upper = np.percentile(boot_preds, 97.5, axis=0)

    plt.figure(figsize=(8,5))
    plt.scatter(df["log_freq"], df["surp"], alpha=0.3, label="Data")
    plt.plot(XX, preds, color='blue', label="Linear fit")
    plt.fill_between(XX.flatten(), lower, upper,
                     color='blue', alpha=0.2, label="95% CI")
    plt.xlabel("log frequency")
    plt.ylabel("surprisal")
    plt.title("Linear Regression with 95% Confidence Interval")
    plt.legend()
    plt.tight_layout()
    plt.savefig("graphs/large_linear_regression_ci.png")
    plt.close()


def plot_gam_vs_linear(df, X, gam, linreg):
    XX = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
    gam_preds = gam.predict(XX)
    lin_preds = linreg.predict(XX)

    plt.figure(figsize=(8,5))
    plt.scatter(df["log_freq"], df["surp"], alpha=0.3, label="Data")
    plt.plot(XX, gam_preds, color='red', label="GAM fit")
    plt.plot(XX, lin_preds, color='blue', linestyle='--', label="Linear fit")
    plt.xlabel("log frequency")
    plt.ylabel("surprisal")
    plt.title("GAM vs Linear Regression Fit")
    plt.legend()
    plt.tight_layout()
    plt.savefig("graphs/large_gam_vs_linear_no_ci.png")
    plt.close()



def main():
    freq_df = pd.read_csv('freqs_coca.tsv', sep='\t')
    freq_dict = dict(zip(freq_df['word'], freq_df['count']))

    evaluate_df = pd.read_csv('results/large_homonym_minimal_pairs_byword_adjusted_GAM.tsv', sep='\t')

    cleaned_words = [
        w.lower().strip().translate(str.maketrans('', '', string.punctuation))
        for w in evaluate_df['word_mod'].tolist()
    ]
    evaluate_df["freqs"] = [freq_dict.get(w, 0) for w in cleaned_words]

    evaluate_df['log_freq'] = np.log1p(evaluate_df['freqs'])

    X = evaluate_df['log_freq'].values.reshape(-1, 1)
    y = evaluate_df['surp'].values

    linreg = LinearRegression().fit(X, y)
    lin_expected = linreg.predict(X)
    evaluate_df['expected_surp_linear'] = lin_expected
    evaluate_df['adjusted_surp_linear'] = y - lin_expected
    evaluate_df['adjusted_prob_linear'] = np.exp(-evaluate_df['adjusted_surp_linear'])

    gam, best_lam, lam_diagnostics = fit_regularized_gam(
        X,
        y,
        lam_values=np.logspace(-3, 2, 8),
        n_splines=10,
        n_splits=5,
        test_size=0.2,
        random_state=42,
        verbose=True,
    )
    # pd.DataFrame(lam_diagnostics).to_csv(
    #     'results/gam_lambda_diagnostics.tsv', sep='\t', index=False
    # )
    print(f"GAM lam selected from sweep: {best_lam:.4f}")

    gam_expected = gam.predict(X)
    evaluate_df['expected_surp_gam'] = gam_expected
    evaluate_df['adjusted_surp'] = y - gam_expected
    evaluate_df['adjusted_prob'] = np.exp(-evaluate_df['adjusted_surp'])

    # evaluate_df.to_csv(
    #     'results/large_homonym_minimal_pairs_with_models.tsv',
    #     sep='\t', index=False
    # )

    plot_gam_with_ci(evaluate_df, X, gam)
    plot_linear_with_ci(evaluate_df, X, linreg)
    plot_gam_vs_linear(evaluate_df, X, gam, linreg)

    print("All regression fits and plots completed.")


if __name__ == "__main__":
    main()
