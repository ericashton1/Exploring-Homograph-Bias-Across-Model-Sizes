# NLP-Final-Project-Exploring-Homograph-Bias-Across-Model-Sizes

## Overview 

Homographs are words that are spelled the same with categorically different meanings. We investigate if BERT and RoBERTa treat homographs differently from their non-homograph synonyms in a minimal pair experiment. Specifically, we look at frequency-adjusted surprisal to see if homographs tend to be more or less surprising than non-homographs and if this varies across model sizes. Both models showed significantly lower surprisal for homographs over their non-homograph synonyms, though this bias was smaller in RoBERTa, the model with a larger training size. These results suggest that as training size increases, models become more familiar with the more subtle contextual variance of non-homographs and thus exhibit a narrower surprisal difference between homographs and non-homographs. Thus, our study implies that models require less training to adequately learn the various contexts of a homograph than to adequately learn the more subtle varying contexts of a non-homograph.


## Experimental setup

First, homographs with meanings in the same part of speech and their non-homographic synonyms were selected to create a minimal pair experiment. The following is an example of the homograph “run”, and two non-homograph synonyms for each meaning:

Run: ([manage, direct] and [sprint, rush])

For each meaning, 3 sets of sentences were created using generative AI and revised by our team to ensure all minimally different words made sense in the context. These were then used to create pairs like the following:

She was hired to run the new marketing department.
She was hired to manage the new marketing department.

The final dataset consists of 34 homographs which yields 680 pairs being tested. With this data, we ran [NLPScholar’s](https://github.com/forrestdavis/NLPScholar/tree/main) Minimal Pair mode with BERT and RoBERTa to get the surprisals of the minimally different words.

An important consideration for the experiment was removing frequency bias, as a model is naturally less surprised by words that appear more frequently in its training data. To account for this, we calculated adjusted surprisal in two ways: one by fitting a linear regression line of surprisal against COCA frequency of each word in the data and another using GAMs, motivated by Smith & Levy et al. (2013) and Rezaii et al. (2023). In both cases, we subtracted the point on the fit line from the original surprisal to get surprisal adjusted for frequency.



## Going from output of NLPScholar to evaluation metrics, tables, and figures

The evaluation of our NLPScholar experiment will give us surprisal of the homograph, and each of its synonyms in the same context. To get our evluation metric, difference in adjusted surprisal, we ditted a regression line against COCA frequency using GAMs. By subtracting the point on the fit line from the original surprisal of each of our words of interest, we got frequency adjusted surprisal. Subtracting the frequency adjusted surprisal of the non-homograph from that of the homograph gave us our evaluation metric: difference in frequency adjusted surprisal between homographs and non-homographs. This is a good measure of "bias" towards/against homographs because it measures how surprised a model is in its placement of the word.


## Results and Disucussion

Both models showed a significant bias towards homographs in the minimal pair experiment. This bias decreased significantly as model size increased. 

These results suggest that the vastly different contexts associated with a homograph’s meanings are what produce the overall bias toward homographs, as well as the stronger bias observed in smaller models. In other words, a language model is likely to be more confident in placing a homograph rather than a non-homograph in our experiment because the homographs have explicitly different contexts in which they belong which are easily recognized. Non-homographs, however, have subtly different contexts in which they belong. These differences require much more training data in order for a model to gain confidence in its predictions.

It is important to note that these conclusions are not absolute as our study is limited in scope. We only looked at two different masked models, meaning perhaps the trend of an increased training size resulting in a decreased bias towards homographs is not always true. Additionally, our assumptions on why this trend seems to exist are not definitive, though our results do suggest that at the very least the relationship between language models and homographs is an interesting area of study.


## References

Smith, N. J., & Levy, R. (2013). The effect of word predictability on reading time is logarithmic. Cognition, 128(3), 302–319. https://doi.org/10.1016/j.cognition.2013.02.013

Rezaii, N., Michaelov, J., Josephy-Hernandez, S., Ren, B., et al. (2023). Measuring sentence information via surprisal: Theoretical and clinical implications in nonfluent aphasia. Annals of Neurology, 94(4), 647–657. DOI: 10.1002/ana.26744.

Wilcox, E. G., et al. (2023). Testing the predictions of surprisal theory in 11 languages. Transactions of the Association for Computational Linguistics (TACL).



Additionally, if we have time and are still interested in embeddings, we can use the output from NLPScholar to potentially show correlation to similarity between static and contextual embeddings. That is, we can plot each pair on a chart where one axis is the word's average difference between its static embedding and contextual embeddings, and the other axis is its bias towards/against contronyms, represented as the difference between the contronym and non-contronym probability for that word. Ideally, we would see a correlation such that as the difference between static and contextual embeddings increases, the bias against the homonym decreases. 
