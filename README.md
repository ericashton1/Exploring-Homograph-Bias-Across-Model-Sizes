# NLP-Final-Project-Exploring-Homonym-Bias-in-LMs


## Going from dataset to tsv format

First, we will create a list of 50 homonyms and 5 synonyms for each definition. We will then download the {open subtitles dataset}(https://github.com/LibrariesHacked/openlibrary-search}(https://huggingface.co/datasets/sentence-transformers/parallel-sentences-opensubtitles) from HuggingFace. Once we have loaded the dataset into python, we can iterate through all the English subtitles, checking each word to see if it is in a set of all the synonyms. If it is, we will add it to our dataset with the ROI (index at which the word was found) and add 1 to a counter which we can use as textid. We will repeat this process until we have 30,000 sentences. Once we have these 30,000 sentences, we can use a dictionary lookup to find another synonym that will replace the word in its pair and the appropriate homonym which will replace the word in its other pair. Then we can format these into a pd dataframe and then a tsv.


## Going from output of NLPScholar to evaluation metrics, tables, and figures

The evaluation of our NLPScholar experiment will give us probability of the original word in the sentence (word a), a non-homonym synonym (word b), and a homonym synonym (word c). We are concerned with the difference between word a and word b compared to the difference between word a and word c. Thus we can iterate through, gathering average difference as well as number of instances where the probability disparity between word a and b is less than the probability disparity between word a and word c (evidence of bias against homonyms). This will then allow us to chart difference between each set of two pairs and give us a percentage of pairs where non-homonyms were favored for each model. We can also create a chart with points plotted for difference between each set of two pairs color-coded by model to potentially show how training size affects homonym bias.

Additionally, we can use the output from NLPScholar to potentially show correlation to similarity between static and contextual embeddings. That is, we can plot each pair on a chart where one axis is the word's average difference between its static embedding and contextual embeddings, and the other axis is its bias towards/against contronyms, represented as the difference between the contronym and non-contronym probability for that word. Ideally, we would see a correlation such that as the difference between static and contextual embeddings increases, so does the bias against the homonym. 
