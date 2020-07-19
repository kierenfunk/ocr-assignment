# OCR assignment report

## Feature Extraction (Max 200 Words)

[The feature extraction approach uses a basic PCA algorithm, taking the 10 axes with the greatest range. PCA was the recommended approach due to consistently producing the best results. A basic feature extraction approach was used as significant computation time is used in classification and error correction. In order to improve feature extraction, components classification and error correction would need to be removed, this would likely decrease performance.]

## Classifier (Max 200 Words)

[The classifier uses a nearest neighbour approach, taking the closest proximal neighbour to the data. Other variants of nearest neighbour were tried including k-nearest neighbours where k=7 to k=25. In general, pages with lower noise perform worse as k gets larger, and pages with higher noise perform better as k gets larger. However, k nearest neighbours does not take into consideration proximity. A harmonic series weightings scheme was applied to attempt to improve on k nearest neighbours. However, the standard nearest neighbour outperformed k nearest neighbours even with harmonic series weightings. Although, there was a difference between harmonic k-NN to standard k-NN such that low noise pages performed better, but high noise pages performed worse. Standard nearest neighbour was chosen but 25 nearest neighbours was stored in the model dictionary for the error correction stage. This gave the best overall performance, because even though high noise pages performed worse in the classification stage, this was mitigated by a well performing error correction stage.]

## Error Correction (Max 200 Words)

[The error correction stage is made up of two components, a Markov transition probability estimation and a word lookup estimation. First, spaces were calculated using box sizes, a space between -20 and 6 pixels inclusive indicated no space between characters, this gave the best results although not with 100% accuracy. The Markov estimation stage uses a 53 by 53 matrix where the rows and columns correspond to letters of both lower case and upper case and the last row/column represents a space. The m,n entry in the matrix is the probability that the nth character follows the mth character in a word. The probabilities are combined with the probabilities in the k-nearest neighbour output so only characters that are feasible are considered. It would make more sense to include this component in the classification stage, although the box sizes to calculate spaces were not available in the classification stage. 
The word lookup is simpler. If a word is not recognised, words with the closest levenshtein distance are found and ranked according to the frequency in the English language. To improve error correction, combining word lookup with Markov transition would be preferable, however this becomes increasingly complicated.]

## Performance

The percentage errors (to 1 decimal place) for the development data are
as follows:

- Page 1: [Insert percentage here, e.g. 97.2%]
- Page 2: [Insert percentage here, e.g. 98.3%]
- Page 3: [Insert percentage here, e.g. 91.5%]
- Page 4: [Insert percentage here, e.g. 77.1%]
- Page 5: [Insert percentage here, e.g. 54.8%]
- Page 6: [Insert percentage here, e.g. 42.7%]

## Other information (Optional, Max 100 words)

[The data used in error correction had a significant effect on the final results. Markov transition probabilities were calculated using the same list of words that the word lookup uses. Originally, Markov probabilities were only estimated using lowercase letters, when introducing uppercase and spaces, there was a significant increase in performance.
More performance was achieved when probabilities were calculated with frequency in consideration. An estimation of frequency using Zipf’s law was used. Where the most frequent word is twice as frequent as the second most frequent word and three times as frequent as the third most frequent word etc.]
