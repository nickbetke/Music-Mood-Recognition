# Music-Mood-Recognition
mood identification system which recognizes the mood of a bollywood song using machine learning models
the dataset of songs annotated with moods waas not available online so we created one. consisting about 400 songs.
our dataset consists of 4 folders namely cheerful, melancholy, lighthearted consisting of songs of these 3 moods.

We have used:  
**librosa** : library for feature extraction purpose.

**pydub** : for audio segmentation and wav format conversion.

**sklearn** : for classification models(training and testing).

**matplotlib** : for visualization purpose.

Four classifiers have been used in this project namely Randomforest, DecisionTree, SVM, and naive bayes.

After training on our curated datasets we have reached upto 65 % accuracy on individual models. which on combining the results would be altogether increased. 
