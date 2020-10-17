# Music-Mood-Recognition
Mood identification system which recognizes the mood of a bollywood song using multiple machine learning models.
The dataset of songs annotated with moods was not available online so we created one, consisting about 400 songs.
Our dataset consists of 3 folders namely cheerful, melancholy, lighthearted consisting of songs of these 3 moods.

*Entire project has been written using python3.*  
### We have mainly used:  
**librosa** : library for feature extraction purpose.

**pydub** : for audio segmentation and wav format conversion.

**sklearn** : for classification models(training and testing).

**matplotlib** : for visualization purpose.  

**Tkinter** : for User Interface.

Four classifiers have been used in this project namely Randomforest, DecisionTree, SVM, and naive bayes.

After training on our curated datasets we have reached upto 65 % accuracy on individual models. which on combining the results would be altogether increased. 
