# Automatic classifier for commits which improve nonfunctional properties.

The reproduce file contains all code for using the classifier along with documentation for other classification techniques.


# Replication Guide

The Script reproduce/reproduce.py serves as a check for the data provided in this repository, as well as a use of the classifier. 
You can reproduce everything by calling `reproduce.py:run()` after data has been reproduced 

## How to reproduce the data

- commit messages must be added to the files found in the data directory data, they can be found by searching github or downloading the repo they are from and using the git log command.
A link to the commit can also be generated using the getCommitLink.py script.
- `ReproductionAndTraining.check()` valdiates the commit count

## How to reproduce the classifier
`ReproductionAndTraining.classifier()` which runs the default classifier as shown in the publication.

If you want to test the alternatives that we also attempted, you will have to uncomment the lines in the run() (lines 30 through 49). Individual classification of the groups unfortunately does not produce good results, this can be seen when uncommenting lines 61-86.

### How to test with different algorithms or settings:
See lines 263-265 for the algorithms. We support:
- Scikit-Learn - RandomForestClassifier; MultinomialNB; SGDClassifier
- Imblearn - BalancedRandomForestClassifier - which is designed for imbalanced data, but produces worse results than Scikit.

### How to test different settings:
We left only in the best setting in so the reproduction is fast. But you can add settings in lines 266-272. 

### How to autimatically mine new commits
Simply run Automine.py, the existing pickle files contain a pretrained classifier or they can be deleted and the classifier retrained.

Settings that were tested (bold is best):
| Setting            | Option                      |
|--------------------|-----------------------------|
| Word Preprocessing | **Lemmatization**, Stemming |
| Featurization      | **Bag of Words**, IDF       |
| n_estimators       | 100, 200, **300**, 400, 500 |
| max_depth          | 100, 1000, **none**         |
| bootstrap          | **false**, true             |
| max_features       | **auto**, sqrt, log2        |
| min_samples_leaf   | **1**,2,5                   |
| min_samples_split  | **2**,3,5                   |

