# question_similarity_nlp_features

My share of the NLP features used in 
[Quora Question Pairs Kaggle competition](https://www.kaggle.com/c/quora-question-pairs),
where we ended up in the top 10% (339/3394), winning a bronze medal.

The processing is done by executing the
[`extract_nlp_features.py`](https://github.com/mbekavac/question_similarity_nlp_features/blob/master/extract_nlp_features.py)
script with two parameters: input dataset file path and the output file path.

```
python extract_nlp_features.py input/train_dataset.csv output/train_features.csv
```

Developed and used with Python 3.5.

Before running, take a look at the
[`utils/constants.py`](https://github.com/mbekavac/question_similarity_nlp_features/blob/master/utils/constants.py)
file and make changes if necessary.
Most of the code can be used with any type of dataset, but some parts had to be tightly coupled with the dataset used
in the competition.

