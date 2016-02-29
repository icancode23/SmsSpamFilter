# SMS Spam Detector
Written in Python 2.7 using Naive Bayes and SVM for Classifier

# How to run
```
$ python sms-spam-detector.py -m "Hello, dude. what's up?"
$ This message is predicted by SVM as ham and Naive Bayes as ham
$ python sms-spam-detector.py -m "WINNER! Credit for free"
$ This message is predicted by SVM as spam and Naive Bayes as spam
```

# TO DO
Don't load the models everytime the program get executed

# Credits
Dataset : https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection

Some codes are adapted from : http://radimrehurek.com/data_science_python/
