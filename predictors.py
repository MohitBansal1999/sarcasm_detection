from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.naive_bayes import MultinomialNB ,GaussianNB, BernoulliNB



bnb=joblib.load('bnb.pkl')

tf=joblib.load('tf.pkl')

def head_line(headline):
    review = re.sub('[^a-zA-Z]', ' ', headline)
    review = review.lower()
    review = review.split()
    
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review]
    review = ' '.join(review)
    review=[review]
    
    test=tf.transform(review).toarray()
    
    labels_pred = bnb.predict(test)
    if labels_pred==1:
        return "Sarcasm"
    else:
        return 'no sarcasm'
