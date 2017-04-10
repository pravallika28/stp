#import regex
import re
import csv
import pprint
import nltk.classify

#start replaceTwoOrMore
def replaceTwoOrMore(s):
    #look for 2 or more repetitions of character
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL) 
    return pattern.sub(r"\1\1", s)
#end

#start process_tweet
def processTweet(tweet):
    # process the tweets
    
    #Convert to lower case
    tweet = tweet.lower()
    #Convert www.* or https?://* to URL
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)
    #Convert @username to AT_USER
    tweet = re.sub('@[^\s]+','AT_USER',tweet)    
    #Remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)
    #Replace #word with word
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    #trim
    tweet = tweet.strip('\'"')
    #print 'TWEET:' + tweet
    return tweet
#end 

#start getStopWordList
def getStopWordList(stopWordListFileName):
    #read the stopwords
    stopWords = []
    stopWords.append('AT_USER')
    stopWords.append('URL')

    fp = open(stopWordListFileName, 'r')
    line = fp.readline()
    while line:
        word = line.strip()
        stopWords.append(word)
        line = fp.readline()
    fp.close()
    return stopWords
#end

#start getfeatureVector
def getFeatureVector(tweet, stopWords):
    featureVector = []  
    words = tweet.split()
    for w in words:
        #replace two or more with two occurrences 
        w = replaceTwoOrMore(w) 
        #strip punctuation
        w = w.strip('\'"?,.')
        #check if it consists of only words
        val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*[a-zA-Z]+[a-zA-Z0-9]*$", w)
        #ignore if it is a stopWord
        if(w in stopWords or val is None):
            continue
        else:
	    #print 'W:' + w.lower()
            featureVector.append(w.lower())
    #print  featureVector
    return featureVector    
#end

#start extract_features
def extract_features(tweet):
    tweet_words = set(tweet)
    features = {}
    for word in featureList:
        features['contains(%s)' % word] = (word in tweet_words)
    return features
#end

Tweets = csv.reader(open('data/full_training_dataset.csv', 'rb'), delimiter=',', quotechar='"')
#for tweet in Tweets:
#   print tweet

#Read the tweets one by one and process it
inpTweets = csv.reader(open('data/sampleTweets.csv', 'rb'), delimiter=',', quotechar='"')
#print inpTweets
stopWords = getStopWordList('data/feature_list/stopwords.txt')
count = 0;
featureList = []
tweets = []
for row in inpTweets:
    print row
    sentiment = row[0]
    tweet = row[1]
    processedTweet = processTweet(tweet)
    featureVector = getFeatureVector(processedTweet, stopWords)
    featureList.extend(featureVector)
    tweets.append((featureVector, sentiment));
#end loop
print '.....\nTRAINING DONE ! \n......'
# Remove featureList duplicates
featureList = list(set(featureList))
print 'featureList made'
# Generate the training set
training_set = nltk.classify.util.apply_features(extract_features, tweets)
print 'training set generated\n'
# Train the Naive Bayes classifier
NBClassifier = nltk.NaiveBayesClassifier.train(training_set)
print 'Bayes trained\n'
# Test the classifier
correct=0;
incorrect=0;
inpTestTweets = csv.reader(open('data/test/training_neatfile_2.csv', 'rb'), delimiter=',', quotechar='"')
#print inpTweets
stopWords = getStopWordList('data/feature_list/stopwords.txt')
count = 0;
featureList = []
tweets = []
for row in inpTestTweets:
    print row
    sentiment = row[0]
    tweet = row[1]
    proctweet = processTweet(tweet)
    senti = NBClassifier.classify(extract_features(getFeatureVector(proctweet, stopWords)))
    print "testTweet = %s, sentiment = %s \n" % (tweet, sentiment)
    if (sentiment == senti or sentiment=='neutral'):
        print " 1 "
        correct=correct+1;
    else: 
        print " 0 " 
        incorrect= incorrect+1;
print "correct = %d , incorrect = %d\n" %(incorrect, correct)

testTweet = 'Congrats @upk, i heard you won the competition'
processedTestTweet = processTweet(testTweet)
sentiment = NBClassifier.classify(extract_features(getFeatureVector(processedTestTweet, stopWords)))
print "testTweet = %s, sentiment = %s\n" % (testTweet, sentiment)
