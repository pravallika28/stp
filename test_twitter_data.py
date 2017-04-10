#!/usr/bin/env python
import get_twitter_data

## PLACE YOUR CREDENTIALS in config.json file or run this file with appropriate arguments from command line
keyword = 'google'
#time = 'since:2017-04-02 until:2017-04-09'
time = 'lastweek'
twitterData = get_twitter_data.TwitterData()
tweets = twitterData.getTwitterData(keyword, time)
print tweets
