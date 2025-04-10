from collections import deque
from bson.json_util import dumps
from textblob import TextBlob
from cleaningPipeline import cleanText
from modelTrain import processCrimeUpdateToTrain
from mongoConnect import sentimentCollection

def processCrimeUpdate(change):
    if change["operationType"] == "insert":
        crimeRecord = change["fullDocument"]
        description = crimeRecord.get("tweet_content", "")
        cleanedDescription = cleanText(description)
        sentiment = analyzeSentiment(cleanedDescription)
        anylises = {
        "crime":crimeRecord,
        "cleanedDescription": cleanedDescription,
        "sentimentAnalysis": sentiment
        }
        sentimentCollection.insert_one(anylises)
        processCrimeUpdateToTrain(anylises)
        print("New crime reported anylises:",anylises)
        print("Watching for real-time crime updates...")
    elif change["operationType"] == "update":
        print("Crime record updated:", dumps(change, indent=4))
    elif change["operationType"] == "delete":
        print("Crime record deleted:", change["documentKey"])

def analyzeSentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    if sentiment > 0:
        return {
            "label": "Positive",
            "score": sentiment
            } 
    elif sentiment < 0:
        return{
            "label": "Negative",
            "score": sentiment
            }
    else:
        return{
            "label": "Neutral",
            "score": sentiment
            } 
    
