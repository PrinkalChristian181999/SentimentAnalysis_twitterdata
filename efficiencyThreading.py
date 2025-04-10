import time
from dataCreation import generateData
from graph import watchSentimentUpdate
from mongoConnect import connectMongoDB
from realtimeTracking import watchCrimeUpdate


def crimeUpdate():
    db = connectMongoDB()
    try:
        print(db)
        watchCrimeUpdate(db)
    except Exception as e:
        print("error occured while performing this action:", e)

def UpdatingTweets():
    time.sleep(5)
    for i in range(1):
        generateData()
        time.sleep(1)
    
def sentimentUpdate():
    db=connectMongoDB()
    try:
        watchSentimentUpdate(db)
    except Exception as e:
        print("error occured while performing this action:", e)