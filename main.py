import time
import threading
from dataCreation import generateData
from graph import updateGraph, watchSentimentUpdate
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
    for i in range(100):
        generateData()
        time.sleep(1)
    
def sentimentUpdate():
    db=connectMongoDB()
    try:
        watchSentimentUpdate(db)
    except Exception as e:
        print("error occured while performing this action:", e)


crimeThread = threading.Thread(target=crimeUpdate)
tweetThread= threading.Thread(target=UpdatingTweets)
sentimentThread=threading.Thread(target=sentimentUpdate)
# graphThread=threading.Thread(target=updateGraph)

if __name__ == "__main__":
  # Start data generation and watchers
    crimeThread = threading.Thread(target=crimeUpdate)
    tweetThread = threading.Thread(target=UpdatingTweets)
    sentimentThread = threading.Thread(target=sentimentUpdate)

    crimeThread.start()
    tweetThread.start()
    sentimentThread.start()

    # Start the graph (MAIN THREAD)
    updateGraph()

    crimeThread.join()
    tweetThread.join()
    sentimentThread.join()
    
        
        
