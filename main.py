
import threading
from dataCreation import generateData
from efficiencyThreading import UpdatingTweets, crimeUpdate, sentimentUpdate
from graph import updateGraph, watchSentimentUpdate
from mongoConnect import connectMongoDB
from realtimeTracking import watchCrimeUpdate

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
    
        
        
