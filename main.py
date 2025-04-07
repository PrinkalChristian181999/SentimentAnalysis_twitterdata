
import threading
from efficiencyThreading import UpdatingTweets, crimeUpdate, sentimentUpdate
from graph import updateGraph

if __name__ == "__main__":
    updateGraph()
    # Start data generation and watchers
    crimeThread = threading.Thread(target=crimeUpdate)
    tweetThread = threading.Thread(target=UpdatingTweets)
    sentimentThread = threading.Thread(target=sentimentUpdate)

    crimeThread.start()
    tweetThread.start()
    sentimentThread.start()


    crimeThread.join()
    tweetThread.join()
    sentimentThread.join()
    
        
        
