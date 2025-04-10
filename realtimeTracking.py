import json
from sentimentAnylize import processCrimeUpdate

def watchCrimeUpdate(db):
    print(db) 
    collection = db["tweets"] 
    pipeline = []  
    
    with collection.watch(pipeline) as stream:
        print("Watching for real-time crime updates...")
        for change in stream:
            print("New Change Detected:")
            print(json.dumps(change, indent=4, default=str))
            processCrimeUpdate(change)