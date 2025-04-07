# Set up MongoDB connection
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

# client = MongoClient("mongodb://localhost:27017")  # Make sure MongoDB is running locally

def connectMongoDB(uri="mongodb://localhost:27017"):
    # """Connects to MongoDB and returns the database object."""
    try:
        client = MongoClient(uri)
        db = client["twitterFakeDb"] 
         # Change to your database name
        print("Connected to MongoDB")
        return db
    except ConnectionFailure as e:
        print("MongoDB connection failed:", e)
        return None

uri="mongodb://localhost:27017"
client = MongoClient(uri)  
db = client["twitterFakeDb"]
userCollection = db['users']  # Users collection
tweetsCollection = db['tweets']  # Tweets collection
sentimentCollection=db["sentimentAnalysis"]