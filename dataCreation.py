from mongoConnect import userCollection,tweetsCollection
from modal import generateFakeTweet,generateFakeUser
import random

# Insert fake user data into MongoDB
def generateFakeUserData(num_users:int):
    users = []
    for _ in range(num_users):
        user = generateFakeUser()
        users.append(user)
    userCollection.insert_many(users)

# Insert fake tweet data into MongoDB
def generateFakeTweetData(num_tweets:int):
    # Fetch all users
    users = list(userCollection.find())
    for _ in range(num_tweets):
        user = random.choice(users)
        tweet = generateFakeTweet(user_id=user["_id"])
        tweetsCollection.insert_one(tweet)

# Generate and insert fake data
def generateData():
    # print("Inserting fake users into MongoDB...")
    # generateFakeUserData(num_users=100)  # Insert 100 fake users
    print("Inserting fake tweets into MongoDB...")
    generateFakeTweetData(num_tweets=1)  # Insert 100000 fake tweets?
