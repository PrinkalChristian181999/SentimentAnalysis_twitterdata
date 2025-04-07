from faker import Faker
import random
from datetime import datetime

# Initialize Faker instance to generate fake data
fake = Faker()

# Function to generate fake user data
def generateFakeUser():
    return {
        "username": fake.user_name(),
        "full_name": fake.name(),
        "email": fake.email(),
        "profile_picture": fake.image_url(),
        "bio": fake.sentence(),
        "location": fake.city(),
        "followers_count": random.randint(100, 10000),
        "following_count": random.randint(100, 1000),
        "join_date": fake.date_this_decade().isoformat(),
        "created_at": datetime.now().isoformat(),
    }

# Function to generate fake tweet data
def generateFakeTweet(user_id):
    return {
        "user_id": user_id,
        "tweet_content": fake.text(max_nb_chars=280),  # Max characters per tweet is 280
        "timestamp": fake.date_this_year().isoformat(),
        "likes_count": random.randint(0, 1000),
        "retweets_count": random.randint(0, 500),
        "replies_count": random.randint(0, 200),
        "hashtags": [fake.word() for _ in range(random.randint(0, 5))],
    }

