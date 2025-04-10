import re

def cleanText(text):
    # """Cleanses crime description text by removing special characters and extra spaces."""
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    text = re.sub(r'http\S+', '', text) #remove urls
    text = re.sub(r'@\w+', '', text) #remove @mentions
    text = re.sub(r'#\w+', '', text)   #remove hashtags
    return text