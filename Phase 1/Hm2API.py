import requests
import csv
import time

review = "This is be one of my favourite animations of all time. It's so mature and deep that you can't find any one like this. There are a lot of different themes which are depicted beautifully such as, adventure, oldness, boyhood, true love, idolization, escapism, nature, endangered animals, ... wrapped in a funny and adventurous piece of work with lavish beauty. Michael Giacchino did a wonderful job here. The way the theme was rearranged for different scenes was extraordinary. Its pace, edits and direction was excellent. So smooth and smart."

def get_sentiment(review):
    url = "https://twinword-sentiment-analysis.p.rapidapi.com/analyze/"
    headers = {
        "x-rapidapi-key": "7aa6af87ccmshe502cd24a962cc1p188c8fjsn6756dfac3fc2",
        "x-rapidapi-host": "twinword-sentiment-analysis.p.rapidapi.com"
    }
    querystring = {"text": review}
    
    for _ in range(3):
        try:
            response = requests.get(url, headers=headers, params=querystring, timeout=5)
            if response.status_code == 200:
                return response.json().get('type', 'unknown')
        except:
            print("Error, retrying...")
        time.sleep(2)
    return 'unknown'

def write_results(data):
    with open("results.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Review", "Sentiment"])
        writer.writerows(data)
    print("Results saved in results.csv")

sentiment = get_sentiment(review)
write_results([[review, sentiment]])
print(f"Review: {review}\n\nSentiment: {sentiment}")