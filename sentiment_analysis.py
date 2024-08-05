
import praw
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT

reddit = praw.Reddit(client_id=REDDIT_CLIENT_ID, client_secret=REDDIT_CLIENT_SECRET, user_agent=REDDIT_USER_AGENT)


subreddit_names = ['learnpython', 'datascience', 'machinelearning', 'programming',
                   'technology', 'business', 'politics', 'news', 'science',
                   'gaming', 'movies', 'music', 'funny', 'aww', 'pics', 
                   'videos', 'askreddit', 'todayilearned', 'showerthoughts', 
                   'lifeprotips']


def collect_reddit_data(subreddit, limit=1000):
    posts = []
    subreddit_instance = reddit.subreddit(subreddit)
    for post in subreddit_instance.new(limit=limit):
        posts.append({'subreddit': subreddit, 'title': post.title, 'selftext': post.selftext})
    return posts


def analyze_sentiment(text):
    return TextBlob(text).sentiment.polarity 

def main():
    all_posts = []

 
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_subreddit = {executor.submit(collect_reddit_data, subreddit, 1000): subreddit for subreddit in subreddit_names}
        for future in as_completed(future_to_subreddit):
            subreddit = future_to_subreddit[future]
            try:
                posts = future.result()
                all_posts.extend(posts)
                print(f"Collected {len(posts)} posts from r/{subreddit}.")
            except Exception as e:
                print(f"Error collecting from r/{subreddit}: {e}")


    df_reddit = pd.DataFrame(all_posts)


    df_reddit['combined_text'] = df_reddit['title'] + ' ' + df_reddit['selftext']
    

    df_reddit['sentiment'] = df_reddit['combined_text'].apply(analyze_sentiment)
    
 
    df_reddit['sentiment_label'] = df_reddit['sentiment'].apply(lambda x: 'positive' if x > 0 else ('negative' if x < 0 else 'neutral'))
    
  
    plt.figure(figsize=(10, 6))
    df_reddit['sentiment_label'].value_counts().plot(kind='bar')
    plt.title('Sentiment Distribution of Reddit Posts')
    plt.xlabel('Sentiment Label')
    plt.ylabel('Count')
    plt.xticks(rotation=45) 
    plt.tight_layout() 
    plt.savefig('sentiment_distribution.png') 
    plt.show() 

  
    print(f"Collected {len(df_reddit)} posts from {len(subreddit_names)} subreddits.")

if __name__ == "__main__":
    main()
