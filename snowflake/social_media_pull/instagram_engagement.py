import os
import pandas as pd
from dotenv import load_dotenv
from instagrapi import Client
from datetime import datetime

load_dotenv()  # reads your .env file

USERNAME = os.getenv("INSTAGRAM_USERNAME")
PASSWORD = os.getenv("INSTAGRAM_PASSWORD")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SESSION_FILE = os.path.join(SCRIPT_DIR, "session.json")

ig = Client()

# reuse session if available to avoid triggering rate limits
if os.path.exists(SESSION_FILE):
    ig.load_settings(SESSION_FILE)
    ig.login(USERNAME, PASSWORD)
else:
    ig.login(USERNAME, PASSWORD)
    ig.dump_settings(SESSION_FILE)

# get my profile information
me = ig.user_info(ig.user_id)

print(f"Logged in as: {me.username}")
print(f"Followers: {me.follower_count}")

PHARMAPLUS_HANDLE = "pharmaplus_kenya"
PHARMAPLUS_ID_FILE = os.path.join(SCRIPT_DIR, "pharmaplus_user_id.txt")

# cache the user_id to avoid hitting the public API every run
if os.path.exists(PHARMAPLUS_ID_FILE):
    user_id = int(open(PHARMAPLUS_ID_FILE).read().strip())
else:
    user_id = ig.user_id_from_username(PHARMAPLUS_HANDLE)
    open(PHARMAPLUS_ID_FILE, "w").write(str(user_id))

print(f"The user_id:{user_id}")


# Fetch the last 200 posts
posts = ig.user_medias(user_id, amount=50)

print(f"Found {len(posts)} posts")

# Loop through each post and extract fields
rows = []
for post in posts:
    views = getattr(post, "view_count", 0) or 0
    likes = post.like_count or 0
    comments = post.comment_count or 0

    # Engagement rate = interactions / views × 100
    eng_rate = round(
        (likes + comments) / max(views, 1) * 100, 2
    )

    rows.append({
        "platform"        : "Instagram",
        "post_id"         : post.id,
        "type"            : post.media_type,  # 1=photo 2=video 8=carousel
        "caption"         : (post.caption_text or "")[:200],
        "published_at"    : post.taken_at,
        "likes"           : likes,
        "comments"        : comments,
        "views"           : views,
        "engagement_rate" : eng_rate,
        "url"             : f"https://instagram.com/p/{post.code}/",
    })



# Convert to a table
df = pd.DataFrame(rows)

# Add today's date to the filename
today = datetime.now().strftime("%Y-%m-%d")
filename = f"pharmaplus_instagram_{today}.csv"

df.to_csv(filename, index=False)
print(f"Saved {len(df)} posts to {filename}")

# Preview the top 5 rows
print(df[["published_at","likes","comments","engagement_rate"]].head())

# Quick stats
print("--- Summary ---")
print(f"Avg likes:       {df['likes'].mean():.0f}")
print(f"Avg comments:    {df['comments'].mean():.0f}")
print(f"Avg engagement:  {df['engagement_rate'].mean():.2f}%")
print(f"Best post likes: {df['likes'].max()}")