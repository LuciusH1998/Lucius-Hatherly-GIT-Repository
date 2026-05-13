"""
First make sure you are in the client folder
You can directly run this file via:
python test_gclient.py
"""

from game_client import GameClient
import time

# Pick any REAL game_id from past seasons
TEST_GAME_ID = "2023020008"  

gclient = GameClient(TEST_GAME_ID)

print("\n First Extraction run (should fetch MANY events)")
df1 = gclient.extract()
print(df1.head())
print("Rows fetched:", len(df1))

# Show new engineered columns
print("\n Columns returned")
print(df1.columns.tolist())

# Validate required feature columns exist
required_cols = ["distance_from_net", "angle_from_net", "is_goal", "empty_net"]
missing = [c for c in required_cols if c not in df1.columns]

if missing:
    print("\n ERROR: Missing required feature columns:", missing)
else:
    print("\n All required feature columns found:", required_cols)

# Second Extraction
print("\n Second Extraction run (should fetch ZERO new events) ")
time.sleep(2)  # simulate live poll
df2 = gclient.extract()
print(df2.head())
print("Rows fetched:", len(df2))

# Third Extraction
print("\n Third Extraction run (should fetch ZERO again) ")
time.sleep(3)
df3 = gclient.extract()
print("Rows fetched:", len(df3))

# Quick checks for expected behavior
if len(df1) > 0 and len(df2) == 0 and len(df3) == 0:
    print("\n Success: GameClient unseen-event filtering behaviour works correctly!")
else:
    print("\n Failure: Unexpected extraction behavior — check unseen-event logic.")

