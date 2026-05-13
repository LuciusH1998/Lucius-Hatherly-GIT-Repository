# Importing relevant libraries
import requests 
import pandas as pd 
import logging 
import numpy as np
from typing import Dict, List, Optional

# Defining Logger 
logger = logging.getLogger(__name__)

# Defining Base_URL derived from Data Acquisition in Milestone 1 and 2 
BASE_URL = "https://api-web.nhle.com/v1/gamecenter"


# Defining game client class
class GameClient:
    """
    A lightweight client for processing events for live NHL games via game_id

    This class will be able to:
    -Fetch JSON for a given game_id 
    -Extracting the relevant plays 
    -Monitoring plays which are already processed 
    -Return only new unseen events in the form of a tidy dataframe
    """

    # Defining constructor method 
    def __init__(self, game_id: str):
        """
        game_id examples:
            Regular season game_id: 202302001
            Playoff gamed_id: 2023030112
        """
        # Defining nhl game_id
        self.game_id = game_id
        # Defining nhl url 
        self.url_api = f"{BASE_URL}/{game_id}/play-by-play"
        # Defining already seen_event_id
        self.already_seen_event_ids: set = set()
        logger.info(f"[GameClient] established for game_id={game_id}")
    
    # Fetching game data through a json file 
    def obtain_game(self) -> Optional[Dict]:
        """Obtaining raw json NHL data, it will return none if nothing is found"""
        try:
            # Request given url with specific game id and return json file 
            r = requests.get(self.url_api)
            r.raise_for_status()
            return r.json()
            # Raise an Exception if there is an error obtaining JSON game 
        except Exception as e:
            logger.error(f"[GameClient] has experienced an error obtaining JSON game.")
            return None
    
    def extract_new_events(self, data: Dict) -> pd.DataFrame:
        """This function extracts only unseen plays and will add new event id to these plays"""
        # If data is None or play not in data return pd.Dataframe
        if data is None or "plays" not in data:
            return pd.DataFrame()
        
        # Defining valid plays
        valid_plays = data["plays"]
        # If not valid plays, return DataFrame 
        if not isinstance(valid_plays, list):
            return pd.DataFrame() 
        # Defining data frame by json normalizing valid plays 
        df = pd.json_normalize(valid_plays)
        
        # Return df if eventId is not in df.columns()
        if "eventId" not in df.columns:
            return pd.DataFrame()
        
        # Keeping only unseen events 
        # Defining eventId as integer 
        df["eventId"] = df["eventId"].astype(int)
        updated_df = df[~df["eventId"].isin(self.already_seen_event_ids)]

        # Update Tracker 
        new_ids = set(updated_df["eventId"].tolist())
        self.already_seen_event_ids.update(new_ids)
        
        # Putting information into the log
        logger.info(f"[GameClient] {len(updated_df)} has a new event extracted.")

        return updated_df
    
    @staticmethod
    def compute_distance_angle(x, y):
        """
        Net assumed at (89, 0) like Milestone 2.
        Method is identical to Milestone 2. 
        """
        x_net, y_net = 89, 0
        dx = x_net - x
        dy = y_net - y
        dist = np.sqrt(dx ** 2 + dy ** 2)
        ang = np.degrees(np.arctan2(abs(dy), abs(dx)))
        return dist, ang
    
    def process_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes:
        - standardized x,y
        - distance_from_net
        - angle_from_net
        - is_goal
        - empty_net
        Method is identical to Milestone 2. 
        """

        if df.empty:
            return df
        
        # Correct IDs will be attached in extract() below.

        # Standardize direction
        df["x_std"] = np.where(df["details.xCoord"] < 0,
                              -df["details.xCoord"],
                               df["details.xCoord"])

        df["y_std"] = np.where(df["details.xCoord"] < 0,
                              -df["details.yCoord"],
                               df["details.yCoord"])

        # Compute distance + angle
        df["distance_from_net"] = None
        df["angle_from_net"] = None

        for idx, row in df.iterrows():
            dist, ang = self.compute_distance_angle(row["x_std"], row["y_std"])
            df.at[idx, "distance_from_net"] = dist
            df.at[idx, "angle_from_net"] = ang

        # is_goal
        df["is_goal"] = (df["typeDescKey"] == "goal").astype(int)

        # empty net (use situationCode)
        def compute_empty(row):
            code = row.get("details.situationCode", None)
            owner = row.get("eventOwnerTeamId", None)

            # assume missing code => 0
            if pd.isna(code) or not isinstance(code, str) or len(code) != 4:
                return 0

            away_goalie = int(code[0])
            home_goalie = int(code[3])

            # if the scoring team has NO goalie on ice, this is an empty net goal
            # eventOwnerTeamId == home team, then check home_goalie
            if row["is_goal"] == 1:
                if owner == row.get("homeTeamId") and home_goalie == 0:
                    return 1
                if owner == row.get("awayTeamId") and away_goalie == 0:
                    return 1
            return 0

        df["empty_net"] = df.apply(compute_empty, axis=1)

        return df

    def extract(self) -> pd.DataFrame:
        """
        Get unseen events, attach the relevant team IDs, and then compute features
        """
        # Defining Data with obtain_game
        data = self.obtain_game()
        # Extracting new events and assigning it to unseen
        unseen = self.extract_new_events(data)

        if unseen.empty:
            return unseen

        # Attach home/away team IDs
        home_id = data.get("homeTeam", {}).get("id", None)
        away_id = data.get("awayTeam", {}).get("id", None)

        unseen["homeTeamId"] = home_id
        unseen["awayTeamId"] = away_id

        # Attach home/away team names
        home_name = data.get("homeTeam", {}).get("commonName", {}).get("default", None)
        away_name = data.get("awayTeam", {}).get("commonName", {}).get("default", None)

        unseen["homeTeamName"] = home_name
        unseen["awayTeamName"] = away_name

        # Attach home/away team logos
        home_logo = data.get("homeTeam", {}).get("logo", None)
        away_logo = data.get("awayTeam", {}).get("logo", None)

        unseen["homeTeamLogo"] = home_logo
        unseen["awayTeamLogo"] = away_logo

        # Creating the necessary features 
        processed = self.process_events(unseen)
        return processed
