import streamlit as st
import requests
from ift6758.client.game_client import GameClient
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from PIL import Image


st.title("Hockey Visualization App")
    
## Sidebar
with st.sidebar:
    st.header("Model Registry")
    # inputs
    workspace = st.text_input("Workspace", value="IFT6758-2025-B08")
    model_name = st.selectbox(
                "Which model would you like to use?",
                ("logreg_distance", "logreg_angle", "logreg_distance_angle"),
                index=None,
                placeholder="Select model...",
                key="model_name"
            )
    if st.session_state["model_name"] in ["logreg_distance", "logreg_angle", "logreg_distance_angle"]:
        model_version = st.selectbox(
                "Which model version would you like to use?",
                tuple(f"v{i}" for i in range(8)) + ("latest",),
                index=None,
                placeholder="Select model version...",
                key="model_version"
            )
    else:
        model_version = st.selectbox(
                "Which model version would you like to use?",
                ("latest",),
                index=None,
                placeholder="Select model version...",
                key="model_version"
            )

    if st.button("Get model"):
        if not workspace or not model_name or not model_version:
            # Clear the history of the previous run
            for key in ["homeTeamId", "awayTeamId", "df", "new_df", "teams", "eventID"]:
                st.session_state.pop(key, None)
            st.error("Please fill out all fields before loading a model.")
            st.stop()

        try:
            # Call serving container to download model
            response = requests.post(
                "http://serving:5000/download_registry_model",
                # "http://127.0.0.1:5000/download_registry_model",
                json={
                    "model": model_name,
                    "version": model_version
                }
            )
            if response.status_code == 200:
                st.success(f" Model {model_name}:{model_version} loaded successfully!")

                # Choosing the correct features to use when making predictions
                if model_name == "logreg_distance":
                    features = ["distance_from_net"]
                elif model_name == "logreg_angle":
                    features = ["angle_from_net"]
                else:
                    features = ["distance_from_net", "angle_from_net"]
                st.session_state["features"] = features

            else:
                st.error(f" Failed to load model: {response.text}")
        except Exception as e:
            st.error(f" Error connecting to serving container: {e}")

        # Clear the history of the previous run
        for key in ["homeTeamId", "awayTeamId", "df", "new_df", "teams", "eventID"]:
            st.session_state.pop(key, None)


## Game ID input
with st.container():
    game_id = str(st.text_input("Game ID", key="game_id"))


## Game info and predictions
with st.container():
    if st.button("Ping game"):
        if game_id:
            if not workspace or not model_name or not model_version:
                st.error("Please choose a model before pinging the game!")
                st.stop()
            else: 
                gclient = GameClient(game_id)
                df = gclient.extract()
                st.session_state["homeTeamId"], st.session_state["awayTeamId"] = df["homeTeamId"].drop_duplicates().values[0], df["awayTeamId"].drop_duplicates().values[0]
                try:
                    # Create test data
                    st.session_state['df']=df.copy()
                    st.session_state['df'].dropna(subset=st.session_state["features"], inplace=True)
                    
                    # Send to serving container
                    response = requests.post(
                        "http://serving:5000/predict",
                        # "http://127.0.0.1:5000/predict",
                        json=st.session_state['df'][st.session_state["features"]].to_dict(orient="list")
                    )
                    
                    if response.status_code == 200:
                        predictions = response.json()["predictions"]
                        st.session_state['df']["model_predictions"] = response.json()["predictions"]
                        st.session_state["teams"] = df[["homeTeamName", "awayTeamName"]].drop_duplicates().values[0]
                        st.success(f"Loaded game succesfully!")
                    else:
                        st.error(f"Prediction failed: {response.text}")

                except Exception as e:
                    st.error(f"Error: {e}")

        else:
            st.warning("Please enter a game ID.")

with st.container():
    if "teams" in st.session_state:
        st.subheader(f"Game {game_id}: {st.session_state['teams'][0]} vs. {st.session_state['teams'][1]}")
        eventID = st.selectbox(
            "Which event ID would you like to consider in the chosen game?",
            sorted(st.session_state['df']["eventId"]),
            index=None,
            placeholder="Select event ID...",
            key="eventID"
        )

    if st.session_state.get("eventID") is not None:
        st.session_state['df'] = st.session_state['df'].reset_index(drop=True)
        event_idx = st.session_state['df'].loc[st.session_state['df']["eventId"] == st.session_state["eventID"]].index[0]
        st.session_state["new_df"] = st.session_state['df'].iloc[:event_idx+1]
        data = st.session_state["new_df"]
        st.text(f"Period {data.iloc[event_idx]['periodDescriptor.number']} - {data.iloc[event_idx]['timeRemaining']} left")

        col1, col2 = st.columns(2)
        with col1:
            st.image(data.iloc[0]["homeTeamLogo"], width="stretch") # setting width='content' or it's default value does not seem to work
            xG_home = data.loc[data["details.eventOwnerTeamId"] == st.session_state["homeTeamId"]]["model_predictions"].sum().round(2)
            goals_home = data.loc[data["details.eventOwnerTeamId"] == st.session_state["homeTeamId"]]["is_goal"].sum()
            st.metric(f"{st.session_state['teams'][0]} xG (actual)", f"{xG_home} ({goals_home})", str((xG_home-goals_home).round(2)), delta_color="off")
            st.write(f"Shots on goal: {data.loc[data['details.eventOwnerTeamId'] == st.session_state['homeTeamId']]['typeDescKey'].value_counts()['shot-on-goal'] if 'shot-on-goal' in data.loc[data['details.eventOwnerTeamId'] == st.session_state['homeTeamId']]['typeDescKey'].value_counts().keys() else 0}")
            st.write(f"Blocked shots: {data.loc[data['details.eventOwnerTeamId'] == st.session_state['homeTeamId']]['typeDescKey'].value_counts()['blocked-shot'] if 'blocked-shot' in data.loc[data['details.eventOwnerTeamId'] == st.session_state['homeTeamId']]['typeDescKey'].value_counts().keys() else 0}")
            st.write(f"Missed shots: {data.loc[data['details.eventOwnerTeamId'] == st.session_state['homeTeamId']]['typeDescKey'].value_counts()['missed-shot'] if 'missed-shot' in data.loc[data['details.eventOwnerTeamId'] == st.session_state['homeTeamId']]['typeDescKey'].value_counts().keys() else 0}")
            st.write(f"Penalties: {data.loc[data['details.eventOwnerTeamId'] == st.session_state['homeTeamId']]['typeDescKey'].value_counts()['penalty'] if 'penalty' in data.loc[data['details.eventOwnerTeamId'] == st.session_state['homeTeamId']]['typeDescKey'].value_counts().keys() else 0}")

        with col2:
            st.image(data.iloc[0]["awayTeamLogo"], width="stretch")
            xG_away = data.loc[data["details.eventOwnerTeamId"] == st.session_state["awayTeamId"]]["model_predictions"].sum().round(2)
            goals_away = data.loc[data["details.eventOwnerTeamId"] == st.session_state["awayTeamId"]]["is_goal"].sum()
            st.metric(f"{st.session_state['teams'][1]} xG (actual)", f"{xG_away} ({goals_away})", str((xG_away-goals_away).round(2)), delta_color="off")
            st.write(f"Shots on goal: {data.loc[data['details.eventOwnerTeamId'] == st.session_state['awayTeamId']]['typeDescKey'].value_counts()['shot-on-goal'] if 'shot-on-goal' in data.loc[data['details.eventOwnerTeamId'] == st.session_state['awayTeamId']]['typeDescKey'].value_counts().keys() else 0}")
            st.write(f"Blocked shots: {data.loc[data['details.eventOwnerTeamId'] == st.session_state['awayTeamId']]['typeDescKey'].value_counts()['blocked-shot'] if 'blocked-shot' in data.loc[data['details.eventOwnerTeamId'] == st.session_state['awayTeamId']]['typeDescKey'].value_counts().keys() else 0}")
            st.write(f"Missed shots: {data.loc[data['details.eventOwnerTeamId'] == st.session_state['awayTeamId']]['typeDescKey'].value_counts()['missed-shot'] if 'missed-shot' in data.loc[data['details.eventOwnerTeamId'] == st.session_state['awayTeamId']]['typeDescKey'].value_counts().keys() else 0}")
            st.write(f"Penalties: {data.loc[data['details.eventOwnerTeamId'] == st.session_state['awayTeamId']]['typeDescKey'].value_counts()['penalty'] if 'penalty' in data.loc[data['details.eventOwnerTeamId'] == st.session_state['awayTeamId']]['typeDescKey'].value_counts().keys() else 0}")

## Data used for predictions
with st.container():
    if st.session_state.get("eventID") is not None:
        st.subheader("Data used for predictions (and predictions)")
        st.dataframe(st.session_state["new_df"][st.session_state["features"] + ["model_predictions"]], use_container_width=True)
    

## Heatmap
# Helper functions
def preProcess(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function performs preprocessing on the dataset to make it suitable for visualization.
    output: a processed dataframe.
    """
    scaler = MinMaxScaler(feature_range=(0, 90))
    df["details.xCoord"] = scaler.fit_transform(df[["details.xCoord"]])

    scaler = MinMaxScaler(feature_range=(-42.5, 42.5))
    df["details.yCoord"] = scaler.fit_transform(df[["details.yCoord"]])

    SHOT_TYPES = {'shot-on-goal', 'goal', 'missed-shot', 'blocked-shot'}
    df = df[df['typeDescKey'].isin(SHOT_TYPES)]

    return df

def avgRate(df: pd.DataFrame, id_col: str = "eventId") -> pd.DataFrame:
    """
    Calculate average rate of shooting at each unique (x, y) position
    for a single team DataFrame.
    """
    df_temp = df.copy()
    n = df_temp[id_col].nunique()  # number of shots/events

    # Count how many events at each (x, y)
    df_temp = df_temp.groupby(["details.xCoord", "details.yCoord"]).count()

    # Keep just one column and rename to 'rate'
    df_temp = pd.DataFrame(df_temp.iloc[:, 0])
    df_temp.columns = ["rate"]

    # Normalize by n - rate per event
    df_temp /= n

    return df_temp

def compute_shot_heatmap(df: pd.DataFrame, id_col: str = "eventId") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the (x, y, z) arrays representing the shot-rate heatmap
    for a single team.
    """
    # Get rate per (x, y)
    df_rate = avgRate(df, id_col=id_col).reset_index()

    # Create target grid
    # x: distance from net (0 to 90 ft)
    # y: lateral distance from center (-42.5 to 42.5 ft)
    X, Y = np.round(
        np.meshgrid(
            np.linspace(0, 90, 91),
            np.linspace(-42.5, 42.5, 86),
        )
    )

    # Interpolate rates onto grid
    z = griddata(
        (df_rate["details.xCoord"], df_rate["details.yCoord"]),
        df_rate["rate"],
        (X, Y),
        method="cubic",
        fill_value=0.0,
    )

    # Convert to percentage and smooth
    z = pd.DataFrame(z) * 100.0
    z_smooth = gaussian_filter(z, sigma=5)

    # Build 1D axes corresponding to grid
    x = np.arange(0, 90).astype(np.float32)
    y = np.arange(-42.5, 42.5).astype(np.float32)

    return x, y, z_smooth


def plot_team_shot_heatmap(
    df: pd.DataFrame,
    team_name: str,
    id_col: str = "eventId",
    rink_image_path: str = "images/nhl_rink_half.png",
) -> go.Figure:
    """
    Create and display a single-team shot-rate heatmap.
    """
    # Compute heatmap arrays
    x, y, z_smooth = compute_shot_heatmap(df, id_col=id_col)

    fig = go.Figure()

    # Contour plot
    fig.add_trace(
        go.Contour(
            z=z_smooth.T,
            x=y,  # lateral axis
            y=x,  # distance-from-net axis
            colorscale=[[0, "blue"], [0.5, "white"], [1.0, "red"]],
            opacity=0.4,
            contours=dict(start=-1, end=1, size=0.1),
            colorbar=dict(
                title=dict(
                    text="Shot Rate per Hour (relative)",
                    font=dict(size=14, family="Arial, sans-serif"),
                )
            ),
            showscale=True,
        )
    )

    # Invert y-axis so 0 (net) is at bottom as in original
    fig["layout"]["yaxis"]["autorange"] = "reversed"
    fig["layout"]["xaxis"]["title"] = "Distance from Center of Rink (ft)"
    fig["layout"]["yaxis"]["title"] = "Distance from Net (ft)"
    fig["layout"]["title"] = f"{team_name} - Shot Heatmap"

    # Add rink background if image is available
    if os.path.exists(rink_image_path):
        fig.add_layout_image(
            dict(
                source=Image.open(rink_image_path),
                xref="x",
                yref="y",
                x=-42.5,
                y=-11,
                sizex=85,
                sizey=101,
                opacity=1,
                sizing="stretch",
                layer="below",
            )
        )

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    fig.update_layout(
        showlegend=False,
        width=910,
        height=630,
        autosize=False,
        margin=dict(t=40, b=0, l=0, r=0),
        template="plotly_white",
    )

    # fig.show()
    return fig

if st.session_state.get("eventID") is not None:
    st.subheader("Shot heatmaps")

with st.container():
    if st.session_state.get("eventID") is not None:
        # Removing the datapoints that have either x or y coordinates missing.
        df = st.session_state["df"][~(st.session_state["df"]["details.xCoord"].isna() | st.session_state["df"]["details.yCoord"].isna())]
        df = df.loc[df["details.eventOwnerTeamId"] == st.session_state["homeTeamId"]]

        # Scaling all points such that they can be displayed on a rink half and plotting them.
        df.loc[df["details.xCoord"]>550, "details.xCoord"] -= 550

        df = preProcess(df)
        fig = plot_team_shot_heatmap(df, team_name=st.session_state["teams"][0], id_col="eventId")
        st.plotly_chart(fig, use_container_width=True)


with st.container():
    if st.session_state.get("eventID") is not None:
        # Removing the datapoints that have either x or y coordinates missing.
        df = st.session_state["df"][~(st.session_state["df"]["details.xCoord"].isna() | st.session_state["df"]["details.yCoord"].isna())]
        df = df.loc[df["details.eventOwnerTeamId"] == st.session_state["awayTeamId"]]

        # Scaling all points such that they can be displayed on a rink half and plotting them.
        df.loc[df["details.xCoord"]>550, "details.xCoord"] -= 550
        df = preProcess(df)
        fig = plot_team_shot_heatmap(df, team_name=st.session_state["teams"][1], id_col="eventId")
        st.plotly_chart(fig, use_container_width=True)

with st.container():
    if st.session_state.get("eventID") is not None:
        st.write("Added elements for section 7:\n" \
        "- Team logos\n" \
        "- Shots on goals, blocked shots, missed shots and penalties\n" \
        "- Shot heatmaps")
