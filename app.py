import streamlit as st
import requests
import pandas as pd
from bs4 import BeautifulSoup
from fuzzywuzzy import process
import re
import matplotlib.pyplot as plt

@st.cache_data(show_spinner=False)
def get_sleeper_player_map():
    """
    Fetch the (huge) dictionary of all NFL players from Sleeper,
    keyed by player_id.
    As recommended by Sleeper, do not call this repeatedly.
    """
    url = "https://api.sleeper.app/v1/players/nfl"
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()
    
    player_map = {}
    for pid, pdata in data.items():
        first = pdata.get("first_name", "").strip()
        last = pdata.get("last_name", "").strip()
        full_name = f"{first} {last}".strip()
        if not full_name or full_name == "None None":
            full_name = pid
        player_map[pid] = full_name
    return player_map

def get_sleeper_user_id(username: str):
    """
    Convert a Sleeper username to user_id.
    """
    url = f"https://api.sleeper.app/v1/user/{username}"
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.json()["user_id"]

def get_sleeper_league_data(league_id: str):
    """
    Fetch all users and rosters for a given Sleeper league.
    """
    users_url = f"https://api.sleeper.app/v1/league/{league_id}/users"
    rosters_url = f"https://api.sleeper.app/v1/league/{league_id}/rosters"

    u_resp = requests.get(users_url)
    r_resp = requests.get(rosters_url)
    u_resp.raise_for_status()
    r_resp.raise_for_status()

    users_data = u_resp.json()
    rosters_data = r_resp.json()
    return users_data, rosters_data

def parse_rosters_into_dataframe(users_data, rosters_data, player_map):
    """
    Build a DataFrame of each roster, replacing numeric player IDs with actual names.
    Handles cases where a roster's player list is None.
    """
    user_map = {u["user_id"]: u.get("display_name", f"User {u['user_id']}") for u in users_data}

    rows = []
    for roster in rosters_data:
        owner_id = roster.get("owner_id", "")
        owner_name = user_map.get(owner_id, f"User {owner_id}")

        players = roster.get("players") or []
        taxi = roster.get("taxi") or []
        reserve = roster.get("reserve") or []

        def id_to_name(pid):
            return player_map.get(str(pid), f"Unknown ({pid})")

        rows.append({
            "Owner": owner_name,
            "Roster ID": roster["roster_id"],
            "Players": [id_to_name(pid) for pid in players],
            "Taxi Squad": [id_to_name(pid) for pid in taxi],
            "Reserve": [id_to_name(pid) for pid in reserve]
        })

    return pd.DataFrame(rows)

@st.cache_data(show_spinner=False)
def get_all_ktc_players_paginated(max_pages=10):
    """
    Scrapes all KTC dynasty rankings pages up to `max_pages`.
    Each page has ~50 players, so page=0 => first 50, page=1 => next 50, etc.
    Returns a DataFrame of all players found.
    """
    base_url = "https://keeptradecut.com/dynasty-rankings"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/91.0.4472.124 Safari/537.36"
        )
    }

    all_players = []
    for page_num in range(max_pages):
        url = f"{base_url}?page={page_num}&filters=QB|WR|RB|TE|RDP&format=1"
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        player_divs = soup.find_all("div", class_="onePlayer")

        if not player_divs:
            break  # no more players

        for player_div in player_divs:
            try:
                rank = player_div.get("data-attr", "").strip()

                name_tag = player_div.find("div", class_="player-name")
                name = name_tag.find("a").text.strip() if name_tag else "Unknown"

                team_span = name_tag.find("span", class_="player-team") if name_tag else None
                team = team_span.text.strip() if team_span else "N/A"

                position_div = player_div.find("div", class_="position-team")
                position = (
                    position_div.find("p", class_="position").text.strip()
                    if position_div else "Unknown"
                )

                # Age is under <p class='position hidden-xs'> with 'y.o.'
                age_span = position_div.find("p", class_="position hidden-xs") if position_div else None
                age = age_span.text.strip() if age_span else "N/A"

                value_div = player_div.find("div", class_="value")
                value = value_div.find("p").text.strip() if value_div else "0"

                all_players.append({
                    "Rank": rank,
                    "Name": name,
                    "Team": team,
                    "Position": position,
                    "Age": age,
                    "KTC_Value": value,
                })
            except:
                continue

        if len(player_divs) < 50:
            break

    return pd.DataFrame(all_players)

def parse_age(age_str: str) -> float:
    """
    Convert a KTC age string like '25.7 y.o.' to a float (25.7).
    Returns 0.0 if parsing fails or if input is not a valid string.
    """
    if not isinstance(age_str, str):
        return 0.0
    match = re.search(r"(\d+(\.\d+)?)", age_str)
    if match:
        return float(match.group(1))
    return 0.0

def classify_team(team_value_df: pd.DataFrame) -> str:
    """
    Given a DataFrame of a team's players (with KTC_Value and Age columns),
    determine if it's a Contender, Tweener, or Rebuild.
    """
    total_ktc = team_value_df["Value_Numeric"].sum()
    if "Age_Numeric" not in team_value_df.columns:
        team_value_df["Age_Numeric"] = team_value_df["Age"].apply(parse_age)
    avg_age = team_value_df["Age_Numeric"].mean()

    # Example thresholds (tweak these)
    if total_ktc > 110000 and avg_age > 25:
        return "Contender"
    elif total_ktc < 90000 and avg_age < 24:
        return "Rebuild"
    else:
        return "Tweener"

# -------------------------------------------------------
# STREAMLIT APP
# -------------------------------------------------------

st.title("Dynasty Dash")

st.markdown("""
Enter Your Username and League ID.
""")

username = st.text_input("Sleeper Username", "")
league_id = st.text_input("Sleeper League ID", "")

if st.button("Fetch My Team's Data"):
    if not username or not league_id:
        st.error("Please provide both Sleeper username and league ID.")
        st.stop()

    try:
        player_map = get_sleeper_player_map()  # cached call
        #st.success("Loaded player map from Sleeper.")

        user_id = get_sleeper_user_id(username)
        #st.success(f"Sleeper user ID for {username}: {user_id}")

        users_data, rosters_data = get_sleeper_league_data(league_id)
        #st.success(f"Fetched {len(rosters_data)} rosters from league {league_id}.")

        rosters_df = parse_rosters_into_dataframe(users_data, rosters_data, player_map)

        # Grab KTC data
        ktc_df = get_all_ktc_players_paginated(max_pages=10)
        #st.success("KeepTradeCut values secured!.")

        # Identify your roster
        my_roster = rosters_df[rosters_df["Owner"] == username]

        # === Create Tabs ===
        tab1, tab2, tab3 = st.tabs(["League", "Keep Trade Cut", "Team"])

        with tab1:
            st.write("### League Rosters")
            st.dataframe(rosters_df)

        with tab2:
            st.write("### KeepTradeCut Rankings")
            st.dataframe(ktc_df)

        with tab3:
            # If the user has no roster, show a warning
            if my_roster.empty:
                st.warning(
                    f"Couldn’t find a roster for {username}. "
                    f"Check if your display name matches exactly."
                )
            else:
                st.write(f"### {username}'s Roster")
                st.dataframe(my_roster[["Players", "Taxi Squad", "Reserve"]])

                # Summarize your main players with KTC
                roster_players = []
                for row_index, row_data in my_roster.iterrows():
                    roster_players.extend(row_data["Players"])

                def normalize_name(name):
                    parts = name.strip().lower().split()
                    first, last = parts[0], parts[-1] if len(parts) > 1 else ""
                    return first, last

                ktc_name_map = {
                    normalize_name(row["Name"]): row
                    for _, row in ktc_df.iterrows()
                }

                your_team_value_data = []
                for p_name in roster_players:
                    p_first, p_last = normalize_name(p_name)
                    # exact match or fallback fuzzy
                    if (p_first, p_last) in ktc_name_map:
                        best_match = ktc_name_map[(p_first, p_last)]
                    else:
                        best_match_name, score = process.extractOne(
                            p_name, ktc_df["Name"].tolist()
                        )
                        best_match = (
                            ktc_df[ktc_df["Name"] == best_match_name].iloc[0]
                            if score and score > 80
                            else None
                        )

                    if best_match is not None:
                        your_team_value_data.append({
                            "Player": p_name,
                            "KTC_Name": best_match["Name"],
                            "Position": best_match["Position"],
                            "KTC_Value": best_match["KTC_Value"],
                            "Age": best_match["Age"],
                        })
                    else:
                        your_team_value_data.append({
                            "Player": p_name,
                            "KTC_Name": "No Match",
                            "Position": None,
                            "KTC_Value": None,
                            "Age": None,
                        })

                team_value_df = pd.DataFrame(your_team_value_data)

                # Convert KTC_Value to float
                def safe_float(v):
                    try:
                        return float(v.replace(",", ""))
                    except:
                        return 0.0

                team_value_df["Value_Numeric"] = team_value_df["KTC_Value"].apply(safe_float)
                st.write("### Keep Trade Cut Values for Your Team")
                st.dataframe(team_value_df)

                total_val = team_value_df["Value_Numeric"].sum()
                st.write(f"**Your Team's Approximate Total Value**: {total_val}")

                # Contender vs Rebuild classification
                label = classify_team(team_value_df)
                st.write(f"**Contention Status**: {label}")

                # Show average age
                team_value_df["Age_Numeric"] = team_value_df["Age"].apply(parse_age)
                mean_age = team_value_df["Age_Numeric"].mean()
                st.write(f"**Average Age**: {mean_age:.1f}")

                # Quick bar chart of top ~12 players by KTC
                top_12 = team_value_df.sort_values("Value_Numeric", ascending=False).head(12)
                st.bar_chart(data=top_12, x="Player", y="Value_Numeric")

    except Exception as e:
        st.error(f"Something went wrong: {e}")


## Future Integrations ##

#-- Future Improvements 
#1. Draft Pick Evaluator - Estimate future rookie pick vallues and use KTC Dynasty Rankings (scrape)
#2. Who should you trade with? (create some formula to identify based on contention and overabundance of position)
#3. Dynasty Power Rankings - identify where you rank in your league
#4. Sqllite - consider integrating a db

########################
