import json
import os
import pandas as pd
from libs.Video_Player import VideoPlayer

def extract_metadata(file_path):
    """Load and return data from a JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_half_start(data):
    """Extract and convert start times of the first and second half into seconds."""
    first_half_start = data["halvesTimestamps"]["1H"]["startTime"]
    second_half_start = data["halvesTimestamps"]["2H"]["startTime"]
    first_half_end = data["halvesTimestamps"]["1H"]["endTime"]

    # Convert HH:MM:SS to total seconds
    def time_to_seconds(time_str):
        h, m, s = map(int, time_str.split(":"))
        return h * 3600 + m * 60 + s

    return time_to_seconds(first_half_start), time_to_seconds(second_half_start), time_to_seconds(first_half_end)

def find_files(match_folder):
    """Finds the .json and .mp4 files in the match folder."""
    json_file, video_file = None, None

    for file in os.listdir(match_folder):
        if file.endswith(".json"):
            json_file = os.path.join(match_folder, file)
        elif file.endswith(".mp4"):
            video_file = os.path.join(match_folder, file)

    if not json_file or not video_file:
        raise FileNotFoundError("Required .json or .mp4 file missing in the folder.")

    return json_file, video_file

def find_match_folder(tournament_folder, match_name):
    """
    Finds the correct match folder inside the tournament folder.
    Match folders contain numbers (e.g., '2036178_Denmark_England'),
    so we search for a folder that contains 'match_name'.
    """
    for folder in os.listdir(tournament_folder):
        if match_name in folder:
            return os.path.join(tournament_folder, folder)
    
    raise FileNotFoundError(f"Match folder for '{match_name}' not found in '{tournament_folder}'")

def process_match_data(tournament_folder, raw_match_name):
    """
    Given the tournament folder and match name from the DataFrame,
    locate the correct match folder, extract halves' start times, and find the video.
    """
    match_folder = find_match_folder(tournament_folder, raw_match_name)
    
    # Find JSON and MP4 files dynamically
    json_file, video_file = find_files(match_folder)

    # Extract data from JSON
    data = extract_metadata(json_file)
    first_half_start, second_half_start, first_half_end = extract_half_start(data)

    return {
        "match_folder": match_folder,
        "json_file": json_file,
        "video_file": video_file,
        "first_half_start": first_half_start,
        "second_half_start": second_half_start,
        "first_half_end" : first_half_end
    }

# **Optimized function to process only unique match names**
def process_dataframe(df, tournament_folder):
    """Processes each unique match in the DataFrame to extract JSON and video details."""
    results = []
    
    unique_match_names = df['match_name'].drop_duplicates().tolist()  # Extract unique match names

    for match_name in unique_match_names:
        print(f"Processing match: {match_name}")
        try:
            match_data = process_match_data(tournament_folder, match_name)
            results.append(match_data)
            # Append the match name to results
            results[-1]["match_name"] = match_name
        except FileNotFoundError as e:
            print(f"Error processing match {match_name}: {e}")
    return pd.DataFrame(results)



def show_reccomendations(processed_video_info, match_data : pd.DataFrame, indices):
    """
    Given the processed video information and the match data DataFrame, display the relevant rows
    and launch the Video Player for each match.
    """
    relevant_rows = match_data.loc[indices]
    print(relevant_rows)
    
    for index, data in relevant_rows.iterrows():
        print(data)

        meta_data = processed_video_info[processed_video_info["match_name"] == data['match_name']].iloc[0]
        indices = [data[["Time [s]","half"]].to_numpy()]
        print(indices)
        video_file = meta_data["video_file"]
        print(meta_data['video_file'])
        video_name = os.path.basename(video_file)
        first_half_start = meta_data["first_half_start"]
        second_half_start = meta_data["second_half_start"]
        first_half_end = meta_data["first_half_end"]
        if (video_name):
            VideoPlayer(video_file, video_name, first_half_start,first_half_end, second_half_start, indices)
    

