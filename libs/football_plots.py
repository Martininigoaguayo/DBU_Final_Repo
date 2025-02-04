# Library for generating the folowing plots
# 1. Heatmap of the pitch
# 2. Plots of pitch
# 3. Graphs for stats
# 4. Graphs for match durations
from libs.folders import *
from libs.data_manipulation import *
from mplsoccer import Pitch
import os
import pandas as pd
import numpy as np
from mplsoccer import Pitch
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


# Function to generate heatmaps with standardized color intensity
def plot_standardized_heatmap(team_identifier, match_specific_data, global_min, global_max):
    # Collect all player positions
    player_positions_x = []
    player_positions_y = []

    # Loop over all player positions for each player (assuming 11 players total)
    for player_number in range(1, 12):  # 11 players
        x_position_column = f'home_{player_number}_x'
        y_position_column = f'home_{player_number}_y'

        if x_position_column in match_specific_data.columns and y_position_column in match_specific_data.columns:
            player_positions_x.extend(match_specific_data[x_position_column].dropna().to_numpy())
            player_positions_y.extend(match_specific_data[y_position_column].dropna().to_numpy())

    # Create the pitch with skillcorner dimensions
    football_pitch = Pitch(pitch_type='skillcorner', pitch_length=105, pitch_width=68, axis=True, label=True, line_color="white", pitch_color="grass")
    figure, axis = football_pitch.draw(figsize=(10, 7))

    # Create a heatmap based on all player positions
    bin_data = football_pitch.bin_statistic(player_positions_x, player_positions_y, statistic='count', bins=(105, 68))
    smoothed_heatmap = gaussian_filter(bin_data['statistic'], sigma=2)

    # Plot the smoothed heatmap with standardized color range
    color_map = axis.imshow(smoothed_heatmap, extent=(-52.5, 52.5, -34, 34), cmap='coolwarm', interpolation='gaussian', vmin=global_min, vmax=global_max)
    
    # Add a colorbar
    figure.colorbar(color_map, ax=axis)

    # Add labels and title
    plt.title(f'Smoothed Heatmap of All Players Movements (Team: {team_identifier})')
    
    # Save the smoothed heatmap
    plt.savefig("HeatMaps"+ "/" +team_identifier + "/" + "smoothed_heatmap_all_players")
    plt.show()

## Example data for three teams
#teams = ["Denmark", "England", "Spain"]
#team_data = {}
#global_min = float('inf')
#global_max = float('-inf')
#
## Loop over teams to get the min and max values for color standardization
#for team_identifier in teams:
#    # Assume the function extract_data and extract_one_match are defined to extract data
#    team_data[team_identifier] = extract_data(team_identifier, "eric_martin")
#    match_specific_data = extract_one_match(team_data[team_identifier], 4)
#    
#    # Collect all player positions to compute the overall min/max values
#    player_positions_x = []
#    player_positions_y = []
#    for player_number in range(1, 12):
#        x_position_column = f'home_{player_number}_x'
#        y_position_column = f'home_{player_number}_y'
#        if x_position_column in match_specific_data.columns and y_position_column in match_specific_data.columns:
#            player_positions_x.extend(match_specific_data[x_position_column].dropna().to_numpy())
#            player_positions_y.extend(match_specific_data[y_position_column].dropna().to_numpy())
#    
#    # Create a heatmap based on all player positions
#    football_pitch = Pitch(pitch_type='skillcorner', pitch_length=105, pitch_width=68, axis=True, label=True, line_color="white", pitch_color="grass")
#    bin_data = football_pitch.bin_statistic(player_positions_x, player_positions_y, statistic='count', bins=(105, 68))
#    
#    # Smooth the heatmap data
#    smoothed_heatmap = gaussian_filter(bin_data['statistic'], sigma=2)
#    
#    # Update the global min/max values for color standardization
#    global_min = min(global_min, smoothed_heatmap.min())
#    global_max = max(global_max, 7)
#
## Now, plot each heatmap using the standardized color range
#for team_identifier in teams:
#    match_specific_data = extract_one_match(team_data[team_identifier], 4)
#    plot_standardized_heatmap(team_identifier, match_specific_data, global_min, global_max)
#




def process_match_durations(base_folder):
    """
    Process match durations from CSV files stored in subdirectories of the base folder.

    Parameters:
    base_folder (str): The path to the directory containing match folders.

    Returns:
    list: A list of match durations in minutes.
    float: The average match duration in minutes.
    """
    # Initialize a list to store match durations
    match_durations = []

    # Loop over each match folder in the base folder
    for match_folder in os.listdir(base_folder):
        match_path = os.path.join(base_folder, match_folder)
        
        # Check if it's a directory (a match folder)
        if os.path.isdir(match_path):
            # List the CSV files inside the match folder (assumes there's a home and away CSV file)
            team_files = os.listdir(match_path)
            
            if len(team_files) > 0:
                # Take only the first CSV file in the folder (assuming this corresponds to the home team)
                home_team_file_path = os.path.join(match_path, team_files[0])
                
                # Read the CSV file for the home team
                try:
                    match_data = pd.read_csv(home_team_file_path)
                    
                    # Check if 'Time [s]' column exists and calculate match duration
                    if 'Time [s]' in match_data.columns:
                        total_match_time_seconds = match_data['Time [s]'].max()
                        match_duration_minutes = total_match_time_seconds / 60  # Convert to minutes
                        match_durations.append(match_duration_minutes)
                        print(f"Processed {team_files[0]}: {match_duration_minutes:.2f} minutes")
                    else:
                        print(f"'Time [s]' column not found in {team_files[0]}")
                
                except Exception as e:
                    print(f"Error processing {home_team_file_path}: {e}")
    
    # Calculate the average match duration
    average_match_duration = np.mean(match_durations) if match_durations else 0

    return match_durations, average_match_duration


def plot_match_durations(match_durations, average_duration, output_file="match_durations_all_teams.png"):
    """
    Plot and save a bar chart of match durations and average duration.

    Parameters:
    match_durations (list): A list of match durations in minutes.
    average_duration (float): The average match duration in minutes.
    output_file (str): The filename to save the plot.
    """
    # Create an x-axis based on the number of matches processed
    x_axis = np.arange(1, len(match_durations) + 1)

    # Plotting the match durations
    plt.figure()
    plt.bar(x_axis, match_durations, color='skyblue', label='Match Durations')
    plt.axhline(y=average_duration, color='r', linestyle='-', label=f'Average Duration: {average_duration:.2f} minutes')
    plt.text(len(match_durations) / 2, average_duration + 1, f'{average_duration:.2f} minutes', color='red', ha='center')

    # Add labels, title, and save the figure
    plt.xlabel('Match Number')
    plt.ylabel('Match Duration (minutes)')
    plt.title('Match Durations')
    plt.legend()

    # Save the figure
    plt.savefig(output_file)
    
    # Show the plot
    plt.show()





def plot_clusters(components, index_labeled_data, team, title):
    """
    Plot clusters using the first two principal components.

    Parameters:
    components (np.ndarray): The principal components.
    index_labeled_data (np.ndarray): The data with cluster labels.
    team (str): The team identifier.
    title (str): The title of the plot.

    Returns:
    None
    """

    # Extract cluster labels from index_labeled_data
    cluster_labels = index_labeled_data[:, 1].astype(int)
    
    # Scatter plot using the first two PCA components
    plt.figure(figsize=(12, 6))

    # Plot each cluster separately
    for cluster in np.unique(cluster_labels):
        cluster_points = components[cluster_labels == cluster]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster}')
    
    # Labeling the plot
    plt.title('K-Means Clustering Using First Two Principal Components')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    
    # Place the legend outside the plot, on the right side
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Clusters")

    # Adjust layout to make space for the legend
    plt.subplots_adjust(right=0.8)

    plt.grid(True)

    # Save the plot
    plt.savefig(team + "/" + "clustering_with_pca_" + title)
    plt.show()




def generate_pitch_with_vectors(df_processed: pd.DataFrame, filename :str, ):
    """
    Generate a pitch plot with player positions and ball movement vectors.

    Parameters:
    df_processed (pd.DataFrame): The processed DataFrame containing player and ball positions.
    filename (str): The filename to save the plot.

    Returns:
    None
    """
    # Extract player positions and ball positions
    df_ball = df_processed[["ball_x", "ball_y"]]
    half = str(df_processed['half'].to_numpy()[0])
    #CHANGE TO REGEX
    df_current = df_processed.filter(regex='^home')
    # Set up the pitch
    pitch = Pitch(pitch_type='skillcorner', pitch_length =105, pitch_width = 68, axis=True, label=True, line_color="white", pitch_color="grass")
    fig, ax = pitch.draw(figsize=(12, 8))

    # Extract player positions
    np_data = df_current.to_numpy()
    player_colors = plt.cm.viridis(np.linspace(0, 1, int((np_data.shape[1]) / 2)))  # Color map for players
    
    # Add markers for legend (not shown on pitch)
    #for color_idx in range(len(player_colors)):
    #    ax.plot([], [], 'o', color=player_colors[color_idx], label=f'Player {color_idx + 1}')

    for i in range(np_data.shape[0]-1):
        for j in range(0, np_data.shape[1] - 1, 2):
            x = np_data[i, j]
            y = np_data[i, j + 1]
            if pd.notna(x) and pd.notna(y):
                ax.scatter(x, y, color = "red", alpha=(i+1)/(np_data.shape[0]-1)) #,color=player_colors[j // 2], edgecolors='black', s=100, alpha=(i+1)/(np_data.shape[0]-1))

    # Plot the ball
    #ax.scatter(df_ball["ball_x"].iloc[1:], df_ball["ball_y"].iloc[1:], s=120, color='yellow', edgecolors='red', linewidth=2, label='Ball')
    ax.scatter(df_ball["ball_x"].iloc[0], df_ball["ball_y"].iloc[0], s=120, color='blue', edgecolors='red', linewidth=2, label='Ball_start')
    
    # Quiver for ball movement
    np_data = df_ball.to_numpy()
    for i in range(np_data.shape[0]-1):
        x = np_data[i, 0]
        y = np_data[i, 1]
        x_1 = np_data[i+1, 0]
        y_1 = np_data[i+1, 1]
        ax.quiver(x, y, (x_1-x), (y_1-y), angles='xy', scale_units='xy', alpha=0.3, width=0.005)


    
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=4)
    plt.xlabel("Meters",fontsize = 14)
    plt.ylabel("Meters",fontsize=14)
    plt.title("Half :" + half + " ,Time [s]: " + str(df_processed["Time [s]"].to_numpy()[0]))
    plt.savefig(filename)
    plt.close()

    
def generate_pitches_from_start_indices(indices: list, src_df : pd.DataFrame, dest : str, step, n_ticks):
    """
    Generate pitch plots with player positions and ball movement vectors at specific indices.

    Parameters:
    indices (list): List of indices to generate pitch plots.
    src_df (pd.DataFrame): The source DataFrame containing player and ball positions.
    dest (str): The destination folder to save the plots.
    step (int): The step size for generating plots.
    n_ticks (int): The number of ticks to plot.

    Returns:
    None
    """
    
    clear_folder(dest)
    for value in indices:
        generate_pitch_with_vectors( src_df.loc[value:value+n_ticks:step], dest+"/ball_stoppage_index_"+str(value)+".png")

import matplotlib.pyplot as plt
from mplsoccer import Pitch
import pandas as pd
import numpy as np

def display_pitches_with_ball_movement(df_processed: pd.DataFrame, indices: list):
    """
    Display a grid of pitch plots with player positions and ball movement for the next 96 ticks for specified indices.

    Parameters:
    df_processed (pd.DataFrame): The processed DataFrame containing player and ball positions.
    indices (list): List of indices in the DataFrame to plot.

    Returns:
    None
    """
    # Set up the subplot layout
    num_pitches = len(indices)
    rows = (num_pitches + 1) // 2  # 2 pitches per row
    fig, axs = plt.subplots(rows, 2, figsize=(16, rows * 8))  # Adjust height as needed
    axs = axs.flatten()  # Flatten in case of odd number of pitches

    # Loop through each index and plot a pitch
    for idx, i in enumerate(indices):
        ax = axs[idx]
        
        # Extract the data for the specific index
        df_ball_start = df_processed.loc[i, ["ball_x", "ball_y"]]
        half = str(df_processed.loc[i, 'half'])
        df_current = df_processed.filter(regex='^home').loc[i]
        
        # Set up the pitch
        pitch = Pitch(pitch_type='skillcorner', pitch_length=105, pitch_width=68, axis=True, label=True, line_color="white", pitch_color="grass")
        pitch.draw(ax=ax)
        
        # Extract player positions
        np_data = df_current.to_numpy().reshape(-1, 2)  # Reshape to get (x, y) pairs
        player_colors = plt.cm.viridis(np.linspace(0, 1, np_data.shape[0]))  # Color map for players
        
        # Plot player positions (static)
        for j, (x, y) in enumerate(np_data):
            if pd.notna(x) and pd.notna(y):
                ax.scatter(x, y, color=player_colors[j], edgecolors='black', s=100, alpha=0.7, label=f'Player {j + 1}')
        
        # Plot ball position over the next 96 ticks (or until the end of the data if fewer than 96 rows are left)
        end_idx = min(i + 96, len(df_processed))
        df_ball_movement = df_processed.loc[i:end_idx, ["ball_x", "ball_y"]]
        
        # Initial ball position (highlighted)
        ax.scatter(df_ball_start["ball_x"], df_ball_start["ball_y"], s=120, color='blue', edgecolors='red', linewidth=2, label='Ball_start')
        
        # Plot ball trail to show movement
        for j in range(1, len(df_ball_movement)):
            x = df_ball_movement.iloc[j]["ball_x"]
            y = df_ball_movement.iloc[j]["ball_y"]
            if pd.notna(x) and pd.notna(y):
                ax.scatter(x, y, s=100, color='yellow', edgecolors='red', alpha=(j / len(df_ball_movement)), label='Ball' if j == 1 else "")

        # Title and legend
        ax.set_title(f"Half: {half}, Time [s]: {df_processed['Time [s]'].iloc[i]}")
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=4)

    # Hide unused subplots if the number of pitches is odd
    for j in range(num_pitches, rows * 2):
        fig.delaxes(axs[j])

    plt.tight_layout()
    plt.show()
