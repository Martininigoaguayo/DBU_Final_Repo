from libs.data_manipulation import *
from libs.feature_generation import *
from libs.dim_reduction import *
from libs.football_plots import *
from libs.clustering import *
from libs.convex_hull import *

import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
from scipy.stats import wasserstein_distance_nd
from mplsoccer import *
import pandas as pd
import numpy as np
import os


def calculate_weights(df: pd.DataFrame, fun, ball_x_col='ball_x', ball_y_col='ball_y', regex="^home"):
    ball_x = df[ball_x_col].values
    ball_y = df[ball_y_col].values
    player_cols = df.filter(regex=regex).columns
    
    # Extract player positions and calculate adjusted inverse distance to the ball
    weights_list = []
    x_cols = [col for col in player_cols if col.endswith('_x')]
    y_cols = [col for col in player_cols if col.endswith('_y')]
    indices = df.index.to_numpy()
    for frame_idx in range(len(df)):
        weights = []
        sum=0
        for i in range(len(x_cols)):  # Loop through all players
            player_x = df.loc[indices[frame_idx], x_cols[i]]
            player_y = df.loc[indices[frame_idx], y_cols[i]]
            
            # Check if player_x or player_y is NaN (inactive player)
            if np.isnan(player_x) or np.isnan(player_y):
                continue  # Skip this player if they are inactive
            
           
            # Calculate the distance to the ball and ensure a small epsilon is added
            distance_to_ball = np.sqrt((player_x - ball_x[frame_idx])**2 + (player_y - ball_y[frame_idx])**2)
          
            weight = fun(distance_to_ball) 
            sum+=weight
            weights.append(weight)

        weights.append(fun(0, sum)) #Adding final weight for ball
        
        weights_list.append(weights)
    print(len(weights_list))
    return weights_list  # Return a list of arrays with normalized weights


def normalize_positions_with_ball(df):
    # Create copies of ball x and y columns for both home and away teams
    df_normalized = df.copy()
    
    # Normalize 'home' team player positions
    home_columns_x = [col for col in df.columns if col.startswith('home') and col.endswith('_x')]
    home_columns_y = [col for col in df.columns if col.startswith('home') and col.endswith('_y')]
    
    for x_col, y_col in zip(home_columns_x, home_columns_y):
        df_normalized[x_col] = df[x_col] - df['ball_x']
        df_normalized[y_col] = df[y_col] - df['ball_y']
    
    # Normalize 'away' team player positions
    away_columns_x = [col for col in df.columns if col.startswith('away') and col.endswith('_x')]
    away_columns_y = [col for col in df.columns if col.startswith('away') and col.endswith('_y')]
    
    for x_col, y_col in zip(away_columns_x, away_columns_y):
        df_normalized[x_col] = df[x_col] - df['ball_x']
        df_normalized[y_col] = df[y_col] - df['ball_y']
    
    return df_normalized


def filter_by_ball_radius(data, index, radius):
    # Get the ball position at the specified index
    ref_ball_x = data.at[index, 'ball_x_team']
    ref_ball_y = data.at[index, 'ball_y_team']
    
    # Calculate the distance of each row's ball position from the reference position
    distances = np.sqrt((data['ball_x_team'] - ref_ball_x)**2 + (data['ball_y_team'] - ref_ball_y)**2)
    
    # Filter rows where the distance is less than or equal to the radius
    filtered_data = data[distances <= radius]
    
    return filtered_data




def most_similar_with_wasserstein(relevant_index, relevant_df,relevant_weights, weighting_function, steps = 48, normalizing_factor = 11, max_weight = 1):
    one_match = relevant_df
    identified_corner_df= relevant_df.loc[relevant_index:relevant_index+1]
    one_match = one_match.iloc[::steps]
    print(weighting_function)
    #####
    inverse_identified_corner_weights = calculate_weights(identified_corner_df,fun = weighting_function)
    inverse_distance_list = relevant_weights
    #one_match = normalize_positions_with_ball(one_match)

    # Filter the columns, then reorder so 'ball_x_team' and 'ball_y_team' are last
    columns_to_select = one_match.filter(regex="^home|ball_x|ball_y").columns
    # Separate ball_x_team and ball_y_team columns and place them at the end
    reordered_columns = [col for col in columns_to_select if not col.startswith("ball")] + \
                        [col for col in columns_to_select if col.startswith("ball")]
    # Apply the reordered columns to the DataFrame, then convert to numpy
    coordinates_numpy = one_match[reordered_columns].to_numpy()

    identified_corner_coordinates_numpy = identified_corner_df[reordered_columns].to_numpy()

    identified_corner_coordinates = [list(zip(row[~np.isnan(row)][::2],row[~np.isnan(row)][1::2])) for row in identified_corner_coordinates_numpy]
    coordinates_zipped = [list(zip(row[~np.isnan(row)][::2],row[~np.isnan(row)][1::2])) for row in coordinates_numpy]
    
    #Get closest situations
    distances = []
    indices = one_match.index.to_numpy()

    i = 0
    for weights, coordinates in zip(inverse_distance_list, coordinates_zipped):

        if(not np.isnan(np.sum(weights)) and (len(weights) == len(inverse_identified_corner_weights[0])) and (len(coordinates) == len(identified_corner_coordinates[0]) )):
  
            distances.append((wasserstein_distance_nd(identified_corner_coordinates[0], coordinates, u_weights= inverse_identified_corner_weights[0], v_weights=weights), indices[i]))
        i+=1
    indices_and_distances = sorted(distances, key = lambda t: t[0])
    indices = [index for _,index in indices_and_distances]
    if (len(indices) == 0):
        raise ValueError("No reccomendations")

    return indices


def most_similar_with_wasserstein_from_row(clicked_row : dict, relevant_df, relevant_weights,weighting_function, steps=48, normalizing_factor=11, max_weight=1):
    """
    Find the most similar situations to a given clicked row using Wasserstein distance.
    
    Parameters:
    - clicked_row: A dictionary representing a specific situation with player positions and ball position.
    - relevant_df: DataFrame containing situations to compare against.
    - weighting_function: Function to calculate weights.
    - steps: Step size for downsampling the relevant_df DataFrame.
    - normalizing_factor: Normalizing factor for the weighting function.
    - max_weight: Maximum weight for the weighting function.
    
    Returns:
    - indices: List of indices in `relevant_df` sorted by similarity to the clicked situation.
    """
    one_match = relevant_df.iloc[::steps]  # Downsample relevant_df
    clicked_df = pd.DataFrame([clicked_row], clicked_row.keys())  # Convert clicked row to DataFrame
    print(clicked_df.head())
    # Calculate weights for clicked situation and for each row in one_match
    clicked_weights = calculate_weights(clicked_df, weighting_function,"ball_x_team","ball_y_team")
    one_match_weights = relevant_weights

    # Prepare and reorder columns for comparison
    columns_to_select = one_match.filter(regex="^home|^ball_x|^ball_y").columns
    reordered_columns = [col for col in columns_to_select if not col.startswith("ball")] + \
                        [col for col in columns_to_select if col.startswith("ball")]
    
    clicked_columns = [col for col in clicked_df.columns if not col.startswith("ball")] + \
                        [col for col in clicked_df.columns if col.startswith("ball")]

    
    coordinates_numpy = one_match[reordered_columns].to_numpy()
    clicked_coordinates_numpy = clicked_df[clicked_columns].to_numpy()


    # Convert coordinates to zipped format
    clicked_coordinates = [list(zip(row[~np.isnan(row)][::2], row[~np.isnan(row)][1::2])) for row in clicked_coordinates_numpy]
    coordinates_zipped = [list(zip(row[~np.isnan(row)][::2], row[~np.isnan(row)][1::2])) for row in coordinates_numpy]


    
    # Calculate Wasserstein distances between clicked situation and each row in one_match
    distances = []
    indices = one_match.index.to_numpy()
    i = 0
    for weights, coordinates in zip(one_match_weights, coordinates_zipped):
        print(len(coordinates),len(clicked_coordinates[0]))
        if not np.isnan(np.sum(weights)) and (len(weights) == len(clicked_weights[0])) and (len(coordinates) == len(clicked_coordinates[0])):
            distance = wasserstein_distance_nd(clicked_coordinates[0], coordinates, u_weights=clicked_weights[0], v_weights=weights)
            distances.append((distance, indices[i]))
        i += 1

    # Sort by distances and extract indices
    indices_and_distances = sorted(distances, key=lambda t: t[0])
    indices = [index for _, index in indices_and_distances]
    
    return indices


def filter_by_ball_radius(data, ball_x, ball_y, radius):
    ref_ball_x = ball_x
    ref_ball_y = ball_y    
    # Calculate the distance of each row's ball position from the reference position
    distances = np.sqrt((data['ball_x'] - ref_ball_x)**2 + (data['ball_y'] - ref_ball_y)**2)
    
    # Filter rows where the distance is less than or equal to the radius
    filtered_data = data[distances <= radius]
    
    return filtered_data


def control_weighting(x, sum=0,scaling_factor = 20):
    return 1

def inverse_weighting(x, sum=0,scaling_factor = 20, n_points = 12, ball_weighting = 0.1):
    if (x > 1):
        return scaling_factor/(x*n_points)
    elif ( x > 0 and x <= 1):
        return scaling_factor/n_points
    else:
        return (scaling_factor - sum) + ball_weighting


def linear_weighting(x, sum=0, y_intercept = 200, n_points=12,ball_weighting = 0.1):
    if (x > 0):
        return (y_intercept-x)/(n_points)
    else:
        return (y_intercept - sum) + ball_weighting

def inverse_exponential_weighting(x, sum=0, scaling_factor=20, n_points=12, ball_weighting = 0.1):
    if (x>0):
        return np.exp(-x/scaling_factor)/n_points
    else:
        return (1 - sum) + ball_weighting




def most_similar_with_wasserstein_closed_interval(relevant_index, relevant_df, weighting_function, steps = 48, normalizing_factor = 11, max_weight = 1,interval_steps=2):
    one_match = relevant_df
    identified_corner_start_df = relevant_df.loc[relevant_index:relevant_index+1]
    identified_corner_stop_df = relevant_df.loc[relevant_index+(steps*interval_steps):relevant_index+(steps*interval_steps)+1]
    one_match = one_match.iloc[::steps]

    #####
    identified_situation_weights_start = calculate_weights(identified_corner_start_df, weighting_function)
    identified_situation_weights_stop = calculate_weights(identified_corner_stop_df,weighting_function)





    inverse_distance_list = calculate_weights(one_match,weighting_function) #Inverse proportionality to distance
    #one_match = normalize_positions_with_ball(one_match)

    # Filter the columns, then reorder so 'ball_x_team' and 'ball_y_team' are last
    columns_to_select = one_match.filter(regex="^home|ball_x|ball_y").columns
    # Separate ball_x_team and ball_y_team columns and place them at the end
    reordered_columns = [col for col in columns_to_select if not col.startswith("ball")] + \
                        [col for col in columns_to_select if col.startswith("ball")]
    
    
    # Apply the reordered columns to the DataFrame, then convert to numpy
    coordinates_numpy = one_match[reordered_columns].to_numpy()

    
    identified_situation_coordinates_start_numpy = identified_corner_start_df[reordered_columns].to_numpy()
    identified_situation_coordinates_stop_numpy = identified_corner_stop_df[reordered_columns].to_numpy()

    identified_situation_coordinates_start = [list(zip(row[~np.isnan(row)][::2],row[~np.isnan(row)][1::2])) for row in identified_situation_coordinates_start_numpy]
    identified_situation_coordinates_stop = [list(zip(row[~np.isnan(row)][::2],row[~np.isnan(row)][1::2])) for row in identified_situation_coordinates_stop_numpy]

    coordinates_zipped = [list(zip(row[~np.isnan(row)][::2],row[~np.isnan(row)][1::2])) for row in coordinates_numpy]
    distances = []
    indices = one_match.index.to_numpy()


    i = 0
    print("Starting length calculation")
    for weights, coordinates, weights_next, coordinates_next  in zip(inverse_distance_list, coordinates_zipped, inverse_distance_list[interval_steps:],coordinates_zipped[interval_steps:]):
        if(not np.isnan(np.sum(weights)) and (len(weights) == len(identified_situation_weights_start[0])) and (len(coordinates) == len(identified_situation_coordinates_start[0]) ) and (len(coordinates_next) == len(coordinates))):
            

            start_distance = wasserstein_distance_nd(identified_situation_coordinates_start[0], coordinates, u_weights= identified_situation_weights_start[0], v_weights=weights)
            stop_distance = wasserstein_distance_nd(identified_situation_coordinates_stop[0], coordinates_next, u_weights= identified_situation_weights_stop[0], v_weights=weights_next)
            distances.append((np.average([start_distance,stop_distance]), indices[i]))


        i+=1
    indices_and_distances = sorted(distances, key = lambda t: t[0])
    indices = [index for _,index in indices_and_distances]
    if (len(indices) == 0):
        raise ValueError("No reccomendations")
    print(len(indices))

    return indices