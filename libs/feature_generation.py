import numpy as np
import pandas as pd
def calculate_msd_for_team(df : pd.DataFrame):
    df = df.filter(regex = "^home")
    n_rows = len(df.index)
    data_np = df.to_numpy()
    msds = np.zeros(n_rows)
    for j in range(0,len(data_np[0]),2):
        point_initial = (data_np[0,j],data_np[0,j+1])
        for i in range(len(data_np)):
            if (pd.notna(data_np[i,j]) and pd.notna(data_np[0,j])):
                msds[i] += (point_initial[0] - data_np[i,j])**2 + (point_initial[1] - data_np[i,j+1])**2
    return msds/11
            







def team_msd_for_dataframe(df, indices, max_time_lag):
    msds = []
    for index in indices:
        working_data = df.loc[index:index + max_time_lag]
        print(working_data.head())
        if(len(working_data) >= max_time_lag):
            if (len(working_data[working_data["half"] == working_data.iloc[0]["half"]]) == len(working_data.index)):
                msd_data = calculate_msd_for_team(working_data)

                # Number of particles and time lags
                msds.append(msd_data)
      
    
    msds_stacked = np.vstack(msds)
    
    return msds_stacked


def calculate_msd(df, window_size):
    """
    Calculate the mean squared displacement for a set of points.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the points.
    window_size (int): The window size for which to calculate the mean squared displacement.

    Returns:
    np.array: The mean squared displacement values for the points.
    """
    
    df = df.filter(regex='^home')
    # Number of time steps
    np_data = df.to_numpy()
    n = len(df.index)
    # Initialize the MSD array
    msds = []
    if (window_size < n):
        # Loop over all time lags
        #Goes through each column
        for j in range(0, np_data.shape[1] - 1, 2):
            msd = np.zeros(window_size)
            #Changes initial start point
            for i in range(n-window_size):
                #Goes through each lag
                for lag in range(1, window_size):
                    if pd.notna(np_data[i, j]) and pd.notna(np_data[i+lag,j]) and pd.notna(np_data[i, j+1]) and pd.notna(np_data[i+lag,j+1]):

                            # Calculate the squared displacement for the current time lag
                        displacement = (np_data[i, j] - np_data[i+lag,j])**2 + \
                                           (np_data[i, j+1] - np_data[i+lag,j+1])**2 

                        msd[lag] += np.mean(displacement)

            msd = msd/window_size 
            msds.append(msd)
    
        return np.array(msds)


def msd_for_dataframe(df, indices, max_time_lag):
    """
    Calculate the mean squared displacement for a set of points.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the points.
    indices (list): The indices of the rows in the DataFrame.
    max_time_lag (int): The maximum time lag for which to calculate the mean squared displacement.

    Returns:
    np.array: The mean squared displacement values for the points.
    """

    msds = []
    for index in indices:
        working_data = df.loc[index:index + max_time_lag]
        if (len(working_data[working_data["half"] == working_data.iloc[0]["half"]]) == len(working_data.index)):
            msd_data = calculate_msd(working_data, int(max_time_lag/2))
       
            # Number of particles and time lags
            msds.append(msd_data)

 
    msds_stacked = np.vstack(msds)
    
    return msds_stacked





import numpy as np
import pandas as pd


def ripley_k_multiclass(points_1: pd.Series,points_2: pd.Series, radii: np.linspace, width: float, height: float):
    
    # Reshape points from flat Series to array of (x, y) pairs
    points_1 = points_1.dropna()
    n_1 = len(points_1) // 2  # Since points come in pairs (x, y)
    points_array_1 = np.array(points_1).reshape(n_1, 2)

    points_2 = points_2.dropna()
    n_2 = len(points_2) // 2  # Since points come in pairs (x, y)
    points_array_2 = np.array(points_2).reshape(n_2, 2)
  
  

    area = width * height
    lambda_density = n_2 / area
    k_values = []

    # Loop through each radius value
    for r in radii:
        count = 0
        
        # Loop through each point and calculate the pairwise distances
        for i in range(n_1):
            for j in range(n_2):
                if i != j:
                    # Calculate Euclidean distance between point i and point j
                    distance = np.linalg.norm(points_array_1[i] - points_array_2[j])
                    if distance < r:
                        count += 1


        # Calculate Ripley's K for the given radius
        k_r = count / (n_2 * lambda_density)
        k_values.append(k_r)

    return k_values


def ripley_k_multiclass_by_indices(df, indices):
    """
    Calculate Ripley's K values for a set of points.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing the points.
    indices (list): The indices of the rows in the DataFrame.

    Returns:
    np.array: The Ripley's K values for the points.
    """
    k_vals = np.array([ripley_k_multiclass(df.filter(regex='^home').loc[i],df.filter(regex='^away').loc[i],np.arange(0, 34), 105.0, 68.0) for i in indices])
    return k_vals

import numpy as np
from scipy.spatial.distance import cdist

def ripley_k(points: pd.Series, radii: np.ndarray, width: float, height: float):
    """
    Ripley's K function for a set of points in a 2D area.

    Parameters:
    points (pd.Series): The points for which to calculate Ripley's K values,
                        expected format: [x_1, y_1, x_2, y_2, ...].
    radii (np.ndarray): The radii for which to calculate Ripley's K values.
    width (float): The width of the area in which the points are located.
    height (float): The height of the area in which the points are located.

    Returns:
    list: The Ripley's K values for the points. 
    """
    # Reshape points from flat Series to array of (x, y) pairs
    points = points.dropna()
    n = len(points) // 2
    points_array = np.array(points).reshape(n, 2)
    
    area = width * height
    lambda_density = n / area  # Intensity of the process
    k_values = []

    # Compute pairwise distances
    distances = cdist(points_array, points_array)
    for r in radii:
        # Count pairs within distance r (ignoring self-pairs)
        count = np.sum((distances < r) & (distances > 0)) /2  # Avoid zero distances
        k_r = count / (n * lambda_density)  # Adjust by intensity
        k_values.append(k_r)

    return k_values




def ripley_k_by_indices(df, indices):
    """
    Calculate Ripley's K values for a set of points.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing the points.
    indices (list): The indices of the rows in the DataFrame.

    Returns:
    np.array: The Ripley's K values for the points.
    """
    k_vals = np.array([ripley_k(df.filter(regex='^home').loc[i],np.arange(0, 34), 105.0, 68.0) for i in indices])
    return k_vals



import numpy as np

def calculate_xy_mean(arr):
    print(arr)
    """
    Given a NumPy array where columns represent x_0, y_0, x_1, y_1, ..., 
    this function calculates the mean x and y for each row.
    
    Parameters:
    arr (np.ndarray): Input 2D NumPy array where columns alternate between x and y coordinates.

    Returns:
    np.ndarray: A 2D NumPy array with two columns (mean_x, mean_y) for each row.
    """
    # Check if the number of columns is even (pairs of x, y)
    if arr.shape[1] % 2 != 0:
        raise ValueError("Number of columns must be even, representing pairs of x and y coordinates.")
    
    # Separate x and y columns: even indices for x, odd indices for y
    x_values = arr[:, ::2]  # x_0, x_1, x_2, ...
    y_values = arr[:, 1::2] # y_0, y_1, y_2, ...
    
    # Calculate the mean along the x and y columns for each row
    mean_x = np.nanmean(x_values, axis=1)  # Mean of x for each row
    mean_y = np.nanmean(y_values, axis=1)  # Mean of y for each row
    
    # Stack mean_x and mean_y into a 2D array
    mean_xy = np.column_stack((mean_x, mean_y))
    
    return mean_xy


