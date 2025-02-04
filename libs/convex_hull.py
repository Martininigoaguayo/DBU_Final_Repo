import numpy as np
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
from libs.feature_generation import *
import pandas as pd

def hull_to_polygon(hull):
    """
    Convert a scipy ConvexHull object to a shapely Polygon.
    
    Parameters:
    hull (scipy.spatial.ConvexHull): A ConvexHull object.

    Returns:
    shapely.geometry.Polygon: A Polygon object representing the convex hull.

    """

    points = hull.points[hull.vertices]
    return Polygon(points)

def find_centroid(hull):
    """
    Find the centroid of a ConvexHull.
    
    Parameters:
    hull (scipy.spatial.ConvexHull): A ConvexHull object.

    Returns:
    numpy.ndarray: The centroid of the ConvexHull.
    """
    points = hull.points[hull.vertices]
    centroid = np.mean(points, axis=0)
    return centroid

def normalize_hull(hull):
    """
    Normalize a ConvexHull by centering it around (0, 0).
    
    Parameters:
    hull (scipy.spatial.ConvexHull): A ConvexHull object.

    Returns:
    scipy.spatial.ConvexHull: A normalized ConvexHull object.
    """
    centroid = find_centroid(hull)
    normalized_points = hull.points - centroid  # Translate points to center around (0, 0)
    return ConvexHull(normalized_points)

def overlapping_area(hull1, hull2):
    """
    Compute the overlapping area between two convex hulls.
    
    Parameters:
    hull1 (scipy.spatial.ConvexHull): The first ConvexHull object.
    hull2 (scipy.spatial.ConvexHull): The second ConvexHull object.
    
    Returns:
    float: The overlapping area between the two convex hulls.
    """
    poly1 = hull_to_polygon(hull1)
    poly2 = hull_to_polygon(hull2)
    
    if not poly1.intersects(poly2):
        return 0.0
    
    intersection_over_union = poly1.intersection(poly2).area/(poly1.area + poly2.area - poly1.intersection(poly2).area)
    
    return intersection_over_union

def top_n_similar_hulls(target_hull, hull_list,index_list, n=10):
    """
    Find the top n hulls with the largest overlapping area with the target hull.
    
    Parameters:
    target_hull (scipy.spatial.ConvexHull): The target ConvexHull object.
    hull_list (list): A list of ConvexHull objects to compare against the target hull.
    n (int): The number of hulls to return (default is 10).

    Returns:
    list: A list of tuples containing the ConvexHull objects and their overlapping area with the target hull.
    """
    # Normalize the target hull
    target_hull_normalized = normalize_hull(target_hull)
    
    # Calculate the overlapping area for each normalized hull in the list
    areas = [(hull,index,  overlapping_area(target_hull_normalized, normalize_hull(hull))) for (hull, index) in zip(hull_list,index_list)]
    
    # Sort the hulls by the overlapping area in descending order
    areas_sorted = sorted(areas, key=lambda x: x[2], reverse=True)
    
    # Return the top n hulls
    return areas_sorted[:n]


def convex_hull(df: pd.DataFrame, regex: str = "^home", num_players: int = None):
    """
    Computes convex hulls for player positions, allowing the selection of a subset of players.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing player positions.
    regex (str): A regex pattern to filter player positions in the DataFrame.
    num_players (int): The number of players to include in the convex hull (optional).
    
    Returns:
    list: A list of ConvexHull objects for each frame of data.
    """
    # Filter columns based on the regex
    df = df.filter(regex=regex)
    np_data = df.to_numpy()  # Convert the DataFrame to a NumPy array
    points = []

    # Process each frame of data to extract player positions
    for row in np_data:
        row = row[~np.isnan(row)]  # Remove NaN values (incomplete player positions)
        player_positions = list(zip(row[0::2], row[1::2]))  # Create (x, y) pairs

        # If num_players is specified, limit the number of players
        if num_players is not None and len(player_positions) > num_players:
            # Sort players by their distance to the center of the field (or other criteria)
            center = np.mean(player_positions, axis=0)  # Calculate the central point (e.g., mean position)
            player_positions = sorted(player_positions, key=lambda pos: np.linalg.norm(np.array(pos) - center))
            player_positions = player_positions[:num_players]  # Select the top N closest players

        points.append(player_positions)

    # Compute convex hulls for each frame
    hulls = []
    for data in points:
        if len(data) >= 3:  # Convex hull requires at least 3 points
            hulls.append(ConvexHull(data))

    return hulls, df.index.to_numpy()

def ripley_k_for_hulls(hulls):
    """
    Compute Ripley's K for a list of convex hulls.

    Parameters:
    hulls (list): A list of ConvexHull objects.

    Returns:
    list: A list of Ripley's K values for each convex hull.
    """
    
    radii = np.arange(0, 34)  # Define radii for Ripley's K
    width = 105.0  # Pitch width
    height = 68.0  # Pitch height

    k_vals = []
    
    # Ensure that we are working with the convex hull objects and not tuples
    for hull_tuple in hulls:
        hull = hull_tuple[0]  # Access the actual convex hull object (first element of the tuple)
        
        points = hull.points  # Get player positions from the convex hull
        flattened_points = points.flatten()  # Flatten into 1D array (x_1, y_1, x_2, y_2, ...)
        
        # Calculate Ripley's K for this hull
        k_val = ripley_k(flattened_points, radii, width, height)
        
        k_vals.append(k_val)
    
    return k_vals
