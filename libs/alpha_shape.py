import alphashape
from shapely.geometry import Polygon
import numpy as np
import gudhi as gd
import matplotlib.pyplot as plt
from mplsoccer import Pitch
import pandas as pd
import matplotlib.patches as patches


# ImpLememted using the source:
# https://alphashape.readthedocs.io/en/latest/alphashape.html
def alpha_shape(df: pd.DataFrame, regex: str = "^home", num_players: int = None, alpha: float = 0.1):
    """
    Computes alpha shapes for player positions, allowing the selection of a subset of players.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing player positions.
    regex (str): A regex pattern to filter player positions in the DataFrame.
    num_players (int): The number of players to include in the alpha shape (optional).
    alpha (float): The alpha value for the alpha shape algorithm.
    
    Returns:
    list: A list of alpha shape polygons for each frame of data.
    """
    # Filter columns based on the regex
    df = df.filter(regex=regex)
    np_data = df.to_numpy()  # Convert the DataFrame to a NumPy array
    points = []
    print(len(np_data))
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

    # Compute alpha shapes for each frame
    alpha_shapes = []
    for data in points:
        if len(data) >= 3:  # Alpha shape requires at least 3 points
            # Compute alpha shape for the points
            alpha_shape_polygon = alphashape.alphashape(data, alpha)
            alpha_shapes.append(alpha_shape_polygon)

    print(points)
    return alpha_shapes, df.index.to_numpy()



# Implemented using the source:
# https://gudhi.inria.fr/python/latest/alpha_complex_user.html
# Define a function to extract player positions and apply alpha complex with circles
def alpha_complex_on_frame(df: pd.DataFrame, regex: str = "^home", frame_idx: int = 0, max_alpha_square=2):
    """
    Applies the Alpha Complex method on the player positions in a specific frame of data.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing player positions.
    regex (str): A regex pattern to filter player positions in the DataFrame.
    frame_idx (int): The specific frame (row) in the DataFrame to visualize.
    max_alpha_square (float): Maximum squared alpha value for the Alpha Complex.
    """
    # Filter columns based on the regex to get player positions
    players = df.iloc[frame_idx].filter(regex=regex).dropna().values.reshape(-1, 2)

    # Create Alpha Complex from points (player positions)
    alpha_complex = gd.AlphaComplex(points=players)
    simplex_tree = alpha_complex.create_simplex_tree(max_alpha_square=max_alpha_square)

    # Create a plot with the football pitch
    football_pitch = Pitch(pitch_type='skillcorner', pitch_length=105, pitch_width=68, axis=True, label=True, line_color="white", pitch_color="grass")
    fig, ax = football_pitch.draw(figsize=(10, 7))

    # Plot player points
    ax.scatter(players[:, 0], players[:, 1], color='red', s=100, zorder=3, label="Player Positions")

    # Loop through simplices to plot edges (1D)
    for simplex in simplex_tree.get_skeleton(1):  # 1D simplices (edges)
        if len(simplex[0]) == 2:  # Only plot edges
            vertices = [players[vertex] for vertex in simplex[0]]
            vertices = np.array(vertices)
            ax.plot(vertices[:, 0], vertices[:, 1], 'b-', lw=2)  # Plot the edges (lines)

    # Loop through simplices and highlight triangles (2D)
    for simplex in simplex_tree.get_skeleton(2):  # 2D simplices (triangles)
        if len(simplex[0]) == 3:  # We want only triangles (2D simplices)
            vertices = [players[vertex] for vertex in simplex[0]]
            vertices = np.array(vertices)
            polygon = patches.Polygon(vertices, closed=True, facecolor="lightblue", edgecolor="blue", lw=2, alpha=0.3)
            ax.add_patch(polygon)

    # Plot circles around each player, proportional to the square root of max_alpha_square
    for point in players:
        circle = plt.Circle(point, np.sqrt(max_alpha_square), color='yellow', fill=False, linestyle='--', alpha=0.7)
        ax.add_patch(circle)

    plt.legend()
    plt.title(f"Alpha Complex for Frame {frame_idx} with Alpha = sqrt({max_alpha_square})")
    plt.show()

def alpha_complex(df, regex="^home", max_alpha_square=2):
    """
    Computes Alpha Complexes for player positions, allowing the selection of a subset of players.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing player positions.
    regex (str): A regex pattern to filter player positions in the DataFrame.
    max_alpha_square (float): Maximum alpha value for the Alpha Complex.
    
    Returns:
    tuple: A tuple containing a list of Alpha Complexes and their indices.
    """
    # Filter columns based on the regex
    df_filtered = df.filter(regex=regex)
    points_list = df_filtered.to_numpy()  # Convert to numpy
    
    alpha_complexes = []
    for points in points_list:
        points = points[~np.isnan(points)]  # Remove NaNs
        player_positions = list(zip(points[0::2], points[1::2]))  # Create (x, y) pairs
        
        if len(player_positions) >= 3:  # Alpha Complex requires at least 3 points
            alpha_complex_obj = gd.AlphaComplex(points=np.array(player_positions))
            simplex_tree = alpha_complex_obj.create_simplex_tree(max_alpha_square=max_alpha_square)
            alpha_complexes.append(simplex_tree)
    
    return alpha_complexes, df.index.to_numpy()

from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
from shapely.ops import unary_union
from shapely.affinity import translate

def normalize_geometry(geometry):
    """
    Normalize a geometry (Polygon, MultiPolygon, or GeometryCollection) by
    translating it such that its centroid is at (0, 0).

    Parameters:
    - geometry: A geometry object (Polygon, MultiPolygon, or GeometryCollection)
    
    Returns:
    - normalized_geometry: The normalized geometry centered at (0, 0)
    """
    # Ensure the geometry is valid and has area
    if geometry.is_empty or geometry.area == 0:
        return geometry  # No normalization for empty or zero-area geometries
    
    # Calculate the centroid of the geometry
    centroid = geometry.centroid
    centroid_x, centroid_y = centroid.x, centroid.y
    
    # Translate the geometry so that the centroid moves to (0, 0)
    normalized_geometry = translate(geometry, xoff=-centroid_x, yoff=-centroid_y)
    
    return normalized_geometry



def intersection_over_union(geom1, geom2):
    """
    Compute the Intersection over Union (IoU) for two geometries.
    
    Parameters:
    - geom1: A geometry object (Polygon, MultiPolygon, or GeometryCollection)
    - geom2: A geometry object (Polygon, MultiPolygon, or GeometryCollection)
    
    Returns:
    - IoU: Intersection over Union value (float)
    """
    # Handle GeometryCollections by combining valid geometries with area
    if isinstance(geom1, GeometryCollection):
        geom1 = unary_union([g for g in geom1.geoms if g.is_valid and g.area > 0 and isinstance(g, (Polygon, MultiPolygon))])
    
    if isinstance(geom2, GeometryCollection):
        geom2 = unary_union([g for g in geom2.geoms if g.is_valid and g.area > 0 and isinstance(g, (Polygon, MultiPolygon))])
    
    # Ensure that both geometries have area (ignore 1D or 0D geometries)
    if geom1.is_empty or geom2.is_empty or geom1.area == 0 or geom2.area == 0:
        return 0.0
    
    # Compute intersection and union
    intersection = geom1.intersection(geom2)
    union = geom1.union(geom2)
    
    # Calculate IoU (intersection area over union area)
    if union.area == 0:  # Avoid division by zero
        return 0.0
    
    iou = intersection.area / union.area
    return iou


def top_n_similar_geometries(target_geometry, geometry_list, index_list, n=10):
    """
    Find the top n geometries with the largest overlapping area with the target geometry.
    
    Parameters:
    - target_geometry (Polygon, MultiPolygon, or GeometryCollection): The target geometry object.
    - geometry_list (list): A list of geometry objects (Polygon, MultiPolygon, or GeometryCollection) to compare against.
    - index_list (list): A list of indices corresponding to each geometry in geometry_list.
    - n (int): The number of geometries to return (default is 10).
    
    Returns:
    - list: A list of tuples containing the geometry object, its index, and the overlapping area with the target geometry.
    """
    # Normalize the target geometry
    target_geometry_normalized = normalize_geometry(target_geometry)
    
    # Calculate the overlapping area for each normalized geometry in the list
    areas = []
    for geometry, index in zip(geometry_list, index_list):
        normalized_geometry = normalize_geometry(geometry)
        overlapping_area_value = intersection_over_union(target_geometry_normalized, normalized_geometry)
        areas.append((geometry, index, overlapping_area_value))
    
    # Sort the geometries by the overlapping area in descending order
    areas_sorted = sorted(areas, key=lambda x: x[2], reverse=True)
    
    # Return the top n geometries
    return areas_sorted[:n]