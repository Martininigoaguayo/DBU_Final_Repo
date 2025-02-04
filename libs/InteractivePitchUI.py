from mplsoccer import Pitch
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from libs.weight_generator import *
from libs.alpha_shape import *

class InteractivePitch:
    def __init__(self, match_data, granularity = 48):
        self.match_data : pd.DataFrame = match_data  # Store match data
        # Initialize pitch
        self.football_pitch = Pitch(
            pitch_type='skillcorner', pitch_length=105, pitch_width=68,
            axis=True, label=True, line_color="white", pitch_color="grass"
        )
        self.fig, self.ax = self.football_pitch.draw(figsize=(10, 7))
        

        self.weights = {}

        self.steps = granularity



        self.selected_index = None
        self.custom_situation = True

        # Data structures for storing points, vectors, situations, and ball position
        self.points = [] 
        self.vectors = []
        self.situations = []
        self.similar_situation_indices = []
        self.ball_position = None
        self.vector_start = None
        
        # Mode flags
        self.draw_vector_mode = False
        self.place_ball_mode = False
        
        # Initialize player dropdowns
        self._initialize_players(match_data)
        
        # Set up UI elements
        self._setup_ui()
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
    
        # Intialize the Alpha Shapes Features on the match
        self.alpha_features, self.alpha_formation_indices = compute_alpha_shape_features(match_data)


    def on_click(self, event):
        if event.inaxes:
            x, y = event.xdata, event.ydata
            if self.place_ball_mode:
                self.ball_position = (x, y)
                self.ax.plot(x, y, 'o', color='green', markersize=10, label="Ball")
                plt.draw()
            elif self.draw_vector_mode:
                if self.vector_start is None:
                    self.vector_start = (x, y)
                    self.ax.plot(x, y, 'bo')
                else:
                    vector_end = (x, y)
                    self.ax.annotate('', xy=vector_end, xytext=self.vector_start,
                                     arrowprops=dict(facecolor='red', shrink=0.05))
                    self.vectors.append((self.vector_start, vector_end))
                    self.vector_start = None
            else:
                self.points.append((x, y))
                self.ax.plot(x, y, 'ro')

    def save_situation(self, _):
        if self.points or self.vectors or self.ball_position:
            self.situations.append({'points': list(self.points), 'vectors': list(self.vectors), 'ball': self.ball_position})
            self._update_situation_dropdown()
            print(f"Situation saved! Total saved situations: {len(self.situations)}")
    
    def toggle_view(self,_):
        if self.custom_situation == True:
            self.custom_situation = False
            self._setup_ui()
        else:
            self.custom_situation = True
            self._setup_ui()
        self.display_ui()
    
    def calculate_wasserstein(self, _):


         # Retrieve selected function from the dropdown
        selected_function = self.function_dropdown.value
        weighting_function = {
            "control": lambda x,sum = 0: 1,
            "function_1": linear_weighting,
            "function_2": inverse_weighting,
            "function_3": inverse_exponential_weighting
        }[selected_function]

        weights = None


        if selected_function in self.weights:
            weights = self.weights[selected_function]
        else:
            #Sample match data at interval
            weights = calculate_weights(self.match_data[::self.steps],weighting_function)
            self.weights[selected_function] = weights


        #### Return indices of neighbours from queried situtations, either with custom_sitations or RL situations

        if self.situations and self.ball_position and self.custom_situation:
           
            # Prepare clicked row from the picked saved situation
            clicked_situation = self.situations[self.situation_dropdown.value]
            clicked_row = self._situation_to_row(clicked_situation)
       

            indices = most_similar_with_wasserstein_from_row(clicked_row, self.match_data, weights, weighting_function,steps=self.steps)
            print("Wasserstein calculated, closest situations:", indices[:10])  # Display the top 10 closest situations
            self.similar_situation_indices = indices
        else:
            relevant_data = self.match_data
            selected_index = relevant_data[
                (relevant_data["match_name"] == self.match_name_dropdown.value) &
                (relevant_data["half"] == self.match_half_dropdown.value) &
                (relevant_data['minute'] == int(self.chosen_minutes.value)) &
                (relevant_data['Time [s]'] < (int(self.chosen_minutes.value) * 60 + int(self.chosen_seconds.value)) + 0.05) &
                (relevant_data['Time [s]'] > (int(self.chosen_minutes.value) * 60 + int(self.chosen_seconds.value)) - 0.05)
                ].index.to_numpy()[0]
            print(selected_index)
            self.selected_index = selected_index
            indices = most_similar_with_wasserstein(selected_index,relevant_data,weights, weighting_function, steps=self.steps)
            print("Wasserstein calculated, closest situations:", indices[:10])  # Display the top 10 closest situations
            self.similar_situation_indices = indices

            
    def _situation_to_row(self, situation):
        """Convert a saved situation (points and ball position) to a 1D row format compatible with the DataFrame."""
        row = {}
        for i, (x, y) in enumerate(situation['points']):
            row[f'home_{i + 1}_x'] = x
            row[f'home_{i + 1}_y'] = y
        if situation['ball']:
            row['ball_x_team'] = situation['ball'][0]
            row['ball_y_team'] = situation['ball'][1]
        return row

    
    def _initialize_players(self, match_data):
        """Initialize home and away player lists from match data."""
        # Extract player numbers using regex and keep backups
        self.home_player_numbers = list(np.unique([player[:-1] for player in match_data.filter(regex="^home").columns.to_numpy()]))
        self.backup_home_player_numbers = self.home_player_numbers.copy()
        
        self.away_player_numbers = list(np.unique([player[:-1] for player in match_data.filter(regex="^away").columns.to_numpy()]))
        self.backup_away_player_numbers = self.away_player_numbers.copy()
        
        # Dropdowns for selecting players
        self.home_players_dropdown = widgets.Dropdown(
            options=[("Select Player", "")] + [(f"{player}", player) for player in self.home_player_numbers],
            description='Home Player:',
            disabled=False,
        )
        
        self.away_players_dropdown = widgets.Dropdown(
            options=[("Select Player", "")] + [(f"{player}", player) for player in self.away_player_numbers],
            description='Away Player:',
            disabled=False,
        )
        
        # Observe dropdown changes
        self.home_players_dropdown.observe(self.home_player_selected, names='value')
        self.away_players_dropdown.observe(self.away_player_selected, names='value')

    def on_click(self, event):
        if event.inaxes:  # Check if click is inside plot
            x, y = event.xdata, event.ydata  # Get coordinates
            
            if self.place_ball_mode:
                # Place the ball at the clicked position
                self.ball_position = (x, y)
                self.ax.plot(x, y, 'o', color='green', markersize=10, label="Ball" if not any(artist.get_label() == "Ball" for artist in self.ax.get_children()) else "")
                plt.draw()  # Update the plot
                print(f"Ball placed at: {self.ball_position}")

            elif self.draw_vector_mode:
                if self.vector_start is None:  # If no start point, set this as start point
                    self.vector_start = (x, y)
                    self.ax.plot(x, y, 'bo')  # Mark the start point with a blue dot
                else:
                    # If there's already a start point, draw the vector from start to this point
                    vector_end = (x, y)
                    self.ax.annotate('', xy=vector_end, xytext=self.vector_start,
                                     arrowprops=dict(facecolor='red', shrink=0.05))  # Draw vector
                    self.vectors.append((self.vector_start, vector_end))  # Save the vector
                    self.vector_start = None  # Reset the start point
            else:
                self.points.append((x, y))  # Add to list of points
                self.ax.plot(x, y, 'ro')  # Plot the point

    def save_situation(self, _):
        """Save the current situation of points, vectors, and ball position."""
        if self.points or self.vectors or self.ball_position:
            self.situations.append({'points': list(self.points), 'vectors': list(self.vectors), 'ball': self.ball_position})
            self._update_situation_dropdown()  # Update dropdown after saving
            print(f"Situation saved! Total saved situations: {len(self.situations)}")
        else:
            print("No players, vectors, or ball to save!")

    def clear_situation(self, _):
        """Clear the current situation and reset UI elements."""
        self.points = []
        self.vectors = []
        self.players = []
        self.vector_start = None
        self.ball_position = None

        # Reset player lists
        self.home_player_numbers = self.backup_home_player_numbers.copy()
        self.away_player_numbers = self.backup_away_player_numbers.copy()
        
        self._update_player_dropdowns()
        
        # Clear plot and redraw pitch
        self.ax.cla()
        self.football_pitch.draw(ax=self.ax)
        plt.draw()
        print("Cleared the current situation. All players are available for selection again.")

    def toggle_draw_vector(self, _):
        """Toggle vector drawing mode."""
        self.draw_vector_mode = not self.draw_vector_mode
        self.place_ball_mode = False
        if self.draw_vector_mode:
            print("Vector drawing mode enabled. Select start and end points for the vector.")
        else:
            print("Switched to player drawing mode.")

    def toggle_place_ball(self, _):
        """Toggle ball placement mode."""
        self.place_ball_mode = not self.place_ball_mode
        self.draw_vector_mode = False
        if self.place_ball_mode:
            print("Ball placement mode enabled. Click to place the ball on the pitch.")
        else:
            print("Ball placement mode disabled.")

    def select_player(self, player_num, dropdown, player_list):
        """Select and add player, remove from dropdown."""
        self.players.append(player_num)
        player_list.remove(player_num)
        self._update_dropdown_options(dropdown, player_list)

    def remove_player(self, player_num, dropdown, player_list):
        """Remove player from list and add back to dropdown."""
        if player_num in self.players:
            self.players.remove(player_num)
            player_list.append(player_num)
            player_list.sort()
            self._update_dropdown_options(dropdown, player_list)

    def _update_dropdown_options(self, dropdown, player_list):
        """Helper to update dropdown options based on player list."""
        dropdown.options = [("Select Player", "")] + [(f"{player}", player) for player in player_list]
        
    def _update_player_dropdowns(self):
        """Reset player dropdowns to full lists."""
        self._update_dropdown_options(self.home_players_dropdown, self.home_player_numbers)
        self._update_dropdown_options(self.away_players_dropdown, self.away_player_numbers)

    def _update_situation_dropdown(self):
        """Update the situation dropdown with the latest saved situations."""
        self.situation_dropdown.options = [("Select Situation", "")] + [(f"Situation {i+1}", i) for i in range(len(self.situations))]

    def home_player_selected(self, change):
        selected_player = change['new']
        if selected_player:
            self.select_player(selected_player, self.home_players_dropdown, self.home_player_numbers)

    def away_player_selected(self, change):
        selected_player = change['new']
        if selected_player:
            self.select_player(selected_player, self.away_players_dropdown, self.away_player_numbers)

    def load_situation(self, change):
        """Load a saved situation and plot it on the pitch."""
        selected_situation = change['new']
        if selected_situation != "":
            situation = self.situations[selected_situation]
            self.clear_situation(None)  # Clear the current plot first

            # Plot the points
            for x, y in situation['points']:
                self.ax.plot(x, y, 'ro')
                self.points.append((x, y))
            
            # Plot the vectors
            for start, end in situation['vectors']:
                self.ax.annotate('', xy=end, xytext=start, arrowprops=dict(facecolor='red', shrink=0.05))
                self.vectors.append((start, end))
            
            # Plot the ball if it exists
            if situation['ball']:
                bx, by = situation['ball']
                self.ax.plot(bx, by, 'o', color='green', markersize=10, label="Ball")
                self.ball_position = (bx, by)
            
            plt.draw()
            print(f"Loaded Situation {selected_situation + 1}")

    def _setup_ui(self):
        # Buttons and dropdowns
        self.save_button = widgets.Button(description="Save Situation", button_style='success')
        self.clear_button = widgets.Button(description="Clear", button_style='warning')
        self.toggle_vector_button = widgets.Button(description="Toggle Draw Vector", button_style='info')
        self.toggle_ball_button = widgets.Button(description="Place Ball", button_style='primary')

        # Function selection dropdown for Wasserstein
        self.function_dropdown = widgets.Dropdown(
            options=[("Control (1)", "control"), 
                     ("200 - x", "function_1"),
                     ("1 / x", "function_2"),
                     ("exp(-x / 40)", "function_3")],
            description='Wasserstein Function:'
        )

        # Toggle for switching between custom situation and real match input
        self.toggle = widgets.ToggleButton(
            value=False,
            description='Toggle Real Match Input',
            disabled=False,
            button_style='info',  # Options: 'success', 'info', 'warning', 'danger' or ''
            tooltip='Click to toggle between real match input and saved situations',
            icon='check'  # (FontAwesome names without the `fa-` prefix)
        )

        self.interval_length_chooser = widgets.Text(
            value='0',
            placeholder='Type something',
            description='Length of situation in seconds:',
            disabled=False
        )

        # Input boxes for real match input
        self.chosen_seconds = widgets.Text(
            value='0',
            placeholder='Type something',
            description='Seconds:',
            disabled=False
        )
        self.chosen_minutes = widgets.Text(
            value='0',
            placeholder='Type something',
            description='Minutes:',
            disabled=False   
        )
        self.match_name_dropdown = widgets.Dropdown(
            options=[(key, key) for key in np.unique(self.match_data.match_name.to_numpy())],
            description="Match name"
        )
        self.match_half_dropdown = widgets.Dropdown(
            options=[("1st Half", "1H"), ("2nd Half", "2H")],
            description="Half"
        )

        # Calculate button for Wasserstein
        self.calculate_wasserstein_button = widgets.Button(description="Calculate Wasserstein", button_style='info')

        # Situation dropdown to reload saved situations
        self.situation_dropdown = widgets.Dropdown(description="Saved Situations:")
        self.situation_dropdown.observe(self.load_situation, names='value')

        # Connect button events
        self.toggle.observe(self.toggle_view, names='value')
        self.save_button.on_click(self.save_situation)
        self.clear_button.on_click(self.clear_situation)
        self.toggle_vector_button.on_click(self.toggle_draw_vector)
        self.toggle_ball_button.on_click(self.toggle_place_ball)
        self.calculate_wasserstein_button.on_click(self.calculate_wasserstein)

        # Create a container that will hold our dynamic layout.
        self.ui_container = widgets.HBox()

        # Set the initial layout based on the toggle’s value.
        self.update_ui()

        # Display the container in the notebook.
        display(self.ui_container)


    def update_ui(self):
        """
        Updates the UI layout based on the toggle value.

        When toggle.value is True, we show the real match input widgets.
        When toggle.value is False, we show the custom situation widgets.
        """
        if self.toggle.value:
            # Layout for real match input
            self.ui_container.children = [
                widgets.VBox([self.function_dropdown, self.toggle]),
                widgets.VBox([
                    
                    self.chosen_minutes, 
                    self.chosen_seconds, 
                    self.match_half_dropdown, 
                    self.match_name_dropdown,
                    self.interval_length_chooser,
                    self.calculate_wasserstein_button
                ])
            ]
        else:
            # Layout for custom (saved) situations
            self.ui_container.children = [
                widgets.VBox([self.situation_dropdown, self.function_dropdown, self.toggle]),
                widgets.VBox([
                    self.save_button, 
                    self.clear_button, 
                    self.toggle_vector_button, 
                    self.toggle_ball_button,
                    self.interval_length_chooser,
                    self.calculate_wasserstein_button
                ])
            ]


    def toggle_view(self, change):
        """
        Callback function triggered when the toggle’s value changes.
        It calls update_ui() to refresh the layout.
        """
        self.update_ui()


import ipywidgets as widgets
from IPython.display import display, clear_output

class PitchDisplay:
    def __init__(self, df_processed: pd.DataFrame, indices: list):
        """
        Initialize an interactive pitch display with a dropdown for index selection.

        Parameters:
        df_processed (pd.DataFrame): The processed DataFrame containing player and ball positions.
        indices (list): List of indices in the DataFrame to plot.
        """
        self.df_processed = df_processed
        self.indices = indices
        self.selected_index = indices[0]  # Initialize with the first index
        
        # Dropdown widget for index selection
        self.dropdown = widgets.Dropdown(
            options=self.indices,
            description='Select Index:',
            value=self.selected_index
        )
        
        # Output widget to display the pitch
        self.output = widgets.Output()
        
        # Initialize display
        self._initialize_display()
        
    def _initialize_display(self):
        """Set up widgets and initial plot."""
        # Initial plot
        self.update_pitch(self.selected_index)
        
        # Observe dropdown changes
        self.dropdown.observe(self._on_dropdown_change, names='value')
        
        # Display widgets
        display(widgets.VBox([self.dropdown, self.output]))
    
    def update_pitch(self, index):
        """Update the pitch plot based on the selected index."""
        with self.output:
            clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(8, 6))

            # Extract data for selected index
            df_ball_start = self.df_processed.loc[index, ["ball_x", "ball_y"]]
            half = str(self.df_processed.loc[index, 'half'])
            df_current = self.df_processed.filter(regex='^home').loc[index]
            
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
            
            # Plot ball movement over next 120 ticks (or until the end of the data if fewer than 96 rows are left)
            end_idx = min(index + 120, len(self.df_processed))
            df_ball_movement = self.df_processed.loc[index:end_idx, ["ball_x", "ball_y"]]
            
            # Initial ball position (highlighted)
            ax.scatter(df_ball_start["ball_x"], df_ball_start["ball_y"], s=120, color='blue', edgecolors='red', linewidth=2, label='Ball_start')
            
            # Plot ball trail to show movement
            for j in range(1, len(df_ball_movement)):
                x = df_ball_movement.iloc[j]["ball_x"]
                y = df_ball_movement.iloc[j]["ball_y"]
                if pd.notna(x) and pd.notna(y):
                    ax.scatter(x, y, s=100, color='yellow', edgecolors='red', alpha=(j / len(df_ball_movement)), label='Ball' if j == 1 else "")

            # Title and legend
            ax.set_title(f"Half: {half}, Time [s]: {self.df_processed['Time [s]'].iloc[index]}")
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=4)
            plt.show()

    def _on_dropdown_change(self, change):
        """Handle dropdown changes and update the selected index."""
        if change['type'] == 'change' and change['name'] == 'value':
            self.selected_index = change['new']
            self.update_pitch(self.selected_index)
    
    def get_selected_index(self):
        """Getter for the selected index."""
        return self.selected_index
