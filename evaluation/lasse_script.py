#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import os
import pandas as pd
import vlc
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QLabel, QHBoxLayout
from PyQt5.QtCore import QTimer

# Video offsets dictionary
kickoffs = {
    'Denmark_England': {'1H': 8.9, '2H': ((60 + 2) * 60 + 31) - 2700},
    'Slovenia_Denmark': {'1H': 8.2, '2H': ((60 + 1) * 60 + 55) - 2700},
    'Germany_Denmark': {'1H': 8.9, '2H': ((60 + 27) * 60 + 36) - 2700},
    'Denmark_Serbia': {'1H': 9, '2H': ((60 + 2) * 60 + 27) - 2700}
}

class VideoPlayer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VLC Video Player")
        self.setGeometry(100, 100, 800, 600)

        # Set up VLC instance
        self.instance = vlc.Instance()
        self.player = self.instance.media_player_new()

        # Create video widget
        self.video_widget = QWidget(self)

        # Delay linking VLC to video widget
        QTimer.singleShot(100, self.initialize_vlc)

        # UI elements
        self.create_ui()

        # State management
        self.dataframe = None
        self.current_index = -1
        self.reference_clip = None
        self.ratings = []  # List to store user ratings

    def initialize_vlc(self):
        """Bind VLC media player to the video widget."""
        try:
            self.video_widget.ensurePolished()
            if sys.platform.startswith("darwin"):
                self.player.set_nsobject(int(self.video_widget.winId()))
            elif sys.platform.startswith("win"):
                self.player.set_hwnd(self.video_widget.winId())
            else:
                self.player.set_xwindow(self.video_widget.winId())
            print("VLC player successfully linked to video widget.")
        except Exception as e:
            print(f"Error linking VLC player: {e}")

    def create_ui(self):
        """Create UI controls and layout."""
        layout = QVBoxLayout()

        # Add the video widget
        layout.addWidget(self.video_widget)

        # Add a label to display the current status
        self.video_label = QLabel("No video playing.")
        layout.addWidget(self.video_label)

        # Add control buttons
        controls_layout = QHBoxLayout()

        play_button = QPushButton("Play Reference")
        play_button.clicked.connect(self.play_reference_clip)
        controls_layout.addWidget(play_button)

        next_button = QPushButton("Next")
        next_button.clicked.connect(self.next_clip)
        controls_layout.addWidget(next_button)

        layout.addLayout(controls_layout)

        # Add Yes/No buttons
        rating_layout = QHBoxLayout()
        self.yes_button = QPushButton("Yes")
        self.yes_button.clicked.connect(lambda: self.rate_clip(1))
        self.yes_button.setEnabled(False)  # Initially disabled
        rating_layout.addWidget(self.yes_button)

        self.no_button = QPushButton("No")
        self.no_button.clicked.connect(lambda: self.rate_clip(0))
        self.no_button.setEnabled(False)  # Initially disabled
        rating_layout.addWidget(self.no_button)

        layout.addLayout(rating_layout)

        # Set the layout to a central container
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def load_clips(self, folder_path):
        """Load reference clip and additional clips."""
        # Load reference info
        ref_file = os.path.join(folder_path, "situation-info.txt")
        with open(ref_file, "r") as f:
            lines = f.readlines()
            match_name = lines[0].split(":")[1].strip()
            time_of_situation = lines[1].split(":")[1].strip()
            self.reference_clip = {
                "match_name": match_name,
                "time": self.time_to_seconds(time_of_situation)
            }

        # Load best clips
        csv_file = os.path.join(folder_path, "best.csv")
        self.dataframe = pd.read_csv(csv_file)

    def play_clip(self, match_name, time, half):
        """Play a specific clip for 10 seconds, starting 5 seconds before the timestamp."""
        offset = kickoffs[match_name][half]
        start_time = time + offset - 5  # Start 5 seconds before the timestamp

        # Ensure start time is not negative
        start_time = max(start_time, 0)

        video_path = f"vids/{match_name}.mov"
        media = self.instance.media_new(video_path)
        self.player.set_media(media)
        self.player.play()

        print(f"Starting playback: {match_name}, {half} @ {start_time}s")

        # Delay before seeking (VLC requires this)
        QTimer.singleShot(1000, lambda: self.seek_and_confirm(start_time))

        # Update UI
        self.video_label.setText(f"Playing: {match_name}, {half} @ {start_time}s (10s clip)")
        self.enable_rating_buttons()

    def seek_and_confirm(self, start_time):
        """Seek to the start time, confirm playback state, and start stop timer."""
        self.player.set_time(int(start_time * 1000))
        print(f"Seeked to {start_time}s, Player state: {self.player.get_state()}")

        # Schedule stop 10 seconds after seeking
        QTimer.singleShot(10000, self.stop_clip)

    def stop_clip(self):
        """Stop playback."""
        self.player.stop()
        self.video_label.setText("Playback stopped.")

    def enable_rating_buttons(self):
        """Enable Yes/No buttons."""
        self.yes_button.setEnabled(True)
        self.no_button.setEnabled(True)

    def disable_rating_buttons(self):
        """Disable Yes/No buttons."""
        self.yes_button.setEnabled(False)
        self.no_button.setEnabled(False)

    def rate_clip(self, rating):
        """Store the user's rating for the current clip."""
        self.ratings.append(rating)
        self.disable_rating_buttons()

        if self.current_index + 1 >= len(self.dataframe):
            # All clips rated, save ratings to the DataFrame
            self.dataframe["rating"] = self.ratings
            print("Ratings saved to DataFrame:")
            print(self.dataframe)
        else:
            # Proceed to the next clip
            self.next_clip()

    def play_reference_clip(self):
        """Play the reference clip."""
        if self.reference_clip:
            match_name = self.reference_clip["match_name"]
            time = self.reference_clip["time"]
            half = "1H" if time < 2700 else "2H"
            self.play_clip(match_name, time, half)

    def next_clip(self):
        """Play the next clip in the list."""
        if self.dataframe is not None:
            self.current_index += 1
            if self.current_index < len(self.dataframe):
                row = self.dataframe.iloc[self.current_index]
                match_name = row["match_name"]
                time = row["Time [s]"]
                half = row["half_team"]
                self.play_clip(match_name, time, half)
            else:
                self.video_label.setText("No more clips to play.")
                self.current_index = -1  # Reset for replay

    @staticmethod
    def time_to_seconds(time_str):
        """Convert time in 'H-M-S' format to seconds."""
        h, m, s = map(int, time_str.split("-"))
        return h * 3600 + m * 60 + s


# In[2]:


# ! ls til_Lasse_4/Til_LASSE_breakthrough-even-Denmark_England-0-46-06
# Til_LASSE_breakthrough-even-Denmark_Serbia-0-47-40
# Til_LASSE_breakthrough-space-Denmark_England-0-30-51
# Til_LASSE_breakthrough-space-Denmark_Serbia-0-15-26
# Til_LASSE_breakthrough-wb-Denmark_England-0-07-22
# Til_LASSE_cross-from-cb-Denmark_Serbia-0-41-40


# In[ ]:


p = "til_Lasse_4/Til_LASSE_cross-from-cb-Denmark_Serbia-0-41-40"

# Run the player
app = QApplication(sys.argv)
player = VideoPlayer()
player.load_clips(p)
player.show()
sys.exit(app.exec_())

# BASELINE
#    Unnamed: 0        match_name  Time [s] half_team        Time Min  rating
# 0      534276   Germany_Denmark   4149.76        2H  1:09:09.760000       0
# 1      198396   Denmark_England   2203.96        1H  0:36:43.960000       0
# 2      310752    Denmark_Serbia   1043.52        1H  0:17:23.520000       0
# 3       12444  Slovenia_Denmark    497.76        1H  0:08:17.760000       0
# 4      534288   Germany_Denmark   4150.24        2H  1:09:10.240000       0
# 5      310776    Denmark_Serbia   1044.48        1H  0:17:24.480000       0
# 6      150324   Denmark_England    281.08        1H  0:04:41.080000       0
# 7      104952  Slovenia_Denmark   4137.44        2H  1:08:57.440000       0
# 8      181908   Denmark_England   1544.44        1H  0:25:44.440000       0
# 9      150336   Denmark_England    281.56        1H  0:04:41.560000       0

# BEST
#    Unnamed: 0        match_name  Time [s] half_team        Time Min  rating
# 0      189912   Denmark_England   1864.60        1H  0:31:04.600000       0
# 1      331872    Denmark_Serbia   1888.32        1H  0:31:28.320000       0
# 2      435768   Germany_Denmark    279.12        1H  0:04:39.120000       0
# 3      511620   Germany_Denmark   3243.52        2H  0:54:03.520000       0
# 4      174636   Denmark_England   1253.56        1H  0:20:53.560000       0
# 5      374364    Denmark_Serbia   3529.72        2H  0:58:49.720000       0
# 6       45828  Slovenia_Denmark   1833.12        1H  0:30:33.120000       0
# 7      393528    Denmark_Serbia   4296.28        2H  1:11:36.280000       1
# 8      291420    Denmark_Serbia    270.24        1H  0:04:30.240000       0
# 9      297408    Denmark_Serbia    509.76        1H  0:08:29.760000       0


# In[ ]:


p = "til_Lasse_4/Til_LASSE_breakthrough-wb-Denmark_England-0-07-22"

# Run the player
app = QApplication(sys.argv)
player = VideoPlayer()
player.load_clips(p)
player.show()
sys.exit(app.exec_())

# BASELINE
#    Unnamed: 0       match_name  Time [s] half_team        Time Min  rating
# 0      525480  Germany_Denmark   3797.92        2H  1:03:17.920000       0
# 1      436680  Germany_Denmark    315.60        1H  0:05:15.600000       0
# 2      350448   Denmark_Serbia   2631.36        1H  0:43:51.360000       0
# 3      257904  Denmark_England   4516.64        2H  1:15:16.640000       0
# 4      257940  Denmark_England   4518.08        2H  1:15:18.080000       0
# 5      177564  Denmark_England   1370.68        1H  0:22:50.680000       0
# 6      257952  Denmark_England   4518.56        2H  1:15:18.560000       0
# 7      257916  Denmark_England   4517.12        2H  1:15:17.120000       0
# 8      257964  Denmark_England   4519.04        2H  1:15:19.040000       0
# 9      257928  Denmark_England   4517.60        2H  1:15:17.600000       0

# BEST
#    Unnamed: 0        match_name  Time [s] half_team        Time Min  rating
# 0       97896  Slovenia_Denmark   3855.20        2H  1:04:15.200000       0
# 1      187848   Denmark_England   1782.04        1H  0:29:42.040000       1
# 2      137736  Slovenia_Denmark   5448.80        2H  1:30:48.800000       1
# 3      408156    Denmark_Serbia   4881.40        2H  1:21:21.400000       0
# 4      501864   Germany_Denmark   2853.28        2H  0:47:33.280000       0
# 5      173868   Denmark_England   1222.84        1H  0:20:22.840000       1
# 6      378972    Denmark_Serbia   3714.04        2H  1:01:54.040000       0
# 7       52500  Slovenia_Denmark   2100.00        1H         0:35:00       1
# 8      139692  Slovenia_Denmark   5527.04        2H  1:32:07.040000       0
# 9      246396   Denmark_England   4056.32        2H  1:07:36.320000       0


# In[3]:


p = "til_Lasse_4/Til_LASSE_breakthrough-space-Denmark_Serbia-0-15-26"

# Run the player
app = QApplication(sys.argv)
player = VideoPlayer()
player.load_clips(p)
player.show()
sys.exit(app.exec_())

# BASELINE
#    Unnamed: 0        match_name  Time [s] half_team        Time Min  rating
# 0       69888  Slovenia_Denmark   2734.88        2H  0:45:34.880000       0
# 1      429120   Germany_Denmark     13.20        1H  0:00:13.200000       0
# 2      465600   Germany_Denmark   1472.40        1H  0:24:32.400000       0
# 3       69876  Slovenia_Denmark   2734.40        2H  0:45:34.400000       0
# 4      465624   Germany_Denmark   1473.36        1H  0:24:33.360000       0
# 5      465648   Germany_Denmark   1474.32        1H  0:24:34.320000       0
# 6       69864  Slovenia_Denmark   2733.92        2H  0:45:33.920000       0
# 7      465612   Germany_Denmark   1472.88        1H  0:24:32.880000       0
# 8      429108   Germany_Denmark     12.72        1H  0:00:12.720000       0
# 9      465636   Germany_Denmark   1473.84        1H  0:24:33.840000       0

# BEST
#    Unnamed: 0        match_name  Time [s] half_team        Time Min  rating
# 0      418224    Denmark_Serbia   5284.12        2H  1:28:04.120000       0
# 1      286692    Denmark_Serbia     81.12        1H  0:01:21.120000       0
# 2      370620    Denmark_Serbia   3379.96        2H  0:56:19.960000       0
# 3       38988  Slovenia_Denmark   1559.52        1H  0:25:59.520000       1
# 4      534288   Germany_Denmark   4150.24        2H  1:09:10.240000       1
# 5      498384   Germany_Denmark   2714.08        2H  0:45:14.080000       0
# 6      219408   Denmark_England   2976.80        2H  0:49:36.800000       0
# 7      384432    Denmark_Serbia   3932.44        2H  1:05:32.440000       0
# 8      456060   Germany_Denmark   1090.80        1H  0:18:10.800000       0
# 9      212844   Denmark_England   2714.24        2H  0:45:14.240000       0


# In[3]:


p = "til_Lasse_4/Til_LASSE_breakthrough-space-Denmark_England-0-30-51"

# Run the player
app = QApplication(sys.argv)
player = VideoPlayer()
player.load_clips(p)
player.show()
sys.exit(app.exec_())

# BASELINE
#    Unnamed: 0        match_name  Time [s] half_team        Time Min  rating
# 0      160344   Denmark_England    681.88        1H  0:11:21.880000       0
# 1      112464  Slovenia_Denmark   4437.92        2H  1:13:57.920000       0
# 2      360168    Denmark_Serbia   2961.88        2H  0:49:21.880000       0
# 3      500460   Germany_Denmark   2797.12        2H  0:46:37.120000       0
# 4      160356   Denmark_England    682.36        1H  0:11:22.360000       0
# 5      360192    Denmark_Serbia   2962.84        2H  0:49:22.840000       0
# 6      160332   Denmark_England    681.40        1H  0:11:21.400000       0
# 7      500448   Germany_Denmark   2796.64        2H  0:46:36.640000       0
# 8      360156    Denmark_Serbia   2961.40        2H  0:49:21.400000       0
# 9      360180    Denmark_Serbia   2962.36        2H  0:49:22.360000       0

# BEST 
#    Unnamed: 0        match_name  Time [s] half_team        Time Min  rating
# 0      181032   Denmark_England   1509.40        1H  0:25:09.400000       0
# 1      253392   Denmark_England   4336.16        2H  1:12:16.160000       0
# 2      102036  Slovenia_Denmark   4020.80        2H  1:07:00.800000       0
# 3      534696   Germany_Denmark   4166.56        2H  1:09:26.560000       0
# 4      309948    Denmark_Serbia   1011.36        1H  0:16:51.360000       0
# 5      243480   Denmark_England   3939.68        2H  1:05:39.680000       1
# 6      176124   Denmark_England   1313.08        1H  0:21:53.080000       0
# 7        2568  Slovenia_Denmark    102.72        1H  0:01:42.720000       0
# 8      201324   Denmark_England   2321.08        1H  0:38:41.080000       0
# 9       58164  Slovenia_Denmark   2326.56        1H  0:38:46.560000       0


# In[ ]:


p = "til_Lasse_4/Til_LASSE_breakthrough-even-Denmark_Serbia-0-47-40"

# Run the player
app = QApplication(sys.argv)
player = VideoPlayer()
player.load_clips(p)
player.show()
sys.exit(app.exec_())

# BASELINE
#    Unnamed: 0        match_name  Time [s] half_team        Time Min  rating
# 0       70824  Slovenia_Denmark   2772.32        2H  0:46:12.320000       0
# 1       28176  Slovenia_Denmark   1127.04        1H  0:18:47.040000       0
# 2       28440  Slovenia_Denmark   1137.60        1H  0:18:57.600000       0
# 3       70812  Slovenia_Denmark   2771.84        2H  0:46:11.840000       0
# 4       28152  Slovenia_Denmark   1126.08        1H  0:18:46.080000       0
# 5       70836  Slovenia_Denmark   2772.80        2H  0:46:12.800000       0
# 6       28188  Slovenia_Denmark   1127.52        1H  0:18:47.520000       0
# 7       28200  Slovenia_Denmark   1128.00        1H         0:18:48       0
# 8       28164  Slovenia_Denmark   1126.56        1H  0:18:46.560000       0
# 9      192192   Denmark_England   1955.80        1H  0:32:35.800000       0

# BEST 
#    Unnamed: 0        match_name  Time [s] half_team        Time Min  rating
# 0      437160   Germany_Denmark    334.80        1H  0:05:34.800000       1
# 1      501156   Germany_Denmark   2824.96        2H  0:47:04.960000       0
# 2      414132    Denmark_Serbia   5120.44        2H  1:25:20.440000       1
# 3       27060  Slovenia_Denmark   1082.40        1H  0:18:02.400000       0
# 4      560184   Germany_Denmark   5186.08        2H  1:26:26.080000       1
# 5      538176   Germany_Denmark   4305.76        2H  1:11:45.760000       0
# 6       12084  Slovenia_Denmark    483.36        1H  0:08:03.360000       0
# 7      269112   Denmark_England   4964.96        2H  1:22:44.960000       0
# 8      168132   Denmark_England    993.40        1H  0:16:33.400000       0
# 9      249612   Denmark_England   4184.96        2H  1:09:44.960000       0


# In[4]:


p = "til_Lasse_4/Til_LASSE_breakthrough-even-Denmark_England-0-46-06"

# Run the player
app = QApplication(sys.argv)
player = VideoPlayer()
player.load_clips(p)
player.show()
sys.exit(app.exec_())

#BASELINE
#    Unnamed: 0        match_name  Time [s] half_team        Time Min  rating
# 0      546228   Germany_Denmark   4627.84        2H  1:17:07.840000       0
# 1      140340  Slovenia_Denmark   5552.96        2H  1:32:32.960000       0
# 2      140352  Slovenia_Denmark   5553.44        2H  1:32:33.440000       0
# 3      299904    Denmark_Serbia    609.60        1H  0:10:09.600000       0
# 4      299892    Denmark_Serbia    609.12        1H  0:10:09.120000       0
# 5      538368   Germany_Denmark   4313.44        2H  1:11:53.440000       0
# 6      140328  Slovenia_Denmark   5552.48        2H  1:32:32.480000       0
# 7      426396    Denmark_Serbia   5611.00        2H         1:33:31       0
# 8      285864    Denmark_Serbia     48.00        1H         0:00:48       0
# 9       28776  Slovenia_Denmark   1151.04        1H  0:19:11.040000       0

# BEST
# 0      469368   Germany_Denmark   1623.12        1H  0:27:03.120000       1
# 1      560100   Germany_Denmark   5182.72        2H  1:26:22.720000       0
# 2      354264    Denmark_Serbia   2725.72        2H  0:45:25.720000       0
# 3      151176   Denmark_England    315.16        1H  0:05:15.160000       0
# 4      549012   Germany_Denmark   4739.20        2H  1:18:59.200000       0
# 5       98520  Slovenia_Denmark   3880.16        2H  1:04:40.160000       0
# 6        1236  Slovenia_Denmark     49.44        1H  0:00:49.440000       0
# 7      498360   Germany_Denmark   2713.12        2H  0:45:13.120000       1
# 8       88476  Slovenia_Denmark   3478.40        2H  0:57:58.400000       1
# 9       63708  Slovenia_Denmark   2548.32        1H  0:42:28.320000       0


# In[ ]:


# p = "TIL_LASSE_3/Til_LASSE_breakthrough-even-Denmark_England-0-46-06"

# # Run the player
# app = QApplication(sys.argv)
# player = VideoPlayer()
# player.load_clips(p)
# player.show()
# sys.exit(app.exec_())

#   Unnamed: 0        match_name  Time [s] half_team        Time Min  rating
# 0      354264    Denmark_Serbia   2725.72        2H  0:45:25.720000       0
# 1       63708  Slovenia_Denmark   2548.32        1H  0:42:28.320000       0
# 2      560100   Germany_Denmark   5182.72        2H  1:26:22.720000       0
# 3       98520  Slovenia_Denmark   3880.16        2H  1:04:40.160000       0
# 4      549012   Germany_Denmark   4739.20        2H  1:18:59.200000       1
# 5      469368   Germany_Denmark   1623.12        1H  0:27:03.120000       1
# 6      498360   Germany_Denmark   2713.12        2H  0:45:13.120000       1
# 7      151176   Denmark_England    315.16        1H  0:05:15.160000       0
# 8       88476  Slovenia_Denmark   3478.40        2H  0:57:58.400000       0
# 9        1236  Slovenia_Denmark     49.44        1H  0:00:49.440000       0


# In[6]:


# p = "TIL_LASSE_3/Til_LASSE_breakthrough-space-Denmark_England-0-30-51"

# # Run the player
# app = QApplication(sys.argv)
# player = VideoPlayer()
# player.load_clips(p)
# player.show()
# sys.exit(app.exec_())

#    Unnamed: 0        match_name  Time [s] half_team        Time Min  rating
# 0      243480   Denmark_England   3939.68        2H  1:05:39.680000       1
# 1      176124   Denmark_England   1313.08        1H  0:21:53.080000       1
# 2       58164  Slovenia_Denmark   2326.56        1H  0:38:46.560000       0
# 3      181032   Denmark_England   1509.40        1H  0:25:09.400000       0
# 4      534696   Germany_Denmark   4166.56        2H  1:09:26.560000       0
# 5      201324   Denmark_England   2321.08        1H  0:38:41.080000       0
# 6      253392   Denmark_England   4336.16        2H  1:12:16.160000       0
# 7      102036  Slovenia_Denmark   4020.80        2H  1:07:00.800000       0
# 8        2568  Slovenia_Denmark    102.72        1H  0:01:42.720000       1
# 9      309948    Denmark_Serbia   1011.36        1H  0:16:51.360000       0


# In[ ]:


# p = "TIL_LASSE_3/Til_LASSE_breakthrough-wb-Denmark_Serbia-0-17-18"

# # Run the player
# app = QApplication(sys.argv)
# player = VideoPlayer()
# player.load_clips(p)
# player.show()
# sys.exit(app.exec_())

#    Unnamed: 0        match_name  Time [s] half_team        Time Min  rating
# 0      299892    Denmark_Serbia    609.12        1H  0:10:09.120000       0
# 1      163848   Denmark_England    822.04        1H  0:13:42.040000       0
# 2      553356   Germany_Denmark   4912.96        2H  1:21:52.960000       0
# 3      490452   Germany_Denmark   3935.12        1H  1:05:35.120000       1
# 4      435816   Germany_Denmark    281.04        1H  0:04:41.040000       0
# 5      497364   Germany_Denmark   4211.60        1H  1:10:11.600000       0
# 6      114516  Slovenia_Denmark   4520.00        2H         1:15:20       0
# 7      241092   Denmark_England   3844.16        2H  1:04:04.160000       0
# 8      368160    Denmark_Serbia   3281.56        2H  0:54:41.560000       0
# 9      296880    Denmark_Serbia    488.64        1H  0:08:08.640000       0


# In[ ]:


get_ipython().system('jupyter nbconvert --to script config_template.ipynb')

