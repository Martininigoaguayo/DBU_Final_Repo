# DBU FINAL REPOSITORY USER GUIDELINE
Final repository for a search tool for DBU video analysts (prototype_UI.ipynb)

## EXPECTED FOLDER STRUCTURE OF DATA
Example:
DBU_Final_Repo/
│
├── data/ 
│   ├── H_EURO2024GERMANY/
│   │   ├── 2036189_England_Slovenia/
│   │   │   ├── 3-2024-2036200-England_Slovenia.mp4
│   │   │   ├── SRB - ENG.json
│   │   │   ├── tracking_away.csv
│   │   │   ├── tracking_home.csv
│   │   ├── 2036161_Germany_Scotland/
│   │   │   ├── 3-2024-2036161-Germany_Scotland.mp4
│   │   │   ├── GER - SCT.json
│   │   │   ├── tracking_away.csv
│   │   │   ├── tracking_home.csv
│   │   ├── ... (more matches)
├── ...
├── protoype_UI.ipynb
├── ...

## MAIN DATA EXTRACTION
The tool begins with extracting all of the match data from the tracking .csv files from the folders and combining all of the home and away tracking positions into one pandas dataframe. 
**First:**
Define the tournament folder nam
```ruby!  
tournament_folder = "data/H_EURO2024GERMANY"
```
We do that using:
```ruby!
data = compile_team_tracking_data(tournament_folder, "England")
```

Then we extract the data for the specified number of matches from the DataFrame, standardizes player positions so that teams always attack from right to left, regardless of their attacking direction in the first half.
```ruby!
one_match = extract_one_match(data, 4)
```


# META DATA EXTRACTION
We use multiple functions for extracting meta data for the specific team that we are searching for. This team is defined in the *data* variable, in the example it is **"England"**. 
```ruby!
processed_data = process_dataframe(one_match, tournament_folder)

```
Using the *display_match_info(processed_data)* we are able to see the extracted meta data. 
An example below:
```!
Processing match: Denmark_England
Processing match: England_Slovenia
Processing match: England_Slovakia
Processing match: England_Switzerland

=========== Processed Match Data ===========

Match Folder: data/H_EURO2024GERMANY\2036178_Denmark_England
Folder Name: 2036178_Denmark_England

Match Name: Denmark_England

JSON File: data/H_EURO2024GERMANY\2036178_Denmark_England\DEN - ENG.json
JSON Name: DEN - ENG.json

Video File: data/H_EURO2024GERMANY\2036178_Denmark_England\3-2024-2036178-Denmark_England.mp4
Video Name: 3-2024-2036178-Denmark_England.mp4

First Half Start: 9 seconds
Second Half Start: 3751 seconds
...
```

# RUNNING THE INTERACTIVE PITCH
Having preproccesed the data we are able to run our InteractivePitch tool.
```ruby! 
interactive_pitch = InteractivePitch(one_match)
```
Here the user is able to either use the drawing feature or the real match feauture as a way to make/select a target situation.

Using this as a target and the processed data, we can show the final similar situation recommendations using the *show_reccommendation()* function. 
Giving us the final results of our model.