---
layout: post
title: "NHL Play-by-Play Data Analysis"
date: 2025-10-04
categories: [Data Science, NHL, Python]
author: Lucius Hatherly (lucius.hatherly@umontreal.ca), Sina Vali (Sina.vali@umontreal.ca), Aymen Boustani (aymen.boustani@umontreal.ca), Shivam Ardeshna (shivam.mayurbhai.ardeshna@umontreal.ca)
---

# NHL Play-by-Play Data Analysis

## Introduction  
In this project, we explored NHL play-by-play data between 2016 and 2024.  
We built Python scripts to automatically download, process, and visualize hockey event data. 

The dataset will include details like season, game date, shot type, team and player information. The project to run and analyze the NHL data is divided into the phases: 1. Data Acquisition, 2. Interactive Debugging Tool, 3. Tiday Data, 4. Simple Visualizations, and 5. Advanced Visualizations: Shot Maps. 

This blog will explain each step in detail and be supported with code and figures used for analysis to support any decided upon conclusions. 

## Data Acquisition
The Data Acquition can be divided into these three steps:

1.	Download raw NHL play-by-play data in JSON form via the NHL API
2.	We will then cache this data locally to avoid repeated downloads
3.	Subsequently, convert the data into a clean pandas data frame for analysis

### a) **Step 1: Understanding the NHL API:**

![Step 1 Screenshot]({{ site.baseurl }}/assets/images/image.png)

The NHL Stats API provides play-by-play data at: https://api-web.nhle.com/v1/gamecenter/[GAME_ID]/play-by-play where the GAME_ID will encode season, game type (regular season or playoffs), and game number. We built helper code to create GAME_IDs from the 2016-2017 to the 2023-2024 season for both the regular season and play-offs.

### b) **Step 2: Explaining our Data Fetcher Class**

We then implemented a NHLDataFetcher which handles three primary tasks:

1.	Extracting raw data from the NHL API
2.	Processing JSON into pandas DataFrame
3.	Looping through entire seasons of NHL data to build a combined data set of regular season and playoff games.

![Step 2 Screenshot]({{ site.baseurl }}/assets/images/image-1.png)

![Step 2 Screenshot]({{ site.baseurl }}/assets/images/image-2.png)


### c) **Step 3: Using the developed NHL Class**

Once we establish this class the downloading and processing of the games is much more straightforward.

**Extracting Data from a Single Game**

![Step 3 Screenshot]({{ site.baseurl }}/assets/images/image-3.png)

This will print a clean DataFrame with columns like gameDate, typeDescKey, details_Xcoord, and details_ycoord, amongst other issues.

**Processing Data for an Entire Season**

![Step 3 Screenshot]({{ site.baseurl }}/assets/images/image-4.png)

This now will give us thousands of rows of data (if we did not use .head()) for the 2016 season.

### d) **Review of Key Design Choices:**

i) **Class-based Approach:** This allows the code to be reusable to extract data from regular games and playoffs across multiple seasons. 
ii) **Select usable columns:** Instead of keeping only raw JSON files where the files are all deeply nested, we keep only the most relevant columns like coordinates, player IDs, details_shotType, typeDescKey. 
iii) **Extensibility:** This code can easily be developed to support other endpoints like more player statistics and team info.

In conclusion, when we structure our data around the NHLDataFetcher class, we developed a reproducible pipeline for downloading real-time NHL data that we can use for analysis. This foundation will enable running further data analysis and training machine learning models if necessary.

## Interactive Debugging Tool 

![Interactive Debugging Tool]({{ site.baseurl }}/assets/images/image-5.png)

![Interactive Debugging Tool]({{ site.baseurl }}/assets/images/image-6.png)

![Interactive Debugging Tool]({{ site.baseurl }}/assets/images/image-7.png)

![Interactive Debugging Tool]({{ site.baseurl }}/assets/images/image-9.png)

![Interactive Debugging Tool]({{ site.baseurl }}/assets/images/image-19.png)

![Interactive Debugging Tool]({{ site.baseurl }}/assets/images/image-18.png)

This interactive debugging tool uses ipywidgets to explore NHL play-by-play data by season, game, and event ID.  The tool then plots the event coordinates directly on a rink image. This will allow us to quickly visualize specific plays, along with metadata such as event type and timing, making it much easier to verify and debug the dataset where anomalous or irregular data occurs.

The dropdown menus and sliders automatically update based on available data, enabling smooth navigation through games and events for both the regular season and playoffs. This helped validate data correctness (e.g., shot coordinates) and served as a useful prototype for debugging and data exploration.
## Tidy Data 

![Tidy Data]({{ site.baseurl }}/assets/images/image-11.png)

![Tidy Data]({{ site.baseurl }}/assets/images/image-12.png)

![Tidy Data]({{ site.baseurl }}/assets/images/image-20.png)

![Tidy Data]({{ site.baseurl }}/assets/images/image-21.png)

![Tidy Data]({{ site.baseurl }}/assets/images/image-22.png)

After we retrieved the 3.2 million play-by-play events for the NHL seasons between 2016 to 2024, we proceeded to consolidate these responses inside a JSON Data Frame. Within the data frame, each row represents a single event during the game. This could include details such as the game date, season, period, time, rink coordinates, shot type, and the teams involved in this said game. The resulting dataset (nhl_all_games_data.csv) provides a structured foundation for data analysis — making it much easier to visualize and model the data generated from the nhl games.

To continue the validation of the dataset, we used .info() and .describe() on the data frame. This confirmed 26 well-defined columns, consistent data types, and key statistical distributions for key numeric variables like coordinates (x, y) and player IDs. This again reinforces that our data is reliable and can be seen as a valid stepping stone for downstream analytics and system use. 

The three new features we could add to further enhance the data set are rebound shots, shots off the rush, along with shot distance and angle metrics from the goal’s position. The first of these features helps us check if a new shot occurred within a few seconds of another unsuccessful shot from a close coordinate.Shots off the rush help us understand whether or not a shot occurred soon after a change in puck possession (within 3 to 5 seconds). Lastly, the shot distance and angle metrics from the goal’s position further helps us analyze player tendencies and shot quality. These additional features could enable deeper insights into how different goals are scored throughout different hockey game contexts. 

## Simple Visualizations 

1) ![Simple Visualizations]({{ site.baseurl }}/assets/images/image-15.png)

After analyzing the figure above which covers shot statistics for the NHL 2021-2022 season, we can see that the most dangerous shots are the backhand and tip-in (where backhand seems to have a slightly higher shot success rate %). Both of these shot types have a shot conversion rate of around 9.5%. According to the results of the graph above, we see that the most common shot type is wrist with around 70000 shots throughout the season. We chose this particular figure as bar charts with their thickness and height are good indicators to see the number of different types of shots throughout the hockey season. Additionally, a line chart was selected to show the Shot Success Rate % as a line chart would easily show the trend or general difference across the different types of shots. Note that these results make intuitive sense as wrist shots due to their ease and speed are the most commonly occurring in a hockey game, while backhand and tip-ins are taken close to the goal so are likely more dangerous to take.  

2) ![Simple Visualizations]({{ site.baseurl }}/assets/images/image-16.png)

The above figure shows that across the three seasons from 2018 to 2021, the closer the distance the shot is from the goal, the probability of scoring the goal is much higher. Specifically, when you are 2.5 ft away from the goal, all three seasons have a probability of scoring of more than 20%. Additionally, when you have a shot that is more than 85 ft from the goal, the probability of a goal is less than 5%. Across the three seasons, the general trend of increasing the distance from the goal leads to a lower probability of scoring a goal is the same. However, the 2019-2020 season has the highest probability of scoring when you are around 2.5ft from the goal compared to the other two seasons. As you increase the distance though, the season with the highest to lowest probability of scoring a goal changes, however, the results are always close together (within only 3 to 4% of difference). A line graph was selected to show these figures as it is easy to use a line graph to see the trend for each season's probability of scoring a goal as the distance from the goal increases.

Overall, these results make intuitive sense as the farther you are away from a goal, the less likely you are to score.

3) ![Simple Visualizations]({{ site.baseurl }}/assets/images/image-17.png)

The figure above shows the goal percentage of various shot types compared to the distance from the hockey goal (ft) during the 2018-2019 season. When analyzing this data, we can observe several important trends expanded upon in more detail below.

When we are between 0 to 5 ft from the goal, tip-in (at over a 45% scoring rate) followed by deflected and backhand shots (with scoring both slightly above 30%) are the most dangerous shots. This result makes sense as both these types of the shot usually take place very close to the goal where the goalkeeper has less time to react and save the shot.
While the distance from the goal increases, the scoring % rate decreases for all types of shots. This decreases to on average less than 10% as we get from 40 to 45 ft from the goal. Between 30 to 65 ft from the goal, deflected shots have the highest success rate at scoring the goal at between 10 to 20%. Greater than 65 ft from the goal sees the most successful type of shot changing between snap shots, wrist shots, and backhand shots. Note though, these success rates for distances more than 60 ft from the goal all are 10% or less. Again, this intuitively makes sense as the farther you are from the goal, the less likely you are to score when you shoot.

Wrist shots are the most common type of shot shown throughout the figure. However, wrist shots have a lower success rate within 5 ft to the goal compared to more dangerous shots like tip-in, backhand, and deflected shots.

We selected a bar chart grouped by different shot types as the figure to analyze this question as it is easy for us to use a bar chart to see the trend of % of shots being goals based on distance from the goal broken down by group type.


## Advanced Visualizations 

### Offensive zone plots:

Shot Maps for Season 2016-2017

{% include shotMap2016.html %}

Shot Maps for Season 2017-2018

{% include shotMap2017.html %}

Shot Maps for Season 2018-2019

{% include shotMap2018.html %}

Shot Maps for Season 2019-2020

{% include shotMap2019.html %}

Shot Maps for Season 2020-2021

{% include shotMap2020.html %}

### Interpreting the plots:

From the figures below, we can see the areas where the teams made most of their shots, revealing potential weaknesses in the opposing defenses. They also illustrate how each team’s offensive effectiveness evolved across different seasons.

### Analyzing the Colorado Avalanche shot map for 2016-17 and 2020-21:

In 2016-17, we can see that compared to the league average shooting rate, Avalanch performed worse in most areas, evident by the larger blue areas in the plot. On the contrary, in 2020-21 the plot is dominated by red areas, i.e., Avalanche made more shots in most spots of the rink.

This is validated by considering the change of the team's standing from last to first.

### Comparison between Buffalo Sabres & Tampa Bay Lightning from 2018-19 to 2020-21:

In 2018-19, the Tampa Bay Lightning was more dominant in the center while the Buffalo Sabres, weren't able to make many shots close to the goal.

In the next year, the heatmap of the Lightning was mostly the same, whereas the attack power of the Sabres diminished.

In 2020-21, the Lightning wasn't able to make shots as close as to the goal as previous years, and the Sabres remained mostly the same.

We can glean a lot of information about the offensive capabilities of one team or the defensive prowess of the other team, just by observing the excess shooting rate heat map.
