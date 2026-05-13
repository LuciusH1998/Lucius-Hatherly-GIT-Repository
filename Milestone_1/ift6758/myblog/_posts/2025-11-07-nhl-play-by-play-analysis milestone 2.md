---
layout: post
title: "NHL Play-by-Play Data Analysis Milestone 2"
date: 2025-11-07
categories: [Data Science, NHL, Python]
author: Lucius Hatherly (lucius.hatherly@umontreal.ca), Sina Vali (Sina.vali@umontreal.ca), Shivam Ardeshna (shivam.mayurbhai.ardeshna@umontreal.ca)
---

# NHL Play-by-Play Data Analysis Milestone 2 

## Introduction  

For this second milestone, we extend upon the previous work on NHL play-by-play data (2016 – 2024) in Milestone I to move past data acquisition and visualization into feature engineering, statistical modeling, and predictive analysis. The data for Milestone I only focuses on NHL seasons between 2016-2021 rather than all seasons between 2016-2024 like in Milestone I. 

Our main goal for this milestone is to estimate the probability that any given shot produces a goal. This in professional hockey analytics is known as Expected Goals (xG).

To start from building on the cleaned and structured datasets from Milestone 1, we design a series of experiments that extract and refine features such as shot distance, angle, rebound conditions, game context, and power-play situations. These features are then used to train and evaluate different machine-learning models. This first starts with simple Logistic Regression baselines, then moving onto XGBoost classifiers, and lastly exploring advanced and custom models to aim for performance improvement and interpretability.

Carrying on, in order to guarantee reproducibility and experiment transparency, all experiments are monitored using Weights & Biases (Wandb). Specifically this includes metrics, model artifacts, and hyperparameter configurations. Additionally, we visualize each model’s performance using calibration plots, ROC/AUC curves, and goal-rate analyses, where we compare how feature complexity and model selection impacts the predictive accuracy for Expected Goals.

This blog post explains every phase of the modeling pipeline starting with Feature Engineering I, Baseline Models, Feature Engineering II, Advanced Models, Give it your best shot, and Evaluate on test set. 

## Feature Engineering I

When we start our modelling pipeline, we concentrated on two key geometric features of a hockey shot — the distance and the angle relative to the net.
Carrying on from the same data-wrangling tools developed in Milestone 1, we then extracted all the given SHOT and GOAL events from the NHL play-by-play data between the 2016/17 and 2019/20 regular seasons.

For our initial tidied data set, we created four key features:

distance_from_net (ft) – this is the euclidean distance of the shot from the net center

angle_from_net (°) – the angle of the shot relative to the goal centerline

is_goal – binary indicator (1 = goal, 0 = no goal)

empty_net – the binary indicator for whether the net was empty at the time of the shot

The final training dataset was designated as the NHL season data between 2016 to 2020 which contained over 1.28 million shot events, while the 2020/21 season data was reserved as the untouched test set.

### 1 a) Histogram Distribution by Distance 

![Histogram Distribution by Distance]({{ site.baseurl }}/assets/images/image-23.png)

The first histogram displays the shots  binned by distance from the net, separated by goal outcome. The histogram reveals that most shots occur within 10–40 feet of the net, and as expected, goals themselves are concentrated in shorter distances. This makes intuitive sense as it is easy to score goals closer to the net. 
The goal frequency decreases drastically beyond 30 feet, this confirms that shot success probability decreases with distance. 
The orange bars (goals) essentially disappears beyond 60 feet, consistent with the intuition that long-range shots are largely unsuccessful unless they occur under special conditions (e.g. deflections). All in all, this makes intuitive sense as it is harder to score farther away from the net. 

### 1 b) Histogram Distribution by Angle

![Histogram Distribution by Angle]({{ site.baseurl }}/assets/images/image-24.png)

The second histogram displays the plot shot counts by angle, where 0° corresponds to shots taken from directly in front of the hockey net and bigger angles displays positions farther away from the hockey net. 
Moreover, we observe shots taken from smaller angles (specifically between 0–30°) likely display a higher proportion of goals compared to shots taken from larger angles which are more difficult to score from. 
In conclusion, the decrease in the number of shots and goal scoring rate at wider angles conveys how shooting geometry minimizes scoring chances, as goalkeepers could better cover the net at these larger angles. 

### c) Histogram Joint Distribution of Distance and Angle 

![2D Histogram Distribution of Angle and Distance]({{ site.baseurl }}/assets/images/image-25.png)

Lastly, the 2D joint plot (distance * angle) displays that the most shots are clustered around 15–35 feet and angles below 35 to 40°. This forms a dense region of high-frequency shooting activity which is situated directly in front of the net, in hockey this is known as the high-danger area.

The heatmap coloring being more red indicates an increased density of events (and goal probabilities) around this given zone. This further reiterates the idea that shots leading to goals is location dependent. 

### Summary 

All in all, these three histograms reconfirm the reliability of specific engineered geometric features.
These plots reveal interpretable and understandable relationships between shot distance, angle, and the probability of scoring a goal. This lays the foundation for future models in which we can use distance and angles to help predict expected goals (xG).

### 2)

Goal Rate Analysis by Distance and Angle: 

This better analysis understands how shot geometry affects scoring outcomes, specifically we defined the goal rate as

Goal Rate = # Goals/(# Goals + # No-Goals)

where goal rate is a function of both distance from the net and shot angle.

![Plots of Goal Rates Compared to Distance and Angle from Net]({{ site.baseurl }}/assets/images/image-26.png)

i) Plot 1: Goal Rate vs. Distance

Figure 1 displays a steep and nonlinear decline in goal rate as distance from the net increases. The plot shows that shots taken within 10 feet of the crease have an approximately 30 % chance of resulting in a goal — the highest success rate in the dataset Subsequently, the probability of scoring decreases rapidly between 10 ft – 30 ft, and ultimately falling below 10 % past 40 ft. After we go past 60 ft, the goal rate decreases below 5 %. This suggests that shots from long range almost never beat the goalie unless these given shots are to an empty-net or deflections. This rapid exponential-like decay reconfirms that the closeness to goal is the most influential spatial factor for scoring probability.

ii) Plot 2: Goal Rate vs. Angle

Figure 2 displays a complementary trend with respect to goal rate in relation to angle. The strongest goal rates (between 12–14 %) occur for shots taken from key positions (between 0° – 20°) taken directly in front of the net. The success rate for the goal rate vs angle diminishes rapidly as we take shots from larger angles. This goal rate then falls below 6% for shots beyond 60°, where hockey players are likely shooting from behind the goal line or from the rink's sideboards. The small bump in goal rate between 25°–35° could be derived from cross-ice passes or one-timers. This momentarily increases scoring chances even if we see bigger angles.

iii) Overall Interpretation

All in all, these results show the importance of spatial dependency of shot quality for hockey. In essence, the closer and more central the shots are significantly more likely to score. 
Significantly, Distance and angle connect together, they help define the “high-danger scoring area” — which corresponds approximately the slot in front of the crease within 25 ft and 30° of center. 
These geometric patterns connect well with common sense intuition on hockey knowledge and validate that engineered geometric features (distance and angle from net) are meaningful predictors for the expected-goals (xG) model which will be developed in the subsequent sections.

### 3) 

![Goal Distance Histogram Empty vs Non-Empty Net]({{ site.baseurl }}/assets/images/image-27.png)

In order to verify the data quality and guarantee realistic shot coordinates, the goal events separated into bins as empty-net and non-empty-net categories were examined. In addition the histogram displays that non-empty-net goals are located between 0–40 ft, this averages around 21.8 ft, and we see that empty-net goals occur a considerable distance farther out, averaging at around 45 ft and extending all the way up to 98 ft. 
From the graph we can see that no non-empty-net goals surpass the cutoff threshold of 110 ft. This confirms that the recorded goals are all located within the real-life common sense on-ice distances.

Essentially, a review of potential outliers found no real anomalies in the coordinate data or given shot type. We can see that occaisional long-range non-empty-net goals were were rare events in a hockey game (e.g., rebounds or delayed-penalty situations) compared to incorrect data errors.

Overall, the histogram distribution connects with common sense hockey logic: close-range shots seriously dominate non-empty-net goals. We also see long-range shots are virtually entirely empty-net situations. This confirms that the engineered distance and event-type features are reliable and consistent.

## Baseline Models 

### 1.

![Logistic Regression Performance Metrics]({{ site.baseurl }}/assets/images/image-28.png)

When we use the distance from the hockey net as the given input, we first trained a baseline Logistic Regression model with set default parameters to predict whether a hockey shot is a successful goal. The validation accuracy of the model was around 0.906, which may seem strong. However, the closer inspection of the given classification report and the confusion matrix displays that the model predicted the “no goal” class for all samples. The models achieves 100 % recall for non-goals and 0 % recall for goals. This predictive behaviour results from the severe class imbalance in the dataset for goals vs no goals — where goals represent a small fraction of all hockey shots.

Although we can see that the prediction accuracy is high, this is highly misleading. This is because the model essentially always learned to always predict the majority class. This signifies that given accuracy is not a reliable evaluation metric for an imbalanced binary classification question that we see here. If we wanted to properly analyze the model's performance, we will examine probability-based metrics such as ROC curves, AUC, and calibration plots in later sections to allow us to better understand the how successful the model is. 

In conclusion, this baseline experiment indicates that while distance clearly influences goal likelihood, a simple linear classifier like a Logistic Regression model trained on raw labels cannot capture the true probabilistic nature of expected goals (xG) without good managing of class imbalances and more complicated features.

### 2. 

After we have observed the downsides of the accuracy-based evaluation, we have shifted our focus to probability-based metrics to better understand the manner in how the model predicts expected goals. Using the predicted probabilities (predict_proba) from our logistic regression model, we produced four diagnostic plots to evaluate the respective calibration and discriminative power.

![ROC FPR vs TPR Curve]({{ site.baseurl }}/assets/images/image-29.png)

a) The ROC curve (orange) lies consistently above the random baseline (blue dashed line), with an AUC = 0.690.
This indicates that though the model has limited predictive strength, it is better than random guessing.
Additionally, given that it relies on a single feature — distance from the net — the curve’s moderate shape reflects the common-sense relationship between proximity and goal likelihood: the closer the shot is to the goal, the higher the true-positive rate for a particular false-positive rate.

![Goal Rate based on Predicted Probability Percentile]({{ site.baseurl }}/assets/images/image-30.png)

b) The bar plot of goal rate by probability percentile displays a significant upward trend — the higher the predicted confidence leads to a higher observed goal frequency.
Ultimately, this confirms that the model’s probability outputs are meaningful: even if these are imperfectly calibrated, they increase monotonically with the given actual scoring likelihood.

![Cumulative Proportion of Goals vs Model Probability Percentile]({{ site.baseurl }}/assets/images/image-31.png)

c) The cumulative goal curve increases drastically as we go above the random baseline, which displays that a relatively small fraction of top-scoring shots explain a large portion of all goals.These results imply that the logistic regression model can effectively rank shots by quality, which then identifies higher-probability events even if we have more limited information.

![Calibration Curve]({{ site.baseurl }}/assets/images/image-32.png)

d) This given calibration curve indicates that predicted probabilities are centered near 0–0.2 and are close to the diagonal. This effectively means that the model is well-calibrated within the lower-probability regions.
However, because the given predicted values are already small, the model seems to underestimate the likelihood of rare goal events. This is another reflection of the dataset’s strong class imbalance.

Summary: Overall, these diagnostics and plots demonstrate that while the distance-only logistic regression model is simple, it generally captures the correct pattern between shot proximity and goal probability. This moderate AUC and reasonable calibration make it a decent baseline, but we will need better and richer contextual features that will be required to improve discrimination and probability accuracy in later stages.

### 3. 

![ROC Curve for Logistic Regression Models]({{ site.baseurl }}/assets/images/image-33.png)

![Goal Rate Compared to given Model Percentile]({{ site.baseurl }}/assets/images/image-34.png)

![Cumulative Number of Goal Compared to Cumulative Number of Shots]({{ site.baseurl }}/assets/images/image-35.png)

To test how certain geometric features contribute to scoring probability, we trained three Logistic Regression classifiers — one which only used distance, one which only used angles, and one which used distance and angles together — and compared these to a random baseline. Subsequently, the model’s outputs were evaluated with the same metrics as before: ROC curve (AUC), goal rate by percentile, cumulative goal curve, and lastly calibration.

ROC Curve

As shown in the ROC comparison, the distance-only model achieved an AUC of 0.690, while the angle-only model performed worse at 0.565, indicating that distance is a much stronger single predictor of scoring likelihood. The combined distance + angle model reached the highest AUC of 0.708, confirming that angle adds complementary information about shot positioning. All three models performed well above the random baseline (AUC ≈ 0.49), demonstrating that even simple geometric inputs capture meaningful spatial structure in scoring events.

Goal Rate by Predicted Percentile

When shots were categorized by predicted probability percentile, the combined model of using both distance and angles showed the clearest monotonic relationship between model confidence and actual goal frequency. The distance-only curve generally followed a similar trend, while the angle-only model remained largely flat. This suggested that considering angle individually is less discriminative but can refine predictions if we use it in conjuction with distance.

Cumulative Goal Proportion

The cumulative goal plots reinforced the findings above: the combined distance and angle model captured goals more efficiently than the other models. The combined model reaches a higher cumulative goal fraction compared to any other given shot percentile. Contrastingly, the angle-only and random models seriously lagged behind. The graph which used only the distance or angle feature showed weaker goal concentration among higher-probability shots.

Interpretation

In conclusion, these comparisons indicate that the distance from the net is the larger geometric determinant of goal probability. However, adding the shot angle marginally improves the model's discrimination ability. The combination of distance and angle model produces a more holistic understanding of the shot quality — this provides a foundation for more advanced feature engineering and hyperparameter tuning later in the milestones.

## Feature Engineering II 

![Tidy_Data_Updated]({{ site.baseurl }}/assets/images/image-36.png)

Feature Engineering II: Contextual and Power-Play Features

Continuing on the transformed shot-level data from the previous sections, the dataset was augmented with new features that convey game-context and spatio-temporal dynamics.
The resulting tidy_data_updated has 21 columns, which combines raw shot information, event-based contexts, and adds special-team situations all together.

The table below is the feature/column list of tidy_data_updated with a description for each feature:

Column Name	Description
1. game_id: The unique identifier for each NHL game.
2. game_seconds: The total completed seconds from the game's start.
3. game_period: The period of the given hockey game (1, 2, 3, or OT) at the moment.
4. x_coord, 5. y_coord:	The given shot coordinates on the rink (in feet).
6. shot_distance: The given distance from the net to the location where the shot was taken.
7. shot_angle: The angle of the shot relative to the centerline goal.
8. shot_type: The given type of shot (e.g., wrist-shot, slap-shot, tip-in, etc.).
9. is_goal: The	Binary indicator showing whether a goal was scored (1 = goal, 0 = no goal).
10. empty_net: The indicator showing whether or not the net was empty:	1 if the shot was an empty net, else 0.
11. last_event_type: The type of the preceding event in the hockey game (e.g., pass, rebound, block, hit).
12. last_event_x_coord, 13. last_event_y_coord: The given x, y coordinates of the last recorded event.
14. time_since_last_event: The number of seconds elapsed since the previous event.
15. distance_from_last_event: The distance (in feet) between the last event and the current shot.
16. rebound: Boolean column which is True if the last event was also a shot, indicating a rebound chance, false otherwise.
17. change_in_shot_angle: The change in shot angle between the last and current events (in degrees).
18. speed: The Average puck speed, computed as distance_from_last_event / time_since_last_event.
19. time_since_power_play_start: The elapsed seconds since the start of a power-play; resets to 0 when the advantage ends.
20. friendly_skaters, 21. opposing_skaters: The number of non-goalie skaters on each team, this accounts for specific penalties and power-play situations.

Summary and Insights

These engineered features transform the dataset with temporal flow, event sequence, and situational awareness, enabling more realistic modeling of how goals occur in-game context.
Specifically, features like rebound, speed, and change_in_shot_angle measure the danger of the shot subsequent to following prior plays, while the power-play indicators (time_since_power_play_start, friendly_skaters, opposing_skaters) indicate how these given situations influence goal probability.
This modified dataset gives a good foundation for training more complex expected goals (xG) models which are capable of reflecting complete dynamics of NHL end-to-end play sequences.

Tidy Data Updated Table Wanb link: https://wandb.ai/IFT6758-2025-B08/my_project/artifacts/dataset/wpg_v_wsh_2017021065/v3/files/wpg_v_wsh_2017021065.table.json

## Advanced Models

## Give it your best shot!

## Evaluate on test set! 