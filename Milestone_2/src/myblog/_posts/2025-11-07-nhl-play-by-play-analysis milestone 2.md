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

season - the given nhl season that the shot event occurs in. 

The final training dataset was designated as the NHL season data between 2016 to 2020 which contained over 1.28 million shot events, while the 2020/21 season data was reserved as the untouched test set. Recall that for the histograms below, 0 = no goal and 1 = goal. 

### 1 a) Histogram Distribution by Distance 

![Histogram Distribution by Distance]({{ site.baseurl }}/assets/images/image-23.png)

The first histogram displays the shots  binned by distance from the net, separated by goal outcome. The histogram reveals that most shots occur within 5–20 feet of the net, and as expected, goals themselves are concentrated in shorter distances. This makes intuitive sense as it is easy to score goals closer to the net. 
The goal frequency decreases drastically beyond 30 feet, this confirms that shot success probability decreases with distance. The orange bars (goals) essentially disappears beyond 60 feet, consistent with the intuition that long-range shots are largely unsuccessful unless they occur under special conditions (e.g. deflections). All in all, this makes intuitive sense as it is harder to score farther away from the net. 

### 1 b) Histogram Distribution by Angle

![Histogram Distribution by Angle]({{ site.baseurl }}/assets/images/image-24.png)

The second histogram displays the plot shot counts by angle, where 0° corresponds to shots taken from directly in front of the hockey net and bigger angles displays positions farther away from the hockey net. Moreover, we observe shots taken from smaller angles (specifically between 0–30°) display a higher proportion of goals compared to shots taken from larger angles which are more difficult to score from. In conclusion, the decrease in the number of shots and goal scoring rate at wider angles conveys how shooting geometry minimizes scoring chances, as goalkeepers could better cover the net at these larger angles. 

### c) Histogram Joint Distribution of Distance and Angle 

![2D Histogram Distribution of Angle and Distance]({{ site.baseurl }}/assets/images/image-25.png)

Lastly, the 2D joint plot (distance * angle) displays that the most shots are clustered around 0–20 feet and angles below 30°. This forms a dense region of high-frequency shooting activity which is situated directly in front of the net, in hockey this is known as the high-danger area.

The heatmap coloring being more red indicates an increased density of events (and goal probabilities) around this given zone. This further reiterates the idea that shots leading to goals is location dependent. 

### Summary 

All in all, these three histograms reconfirm the reliability of specific engineered geometric features. These plots reveal interpretable and understandable relationships between shot distance, angle, and the probability of scoring a goal. This lays the foundation for future models in which we can use distance and angles to help predict expected goals (xG).

### 2)

**Goal Rate Analysis by Distance and Angle:** 

This analysis understands how shot geometry affects scoring outcomes, specifically we defined the goal rate as

Goal Rate = # Goals/(# Goals + # No-Goals)

where goal rate is a function of both distance from the net and shot angle.

![Plots of Goal Rates Compared to Distance from Net]({{ site.baseurl }}/assets/images/image-26.png)

![Plots of Goal Rates Compared to Angle from Net]({{ site.baseurl }}/assets/images/image-27.png)


i) **Plot 1: Goal Rate vs. Distance**

Figure 1 displays a steep and nonlinear decline in goal rate as distance from the net increases. The plot shows that shots taken within 10 feet of the crease have an approximately 25 % chance of resulting in a goal — the highest success rate in the dataset. Subsequently, the probability of scoring decreases rapidly between 5 ft – 17.5 ft, and ultimately falling below 5 % just after 20 ft. The goal rate  then proceeds to decreases rapidly as we increase the distance from the net. This suggests that shots from long range almost never beat the goalie unless these given shots are to an empty-net or deflections. This rapid decay reconfirms that the closeness to goal is the most influential spatial factor for scoring probability.


ii) **Plot 2: Goal Rate vs. Angle**

This figure illustrates how scoring probability decreases as the shooting angle widens from the net. The highest goal rates, around 6–7%, occur for shots taken nearly straight on (0°–10°), where players have optimal shooting alignment and reduced goalie coverage. Beyond 15°, the success rate declines steadily, showing that shots from wider angles are far less effective. Small peaks between 35° and 55° suggest that certain situations—such as quick one-timers, rebounds, or cross-ice passes—temporarily improve goal chances even from less favorable angles. However, shots taken beyond 70° have goal rates below 4%, as these attempts often come from the boards or behind the goal line. Overall, the trend highlights the geometric disadvantage of shooting from the sides, emphasizing the strategic importance of generating central, low-angle opportunities for higher scoring efficienc

iii) **Overall Interpretation**

All in all, these results show the importance of spatial dependency of shot quality for hockey. In essence, the closer and more central the shots are significantly more likely to score. Significantly, Distance and angle connect together, they help define the “high-danger scoring area” — which corresponds approximately the slot in front of the crease within 10 ft and 10° of center. These geometric patterns connect well with common sense intuition on hockey knowledge and validate that engineered geometric features (distance and angle from net) are meaningful predictors for the expected-goals (xG) model which will be developed in the subsequent sections.

### 3) 

![Goal Distance Histogram Empty vs Non-Empty Net]({{ site.baseurl }}/assets/images/image-28.png)

In order to verify the data quality and guarantee realistic shot coordinates, the goal events separated into bins as empty-net and non-empty-net categories were examined. The histogram displays that non-empty-net goals are located between 0–40 ft, this averages around 21.69 ft, and we see that empty-net goals occur a considerable distance farther out, averaging at around 44.3 ft and extending all the way up to around 98 ft. 
From the graph we can see that no non-empty-net goals surpass the cutoff threshold of 110 ft. This confirms that the recorded goals are all located within the real-life common sense on-ice distances.

Essentially, a review of potential outliers found no real anomalies in the coordinate data or given shot type. We can see that occaisional long-range non-empty-net goals were rare events in a hockey game (e.g., rebounds or delayed-penalty situations) compared to incorrect data errors.

Overall, the histogram distribution connects with common sense hockey logic: close-range shots seriously dominate non-empty-net goals. We also see long-range shots are virtually entirely empty-net situations. This confirms that the engineered distance and event-type features are reliable and consistent.

## Baseline Models 

### 1.

![Logistic Regression Performance Metrics]({{ site.baseurl }}/assets/images/image-29.png)

When we use the distance from the hockey net as the given input, we first trained a baseline Logistic Regression model with set default parameters to predict whether a hockey shot is a successful goal. The validation accuracy of the model was around 0.9486, which may seem strong. However, the closer inspection of the given classification report and the confusion matrix displays that the model predicted the “no goal” class for all samples. The models achieves 100 % recall for non-goals and 0 % recall for goals. This predictive behaviour results from the severe class imbalance in the dataset for goals vs no goals — where goals represent a small fraction of all hockey shots.

Although we can see that the prediction accuracy is high, this is misleading. This is because the model essentially always learned to predict the majority class. This signifies that given accuracy is not a reliable evaluation metric for an imbalanced binary classification question. If we wanted to properly analyze the model's performance, we will examine probability-based metrics such as ROC curves, and calibration plots in later sections to allow us to better understand the how successful the model is. 

In conclusion, this baseline experiment indicates that while distance clearly influences goal likelihood, a simple linear classifier like a Logistic Regression model trained on raw labels cannot capture the true probabilistic nature of expected goals (xG) without good managing of class imbalances and more complicated feature engineering.

### 2. 

After we have observed the downsides of the accuracy-based evaluation, we have shifted our focus to probability-based metrics to better understand the manner in how the model predicts expected goals. Using the predicted probabilities (predict_proba) from our logistic regression model, we produced four diagnostic plots to evaluate the respective calibration and discriminative power.

![ROC FPR vs TPR Curve]({{ site.baseurl }}/assets/images/image-30.png)

a) The ROC curve (orange) lies consistently above the random baseline (blue dashed line), with an AUC = 0.682. This indicates that though the model has limited predictive strength, it is better than random guessing. Additionally, given that it relies on a single feature — distance from the net — the curve’s moderate shape reflects the common-sense relationship between proximity and goal likelihood: the closer the shot is to the goal, the higher the true-positive rate for a particular false-positive rate.

![Goal Rate based on Predicted Probability Percentile]({{ site.baseurl }}/assets/images/image-31.png)

b) The plot of goal rate by probability percentile displays a significant upward trend — the higher the predicted probability percentile leads to a higher goal rate. The highest goal rate is slightly above 0.14 with a probability percentile of 100. This means that if the probability percentile or confidence of scoring is 100, the actual goal rate is around 0.15. Ultimately, this confirms that the model’s probability outputs are meaningful: even if these are imperfectly calibrated, they increase monotonically with the goal rate.

However, you can see some variation exists across percentiles, the general pattern confirms that the model captures that distance influences scoring probability. However, the curve’s fairly gradual slope also suggests that the model’s discrimination power is moderate — meaning it separates likely from unlikely goals, although imperfectly.

In conclusion, the logistic regression model’s predicted probabilities does correlate positively with the actual goal frequency. This demonstrates that the baseline Logistic Regression Model effectively provides useful, though imperfectly calibrated, expected-goal estimates.

![Cumulative Proportion of Goals vs Model Probability Percentile]({{ site.baseurl }}/assets/images/image-32.png)

c) The cumulative goal curve increases drastically as we increase the probability percentile, which displays that a relatively small fraction of top-scoring shots explain a large portion of all goals. These results imply that the logistic regression model can effectively rank shots by quality, which then identifies higher-probability events even if we have more limited information. This curve ultimately demonstrates that the logistic regression model effectively discriminates between high- and low-quality scoring opportunities, providing meaningful expected-goal estimates even if the logistic regression calibration is again not perfect.


![Calibration Curve]({{ site.baseurl }}/assets/images/image-33.png)

d) This given calibration curve indicates that predicted probabilities are centered near 0–0.15 and are close to the diagonal. This effectively means that the Logistic Regression model is well-calibrated within the lower-probability regions. However, because the given predicted values are already small, the model seems to underestimate the likelihood of rare goal events. This is another reflection of the dataset’s strong class imbalance.

Overall, these diagnostics and plots demonstrate that while the distance-only logistic regression model is simple, it generally captures the correct pattern between shot proximity and goal probability. This moderate AUC and reasonable calibration make it a decent baseline, but we will need better and richer contextual features that will be required to improve discrimination and probability accuracy in later stages.

### 3. 

![ROC Curve for Logistic Regression Models]({{ site.baseurl }}/assets/images/image-34.png)

**ROC Curve for Logistic Regression Models on WANB**: [View Run Summary on W&B](https://wandb.ai/IFT6758-2025-B08/IFT6758-Milestone2/runs/u890bjv0)

![Goal Rate Compared to given Model Percentile]({{ site.baseurl }}/assets/images/image-35.png)

**Goal Rate Curve on WANB**: [View Run Summary on W&B](https://wandb.ai/IFT6758-2025-B08/IFT6758-Milestone2/runs/joq4jnr0)

![Cumulative Number of Goal Compared to Cumulative Number of Shots]({{ site.baseurl }}/assets/images/image-36.png)

**Cumulative Goal Curve**: [View Run Summary on W&B](https://wandb.ai/IFT6758-2025-B08/IFT6758-Milestone2/runs/1seszh76)

![Calibration Curve for Different Feature Combinations]({{ site.baseurl }}/assets/images/image-37.png)

![Calibration Curve for Different Feature Combinations Brier Score]({{ site.baseurl }}/assets/images/image-38.png)

**Calibration Curve**: [View Run Summary on W&B](https://wandb.ai/IFT6758-2025-B08/IFT6758-Milestone2/runs/a1f7j105)

To test how certain geometric features contribute to scoring probability, we trained three Logistic Regression classifiers — one which only used distance, one which only used angles, and one which used distance and angles together — and compared these to a random baseline. Subsequently, the model’s outputs were evaluated with the same metrics as before: ROC curve (AUC), goal rate by percentile, and cumulative goal curve.

**ROC Curve**

As shown in the ROC comparison, the distance-only model achieved an AUC of 0.682, while the angle-only model performed worse at 0.539, indicating that distance is a much stronger single predictor of scoring likelihood. The combined distance + angle model reached the highest AUC of 0.686, confirming that angle adds complementary information about shot positioning. All three models performed well above the random baseline (AUC ≈ 0.50), demonstrating that even simple geometric inputs capture meaningful spatial structure in scoring events.

**Goal Rate by Predicted Percentile**

When shots were categorized by predicted probability percentile, the combined model of using both distance and angles showed the clearest monotonic relationship between model predictive probability and actual goal frequency with both feature combinations having an actual goal frequency of just below 0.16 at the 100th percentile. The distance-only curve generally followed a similar trend, while the angle-only model remained remained small and ended at 0.07 at the 100th percentile. This suggested that considering angle individually is less discriminative but can refine predictions if we use it in conjuction with distance.

**Cumulative Goal Proportion**

This figure compares the cumulative proportion of goals captured across models as the predicted probability percentile increases. The distance + angle model (yellow) consistently accumulates goals faster than all others, indicating its superior ability to rank high-probability scoring opportunities. The distance-only model (red) performs nearly as well, confirming that shot distance remains a strong individual predictor of goal likelihood. In contrast, the angle-only model (blue) shows a noticeably flatter curve, reflecting its weaker discriminative power in isolating high-quality scoring chances. The random baseline (purple) follows a near-diagonal line, as expected, representing no predictive skill. Overall, the combined model’s steeper rise demonstrates more efficient goal capture and highlights how integrating geometric features—both distance and angle—substantially improves the model’s ability to separate dangerous shots from low-probability ones.

**Calibration Curve**

The calibration curve visualizes how closely the predicted goal probabilities match the actual scoring frequencies for each logistic regression model. As shown, the distance + angle model (blue) lies nearest to the diagonal line of perfect calibration, demonstrating the most reliable probability estimates and the lowest Brier score (~0.0476). The distance-only (green) and angle-only (red) models exhibit moderate calibration, though both slightly underpredict goal likelihoods at higher probability ranges. In contrast, the random baseline (orange) remains nearly flat, confirming its lack of predictive value. Overall, these results indicate that combining spatial features—distance and angle—yields the best-calibrated model, where predicted probabilities accurately reflect real-world scoring chances.


**Interpretation**

In conclusion, these comparisons indicate that the distance from the net is the larger geometric determinant of goal probability. However, adding the shot angle marginally improves the model's discrimination ability. The combination of distance and angle model produces a more holistic understanding of the shot quality — this provides a foundation for more advanced feature engineering and hyperparameter tuning later in the milestone.

## Feature Engineering II 

![Tidy_Data_Updated]({{ site.baseurl }}/assets/images/image-39.png)

**Feature Engineering II: Contextual and Power-Play Features**

Continuing on the transformed shot-level data from the previous sections, the dataset was augmented with new features that convey game-context and spatio-temporal dynamics.
The resulting tidy_data_updated has 22 columns, which combines raw shot information, event-based contexts, and adds special-team situations all together.

The table below is the feature/column list of tidy_data_updated with a description for each feature:

**Column Name	Description**
1. game_id: The unique identifier for each NHL game.
2. game_seconds: The total completed seconds from the game's start.
3. game_period: The period of the given hockey game (1, 2, 3, or OT) at the moment.
4. x_coord, 5. y_coord:	The given shot coordinates on the rink (in feet).
6. shot_distance: The given distance from the net to the location where the shot was taken.
7. shot_angle: The angle of the shot relative to the centerline goal.
8. shot_type: The given type of shot (e.g., wrist-shot, slap-shot, tip-in, etc.).
9. is_goal: The	Binary indicator showing whether a goal was scored (1 = goal, 0 = no goal).
10. empty_net: The indicator showing whether or not the net was empty:	1 if the shot was an empty net, else 0.
11. season: Given NHL season for the particular game and event. 
12. last_event_type: The type of the preceding event in the hockey game (e.g., pass, rebound, block, hit).
13. last_event_x_coord, 14. last_event_y_coord: The given x, y coordinates of the last recorded event.
15. time_since_last_event: The number of seconds elapsed since the previous event.
16. distance_from_last_event: The distance (in feet) between the last event and the current shot.
17. rebound: Boolean column which is True if the last event was also a shot, indicating a rebound chance, false otherwise.
18. change_in_shot_angle: The change in shot angle between the last and current events (in degrees).
19. speed: The Average puck speed, computed as distance_from_last_event / time_since_last_event.
20. time_since_power_play_start: The elapsed seconds since the start of a power-play; resets to 0 when the advantage ends.
21. friendly_skaters, 22. opposing_skaters: The number of non-goalie skaters on each team, this accounts for specific penalties and power-play situations.

**Summary and Insights**

These engineered features transform the dataset with temporal flow, event sequence, and situational awareness, enabling more realistic modeling of how goals occur in-game context.
Specifically, features like rebound, speed, and change_in_shot_angle measure the danger of the shot subsequent to following prior plays, while the power-play indicators (time_since_power_play_start, friendly_skaters, opposing_skaters) indicate how these given situations influence goal probability.
This modified dataset gives a good foundation for training more complex expected goals (xG) models which are capable of reflecting complete dynamics of NHL end-to-end play sequences.

**Tidy Data Updated Table W&B link:** [View 2017 NHL Game Dataset on W&B](https://wandb.ai/IFT6758-2025-B08/my_project/artifacts/dataset/wpg_v_wsh_2017021065/v5/files/wpg_v_wsh_2017021065.table.json)


## Advanced Models

In this section, we advanced beyond the baseline Logistic Regression to build **high-performance XGBoost classifiers**. Our goal was to analyze whether the inclusion of all engineered features, combined with **hyperparameter tuning, calibration**, and **feature selection**, can enhance predictive performance. All experiments were tracked through **Weights & Biases (wandb)** for transparency and reproducibility.

---

### Q1 — Baseline XGBoost (Distance & Angle Only)

Using the transformed tidy data, we proceeded to split the tidy data into a training and validation set to evaluate the effectiveness of the XGBoost baseline classifier. The first step was to train a **baseline XGBoost model** using only the `shot_distance` and `shot_angle` features.  This model served as a comparison to the Logistic Regression baseline model defined in section 3, and other fully tuned models trained later with the complete feature set.

Note in section 5 in order to ensure testing could be completed in a timely manner that minimized memory usage and computation cost, the training and validation sets used in section 5 comprised 10% of the entire transformed tidy data which 5 seasons worth of NHL data. 

- Optimizer: `XGBoostClassifier` (default parameters)
- Evaluation metric: `roc_auc`, goal rate curve, cumulative goal rate, and reliability curve 

**Results:**
- Validation AUC: **0.664**

**Visualizations:**

![Baseline XGBoost (AUC)]({{ site.baseurl }}/assets/images/image-40.png) 
![Baseline XGBoost Goal Rate Curve]({{ site.baseurl }}/assets/images/image-41.png)
![Baseline XGBoost Cumulative Goal Rate Curve]({{ site.baseurl }}/assets/images/image-42.png)
![Baseline XGBoost Reliability Curve]({{ site.baseurl }}/assets/images/image-43.png)

**WandB Run:** [View Baseline XGBoost AUC](https://wandb.ai/IFT6758-2025-B08/IFT6758-Milestone2/runs/soiflbit)  
**WandB Run:** [View Baseline XGBoost Goal Rate Curve](https://wandb.ai/IFT6758-2025-B08/IFT6758-Milestone2/runs/5vgwclnq)  
**WandB Run:** [View Baseline XGBoost Cumulative Goal Rate Curve](https://wandb.ai/IFT6758-2025-B08/IFT6758-Milestone2/runs/acq0dg4p)  
**WandB Run:** [View Baseline XGBoost Experiment](https://wandb.ai/IFT6758-2025-B08/IFT6758-Milestone2/runs/funh0so2)

**Analysis**

Suprisingly, the XGB AUC of 0.664 is smaller in comparison to the Logistic Regression's baseline model, where the AUC is 0.686 from Task 3. This shows XGBoost’s comparative lack of strength in capturing nonlinear patterns in comparison to Logistic Regression. In summary, prior to more sophisticated hyper parameter tuning and feature selection, XGBoost performs worse as an Expected Goal classifier than Logistic Regression. This could be derived from the smaller number of training samples used to train XGBoost baseline compared to that of Logistic Regression's baseline, or XGBoost's baseline being more sensitive to the class imbalance compared to Logistic Regression. 

With respect to the Goal Rate and Cumulative Goal curve, the general monotonically increasing trend which aligns with that of the Logistic Regression. However, the graphs are overall less smooth than the curves of Logistic Regression. The higher the probability percentile, the higher the goal scoring rate. The Calibration tool for distance and angle is initially close to the perfect classifier but diverges above and away the perfectly calibrated curve for larger probability percentiles. These all reinforce that the XGBoost is a better than random but an imperfect classifier for Expected Goals possibly due to the sensitivity of a serious class imbalance. 

### Q2 — XGBoost with All Features and Hyperparameter Tuning

We next trained an **XGBoost classifier** using all 22 engineered tidy updated features and applied **GridSearchCV** for fine-tuned optimization, and evaluated the tuned Hyperparameters on the validation set.

**Grid Search Parameters:**

| Hyperparameter | Tested Range |
|----------------|---------------|
| `learning_rate` | [0.01, 0.05, 0.1] |
| `max_depth` | [1, 3, 5] |
| `n_estimators` | [100, 200, 300] |
| `eval_metric` | [logloss, auc] |

**Best Configuration:**
learning_rate = 0.05
max_depth = 5
n_estimators = 100
eval_metric = logloss

**Best Cross Validation Accuracy**: 0.8214590354826432

![Optimal Parameters XGBoost (AUC)]({{ site.baseurl }}/assets/images/image-44.png)

![Optimal Parameters XGBoost Goal Rate Curve]({{ site.baseurl }}/assets/images/image-45.png)

![Optimal Parameters XGBoost Cumulative Goal Curve]({{ site.baseurl }}/assets/images/image-46.png)

![Optimal Parameters XGBoost Cumulative Goal Curve]({{ site.baseurl }}/assets/images/image-47.png)

After hyperparameter tuning, we can see that the best cross validation AUC for XGBoost is 0.82, far better than that of any of the Logistic Regression baseline models and the XGBoost baseline. Moreover, the hyperparameter tuned XGBoost with optimized selected features has a signficantly better AUC curve of 0.811 than the XGBoost and Logistic Regression baselines. In addition, the Goal Rate curve, Cumulative Goal curve, and Calibration curve all further show trends and figures that are better predictors for exepected goals than the baseline XGBoost and Logistic Regression models. This indicates that post hyperparameter tuning, if we optimized learning rate, max_depth, n_estimators, and eval_metric then we can see that the XGBoost will be a better classifier for expected goals. 

### Q3 — Feature Selection and Model Simplification

![Recursive Feature Elimination with Correlated Features]({{ site.baseurl }}/assets/images/image-48.png)

![Feature Importance via Coefficients]({{ site.baseurl }}/assets/images/image-49.png)

![Feature Importance via MDI]({{ site.baseurl }}/assets/images/image-50.png)

![SHAP Graph]({{ site.baseurl }}/assets/images/image-51.png)

**WandB Run:** [View XGBoost Optimal Parameters ROC Curve](https://wandb.ai/IFT6758-2025-B08/IFT6758-Milestone2/runs/7e6usrc3)

**WandB Run:** [View XGBoost Optimal Goal Rate Curve](https://wandb.ai/IFT6758-2025-B08/IFT6758-Milestone2/runs/w5qpsrrr)

**WandB Run:** [View XGBoost Cumulative Goal Rate Curve](https://wandb.ai/IFT6758-2025-B08/IFT6758-Milestone2/runs/xvo1su3s)

**WandB Run:** [View Calibration Curve](https://wandb.ai/IFT6758-2025-B08/IFT6758-Milestone2/runs/geor8nf8)

To simplify the input feature space and reduce redundancy, several feature selection and interpretability techniques were applied, including Recursive Feature Elimination (RFE), coefficient-based importance, Mean Decrease in Impurity (MDI), and SHAP analysis. RFE results showed that test accuracy stabilized after roughly 5–10 features, suggesting that only a small subset of variables carried meaningful predictive information. Coefficient and MDI analyses consistently ranked shot_distance, shot_angle, empty_net, and contextual variables like game_period and time_since_last_event as the strongest predictors of scoring likelihood.

The SHAP feature impact plot further confirmed these findings—shot_distance had the most substantial negative effect on predicted goal probability (closer shots increased scoring chances), while empty_net and shot_angle had positive contributions. Contextual features such as game_seconds and last_event_type_penalty provided situational context that slightly refined predictions. These results highlight that shot geometry and situational context drive goal probability far more than other event-based variables.

After applying these feature selection techniques, the final XGBoost model retained only the most influential variables. Importantly, the optimal hyperparameters matched those found in Q2, indicating model stability across feature subsets. The reduced-feature XGBoost model achieved an AUC of 0.811, identical to the full model, confirming that dimensionality reduction preserved predictive performance while improving interpretability and computational efficiency.

## Give it your best shot!
For preprocessing, we have standardized the data, for equalizing feature scales, stabilizing gradients, preventing domination of high-magnitude features and improving convergence. PCA has also been implemented, but later removed due to not having a noticeable impact. Finally, since the dataset is imbalanced, we have used SMOTE to mitigate it’s effect.

We have considered four models for this section: logisitic regression, with regularization, KNN with k=7, random forest, and a small 2-layer neural network ReLU activation and SGD optimizer. GridSearchCV has been used to find the best hyperparameters from a defined parameter grid.

The results are as follows:

**Logistic regression model:**
![ROC curve]({{ site.baseurl }}/assets/images/img1.png)
![Goal rate by probability percentile curve]({{ site.baseurl }}/assets/images/img2.png)
![Cumulative goals by probability percentile curve]({{ site.baseurl }}/assets/images/img3.png)
![Calibration curve]({{ site.baseurl }}/assets/images/img4.png)<br>
**WandB Run:** [View quantitative metrics](https://wandb.ai/IFT6758-2025-B08/IFT6758-Milestone2/runs/68iuetrf)<br>
**WandB Run:** [View ROC curve](https://wandb.ai/IFT6758-2025-B08/IFT6758-Milestone2/runs/6dchl54t)<br>
**WandB Run:** [View Goal rate by probability percentile curve](https://wandb.ai/IFT6758-2025-B08/IFT6758-Milestone2/runs/v786jsu2)<br>
**WandB Run:** [View Cumulative goals by probability percentile curve](https://wandb.ai/IFT6758-2025-B08/IFT6758-Milestone2/runs/d3piozyb)<br>
**WandB Run:** [View Calibration curve](https://wandb.ai/IFT6758-2025-B08/IFT6758-Milestone2/runs/nazwc5o2)


**KNN model:**



![ROC curve]({{ site.baseurl }}/assets/images/img5.png)


![Goal rate by probability percentile curve]({{ site.baseurl }}/assets/images/img6.png)


![Cumulative goals by probability percentile curve]({{ site.baseurl }}/assets/images/img7.png)


![Calibration curve]({{ site.baseurl }}/assets/images/img8.png)<br>
**WandB Run:** [View quantitative metrics](https://wandb.ai/IFT6758-2025-B08/IFT6758-Milestone2/runs/q4xdubw3)<br>
**WandB Run:** [View ROC curve](https://wandb.ai/IFT6758-2025-B08/IFT6758-Milestone2/runs/9wndfi7m)<br>
**WandB Run:** [View Goal rate by probability percentile curve](https://wandb.ai/IFT6758-2025-B08/IFT6758-Milestone2/runs/xt601bkw)<br>
**WandB Run:** [View Cumulative goals by probability percentile curve](https://wandb.ai/IFT6758-2025-B08/IFT6758-Milestone2/runs/dmalxrx7)<br>
**WandB Run:** [View Calibration curve](https://wandb.ai/IFT6758-2025-B08/IFT6758-Milestone2/runs/n94g9h0z)<br>


**Random forest model:**



![ROC curve]({{ site.baseurl }}/assets/images/img9.png)


![Goal rate by probability percentile curve]({{ site.baseurl }}/assets/images/img10.png)


![Cumulative goals by probability percentile curve]({{ site.baseurl }}/assets/images/img11.png)


![Calibration curve]({{ site.baseurl }}/assets/images/img12.png)<br>
**WandB Run:** [View quantitative metrics](https://wandb.ai/IFT6758-2025-B08/IFT6758-Milestone2/runs/e69e50jy)<br>
**WandB Run:** [View ROC curve](https://wandb.ai/IFT6758-2025-B08/IFT6758-Milestone2/runs/6x7mdv7m)<br>
**WandB Run:** [View Goal rate by probability percentile curve](https://wandb.ai/IFT6758-2025-B08/IFT6758-Milestone2/runs/bcflkmrb)<br>
**WandB Run:** [View Cumulative goals by probability percentile curve](https://wandb.ai/IFT6758-2025-B08/IFT6758-Milestone2/runs/52y4soa0)<br>
**WandB Run:** [View Calibration curve](https://wandb.ai/IFT6758-2025-B08/IFT6758-Milestone2/runs/tdte8gdz)


**Neural network model:**



![ROC curve]({{ site.baseurl }}/assets/images/img13.png)


![Goal rate by probability percentile curve]({{ site.baseurl }}/assets/images/img14.png)


![Cumulative goals by probability percentile curve]({{ site.baseurl }}/assets/images/img15.png)


![Calibration curve]({{ site.baseurl }}/assets/images/img16.png)<br>
**WandB Run:** [View quantitative metrics](https://wandb.ai/IFT6758-2025-B08/IFT6758-Milestone2/runs/1s32hvp2)<br>
**WandB Run:** [View ROC curve](https://wandb.ai/IFT6758-2025-B08/IFT6758-Milestone2/runs/1co81jnl)<br>
**WandB Run:** [View Goal rate by probability percentile curve](https://wandb.ai/IFT6758-2025-B08/IFT6758-Milestone2/runs/wzdwfepa)<br>
**WandB Run:** [View Cumulative goals by probability percentile curve](https://wandb.ai/IFT6758-2025-B08/IFT6758-Milestone2/runs/hb8bhld5)<br>
**WandB Run:** [View Calibration curve](https://wandb.ai/IFT6758-2025-B08/IFT6758-Milestone2/runs/rxu0hqes)

Out of the four models in this section, logistic regression had the best performance with an AUC score of 0.767


## Evaluate on test set

**Testing on the regular season dataset:**

- section 3 logistic regression models:
**WandB Run:** [View plots for section 3 logistic regression models](https://wandb.ai/IFT6758-2025-B08/IFT6758-Milestone2/runs/fn8hklmo)
![ROC curve]({{ site.baseurl }}/assets/images/output1.1.png) 
![Goal rate by probability percentile curve]({{ site.baseurl }}/assets/images/output1.2.png) 
![Cumulative goals by probability percentile curve]({{ site.baseurl }}/assets/images/output1.3.png) 
![Calibration curve]({{ site.baseurl }}/assets/images/output1.4.png) 

- XGBoost model:
**WandB Run:** [View plots for the XGBoost model](https://wandb.ai/IFT6758-2025-B08/IFT6758-Milestone2/runs/y0389i6a)
![ROC curve]({{ site.baseurl }}/assets/images/xgb1.1.png) 
![Goal rate by probability percentile curve]({{ site.baseurl }}/assets/images/xgb1.2.png) 
![Cumulative goals by probability percentile curve]({{ site.baseurl }}/assets/images/xgb1.3.png) 
![Calibration curve]({{ site.baseurl }}/assets/images/xgb1.4.png) 

- Best model in section 6:
**WandB Run:** [View plots for the best model in section 6](https://wandb.ai/IFT6758-2025-B08/IFT6758-Milestone2/runs/96ew6k8a)
![ROC curve]({{ site.baseurl }}/assets/images/lr1.1.png) 
![Goal rate by probability percentile curve]({{ site.baseurl }}/assets/images/lr1.2.png) 
![Cumulative goals by probability percentile curve]({{ site.baseurl }}/assets/images/lr1.3.png) 
![Calibration curve]({{ site.baseurl }}/assets/images/lr1.4.png) 


**Testing on the playoff games:**

- section 3 logistic regression models:
**WandB Run:** [View plots for section 3 logistic regression models](https://wandb.ai/IFT6758-2025-B08/IFT6758-Milestone2/runs/rmf8g65b)
![ROC curve]({{ site.baseurl }}/assets/images/output2.1.png) 
![Goal rate by probability percentile curve]({{ site.baseurl }}/assets/images/output2.2.png) 
![Cumulative goals by probability percentile curve]({{ site.baseurl }}/assets/images/output2.3.png) 
![Calibration curve]({{ site.baseurl }}/assets/images/output2.4.png) 

- XGBoost model:
**WandB Run:** [View plots for the XGBoost model](https://wandb.ai/IFT6758-2025-B08/IFT6758-Milestone2/runs/7uebjhn8)
![ROC curve]({{ site.baseurl }}/assets/images/xgb2.1.png) 
![Goal rate by probability percentile curve]({{ site.baseurl }}/assets/images/xgb2.2.png) 
![Cumulative goals by probability percentile curve]({{ site.baseurl }}/assets/images/xgb2.3.png) 
![Calibration curve]({{ site.baseurl }}/assets/images/xgb2.4.png) 

- Best model in section 6:
**WandB Run:** [View plots for the best model in section 6](https://wandb.ai/IFT6758-2025-B08/IFT6758-Milestone2/runs/9k4vby3a)
![ROC curve]({{ site.baseurl }}/assets/images/lr2.1.png) 
![Goal rate by probability percentile curve]({{ site.baseurl }}/assets/images/lr2.2.png) 
![Cumulative goals by probability percentile curve]({{ site.baseurl }}/assets/images/lr2.3.png) 
![Calibration curve]({{ site.baseurl }}/assets/images/lr2.4.png) 


Between the three logistic regression models from section 3, the model trained on both distance and angle features is, as one would expect, performing better compared to the models trained on only one of the features. Also, by comparing the two models trained on only one feature, one can deduce that the distance feature (AUC = 0.682) is more important than the angle feature (AUC = 0.533).

The calibration curves for the logistic regression models in part 3, each collapse into one point. This is because the models’ predicted probabilities on the test set all fall into only one calibration bin. One reason for this could be because the dataset is imbalanced.

On the test set, XGBoost has the best results, with an AUC score of 0.819 and 0.808 on the regular seasons and playoffs respectively. Also, by comparing the scores from validation and test set for this model, it can be deduced that it is not over-fitting, and overall can generalize well.


The best logistic regression model’s AUC score on the test set is 0.539, which is considerably lower compared to it’s score on the validation and train sets. This can be due to over-fitting, or the train and validation sets being relatively simple and small. 


All of the tested models show similar generalization performance in regular season compared to playoff games.

---

*All images referenced in this blog are outputs generated during model evaluation and saved locally.*
