---
title: Project Proposal
layout: home
---
# Beat Buddy
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

## Introduction

<iframe src ="GenreDeminationOverYears.html" height = "500" width = "500"></iframe>
Phrases such as "Alexa, play songs like 'Girls like you' on Spotify" and "Play today's hit songs'", have become ubiquitous this decade and the core algorithms powering them are recommender and ranking systems. Such systems find applications in almost every music software product we interact with and are greatly responsible for making them more enjoyable for us.
<br /> 
Simple pattern matching hints that popular songs have similar musical attributes common amongst them. These attributes can be in turn leveraged by systems and artists to predict whether a new song conforms to them and will be a hit or not. [1]
Music recommendation systems generally operate by analyzing a user's music preferences and mapping another song closest to the preference. Some recommendation models operate via collaborative filtering on audio properties of the song [2]. Some recommendation models also operate via clustering similar songs together and recommending new songs from the cluster [3].
<br /> 
For this project, we will be working on the Million Song dataset [4] which contains 300GB of metadata of 1 million songs, as the name suggests. Some of the features of this dataset include beat frequency, artist tags, energy, danceability, segments_timbre_shape. While we do not know what every feature represents in terms of audio properties, the plenty of features do give us enough playroom to engineer more meaningful features for the model.

<br /> 

## Problem Definition

In this project we aim to build a music system that analyzes a song to predict whether it will be a hit song or not and provide recommendations to other such similar songs. Our system will be based on the Million Song dataset which contains metadata for 1 million songs. We will use the Echo Nest Taste Profile Subset [4] to obtain data on user preferences.
<br /> 

## Methods

For hotness score prediction of songs, we will benchmark the performance of three supervised learning algorithms- Random Forest, Naive Bayes and SVM. For song recommendations, we will be benchmarking the performance of three unsupervised learning algorithms- Collaborative Filtering, DBSCAN and GMM.

These algorithms will be evaluated on their ability to predict the hotness of songs and their ability to group songs together based on some characteristics and provide recommendations to users. By benchmarking the performance of these algorithm, we hope to identify effective methods that can be used to predict some characteristics and recommend songs to users.


<br /> 

## Potential Results and Discussion

We plan on measuring the impact of recommendations/predictions using four metrics over an m-fold cross validation [5] set- RMSE, Precision@k, Recall@k or HitRatio@k, Accuracy@k of the list of recommendations/predictions. Furthermore, conducting an EDA on this dataset should help us visualize the intricacies between various genres like their BPM, loudness, etc [6]. This project would engender discussion topics like the sensitivity of the models to various hyperparameters, trends of song characteristics and popularity of songs for the entire dataset.
<br />     
      
![Heat_map](heat_map.png)
![Scatter plot](t.png)
![Correlation plot](correlation plot.png)
## Proposed Timeline

![Gantt Chart](gantthighres.png )

[Link to Full Gantt Chart](https://drive.google.com/file/d/1kYv0eMd6moiMXjqHtyMAOoOe5MlqGz8l/view?usp=sharing)

<br /> 

## Contribution table

| Name              | Task        |
| ----------------- | ----------- |
| Aditya Salian     | Dataset Search, Results and Discussions, Presentation Recording       |
| Shlok Shah        | Github Page Creation, Introduction/Background, Presentation Recording        |
| Anirudh Mukherjee | Dataset Search, Gantt Chart, Github page creation, Presentation Recording       |
| Vidit Jain        | Problem Definition, Methods, Presentation Recording        |
| Shivam Agarwal    | Problem Definition, Methods, Presentation Recording        |
 

<br /> 

## References

[1] Dimolitsas I, Kantarelis S, Fouka A. SpotHitPy: A Study For ML-Based Song Hit Prediction Using Spotify. arXiv preprint arXiv:2301.07978. 2023 Jan 19.

[2] K. Yoshii, M. Goto, K. Komatani, T. Ogata and H. G. Okuno, "An Efficient Hybrid Music Recommender System Using an Incrementally Trainable Probabilistic Generative Model," in IEEE Transactions on Audio, Speech, and Language Processing, vol. 16, no. 2, pp. 435-447, Feb. 2008, doi: 10.1109/TASL.2007.911503.

[3] P. N, D. Khanwelkar, H. More, N. Soni, J. Rajani and C. Vaswani, "Analysis of Clustering Algorithms for Music Recommendation," 2022 IEEE 7th International conference for Convergence in Technology (I2CT), Mumbai, India, 2022, pp. 1-6, doi: 10.1109/I2CT54291.2022.9824160.

[4] Thierry Bertin-Mahieux, Daniel P.W. Ellis, Brian Whitman, and Paul Lamere. The Million Song Dataset. In Proceedings of the 12th International Society for Music Information Retrieval Conference (ISMIR 2011), 2011.

[5] Schindler A, Mayer R, Rauber A. Facilitating Comprehensive Benchmarking Experiments on the Million Song Dataset. InISMIR 2012 Oct (pp. 469-474).

[6] Cascante J. Song Genre Identification: The Million Song Dataset. Machine Learning Homework. 2017 Sep.


