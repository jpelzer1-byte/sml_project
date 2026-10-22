# Creator Engagement Classification 
Python project exploring the ways a decision tree classifies video engagement using a small sample of social media content. 

I built this over the summer to practice working with structured data, creating classification labels, supervised machine learning, and evaluating its predictions. There are still many more things to learn but it was fun to connect a hobby of mine to aspects of data science. 

## Project Overview

The project uses a sample dataset of 12 videos across TikTok, Instagram, and YouTube. Each video is assigned an engagement category based on its provided engagement rate:

- **Strong:** engagement rate greater than or equal to 10%.
- **Needs Improvement:** engagement rate below 10%.

These categories are learning labels defined for this project, rather than universal measures of content quality or success.

A decision tree classifier then attempts to predict the category using three features:

| Feature | Description |
|---|---|
| `views` | Video view count |
| `length_seconds` | Video duration in seconds |
| `post_hour` | Hour of the day the video was posted |

The engagement rate determines the target label. Not included as an input feature.

## Tools Used

- **pandas** for reading and organizing data
- **NumPy** for conditional labeling
- **scikit-learn** for model training and evaluation

## How It Works

1. Loads `sample_videos.csv`.
2. Assigns engagement labels and exports `creator_report.csv`.
3. Creates binary target representing whether engagement meets the 10% threshold.
4. Splits data set into 75% training data and 25% test data.
5. Trains a `DecisionTreeClassifier`.
6. Prints test accuracy and a classification report containing precision, recall, F1-score, and support.


## Limitations

This project only demonstrates a basic machine-learning workflow. Data set is currently too small to establish reliable predictions. 

## Next Steps

- Expand the dataset and document its source and collection process.
- Compare the decision tree against a simple baseline classifier.
- Evaluate performance across multiple splits once sufficient data is available.
- Add visualizations for engagement patterns and model results.
- Refactor the script into reusable functions and add input validation.

## Learning Focus

Understanding basics of machine learning and its relationship with data science. 
