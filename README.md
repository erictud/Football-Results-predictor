# Football result predictor

This projects implements a ML model that predicts the result of a match based one the stats at the half-time. The model is based on a clasification problem, being trained by an extensive dataframe. 

## Extraction of train & testing dataframes
The dataset is based on data collected from the RapidAPI platform, the football section. We use this API to extract match data from six seasons of the Romanian Superliga (from 20-21 season to 24-25 season). From each fixture, we extract the names of the home & away teams, the goals scored by each team at HT and FT. From these stats, we calculate a goal diffrence at halftime based on which we calculate an advantage variable (0 for draw, 1 if home leads, 2 if away leads). We also calculate the target label which is a code for the winning team (0 - draw, 1 - home wins, 2 - away wins). Another metric we calculate is the goals per minute metric which is the result of dividing the total goals at half time by 45 (the length of a football half in minutes). We split the calculated dataset in one for training (70% of the dataset) and the rest for testing. So, the dataset is composed by 9 fields (home/away team name, half time goals of both teams, total half time goals, goal diffrence, total goals, advantage at halftime, goals per minute and result) of diffrent types (strings, integers, categorical values and floats).

## Exploratory data analysis
### Analyzing missing values
We firstly analyze the number of percent of missing values for each column. When running the analysis we observe that we have a 0% missing values. That's because we get our data from an official API, we dont have any cases were we have missing data. So treating cases in which we have missing data for columns is redundant.
### Getting descriptive stats
We treat numerical values (ints and floats) diffrent from categorical values. Firstly for every numerical value in each dataset we show the count, mean, standard deviation, maximum and minimum. For categorical values, we determine the number of unique teams, the most played team and the frequency.
### Analyzing variable distribution
We generate a histogram for every numerical variable and a countplot for every categorical value. Analyzing goal diff at half time column, we observe that most matches have no advantage at half time, while at HT the best goal diff is 2. We can also analyze the HT advantage coutplot, and we can observe in both data sets that most matches are a tie at halftime and in the rest of cases home team si favoured.
### Detecting outliers
We use the IQR (Interquartile Range) rule and boxplot visualizations, we analyzed the numerical columns for extreme values. For the HT total goals column, the boxplot revealed a few outliers on the upper side. Because these represent real, valid football match outcomes rather than recording errors, we decided to keep these instances in the dataset.
### Analyzing corelation
We analyze the correlation between halftime goals metrics and we observe a near-zero correlation (-0.03) between home and away goals, meaning their scoring performances don't influence each other. Naturally, individual team goals strongly correlate (0.71 and 0.68) with the total HT goals. In order to avoid data redundancy, we can drop the Total goals HT column before training the final machine learning model.
### Analyzing relations with target variable
We used a Violin Plot because we are solving a classification problem. The plot clearly shows how the half-time goal difference impacts the final match result:
class 1 (Home win): The distribution shows mostly positive half-time goal differences., class 2 (Away win): The distribution leans heavily towards negative differences, class 0 (Draw): The data is strongly concentrated at 0.

## Training and evaluating base model
We trained a random forest model due to the random nature of football (we dont have a clear relation between stats at half time and the final result). After training, we get a 57.29% accuracy. We assess the result based on the testing dataset and the confusion matrix that we generated. We see that the model has predicted 84/122 home wins, 47/85 away wins and 34/81 ties. Altough, the score may seem low, due to the unpredictable nature of football results and the small training dataset, the result is quite good.