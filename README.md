## Problem 
In this project, we try to contribute using Home Credit Data to forecast better for a credit application whether it will be paid or not. It relaesed 7 dataframes.

## objective :
The main training data with information about each loan application at Home Credit. Every loan has its own row and is identified by the feature SK_ID_CURR. The training application data comes with the TARGET indicating 0: the loan was repaid or 1: the loan was not repaid.

EDA
In this section, we apply basic Python data analyses tools like Pandas, Numpy, Matplotlib, and Seaborn. We will face some outliers and missing values. For missing values, we determine a threshold and accordingly drop some missing values. For the outliers, we change some of them with nan values. However, we left most of the job to the Gradient Boosting Models which are very capable of dealing with them.

## modeling :
For prediction, we focus on two models, LightGBM and XGBoost. Both are based on decision tree algorithms. XGBoost is very much like LightGBM except that it works by splitting the trees level-wise while the latter works by splitting leaf-wise.

## Performance Metric:
In this problem, the data is imbalanced. So we can’t use accuracy as an error metric. When data is imbalanced we can use Log loss, F1-score and AUC. Here we are sticking to AUC which can handle imbalanced datasets.

## modeling :
Using Streamlet , amodel was deployed which recieve applicant feautres and predict if he is high risk or low risk one 


https://github.com/AliNasserMohamed/Home_credit_default_risk/assets/67801762/9f494434-b62b-4496-ba18-d602840ca505




 


