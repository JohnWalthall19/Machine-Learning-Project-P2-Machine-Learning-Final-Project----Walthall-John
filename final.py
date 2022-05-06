#!/bin/python3

import pandas               as pd
import seaborn              as sns
import matplotlib.pyplot    as plt

from sklearn                    import linear_model
from sklearn.preprocessing      import StandardScaler
from sklearn.model_selection    import train_test_split
from sklearn.metrics            import mean_squared_error

from yellowbrick.regressor import ResidualsPlot

df = pd.read_csv("./data.csv") 

# Part 1
# Plot the distribution of target values in your dataset. Give a high level summary of
# any other pertinent statistics / distributions relevant to your dataset.

sns.histplot( data=df, x="MntWines" )
plt.savefig("distributionOfMntwines.png")

# 2. Look for NA or missing values, impute with a value that makes sense, drop the row as a
# last resort.

"""No values NA or missing"""

# 3. Looking at the non-numeric columns in your dataset, decide which columns should be
# kept for model training. (hint: any unique identifiers should be dropped)

"""They should be kept, no unique identifiers"""

# 4. Of the set of non-numeric columns, decide which should be label-encoded, and which
# should be one-hot encoded, and then carry this out on the dataset.

"""Both education and marital status should be one-hot encoded, because there is no logical
ordering or priority to either in terms of this dataset"""

df = pd.get_dummies( data=df, columns=["Education"] )
df = pd.get_dummies( data=df, columns=["Marital"] )

# 5. Create at least 3 aggregation columns. This could be mean, std deviation, min/max,
# counts, and so on of other likely important columns. It is your choice which columns and
# which functions. This can easily be done with transforms(as we have done numerous
# times before) or pd.agg

mini = df["MntWines"].min()
maxi = df["MntWines"].max()
avg = df["MntWines"].mean()

df["MinimumWines"] = df.apply( lambda x: mini, axis=1 )
df["MaximumWines"] = df.apply( lambda x: maxi, axis=1 )
df["AverageWines"] = df.apply( lambda x: avg, axis=1 )

# 7. Split your data into features (x_data) and labels (y_data)

x_data = df.drop( columns=["MntWines", "MntTotal", "MntRegularProds"] )
y_data = df["MntWines"]

# 8. Using a scaling/normalization library of your choice, scale / normalize your data 
feat_names = x_data.columns
scaler = StandardScaler()
x_data = scaler.fit_transform(x_data)

# 9. Split your data into a test and training set randomly, with 10% of the data for the test set.

x_train, x_test, y_train, y_test = train_test_split( x_data, y_data, test_size=0.10, random_state=1 )

# Part 2
# 1. Import linear regression and logistic regression.

reg = linear_model.LinearRegression()

# 2. Fit / train both models on the training dataset / labels

reg = reg.fit( x_train, y_train )

# Part 3
# Use model.score on your test set / labels to get an idea of accuracy

print( reg.score(x_test,y_test) )

# 2. Print a comparison of the two models’ accuracy

# 3. Change the distribution of training-test data to 20% test, 30% test.. All the way to 90% test 10% train. 
# Store this data and plot the relationship between accuracy and percent of training data.

for i in range( 20, 100, 10 ):
    x_train, x_test, y_train, y_test = train_test_split( x_data, y_data, test_size=float(i)/100.0, random_state=5 )
    reg = linear_model.LinearRegression()
    reg = reg.fit( x_train, y_train )
    print( f"Training with {i}% test: {reg.score( x_test, y_test )*100}% accuracy" )
    
importance = reg.coef_
# summarize feature importance
for i,v in enumerate(importance):
    name = feat_names[i]
    print('Feature: ', name, ', Score: ', v)
 
# plot feature importance
plt.bar([x for x in range(len(importance))], importance)
plt.show()

# 4. Using a residuals library of your choice, plot the residuals from the linear regression
# model.

#visual = ResidualsPlot(reg)
#visual.fit( x_train, y_train )
#visual.score( x_test, y_test )
#visual.show()
#visual.savefig("residuals.png")

    # a. Write a few sentences analyzing this plot.

"""
The distribution of residuals is roughly a bell curve centered at 0.
The models seems to have greater potential for over-estimation than for underestimation, as the largest 
positive residual is around 1200 while the smallest negative residual is only about 800.
The range of residuals increases as the predicted value gets larger.
"""

# 5. If you chose classification, find a way to plot or visualize the performance of the decision
# tree classifier, if regression, do this for the logistic regression model. I expect you to do
# your own research on the model and check out a few articles showing how people
# typically plot this, and explore the performance on your own. I am giving flexibility here
# as long as I see you did some research.

# 6. Make an argument for which of the two models for the problem type you chose works
# better in this case

scores_train = []
scores_test = []

for i in range( 20, 100, 10 ):
    x_train, x_test, y_train, y_test = train_test_split( x_data, y_data, test_size=float(i)/100.0, random_state=5 )
    reg = linear_model.SGDRegressor()
    reg = reg.fit( x_train, y_train )

    ypred_train = reg.predict( x_train )
    ypred_test = reg.predict( x_test )

    scores_train.append( mean_squared_error( y_train, ypred_train ) )
    scores_test.append( mean_squared_error( y_test, ypred_test ) )
    print( f"Training with {i}% test: {reg.score( x_test, y_test )*100}% accuracy" )

plt.plot( [ x for x in range(len(scores_train)) ], scores_train, scores_test );
#plt.show()
print( scores_train, scores_test );
"""
In this case, I would argue that linear regression is a better choice for this problem.
There is no need for the optimizations provided by SGDRegression in this case, both
models seem to take roughly the same amount of time and provide roughly equal accuracy.
In fact, in all but 2 runs of the algorithm, linear regression was actually slightly more
accurate than SGDRegression.
"""



