#!/usr/bin/env python
# coding: utf-8

# In[45]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
from scipy.stats import ttest_ind
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import warnings
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import missingno as no
import datetime
import numpy as np
import plotly.graph_objects as go
import scipy.stats as stats
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv('Walmart Sales.csv')


# In[3]:


df.head()


# In[4]:


df.tail()


# In[5]:


df.info()


# In[6]:


no.bar(df)


# In[7]:


df['Date'] = pd.to_datetime(df['Date'])


# In[8]:


df.describe()


# In[9]:


df.isnull().sum()


# In[10]:


#What is the total weekly sales for all stores?
total_weekly_sales = df['Weekly_Sales'].sum()
print("Total weekly sales for all stores:", total_weekly_sales)


# In[11]:


#What is the total number of stores?
stores_cnt = df[['Store']].nunique(axis=0).values[0]
print(" total number of stores:", stores_cnt)


# In[12]:


#What is the total number of departmants?
stores_cnt = df[['Dept']].nunique(axis=0).values[0]
print(" total number of departmants:", stores_cnt)


# In[13]:


#What is the total sales for each store?
total_sales = df.groupby(['Store', 'Dept'], as_index=False)['Weekly_Sales'].sum().reset_index()

fig = px.bar(total_sales, x='Store', y='Weekly_Sales', color='Dept', title="Total Weekly Sales by Store")
fig.show()


# In[14]:


#What is the average weekly sales for each store?
mean_sales = df.groupby(['Store', 'Dept'], as_index=False)['Weekly_Sales'].mean().reset_index()

fig = px.bar(mean_sales, x='Store', y='Weekly_Sales', color='Dept', title="Total Weekly Sales by Store")
fig.show()


# In[15]:


#Which store has the highest weekly sales?
max_weekly_sales = df.groupby('Store')['Weekly_Sales'].sum().idxmax()
print("Store with the highest weekly sales:", max_weekly_sales)


# In[16]:


#Which store has the lowest weekly sales?
min_weekly_sales = df.groupby('Store')['Weekly_Sales'].sum().idxmin()
print("Store with the lowest weekly sales:", min_weekly_sales)


# In[17]:


# Visualizing the distribution of the Weekly Sales variable
sns.histplot(df['Weekly_Sales'], bins=50)
plt.show()


# In[18]:


# Checking for outliers or unusual observations in the data
px.box(df.Weekly_Sales,color = df.IsHoliday)


# In[59]:


px.box(df.Weekly_Sales,color = df.Store)


# In[60]:


px.box(df.Weekly_Sales,color = df.Dept)


# In[62]:


from statsmodels.formula.api import ols

# Assuming you have your data in a dataframe named 'df'

# Performing the ANOVA test
model = ols('Weekly_Sales ~ Store + Dept + IsHoliday', data=df).fit()
anova_table = sm.stats.anova_lm(model)

# Analyzing the ANOVA result
p_values = anova_table['PR(>F)']
alpha = 0.05  # Significance level

# Checking the significance of each variable
store_significant = p_values['Store'] < alpha
dept_significant = p_values['Dept'] < alpha
isholiday_significant = p_values['IsHoliday'] < alpha

# Printing the significance of each variable
print("Store is significant:", store_significant)
print("Dept is significant:", dept_significant)
print("IsHoliday is significant:", isholiday_significant)


# In[61]:


df.info()


# In[19]:


#What is the average weekly sales for each store?
mean_sales = df.groupby(['Date','Store'], as_index=False)['Weekly_Sales'].sum().reset_index()

fig = px.line(mean_sales, x='Date', y='Weekly_Sales', color='Store', title="Total Weekly Sales by Store")
fig.show()


# In[20]:


#What is the average weekly sales for each store?
mean_sales = df.groupby(['Date','IsHoliday'], as_index=False)['Weekly_Sales'].sum().reset_index()
fig = px.line(mean_sales, x='Date', y='Weekly_Sales', title="Total Weekly Sales(H:Holiday)")

# Add annotation when IsHoliday is true
holiday_sales = mean_sales[mean_sales['IsHoliday'] == True]
for i in range(len(holiday_sales)):
    fig.add_annotation(x=holiday_sales.iloc[i]['Date'], y=holiday_sales.iloc[i]['Weekly_Sales'],
                       text="H", showarrow=True, arrowhead=1)

fig.show()


# In[21]:


df['Year'] = df['Date'].dt.year

# Grouping by year and the week number, and summing the weekly sales
weekly_sales_yearly = df.groupby(['Year', df['Date'].dt.week])['Weekly_Sales'].sum().reset_index()
weekly_sales_yearly.rename(columns={'Date': 'Week'}, inplace=True)

# Creating the line plot using Plotly
fig = px.line(weekly_sales_yearly, x='Week', y='Weekly_Sales', color='Year', 
              title='Sum of Weekly Sales for Each Year Over Weeks')
fig.update_layout(
    xaxis_title='Week',
    yaxis_title='Sum of Weekly Sales',
    legend_title='Year'
)
fig.show()


# In[22]:


df['IsHoliday'].replace({True: 1, False: 0}, inplace=True)


# In[23]:


df


# In[24]:


# Calculate the sum of Weekly_Sales
sum_weekly_sales = df['Weekly_Sales'].sum()

# Plot the distribution of the 'Weekly_Sales' column
fig = go.Figure(data=[go.Histogram(x=df['Weekly_Sales'])])

# Update layout
fig.update_layout(title_text=f"Distribution of Weekly_Sales (Total Sum: {sum_weekly_sales})",
                  xaxis_title="Weekly_Sales",
                  yaxis_title="Count")

# Show the figure
fig.show()


# In[25]:


df = df[df.Weekly_Sales>0]


# In[26]:


# Apply Box-Cox transformation to make the distribution more normally distributed
transformed_values, lambda_value = stats.boxcox(df['Weekly_Sales'])
df['transformed_values'] = transformed_values
# Print the estimated lambda value
print("Estimated lambda value:", lambda_value)

# Plot the distribution of the transformed values
fig = go.Figure(data=[go.Histogram(x=transformed_values, nbinsx=5)])

# Update layout
fig.update_layout(title_text="Distribution of Transformed_Values (Box-Cox Transformation)",
                  xaxis_title="Transformed_Values",
                  yaxis_title="Count")

# Show the figure
fig.show()


# In[27]:


# Create the Q-Q plot
stats.probplot(df['transformed_values'], dist="norm", plot=plt)

# Customize the plot
plt.title("Q-Q Plot")
plt.xlabel("Theoretical Quantiles")
plt.ylabel("Sample Quantiles")

# Display the plot
plt.show()


# On a Q-Q plot left-skewed data appears as a concave curve (the opposite of right-skewed data)

# In[29]:


# Calculate the skewness of the 'Values' column
skewness = df['transformed_values'].skew()
print("Skewness:", skewness)


# The skewness transformation is -0.0757, which is very close to zero. This indicates that  distribution is approximately symmetrical, and does not need any transformation to improve normality.

# In[30]:


# Convert 'Date' to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Sort the DataFrame by 'Date', 'Store', and 'Dept' if it's not already sorted
df = df.sort_values(by=['Date', 'Store', 'Dept','IsHoliday'])

# Define the number of lags to create
num_lags = 5  # Example: Creating 2 lagged values

# Create lagged features based on 'Store' and 'Dept' for the 'Sales' variable
for lag in range(1, num_lags + 1):
    df[f'transformed_values_{lag}'] = df.groupby(['Store', 'Dept', 'IsHoliday'])['transformed_values'].shift(lag)

df = pd.DataFrame(df)

df.dropna(inplace=True)


# In[36]:


import category_encoders as ce

# Create a target encoder object
encoder = ce.TargetEncoder(cols=['Store','Dept', 'IsHoliday'])

# Fit and transform the data
encoded_data = encoder.fit_transform(df[['Store','Dept', 'IsHoliday']], df['transformed_values'])


# In[38]:


df[['Store_e','Dept_e', 'IsHoliday_e']] = encoded_data


# In[40]:


corr = df[['transformed_values','transformed_values_1','transformed_values_2','transformed_values_3',
          'transformed_values_4','transformed_values_5','Store_e','Dept_e', 'IsHoliday_e']].corr()

sns.heatmap(corr, annot=True, cmap="Blues")


# In[46]:


X = df[['transformed_values_1','transformed_values_2','transformed_values_3',
          'transformed_values_4','transformed_values_5','Store_e','Dept_e', 'IsHoliday_e']]

y = df['transformed_values']


# In[48]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[49]:


# Linear regression
lr = LinearRegression()
lr.fit(X_train, y_train)

# Decision tree regression
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)

# Random forest regression
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)

# Gradient boosting regression
gb = GradientBoostingRegressor(random_state=42)
gb.fit(X_train, y_train)


# In[50]:


# Make predictions
y_pred_lr = lr.predict(X_test)
y_pred_dt = dt.predict(X_test)
y_pred_rf = rf.predict(X_test)
y_pred_gb = gb.predict(X_test)

# Evaluate performance
mse_lr = mean_squared_error(y_test, y_pred_lr)
mse_dt = mean_squared_error(y_test, y_pred_dt)
mse_rf = mean_squared_error(y_test, y_pred_rf)
mse_gb = mean_squared_error(y_test, y_pred_gb)


# In[56]:


r2_lr = r2_score(y_test, y_pred_lr)
r2_dt = r2_score(y_test, y_pred_dt)
r2_rf = r2_score(y_test, y_pred_rf)
r2_gb = r2_score(y_test, y_pred_gb)


# In[52]:


# Perform cross-validation
scores_lr = cross_val_score(lr, X, y, cv=5, scoring="neg_mean_squared_error")
scores_dt = cross_val_score(dt, X, y, cv=5, scoring="neg_mean_squared_error")
scores_rf = cross_val_score(rf, X, y, cv=5, scoring="neg_mean_squared_error")
scores_gb = cross_val_score(gb, X, y, cv=5, scoring="neg_mean_squared_error")


# In[57]:


# Print results
print("Linear regression:")
print("MSE: {:.4f}".format(mse_lr))
print("R2: {:.4f}".format(r2_lr))
print("Cross-validation scores: {}".format(-scores_lr))
print("Mean cross-validation score: {:.4f}".format(-scores_lr.mean()))
print("Standard deviation of cross-validation score: {:.4f}".format(scores_lr.std()))
print()

print("Decision tree regression:")
print("MSE: {:.4f}".format(mse_dt))
print("R2: {:.4f}".format(r2_dt))
print("Cross-validation scores: {}".format(-scores_dt))
print("Mean cross-validation score: {:.4f}".format(-scores_dt.mean()))
print("Standard deviation of cross-validation score: {:.4f}".format(scores_dt.std()))
print()

print("Random forest regression:")
print("MSE: {:.4f}".format(mse_rf))
print("R2: {:.4f}".format(r2_rf))
print("Cross-validation scores: {}".format(-scores_rf))
print("Mean cross-validation score: {:.4f}".format(-scores_rf.mean()))
print("Standard deviation of cross-validation score: {:.4f}".format(scores_rf.std()))
print()

print("Gradient boosting regression:")
print("MSE: {:.4f}".format(mse_gb))
print("R2: {:.4f}".format(r2_gb))
print("Cross-validation scores: {}".format(-scores_gb))
print("Mean cross-validation score: {:.4f}".format(-scores_gb.mean()))
print("Standard deviation of cross-validation score: {:.4f}".format(scores_gb.std()))
print()

