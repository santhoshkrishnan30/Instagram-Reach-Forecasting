# Instagram-Reach-Forecasting

Instagram reach forecasting is the process of predicting the number of people that an Instagram post, story, or other content will be reached, based on historical data and various other factors.

For content creators and anyone using Instagram professionally, predicting the reach can be valuable for planning and optimizing their social media strategy. By understanding how their content is performing, creators can make informed decisions about when to publish, what types of content to create, and how to engage their audience. It can lead to increased engagement, better performance metrics, and ultimately, greater success on the platform.

For the task of Instagram Reach Forecasting, we need to have data about Instagram reach for a particular time period.

# Instagram Reach Forecasting using Python
Let’s start this task by importing the necessary Python libraries and the dataset:

```python
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "plotly_white"

data = pd.read_csv("Instagram-Reach.csv", encoding = 'latin-1')
print(data.head())
```
![image](https://github.com/santhoshkrishnan30/Instagram-Reach-Forecasting/assets/145760700/103c660e-b2a6-48f8-a69a-5001d568a061)

convert the Date column into datetime datatype to move forward:
```python
data['Date'] = pd.to_datetime(data['Date'])
print(data.head())
```
![image](https://github.com/santhoshkrishnan30/Instagram-Reach-Forecasting/assets/145760700/b1df1c5e-95b0-4bf0-bfd4-d9aba3a7c3fb)

# Analyzing Reach

analyze the trend of Instagram reach over time using a line chart:

```python
fig = go.Figure()
fig.add_trace(go.Scatter(x=data['Date'], 
                         y=data['Instagram reach'], 
                         mode='lines', name='Instagram reach'))
fig.update_layout(title='Instagram Reach Trend', xaxis_title='Date', 
                  yaxis_title='Instagram Reach')
fig.show()
```
![image](https://github.com/santhoshkrishnan30/Instagram-Reach-Forecasting/assets/145760700/5d7b2b52-bb4f-4f41-853f-8e95de7ebef3)

 let’s analyze Instagram reach for each day using a bar chart:

 ```python
fig = go.Figure()
fig.add_trace(go.Bar(x=data['Date'], 
                     y=data['Instagram reach'], 
                     name='Instagram reach'))
fig.update_layout(title='Instagram Reach by Day', 
                  xaxis_title='Date', 
                  yaxis_title='Instagram Reach')
fig.show()
```
![image](https://github.com/santhoshkrishnan30/Instagram-Reach-Forecasting/assets/145760700/56d1c069-0656-4d91-b78d-8e6f4c3616e4)

let’s analyze the distribution of Instagram reach using a box plot:

```python
fig = go.Figure()
fig.add_trace(go.Box(y=data['Instagram reach'], 
                     name='Instagram reach'))
fig.update_layout(title='Instagram Reach Box Plot', 
                  yaxis_title='Instagram Reach')
fig.show()
```
![image](https://github.com/santhoshkrishnan30/Instagram-Reach-Forecasting/assets/145760700/96447a75-eac3-48b2-8022-77f10eca2e23)

let’s create a day column and analyze reach based on the days of the week. To create a day column, we can use the dt.day_name() method to extract the day of the week from the Date column:

```python
data['Day'] = data['Date'].dt.day_name()
print(data.head())
```
![image](https://github.com/santhoshkrishnan30/Instagram-Reach-Forecasting/assets/145760700/34de10b5-2bf0-411f-9662-190bfe146af4)
let’s analyze the reach based on the days of the week. For this, we can group the DataFrame by the Day column and calculate the mean, median, and standard deviation of the Instagram reach column for each day:

```python
import numpy as np

day_stats = data.groupby('Day')['Instagram reach'].agg(['mean', 'median', 'std']).reset_index()
print(day_stats)
```
![image](https://github.com/santhoshkrishnan30/Instagram-Reach-Forecasting/assets/145760700/94e27ca9-ba05-4155-88aa-cc3a29718f04)

let’s create a bar chart to visualize the reach for each day of the week:

```python
fig = go.Figure()
fig.add_trace(go.Bar(x=day_stats['Day'], 
                     y=day_stats['mean'], 
                     name='Mean'))
fig.add_trace(go.Bar(x=day_stats['Day'], 
                     y=day_stats['median'], 
                     name='Median'))
fig.add_trace(go.Bar(x=day_stats['Day'], 
                     y=day_stats['std'], 
                     name='Standard Deviation'))
fig.update_layout(title='Instagram Reach by Day of the Week', 
                  xaxis_title='Day', 
                  yaxis_title='Instagram Reach')
fig.show()
```
![image](https://github.com/santhoshkrishnan30/Instagram-Reach-Forecasting/assets/145760700/ce1c75a5-346f-4720-82b6-147d48fd19cb)

# Instagram Reach Forecasting using Time Series Forecasting
To forecast reach, we can use Time Series Forecasting. Let’s see how to use Time Series Forecasting to forecast the reach of my Instagram account step-by-step.

Let’s look at the Trends and Seasonal patterns of Instagram reach:

```python
from plotly.tools import mpl_to_plotly
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

data = data[["Date", "Instagram reach"]]

result = seasonal_decompose(data['Instagram reach'], 
                            model='multiplicative', 
                            period=100)

fig = plt.figure()
fig = result.plot()

fig = mpl_to_plotly(fig)
fig.show()
```
![image](https://github.com/santhoshkrishnan30/Instagram-Reach-Forecasting/assets/145760700/cf10c40a-58b2-400a-9d60-6af43b8ea30c)

The reach is affected by seasonality, so we can use the SARIMA model to forecast the reach of the Instagram account. We need to find p, d, and q values to forecast the reach of Instagram. To find the value of d, we can use the autocorrelation plot, and to find the value of q, we can use a partial autocorrelation plot. The value of d will be 1. 

Now here’s how to visualize an autocorrelation plot to find the value of p:

```python
pd.plotting.autocorrelation_plot(data["Instagram reach"])
```
![image](https://github.com/santhoshkrishnan30/Instagram-Reach-Forecasting/assets/145760700/9d0d0b18-6297-487f-956b-07d5c6c9eb97)

And now here’s how to visualize a partial autocorrelation plot to find the value of q:

```python
from statsmodels.graphics.tsaplots import plot_pacf
plot_pacf(data["Instagram reach"], lags = 100)
```
![image](https://github.com/santhoshkrishnan30/Instagram-Reach-Forecasting/assets/145760700/2c8c12e3-f089-440e-83a2-a5c5f2651971)

```
Now here’s how to train a model using SARIMA:
```python
p, d, q = 8, 1, 2

import statsmodels.api as sm
import warnings
model=sm.tsa.statespace.SARIMAX(data['Instagram reach'],
                                order=(p, d, q),
                                seasonal_order=(p, d, q, 12))
model=model.fit()
print(model.summary())
```
![image](https://github.com/santhoshkrishnan30/Instagram-Reach-Forecasting/assets/145760700/36af5670-6cfd-4660-89b2-b99fb9ac158b)
![image](https://github.com/santhoshkrishnan30/Instagram-Reach-Forecasting/assets/145760700/bad7467a-e87d-4efa-88e0-0d27a58010ed)

Now let’s make predictions using the model and have a look at the forecasted reach:

```python
predictions = model.predict(len(data), len(data)+100)

trace_train = go.Scatter(x=data.index, 
                         y=data["Instagram reach"], 
                         mode="lines", 
                         name="Training Data")
trace_pred = go.Scatter(x=predictions.index, 
                        y=predictions, 
                        mode="lines", 
                        name="Predictions")

layout = go.Layout(title="Instagram Reach Time Series and Predictions", 
                   xaxis_title="Date", 
                   yaxis_title="Instagram Reach")

fig = go.Figure(data=[trace_train, trace_pred], layout=layout)
fig.show()
```
![image](https://github.com/santhoshkrishnan30/Instagram-Reach-Forecasting/assets/145760700/ccaaf21e-29fb-4374-9e8d-60353b913b9e)

So this is how we can forecast the reach of an Instagram account using Time Series Forecasting.

# Summary

Instagram reach prediction is the process of predicting the number of people that an Instagram post, story, or other content will be reached, based on historical data and various other factors.














