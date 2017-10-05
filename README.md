# rnn_sunspots

## Forecasting number of sunspots with R package "rnn"

This is an example for using R package "rnn" in a time series forecasting situation. 

The data used is R's in-built time series "sunspots", containing data about the number of sunspots observed between 1749 and 1984. The data seems difficult to analyze as it does not show any clear trends or seasonality (the cycles have different lengths throughout the observation period).

Neural networks represent an alternative of traditional time series methods. Package "rnn" contains functions for creating recurrent neural networks, capable of learning temporal patterns. Loading data into the functions of "rnn", however, is somewhat tricky. This example aims to facilitate the data uploading process, and gives an idea for getting the most of your data when only a few predictors are available.
