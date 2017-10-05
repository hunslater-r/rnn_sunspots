######################
## Data preparation ##
######################

# Importing data:
data <- sunspots
head(data)

# General info about the data:
is.ts(data)
#[1] TRUE

plot.ts(data)
mean(data)
# [1] 51.26596

sd(data)
# [1] 43.44897

# Turning time series data into data frame:
data_df <- as.data.frame(data)
names(data_df) <- c("qty")

# Preparing time related variables: month, c ("century"), d ("decade"), y ("year")
library(zoo)

times <- as.data.frame(as.yearmon(time(data)))
years <- as.data.frame(substr(times[, 1], 5, 8))

months <- as.data.frame(substr(times[, 1], 1, 3))
months_numeric <- match(months[, 1], month.abb)

data_df <- cbind(data_df, years, months_numeric)
names(data_df) <- c("qty", "year", "month")

data_df$c <- NA
data_df$d <- NA
data_df$y <- NA

for (i in 1:nrow(data_df)) {
  data_df$c[i] <- as.numeric(as.character(substr(as.character(data_df$year[i]), 2, 2)))
  data_df$d[i] <- as.numeric(as.character(substr(as.character(data_df$year[i]), 3, 3)))
  data_df$y[i] <- as.numeric(as.character(substr(as.character(data_df$year[i]), 4, 4)))
}

# Separating training & test sets:
train <- as.data.frame(data_df[as.numeric(as.character(data_df$year)) < 1960,]) #Training: data before 1960
names(train) <- names(data_df)

test <- as.data.frame(data_df[as.numeric(as.character(data_df$year)) >= 1960,]) #Test: 1960-1984
names(test) <- names(data_df)


##################################
## Modelling with package "rnn  ##
##################################

library(rnn) #Importing package "rnn"
set.seed(12345)


#Normalization:

library(plyr)
train_min <- apply(data.matrix(train), 2, FUN = min) #Calculating the minimum of each column of training data
train_max <- apply(data.matrix(train), 2, FUN = max) #Calculating the maximum of each column of training data

train_norm <- data.frame(matrix(nrow = nrow(train), ncol = ncol(train)))

for (i in 1:nrow(train)) {
  for (j in 1:ncol(train)) {
    train_norm[i, j] <- (as.numeric(as.character(train[i, j])) - train_min[j])/
      (train_max[j] - train_min[j])
  }
}

names(train_norm) <- names(data_df)

# Step 1: Data frame

train_norm_qty_df <- data.frame(matrix(nrow = nrow(train_norm)  - 12, ncol = 12))
train_norm_c_df <- data.frame(matrix(nrow = nrow(train_norm)  - 12, ncol = 12))
train_norm_d_df <- data.frame(matrix(nrow = nrow(train_norm)  - 12, ncol = 12))
train_norm_y_df <- data.frame(matrix(nrow = nrow(train_norm)  - 12, ncol = 12))
train_norm_month_df <- data.frame(matrix(nrow = nrow(train_norm)  - 12, ncol = 12))


for (i in 1:nrow(train_norm_qty_df)) {
  train_norm_qty_df[i, ] = train_norm[i:(i+11), "qty"] 
  train_norm_c_df[i, ] = train_norm[i:(i+11), "c"]
  train_norm_d_df[i, ] = train_norm[i:(i+11), "d"] 
  train_norm_y_df[i, ] = train_norm[i:(i+11), "y"] 
  train_norm_month_df[i, ] = train_norm[i:(i+11), "month"]
  
}

# Step 2: Data matrix (numeric matrix)

train_norm_qty_m <- data.matrix(train_norm_qty_df)
train_norm_c_m <- data.matrix(train_norm_c_df)
train_norm_d_m <- data.matrix(train_norm_d_df)
train_norm_y_m <- data.matrix(train_norm_y_df)
train_norm_month_m <- data.matrix(train_norm_month_df)


# Step 3: 3D Array

train_norm_qty_fin <- array(c(train_norm_qty_m), 
                            dim=c(dim(train_norm_qty_m), 1))

train_norm_predictors_fin <- array(c(train_norm_c_m, train_norm_d_m, train_norm_y_m, train_norm_month_m), 
                                   dim=c(dim(train_norm_month_m), 4))

# Training the model:
model <- trainr(Y=train_norm_qty_fin[,1:dim(train_norm_qty_fin)[2],],
                X=train_norm_predictors_fin[,1:dim(train_norm_predictors_fin)[2],],
                learningrate   =  0.001,
                hidden_dim     = 10,
                numepochs = 100,
                network_type = "lstm")


##################
## Forecasting: ##
##################

# Normalization:

test_norm <- data.frame(matrix(nrow = nrow(test), ncol = ncol(test)))

for (i in 1:nrow(test)) {
  for (j in 1:ncol(test)) {
    test_norm[i, j] <- (as.numeric(as.character(test[i, j])) - train_min[j])/
      (train_max[j] - train_min[j])
  }
}

names(test_norm) <- names(test)

# Step 1: data frame

test_norm_c_df <- data.frame(matrix(nrow = nrow(test), ncol = 12))
test_norm_d_df <- data.frame(matrix(nrow = nrow(test), ncol = 12))
test_norm_y_df <- data.frame(matrix(nrow = nrow(test), ncol = 12))
test_norm_month_df <- data.frame(matrix(nrow = nrow(test), ncol = 12))

test_norm_c_df[1, ] <- cbind(train_norm_c_df[nrow(train_norm_c_df), 2:12], test_norm$c[1])
test_norm_d_df[1, ] <- cbind(train_norm_d_df[nrow(train_norm_d_df), 2:12], test_norm$d[1])
test_norm_y_df[1, ] <- cbind(train_norm_y_df[nrow(train_norm_y_df), 2:12], test_norm$y[1])
test_norm_month_df[1, ] <- cbind(train_norm_month_df[nrow(train_norm_month_df), 2:12], test_norm$month[1])

for (i in 2:nrow(test_norm_c_df)) {
  test_norm_c_df[i, ] <- cbind(test_norm_c_df[i-1, 2:12], test_norm$c[i])
  test_norm_d_df[i, ] <- cbind(test_norm_d_df[i-1, 2:12], test_norm$d[i])
  test_norm_y_df[i, ] <- cbind(test_norm_y_df[i-1, 2:12], test_norm$y[i])
  test_norm_month_df[i, ] <- cbind(test_norm_month_df[i-1, 2:12], test_norm$month[i])
}

# Step 2: data.matrix

test_norm_c_m <- data.matrix(test_norm_c_df)
test_norm_d_m <- data.matrix(test_norm_d_df)
test_norm_y_m <- data.matrix(test_norm_y_df)
test_norm_month_m <- data.matrix(test_norm_month_df)

# Step 3: 3D Array

test_norm_predictors_fin <- array(c(test_norm_c_m, test_norm_d_m, test_norm_y_m, test_norm_month_m), 
                                  dim=c(dim(test_norm_month_m), 4))


# Forecasting:

predictions <- predictr(model, test_norm_predictors_fin)

predictions_final <- list()

for(i in 1:nrow(test)) {
  predictions_final[i] <- as.data.frame(predictions)[i, 12] * (train_max[1] - train_min[1]) + train_min[1]
}

# Comparing real and forecasted data on ts.plot:

real <- ts(data_df[(nrow(data_df)-(nrow(test)-1)):nrow(data_df), "qty"])
pred <- ts(unlist(predictions_final))

ts.plot(real, pred, gpars = list(col = c("blue", "red")), main = "LSTM results")
legend("topleft", legend = c("predicted", "real"), col = c("dark red", "blue"), lty = 1:1)



