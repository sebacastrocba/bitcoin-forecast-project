library(tidymodels)
library(tidyverse)
library(modeltime)
library(timetk)

# Load dataset
data <- read_csv(file = "/home/seba/Documentos/Proyectos_personales/bitcoin-project/data/bitcoin_prices.csv")
data %>% head()

# Plot time series

data %>% plot_time_series(Datetime, `Adj Close`)

# Data preparation

data_prepared_tbl <- data %>% 
  select(Datetime, `Adj Close`) %>% 
  rename(timestamp = Datetime, adj_close = `Adj Close`)

# Train test split

splits <- time_series_split(
  data_prepared_tbl,
  assess = 5,
  cumulative = TRUE
)

# Modeling

wflw_fit_arima <- workflow() %>% 
  add_model(
    arima_reg() %>% set_engine("auto_arima") 
  ) %>% 
  add_recipe(
    recipe = recipe(adj_close ~ timestamp, training(splits))
  ) %>% 
  fit(training(splits))

model_tbl <- modeltime_table(wflw_fit_arima)

calib_tbl <- model_tbl %>% 
  modeltime_calibrate(testing(splits))

acc_tbl <- modeltime_accuracy(calib_tbl)

acc_tbl %>% table_modeltime_accuracy()

test_forecast_tbl <- calib_tbl %>% 
  modeltime_forecast(
    new_data = testing(splits),
    actual_data = data_prepared_tbl
  )

test_forecast_tbl %>% 
  plot_modeltime_forecast()

# Refit and Future Forecast ----

future_forecast_tbl <- calib_tbl %>% 
  modeltime_refit(data_prepared_tbl) %>% 
  modeltime_forecast(
    h = 6,
    actual_data = data_prepared_tbl
  )

future_forecast_tbl %>% 
  plot_modeltime_forecast()

# Write CSV ----

future_forecast_tbl %>% 
  select(.index:.conf_hi) %>% 
  set_names(c("Datetime", "Adj Close", "Lower Bound", "Upper Bound")) %>% 
  write_csv("data/bitcoin_prices_forecast.csv")
