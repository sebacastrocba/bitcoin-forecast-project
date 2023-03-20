from rpy2.robjects.packages import importr

modeltime = importr("modeltime")
parsnip = importr("parsnip")
workflows = importr("workflows")
recipes = importr("recipes")
rsample = importr("rsample")
timetk = importr("timetk")
dplyr = importr("dplyr")
readr = importr("readr")
base = importr("base")
stats = importr("stats")
generics = importr("generics")


def make_modeltime_forecast(path_1, path_2) -> None:
    """Modeltime Forecasting Engine to interface with R via rpy2. Requires:
    - modeltime
    - parsnip
    - workflows
    - recipes
    - rsample
    - timetk
    - dplyr
    - readr
    - base
    - stats
    - generics

    Args:
        path_1 (str): Location of the incoming CSV (Raw Bitcoin Prices)
        path_2 (str): Location to store the outgoing CSV (Forecast)
    """
    
    bitcoin_prices_tbl = readr.read_csv(path_1)
    
    # Data Preparation 
    
    data_prepared_tbl = dplyr.select(bitcoin_prices_tbl, "Datetime", "Adj Close")
    
    data_prepared_tbl = stats.setNames(data_prepared_tbl, base.c("Datetime", "Adj_Close"))
    
    splits = timetk.time_series_split(
        data_prepared_tbl,
        assess     = 5,
        cumulative = True
    )
    
    # Modeling
    
    wflw_fit = workflows.workflow()
    wflw_fit = workflows.add_model(
        wflw_fit, 
        parsnip.set_engine(
            modeltime.arima_reg(), 
            "auto_arima"
        )
    )
    wflw_fit = workflows.add_recipe(
        wflw_fit,
        recipe = recipes.recipe(
            stats.as_formula("Adj_Close ~ Datetime"), 
            rsample.training(splits)
        )
    )
    wflw_fit = generics.fit(wflw_fit, rsample.training(splits))
    
    print(wflw_fit)
    
    # Modeltime Workflow
    
    model_tbl = modeltime.modeltime_table(wflw_fit)
    
    calib_tbl = modeltime.modeltime_calibrate(
        model_tbl,
        rsample.testing(splits)
    )
    
    refit_tbl = modeltime.modeltime_refit(calib_tbl, data_prepared_tbl)
    
    future_forecast_tbl = modeltime.modeltime_forecast(
        refit_tbl,
        h = 6,
        actual_data = data_prepared_tbl
    )
    
    future_forecast_tbl = dplyr.select(future_forecast_tbl, base.c(4,5,6,7))
    
    future_forecast_tbl = stats.setNames(future_forecast_tbl, base.c("Datetime", "Adj Close", "Lower Bound", "Upper Bound"))
    
    readr.write_csv(future_forecast_tbl, path_2)
    