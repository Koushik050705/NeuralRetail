import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error

def run_forecast(daily_sales, periods=90):
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True,
                     daily_seasonality=False, seasonality_mode='additive')
    model.fit(daily_sales)

    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)

    holdout = daily_sales.tail(30)
    merged = holdout.merge(forecast[['ds', 'yhat']], on='ds', how='left')
    mape = mean_absolute_percentage_error(merged['y'], merged['yhat'])
    print(f"MAPE: {mape:.2%}")

    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

if __name__ == '__main__':
    daily = pd.read_csv('data/daily_sales.csv', parse_dates=['ds'])
    forecast = run_forecast(daily)
    forecast.to_csv('data/forecast.csv', index=False)