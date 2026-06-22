import pandas as pd

def build_rfm(df):
    snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    rfm = df.groupby('Customer ID').agg(
        Recency   = ('InvoiceDate', lambda x: (snapshot_date - x.max()).days),
        Frequency = ('Invoice',     'nunique'),
        Monetary  = ('TotalPrice',  'sum')
    ).reset_index()
    return rfm

def build_daily_sales(df):
    daily = df.resample('D', on='InvoiceDate')['TotalPrice'].sum().reset_index()
    daily.columns = ['ds', 'y']
    return daily[daily['y'] > 0]

if __name__ == '__main__':
    df = pd.read_csv('data/clean_df.csv', parse_dates=['InvoiceDate'])
    build_rfm(df).to_csv('data/rfm_base.csv', index=False)
    build_daily_sales(df).to_csv('data/daily_sales.csv', index=False)