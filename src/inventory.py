import pandas as pd
import numpy as np

def abc_category(pct):
    if pct <= 0.70: return 'A'
    elif pct <= 0.90: return 'B'
    return 'C'

def run_inventory_optimization(df, ordering_cost=50, holding_rate=0.20):
    pr = df.groupby('StockCode').agg(
        TotalRevenue = ('TotalPrice', 'sum'),
        SalesPrice   = ('Price', 'mean'),
        AnnualDemand = ('Quantity', 'sum')
    ).reset_index()

    pr = pr.sort_values('TotalRevenue', ascending=False)
    pr['CumulativePct'] = pr['TotalRevenue'].cumsum() / pr['TotalRevenue'].sum()
    pr['ABC_Category'] = pr['CumulativePct'].apply(abc_category)

    pr['HoldingCost'] = pr['SalesPrice'] * holding_rate
    pr['EOQ'] = np.sqrt(
        (2 * pr['AnnualDemand'] * ordering_cost) / pr['HoldingCost'].replace(0, np.nan)
    ).fillna(0).round(0)

    return pr

if __name__ == '__main__':
    df = pd.read_csv('data/clean_df.csv', parse_dates=['InvoiceDate'])
    inv = run_inventory_optimization(df)
    inv.to_csv('data/inventory.csv', index=False)