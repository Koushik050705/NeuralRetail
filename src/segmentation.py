import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def segment_customers(rfm):
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

    means = rfm.groupby('Cluster')['Monetary'].mean().sort_values(ascending=False)
    rank_to_label = {means.index[0]: 'Champions', means.index[1]: 'Loyal',
                      means.index[2]: 'At Risk',   means.index[3]: 'Hibernating'}
    rfm['Segment'] = rfm['Cluster'].map(rank_to_label)
    return rfm

if __name__ == '__main__':
    rfm = pd.read_csv('data/rfm_base.csv')
    rfm = segment_customers(rfm)
    rfm.to_csv('data/rfm_segmented.csv', index=False)