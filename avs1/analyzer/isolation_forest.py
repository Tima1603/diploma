# import pandas as pd
# from sklearn.ensemble import IsolationForest
#
#
# def main():
#     # Load clustered data from clustered_data.csv
#     clustered_df = pd.read_csv(r".\data\clustered_data.csv")
#
#     # Use Isolation Forest for anomaly detection
#     clf = IsolationForest(contamination=0.1)  # 10% of data is considered as anomalies, you can adjust this value
#     clf.fit(clustered_df[['size_kb', 'claster_id', 'cluster']])
#
#     # Predict anomalies
#     clustered_df['anomaly'] = clf.predict(clustered_df[['size_kb', 'claster_id', 'cluster']])
#
#     # Display anomalies
#     anomalies = clustered_df[clustered_df['anomaly'] == -1]
#     print("Detected anomalies:")
#     print(anomalies)
#
#
# if __name__ == "__main__":
#     main()


# import pandas as pd
# from sklearn.ensemble import IsolationForest
#
#
# def main():
#     # Load clustered data from clustered_data.csv
#     clustered_df = pd.read_csv(r".\data\clustered_data2.csv")
#     # Use Isolation Forest for anomaly detection
#     clf = IsolationForest(contamination=0.1)  # 10% of data is considered as anomalies, you can adjust this value
#     clf.fit(clustered_df[['size_kb', 'cluster']])
#     # Predict anomalies
#     clustered_df['anomaly'] = clf.predict(clustered_df[['size_kb', 'cluster']])
#     # # Add 'id' column
#     # clustered_df.insert(loc=1, column='id', value='')  # Insert 'id' column at index 1
#     # # Set value 'id' at row 2
#     # clustered_df.at[0, 'id'] = 'id'
#     # Display anomalies
#     print("Detected anomalies:")
#     print(clustered_df[clustered_df['anomaly'] == -1])
#
#
# if __name__ == "__main__":
#     main()




import pandas as pd
from sklearn.ensemble import IsolationForest

class AnomalyDetector:
    def __init__(self, input_file):
        self.input_file = input_file

    def detect_anomalies(self, contamination=0.1):
        # Load clustered data
        clustered_df = pd.read_csv(self.input_file)

        # Use Isolation Forest for anomaly detection
        clf = IsolationForest(contamination=contamination)
        clf.fit(clustered_df[['size_kb', 'cluster']])

        # Predict anomalies
        clustered_df['anomaly'] = clf.predict(clustered_df[['size_kb', 'cluster']])

        return clustered_df[clustered_df['anomaly'] == -1]
