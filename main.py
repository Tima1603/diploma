import avs1.analyzer.box_cox as box
from avs1.analyzer.kmeans import K_Means
import avs1.analyzer.isolation_forest as iso_forest


def main():
    def run_kmeans(input_file, output_file):
        kmeans = K_Means(input_file, output_file)
        kmeans.fit()
        kmeans.plot_clusters()
        kmeans.save_clusters()



    print("First phase - preprocess data with Box Cox method for correct Machine Learning work:\n"
          "\npreprocessed data:")
    preprocess_data = box.preprocess_data(r".\avs1\analyzer\data\data2.csv", r".\avs1\analyzer\data\boxcoxed_data.csv")
    print()
    print(preprocess_data)
    print()

    print("Second phase - clustering data with K-Means Machine Learning algorithm")
    print()
    run_kmeans(r".\avs1\analyzer\data\data2.csv", r".\avs1\analyzer\data\cluster.csv")
    print()

    print("Final phase - detect anomaly from clustered data with Isolation Forest method using by Machine Learning:")
    print()
    detector = iso_forest.AnomalyDetector(r".\avs1\analyzer\data\cluster.csv")
    anomalies = detector.detect_anomalies(contamination=0.1)
    print("Detected anomalies:")
    print(anomalies)


if __name__ == "__main__":
    main()
