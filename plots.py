import pandas as pd
import matplotlib.pyplot as plt

def create_plot(results_path="results/anomaly_predictions.csv",
                output_path="results/anomaly_plot.png"):
    df = pd.read_csv(results_path)

    plt.figure(figsize=(10, 6))
    normal = df[df["predicted_anomaly"] == 0]
    anomaly = df[df["predicted_anomaly"] == 1]

    plt.scatter(normal["transaction_amount"],
                normal["transaction_frequency"],
                alpha=0.6, label="Predicted Normal")
    plt.scatter(anomaly["transaction_amount"],
                anomaly["transaction_frequency"],
                alpha=0.8, label="Predicted Anomaly")

    plt.xlabel("Transaction Amount")
    plt.ylabel("Transaction Frequency")
    plt.title("AI-Based Anomaly Detection")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

if __name__ == "__main__":
    create_plot()
