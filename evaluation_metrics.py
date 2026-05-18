"""
Generate evaluation metric figures for the crop disease detection prototype.

The default data below is a small demonstration set so the script can create a
report-ready image immediately. Replace y_true and y_score with your real test
labels and model confidence scores when you evaluate on a labelled dataset.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)


OUTPUT_DIR = Path("static/evaluation")
OUTPUT_IMAGE = OUTPUT_DIR / "evaluation_metrics.png"
OUTPUT_EXPLANATION = OUTPUT_DIR / "evaluation_metrics_explanation.txt"
ROC_IMAGE = OUTPUT_DIR / "roc_auc_curve.png"
CONFUSION_IMAGE = OUTPUT_DIR / "confusion_matrix.png"
PR_IMAGE = OUTPUT_DIR / "precision_recall_curve.png"
SCORE_IMAGE = OUTPUT_DIR / "score_distribution.png"


class ModelPerformance:
    """Class to display overall model performance and component contribution in tabular format."""
    
    def __init__(self):
        """Initialize ModelPerformance with component data."""
        self.models = {}
        self.overall_metrics = {}
    
    def add_model(self, model_name, accuracy, precision, recall, f1_score, 
                  auc_score, inference_time, contribution_percentage):
        """
        Add a model's performance metrics.
        
        Args:
            model_name: Name of the model/component
            accuracy: Accuracy score (0-1)
            precision: Precision score (0-1)
            recall: Recall score (0-1)
            f1_score: F1 score (0-1)
            auc_score: AUC score (0-1)
            inference_time: Inference time in milliseconds
            contribution_percentage: Component contribution %
        """
        self.models[model_name] = {
            "Accuracy": f"{accuracy:.4f}",
            "Precision": f"{precision:.4f}",
            "Recall": f"{recall:.4f}",
            "F1 Score": f"{f1_score:.4f}",
            "AUC": f"{auc_score:.4f}",
            "Inference (ms)": f"{inference_time:.2f}",
            "Contribution %": f"{contribution_percentage:.2f}"
        }
    
    def set_overall_metrics(self, accuracy, precision, recall, f1_score, auc_score):
        """Set overall model performance metrics."""
        self.overall_metrics = {
            "Accuracy": f"{accuracy:.4f}",
            "Precision": f"{precision:.4f}",
            "Recall": f"{recall:.4f}",
            "F1 Score": f"{f1_score:.4f}",
            "AUC": f"{auc_score:.4f}"
        }
    
    def display_overall_performance(self):
        """Display overall model performance in table format."""
        print("\n" + "="*70)
        print("OVERALL MODEL PERFORMANCE".center(70))
        print("="*70)
        
        df_overall = pd.DataFrame([self.overall_metrics])
        print(df_overall.to_string(index=False))
        print("="*70 + "\n")
    
    def display_component_contribution(self):
        """Display component contribution in table format."""
        print("\n" + "="*100)
        print("COMPONENT CONTRIBUTION & PERFORMANCE METRICS".center(100))
        print("="*100)
        
        df_components = pd.DataFrame(self.models).T
        print(df_components.to_string())
        print("="*100 + "\n")
    
    def display_performance_summary(self):
        """Display comprehensive performance summary."""
        print("\n" + "="*110)
        print("FARMER AI CHATBOT - MODEL PERFORMANCE SUMMARY".center(110))
        print("="*110)
        
        # Overall metrics
        print("\n📊 OVERALL SYSTEM PERFORMANCE")
        print("-" * 110)
        df_overall = pd.DataFrame([self.overall_metrics])
        print(df_overall.to_string(index=False))
        
        # Component metrics
        print("\n\n🔧 COMPONENT-WISE CONTRIBUTION & METRICS")
        print("-" * 110)
        df_components = pd.DataFrame(self.models).T
        print(df_components.to_string())
        
        print("\n" + "="*110 + "\n")
    
    def get_dataframe(self):
        """Return component data as pandas DataFrame."""
        return pd.DataFrame(self.models).T
    
    def export_to_csv(self, filename="model_performance.csv"):
        """Export performance metrics to CSV file."""
        df = pd.DataFrame(self.models).T
        df.to_csv(filename)
        print(f"✓ Exported to {filename}")
    
    def export_to_html(self, filename="model_performance.html"):
        """Export performance metrics to HTML table."""
        df = pd.DataFrame(self.models).T
        html = df.to_html()
        
        full_html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; text-align: center; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th {{ background-color: #4CAF50; color: white; padding: 12px; text-align: left; }}
                td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Farmer AI Chatbot - Component Performance Metrics</h1>
            {html}
        </body>
        </html>
        """
        
        with open(filename, 'w') as f:
            f.write(full_html)
        print(f"✓ Exported to {filename}")


class ModelComparisonChart:
    """Create comparison visualizations for multiple models."""
    
    def __init__(self, performance_data):
        """
        Initialize with ModelPerformance data.
        
        Args:
            performance_data: Dict of model names and their metrics
        """
        self.data = performance_data
    
    def plot_metrics_comparison(self, metrics=["Accuracy", "Precision", "Recall", "F1 Score"]):
        """
        Plot comparison of metrics across models.
        
        Args:
            metrics: List of metrics to compare
        """
        df = pd.DataFrame(self.data).T
        
        # Convert string values to float
        for metric in metrics:
            df[metric] = df[metric].astype(float)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Model Metrics Comparison", fontsize=16, fontweight='bold')
        
        metrics_to_plot = metrics[:4]  # First 4 metrics
        
        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx // 2, idx % 2]
            values = df[metric].values
            models = df.index
            
            colors = ['#0F766E', '#B45309', '#2563EB', '#DC2626', '#7C3AED'][:len(models)]
            bars = ax.bar(models, values, color=colors, alpha=0.7, edgecolor='black')
            
            ax.set_title(metric, fontsize=12, fontweight='bold')
            ax.set_ylabel("Score", fontsize=10)
            ax.set_ylim(0, 1.1)
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)
            
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        fig.tight_layout()
        plt.show()
        return fig
    
    def plot_contribution_heatmap(self):
        """Plot contribution percentages as heatmap."""
        df = pd.DataFrame(self.data).T
        
        # Extract numeric contribution values
        contribution = df["Contribution %"].astype(float).values.reshape(-1, 1)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(contribution, cmap="YlGnBu", aspect='auto')
        
        ax.set_xticks([0])
        ax.set_xticklabels(["Contribution %"])
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(df.index)
        ax.set_title("Component Contribution Percentage", fontsize=14, fontweight='bold')
        
        # Add text annotations
        for i in range(len(df)):
            text = ax.text(0, i, f"{contribution[i, 0]:.2f}%",
                          ha="center", va="center", color="black", fontweight='bold', fontsize=12)
        
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        plt.show()
        return fig


class ConfusionMatrix:
    """A class to create, analyze, and visualize confusion matrices."""
    
    def __init__(self, y_true, y_pred, labels=None, class_names=None):
        """
        Initialize the ConfusionMatrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: List of labels to index the matrix (default: [0, 1])
            class_names: List of class names for visualization (default: ["Class 0", "Class 1"])
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.labels = labels if labels is not None else [0, 1]
        self.class_names = class_names if class_names is not None else [f"Class {i}" for i in self.labels]
        self.matrix = confusion_matrix(y_true, y_pred, labels=self.labels)
    
    def get_matrix(self):
        """Return the confusion matrix as a numpy array."""
        return self.matrix
    
    def get_metrics(self):
        """
        Calculate and return basic metrics from the confusion matrix.
        
        Returns:
            dict: Dictionary containing TP, TN, FP, FN
        """
        tn = self.matrix[0, 0]
        fp = self.matrix[0, 1]
        fn = self.matrix[1, 0]
        tp = self.matrix[1, 1]
        
        return {
            "TP": tp,  # True Positive
            "TN": tn,  # True Negative
            "FP": fp,  # False Positive
            "FN": fn   # False Negative
        }
    
    def get_performance_metrics(self):
        """
        Calculate and return performance metrics.
        
        Returns:
            dict: Dictionary containing accuracy, precision, recall, F1 score
        """
        metrics = self.get_metrics()
        tp = metrics["TP"]
        tn = metrics["TN"]
        fp = metrics["FP"]
        fn = metrics["FN"]
        
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
    
    def visualize(self, figsize=(6, 5), cmap="YlGnBu", title=None):
        """
        Visualize the confusion matrix.
        
        Args:
            figsize: Figure size (width, height)
            cmap: Colormap name
            title: Title for the plot
            
        Returns:
            tuple: (fig, ax) matplotlib objects
        """
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(self.matrix, cmap=cmap)
        
        if title is None:
            title = "Confusion Matrix"
        ax.set_title(title)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("Actual Label")
        ax.set_xticks(range(len(self.labels)), self.class_names)
        ax.set_yticks(range(len(self.labels)), self.class_names)
        
        # Add text annotations
        for i in range(self.matrix.shape[0]):
            for j in range(self.matrix.shape[1]):
                ax.text(j, i, str(self.matrix[i, j]), 
                       ha="center", va="center", fontweight="bold", color="black")
        
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        
        return fig, ax
    
    def print_metrics(self):
        """Print all metrics in a readable format."""
        basic = self.get_metrics()
        performance = self.get_performance_metrics()
        
        print("\n=== Confusion Matrix ===")
        print(f"                Predicted")
        print(f"                Healthy  Diseased")
        print(f"Actual Healthy  {basic['TN']:6d}    {basic['FP']:6d}")
        print(f"       Diseased {basic['FN']:6d}    {basic['TP']:6d}")
        
        print("\n=== Performance Metrics ===")
        print(f"Accuracy:  {performance['accuracy']:.4f}")
        print(f"Precision: {performance['precision']:.4f}")
        print(f"Recall:    {performance['recall']:.4f}")
        print(f"F1 Score:  {performance['f1_score']:.4f}")


def demo_predictions():
    """Return demo binary labels and confidence scores for diseased leaves."""
    y_true = np.array(
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    )
    y_score = np.array(
        [0.05, 0.12, 0.18, 0.22, 0.28, 0.35, 0.41, 0.48, 0.55, 0.62,
         0.38, 0.52, 0.61, 0.67, 0.73, 0.79, 0.84, 0.89, 0.93, 0.97]
    )
    return y_true, y_score


def build_metrics_figure(y_true, y_score, threshold=0.50):
    y_pred = (y_score >= threshold).astype(int)

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Disease Detection Evaluation Metrics", fontsize=18, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(fpr, tpr, color="#0F766E", linewidth=3, label=f"AUC = {roc_auc:.2f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#94A3B8", label="Random baseline")
    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")

    ax = axes[0, 1]
    im = ax.imshow(cm, cmap="YlGnBu")
    ax.set_title(f"Confusion Matrix at Threshold {threshold:.2f}")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    ax.set_xticks([0, 1], ["Healthy", "Diseased"])
    ax.set_yticks([0, 1], ["Healthy", "Diseased"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1, 0]
    ax.plot(recall, precision, color="#B45309", linewidth=3, label=f"PR AUC = {pr_auc:.2f}")
    ax.set_title("Precision-Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left")

    ax = axes[1, 1]
    ax.hist(y_score[y_true == 0], bins=np.linspace(0, 1, 8), alpha=0.75, color="#2563EB", label="Healthy")
    ax.hist(y_score[y_true == 1], bins=np.linspace(0, 1, 8), alpha=0.75, color="#DC2626", label="Diseased")
    ax.axvline(threshold, color="#111827", linestyle="--", linewidth=2, label=f"Threshold {threshold:.2f}")
    ax.set_title("Score Distribution")
    ax.set_xlabel("Model disease confidence score")
    ax.set_ylabel("Number of samples")
    ax.set_xlim(0, 1)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig, roc_auc, pr_auc, cm


def save_individual_figures(y_true, y_score, threshold=0.50):
    y_pred = (y_score >= threshold).astype(int)

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color="#0F766E", linewidth=3, label=f"AUC = {roc_auc:.2f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#94A3B8", label="Random baseline")
    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(ROC_IMAGE, dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="YlGnBu")
    ax.set_title(f"Confusion Matrix at Threshold {threshold:.2f}")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    ax.set_xticks([0, 1], ["Healthy", "Diseased"])
    ax.set_yticks([0, 1], ["Healthy", "Diseased"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(CONFUSION_IMAGE, dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, color="#B45309", linewidth=3, label=f"PR AUC = {pr_auc:.2f}")
    ax.set_title("Precision-Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(PR_IMAGE, dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    bins = np.linspace(0, 1, 8)
    ax.hist(y_score[y_true == 0], bins=bins, alpha=0.75, color="#2563EB", label="Healthy")
    ax.hist(y_score[y_true == 1], bins=bins, alpha=0.75, color="#DC2626", label="Diseased")
    ax.axvline(threshold, color="#111827", linestyle="--", linewidth=2, label=f"Threshold {threshold:.2f}")
    ax.set_title("Score Distribution")
    ax.set_xlabel("Model disease confidence score")
    ax.set_ylabel("Number of samples")
    ax.set_xlim(0, 1)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(SCORE_IMAGE, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_explanation(roc_auc, pr_auc, cm):
    tn, fp, fn, tp = cm.ravel()
    text = f"""Evaluation Metrics Explanation

ROC Curve and AUC:
The ROC curve shows the trade-off between true positive rate and false positive
rate at different confidence thresholds. AUC summarizes this curve into one
score. A higher AUC means the model separates diseased and healthy leaves more
reliably. Demo AUC: {roc_auc:.2f}.

Confusion Matrix:
The confusion matrix counts correct and incorrect predictions at one selected
threshold. TN={tn}, FP={fp}, FN={fn}, TP={tp}. In plant disease detection,
false negatives are especially important because a diseased crop may be missed.

Precision-Recall Curve:
Precision measures how many predicted diseased samples are truly diseased.
Recall measures how many actual diseased samples the model detects. This curve
is useful when disease cases are less frequent than healthy cases. Demo PR AUC:
{pr_auc:.2f}.

Score Distribution:
The score distribution shows confidence scores for healthy and diseased samples.
Good separation means healthy samples cluster near 0 and diseased samples
cluster near 1. Overlap near the threshold indicates uncertain predictions.

Note:
These values are generated from demo labels and scores. Replace the demo data in
evaluation_metrics.py with real test-set labels and model confidence scores for
final paper or project results.
"""
    OUTPUT_EXPLANATION.write_text(text, encoding="utf-8")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    y_true, y_score = demo_predictions()
    fig, roc_auc, pr_auc, cm = build_metrics_figure(y_true, y_score)
    fig.savefig(OUTPUT_IMAGE, dpi=300, bbox_inches="tight")
    plt.close(fig)
    save_individual_figures(y_true, y_score)
    write_explanation(roc_auc, pr_auc, cm)
    print(f"Saved image: {OUTPUT_IMAGE}")
    print(f"Saved image: {ROC_IMAGE}")
    print(f"Saved image: {CONFUSION_IMAGE}")
    print(f"Saved image: {PR_IMAGE}")
    print(f"Saved image: {SCORE_IMAGE}")
    print(f"Saved explanation: {OUTPUT_EXPLANATION}")


def demo_model_performance():
    """
    Demonstrate ModelPerformance class with sample data from Farmer AI Chatbot components.
    Shows overall performance and component contribution in tabular format.
    """
    print("\n\n" + "="*110)
    print("FARMER AI CHATBOT - COMPREHENSIVE MODEL PERFORMANCE ANALYSIS".center(110))
    print("="*110)
    
    # Initialize ModelPerformance tracker
    perf = ModelPerformance()
    
    # Set overall system performance metrics
    perf.set_overall_metrics(
        accuracy=0.9125,
        precision=0.9050,
        recall=0.9200,
        f1_score=0.9125,
        auc_score=0.9680
    )
    
    # Add individual component metrics
    # Disease Detection (YOLOv8 Segmentation)
    perf.add_model(
        "YOLOv8 Disease Detection",
        accuracy=0.9400,
        precision=0.9350,
        recall=0.9450,
        f1_score=0.9400,
        auc_score=0.9750,
        inference_time=45.23,
        contribution_percentage=30.0
    )
    
    # LLM Response Generation
    perf.add_model(
        "LLM (Groq llama-3.1-8b)",
        accuracy=0.8950,
        precision=0.8900,
        recall=0.9000,
        f1_score=0.8950,
        auc_score=0.9500,
        inference_time=152.45,
        contribution_percentage=35.0
    )
    
    # RAG Retrieval System
    perf.add_model(
        "RAG Retrieval (Keyword+FAISS)",
        accuracy=0.9050,
        precision=0.9000,
        recall=0.9100,
        f1_score=0.9050,
        auc_score=0.9600,
        inference_time=18.67,
        contribution_percentage=20.0
    )
    
    # Speech-to-Text (ASR)
    perf.add_model(
        "ASR (Speech Recognition)",
        accuracy=0.8750,
        precision=0.8800,
        recall=0.8700,
        f1_score=0.8750,
        auc_score=0.9200,
        inference_time=89.34,
        contribution_percentage=10.0
    )
    
    # Text-to-Speech (TTS)
    perf.add_model(
        "TTS (Audio Generation)",
        accuracy=0.9200,
        precision=0.9150,
        recall=0.9250,
        f1_score=0.9200,
        auc_score=0.9700,
        inference_time=56.78,
        contribution_percentage=5.0
    )
    
    # Display results
    perf.display_overall_performance()
    perf.display_component_contribution()
    perf.display_performance_summary()
    
    # Export results
    print("\n📁 EXPORTING RESULTS TO FILES...")
    perf.export_to_csv("static/evaluation/model_performance.csv")
    perf.export_to_html("static/evaluation/model_performance.html")
    
    # Display comparison chart
    print("\n📊 GENERATING PERFORMANCE COMPARISON CHART...")
    chart = ModelComparisonChart(perf.models)
    chart.plot_metrics_comparison()
    chart.plot_contribution_heatmap()
    
    return perf


if __name__ == "__main__":
    # Uncomment to run the demo model performance analysis
    # demo_model_performance()
    
    main()
