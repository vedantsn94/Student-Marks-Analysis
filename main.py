"""
Student Marks Analysis
======================
Loads student data, performs EDA, generates visualizations,
trains ML models to predict Pass/Fail, and saves all outputs.

Usage:
    python main.py                    # default CSV in same folder
    python main.py --csv path/to.csv  # custom CSV path
    python main.py --pass-mark 40     # change pass threshold (default 35)
    python main.py --save-plots       # save charts as PNG instead of showing
"""

import argparse
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

# ── CLI arguments ──────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Student Marks Analysis")
    parser.add_argument("--csv",        default="student_marks.csv", help="Path to CSV file")
    parser.add_argument("--pass-mark",  type=float, default=35,      help="Minimum marks per subject to pass")
    parser.add_argument("--save-plots", action="store_true",          help="Save plots as PNG files instead of displaying")
    parser.add_argument("--output-dir", default="output",             help="Directory to save plots and reports")
    return parser.parse_args()

# ── Helpers ────────────────────────────────────────────────────────────────────

def show_or_save(fig, name, save, out_dir):
    if save:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{name}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {path}")
    else:
        plt.show()
    plt.close(fig)


def grade(avg):
    if avg >= 85: return "A"
    if avg >= 70: return "B"
    if avg >= 55: return "C"
    if avg >= 40: return "D"
    return "F"


def percentile_rank(series, value):
    """Return the percentile rank of a value within a series."""
    return round((series < value).mean() * 100, 1)


def detect_subject_cols(df):
    possible_non_subject = {"StudentID", "RollNo", "Name", "Attendance", "Section", "Class"}
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    return [c for c in numeric_cols if c not in possible_non_subject]


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── 1. Load & Validate ────────────────────────────────────────────────────
    print("=" * 60)
    print("  STUDENT MARKS ANALYSIS")
    print("=" * 60)

    if not os.path.exists(args.csv):
        print(f"ERROR: CSV file not found → '{args.csv}'")
        sys.exit(1)

    df = pd.read_csv(args.csv)
    print(f"\n[1] Data loaded: {df.shape[0]} students, {df.shape[1]} columns")
    print(f"    Columns: {list(df.columns)}")

    subject_cols = detect_subject_cols(df)
    if not subject_cols:
        print("ERROR: No numeric subject columns detected.")
        sys.exit(1)
    print(f"    Detected subject columns: {subject_cols}")

    # ── 2. Cleaning ───────────────────────────────────────────────────────────
    missing_before = df.isnull().sum().sum()
    for c in subject_cols:
        df[c] = df[c].fillna(df[c].median())
        df[c] = df[c].clip(0, 100)

    if "Attendance" in df.columns:
        df["Attendance"] = df["Attendance"].fillna(df["Attendance"].median()).clip(0, 100)

    duplicates = df.duplicated(subset=["Name"]).sum() if "Name" in df.columns else 0
    print(f"\n[2] Cleaning: filled {missing_before} missing values, found {duplicates} duplicate names")

    # ── 3. Feature Engineering ────────────────────────────────────────────────
    df["Total"]       = df[subject_cols].sum(axis=1)
    df["Average"]     = df[subject_cols].mean(axis=1).round(2)
    df["MaxSubject"]  = df[subject_cols].max(axis=1)
    df["MinSubject"]  = df[subject_cols].min(axis=1)
    df["StdDevMarks"] = df[subject_cols].std(axis=1).round(2)   # consistency metric
    df["SubjectsPassed"] = (df[subject_cols] >= args.pass_mark).sum(axis=1)

    # Pass/Fail: must pass ALL subjects
    df["PassFail"]    = (df["MinSubject"] >= args.pass_mark).astype(int)
    df["Grade"]       = df["Average"].apply(grade)
    df["Percentile"]  = df["Average"].rank(pct=True).mul(100).round(1)

    # Attendance tier
    if "Attendance" in df.columns:
        df["AttTier"] = pd.cut(
            df["Attendance"],
            bins=[0, 60, 75, 85, 100],
            labels=["Poor", "Average", "Good", "Excellent"]
        )

    print(f"\n[3] Features engineered — new columns: Total, Average, Grade, Percentile, StdDevMarks, SubjectsPassed")

    # ── 4. EDA ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    pass_rate = df["PassFail"].mean() * 100
    print(f"\nTotal students : {len(df)}")
    print(f"Pass rate      : {pass_rate:.1f}%  ({df['PassFail'].sum()} pass / {(df['PassFail']==0).sum()} fail)")
    print(f"Average marks  : {df['Average'].mean():.2f}  (std: {df['Average'].std():.2f})")
    print(f"Highest scorer : {df.loc[df['Average'].idxmax(), 'Name']} — {df['Average'].max():.1f}")
    print(f"Lowest scorer  : {df.loc[df['Average'].idxmin(), 'Name']} — {df['Average'].min():.1f}")

    print("\nGrade distribution:")
    print(df["Grade"].value_counts().sort_index().to_string())

    print("\nSubject-wise average marks:")
    print(df[subject_cols].mean().sort_values(ascending=False).round(2).to_string())

    print("\nSubject-wise pass rates (%):")
    for s in subject_cols:
        r = (df[s] >= args.pass_mark).mean() * 100
        print(f"  {s:<12}: {r:.1f}%")

    print("\nTop 10 students by Average:")
    top10 = df.sort_values("Average", ascending=False).head(10)
    cols_show = ["Name"] + subject_cols + ["Average", "Grade"]
    print(top10[cols_show].to_string(index=False))

    print("\nBottom 10 students by Average:")
    bot10 = df.sort_values("Average").head(10)
    print(bot10[cols_show].to_string(index=False))

    if "Attendance" in df.columns:
        print("\nAverage marks by Attendance tier:")
        print(df.groupby("AttTier", observed=True)["Average"].mean().round(2).to_string())

    # ── 5. Export enriched CSV ────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    enriched_path = os.path.join(args.output_dir, "student_marks_enriched.csv")
    df.to_csv(enriched_path, index=False)
    print(f"\n[5] Enriched CSV saved → {enriched_path}")

    # ── 6. Visualizations ────────────────────────────────────────────────────
    print("\n[6] Generating visualizations…")
    sns.set_theme(style="whitegrid", palette="muted")

    # 6a. Subject-wise average bar chart
    fig, ax = plt.subplots(figsize=(8, 4))
    means = df[subject_cols].mean().sort_values()
    colors = ["#e74c3c" if v < args.pass_mark else "#2ecc71" if v >= 70 else "#f39c12" for v in means]
    means.plot(kind="barh", ax=ax, color=colors)
    ax.axvline(args.pass_mark, color="red", linestyle="--", linewidth=1, label=f"Pass mark ({args.pass_mark})")
    ax.set_title("Subject-wise Average Marks", fontsize=13, fontweight="bold")
    ax.set_xlabel("Average Marks")
    ax.legend()
    show_or_save(fig, "01_subject_averages", args.save_plots, args.output_dir)

    # 6b. Grade distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    grade_order = ["A", "B", "C", "D", "F"]
    grade_colors = {"A": "#2ecc71", "B": "#27ae60", "C": "#f39c12", "D": "#e67e22", "F": "#e74c3c"}
    counts = df["Grade"].value_counts().reindex(grade_order, fill_value=0)
    bars = ax.bar(counts.index, counts.values, color=[grade_colors[g] for g in counts.index], edgecolor="white")
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, str(val), ha="center", fontsize=10)
    ax.set_title("Grade Distribution", fontsize=13, fontweight="bold")
    ax.set_ylabel("Number of Students")
    show_or_save(fig, "02_grade_distribution", args.save_plots, args.output_dir)

    # 6c. Average marks distribution (histogram + KDE)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df["Average"], bins=15, color="#3498db", edgecolor="white", alpha=0.8, density=True, label="Histogram")
    df["Average"].plot.kde(ax=ax, color="#e74c3c", linewidth=2, label="KDE")
    ax.axvline(df["Average"].mean(), color="orange", linestyle="--", linewidth=1.5, label=f"Mean ({df['Average'].mean():.1f})")
    ax.set_title("Distribution of Average Marks", fontsize=13, fontweight="bold")
    ax.set_xlabel("Average Marks")
    ax.legend()
    show_or_save(fig, "03_average_distribution", args.save_plots, args.output_dir)

    # 6d. Correlation heatmap
    corr_cols = subject_cols + ["Average", "Total"]
    if "Attendance" in df.columns:
        corr_cols.append("Attendance")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df[corr_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax, linewidths=0.5)
    ax.set_title("Correlation Heatmap", fontsize=13, fontweight="bold")
    show_or_save(fig, "04_correlation_heatmap", args.save_plots, args.output_dir)

    # 6e. Boxplot per subject
    fig, ax = plt.subplots(figsize=(9, 5))
    df[subject_cols].plot.box(ax=ax, patch_artist=True)
    ax.axhline(args.pass_mark, color="red", linestyle="--", linewidth=1, label=f"Pass mark ({args.pass_mark})")
    ax.set_title("Subject-wise Marks Distribution (Boxplot)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Marks")
    ax.legend()
    plt.xticks(rotation=30, ha="right")
    show_or_save(fig, "05_subject_boxplots", args.save_plots, args.output_dir)

    # 6f. Pass/Fail pie chart
    fig, ax = plt.subplots(figsize=(5, 5))
    labels = ["Pass", "Fail"]
    sizes  = [df["PassFail"].sum(), (df["PassFail"] == 0).sum()]
    colors = ["#2ecc71", "#e74c3c"]
    ax.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=140,
           wedgeprops=dict(edgecolor="white", linewidth=2))
    ax.set_title("Overall Pass / Fail", fontsize=13, fontweight="bold")
    show_or_save(fig, "06_pass_fail_pie", args.save_plots, args.output_dir)

    # 6g. Scatter: Average vs Attendance (if available)
    if "Attendance" in df.columns:
        fig, ax = plt.subplots(figsize=(7, 5))
        scatter = ax.scatter(df["Attendance"], df["Average"],
                             c=df["PassFail"], cmap="RdYlGn", alpha=0.7, edgecolors="grey", linewidths=0.3)
        plt.colorbar(scatter, ax=ax, label="Pass (1) / Fail (0)")
        ax.set_title("Attendance vs Average Marks", fontsize=13, fontweight="bold")
        ax.set_xlabel("Attendance (%)")
        ax.set_ylabel("Average Marks")
        show_or_save(fig, "07_attendance_vs_average", args.save_plots, args.output_dir)

    # 6h. Top 10 students bar chart
    fig, ax = plt.subplots(figsize=(9, 4))
    top10_sorted = top10.sort_values("Average")
    ax.barh(top10_sorted["Name"], top10_sorted["Average"], color="#3498db", edgecolor="white")
    ax.set_title("Top 10 Students by Average", fontsize=13, fontweight="bold")
    ax.set_xlabel("Average Marks")
    show_or_save(fig, "08_top10_students", args.save_plots, args.output_dir)

    # ── 7. ML: Predict Pass/Fail ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  MACHINE LEARNING — PASS / FAIL PREDICTION")
    print("=" * 60)

    feature_cols = subject_cols + ["Average", "MaxSubject", "MinSubject", "StdDevMarks", "SubjectsPassed"]
    if "Attendance" in df.columns:
        feature_cols.append("Attendance")

    X = df[feature_cols]
    y = df["PassFail"]

    class_counts = y.value_counts()
    print(f"\nClass distribution — Pass: {class_counts.get(1,0)}, Fail: {class_counts.get(0,0)}")

    if class_counts.min() < 2:
        print("WARNING: Not enough samples in one class for ML. Skipping.")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        models = {
            "Logistic Regression": Pipeline([
                ("scaler", StandardScaler()),
                ("clf",    LogisticRegression(max_iter=2000))
            ]),
            "Random Forest": RandomForestClassifier(
                n_estimators=300, random_state=42, class_weight="balanced"
            ),
            "Gradient Boosting": GradientBoostingClassifier(
                n_estimators=200, random_state=42
            ),
            "SVM": Pipeline([
                ("scaler", StandardScaler()),
                ("clf",    SVC(kernel="rbf", probability=True, class_weight="balanced"))
            ]),
        }

        results = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred  = model.predict(X_test)
            acc     = accuracy_score(y_test, y_pred)
            cv_mean = cross_val_score(model, X, y, cv=5, scoring="accuracy").mean()
            results[name] = {"model": model, "acc": acc, "cv": cv_mean, "y_pred": y_pred}
            print(f"\n--- {name} ---")
            print(f"  Test accuracy  : {acc:.4f}")
            print(f"  5-fold CV mean : {cv_mean:.4f}")
            print(classification_report(y_test, y_pred, target_names=["Fail", "Pass"]))

        # Confusion matrices
        fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 4))
        for ax, (name, r) in zip(axes, results.items()):
            cm = confusion_matrix(y_test, r["y_pred"])
            ConfusionMatrixDisplay(cm, display_labels=["Fail", "Pass"]).plot(ax=ax, colorbar=False)
            ax.set_title(f"{name}\nAcc={r['acc']:.3f}", fontsize=10)
        plt.suptitle("Confusion Matrices", fontsize=13, fontweight="bold")
        plt.tight_layout()
        show_or_save(fig, "09_confusion_matrices", args.save_plots, args.output_dir)

        # Model accuracy comparison
        fig, ax = plt.subplots(figsize=(7, 4))
        names = list(results.keys())
        test_accs = [results[n]["acc"] for n in names]
        cv_accs   = [results[n]["cv"]  for n in names]
        x = np.arange(len(names))
        ax.bar(x - 0.2, test_accs, 0.35, label="Test Accuracy",  color="#3498db")
        ax.bar(x + 0.2, cv_accs,   0.35, label="5-Fold CV Mean", color="#e67e22")
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=15, ha="right")
        ax.set_ylim(0, 1.1); ax.set_ylabel("Accuracy")
        ax.set_title("Model Accuracy Comparison", fontsize=13, fontweight="bold")
        ax.legend()
        show_or_save(fig, "10_model_comparison", args.save_plots, args.output_dir)

        # Best model → feature importances (RF only)
        rf = results["Random Forest"]["model"]
        importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
        print("\nTop Feature Importances (Random Forest):")
        print(importances.head(10).round(4).to_string())

        fig, ax = plt.subplots(figsize=(8, 4))
        importances.head(10).sort_values().plot(kind="barh", ax=ax, color="#9b59b6")
        ax.set_title("Top 10 Feature Importances (Random Forest)", fontsize=13, fontweight="bold")
        ax.set_xlabel("Importance")
        show_or_save(fig, "11_feature_importances", args.save_plots, args.output_dir)

        # ── 8. Predict example student ────────────────────────────────────────
        print("\n" + "=" * 60)
        print("  EXAMPLE STUDENT PREDICTION (Random Forest)")
        print("=" * 60)

        example = {c: 68 for c in subject_cols}
        example["Average"]       = np.mean(list(example.values()))
        example["MaxSubject"]    = max(example[c] for c in subject_cols)
        example["MinSubject"]    = min(example[c] for c in subject_cols)
        example["StdDevMarks"]   = 0.0
        example["SubjectsPassed"] = sum(1 for c in subject_cols if example[c] >= args.pass_mark)
        if "Attendance" in df.columns:
            example["Attendance"] = 85

        ex_df = pd.DataFrame([example])[feature_cols]
        pred  = rf.predict(ex_df)[0]
        prob  = rf.predict_proba(ex_df)[0][1]
        pct   = percentile_rank(df["Average"], example["Average"])

        print(f"\nExample student marks (all subjects = 68):")
        print(f"  Predicted result  : {'PASS ✓' if pred == 1 else 'FAIL ✗'}")
        print(f"  Pass probability  : {prob:.1%}")
        print(f"  Percentile rank   : {pct}th")
        print(f"  Grade             : {grade(example['Average'])}")

    # ── 9. Summary report ─────────────────────────────────────────────────────
    report_path = os.path.join(args.output_dir, "analysis_report.txt")
    with open(report_path, "w") as f:
        f.write("STUDENT MARKS ANALYSIS — SUMMARY REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total students   : {len(df)}\n")
        f.write(f"Subjects         : {', '.join(subject_cols)}\n")
        f.write(f"Pass mark        : {args.pass_mark}\n")
        f.write(f"Pass rate        : {pass_rate:.1f}%\n")
        f.write(f"Class average    : {df['Average'].mean():.2f}\n\n")
        f.write("Grade distribution:\n")
        f.write(df["Grade"].value_counts().sort_index().to_string() + "\n\n")
        f.write("Subject averages:\n")
        f.write(df[subject_cols].mean().sort_values(ascending=False).round(2).to_string() + "\n\n")
        f.write("Top 5 students:\n")
        f.write(top10.head(5)[["Name", "Average", "Grade"]].to_string(index=False) + "\n")
    print(f"\n[9] Summary report saved → {report_path}")

    print("\n" + "=" * 60)
    print("  DONE ✓")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
