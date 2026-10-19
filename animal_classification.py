# Animal Species Classification
# -----------------------------------------------------------------------------
# Copy each block (separated by "# %%") into its own Jupyter notebook cell.
# Lines starting with "# [markdown]" are notes you can paste into a Markdown cell.
# -----------------------------------------------------------------------------

# %% [markdown]
# # Animal Species Classification
#
# ## 1. The Idea
# Predict an animal's species type (mammal, bird, reptile, fish, amphibian,
# insect, invertebrate) from simple observable traits (hair, feathers, eggs,
# milk, legs, etc.). It is a clean multi-class classification problem.
#
# Dataset: UCI Zoo dataset (101 animals, 16 features).
#   - Source: https://archive.ics.uci.edu/dataset/111/zoo
#   - OpenML mirror: https://www.openml.org/d/62

# %%
# 2.1 Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

sns.set_theme(style="whitegrid")
RANDOM_STATE = 42
print("Libraries imported successfully.")

# %%
# 2.2 Load the dataset
zoo = fetch_openml("zoo", version=1, as_frame=True)
df = zoo.frame.copy()
print(f"Dataset shape: {df.shape[0]} rows (animals), {df.shape[1]} columns")
df.head()

# %%
df.info()

# %%
# 2.3 Check the target balance
print(df["type"].value_counts())
print()
print("Proportions:")
print(df["type"].value_counts(normalize=True).round(3))

plt.figure(figsize=(8, 4))
order = df["type"].value_counts().index
sns.countplot(y="type", data=df, order=order, hue="type", palette="Set2", legend=False)
plt.title("Animal type distribution")
plt.xlabel("Count")
plt.ylabel("Species type")
plt.tight_layout()
plt.show()

# %%
# 2.4 Check for missing values
missing = df.isnull().sum()
print("Missing values per column:")
print(missing[missing > 0] if missing.sum() > 0 else "None - the dataset is complete.")
print(f"\nTotal missing values: {df.isnull().sum().sum()}")

# %%
# 2.5 Prepare features (X) and target (y)
X = df.drop(columns=["type"]).copy()
y_raw = df["type"].astype(str)

# Convert boolean-like 'true'/'false' columns to 1/0.
for col in X.columns:
    if X[col].dtype.name in ("category", "object"):
        X[col] = X[col].astype(str).map({"true": 1, "false": 0})
X = X.apply(pd.to_numeric)  # 'legs' stays as its integer count

# Encode the target labels to integers.
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y_raw)

print("Label encoding:")
for i, name in enumerate(target_encoder.classes_):
    print(f"  {i} -> {name}")
print(f"\nFeature matrix shape: {X.shape}")
X.head()

# %%
# 2.6 Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
)
print(f"Training set: {X_train.shape[0]} animals")
print(f"Test set:     {X_test.shape[0]} animals")

# %%
# 3.1 Model A - Decision Tree
dtree = DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE)
dtree.fit(X_train, y_train)
dtree_pred = dtree.predict(X_test)
dtree_acc = accuracy_score(y_test, dtree_pred)
print(f"Decision Tree test accuracy: {dtree_acc:.4f}")

# %%
# 3.2 Model B - K-Nearest Neighbors (KNN)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)
knn_acc = accuracy_score(y_test, knn_pred)
print(f"KNN test accuracy: {knn_acc:.4f}")

# %%
# 4.1 Accuracy comparison
results = pd.DataFrame({
    "Model": ["Decision Tree", "KNN"],
    "Accuracy": [dtree_acc, knn_acc],
}).sort_values("Accuracy", ascending=False).reset_index(drop=True)
print(results.to_string(index=False))

plt.figure(figsize=(6, 4))
sns.barplot(x="Model", y="Accuracy", data=results, hue="Model", palette="Set2", legend=False)
plt.ylim(0, 1.05)
plt.title("Model accuracy comparison")
for i, v in enumerate(results["Accuracy"]):
    plt.text(i, v + 0.02, f"{v:.4f}", ha="center", fontweight="bold")
plt.show()

# %%
# 4.2 Confusion matrices
labels = target_encoder.classes_
present = np.unique(np.concatenate([y_test, dtree_pred, knn_pred]))
display_labels = [labels[i] for i in present]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, (name, pred) in zip(axes, [("Decision Tree", dtree_pred), ("KNN", knn_pred)]):
    cm = confusion_matrix(y_test, pred, labels=present)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation=45)
    ax.set_title(f"{name}\nAccuracy = {accuracy_score(y_test, pred):.4f}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
plt.tight_layout()
plt.show()

# %%
print("=== Decision Tree ===")
print(classification_report(y_test, dtree_pred, labels=present,
                            target_names=display_labels, zero_division=0))
print("=== KNN ===")
print(classification_report(y_test, knn_pred, labels=present,
                            target_names=display_labels, zero_division=0))

# %%
# 4.3 What the Decision Tree learned
plt.figure(figsize=(20, 10))
plot_tree(dtree, feature_names=X.columns, class_names=labels,
          filled=True, rounded=True, fontsize=9, max_depth=3)
plt.title("Decision Tree (top levels)")
plt.show()

# %%
importances = (pd.Series(dtree.feature_importances_, index=X.columns)
               .sort_values(ascending=False).head(10))
plt.figure(figsize=(8, 5))
sns.barplot(x=importances.values, y=importances.index, hue=importances.index,
            palette="viridis", legend=False)
plt.title("Top feature importances (Decision Tree)")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()
print(importances)

# %%
# 5. Conclusion
better = "Decision Tree" if dtree_acc >= knn_acc else "KNN"
worse = "KNN" if better == "Decision Tree" else "Decision Tree"
better_acc = max(dtree_acc, knn_acc)
worse_acc = min(dtree_acc, knn_acc)

print("FINAL RESULTS")
print("-" * 40)
print(f"Decision Tree accuracy : {dtree_acc:.4f}")
print(f"KNN accuracy           : {knn_acc:.4f}")
print("-" * 40)
print(f"Best model: {better} ({better_acc:.4f}) vs {worse} ({worse_acc:.4f})")
