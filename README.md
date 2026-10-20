# Animal Species Classification 

A beginner friendly machine learning project that predicts an animal's **species type**
(mammal, bird, reptile, fish, amphibian, insect, or invertebrate) from simple
observable traits such as whether it has hair, feathers, produces milk, lays eggs,
breathes air, has a backbone, and how many legs it has.

## Project Overview

- **Problem type:** Multi-class classification (7 animal groups)
- **Goal:** Given a short checklist of physical traits, predict which species group an animal belongs to.
- **Why it's useful:** Grouping organisms by traits is a core task in biology and ecology. It's also a clean demonstration of how just a few well-chosen features can separate very different kinds of animals.
- **Models compared:** Decision Tree and K-Nearest Neighbors (KNN).

## Dataset

**UCI Zoo dataset** — 101 animals described by 16 features plus a species-type label.

- UCI page: https://archive.ics.uci.edu/dataset/111/zoo
- OpenML mirror (used in the code): https://www.openml.org/d/62

The dataset is downloaded automatically in the notebook via
`sklearn.datasets.fetch_openml("zoo", version=1)`, so there is **no manual download step**.

**Features:** 15 boolean traits (`hair`, `feathers`, `eggs`, `milk`, `airborne`, `aquatic`,
`predator`, `toothed`, `backbone`, `breathes`, `venomous`, `fins`, `tail`, `domestic`,
`catsize`) and 1 numeric trait (`legs`).

**Target:** `type` — one of `mammal`, `bird`, `fish`, `invertebrate`, `insect`, `reptile`, `amphibian`.

## Project Structure

```
.
├── animal_classification.ipynb   # Main Jupyter notebook (run this)
├── animal_classification.py      # Same code as copy-paste cells (# %% markers)
├── requirements.txt              # Python dependencies
├── .gitignore
└── README.md
```

## Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd <repo-folder>
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python3 -m venv .venv
   mac: source .venv/bin/activate        # On Windows: .venv\Scripts\activate
   ```

3. **Install the dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter and open the notebook**
   ```bash
   jupyter notebook animal_classification.ipynb
   ```
   Then run all cells (Kernel → Restart & Run All). An internet connection is
   needed the first time so scikit-learn can download the dataset from OpenML.

> Prefer not to use the notebook? Paste the blocks from `animal_classification.py`
> (split by `# %%`) into notebook cells, or run `python animal_classification.py`.

## Workflow

1. **Data Preparation** — load the dataset, check the target balance, confirm there are no missing values, convert the `true`/`false` traits to `1`/`0`, and encode the target labels.
2. **Model Building** — train a Decision Tree and a KNN classifier on a stratified 75/25 train/test split.
3. **Evaluation** — compare accuracy and display a confusion matrix for each model, plus per-class precision/recall.
4. **Interpretation** — plot the Decision Tree and its feature importances.

## Results

Evaluated on a stratified held-out test set of 26 animals (`random_state=42`):

| Model          | Test Accuracy |
|----------------|:-------------:|
| Decision Tree  | **0.9231**    |
| KNN (k=3)      | **0.9231**    |

Both models reached **92.3% accuracy**. They tie on overall accuracy but differ on
the rarest groups: with only a single test example each for *amphibian* and *reptile*,
the Decision Tree got the reptile right while KNN got the amphibian right. The common,
well-represented groups (mammal, bird, fish) were classified perfectly by both.

The Decision Tree's feature-importance analysis shows the classification is driven by a
few decisive traits — **milk** (identifies mammals), **feathers** (identifies birds),
and **backbone**/**fins** (help separate fish and invertebrates).

## Conclusion

- **Better model:** Both tie at 0.9231 accuracy on this split. The **Decision Tree** is the
  more practical choice here because it is just as accurate, trains instantly, and is
  **interpretable** — it shows the exact rules it used (e.g. "produces milk → mammal").
- **What I learned:**
  1. A few good features (milk, feathers, backbone) explain most of the classification.
  2. Categorical/boolean data must be encoded into numbers before modeling.
  3. Class imbalance hurts rare groups — amphibians and reptiles have too few examples for the models to learn confidently.
  4. Interpretable models like the Decision Tree make it easy to understand *why* a prediction was made, not just *what* it predicted.

## Tech Stack

Python · scikit-learn · pandas · NumPy · matplotlib · seaborn · Jupyter
