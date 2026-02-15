"""
Sports vs Politics Text Classifier — Comparing ML Techniques.

This script builds and compares three machine learning classifiers:
  1. Multinomial Naive Bayes
  2. Logistic Regression
  3. Support Vector Machine (Linear SVM)

Each classifier is tested with three feature representations:
  - Bag of Words (BoW) with unigrams
  - Bag of Words with bigrams (n-grams)
  - TF-IDF (Term Frequency - Inverse Document Frequency)

The script performs stratified 5-fold cross-validation and reports
accuracy, precision, recall, and F1-score for each combination.

Usage:
    pip install scikit-learn
    python classifier.py

Author: Saurav Soni (B22AI035)
Course: CSL 7640 — Natural Language Understanding
"""

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════
# SECTION 1 — DATASET
#
# Curated dataset of 178 sentences covering diverse sub-topics.
# Sports: cricket, football, basketball, tennis, olympics, general
# Politics: elections, parliament, policy, international, governance
#
# Data was collected by paraphrasing real news headlines from sources
# like BBC Sport, ESPN, Reuters Politics, The Hindu — ensuring
# originality while maintaining realistic vocabulary and phrasing.
# ══════════════════════════════════════════════════════════════════

# ── SPORTS sentences (93 total) ──
sports_sentences = [
    # Cricket (15 sentences)
    "the batsman scored a brilliant century in the test match",
    "india won the cricket world cup after a thrilling final",
    "the bowler took five wickets in the innings",
    "virat kohli played an outstanding knock in the odi series",
    "the ipl auction saw record breaking bids for young players",
    "the cricket team celebrated their victory with a lap of honor",
    "the spinner bowled a magnificent spell to restrict the opponents",
    "the opening partnership was crucial in chasing the target",
    "the umpire gave a controversial decision during the match",
    "the cricket board announced the schedule for the upcoming series",
    "the wicketkeeper took a stunning catch behind the stumps",
    "the test match ended in a draw after five days of play",
    "the captain won the toss and elected to bat first",
    "the fast bowler clocked speeds above 150 kmph consistently",
    "the cricket stadium was packed with enthusiastic fans",
    # Football / Soccer (15 sentences)
    "the striker scored a hat trick in the champions league match",
    "the goalkeeper made an incredible save to keep the team ahead",
    "the football club signed a new midfielder for fifty million euros",
    "the world cup final attracted millions of viewers worldwide",
    "the coach implemented a new tactical formation for the season",
    "the defender received a red card for a dangerous tackle",
    "the team won the league title after a dominant season",
    "the penalty shootout decided the winner of the tournament",
    "the footballer celebrated his goal with the fans",
    "the transfer window saw several high profile deals completed",
    "the football association banned the player for unsporting behavior",
    "the referee awarded a controversial penalty in stoppage time",
    "the team qualified for the knockout stage of the competition",
    "the football match was postponed due to heavy rain",
    "the youth academy produced several talented players this year",
    # Basketball (10 sentences)
    "the point guard scored thirty points in the nba finals",
    "the basketball team clinched the championship in seven games",
    "the center dominated the paint with rebounds and blocks",
    "the three point shooting was exceptional throughout the game",
    "the basketball coach drew up a perfect play during the timeout",
    "the slam dunk contest thrilled fans at the all star weekend",
    "the rookie showed immense potential in his debut season",
    "the free throw shooting was crucial in the close game",
    "the basketball player signed a max contract extension",
    "the arena erupted as the buzzer beater went through the net",
    # Tennis (8 sentences)
    "the tennis player won the grand slam title in straight sets",
    "the serve and volley strategy worked perfectly on grass courts",
    "the match went to five sets in an epic wimbledon final",
    "the tennis ranking changed after the tournament results",
    "the forehand winner sealed the championship point",
    "the tennis academy trains young athletes from around the world",
    "the doubles pair won their first major title together",
    "the clay court season began with the monte carlo masters",
    # Olympics (6 sentences)
    "the athlete won a gold medal in the hundred meter sprint",
    "the olympic games showcased incredible sporting talent",
    "the swimmer broke the world record in the butterfly event",
    "the gymnast performed a flawless routine on the balance beam",
    "the relay team set a new national record at the olympics",
    "the marathon runner crossed the finish line in first place",
    # General sports (39 sentences)
    "the sports federation announced new rules for the competition",
    "the athlete tested positive for a banned substance",
    "the coach praised the team for their disciplined performance",
    "the sports ministry allocated more funds for athlete training",
    "the championship trophy was lifted by the winning captain",
    "the tournament draw was announced at the press conference",
    "the training camp prepared the players for the upcoming season",
    "the fitness coach designed a specialized training program",
    "the sports injury forced the player to miss the tournament",
    "the commentator praised the outstanding display of sportsmanship",
    "the home crowd gave the team a standing ovation after the win",
    "the playoff series was one of the most exciting in recent memory",
    "the team roster was finalized before the tournament deadline",
    "the coach announced the starting lineup for the big game",
    "the athlete broke through the tape at the finish line",
    "the sports analyst predicted a close contest between the rivals",
    "the halftime show entertained fans during the break",
    "the sports channel broadcast the match live to global audiences",
    "the player of the match award went to the star performer",
    "the victory parade was held in the city center",
    "the under nineteen team won the junior world championship",
    "the sports event attracted sponsors from major corporations",
    "the athletic meet saw several records being broken",
    "the fans celebrated wildly after the last minute goal",
    "the league standings changed dramatically after the weekend results",
    "the franchise drafted the top pick in the annual selection",
    "the stadium renovation was completed before the new season",
    "the esports tournament attracted thousands of online viewers",
    "the racing driver won the grand prix after a brilliant overtake",
    "the boxing champion defended his title successfully",
    "the hockey team scored in overtime to win the series",
    "the volleyball team qualified for the world championship",
    "the table tennis player won a bronze medal at the games",
    "the badminton final was a closely contested affair",
    "the wrestling bout went into extra time before a winner emerged",
    "the archery competition saw a new record in the qualifying round",
    "the cycling race covered over two hundred kilometers",
    "the golf tournament ended with a dramatic playoff hole",
    "the rugby team scored a last minute try to win the match",
]

# ── POLITICS sentences (85 total) ──
politics_sentences = [
    # Elections (15 sentences)
    "the prime minister won the general election with a huge majority",
    "the opposition leader challenged the ruling party policies",
    "the election commission announced the dates for state elections",
    "the political rally attracted thousands of supporters",
    "voter turnout increased significantly in the recent elections",
    "the exit polls predicted a close contest between the two parties",
    "the candidate filed nomination papers for the constituency",
    "the election results surprised political analysts across the nation",
    "the coalition government was formed after lengthy negotiations",
    "the by election was held to fill the vacant parliamentary seat",
    "the political campaign focused on development and employment",
    "the debate between candidates was broadcast on national television",
    "the electoral reforms were proposed to ensure fair elections",
    "the voting machines were tested before the election day",
    "the political party released its manifesto ahead of the polls",
    # Parliament / Legislature (15 sentences)
    "the parliament passed a new bill on education reform",
    "the budget session saw heated debates on fiscal policy",
    "the opposition staged a walkout during the parliamentary session",
    "the speaker of the house called for order during the debate",
    "the legislative committee reviewed the proposed amendments",
    "the senate voted to approve the trade agreement",
    "the parliamentary inquiry investigated allegations of corruption",
    "the government introduced a new policy on healthcare",
    "the constitutional amendment required a two thirds majority",
    "the minister presented the annual report to the parliament",
    "the bill was referred to a select committee for review",
    "the floor leader urged members to support the legislation",
    "the joint session of parliament discussed national security",
    "the parliamentary session was adjourned due to disruptions",
    "the new law was enacted after receiving presidential assent",
    # Policy and Governance (15 sentences)
    "the government announced a new economic stimulus package",
    "the tax reform bill aims to simplify the taxation system",
    "the foreign minister met with international diplomats",
    "the defense budget was increased by fifteen percent this year",
    "the healthcare policy covers all citizens below the poverty line",
    "the education minister launched a new digital literacy program",
    "the infrastructure project received government approval",
    "the trade policy was revised to boost exports",
    "the environmental regulation was strengthened to reduce pollution",
    "the central bank announced changes to the monetary policy",
    "the social welfare scheme benefited millions of families",
    "the government imposed sanctions on the neighboring country",
    "the public distribution system was reformed for better efficiency",
    "the labor law amendments were met with mixed reactions",
    "the regulatory body issued new guidelines for the industry",
    # International Relations (15 sentences)
    "the bilateral summit focused on trade and security cooperation",
    "the united nations general assembly discussed climate change",
    "the diplomatic talks between the two nations reached a breakthrough",
    "the peace agreement was signed after years of conflict",
    "the foreign policy shift caused controversy in diplomatic circles",
    "the international sanctions were imposed on the regime",
    "the ambassador presented credentials to the head of state",
    "the treaty was ratified by both countries after lengthy talks",
    "the geopolitical tensions escalated in the region",
    "the humanitarian aid was sent to the conflict affected areas",
    "the international community condemned the military action",
    "the trade war between the superpowers affected global markets",
    "the diplomatic relations were restored after a decade of tension",
    "the strategic partnership was strengthened through the new accord",
    "the border dispute was discussed at the international tribunal",
    # Governance and Administration (25 sentences)
    "the chief minister inaugurated the new government hospital",
    "the bureaucratic reforms aimed to reduce red tape and corruption",
    "the governor signed the executive order on public safety",
    "the municipal corporation increased property tax rates",
    "the district administration took measures for disaster management",
    "the government official was charged with misuse of public funds",
    "the transparency act requires disclosure of government spending",
    "the public hearing was held regarding the new zoning regulations",
    "the administrative reforms commission submitted its recommendations",
    "the government launched a campaign against black money",
    "the political scandal led to the resignation of the minister",
    "the civic body elections were held across the state",
    "the government advisory committee met to discuss economic reforms",
    "the federal structure allows states to have their own legislation",
    "the judiciary upheld the constitutional validity of the new law",
    "the president addressed the nation on the anniversary of independence",
    "the government formed a task force to tackle unemployment",
    "the political alliance broke apart over policy disagreements",
    "the supreme court delivered a landmark judgment on civil rights",
    "the protest movement demanded changes to the agricultural policy",
    "the government allocated funds for rural development programs",
    "the ministry of finance released the quarterly economic report",
    "the political leadership met to discuss the national crisis",
    "the governor declared a state of emergency in the region",
    "the diplomatic envoy was recalled following the border incident",
]


def get_dataset():
    """Return the full labelled dataset as (texts, labels) lists.
    Combines sports and politics sentences with corresponding labels."""
    texts = sports_sentences + politics_sentences
    labels = (["sports"] * len(sports_sentences) +
              ["politics"] * len(politics_sentences))
    return texts, labels


# ══════════════════════════════════════════════════════════════════
# SECTION 2 — PIPELINE CONSTRUCTION
#
# We build 9 pipelines: 3 feature extractors × 3 classifiers.
# Each pipeline is a sklearn Pipeline with a vectorizer + classifier.
# ══════════════════════════════════════════════════════════════════

def build_pipelines():
    """
    Build all 9 combinations of (feature representation × ML technique).

    Feature representations:
      - BoW unigram:  CountVectorizer with unigrams only
      - BoW bigram:   CountVectorizer with unigrams + bigrams
      - TF-IDF:       TfidfVectorizer with unigrams + bigrams

    ML Techniques:
      - Multinomial Naive Bayes (with Laplace smoothing alpha=1)
      - Logistic Regression (L2 regularized, C=1.0)
      - Linear SVM (hinge loss, C=1.0)

    Returns a dict mapping (feature_name, model_name) → Pipeline.
    """
    # Define feature extractors
    feature_extractors = {
        "BoW (Unigram)": CountVectorizer(
            lowercase=True, stop_words="english", ngram_range=(1, 1)
        ),
        "BoW (Bigram)": CountVectorizer(
            lowercase=True, stop_words="english", ngram_range=(1, 2)
        ),
        "TF-IDF": TfidfVectorizer(
            lowercase=True, stop_words="english", ngram_range=(1, 2)
        ),
    }

    # Define classifiers
    classifiers = {
        "Naive Bayes": MultinomialNB(alpha=1.0),        # Laplace smoothing
        "Logistic Regression": LogisticRegression(
            max_iter=1000, C=1.0, solver="lbfgs"
        ),
        "Linear SVM": LinearSVC(max_iter=2000, C=1.0),
    }

    # Build all 9 pipelines
    pipelines = {}
    for feat_name, feat_ext in feature_extractors.items():
        for clf_name, clf in classifiers.items():
            pipe = Pipeline([
                ("vectorizer", feat_ext),
                ("classifier", clf),
            ])
            pipelines[(feat_name, clf_name)] = pipe

    return pipelines


# ══════════════════════════════════════════════════════════════════
# SECTION 3 — EVALUATION (Stratified 5-Fold Cross-Validation)
# ══════════════════════════════════════════════════════════════════

def evaluate_pipelines(pipelines, texts, labels, n_folds=5):
    """
    Evaluate each pipeline using stratified k-fold cross-validation.

    Stratified folds ensure each fold has roughly the same class
    distribution as the full dataset. We collect accuracy, precision,
    recall, and F1 for each fold and report the mean ± std.

    Args:
        pipelines: dict of (feat, model) → Pipeline
        texts: list of sentences
        labels: list of class labels
        n_folds: number of CV folds (default 5)

    Returns:
        results: dict mapping (feat, model) → dict of metric means/stds
    """
    texts_arr = np.array(texts)
    labels_arr = np.array(labels)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    results = {}

    for (feat_name, clf_name), pipe in pipelines.items():
        # Collect metrics across folds
        fold_metrics = {"accuracy": [], "precision": [], "recall": [], "f1": []}

        for train_idx, test_idx in skf.split(texts_arr, labels_arr):
            X_train, X_test = texts_arr[train_idx], texts_arr[test_idx]
            y_train, y_test = labels_arr[train_idx], labels_arr[test_idx]

            # Train and predict
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)

            # Calculate metrics (Sports as positive class)
            fold_metrics["accuracy"].append(accuracy_score(y_test, y_pred))
            fold_metrics["precision"].append(
                precision_score(y_test, y_pred, pos_label="sports", zero_division=0)
            )
            fold_metrics["recall"].append(
                recall_score(y_test, y_pred, pos_label="sports", zero_division=0)
            )
            fold_metrics["f1"].append(
                f1_score(y_test, y_pred, pos_label="sports", zero_division=0)
            )

        # Compute mean and std for each metric
        results[(feat_name, clf_name)] = {
            metric: (np.mean(vals), np.std(vals))
            for metric, vals in fold_metrics.items()
        }

    return results


# ══════════════════════════════════════════════════════════════════
# SECTION 4 — OUTPUT FORMATTING
# ══════════════════════════════════════════════════════════════════

def print_results_table(results):
    """Print a formatted comparison table of all results."""
    print("\n" + "=" * 90)
    print(f"{'Feature':<18} {'Classifier':<22} {'Accuracy':>10} {'Precision':>10} "
          f"{'Recall':>10} {'F1-Score':>10}")
    print("=" * 90)

    for (feat, clf), metrics in sorted(results.items()):
        acc_m, acc_s = metrics["accuracy"]
        pre_m, pre_s = metrics["precision"]
        rec_m, rec_s = metrics["recall"]
        f1_m, f1_s = metrics["f1"]
        print(f"{feat:<18} {clf:<22} {acc_m:>8.2%}±{acc_s:.2f} "
              f"{pre_m:>8.2%}±{pre_s:.2f} {rec_m:>8.2%}±{rec_s:.2f} "
              f"{f1_m:>8.2%}±{f1_s:.2f}")

    print("=" * 90)


def generate_latex_table(results):
    """Generate a LaTeX table string for the report."""
    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\caption{5-Fold Cross-Validation Results (\%)}")
    lines.append(r"\label{tab:results}")
    lines.append(r"\begin{tabular}{llcccc}")
    lines.append(r"\hline")
    lines.append(r"\textbf{Feature} & \textbf{Classifier} & \textbf{Accuracy} & "
                 r"\textbf{Precision} & \textbf{Recall} & \textbf{F1} \\")
    lines.append(r"\hline")

    for (feat, clf), metrics in sorted(results.items()):
        acc_m, acc_s = metrics["accuracy"]
        pre_m, pre_s = metrics["precision"]
        rec_m, rec_s = metrics["recall"]
        f1_m, f1_s = metrics["f1"]
        lines.append(
            f"{feat} & {clf} & {acc_m*100:.1f}$\\pm${acc_s*100:.1f} & "
            f"{pre_m*100:.1f}$\\pm${pre_s*100:.1f} & "
            f"{rec_m*100:.1f}$\\pm${rec_s*100:.1f} & "
            f"{f1_m*100:.1f}$\\pm${f1_s*100:.1f} \\\\"
        )

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════
# SECTION 5 — FINAL TRAINING AND INTERACTIVE DEMO
# ══════════════════════════════════════════════════════════════════

def final_train_and_demo(texts, labels):
    """Train the best model on full data and run interactive demo.
    Uses TF-IDF + Linear SVM which is a strong general-purpose combo."""
    best_pipe = Pipeline([
        ("vectorizer", TfidfVectorizer(
            lowercase=True, stop_words="english", ngram_range=(1, 2)
        )),
        ("classifier", LinearSVC(max_iter=2000, C=1.0)),
    ])
    best_pipe.fit(texts, labels)

    # Print full-data classification report for reference
    y_pred = best_pipe.predict(texts)
    print("\nClassification Report (on full training data, TF-IDF + SVM):")
    print(classification_report(labels, y_pred, target_names=["politics", "sports"]))

    cm = confusion_matrix(labels, y_pred, labels=["politics", "sports"])
    print(f"Confusion Matrix:\n{cm}\n")

    # Interactive demo loop
    print("-" * 60)
    print("Interactive Sports vs Politics Classifier")
    print("(Type a sentence, or 'quit' to exit)")
    print("-" * 60)

    while True:
        try:
            user_input = input("\nEnter text: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if not user_input or user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        prediction = best_pipe.predict([user_input])[0]
        print(f"Prediction: {prediction.upper()}")


# ══════════════════════════════════════════════════════════════════
# SECTION 6 — MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════

def main():
    """Main entry point: load data, evaluate, display results, demo."""
    print("=" * 60)
    print("  Sports vs Politics Text Classifier")
    print("  CSL 7640: Natural Language Understanding")
    print("  Saurav Soni (B22AI035)")
    print("=" * 60)

    # Load dataset from embedded data above
    texts, labels = get_dataset()
    print(f"\nDataset: {len(texts)} sentences "
          f"({labels.count('sports')} sports, {labels.count('politics')} politics)")

    # Build all 9 pipeline combinations
    pipelines = build_pipelines()
    print(f"Evaluating {len(pipelines)} model-feature combinations "
          f"with 5-fold cross-validation...\n")

    # Evaluate with stratified 5-fold CV
    results = evaluate_pipelines(pipelines, texts, labels, n_folds=5)

    # Print results comparison table
    print_results_table(results)

    # Generate and save LaTeX table for report
    latex_table = generate_latex_table(results)
    print("\nLaTeX Table (for report):")
    print(latex_table)

    with open("results_table.tex", "w") as f:
        f.write(latex_table)
    print("\n(Table saved to results_table.tex)")

    # Train best model on full data and run interactive demo
    final_train_and_demo(texts, labels)


if __name__ == "__main__":
    main()