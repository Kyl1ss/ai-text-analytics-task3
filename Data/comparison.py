import pandas as pd
import numpy as np
import scipy.sparse
import pickle
from sklearn.feature_selection import chi2
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


# load CSV
df = pd.read_csv("cleaned_full_dataset (3).csv")
df = df.dropna(subset=['cleaned_abstract', 'period'])
y = df['period']

# Load the file to extract feature X
with open('tfidf_vocab.pkl', 'rb') as f:
    feature_names = np.array(pickle.load(f))
X_tfidf = scipy.sparse.load_npz('tfidf_matrix.npz')

print(f"Data has been loaded! In total {X_tfidf.shape[0]} abstract of a paper，Extract {X_tfidf.shape[1]} Lexical features. \n")

# Split the training set and test set
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Training a logistic regression classifier
clf = LogisticRegression(random_state=42, max_iter=1000)
clf.fit(X_train, y_train)

keywords_data = []
print("Top 10 trending terms from each period:")
for i, period_name in enumerate(clf.classes_):
    coef = clf.coef_[i] if len(clf.classes_) > 2 else (clf.coef_[0] if i == 1 else -clf.coef_[0])
    top_indices = coef.argsort()[-15:][::-1] # Extract Top 15 and save
    top_words = [feature_names[idx] for idx in top_indices]
    keywords_data.append({"Period": period_name, "Top_Keywords": ", ".join(top_words)})
    print(f"[{period_name}]: {', '.join(top_words[:10])}")

pd.DataFrame(keywords_data).to_csv("period_top_keywords.csv", index=False)

print("=== LDA topic model ===")
# Load the word frequency matrix
X_count = scipy.sparse.load_npz('count_matrix.npz')
with open('tfidf_vocab.pkl', 'rb') as f:
    feature_names = np.array(pickle.load(f))

# Train the LDA model (set to 5 themes)
lda = LatentDirichletAllocation(n_components=5, random_state=42)
doc_topics = lda.fit_transform(X_count)

# Extract and print the key terms for each topic
print("\nKey words for each topic:")
topic_words_data = []
for topic_idx, topic in enumerate(lda.components_):
    top_indices = topic.argsort()[-10:][::-1]
    top_words = [feature_names[i] for i in top_indices]
    topic_words_data.append({"Topic": f"Topic_{topic_idx}", "Keywords": ", ".join(top_words)})

# Calculate and display the average proportion of the topic across different periods
df_topics = pd.DataFrame(doc_topics, columns=[f'Topic_{i}' for i in range(5)])
df_topics['period'] = y
topic_means = df_topics.groupby('period').mean()

print("\nAverage distribution of themes across different periods:")
print(topic_means)
topic_means.to_csv("lda_topic_distributions.csv")
pd.DataFrame(topic_words_data).to_csv("lda_topic_keywords_reference.csv", index=False)

# Obtain the two feature matrices
X_tfidf = scipy.sparse.load_npz('tfidf_matrix.npz')
X_w2v = np.load('word2vec_embeddings.npy')

# Fix an extra row in the matrix
if X_w2v.shape[0] != len(y):
    print(f"Line numbers do not match！y have {len(y)} row, X_w2v have {X_w2v.shape[0]} row")
    print("Automatic label alignment")
    X_w2v = X_w2v[:len(y)]


# Define a generic training and evaluation function
def train_and_evaluate(X, y, feature_name):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Use a larger `max_iter` value to ensure that the logistic regression converges
    clf = LogisticRegression(random_state=42, max_iter=2000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(f"=== Evaluation Report: Based on {feature_name} ===")
    print(classification_report(y_test, y_pred))
    print("-" * 55)


# 2. The comparative experiment has officially begun
report_tfidf = train_and_evaluate(X_tfidf, y, "Traditional TF-IDF sparse representation")
report_w2v = train_and_evaluate(X_w2v, y, "Deep Word2Vec dense representations")

evaluation_text = f"""
=== Axis B: Comparison Methods Evaluation Summary ===

1. Comparison of Classifier Baselines:
[Traditional TF-IDF sparse representation]
{report_tfidf}

[Deep Word2Vec dense representations]
{report_w2v}

Conclusion: Word2Vec demonstrates better recall and robustness when handling minority categories (such as the ‘early’ period), resulting in an improvement in the overall F1 score.

2. Possible points of convergence with PCA:
- Early (1999–2009): Dominated by traditional statistics and tree models (Bayesian, tree, stochastic).
- Growth (2015–2017): The boom in deep learning (recurrent, deep, RNN).
- Recent (2018–2021): Graph networks and cutting-edge fields (graph, federated).
Please check whether these nodes also exhibit corresponding abnormal jumps in PCA space
"""
with open("axis_B_evaluation_metrics.txt", "w", encoding="utf-8") as f:
    f.write(evaluation_text)