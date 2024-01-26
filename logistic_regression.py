from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

def log_reg_probability(df):
    #create new column called label and map positive/negative to 1/0 for use in logistic regression model
    df['label'] = df['argument_type'].map({'positive': 1, 'negative': 0})

    # Splitting the dataset into training and testing sets
    X_train, X_test = train_test_split(df, test_size=0.3, stratify=df['argument_type'], random_state=42)

    #verify split
    #print(f"Training set: \n {X_train['argument_type'].value_counts(normalize=True)}")
    #print(f"Testing set: \n {X_test['argument_type'].value_counts(normalize=True)}")

    # Feature and Label Selection
    features = ['formality', 'subjectivity', 'optimistic vs. cynical tone', 'extremity', 'lexical density']
    X_train_features = X_train[features]
    X_test_features = X_test[features]

    y_train = X_train['label']
    y_test = X_test['label']

    # classification model & evaluation
    model = LogisticRegression()
    model.fit(X_train_features, y_train)
    y_pred = model.predict(X_test_features)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    # Model Initialization and Training
    model = LogisticRegression()
    model.fit(X_train_features, y_train)

    #probability score generation
    y_prob = model.predict_proba(df[features])[:, 1]  # Get probabilities for the positive class
    return y_prob