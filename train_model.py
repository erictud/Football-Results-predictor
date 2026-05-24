import pandas as pnd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

def train_and_test(df_train, df_test):
    # data cleanup
    to_be_deleted_cols = ['Home team', 'Away team', 'Total goals HT', 'Result']
    
    # defining characteristics (input data)
    input_train = df_train.drop(columns=to_be_deleted_cols, errors='ignore')
    input_test = df_test.drop(columns=to_be_deleted_cols, errors='ignore')
    
    # defining target variable
    target_train = df_train['Result']
    target_test = df_test['Result']
    
    # training model
    print("Training random forest model...")
    model = RandomForestClassifier(random_state=42)
    model.fit(input_train, target_train)
    
    # testing model based on test.csv data frame
    print("Generating predictions...")
    predictions = model.predict(input_test)
    
    # Accuracy
    accuracy = accuracy_score(target_test, predictions)
    print(f"Accuracy: {accuracy * 100:.2f}%\n")
    
    # 5. generating confision matrix
    fig, ax = plt.subplots(figsize=(7, 5))
    
    disp = ConfusionMatrixDisplay.from_estimator(
        model, 
        input_test, 
        target_test,
        display_labels=['Class 0 (Tie)', 'Class 1 (home)', 'Class 2 (away)'],
        cmap='Blues',
        ax=ax
    )
    
    plt.title('Confusion matrix- based on test data')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
    print("SUCCES: Confusion matrix has been succesfully generated!")

# MAIN PROGRAMM

# importing datasets
df_train = pnd.read_csv("train.csv")
df_test = pnd.read_csv("test.csv")

# training
train_and_test(df_train, df_test)