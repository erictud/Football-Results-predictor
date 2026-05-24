import pandas as pnd
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore') # ignoring warnings

# AUXILIARY FUNCTIONS

def analyze_missing_vals(dataframe, dataset_name):
    print(f"--- Analyzing missing values for {dataset_name} ---")

    # extracting the number of missing values on each column
    missing_values = dataframe.isnull().sum()

    # caclulating the % of missing values on each column
    missing_percent = (dataframe.isnull().sum() / len(dataframe)) * 100

    # creating a table with the result
    missing_vals_table = pnd.DataFrame({
        "Number of missing values: ": missing_values,
        "Percent of missing values: ": missing_percent
    })

    # showing results
    print(missing_vals_table)

def get_descriptive_stats(dataframe, dataset_name):
    print(f"\n--- Descriptive stats for: {dataset_name} ---")

    # 1. Treating numerical valules
    print("Stats for numerical values:")
    print(dataframe.describe().round(2)) # Folosim round(2) pentru a nu afișa prea multe zecimale

    # 2. Categorical values
    print("Categorical values:")
    print(dataframe.describe(include=['object']))

def generate_distribution_charts(dataframe, dataset_name):
    print(f"Generate distribution charts for {dataset_name}...")

    # generating histograms for numerical values
    numerical_cols = ["HT goals of home team", "HT goals away team", "Goal diff HT", "Total goals HT", "Goals per minute at HT"]
    
    for col in numerical_cols:
        df_train[col] = pnd.to_numeric(df_train[col])
        df_test[col] = pnd.to_numeric(df_test[col])
        plt.figure(figsize=(8, 5))
        
        # setting styiling
        dataframe[col].plot(kind='hist', bins=10, color='skyblue', edgecolor='black')
        
        plt.title(f'Historgram: Distribution of {col} (in {dataset_name})')
        plt.xlabel(col)
        plt.ylabel('Number of matches (freq)')
        
        # saving chart as an image
        file_name = f'./distribution/hist_{col}_{dataset_name}.png'
        plt.savefig(file_name)
        plt.close()

    # countplot for categorical values
    categorical_cols = ['Result', 'HT advantage']
    
    for col in categorical_cols:
        plt.figure(figsize=(8, 5))
        
        numbered_vals = dataframe[col].value_counts().sort_index()
        numbered_vals.plot(kind='bar', color=['#4C72B0', '#DD8452', '#55A868'], edgecolor='black')
        
        plt.title(f'Countplot: Distribution of {col} (in {dataset_name})')
        plt.xlabel(col)
        plt.ylabel('Number of matches')
        
        # rotating
        plt.xticks(rotation=0)
        
        # saving coutplot as image
        file_name = f'./distribution/countplot_{col}_{dataset_name}.png'
        plt.savefig(file_name)
        plt.close()

def detect_outlier_iqr(dataframe, dataset_name):
    print(f"\n--- Outlier detect ({dataset_name}) ---")
    
    numerical_cols = ["HT goals of home team", "HT goals away team", "Total goals HT"]
    
    for col in numerical_cols:
        # generating chart
        plt.figure(figsize=(6, 4))
        dataframe[col].plot(kind='box', vert=False, color='red')
        plt.title(f'Boxplot: Outlieri for {col} (in {dataset_name})')
        
        # saving chart
        file_name = f'./outlier/boxplot_{col}_{dataset_name}.png'
        plt.savefig(file_name)
        plt.close()
        
        # IQR method
        # generating Quartila 1 (25%) and Quartila 3 (75%)
        Q1 = dataframe[col].quantile(0.25)
        Q3 = dataframe[col].quantile(0.75)
        IQR = Q3 - Q1
        
        # defining inferior/superior limits
        inf_lim = Q1 - 1.5 * IQR
        sup_lim = Q3 + 1.5 * IQR
        
        # Counting the num of outliers
        outlieri = dataframe[(dataframe[col] < inf_lim) | (dataframe[col] > sup_lim)]
        numar_outlieri = len(outlieri)
        
        print(f"{col}: detected {numar_outlieri} outliers.")

def generate_heatmap_corelation(dataframe, dataset_name):
    print(f"Generating corelation matrix for {dataset_name}...")
    
    # selecting numerical cols
    numerical_cols = [
        "HT goals of home team", "HT goals away team", "Total goals HT"
    ]
    
    # calculating correlation matrix
    corr_matrix = dataframe[numerical_cols].corr()
    
    # constructing chart
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # coloring cells
    cax = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    
    # adding legend
    fig.colorbar(cax)
    
    # setting name on axis
    ax.set_xticks(np.arange(len(numerical_cols)))
    ax.set_yticks(np.arange(len(numerical_cols)))
    
    # rotating test
    ax.set_xticklabels(numerical_cols, rotation=45, ha='right')
    ax.set_yticklabels(numerical_cols)
    
    # adding numbers
    for i in range(len(numerical_cols)):
        for j in range(len(numerical_cols)):
            valoare = round(corr_matrix.iloc[i, j], 2)
            
            # changing color
            text_color = "white" if abs(valoare) > 0.6 else "black"
            
            ax.text(j, i, f"{valoare}", ha='center', va='center', color=text_color)
            
    plt.title(f'Heatmap: corelation matrix ({dataset_name})', pad=20)
    plt.tight_layout()
    
    # saving photo
    file_name = f'./corelation/heatmap_{dataset_name}.png'
    plt.savefig(file_name)
    plt.close()

def generate_violin_chart(dataframe, dataset_name):
    print(f"Generating violion plot for target variable ({dataset_name})...")
    
    col_target = 'Result' 
    col_feature = 'Goal diff HT' # we choose goal diff
    
    # finding unique classes
    class_target = sorted(dataframe[col_target].unique())
    
    # creating a list of lists for each class
    date_pentru_grafic = [dataframe[dataframe[col_target] == Class][col_feature].values for Class in class_target]
    
    # initialize chart
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # generating violin plot
    parts = ax.violinplot(date_pentru_grafic, positions=class_target, showmeans=True)
    
    # styling
    for pc in parts['bodies']:
        pc.set_facecolor('skyblue')
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)
        
    # adding titles 
    plt.title(f'Violin Plot: realtion betwen goal diff at HT and Result ({dataset_name})', pad=15)
    plt.xlabel('Target (final result: 0=tie, 1=home, 2=away)')
    plt.ylabel('Goal diff at HT (home - away)')
    
    # setting labels
    plt.xticks(class_target, [f'Class {int(c)}' for c in class_target])
    
    # saving img
    file_name = f'./relations_target/violin_target_{dataset_name}.png'
    plt.savefig(file_name)
    plt.close()

# MAIN PROGRAMM

# importing datasets
df_train = pnd.read_csv("train.csv")
df_test = pnd.read_csv("test.csv")

# analizing missing values for each data frame
analyze_missing_vals(df_train, "training dataset")
analyze_missing_vals(df_test, "testing dataset")

# getting descriptive stats
get_descriptive_stats(df_train, "training dataset")
get_descriptive_stats(df_test, "testing dataset")

# generating histograms for the distribution of variables
generate_distribution_charts(df_train, "train dataset")
generate_distribution_charts(df_test, "test dataset")

# detecting outliers using IQR
detect_outlier_iqr(df_train, "train dataset")
detect_outlier_iqr(df_test, "test dataset")

# analyzing correlations
generate_heatmap_corelation(df_train, "train dataset")
generate_heatmap_corelation(df_test, "test dataset")

# analyzing relations between columns and target var
generate_violin_chart(df_train, "train dataset")
generate_violin_chart(df_test, "test dataset")