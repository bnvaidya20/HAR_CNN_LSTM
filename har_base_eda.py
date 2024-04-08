
import os
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


# Data Loading
    
class DataLoader: 
    def __init__(self, directory):
        self.directory = directory

    def load_data(self):

        data_files = {}  

        # Walk through the directory
        for dirname, _, filenames in os.walk(self.directory):
            for filename in filenames:
                # Check if the file is a CSV
                if filename.endswith('.csv'):
                    file_path = os.path.join(dirname, filename)
                    print(f"Loading {file_path}...")  
                    # Load the CSV file into a DataFrame
                    data_files[filename] = pd.read_csv(file_path)
        
        return data_files  
     

class DataPreprocessor:
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data

    def get_basic_info(self):
        print(f"Training data shape: {self.train_data.shape}")
        print(f"Training data:\n {self.train_data.head()}")
        print(f"Training data type:\n {self.train_data.dtypes}")
        print(f"Training data desc:\n {self.train_data.describe().T}")
        print(f"Test data shape: {self.test_data.shape}")

    def check_basic_values(self):
        print(f"Duplicates in Training Data: {self.train_data.duplicated().sum()}")
        print(f"Missing Values in Training Data:\n{self.train_data.isna().sum()}")

    def handle_missing_values(self):
        self.train_data.dropna(inplace=True)
        self.test_data.dropna(inplace=True)

    def get_counts_activity(self):
        return print(self.train_data['Activity'].groupby(self.train_data['Activity']).size())




# Exploratory Data Analysis (EDA)

class EDAVisualizer:
    def __init__(self, df):
        self.df=df

    # Sensor data usage count
    def plot_sensor_counts(self):
        sensor_counts = [self.df.columns.str.contains('Acc').sum(), self.df.columns.str.contains('Gyro').sum()]
        plt.figure(figsize=(10, 6))
        plt.bar(['Accelerometer', 'Gyroscope'], sensor_counts, color=['skyblue', 'salmon'])
        plt.title("Sensor Data Usage", fontsize=20)
        plt.xlabel("Sensor Type", fontsize=16)
        plt.ylabel("Count", fontsize=16)
        plt.show()

    # Data distribution by user and activity
    def plot_user_activity_distribution(self):
        plt.figure(figsize=(16, 8))
        sns.countplot(x='subject', hue='Activity', data=self.df, palette="tab10")
        plt.title('Data Provided by Each User by Activity', fontsize=16)
        plt.xlabel("User", fontsize=14)
        plt.ylabel("Count", fontsize=14)
        plt.xticks(fontsize=12)
        plt.legend(title='Activity', fontsize=12, title_fontsize='13')
        plt.show()

    # Distribution of Sensor Values by Activity
    def plot_bodyaccmean_activity_distribution(self):
        plt.figure(figsize=(16, 6))
        sns.kdeplot(data=df_train, x='tBodyAcc-mean()-X', hue='Activity', fill=True, common_norm=False, palette="crest", alpha=0.5, linewidth=0)
        plt.title("Distribution of Body Acceleration Mean (X) by Activity", fontsize=16)
        plt.xlabel("Body Acceleration Mean (X)", fontsize=14)
        plt.ylabel("Density", fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.show()

    def plot_bodyaccmagmean_activity_distribution(self):
        plt.figure(figsize=(16, 6))
        sns.kdeplot(data=df_train, x='tBodyAccMag-mean()', hue='Activity', fill=True, common_norm=False, palette="crest", alpha=0.5, linewidth=0)
        plt.title("Distribution of Body Acceleration Mag Mean by Activity", fontsize=16)
        plt.xlabel("Body Acceleration Mag Mean ", fontsize=14)
        plt.ylabel("Density", fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.show()

    # Box Plots for Feature by Activity
    def plot_feature_by_activity(self, feature):
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=self.df, x='Activity', y=feature)
        plt.title(f'Distribution of {feature} by Activity')
        plt.xlabel("Activity", size=14)
        plt.ylabel(feature, size=14)
        plt.xticks(rotation=45)
        plt.show()

    # Body Accelerometer Magnitude Mean by Activity
    def plot_magnitude_mean_activity(self):

        plt.figure(figsize=(10, 7))
        sns.boxplot(x='Activity', y='tBodyAccMag-mean()', data=self.df, showfliers=False, saturation=1)
        plt.axhline(y=-0.7, xmin=0.1, xmax=0.9, dashes=(5, 5), c='green')
        plt.axhline(y=-0.05, xmin=0.4, dashes=(5, 5), c='magenta')
        plt.title("Accelerometer Magnitude Mean by Activity", fontsize=16)
        plt.xlabel("Activity", fontsize=14)
        plt.ylabel('Accelerometer Magnitude Mean', fontsize=14)
        plt.xticks(rotation=45, fontsize=12)
        plt.show()

    # Activity Distribution
    def plot_activity_distribution(self, datatype='Training'):
        plt.figure(figsize=(10, 6))
        sns.countplot(data=self.df, x='Activity')
        plt.title(f'Activity Distribution in {datatype} Data')
        plt.xlabel("Activity", size=14)
        plt.xticks(rotation=45)
        plt.show()

    # Pair Plot for Selected Features:
    def pairplot_features(self):
        sns.pairplot(df_train, vars=['tBodyAcc-mean()-X', 'tBodyAcc-mean()-Y', 'tBodyAcc-mean()-Z', 'tBodyAccMag-mean()'], hue='Activity')
        plt.suptitle("Pairwise Feature Distribution by Activity", y=1.02, fontsize=16)
        plt.show()

    # Correlation Heatmap
    def plot_correlation_matrix(self):
        corr = self.df.corr(numeric_only=True)
        plt.figure(figsize=(15, 10))
        sns.heatmap(corr, annot=False, fmt=".1f", cmap='coolwarm')
        plt.title('Correlation Matrix')
        plt.show()



loader=DataLoader(directory='./input')
data_files = loader.load_data()

df_train=data_files['train.csv']
df_test=data_files['test.csv']



preprocessor= DataPreprocessor(df_train, df_test)

preprocessor.get_basic_info()
preprocessor.check_basic_values()
preprocessor.handle_missing_values()
preprocessor.get_counts_activity()

visualizer=EDAVisualizer(df_train)

visualizer1=EDAVisualizer(df_test)

visualizer.plot_activity_distribution()

visualizer1.plot_activity_distribution(datatype='Test')

visualizer.plot_user_activity_distribution()

visualizer.plot_bodyaccmean_activity_distribution()

visualizer.plot_bodyaccmagmean_activity_distribution()

visualizer.plot_sensor_counts()

feature='angle(X,gravityMean)'
visualizer.plot_feature_by_activity(feature)

visualizer.plot_magnitude_mean_activity()

visualizer.pairplot_features()

visualizer.plot_correlation_matrix()


df_train.to_csv('./input/df_train.csv', index=False)
df_test.to_csv('./input/df_test.csv', index=False)
