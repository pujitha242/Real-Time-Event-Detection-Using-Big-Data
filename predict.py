from pyspark.ml.linalg import Vectors
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.ml.feature import StringIndexerModel, StandardScalerModel
from pyspark.sql import SparkSession

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("Single Prediction with Spark Model") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "8g") \
    .config("spark.sql.warehouse.dir", "/tmp/spark-warehouse") \
    .getOrCreate()



# Load the trained model and preprocessors
model = LogisticRegressionModel.load("event_detection_model")
scaler_model = StandardScalerModel.load("scaler_model")

# Load the StringIndexer models for categorical variables
indexer_paths = [
    ('country_txt_index', 'country_txt'),
    ('region_txt_index', 'region_txt'),
    ('provstate_index', 'provstate'),
    ('city_index', 'city'),
    ('targtype1_txt_index', 'targtype1_txt'),
    ('natlty1_txt_index', 'natlty1_txt'),
    ('weaptype1_txt_index', 'weaptype1_txt'),
]
indexers = {original: StringIndexerModel.load(path) for path, original in indexer_paths}

# Load StringIndexer for the target variable
target_indexer = StringIndexerModel.load("attacktype1_txt_index")

# Input Data: Replace with your input values
input_data = { 
    'iyear': 1970,
    'imonth': 1,
    'iday': 10,
    'country_txt': 'East Germany (GDR)',
    'region_txt': 'Eastern Europe',
    'provstate': 'Berlin',
    'city': 'Berlin',
    'targtype1_txt': 'Government (General)',
    'natlty1_txt': 'Germany',
    'weaptype1_txt': 'Explosives',
    'nkill': 0,
    'nkillus': 0,
    'nkillter': 0,
    'nwound': 0,
    'nwoundus': 0,
    'nwoundte': 0,
}

# Convert categorical variables to indices using StringIndexer models
for col, indexer in indexers.items():
    if col in input_data:
        input_data[col] = indexer.labels.index(input_data[col]) if input_data[col] in indexer.labels else len(indexer.labels)  # Default to unknown index
    else:
        raise KeyError(f"Missing required key '{col}' in input_data")

# Assemble features into a vector
features = [
    input_data['iyear'],
    input_data['imonth'],
    input_data['iday'],
    input_data['nkill'],
    input_data['nkillus'],
    input_data['nkillter'],
    input_data['nwound'],
    input_data['nwoundus'],
    input_data['nwoundte'],
    input_data['country_txt'],
    input_data['region_txt'],
    input_data['provstate'],
    input_data['city'],
    input_data['targtype1_txt'],
    input_data['natlty1_txt'],
    input_data['weaptype1_txt'],
]

# Create a DataFrame for the single input
single_input_df = spark.createDataFrame([(Vectors.dense(features),)], ["features"])

# Scale the features
single_input_df = scaler_model.transform(single_input_df)

# Predict using the trained model
predictions = model.transform(single_input_df)

# Show prediction results
predictions.select("features", "prediction", "probability").show()

# Decode the prediction to the target label
predicted_label_index = int(predictions.collect()[0]['prediction'])
predicted_label = target_indexer.labels[predicted_label_index]
print(f"Predicted Attack Type: {predicted_label}")
import matplotlib.pyplot as plt
# Load dataset (replace 'your_data.csv' with the actual path to your data file)
df = spark.read.csv("dataset.csv", header=True, inferSchema=True)

# Now, you can proceed with your analysis code
attack_counts = df.groupBy('attacktype1_txt').count().orderBy('count', ascending=False).limit(10).toPandas()

# Create a bar plot
plt.bar(attack_counts['attacktype1_txt'], attack_counts['count'])
plt.xlabel("Attack Type")
plt.ylabel("Frequency")
plt.title("Top 10 Most Frequent Attack Types")
plt.xticks(rotation=90)
plt.show()



import seaborn as sns

# Select the numerical features for correlation analysis
numerical_features_df = df.select('nkill', 'nkillus', 'nkillter', 'nwound', 'nwoundus', 'nwoundte').toPandas()

# Calculate and plot the correlation matrix
correlation_matrix = numerical_features_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix of Numerical Features")
plt.show()


# Group the data by region and attack type
attack_by_region = df.groupBy('region_txt', 'attacktype1_txt').count().toPandas()

# Pivot the data for a stacked bar chart
attack_by_region_pivot = attack_by_region.pivot_table(index='region_txt', columns='attacktype1_txt', values='count', aggfunc='sum')

# Plot the data
attack_by_region_pivot.plot(kind='bar', stacked=True)
plt.title("Attack Types by Region")
plt.ylabel("Count")
plt.show()


import seaborn as sns

# Prepare the data for boxplot
attack_wounds_df = df.select('attacktype1_txt', 'nkill').toPandas()

# Plot the deaths by attack type
sns.boxplot(x='attacktype1_txt', y='nkill', data=attack_wounds_df)
plt.xticks(rotation=90)
plt.title("Deaths by Attack Type")
plt.show()


# Prepare the data for boxplot
attack_wounds_df = df.select('attacktype1_txt', 'nwound').toPandas()

# Plot the wounds by attack type
sns.boxplot(x='attacktype1_txt', y='nwound', data=attack_wounds_df)
plt.xticks(rotation=90)
plt.title("Wounds by Attack Type")
plt.show()


import matplotlib.pyplot as plt

# Extract the probability column and plot its distribution
predictions.select('probability').toPandas()['probability'].apply(lambda x: x[1]).hist(bins=30)
plt.title("Prediction Probability Distribution")
plt.xlabel("Probability")
plt.ylabel("Frequency")
plt.show()

