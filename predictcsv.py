from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.ml.feature import StringIndexerModel, StandardScalerModel
from pyspark.sql.functions import col

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("Event Detection Prediction") \
    .config("spark.sql.warehouse.dir", "/tmp/spark-warehouse") \
    .getOrCreate()

# Load the trained model and preprocessors
model = LogisticRegressionModel.load("event_detection_model")
scaler_model = StandardScalerModel.load("scaler_model")

# Load the StringIndexer models for categorical variables
indexer_paths = [
    ('country_txt_index_indexer', 'country_txt'),
    ('region_txt_index_indexer', 'region_txt'),
    ('provstate_index_indexer', 'provstate'),
    ('city_index_indexer', 'city'),
    ('targtype1_txt_index_indexer', 'targtype1_txt'),
    ('natlty1_txt_index_indexer', 'natlty1_txt'),
    ('weaptype1_txt_index_indexer', 'weaptype1_txt'),
]
indexers = {original: StringIndexerModel.load(path) for path, original in indexer_paths}

# Load StringIndexer for the target variable
target_indexer = StringIndexerModel.load("attacktype1_txt_index_indexer")

# Load the dataset
df = spark.read.csv("dataset.csv", header=True, inferSchema=True)

# Preprocess and convert categorical variables to indices (similar to single prediction)
for col_name, indexer in indexers.items():
    df = indexer.transform(df)

# Assemble features into a vector
df = df.withColumn("features", Vectors.dense(
    col('iyear'),
    col('imonth'),
    col('iday'),
    col('nkill'),
    col('nkillus'),
    col('nkillter'),
    col('nwound'),
    col('nwoundus'),
    col('nwoundte'),
    col('country_txt'),
    col('region_txt'),
    col('provstate'),
    col('city'),
    col('targtype1_txt'),
    col('natlty1_txt'),
    col('weaptype1_txt'),
))

# Scale the features
scaled_df = scaler_model.transform(df)

# Make predictions using the trained model
predictions = model.transform(scaled_df)

# Add the actual label (assuming you have a column for the ground truth, e.g., 'attacktype1_txt')
predictions = predictions.withColumn("actual_label", predictions['attacktype1_txt'])

# Compare predicted and actual labels, and filter rows where they match
correct_predictions_df = predictions.filter(col("prediction") == col("actual_label"))

# Show or save the filtered dataset
correct_predictions_df.select("features", "prediction", "probability", "actual_label").show()

# Optionally, save the filtered dataframe to a new CSV file
correct_predictions_df.coalesce(1).write.csv("filtered_predictions.csv", header=True)
