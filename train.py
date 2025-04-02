from pyspark.sql import SparkSession
from pyspark.sql.functions import when, col
from pyspark.ml.feature import StringIndexer, StandardScaler, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import logging

# Suppress Spark warnings
logging.getLogger("py4j").setLevel(logging.ERROR)
logging.getLogger("pyspark").setLevel(logging.ERROR)

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("Event Detection with Spark") \
    .config("spark.sql.warehouse.dir", "/tmp/spark-warehouse") \
    .getOrCreate()

# Suppress additional logging
spark.sparkContext.setLogLevel("ERROR")

# Load dataset (replace with your actual dataset file path)
file_path = 'datasettrain.csv'  # Replace with your dataset's file path
df = spark.read.csv(file_path, header=True, inferSchema=True)

# Define the features and target
features = ['iyear', 'imonth', 'iday', 'country_txt', 'region_txt', 'provstate', 'city',
            'targtype1_txt', 'natlty1_txt', 'weaptype1_txt', 'nkill', 'nkillus', 'nkillter', 
            'nwound', 'nwoundus', 'nwoundte']
target = 'attacktype1_txt'

# Drop rows with missing target
df = df.na.drop(subset=[target])

# Fill missing values for categorical columns with 'Unknown'
df = df.fillna('Unknown')

# Convert numerical columns to float and handle invalid values
numerical_features = ['nkill', 'nkillus', 'nkillter', 'nwound', 'nwoundus', 'nwoundte']
for col_name in numerical_features:
    df = df.withColumn(
        col_name,
        when(col(col_name).rlike("^[0-9.]+$"), col(col_name).cast("float")).otherwise(0.0)
    )

# Encode categorical features using StringIndexer
indexers = [
    StringIndexer(inputCol=column, outputCol=f"{column}_index").fit(df)
    for column in ['country_txt', 'region_txt', 'provstate', 'city', 'targtype1_txt', 'natlty1_txt', 'weaptype1_txt', target]
]
for indexer in indexers:
    df = indexer.transform(df)

# Rename the target column index
df = df.withColumnRenamed(f"{target}_index", "label")

# Assemble features into a single vector column
indexed_features = [f"{col}_index" for col in ['country_txt', 'region_txt', 'provstate', 'city', 'targtype1_txt', 'natlty1_txt', 'weaptype1_txt']]
all_features = ['iyear', 'imonth', 'iday'] + numerical_features + indexed_features

assembler = VectorAssembler(inputCols=all_features, outputCol="features")
df = assembler.transform(df)

# Standardize numerical features using StandardScaler
scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withStd=True, withMean=False)
scaler_model = scaler.fit(df)
df = scaler_model.transform(df)

# Split the dataset into training and testing sets
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

# Train a Logistic Regression model
lr = LogisticRegression(featuresCol="scaled_features", labelCol="label")
model = lr.fit(train_df)

# Save the trained model
model.write().overwrite().save("event_detection_model")

# Save encoders and scaler
for indexer in indexers:
    indexer.write().overwrite().save(f"{indexer.getOutputCol()}")
scaler_model.write().overwrite().save("scaler_model")

# Evaluate the model
predictions = model.transform(test_df)

# Calculate accuracy
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"Accuracy: {accuracy:.4f}")


import matplotlib.pyplot as plt
import pandas as pd

# Convert Spark DataFrame to Pandas for plotting
df_pd = df.toPandas()

# Plot the number of attacks by year
plt.figure(figsize=(10,6))
df_pd['iyear'].value_counts().sort_index().plot(kind='bar', color='skyblue')
plt.title('Distribution of Attacks by Year')
plt.xlabel('Year')
plt.ylabel('Number of Attacks')
plt.xticks(rotation=90)
plt.show()


# Plot the number of attacks by country
plt.figure(figsize=(10,6))
df_pd['country_txt'].value_counts().head(10).plot(kind='bar', color='salmon')
plt.title('Top 10 Countries by Number of Attacks')
plt.xlabel('Country')
plt.ylabel('Number of Attacks')
plt.xticks(rotation=45)
plt.show()


# Plot the distribution of attack types
plt.figure(figsize=(8,8))
df_pd['attacktype1_txt'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=['lightgreen', 'lightcoral', 'lightblue'])
plt.title('Distribution of Attack Types')
plt.ylabel('')
plt.show()


# Plot the distribution of weapon types
plt.figure(figsize=(10,6))
df_pd['weaptype1_txt'].value_counts().head(10).plot(kind='bar', color='lightseagreen')
plt.title('Top 10 Weapon Types Used in Attacks')
plt.xlabel('Weapon Type')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.show()


import seaborn as sns

# Create a new column to calculate total casualties (nkill + nwound)
df_pd['total_casualties'] = df_pd['nkill'] + df_pd['nwound']

# Plot boxplot for casualties by attack type
plt.figure(figsize=(12,6))
sns.boxplot(x='attacktype1_txt', y='total_casualties', data=df_pd)
plt.xticks(rotation=90)
plt.title('Number of Victims by Attack Type')
plt.xlabel('Attack Type')
plt.ylabel('Total Casualties')
plt.show()


import seaborn as sns

# Pivot table to count attacks by country and attack type
country_attack_pivot = df_pd.groupby(['country_txt', 'attacktype1_txt']).size().unstack(fill_value=0)

# Plot heatmap
plt.figure(figsize=(12,8))
sns.heatmap(country_attack_pivot, cmap='YlGnBu', annot=True, fmt="d", linewidths=0.5)
plt.title('Country vs. Attack Type Heatmap')
plt.xlabel('Attack Type')
plt.ylabel('Country')
plt.show()


attack_casualties = df_pd.groupby('attacktype1_txt')['total_casualties'].sum().sort_values(ascending=False).head(10)

plt.figure(figsize=(10,6))
attack_casualties.plot(kind='bar', color='orange')
plt.title('Top 10 Attack Types by Number of Casualties')
plt.xlabel('Attack Type')
plt.ylabel('Total Casualties')
plt.xticks(rotation=45)
plt.show()


region_casualties = df_pd.groupby('region_txt')[['nkill', 'nwound']].sum().sort_values(by='nkill', ascending=False)

region_casualties.plot(kind='bar', figsize=(12,6), stacked=True, color=['red', 'blue'])
plt.title('Casualties by Region')
plt.xlabel('Region')
plt.ylabel('Number of Casualties')
plt.xticks(rotation=45)
plt.show()


plt.figure(figsize=(12,6))
sns.violinplot(x='weaptype1_txt', y='total_casualties', data=df_pd)
plt.xticks(rotation=90)
plt.title('Casualties by Weapon Type')
plt.xlabel('Weapon Type')
plt.ylabel('Total Casualties')
plt.show()


monthly_attacks = df_pd.groupby('imonth').size()

plt.figure(figsize=(10,6))
monthly_attacks.plot(kind='line', marker='o', color='purple')
plt.title('Monthly Distribution of Attacks')
plt.xlabel('Month')
plt.ylabel('Number of Attacks')
plt.xticks(range(1, 13))
plt.grid(True)
plt.show()
