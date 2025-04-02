from flask import Flask, render_template, request
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.ml.linalg import Vectors
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.ml.feature import StringIndexerModel, StandardScalerModel
from pyspark.sql import SparkSession

# Initialize Flask App
app = Flask(__name__)

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("Single Prediction with Spark Model") \
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

# Prepare static path to save graphs
static_path = "static/images"
os.makedirs(static_path, exist_ok=True)

# Flask Route for the form to get user input
@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the form data
        input_data = {
            'iyear': int(request.form['iyear']),
            'imonth': int(request.form['imonth']),
            'iday': int(request.form['iday']),
            'country_txt': request.form['country_txt'],
            'region_txt': request.form['region_txt'],
            'provstate': request.form['provstate'],
            'city': request.form['city'],
            'targtype1_txt': request.form['targtype1_txt'],
            'natlty1_txt': request.form['natlty1_txt'],
            'weaptype1_txt': request.form['weaptype1_txt'],
            'nkill': int(request.form['nkill']),
            'nkillus': int(request.form['nkillus']),
            'nkillter': int(request.form['nkillter']),
            'nwound': int(request.form['nwound']),
            'nwoundus': int(request.form['nwoundus']),
            'nwoundte': int(request.form['nwoundte']),
        }

        # Convert categorical variables to indices using StringIndexer models
        for col, indexer in indexers.items():
            if col in input_data:
                input_data[col] = indexer.labels.index(input_data[col]) if input_data[col] in indexer.labels else len(indexer.labels)  # Default to unknown index

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

        # Decode the prediction to the target label
        predicted_label_index = int(predictions.collect()[0]['prediction'])
        predicted_label = target_indexer.labels[predicted_label_index]

        # Prepare and save plots
        save_plots()

        # Render the result page
        return render_template('prediction.html', prediction=predicted_label)

    return render_template('index.html')

# Function to generate and save plots
def save_plots():
    # Load dataset (replace 'your_data.csv' with the actual path to your data file)
    df = spark.read.csv("dataset.csv", header=True, inferSchema=True)

    # Generate Top 10 Most Frequent Attack Types Bar Plot
    attack_counts = df.groupBy('attacktype1_txt').count().orderBy('count', ascending=False).limit(10).toPandas()
    plt.bar(attack_counts['attacktype1_txt'], attack_counts['count'])
    plt.xlabel("Attack Type")
    plt.ylabel("Frequency")
    plt.title("Top 10 Most Frequent Attack Types")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(static_path, "attack_types.png"))
    plt.close()

    # Generate Correlation Matrix Heatmap
    numerical_features_df = df.select('nkill', 'nkillus', 'nkillter', 'nwound', 'nwoundus', 'nwoundte').toPandas()
    correlation_matrix = numerical_features_df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
    plt.title("Correlation Matrix of Numerical Features")
    plt.tight_layout()
    plt.savefig(os.path.join(static_path, "correlation_matrix.png"))
    plt.close()

    # Generate Attack Types by Region Stacked Bar Plot
    attack_by_region = df.groupBy('region_txt', 'attacktype1_txt').count().toPandas()
    attack_by_region_pivot = attack_by_region.pivot_table(index='region_txt', columns='attacktype1_txt', values='count', aggfunc='sum')
    attack_by_region_pivot.plot(kind='bar', stacked=True)
    plt.title("Attack Types by Region")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(static_path, "attack_types_by_region.png"))
    plt.close()

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
