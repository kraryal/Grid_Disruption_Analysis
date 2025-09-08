import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg for thread safety

from flask import Flask, render_template, url_for, request, jsonify
import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import os
import networkx as nx
from itertools import combinations
import matplotlib.cm as cm
from sklearn.ensemble import IsolationForest, RandomForestClassifier as RFC, VotingClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
import time
import joblib
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, precision_score, recall_score
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE
import socket  # For retrieving IP address
import warnings  # For suppressing warnings

# Add initial logging to confirm app startup
print("Starting Flask app initialization...")

app = Flask(__name__)

# Add logging to confirm Flask app creation
print("Flask app created successfully.")

# Load data globally at app startup
def load_training_data():
    try:
        print("Loading training data from 'data/processed_training_data.csv'...")
        data = pd.read_csv('data/processed_training_data.csv')
        print(f"Training data loaded successfully. Shape: {data.shape}")
        print(f"Columns in training data: {data.columns.tolist()}")
        return data
    except Exception as e:
        print(f"Error loading training data: {str(e)}")
        raise

training_data = load_training_data()

# Load pre-trained models
try:
    print("Loading pre-trained models...")
    model_rf_dict = joblib.load('model_results_rfc.joblib')
    model_xgb_dict = joblib.load('model_results_xgb.joblib')
    model_lgb_dict = joblib.load('model_results_lgb.joblib')
    model_best_dict = joblib.load('model_results_best.joblib')

    # Extract the model instances
    model_rf = model_rf_dict['best_model']
    model_xgb = model_xgb_dict['best_model']
    model_lgb = model_lgb_dict['best_model']
    model_best = model_best_dict['best_model']
    best_model_name = model_best_dict.get('best_model_name', 'Unknown')  # Extract best_model_name if available

    print("Models loaded successfully.")
    print(f"Best model name from model_results_best.joblib: {best_model_name}")
except Exception as e:
    print(f"Error loading models: {str(e)}")
    raise

# Evaluate models and print F1-score, precision, recall, and accuracy
def evaluate_models():
    try:
        print("Starting model evaluation...")
        # Prepare data for evaluation
        if 'duration_category' not in training_data.columns:
            print("Error: Target column 'duration_category' not found in training data.")
            print(f"Available columns: {training_data.columns.tolist()}")
            raise ValueError("Target column 'duration_category' not found in training data.")
        print("Preparing features and target...")
        # Select only the features used during training
        expected_features = ['state_encoded', 'event_type_encoded', 'fips', 'mean_customers', 'month', 'weekday']
        missing_features = [feat for feat in expected_features if feat not in training_data.columns]
        if missing_features:
            print(f"Error: Missing expected features {missing_features} in training data.")
            print(f"Available columns: {training_data.columns.tolist()}")
            raise ValueError(f"Missing expected features {missing_features} in training data.")
        X = training_data[expected_features]
        y = training_data['duration_category']
        print(f"Features shape: {X.shape}, Target shape: {y.shape}")
        print(f"Features columns: {X.columns.tolist()}")

        # Check for missing or invalid data
        if X.isnull().any().any() or y.isnull().any():
            print("Error: Missing values detected in features or target.")
            print(f"Missing values in X:\n{X.isnull().sum()}")
            print(f"Missing values in y: {y.isnull().sum()}")
            raise ValueError("Missing values detected in features or target.")

        # Check data types
        print("Checking data types of features...")
        print(f"X dtypes:\n{X.dtypes}")
        print(f"y dtype: {y.dtype}")
        # Ensure all features are numerical
        non_numerical_features = X.select_dtypes(exclude=['int', 'float', 'bool']).columns
        if not non_numerical_features.empty:
            print(f"Error: Non-numerical features detected: {non_numerical_features.tolist()}")
            raise ValueError(f"Non-numerical features detected: {non_numerical_features.tolist()}")

        # Split data
        print("Splitting data into training and validation sets...")
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"Training set shape: {X_train.shape}, Validation set shape: {X_val.shape}")
        print(f"Training target distribution:\n{y_train.value_counts()}")
        print(f"Validation target distribution:\n{y_val.value_counts()}")

        # Models dictionary
        models_dict = {
            'RandomForest': model_rf,
            'XGBoost': model_xgb,
            'LightGBM': model_lgb,
            'Best Model (VotingClassifier)': model_best
        }

        # Evaluate each model
        best_f1 = 0
        best_model_name = ''
        # Suppress UndefinedMetricWarning during metric calculation
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message="Recall is ill-defined")
            for model_name, model in models_dict.items():
                print(f"Evaluating {model_name}...")
                try:
                    # Check model attributes
                    if hasattr(model, 'feature_names_in_'):
                        print(f"{model_name} expected features: {model.feature_names_in_}")
                    else:
                        print(f"{model_name} does not have feature_names_in_ attribute (likely an ensemble model).")

                    # Make predictions
                    print(f"Making predictions with {model_name}...")
                    y_pred = model.predict(X_val)
                    print(f"Predictions shape: {y_pred.shape}")
                    print(f"Sample predictions: {y_pred[:5]}")

                    # Calculate metrics
                    print(f"Calculating metrics for {model_name}...")
                    f1 = f1_score(y_val, y_pred, average='weighted')
                    precision = precision_score(y_val, y_pred, average='weighted')
                    recall = recall_score(y_val, y_pred, average='weighted')
                    accuracy = accuracy_score(y_val, y_pred)
                    print(f"{model_name} Metrics:")
                    print(f"  F1-score: {f1:.3f}")
                    print(f"  Precision: {precision:.3f}")
                    print(f"  Recall: {recall:.3f}")
                    print(f"  Accuracy: {accuracy:.3f}\n")
                    if f1 > best_f1:
                        best_f1 = f1
                        best_model_name = model_name
                except Exception as e:
                    print(f"Error evaluating {model_name}: {str(e)}")
                    continue

        print(f"Best Model: {best_model_name} (F1-score: {best_f1:.3f})")
    except Exception as e:
        print(f"Error during model evaluation: {str(e)}")
        raise

# Run evaluation at startup with confirmation
print("Calling evaluate_models() to evaluate models at startup...")
try:
    evaluate_models()
    print("Model evaluation completed successfully.")
except Exception as e:
    print(f"Failed to evaluate models: {str(e)}")
finally:
    # Get the machine's IP address
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))  # Connect to a public DNS server (Google's DNS)
        ip_address = s.getsockname()[0]
        s.close()
    except Exception as e:
        print(f"Error getting IP address: {str(e)}")
        ip_address = "127.0.0.1"  # Fallback to localhost if IP retrieval fails

    port = 5000
    # Print the network-accessible link
    print(f"\nFlask app is running! Access it at:")
    print(f"Local: http://127.0.0.1:{port}/")
    print(f"Network: http://{ip_address}:{port}/")
    print("\nNote: The network link (http://{ip_address}:{port}/) can be used by others on the same network to access the app.")

# Load Data Functions
def load_aggregated_data():
    return pd.read_csv('data/Combined_Aggregated_Outage_Data.csv')

def load_events_data():
    return pd.read_csv('data/Combined_events.csv')

def load_merged_data():
    return pd.read_csv('data/Combined_merged_Data.csv')

# Function to preprocess the data (used for both training and prediction)
def preprocess_data(df, state_encoder=None, event_encoder=None, scaler=None, county_fips_dict=None, for_training=False):
    # Calculate duration if not present
    if 'duration' not in df.columns and 'start_time' in df.columns and 'end_time' in df.columns:
        df['start_time'] = pd.to_datetime(df['start_time'])
        df['end_time'] = pd.to_datetime(df['end_time'])
        df['duration'] = (df['end_time'] - df['start_time']).dt.total_seconds() / 3600  # Duration in hours
    elif 'duration' not in df.columns:
        df['duration'] = 2  # Default to 2 hours if duration cannot be calculated

    # Create duration categories (0 to 4)
    def map_duration_to_category(duration):
        if duration <= 4:
            return 0  # 0-4 hours
        elif duration <= 24:
            return 1  # 4-24 hours
        elif duration <= 72:
            return 2  # 24-72 hours
        elif duration <= 168:
            return 3  # 72-168 hours
        else:
            return 4  # 168+ hours

    if for_training:
        df['duration_category'] = df['duration'].apply(map_duration_to_category)

    # Extract features
    df['start_time'] = pd.to_datetime(df['start_time'])
    df['month'] = df['start_time'].dt.month
    df['weekday'] = df['start_time'].dt.day_name()

    # Encode categorical features
    if for_training:
        state_encoder = LabelEncoder()
        event_encoder = LabelEncoder()
        df['state_encoded'] = state_encoder.fit_transform(df['state'])
        df['event_type_encoded'] = event_encoder.fit_transform(df['event_type'])
        joblib.dump(state_encoder, 'state_encoder.joblib')
        joblib.dump(event_encoder, 'event_encoder.joblib')
    else:
        df['state_encoded'] = state_encoder.transform(df['state'])
        df['event_type_encoded'] = event_encoder.transform(df['event_type'])

    # Map county to FIPS
    df['fips'] = df['county'].map(county_fips_dict)
    if df['fips'].isnull().any():
        raise ValueError("Some counties could not be mapped to FIPS codes. Check the county_fips_mapping.joblib file.")

    # Handle mean_customers
    if 'mean_customers' in df.columns:
        df['mean_customers'] = df['mean_customers'].fillna(0).astype('float64')
    else:
        df['mean_customers'] = 0  # Placeholder if not available

    # Map weekday to numerical values
    Weekday_dict = {
        'Sunday': 0, 'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6
    }
    df['weekday'] = df['weekday'].map(Weekday_dict)

    # Apply log transformation to mean_customers
    df['mean_customers'] = np.log1p(df['mean_customers'])

    # Scale mean_customers
    if for_training:
        scaler = StandardScaler()
        df['mean_customers'] = scaler.fit_transform(df[['mean_customers']])
        joblib.dump(scaler, 'scaler.joblib')
    else:
        # Ensure df[['mean_customers']] is a 2D DataFrame before passing to scaler.transform
        scaled_values = scaler.transform(df[['mean_customers']])
        df['mean_customers'] = scaled_values.flatten()  # Flatten to match the Series shape

    return df, state_encoder, event_encoder, scaler

# Function to generate the network graph image
def generate_network_graph():
    df = load_merged_data()
    df['start_time'] = pd.to_datetime(df['start_time'])
    df['node'] = df['state'] + ' - ' + df['county']
    df['date'] = df['start_time'].dt.date

    sample_dates = df['date'].sort_values(ascending=False).unique()[:7]
    G = nx.Graph()

    for date in sample_dates:
        nodes_on_date = df[df['date'] == date]['node'].unique()
        for node_a, node_b in combinations(nodes_on_date, 2):
            if G.has_edge(node_a, node_b):
                G[node_a][node_b]['weight'] += 1
            else:
                G.add_edge(node_a, node_b, weight=1)

    edges_limited = sorted(G.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)[:50]
    limited_graph = nx.Graph()
    for u, v, d in edges_limited:
        limited_graph.add_edge(u, v, weight=d['weight'])

    degrees = dict(limited_graph.degree())
    degree_values = list(degrees.values())
    norm = plt.Normalize(min(degree_values), max(degree_values))
    node_colors = [cm.viridis(norm(degrees[node])) for node in limited_graph.nodes()]

    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(limited_graph, seed=42)
    nx.draw_networkx_nodes(limited_graph, pos, node_size=600, node_color=node_colors)
    nx.draw_networkx_edges(limited_graph, pos, width=1, alpha=0.7)
    nx.draw_networkx_labels(limited_graph, pos, font_size=8)
    plt.title("Simulated Outage Connection Network (Colorful Nodes by Connectivity)")
    plt.axis('off')
    plt.tight_layout()

    output_path = 'static/images/network_graph.png'
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(output_path, dpi=300)
    plt.close()

# Generate the network graph image when the app starts
with app.app_context():
    generate_network_graph()

# Preprocess and save the dataset if not already done
if not os.path.exists('data/processed_training_data.csv'):
    # Load and preprocess the real data
    df = load_events_data()  # Try Combined_events.csv first
    print("Columns in Combined_events.csv:", df.columns)
    print("Original dataset size:", len(df))

    # Load county FIPS mapping
    county_fips_dict = joblib.load("county_fips_mapping.joblib")

    # Preprocess the data
    training_df, state_encoder, event_encoder, scaler = preprocess_data(
        df,
        county_fips_dict=county_fips_dict,
        for_training=True
    )

    # Sample the dataset if it's too large (e.g., limit to 50,000 rows)
    max_rows = 50000
    if len(training_df) > max_rows:
        training_df = training_df.sample(n=max_rows, random_state=42)
        print(f"Dataset sampled to {max_rows} rows to speed up training.")
    else:
        print("Dataset size is less than 50,000 rows; using full dataset:", len(training_df))

    # Save the preprocessed dataset
    training_df.to_csv('data/processed_training_data.csv', index=False)
    print("Preprocessed dataset saved as 'data/processed_training_data.csv' with size:", len(training_df))

# Train models if not already trained
if not os.path.exists('model_results_rfc.joblib') or not os.path.exists('model_results_xgb.joblib') or \
   not os.path.exists('model_results_lgb.joblib') or not os.path.exists('model_results_ensemble.joblib') or \
   not os.path.exists('model_results_best.joblib'):
    # Load the preprocessed dataset
    training_df = pd.read_csv('data/processed_training_data.csv')
    print("Loaded preprocessed dataset with size:", len(training_df))

    X = training_df[['state_encoded', 'event_type_encoded', 'fips', 'mean_customers', 'month', 'weekday']]
    y = training_df['duration_category']

    print("Training features:", X.columns.tolist())
    print("Target labels before adjustment:", y.unique())
    print("Class distribution before SMOTE:", y.value_counts().to_dict())

    # Adjust target labels to start from 0 if necessary
    if y.min() != 0:
        print("Adjusting target labels to start from 0...")
        y = y - y.min()
        print("Target labels after adjustment:", y.unique())

    # Split the data before SMOTE to avoid data leakage
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Apply SMOTE to balance the training data
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    print("Class distribution after SMOTE:", pd.Series(y_train_smote).value_counts().to_dict())

    # Define individual models
    rfc = RFC(class_weight='balanced', random_state=42)
    xgb = XGBClassifier(random_state=42)
    lgb = LGBMClassifier(class_weight='balanced', random_state=42)

    # Define hyperparameter grids for lightweight tuning
    param_grid_rfc = {
        'n_estimators': [100],
        'max_depth': [10, None],
        'min_samples_split': [2],
        'min_samples_leaf': [1]
    }
    param_grid_xgb = {
        'n_estimators': [100],
        'max_depth': [3, 6],
        'learning_rate': [0.1]
    }
    param_grid_lgb = {
        'n_estimators': [100],
        'max_depth': [3, 6],
        'learning_rate': [0.1]
    }

    # Perform GridSearchCV for each model
    print("Training RFC...")
    grid_search_rfc = GridSearchCV(rfc, param_grid_rfc, cv=3, scoring='f1_weighted', n_jobs=-1)
    grid_search_rfc.fit(X_train_smote, y_train_smote)
    best_rfc = grid_search_rfc.best_estimator_
    joblib.dump({'best_model': best_rfc}, 'model_results_rfc.joblib')
    print("Best RFC parameters:", grid_search_rfc.best_params_)

    print("Training XGBoost...")
    grid_search_xgb = GridSearchCV(xgb, param_grid_xgb, cv=3, scoring='f1_weighted', n_jobs=-1)
    grid_search_xgb.fit(X_train_smote, y_train_smote)
    best_xgb = grid_search_xgb.best_estimator_
    joblib.dump({'best_model': best_xgb}, 'model_results_xgb.joblib')
    print("Best XGBoost parameters:", grid_search_xgb.best_params_)

    print("Training LightGBM...")
    grid_search_lgb = GridSearchCV(lgb, param_grid_lgb, cv=3, scoring='f1_weighted', n_jobs=-1)
    grid_search_lgb.fit(X_train_smote, y_train_smote)
    best_lgb = grid_search_lgb.best_estimator_
    joblib.dump({'best_model': best_lgb}, 'model_results_lgb.joblib')
    print("Best LightGBM parameters:", grid_search_lgb.best_params_)

    # Create an ensemble model using VotingClassifier
    ensemble_model = VotingClassifier(
        estimators=[
            ('rfc', best_rfc),
            ('xgb', best_xgb),
            ('lgb', best_lgb)
        ],
        voting='soft'  # Use soft voting for probability-based predictions
    )

    print("Training ensemble model...")
    ensemble_model.fit(X_train_smote, y_train_smote)
    joblib.dump({'best_model': ensemble_model}, 'model_results_ensemble.joblib')
    print("Ensemble model saved.")

    # Evaluate all models
    models_to_evaluate = {
        'RandomForest': best_rfc,
        'XGBoost': best_xgb,
        'LightGBM': best_lgb,
        'Ensemble': ensemble_model
    }

    best_model_name = None
    best_f1_score = 0
    for model_name, model in models_to_evaluate.items():
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='weighted')
        print(f"{model_name} F1-score:", f1)
        print(f"{model_name} Classification Report:\n", classification_report(y_test, y_pred))
        if f1 > best_f1_score:
            best_f1_score = f1
            best_model_name = model_name

    print(f"Best model: {best_model_name} with F1-score: {best_f1_score}")

    # Save the best model for prediction along with its name
    joblib.dump({'best_model': models_to_evaluate[best_model_name], 'best_model_name': best_model_name}, 'model_results_best.joblib')

# Load the scaler and encoders for prediction
scaler = joblib.load('scaler.joblib')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/dashboard')
def dashboard():
    df = load_aggregated_data()
    states = df['state'].unique().tolist()
    selected_state = request.args.get('state', 'all')

    if selected_state == 'all':
        filtered_df = df
    else:
        filtered_df = df[df['state'] == selected_state]

    filtered_df['month'] = filtered_df['month'].replace(0, 1)
    filtered_df['date'] = pd.to_datetime(filtered_df[['year', 'month']].assign(day=1))

    min_date = filtered_df['date'].min()
    max_date = filtered_df['date'].max()
    all_dates = pd.date_range(start=min_date, end=max_date, freq='MS')
    monthly_outages = filtered_df.groupby('date').sum()['outage_count'].reset_index()
    monthly_outages = monthly_outages.set_index('date').reindex(all_dates, fill_value=0).reset_index().rename(
        columns={'index': 'date'})

    monthly_outages = monthly_outages.rename(columns={'date': 'ds', 'outage_count': 'y'})

    model = Prophet()
    model.fit(monthly_outages)

    future = model.make_future_dataframe(periods=12, freq='M')
    forecast = model.predict(future)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly_outages['ds'],
        y=monthly_outages['y'],
        mode='lines',
        name='Historical',
        line=dict(color='#1f77b4'),
        hovertemplate='Date: %{x}<br>Outages: %{y}<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat'],
        mode='lines',
        name='Forecast',
        line=dict(color='#ff7f0e'),
        hovertemplate='Date: %{x}<br>Forecasted Outages: %{y}<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat_upper'],
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat_lower'],
        line=dict(width=0),
        showlegend=False,
        fill='tonexty',
        fillcolor='rgba(255,127,14,0.2)',
        hoverinfo='skip'
    ))

    last_hist_date = monthly_outages['ds'].max()
    fig.add_vline(x=last_hist_date, line_width=1, line_dash="dash", line_color="gray")

    historical_forecast = model.predict(monthly_outages[['ds']])
    residuals = monthly_outages['y'] - historical_forecast['yhat']
    std_res = residuals.std()
    anomalies = monthly_outages[abs(residuals) > 2 * std_res]
    if not anomalies.empty:
        fig.add_trace(go.Scatter(
            x=anomalies['ds'],
            y=anomalies['y'],
            mode='markers',
            marker=dict(color='red', size=10),
            name='Anomalies',
            hovertemplate='Date: %{x}<br>Outages: %{y}<br>Anomaly<extra></extra>'
        ))

    title = f'Monthly Outage Count{" for " + selected_state if selected_state != "all" else ""} with Forecast (Prophet)'
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Outage Count',
        legend_title='Legend',
        template='plotly_white',
        hovermode='x unified',
        margin=dict(l=50, r=50, t=50, b=50),
        font=dict(size=14),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )

    graph_html = fig.to_html(full_html=False)
    return render_template('dashboard.html', graph_html=graph_html, states=states, selected_state=selected_state)

@app.route('/state_heatmap')
def state_monthly_heatmap():
    try:
        df = load_aggregated_data()
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return render_template('state_heatmap.html', error=f"Failed to load data: {str(e)}")

    try:
        df['month'] = df['month'].replace(0, 1)
        df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))

        state_month_outage = df.pivot_table(
            index='state', columns='date', values='outage_count', aggfunc='sum', fill_value=0
        )

        plt.figure(figsize=(32, 24))
        sns.heatmap(state_month_outage, cmap='Reds', linewidths=0.9)

        all_dates = state_month_outage.columns
        visible_ticks = list(range(0, len(all_dates), 22))
        visible_labels = [all_dates[i].strftime('%Y') for i in visible_ticks]

        plt.xticks(visible_ticks, visible_labels, rotation=45, ha='right', fontsize=25)
        plt.yticks(fontsize=30)

        plt.title("State-wise Monthly Outage Count Heatmap", fontsize=30)
        plt.xlabel("Date", fontsize=30)
        plt.ylabel("State", fontsize=30)
        plt.tight_layout()

        output_path = os.path.join(app.root_path, 'static', 'images', 'state_monthly_heatmap.png')
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        if os.path.exists(output_path):
            os.remove(output_path)

        plt.savefig(output_path, dpi=500)
        plt.close()

        if not os.path.exists(output_path):
            raise FileNotFoundError("Heatmap image was not generated successfully.")

        timestamp = int(time.time())
        return render_template('state_heatmap.html', image_path='images/state_monthly_heatmap.png', timestamp=timestamp)
    except Exception as e:
        print(f"Error generating heatmap: {str(e)}")
        return render_template('state_heatmap.html', error=f"Failed to generate heatmap: {str(e)}")

@app.route('/map')
def choropleth_map():
    df = load_events_data()

    state_name_to_abbr = {
        'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA',
        'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE', 'Florida': 'FL', 'Georgia': 'GA',
        'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS',
        'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD', 'Massachusetts': 'MA',
        'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS', 'Missouri': 'MO', 'Montana': 'MT',
        'Nebraska': 'NE', 'Nevada': 'NV', 'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM',
        'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK',
        'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC',
        'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT',
        'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY'
    }

    df['state_abbr'] = df['state'].map(state_name_to_abbr)

    state_counts = df['state_abbr'].value_counts().reset_index()
    state_counts.columns = ['state', 'event_count']
    state_counts = state_counts.dropna()

    fig = px.choropleth(
        state_counts,
        locations='state',
        locationmode='USA-states',
        color='event_count',
        color_continuous_scale='Reds',
        scope='usa',
        title='U.S. Grid Outages by State (2014–2023)',
        hover_data=['event_count'],
        labels={'event_count': 'Event Count'}
    )

    fig.update_layout(
        title=dict(text='U.S. Grid Outages by State (2014–2023)', font=dict(size=20)),
        template='plotly_white',
        margin=dict(l=50, r=50, t=50, b=50),
        font=dict(size=14),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )

    graph_html = fig.to_html(full_html=False)
    return render_template('map.html', graph_html=graph_html)

@app.route('/hourly_heatmap')
def outage_hourly_heatmap():
    df = load_events_data()

    df['start_time'] = pd.to_datetime(df['start_time'])
    df['hour'] = df['start_time'].dt.hour
    df['weekday'] = df['start_time'].dt.day_name()

    heatmap_data = df.groupby(['weekday', 'hour']).size().unstack(fill_value=0)

    ordered_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    heatmap_data = heatmap_data.reindex(ordered_days)

    plt.figure(figsize=(14, 6))
    sns.heatmap(heatmap_data, cmap='YlGnBu', linewidths=0.5)
    plt.title('Outage Event Frequency by Hour and Day of Week (2014–2023)', fontsize=14)
    plt.xlabel('Hour of Day', fontsize=12)
    plt.ylabel('Day of Week', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()

    output_path = os.path.join(app.root_path, 'static', 'images', 'heatmap_hour_day.png')
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if os.path.exists(output_path):
        os.remove(output_path)

    plt.savefig(output_path, dpi=300)
    plt.close()

    timestamp = int(time.time())
    return render_template('heatmap_hourly.html', image_path='images/heatmap_hour_day.png', timestamp=timestamp)

@app.route('/network')
def network_graph():
    timestamp = int(time.time())
    return render_template('network.html', image_path='images/network_graph.png', timestamp=timestamp)

@app.route('/anomaly_detection')
def anomaly_detection():
    df = load_events_data()

    df['start_time'] = pd.to_datetime(df['start_time'])
    df['timestamp'] = df['start_time'].astype(int) / 10 ** 9
    df['event_count'] = 1

    daily_events = df.groupby(df['start_time'].dt.date)['event_count'].sum().reset_index()
    daily_events['start_time'] = pd.to_datetime(daily_events['start_time'])
    daily_events['timestamp'] = daily_events['start_time'].astype(int) / 10 ** 9

    X = daily_events[['timestamp', 'event_count']].values
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    daily_events['anomaly'] = iso_forest.fit_predict(X)

    fig = go.Figure()
    normal = daily_events[daily_events['anomaly'] == 1]
    fig.add_trace(go.Scatter(
        x=normal['start_time'],
        y=normal['event_count'],
        mode='markers',
        name='Normal Events',
        marker=dict(color='#1f77b4', size=8),
        hovertemplate='Date: %{x}<br>Events: %{y}<extra></extra>'
    ))
    anomalies = daily_events[daily_events['anomaly'] == -1]
    fig.add_trace(go.Scatter(
        x=anomalies['start_time'],
        y=anomalies['event_count'],
        mode='markers',
        name='Anomalies (Isolation Forest)',
        marker=dict(color='red', size=10, symbol='x'),
        hovertemplate='Date: %{x}<br>Events: %{y}<br>Anomaly<extra></extra>'
    ))

    fig.update_layout(
        title='Anomaly Detection in Outage Events (2014–2023)',
        xaxis_title='Date',
        yaxis_title='Daily Event Count',
        legend_title='Legend',
        template='plotly_white',
        hovermode='x unified',
        margin=dict(l=50, r=50, t=50, b=50),
        font=dict(size=14),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )

    graph_html = fig.to_html(full_html=False)
    return render_template('anomaly_detection.html', graph_html=graph_html)

@app.route('/clustering')
def clustering():
    df = load_aggregated_data()

    state_year_outages = df.groupby(['state', 'year'])['outage_count'].sum().unstack(fill_value=0)

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(state_year_outages.values)

    kmeans = KMeans(n_clusters=4, random_state=42)
    state_year_outages['cluster'] = kmeans.fit_predict(scaled_data)

    state_clusters = state_year_outages.reset_index()[['state', 'cluster']]
    state_clusters['state_abbr'] = state_clusters['state'].map({
        'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA',
        'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE', 'Florida': 'FL', 'Georgia': 'GA',
        'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS',
        'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD', 'Massachusetts': 'MA',
        'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS', 'Missouri': 'MO', 'Montana': 'MT',
        'Nebraska': 'NE', 'Nevada': 'NV', 'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM',
        'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK',
        'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC',
        'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT',
        'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY'
    })

    fig = px.choropleth(
        state_clusters,
        locations='state_abbr',
        locationmode='USA-states',
        color='cluster',
        scope='usa',
        title='Clustering of States by Outage Patterns (2014–2023) (K-Means)',
        color_discrete_sequence=px.colors.qualitative.Set2,
        hover_data=['state', 'cluster'],
        labels={'cluster': 'Cluster'}
    )

    fig.update_layout(
        title=dict(text='Clustering of States by Outage Patterns (2014–2023) (K-Means)', font=dict(size=20)),
        template='plotly_white',
        margin=dict(l=50, r=50, t=50, b=50),
        font=dict(size=14),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )

    graph_html = fig.to_html(full_html=False)
    return render_template('clustering.html', graph_html=graph_html)

state_county_map = joblib.load("state_county_mapping.joblib")

def get_user_input(State, EventType, county, mean_customers, month, weekday):
    state_encoder = joblib.load('state_encoder.joblib')
    event_encoder = joblib.load('event_encoder.joblib')
    county_fips_dict = joblib.load("county_fips_mapping.joblib")
    Weekday_dict = {
        'Sunday': 0, 'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6
    }
    month_dict = {
        'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
        'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
    }

    # Create a DataFrame with a single row for prediction
    encoded_df = pd.DataFrame({
        'state': [State],
        'event_type': [EventType],
        'county': [county],
        'mean_customers': [mean_customers],
        'month': [month],
        'weekday': [weekday],
        'start_time': [pd.Timestamp('2025-01-01')]  # Placeholder; only needed for preprocessing
    })

    # Preprocess the data for prediction
    encoded_df, _, _, _ = preprocess_data(
        encoded_df,
        state_encoder=state_encoder,
        event_encoder=event_encoder,
        scaler=scaler,
        county_fips_dict=county_fips_dict,
        for_training=False
    )

    return encoded_df[['state_encoded', 'event_type_encoded', 'fips', 'mean_customers', 'month', 'weekday']]
@app.route('/get_counties', methods=['POST'])
def get_counties():
    state = request.json.get('state')
    counties = list(set(state_county_map.get(state, [])))
    return jsonify(counties)

def make_predict(df):
    result_dict = joblib.load("model_results_best.joblib")
    model = result_dict['best_model']
    best_model_name = result_dict['best_model_name']
    print(f"Model used: {best_model_name}")
    print(f"Model expected features: {model.feature_names_in_ if hasattr(model, 'feature_names_in_') else 'Ensemble model'}")
    print(f"Input DataFrame columns: {df.columns}")

    # Make prediction
    pred = model.predict(df)[0]
    pred_map = {
        0: "0 to 4 hours",
        1: "4 to 24 hours",
        2: "24 to 72 hours",
        3: "3 to 7 days",
        4: "Longer than 7 days"
    }
    predicted_category = pred_map[pred]

    # Get prediction probabilities
    pred_proba = model.predict_proba(df)[0]
    proba_dict = {pred_map[i]: round(float(prob) * 100, 2) for i, prob in enumerate(pred_proba)}

    # Compute feature importance using RandomForest's built-in feature_importances_
    if isinstance(model, VotingClassifier):
        # If ensemble, use the RandomForest estimator
        rfc = next((est for name, est in model.estimators_ if name == 'rfc'), None)
        if rfc is None:
            rfc = model.estimators_[0]  # Fallback to first estimator
    else:
        rfc = model

    feature_names = ['State', 'Event Type', 'FIPS (County)', 'Customers Affected', 'Month', 'Weekday']
    importances = rfc.feature_importances_
    importance_dict = {name: round(float(value), 4) for name, value in zip(feature_names, importances)}

    # Note: Feature importance reflects the training data's overall trends and may not vary with specific input (e.g., state or county).
    print("Note: Feature importance is based on the training dataset's global feature distribution.")

    # Create a bar chart for probabilities
    proba_fig = go.Figure(data=[
        go.Bar(
            x=list(proba_dict.values()),
            y=list(proba_dict.keys()),
            orientation='h',
            marker=dict(color='rgba(50, 171, 96, 0.6)'),
            text=[f"{val}%" for val in proba_dict.values()],
            textposition='auto'
        )
    ])
    proba_fig.update_layout(
        title="Prediction Probabilities",
        xaxis_title="Probability (%)",
        yaxis_title="Outage Duration Category",
        template='plotly_white',
        margin=dict(l=50, r=50, t=50, b=50),
        font=dict(size=14)
    )
    proba_chart = proba_fig.to_html(full_html=False)

    # Create a bar chart for feature importances
    importance_fig = go.Figure(data=[
        go.Bar(
            x=list(importance_dict.values()),
            y=list(importance_dict.keys()),
            orientation='h',
            marker=dict(color='rgba(255, 99, 71, 0.6)'),
            text=[f"{val:.4f}" for val in importance_dict.values()],
            textposition='auto'
        )
    ])
    importance_fig.update_layout(
        title="Feature Importance (Based on Training Data)",
        xaxis_title="Importance",
        yaxis_title="Feature",
        template='plotly_white',
        margin=dict(l=50, r=50, t=50, b=50),
        font=dict(size=14)
    )
    importance_chart = importance_fig.to_html(full_html=False)

    return predicted_category, proba_dict, proba_chart, importance_dict, importance_chart, best_model_name

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    states = joblib.load("state_list.joblib")
    event_types = joblib.load("event_type_list.joblib")
    months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
              'November', 'December']
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    models = ['Best Model']  # Using the best model selected during training

    prediction = None
    selected_state = selected_county = selected_event = selected_month = selected_weekday = customers = selected_model = None
    proba_dict = proba_chart = importance_dict = importance_chart = best_model_name = None

    if request.method == 'POST':
        try:
            selected_state = request.form.get('state')
            selected_county = request.form.get('county')
            selected_event_type = request.form.get('event_type')
            selected_month = request.form.get('month')
            selected_weekday = request.form.get('weekday')
            customers = request.form.get('customers')
            selected_model = request.form.get('model')

            # Validate required fields
            required_fields = {
                'state': selected_state,
                'county': selected_county,
                'event_type': selected_event_type,
                'month': selected_month,
                'weekday': selected_weekday,
                'customers': customers,
                'model': selected_model
            }
            missing_fields = [field for field, value in required_fields.items() if not value]
            if missing_fields:
                return render_template(
                    'predict.html',
                    states=states,
                    event_types=event_types,
                    months=months,
                    weekdays=weekdays,
                    models=models,
                    error=f"Missing required fields: {', '.join(missing_fields)}.",
                    selected_state=selected_state,
                    selected_county=selected_county,
                    selected_event=selected_event_type,
                    selected_month=selected_month,
                    selected_weekday=selected_weekday,
                    customers=customers,
                    selected_model=selected_model,
                    best_model_name=best_model_name,
                    note="Feature importance reflects training data trends and does not vary with input."
                )

            # Validate customers
            try:
                customers = int(customers)
                if customers < 0:
                    raise ValueError("Number of customers must be a non-negative integer.")
            except ValueError:
                return render_template(
                    'predict.html',
                    states=states,
                    event_types=event_types,
                    months=months,
                    weekdays=weekdays,
                    models=models,
                    error="Please enter a valid non-negative integer for the number of customers affected (e.g., 0, 100, 1000).",
                    selected_state=selected_state,
                    selected_county=selected_county,
                    selected_event=selected_event_type,
                    selected_month=selected_month,
                    selected_weekday=selected_weekday,
                    customers=customers,
                    selected_model=selected_model,
                    best_model_name=best_model_name,
                    note="Feature importance reflects training data trends and does not vary with input."
                )

            user_df = get_user_input(selected_state, selected_event_type, selected_county, customers, selected_month, selected_weekday)
            prediction, proba_dict, proba_chart, importance_dict, importance_chart, best_model_name = make_predict(user_df)
            return render_template(
                'predict.html',
                states=states,
                event_types=event_types,
                months=months,
                weekdays=weekdays,
                models=models,
                prediction=prediction,
                proba_dict=proba_dict,
                proba_chart=proba_chart,
                importance_dict=importance_dict,
                importance_chart=importance_chart,
                best_model_name=best_model_name,
                selected_state=selected_state,
                selected_county=selected_county,
                selected_event=selected_event_type,
                selected_month=selected_month,
                selected_weekday=selected_weekday,
                customers=customers,
                selected_model=selected_model,
                note="Feature importance reflects training data trends and does not vary with input."
            )

        except Exception as e:
            return render_template(
                'predict.html',
                states=states,
                event_types=event_types,
                months=months,
                weekdays=weekdays,
                models=models,
                error=f"An error occurred: {str(e)}",
                selected_state=selected_state,
                selected_county=selected_county,
                selected_event=selected_event_type,
                selected_month=selected_month,
                selected_weekday=selected_weekday,
                customers=customers,
                selected_model=selected_model,
                best_model_name=best_model_name,
                note="Feature importance reflects training data trends and does not vary with input."
            )

    return render_template(
        'predict.html',
        states=states,
        event_types=event_types,
        months=months,
        weekdays=weekdays,
        models=models,
        best_model_name=best_model_name,
        note="Feature importance reflects training data trends and does not vary with input."
    )


@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    # Get the machine's IP address
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))  # Connect to a public DNS server (Google's DNS)
        ip_address = s.getsockname()[0]
        s.close()
    except Exception as e:
        print(f"Error getting IP address: {str(e)}")
        ip_address = "127.0.0.1"  # Fallback to localhost if IP retrieval fails

    port = 5000
    print(f"\nStarting Flask app on host 0.0.0.0, port {port}...")
    app.run(host='0.0.0.0', port=port, debug=True)