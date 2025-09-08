# ⚡ Grid Disruption Analysis

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📊 Overview

Grid Disruption Analysis is a comprehensive Flask-based web application that visualizes and analyzes U.S. power grid outage data spanning from 2014 to 2023. This project combines interactive data visualization with advanced machine learning techniques to provide actionable insights for researchers, utility companies, and policymakers.

### 🎯 Target Audience
- Energy researchers and analysts
- Utility companies and grid operators
- Policy makers and government agencies
- Data science enthusiasts

## ✨ Key Features

### 📈 Interactive Visualizations
- **State-wise Monthly Heatmap**: Track outage patterns across states and time
- **Hourly Outage Analysis**: Identify peak disruption times
- **Choropleth Maps**: Geographic visualization of outage distribution
- **Network Graphs**: Visualize connections between outage events

### 🤖 Machine Learning Models
- **Forecasting**: Prophet-based time series prediction for outage trends
- **Anomaly Detection**: Isolation Forest algorithm to identify unusual events
- **Clustering Analysis**: K-Means clustering to group states by outage patterns
- **Duration Prediction**: Predictive models for outage duration estimation

### 💡 Key Insights
- Seasonal and temporal outage patterns
- Regional vulnerability analysis
- Peak disruption time identification
- Grid resilience enhancement recommendations

## 🛠️ Tech Stack

- **Backend**: Flask (Python web framework)
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Machine Learning**: Scikit-learn, Prophet
- **Network Analysis**: NetworkX
- **Frontend**: HTML, CSS, Bootstrap

## 🚀 Quick Start

### Prerequisites

```bash
# Check Python version (3.8+ required)
python --version

# Ensure pip is installed
pip --version
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/kraryal/Grid_Disruption_Analysis.git
cd Grid_Disruption_Analysis
```

2. **Set up virtual environment** (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

3. **Install dependencies**
```bash
# Install from requirements.txt
pip install -r requirements.txt

# OR install manually if requirements.txt is missing
pip install flask pandas prophet plotly seaborn matplotlib networkx scikit-learn numpy
```

4. **Verify installation**
```bash
python -c "import flask, pandas, prophet, plotly, seaborn, matplotlib, networkx, sklearn; print('✅ All dependencies installed successfully!')"
```

## 🔧 Post-Installation Setup

### 1. **Directory Structure Verification**
```bash
# Ensure you're in the correct directory
ls -la
# You should see: CODE/ folder, README.md, etc.

# Navigate to CODE directory
cd CODE

# Verify Flask app exists
ls -la app.py
```

### 2. **Data Setup** (if required)
```bash
# If data folder doesn't exist, create it
mkdir -p data

# Verify data files are present
ls -la data/
# Should contain your grid outage datasets
```

### 3. **Configuration Check**
```python
# Optional: Create a config check script (config_check.py)
import os
import sys

def check_setup():
    """Verify the application setup"""
    
    # Check if we're in the right directory
    if not os.path.exists('app.py'):
        print("❌ app.py not found. Make sure you're in the CODE directory.")
        return False
    
    # Check data directory
    if not os.path.exists('data'):
        print("⚠️  Data directory not found. Creating...")
        os.makedirs('data', exist_ok=True)
    
    # Check templates directory
    if not os.path.exists('templates'):
        print("⚠️  Templates directory not found.")
        return False
    
    # Check static directory
    if not os.path.exists('static'):
        print("⚠️  Static directory not found.")
        return False
    
    print("✅ Setup verification complete!")
    return True

if __name__ == "__main__":
    check_setup()
```

### 4. **Environment Variables** (Optional)
```bash
# Create .env file for configuration (optional)
cat > .env << EOF
FLASK_APP=app.py
FLASK_ENV=development
FLASK_DEBUG=1
PORT=5000
EOF

# Load environment variables
# On Windows:
set FLASK_APP=app.py
set FLASK_ENV=development

# On macOS/Linux:
export FLASK_APP=app.py
export FLASK_ENV=development
```

## 🏃‍♂️ Running the Application

### Method 1: Direct Python Execution
```bash
# Navigate to CODE directory
cd CODE

# Run the Flask application
python app.py

# Expected output:
# * Running on http://127.0.0.1:5000
# * Debug mode: on
# * Restarting with stat
# * Debugger is active!
```


## 🖥️ Application Usage

### Accessing the Application
```bash
# Local access
http://127.0.0.1:5000
# or
http://localhost:5000

# Network access (if running with --host=0.0.0.0)
http://YOUR_IP_ADDRESS:5000
```

### API Endpoints (if available)
```python
# Example API calls using requests library
import requests

# Get monthly trends for a specific state
response = requests.get('http://127.0.0.1:5000/api/monthly_trend?state=California')
data = response.json()

# Predict outage duration
payload = {
    'duration': 120,
    'customers_affected': 5000,
    'demand_loss': 250
}
response = requests.post('http://127.0.0.1:5000/api/predict', json=payload)
prediction = response.json()
```

### Sample Usage Code
```python
# Example: Programmatic access to core functions
from app import app

# Create application context
with app.app_context():
    # Example: Get data for specific state
    def get_state_data(state_name):
        # Your data processing logic here
        pass
    
    # Example: Run prediction model
    def predict_outage_duration(features):
        # Your ML prediction logic here
        pass
```

## 🔧 Development Setup

### Setting up for Development
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Or install individual dev packages
pip install pytest flask-testing black flake8 isort

# Run tests (if available)
python -m pytest tests/

# Format code with Black
black app.py

# Check code style
flake8 app.py

# Sort imports
isort app.py
```

### Creating Custom Configurations
```python
# config.py - Custom configuration file
import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key'
    DEBUG = True
    TESTING = False
    
class ProductionConfig(Config):
    DEBUG = False
    SECRET_KEY = os.environ.get('SECRET_KEY')
    
class DevelopmentConfig(Config):
    DEBUG = True
    
class TestingConfig(Config):
    TESTING = True
    DEBUG = True

# Usage in app.py
# app.config.from_object('config.DevelopmentConfig')
```

## 🐛 Troubleshooting

### Common Issues and Solutions

1. **Port Already in Use**
```bash
# Find process using port 5000
lsof -i :5000  # Mac/Linux
netstat -ano | findstr :5000  # Windows

# Kill the process or use different port
python app.py --port 8000
```

2. **Module Not Found Errors**
```bash
# Reinstall dependencies
pip install --force-reinstall -r requirements.txt

# Check Python path
python -c "import sys; print('\n'.join(sys.path))"

# Ensure virtual environment is activated
which python  # Should show venv path
```

3. **Data Loading Issues**
```python
# Debug data loading
import pandas as pd
import os

print("Current directory:", os.getcwd())
print("Files in data/:", os.listdir('data/') if os.path.exists('data/') else "Data directory not found")
```

4. **Template Not Found**
```bash
# Verify template structure
ls -la templates/
# Should contain: index.html, about.html, etc.
```

### Debug Mode
```python
# Add debug prints to app.py
if __name__ == '__main__':
    print("🚀 Starting Grid Disruption Analysis App...")
    print("📍 Current directory:", os.getcwd())
    print("📊 Checking data availability...")
    
    app.run(debug=True, host='127.0.0.1', port=5000)
```

## 📊 Data Management

### Data File Structure
```
data/
├── grid_outages_2014_2023.csv    # Main dataset
├── state_coordinates.json        # Geographic data
├── processed/                    # Processed datasets
│   ├── monthly_aggregated.csv
│   ├── hourly_patterns.csv
│   └── anomaly_scores.csv
└── models/                       # Saved ML models
    ├── prophet_model.pkl
    ├── isolation_forest.pkl
    └── kmeans_clusters.pkl
```

### Data Processing Examples
```python
# Example data processing script
import pandas as pd
from datetime import datetime

def load_and_preprocess_data():
    """Load and preprocess grid outage data"""
    
    # Load main dataset
    df = pd.read_csv('data/grid_outages_2014_2023.csv')
    
    # Convert date columns
    df['date'] = pd.to_datetime(df['date'])
    
    # Create additional features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['hour'] = df['date'].dt.hour
    df['day_of_week'] = df['date'].dt.dayofweek
    
    return df

# Usage
if __name__ == "__main__":
    data = load_and_preprocess_data()
    print(f"✅ Loaded {len(data)} records")
```

## 📁 Complete Project Structure

```
Grid_Disruption_Analysis/
├── CODE/
│   ├── app.py                     # Main Flask application
│   ├── requirements.txt           # Python dependencies
│   ├── config.py                  # Configuration settings
│   ├── models/                    # ML models directory
│   │   ├── __init__.py
│   │   ├── forecasting.py         # Prophet models
│   │   ├── anomaly_detection.py   # Isolation Forest
│   │   └── clustering.py          # K-Means clustering
│   ├── templates/                 # HTML templates
│   │   ├── base.html              # Base template
│   │   ├── index.html             # Homepage
│   │   ├── monthly_trend.html     # Monthly analysis
│   │   ├── predict.html           # Prediction interface
│   │   ├── heatmap.html           # State heatmap
│   │   ├── choropleth.html        # Geographic map
│   │   ├── hourly_heatmap.html    # Hourly patterns
│   │   ├── network_graph.html     # Network visualization
│   │   ├── anomaly.html           # Anomaly detection
│   │   ├── clustering.html        # State clustering
│   │   └── about.html             # About page
│   ├── static/                    # Static files
│   │   ├── css/
│   │   │   └── style.css          # Custom styles
│   │   ├── js/
│   │   │   └── app.js             # JavaScript functionality
│   │   └── images/                # Images and icons
│   ├── data/                      # Dataset files
│   │   ├── grid_outages_2014_2023.csv
│   │   └── processed/             # Processed data
│   └── utils/                     # Utility functions
│       ├── __init__.py
│       ├── data_processing.py     # Data manipulation
│       └── visualization.py       # Chart generation
├── tests/                         # Test files
│   ├── __init__.py
│   ├── test_app.py               # App tests
│   └── test_models.py            # Model tests
├── docs/                         # Documentation
│   ├── api.md                    # API documentation
│   └── deployment.md             # Deployment guide
├── .env.example                  # Environment variables template
├── .gitignore                    # Git ignore rules
├── requirements.txt              # Production dependencies
├── requirements-dev.txt          # Development dependencies
├── Dockerfile                    # Docker configuration
├── docker-compose.yml           # Docker Compose setup
├── README.md                     # This file
└── LICENSE                       # License information
```

## 🚀 Advanced Usage

### Custom Model Training
```python
# train_models.py - Script to retrain models with new data
from models.forecasting import ProphetForecaster
from models.anomaly_detection import AnomalyDetector
from utils.data_processing import load_and_preprocess_data

def retrain_models():
    """Retrain all ML models with latest data"""
    
    # Load latest data
    data = load_and_preprocess_data()
    
    # Retrain forecasting model
    forecaster = ProphetForecaster()
    forecaster.fit(data)
    forecaster.save('models/prophet_model_updated.pkl')
    
    # Retrain anomaly detection
    detector = AnomalyDetector()
    detector.fit(data)
    detector.save('models/isolation_forest_updated.pkl')
    
    print("✅ All models retrained successfully!")

if __name__ == "__main__":
    retrain_models()
```

### Batch Processing
```python
# batch_analysis.py - Process multiple states or time periods
def batch_analyze_states(states_list):
    """Analyze multiple states in batch"""
    results = {}
    
    for state in states_list:
        print(f"🔄 Processing {state}...")
        # Your analysis logic here
        results[state] = analyze_state(state)
    
    return results

# Usage
states = ['California', 'Texas', 'New York', 'Florida']
batch_results = batch_analyze_states(states)
```

---

## 👥 Team

**Team 155** - Data Analysis Initiative

- **Krishna Aryal** - [@kraryal](https://github.com/kraryal) - Project Lead & Backend Development
- **Crystal Vandekerkhove** - Data Analysis & Visualization
- **Jinesh Patel** - Machine Learning & Modeling

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🔮 Roadmap

- [ ] Real-time data integration with utility APIs
- [ ] Docker containerization for easy deployment
- [ ] RESTful API endpoints for external integration
- [ ] Mobile-responsive UI improvements
- [ ] Advanced ML model comparison dashboard
- [ ] Automated testing and CI/CD pipeline
- [ ] Performance optimization and caching
- [ ] Multi-language support

## 📞 Support

Need help? Here's how to get support:

- 📖 Check the [Documentation](/DOC))
- 🐛 Report bugs via [GitHub Issues](https://github.com/kraryal/Grid_Disruption_Analysis/issues)
- 💬 Join discussions in [GitHub Discussions](https://github.com/kraryal/Grid_Disruption_Analysis/discussions)
- 📧 Contact: [aryalkris9@gmail.com]

---

⭐ **Star this repository** if you find it helpful!

**Made with ❤️ by Team 155**
```
================================================================================
