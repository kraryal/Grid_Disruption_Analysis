================================================================================
Grid Disruption Analysis - README
================================================================================

DESCRIPTION
--------------------------------------------------------------------------------
Grid Disruption Analysis is a Flask-based web application designed to visualize 
and analyze U.S. grid outage data from 2014 to 2023. Developed by Team 155
(Krishna Aryal, Crystal Vandekerkhove, Jinesh Patel) for a data analysis
initiative, it targets researchers and utility companies with machine learning
models to predict outage durations, detect anomalies, and uncover patterns. Key
features include:
- Interactive visualizations: state-wise monthly outage heatmap, hourly outage
  heatmap, choropleth map, and network graph.
- Forecasting with Prophet, anomaly detection with Isolation Forest, and
  clustering with K-Means.
- Insights into outage trends, peak times, and regional patterns to aid
  mitigation and enhance grid resilience.
The app uses Flask as the web framework, with Python libraries like Pandas,
Matplotlib, Seaborn, Plotly, and Scikit-learn for data processing and
visualization. It features a user-friendly homepage with a light blue aesthetic
for visualization links and an About page with a similar design for machine
learning model details, ensuring a consistent and appealing experience.

INSTALLATION
--------------------------------------------------------------------------------
1. **Prerequisites**:
   - Install Python 3.8 or higher.
   - Ensure `pip` is available for dependency installation.

2. **Extract the CODE Folder**:
   - Copy the `CODE` folder to your desired directory on your local machine.

3. **Set Up a Virtual Environment (Recommended)**:
   - Create a virtual environment: `python -m venv venv`
   - Activate it:
     - On Linux/Mac: `source venv/bin/activate`
     - On Windows: `venv\Scripts\activate`

4. **Install Dependencies**:
   - Install required Python libraries using the provided `requirements.txt`:
     `pip install -r requirements.txt`
   - If `requirements.txt` is missing, install manually:
     `pip install flask pandas prophet plotly seaborn matplotlib networkx scikit-learn`

5. **Verify Installation**:
   - Run the following to check dependencies:
     `python -c "import flask, pandas, prophet, plotly, seaborn, matplotlib, networkx, sklearn"`
   - If no errors occur, the setup is complete.

EXECUTION
--------------------------------------------------------------------------------
1. **Navigate to the Project Directory**:
   - Change to the directory containing `app.py` (inside the `CODE` folder):
     `cd <path-to-CODE-folder>`

2. **Run the Flask App**:
   - Start the server: `python app.py`
   - The app will run on `http://127.0.0.1:5000`. Terminal output will confirm the server is active.
   - To stop the server, press `Ctrl+C`.

3. **Access the App**:
   - Open a web browser and go to `http://127.0.0.1:5000`.
   - The homepage offers links to:
     - **Monthly Trend**: View interactive monthly outage trends with forecasting.
     - **Predict**: Input data to predict outage durations.
     - **State-wise Outage Heatmap**: Explore outage counts by state and month.
     - **Choropleth Map**: See outages by state on a map.
     - **Hourly Outage Heatmap**: Analyze outage frequency by hour and day.
     - **Network Graph**: Visualize connections between outage events.
     - **Anomaly Detection**: Identify unusual outage events.
     - **Clustering of States**: Group states by outage patterns.
     - **About Project**: Learn about the project and team.

4. **Interact with the App**:
   - On the Monthly Trend page, use the dropdown to select a state.
   - Navigate other pages via links to explore static visualizations.

================================================================================
