# ⚡ Optivolt

## AI-Powered EV Depot Demand Forecasting and Grid Capacity Planning

**An AI-driven decision-support application for forecasting electric vehicle depot electricity demand, identifying grid-capacity risks and supporting infrastructure planning.**

Optivolt combines Long Short-Term Memory (LSTM) neural networks, historical electricity demand analysis and interactive scenario planning in a Streamlit application.

**Live Application:** [Optivolt EV Depot Intelligence Portal](https://optivolt-b7gtuketqytpwukkwonf7f.streamlit.app/)

---

## 1. Project Overview

The transition to electric vehicles (EVs) creates new challenges for depot operators and electricity network planners. Charging demand can vary significantly throughout the day, creating uncertainty about future electricity requirements and the risk of exceeding available grid capacity.

Optivolt was developed to investigate how artificial intelligence can improve electricity demand forecasting and support evidence-based operational planning.

The project combines an experimental machine-learning workflow with a web-based decision-support application.

Its principal objectives are to:

- Forecast short-term and longer-term depot electricity demand.
- Evaluate LSTM against alternative forecasting approaches.
- Identify periods of potential grid-capacity stress.
- Assess the impact of fleet growth and changing charging conditions.
- Support planning decisions through interactive visualisations, alerts and downloadable reports.

## 2. Key Application Features

### Demand Prediction

The application uses saved LSTM models to generate electricity demand forecasts for supported depots.

Users can select a depot, configure the forecast period and assess predicted demand against grid-capacity limits.

Forecast information includes predicted peak demand, forecast energy consumption, available grid headroom, peak-demand timing and potential capacity breaches.

### Fleet Growth & Capacity Planning

Users can examine how operational changes may affect electricity demand.

Planning conditions include fleet and charging-demand growth, additional fixed site loads, charging time shifts, flexible charging, temporary demand surges and peak-demand management.

The application compares the saved LSTM baseline with an adjusted planning scenario, providing an indication of the potential impact on grid-capacity utilisation.

### System Alerts

System Alerts evaluates the saved LSTM forecast against the capacity threshold associated with that forecast.

It identifies potential capacity warnings and breaches, helping users understand when operational attention may be required.

### Model Performance

The application provides model evaluation information to support interpretation of forecasting reliability, prediction errors and operational suitability.

### Reports and Historical Records

Users can review operational forecast information and export reports in CSV and PDF formats. The application also provides historical records and administrative settings.

## 3. Data Sources

The project uses electricity demand datasets associated with the UK Power Networks Optimise Prime project.

Historical demand records are available for nine depot locations:

1. Bexleyheath
2. Camden
3. Dartford
4. Islington
5. Mount Pleasant
6. Orpington
7. Premier Park
8. Victoria
9. Whitechapel

The datasets contain time-series electricity demand values at **15-minute intervals**.

### Data Preparation

The experimental workflow includes:

- Data loading and quality assessment.
- Timestamp parsing and chronological ordering.
- Missing-value and data-quality checks.
- Time-series feature engineering.
- Chronological training, validation and testing.
- Synthetic training-data generation for designated experiments.
- Model fitting and held-out evaluation.

Synthetic samples are generated using historical training data, with validation and test periods kept separate.

## 4. Machine-Learning Methodology

The project evaluates four short-term forecasting approaches:

| Approach              | Description                                                      |
| --------------------- | ---------------------------------------------------------------- |
| Previous-day baseline | Uses demand from the corresponding time on the previous day      |
| Random Forest         | Ensemble-based machine-learning regression                       |
| XGBoost               | Gradient-boosted decision-tree regression                        |
| LSTM                  | Recurrent neural network designed to learn temporal dependencies |

The short-term experimental comparison uses the Bexleyheath dataset.

LSTM is the principal operational forecasting approach in the Streamlit application, using saved depot-specific models where available.

The experiment also examines recursive 24-hour forecasting and 30-day LSTM demand prediction across nine depots.

### Model Evaluation

Forecasting performance is assessed through nine criteria:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Coefficient of Determination (R²)
- F1-score
- Balanced Accuracy
- Capacity-Breach Recall
- False Alarm Ratio
- Peak-Demand MAE
- Model Training Time

Additional analysis includes residual distributions, feature importance, confusion matrices, historical forecasting accuracy and comparisons between actual and predicted electricity demand.

The experimental results are generated from held-out predictions rather than pre-entered performance values.

## 5. Technology Stack

The project uses:

| Technology                | Purpose                                        |
| ------------------------- | ---------------------------------------------- |
| Python                    | Data processing and application development    |
| Streamlit                 | Interactive web application                    |
| Pandas and NumPy          | Data manipulation and numerical analysis       |
| Keras / TensorFlow        | Operational LSTM model loading and forecasting |
| Keras / PyTorch backend   | Reproducible LSTM experiment                   |
| Scikit-learn              | Random Forest, evaluation and preprocessing    |
| XGBoost                   | Gradient-boosted forecasting comparison        |
| Matplotlib and Plotly     | Data visualisation                             |
| GitHub                    | Source-code management                         |
| Streamlit Community Cloud | Web application deployment                     |

## 6. Repository Structure

```
Optivolt/
├── app.py
├── optivolt_lstm.py
├── optivolt_experiment.py
├── requirements.txt
├── experiment_requirements.txt
├── README.md
├── EXPERIMENT_README.md
├── 30days(1).zip
├── dashboard/
├── data/
├── models/
├── synthetic_training_data/
└── UKPN_OptimisePrime_*.csv
```

Additional Python modules support forecasting, data processing, model loading, evaluation, reporting and application functionality.

## 7. Running the Application Locally

### Requirements

Python and the project dependencies are required.

Install the application dependencies:

```
pip install -r requirements.txt
```

Launch Streamlit:

```
streamlit run app.py
```

Streamlit will provide a local URL for accessing the application.

### Running the Forecasting Experiment

Install the separate experimental dependencies:

```
pip install -r experiment_requirements.txt
```

Run the experiment:

```
python optivolt_experiment.py --archive "30days(1).zip" --output results --epochs-short 25 --epochs-long 20 --accuracy-tolerance-kw 1.0
```

The experiment creates numerical result tables and plots in the selected output directory.

For detailed experimental methodology and interpretation guidance, see `EXPERIMENT_README.md`.

## 8. Business Value

Optivolt demonstrates how AI-based electricity demand forecasting can contribute to better EV depot planning.

Potential business benefits include:

- Earlier identification of forecast capacity risks.
- Improved understanding of future charging demand.
- Better-informed fleet electrification decisions.
- Assessment of alternative charging-management strategies.
- Support for grid infrastructure investment planning.
- Reduced reliance on purely reactive capacity management.

These benefits represent potential applications of the system; operational or financial savings would require validation in a real deployment.

## 9. Limitations and Future Development

Forecast accuracy depends on the quality and representativeness of historical electricity demand data.

Synthetic training data can support experimental development but does not replace independently collected observations.

The deployed application uses saved forecasting models and is intended as a decision-support prototype rather than an automated grid-control system.

Future improvements could include:

- Integration with live depot charging and metering data.
- Continuous model monitoring and retraining.
- Additional depot-specific LSTM models.
- Forecast uncertainty and confidence-interval analysis.
- Integration with charging schedules and electricity tariffs.
- Wider operational validation across depots.

Grid-capacity thresholds used within the application are configurable planning values and should not be assumed to represent verified contractual connection limits.

## 10. Project Deliverables

The project provides:

- A Streamlit-based AI forecasting and capacity-planning application.
- A reproducible Python machine-learning experiment.
- Historical electricity demand datasets for nine depots.
- Model evaluation tables and visualisations.
- Capacity-risk assessment and scenario-planning functionality.
- Technical documentation supporting interpretation and reproduction of the experimental work.

## 11. Author and Project Context

**Project:** AI-Powered EV Depot Demand Forecasting and Grid Capacity Planning

**Application:** Optivolt EV Depot Intelligence Portal

**Project Area:** Artificial Intelligence, Machine Learning, Time-Series Forecasting, Electric Vehicle Infrastructure and Energy Demand Management

**Academic Context:** Level 7 AI Data Specialist Work-Based Project

---

*Optivolt is a forecasting and decision-support prototype. Its predictions should be interpreted alongside operational knowledge, verified network constraints and appropriate human oversight.*

