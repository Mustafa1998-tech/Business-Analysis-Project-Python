# Business Analysis Project with Python

A comprehensive business analysis project that demonstrates data generation, cleaning, visualization, and customer segmentation using Python.

## 📋 Project Structure

```
business_analysis_project/
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── app.py                   # Main application
├── data_generator.py        # Script to generate sample data
├── data/                    # Data directory
│   ├── sample_sales.csv     # Sample sales data
│   ├── sample_sales_100.csv # Additional sample data
│   └── customers.csv        # Customer information
├── notebooks/               # Jupyter notebooks for analysis
│   └── analysis.py         # Data analysis scripts
├── reports/                 # Generated reports
│   └── insights.md         # Analysis insights
└── assets/plots/            # Visualization outputs
    ├── sales_by_category.png
    └── total_sales_by_month.png
```

## 🚀 Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Mustafa1998-tech/Business-Analysis-Project-Python.git
   cd Business-Analysis-Project-Python
   ```

2. **Create and activate virtual environment** (recommended):
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate
   
   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Generate sample data**:
   ```bash
   python data_generator.py
   ```

5. **Run the analysis**:
   ```bash
   python notebooks/analysis.py
   ```

6. **Start the application**:
   ```bash
   python app.py
   ```

## 📊 Project Components

1. **Data Generation**
   - Generates realistic sales data with customers, products, and transactions
   - Includes regional distribution and loyalty programs

2. **Analysis**
   - Time series analysis of sales
   - Product category performance
   - RFM (Recency, Frequency, Monetary) analysis
   - Customer segmentation using K-means clustering

3. **Outputs**
   - Visualizations in `assets/plots/`
   - Customer segments in `data/rfm_segments.csv`
   - Business insights in `reports/insights.md`

## Dependencies

- Python 3.7+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- plotly (optional for interactive visualizations)
- jupyter (optional for notebook interface)

## Customization

1. Edit `data_generator.py` to modify data generation parameters
2. Update the analysis in `notebooks/analysis.py`
3. Add more visualizations or analysis as needed

## License

This project is open source and available under the MIT License.
