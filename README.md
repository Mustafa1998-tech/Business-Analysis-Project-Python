# Business Analysis Project with Python

A comprehensive business analysis project that demonstrates data generation, cleaning, visualization, and customer segmentation using Python.

## Project Structure

```
business_analysis_project/
├── README.md
├── requirements.txt
├── data_generator.py
├── data/
│   ├── sample_sales.csv
│   ├── customers.csv
│   └── rfm_segments.csv
├── notebooks/
│   └── analysis.py
├── reports/
│   └── insights.md
└── assets/plots/
    ├── sales_by_category.png
    └── total_sales_by_month.png
```

## Setup Instructions

1. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # On Windows
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Generate sample data**:
   ```bash
   python data_generator.py
   ```

4. **Run the analysis**:
   ```bash
   python notebooks/analysis.py
   ```

## Project Components

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
