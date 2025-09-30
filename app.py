import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set page config with a nice theme and favicon
st.set_page_config(
    page_title="📊 Business Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
def load_css():
    st.markdown("""
    <style>
        /* Main app styling */
        .main {
            background-color: #f8f9fa;
        }
        
        /* Sidebar styling */
        .css-1d391kg {
            background-color: #2c3e50;
            color: white;
        }
        
        /* Card styling */
        .card {
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }
        
        /* Title styling */
        .title {
            color: #2c3e50;
            margin-bottom: 20px;
            font-weight: 700;
        }
        
        /* Button styling */
        .stButton>button {
            background-color: #3498db;
            color: white;
            border-radius: 5px;
            padding: 0.5rem 1rem;
            border: none;
            font-weight: 500;
        }
        
        .stButton>button:hover {
            background-color: #2980b9;
        }
        
        /* Metric cards */
        .metric-card {
            background: linear-gradient(135deg, #3498db, #2c3e50);
            color: white;
            border-radius: 10px;
            padding: 15px;
            margin: 5px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        .metric-value {
            font-size: 24px;
            font-weight: 700;
            margin: 5px 0;
        }
        
        .metric-label {
            font-size: 14px;
            opacity: 0.9;
        }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

# Set page config
# Load CSS
load_css()

# Sidebar for file upload and settings
with st.sidebar:
    st.title('⚙️ الإعدادات')
    st.markdown("---")
    st.subheader('تحميل البيانات')
    uploaded_file = st.file_uploader("اختر ملف CSV", type=["csv"], help="قم بتحميل ملف CSV يحتوي على بيانات المبيعات")
    
    st.markdown("---")
    st.subheader('إعدادات التجميع')
    n_clusters = st.slider("عدد المجموعات", 2, 6, 4, help="اختر عدد المجموعات المراد تقسيم العملاء إليها")
    
    st.markdown("---")
    st.info("""
    ### تعليمات الاستخدام:
    1. قم بتحميل ملف CSV يحتوي على بيانات المبيعات
    2. اضبط عدد المجموعات المطلوبة
    3. استكشف التحليلات والرسومات التفاعلية
    """)

# Main content
st.title('📊 لوحة تحليل المبيعات')
st.markdown("---")

# Data loading and processing
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file, parse_dates=['date'])
    elif os.path.exists('data/sample_sales.csv'):
        return pd.read_csv('data/sample_sales.csv', parse_dates=['date'])
    return None

# Load data
sales = load_data(uploaded_file)

if sales is None:
    st.warning("⚠️ الرجاء تحميل ملف CSV أو التأكد من وجود ملف sample_sales.csv في مجلد data")
    st.stop()

# Data processing
sales['year_month'] = sales['date'].dt.to_period('M')
sales['month_name'] = sales['date'].dt.month_name(locale='ar')
sales['day_of_week'] = sales['date'].dt.day_name(locale='ar')
sales['hour'] = sales['date'].dt.hour

# KPI Cards
st.subheader("📊 النظرة العامة")

# Calculate KPIs
total_revenue = sales['total_amount'].sum()
avg_order_value = sales['total_amount'].mean()
total_customers = sales['customer_id'].nunique()
total_orders = len(sales)

# Create columns for KPI cards
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">إجمالي الإيرادات</div>
            <div class="metric-value">${total_revenue:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)

with col2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">متوسط قيمة الطلب</div>
            <div class="metric-value">${avg_order_value:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)

with col3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">عدد العملاء</div>
            <div class="metric-value">{total_customers:,}</div>
        </div>
        """, unsafe_allow_html=True)

with col4:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">إجمالي الطلبات</div>
            <div class="metric-value">{total_orders:,}</div>
        </div>
        """, unsafe_allow_html=True)

# Data preview with expander
with st.expander("👁️ معاينة البيانات", expanded=False):
    st.dataframe(sales.head(20).style.background_gradient(cmap='Blues'), use_container_width=True)

# Sales Trends and Category Analysis
st.markdown("---")
st.subheader("📈 اتجاهات المبيعات والتحليل")

# Create tabs for different visualizations
tab1, tab2, tab3 = st.tabs(["المبيعات الشهرية", "توزيع الفئات", "التحليل الزمني"])

with tab1:
    # Monthly sales trend
    monthly_sales = sales.groupby(['year_month', 'month_name'])['total_amount'].sum().reset_index()
    monthly_sales = monthly_sales.sort_values('year_month')
    
    fig = px.line(
        monthly_sales, 
        x='month_name', 
        y='total_amount',
        title='<b>الاتجاه الشهري للمبيعات</b>',
        labels={'total_amount': 'إجمالي المبيعات', 'month_name': 'الشهر'},
        markers=True
    )
    fig.update_traces(line=dict(width=3), marker=dict(size=8))
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    # Sales by category
    category_sales = sales.groupby('category')['total_amount'].sum().reset_index()
    category_sales = category_sales.sort_values('total_amount', ascending=False)
    
    fig = px.pie(
        category_sales, 
        values='total_amount', 
        names='category',
        title='<b>توزيع المبيعات حسب الفئة</b>',
        hole=0.4,
        color_discrete_sequence=px.colors.sequential.Blues_r
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    # Time-based analysis
    col1, col2 = st.columns(2)
    
    with col1:
        # Daily pattern
        daily_sales = sales.groupby('day_of_week')['total_amount'].sum().reindex([
            'السبت', 'الأحد', 'الإثنين', 'الثلاثاء', 'الأربعاء', 'الخميس', 'الجمعة'
        ]).reset_index()
        
        fig = px.bar(
            daily_sales,
            x='day_of_week',
            y='total_amount',
            title='<b>المبيعات حسب اليوم</b>',
            labels={'total_amount': 'إجمالي المبيعات', 'day_of_week': 'اليوم'}
        )
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Hourly pattern
        hourly_sales = sales.groupby('hour')['total_amount'].sum().reset_index()
        
        fig = px.area(
            hourly_sales,
            x='hour',
            y='total_amount',
            title='<b>المبيعات حسب الساعة</b>',
            labels={'total_amount': 'إجمالي المبيعات', 'hour': 'ساعة اليوم'}
        )
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

# Customer Segmentation with RFM Analysis
st.markdown("---")
st.subheader("👥 تحليل وتقسيم العملاء")

# RFM Calculation
@st.cache_data
def calculate_rfm(data):
    rfm = data.groupby('customer_id').agg({
        'date': lambda x: (data['date'].max() - x.max()).days,
        'customer_id': 'count',
        'total_amount': 'sum'
    }).rename(columns={
        'date': 'recency',
        'customer_id': 'frequency',
        'total_amount': 'monetary'
    })
    rfm['recency'] = rfm['recency'].astype(int)
    return rfm

rfm = calculate_rfm(sales)

# KMeans Clustering
@st.cache_data
def perform_clustering(rfm_data, n_clusters=4):
    scaler = StandardScaler()
    X = scaler.fit_transform(rfm_data[['recency', 'frequency', 'monetary']])
    km = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = km.fit_predict(X)
    return clusters, km.cluster_centers_

rfm['cluster'] = perform_clustering(rfm, n_clusters)[0]

# Visualizations
col1, col2 = st.columns(2)

with col1:
    # RFM Distribution
    st.markdown("### توزيع RFM")
    
    fig = make_subplots(rows=1, cols=3, subplot_titles=('الأخيرة', 'التكرار', 'القيمة النقدية'))
    
    fig.add_trace(go.Histogram(x=rfm['recency'], name='الأخيرة', marker_color='#3498db'), row=1, col=1)
    fig.add_trace(go.Histogram(x=rfm['frequency'], name='التكرار', marker_color='#2ecc71'), row=1, col=2)
    fig.add_trace(go.Histogram(x=rfm['monetary'], name='القيمة', marker_color='#e74c3c'), row=1, col=3)
    
    fig.update_layout(
        showlegend=False,
        height=300,
        margin=dict(l=20, r=20, t=30, b=20),
        plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Cluster Distribution
    cluster_counts = rfm['cluster'].value_counts().sort_index()
    fig = px.pie(
        values=cluster_counts.values,
        names=[f'المجموعة {i+1}' for i in cluster_counts.index],
        title='<b>توزيع العملاء حسب المجموعات</b>',
        hole=0.5,
        color_discrete_sequence=px.colors.sequential.Blues_r
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # 3D Scatter Plot of Clusters
    st.markdown("### تجزئة العملاء ثلاثية الأبعاد")
    
    fig = px.scatter_3d(
        rfm,
        x='recency',
        y='frequency',
        z='monetary',
        color='cluster',
        color_continuous_scale='blues',
        labels={
            'recency': 'الأخيرة',
            'frequency': 'التكرار',
            'monetary': 'القيمة النقدية',
            'cluster': 'المجموعة'
        },
        title='<b>توزيع العملاء في الفضاء ثلاثي الأبعاد</b>'
    )
    
    fig.update_layout(
        scene=dict(
            xaxis_title='الأخيرة (أيام)',
            yaxis_title='التكرار',
            zaxis_title='القيمة النقدية ($)'
        ),
        height=600,
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Cluster Analysis
expander = st.expander("📊 تحليل المجموعات", expanded=False)
with expander:
    # Calculate cluster statistics
    cluster_stats = rfm.groupby('cluster').agg({
        'recency': 'mean',
        'frequency': 'mean',
        'monetary': ['mean', 'count']
    }).round(2)
    
    # Rename columns for better display
    cluster_stats.columns = ['متوسط الأخيرة', 'متوسط التكرار', 'متوسط القيمة', 'عدد العملاء']
    cluster_stats['نسبة العملاء'] = (cluster_stats['عدد العملاء'] / len(rfm) * 100).round(1).astype(str) + '%'
    
    # Display cluster statistics
    st.dataframe(
        cluster_stats.style.background_gradient(cmap='Blues'),
        use_container_width=True
    )
    
    # Add interpretation
    st.markdown("### تفسير المجموعات:")
    st.markdown("""
    - **المجموعة 0**: عملاء جدد مع إنفاق منخفض
    - **المجموعة 1**: عملاء مخلصين مع إنفاق مرتفع
    - **المجموعة 2**: عملاء معرضين للضياع
    - **المجموعة 3**: عملاء محتملين يحتاجون إلى عناية
    """)

# Insights and Recommendations
st.markdown("---")
st.subheader("💡 رؤى وتوصيات")

# Calculate key metrics
top_category = sales.groupby('category')['total_amount'].sum().idxmax()
top_month = sales.groupby('month_name')['total_amount'].sum().idxmax()
best_customer = rfm.sort_values('monetary', ascending=False).iloc[0].name

# Create columns for insights
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🎯 النتائج الرئيسية")
    st.markdown(f"""
    <div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
        <div style='display: flex; align-items: center; margin-bottom: 15px;'>
            <span style='font-size: 24px; margin-left: 10px;'>🔝</span>
            <div>
                <div style='font-weight: bold;'>أعلى فئة مبيعاً</div>
                <div style='font-size: 18px;'>{top_category}</div>
            </div>
        </div>
        <div style='display: flex; align-items: center; margin-bottom: 15px;'>
            <span style='font-size: 24px; margin-left: 10px;'>📅</span>
            <div>
                <div style='font-weight: bold;'>أفضل شهر في الأداء</div>
                <div style='font-size: 18px;'>{top_month}</div>
            </div>
        </div>
        <div style='display: flex; align-items: center;'>
            <span style='font-size: 24px; margin-left: 10px;'>👑</span>
            <div>
                <div style='font-weight: bold;'>أفضل عميل من حيث القيمة</div>
                <div style='font-size: 18px;'>{best_customer}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("### 📊 تحليل RFM")
    st.markdown("""
    <div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px;'>
        <div style='display: flex; align-items: center; margin-bottom: 15px;'>
            <span style='font-size: 24px; margin-left: 10px;'>🔄</span>
            <div>
                <div style='font-weight: bold;'>الأخيرة (Recency)</div>
                <div>عدد الأيام منذ آخر عملية شراء</div>
            </div>
        </div>
        <div style='display: flex; align-items: center; margin-bottom: 15px;'>
            <span style='font-size: 24px; margin-left: 10px;'>🔢</span>
            <div>
                <div style='font-weight: bold;'>التكرار (Frequency)</div>
                <div>عدد مرات الشراء</div>
            </div>
        </div>
        <div style='display: flex; align-items: center;'>
            <span style='font-size: 24px; margin-left: 10px;'>💰</span>
            <div>
                <div style='font-weight: bold;'>القيمة النقدية (Monetary)</div>
                <div>إجمالي المبلغ المنفق</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Recommendations
st.markdown("### 🚀 توصيات قابلة للتنفيذ")

rec_col1, rec_col2, rec_col3 = st.columns(3)

with rec_col1:
    st.markdown("""
    <div style='background-color: #e3f2fd; padding: 15px; border-radius: 10px; height: 100%;'>
        <h4>🎯 استهداف العملاء</h4>
        <ul style='padding-right: 20px;'>
            <li>ركز على العملاء ذوي القيمة العالية</li>
            <li>أعد استهداف العملاء المعرضين للضياع</li>
            <li>حفز العملاء النشطين مؤخرًا</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with rec_col2:
    st.markdown("""
    <div style='background-color: #e8f5e9; padding: 15px; border-radius: 10px; height: 100%;'>
        <h4>📈 تحسين المبيعات</h4>
        <ul style='padding-right: 20px;'>
            <li>عزز الفئات الأكثر مبيعًا</li>
            <li>قدم عروضًا ترويجية موسمية</li>
            <li>حسن تجربة الشراء</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with rec_col3:
    st.markdown("""
    <div style='background-color: #fff3e0; padding: 15px; border-radius: 10px; height: 100%;'>
        <h4>💡 تحليلات متقدمة</h4>
        <ul style='padding-right: 20px;'>
            <li>حلل أنماط الشراء</li>
            <li>توقع المبيعات المستقبلية</li>
            <li>حسن استراتيجية التسعير</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Download Section
st.markdown("---")
st.subheader("📥 تصدير البيانات")

# Add download buttons
dl_col1, dl_col2 = st.columns(2)

with dl_col1:
    # Download RFM Data
    @st.cache_data
    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')
    
    csv = convert_df(rfm)
    st.download_button(
        label="📊 تحميل بيانات RFM",
        data=csv,
        file_name=f'rfm_analysis_{datetime.now().strftime("%Y%m%d")}.csv',
        mime='text/csv',
        use_container_width=True
    )

with dl_col2:
    # Download Sales Data
    sales_csv = convert_df(sales)
    st.download_button(
        label="📈 تحميل بيانات المبيعات",
        data=sales_csv,
        file_name=f'sales_data_{datetime.now().strftime("%Y%m%d")}.csv',
        mime='text/csv',
        use_container_width=True
    )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d; font-size: 14px; margin-top: 30px;'>
    تم تطوير لوحة التحليل بواسطة فريق التحليلات | {}
</div>
""".format(datetime.now().year), unsafe_allow_html=True)
