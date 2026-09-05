import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from lifetimes import BetaGeoFitter, GammaGammaFitter, ParetoNBDFitter
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# Set page config
st.set_page_config(
    page_title="Customer Lifetime Value Prediction",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
    }
    .stDownloadButton>button {
        width: 100%;
        border-radius: 20px;
    }
    .css-1d391kg {
        padding: 2rem 1rem;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 10px 20px;
        border-radius: 10px 10px 0 0;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown("""
# 📊 Customer Lifetime Value Prediction Dashboard

This interactive dashboard helps you analyze and predict Customer Lifetime Value (CLV) using machine learning models.
""")

# Sidebar inputs
# st.sidebar.title("⚙️ Model Settings")

model_type = "BG/NBD + Gamma-Gamma"
# Model selection
# st.sidebar.subheader("Model Selection")
# model_type = st.sidebar.selectbox(
#     "Select Model Type",
#     ["BG/NBD + Gamma-Gamma (Recommended)", "Pareto/NBD + Gamma-Gamma"],
#     help="Choose between BG/NBD or Pareto/NBD model for purchase prediction"
# )

# File upload
# uploaded_file = st.sidebar.file_uploader(
#     "📤 Upload your CSV file",
#     type=["csv"],
#     help="Upload a CSV file with columns: frequency, recency, T, monetary_value"
# )

# Model parameters
st.sidebar.subheader("Prediction Settings")
prediction_days = st.sidebar.slider(
    "📅 Prediction Period (days)",
    min_value=7,
    max_value=365,
    value=30,
    step=7,
    help="Select the number of days to predict CLV for"
)

discount_rate = st.sidebar.slider(
    "💵 Annual Discount Rate (%)",
    min_value=0.0,
    max_value=20.0,
    value=10.0,
    step=0.5,
    help="Annual discount rate for CLV calculation"
) / 100.0  # Convert to decimal

# Add customer data input form
# st.sidebar.subheader("🔍 Check Individual Customer")
# with st.sidebar.form("customer_form"):
#     st.markdown("### Enter Customer Details")
    
#     # Input fields for customer data
#     frequency = st.number_input(
#         "Frequency (total transactions - 1)",
#         min_value=0,
#         step=1,
#         value=1,
#         help="Number of repeat purchases (total transactions - 1)"
#     )
    
#     recency = st.number_input(
#         "Recency (days since last purchase)",
#         min_value=0,
#         step=1,
#         value=30,
#         help="Days since the last purchase"
#     )
    
#     T = st.number_input(
#         "Customer Age (days since first purchase)",
#         min_value=recency + 1 if 'recency' in locals() else 31,
#         step=1,
#         value=60,
#         help="Days since the first purchase (must be > recency)"
#     )
    
#     monetary_value = st.number_input(
#         "Average Order Value ($)",
#         min_value=0.01,
#         step=1.0,
#         value=100.0,
#         format="%.2f",
#         help="Average monetary value per transaction"
#     )
    
#     # Submit button for the form
#     submitted = st.form_submit_button("Calculate CLV")

# Add a button to retrain models
# if st.sidebar.button("🔄 Retrain Models"):
#     import subprocess
#     with st.spinner("Training models... This may take a few minutes."):
#         result = subprocess.run(["python", "train_clv_model.py"], capture_output=True, text=True)
#         if result.returncode == 0:
#             st.sidebar.success("Models retrained successfully!")
#         else:
#             st.sidebar.error("Error during model training!")

uploaded_file = None
# Load sample data if no file uploaded
if uploaded_file is None:
    try:
        df = pd.read_csv('customer_lifetime_value_prediction.csv')
        # st.sidebar.info("Using sample data. Upload your own file to analyze your customers.")
    except Exception as e:
        st.error(f"Error loading sample data: {str(e)}")
        st.stop()
else:
    try:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("Uploaded data loaded successfully!")
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        st.stop()

# Check required columns
required_columns = ['frequency', 'recency', 'T', 'monetary_value']
if not all(col in df.columns for col in required_columns):
    missing = [col for col in required_columns if col not in df.columns]
    st.error(f"Error: The uploaded file is missing required columns: {', '.join(missing)}")
    st.stop()

# Load models
@st.cache_resource
def load_models():
    try:
        bgf = joblib.load('bgf_model.pkl')
        ggf = joblib.load('ggf_model.pkl')
        pnbd = joblib.load('pnbd_model.pkl') if model_type == "Pareto/NBD + Gamma-Gamma" else None
        return bgf, ggf, pnbd
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.info("Please make sure to run 'train_clv_model.py' first to train the models.")
        st.stop()

try:
    bgf, ggf, pnbd = load_models()
    
    # Process individual customer form submission
    if 'submitted' in locals() and submitted:
        # Create a single customer DataFrame
        customer_data = pd.DataFrame({
            'frequency': [frequency],
            'recency': [recency],
            'T': [T],
            'monetary_value': [monetary_value]
        })
        
        try:
            # Make predictions based on selected model
            if model_type == "Pareto/NBD + Gamma-Gamma":
                pred_purchases = pnbd.conditional_expected_number_of_purchases_up_to_time(
                    prediction_days, frequency, recency, T
                )
                model_used = "Pareto/NBD"
            else:
                pred_purchases = bgf.conditional_expected_number_of_purchases_up_to_time(
                    prediction_days, frequency, recency, T
                )
                model_used = "BG/NBD"
            
            # Use Gamma-Gamma for monetary value prediction
            pred_clv = ggf.customer_lifetime_value(
                pnbd if model_type == "Pareto/NBD + Gamma-Gamma" else bgf,
                frequency,
                recency,
                T,
                monetary_value,
                time=prediction_days/30,
                discount_rate=discount_rate/12,
                freq='D'
            )
            
            # Store in session state
            st.session_state.customer_data = {
                'frequency': frequency,
                'recency': recency,
                'T': T,
                'monetary_value': monetary_value,
                'predicted_purchases': pred_purchases,
                'predicted_clv': pred_clv[0],
                'model_used': model_used
            }
            
        except Exception as e:
            st.sidebar.error(f"Error making prediction: {str(e)}")
    
    # Show individual customer results if available
    if 'customer_data' in st.session_state:
        cust = st.session_state.customer_data
        with st.sidebar.expander("📊 Individual Customer Results", expanded=True):
            st.metric("Model Used", cust['model_used'])
            st.metric("Predicted Purchases", f"{cust['predicted_purchases']:.2f}")
            st.metric("Predicted CLV", f"${cust['predicted_clv']:.2f}")
            
            # Add to main dataframe for visualization
            customer_df = pd.DataFrame([{
                'frequency': cust['frequency'],
                'recency': cust['recency'],
                'T': cust['T'],
                'monetary_value': cust['monetary_value'],
                'predicted_purchases': cust['predicted_purchases'],
                'predicted_clv': cust['predicted_clv'],
                'is_custom': True
            }])
            
            # Add to main df for visualization
            if 'df' in locals():
                df['is_custom'] = False
                combined_df = pd.concat([df, customer_df], ignore_index=True)
            else:
                combined_df = customer_df
    
    # Calculate predictions
    with st.spinner("Calculating predictions..."):
        df['predicted_purchases'] = bgf.conditional_expected_number_of_purchases_up_to_time(
            prediction_days,
            df['frequency'],
            df['recency'],
            df['T']
        )

        df['predicted_clv'] = ggf.customer_lifetime_value(
            bgf,
            df['frequency'],
            df['recency'],
            df['T'],
            df['monetary_value'],
            time=prediction_days/30,  # Convert to months
            discount_rate=discount_rate/12,  # Monthly discount rate
            freq='D'  # Daily frequency
        )
        
        # Calculate customer segments
        df['clv_segment'] = pd.qcut(df['predicted_clv'], 
                                   q=[0, 0.25, 0.5, 0.75, 1], 
                                   labels=['Low', 'Medium', 'High', 'VIP'])
        
        # Calculate recency segments
        df['recency_segment'] = pd.qcut(df['recency'], 
                                       q=[0, 0.25, 0.5, 0.75, 1], 
                                       labels=['Very Recent', 'Recent', 'Less Recent', 'Not Recent'])
        
        # Calculate frequency segments
        df['frequency_segment'] = pd.qcut(df['frequency'], 
                                         q=[0, 0.25, 0.5, 0.75, 1], 
                                         labels=['Low', 'Medium', 'High', 'Very High'])

    # Display results
    st.subheader("📈 Customer Insights Dashboard")
    
    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>Total Customers</h3>
            <h2>{:,}</h2>
        </div>
        """.format(len(df)), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>Avg. Predicted CLV</h3>
            <h2>${:,.2f}</h2>
        </div>
        """.format(df['predicted_clv'].mean()), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>Total Predicted Value</h3>
            <h2>${:,.2f}</h2>
        </div>
        """.format(df['predicted_clv'].sum()), unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>Avg. Predicted Purchases</h3>
            <h2>{:.2f}</h2>
        </div>
        """.format(df['predicted_purchases'].mean()), unsafe_allow_html=True)
    
    # Tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Overview", 
        "📈 CLV Analysis", 
        "👥 Customer Segments", 
        "🔍 Customer Details"
    ])
    
    with tab1:
        # Row 1: CLV Distribution and Recency vs Frequency
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### CLV Distribution")
            fig1 = px.histogram(
                df, 
                x='predicted_clv', 
                nbins=50,
                labels={'predicted_clv': 'Predicted CLV ($)'},
                color_discrete_sequence=['#636EFA']
            )
            fig1.update_layout(
                xaxis_title="Predicted CLV ($)",
                yaxis_title="Number of Customers",
                showlegend=False
            )
            st.plotly_chart(fig1, use_container_width=True)
            
        with col2:
            st.markdown("### Recency vs Frequency (Color by CLV)")
            
            # Prepare data for plotting
            plot_df = df.sample(min(1000, len(df))).copy() if 'df' in locals() else pd.DataFrame()
            
            # Add custom customer if available
            if 'customer_data' in st.session_state and 'combined_df' in locals():
                custom_cust = st.session_state.customer_data
                fig2 = px.scatter(
                    combined_df,
                    x='recency',
                    y='frequency',
                    color='predicted_clv',
                    hover_data=['Customer ID'] if 'Customer ID' in combined_df.columns else None,
                    labels={
                        'recency': 'Recency (days)',
                        'frequency': 'Frequency',
                        'predicted_clv': 'Predicted CLV ($)'
                    },
                    color_continuous_scale='viridis'
                )
                
                # Add custom marker for the individual customer
                fig2.add_trace(
                    go.Scatter(
                        x=[custom_cust['recency']],
                        y=[custom_cust['frequency']],
                        mode='markers',
                        marker=dict(
                            color='red',
                            size=12,
                            line=dict(color='white', width=2)
                        ),
                        name='Your Customer',
                        hovertext=f"CLV: ${custom_cust['predicted_clv']:.2f}<br>"
                                f"Model: {custom_cust['model_used']}"
                    )
                )
            else:
                fig2 = px.scatter(
                    plot_df,
                    x='recency',
                    y='frequency',
                    color='predicted_clv',
                    hover_data=['Customer ID'] if 'Customer ID' in plot_df.columns else None,
                    labels={
                        'recency': 'Recency (days)',
                        'frequency': 'Frequency',
                        'predicted_clv': 'Predicted CLV ($)'
                    },
                    color_continuous_scale='viridis'
                )
                
            fig2.update_layout(showlegend=True)
            st.plotly_chart(fig2, use_container_width=True)
        
        # Row 2: Monetary Value vs Frequency and Correlation Heatmap
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown("### Monetary Value vs Frequency")
            fig3 = px.scatter(
                df.sample(min(1000, len(df))),  # Sample for better performance
                x='frequency',
                y='monetary_value',
                color='predicted_clv',
                hover_data=['Customer ID'],
                labels={
                    'frequency': 'Frequency',
                    'monetary_value': 'Monetary Value ($)',
                    'predicted_clv': 'Predicted CLV ($)'
                },
                color_continuous_scale='plasma'
            )
            st.plotly_chart(fig3, use_container_width=True)
            
        with col4:
            st.markdown("### Feature Correlation")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            corr = df[numeric_cols].corr()
            fig4 = px.imshow(
                corr,
                text_auto=True,
                aspect="auto",
                color_continuous_scale='RdBu',
                zmin=-1,
                zmax=1
            )
            st.plotly_chart(fig4, use_container_width=True)
    
    with tab2:
        # CLV Analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### CLV by Recency Segment")
            fig5 = px.box(
                df,
                x='recency_segment',
                y='predicted_clv',
                color='recency_segment',
                labels={
                    'recency_segment': 'Recency Segment',
                    'predicted_clv': 'Predicted CLV ($)'
                }
            )
            st.plotly_chart(fig5, use_container_width=True)
            
        with col2:
            st.markdown("### CLV by Frequency Segment")
            fig6 = px.box(
                df,
                x='frequency_segment',
                y='predicted_clv',
                color='frequency_segment',
                category_orders={"frequency_segment": ["Low", "Medium", "High", "Very High"]},
                labels={
                    'frequency_segment': 'Frequency Segment',
                    'predicted_clv': 'Predicted CLV ($)'
                }
            )
            st.plotly_chart(fig6, use_container_width=True)
        
        # CLV Distribution by Segment
        st.markdown("### CLV Distribution by Segment")
        fig7 = px.violin(
            df,
            x='clv_segment',
            y='predicted_clv',
            color='clv_segment',
            box=True,
            points="all",
            category_orders={"clv_segment": ["Low", "Medium", "High", "VIP"]},
            labels={
                'clv_segment': 'CLV Segment',
                'predicted_clv': 'Predicted CLV ($)'
            }
        )
        st.plotly_chart(fig7, use_container_width=True)
        
    with tab3:
        # Customer Segments Analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Customer Distribution by CLV Segment")
            segment_counts = df['clv_segment'].value_counts().reset_index()
            segment_counts.columns = ['CLV Segment', 'Count']
            fig8 = px.pie(
                segment_counts,
                values='Count',
                names='CLV Segment',
                hole=0.4,
                category_orders={"CLV Segment": ["Low", "Medium", "High", "VIP"]}
            )
            st.plotly_chart(fig8, use_container_width=True)
            
        with col2:
            st.markdown("### Average CLV by Segment")
            segment_avg = df.groupby('clv_segment')['predicted_clv'].mean().reset_index()
            fig9 = px.bar(
                segment_avg,
                x='clv_segment',
                y='predicted_clv',
                color='clv_segment',
                category_orders={"clv_segment": ["Low", "Medium", "High", "VIP"]},
                labels={
                    'clv_segment': 'CLV Segment',
                    'predicted_clv': 'Average CLV ($)'
                }
            )
            st.plotly_chart(fig9, use_container_width=True)
        
        # RFM Analysis
        st.markdown("### RFM Analysis")
        rfm = df[['recency', 'frequency', 'monetary_value', 'predicted_clv']].copy()
        rfm['r_quartile'] = pd.qcut(rfm['recency'], 4, ['1', '2', '3', '4'])
        rfm['f_quartile'] = pd.qcut(rfm['frequency'], 4, ['4', '3', '2', '1'])
        rfm['m_quartile'] = pd.qcut(rfm['monetary_value'], 4, ['4', '3', '2', '1'])
        rfm['RFM_Score'] = rfm['r_quartile'].astype(str) + rfm['f_quartile'].astype(str) + rfm['m_quartile'].astype(str)
        
        # Create a 3D scatter plot for RFM
        fig10 = px.scatter_3d(
            rfm.sample(min(500, len(rfm))),  # Sample for better performance
            x='recency',
            y='frequency',
            z='monetary_value',
            color='predicted_clv',
            size='predicted_clv',
            opacity=0.7,
            labels={
                'recency': 'Recency',
                'frequency': 'Frequency',
                'monetary_value': 'Monetary Value ($)',
                'predicted_clv': 'Predicted CLV ($)'
            }
        )
        st.plotly_chart(fig10, use_container_width=True, height=600)
        
    with tab4:
        # Customer Details Table
        st.markdown("### Customer Details")
        st.dataframe(
            df[['Customer ID', 'recency', 'frequency', 'monetary_value', 'predicted_clv', 'clv_segment']].sort_values(
                'predicted_clv', ascending=False
            ).reset_index(drop=True),
            height=400,
            use_container_width=True
        )
        
        # Download results
        st.markdown("### Download Predictions")
        
        # Create a download button for the main dataset
        if 'df' in locals():
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download Full Dataset as CSV",
                data=csv,
                file_name="customer_clv_predictions.csv",
                mime="text/csv"
            )
        
        # Add a button to download the individual customer prediction
        if 'customer_data' in st.session_state:
            cust = st.session_state.customer_data
            cust_df = pd.DataFrame([{
                'frequency': cust['frequency'],
                'recency': cust['recency'],
                'T': cust['T'],
                'monetary_value': cust['monetary_value'],
                'predicted_purchases': cust['predicted_purchases'],
                'predicted_clv': cust['predicted_clv'],
                'model_used': cust['model_used']
            }])
            
            csv = cust_df.to_csv(index=False)
            st.download_button(
                label="Download Individual Prediction",
                data=csv,
                file_name="individual_clv_prediction.csv",
                mime="text/csv"
            )
    
    # Add some space at the bottom
    st.markdown("---")
    st.info("""
    💡 **Tips**: 
    - Use the sidebar to adjust prediction settings and upload your own data
    - Hover over data points in the charts for more details
    - Click and drag to zoom in on specific areas of the charts
    """)

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.info("Please check if the required model files exist and the data format is correct.")