import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Bangalore Restaurant Recommender", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for unified styling
st.markdown("""
<style>
    .main {
        padding-top: 1rem;
        background-color: #f8f9fa;
    }
    
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* Sidebar styling - Fixed for better visibility */
    .stSidebar {
        background-color: #ffffff !important;
        border-right: 2px solid #e9ecef;
        padding-top: 1rem;
    }
    
    .stSidebar .stSidebar-content {
        background-color: #ffffff !important;
        padding: 1rem;
    }
    
    /* Fix sidebar text and components */
    .stSidebar h1, .stSidebar h2, .stSidebar h3, .stSidebar h4, .stSidebar h5, .stSidebar h6,
    .stSidebar p, .stSidebar div, .stSidebar span, .stSidebar label {
        color: #2c3e50 !important;
    }
    
    .stSidebar .stMarkdown {
        color: #2c3e50 !important;
    }
    
    /* Fix radio buttons in sidebar */
    .stSidebar .stRadio > div {
        background-color: #f8f9fa !important;
        padding: 1rem !important;
        border-radius: 8px !important;
        border: 2px solid #dee2e6 !important;
    }
    
    .stSidebar .stRadio label {
        color: #2c3e50 !important;
        font-weight: 600 !important;
    }
    
    /* Fix selectbox in sidebar */
    .stSidebar .stSelectbox > div > div {
        background-color: #ffffff !important;
        color: #2c3e50 !important;
        border: 2px solid #dee2e6 !important;
        border-radius: 8px !important;
    }
    
    .stSidebar .stSelectbox label {
        color: #2c3e50 !important;
        font-weight: 600 !important;
    }
    
    /* Sidebar metrics styling */
    .stSidebar .stMetric {
        background-color: #f8f9fa !important;
        padding: 0.75rem !important;
        border-radius: 8px !important;
        border: 1px solid #dee2e6 !important;
        margin-bottom: 0.5rem !important;
    }
    
    .stSidebar .stMetric label, 
    .stSidebar .stMetric div[data-testid="metric-container"] > div {
        color: #2c3e50 !important;
        font-weight: 500 !important;
    }
    
    /* Main content styling */
    .restaurant-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        border: 1px solid #e9ecef;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .restaurant-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.15);
    }
    
    .restaurant-name {
        color: #2c3e50;
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .restaurant-details {
        color: #495057;
        font-size: 0.95rem;
        line-height: 1.5;
        margin-bottom: 6px;
        display: flex;
        align-items: flex-start;
        gap: 8px;
    }
    
    .rating-cost {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-top: 12px;
        padding-top: 12px;
        border-top: 1px solid #f1f3f4;
        gap: 10px;
        flex-wrap: wrap;
    }
    
    .rating {
        background: linear-gradient(135deg, #28a745, #20c997);
        color: white;
        padding: 6px 12px;
        border-radius: 15px;
        font-weight: 600;
        font-size: 0.85rem;
        display: flex;
        align-items: center;
        gap: 4px;
        white-space: nowrap;
    }
    
    .cost {
        background: linear-gradient(135deg, #007bff, #0056b3);
        color: white;
        padding: 6px 12px;
        border-radius: 15px;
        font-weight: 600;
        font-size: 0.85rem;
        display: flex;
        align-items: center;
        gap: 4px;
        white-space: nowrap;
    }
    
    .features-row {
        display: flex;
        gap: 8px;
        margin-top: 8px;
        flex-wrap: wrap;
    }
    
    .feature-badge {
        background: #f8f9fa;
        color: #495057;
        padding: 4px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        border: 1px solid #dee2e6;
    }
    
    .header-container {
        text-align: center;
        padding: 1.5rem 0;
        margin-bottom: 1.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        color: white;
    }
    
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        color: white !important;
    }
    
    .subtitle {
        font-size: 1.1rem;
        opacity: 0.9;
        font-weight: 400;
        color: white !important;
    }
    
    .section-header {
        color: #2c3e50 !important;
        font-size: 1.4rem;
        font-weight: 600;
        margin: 1.5rem 0 1rem 0;
        text-align: center;
        padding: 0.8rem;
        background: #ffffff;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    
    .no-results {
        text-align: center;
        color: #6c757d;
        font-size: 1.1rem;
        padding: 2rem;
        background: #ffffff;
        border-radius: 12px;
        margin: 2rem 0;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
    }
    
    .sidebar-section {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border: 1px solid #dee2e6;
    }
    
    .sidebar-title {
        color: #2c3e50 !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Info boxes */
    .stInfo {
        background-color: #e3f2fd !important;
        border: 1px solid #90caf9 !important;
        border-radius: 8px !important;
    }
    
    .stInfo > div {
        color: #1565c0 !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and preprocess the data"""
    try:
        # Try to load the CSV file
        df = pd.read_csv("zomato_cleaned.csv")
        
        # Data cleaning and preprocessing
        df['cuisines'] = df['cuisines'].fillna('Unknown')
        df['name'] = df['name'].fillna('Unknown Restaurant')
        df['full_address'] = df['full_address'].fillna('Address not available')
        
        # Handle ratings - use 'dinner ratings' as primary rating column
        if 'dinner ratings' in df.columns:
            df['rating'] = pd.to_numeric(df['dinner ratings'], errors='coerce').fillna(0)
        else:
            # Fallback to other rating columns if dinner ratings not available
            rating_columns = ['rate', 'rating', 'aggregate rating']
            df['rating'] = 0
            for col in rating_columns:
                if col in df.columns:
                    df['rating'] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                    break
        
        # Handle cost - use 'averagecost' column directly since it exists in your data
        if 'averagecost' in df.columns:
            df['cost'] = pd.to_numeric(df['averagecost'], errors='coerce').fillna(0)
        else:
            # Fallback for other possible cost columns
            cost_columns = ['average_cost_for_two', 'cost', 'approx_cost(for two people)']
            df['cost'] = 0
            for col in cost_columns:
                if col in df.columns:
                    if df[col].dtype == 'object':
                        df['cost'] = df[col].astype(str).str.replace(',', '').str.replace('₹', '').str.replace('Rs.', '').str.replace('Rs', '').str.extract('(\d+)').astype(float).fillna(0)
                    else:
                        df['cost'] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                    break
        
        # Add service features (convert boolean columns to more readable names)
        if 'ishomedelivery' in df.columns:
            df['home_delivery'] = df['ishomedelivery'].fillna(False)
        else:
            df['home_delivery'] = False
            
        if 'istakeaway' in df.columns:
            df['takeaway'] = df['istakeaway'].fillna(False)
        else:
            df['takeaway'] = False
            
        if 'isindoorseating' in df.columns:
            df['indoor_seating'] = df['isindoorseating'].fillna(False)
        else:
            df['indoor_seating'] = False
            
        if 'isvegonly' in df.columns:
            df['veg_only'] = df['isvegonly'].fillna(False)
        else:
            df['veg_only'] = False
        
        # Remove duplicates and invalid entries
        df = df.drop_duplicates(subset=['name'], keep='first')
        df = df[df['name'] != 'Unknown Restaurant']
        
        # IMPORTANT: Reset index to ensure continuity and avoid index mismatches
        df = df.reset_index(drop=True)
        
        return df
        
    except FileNotFoundError:
        st.error("Data file 'zomato_cleaned.csv' not found in the current directory.")
        st.info("Please make sure the CSV file is in the same folder as this script.")
        return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

@st.cache_data
def compute_similarity(df):
    """Compute TF-IDF and cosine similarity matrix"""
    try:
        # Combine cuisines and other features for better recommendations
        df_copy = df.copy()
        df_copy['features'] = (
            df_copy['cuisines'].fillna('') + ' ' + 
            df_copy['name'].fillna('') + ' ' +
            df_copy['area'].fillna('')
        )
        
        tfidf = TfidfVectorizer(
            stop_words='english', 
            max_features=1000,
            ngram_range=(1, 2),
            min_df=1
        )
        
        tfidf_matrix = tfidf.fit_transform(df_copy['features'])
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        
        return cosine_sim
    except Exception as e:
        st.error(f"Error computing similarity: {str(e)}")
        return None

def recommend_by_restaurant(df, cosine_sim, name, num_recommendations=6):
    """Recommend restaurants similar to a given restaurant"""
    try:
        # Find the restaurant index
        matches = df[df['name'].str.contains(name, case=False, na=False)]
        if matches.empty:
            return pd.DataFrame()
        
        # Get the first match index
        idx = matches.index[0]
        
        # Safety check: ensure index is within bounds
        if idx >= len(df) or idx >= cosine_sim.shape[0]:
            st.error(f"Index error: Restaurant index {idx} is out of bounds.")
            return pd.DataFrame()
        
        # Get similarity scores
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get top similar restaurants (excluding the selected one)
        restaurant_indices = []
        for i, score in sim_scores[1:num_recommendations+1]:
            if i < len(df):  # Safety check
                restaurant_indices.append(i)
        
        if not restaurant_indices:
            return pd.DataFrame()
        
        recommendations = df.iloc[restaurant_indices]
        return recommendations.reset_index(drop=True)
        
    except Exception as e:
        st.error(f"Error in recommendation: {str(e)}")
        return pd.DataFrame()

def recommend_by_cuisine(df, cuisine, num_recommendations=8):
    """Recommend restaurants by cuisine type"""
    try:
        # Filter restaurants that have the specified cuisine
        cuisine_restaurants = df[df['cuisines'].str.contains(cuisine, case=False, na=False)]
        
        if len(cuisine_restaurants) == 0:
            return pd.DataFrame()
        
        # Sort by rating (descending) and then by cost (ascending) for better recommendations
        top_restaurants = cuisine_restaurants.sort_values(
            ['rating', 'cost'], 
            ascending=[False, True]
        ).head(num_recommendations)
        
        return top_restaurants.reset_index(drop=True)
        
    except Exception as e:
        st.error(f"Error in cuisine recommendation: {str(e)}")
        return pd.DataFrame()

def display_restaurant_card(restaurant):
    """Display a restaurant in a styled card format"""
    # Format rating
    if restaurant['rating'] > 0:
        rating_display = f"★ {restaurant['rating']:.1f}"
        rating_color = "#28a745" if restaurant['rating'] >= 4.0 else "#ffc107" if restaurant['rating'] >= 3.0 else "#dc3545"
    else:
        rating_display = "★ N/A"
        rating_color = "#6c757d"
    
    # Format cost
    cost_display = "Cost N/A"
    if 'cost' in restaurant and not pd.isna(restaurant['cost']) and restaurant['cost'] > 0:
        cost_value = int(restaurant['cost'])
        cost_display = f"₹{cost_value:,} for 2"
    elif 'averagecost' in restaurant and not pd.isna(restaurant['averagecost']) and restaurant['averagecost'] > 0:
        cost_value = int(restaurant['averagecost'])
        cost_display = f"₹{cost_value:,} for 2"
    
    # Truncate long addresses
    address = str(restaurant['full_address'])
    if len(address) > 80:
        address = address[:80] + "..."
    
    # Truncate long cuisine lists
    cuisines = str(restaurant['cuisines'])
    if len(cuisines) > 50:
        cuisines = cuisines[:50] + "..."
    
    # Service features
    service_badges = []
    if 'home_delivery' in restaurant and restaurant['home_delivery']:
        service_badges.append("Delivery")
    if 'takeaway' in restaurant and restaurant['takeaway']:
        service_badges.append("Takeaway")
    if 'indoor_seating' in restaurant and restaurant['indoor_seating']:
        service_badges.append("Dine-in")
    if 'veg_only' in restaurant and restaurant['veg_only']:
        service_badges.append("Veg Only")
    
    service_html = ""
    if service_badges:
        badges_html = " ".join([f'<span class="feature-badge">{badge}</span>' for badge in service_badges])
        service_html = f'<div class="features-row">{badges_html}</div>'
    
    st.markdown(f"""
    <div class="restaurant-card">
        <div class="restaurant-name">{restaurant['name']}</div>
        <div class="restaurant-details"><strong>Address:</strong> {address}</div>
        <div class="restaurant-details"><strong>Cuisines:</strong> {cuisines}</div>
        {service_html}
        <div class="rating-cost">
            <span class="rating" style="background-color: {rating_color};">{rating_display}</span>
            <span class="cost">{cost_display}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

def main():
    # Header
    st.markdown("""
    <div class="header-container">
        <h1 class="main-title">Bangalore Restaurant Recommender</h1>
        <p class="subtitle">Discover amazing restaurants tailored to your taste</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    if df is None:
        st.stop()
    
    # Check if dataframe is empty
    if len(df) == 0:
        st.error("No restaurant data found in the CSV file.")
        st.stop()
    
    # Compute similarity matrix
    with st.spinner("Loading restaurant data..."):
        cosine_sim = compute_similarity(df)
        if cosine_sim is None:
            st.stop()
    
    # Sidebar for filters
    with st.sidebar:
        
        st.markdown('<h3 class="sidebar-title">Recommendation Settings</h3>', unsafe_allow_html=True)
        
        option = st.radio(
            "Choose your recommendation method:",
            ["By Restaurant Name", "By Cuisine Type"],
            help="Select how you'd like to discover new restaurants"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Quick Stats in sidebar
        
        st.markdown('<h3 class="sidebar-title">Quick Stats</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Restaurants", len(df))
            
        with col2:
            avg_rating = df[df['rating'] > 0]['rating'].mean() if len(df[df['rating'] > 0]) > 0 else 0
            st.metric("Avg Rating", f"{avg_rating:.1f}")
        
        # Calculate unique cuisines
        all_cuisines = []
        for cuisines in df['cuisines'].dropna():
            if cuisines and cuisines != 'Unknown':
                all_cuisines.extend([c.strip() for c in str(cuisines).split(',')])
        unique_cuisines_count = len(set([c for c in all_cuisines if c and c != '']))
        
        # Cost statistics
        cost_data = df[df['cost'] > 0]['cost'] if 'cost' in df.columns else df[df['averagecost'] > 0]['averagecost']
        avg_cost = cost_data.mean() if len(cost_data) > 0 else 0
        
        st.metric("Unique Cuisines", unique_cuisines_count)
        st.metric("Avg Cost (₹)", f"{avg_cost:,.0f}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Tips section
        
        st.markdown('<h3 class="sidebar-title">Tips</h3>', unsafe_allow_html=True)
        st.info("Try different restaurants to discover new flavors")
        st.info("Use cuisine search to explore specific food types")
        st.info("Look for service badges for delivery options")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Main content area
    if option == "By Restaurant Name":
        st.markdown('<h2 class="section-header">Find Similar Restaurants</h2>', unsafe_allow_html=True)
        
        # Restaurant selection
        restaurant_names = sorted([name for name in df['name'].dropna().unique() if name != 'Unknown Restaurant'])
        
        if len(restaurant_names) == 0:
            st.error("No valid restaurant names found in the data.")
            return
        
        selected_restaurant = st.selectbox(
            "Select a restaurant you like:",
            restaurant_names,
            help="Choose a restaurant to find similar options"
        )
        
        if selected_restaurant:
            with st.spinner(f"Finding restaurants similar to {selected_restaurant}..."):
                results = recommend_by_restaurant(df, cosine_sim, selected_restaurant)
            
            if not results.empty:
                st.markdown(f'<h3 class="section-header">Restaurants similar to "{selected_restaurant}"</h3>', unsafe_allow_html=True)
                
                # Display results in columns for better layout
                for i in range(len(results)):
                    if i % 2 == 0:
                        cols = st.columns(2)
                    
                    with cols[i % 2]:
                        display_restaurant_card(results.iloc[i])
            else:
                st.markdown('<div class="no-results">No similar restaurants found. Try selecting a different restaurant.</div>', unsafe_allow_html=True)
    
    elif option == "By Cuisine Type":
        st.markdown('<h2 class="section-header">Explore by Cuisine</h2>', unsafe_allow_html=True)
        
        # Get unique cuisines
        all_cuisines = []
        for cuisines in df['cuisines'].dropna():
            if cuisines and cuisines != 'Unknown':
                all_cuisines.extend([c.strip() for c in str(cuisines).split(',')])
        
        unique_cuisines = sorted(set([c for c in all_cuisines if c and c != '' and len(c) > 1]))
        
        if len(unique_cuisines) == 0:
            st.error("No valid cuisines found in the data.")
            return
        
        cuisine_input = st.selectbox(
            "Choose your preferred cuisine:",
            unique_cuisines,
            help="Select a cuisine type to discover top-rated restaurants"
        )
        
        if cuisine_input:
            with st.spinner(f"Finding the best {cuisine_input} restaurants..."):
                results = recommend_by_cuisine(df, cuisine_input)
            
            if not results.empty:
                st.markdown(f'<h3 class="section-header">Top {cuisine_input} Restaurants</h3>', unsafe_allow_html=True)
                
                # Display results in columns for better layout
                for i in range(len(results)):
                    if i % 2 == 0:
                        cols = st.columns(2)
                    
                    with cols[i % 2]:
                        display_restaurant_card(results.iloc[i])
            else:
                st.markdown(f'<div class="no-results">No restaurants found for {cuisine_input} cuisine. Try selecting a different cuisine.</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()