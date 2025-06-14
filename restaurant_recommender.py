import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# page configuration
st.set_page_config(
    page_title="Bangalore Restaurant Recommender", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# custom CSS
st.markdown("""
<style>
    :root {
        /* Professional Blue Color Palette */
        --primary-blue: #2563eb;
        --primary-blue-light: #3b82f6;
        --primary-blue-dark: #1d4ed8;
        
        /* Background Colors */
        --bg-primary: #ffffff;
        --bg-secondary: #f8fafc;
        --bg-tertiary: #f1f5f9;
        --bg-accent: #eff6ff;
        
        /* Text Colors */
        --text-primary: #1e293b;
        --text-secondary: #64748b;
        --text-accent: #2563eb;
        
        /* Border Colors */
        --border-light: #e2e8f0;
        --border-medium: #cbd5e1;
        --border-accent: #bfdbfe;
        
        /* Status Colors */
        --success: #10b981;
        --success-light: #d1fae5;
        --warning: #f59e0b;
        --warning-light: #fef3c7;
    }
    
    .main {
        padding-top: 1rem;
        background: var(--bg-secondary);
        min-height: 100vh;
    }
    
    .stApp {
        background: var(--bg-secondary);
    }
    
    /* Sidebar */
    .stSidebar {
        background: var(--bg-primary) !important;
        border-right: 1px solid var(--border-light);
        padding-top: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    .stSidebar .stSidebar-content {
        background: var(--bg-primary) !important;
        padding: 1.5rem;
    }
    
    /* sidebartext */
    .stSidebar h1, .stSidebar h2, .stSidebar h3, .stSidebar h4, .stSidebar h5, .stSidebar h6,
    .stSidebar p, .stSidebar div, .stSidebar span, .stSidebar label {
        color: var(--text-primary) !important;
        font-weight: 500;
    }
    
    .stSidebar .stMarkdown {
        color: var(--text-primary) !important;
    }
    
    /* radio buttons */
    .stSidebar .stRadio > div {
        background: var(--bg-primary) !important;
        padding: 1rem !important;
        border-radius: 8px !important;
        border: 1px solid var(--border-light) !important;
        transition: all 0.2s ease;
    }
    
    .stSidebar .stRadio > div:hover {
        border-color: var(--border-accent) !important;
        background: var(--bg-accent) !important;
    }
    
    .stSidebar .stRadio label {
        color: var(--text-primary) !important;
        font-weight: 500 !important;
    }
    
    /* selectbox */
    .stSidebar .stSelectbox > div > div {
        background: var(--bg-primary) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-light) !important;
        border-radius: 6px !important;
        transition: all 0.2s ease;
    }
    
    .stSidebar .stSelectbox > div > div:hover {
        border-color: var(--border-accent) !important;
        background: var(--bg-accent) !important;
    }
    
    .stSidebar .stSelectbox label {
        color: var(--text-primary) !important;
        font-weight: 600 !important;
    }
    
    /*metrics */
    .stSidebar .stMetric {
        background: var(--bg-accent) !important;
        border: 1px solid var(--border-accent) !important;
        color: var(--text-accent) !important;
        padding: 1rem !important;
        border-radius: 8px !important;
        margin-bottom: 1rem !important;
        transition: all 0.2s ease;
    }
    
    .stSidebar .stMetric:hover {
        box-shadow: 0 2px 8px rgba(37, 99, 235, 0.1);
        transform: translateY(-1px);
    }
    
    .stSidebar .stMetric label, 
    .stSidebar .stMetric div[data-testid="metric-container"] > div {
        font-weight: 600 !important;
        color: var(--text-accent) !important;
    }
    
    /* Restaurant Cards */
    .restaurant-card {
        background: var(--bg-primary);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
        border: 1px solid var(--border-light);
        transition: all 0.2s ease;
    }
    
    .restaurant-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
        border-color: var(--border-accent);
    }
    
    .restaurant-name {
        color: var(--text-primary);
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .restaurant-details {
        color: var(--text-secondary);
        font-size: 0.9rem;
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
        margin-top: 1rem;
        padding-top: 1rem;
        border-top: 1px solid var(--border-light);
        gap: 12px;
        flex-wrap: wrap;
    }
    
    /* rating */
    .rating {
        background: var(--success-light);
        color: #000000;
        padding: 6px 12px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
        display: flex;
        align-items: center;
        gap: 4px;
        white-space: nowrap;
        border: 1px solid #a7f3d0;
    }
    
    /* cost */
    .cost {
        background: var(--primary-blue);
        color: var(--bg-primary);
        padding: 6px 12px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
        display: flex;
        align-items: center;
        gap: 4px;
        white-space: nowrap;
    }
    
    .features-row {
        display: flex;
        gap: 6px;
        margin-top: 0.75rem;
        flex-wrap: wrap;
    }
    
    /* badges */
    .feature-badge {
        background: var(--bg-tertiary);
        color: var(--text-secondary);
        border: 1px solid var(--border-light);
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    
    .feature-badge:hover {
        background: var(--bg-accent);
        color: var(--text-accent);
        border-color: var(--border-accent);
    }
    
    /* header */
    .header-container {
        text-align: center;
        padding: 2rem;
        margin-bottom: 2rem;
        background: linear-gradient(135deg, #1e293b 0%, #334155 50%, #475569 100%);
        border-radius: 12px;
        color: var(--bg-primary);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
        border: 1px solid #334155;
        position: relative;
        overflow: hidden;
    }
    
    .header-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, rgba(255, 255, 255, 0.1) 0%, transparent 50%, rgba(255, 255, 255, 0.05) 100%);
        pointer-events: none;
    }
    
    .main-title {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        color: var(--bg-primary) !important;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
        position: relative;
        z-index: 1;
    }
    
    .subtitle {
        font-size: 1rem;
        color: rgba(255, 255, 255, 0.9) !important;
        font-weight: 400;
        position: relative;
        z-index: 1;
    }
    
    /* section headers */
    .section-header {
        color: var(--text-primary) !important;
        font-size: 1.2rem;
        font-weight: 600;
        margin: 1.5rem 0 1rem 0;
        text-align: center;
        padding: 1rem;
        background: var(--bg-accent);
        border-radius: 8px;
        border: 1px solid var(--border-accent);
    }
    
    .no-results {
        text-align: center;
        color: var(--text-secondary);
        font-size: 1rem;
        padding: 2rem;
        background: var(--bg-primary);
        border-radius: 12px;
        margin: 1.5rem 0;
        border: 1px solid var(--border-light);
    }
    
    /* sidebar sections */
    .sidebar-section {
        background: var(--bg-accent);
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border: 1px solid var(--border-accent);
    }
    
    .sidebar-title {
        color: var(--text-accent) !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        margin-bottom: 0.75rem !important;
    }
    
    /* info */
    .stInfo {
        background: var(--bg-accent) !important;
        border: 1px solid var(--border-accent) !important;
        border-radius: 8px !important;
    }
    
    .stInfo > div {
        color: var(--text-accent) !important;
        font-weight: 500 !important;
    }
    
    /* Dropdown  */
    .stSelectbox > div,
    div[data-baseweb="select"] > div {
        background: var(--bg-primary) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-light) !important;
        border-radius: 6px !important;
        transition: all 0.2s ease;
    }
    
    .stSelectbox > div:hover,
    div[data-baseweb="select"] > div:hover {
        border-color: var(--border-accent) !important;
        background: var(--bg-accent) !important;
    }
    
    div[data-baseweb="select"] div[role="option"] {
        background: var(--bg-primary) !important;
        color: var(--text-primary) !important;
        padding: 10px 12px;
    }
    
    div[data-baseweb="select"] div[role="option"]:hover {
        background: var(--bg-accent) !important;
    }
    
    /* focus */
    div[data-baseweb="select"]:focus-within,
    div[data-baseweb="input"]:focus-within,
    .stSelectbox > div:focus-within,
    .stTextInput > div:focus-within {
        border-color: var(--primary-blue) !important;
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
    }
    
    input:focus,
    select:focus,
    textarea:focus {
        outline: none;
        border-color: var(--primary-blue) !important;
    }
    
    button:focus,
    input:focus,
    select:focus {
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
        outline: none;
    }
    
    /* typograohy */
    * {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
    }
    
    /*Transitions */
    .restaurant-card,
    .rating,
    .cost,
    .feature-badge,
    .sidebar-section {
        transition: all 0.2s ease;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    try:
        #load the CSV file
        df = pd.read_csv("zomato_cleaned.csv")
        
        # Data cleaning and preprocessing
        df['cuisines'] = df['cuisines'].fillna('Unknown')
        df['name'] = df['name'].fillna('Unknown Restaurant')
        df['full_address'] = df['full_address'].fillna('Address not available')
        
        # Handle ratings
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
        
        # Handle cost 
        if 'averagecost' in df.columns:
            df['cost'] = pd.to_numeric(df['averagecost'], errors='coerce').fillna(0)
        else:
            cost_columns = ['average_cost_for_two', 'cost', 'approx_cost(for two people)']
            df['cost'] = 0
            for col in cost_columns:
                if col in df.columns:
                    if df[col].dtype == 'object':
                        df['cost'] = df[col].astype(str).str.replace(',', '').str.replace('₹', '').str.replace('Rs.', '').str.replace('Rs', '').str.extract('(\d+)').astype(float).fillna(0)
                    else:
                        df['cost'] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                    break
        
        # Add service features 
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
        # restaurant index
        matches = df[df['name'].str.contains(name, case=False, na=False)]
        if matches.empty:
            return pd.DataFrame()
        
        # irst match index
        idx = matches.index[0]
        
        # ensure index is within bounds
        if idx >= len(df) or idx >= cosine_sim.shape[0]:
            st.error(f"Index error: Restaurant index {idx} is out of bounds.")
            return pd.DataFrame()
        
        # similarity scores
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        #  top similar restaurants 
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
                st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)
                
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
        
        # unique cuisines
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
                
                # Display results 
                for i in range(len(results)):
                    if i % 2 == 0:
                        cols = st.columns(2)
                    
                    with cols[i % 2]:
                        display_restaurant_card(results.iloc[i])
            else:
                st.markdown(f'<div class="no-results">No restaurants found for {cuisine_input} cuisine. Try selecting a different cuisine.</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()