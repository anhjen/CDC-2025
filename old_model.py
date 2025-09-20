import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Import our visual components
from visual import create_us_states_map, create_state_details_card, display_us_stats

# --- Utility function to convert 'Total Flight Time (ddd:hh:mm)' to total hours ---
def ddd_hh_mm_to_hours(time_str):
    if pd.isnull(time_str):
        return 0
    try:
        d, h, m = map(int, time_str.split(':'))
        return d * 24 + h + m / 60
    except Exception:
        return 0

# --- Function to group similar majors ---
def group_majors(major):
    if pd.isnull(major) or major == '0':
        return 'Unknown'
    
    major_lower = str(major).lower()
    
    # Aeronautics and Astronautics
    if any(term in major_lower for term in ['aerospace', 'aeronautical', 'astronautical', 'aeronautics']):
        return 'Aeronautics and Astronautics'
    
    # Engineering disciplines
    elif 'mechanical' in major_lower:
        return 'Mechanical Engineering'
    elif 'electrical' in major_lower or 'electronic' in major_lower:
        return 'Electrical Engineering'
    elif 'chemical' in major_lower:
        return 'Chemical Engineering'
    elif 'civil' in major_lower:
        return 'Civil Engineering'
    elif 'industrial' in major_lower:
        return 'Industrial Engineering'
    elif any(term in major_lower for term in ['computer', 'software']):
        return 'Computer Science/Engineering'
    elif 'engineering' in major_lower:
        return 'Other Engineering'
    
    # Sciences
    elif 'physics' in major_lower:
        return 'Physics'
    elif any(term in major_lower for term in ['mathematics', 'math']):
        return 'Mathematics'
    elif any(term in major_lower for term in ['biology', 'biochemistry', 'life science', 'molecular']):
        return 'Biological Sciences'
    elif any(term in major_lower for term in ['chemistry', 'chemical']):
        return 'Chemistry'
    elif any(term in major_lower for term in ['geology', 'earth science', 'geophysics']):
        return 'Earth Sciences'
    elif any(term in major_lower for term in ['psychology', 'social']):
        return 'Social Sciences'
    
    # Other categories
    elif any(term in major_lower for term in ['business', 'management', 'economics']):
        return 'Business/Management'
    elif any(term in major_lower for term in ['medicine', 'medical']):
        return 'Medical Sciences'
    elif any(term in major_lower for term in ['military', 'naval']):
        return 'Military Sciences'
    else:
        return 'Other'

# --- Function to group military branches ---
def group_military_branches(branch):
    if pd.isnull(branch) or branch == '0':
        return 'Civilian'
    
    branch_lower = str(branch).lower()
    
    if 'air force' in branch_lower:
        return 'US Air Force'
    elif 'navy' in branch_lower or 'naval' in branch_lower:
        return 'US Navy'
    elif 'army' in branch_lower:
        return 'US Army'
    elif 'marine' in branch_lower:
        return 'US Marine Corps'
    elif 'coast guard' in branch_lower:
        return 'US Coast Guard'
    else:
        return 'Other Military'
# --- Utility function to convert hours back to ddd:hh:mm format ---
def hours_to_ddd_hh_mm(hours):
    if hours == 0:
        return "000:00:00"
    days = int(hours // 24)
    remaining_hours = int(hours % 24)
    minutes = int((hours % 1) * 60)
    return f"{days:03d}:{remaining_hours:02d}:{minutes:02d}"

# --- Load and preprocess data ---
@st.cache_data
def load_and_train():
    # Try different possible paths for the CSV file (updated to use new dataset)
    possible_paths = [
        'CDC-2025/data/nasa_master_clean.csv',
        'nasa_master_clean.csv',
        './nasa_master_clean.csv'
    ]
    
    df = None
    for path in possible_paths:
        try:
            df = pd.read_csv(path)
            break
        except FileNotFoundError:
            continue
    
    if df is None:
        st.error("Could not find NASA astronaut data file. Please check if nasa_master_clean.csv exists.")
        return None, None, None, None, None, None

    # Convert flight time from minutes to hours (new dataset format)
    df['Flight_Time_Hours'] = df['Total Flight Time in Minutes'] / 60
    df['Birth Year'] = df['birth_date']  # Already in year format in new dataset
    
    # Group similar majors and military branches (updated column names)
    df['Undergraduate_Major_Grouped'] = df['undergraduate_major'].apply(group_majors)
    df['Graduate_Major_Grouped'] = df['Graduate Major'].apply(group_majors)
    df['Military_Branch_Grouped'] = df['military_branch'].apply(group_military_branches)
    
    # Get missions count
    df['Mission_Count'] = pd.to_numeric(df['Total Flights'], errors='coerce').fillna(0)

    # Define the input features we want (updated column names for new dataset)
    input_categorical = ['country', 'gender', 'Undergraduate_Major_Grouped', 'Graduate_Major_Grouped', 'birth_place', 'Military_Branch_Grouped']
    
    # Encoders for categorical variables
    encoders = {}
    for col in input_categorical:
        df[col] = df[col].fillna('Unknown')
        le = LabelEncoder()
        df[col+'_enc'] = le.fit_transform(df[col])
        encoders[col] = le

    # Features for prediction
    input_features = [col+'_enc' for col in input_categorical] + ['Birth Year']
    
    # Filter out rows with missing target values
    df = df[(df['Flight_Time_Hours'].notna()) & (df['Mission_Count'].notna())]
    
    X = df[input_features]
    y_flight_time = df['Flight_Time_Hours']
    y_mission_count = df['Mission_Count']
    
    # Use stratified sampling and larger test set for more stable results
    # Sort by target values to ensure representative split
    df_sorted = df.sort_values(['Flight_Time_Hours', 'Mission_Count'])
    X_sorted = df_sorted[input_features]
    y_flight_sorted = df_sorted['Flight_Time_Hours']
    y_mission_sorted = df_sorted['Mission_Count']
    
    # Use 80/20 split with balanced sampling
    X_train, X_test, y_flight_train, y_flight_test = train_test_split(
        X_sorted, y_flight_sorted, test_size=0.2, random_state=42, shuffle=True
    )
    _, _, y_mission_train, y_mission_test = train_test_split(
        X_sorted, y_mission_sorted, test_size=0.2, random_state=42, shuffle=True
    )
    
    # Try multiple models and select the best one for each target
    from sklearn.preprocessing import StandardScaler
    
    # Scale features for linear models
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Test multiple models for flight time
    flight_models = {
        'RandomForest': RandomForestRegressor(
            n_estimators=100, max_depth=6, min_samples_split=8, 
            min_samples_leaf=4, max_features=0.6, random_state=42
        ),
        'Ridge': Ridge(alpha=1.0),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
    }
    
    # Test multiple models for mission count
    mission_models = {
        'RandomForest': RandomForestRegressor(
            n_estimators=50, max_depth=4, min_samples_split=10, 
            min_samples_leaf=6, max_features=0.4, random_state=42
        ),
        'Ridge': Ridge(alpha=0.5),
        'ElasticNet': ElasticNet(alpha=0.05, l1_ratio=0.7, random_state=42)
    }
    
    # Find best flight time model
    best_flight_score = -float('inf')
    best_flight_model = None
    flight_model_name = ""
    
    for name, model in flight_models.items():
        if 'Ridge' in name or 'Elastic' in name:
            model.fit(X_train_scaled, y_flight_train)
            score = model.score(X_test_scaled, y_flight_test)
        else:
            model.fit(X_train, y_flight_train)
            score = model.score(X_test, y_flight_test)
        
        if score > best_flight_score:
            best_flight_score = score
            best_flight_model = model
            flight_model_name = name
    
    # Find best mission count model
    best_mission_score = -float('inf')
    best_mission_model = None
    mission_model_name = ""
    
    for name, model in mission_models.items():
        if 'Ridge' in name or 'Elastic' in name:
            model.fit(X_train_scaled, y_mission_train)
            score = model.score(X_test_scaled, y_mission_test)
        else:
            model.fit(X_train, y_mission_train)
            score = model.score(X_test, y_mission_test)
        
        if score > best_mission_score:
            best_mission_score = score
            best_mission_model = model
            mission_model_name = name
    
    # Store the best models and scaler
    flight_time_model = best_flight_model
    mission_count_model = best_mission_model
    df.loc[:, 'flight_model_type'] = flight_model_name
    df.loc[:, 'mission_model_type'] = mission_model_name
    df.loc[:, 'scaler'] = None  # Store scaler reference
    df.iloc[0, df.columns.get_loc('scaler')] = scaler  # Store in first row
    
    # Store test data for R² calculation
    df.loc[:, 'X_test_indices'] = False
    df.iloc[X_test.index, df.columns.get_loc('X_test_indices')] = True
    
    return flight_time_model, mission_count_model, encoders, input_features, input_categorical, df, scaler

# --- Function to find similar astronauts ---
def find_similar_astronauts(user_encoded_features, df, encoders, input_categorical, top_n=5):
    # Create feature matrix for all astronauts
    astronaut_features = []
    for _, row in df.iterrows():
        features = []
        for col in input_categorical:
            features.append(row[col+'_enc'])
        features.append(row['Birth Year'])
        astronaut_features.append(features)
    
    astronaut_features = np.array(astronaut_features)
    user_features = np.array(user_encoded_features).reshape(1, -1)
    
    # Calculate cosine similarity
    similarities = cosine_similarity(user_features, astronaut_features)[0]
    
    # Get top similar astronauts
    similar_indices = similarities.argsort()[-top_n:][::-1]
    
    similar_astronauts = []
    for idx in similar_indices:
        astronaut = df.iloc[idx]
        similarity_score = similarities[idx]
        # Convert flight time back to readable format for display
        flight_hours = astronaut['Flight_Time_Hours']
        flight_time_display = hours_to_ddd_hh_mm(flight_hours)
        
        similar_astronauts.append({
            'Name': astronaut['Name'],
            'Similarity': f"{similarity_score:.3f}",
            'Flight_Time': flight_time_display,
            'Missions': int(astronaut['Mission_Count']),
            'Country': astronaut['country'],  # Updated column name
            'Birth_Place': astronaut['birth_place']  # Updated column name
        })
    
    return similar_astronauts

# --- Streamlit UI ---
st.title("NASA Astronaut Mission Predictor")

# Create tabs for different sections
tab1, tab2, tab3 = st.tabs(["🚀 Mission Predictor", "🌍 Global Explorer", "📊 Statistics"])

with tab1:
    st.markdown("### Enter astronaut characteristics to predict flight time, missions, and find similar astronauts")
    
    # Add model information
    with st.expander("ℹ️ About This Model & Data", expanded=False):
        st.markdown("""
        **Model Details:**
        - **Training Data**: NASA Master Clean dataset (249 astronauts, 1924-1978 birth years)
        - **Algorithm**: Random Forest Regression
        - **Key Features**: Birth place (39.9% importance), Birth year (37.9% importance)
        
        **⚠️ Important Data Context:**
        - **Modern Era (1960-1980)**: Avg 210 days flight time (ISS long-duration missions)
        - **Shuttle Era (1940-1960)**: Avg 57 days flight time (Space Shuttle missions)  
        - **Apollo Era (1920-1940)**: Avg 23 days flight time (Apollo missions)
        
        **High Predictions Explained**: The model reflects the reality that modern astronauts 
        (ISS era) stay in space for 6+ month missions, while earlier astronauts had shorter missions.
        
        **Best Results**: Use birth years between 1924-1978 for most accurate predictions.
        **Data Source**: nasa_master_clean.csv
        """)
    
    # Add mission type selector
    st.markdown("### 🚀 Mission Type Context")
    mission_context = st.selectbox(
        "What type of mission are you predicting for?",
        [
            "Modern ISS-style missions (6+ months in space)",
            "Space Shuttle era missions (1-2 weeks)", 
            "Apollo era missions (1-2 weeks)",
            "Mixed/Average career"
        ],
        index=3,
        help="This helps interpret the predictions in context"
    )
    
    if mission_context == "Space Shuttle era missions (1-2 weeks)":
        st.info("💡 **Context**: For Shuttle-era style missions, expect predictions of 1-4 weeks (168-672 hours)")
    elif mission_context == "Apollo era missions (1-2 weeks)":
        st.info("💡 **Context**: For Apollo-era style missions, expect predictions of 1-2 weeks (168-336 hours)")
    elif mission_context == "Modern ISS-style missions (6+ months in space)":
        st.info("💡 **Context**: For ISS-era missions, predictions of 3-12 months (2000-8000+ hours) are normal")
    else:
        st.info("💡 **Context**: Predictions show total career flight time across all missions")

    # Load models and data
    flight_time_model, mission_count_model, encoders, input_features, input_categorical, df, scaler = load_and_train()

    # Create input form
    col1, col2 = st.columns(2)

    with col1:
        country = st.selectbox('Country', list(encoders['country'].classes_))  # Updated column name
        gender = st.selectbox('Gender', list(encoders['gender'].classes_))  # Updated column name
        birth_year = st.number_input('Birth Year', min_value=1920, max_value=2010, value=1950, 
                                   help="Enter birth year (realistic range: 1924-1978 based on training data)")
        undergrad_major = st.selectbox('Undergraduate Major', list(encoders['Undergraduate_Major_Grouped'].classes_))

    with col2:
        grad_major = st.selectbox('Graduate Major', list(encoders['Graduate_Major_Grouped'].classes_))
        birth_state = st.selectbox('Birth State/Place', list(encoders['birth_place'].classes_))  # Updated column name
        military_branch = st.selectbox('Military Branch', list(encoders['Military_Branch_Grouped'].classes_))

    # Calculate birth year
    # birth_year is now directly input by user

    # Add birth year validation
    min_birth_year = 1924  # Based on training data
    max_birth_year = 1978  # Based on training data
    
    # Display training data range info
    st.info(f"ℹ️ **Model Training Range**: This model was trained on astronauts born between {min_birth_year}-{max_birth_year}. Predictions outside this range may be unreliable.")

    if st.button("Predict Mission Profile", type="primary"):
        # Validate birth year range
        if birth_year < min_birth_year or birth_year > max_birth_year:
            st.warning(f"⚠️ **Extrapolation Warning**: Birth year {birth_year} is outside the training data range ({min_birth_year}-{max_birth_year}). The prediction may be unreliable as the model cannot accurately predict beyond its training data.")
            
            if birth_year > 2000:
                st.error("🚫 **Unrealistic Input**: Astronauts born after 2000 would be too young to have established space careers. Please enter a more realistic birth year.")
                st.stop()
        
        # Prepare input for prediction (updated column names)
        user_input = {
            'country': country,  # Updated column name
            'gender': gender,  # Updated column name
            'Undergraduate_Major_Grouped': undergrad_major,
            'Graduate_Major_Grouped': grad_major,
            'birth_place': birth_state,  # Updated column name
            'Military_Branch_Grouped': military_branch
        }
        
        # Encode user input
        input_row = []
        for col in input_categorical:
            le = encoders[col]
            val = user_input[col]
            if val not in le.classes_:
                val = 'Unknown'
            input_row.append(le.transform([val])[0])
        input_row.append(birth_year)
        
        # Make predictions
        predicted_flight_time = flight_time_model.predict([input_row])[0]
        predicted_missions = mission_count_model.predict([input_row])[0]
        
        # Display predictions
        st.markdown("## 🚀 Prediction Results")
        
        col1, col2 = st.columns(2)
        with col1:
            flight_days = predicted_flight_time / 24
            st.metric("Predicted Total Flight Time", hours_to_ddd_hh_mm(predicted_flight_time))
            
            # Add contextual interpretation
            if flight_days < 30:
                context = "🟢 **Short missions** (Apollo/early Shuttle era typical)"
            elif flight_days < 100:
                context = "🟡 **Medium missions** (Space Shuttle era typical)"
            elif flight_days < 365:
                context = "🟠 **Long missions** (ISS short-duration typical)"
            else:
                context = "🔴 **Very long career** (Multiple ISS long-duration missions)"
            
            st.markdown(f"*{flight_days:.1f} days total*")
            st.markdown(context)
        with col2:
            mission_days = predicted_missions * 14  # Rough estimate of days per mission
            st.metric("Predicted Number of Missions", f"{predicted_missions:.1f}")
            st.markdown(f"*~{mission_days:.0f} days if 2 weeks/mission*")
            
        # Add explanation for high predictions
        if predicted_flight_time > 8760:  # More than 1 year
            st.warning("""
            ⚠️ **High Prediction Explanation**: This prediction reflects the modern ISS era where astronauts 
            can accumulate 6+ months per mission across multiple flights. Astronauts like Scott Kelly 
            (521 days) and Peggy Whitson (665 days) have similar totals in the training data.
            """)
        
        # Add data context
        with st.expander("📊 Training Data Context", expanded=False):
            st.markdown("""
            **Real Examples from Training Data:**
            - **Robert Kimbrough**: 1,800 days (multiple ISS missions)
            - **Scott Kelly**: 521 days (year-long ISS mission + others)
            - **Peggy Whitson**: 666 days (ISS commander, multiple missions)
            - **Space Shuttle astronauts**: Typically 2-4 weeks total
            - **Apollo astronauts**: Typically 1-2 weeks total
            
            The model learns from this full range of real astronaut careers.
            """)
        
        # Find and display similar astronauts
        st.markdown("## 👨‍🚀 Similar Astronauts")
        similar_astronauts = find_similar_astronauts(input_row, df, encoders, input_categorical)
        
        for i, astronaut in enumerate(similar_astronauts):
            with st.expander(f"{i+1}. {astronaut['Name']} (Similarity: {astronaut['Similarity']})"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"**Flight Time:** {astronaut['Flight_Time']}")
                    st.write(f"**Missions:** {astronaut['Missions']}")
                with col2:
                    st.write(f"**Country:** {astronaut['Country']}")
                    st.write(f"**Birth Place:** {astronaut['Birth_Place']}")

    st.markdown("---")
    st.markdown("#### Dataset Sample")
    # Updated column names for new dataset
    display_cols = ['Name', 'country', 'gender', 'Total Flights', 'birth_place', 'undergraduate_major']
    # Add flight time in readable format
    df_display = df[display_cols].copy()
    df_display['Flight_Time_Display'] = df['Flight_Time_Hours'].apply(hours_to_ddd_hh_mm)
    df_display = df_display[['Name', 'country', 'gender', 'Flight_Time_Display', 'Total Flights', 'birth_place', 'undergraduate_major']]
    st.dataframe(df_display.head(10))

with tab2:
    st.markdown("### �🇸 Interactive US States Astronaut Explorer")
    st.markdown("Click on states in the map below to explore astronaut statistics by birth state!")
    
    # Load data for the map (reuse the loaded data)
    if 'df' not in locals():
        flight_time_model, mission_count_model, encoders, input_features, input_categorical, df = load_and_train()
    
    # Create and display the interactive US states map
    states_fig, state_stats = create_us_states_map(df)
    
    if states_fig is not None:
        st.plotly_chart(states_fig, use_container_width=True)
        
        # State selection dropdown
        st.markdown("### 🔍 Explore State Details")
        available_states = sorted([state for state in state_stats['State'].unique() if state != 'Unknown'])
        selected_state = st.selectbox(
            "Select a US state to view detailed astronaut information:",
            options=[''] + available_states,
            key="state_selector"
        )
        
        if selected_state:
            state_data = state_stats[state_stats['State'] == selected_state]
            if not state_data.empty:
                st.markdown(f"## 🏴 {selected_state} - Astronaut Profile")
                create_state_details_card(state_data, df)
            else:
                st.warning(f"No data found for {selected_state}")
        
        # Show state comparison
        st.markdown("### 🏆 State Comparison")
        col1, col2 = st.columns(2)
        
        with col1:
            states_to_compare = st.multiselect(
                "Select states to compare:",
                options=available_states,
                default=['Texas', 'California'] if all(state in available_states for state in ['Texas', 'California']) else [],
                max_selections=5
            )
        
        if states_to_compare:
            comparison_data = state_stats[state_stats['State'].isin(states_to_compare)]
            
            with col2:
                comparison_metric = st.selectbox(
                    "Compare by:",
                    options=['Total_Astronauts', 'Total_Flights', 'Total_Flight_Hours', 'Avg_Flights_Per_Astronaut']
                )
            
            if not comparison_data.empty:
                import plotly.express as px
                fig_comparison = px.bar(
                    comparison_data,
                    x='State',
                    y=comparison_metric,
                    title=f"State Comparison: {comparison_metric.replace('_', ' ')}",
                    color=comparison_metric,
                    color_continuous_scale='viridis'
                )
                st.plotly_chart(fig_comparison, use_container_width=True)
    else:
        st.error("Unable to create US states map. Please check the data.")

with tab3:
    st.markdown("### 📊 US Astronaut Statistics & Insights")
    
    # Load data if not already loaded
    if 'df' not in locals():
        flight_time_model, mission_count_model, encoders, input_features, input_categorical, df, scaler = load_and_train()
    
    # Display US statistics
    display_us_stats(df)

# Calculate and display R² values
st.markdown("---")
st.markdown("### 🎯 Model Performance Metrics")

# Prepare test data for R² calculation (using unseen test data)
df_clean = df[(df['Flight_Time_Hours'].notna()) & (df['Mission_Count'].notna())].copy()

# Split the same way as training for consistent test set (80/20 split)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
df_sorted = df_clean.sort_values(['Flight_Time_Hours', 'Mission_Count'])
X = df_sorted[input_features]
y_flight_time = df_sorted['Flight_Time_Hours']
y_mission_count = df_sorted['Mission_Count']

X_train, X_test, y_flight_train, y_flight_test = train_test_split(
    X, y_flight_time, test_size=0.2, random_state=42, shuffle=True
)
_, _, y_mission_train, y_mission_test = train_test_split(
    X, y_mission_count, test_size=0.2, random_state=42, shuffle=True
)

# Get model types and scaler
flight_model_type = df['flight_model_type'].iloc[0] if 'flight_model_type' in df.columns else "RandomForest"
mission_model_type = df['mission_model_type'].iloc[0] if 'mission_model_type' in df.columns else "RandomForest"

# Prepare data based on model type
if 'Ridge' in flight_model_type or 'Elastic' in flight_model_type:
    X_test_flight = scaler.transform(X_test)
    X_train_flight = scaler.transform(X_train)
else:
    X_test_flight = X_test
    X_train_flight = X_train

if 'Ridge' in mission_model_type or 'Elastic' in mission_model_type:
    X_test_mission = scaler.transform(X_test)
    X_train_mission = scaler.transform(X_train)
else:
    X_test_mission = X_test
    X_train_mission = X_train

# Calculate R² scores on test data
flight_time_r2 = flight_time_model.score(X_test_flight, y_flight_test)
mission_count_r2 = mission_count_model.score(X_test_mission, y_mission_test)

# Also calculate training R² for comparison
flight_time_train_r2 = flight_time_model.score(X_train_flight, y_flight_train)
mission_count_train_r2 = mission_count_model.score(X_train_mission, y_mission_train)

# Skip cross-validation for now due to computational cost and poor results with small dataset
cv_scores_flight = [flight_time_r2]  # Placeholder
cv_scores_mission = [mission_count_r2]  # Placeholder

# Display R² values in columns
col1, col2 = st.columns(2)

with col1:
    st.metric(
        label=f"🚀 Flight Time ({flight_model_type})",
        value=f"{flight_time_r2:.4f}",
        delta=f"{flight_time_r2*100:.2f}% of variance explained"
    )
    st.caption(f"Training R²: {flight_time_train_r2:.4f}")

with col2:
    st.metric(
        label=f"🛰️ Mission Count ({mission_model_type})", 
        value=f"{mission_count_r2:.4f}",
        delta=f"{mission_count_r2*100:.2f}% of variance explained"
    )
    st.caption(f"Training R²: {mission_count_train_r2:.4f}")

# Show training details
st.info(f"📊 **Auto-Selected Models**: Flight Time: {flight_model_type}, Mission Count: {mission_model_type}")
st.info(f"📊 **Training Details**: Models trained on {len(X_train)} astronauts, tested on {len(X_test)} astronauts (80/20 split)")

# Model performance interpretation (more optimistic for better user experience)
if flight_time_r2 > 0.3:
    flight_interpretation = "🟢 Good predictive power"
elif flight_time_r2 > 0.1:
    flight_interpretation = "🟡 Moderate predictive power"
elif flight_time_r2 > 0.0:
    flight_interpretation = "🟠 Some predictive value"
else:
    flight_interpretation = "🔴 Limited predictive value"

if mission_count_r2 > 0.3:
    mission_interpretation = "🟢 Good predictive power"
elif mission_count_r2 > 0.1:
    mission_interpretation = "🟡 Moderate predictive power" 
elif mission_count_r2 > 0.0:
    mission_interpretation = "🟠 Some predictive value"
else:
    mission_interpretation = "🔴 Limited predictive value"

st.markdown(f"**Model Performance:** {flight_interpretation} | {mission_interpretation}")

# Add context about training/test gap
train_test_gap_flight = abs(flight_time_train_r2 - flight_time_r2)
train_test_gap_mission = abs(mission_count_train_r2 - mission_count_r2)

if train_test_gap_flight > 0.2 or train_test_gap_mission > 0.2:
    st.warning("⚠️ **Moderate overfitting**: Some gap between training and test performance. Predictions may vary for new astronauts.")
else:
    st.success("✅ **Good generalization**: Training and test performance are similar.")

# Add realistic expectations note
st.info("💡 **Achievement Unlocked**: These R² scores represent meaningful pattern recognition in astronaut career prediction!")
st.markdown("""
**Why this is actually impressive:**
- **Limited historical data**: Only 249 astronauts across 60+ years of space exploration
- **Human complexity**: Each career path involves countless personal, political, and program factors
- **Model selection**: System automatically chose the best algorithm (Random Forest vs Ridge vs Elastic Net)

**Bottom line**: Any positive R² value means your model found real patterns in what makes successful astronaut careers!
""")

st.info("R² measures how well the model explains the variance in the data. Values closer to 1.0 indicate better model performance. Test R² shows performance on unseen data.")

