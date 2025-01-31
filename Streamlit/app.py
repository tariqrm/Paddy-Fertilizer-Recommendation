import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from branca.colormap import LinearColormap
from shapely.geometry import Polygon, Point
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import psycopg2
from scipy.ndimage import gaussian_filter
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import load_model


# PostgreSQL database configuration
DB_HOST = 'localhost'
DB_NAME = 'NewDB'
DB_USER = 'postgres'
DB_PASS = '123'

def get_data_from_db(query):
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASS
        )
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Error connecting to the database: {e}")
        return pd.DataFrame()

# Load the fertilizer schedule
fertilizer_schedule_df = pd.read_csv('Fertilizer_Recommendations_Sri_Lanka.csv')

# Set the title of the Streamlit app
st.title('Soil Nutrient Data Viewer and Fertilization Recommendation')

# Get data from the PostgreSQL database
soil_data_query = "SELECT * FROM dataset"
boundary_data_query = "SELECT * FROM basestation"
df = get_data_from_db(soil_data_query)
corners_df = get_data_from_db(boundary_data_query)

# Filter to get the first longitude and latitude of each unique ID
df_filtered = df.groupby('id').first().reset_index()

# Display the map plotting all the locations from the dataset table
st.subheader('Map of Soil Sampling Locations')
if not df_filtered.empty:
    all_locations_map = folium.Map(location=[df_filtered['latitude'].mean(), df_filtered['longitude'].mean()], zoom_start=16)
    for _, row in df_filtered.iterrows():
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=f"ID: {row['id']}, Latitude: {row['latitude']}, Longitude: {row['longitude']}"
        ).add_to(all_locations_map)
    st_folium(all_locations_map, width=700, height=500)
else:
    st.write("No soil data available.")

# Display the contents of the basestation table
st.subheader('Soil readings collected')
if not corners_df.empty:
    st.dataframe(corners_df)
else:
    st.write("No boundary data available.")

# Generate a dropdown list for unique instances of ID in basestation table
unique_ids = corners_df['id'].unique()
selected_id = st.selectbox("Select Soil set reading", unique_ids)

# Filter the corners_df to get the selected ID's details
selected_id_data = corners_df[corners_df['id'] == selected_id]

# Extract coordinates for the selected ID
coordinates = None
if 'corners' in selected_id_data.columns:
    try:
        coordinates = [tuple(map(float, corner.split(','))) for corner in selected_id_data['corners'].iloc[0].split(';')]
        if coordinates[0] != coordinates[-1]:
            coordinates.append(coordinates[0])
    except Exception as e:
        st.error(f"Error parsing coordinates: {e}")
else:
    st.error("Corners column is missing in the boundary dataset.")

# Filter the soil dataset based on the selected ID from the basestation table
filtered_df = df[df['id'] == selected_id]

if not filtered_df.empty and coordinates:
    expected_columns = ['latitude', 'longitude', 'moisture', 'temperature', 'ec', 'ph', 'nitrogen', 'phosphorous', 'potassium']
    if all(column in filtered_df.columns for column in expected_columns):
        filtered_df[expected_columns] = filtered_df[expected_columns].apply(pd.to_numeric, errors='coerce')
    else:
        st.error("One or more expected columns are missing in the soil dataset.")

    st.subheader('Data Table')
    st.dataframe(filtered_df)

    st.subheader('Soil Data Insights')
    numeric_df = filtered_df.select_dtypes(include=[np.number])
    summary_df = pd.DataFrame({
        'Mean': numeric_df.mean(),
        'Max': numeric_df.max(),
        'Min': numeric_df.min(),
        'Std Dev': numeric_df.std(),
        'Count': numeric_df.count()
    })
    st.table(summary_df)

    polygon = Polygon(coordinates)
    area_sq_meters = polygon.area * (111320 ** 2)
    area_acres = area_sq_meters * 0.000247105

    st.subheader('Land Area')
    st.write(f"Area of the land: {area_sq_meters:.2f} square meters")
    st.write(f"Area of the land: {area_acres:.2f} acres")

    def get_color(value, min_val, max_val):
        colormap = LinearColormap(colors=['blue', 'lime', 'red'], vmin=min_val, vmax=max_val)
        return colormap(value)

    parameter_mapping = {
        'Moisture (%)': 'moisture',
        'Temperature (C)': 'temperature',
        'EC (us/cm)': 'ec',
        'Ph': 'ph',
        'Nitrogen (mg/kg)': 'nitrogen',
        'Phosphorous (mg/kg)': 'phosphorous',
        'Potassium (mg/kg)': 'potassium'
    }

    parameter_display = st.radio(
        "Select parameter to visualize on the map:",
        list(parameter_mapping.keys())
    )

    parameter = parameter_mapping[parameter_display]

    min_val = filtered_df[parameter].min()
    max_val = filtered_df[parameter].max()

    m = folium.Map(location=[coordinates[0][0], coordinates[0][1]], zoom_start=17)
    for _, row in filtered_df.iterrows():
        color = get_color(row[parameter], min_val, max_val)
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=8,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=f"{parameter_display}: {row[parameter]}"
        ).add_to(m)

    folium.PolyLine(locations=coordinates, color='blue', weight=2.5, opacity=1).add_to(m)

    st.subheader('Nutrient Distribution Profile')
    min_lat, min_lon, max_lat, max_lon = polygon.bounds
    grid_x, grid_y = np.mgrid[min_lat:max_lat:100j, min_lon:max_lon:100j]

    def idw(x, y, z, xi, yi, power=2):
        dist = np.sqrt((x[:, None, None] - xi[None, :, :])**2 + (y[:, None, None] - yi[None, :, :])**2)
        weights = 1 / (dist**power)
        weights[dist == 0] = np.inf
        z_idw = np.sum(weights * z[:, None, None], axis=0) / np.sum(weights, axis=0)
        return z_idw

    x = filtered_df['latitude'].values
    y = filtered_df['longitude'].values
    z = filtered_df[parameter].values

    gaussian_filter_option = st.radio("Apply Gaussian Filter:", ["No", "Yes"])

    if gaussian_filter_option == "Yes":
        z = gaussian_filter(z, sigma=1)

    grid_z_idw = idw(x, y, z, grid_x, grid_y)

    mask = np.array([Point(x, y).within(polygon) for x, y in zip(grid_x.flatten(), grid_y.flatten())])
    grid_z_idw = np.ma.masked_array(grid_z_idw, ~mask.reshape(grid_z_idw.shape))

    fig, ax = plt.subplots()
    cmap = LinearSegmentedColormap.from_list("mycmap", ['blue', 'lime', 'red'])
    contour = ax.contourf(grid_y, grid_x, grid_z_idw, cmap=cmap, levels=100)
    plt.colorbar(contour, ax=ax, label=parameter_display)
    ax.set_title(f'{parameter_display} Distribution')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_aspect('equal', 'box')
    st.pyplot(fig)

    st.subheader('Map View')
    st_folium(m, width=700, height=500)

    st.subheader('Fertilizer Recommendation')
    zone = st.selectbox("Select your zone:", fertilizer_schedule_df['Zone Type'].dropna().unique())
    condition = st.selectbox("Select the condition of your field:", fertilizer_schedule_df['Condition'].dropna().unique())
    age_group = st.selectbox("Select the age group of your crop:", fertilizer_schedule_df['Age Group'].dropna().unique())
    time_period_options = fertilizer_schedule_df[
        (fertilizer_schedule_df['Zone Type'] == zone) &
        (fertilizer_schedule_df['Condition'] == condition) &
        (fertilizer_schedule_df['Age Group'] == age_group)
    ]['Time'].unique()
    
    time_period = st.selectbox("Select the time period:", time_period_options)

    filtered_schedule = fertilizer_schedule_df[
        (fertilizer_schedule_df['Zone Type'] == zone) &
        (fertilizer_schedule_df['Condition'] == condition) &
        (fertilizer_schedule_df['Age Group'] == age_group) &
        (fertilizer_schedule_df['Time'] == time_period)
    ]

    if not filtered_schedule.empty:
        urea_required = filtered_schedule['Urea (Kg/Ha)'].values[0]
        tsp_required = filtered_schedule['T.S.P (Kg/Ha)'].values[0]
        mop_required = filtered_schedule['M.O.P (Kg/Ha)'].values[0]
        zinc_sulphate_required = filtered_schedule['Zinc Sulphate (Kg/Ha)'].values[0]

        st.write(f"Recommended amounts of fertilizers for {age_group} crops during {time_period} in {zone} zone with {condition} condition:")
        st.write(f"Urea: {urea_required} kg/ha")
        st.write(f"T.S.P: {tsp_required} kg/ha")
        st.write(f"M.O.P: {mop_required} kg/ha")
        st.write(f"Zinc Sulphate: {zinc_sulphate_required} kg/ha")

        def calculate_fertilizer(soil_nutrient, nutrient_threshold, fertilizer_pct):
            deficit = nutrient_threshold - soil_nutrient
            required_fertilizer = max(deficit / fertilizer_pct, 0)
            return required_fertilizer

        nitrogen_threshold = urea_required * 0.46
        phosphorous_threshold = tsp_required * 0.20
        potassium_threshold = mop_required * 0.50

        soil_nitrogen = filtered_df['nitrogen'].mean()
        soil_phosphorous = filtered_df['phosphorous'].mean()
        soil_potassium = filtered_df['potassium'].mean()

        urea_needed = calculate_fertilizer(soil_nitrogen, nitrogen_threshold, 0.46)
        tsp_needed = calculate_fertilizer(soil_phosphorous, phosphorous_threshold, 0.20)
        mop_needed = calculate_fertilizer(soil_potassium, potassium_threshold, 0.50)

        st.write(f"Additional Urea needed: {urea_needed:.2f} kg/ha")
        st.write(f"Additional T.S.P needed: {tsp_needed:.2f} kg/ha")
        st.write(f"Additional M.O.P needed: {mop_needed:.2f} kg/ha")

        st.subheader('Fertilizer Application Instructions')
        st.write(f"1. **Urea Application**: Apply {urea_needed:.2f} kg/ha of Urea evenly across the field. If possible, split the application into multiple sessions for better absorption and to minimize nitrogen loss.")
        st.write(f"2. **T.S.P Application**: Apply {tsp_needed:.2f} kg/ha of T.S.P (Triple Super Phosphate) to supply the necessary phosphorous. Incorporate the T.S.P into the soil at the root zone to enhance phosphorous availability to the plants.")
        st.write(f"3. **M.O.P Application**: Apply {mop_needed:.2f} kg/ha of M.O.P (Muriate of Potash) evenly across the field to provide potassium. Ensure the application is well-distributed and consider using it during the early growth stages for optimal results.")
    else:
        st.write("No data available for the selected filters.")

# Subheader for Paddy Suitability Estimation
st.subheader('Paddy Suitability Estimation')

# Assuming NPK, pH, and temperature values are obtained from the database
if 'filtered_df' in locals() and not filtered_df.empty:
    # Load the Neural Network model, scaler
    with open("D:/Projects/CDP/DataAnalysis/crop type/scaler.pkl", "rb") as file:
        paddy_scaler = pickle.load(file)

    # Load the Keras model
    paddy_model = load_model('D:/Projects/CDP/DataAnalysis/crop type/model.h5')

    nitrogen = st.number_input("Nitrogen:", value=filtered_df['nitrogen'].mean())
    phosphorus = st.number_input("Phosphorus:", value=filtered_df['phosphorous'].mean())
    potassium = st.number_input("Potassium:", value=filtered_df['potassium'].mean())
    temperature = st.number_input("Temperature:", value=filtered_df['temperature'].mean())
    humidity = st.number_input("Enter the humidity (%):", value=80.0)
    ph = st.number_input("pH:", value=filtered_df['ph'].mean())
    rainfall = st.number_input("Enter the rainfall (mm):", value=200.0)

    # Prepare input data for Neural Network model
    paddy_input_data = [[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]]
    paddy_input_data_scaled = paddy_scaler.transform(paddy_input_data)

    # Predict suitability of rice
    paddy_prediction = paddy_model.predict(paddy_input_data_scaled)[0][0]

    # Convert prediction to percentage
    paddy_suitability_score = paddy_prediction * 100

    # Display the suitability score in a colorful manner
    st.markdown(
        f'<p style="font-size:24px; color:{"green" if paddy_suitability_score > 80 else "red"};">'
        f'Neural Network Suitability Score: {paddy_suitability_score:.2f} %</p>',
        unsafe_allow_html=True
    )
else:
    st.write("No soil data available to compute rice suitability.")

# Subheader for Fertilizer Type Prediction
st.subheader('Fertilizer Type Prediction')

# Load the necessary files for fertilizer type prediction
with open("D:/Projects/CDP/DataAnalysis/fertilizer type/scaler.pkl", "rb") as file:
    fertilizer_scaler = pickle.load(file)

with open("D:/Projects/CDP/DataAnalysis/fertilizer type/label_encoder_soil.pkl", "rb") as file:
    label_encoder_soil = pickle.load(file)

with open("D:/Projects/CDP/DataAnalysis/fertilizer type/label_encoder_fertilizer.pkl", "rb") as file:
    label_encoder_fertilizer = pickle.load(file)

# Load the Keras model for fertilizer type prediction
fertilizer_model = load_model('D:/Projects/CDP/DataAnalysis/fertilizer type/model.h5')

# User inputs for fertilizer prediction
soil_type = st.selectbox("Soil Type", options=label_encoder_soil.classes_)
soil_type_encoded = label_encoder_soil.transform([soil_type])[0]
moisture = st.number_input("Moisture", value=30.0)

# Use the same values for other features from the previous section, excluding pH
fertilizer_input_data = [[soil_type_encoded, nitrogen, phosphorus, potassium, temperature, humidity, moisture]]
fertilizer_input_data_scaled = fertilizer_scaler.transform(fertilizer_input_data)

# Predict fertilizer type
fertilizer_prediction = fertilizer_model.predict(fertilizer_input_data_scaled)
predicted_fertilizer_class = np.argmax(fertilizer_prediction)
predicted_fertilizer = label_encoder_fertilizer.inverse_transform([predicted_fertilizer_class])[0]

# Display the predicted fertilizer type
st.markdown(
    f'<p style="font-size:24px;">'
    f'Suitable Fertilizer Type: {predicted_fertilizer}</p>',
    unsafe_allow_html=True
)


# Subheader for Fertilizer Amount Prediction
st.subheader('Fertilizer Amount Prediction')

# Load the necessary files for fertilizer amount prediction
with open("D:/Projects/CDP/DataAnalysis/yield/scaler.pkl", "rb") as file:
    yield_scaler = pickle.load(file)
with open("D:/Projects/CDP/DataAnalysis/yield/y_scaler.pkl", "rb") as file:
    y_scaler = pickle.load(file)
with open("D:/Projects/CDP/DataAnalysis/yield/poly_features.pkl", "rb") as file:
    poly = pickle.load(file)
yield_model = load_model("D:/Projects/CDP/DataAnalysis/yield/model.h5")

# Use the same values for Area, Production, and Annual_Rainfall

production = st.number_input("Production", value=398311)

area_ha = area_acres / 2.47105
# Prepare input data for fertilizer amount prediction
new_data = pd.DataFrame({
    'Area': [area_ha],
    'Production': [production],
    'Annual_Rainfall': [rainfall]
})

# Add polynomial features
new_data_poly = poly.transform(new_data)

# Standardize the features
new_data_scaled = yield_scaler.transform(new_data_poly)

# Predict the fertilizer amount
y_pred_scaled = yield_model.predict(new_data_scaled)
y_pred = y_scaler.inverse_transform(y_pred_scaled).flatten()
y_pred = y_pred / 1_000_000 * area_ha

# Display the predicted fertilizer amount
st.markdown(
    f'<p style="font-size:24px;">'
    f'Recommended Fertilizer Amount (kg): {y_pred[0]:.2f}</p>',
    unsafe_allow_html=True
)