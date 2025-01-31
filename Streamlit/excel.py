import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from branca.colormap import LinearColormap
from shapely.geometry import Polygon, Point
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Load the fertilizer schedule
fertilizer_schedule_df = pd.read_csv('Fertilizer_Schedule.csv')

# Set the title of the Streamlit app
st.title('Upload custom data')

# File uploader for Excel files
uploaded_file = st.file_uploader("Choose an Excel file for soil data", type=["xlsx", "xls"])
corners_file = st.file_uploader("Choose an Excel file for land boundaries", type=["xlsx", "xls"])

if uploaded_file is not None and corners_file is not None:
    # Read the uploaded files into DataFrames
    df = pd.read_excel(uploaded_file)
    corners_df = pd.read_excel(corners_file)

    # Display the data in a table
    st.subheader('Data Table')
    st.dataframe(df)

    # Provide detailed insights on the soil data
    st.subheader('Soil Data Insights')
    summary_df = pd.DataFrame({
        'Mean': df.mean(),
        'Max': df.max(),
        'Min': df.min(),
        'Std Dev': df.std(),
        'Count': df.count()
    })
    st.table(summary_df)

    # Extract coordinates
    coordinates = [tuple(map(float, corner.split(','))) for corner in corners_df['Corners']]
    
    # Ensure the first and last points are the same to form a closed boundary
    if coordinates[0] != coordinates[-1]:
        coordinates.append(coordinates[0])

    # Calculate the area of the land
    polygon = Polygon(coordinates)
    area_sq_meters = polygon.area * (111320 ** 2)  # approximate conversion from degrees to meters squared
    area_acres = area_sq_meters * 0.000247105  # conversion from square meters to acres

    st.subheader('Land Area')
    st.write(f"Area of the land: {area_sq_meters:.2f} square meters")
    st.write(f"Area of the land: {area_acres:.2f} acres")

    # Function to map values to colors
    def get_color(value, min_val, max_val):
        colormap = LinearColormap(colors=['blue', 'lime', 'red'], vmin=min_val, vmax=max_val)
        return colormap(value)

    # Create a colormap for the selected parameter
    parameter = st.radio(
        "Select parameter to visualize on the map:",
        ('Moisture (%)', 'Temperature (C)', 'EC (us/cm)', 'Ph', 'Nitrogen (mg/kg)', 'Posphorous (mg/kg)', 'Potassium (mg/kg)')
    )

    min_val = df[parameter].min()
    max_val = df[parameter].max()

    # Create a Folium map centered around the first point
    m = folium.Map(location=[coordinates[0][0], coordinates[0][1]], zoom_start=17)

    # Add points to the map with color intensity based on the selected parameter
    for _, row in df.iterrows():
        color = get_color(row[parameter], min_val, max_val)
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=8,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=f"{parameter}: {row[parameter]}"
        ).add_to(m)

    # Add a line to the map to mark the boundary of the land
    folium.PolyLine(locations=coordinates, color='blue', weight=2.5, opacity=1).add_to(m)

    # Interpolation
    st.subheader('Nutrient Distribution Profile')

    # Define grid within the bounding box of the polygon
    min_lat, min_lon, max_lat, max_lon = polygon.bounds
    grid_x, grid_y = np.mgrid[min_lat:max_lat:100j, min_lon:max_lon:100j]

    def idw(x, y, z, xi, yi, power=2):
        dist = np.sqrt((x[:, None, None] - xi[None, :, :])**2 + (y[:, None, None] - yi[None, :, :])**2)
        weights = 1 / (dist**power)
        weights[dist == 0] = np.inf  # Handle zero distance (to avoid division by zero)
        z_idw = np.sum(weights * z[:, None, None], axis=0) / np.sum(weights, axis=0)
        return z_idw

    # Perform IDW interpolation
    x = df['Latitude'].values
    y = df['Longitude'].values
    z = df[parameter].values
    grid_z_idw = idw(x, y, z, grid_x, grid_y)

    # Mask the grid outside the polygon
    mask = np.array([Point(x, y).within(polygon) for x, y in zip(grid_x.flatten(), grid_y.flatten())])
    grid_z_idw = np.ma.masked_array(grid_z_idw, ~mask.reshape(grid_z_idw.shape))

    # Plot the interpolated data
    fig, ax = plt.subplots()
    cmap = LinearSegmentedColormap.from_list("mycmap", ['blue', 'lime', 'red'])
    contour = ax.contourf(grid_y, grid_x, grid_z_idw, cmap=cmap, levels=100)  # Swap grid_x and grid_y for correct orientation
    plt.colorbar(contour, ax=ax, label=parameter)
    ax.set_title(f'{parameter} Distribution')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_aspect('equal', 'box')
    st.pyplot(fig)

    # Display the map
    st.subheader('Map View')
    st_folium(m, width=700, height=500)

    # Fertilizer recommendation
    st.subheader('Fertilizer Recommendation')

    # Select crop age group
    age_group = st.selectbox("Select the age group of your crop:", fertilizer_schedule_df['Age Group'].dropna().unique())

    # Filter the time periods based on the selected age group
    time_period_options = fertilizer_schedule_df[fertilizer_schedule_df['Age Group'] == age_group]['Time *'].unique()
    time_period = st.selectbox("Select the time period:", time_period_options)

    # Filter the fertilizer schedule based on the selected age group and time period
    filtered_schedule = fertilizer_schedule_df[
        (fertilizer_schedule_df['Age Group'] == age_group) & (fertilizer_schedule_df['Time *'] == time_period)
    ]

    # Display the recommended fertilizer amounts
    if not filtered_schedule.empty:
        urea_required = filtered_schedule['Urea'].values[0]
        tsp_required = filtered_schedule['T.S.P'].values[0]
        mop_required = filtered_schedule['M.O.P'].values[0]

        st.write(f"Recommended amounts of fertilizers for {age_group} crops during {time_period}:")
        st.write(f"Urea: {urea_required} kg/ha")
        st.write(f"T.S.P: {tsp_required} kg/ha")
        st.write(f"M.O.P: {mop_required} kg/ha")

        # Calculate the deficit and additional fertilizer needed
        def calculate_fertilizer(soil_nutrient, nutrient_threshold, fertilizer_pct):
            deficit = nutrient_threshold - soil_nutrient
            required_fertilizer = max(deficit / fertilizer_pct, 0)  # Ensure no negative values
            return required_fertilizer

        nitrogen_threshold = urea_required * 0.46  # Assuming Urea is 46% Nitrogen
        phosphorous_threshold = tsp_required * 0.20  # Assuming T.S.P is 20% Phosphorous
        potassium_threshold = mop_required * 0.50  # Assuming M.O.P is 50% Potassium

        soil_nitrogen = df['Nitrogen (mg/kg)'].mean()
        soil_phosphorous = df['Posphorous (mg/kg)'].mean()
        soil_potassium = df['Potassium (mg/kg)'].mean()

        urea_needed = calculate_fertilizer(soil_nitrogen, nitrogen_threshold, 0.46)
        tsp_needed = calculate_fertilizer(soil_phosphorous, phosphorous_threshold, 0.20)
        mop_needed = calculate_fertilizer(soil_potassium, potassium_threshold, 0.50)

        st.write(f"Additional Urea needed: {urea_needed:.2f} kg/ha")
        st.write(f"Additional T.S.P needed: {tsp_needed:.2f} kg/ha")
        st.write(f"Additional M.O.P needed: {mop_needed:.2f} kg/ha")
else:
    st.write("Please upload both Excel files to view the map.")
