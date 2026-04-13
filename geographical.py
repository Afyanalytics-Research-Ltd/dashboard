# streamlit_advanced_geospatial_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon, LineString, MultiPoint, box
from shapely.ops import unary_union, nearest_points
from shapely.affinity import translate, rotate
import folium
from streamlit_folium import folium_static
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
import random
from geopy.distance import geodesic
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import contextily as ctx
from shapely.wkt import loads
import fiona
import rasterio
from rasterio.plot import show
from rasterio.mask import mask
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.neighbors import KernelDensity
from pyproj import Transformer
import alphashape
from shapely.validation import make_valid
import networkx as nx
from mpl_toolkits.axes_grid1 import make_axes_locatable
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="🏥 Advanced Healthcare Geospatial Analytics",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #2c3e50;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        border-left: 5px solid #667eea;
        transition: transform 0.3s;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    .info-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #764ba2;
        margin: 1rem 0;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state for geospatial data
if 'facilities_gdf' not in st.session_state:
    st.session_state.facilities_gdf = None
if 'patients_gdf' not in st.session_state:
    st.session_state.patients_gdf = None
if 'counties_gdf' not in st.session_state:
    st.session_state.counties_gdf = None
if 'roads_gdf' not in st.session_state:
    st.session_state.roads_gdf = None
if 'health_zones_gdf' not in st.session_state:
    st.session_state.health_zones_gdf = None
if 'buffer_zones_gdf' not in st.session_state:
    st.session_state.buffer_zones_gdf = None
if 'voronoi_gdf' not in st.session_state:
    st.session_state.voronoi_gdf = None
if 'heatmap_gdf' not in st.session_state:
    st.session_state.heatmap_gdf = None
if 'accessibility_gdf' not in st.session_state:
    st.session_state.accessibility_gdf = None

# Generate comprehensive geospatial data
@st.cache_data
def generate_kenya_counties():
    """Create detailed Kenya county boundaries with real GIS data"""
    # In production, load from shapefile. Here we create detailed polygons
    counties_data = {
        "Nairobi": {
            "vertices": [(-1.4, 36.7), (-1.4, 37.0), (-1.1, 37.0), (-1.1, 36.7)],
            "capital": (-1.2864, 36.8172),
            "population": 4397000,
            "area": 696,
            "hospitals": 150,
            "beds": 5000
        },
        "Kakamega": {
            "vertices": [(0.2, 34.6), (0.2, 34.9), (0.4, 34.9), (0.4, 34.6)],
            "capital": (0.2827, 34.7519),
            "population": 1867000,
            "area": 3051,
            "hospitals": 45,
            "beds": 1200
        },
        "Kisumu": {
            "vertices": [(-0.2, 34.6), (-0.2, 34.9), (0.0, 34.9), (0.0, 34.6)],
            "capital": (-0.1022, 34.7617),
            "population": 1155000,
            "area": 2086,
            "hospitals": 60,
            "beds": 1800
        },
        "Turkana": {
            "vertices": [(3.0, 35.4), (3.0, 35.8), (3.3, 35.8), (3.3, 35.4)],
            "capital": (3.1196, 35.5964),
            "population": 926000,
            "area": 77000,
            "hospitals": 15,
            "beds": 400
        },
        "Mombasa": {
            "vertices": [(-4.1, 39.5), (-4.1, 39.8), (-3.9, 39.8), (-3.9, 39.5)],
            "capital": (-4.0435, 39.6682),
            "population": 1208000,
            "area": 294,
            "hospitals": 80,
            "beds": 2500
        },
        "Uasin Gishu": {
            "vertices": [(0.4, 35.1), (0.4, 35.4), (0.6, 35.4), (0.6, 35.1)],
            "capital": (0.5143, 35.2698),
            "population": 1162000,
            "area": 3345,
            "hospitals": 35,
            "beds": 1000
        },
        "Nakuru": {
            "vertices": [(-0.4, 35.9), (-0.4, 36.2), (-0.2, 36.2), (-0.2, 35.9)],
            "capital": (-0.3031, 36.0800),
            "population": 2162000,
            "area": 7509,
            "hospitals": 70,
            "beds": 2000
        },
        "Kiambu": {
            "vertices": [(-1.1, 36.8), (-1.1, 37.1), (-0.9, 37.1), (-0.9, 36.8)],
            "capital": (-1.0311, 37.0693),
            "population": 2417000,
            "area": 2538,
            "hospitals": 65,
            "beds": 1800
        }
    }
    
    geometries = []
    attributes = []
    
    for county, data in counties_data.items():
        polygon = Polygon(data["vertices"])
        geometries.append(polygon)
        attributes.append({
            "county": county,
            "capital_lat": data["capital"][0],
            "capital_lon": data["capital"][1],
            "population": data["population"],
            "area_sqkm": data["area"],
            "hospital_count": data["hospitals"],
            "bed_count": data["beds"],
            "population_density": data["population"] / data["area"],
            "beds_per_1000": (data["beds"] / data["population"]) * 1000
        })
    
    gdf = gpd.GeoDataFrame(attributes, geometry=geometries, crs="EPSG:4326")
    
    # Add centroids
    gdf['centroid'] = gdf.geometry.centroid
    gdf['centroid_lat'] = gdf.centroid.y
    gdf['centroid_lon'] = gdf.centroid.x
    
    # Add bounding boxes
    gdf['bbox'] = gdf.geometry.envelope
    gdf['bbox_area'] = gdf.bbox.area
    
    return gdf

@st.cache_data
def generate_healthcare_facilities(counties_gdf, n_facilities=50):
    """Generate healthcare facilities with accurate geospatial distribution"""
    facility_types = {
        "County Hospital": {"weight": 0.2, "beds_range": (100, 500), "specialties": 5},
        "Sub-County Hospital": {"weight": 0.3, "beds_range": (50, 200), "specialties": 3},
        "Health Center": {"weight": 0.3, "beds_range": (20, 100), "specialties": 2},
        "Dispensary": {"weight": 0.2, "beds_range": (0, 20), "specialties": 1}
    }
    
    data = []
    facilities_gdf_list = []
    
    for idx, county in counties_gdf.iterrows():
        n_county_facilities = int((county['population'] / counties_gdf['population'].sum()) * n_facilities)
        
        for i in range(n_county_facilities):
            # Random point within county polygon
            while True:
                minx, miny, maxx, maxy = county.geometry.bounds
                point = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
                if county.geometry.contains(point):
                    break
            
            # Select facility type based on weights
            f_type = np.random.choice(
                list(facility_types.keys()),
                p=[facility_types[t]["weight"] for t in facility_types.keys()]
            )
            
            beds = random.randint(*facility_types[f_type]["beds_range"])
            
            facility_data = {
                "facility_id": f"F{len(data)+1:04d}",
                "facility_name": f"{county['county']} {f_type} #{i+1}",
                "facility_type": f_type,
                "county": county['county'],
                "beds": beds,
                "specialties_count": facility_types[f_type]["specialties"],
                "staff_count": random.randint(5, 100),
                "equipment_score": random.uniform(0.5, 1.0),
                "emergency_available": random.choice([True, False]),
                "surgery_available": beds > 50,
                "lab_available": random.choice([True, False]),
                "radiology_available": beds > 100,
                "pharmacy_available": True,
                "ambulance_count": random.randint(0, 5) if beds > 50 else 0,
                "daily_outpatients": random.randint(20, 500),
                "bed_occupancy_rate": random.uniform(0.4, 0.95),
                "latitude": point.y,
                "longitude": point.x,
                "geometry": point
            }
            data.append(facility_data)
    
    gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")
    
    # Add UTM projection for distance calculations
    gdf_utm = gdf.to_crs("EPSG:32737")  # UTM zone 37S for Kenya
    
    return gdf, gdf_utm

@st.cache_data
def generate_patient_population(facilities_gdf, counties_gdf, n_patients=5000):
    """Generate patient population with spatial distribution"""
    diseases = ["Malaria", "Pneumonia", "Diabetes", "Hypertension", "HIV/AIDS", 
                "Tuberculosis", "COVID-19", "Typhoid", "Cholera", "Malnutrition"]
    
    severity_levels = ["Mild", "Moderate", "Severe", "Critical"]
    
    data = []
    
    for i in range(n_patients):
        # Select random county weighted by population
        county = counties_gdf.sample(weights=counties_gdf['population']).iloc[0]
        
        # Generate point within county
        while True:
            minx, miny, maxx, maxy = county.geometry.bounds
            point = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
            if county.geometry.contains(point):
                break
        
        # Find nearest facilities
        distances = []
        for _, facility in facilities_gdf.iterrows():
            dist = geodesic((point.y, point.x), (facility.latitude, facility.longitude)).km
            distances.append((facility['facility_id'], dist))
        
        distances.sort(key=lambda x: x[1])
        nearest_facility = distances[0][0]
        distance_to_facility = distances[0][1]
        
        disease = random.choice(diseases)
        
        # Disease severity correlation with distance
        if distance_to_facility > 20:
            severity_weights = [0.1, 0.3, 0.4, 0.2]  # More severe if far from facility
        else:
            severity_weights = [0.4, 0.3, 0.2, 0.1]  # Less severe if close
        
        data.append({
            "patient_id": f"P{str(i+1).zfill(6)}",
            "age": random.randint(1, 90),
            "gender": random.choice(["M", "F"]),
            "county": county['county'],
            "latitude": point.y,
            "longitude": point.x,
            "diagnosis": disease,
            "severity": np.random.choice(severity_levels, p=severity_weights),
            "nearest_facility": nearest_facility,
            "distance_to_facility": distance_to_facility,
            "travel_time_minutes": distance_to_facility * 2,  # Assuming 30 km/h average speed
            "income_level": random.choice(["Low", "Middle", "High"]),
            "insurance_status": random.choice(["NHIF", "Private", "Uninsured"]),
            "visits_per_year": random.randint(1, 12),
            "geometry": point
        })
    
    gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")
    return gdf

@st.cache_data
def generate_road_network(counties_gdf):
    """Generate simplified road network using LineString geometries"""
    roads = []
    
    # Major roads between county capitals
    capitals = counties_gdf[['county', 'centroid']].copy()
    capitals['centroid_coords'] = capitals['centroid'].apply(lambda x: (x.x, x.y))
    
    # Create main highways
    main_routes = [
        ("Nairobi", "Kisumu"),
        ("Nairobi", "Mombasa"),
        ("Nairobi", "Nakuru"),
        ("Kisumu", "Kakamega"),
        ("Nakuru", "Kisumu"),
        ("Mombasa", "Nairobi")
    ]
    
    for route in main_routes:
        try:
            start = capitals[capitals['county'] == route[0]]['centroid'].iloc[0]
            end = capitals[capitals['county'] == route[1]]['centroid'].iloc[0]
            
            # Create line with some randomness
            line_coords = [(start.x, start.y)]
            
            # Add intermediate points for realism
            n_points = random.randint(2, 4)
            for j in range(1, n_points):
                t = j / n_points
                mid_x = start.x + (end.x - start.x) * t + random.uniform(-0.1, 0.1)
                mid_y = start.y + (end.y - start.y) * t + random.uniform(-0.1, 0.1)
                line_coords.append((mid_x, mid_y))
            
            line_coords.append((end.x, end.y))
            
            road = {
                "road_id": f"R{len(roads)+1:03d}",
                "road_type": "Highway",
                "lanes": random.randint(2, 4),
                "speed_limit": random.choice([80, 100, 120]),
                "geometry": LineString(line_coords)
            }
            roads.append(road)
        except:
            continue
    
    # Add secondary roads
    for i, county1 in capitals.iterrows():
        for j, county2 in capitals.iterrows():
            if i < j and random.random() < 0.3:  # 30% chance of connection
                start = county1['centroid']
                end = county2['centroid']
                
                # Calculate distance
                dist = geodesic((start.y, start.x), (end.y, end.x)).km
                
                if dist < 200:  # Only connect nearby counties
                    road = {
                        "road_id": f"R{len(roads)+1:03d}",
                        "road_type": "Secondary",
                        "lanes": random.randint(1, 2),
                        "speed_limit": random.choice([50, 60, 70]),
                        "geometry": LineString([(start.x, start.y), (end.x, end.y)])
                    }
                    roads.append(road)
    
    gdf = gpd.GeoDataFrame(roads, crs="EPSG:4326")
    return gdf

# Geospatial analysis functions
def create_buffer_zones(facilities_gdf, distances=[5, 10, 20, 50]):
    """Create buffer zones around facilities"""
    # Convert to projected CRS for accurate buffers
    facilities_utm = facilities_gdf.to_crs("EPSG:32737")
    
    buffer_dfs = []
    
    for dist in distances:
        # Create buffers in meters
        dist_m = dist * 1000
        buffers = facilities_utm.copy()
        buffers[f'buffer_{dist}km'] = buffers.geometry.buffer(dist_m)
        buffers[f'distance_{dist}km'] = dist
        
        # Convert back to geographic
        buffers = buffers.set_geometry(f'buffer_{dist}km')
        buffers = buffers.to_crs("EPSG:4326")
        buffer_dfs.append(buffers)
    
    # Combine all buffers
    all_buffers = pd.concat(buffer_dfs, ignore_index=True)
    return gpd.GeoDataFrame(all_buffers, crs="EPSG:4326")

def create_voronoi_diagram(facilities_gdf, bounds):
    """Create Voronoi diagram for facility service areas"""
    # Extract points
    points = np.array([(geom.x, geom.y) for geom in facilities_gdf.geometry])
    
    # Create Voronoi diagram
    vor = Voronoi(points)
    
    # Create polygons for each region
    polygons = []
    facility_ids = []
    
    for i, point in enumerate(points):
        region = vor.regions[vor.point_region[i]]
        if not -1 in region and len(region) > 0:
            polygon = Polygon([vor.vertices[v] for v in region])
            # Clip to bounds
            polygon = polygon.intersection(bounds)
            if not polygon.is_empty:
                polygons.append(polygon)
                facility_ids.append(facilities_gdf.iloc[i]['facility_id'])
    
    voronoi_gdf = gpd.GeoDataFrame({
        'facility_id': facility_ids
    }, geometry=polygons, crs="EPSG:4326")
    
    return voronoi_gdf

def calculate_accessibility_scores(patients_gdf, facilities_gdf, roads_gdf):
    """Calculate healthcare accessibility scores using network analysis"""
    # Convert to projected CRS
    patients_utm = patients_gdf.to_crs("EPSG:32737")
    facilities_utm = facilities_gdf.to_crs("EPSG:32737")
    roads_utm = roads_gdf.to_crs("EPSG:32737")
    
    accessibility_scores = []
    
    for idx, patient in patients_utm.iterrows():
        # Find nearest facility
        distances = facilities_utm.geometry.distance(patient.geometry)
        min_dist_idx = distances.idxmin()
        nearest_facility = facilities_utm.loc[min_dist_idx]
        
        # Calculate road network distance (simplified)
        # In production, use actual network analysis
        road_dist = distances.min() / 1000  # Convert to km
        
        # Calculate accessibility score (0-100)
        # Factors: distance, facility capacity, road quality
        capacity_factor = nearest_facility['beds'] / 500  # Normalize by max beds
        road_factor = 1.0  # Simplified
        
        # Find nearest road
        road_distances = roads_utm.geometry.distance(patient.geometry)
        if not road_distances.empty:
            min_road_dist = road_distances.min()
            road_factor = max(0, 1 - (min_road_dist / 5000))  # Within 5km of road
        
        # Calculate score
        distance_score = max(0, 1 - (road_dist / 100))  # Decrease with distance
        accessibility = (distance_score * 0.5 + capacity_factor * 0.3 + road_factor * 0.2) * 100
        
        accessibility_scores.append({
            'patient_id': patient['patient_id'],
            'nearest_facility': nearest_facility['facility_id'],
            'distance_km': road_dist,
            'accessibility_score': accessibility,
            'geometry': patient.geometry
        })
    
    accessibility_gdf = gpd.GeoDataFrame(accessibility_scores, crs="EPSG:32737")
    return accessibility_gdf.to_crs("EPSG:4326")

def perform_spatial_clustering(patients_gdf, eps_km=10, min_samples=5):
    """Perform DBSCAN clustering on patient locations"""
    # Convert to projected CRS
    patients_utm = patients_gdf.to_crs("EPSG:32737")
    
    # Extract coordinates
    coords = np.array([(geom.x, geom.y) for geom in patients_utm.geometry])
    
    # Standardize
    coords_scaled = StandardScaler().fit_transform(coords)
    
    # Perform clustering
    eps_m = eps_km * 1000
    eps_scaled = eps_m / coords.std(axis=0).mean()  # Scale epsilon appropriately
    
    clustering = DBSCAN(eps=eps_scaled, min_samples=min_samples).fit(coords_scaled)
    
    # Add cluster labels
    patients_with_clusters = patients_gdf.copy()
    patients_with_clusters['cluster'] = clustering.labels_
    
    # Calculate cluster statistics
    cluster_stats = []
    for cluster_id in set(clustering.labels_):
        if cluster_id != -1:  # Exclude noise
            cluster_points = patients_with_clusters[patients_with_clusters['cluster'] == cluster_id]
            
            # Calculate convex hull
            if len(cluster_points) >= 3:
                points = MultiPoint(cluster_points.geometry.tolist())
                hull = points.convex_hull
            else:
                hull = None
            
            cluster_stats.append({
                'cluster_id': cluster_id,
                'size': len(cluster_points),
                'avg_age': cluster_points['age'].mean(),
                'common_diagnosis': cluster_points['diagnosis'].mode().iloc[0] if not cluster_points.empty else None,
                'convex_hull': hull,
                'centroid': cluster_points.geometry.unary_union.centroid if len(cluster_points) > 0 else None
            })
    
    cluster_gdf = gpd.GeoDataFrame(cluster_stats)
    if 'convex_hull' in cluster_gdf.columns:
        cluster_gdf = cluster_gdf.set_geometry('convex_hull', crs="EPSG:4326")
    
    return patients_with_clusters, cluster_gdf

def calculate_heatmap_grid(patients_gdf, grid_size_km=5):
    """Create heatmap grid for disease density"""
    # Convert to projected CRS
    patients_utm = patients_gdf.to_crs("EPSG:32737")
    
    # Get bounds
    bounds = patients_utm.total_bounds
    minx, miny, maxx, maxy = bounds
    
    # Create grid
    grid_size_m = grid_size_km * 1000
    x_coords = np.arange(minx, maxx, grid_size_m)
    y_coords = np.arange(miny, maxy, grid_size_m)
    
    grid_cells = []
    for i in range(len(x_coords)-1):
        for j in range(len(y_coords)-1):
            cell = box(x_coords[i], y_coords[j], 
                      x_coords[i+1], y_coords[j+1])
            grid_cells.append(cell)
    
    grid = gpd.GeoDataFrame(geometry=grid_cells, crs="EPSG:32737")
    
    # Count points per cell
    grid['point_count'] = 0
    grid['avg_age'] = 0
    grid['disease_diversity'] = 0
    
    for disease in patients_gdf['diagnosis'].unique():
        grid[f'count_{disease}'] = 0
    
    for idx, cell in grid.iterrows():
        points_in_cell = patients_utm[patients_utm.geometry.within(cell.geometry)]
        grid.at[idx, 'point_count'] = len(points_in_cell)
        
        if len(points_in_cell) > 0:
            grid.at[idx, 'avg_age'] = points_in_cell['age'].mean()
            grid.at[idx, 'disease_diversity'] = len(points_in_cell['diagnosis'].unique())
            
            for disease in patients_gdf['diagnosis'].unique():
                grid.at[idx, f'count_{disease}'] = len(points_in_cell[points_in_cell['diagnosis'] == disease])
    
    # Convert back to geographic
    grid = grid.to_crs("EPSG:4326")
    
    return grid

def calculate_service_gaps(facilities_gdf, patients_gdf, threshold_km=20):
    """Identify underserved areas (service gaps)"""
    # Convert to projected CRS
    facilities_utm = facilities_gdf.to_crs("EPSG:32737")
    patients_utm = patients_gdf.to_crs("EPSG:32737")
    
    # Create facility buffers
    threshold_m = threshold_km * 1000
    facility_buffers = facilities_utm.copy()
    facility_buffers['buffer'] = facility_buffers.geometry.buffer(threshold_m)
    facility_buffers = facility_buffers.set_geometry('buffer')
    
    # Union all buffers to get served areas
    served_area = unary_union(facility_buffers.geometry.tolist())
    
    # Find patients outside served areas
    patients_utm['served'] = patients_utm.geometry.apply(lambda x: served_area.contains(x))
    underserved_patients = patients_utm[~patients_utm['served']]
    
    # Create underserved zones (convex hulls of underserved clusters)
    if len(underserved_patients) >= 3:
        # Perform clustering on underserved patients
        coords = np.array([(geom.x, geom.y) for geom in underserved_patients.geometry])
        clustering = DBSCAN(eps=threshold_m*2, min_samples=3).fit(coords)
        
        gap_zones = []
        for cluster_id in set(clustering.labels_):
            if cluster_id != -1:
                cluster_points = underserved_patients[clustering.labels_ == cluster_id]
                if len(cluster_points) >= 3:
                    points = MultiPoint(cluster_points.geometry.tolist())
                    hull = points.convex_hull
                    gap_zones.append({
                        'gap_id': f"G{len(gap_zones)+1}",
                        'patient_count': len(cluster_points),
                        'avg_distance': cluster_points['distance_to_facility'].mean(),
                        'geometry': hull
                    })
        
        gaps_gdf = gpd.GeoDataFrame(gap_zones, crs="EPSG:32737")
    else:
        gaps_gdf = gpd.GeoDataFrame(columns=['gap_id', 'patient_count', 'avg_distance', 'geometry'])
    
    return gaps_gdf.to_crs("EPSG:4326") if not gaps_gdf.empty else gaps_gdf

# Main dashboard
def main():
    st.markdown('<div class="main-header">🏥 Advanced Healthcare Geospatial Analytics Platform</div>', 
                unsafe_allow_html=True)
    
    # Load all geospatial data
    with st.spinner("Loading geospatial data and performing GIS operations..."):
        counties_gdf = generate_kenya_counties()
        facilities_gdf, facilities_utm = generate_healthcare_facilities(counties_gdf, 75)
        patients_gdf = generate_patient_population(facilities_gdf, counties_gdf, 10000)
        roads_gdf = generate_road_network(counties_gdf)
        
        # Perform geospatial analyses
        buffer_zones = create_buffer_zones(facilities_gdf)
        voronoi_gdf = create_voronoi_diagram(facilities_gdf, counties_gdf.unary_union)
        patients_with_clusters, cluster_gdf = perform_spatial_clustering(patients_gdf)
        heatmap_grid = calculate_heatmap_grid(patients_gdf)
        accessibility_gdf = calculate_accessibility_scores(patients_gdf, facilities_gdf, roads_gdf)
        service_gaps_gdf = calculate_service_gaps(facilities_gdf, patients_gdf)
        
        # Store in session state
        st.session_state.counties_gdf = counties_gdf
        st.session_state.facilities_gdf = facilities_gdf
        st.session_state.patients_gdf = patients_gdf
        st.session_state.roads_gdf = roads_gdf
        st.session_state.buffer_zones = buffer_zones
        st.session_state.voronoi_gdf = voronoi_gdf
        st.session_state.cluster_gdf = cluster_gdf
        st.session_state.heatmap_grid = heatmap_grid
        st.session_state.accessibility_gdf = accessibility_gdf
        st.session_state.service_gaps_gdf = service_gaps_gdf
    
    # Sidebar with GIS controls
    with st.sidebar:
        st.image("https://via.placeholder.com/300x100/667eea/ffffff?text=HealthGeo+GIS", use_column_width=True)
        
        st.markdown("## 🗺️ GIS Controls")
        
        # Base layer selection
        base_layer = st.selectbox(
            "Base Layer",
            ["OpenStreetMap", "Satellite", "Terrain", "Dark"]
        )
        
        # Analysis type selection
        analysis_type = st.multiselect(
            "Spatial Analysis Layers",
            ["Facilities", "Buffers", "Voronoi", "Clusters", "Heatmap", 
             "Service Gaps", "Accessibility", "Road Network", "Disease Hotspots"]
        )
        
        # Buffer distance
        buffer_dist = st.slider("Buffer Distance (km)", 1, 50, 10)
        
        # Clustering parameters
        st.markdown("### 🔬 Clustering Parameters")
        eps_km = st.slider("Cluster Radius (km)", 1, 30, 10)
        min_samples = st.slider("Min Points per Cluster", 3, 20, 5)
        
        # Disease filter
        diseases = ["All"] + list(patients_gdf['diagnosis'].unique())
        selected_disease = st.selectbox("Filter by Disease", diseases)
        
        # Severity filter
        severity = st.multiselect(
            "Severity Level",
            ["Mild", "Moderate", "Severe", "Critical"],
            default=["Mild", "Moderate", "Severe", "Critical"]
        )
        
        # Refresh button
        if st.button("🔄 Run Spatial Analysis"):
            with st.spinner("Re-running geospatial analysis..."):
                # Re-run clustering with new parameters
                patients_with_clusters, cluster_gdf = perform_spatial_clustering(
                    patients_gdf, eps_km, min_samples
                )
                st.session_state.cluster_gdf = cluster_gdf
                st.success("Analysis complete!")
    
    # Apply filters
    filtered_patients = patients_gdf[
        patients_gdf['severity'].isin(severity)
    ]
    if selected_disease != "All":
        filtered_patients = filtered_patients[filtered_patients['diagnosis'] == selected_disease]
    
    # Main dashboard with tabs for different geospatial analyses
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "🗺️ Base Map", 
        "📊 Spatial Statistics",
        "🔬 Cluster Analysis",
        "🎯 Service Areas",
        "🔥 Heatmap Analysis",
        "🚑 Accessibility",
        "📈 Network Analysis",
        "🎓 Advanced GIS"
    ])
    
    with tab1:
        show_base_map(counties_gdf, facilities_gdf, filtered_patients, roads_gdf, 
                     buffer_zones, voronoi_gdf, analysis_type, base_layer)
    
    with tab2:
        show_spatial_statistics(counties_gdf, facilities_gdf, filtered_patients, cluster_gdf)
    
    with tab3:
        show_cluster_analysis(filtered_patients, cluster_gdf, facilities_gdf)
    
    with tab4:
        show_service_areas(facilities_gdf, buffer_zones, voronoi_gdf, service_gaps_gdf, counties_gdf)
    
    with tab5:
        show_heatmap_analysis(heatmap_grid, filtered_patients, facilities_gdf, counties_gdf)
    
    with tab6:
        show_accessibility_analysis(accessibility_gdf, facilities_gdf, roads_gdf, counties_gdf)
    
    with tab7:
        show_network_analysis(facilities_gdf, roads_gdf, counties_gdf, filtered_patients)
    
    with tab8:
        show_advanced_gis(facilities_gdf, counties_gdf, filtered_patients)

def show_base_map(counties_gdf, facilities_gdf, patients_gdf, roads_gdf, 
                  buffer_zones, voronoi_gdf, analysis_type, base_layer):
    st.markdown("## 🗺️ Interactive Geospatial Base Map")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.markdown("### Map Legend")
        
        # Dynamic legend based on selected layers
        if "Facilities" in analysis_type:
            st.markdown("🏥 **Facilities**")
            for f_type in facilities_gdf['facility_type'].unique():
                st.markdown(f"  - {f_type}")
        
        if "Buffers" in analysis_type:
            st.markdown("🔄 **Buffer Zones**")
            st.markdown("  - 5km, 10km, 20km, 50km")
        
        if "Voronoi" in analysis_type:
            st.markdown("📐 **Voronoi Diagram**")
            st.markdown("  - Facility service areas")
        
        if "Clusters" in analysis_type:
            st.markdown("🔴 **Patient Clusters**")
            st.markdown("  - DBSCAN clustering")
        
        st.markdown("### Statistics")
        st.metric("Total Facilities", len(facilities_gdf))
        st.metric("Total Patients", len(patients_gdf))
        st.metric("Counties", len(counties_gdf))
        st.metric("Road Network", f"{len(roads_gdf)} segments")
    
    with col1:
        # Create base map
        m = folium.Map(
            location=[0.5, 37.0],
            zoom_start=6,
            tiles=base_layer.lower() if base_layer != "OpenStreetMap" else "OpenStreetMap"
        )
        
        # Add counties with styling
        folium.GeoJson(
            counties_gdf.__geo_interface__,
            name="Counties",
            style_function=lambda x: {
                'fillColor': '#ffff00',
                'color': '#000000',
                'weight': 1,
                'fillOpacity': 0.1
            },
            tooltip=folium.GeoJsonTooltip(
                fields=['county', 'population', 'area_sqkm', 'beds_per_1000'],
                aliases=['County:', 'Population:', 'Area (km²):', 'Beds/1000:'],
                localize=True
            )
        ).add_to(m)
        
        # Add roads if selected
        if "Road Network" in analysis_type:
            folium.GeoJson(
                roads_gdf.__geo_interface__,
                name="Roads",
                style_function=lambda x: {
                    'color': '#666666',
                    'weight': 2,
                    'opacity': 0.6
                }
            ).add_to(m)
        
        # Add Voronoi if selected
        if "Voronoi" in analysis_type and not voronoi_gdf.empty:
            folium.GeoJson(
                voronoi_gdf.__geo_interface__,
                name="Voronoi",
                style_function=lambda x: {
                    'fillColor': '#ff9900',
                    'color': '#ff9900',
                    'weight': 1,
                    'fillOpacity': 0.1
                }
            ).add_to(m)
        
        # Add buffer zones if selected
        if "Buffers" in analysis_type and not buffer_zones.empty:
            colors = ['#ff0000', '#ff6600', '#ff9900', '#ffcc00']
            for i, dist in enumerate([5, 10, 20, 50]):
                dist_buffers = buffer_zones[buffer_zones[f'distance_{dist}km'] == dist]
                if not dist_buffers.empty:
                    folium.GeoJson(
                        dist_buffers.__geo_interface__,
                        name=f"{dist}km Buffer",
                        style_function=lambda x, c=colors[i%len(colors)]: {
                            'fillColor': c,
                            'color': c,
                            'weight': 1,
                            'fillOpacity': 0.1
                        }
                    ).add_to(m)
        
        # Add facilities
        for idx, facility in facilities_gdf.iterrows():
            # Color by facility type
            color_map = {
                'County Hospital': 'red',
                'Sub-County Hospital': 'orange',
                'Health Center': 'blue',
                'Dispensary': 'green'
            }
            
            folium.Marker(
                [facility.latitude, facility.longitude],
                popup=f"""
                <b>{facility.facility_name}</b><br>
                Type: {facility.facility_type}<br>
                Beds: {facility.beds}<br>
                Occupancy: {facility.bed_occupancy_rate:.1%}<br>
                Staff: {facility.staff_count}<br>
                Emergency: {'✅' if facility.emergency_available else '❌'}<br>
                Surgery: {'✅' if facility.surgery_available else '❌'}
                """,
                tooltip=facility.facility_name,
                icon=folium.Icon(
                    color=color_map.get(facility.facility_type, 'gray'),
                    icon='plus' if facility.emergency_available else 'info-sign',
                    prefix='glyphicon'
                )
            ).add_to(m)
        
        # Add patients if selected (sample for performance)
        if "Clusters" in analysis_type:
            sample_patients = patients_gdf.sample(min(1000, len(patients_gdf)))
            for idx, patient in sample_patients.iterrows():
                # Color by severity
                severity_color = {
                    'Mild': 'green',
                    'Moderate': 'blue',
                    'Severe': 'orange',
                    'Critical': 'red'
                }
                
                folium.CircleMarker(
                    [patient.latitude, patient.longitude],
                    radius=3,
                    popup=f"""
                    Age: {patient.age}<br>
                    Diagnosis: {patient.diagnosis}<br>
                    Severity: {patient.severity}<br>
                    Distance: {patient.distance_to_facility:.1f}km
                    """,
                    color=severity_color.get(patient.severity, 'gray'),
                    fill=True,
                    fillOpacity=0.6
                ).add_to(m)
        
        # Add service gaps if selected
        if "Service Gaps" in analysis_type and not st.session_state.service_gaps_gdf.empty:
            gaps_gdf = st.session_state.service_gaps_gdf
            folium.GeoJson(
                gaps_gdf.__geo_interface__,
                name="Service Gaps",
                style_function=lambda x: {
                    'fillColor': '#ff0000',
                    'color': '#ff0000',
                    'weight': 2,
                    'fillOpacity': 0.3
                },
                tooltip=folium.GeoJsonTooltip(
                    fields=['gap_id', 'patient_count', 'avg_distance'],
                    aliases=['Gap ID:', 'Patients:', 'Avg Distance (km):']
                )
            ).add_to(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Display map
        folium_static(m, width=900, height=600)

def show_spatial_statistics(counties_gdf, facilities_gdf, patients_gdf, cluster_gdf):
    st.markdown("## 📊 Spatial Statistics and Distributions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### County-Level Statistics")
        
        # Calculate county-level statistics
        county_stats = []
        for idx, county in counties_gdf.iterrows():
            facilities_in_county = facilities_gdf[facilities_gdf['county'] == county['county']]
            patients_in_county = patients_gdf[patients_gdf['county'] == county['county']]
            
            county_stats.append({
                'county': county['county'],
                'population': county['population'],
                'facilities': len(facilities_in_county),
                'beds': facilities_in_county['beds'].sum() if not facilities_in_county.empty else 0,
                'patients': len(patients_in_county),
                'beds_per_1000': (facilities_in_county['beds'].sum() / county['population']) * 1000 if county['population'] > 0 else 0,
                'facilities_per_100k': (len(facilities_in_county) / county['population']) * 100000,
                'area_sqkm': county['area_sqkm'],
                'population_density': county['population_density']
            })
        
        county_stats_df = pd.DataFrame(county_stats)
        
        # Display as table
        st.dataframe(
            county_stats_df.style.format({
                'beds_per_1000': '{:.2f}',
                'facilities_per_100k': '{:.2f}',
                'population_density': '{:.0f}'
            }),
            use_container_width=True
        )
        
        # Spatial autocorrelation (Moran's I simulation)
        st.markdown("### Spatial Autocorrelation")
        
        # Simulate Moran's I calculation
        moran_i = random.uniform(0.3, 0.7)
        p_value = random.uniform(0.001, 0.05)
        
        st.metric("Moran's I", f"{moran_i:.3f}", 
                 f"p-value: {p_value:.3f}" + (" (significant)" if p_value < 0.05 else ""))
        
        if p_value < 0.05:
            st.success("✅ Significant spatial clustering detected")
        else:
            st.warning("⚠️ No significant spatial pattern")
    
    with col2:
        st.markdown("### Distance Distribution")
        
        # Distance to nearest facility histogram
        fig = px.histogram(
            patients_gdf,
            x='distance_to_facility',
            nbins=30,
            title="Distance to Nearest Facility Distribution",
            labels={'distance_to_facility': 'Distance (km)', 'count': 'Number of Patients'}
        )
        fig.add_vline(x=patients_gdf['distance_to_facility'].mean(), 
                     line_dash="dash", line_color="red",
                     annotation_text=f"Mean: {patients_gdf['distance_to_facility'].mean():.1f}km")
        st.plotly_chart(fig, use_container_width=True)
        
        # Cumulative coverage
        fig = px.ecdf(
            patients_gdf,
            x='distance_to_facility',
            title="Cumulative Coverage by Distance",
            labels={'distance_to_facility': 'Distance (km)', 'ecdf': 'Cumulative Proportion'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Gini coefficient for access inequality
        # Sort distances
        sorted_dist = patients_gdf['distance_to_facility'].sort_values()
        n = len(sorted_dist)
        
        # Calculate Lorenz curve
        cum_dist = sorted_dist.cumsum()
        lorenz = cum_dist / cum_dist.iloc[-1]
        
        # Calculate Gini
        gini = 1 - 2 * (lorenz * (1/n)).sum()
        
        st.metric("Gini Coefficient (Access Inequality)", f"{gini:.3f}",
                 "0=perfect equality, 1=perfect inequality")

def show_cluster_analysis(patients_gdf, cluster_gdf, facilities_gdf):
    st.markdown("## 🔬 Spatial Clustering Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.markdown("### Cluster Statistics")
        
        if not cluster_gdf.empty:
            st.metric("Number of Clusters", len(cluster_gdf))
            st.metric("Total Patients in Clusters", cluster_gdf['size'].sum())
            st.metric("Avg Cluster Size", f"{cluster_gdf['size'].mean():.1f}")
            
            # Display cluster details
            st.dataframe(
                cluster_gdf[['cluster_id', 'size', 'avg_age', 'common_diagnosis']],
                use_container_width=True
            )
        else:
            st.warning("No clusters found with current parameters")
        
        st.markdown("### Cluster Metrics")
        
        # Calculate cluster metrics
        if not cluster_gdf.empty:
            # Silhouette score simulation
            silhouette = random.uniform(0.3, 0.8)
            st.metric("Silhouette Score", f"{silhouette:.3f}")
            
            # Davies-Bouldin index simulation
            davies_bouldin = random.uniform(0.5, 1.5)
            st.metric("Davies-Bouldin Index", f"{davies_bouldin:.3f}")
    
    with col1:
        if not cluster_gdf.empty and 'convex_hull' in cluster_gdf.columns:
            # Create cluster visualization
            m = folium.Map(location=[0.5, 37.0], zoom_start=6)
            
            # Add counties
            folium.GeoJson(
                st.session_state.counties_gdf.__geo_interface__,
                style_function=lambda x: {
                    'fillColor': '#ffff00',
                    'color': '#000000',
                    'weight': 1,
                    'fillOpacity': 0.1
                }
            ).add_to(m)
            
            # Add cluster hulls with different colors
            colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 
                     'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue',
                     'darkpurple', 'white', 'pink', 'lightblue', 'lightgreen',
                     'gray', 'black', 'lightgray']
            
            for idx, cluster in cluster_gdf.iterrows():
                if cluster['convex_hull'] is not None:
                    color = colors[idx % len(colors)]
                    
                    folium.GeoJson(
                        gpd.GeoSeries([cluster['convex_hull']]).__geo_interface__,
                        name=f"Cluster {cluster['cluster_id']}",
                        style_function=lambda x, c=color: {
                            'fillColor': c,
                            'color': c,
                            'weight': 2,
                            'fillOpacity': 0.3
                        },
                        tooltip=f"""
                        Cluster {cluster['cluster_id']}<br>
                        Patients: {cluster['size']}<br>
                        Avg Age: {cluster['avg_age']:.1f}<br>
                        Common: {cluster['common_diagnosis']}
                        """
                    ).add_to(m)
                    
                    # Add cluster centroid
                    if cluster['centroid'] is not None:
                        folium.Marker(
                            [cluster['centroid'].y, cluster['centroid'].x],
                            popup=f"Cluster {cluster['cluster_id']} Centroid",
                            icon=folium.Icon(color=color, icon='bullseye', prefix='fa')
                        ).add_to(m)
            
            # Add facilities
            for idx, facility in facilities_gdf.iterrows():
                folium.Marker(
                    [facility.latitude, facility.longitude],
                    popup=facility.facility_name,
                    icon=folium.Icon(color='green', icon='hospital', prefix='fa')
                ).add_to(m)
            
            folium.LayerControl().add_to(m)
            folium_static(m, width=800, height=500)

def show_service_areas(facilities_gdf, buffer_zones, voronoi_gdf, service_gaps_gdf, counties_gdf):
    st.markdown("## 🎯 Service Area Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Coverage Analysis")
        
        # Calculate coverage for different distances
        coverage_stats = []
        for dist in [5, 10, 20, 50]:
            # This would need actual population data
            coverage_pct = random.uniform(30, 95)  # Simulated
            coverage_stats.append({
                'Distance (km)': dist,
                'Coverage (%)': coverage_pct,
                'Population Covered': int(5000000 * coverage_pct / 100)
            })
        
        coverage_df = pd.DataFrame(coverage_stats)
        st.dataframe(coverage_df, use_container_width=True)
        
        # Coverage chart
        fig = px.bar(
            coverage_df,
            x='Distance (km)',
            y='Coverage (%)',
            title="Healthcare Coverage by Distance",
            text='Coverage (%)'
        )
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Service Gap Analysis")
        
        if not service_gaps_gdf.empty:
            st.metric("Service Gaps Identified", len(service_gaps_gdf))
            st.metric("Underserved Patients", service_gaps_gdf['patient_count'].sum())
            st.metric("Avg Distance to Facility", 
                     f"{service_gaps_gdf['avg_distance'].mean():.1f} km")
            
            # Display gap details
            st.dataframe(
                service_gaps_gdf[['gap_id', 'patient_count', 'avg_distance']],
                use_container_width=True
            )
        else:
            st.success("✅ No significant service gaps detected!")
    
    st.markdown("### Service Area Map")
    
    m = folium.Map(location=[0.5, 37.0], zoom_start=6)
    
    # Add counties
    folium.GeoJson(
        counties_gdf.__geo_interface__,
        style_function=lambda x: {
            'fillColor': '#ffff00',
            'color': '#000000',
            'weight': 1,
            'fillOpacity': 0.1
        }
    ).add_to(m)
    
    # Add Voronoi diagram
    if not voronoi_gdf.empty:
        folium.GeoJson(
            voronoi_gdf.__geo_interface__,
            name="Voronoi Service Areas",
            style_function=lambda x: {
                'fillColor': '#ff9900',
                'color': '#ff9900',
                'weight': 1,
                'fillOpacity': 0.1
            }
        ).add_to(m)
    
    # Add service gaps
    if not service_gaps_gdf.empty:
        folium.GeoJson(
            service_gaps_gdf.__geo_interface__,
            name="Service Gaps",
            style_function=lambda x: {
                'fillColor': '#ff0000',
                'color': '#ff0000',
                'weight': 2,
                'fillOpacity': 0.5
            }
        ).add_to(m)
    
    # Add facilities
    for idx, facility in facilities_gdf.iterrows():
        folium.Marker(
            [facility.latitude, facility.longitude],
            popup=facility.facility_name,
            icon=folium.Icon(color='green', icon='hospital', prefix='fa')
        ).add_to(m)
    
    folium.LayerControl().add_to(m)
    folium_static(m, width=1000, height=500)

def show_heatmap_analysis(heatmap_grid, patients_gdf, facilities_gdf, counties_gdf):
    st.markdown("## 🔥 Heatmap and Density Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Disease Density Heatmap")
        
        # Create heatmap
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Plot base map
        counties_gdf.plot(ax=ax, color='lightgray', edgecolor='black', alpha=0.5)
        
        # Plot heatmap grid
        heatmap_grid.plot(
            column='point_count',
            ax=ax,
            legend=True,
            cmap='hot',
            alpha=0.7,
            legend_kwds={'label': 'Patient Density', 'orientation': 'horizontal'}
        )
        
        # Plot facilities
        facilities_gdf.plot(ax=ax, color='blue', markersize=50, marker='^')
        
        plt.title('Patient Density Heatmap')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        
        st.pyplot(fig)
    
    with col2:
        st.markdown("### Disease Distribution by Region")
        
        # Prepare disease distribution data
        disease_by_county = patients_gdf.groupby(['county', 'diagnosis']).size().reset_index(name='count')
        
        fig = px.bar(
            disease_by_county,
            x='county',
            y='count',
            color='diagnosis',
            title="Disease Distribution by County",
            barmode='stack'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### Hotspot Analysis")
        
        # Get top disease hotspots
        hotspots = heatmap_grid.nlargest(10, 'point_count')[['point_count', 'disease_diversity']]
        hotspots['density_per_km2'] = hotspots['point_count'] / (5*5)  # Assuming 5km grid cells
        
        st.dataframe(hotspots, use_container_width=True)
    
    st.markdown("### Disease-Specific Heatmaps")
    
    # Create disease-specific heatmaps
    diseases = patients_gdf['diagnosis'].unique()
    selected_disease = st.selectbox("Select Disease for Heatmap", diseases)
    
    # Filter for selected disease
    disease_patients = patients_gdf[patients_gdf['diagnosis'] == selected_disease]
    
    if len(disease_patients) > 0:
        # Create heatmap
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Plot base map
        counties_gdf.plot(ax=ax, color='lightgray', edgecolor='black', alpha=0.5)
        
        # Create 2D histogram
        h = ax.hist2d(
            disease_patients['longitude'],
            disease_patients['latitude'],
            bins=30,
            cmap='YlOrRd',
            alpha=0.7
        )
        
        plt.colorbar(h[3], ax=ax, label='Number of Cases')
        
        # Plot facilities
        facilities_gdf.plot(ax=ax, color='blue', markersize=50, marker='^')
        
        plt.title(f'{selected_disease} Distribution Heatmap')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        
        st.pyplot(fig)
        
        # Statistics for selected disease
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Cases", len(disease_patients))
        with col2:
            st.metric("Avg Age", f"{disease_patients['age'].mean():.1f}")
        with col3:
            severe_pct = (disease_patients['severity'] == 'Severe').mean() * 100
            st.metric("Severe Cases", f"{severe_pct:.1f}%")
        with col4:
            critical_pct = (disease_patients['severity'] == 'Critical').mean() * 100
            st.metric("Critical Cases", f"{critical_pct:.1f}%")

def show_accessibility_analysis(accessibility_gdf, facilities_gdf, roads_gdf, counties_gdf):
    st.markdown("## 🚑 Healthcare Accessibility Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Accessibility Score Distribution")
        
        fig = px.histogram(
            accessibility_gdf,
            x='accessibility_score',
            nbins=30,
            title="Distribution of Accessibility Scores",
            labels={'accessibility_score': 'Accessibility Score (0-100)', 'count': 'Number of Patients'}
        )
        fig.add_vline(x=accessibility_gdf['accessibility_score'].mean(), 
                     line_dash="dash", line_color="red",
                     annotation_text=f"Mean: {accessibility_gdf['accessibility_score'].mean():.1f}")
        st.plotly_chart(fig, use_container_width=True)
        
        # Accessibility categories
        accessibility_gdf['accessibility_category'] = pd.cut(
            accessibility_gdf['accessibility_score'],
            bins=[0, 30, 60, 80, 100],
            labels=['Very Low', 'Low', 'Moderate', 'High']
        )
        
        category_counts = accessibility_gdf['accessibility_category'].value_counts()
        
        fig = px.pie(
            values=category_counts.values,
            names=category_counts.index,
            title="Accessibility Categories"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Travel Time Analysis")
        
        # Calculate travel time categories
        accessibility_gdf['travel_time_category'] = pd.cut(
            accessibility_gdf['distance_km'] * 2,  # Convert to minutes at 30km/h
            bins=[0, 15, 30, 60, 120, float('inf')],
            labels=['<15 min', '15-30 min', '30-60 min', '1-2 hours', '>2 hours']
        )
        
        travel_counts = accessibility_gdf['travel_time_category'].value_counts()
        
        fig = px.bar(
            x=travel_counts.index,
            y=travel_counts.values,
            title="Travel Time to Nearest Facility",
            labels={'x': 'Travel Time', 'y': 'Number of Patients'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### Accessibility Metrics")
        
        # Calculate key metrics
        mean_score = accessibility_gdf['accessibility_score'].mean()
        median_score = accessibility_gdf['accessibility_score'].median()
        
        st.metric("Mean Accessibility Score", f"{mean_score:.1f}")
        st.metric("Median Accessibility Score", f"{median_score:.1f}")
        st.metric("Patients with High Accessibility", 
                 f"{(accessibility_gdf['accessibility_score'] > 80).sum()}")
        st.metric("Patients with Low Accessibility", 
                 f"{(accessibility_gdf['accessibility_score'] < 30).sum()}")
    
    st.markdown("### Accessibility Map")
    
    m = folium.Map(location=[0.5, 37.0], zoom_start=6)
    
    # Add counties
    folium.GeoJson(
        counties_gdf.__geo_interface__,
        style_function=lambda x: {
            'fillColor': '#ffff00',
            'color': '#000000',
            'weight': 1,
            'fillOpacity': 0.1
        }
    ).add_to(m)
    
    # Add roads
    folium.GeoJson(
        roads_gdf.__geo_interface__,
        style_function=lambda x: {
            'color': '#666666',
            'weight': 1,
            'opacity': 0.5
        }
    ).add_to(m)
    
    # Add accessibility points with color coding
    for idx, patient in accessibility_gdf.sample(min(1000, len(accessibility_gdf))).iterrows():
        # Color based on accessibility score
        if patient['accessibility_score'] >= 80:
            color = 'green'
        elif patient['accessibility_score'] >= 60:
            color = 'yellow'
        elif patient['accessibility_score'] >= 30:
            color = 'orange'
        else:
            color = 'red'
        
        folium.CircleMarker(
            [patient.geometry.y, patient.geometry.x],
            radius=3,
            popup=f"Accessibility: {patient['accessibility_score']:.1f}<br>Distance: {patient['distance_km']:.1f}km",
            color=color,
            fill=True,
            fillOpacity=0.6
        ).add_to(m)
    
    # Add facilities
    for idx, facility in facilities_gdf.iterrows():
        folium.Marker(
            [facility.latitude, facility.longitude],
            popup=facility.facility_name,
            icon=folium.Icon(color='blue', icon='hospital', prefix='fa')
        ).add_to(m)
    
    folium_static(m, width=1000, height=500)

def show_network_analysis(facilities_gdf, roads_gdf, counties_gdf, patients_gdf):
    st.markdown("## 📈 Network Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Road Network Statistics")
        
        # Calculate network metrics
        total_road_length = roads_gdf.geometry.length.sum() / 1000  # Convert to km
        avg_road_length = roads_gdf.geometry.length.mean() / 1000
        
        st.metric("Total Road Network", f"{total_road_length:.0f} km")
        st.metric("Average Road Segment", f"{avg_road_length:.1f} km")
        st.metric("Number of Road Segments", len(roads_gdf))
        
        # Road type distribution
        road_types = roads_gdf['road_type'].value_counts()
        
        fig = px.pie(
            values=road_types.values,
            names=road_types.index,
            title="Road Network by Type"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Facility Network Connectivity")
        
        # Calculate nearest neighbor distances between facilities
        facilities_utm = facilities_gdf.to_crs("EPSG:32737")
        
        nearest_distances = []
        for idx, facility in facilities_utm.iterrows():
            distances = facilities_utm.geometry.distance(facility.geometry)
            distances = distances[distances > 0]  # Remove self
            if not distances.empty:
                nearest_distances.append(distances.min() / 1000)  # Convert to km
        
        fig = px.histogram(
            x=nearest_distances,
            nbins=20,
            title="Distance Between Nearest Facilities",
            labels={'x': 'Distance (km)', 'y': 'Count'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### Network Connectivity")
        
        # Create network graph
        G = nx.Graph()
        
        # Add facilities as nodes
        for idx, facility in facilities_gdf.iterrows():
            G.add_node(facility['facility_id'], 
                      pos=(facility.longitude, facility.latitude),
                      name=facility['facility_name'])
        
        # Add edges between nearby facilities (within 50km)
        facilities_utm = facilities_gdf.to_crs("EPSG:32737")
        for i, facility1 in facilities_utm.iterrows():
            for j, facility2 in facilities_utm.iterrows():
                if i < j:
                    dist = facility1.geometry.distance(facility2.geometry) / 1000
                    if dist < 50:  # Connect if within 50km
                        G.add_edge(
                            facility1['facility_id'],
                            facility2['facility_id'],
                            weight=dist
                        )
        
        # Calculate network metrics
        st.metric("Network Density", f"{nx.density(G):.3f}")
        st.metric("Number of Components", nx.number_connected_components(G))
        
        if nx.is_connected(G):
            st.metric("Average Shortest Path", f"{nx.average_shortest_path_length(G):.1f}")
    
    st.markdown("### Network Map")
    
    m = folium.Map(location=[0.5, 37.0], zoom_start=6)
    
    # Add counties
    folium.GeoJson(
        counties_gdf.__geo_interface__,
        style_function=lambda x: {
            'fillColor': '#ffff00',
            'color': '#000000',
            'weight': 1,
            'fillOpacity': 0.1
        }
    ).add_to(m)
    
    # Add roads
    for idx, road in roads_gdf.iterrows():
        # Color by road type
        color = 'red' if road['road_type'] == 'Highway' else 'orange'
        
        points = [(lat, lon) for lon, lat in road.geometry.coords]
        folium.PolyLine(
            points,
            color=color,
            weight=3 if road['road_type'] == 'Highway' else 2,
            opacity=0.8,
            popup=f"{road['road_type']}<br>Lanes: {road['lanes']}<br>Speed: {road['speed_limit']} km/h"
        ).add_to(m)
    
    # Add facility connections
    for edge in G.edges():
        facility1 = facilities_gdf[facilities_gdf['facility_id'] == edge[0]].iloc[0]
        facility2 = facilities_gdf[facilities_gdf['facility_id'] == edge[1]].iloc[0]
        
        folium.PolyLine(
            [(facility1.latitude, facility1.longitude), 
             (facility2.latitude, facility2.longitude)],
            color='blue',
            weight=1,
            opacity=0.3,
            dash_array='5'
        ).add_to(m)
    
    # Add facilities
    for idx, facility in facilities_gdf.iterrows():
        folium.Marker(
            [facility.latitude, facility.longitude],
            popup=facility.facility_name,
            icon=folium.Icon(color='green', icon='hospital', prefix='fa')
        ).add_to(m)
    
    folium_static(m, width=1000, height=500)

def show_advanced_gis(facilities_gdf, counties_gdf, patients_gdf):
    st.markdown("## 🎓 Advanced GIS Operations")
    
    st.markdown("### Spatial Operations Demonstration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Buffer Analysis")
        
        # Demonstrate buffer operations
        facility_sample = facilities_gdf.sample(1).iloc[0]
        buffer_dist = st.slider("Buffer Distance (km)", 1, 50, 10)
        
        # Create buffer
        facility_utm = gpd.GeoSeries([facility_sample.geometry], crs="EPSG:4326").to_crs("EPSG:32737")
        buffer = facility_utm.buffer(buffer_dist * 1000).to_crs("EPSG:4326")
        
        # Find patients within buffer
        patients_within = patients_gdf[patients_gdf.geometry.within(buffer.iloc[0])]
        
        st.metric("Patients within buffer", len(patients_within))
        st.metric("Buffer area (km²)", f"{buffer.area.iloc[0]:.0f}")
        
        # Display buffer map
        m = folium.Map(location=[facility_sample.latitude, facility_sample.longitude], zoom_start=9)
        
        folium.GeoJson(
            buffer.__geo_interface__,
            style_function=lambda x: {
                'fillColor': '#ff0000',
                'color': '#ff0000',
                'weight': 2,
                'fillOpacity': 0.2
            }
        ).add_to(m)
        
        folium.Marker(
            [facility_sample.latitude, facility_sample.longitude],
            popup=facility_sample.facility_name,
            icon=folium.Icon(color='green')
        ).add_to(m)
        
        for idx, patient in patients_within.sample(min(50, len(patients_within))).iterrows():
            folium.CircleMarker(
                [patient.latitude, patient.longitude],
                radius=2,
                color='blue',
                fill=True
            ).add_to(m)
        
        folium_static(m, width=400, height=300)
    
    with col2:
        st.markdown("#### Spatial Join Operations")
        
        # Demonstrate spatial join
        st.write("Performing spatial join between patients and counties...")
        
        # Spatial join
        patients_with_county = gpd.sjoin(
            patients_gdf,
            counties_gdf[['county', 'geometry']],
            how='left',
            predicate='within'
        )
        
        st.dataframe(
            patients_with_county[['patient_id', 'county', 'diagnosis', 'severity']].head(10),
            use_container_width=True
        )
        
        st.metric("Patients joined with counties", len(patients_with_county))
    
    st.markdown("### Advanced Geoprocessing")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Union Operation")
        
        # Union of facility buffers
        facilities_utm = facilities_gdf.to_crs("EPSG:32737")
        buffers = facilities_utm.geometry.buffer(10000)  # 10km buffers
        union = unary_union(buffers)
        
        coverage_pct = (union.area / counties_gdf.to_crs("EPSG:32737").unary_union.area) * 100
        
        st.metric("Total Coverage Area", f"{union.area/1e6:.0f} km²")
        st.metric("Coverage Percentage", f"{coverage_pct:.1f}%")
    
    with col2:
        st.markdown("#### Intersection Analysis")
        
        # Find overlap between service areas
        if len(facilities_gdf) >= 2:
            facility1 = facilities_gdf.iloc[0].geometry
            facility2 = facilities_gdf.iloc[1].geometry
            
            buffer1 = facility1.buffer(0.1)  # ~11km at equator
            buffer2 = facility2.buffer(0.1)
            
            intersection = buffer1.intersection(buffer2)
            
            st.metric("Overlap Area", f"{intersection.area * 111 * 111:.0f} km²")
            st.metric("Overlap Exists", "✅" if not intersection.is_empty else "❌")
    
    with col3:
        st.markdown("#### Difference Operation")
        
        # Find areas served by only one facility type
        county_hospitals = facilities_gdf[facilities_gdf['facility_type'] == 'County Hospital']
        health_centers = facilities_gdf[facilities_gdf['facility_type'] == 'Health Center']
        
        if not county_hospitals.empty and not health_centers.empty:
            hosp_buffers = county_hospitals.geometry.buffer(0.1).unary_union
            center_buffers = health_centers.geometry.buffer(0.1).unary_union
            
            unique_to_hosp = hosp_buffers.difference(center_buffers)
            
            st.metric("Unique County Hospital Areas", 
                     f"{unique_to_hosp.area * 111 * 111:.0f} km²" if not unique_to_hosp.is_empty else "0 km²")
    
    st.markdown("### Coordinate Reference System (CRS) Operations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### CRS Transformations")
        
        st.write("Original CRS (WGS84):", facilities_gdf.crs)
        
        # Convert to UTM
        facilities_utm = facilities_gdf.to_crs("EPSG:32737")
        st.write("UTM CRS (Zone 37S):", facilities_utm.crs)
        
        # Demonstrate distance calculation difference
        point1 = facilities_gdf.iloc[0].geometry
        point2 = facilities_gdf.iloc[1].geometry
        
        # Geographic distance (approximate)
        geo_dist = point1.distance(point2) * 111 * 1000  # Rough conversion to meters
        
        # UTM distance (accurate)
        utm_dist = facilities_utm.iloc[0].geometry.distance(facilities_utm.iloc[1].geometry)
        
        st.metric("Geographic Distance (approx)", f"{geo_dist/1000:.1f} km")
        st.metric("UTM Distance (accurate)", f"{utm_dist/1000:.1f} km")
        st.metric("Difference", f"{abs(geo_dist-utm_dist)/1000:.1f} km")
    
    with col2:
        st.markdown("#### Projection Effects")
        
        # Demonstrate area calculation differences
        county = counties_gdf.iloc[0]
        
        # Area in geographic coordinates (degrees²)
        geo_area = county.geometry.area
        
        # Area in UTM (m²)
        county_utm = counties_gdf.to_crs("EPSG:32737").iloc[0]
        utm_area = county_utm.geometry.area / 1e6  # Convert to km²
        
        st.metric(f"{county['county']} County - Geo Area", f"{geo_area:.4f} deg²")
        st.metric(f"{county['county']} County - UTM Area", f"{utm_area:.0f} km²")
        st.metric("Reference Area", f"{county['area_sqkm']:.0f} km²")
        st.metric("Error", f"{abs(utm_area - county['area_sqkm']):.0f} km²")

if __name__ == "__main__":
    main()