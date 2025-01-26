# Street coordinates in Kingston (approximate centers)
STREET_COORDINATES = {
    "Bagot Street": (-76.4853, 44.2307),
    "Queen Street": (-76.4800, 44.2298),
    "Johnson Street": (-76.4912, 44.2277),
    "Wellington Street": (-76.4800, 44.2321),
    "Clergy Street": (-76.4930, 44.2302),
    "Brock Street": (-76.4860, 44.2295),
    "Ontario Street": (-76.4798, 44.2291),
    "King Street": (-76.4810, 44.2285),
    "Division Street": (-76.4950, 44.2315),
    "Princess Street": (-76.4816, 44.2314)
}

# Crime severity weights (0-1, where 1 is most severe)
CRIME_SEVERITY = {
    "Homicide": 1.0,
    "Assault": 0.8,
    "Robbery": 0.7,
    "Burglary": 0.6,
    "Arson": 0.7,
    "Drug Offense": 0.5,
    "Theft": 0.4,
    "Fraud": 0.3,
    "Vandalism": 0.3,
    "Public Disturbance": 0.2
}

KINGSTON_CRIME_DATA = [
    {
        "type": "Feature",
        "properties": {
            "crime_type": "Public Disturbance",
            "date": "2021-09-09",
            "time": "18:47:27",
            "severity": CRIME_SEVERITY["Public Disturbance"],
            "street": "Bagot Street"
        },
        "geometry": {
            "type": "Point",
            "coordinates": STREET_COORDINATES["Bagot Street"]
        }
    },
    # ... more entries following the same pattern
]

def initialize_crime_data(crime_records):
    """Convert raw crime records into formatted crime data"""
    formatted_data = []
    for record in crime_records:
        if record["Street"] in STREET_COORDINATES:
            formatted_data.append({
                "type": "Feature",
                "properties": {
                    "crime_type": record["Crime Type"],
                    "date": record["Crime Date"],
                    "time": record["Crime Time"],
                    "severity": CRIME_SEVERITY.get(record["Crime Type"], 0.1),
                    "street": record["Street"]
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": STREET_COORDINATES[record["Street"]]
                }
            })
    return formatted_data

def get_nearby_crimes(location: tuple, radius: float = 0.005) -> list:
    """Get crimes within radius (in degrees) of the given location"""
    lat, lon = location
    nearby = []
    for crime in KINGSTON_CRIME_DATA:
        crime_lon, crime_lat = crime["geometry"]["coordinates"]
        if (abs(crime_lat - lat) < radius and 
            abs(crime_lon - lon) < radius):
            nearby.append(crime)
    return nearby

def calculate_area_safety(location: tuple, radius: float = 0.005) -> float:
    """Calculate safety score for an area (0-1, where 1 is safest)"""
    nearby_crimes = get_nearby_crimes(location, radius)
    if not nearby_crimes:
        return 1.0
    
    # Calculate weighted safety score
    total_severity = sum(crime["properties"]["severity"] for crime in nearby_crimes)
    # More crimes and higher severity = lower safety score
    safety_score = max(0, 1 - (total_severity / 10))  # Adjust denominator as needed
    return safety_score

# Define areas with known higher risk
HIGH_RISK_AREAS = [
    {
        "name": "Area 1",
        "center": [-76.4951, 44.2312],
        "radius": 0.2,  # kilometers
        "risk_level": 0.8  # 0 to 1, where 1 is highest risk
    },
    # Add more high-risk areas
]

def get_crime_data_for_location(lat: float, lon: float, radius: float = 0.5) -> list:
    """
    Get crime data within radius km of the given coordinates
    """
    nearby_crimes = []
    for crime in KINGSTON_CRIME_DATA:
        crime_lat = crime["geometry"]["coordinates"][1]
        crime_lon = crime["geometry"]["coordinates"][0]
        
        # Simple distance check (you might want to use a more sophisticated method)
        if (abs(crime_lat - lat) < radius/111) and (abs(crime_lon - lon) < radius/111):
            nearby_crimes.append(crime)
    
    return nearby_crimes 