from typing import List, Tuple, Dict
import logging
import requests
import os
import pandas as pd
from datetime import datetime
import numpy as np
from math import radians, sin, cos, sqrt, atan2
from openai import OpenAI
from dotenv import load_dotenv
import json
from scipy.spatial import KDTree
import math

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SafeRouter:
    def __init__(self):
        self.mapbox_token = os.getenv('MAPBOX_TOKEN')
        if not self.mapbox_token:
            logger.error("MAPBOX_TOKEN not found!")
            raise ValueError("Mapbox token is required")
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        # Initialize safety_insights
        self.safety_insights = {}
        
        try:
            # Load the new CSV with Brock and Alfred data
            self.crime_data = pd.read_csv('brock_alfred_incident_data.csv')
            self.process_crime_data()
            self.streetlight_data = pd.read_csv('data/kingston_streetlights.csv')
            logger.info(f"Loaded {len(self.crime_data)} crime records")
            
            # Create KDTree for efficient spatial queries
            self.crime_locations = KDTree(self.crime_data[['latitude', 'longitude']].values)
            # Normalize severity scores
            self.crime_data['severity_score'] = self.crime_data['severity'].fillna(0) ** 2  # Square severity for stronger effect
            max_severity = self.crime_data['severity_score'].max()
            self.crime_data['severity_score'] = self.crime_data['severity_score'] / max_severity if max_severity > 0 else 0
            
            # Initialize AI analysis of crime patterns
            self.analyze_crime_patterns()
            self.setup_spatial_index()
            # Create a heatmap of high-risk areas
            self.create_risk_zones()
            self.high_risk_locations = self.identify_high_risk_areas()
            self.initialize_safe_corridors()
        except Exception as e:
            logger.error(f"Error loading crime data: {e}")
            self.crime_data = pd.DataFrame()
            self.streetlight_data = pd.DataFrame()
            self.safety_insights = {
                "high_risk_patterns": [],
                "safe_times": [],
                "avoid_areas": [],
                "recommendations": ["No crime data available"]
            }

    def process_crime_data(self):
        """Process crime data with enhanced context"""
        if not self.crime_data.empty:
            self.risk_summary = self.crime_data.groupby('location').agg({
                'description': ' '.join,
                'severity': 'max',
                'time': lambda x: ', '.join(x.unique())
            }).to_dict('index')
            
            # Create weighted risk zones
            self.risk_zones = {}
            for loc, data in self.risk_summary.items():
                coords = self.get_location_coordinates(loc)
                if coords:
                    self.risk_zones[loc] = {
                        'coordinates': coords,
                        'severity': data['severity'],
                        'radius': 0.002,  # About 200 meters
                        'times': data['time']
                    }

    def initialize_safe_corridors(self):
        """Define known safe corridors and alternative routes"""
        self.safe_corridors = {
            'north_south': [
                {
                    'name': 'Alfred Street (Safe Section)',
                    'coords': [(44.2315, -76.4935), (44.2295, -76.4935)],
                    'preference': 0.8
                },
                {
                    'name': 'University Avenue',
                    'coords': [(44.2315, -76.4955), (44.2295, -76.4955)],
                    'preference': 0.9
                }
            ],
            'east_west': [
                {
                    'name': 'Colborne Street',
                    'coords': [(44.2315, -76.4925), (44.2315, -76.4945)],
                    'preference': 1.0
                },
                {
                    'name': 'Johnson Street',
                    'coords': [(44.2285, -76.4925), (44.2285, -76.4945)],
                    'preference': 0.9
                }
            ]
        }

    def get_location_coordinates(self, location: str) -> Tuple[float, float]:
        """Get coordinates for a location using Mapbox Geocoding"""
        try:
            url = f"https://api.mapbox.com/geocoding/v5/mapbox.places/{location}, Kingston, ON.json"
            params = {
                'access_token': self.mapbox_token,
                'limit': 1
            }
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                if data['features']:
                    coords = data['features'][0]['center']
                    return (coords[1], coords[0])
        except Exception as e:
            print(f"Error geocoding location: {e}")
        return None

    def analyze_crime_patterns(self):
        """Use OpenAI to analyze crime patterns and create safety heuristics"""
        try:
            # Prepare crime data summary
            if not self.crime_data.empty:
                crime_summary = self.crime_data.groupby('incident_type').size().to_dict()
                # Ensure we have a time column, if not use a default summary
                if 'time' in self.crime_data.columns:
                    time_patterns = self.crime_data['time'].value_counts().head(5).to_dict()
                else:
                    time_patterns = {"No time data available": 0}
            else:
                crime_summary = {"No crime data available": 0}
                time_patterns = {"No time data available": 0}
            
            prompt = f"""
            Analyze these crime patterns in Kingston and create safety heuristics:
            
            Crime Types and Frequencies:
            {crime_summary}
            
            Time Patterns:
            {time_patterns}
            
            Based on this data, provide:
            1. High-risk areas or patterns
            2. Safest times for travel
            3. Areas to avoid
            4. Safety recommendations
            
            Respond with a JSON object containing these keys: high_risk_patterns, safe_times, avoid_areas, recommendations
            """

            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a safety analysis expert. Provide specific, actionable insights in JSON format."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Parse the JSON response
            try:
                self.safety_insights = json.loads(response.choices[0].message.content)
            except json.JSONDecodeError:
                logger.error("Failed to parse AI response as JSON")
                self.safety_insights = {
                    "high_risk_patterns": [],
                    "safe_times": [],
                    "avoid_areas": [],
                    "recommendations": ["Error analyzing crime patterns"]
                }
            
            logger.info("Successfully generated AI safety insights")
            
        except Exception as e:
            logger.error(f"Error in AI analysis: {e}")
            self.safety_insights = {
                "high_risk_patterns": [],
                "safe_times": [],
                "avoid_areas": [],
                "recommendations": ["Error analyzing crime patterns"]
            }

    def get_ai_safety_recommendation(self, lat: float, lon: float, time: str) -> float:
        """Get AI-based safety recommendation for a specific location and time"""
        try:
            # Get nearby crimes
            nearby_crimes = self.get_nearby_crimes(lat, lon)
            
            if nearby_crimes.empty:
                return 1.0
                
            crime_summary = nearby_crimes.to_dict('records')
            
            prompt = f"""
            Analyze the safety of this location:
            Coordinates: {lat}, {lon}
            Time: {time}
            
            Nearby incidents:
            {crime_summary}
            
            Previous analysis:
            {json.dumps(self.safety_insights)}
            
            Rate the safety from 0 to 1 and explain why.
            Respond with a JSON object containing these keys: safety_score, explanation
            """

            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a safety analysis expert. Provide numerical safety scores and explanations in JSON format."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            try:
                result = json.loads(response.choices[0].message.content)
                return float(result['safety_score'])
            except (json.JSONDecodeError, KeyError, ValueError):
                logger.error("Failed to parse AI safety recommendation")
                return 0.5
            
        except Exception as e:
            logger.error(f"Error in AI safety recommendation: {e}")
            return 0.5

    def calculate_segment_safety(self, coord: List[float], time_of_day: str) -> Dict:
        """Calculate comprehensive safety score for a route segment"""
        try:
            lon, lat = coord
            nearby_crimes = self.get_nearby_crimes(lat, lon)
            
            # Base safety score starts at 0.85 (85%)
            base_safety_score = 0.85
            risk_factors = []
            
            # Time-based factors
            hour = int(time_of_day.split(':')[0])
            is_night = hour < 6 or hour > 20
            time_multiplier = 0.8 if is_night else 1.0
            
            if not nearby_crimes.empty:
                # Calculate crime impact
                crime_count = len(nearby_crimes)
                
                # More severe safety reduction for higher crime counts
                crime_penalty = min(0.5, crime_count * 0.15)  # Each crime reduces safety by 15%, max 50%
                
                # Weight recent crimes more heavily
                if 'date' in nearby_crimes.columns:
                    recent_crimes = nearby_crimes[
                        pd.to_datetime(nearby_crimes['date']) > (datetime.now() - pd.Timedelta(days=30))
                    ]
                    if not recent_crimes.empty:
                        crime_penalty += 0.1  # Additional penalty for recent crimes
                        risk_factors.append("Recent criminal activity in area")
                
                # Crime type severity weighting
                if 'type' in nearby_crimes.columns:
                    crime_types = nearby_crimes['type'].value_counts()
                    for crime_type, count in crime_types.items():
                        severity_weight = self.get_crime_severity_weight(crime_type)
                        crime_penalty += (count * severity_weight)
                        risk_factors.append(f"{count} incidents of {crime_type} nearby")
                
                # Apply time-based factors
                if is_night:
                    risk_factors.append("Reduced visibility at night")
                    if crime_count > 0:
                        crime_penalty *= 1.2  # 20% worse at night
                
                # Calculate final safety score
                safety_score = max(0.2, min(1.0, base_safety_score - crime_penalty)) * time_multiplier
                
                logger.debug(f"Location {lat}, {lon} - Safety: {safety_score:.2f}, Crimes: {crime_count}")
            else:
                # No crimes nearby, but still consider time of day
                safety_score = base_safety_score * time_multiplier
                risk_factors.append("No recorded incidents nearby")
            
            return {
                'safety_score': safety_score,
                'risk_factors': risk_factors
            }
            
        except Exception as e:
            logger.error(f"Error calculating segment safety: {e}")
            return {'safety_score': 0.5, 'risk_factors': ['Error in calculation']}

    def get_crime_severity_weight(self, crime_type: str) -> float:
        """
        Return severity weight based on crime type
        Higher weights mean more severe crimes
        """
        severity_weights = {
            'assault': 0.15,
            'robbery': 0.12,
            'theft': 0.08,
            'vandalism': 0.05,
            'disturbance': 0.03,
            # Add more crime types as needed
        }
        
        # Case-insensitive lookup with default weight
        for key, weight in severity_weights.items():
            if key.lower() in crime_type.lower():
                return weight
        return 0.05  # Default weight for unknown crime types

    def find_safe_route(self, start: Tuple[float, float], end: Tuple[float, float], 
                       time_of_day: str = None) -> Dict:
        """Enhanced safe route finding with multiple strategies"""
        try:
            # Get initial route options
            direct_routes = self.get_route_alternatives(start, end)
            
            # Try direct routes first
            for route in direct_routes:
                is_safe, alternative, reason, recommendations = self.analyze_route_safety(
                    route['geometry']['coordinates'], 
                    time_of_day
                )
                
                if is_safe:
                    return {
                        'success': True,
                        'route': {
                            'geometry': route['geometry']
                        },
                        'duration': route.get('duration', 0),
                        'distance': route.get('distance', 0),
                        'safety_analysis': reason,
                        'recommendations': recommendations
                    }
            
            # If no safe direct route, try safe corridors
            safe_route = self.find_route_via_safe_corridors(start, end)
            if safe_route:
                return safe_route
            
            # Final fallback to northern bypass
            return self.get_route_via_street(start, end, 44.2315, -76.4935)

        except Exception as e:
            print(f"Routing error: {str(e)}")
            return self.get_route_via_street(start, end, 44.2315, -76.4935)

    def find_route_via_safe_corridors(self, start: Tuple[float, float], end: Tuple[float, float]) -> Dict:
        """Find route using pre-defined safe corridors"""
        best_route = None
        min_distance = float('inf')
        
        for direction in self.safe_corridors.values():
            for corridor in direction:
                try:
                    # Create route through safe corridor
                    waypoints = [
                        f"{start[1]},{start[0]}",
                        f"{corridor['coords'][0][1]},{corridor['coords'][0][0]}",
                        f"{corridor['coords'][1][1]},{corridor['coords'][1][0]}",
                        f"{end[1]},{end[0]}"
                    ]
                    
                    route = self.get_route_through_waypoints(waypoints)
                    if route and route.get('distance', float('inf')) < min_distance:
                        min_distance = route.get('distance', float('inf'))
                        best_route = route
                        best_route['safety_analysis'] = f"Route via {corridor['name']} (safe corridor)"
                
                except Exception as e:
                    print(f"Error finding route via corridor: {e}")
                    continue
        
        return best_route

    def get_route_through_waypoints(self, waypoints: list) -> Dict:
        """Get a route through a series of waypoints"""
        try:
            url = f"https://api.mapbox.com/directions/v5/mapbox/walking/{';'.join(waypoints)}"
            params = {
                'access_token': self.mapbox_token,
                'geometries': 'geojson',
                'steps': 'true'
            }
            
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                if 'routes' in data and data['routes']:
                    return {
                        'success': True,
                        'route': {
                            'geometry': data['routes'][0]['geometry']
                        },
                        'duration': data['routes'][0].get('duration', 0),
                        'distance': data['routes'][0].get('distance', 0)
                    }
        except Exception as e:
            print(f"Error getting route through waypoints: {e}")
        return None

    def get_route_via_street(self, start: Tuple[float, float], end: Tuple[float, float], 
                           waypoint_lat: float, waypoint_lon: float) -> Dict:
        """Get a route through a specific street waypoint"""
        try:
            coords = f"{start[1]},{start[0]};{waypoint_lon},{waypoint_lat};{end[1]},{end[0]}"
            
            url = f"https://api.mapbox.com/directions/v5/mapbox/walking/{coords}"
            params = {
                'access_token': self.mapbox_token,
                'geometries': 'geojson',
                'steps': 'true'
            }
            
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                if 'routes' in data and len(data['routes']) > 0:
                    return {
                        'success': True,
                        'route': {
                            'geometry': data['routes'][0]['geometry']
                        },
                        'duration': data['routes'][0].get('duration', 0),
                        'distance': data['routes'][0].get('distance', 0),
                        'safety_analysis': "Route modified to use safer streets."
                    }
        except Exception as e:
            print(f"Error getting route via street: {e}")
        
        return {
            'success': True,
            'route': {
                'geometry': {
                    'type': 'LineString',
                    'coordinates': [[start[1], start[0]], [waypoint_lon, waypoint_lat], [end[1], end[0]]]
                }
            },
            'duration': 0,
            'distance': 0,
            'safety_analysis': "Using fallback safe route."
        }

    def get_route_alternatives(self, start: Tuple[float, float], end: Tuple[float, float]) -> list:
        """Get multiple route options from Mapbox"""
        try:
            url = f"https://api.mapbox.com/directions/v5/mapbox/walking/{start[1]},{start[0]};{end[1]},{end[0]}"
            params = {
                'access_token': self.mapbox_token,
                'geometries': 'geojson',
                'alternatives': 'true',
                'steps': 'true',
                'overview': 'full'
            }
            
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                if 'routes' in data:
                    return data['routes']
        except Exception as e:
            print(f"Error getting route alternatives: {e}")
        return []

    def analyze_route_safety(self, route_coords: list, time_of_day: str = None) -> tuple:
        """Enhanced route safety analysis considering time and context"""
        try:
            route_description = self.convert_route_to_description(route_coords)
            current_time = time_of_day or datetime.now().strftime("%H:%M")
            
            # Create detailed safety context
            safety_context = self.create_safety_context(route_coords, current_time)
            
            prompt = f"""
            Analyze this Kingston walking route with detailed safety considerations:

            Route: {route_description}
            Time: {current_time}

            Safety Context:
            {safety_context}

            Known Safe Alternatives:
            - Colborne Street (preferred for east-west travel)
            - Johnson Street (good alternative for east-west)
            - University Avenue (preferred for north-south travel)
            - Division Street (good alternative for north-south)

            Requirements:
            1. Completely avoid high-risk intersections
            2. Consider time of day in route selection
            3. Prefer well-lit main streets
            4. Suggest specific alternative streets if needed

            Analyze and respond in this format:
            SAFE: true/false
            ALTERNATIVE: [specific street-by-street directions if needed]
            REASON: [detailed explanation]
            RECOMMENDATIONS: [additional safety tips]
            """

            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert in Kingston safety and navigation, with detailed knowledge of street patterns and safety considerations."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            return self.parse_enhanced_analysis(response.choices[0].message.content)

        except Exception as e:
            print(f"Error in route analysis: {e}")
            return False, "Colborne Street", "Error analyzing route", []

    def create_safety_context(self, route_coords: list, time: str) -> str:
        """Create detailed safety context for the route"""
        context = []
        for loc, zone in self.risk_zones.items():
            for coord in route_coords:
                if self.is_point_in_zone(coord, zone['coordinates'], zone['radius']):
                    context.append(f"""
                    Risk Area: {loc}
                    Severity: {zone['severity']}/10
                    Active Times: {zone['times']}
                    Current Time: {time}
                    Distance from route: {self.calculate_distance(coord, zone['coordinates'])}m
                    """)
        return "\n".join(context)

    def is_point_in_zone(self, point: list, center: Tuple[float, float], radius: float) -> bool:
        """Check if a point is within a risk zone"""
        return abs(point[1] - center[0]) < radius and abs(point[0] - center[1]) < radius

    def calculate_distance(self, point1: list, point2: Tuple[float, float]) -> float:
        """Calculate distance between two points in meters"""
        return int(np.sqrt(
            (point1[1] - point2[0])**2 + 
            (point1[0] - point2[1])**2
        ) * 111000)  # Convert to meters

    def parse_enhanced_analysis(self, analysis: str) -> tuple:
        """Parse the OpenAI response into usable components"""
        try:
            is_safe = "SAFE: true" in analysis.lower()
            alternative = ""
            reason = ""
            recommendations = []

            if "ALTERNATIVE:" in analysis:
                alternative = analysis.split("ALTERNATIVE:")[1].split("REASON:")[0].strip()
            if "REASON:" in analysis:
                reason = analysis.split("REASON:")[1].strip()
            if "RECOMMENDATIONS:" in analysis:
                recommendations = analysis.split("RECOMMENDATIONS:")[1].strip().split(", ")

            return is_safe, alternative, reason, recommendations
        except:
            return False, "Colborne Street", "Error parsing analysis", []

    def convert_route_to_description(self, coords: list) -> str:
        """Convert coordinates to human-readable street descriptions"""
        try:
            descriptions = []
            for coord in coords[::10]:  # Sample every 10th coordinate to reduce API calls
                url = f"https://api.mapbox.com/geocoding/v5/mapbox.places/{coord[0]},{coord[1]}.json"
                params = {
                    'access_token': self.mapbox_token,
                    'types': 'address'
                }
                
                response = requests.get(url, params=params)
                if response.status_code == 200:
                    data = response.json()
                    if 'features' in data and data['features']:
                        descriptions.append(data['features'][0]['place_name'])
            
            return " → ".join(descriptions)
        except Exception as e:
            print(f"Error converting coordinates to description: {e}")
            return str(coords)

    def setup_spatial_index(self):
        if not self.crime_data.empty:
            try:
                self.crime_locations = KDTree(self.crime_data[['latitude', 'longitude']].values)
                self.crime_data['severity_score'] = self.crime_data['severity'].fillna(0) ** 2
                max_severity = self.crime_data['severity_score'].max()
                self.crime_data['severity_score'] = self.crime_data['severity_score'] / max_severity if max_severity > 0 else 0
            except Exception as e:
                print(f"Error setting up spatial index: {e}")

    def create_risk_zones(self):
        """Create zones of high risk based on crime density and severity"""
        if not self.crime_data.empty:
            # Group incidents by location and calculate risk score
            self.risk_areas = self.crime_data.groupby(['latitude', 'longitude']).agg({
                'severity': 'sum'
            }).reset_index()
            
            # Normalize severity scores
            max_severity = self.risk_areas['severity'].max()
            self.risk_areas['risk_score'] = self.risk_areas['severity'] / max_severity if max_severity > 0 else 0
            
            # Create KDTree for risk zones
            self.risk_zone_tree = KDTree(self.risk_areas[['latitude', 'longitude']].values)

    def get_safe_waypoints(self, start: Tuple[float, float], end: Tuple[float, float]) -> list:
        """Generate waypoints to avoid high-risk areas"""
        # Calculate midpoint
        mid_lat = (start[0] + end[0]) / 2
        mid_lon = (start[1] + end[1]) / 2
        
        # Calculate perpendicular offset
        dx = end[1] - start[1]
        dy = end[0] - start[0]
        distance = math.sqrt(dx*dx + dy*dy)
        
        # Create offset points (both left and right of the direct path)
        offset = 0.005  # Approximately 500 meters
        perpendicular_lat = [-dy * offset / distance, dy * offset / distance]
        perpendicular_lon = [dx * offset / distance, -dx * offset / distance]
        
        waypoints = []
        for i in range(2):
            point = (
                mid_lat + perpendicular_lat[i],
                mid_lon + perpendicular_lon[i]
            )
            if not self.is_high_risk_area(point):
                waypoints.append(point)
        
        return waypoints

    def is_high_risk_area(self, point: Tuple[float, float], buffer: float = 0.002) -> bool:
        """Check if a point is within a high-risk area (including buffer zone)"""
        for risk_point in self.high_risk_locations:
            if abs(point[0] - risk_point[0]) < buffer and abs(point[1] - risk_point[1]) < buffer:
                return True
        return False

    def identify_high_risk_areas(self) -> list:
        """Identify areas with high concentration of severe incidents"""
        if self.crime_data.empty:
            return []
            
        high_risk = self.risk_areas[self.risk_areas['risk_score'] > 0.7]  # Adjust threshold as needed
        return high_risk[['latitude', 'longitude']].values.tolist()

    def get_nearby_crimes(self, lat: float, lon: float, radius: float = 0.003) -> list:
        """Get crime incidents near a point"""
        if hasattr(self, 'crime_locations'):
            nearby_indices = self.crime_locations.query_ball_point([lat, lon], radius)
            return [
                {
                    'type': self.crime_data.iloc[i]['incident_type'],
                    'severity': self.crime_data.iloc[i]['severity'],
                    'time': self.crime_data.iloc[i]['time_of_day'],
                    'location': f"{self.crime_data.iloc[i]['latitude']}, {self.crime_data.iloc[i]['longitude']}"
                }
                for i in nearby_indices
            ]
        return []

    def get_streetlight_coverage(self, lat: float, lon: float, radius_km: float = 0.1) -> float:
        """Calculate streetlight coverage score"""
        if self.streetlight_data.empty:
            return 0.5

        nearby_lights = 0
        for _, light in self.streetlight_data.iterrows():
            distance = self.calculate_distance(lat, lon, 
                                            light['latitude'], 
                                            light['longitude'])
            if distance <= radius_km:
                nearby_lights += 1

        # Score based on number of nearby lights
        return min(1.0, nearby_lights / 5)  # Normalize to 0-1

    def calculate_safety_score(self, lat: float, lon: float, radius: float = 0.005) -> float:
        """Calculate safety score for a given point based on nearby incidents"""
        # Find incidents within radius
        nearby_indices = self.crime_locations.query_ball_point([lat, lon], radius)
        
        if not nearby_indices:
            return 1.0  # Maximum safety if no incidents nearby
        
        # Calculate weighted score based on incidents and distance
        total_weight = 0
        for idx in nearby_indices:
            distance = np.sqrt(
                (lat - self.crime_data.iloc[idx]['latitude'])**2 +
                (lon - self.crime_data.iloc[idx]['longitude'])**2
            )
            # Inverse distance weighting
            weight = self.crime_data.iloc[idx]['severity_score'] * (1 - distance/radius)
            total_weight += weight
        
        # Convert to safety score (inverse of risk)
        safety_score = 1 / (1 + total_weight)
        return safety_score

    def get_address_from_coords(self, lat: float, lon: float) -> str:
        """Convert coordinates to human-readable address using Mapbox reverse geocoding"""
        try:
            url = f"https://api.mapbox.com/geocoding/v5/mapbox.places/{lon},{lat}.json"
            params = {
                'access_token': self.mapbox_token,
                'types': 'address,poi',  # Include both addresses and points of interest
                'limit': 1,
                'country': 'ca',  # Limit to Canadian addresses
                'proximity': f"{lon},{lat}"  # Prioritize nearby results
            }
            
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                if data.get('features') and len(data['features']) > 0:
                    feature = data['features'][0]
                    
                    # Extract the relevant parts of the address
                    place_parts = feature['place_name'].split(', ')
                    
                    # Check if it's a POI (Point of Interest)
                    if feature.get('properties', {}).get('category'):
                        # For POIs, use the name and the street
                        return f"{place_parts[0]} ({place_parts[1]})"
                    else:
                        # For regular addresses, use street number and name
                        address = place_parts[0]
                        # Add neighborhood or area if available
                        if len(place_parts) > 2:
                            address += f" in {place_parts[1]}"
                        return address
                
            logger.warning(f"Could not get address for coordinates: {lat}, {lon}")
            return "location"  # Generic fallback
        except Exception as e:
            logger.error(f"Error in reverse geocoding: {e}")
            return "location"  # Generic fallback

    def chat_with_context(self, user_message: str) -> str:
        """Enhanced chat function with route context using addresses"""
        try:
            logger.info("Starting chat with context")
            
            # Build context about the current route and area
            context_parts = []
            
            # Add route information if available
            if self.current_route.get('start') and self.current_route.get('end'):
                try:
                    start_lat, start_lon = self.current_route['start']
                    end_lat, end_lon = self.current_route['end']
                    
                    # Convert coordinates to addresses with better error handling
                    start_address = self.get_address_from_coords(start_lat, start_lon)
                    end_address = self.get_address_from_coords(end_lat, end_lon)
                    
                    route_description = f"The user is planning to walk from {start_address} to {end_address} in Kingston."
                    context_parts.append(route_description)
                    logger.debug(f"Route description: {route_description}")
                    
                    # Add safety insights for the route
                    try:
                        start_safety = self.calculate_safety_score(start_lat, start_lon)
                        end_safety = self.calculate_safety_score(end_lat, end_lon)
                        safety_info = f"The {start_address} area has a safety score of {start_safety:.1%} and the {end_address} area has a safety score of {end_safety:.1%}."
                        context_parts.append(safety_info)
                    except Exception as e:
                        logger.error(f"Error calculating safety scores: {e}")
                except Exception as e:
                    logger.error(f"Error processing route addresses: {e}")
                    context_parts.append("The user is planning a route in Kingston.")

            # Add general safety insights
            try:
                if isinstance(self.safety_insights, dict):
                    context_parts.append("\nRelevant safety information:")
                    if self.safety_insights.get('high_risk_patterns'):
                        context_parts.append("- High risk patterns: " + "; ".join(self.safety_insights['high_risk_patterns']))
                    if self.safety_insights.get('safe_times'):
                        context_parts.append("- Safe times: " + "; ".join(self.safety_insights['safe_times']))
            except Exception as e:
                logger.error(f"Error processing safety insights: {e}")

            # Build the final context
            context = "\n".join(context_parts) if context_parts else "No specific route information available."

            prompt = f"""As a safety assistant for Kingston, Ontario, help the user with their question.

Available Context:
{context}

User Question: {user_message}

Provide specific, relevant safety advice for Kingston. If the user has a planned route, include specific recommendations for that journey, mentioning specific street names and landmarks when possible."""

            logger.debug(f"Sending prompt to OpenAI: {prompt}")

            try:
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {
                            "role": "system", 
                            "content": "You are a knowledgeable safety assistant for Kingston, Ontario, specializing in pedestrian safety and local area knowledge. Provide specific advice using street names and landmarks."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=200
                )
                
                ai_response = response.choices[0].message.content
                logger.info("Successfully received response from OpenAI")
                return ai_response

            except Exception as e:
                logger.error(f"OpenAI API error: {e}")
                raise

        except Exception as e:
            logger.error(f"Error in chat_with_context: {str(e)}", exc_info=True)
            return "I'm here to help with safety advice in Kingston. What would you like to know?" 