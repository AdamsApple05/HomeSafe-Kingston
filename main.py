from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv
import mapbox
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
import requests

# Load environment variables
load_dotenv()
MAPBOX_TOKEN = os.getenv('MAPBOX_TOKEN')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

app = Flask(__name__)

class SafeRouter:
    def __init__(self):
        self.mapbox_client = mapbox.Directions()
        
    def calculate_safety_score(self, lat, lon):
        # Using a simple example with crime data from data.gov
        # You would need to replace this URL with actual crime data API
        try:
            # Example radius in meters
            radius = 500
            crime_url = f"https://data.police.uk/api/crimes-street/all-crime?lat={lat}&lng={lon}"
            response = requests.get(crime_url)
            if response.status_code == 200:
                crimes = response.json()
                # Simple scoring: more crimes = lower safety score
                # Score from 0 (unsafe) to 1 (safe)
                safety_score = max(0, 1 - (len(crimes) / 100))
                return safety_score
            return 0.5  # Default medium safety if API fails
        except Exception as e:
            print(f"Error calculating safety score: {e}")
            return 0.5
    
    def find_safe_route(self, start_coords, end_coords):
        try:
            # Get the base route from Mapbox
            response = self.mapbox_client.directions([
                start_coords,
                end_coords
            ], profile='walking')
            
            route_data = response.json()
            
            if 'routes' not in route_data or not route_data['routes']:
                return {"error": "No route found"}
                
            # Get the main route
            main_route = route_data['routes'][0]
            
            # Calculate safety scores for key points along the route
            coordinates = main_route['geometry']['coordinates']
            safety_scores = []
            
            # Sample safety scores every few points
            for coord in coordinates[::5]:  # Check every 5th coordinate to reduce API calls
                safety_score = self.calculate_safety_score(coord[1], coord[0])
                safety_scores.append(safety_score)
            
            # Calculate average safety score for the route
            avg_safety_score = sum(safety_scores) / len(safety_scores) if safety_scores else 0
            
            return {
                "route": main_route,
                "safety_score": avg_safety_score,
                "message": f"Route found with safety score: {avg_safety_score:.2f}"
            }
            
        except Exception as e:
            return {"error": str(e)}

@app.route('/')
def index():
    return render_template('index.html', mapbox_token=MAPBOX_TOKEN)

@app.route('/get_route', methods=['POST'])
def get_route():
    data = request.get_json()
    start = data.get('start')
    end = data.get('end')
    
    if not start or not end:
        return jsonify({"error": "Missing start or end coordinates"}), 400
    
    router = SafeRouter()
    safe_route = router.find_safe_route(start, end)
    
    return jsonify(safe_route)

if __name__ == '__main__':
    app.run(debug=True)
