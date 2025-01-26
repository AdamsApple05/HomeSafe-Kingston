from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from safe_router import SafeRouter
import os
import openai
import logging
from datetime import datetime
from openai import OpenAI

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Load environment variables
load_dotenv()
MAPBOX_TOKEN = os.getenv('MAPBOX_TOKEN')
openai.api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def analyze_route_safety(route_data: dict) -> str:
    """Use OpenAI to analyze route safety and provide recommendations"""
    try:
        overall_safety = route_data.get('overall_safety', 0) * 100
        duration = route_data.get('duration', 0) / 60  # Convert to minutes
        distance = route_data.get('distance', 0) / 1000  # Convert to kilometers

        prompt = f"""
        Analyze this walking route in Kingston:
        - Overall safety score: {overall_safety:.1f}%
        - Duration: {duration:.1f} minutes
        - Distance: {distance:.2f} km

        Please provide:
        1. A brief safety assessment
        2. Specific recommendations for this route
        3. Time of day considerations
        Keep the response concise and practical.
        """

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a local safety expert in Kingston, Ontario."},
                {"role": "user", "content": prompt}
            ]
        )

        return response.choices[0].message.content

    except Exception as e:
        logger.error(f"Error in AI analysis: {str(e)}")
        return "Unable to generate safety analysis at this time."

@app.route('/')
def index():
    mapbox_token = os.getenv('MAPBOX_TOKEN')
    logger.debug(f"Mapbox token available: {bool(mapbox_token)}")
    return render_template('index.html', mapbox_token=mapbox_token)

@app.route('/get_route', methods=['POST'])
def get_route():
    try:
        data = request.get_json()
        logger.debug(f"Received route request with data: {data}")
        
        if not data or 'start' not in data or 'end' not in data:
            logger.error("Missing start or end coordinates in request")
            return jsonify({'error': 'Missing coordinates'}), 400

        start_coords = data['start'].split(',')
        end_coords = data['end'].split(',')
        
        logger.debug(f"Processing coordinates - Start: {start_coords}, End: {end_coords}")
        
        # Validate coordinates
        try:
            start_coords = [float(x) for x in start_coords]
            end_coords = [float(x) for x in end_coords]
        except ValueError as e:
            logger.error(f"Invalid coordinate format: {e}")
            return jsonify({'error': 'Invalid coordinate format'}), 400

        router = SafeRouter()
        route_data = router.find_safe_route(
            start=tuple(start_coords),
            end=tuple(end_coords)
        )
        
        logger.debug(f"Route data received: {route_data}")
        
        if route_data.get('success', False) and route_data.get('route'):
            return jsonify({
                'route': route_data['route'],
                'duration': route_data['duration'],
                'distance': route_data['distance']
            })
        else:
            logger.warning("No route found")
            return jsonify({'route': None}), 200
            
    except Exception as e:
        logger.error(f"Route error: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        route_data = data.get('route', {})
        
        # Initialize router with current route if available
        router = SafeRouter()
        if route_data and 'start' in route_data and 'end' in route_data:
            start_coords = [float(x) for x in route_data['start'].split(',')]
            end_coords = [float(x) for x in route_data['end'].split(',')]
            router.current_route = {
                'start': tuple(start_coords),
                'end': tuple(end_coords)
            }

        # Use the enhanced chat method that includes route context
        ai_response = router.chat_with_context(user_message)
        return jsonify({'response': ai_response})

    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        return jsonify({
            'response': "I apologize, but I'm having trouble processing your request. Please try again."
        }), 200

if __name__ == '__main__':
    # Make sure the Mapbox token is available
    if not os.getenv('MAPBOX_TOKEN'):
        logger.error("MAPBOX_TOKEN not found in environment variables!")
    
    app.run(debug=True) 