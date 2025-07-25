from flask import Flask, jsonify, request
from flask_cors import CORS
from starlette.middleware.wsgi import WSGIMiddleware
import json
from datetime import datetime, timedelta
import random

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Wrap the Flask app with WSGIMiddleware
asgi_app = WSGIMiddleware(app)

# Mock data for different time series
time_series_data = {
    "Inflation": {
        "description": "Inflation rate over time",
        "data": [
            {
                "date": (datetime.now() - timedelta(days=i*30)).strftime("%Y-%m-%d"),
                "value": round(random.uniform(1.5, 4.5), 1),
                "id": f"inflation_{i}",
                "short_desc": f"Inflation report for {(datetime.now() - timedelta(days=i*30)).strftime('%B %Y')}"
            } for i in range(24)
        ]
    },
    "Labour Market": {
        "description": "Unemployment rate over time",
        "data": [
            {
                "date": (datetime.now() - timedelta(days=i*30)).strftime("%Y-%m-%d"),
                "value": round(random.uniform(3.0, 6.5), 1),
                "id": f"labour_{i}",
                "short_desc": f"Labour market statistics for {(datetime.now() - timedelta(days=i*30)).strftime('%B %Y')}"
            } for i in range(24)
        ]
    },
    "Interest Rate Guidance": {
        "description": "Central bank interest rate guidance",
        "data": [
            {
                "date": (datetime.now() - timedelta(days=i*30)).strftime("%Y-%m-%d"),
                "value": round(random.uniform(2.0, 5.0), 2),
                "id": f"interest_{i}",
                "short_desc": f"Interest rate decision for {(datetime.now() - timedelta(days=i*30)).strftime('%B %Y')}"
            } for i in range(24)
        ]
    }
}

# Define highlight categories
highlight_categories = [
    {"id": "inflation", "color": "#FFCDD2", "label": "Inflation"},
    {"id": "employment", "color": "#C8E6C9", "label": "Employment"},
    {"id": "interest rate", "color": "#BBDEFB", "label": "Interest Rate"},
    {"id": "balance sheet", "color": "#D1C4E9", "label": "Balance Sheet"},
]

# Mock detailed data for each point
detailed_data = {}

with open("/Users/dzz1th/Job/mgi/Soroka/engine/engine/src/processing/parsed_results.json", "r") as f:
    fed_transcript = json.load(f)

with open("/Users/dzz1th/Job/mgi/Soroka/data/pc_data/recent_statement_reasonings_summary.json", "r") as f:
    summary = json.load(f)

# Generate detailed data for each time series and point
for series_name, series_info in time_series_data.items():
    for point in series_info["data"]:
        point_id = point["id"]
        date_obj = datetime.strptime(point["date"], "%Y-%m-%d")
        month_year = date_obj.strftime("%B %Y")
        
        detailed_data[point_id] = {
            "title": f"{series_name} Report - {month_year}",
            "date": point["date"],
            "value": point["value"],
            "structured_summary": summary,
            "transcript": fed_transcript,
            "highlightCategories": highlight_categories
        }

@app.route('/api/time-series', methods=['GET'])
def get_time_series():
    series_name = request.args.get('name', 'Inflation')
    if series_name in time_series_data:
        return jsonify(time_series_data[series_name])
    else:
        return jsonify({"error": "Time series not found"}), 404

@app.route('/api/series-list', methods=['GET'])
def get_series_list():
    return jsonify(list(time_series_data.keys()))

@app.route('/api/point-details/<point_id>', methods=['GET'])
def get_point_details(point_id):
    if point_id in detailed_data:
        return jsonify(detailed_data[point_id])
    else:
        return jsonify({"error": "Point not found"}), 404

@app.route('/api/test', methods=['GET'])
def test():
    return jsonify({"status": "API is working!"})

if __name__ == '__main__':
    print("Starting Flask server on http://localhost:8000")
    app.run(debug=True, port=8000, host='0.0.0.0')