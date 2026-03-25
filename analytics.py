import sqlite3
import datetime
import os

DB_PATH = "analytics.db"

def init_db():
    """Initialize the database and create the table if it doesn't exist."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS traffic_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            direction TEXT,
            vehicle_count INTEGER,
            avg_speed REAL
        )
    ''')
    conn.commit()
    conn.close()

def save_metrics(direction_counts, avg_speed):
    """Save the current processing results to the database."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    now = datetime.datetime.now()
    for direction, count in direction_counts.items():
        c.execute('''
            INSERT INTO traffic_history (timestamp, direction, vehicle_count, avg_speed)
            VALUES (?, ?, ?, ?)
        ''', (now, direction, count, avg_speed))
    conn.commit()
    conn.close()

def get_historical_average(direction):
    """
    Get the historical average for a specific direction during the current hour.
    This helps in comparing current traffic with 'normal' traffic for this time.
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    current_hour = datetime.datetime.now().hour
    
    # Query for averages of the same hour in the past
    c.execute('''
        SELECT AVG(vehicle_count) 
        FROM traffic_history 
        WHERE direction = ? AND strftime('%H', timestamp) = ?
    ''', (direction, f"{current_hour:02d}"))
    
    res = c.fetchone()[0]
    conn.close()
    return res if res is not None else 0

def get_signal_recommendation(current_counts):
    """
    Logic to determine signal states and timings.
    Returns a dict with state (RED/GREEN) and recommended duration for each direction.
    """
    recommendations = {}
    directions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    
    for d in directions:
        curr = current_counts.get(d, 0)
        hist = get_historical_average(d)
        
        # Logic: If current traffic is 20% higher than historical average,
        # or if there are more than 5 vehicles, prioritize with a longer GREEN.
        
        duration = 30 # Default 30s
        if curr > (hist * 1.2) or curr > 5:
            state = "GREEN"
            duration = min(60, 30 + int(curr * 2)) # Dynamic extension
        else:
            state = "RED"
            duration = 20
            
        recommendations[d] = {
            "state": state,
            "duration": duration,
            "current": curr,
            "historical_avg": round(hist, 1)
        }
    
    return recommendations

if __name__ == "__main__":
    init_db()
    print("Database initialized.")
