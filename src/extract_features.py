import fastf1
import pandas as pd
import os

def extract_features(year=2024, event='Monza', session_type='Q'):
    
    fastf1.Cache.enable_cache('cache')

    
    session = fastf1.get_session(year, event, session_type)
    session.load()

    
    laps = session.laps.pick_quicklaps()

    data = []

    for _, lap in laps.iterrows():
        if pd.isnull(lap.LapTime):
            continue

        try:
            telemetry = lap.get_car_data()
            if telemetry.empty:
                continue

            avg_throttle = telemetry['Throttle'].mean()
            avg_drs = telemetry['DRS'].mean()
            avg_speed = telemetry['Speed'].mean()

            max_fuel = 100
            fuel_per_lap = 5
            fuel_load = max_fuel - lap.LapNumber * fuel_per_lap

            data.append({
                'Driver': lap.Driver,
                'Team': lap.Team,
                'LapNumber': lap.LapNumber,
                'LapTime': lap.LapTime.total_seconds(),
                'AvgThrottle': avg_throttle,
                'AvgDRS': avg_drs,
                'FuelLoad': fuel_load,
                'AvgSpeed': avg_speed,
            })

        except Exception as e:
            print(f"Skipping lap due to error: {e}")

    
    df = pd.DataFrame(data)
    os.makedirs("data", exist_ok=True)
    df.to_csv('data/lap_features.csv', index=False)

    print("Features saved to data/lap_features.csv")
    print(f"Extracted {len(df)} laps")
    return df
