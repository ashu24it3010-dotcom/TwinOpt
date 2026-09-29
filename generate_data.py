import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_machine_dataset(machine_id: str, name: str, base_rpm: float, initial_temp: float, base_vib: float, num_records: int = 1000) -> pd.DataFrame:
    np.random.seed(42 if machine_id == "M1" else (43 if machine_id == "M2" else 44))
    
    start_time = datetime.now() - timedelta(minutes=5 * num_records)
    records = []

    current_temp = initial_temp
    current_health = 1.0
    operating_hours = 0.0

    for i in range(num_records):
        timestamp = start_time + timedelta(minutes=5 * i)
        operating_hours += 5 / 60.0  # 5-minute time steps

        # Operating modes: 1.0 (Normal), 1.2 (High load), 0.8 (Eco), 0.5 (Reduced), 0.0 (Stopped)
        speed_factor = float(np.random.choice([1.0, 1.2, 0.8, 0.5, 0.0], p=[0.60, 0.15, 0.15, 0.07, 0.03]))

        if speed_factor == 0.0:
            rpm = 0.0
            power = 0.5
            current_temp = max(25.0, current_temp - 1.2)
            vib = 0.05
            status = "STOPPED"
            produced = 0
        else:
            rpm = base_rpm * speed_factor + np.random.normal(0, 8.0)
            power = 2.0 + (speed_factor ** 1.5) * 12.0 + np.random.normal(0, 0.2)

            # Thermal & mechanical stress accumulation
            thermal_increase = (speed_factor ** 1.8) * 2.0
            current_temp = min(110.0, max(30.0, current_temp + thermal_increase - 1.2 + np.random.normal(0, 0.3)))

            vib = max(0.05, (speed_factor ** 2.0) * base_vib + (1.0 - current_health) * 0.35 + np.random.normal(0, 0.015))

            # Health degradation formula
            stress = max(0.0, (current_temp - 80.0) / 1000.0) + max(0.0, (vib - 0.5) / 500.0)
            current_health = max(0.05, current_health - stress - 0.0001)

            # Assign machine status threshold
            if current_health < 0.3 or current_temp > 95.0 or vib > 0.8:
                status = "CRITICAL"
            elif current_health < 0.6 or current_temp > 80.0 or vib > 0.5:
                status = "WARNING"
            else:
                status = "NORMAL"

            produced = int(10 * speed_factor)

        records.append({
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "machine_id": machine_id,
            "machine_name": name,
            "rpm": round(rpm, 2),
            "temperature": round(current_temp, 2),
            "vibration": round(vib, 4),
            "power": round(power, 2),
            "health": round(current_health, 4),
            "status": status,
            "operating_hours": round(operating_hours, 2),
            "produced_units": produced
        })

    return pd.DataFrame(records)

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    df_m1 = generate_machine_dataset("M1", "Cutting", base_rpm=1500.0, initial_temp=65.0, base_vib=0.25)
    df_m2 = generate_machine_dataset("M2", "Processing", base_rpm=1400.0, initial_temp=68.0, base_vib=0.35)
    df_m3 = generate_machine_dataset("M3", "Assembly", base_rpm=1300.0, initial_temp=60.0, base_vib=0.20)

    df_m1.to_csv("data/machine_1.csv", index=False)
    df_m2.to_csv("data/machine_2.csv", index=False)
    df_m3.to_csv("data/machine_3.csv", index=False)

    print("Successfully generated historical telemetry data:")
    print(" - data/machine_1.csv (Cutting)")
    print(" - data/machine_2.csv (Processing)")
    print(" - data/machine_3.csv (Assembly)")