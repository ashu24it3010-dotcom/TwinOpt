import numpy as np
from typing import Dict, Any

class Machine:
    def __init__(self, machine_id: str, name: str, base_rpm: float = 1500.0, initial_temp: float = 65.0, base_vib: float = 0.25):
        self.machine_id = machine_id
        self.name = name
        self.base_rpm = base_rpm
        self.temperature = initial_temp
        self.vibration = base_vib
        self.base_vib = base_vib
        self.power = 2.0
        self.health = 1.0
        self.status = "NORMAL"
        self.operating_hours = 0.0
        self.produced_units = 0
        self.rpm = base_rpm

    def step(self, speed_factor: float = 1.0) -> Dict[str, Any]:
        self.operating_hours += 5 / 60.0  # 5-minute time step

        if speed_factor <= 0.0:
            self.rpm = 0.0
            self.power = 0.5
            self.temperature = max(25.0, self.temperature - 1.2)
            self.vibration = 0.05
            self.status = "STOPPED"
            units_step = 0
        else:
            self.rpm = max(0.0, self.base_rpm * speed_factor + np.random.normal(0, 5.0))
            self.power = max(0.5, 2.0 + (speed_factor ** 1.5) * 12.0 + np.random.normal(0, 0.2))

            # Thermal and mechanical dynamics
            thermal_increase = (speed_factor ** 1.8) * 2.0
            self.temperature = min(110.0, max(25.0, self.temperature + thermal_increase - 1.2 + np.random.normal(0, 0.2)))

            self.vibration = max(0.05, (speed_factor ** 2.0) * self.base_vib + (1.0 - self.health) * 0.35 + np.random.normal(0, 0.01))

            # Health degradation
            stress = max(0.0, (self.temperature - 80.0) / 1000.0) + max(0.0, (self.vibration - 0.5) / 500.0)
            self.health = max(0.05, self.health - stress - 0.0001)

            # Update status
            if self.health < 0.3 or self.temperature > 95.0 or self.vibration > 0.8:
                self.status = "CRITICAL"
            elif self.health < 0.6 or self.temperature > 80.0 or self.vibration > 0.5:
                self.status = "WARNING"
            else:
                self.status = "NORMAL"

            units_step = int(10 * speed_factor)
            self.produced_units += units_step

        return self.get_state()

    def get_state(self) -> Dict[str, Any]:
        return {
            "machine_id": self.machine_id,
            "name": self.name,
            "rpm": round(self.rpm, 2),
            "temperature": round(self.temperature, 2),
            "vibration": round(self.vibration, 4),
            "power": round(self.power, 2),
            "health": round(self.health, 4),
            "status": self.status,
            "operating_hours": round(self.operating_hours, 2),
            "produced_units": self.produced_units
        }