from typing import Dict, Any
from digital_twin.machine import Machine

class FactoryLine:
    def __init__(self):
        self.m1 = Machine("M1", "Cutting", base_rpm=1500.0, initial_temp=65.0, base_vib=0.25)
        self.m2 = Machine("M2", "Processing", base_rpm=1400.0, initial_temp=68.0, base_vib=0.35)
        self.m3 = Machine("M3", "Assembly", base_rpm=1300.0, initial_temp=60.0, base_vib=0.20)
        self.time_step = 0

    def step(self, actions: Dict[str, float] = None) -> Dict[str, Any]:
        if actions is None:
            actions = {"M1_speed": 1.0, "M2_speed": 1.0, "M3_speed": 1.0}

        self.time_step += 1

        s1 = self.m1.step(actions.get("M1_speed", 1.0))
        s2 = self.m2.step(actions.get("M2_speed", 1.0))
        s3 = self.m3.step(actions.get("M3_speed", 1.0))

        total_power = s1["power"] + s2["power"] + s3["power"]
        total_finished_goods = min(s1["produced_units"], s2["produced_units"], s3["produced_units"])

        return {
            "step": self.time_step,
            "M1": s1,
            "M2": s2,
            "M3": s3,
            "total_power_kw": round(total_power, 2),
            "total_finished_goods": total_finished_goods
        }

    def reset(self):
        self.__init__()
        return self.step()