"""Telemetry and decision-analysis tools for the industrial agent."""

from collections.abc import Mapping


class IndustrialTools:
    """Format factory telemetry and explain reinforcement-learning actions."""

    ACTION_DESCRIPTIONS = {
        0: "reduce Machine 2 speed to 0.5x",
        1: "slightly reduce Machine 2 speed to 0.8x",
        2: "maintain Machine 2 speed at 1.0x",
        3: "increase Machine 2 speed to 1.2x",
        4: "stop Machine 2 for cooling",
    }

    @staticmethod
    def inspect_sensor_data(sensor_data: Mapping[str, object]) -> str:
        """Summarize the Machine 2 fields supplied by the factory state."""
        required_fields = ("status", "temperature", "vibration", "health", "rpm")
        missing_fields = [field for field in required_fields if field not in sensor_data]
        if missing_fields:
            raise ValueError(
                f"Sensor data is missing required fields: {', '.join(missing_fields)}"
            )

        status = sensor_data["status"]
        temperature = float(sensor_data["temperature"])
        vibration = float(sensor_data["vibration"])
        health = float(sensor_data["health"])
        rpm = float(sensor_data["rpm"])

        return (
            f"Machine 2 status: {status}; temperature: {temperature:.1f} °C; "
            f"vibration: {vibration:.3f} mm/s; health: {health:.1%}; "
            f"speed: {rpm:.0f} RPM."
        )

    @staticmethod
    def calculate_efficiency_score(produced_units: int, power_kw: float) -> str:
        """Report the production-to-power ratio for the current factory step."""
        if power_kw <= 0:
            raise ValueError("Power consumption must be greater than zero.")

        score = produced_units / power_kw
        return f"Production-to-power ratio: {score:.2f} units/kW."

    @staticmethod
    def evaluate_rl_decision(last_action: int, predicted_temp: float) -> str:
        """Explain the selected Machine 2 action and its temperature context."""
        if last_action not in IndustrialTools.ACTION_DESCRIPTIONS:
            raise ValueError(f"Unknown reinforcement-learning action: {last_action}")

        action_description = IndustrialTools.ACTION_DESCRIPTIONS[last_action]
        if predicted_temp > 80.0:
            temperature_context = (
                f"Predicted temperature is elevated at {predicted_temp:.1f} °C."
            )
        else:
            temperature_context = (
                f"Predicted temperature is {predicted_temp:.1f} °C."
            )

        return f"The agent chose to {action_description}. {temperature_context}"
