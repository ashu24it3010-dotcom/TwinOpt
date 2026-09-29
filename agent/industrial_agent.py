import json
import os
from typing import Any, Dict


class IndustrialCopilot:

    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY", None)

        if self.api_key:
            import openai

            self.client = openai.OpenAI(api_key=self.api_key)
        else:
            self.client = None

    def query(
        self,
        prompt: str,
        current_state: Dict[str, Any],
        predicted_temp: float,
        last_action: int,
    ) -> str:
        if self.client:
            try:
                system_prompt = (
                    "You are an AI Industrial Assistant for a Smart Factory Digital Twin. "
                    "Answer the user's query naturally. Use the provided machine state and telemetry data "
                    "to answer questions about machine status, actions, or forecasts. If the user asks general or greeting questions "
                    "(e.g., 'what is your name'), respond politely without forcing the machine statistics."
                )

                user_content = f"""User Question: {prompt}

Current Machine Telemetry Context:
- State: {json.dumps(current_state)}
- Predicted M2 Temp: {predicted_temp}°C
- Last RL Speed Action: {last_action}"""

                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    max_tokens=250,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                # Log the error to console so silent API failures don't hide issues
                print(f"[IndustrialCopilot API Error]: {e}")

        # Fallback rule-based offline engine (only used if OPENAI_API_KEY is missing or API call fails)
        m2_info = current_state.get("M2", {})
        temp = m2_info.get("temperature", 70.0)
        vib = m2_info.get("vibration", 0.3)
        health = m2_info.get("health", 1.0) * 100.0

        explanation = []
        explanation.append(
            f"• **Analysis:** Machine 2 is operating at {temp}°C with vibration level {vib} mm/s and health status {health:.1f}%."
        )

        if temp > 80.0 or predicted_temp > 85.0:
            explanation.append(
                "• **RL Action Reason:** Speed was reduced to lower thermal buildup and avoid critical over-temperature degradation."
            )
        else:
            explanation.append(
                "• **RL Action Reason:** Thermal levels are stable. Operating speed is set to maximize line throughput."
            )

        explanation.append(
            f"• **Forecasting:** Next thermal checkpoint predicted at {predicted_temp:.1f}°C."
        )
        return "\n".join(explanation)