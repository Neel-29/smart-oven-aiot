import time

class SimulatedOven:
    """
    Simulates the physical hardware of the Smart Oven.
    It holds a state (temp, humidity, relay) and simulates
    the cooking process over time.
    """
    
    def __init__(self):
        self.current_temp = 20.0     # Start at room temp (C)
        self.current_humidity = 45.0   # Start at room humidity (%)
        self.relay_state = "OFF"     # Heating element
        self.cook_time_remaining_s = 0
        self.target_temp = 20.0
        self.is_cooking = False

    def start_cooking(self, target_temp_f, duration_min):
        """
        Public command to start the oven, sent from the API.
        Converts F to C for simulation.
        """
        self.target_temp = (target_temp_f - 32) * 5.0 / 9.0
        self.cook_time_remaining_s = duration_min * 60
        
        self.relay_state = "ON"
        self.is_cooking = True
        
        print(f"[OVEN_SIM] STARTING. Target: {self.target_temp:.0f}°C ({target_temp_f}°F) for {duration_min} min.")

    def stop_cooking(self):
        """Public command to stop the oven."""
        self.relay_state = "OFF"
        self.is_cooking = False
        self.cook_time_remaining_s = 0
        print("[OVEN_SIM] STOPPED MANUALLY.")
        
    def get_sensor_values(self):
        """
        Simulates the oven's internal sensors.
        This would be real sensor data on an ESP32.
        """
        if self.relay_state == "ON":
            if self.current_temp < self.target_temp:
                self.current_temp += 5.0
            else:
                self.current_temp += (self.target_temp - self.current_temp) * 0.1
            
            if self.current_humidity > 15.0:
                self.current_humidity -= 0.1 
        
        elif self.relay_state == "OFF" and self.current_temp > 20.0:
            self.current_temp -= 1.0

        if self.is_cooking and self.cook_time_remaining_s > 0:
            self.cook_time_remaining_s -= 1
            
            if self.cook_time_remaining_s == 0:
                print("[OVEN_SIM] COOKING COMPLETE. DING!")
                self.relay_state = "OFF"
                self.is_cooking = False

        return {
            "temperature_c": round(self.current_temp, 2),
            "humidity_percent": round(self.current_humidity, 2),
            "relay_state": self.relay_state,
            "time_remaining_s": self.cook_time_remaining_s,
            "is_cooking": self.is_cooking
        }

if __name__ == '__main__':
    print("--- Running Oven Simulation Test ---")
    oven = SimulatedOven()
    
    oven.start_cooking(target_temp_f=400, duration_min=1)
    
    for i in range(70):
        time.sleep(0.1) 
        sensors = oven.get_sensor_values()
        
        if i % 5 == 0:
            print(f"[LOG] Sec {i}: Temp={sensors['temperature_c']:.0f}C, "
                  f"Time Left={sensors['time_remaining_s']}s, "
                  f"Heating={sensors['relay_state']}")