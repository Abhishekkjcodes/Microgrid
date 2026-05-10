import numpy as np
import gymnasium as gym
from gymnasium import spaces

class MicrogridEnv(gym.Env):
    """
    Custom Environment for a Smart Energy Microgrid that follows the Gymnasium API.
    The agent controls battery charging/discharging and load management to minimize energy costs
    and maximize renewable usage (SDG 7).
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super().__init__()
        
        # Action Space (Discrete):
        # 0: Charge battery (from solar or grid)
        # 1: Discharge battery (to serve load)
        # 2: Import from grid directly
        # 3: Shed non-critical load
        self.action_space = spaces.Discrete(4)
        
        # Observation Space (Discrete/MultiDiscrete):
        # SOC: 4 bins (0-25%, 25-50%, 50-75%, 75-100%)
        # Solar: 3 bins (Low, Med, High)
        # Load: 2 bins (Low, High)
        # Price: 2 bins (Off-peak, Peak)
        self.observation_space = spaces.MultiDiscrete([4, 3, 2, 2])
        
        # Physical system constraints
        self.battery_capacity = 100.0  # Total capacity in kWh
        self.max_power_rate = 20.0     # Max charge/discharge rate in kW
        self.blackout_penalty = 10.0   # Large penalty for shedding load ($)
        
        # Episode tracking variables
        self.current_hour = 0
        self.battery_soc_kwh = 0.0
        
    def _get_solar_output(self, hour):
        """Synthetic solar data generation: Sine wave peaking at midday (hour 12)"""
        # Solar is only active between hours 6 and 18
        if 6 <= hour <= 18:
            return 40.0 * np.sin(np.pi * (hour - 6) / 12.0)
        return 0.0
        
    def _get_load_demand(self, hour):
        """Synthetic load data generation: Daily profile"""
        # High demand in the morning (7-9) and evening (17-21)
        if 7 <= hour <= 9 or 17 <= hour <= 21:
            return 30.0
        return 10.0  # Low demand at night and mid-day
        
    def _get_grid_price(self, hour):
        """Synthetic grid price: Time-of-Use (ToU) pricing"""
        # Peak prices during hours 17-21
        if 17 <= hour <= 21:
            return 0.5  # Peak: $0.5/kWh
        return 0.1      # Off-peak: $0.1/kWh
        
    def _get_state(self):
        """Map the current continuous environment state into the discrete state bins"""
        # 1. SOC Bins (0-25%, 25-50%, 50-75%, 75-100%)
        soc_pct = self.battery_soc_kwh / self.battery_capacity
        soc_bin = min(3, int(soc_pct // 0.25))
        
        # 2. Solar Bins (Low: <10, Med: 10-25, High: >25)
        solar = self._get_solar_output(self.current_hour)
        if solar < 10: solar_bin = 0
        elif solar < 25: solar_bin = 1
        else: solar_bin = 2
        
        # 3. Load Bins (Low: <20, High: >=20)
        load = self._get_load_demand(self.current_hour)
        load_bin = 0 if load < 20 else 1
        
        # 4. Price Bins (Off-peak: 0, Peak: 1)
        price_bin = 1 if (17 <= self.current_hour <= 21) else 0
        
        return np.array([soc_bin, solar_bin, load_bin, price_bin], dtype=np.int32)
        
    def reset(self, seed=None, options=None):
        """Reset environment to initial state (hour 0) for a new episode"""
        super().reset(seed=seed)
        self.current_hour = 0
        self.battery_soc_kwh = self.battery_capacity * 0.5  # Start at 50% SOC
        return self._get_state(), {}
        
    def step(self, action):
        """Execute one time step within the environment"""
        solar = self._get_solar_output(self.current_hour)
        load = self._get_load_demand(self.current_hour)
        price = self._get_grid_price(self.current_hour)
        
        grid_import_units = 0.0
        blackout_occurred = 0.0
        
        # Net demand is the load remaining after using available solar energy
        net_demand = load - solar
        
        # Apply logic based on chosen action
        if action == 0:  # 0: Charge battery
            # Calculate how much battery space is available
            charge_amount = min(self.max_power_rate, self.battery_capacity - self.battery_soc_kwh)
            
            if net_demand > 0:
                # We need to serve the net demand AND charge the battery, both require grid import
                grid_import_units = net_demand + charge_amount
            else:
                # There is excess solar available
                excess_solar = -net_demand
                if excess_solar >= charge_amount:
                    grid_import_units = 0.0 # Everything charged directly from free solar
                else:
                    grid_import_units = charge_amount - excess_solar # Charge remainder from grid
                    
            self.battery_soc_kwh += charge_amount
            
        elif action == 1:  # 1: Discharge battery
            if net_demand > 0:
                # Attempt to discharge to meet the net demand
                discharge_amount = min(self.max_power_rate, self.battery_soc_kwh, net_demand)
                # Remainder of demand must be imported from the grid
                grid_import_units = net_demand - discharge_amount
                self.battery_soc_kwh -= discharge_amount
            else:
                # No net demand, discharging is a wasted action here, import nothing
                grid_import_units = 0.0
                
        elif action == 2:  # 2: Import directly from grid
            # Import everything needed from grid, bypass battery
            if net_demand > 0:
                grid_import_units = net_demand
            else:
                grid_import_units = 0.0
                
        elif action == 3:  # 3: Shed non-critical load
            # Accept a partial or full blackout to avoid grid costs
            if net_demand > 0:
                grid_import_units = 0.0
                blackout_occurred = net_demand # The amount of load we failed to serve
            else:
                grid_import_units = 0.0
                
        # ---------------------------------------------------------
        # REWARD FUNCTION
        # Objective: Minimize cost. Maximize negative of total cost.
        # Cost = (grid import * TOU price) + (unmet demand * blackout penalty)
        # ---------------------------------------------------------
        cost = (grid_import_units * price) + (blackout_occurred * self.blackout_penalty)
        reward = -cost
        
        # Advance simulation time
        self.current_hour += 1
        
        # Check termination condition (episode ends after 24 hours)
        terminated = self.current_hour >= 24
        truncated = False
        
        # Track important metrics for evaluation
        info = {
            "hour": self.current_hour - 1,
            "soc": self.battery_soc_kwh,
            "cost": grid_import_units * price,
            "blackout": blackout_occurred,
            "grid_import": grid_import_units
        }
        
        return self._get_state(), reward, terminated, truncated, info
