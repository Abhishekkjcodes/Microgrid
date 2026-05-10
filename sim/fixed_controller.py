class FixedController:
    """
    Baseline rule-based heuristic controller for the Smart Microgrid.
    This provides a sensible standard logic baseline to compare against the RL Agent.
    """
    def __init__(self, env):
        self.env = env
        
    def act(self, obs):
        """
        Determines action based on a set of fixed heuristic rules.
        Args:
            obs: The discretized observation space [soc_bin, solar_bin, load_bin, price_bin]
        Returns:
            action (int): The chosen action.
        """
        soc_bin, solar_bin, load_bin, price_bin = obs
        
        # Rule 1: If battery is empty (bin 0) and the price is currently high (bin 1),
        # we try to shed load to avoid exorbitant grid import costs.
        if soc_bin == 0 and price_bin == 1:
            return 3  # Shed Load
            
        # Rule 2: If the price is currently high (bin 1), 
        # we always prefer to discharge the battery to serve the load and avoid grid fees.
        if price_bin == 1:
            return 1  # Discharge
            
        # Rule 3: If the price is currently low (bin 0) and the battery is not full,
        # we take advantage of the off-peak rates to charge up the battery.
        if price_bin == 0 and soc_bin < 3:
            return 0  # Charge
            
        # Default fallback action: just import from the grid
        return 2  # Import
