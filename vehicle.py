class Vehicle:
    def __init__(self, vehicle_type, centroid_location, ID, intersection_number, lane_location, action):
        self.vehicle_type = vehicle_type  # "normal_vehicle" or "emergency_vehicle"
        self.centroid_location = centroid_location  # (x, y) of the vehicle's centroid
        self.ID = ID 
        self.intersection_number = intersection_number  # "1", "2", "3", "4"
        self.lane_location = lane_location  # "left", "middle", "right"
        self.action = action  # "stop", "yield_left", "yield_right", "continue"

    