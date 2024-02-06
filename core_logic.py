from vehicle import *

def instruction_delivery_Emergency_Vehicle_At_1left(vehicle):
    
    if vehicle.vehicle_type == "normal_vehicle":

        if vehicle.intersection_number == "1":
            vehicle.action = "yield_left"

        elif vehicle.intersection_number == "2" or vehicle.intersection_number == "3" or vehicle.intersection_number == "4":
            vehicle.action = "stop"
        
    return vehicle.action

def instruction_delivery_Emergency_Vehicle_At_1middle(vehicle):
    
    if vehicle.vehicle_type == "normal_vehicle":

        if (vehicle.intersection_number == "2" and vehicle.lane_location == "right") or (vehicle.intersection_number == "3" and (vehicle.lane_location == "right" or vehicle.lane_location == "straight")):
            vehicle.action = "continue"
        elif (vehicle.intersection_number == "1") and (vehicle.lane_location == "left"):
            vehicle.action = "yeild_left"
        elif (vehicle.intersection_number == "1") and (vehicle.lane_location == "middle") or (vehicle.intersection_number == "1") and (vehicle.lane_location == "right"):
            vehicle.action = "yeild_right"
        else:
            vehicle.action = "stop"
        
    return vehicle.action

def instruction_delivery_Emergency_Vehicle_At_1right(vehicle):
        
    if vehicle.vehicle_type == "normal_vehicle":

        if vehicle.intersection_number == "1":
            vehicle.action = "yield_left"

        elif vehicle.intersection_number == "2" and vehicle.lane_location == "middle":
            vehicle.action = "stop"
        
        else:
            vehicle.action = "continue"
        
    return vehicle.action