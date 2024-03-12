from vehicle import *

def normal_vehicle_number_change(intersection_number):
    if intersection_number == "1":
        return ("1", "2", "3", "4")
    if intersection_number == "2":
        return ("2", "3", "4", "1")
    if intersection_number == "3":
        return ("3", "4", "1", "2")
    if intersection_number == "4":
        return ("4", "1", "2", "3")
    pass

def instruction_delivery_Emergency_Vehicle_At_left(vehicle,emergency_intersection_number):
    section_1, section_2, section_3, section_4 = normal_vehicle_number_change(emergency_intersection_number)
    
    if vehicle.vehicle_type == "normal_vehicle":

        if vehicle.intersection_number == section_1:
            vehicle.action = "yield_right"

        elif vehicle.intersection_number == section_2 or vehicle.intersection_number == section_3 or vehicle.intersection_number == section_4:
            vehicle.action = "stop"
        
    return vehicle.action

def instruction_delivery_Emergency_Vehicle_At_straight(vehicle, emergency_intersection_number):
    section_1, section_2, section_3, section_4 = normal_vehicle_number_change(emergency_intersection_number)
    
    if vehicle.vehicle_type == "normal_vehicle":

        if (vehicle.intersection_number == section_2 and vehicle.lane_location == "right") or (vehicle.intersection_number == section_3 and (vehicle.lane_location == "right" or vehicle.lane_location == "straight")):
            vehicle.action = "continue"
        elif (vehicle.intersection_number == section_1) and (vehicle.lane_location == "left"):
            vehicle.action = "yeild_left"
        elif (vehicle.intersection_number == section_1 and vehicle.lane_location == "straight") or (vehicle.intersection_number == section_1 and vehicle.lane_location == "right"):
            vehicle.action = "yeild_right"
        else:
            vehicle.action = "stop"
        
    return vehicle.action

def instruction_delivery_Emergency_Vehicle_At_right(vehicle, emergency_intersection_number):
    section_1, section_2, section_3, section_4 = normal_vehicle_number_change(emergency_intersection_number)
        
    if vehicle.vehicle_type == "normal_vehicle":

        if vehicle.intersection_number == section_1:
            vehicle.action = "yield_left"

        elif (vehicle.intersection_number == section_2 and vehicle.lane_location == "straight") or (vehicle.intersection_number == section_3 and vehicle.lane_location == "left"):
            vehicle.action = "stop"
        
        else:
            vehicle.action = "continue"
        
    return vehicle.action
