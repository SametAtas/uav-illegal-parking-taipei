import math
import numpy as np
from pyproj import Transformer

# --- Define the Math as a reusable Function ---
def calculate_car_error(car_name, x_pix, y_pix, true_X, true_Y):
    # 1. Camera Parameters
    f = 2873.64      
    cx = 15.89       
    cy = -6.54       
    img_w, img_h = 4000, 3000

    # 2. Drone Flight Data (Raw EXIF)
    omega = math.radians(0.00)     
    phi = math.radians(0.10)       
    kappa = math.radians(-55.30)   
    delta_Z = -114.80            
    
    drone_lat = 25.0069835
    drone_lon = 121.5343854

    # Convert Drone WGS84 to TWD97
    transformer = Transformer.from_crs("epsg:4326", "epsg:3826", always_xy=True)
    X_O, Y_O = transformer.transform(drone_lon, drone_lat)       

    # 3. Rotation Matrices
    M_omega = np.array([[1, 0, 0],[0, math.cos(omega), math.sin(omega)],[0, -math.sin(omega), math.cos(omega)]])
    M_phi = np.array([[math.cos(phi), 0, -math.sin(phi)],[0, 1, 0],[math.sin(phi), 0, math.cos(phi)]])
    M_kappa = np.array([[math.cos(kappa), math.sin(kappa), 0],[-math.sin(kappa), math.cos(kappa), 0],[0, 0, 1]])
    
    M = M_kappa @ M_phi @ M_omega
    
    # 4. The Collinearity Math
    x_a = x_pix - (img_w / 2)
    y_a = (img_h / 2) - y_pix 

    num_X = M[0,0]*(x_a - cx) + M[1,0]*(y_a - cy) + M[2,0]*(-f)
    den   = M[0,2]*(x_a - cx) + M[1,2]*(y_a - cy) + M[2,2]*(-f)
    X_A = X_O + delta_Z * (num_X / den)

    num_Y = M[0,1]*(x_a - cx) + M[1,1]*(y_a - cy) + M[2,1]*(-f)
    Y_A = Y_O + delta_Z * (num_Y / den)

    # 5. Calculate Error
    error_x = abs(true_X - X_A)
    error_y = abs(true_Y - Y_A)
    total_error = math.sqrt(error_x**2 + error_y**2)

    print(f"--- {car_name} ---")
    print(f"Calculated: X = {X_A:.2f}, Y = {Y_A:.2f}")
    print(f"True Map:   X = {true_X:.2f}, Y = {true_Y:.2f}")
    print(f"Total Error (誤差): {total_error:.2f} meters\n")
    
    return total_error


print("Starting Error Calculations...\n")

# Car 1
calculate_car_error("Car 1", x_pix=3317, y_pix=2705, true_X=303904.83, true_Y=2766577.38)

# Car 2
calculate_car_error("Car 2", x_pix=3344, y_pix=2688, true_X=303905.86, true_Y=2766577.02)

# Car 3 (Using your exact pixels and TWD97 map coordinates)
calculate_car_error("Car 3", x_pix=3185, y_pix=2253, true_X=303971.38, true_Y=2766581.76)