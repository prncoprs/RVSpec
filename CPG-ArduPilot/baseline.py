import time
import subprocess
import shutil
from pathlib import Path
from pymavlink import mavutil

# Define paths
ARDUPILOT_PATH = Path("<RVSPEC_ROOT>/UAV/ardupilot")
LOGS_DIR = ARDUPILOT_PATH / "logs"
MISSION_FILE = ARDUPILOT_PATH / "Tools/autotest/Generic_Missions/CMAC-copter-navtest.txt"
FLIGHT_LOG_FILE = Path.cwd() / "flight_log.csv"

# Start SITL using MAVProxy
def start_sitl():
    print("Starting ArduPilot SITL...")
    sitl_command = [
        "sim_vehicle.py",
        "-v", "ArduCopter",
        "-w",  # Wipe settings for a clean start
        "--console",
        "--map",
        "--debug",
        "--aircraft", "sitl_test"
    ]
    subprocess.Popen(sitl_command)
    time.sleep(10)  # Give time for SITL to initialize

def connect_mavlink():
    print("Connecting to MAVProxy SITL...")
    
    # Retry connecting until successful
    while True:
        try:
            master = mavutil.mavlink_connection("tcp:127.0.0.1:5760")
            master.wait_heartbeat()
            print("MAVLink connection established.")
            return master
        except Exception as e:
            print(f"Connection failed: {e}, retrying...")
            time.sleep(2)  # Wait and retry

# Load mission using MAVProxy
def load_mission(mission_file):
    print(f"Loading mission from {mission_file}...")
    mission_cmd = f"wp load {mission_file}"
    subprocess.run(["mavproxy.py", "--master=tcp:127.0.0.1:5760", "--cmd", mission_cmd])
    print("Mission uploaded successfully!")

# Disable sensor noise using MAVProxy commands
def disable_noise():
    print("Disabling sensor noise...")
    noise_params = [
        "SIM_GPS_NOISE", "SIM_ACC_NOISE", "SIM_GYRO_NOISE",
        "SIM_BARO_NOISE", "SIM_AIRSIM_NOISE", "SIM_MAG_NOISE",
        "SIM_FLOW_NOISE", "SIM_LIDAR_NOISE"
    ]
    
    for param in noise_params:
        cmd = f"param set {param} 0"
        subprocess.run(["mavproxy.py", "--master=tcp:127.0.0.1:5760", "--cmd", cmd])

    print("Sensor noise disabled.")

# Arm the vehicle and start the mission
def start_mission(master):
    print("Arming the vehicle...")
    
    # Ensure GPS lock
    while True:
        msg = master.recv_match(type="GPS_RAW_INT", blocking=True)
        if msg and msg.fix_type >= 2:
            break
        print("Waiting for GPS lock...")
        time.sleep(1)

    # Send arm command
    master.arducopter_arm()
    print("Waiting for arming...")
    master.motors_armed_wait()

    # Takeoff to 10m
    print("Taking off...")
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0,
        0, 0, 0, 0, 0, 0, 10
    )

    time.sleep(10)  # Allow time to reach altitude

    # Switch to AUTO mode
    print("Switching to AUTO mode...")
    master.set_mode("AUTO")
    time.sleep(5)

# Log flight data
def log_flight(master):
    print("Logging flight data...")
    
    with FLIGHT_LOG_FILE.open("w") as f:
        f.write("Time, Lat, Lon, Alt, Roll, Pitch, Yaw, VelocityX, VelocityY, VelocityZ\n")

        while True:
            msg = master.recv_match(type="GLOBAL_POSITION_INT", blocking=True)
            if msg:
                lat, lon, alt = msg.lat / 1e7, msg.lon / 1e7, msg.relative_alt / 1000

                attitude = master.recv_match(type="ATTITUDE", blocking=True)
                roll, pitch, yaw = attitude.roll, attitude.pitch, attitude.yaw

                velocity = master.recv_match(type="VFR_HUD", blocking=True)
                vx, vy, vz = velocity.airspeed, velocity.groundspeed, velocity.climb

                data = [time.time(), lat, lon, alt, roll, pitch, yaw, vx, vy, vz]
                f.write(",".join(map(str, data)) + "\n")

                print(f"Logged: {lat}, {lon}, {alt}")

            time.sleep(0.5)

# Retrieve flight log (BIN file)
def retrieve_logs():
    print("Retrieving flight logs...")
    log_files = list(LOGS_DIR.glob("*.BIN"))

    if not log_files:
        print("No log files found.")
        return

    latest_log = max(log_files, key=lambda f: f.stat().st_ctime)
    log_path = latest_log.resolve()

    shutil.copy(log_path, Path.cwd() / latest_log.name)
    print(f"Log file {latest_log.name} copied to {Path.cwd()}")

# Main execution
if __name__ == "__main__":
    start_sitl()
    master = connect_mavlink()
    disable_noise()
    load_mission(MISSION_FILE)
    start_mission(master)
    log_flight(master)
    retrieve_logs()
    print("Mission complete!")
