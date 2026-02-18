#!/usr/bin/env python

import argparse
import time
import os
import sys
import subprocess
import signal
import logging
import re
import psutil
import math
import threading
from datetime import datetime
from pymavlink import mavutil
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


mode_mapping_acm = {
    0 : 'STABILIZE',
    1 : 'ACRO',
    2 : 'ALT_HOLD',
    3 : 'AUTO',
    4 : 'GUIDED',
    5 : 'LOITER',
    6 : 'RTL',
    7 : 'CIRCLE',
    8 : 'POSITION',
    9 : 'LAND',
    10 : 'OF_LOITER',
    11 : 'DRIFT',
    13 : 'SPORT',
    14 : 'FLIP',
    15 : 'AUTOTUNE',
    16 : 'POSHOLD',
    17 : 'BRAKE',
    18 : 'THROW',
    19 : 'AVOID_ADSB',
    20 : 'GUIDED_NOGPS',
    21 : 'SMART_RTL',
    22 : 'FLOWHOLD',
    23 : 'FOLLOW',
    24 : 'ZIGZAG',
    25 : 'SYSTEMID',
    26 : 'AUTOROTATE',
    27 : 'AUTO_RTL',
}



class SimTask:
    def __init__(self, connect, mission):
        self.connect = connect
        self.mission = mission
        self.sitl_proc = None
        self.master = None  # pymavlink connection
        self.parm_file = None
        self.rc_override_threads = {}
        self.rc_override_stop_flags = {}

    def start_sitl(self):
        """
        Start SITL using sim_vehicle.py with --console.
        """
        logger.info("Starting SITL simulation environment...")
        
        sitl_cmd = [
            "sim_vehicle.py", "-f", "quad", "-v", "ArduCopter",
            "--console", "--map", 
            "--debug", 
            "--speedup", "2", "--wipe-eeprom",
            "--aircraft", "sitl_test",
            "--out=udp:127.0.0.1:14550"
        ]
        
        # sitl_cmd = [
        #     "sim_vehicle.py", 
        #     "-f", "gazebo-iris", 
        #     "-v", "ArduCopter",
        #     "--console", "--map", 
        #     "--model", "JSON",
        #     "--debug", 
        #     "--speedup", "2", 
        #     "--wipe-eeprom",
        #     "--aircraft", "sitl_test",
        #     "--out=udp:127.0.0.1:14550"
        # ]
        
        # sitl_cmd = [
        #     "sim_vehicle.py", 
        #     "-v", "ArduCopter",
        #     "-w",
        #     "--model", "webots-python",
        #     "--add-param-file=<RVSPEC_ROOT>/UAV/ardupilot/libraries/SITL/examples/Webots_Python/params/iris.parm", 
        #     "--console", "--map", 
        #     "--debug", 
        #     "--speedup", "2", 
        #     "--wipe-eeprom",
        #     "--aircraft", "sitl_test"
        #     # "--out=udp:127.0.0.1:9002"
        # ]
        
        # Start sim_vehicle.py in a new session (process group)
        self.sitl_proc = subprocess.Popen(sitl_cmd, preexec_fn=os.setsid)
        # Wait a sufficient amount of time for SITL to fully start.
        time.sleep(3)
        logger.info("SITL started.")

    def connect_vehicle(self):
        """
        Connect to the vehicle using pymavlink's mavutil.
        """
        logger.info(f"Connecting to vehicle at {self.connect}...")
        self.master = mavutil.mavlink_connection(self.connect, source_system=255)
        self.master.wait_heartbeat()
        logger.info(f"Heartbeat received from system {self.master.target_system} component {self.master.target_component}")
        
        # Register STATUSTEXT logger
        self.master.message_hooks.append(self._log_statustext)
        
    def _log_statustext(self, mav, msg):
        MAV_SEVERITY = {
            0: "EMERGENCY", 1: "ALERT", 2: "CRITICAL", 3: "ERROR",
            4: "WARNING", 5: "NOTICE", 6: "INFO", 7: "DEBUG"
        }
        if msg.get_type() == 'STATUSTEXT' and hasattr(msg, 'text'):
            severity = MAV_SEVERITY.get(msg.severity, "UNKNOWN")
            logger.info(f"[STATUSTEXT:{severity}] {msg.text}")

    def wait_for_param_ack(self, param_name, expected_value, timeout=10):
        """
        Wait for a PARAM_VALUE message for param_name with a value close to expected_value.
        Returns True if the expected parameter is acknowledged within timeout, False otherwise.
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            msg = self.master.recv_match(type='PARAM_VALUE', blocking=True, timeout=1)
            if msg is not None:
                # param_id might be bytes, so decode if necessary
                if isinstance(msg.param_id, bytes):
                    pid = msg.param_id.decode('utf-8').strip('\x00')
                else:
                    pid = msg.param_id
                if pid == param_name:
                    # Use a tolerance when comparing float values
                    if abs(float(msg.param_value) - expected_value) < 1e-3:
                        logger.info(f"Parameter {param_name} acknowledged with value {msg.param_value}")
                        return True
                    else:
                        logger.warning(f"Parameter {param_name} value mismatch: expected {expected_value} got {msg.param_value}")
        return False

    def load_parm_file(self, default_parm=False):
        """
        Load parameters from a .parm file and set them via MAVLink.
        The .parm file should have one parameter per line in the format:
        PARAM_NAME VALUE
        Lines starting with '#' or empty lines are skipped.
        """
        if default_parm:
            logger.info("Using default parameters.")
            parm_file = os.path.join(os.path.dirname(__file__), "00-default.parm")
        else:
            logger.info("Using Params parameters.")
            parm_file = self.parm_file
        
        if not parm_file:
            logger.info("No parameter file specified, skipping parameter load.")
            return

        if not os.path.exists(parm_file):
            logger.info(f"Parameter file not found: {parm_file}")
            return

        with open(parm_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                param_name, param_value = parts[0], parts[1]
                try:
                    value = float(param_value)
                except ValueError:
                    logger.error(f"Invalid value for {param_name}: {param_value}")
                    continue
                # Send the parameter using param_set_send
                self.master.mav.param_set_send(
                    self.master.target_system,
                    self.master.target_component,
                    param_name.encode('utf-8'),
                    value,
                    mavutil.mavlink.MAV_PARAM_TYPE_REAL32
                )
                logger.info(f"Sent request to set {param_name} to {value}")
                
                # Wait for the acknowledgement
                if not self.wait_for_param_ack(param_name, value, timeout=10):
                    logger.error(f"Timeout or value mismatch waiting for parameter {param_name}")
                else:
                    logger.info(f"{param_name} successfully updated.")
                # time.sleep(0.5)  # Avoid flooding the autopilot with requests

    def load_mission_from_file(self):
        """
        Load the mission file (QGC WPL 110 format) and return a list of mission items.
        """
        mission_list = []
        if not os.path.exists(self.mission):
            logger.error(f"Mission file not found: {self.mission}")
            self.exit_task()
        logger.info(f"Loading mission from {self.mission} ...")
        with open(self.mission, "r") as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            line = line.strip()
            if i == 0 and line.startswith("QGC"):
                continue
            if not line:
                continue
            parts = line.split()
            if len(parts) < 12:
                logger.warning(f"Skipping invalid mission line: {line}")
                continue
            mission_item = {
                "seq": int(parts[0]),
                "current": int(parts[1]),
                "frame": int(parts[2]),
                "command": int(parts[3]),
                "param1": float(parts[4]),
                "param2": float(parts[5]),
                "param3": float(parts[6]),
                "param4": float(parts[7]),
                "x": float(parts[8]),
                "y": float(parts[9]),
                "z": float(parts[10]),
                "autocontinue": int(parts[11])
            }
            mission_list.append(mission_item)
        logger.info(f"Loaded {len(mission_list)} mission items.")
        return mission_list

    def _send_mission_item_int(self, item, seq):
        """
        Helper method to send one mission item using MISSION_ITEM_INT.
        Convert latitude and longitude from degrees to int (degrees * 1e7).
        """
        # TODO: need to check frame again
        # If the mission file uses frame 3, convert it to frame 6?
        frame = item["frame"]
        # if frame == 3:
        #     frame = 6

        # local: x position in meters * 1e4, global: latitude in degrees * 10^7
        # local: y position in meters * 1e4, global: longitude in degrees *10^7
        lat_int = int(round(item["x"] * 1e7))
        lon_int = int(round(item["y"] * 1e7))
        logger.info(f"Sending mission item {seq}...")
        self.master.mav.mission_item_int_send(
            self.master.target_system,
            self.master.target_component,
            seq,
            frame,
            item["command"],
            item["current"],
            item["autocontinue"],
            item["param1"],
            item["param2"],
            item["param3"],
            item["param4"],
            lat_int,
            lon_int,
            item["z"]
        )

    def _send_mission_item(self, item, seq):
        """
        Helper method to send one mission item using MISSION_ITEM.
        For MISSION_ITEM, latitude, longitude, and altitude are sent as floats.
        """
        logger.info(f"Sending mission item {seq}...")
        self.master.mav.mission_item_send(
            self.master.target_system,
            self.master.target_component,
            seq,
            item["frame"],          # For QGC WPL 110, typically frame 3 (MAV_FRAME_GLOBAL) is used.
            item["command"],
            item["current"],
            item["autocontinue"],
            item["param1"],
            item["param2"],
            item["param3"],
            item["param4"],
            item["x"],              # latitude in degrees
            item["y"],              # longitude in degrees
            item["z"]               # altitude in meters
        )

    def upload_mission(self):
        """
        Upload the mission to the autopilot using the MAVLink mission protocol with MISSION_ITEM_INT.
        Follows the protocol:
        1. Send MISSION_COUNT with the total number of items.
        2. For each mission item, wait for a MISSION_REQUEST. 
            Although the MISSION_REQUEST is deprecated, but the autopilot still uses it. Weired!
            - If a duplicate (req.seq < expected), re-send that item.
            - If req.seq > expected, abort.
        3. After sending all items, wait for a MISSION_ACK.
        """
        MAX_ATTEMPTS = 3
        mission_list = self.load_mission_from_file()  # List of mission items from file.
        count = len(mission_list)
        logger.info(f"Uploading mission with {count} items...")
        
        # Send the mission count.
        self.master.mav.mission_count_send(self.master.target_system, self.master.target_component, count)

        for seq in range(count):
            while True:
                req = self.master.recv_match(type='MISSION_REQUEST', blocking=True, timeout=10)
                if req is None:
                    # Handle timeout (retry or abort)
                    pass
                if req.seq < seq:
                    # Duplicate request for an item we've already sent: re-send mission item for that sequence.
                    self._send_mission_item_int(mission_list[req.seq], req.seq)
                    continue  # Wait for the proper request.
                elif req.seq > seq:
                    # Out-of-sequence: this is an error.
                    # self.exit_task()
                    pass
                else:
                    # req.seq == seq, as expected.
                    break
            # Now send mission item seq.
            self._send_mission_item_int(mission_list[seq], seq)

        # Wait for a mission acknowledgment.
        ack = self.master.recv_match(type='MISSION_ACK', blocking=True, timeout=10)
        if ack:
            if ack.type == 0:   # MAV_MISSION_ACCEPTED
                logger.info("Mission upload acknowledged (accepted).")
                return True
            else:
                logger.error(f"Mission upload acknowledged with error (type: {ack.type}).")
                # self.exit_task()
        else:
            logger.error("No mission acknowledgment received; upload may have failed.")
            # self.exit_task()
        return False

    def set_home_position(self, lat, lon, alt, current_location=False):
        """
        Set home position explicitly using MAV_CMD_DO_SET_HOME.
        If current_location is True, uses the drone's current position.
        """
        logger.info(f"Setting home position: lat={lat}, lon={lon}, alt={alt} (current_location={current_location})")
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_DO_SET_HOME,
            0,  # confirmation
            1 if current_location else 0,  # use current location or not
            0, 0, 0,  # unused
            lat,
            lon,
            alt
        )

    def get_current_position(self, timeout=5):
        """
        Retrieve the current GPS position (lat, lon, relative alt).
        Args:
            timeout (int): Timeout in seconds for waiting for a GPS message.
        Returns:
            tuple or None: (latitude, longitude, relative_altitude) in degrees and meters, or None if unavailable.
        """
        msg = self.master.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=timeout)
        if not msg:
            logger.warning(f"Failed to receive GLOBAL_POSITION_INT within {timeout}s.")
            return None
        lat = msg.lat / 1e7
        lon = msg.lon / 1e7
        alt = msg.relative_alt / 1000.0  # mm to meters
        logger.info(f"Current Position → lat={lat:.7f}, lon={lon:.7f}, alt={alt:.2f}m")
        return (lat, lon, alt)
    
    def has_reached_position(self, target_lat, target_lon, target_alt, threshold=2.0):
        """
        Check if the drone has reached the target position (lat, lon, alt).
        
        Args:
            target_lat (float): Target latitude in degrees.
            target_lon (float): Target longitude in degrees.
            target_alt (float): Target relative altitude in meters.
            threshold (float): Allowed distance threshold in meters.

        Returns:
            bool: True if within threshold distance, False otherwise.
        """
        import math

        msg = self.master.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=5)
        if not msg:
            logger.warning("Failed to get GPS position for verification.")
            return False

        current_lat = msg.lat / 1e7
        current_lon = msg.lon / 1e7
        current_alt = msg.relative_alt / 1000.0

        # Haversine distance
        dlat = math.radians(target_lat - current_lat)
        dlon = math.radians(target_lon - current_lon)
        a = math.sin(dlat/2)**2 + math.cos(math.radians(current_lat)) * math.cos(math.radians(target_lat)) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        horizontal_distance = 6378137.0 * c

        vertical_diff = abs(target_alt - current_alt)
        distance_3d = math.sqrt(horizontal_distance**2 + vertical_diff**2)

        logger.info(f"Distance to target: {distance_3d:.2f} m")
        return distance_3d <= threshold
    
    def wait_until_reach_position(self, target_lat, target_lon, target_alt, threshold=2.0, timeout=60):
        """
        Wait until the drone reaches the target GPS position within a given threshold.

        Args:
            target_lat (float): Target latitude in degrees.
            target_lon (float): Target longitude in degrees.
            target_alt (float): Target relative altitude in meters.
            threshold (float): Allowed distance to be considered as "arrived" (meters).
            timeout (int): Max time to wait in seconds.

        Returns:
            bool: True if the target is reached within timeout, False otherwise.
        """
        logger.info("Waiting to reach target position...")
        start_time = time.time()

        while time.time() - start_time < timeout:
            if self.has_reached_position(target_lat, target_lon, target_alt, threshold):
                logger.info("Target position reached.")
                return True
            time.sleep(1)

        logger.warning("Timeout waiting for target position.")
        return False

    def set_mode(self, mode):
        """
        Set flight mode using MAVLink SET_MODE.
        For ArduCopter, a typical mapping might be:
          STABILIZE: 0, AUTO: 3, GUIDED: 4
        """
        mode_mapping = {
            "STABILIZE": 0,
            "ALT_HOLD": 2,
            "AUTO": 3,
            "GUIDED": 4,
            "LOITER": 5,
            "RTL": 6,
            "CIRCLE": 7,
            "LAND": 9,
            "DRIFT": 11,
            "SPORT": 13,
            "FLIP": 14,
            "BRAKE": 17,
        }
        if mode not in mode_mapping:
            logger.error(f"Mode {mode} not recognized.")
            self.exit_task()
        mode_id = mode_mapping[mode]
        logger.info(f"Setting mode to {mode} (mode_id={mode_id})...")
        self.master.mav.set_mode_send(self.master.target_system,
                                      mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                                      mode_id)
        # for FLIP mode, hard to catch response
        if mode == "FLIP":
            return
        
        # Wait until the heartbeat indicates the mode has changed.
        while True:
            hb = self.master.recv_match(type='HEARTBEAT', blocking=True)
            if hb and hasattr(hb, 'custom_mode') and hb.custom_mode == mode_id:
                logger.info(f"Mode set to {mode}.")
                break
            time.sleep(1)

    def wait_for_statustext(self, stext, timeout=60):
        logger.info(f"Waiting for STATUSTEXT: '{stext}'...")
        end_time = time.time() + timeout
        pattern = re.compile(stext, re.IGNORECASE)
        while time.time() < end_time:
            msg = self.master.recv_match(type='STATUSTEXT', blocking=True, timeout=5)
            if msg is not None and hasattr(msg, 'text'):
                text = msg.text
                # logger.info(f"Received STATUSTEXT: {text}")
                if pattern.search(text):
                    logger.info(f"'{stext}' detected.")
                    return True
            else:
                logger.info("No STATUSTEXT received; waiting...")
        logger.error(f"Timeout waiting for '{stext}'.")
        return False

    def wait_for_prearm_good(self, timeout=60):
        # Wait for "EKF3 IMU1 is using GPS" STATUSTEXT. Usually it indicates pre-arm checks are good.
        return self.wait_for_statustext(stext="EKF3 IMU1 is using GPS")
    
    def wait_until_ready_to_arm(self, timeout=60):
        """
        Wait until the vehicle is ready to arm (pre-arm checks pass).
        Args:
            timeout (int): Maximum time to wait in seconds.
        Returns:
            bool: True if ready to arm, False if timeout or error occurs.
        """
        logger.info("Waiting for vehicle to become ready to arm...")

        start_time = time.time()
        while time.time() - start_time < timeout:
            # Request SYS_STATUS message explicitly
            self.master.mav.command_long_send(
                self.master.target_system,
                self.master.target_component,
                mavutil.mavlink.MAV_CMD_REQUEST_MESSAGE,
                0,
                mavutil.mavlink.MAVLINK_MSG_ID_SYS_STATUS,
                0, 0, 0, 0, 0, 0
            )

            msg = self.master.recv_match(type="SYS_STATUS", blocking=True, timeout=2)
            if msg:
                # Bitmask of onboard_control_sensors_health
                health_flags = msg.onboard_control_sensors_health
                present_flags = msg.onboard_control_sensors_present
                enabled_flags = msg.onboard_control_sensors_enabled

                # Check required sensors are healthy and enabled
                required_flags = (
                    mavutil.mavlink.MAV_SYS_STATUS_SENSOR_3D_GYRO |
                    mavutil.mavlink.MAV_SYS_STATUS_SENSOR_3D_ACCEL |
                    mavutil.mavlink.MAV_SYS_STATUS_SENSOR_ABSOLUTE_PRESSURE |
                    mavutil.mavlink.MAV_SYS_STATUS_SENSOR_GPS
                )

                all_ok = (
                    (health_flags & required_flags) == required_flags and
                    (enabled_flags & required_flags) == required_flags and
                    (present_flags & required_flags) == required_flags
                )

                if all_ok:
                    logger.info("Vehicle passed pre-arm checks.")
                    return True
            time.sleep(1)
        logger.warning("Timeout waiting for pre-arm readiness.")
        return False
    
    def arm_vehicle(self):
        """
        Arm the vehicle and wait until it is armed (as indicated by the heartbeat).
        """
        logger.info("Arming vehicle...")
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            1, 0, 0, 0, 0, 0, 0
        )

        # TODO: check ARMING_ACK message
        # wait until arming confirmed (can manually check with master.motors_armed())
        logger.info("Waiting for the vehicle to arm")
        self.master.motors_armed_wait()
        logger.info('Armed!')
        return True
    
    # def arm_vehicle(self, timeout=20):
    #     """
    #     Arm the vehicle, retrying until success or timeout (based on heartbeat status).
    #     """
    #     logger.info("Attempting to arm vehicle...")
    #     start_time = time.time()

    #     while time.time() - start_time < timeout:
    #         # Send arm command
    #         self.master.mav.command_long_send(
    #             self.master.target_system,
    #             self.master.target_component,
    #             mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
    #             0,
    #             1, 0, 0, 0, 0, 0, 0
    #         )
    #         logger.info("Sent arm command. Waiting for vehicle to arm...")

    #         # Wait briefly and check armed status
    #         time.sleep(1)
    #         if self.master.motors_armed():
    #             logger.info("Vehicle is armed!")
    #             return True
    #         else:
    #             logger.warning("Vehicle not armed yet, retrying...")

    #     logger.error("Failed to arm vehicle within timeout.")
    #     return False

    def takeoff(self, alt):
        """
        Command the vehicle to take off to a specified altitude (in meters) using MAV_CMD_NAV_TAKEOFF.
        """
        logger.info(f"Sending takeoff command to {alt} meters...")
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0,
            0, 0, 0, 0,
            0, 0,
            alt)
        # Wait until the vehicle's altitude is at least the target.
        while True:
            msg = self.master.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=5)
            if msg:
                current_alt = msg.relative_alt / 1000.0  # relative_alt is in mm
                logger.info(f"Current altitude: {current_alt:.2f} m")
                if current_alt >= alt:
                    logger.info("Desired altitude reached.")
                    break
            time.sleep(1)
            
    def list_dataflash_logs(self):
        """
        Requests a list of available DataFlash logs from the autopilot and prints them.
        Returns a list of log entries.
        """
        # Request log list from index 0 to 255 (adjust as needed)
        self.master.mav.log_request_list_send(
            self.master.target_system,
            self.master.target_component,
            0,   # first log index
            255  # last log index
        )
        
        logs = []
        logger.info("Waiting for log entries...")
        while True:
            msg = self.master.recv_match(type='LOG_ENTRY', blocking=True, timeout=5)
            if msg is None:
                break
            # Each LOG_ENTRY message contains info about one log.
            logs.append(msg)
            logger.info(f"Log ID: {msg.id}, Size: {msg.size} bytes")
        return logs
            
    def download_dataflash_log(self, log_id, output_file):
        """
        Downloads a DataFlash log from the autopilot using LOG_REQUEST_DATA.
        Parameters:
        log_id       : the identifier of the log file to download
        output_file  : path to the file where the log will be saved
        """
        offset = 0
        block_size = 16  # Maximum number of bytes per LOG_DATA message
        with open(output_file, 'wb') as f:
            while True:
                # Request a block of log data
                self.master.mav.log_request_data_send(
                    self.master.target_system,
                    self.master.target_component,
                    log_id,
                    offset,
                    block_size
                )
                # Wait for the corresponding LOG_DATA message
                msg = self.master.recv_match(type='LOG_DATA', blocking=True, timeout=5)
                if msg is None:
                    logger.info("Timeout: no LOG_DATA received. Exiting.")
                    break

                # Check if the received message corresponds to our request
                if msg.id != log_id or msg.ofs != offset:
                    logger.info(f"Unexpected log data. Expected id: {log_id} at offset {offset}; got id: {msg.id} at offset {msg.ofs}")
                    break

                # Write the valid data bytes
                f.write(bytes(msg.data[:msg.count]))
                logger.info(f"Received {msg.count} bytes from offset {offset}.")

                # A block smaller than block_size indicates end of log
                if msg.count < block_size:
                    logger.info("End of log reached.")
                    break

                offset += msg.count
    
    def store_dataflash_logs(self):
        dataflash_logs = self.list_dataflash_logs()
        if not dataflash_logs:
            logger.info("No DataFlash logs found.")
            return

        # Assume the log with the highest id is the latest one.
        latest_log = max(dataflash_logs, key=lambda msg: msg.id)
        log_id = latest_log.id

        # Build the output file path
        date_str = datetime.now().strftime("%Y%m%d-%H:%M:%S")
        config_name = self.get_config_name()
        output_dir = os.path.join(".", "flight_data", "dataflash_logs")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"flight_dataflash_{config_name}_{date_str}_{log_id}.bin")
        
        logger.info(f"\nDownloading latest log id {log_id} to file '{output_file}'...")
        self.download_dataflash_log(log_id, output_file)
        logger.info(f"Download of latest log {log_id} completed.\n")
        
    def store_flight_data(self):
        """
        Store flight data to file for analysis.
        """
        self.store_dataflash_logs()
    
    def get_config_name(self):
        if self.parm_file is None:
            return "default"
        return os.path.splitext(os.path.basename(self.parm_file))[0]
    
    def set_rc_channel_pwm(self, channel_id, pwm=1500):
        """
        Set a specific RC channel PWM value (channels 1-8) reliably by sending it multiple times.
        pwm: 1000-2000 (override), 0 (release), 65535 (ignore).
        """
        if channel_id < 1 or channel_id > 8:
            logger.error("Only channels 1-8 are supported in this implementation.")
            return

        rc_channel_values = [65535 for _ in range(8)]
        rc_channel_values[channel_id - 1] = pwm

        logger.info(f"Setting RC channel {channel_id} to PWM {pwm}")
        self.master.mav.rc_channels_override_send(
            self.master.target_system,
            self.master.target_component,
            *rc_channel_values
        )
        
    def verify_rc_channel_pwm(self, channel_id, expected_pwm, timeout=5):
        logger.info(f"Verifying RC channel {channel_id} is set to PWM {expected_pwm}")
        start_time = time.time()
        while time.time() - start_time < timeout:
            msg = self.master.recv_match(type='RC_CHANNELS', blocking=True, timeout=1)
            if msg:
                actual_pwm = getattr(msg, f'chan{channel_id}_raw', None)
                if actual_pwm is not None:
                    if abs(actual_pwm - expected_pwm) < 10:  # tolerance +/-10
                        logger.info(f"Channel {channel_id} verified at PWM {actual_pwm}")
                        return True
                    else:
                        logger.warning(f"Channel {channel_id} PWM mismatch: expected {expected_pwm}, got {actual_pwm}")
            time.sleep(0.5)
        logger.error(f"Timeout verifying PWM value for channel {channel_id}")
        return False
    
    def start_rc_override(self, channel_id, pwm, interval=0.2):
        """
        Start persistently overriding a specific RC channel with a PWM value.
        Can be stopped with stop_rc_override(channel_id).
        """
        # Stop any existing override on the same channel first
        self.stop_rc_override(channel_id)

        stop_flag = threading.Event()
        self.rc_override_stop_flags[channel_id] = stop_flag

        def _override_loop():
            logger.info(f"Starting persistent RC override on channel {channel_id} to {pwm}")
            rc_values = [65535] * 8
            rc_values[channel_id - 1] = pwm

            while not stop_flag.is_set():
                self.master.mav.rc_channels_override_send(
                    self.master.target_system,
                    self.master.target_component,
                    *rc_values
                )
                time.sleep(interval)
            logger.info(f"Stopped RC override on channel {channel_id}")

        thread = threading.Thread(target=_override_loop, daemon=True)
        self.rc_override_threads[channel_id] = thread
        thread.start()

    def stop_rc_override(self, channel_id):
        """
        Stop the persistent RC override on a specific channel.
        """
        if channel_id in self.rc_override_stop_flags:
            self.rc_override_stop_flags[channel_id].set()
            del self.rc_override_stop_flags[channel_id]
        
        if channel_id in self.rc_override_threads:
            thread = self.rc_override_threads[channel_id]
            thread.join(timeout=5)  # Wait briefly for the thread to finish
            del self.rc_override_threads[channel_id]
            
    def set_guided_speed(self, speed_m_s, speed_type="groundspeed"):
        """
        Set the horizontal movement speed of the drone in GUIDED mode.
        Args:
            speed_m_s (float): Desired speed in meters per second (must be > 0)
        """
        if speed_m_s <= 0:
            logger.warning("Speed must be positive. Ignoring request.")
            return
        logger.info(f"Setting GUIDED {speed_type} speed to {speed_m_s:.2f} m/s")
        SPEED_TYPE = {
            "airspeed": 0,
            "groundspeed": 1,
            "climbspeed": 2,
            "descentspeed": 3
        }
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_DO_CHANGE_SPEED,  # Command to change speed
            0,          # Confirmation
            SPEED_TYPE.get(speed_type, 0),  # Speed type (airspeed, groundspeed, etc.)
            speed_m_s,  # Speed in m/s
            -1,         # Throttle (negative = no change)
            0, 0, 0, 0  # Unused parameters
        )

    def set_wpnav_speed(self, speed_m_s):
        """
        Set the WPNAV_SPEED parameter (in cm/s) to control speed in GUIDED mode.
        Args:
            speed_m_s (float): Desired speed in meters/second
        """
        speed_cms = int(speed_m_s * 100)
        logger.info(f"Setting WPNAV_SPEED to {speed_cms} cm/s")
        self.master.mav.param_set_send(
            self.master.target_system,
            self.master.target_component,
            b'WPNAV_SPEED',
            float(speed_cms),
            mavutil.mavlink.MAV_PARAM_TYPE_REAL32
        )
        
    def send_guided_velocity(self, vx, vy, vz):
        """
        Sends a velocity command (m/s) in NED frame.
        Positive x = north, y = east, z = down
        """
        logger.info(f"Sending velocity command vx={vx:.2f} vy={vy:.2f} vz={vz:.2f}")
        self.master.mav.set_position_target_local_ned_send(
            0,
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_FRAME_BODY_NED,
            0b0000111111000111,  # Bitmask: velocity only
            0, 0, 0,              # Position
            vx, vy, vz,           # Velocity in m/s
            0, 0, 0,              # Acceleration
            0, 0                  # Yaw
        )
    
    def send_velocity_for_duration(self, vx, vy, vz, duration=5.0):
        """
        Continuously send velocity command for a duration in seconds.
        """
        rate_hz = 30
        interval = 1.0 / rate_hz
        count = int(duration * rate_hz)
        logger.info(f"Sending velocity command vx={vx:.2f} vy={vy:.2f} vz={vz:.2f} for {duration:.2f} seconds")
        for _ in range(count):
            self.master.mav.set_position_target_local_ned_send(
                0,
                self.master.target_system,
                self.master.target_component,
                mavutil.mavlink.MAV_FRAME_BODY_NED,
                0b0000111111000111,  # Velocity only
                0, 0, 0,             # position
                vx, vy, vz,          # velocity in m/s
                0, 0, 0,             # acceleration
                0, 0                 # yaw
            )
            time.sleep(interval)

    def goto_north_meters(self, distance_m, altitude=None):
        """Move drone northward by a specified distance (in meters)."""
        logger.info(f"Moving North by {distance_m} meters...")
        if altitude is None:
            altitude = 50  # Default altitude if not provided

        msg = self.master.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=5)
        if msg:
            current_lat = msg.lat / 1e7
            current_lon = msg.lon / 1e7
            target_lat = current_lat + (distance_m / 111139)  # Approximate conversion

            self.master.mav.send(
                mavutil.mavlink.MAVLink_set_position_target_global_int_message(
                    10, self.master.target_system, self.master.target_component,
                    mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
                    int(0b110111111000),
                    int(target_lat * 1e7), int(current_lon * 1e7), altitude,
                    0, 0, 0, 0, 0, 0, 0, 0
                )
            )

            while True:
                msg = self.master.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=5)
                if msg:
                    lat_now = msg.lat / 1e7
                    distance = abs(lat_now - target_lat) * 111139
                    logger.info(f"Distance to target: {distance:.2f} m")
                    if distance < 1.0:
                        logger.info("Reached target position.")
                        break
                time.sleep(1)
        else:
            logger.error("Unable to retrieve current position.")


    def goto_offset_from_home(self, north_meters=0, east_meters=0, alt=None):
        """
        Move to a position relative to home (north/east offset in meters).
        """
        # Wait until HOME_POSITION is received
        home_msg = self.master.recv_match(type='HOME_POSITION', blocking=True, timeout=10)
        if not home_msg:
            logger.error("Home position not available!")
            return

        home_lat = home_msg.latitude / 1e7
        home_lon = home_msg.longitude / 1e7
        home_alt = home_msg.altitude / 1000.0  # mm to m

        # Calculate new lat/lon
        earth_radius = 6378137.0
        delta_lat = north_meters / earth_radius
        delta_lon = east_meters / (earth_radius * math.cos(math.pi * home_lat / 180))

        target_lat = home_lat + delta_lat * 180 / math.pi
        target_lon = home_lon + delta_lon * 180 / math.pi
        target_alt = alt if alt is not None else home_alt

        logger.info(f"Navigating to offset: North={north_meters}m, East={east_meters}m, Alt={target_alt}m")
        self.goto_gps_position(target_lat, target_lon, target_alt)
        
    def goto_offset_from_position(self, ref_lat, ref_lon, ref_alt, north_meters=0, east_meters=0, alt=None):
        """
        Move to a GPS position that is an offset (in meters) from the given reference position.
        
        Args:
            ref_lat (float): Reference latitude in degrees
            ref_lon (float): Reference longitude in degrees
            ref_alt (float): Reference altitude in meters
            north_meters (float): Offset in the north direction (positive is north)
            east_meters (float): Offset in the east direction (positive is east)
            alt (float): Optional target altitude; if None, use ref_alt
        """
        # Calculate new latitude/longitude using the given offset
        earth_radius = 6378137.0  # meters
        delta_lat = north_meters / earth_radius
        delta_lon = east_meters / (earth_radius * math.cos(math.pi * ref_lat / 180))

        target_lat = ref_lat + delta_lat * 180 / math.pi
        target_lon = ref_lon + delta_lon * 180 / math.pi
        target_alt = alt if alt is not None else ref_alt

        logger.info(f"Navigating from reference position to offset: North={north_meters}m, East={east_meters}m")
        logger.info(f"Target GPS position: lat={target_lat:.7f}, lon={target_lon:.7f}, alt={target_alt:.2f}m")
        self.goto_gps_position(target_lat, target_lon, target_alt)
        
    def goto_gps_position(self, lat, lon, alt, threshold=2.0, timeout=200):
        """
        Navigate to a global GPS coordinate (lat, lon, alt) and wait until the drone reaches it.
        Args:
            lat (float): Target latitude in degrees.
            lon (float): Target longitude in degrees.
            alt (float): Target relative altitude in meters.
            threshold (float): Distance threshold (in meters) for "arrival".
            timeout (int): Timeout in seconds to wait for arrival.
        """
        ### TODO: altitude is not working here!
        logger.info(f"Going to GPS position: lat={lat:.7f}, lon={lon:.7f}, alt={alt:.2f}m")

        # Send position target
        self.master.mav.set_position_target_global_int_send(
            0,
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
            0b0000111111111000,  # Bitmask: position only
            int(lat * 1e7),
            int(lon * 1e7),
            alt,
            0, 0, 0,  # velocity
            0, 0, 0,  # acceleration
            0, 0      # yaw, yaw_rate
        )

        logger.info("Waiting to reach GPS target...")
        start_time = time.time()
        while time.time() - start_time < timeout:
            msg = self.master.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=1)
            if msg:
                current_lat = msg.lat / 1e7
                current_lon = msg.lon / 1e7
                current_alt = msg.relative_alt / 1000.0  # mm → meters

                # Haversine distance calculation
                dlat = math.radians(lat - current_lat)
                dlon = math.radians(lon - current_lon)
                a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(current_lat)) * math.cos(math.radians(lat)) * math.sin(dlon / 2) ** 2
                c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
                distance = 6378137.0 * c  # Earth radius in meters

                logger.info(f"Distance to target: {distance:.2f} m")
                if distance <= threshold:
                    logger.info("Reached target position.")
                    return
            time.sleep(1)

        logger.warning("Timeout: did not reach target GPS position.")


    def goto_offset_from_position_nowait(self, ref_lat, ref_lon, ref_alt, north_meters=0, east_meters=0, alt=None):
        """
        Send navigation command to a GPS offset from a given position, without waiting for arrival.

        Args:
            ref_lat (float): Reference latitude in degrees
            ref_lon (float): Reference longitude in degrees
            ref_alt (float): Reference altitude in meters
            north_meters (float): Offset in the north direction (positive is north)
            east_meters (float): Offset in the east direction (positive is east)
            alt (float): Optional target altitude; if None, use ref_alt
        """
        earth_radius = 6378137.0  # meters
        delta_lat = north_meters / earth_radius
        delta_lon = east_meters / (earth_radius * math.cos(math.pi * ref_lat / 180))

        target_lat = ref_lat + delta_lat * 180 / math.pi
        target_lon = ref_lon + delta_lon * 180 / math.pi
        target_alt = alt if alt is not None else ref_alt

        logger.info(f"Sending async nav to offset: North={north_meters}m, East={east_meters}m")
        logger.info(f"Target GPS position: lat={target_lat:.7f}, lon={target_lon:.7f}, alt={target_alt:.2f}m")
        self.goto_gps_position_nowait(target_lat, target_lon, target_alt)

    def goto_gps_position_nowait(self, lat, lon, alt):
        """
        Send a position command to a GPS coordinate without waiting for arrival.

        Args:
            lat (float): Target latitude in degrees
            lon (float): Target longitude in degrees
            alt (float): Target relative altitude in meters
        """
        logger.info(f"Sending async GPS position target: lat={lat:.7f}, lon={lon:.7f}, alt={alt:.2f}m")
        self.master.mav.set_position_target_global_int_send(
            0,
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
            0b0000111111111000,  # Bitmask: position only
            int(lat * 1e7),
            int(lon * 1e7),
            alt,
            0, 0, 0,  # velocity
            0, 0, 0,  # acceleration
            0, 0      # yaw, yaw_rate
        )
            
    def wait_until_reach_home(self, tolerance_m=2, timeout=120):
        """
        Wait until drone returns close to home position.
        """
        logger.info("Waiting until drone reaches home position...")
        home_position = self.master.recv_match(type='HOME_POSITION', blocking=True, timeout=5)
        if not home_position:
            logger.error("Home position not set.")
            return False
        
        home_lat = home_position.latitude / 1e7
        home_lon = home_position.longitude / 1e7

        end_time = time.time() + timeout
        while time.time() < end_time:
            msg = self.master.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=5)
            if msg:
                current_lat = msg.lat / 1e7
                current_lon = msg.lon / 1e7
                distance = math.sqrt(((current_lat - home_lat) * 111139)**2 + 
                                    ((current_lon - home_lon) * 111139 * math.cos(math.radians(home_lat)))**2)
                logger.info(f"Distance from home: {distance:.2f} m")
                if distance <= tolerance_m:
                    logger.info("Drone reached home position.")
                    return True
            time.sleep(2)
        
        logger.error("Timeout waiting for drone to reach home position.")
        return False
    
    def set_home_origin(self, lat=-35.363262, lon=149.165237, alt=584.09):
        """
        Sets the home/origin location for the simulated vehicle.

        Parameters:
        - lat: Latitude in degrees
        - lon: Longitude in degrees
        - alt: Altitude in meters (above mean sea level)
        """
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_DO_SET_HOME,
            1,  # 1 = Use specified location, not current GPS
            0, 0, 0, 0,
            lat, lon, alt
        )
        logger.info(f"Home origin set to lat={lat}, lon={lon}, alt={alt}")

    def exit_task(self):
        """
        Graceful termination of subprocesses (SITL, MAVProxy) and logging handlers.
        """
        if self.sitl_proc:
            try:
                pgid = os.getpgid(self.sitl_proc.pid)
                logger.info(f"Gracefully terminating process group {pgid}...")
                os.killpg(pgid, signal.SIGINT)  # Gentle interrupt first
                time.sleep(1)
                if self.sitl_proc.poll() is None:
                    logger.info("Forcing termination...")
                    os.killpg(pgid, signal.SIGTERM)
                    time.sleep(1)
            except Exception as e:
                logger.error(f"Error terminating SITL process group: {e}")

        # Explicitly terminate MAVProxy processes
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                if proc.info['cmdline'] and any("mavproxy.py" in arg for arg in proc.info['cmdline']):
                    logger.info(f"Terminating MAVProxy process PID {proc.pid} gracefully...")
                    proc.send_signal(signal.SIGINT)
                    proc.wait(timeout=3)
                    if proc.is_running():
                        logger.info(f"Forcing termination of PID {proc.pid}...")
                        proc.kill()
        except ImportError:
            logger.error("psutil not installed; MAVProxy processes might linger.")

        # Ensure logging handlers are closed cleanly
        handlers = logger.handlers[:]
        for handler in handlers:
            handler.close()
            logger.removeHandler(handler)

        logger.info("Clean shutdown completed.")
        sys.exit(0)

    def signal_handler(self, sig, frame):
        logger.info("KeyboardInterrupt (Ctrl+C) detected. Terminating processes...")
        self.exit_task()

    def run_mission1(self):
        """
        Run the mission sequence:
          - Set mode to STABILIZE
          - Upload the mission via pymavlink
          - Set mode to GUIDED
          - Wait for "pre-arm good" STATUSTEXT
          - Arm the vehicle
          - Command takeoff to 20 m
          - Set mode to AUTO
        """
        self.set_mode("STABILIZE")
        
        
        # Wait until pre-arm checks pass.
        if not self.wait_for_prearm_good(timeout=60):
            logger.error("Prearm condition not met. Aborting mission sequence.")
            self.exit_task()

        self.upload_mission()
        self.set_mode("GUIDED")

        if not self.arm_vehicle():
            logger.error("Failed to arm vehicle. Aborting mission sequence.")
            self.exit_task()
        
        self.takeoff(20)
        self.set_mode("AUTO")

        # wait until disarming confirmed
        self.master.motors_disarmed_wait()

    def run_mission(self):
        """Execute specified custom mission."""
        logger.info("")
        logger.info("************************************************")
        logger.info("**************** MISSION START! ****************")
        logger.info("")
        
        # Wait until pre-arm checks pass.
        if not self.wait_for_prearm_good(timeout=60):
        # if not self.wait_until_ready_to_arm(timeout=60):
            logger.error("Prearm condition not met. Aborting mission sequence.")
            self.exit_task()
        time.sleep(5)
            
        self.set_mode("GUIDED")

        if not self.arm_vehicle():
            logger.error("Arming failed.")
            self.exit_task()
            
         # record pos here
        to_lat, to_lon, to_alt = self.get_current_position()
        to_alt += 50
        time.sleep(1)

        self.takeoff(50)
        logger.info("Hovering at 50m for 10 seconds...")
        time.sleep(5)
       
        # self.set_rc_channel_pwm(3, 1500)
        # self.verify_rc_channel_pwm(3, 1500)
        
        self.load_parm_file()
        time.sleep(5)
        
        # we need to keep send the rc override command to keep the altitude
        self.start_rc_override(channel_id=3, pwm=1500)
        
        self.set_mode("ALT_HOLD")
        logger.info("ALT_HOLD mode for 10 seconds...")
        
        time.sleep(10)

        self.set_mode("FLIP")
        logger.info("Performing FLIP mode for 10 seconds...")
        time.sleep(3)

        self.set_mode("CIRCLE")
        self.start_rc_override(channel_id=2, pwm=1600)
        # self.set_rc_channel_pwm(2, 1600)
        time.sleep(10)
        self.start_rc_override(channel_id=2, pwm=1400)
        # self.set_rc_channel_pwm(2, 1400)
        time.sleep(10)
        self.start_rc_override(channel_id=2, pwm=1500)
        self.set_rc_channel_pwm(2, 1500)

        self.start_rc_override(channel_id=1, pwm=1600)
        # self.set_rc_channel_pwm(1, 1600)
        time.sleep(10)
        self.start_rc_override(channel_id=1, pwm=1400)
        # self.set_rc_channel_pwm(1, 1400)
        time.sleep(10)
        self.start_rc_override(channel_id=1, pwm=1500)
        # self.set_rc_channel_pwm(1, 1500)
        self.stop_rc_override(1)
        self.stop_rc_override(2)

        self.set_mode("LOITER")
        logger.info("LOITER mode for 10 seconds...")
        time.sleep(10)

        self.set_mode("DRIFT")
        logger.info("DRIFT mode for 10 seconds...")
        time.sleep(10)

        self.set_mode("GUIDED")
        logger.info("Switched to GUIDED mode for directional command.")
        # self.goto_north_meters(100)
        self.get_current_position()
        self.goto_offset_from_position_nowait(to_lat, to_lon, to_alt, north_meters=400, east_meters=0, alt=50)
        # self.goto_offset_from_home(north_meters=200, east_meters=0, alt=50)
        time.sleep(5)
        self.set_mode("BRAKE")
        time.sleep(5)
        self.set_mode("GUIDED")
        logger.info("Switched to GUIDED mode for directional command.")
        self.goto_offset_from_position(to_lat, to_lon, to_alt, north_meters=400, east_meters=0, alt=50)
        # self.goto_offset_from_home(north_meters=200, east_meters=0, alt=50)
        

        self.set_mode("RTL")
        logger.info("Returning to Launch...")
        if not self.wait_until_reach_position(to_lat, to_lon, to_alt, threshold=2.0, timeout=200):
            logger.error("Drone did not reach takeoff pos within expected time.")
            self.exit_task()

        self.set_mode("LAND")
        logger.info("Landing initiated...")

        self.master.motors_disarmed_wait()
        logger.info("Mission successfully completed and vehicle disarmed.")
        
        
        
    def run_horizontal_brake_mission(self):
        """Execute specified custom mission."""
        logger.info("")
        logger.info("************************************************")
        logger.info("********** HORIONTAL BRAKE MISSION START *******")
        logger.info("")
        
        # Wait until pre-arm checks pass.
        if not self.wait_for_prearm_good(timeout=60):
        # if not self.wait_until_ready_to_arm(timeout=60):
            logger.error("Prearm condition not met. Aborting mission sequence.")
            self.exit_task()
        time.sleep(5)
            
        self.set_mode("GUIDED")
        if not self.arm_vehicle():
            logger.error("Arming failed.")
            self.exit_task()

        self.takeoff(20)
        logger.info("Hovering for 10 seconds...")
        time.sleep(5)
        
        for spd in range(1, 11): # 1-10 m/s
            self.set_mode("GUIDED")
            self.send_velocity_for_duration(float(spd), 0.0, 0.0)
            # time.sleep(5)
            self.set_mode("BRAKE")
            time.sleep(5)

        self.set_mode("RTL")
        logger.info("Returning to Launch...")

        self.master.motors_disarmed_wait()
        logger.info("Mission successfully completed and vehicle disarmed.")

    def run_vertical_brake_mission(self, start=-10, end=10, step=0.1):
        """Test BRAKE mode with different initial vertical speeds (up & down)."""
        logger.info("")
        logger.info("************************************************")
        logger.info("*********** VERTICAL BRAKE MISSION START *******")
        logger.info("")
        
        # Pre-arm checks
        if not self.wait_for_prearm_good(timeout=60):
            logger.error("Prearm condition not met. Aborting mission sequence.")
            self.exit_task()
        time.sleep(5)

        # Set to GUIDED mode and arm
        self.set_mode("GUIDED")
        to_lat, to_lon, to_alt = self.get_current_position()
        to_alt += 50
        if not self.arm_vehicle():
            logger.error("Arming failed.")
            self.exit_task()

        # Takeoff 
        self.takeoff(100)
        logger.info("Hovering before vertical speed tests...")
        time.sleep(5)

        # Test vertical speeds: positive = ascend, negative = descend
        for vz in np.arange(start, end, step):  
            if vz == 0:
                continue  # skip zero-speed case
            logger.info(f"Testing initial vertical speed: {vz} m/s")
            self.set_mode("GUIDED")
            self.send_velocity_for_duration(0.0, 0.0, float(vz))  # vertical motion only
            self.set_mode("BRAKE")
            time.sleep(5)  # let it brake vertically
            
            # self.goto_gps_position(to_lat, to_lon, to_alt)  # go back to original position
            
        self.set_mode("RTL")
        logger.info("Returning to Launch...")

        self.master.motors_disarmed_wait()
        logger.info("Vertical brake mission completed and vehicle disarmed.")
        
        
    def run_alt_hold_mission(self):    
        """Execute specified custom mission."""
        logger.info("")
        logger.info("************************************************")
        logger.info("************ ALT_HOLD MISSION START! ***********")
        logger.info("")
        
        # Wait until pre-arm checks pass.
        if not self.wait_for_prearm_good(timeout=60):
        # if not self.wait_until_ready_to_arm(timeout=60):
            logger.error("Prearm condition not met. Aborting mission sequence.")
            self.exit_task()
        time.sleep(5)
            
        self.set_mode("GUIDED")

        if not self.arm_vehicle():
            logger.error("Arming failed.")
            self.exit_task()

        self.takeoff(20)
        logger.info("Hovering for 10 seconds...")
        time.sleep(5)
        
        for spd in range(1, 11): # 1-10 m/s
            # record pos here
            # curr_lat, curr_lon, curr_alt = self.get_current_position()
            # time.sleep(1)
            self.set_mode("GUIDED")
            self.send_velocity_for_duration(float(spd), 0.0, 0.0)
            # time.sleep(5)
            self.start_rc_override(channel_id=3, pwm=1500)
            self.set_mode("ALT_HOLD")
            time.sleep(10)
            self.stop_rc_override(3)

        self.set_mode("RTL")
        logger.info("Returning to Launch...")

        self.master.motors_disarmed_wait()
        logger.info("Mission successfully completed and vehicle disarmed.")
    
    def run(self):
        # Register the signal handler
        signal.signal(signal.SIGINT, self.signal_handler)
    
        self.start_sitl()
        self.connect_vehicle()
        self.load_parm_file(default_parm=True)
        # self.load_parm_file()
        self.set_home_origin()
        
        
        self.run_mission()
        # self.run_horizontal_brake_mission()
        # self.run_vertical_brake_mission(-10, 10, 1)
        # self.run_alt_hold_mission()
        
        self.store_flight_data()
        logger.info("Mission complete. Exit...")
        # TODO: exit task on KeyboardInterrupt
        self.exit_task()


def main():
    parser = argparse.ArgumentParser(description="SimTask using pymavlink")
    parser.add_argument('--connect', help="Connection string (default: udp:127.0.0.1:14550)", default="udp:127.0.0.1:14550")
    parser.add_argument('--mission', help="Mission file (QGC WPL 110 format)", 
                        default=os.path.join("Tools", "autotest", "ArduCopter_Tests", "CopterMission", "copter_mission.txt"))
    parser.add_argument('--parm_file', help="Configuration Parameter file (default: None)", default=None)
    args = parser.parse_args()

    task = SimTask(args.connect, args.mission)
    task.parm_file = args.parm_file 
    task.run()

if __name__ == '__main__':
    main()
