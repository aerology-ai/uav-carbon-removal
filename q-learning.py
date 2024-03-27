import logging
import logging.handlers

import pickle
import math
import pytz
import random

import numpy as np

from Py6S import *
from datetime import datetime, timedelta
from timezonefinder import TimezoneFinder

# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

syslog_handler = logging.handlers.SysLogHandler(address='/dev/log')
formatter = logging.Formatter('%(levelname)s - %(module)s.%(funcName)s: %(message)s')
syslog_handler.setFormatter(formatter)

logger = logging.getLogger()
logger.addHandler(syslog_handler)
logger.setLevel(logging.INFO)


with open('/tmp/pycharm_project_709/nasa_power_dict.pkl', 'rb') as pickle_file:
    data_dict = pickle.load(pickle_file)
logging.info(f"Loaded NASA POWER data dictionary with {len(data_dict)} latitudes")

time_step = 3  # Time step in hours
speed = 13  # Speed in m/s

# Constants
earth_circumference_km = 40075
degrees_per_latitude = earth_circumference_km / 360
travel_distance_km = speed * time_step * 3600 / 1000

lat_range = 61  # 0 to 60 degrees inclusive is 61 degrees
degrees_per_step_lat = travel_distance_km / degrees_per_latitude
logging.info(f"Vehicle traveling at {travel_distance_km} km/{time_step} hour(s) can move {degrees_per_step_lat:.4f} "
             f"degrees latitudinally")
num_lat_steps = int(lat_range / degrees_per_step_lat)
logging.info(f"Domain defined by {lat_range} degrees of latitude discretized into {num_lat_steps} steps")

# Longitude steps for the range -20 to -50 at a representative latitude (e.g., 45 degrees)
# TODO - dynamically handle variation in longitudinal distance (but requires 3-d Q-table for each lat)
lon_range = 31
representative_lat = 30
degrees_per_longitude_at_lat = np.cos(np.radians(representative_lat)) * degrees_per_latitude
degrees_per_step_lon = travel_distance_km / degrees_per_longitude_at_lat
logging.info(f"Vehicle traveling at {travel_distance_km} km/{time_step} hour(s) can move {degrees_per_step_lon:.4f} "
             f"degrees longitudinally (representative latitude: {representative_lat})")
num_lon_steps = int(lon_range / degrees_per_step_lon)
logging.info(f"Domain defined by {lon_range} degrees of longitude discretized into {num_lon_steps} steps")

num_days = int(365 / 4)  # 3 months
num_daily_timesteps = int(24 / time_step)
num_time_steps = num_days * num_daily_timesteps
logging.info(f"Temporal space discretized into {num_days} days with {num_daily_timesteps} time steps per day ("
             f"{num_time_steps} total steps)")

num_actions = 9

LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.95
EPISODES = 600000

actual_altitude = 5400

epsilon = 1.0  # Initial epsilon
epsilon_min = 0.01  # Minimum epsilon value
epsilon_decay = 0.9999925  # Decay factor

q_table = np.zeros((num_lat_steps, num_lon_steps, num_days, num_daily_timesteps, num_actions))
logging.info(f"Q-table created with shape: {q_table.shape}")

initial_lat_index = 22
initial_lon_index = 4
# TODO - allow agent to learn optimal starting lat/lon


def est_dni_from_py6s(latitude, longitude, altitude, dthz):
    s = SixS()
    dthz_30 = dthz + timedelta(minutes=30)
    logging.debug(f"Will use midpoint of hour (calculated as {dthz_30})")

    datetime_str = dthz_30.strftime('%Y-%m-%dT%H:%M:%S')
    s.atmos_profile = AtmosProfile.FromLatitudeAndDate(latitude, datetime_str)
    s.altitudes.set_target_custom_altitude(altitude / 1000)

    s.geometry = Geometry.User()
    s.geometry.from_time_and_location(latitude, longitude, datetime_str, 0, 0)

    if s.geometry.solar_z > 90:
        logging.debug(f"Sun is below the horizon at {latitude:.2f}, {longitude:.2f}, {altitude}m at {dthz_30}")
        return 0

    s.run()
    logging.debug(f"DHI at {latitude:.2f}, {longitude:.2f}, {altitude}m at {dthz_30} is "
                  f"{s.outputs.direct_solar_irradiance:.2f} (solar_z: {s.outputs.solar_z})")

    dni = s.outputs.direct_solar_irradiance / math.cos(math.radians(s.outputs.solar_z))
    logging.debug(f"Est. DNI at {latitude:.2f}, {longitude:.2f}, {altitude}m at {dthz_30} is {dni:.2f}")

    return dni


def adjust_dni_for_altitude(lat, lon, dthz, interpolated_sfc_dni):
    # Get DNI at 0m and 5400m
    dni_at_0m = est_dni_from_py6s(lat, lon, 0, dthz)
    dni_at_5400m = est_dni_from_py6s(lat, lon, 5400, dthz)

    # If dni_at_0m is 0, set the adjustment factor to 1, so the original NASA DNI is used
    if dni_at_0m == 0:
        adjustment_factor = 1
    else:
        adjustment_factor = dni_at_5400m / dni_at_0m

    logging.debug(f"Altitude factor for {dthz} at {lat:.2f}, {lon:.2f} is {adjustment_factor:.3f}")

    # Calculate the adjusted DNI
    adjusted_dni = adjustment_factor * interpolated_sfc_dni
    logging.debug(f"Altitude-adjusted DNI at {lat:.2f}, {lon:.2f} at {dthz} is {adjusted_dni:.2f}")

    return adjusted_dni


def bilinear_interpolation(lat, lon, datetime_obj, latitude_range=(0, 60), longitude_range=(-50, -20)):
    """
    Perform bilinear interpolation for a point (lat, lon) at a given datetime using a nested dictionary,
    adjust the DNI value based on altitude, and add both interpolated and adjusted values to the dictionary.
    lat - latitude, lon - longitude, datetime_obj - the specific datetime object for DNI values in UTC.
    lat_range - Tuple indicating the range of latitude values (default is (0, 60)).
    lon_range - Tuple indicating the range of longitude values (default is (-50, -20)).
    data_dict - nested dictionary structure {latitude: {longitude: {unix_timestamp: {'sfc_dni': value}}}}
    """
    unix_timestamp = int(datetime_obj.timestamp())

    # Slight perturbation inward for boundary points
    delta = 1e-6  # Small perturbation value
    if lat == latitude_range[1]:
        lat -= delta  # Slightly decrease lat if it's on the upper boundary
        logging.debug(f"Adjusted latitude to {lat:.7f} for upper boundary")
    elif lat == latitude_range[0]:
        lat += delta  # Slightly increase lat if it's on the lower boundary
        logging.debug(f"Adjusted latitude to {lat:.7f} for lower boundary")

    if lon == longitude_range[0]:
        lon += delta  # Slightly increase lon if it's on the left boundary
        logging.debug(f"Adjusted longitude to {lon:.7f} for left boundary")
    elif lon == longitude_range[1]:
        lon -= delta  # Slightly decrease lon if it's on the right boundary
        logging.debug(f"Adjusted longitude to {lon:.7f} for right boundary")

    if data_dict.get(lat) and data_dict[lat].get(lon) and data_dict[lat][lon].get(unix_timestamp):
        sfc_dni = data_dict[lat][lon][unix_timestamp].get('sfc_dni', float('nan'))
        if math.isnan(sfc_dni):
            logging.warning(f"NaN DNI value found for {lat}, {lon} at {datetime_obj}")

        if 'alt_adj_dni' not in data_dict[lat][lon][unix_timestamp]:
            data_dict[lat][lon][unix_timestamp]['alt_adj_dni'] = adjust_dni_for_altitude(lat,
                                                                                         lon,
                                                                                         datetime_obj,
                                                                                         sfc_dni)
        alt_adj_dni = data_dict[lat][lon][unix_timestamp]['alt_adj_dni']

        logging.debug(f"DNI values (sfc: {sfc_dni:.2f}, alt-adj: {alt_adj_dni:.2f}) found for "
                      f"{lat:.2f}, {lon:.2f} at {datetime_obj}")
        return sfc_dni, alt_adj_dni

    # Identify the grid cell boundaries
    lat_lower = float(int(lat))
    lat_upper = lat_lower + 1
    lon_left = float(int(lon)) if lon > 0 else float(int(lon) - 1)
    lon_right = lon_left + 1
    logging.debug(f"Interpolation for {lat:.2f}, {lon:.2f} uses cell boundaries: lat [{lat_lower}, {lat_upper}], "
                  f"lon [{lon_left}, {lon_right}]")

    values = {}
    for lat_bound in ('lower', 'upper'):
        for lon_bound in ('left', 'right'):
            current_lat = lat_lower if lat_bound == 'lower' else lat_upper
            current_lon = lon_left if lon_bound == 'left' else lon_right

            key_name = f"value_{lat_bound}_{lon_bound}"
            values[key_name] = data_dict.get(
                current_lat, {}
            ).get(
                current_lon, {}
            ).get(
                unix_timestamp, {'sfc_dni': float('nan')}
            )['sfc_dni']
            logging.debug(f"Retrieved {values[key_name]:.2f} DNI for {current_lat}, {current_lon} "
                          f"({lat_bound}-{lon_bound}) at {datetime_obj}")

    # Normalized distances
    lat_diff = lat - lat_lower
    lon_diff = lon - lon_left

    # Bilinear interpolation formula
    interpolated_sfc_dni = (values['value_lower_left'] * (1 - lat_diff) * (1 - lon_diff) +
                            values['value_lower_right'] * (1 - lat_diff) * lon_diff +
                            values['value_upper_left'] * lat_diff * (1 - lon_diff) +
                            values['value_upper_right'] * lat_diff * lon_diff)
    if math.isnan(interpolated_sfc_dni):
        logging.warning(f"NaN DNI value interpolated for {lat}, {lon} at {datetime_obj} (latitude_range[0] is "
                        f"{latitude_range[0]}, latitude_range[1] is {latitude_range[1]})")
    logging.debug(f"Interpolated DNI at {lat:.2f}, {lon:.2f} at {datetime_obj} is {interpolated_sfc_dni:.2f}")

    alt_adj_dni = adjust_dni_for_altitude(lat, lon, datetime_obj, interpolated_sfc_dni)
    if math.isnan(alt_adj_dni):
        logging.warning(f"NaN altitude-adjusted DNI calculated for {lat}, {lon} at {datetime_obj} (latitude_range[0] "
                        f"is {latitude_range[0]}, latitude_range[1] is {latitude_range[1]})")

    data_dict.setdefault(lat, {}).setdefault(lon, {})[unix_timestamp] = {'sfc_dni': interpolated_sfc_dni,
                                                                         'alt_adj_dni': alt_adj_dni}

    return interpolated_sfc_dni, alt_adj_dni


def take_action(current_state, action_key):
    lat_index, lon_index, day_index, hour_index = current_state

    action_effects = {
        0: (1, 0),  # 'N'
        1: (1, 1),  # 'NE'
        2: (0, 1),  # 'E'
        3: (-1, 1),  # 'SE'
        4: (-1, 0),  # 'S'
        5: (-1, -1),  # 'SW'
        6: (0, -1),  # 'W'
        7: (1, -1),  # 'NW'
        8: (0, 0),  # 'STAY'
    }
    logging.debug(f"Vehicle moves {action_effects[action_key]} from {lat_index}, {lon_index}")

    lat_change, lon_change = action_effects[action_key]

    new_lat_index = max(0, min(num_lat_steps - 1, lat_index + lat_change))
    new_lon_index = max(0, min(num_lon_steps - 1, lon_index + lon_change))

    new_time_index = min(day_index * num_daily_timesteps + hour_index + 1, num_time_steps - 1)
    new_day_index = new_time_index // num_daily_timesteps
    new_hour_index = new_time_index % num_daily_timesteps
    logging.debug(f"State at day index {new_day_index}, hour index {new_hour_index} is lat, lon indices "
                  f"{new_lat_index}, {new_lon_index}")

    return new_lat_index, new_lon_index, new_day_index, new_hour_index


def convert_state_to_sensible_values(state_indices, latitude_range=(0, 60), longitude_range=(-50, -20)):
    lat_index, lon_index, day_index, hour_index = state_indices

    # Adjust geo_lat calculation based on the provided lat_range
    geo_lat = ((lat_index / (num_lat_steps - 1)) * abs(latitude_range[0] - latitude_range[1])) + latitude_range[0]

    # Adjust geo_lon calculation based on the provided lon_range
    # Using abs() to ensure the correct step calculation for negative ranges
    geo_lon = ((lon_index / (num_lon_steps - 1)) * abs(longitude_range[0] - longitude_range[1])) + longitude_range[0]

    hours = day_index * 24 + hour_index * time_step
    dthz = datetime(2022, 3, 20, tzinfo=pytz.utc) + timedelta(hours=hours)

    dthl = dthz
    if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
        tf = TimezoneFinder()
        time_zone_str = tf.timezone_at(lat=geo_lat, lng=geo_lon)

        if time_zone_str:
            time_zone = pytz.timezone(time_zone_str)
            dthl = dthz.astimezone(time_zone)
    else:
        # Calculate local time adjustment based on longitude
        local_time_offset = timedelta(hours=geo_lon / 15)
        dthl = dthz + local_time_offset

    return geo_lat, geo_lon, dthz, dthl


def take_action_and_get_reward(start_state, action_integer, altitude_adjusted=True):
    end_state = take_action(start_state, action_integer)
    geo_lat, geo_lon, dthz, dthl = convert_state_to_sensible_values(end_state)
    geo_coords = (geo_lat, geo_lon)

    sfc_dni, alt_adj_dni = bilinear_interpolation(geo_lat, geo_lon, dthz)
    irradiance_reward = alt_adj_dni if altitude_adjusted else sfc_dni
    if irradiance_reward == np.nan:
        logging.warning(f"Reward is nan for {geo_lat}, {geo_lon} at {dthz}")

    logging.debug(f"Vehicle moved to {geo_lat:.2f}, {geo_lon:.2f} at {dthz} ({dthl}) with reward "
                  f"{irradiance_reward:.2f} w/m^2")

    return end_state, irradiance_reward, geo_coords, dthz


previous_q_sum = 0

for episode in range(EPISODES):
    total_reward = 0

    state = (initial_lat_index, initial_lon_index, 0, 0)
    init_geo_lat, init_geo_lon, init_dtmz, init_dthl = convert_state_to_sensible_values(state)

    if episode % 100 == 0 or episode == 0:
        logging.info(f"Episode {episode} - Epsilon: {epsilon:.6f}")
    logging.debug(
        f"Ep. {episode} initialized at {init_dtmz} ({init_dthl}), {init_geo_lat:.2f}, {init_geo_lon:.2f} "
        f"(index state: {state}); epsilon: {epsilon}")

    for step in range(num_time_steps):
        if step == num_time_steps - 1:  # Skip taking action at the last step
            break

        if random.uniform(0, 1) < epsilon:
            action_int = random.randint(0, num_actions - 1)  # Explore
        else:
            action_int = np.argmax(q_table[state])  # Exploit

        new_state, reward, geo_coordinates, datehour_utc = take_action_and_get_reward(state, action_int)
        total_reward += reward

        # Q-table update
        best_future_q = np.max(q_table[new_state])
        q_table[state + (action_int,)] = q_table[state + (action_int,)] + LEARNING_RATE * (
                    reward + DISCOUNT_FACTOR * best_future_q - q_table[state + (action_int,)])
        state = new_state

    average_reward = total_reward / num_time_steps

    if (episode + 1) % 10 == 0:
        previous_q_sum = np.sum(np.abs(q_table))
    if episode % 10 == 0:
        current_q_sum = np.sum(np.abs(q_table))
        q_change = current_q_sum - previous_q_sum
        logging.info(f"Episode {episode} - Q-table change: {q_change:.3f}")

    if episode % 100 == 0:
        logging.info(f"Episode {episode} - average reward: {average_reward:.2f}")
    logging.debug(f"Ep. {episode} ended with average reward: {average_reward:.2f}")

    # Decay epsilon
    epsilon = max(epsilon_min, epsilon_decay * epsilon)

time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
q_table_filename = f'/tmp/pycharm_project_709/q_table_{time}.npy'
np.save(q_table_filename, q_table)
