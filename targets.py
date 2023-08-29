import numpy as np
import logging

c = 3e8

def get_targets(radar, config_data, logging):
    """
    this function returns all the targets as an array based on the params
    Assume z angle and speed to be 0 for simplicity
    
    # target_3 = dict(location=(4*np.cos(np.radians(true_theta[2])), 4*np.sin(np.radians(true_theta[2])),
    #                 0), speed=(0, -10, 0), rcs=10, phase=0)

    # We will use angle range from -90 to 90 with 
    """
    params = {}
    params["num_target"] = config_data["target"]["num_target"]
    params["x_angles"] = config_data["target"]["x_angle"] # put the angles based on the num target with ind 0 for target 1
    params["z_angles"] = config_data["target"]["z_angle"]
    params["rcs"]=config_data["target"]["rcs"]
    params["x_speed"] = config_data["target"]["x_speed"]
    params["y_speed"] = config_data["target"]["y_speed"]
    params["z_speed"] = config_data["target"]["z_speed"]
    params["phase"] = config_data["target"]["phase"]
    params["dist"]=config_data["target"]["dist"]

    targets = []
    '''
    #Calculate the minimum and maximum range of the radar      
    Here the maximum speed is halved compared to the normal case
    this is due to the positive and negative speed.
    '''
    
    max_range = (c * radar.receiver.fs *
             radar.transmitter.pulse_length /
             radar.transmitter.bandwidth / 2)
    max_speed = (c /radar.transmitter.prp[0] / \
        radar.transmitter.fc_vect[0] / 2) / 2
    
    min_angle = config_data["run_params"]["min_angle"]
    max_angle = config_data["run_params"]["max_angle"]
    
    ranges_ = []
    angles_ = []
    speed_ = []
    for i in range(0, params["num_target"]):
        rand_range = np.random.uniform(0, max_range)
        rand_speed = np.random.uniform(-1*max_speed, max_speed)
        rand_angle = np.random.uniform(min_angle, max_angle)
        ranges_.append(rand_range)
        angles_.append(rand_angle)
        speed_.append(rand_speed)
        """
            Assumptions
                1. Object is in the X-Y plane
                2. Antennas are in Y direction so object is moving in X direction, Y and Z direction speed is 0
                3. rcs co-efficient of all the objects is 10
                4. Phase is 0 for all the objects

        """
        t_ = dict(location=(rand_range*np.cos(np.radians(rand_angle)), rand_range*np.sin(np.radians(rand_angle)), 
                            0), speed=(rand_speed, 0, 0) , rcs= 10,  phase= 0)
        
        logging.info("Object Index " + str(i) )
        logging.info("Rand Range " + str(rand_range))
        logging.info("Rand Speed " + str(rand_speed))
        logging.info("Rand Angle " + str(rand_angle))

        # t_ = dict(location=(params["dist"][i]*np.cos(np.radians(params["x_angles"][i])), params["dist"][i]*np.sin(np.radians(params["x_angles"][i])), 
                            # 0), speed=(params["x_speed"][i], params["y_speed"][i], params["z_speed"][i]), rcs=params["rcs"][i], phase=params["phase"][i])
        targets.append(t_)

    return np.array(ranges_), np.array(angles_), np.array(speed_), targets

    
def main():
    '''
    params
    '''
    # config_data = read_config()
    with open('config.json', 'r') as f:
        config_data = json.load(f)
    params = {}
    params["num_target"] = config_data["target"]["num_targer"]
    params["x_angles"] = config_data["target"]["x_angle"] # put the angles based on the num target with ind 0 for target 1
    params["z_angles"] = config_data["target"]["z_angle"]
    params["rcs"]=config_data["target"]["rcs"]
    params["x_speed"] = config_data["target"]["x_speed"]
    params["y_speed"] = config_data["target"]["y_speed"]
    params["z_speed"] = config_data["target"]["z_speed"]
    params["phase"] = config_data["target"]["phase"]
    params["dist"]=config_data["target"]["dist"]
    targets = get_targets(params)
    print(targets)

if(__name__ == "__main__"):
    main()