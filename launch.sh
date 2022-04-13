#!/bin/bash

set -e

# YOUR CODE BELOW THIS LINE
# ----------------------------------------------------------------------------
echo "This is an empty launch script. Update it to launch your application."

roslaunch circle_drive circle_drive.launch
#roslaunch duckietown_demos lane_following.launch
#roslaunch --wait circle_drive indefinite_navigation2.launch &

#roslaunch --wait <package> launchfile.launch &
#sleep 5
# we put a short sleep in here because rostopic will fail if there's no roscore yet
#rostopic pub /$VEHICLE_NAME/fsm_node/mode duckietown_msgs/FSMState '{header: {}, state: "LANE_FOLLOWING"}'
