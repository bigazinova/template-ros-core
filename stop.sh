#rostopic pub /$VEHICLE_NAME/fsm_node/mode duckietown_msgs/FSMState '{header: {}, state: "NORMAL_JOYSTICK_CONTROL"}'
rostopic pub -r 10 /{hostname}/joy sensor_msgs/Joy '{header:{},axes:[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],buttons:[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]}'
