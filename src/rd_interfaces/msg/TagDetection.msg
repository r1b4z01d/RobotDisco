# One AprilTag seen by a camera, in the camera's optical frame. Produced by rd_perception from the
# detector in use (Isaac ROS AprilTag on the robot; a scripted publisher in the rig), consumed by rd_nav's
# tag corrector. Keeps the Isaac message type out of every other package.
int32 id
string family                       # "tag36h11"
float32 size                        # edge length of the black border, metres
geometry_msgs/PoseStamped pose      # tag pose in header.frame_id (the camera optical frame)
float32 decision_margin             # detector confidence, 0 when unknown
