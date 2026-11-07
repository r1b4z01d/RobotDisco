# The off-robot test rig: ROS 2 Jazzy with everything the rd_* packages need except the Isaac ROS and
# CUDA pieces (those are tested on the GPU box, phase 7). Built by rig/rig.sh; the workspace is mounted,
# not copied, so edits on the host are what the container builds.
FROM ros:jazzy-ros-base
ARG DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-lc"]

COPY provision/apt-packages.txt provision/ros-packages.txt /tmp/
RUN apt-get update && apt-get install -y --no-install-recommends \
      $(sed -e 's/#.*//' -e '/^[[:space:]]*$/d' /tmp/apt-packages.txt | grep -vE '^(network-manager|openssh-server|supervisor|chrony|libglfw3|libgl1-mesa-dri|libegl1|mesa-utils)$') \
      $(sed -e 's/#.*//' -e '/^[[:space:]]*$/d' /tmp/ros-packages.txt) \
      ros-jazzy-nav2-loopback-sim \
      \
      ros-jazzy-launch-testing ros-jazzy-launch-testing-ros ros-jazzy-ament-cmake-pytest \
      ros-jazzy-rmw-cyclonedds-cpp ros-jazzy-ros2controlcli ros-jazzy-tf2-geometry-msgs \
      python3-pytest python3-pytest-timeout python3-yaml python3-numpy python3-opencv python3-websockets \
      python3-fastapi python3-uvicorn python3-httpx python3-pil \
    && rm -rf /var/lib/apt/lists/*

ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
    ROS_DOMAIN_ID=0 \
    RCUTILS_COLORIZED_OUTPUT=0 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /ws
RUN echo 'source /opt/ros/jazzy/setup.bash; [ -f /ws/install/setup.bash ] && source /ws/install/setup.bash; export RD_ROBOT_CONFIG=/ws/config/robot.yaml' > /etc/profile.d/rd.sh
ENTRYPOINT ["/bin/bash", "-lc"]
CMD ["bash"]
