# firmware/

`AmazingHand/`: the 8-DOF hand, as a submodule of the fork `r1b4z01d/AmazingHand` (CAD, ESP32 firmware
with the UDP `:8766` / TCP `:8765` line protocol, demos). Flashing is manual (Arduino IDE / PlatformIO,
see its README); rd_arm's hand node is the only client on the robot.

`RD_Battery_Monitor/`: an ESP32 with an INA228 (I2C address 0x40, 2 mOhm shunt, 10 A range) on the battery. Every
250 ms it sends one JSON datagram, `{"Voltage": V, "Current": mA, "Temperature": degC}`, over UDP to port 8889 and
prints the same line on serial at 115200. Before flashing (Arduino IDE; libraries INA228 and ArduinoJson) set the
Wi-Fi name and password and the receiver's address at the top of the sketch; do not commit them. Nothing on the robot
reads the datagrams yet.
