#include <WiFi.h>
#include <WiFiUdp.h>
#include "INA228.h"
#include <ArduinoJson.h>


INA228 INA(0x40);
JsonDocument batteryPayload;

const char* ssid = "YOUR_SSID";
const char* password = "YOUR_PASSWORD";

WiFiUDP udp;
const unsigned int localPort = 8889;
IPAddress remoteIp(192, 168, 1, 100);  // Replace with your PC/receiver IP
const unsigned int remotePort = 8889;

void setup() {
  Serial.begin(115200);
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  udp.begin(localPort);

  Wire.begin();
  if (!INA.begin()) {
    Serial.println("Could not connect. Fix and Reboot");
    while (1)
      ;
  }

  INA.setMaxCurrentShunt(10, 0.002);
}

void loop() {

  batteryPayload["Voltage"] = INA.getBusVoltage();
  batteryPayload["Current"] = INA.getMilliAmpere();
  batteryPayload["Temperature"] = INA.getTemperature();

  String jsonOutput;
  serializeJson(batteryPayload, jsonOutput);

  Serial.println(jsonOutput);

  udp.beginPacket(remoteIp, remotePort);
  udp.print(jsonOutput);
  udp.endPacket();

  delay(250);  // ~50 fps / updates per second
}