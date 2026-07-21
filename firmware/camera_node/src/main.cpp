#include <Arduino.h>

int iter = 0;

void setup()
{
  Serial.begin(115200);
  delay(2000);
  Serial.println("Hello from keshik camera_node");
}

void loop()
{
  Serial.print("Alive, ");
  Serial.println(iter);
  iter++;
  delay(1000);
}