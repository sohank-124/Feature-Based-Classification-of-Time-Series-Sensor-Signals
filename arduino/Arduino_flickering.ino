// Streams timestamped photoresistor data 
// Output format: "milliseconds,ADC"

const int sensorPin = A0;        // Photoresistor input

// LED used as a controlled optical noise source
const int ledPin = 6;            // Digital output driving LED

// Allows the user to quickly change the frequency of LED flickering
const int MODE = 2;  
// 0 = Off
// 1 = 10Hz
// 2 = 20Hz


const int sampleRate = 50;       // Target sampling rate (Hz)
const unsigned long sampleInterval = 1000UL / sampleRate;

// Flicker timing
unsigned long blinkMs;
unsigned long lastBlink = 0;
bool ledOn = false;

unsigned long lastSample = 0;

void setup() {
  Serial.begin(115200);          // Must match analysis-side baud rate
  pinMode(ledPin, OUTPUT);
  digitalWrite(ledPin, LOW);     // Default OFF

  // Set flicker rate based on MODE
  if (MODE == 1) {
    blinkMs = 50;    // 10 Hz (toggle every 50 ms)
  } else if (MODE == 2) {
    blinkMs = 25;    // 20 Hz (toggle every 25 ms)
  } else {
    blinkMs = 0;     // Off
  }

  delay(800);                    // Allow serial connection to stabilize
}

void loop() {
  unsigned long t = millis();

  // blinks at a Hz depending on the mode
  if (MODE != 0 && (t - lastBlink >= blinkMs)) {
    ledOn = !ledOn;
    digitalWrite(ledPin, ledOn);
    lastBlink = t;
  }

  // Sample sensor and transmit timestamped value
  if (t - lastSample >= sampleInterval) {
    int adc = analogRead(sensorPin);
    Serial.print(t);
    Serial.print(",");
    Serial.println(adc);
    lastSample = t;
  }
}
