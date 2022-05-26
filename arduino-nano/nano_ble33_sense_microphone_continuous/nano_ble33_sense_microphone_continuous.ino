/* Edge Impulse Arduino examples
 * Copyright (c) 2021 EdgeImpulse Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

// If your target is limited in memory remove this macro to save 10K RAM
#define EIDSP_QUANTIZE_FILTERBANK   0

/**
 * Define the number of slices per model window. E.g. a model window of 1000 ms
 * with slices per model window set to 4. Results in a slice size of 250 ms.
 * For more info: https://docs.edgeimpulse.com/docs/continuous-audio-sampling
 */
#define EI_CLASSIFIER_SLICES_PER_MODEL_WINDOW 3

/*
 ** NOTE: If you run into TFLite arena allocation issue.
 **
 ** This may be due to may dynamic memory fragmentation.
 ** Try defining "-DEI_CLASSIFIER_ALLOCATION_STATIC" in boards.local.txt (create
 ** if it doesn't exist) and copy this file to
 ** `<ARDUINO_CORE_INSTALL_PATH>/arduino/hardware/<mbed_core>/<core_version>/`.
 **
 ** See
 ** (https://support.arduino.cc/hc/en-us/articles/360012076960-Where-are-the-installed-cores-located-)
 ** to find where Arduino installs cores on your machine.
 **
 ** If the problem persists then there's not enough memory for this model and application.
 */

/* Includes ---------------------------------------------------------------- */
#include "mbed.h"
#include <PDM.h>
#include <cough-monitor-audio_Created_by_Eivind_Holt__inferencing.h>
#include <ArduinoBLE.h>

const char* uuidOfService = "00001101-0000-1000-8000-00805f9b34fb";
const char* uuidOfTxChar = "00001143-0000-1000-8000-00805f9b34fb";
BLEService coughService(uuidOfService);
BLEUnsignedIntCharacteristic coughCounterCharacteristic(uuidOfTxChar, BLERead | BLENotify | BLEBroadcast);

/** Audio buffers, pointers and selectors */
typedef struct {
    signed short *buffers[2];
    unsigned char buf_select;
    unsigned char buf_ready;
    unsigned int buf_count;
    unsigned int n_samples;
} inference_t;

static inference_t inference;
static bool record_ready = false;
static signed short *sampleBuffer;
static bool debug_nn = false; // Set this to true to see e.g. features generated from the raw signal
static int print_results = -(EI_CLASSIFIER_SLICES_PER_MODEL_WINDOW);
static bool test_leds = false; // Set this to true to test the LEDs
static u_int16_t number_of_coughs = 0;
static unsigned long cough_led_start_time = 0;
static const u_int16_t COUGH_LED_DURATION = 500;
#define ON          0
#define OFF         1
#define LEDR        p24
#define LEDG        p16
#define LEDB        p6
#define PROBABILITY_THR 0.5

static mbed::DigitalOut rgb[] = {LEDR, LEDG, LEDB};

static size_t current_color = 0;

static bool is_cough(size_t ix) {
  if(ix == 0) {
    return true;
  }
  else {
    return false;
  }
}

void setup()
{
    Serial.begin(115200);
    waitForSerial(3000);
    Serial1.begin(115200);
    ei_printf("ei-cough-monitor-audio-(created-by-eivind-holt)-arduino-1.0.5");

    rgb[0] = OFF;
    rgb[1] = OFF;
    rgb[2] = OFF;
  
    if(test_leds)
    {
      delay(2000);
      Serial.print("Testing LEDs\n");
      Serial.print("RED\n");
      rgb[0] = ON;
      delay(3000);
      rgb[0] = OFF;
      Serial.print("GREEN\n");
      rgb[1] = ON;
      delay(3000);
      rgb[1] = OFF;
      Serial.print("BLUE\n");
      rgb[2] = ON;
      delay(3000);
      rgb[2] = OFF;
    }

    // summary of inferencing settings (from model_metadata.h)
    ei_printf("Inferencing settings:\n");
    ei_printf("\tInterval: %.2f ms.\n", (float)EI_CLASSIFIER_INTERVAL_MS);
    ei_printf("\tFrame size: %d\n", EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE);
    ei_printf("\tSample length: %d ms.\n", EI_CLASSIFIER_RAW_SAMPLE_COUNT / 16);
    ei_printf("\tNo. of classes: %d\n", sizeof(ei_classifier_inferencing_categories) /
                                            sizeof(ei_classifier_inferencing_categories[0]));

    run_classifier_init();
    if (microphone_inference_start(EI_CLASSIFIER_SLICE_SIZE) == false) {
        ei_printf("ERR: Failed to setup audio sampling\r\n");
        return;
    }

    if (!BLE.begin()) {
        ei_printf("starting BLE failed!");
        while (1);
    }

    BLE.setLocalName("CoughMonitor");
    BLE.setAdvertisedService(coughService); // add the service UUID
    coughService.addCharacteristic(coughCounterCharacteristic); // add the cough counter characteristic
    BLE.addService(coughService); // Add the cough service
    BLE.setEventHandler(BLEConnected, blePeripheralConnectHandler);
    BLE.setEventHandler(BLEDisconnected, blePeripheralDisconnectHandler);
    coughCounterCharacteristic.writeValue(0); // set initial value for this characteristic
    BLE.advertise();
    ei_printf("Bluetooth® device active, waiting for connections...");
}

void loop()
{
    disableUnusedComponents();
    BLE.poll();
    unsigned long currentMillis = millis();
    if ((cough_led_start_time + COUGH_LED_DURATION) < currentMillis){
        rgb[current_color] = OFF;
    }
    
    bool m = microphone_inference_record();
    if (!m) {
        ei_printf("ERR: Failed to record audio...\n");
        return;
    }

    signal_t signal;
    signal.total_length = EI_CLASSIFIER_SLICE_SIZE;
    signal.get_data = &microphone_audio_signal_get_data;
    ei_impulse_result_t result = {0};

    EI_IMPULSE_ERROR r = run_classifier_continuous(&signal, &result, debug_nn);
    if (r != EI_IMPULSE_OK) {
        ei_printf("ERR: Failed to run classifier (%d)\n", r);
        return;
    }

    if (++print_results >= (EI_CLASSIFIER_SLICES_PER_MODEL_WINDOW))
    {
        // print the predictions
        ei_printf("Predictions ");
        ei_printf("(DSP: %d ms., Classification: %d ms., Anomaly: %d ms.)",
                  result.timing.dsp, result.timing.classification, result.timing.anomaly);
        ei_printf(": \n");

        // Get the index with higher probability
        size_t ix_max = 0;
        float  pb_max = 0;
        for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
            ei_printf("    %s: %.2f\n", result.classification[ix].label,
                result.classification[ix].value);

            if(result.classification[ix].value > pb_max) {
                ix_max = ix;
                pb_max = result.classification[ix].value;
            }
        }

    if(pb_max > PROBABILITY_THR) {
        if(is_cough(ix_max)) {
            /* ei_printf("Predictions ");
            ei_printf("(DSP: %d ms., Classification: %d ms., Anomaly: %d ms.)",
                    result.timing.dsp, result.timing.classification, result.timing.anomaly);
            ei_printf(": \n");

            ei_printf("    %s: %.2f\n", result.classification[0].label,
                result.classification[0].value); */
            updateCoughCounter();

            cough_led_start_time = millis();
            rgb[current_color] = ON;
        }

        #if EI_CLASSIFIER_HAS_ANOMALY == 1
                ei_printf("    anomaly score: %.3f\n", result.anomaly);
        #endif

        print_results = 0;
        record_ready = true;
        }
    }
}

void waitForSerial(unsigned long timeout_millis) {
  unsigned long start = millis();
  while (!Serial) {
    if (millis() - start > timeout_millis)
      break;
  }
}

void disableUnusedComponents() {
    digitalWrite(LED_PWR, LOW); // Disable power led after setup
    digitalWrite(PIN_ENABLE_I2C_PULLUP, LOW);     //Turn off the I2C pull-up resistors
    //digitalWrite(PIN_ENABLE_SENSORS_3V3, LOW); // Disable sensors
}

/**
 * @brief      PDM buffer full callback
 *             Get data and call audio thread callback
 */
static void pdm_data_ready_inference_callback(void)
{
    int bytesAvailable = PDM.available();

    // read into the sample buffer
    int bytesRead = PDM.read((char *)&sampleBuffer[0], bytesAvailable);

    if (record_ready == true) {
        for (int i = 0; i<bytesRead>> 1; i++) {
            inference.buffers[inference.buf_select][inference.buf_count++] = sampleBuffer[i];

            if (inference.buf_count >= inference.n_samples) {
                inference.buf_select ^= 1;
                inference.buf_count = 0;
                inference.buf_ready = 1;
            }
        }
    }
}

/**
 * @brief      Init inferencing struct and setup/start PDM
 *
 * @param[in]  n_samples  The n samples
 *
 * @return     { description_of_the_return_value }
 */
static bool microphone_inference_start(uint32_t n_samples)
{
    inference.buffers[0] = (signed short *)malloc(n_samples * sizeof(signed short));

    if (inference.buffers[0] == NULL) {
        return false;
    }

    inference.buffers[1] = (signed short *)malloc(n_samples * sizeof(signed short));

    if (inference.buffers[1] == NULL) {
        free(inference.buffers[0]);
        return false;
    }

    sampleBuffer = (signed short *)malloc((n_samples >> 1) * sizeof(signed short));

    if (sampleBuffer == NULL) {
        free(inference.buffers[0]);
        free(inference.buffers[1]);
        return false;
    }

    inference.buf_select = 0;
    inference.buf_count = 0;
    inference.n_samples = n_samples;
    inference.buf_ready = 0;

    // configure the data receive callback
    PDM.onReceive(&pdm_data_ready_inference_callback);

    PDM.setBufferSize((n_samples >> 1) * sizeof(int16_t));

    // initialize PDM with:
    // - one channel (mono mode)
    // - a 16 kHz sample rate
    if (!PDM.begin(1, EI_CLASSIFIER_FREQUENCY)) {
        ei_printf("Failed to start PDM!");
    }

    // set the gain, defaults to 20
    PDM.setGain(127); //127

    record_ready = true;

    return true;
}

/**
 * @brief      Wait on new data
 *
 * @return     True when finished
 */
static bool microphone_inference_record(void)
{
    bool ret = true;

    if (inference.buf_ready == 1) {
        ei_printf(
            "Error sample buffer overrun. Decrease the number of slices per model window "
            "(EI_CLASSIFIER_SLICES_PER_MODEL_WINDOW)\n");
        ret = false;
    }

    while (inference.buf_ready == 0) {
        delay(1);
    }

    inference.buf_ready = 0;

    return ret;
}

/**
 * Get raw audio signal data
 */
static int microphone_audio_signal_get_data(size_t offset, size_t length, float *out_ptr)
{
    numpy::int16_to_float(&inference.buffers[inference.buf_select ^ 1][offset], out_ptr, length);

    return 0;
}

/**
 * @brief      Stop PDM and release buffers
 */
static void microphone_inference_end(void)
{
    PDM.end();
    free(inference.buffers[0]);
    free(inference.buffers[1]);
    free(sampleBuffer);
}

#if !defined(EI_CLASSIFIER_SENSOR) || EI_CLASSIFIER_SENSOR != EI_CLASSIFIER_SENSOR_MICROPHONE
#error "Invalid model for current sensor."
#endif

void updateCoughCounter() {
    coughCounterCharacteristic.writeValue(++number_of_coughs);  // and update the cough counter characteristic
    ei_printf("Cough count: %d\n", number_of_coughs); // print it
    Serial1.print("Cough count: ");
    Serial1.println(number_of_coughs);
}

void blePeripheralConnectHandler(BLEDevice central) {
  // central connected event handler
  ei_printf("Connected event, central: %s+n", central.address());
  coughCounterCharacteristic.broadcast();// .writeValue(number_of_coughs);
}

void blePeripheralDisconnectHandler(BLEDevice central) {
  // central disconnected event handler
  ei_printf("Disconnected event, central: %s\n", central.address());
}