#include "accelerometer_handler.h"
#include "config.h"
#include "magic_wand_model_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include "mbed.h"
#include "DA7212.h"
#include "uLCD_4DGL.h"
#include <cmath>

Serial pc(USBTX, USBRX);
DA7212 audio;
// the thread to run the dnn independently
Thread DNN_thread(osPriorityNormal, 120 * 1024 /*120K stack size*/); 
// the thread for mode selection and song selection
Thread selection_thread(osPriorityBelowNormal, 50 * 1024 /*50K stack size*/);
// the thread to use to play note
Thread song_thread(osPriorityLow);
// the thread to judge if we hit the taiko note
Thread taiko_thread(osPriorityLow);

EventQueue selection_queue(32 * EVENTS_EVENT_SIZE);
EventQueue song_queue(32 * EVENTS_EVENT_SIZE);
EventQueue taiko_queue(32 * EVENTS_EVENT_SIZE);

uLCD_4DGL uLCD(D1, D0, D2); // serial tx, serial rx, reset pin;
InterruptIn sw2(SW2); // use to pause the song
DigitalIn sw3(SW3);

int which_modeORsong; // control by DNN, will be 0, 1, 2, 3, 4
bool song_set = true; // the song set, and there is 2 set, each set with 3 songs
int tone_array[4][10] = 
{{330, 294, 262, 294, 330, 350, 392, 440, 494, 524},
{0,0,0,0,0,0,0,0,0,0},
{0,0,0,0,0,0,0,0,0,0},
{0,0,0,0,0,0,0,0,0,0}
} 
// there is 4 song and with each have total 10 notes
char taiko_array[13] = 
{'a', 'b', 'b', 'a', 'b', 'b', 'a', 'a', 'b', 'a', '\0', '\0', '\0'};
// the taiko_array note array, and will be 'a' or 'b'
bool taiko_hit;
// to record that we hit the taiko note or not

bool play_on = true;
bool taiko_on = false;
int which_song = 0; // will be 0, 1, 2, 3
// when we select a song we will give this variable a value
// maybe depend on which_modeORsong

void taiko_hit_judge(char taiko_note);
void playNote(int freq);
void pause(void); 
// to be triggered after interrupt

void mode_selection(void);
void song_selection(void);

int PredictGesture(float* output);
void DNN(void);

extern float x, y, z;

int main(void) 
{
    DNN_thread.start(DNN);
    selection_thread.start(callback(&selection_queue, &EventQueue::dispatch_forever));
    song_thread.start(callback(&song_queue, &EventQueue::dispatch_forever));
    taiko_thread.start(callback(&taiko_queue, &EventQueue::dispatch_forever));
    sw2.rise(&pause);

    // the infinite loop to wait for plaing music
    while (1) {
        int i;
        int taiko_score;
        
        // if the play_on is false, we will jumpt out from this song immediately
        for (i = 0, taiko_hit = 0, taiko_score = 0; i < 10 && play_on; i++) {
            uLCD.reset();
            uLCD.printf("\n Now playing sond: %d \n", which_song); 
            if (taiko_on)
                uLCD.printf("%c %c %c %c\n", taiko_array[i], taiko_array[i+1], taiko_array[i+2], taiko_array[i+3]);
            // the loop below will play the note for the duration of 1s
            for(int j = 0; j < kAudioSampleFrequency / kAudioTxBufferSize; ++j) {
                song_queue.call(playNote, tone_array[which_song][i]);
            }
            
            // set the initial value to false
            taiko_hit = false;
            int idC = taiko_queue.call_every(0.2, taiko_hit_judge, taiko_array[i]);
            wait(1.0);
            taiko_queue.cancel(idC); // cancel the task
            
            if (taiko_hit) taiko_score++;
        }
        // i == 10 means that the songs complete without interrupt
        // so we have to manually turn off the song
        if (i == 10) {
            play_on = false; // manually turn off
            if (taiko_on) {
                uLCD.printf("the final score is %d\n", taiko_score);
                taiko_on = false; // turn this off, so that we just print one time
            }
        }
    }
}

// Return the result of the last prediction
int PredictGesture(float* output) 
{
    // How many times the most recent gesture has been matched in a row
    static int continuous_count = 0;
    // The result of the last prediction
    static int last_predict = -1;

    // Find whichever output has a probability > 0.8 (they sum to 1)
    int this_predict = -1;
    for (int i = 0; i < label_num; i++) {
        if (output[i] > 0.8) this_predict = i;
    }

    // No gesture was detected above the threshold
    if (this_predict == -1) {
        continuous_count = 0;
        last_predict = label_num;
        return label_num;
    }

    if (last_predict == this_predict) {
        continuous_count += 1;
    } 
    else {
        continuous_count = 0;
    }
    last_predict = this_predict;

    // If we haven't yet had enough consecutive matches for this gesture,
    // report a negative result
    if (continuous_count < config.consecutiveInferenceThresholds[this_predict]) {
        return label_num;
    }
    // Otherwise, we've seen a positive result, so clear all our variables
    // and report it
    continuous_count = 0;
    last_predict = -1;

    return this_predict;
}

void DNN(void) 
{
    // Create an area of memory to use for input, output, and intermediate arrays.
    // The size of this will depend on the model you're using, and may need to be
    // determined by experimentation.
    constexpr int kTensorArenaSize = 60 * 1024;
    uint8_t tensor_arena[kTensorArenaSize];

    // Whether we should clear the buffer next time we fetch data
    bool should_clear_buffer = false;
    bool got_data = false;

    // The gesture index of the prediction
    int gesture_index;

    // Set up logging.
    static tflite::MicroErrorReporter micro_error_reporter;
    tflite::ErrorReporter* error_reporter = &micro_error_reporter;

    // Map the model into a usable data structure. This doesn't involve any
    // copying or parsing, it's a very lightweight operation.
    const tflite::Model* model = tflite::GetModel(g_magic_wand_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        error_reporter->Report(
            "Model provided is schema version %d not equal "
            "to supported version %d.",
            model->version(), TFLITE_SCHEMA_VERSION);
        return -1;
    }

    // Pull in only the operation implementations we need.
    // This relies on a complete list of all the ops needed by this graph.
    // An easier approach is to just use the AllOpsResolver, but this will
    // incur some penalty in code space for op implementations that are not
    // needed by this graph.
    static tflite::MicroOpResolver<6> micro_op_resolver;
    micro_op_resolver.AddBuiltin(
        tflite::BuiltinOperator_DEPTHWISE_CONV_2D,
        tflite::ops::micro::Register_DEPTHWISE_CONV_2D());
    micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_MAX_POOL_2D,
                                tflite::ops::micro::Register_MAX_POOL_2D());
    micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_CONV_2D,
                                tflite::ops::micro::Register_CONV_2D());
    micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_FULLY_CONNECTED,
                                tflite::ops::micro::Register_FULLY_CONNECTED());
    micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_SOFTMAX,
                                tflite::ops::micro::Register_SOFTMAX());
    micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_RESHAPE,
                                tflite::ops::micro::Register_RESHAPE(), 1);
    // Build an interpreter to run the model with
    static tflite::MicroInterpreter static_interpreter(
        model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
    tflite::MicroInterpreter* interpreter = &static_interpreter;

    // Allocate memory from the tensor_arena for the model's tensors
    interpreter->AllocateTensors();

    // Obtain pointer to the model's input tensor
    TfLiteTensor* model_input = interpreter->input(0);
    if ((model_input->dims->size != 4) || (model_input->dims->data[0] != 1) ||
        (model_input->dims->data[1] != config.seq_length) ||
        (model_input->dims->data[2] != kChannelNumber) ||
        (model_input->type != kTfLiteFloat32)) {
        error_reporter->Report("Bad input tensor parameters in model");
        return -1;
    }

    int input_length = model_input->bytes / sizeof(float);

    TfLiteStatus setup_status = SetupAccelerometer(error_reporter);
    if (setup_status != kTfLiteOk) {
        error_reporter->Report("Set up failed\n");
        return -1;
    }

    error_reporter->Report("Set up successful...\n");

    while (true){

        // Attempt to read new data from the accelerometer
        got_data = ReadAccelerometer(error_reporter, model_input->data.f,
                                    input_length, should_clear_buffer);

        // If there was no new data,
        // don't try to clear the buffer again and wait until next time
        if (!got_data) {
            should_clear_buffer = false;
            continue;
        }

        // Run inference, and report any error
        TfLiteStatus invoke_status = interpreter->Invoke();
        if (invoke_status != kTfLiteOk) {
            error_reporter->Report("Invoke failed on index: %d\n", begin_index);
            continue;
        }

        // Analyze the results to obtain a prediction
        gesture_index = PredictGesture(interpreter->output(0)->data.f);

        // Clear the buffer next time we read data
        should_clear_buffer = gesture_index < label_num;

        // Produce an output
        // the gestrue_index returned by the PredictGesture() will return label_num
        // if it did'nt detect anything
        if (gesture_index < label_num) {
            error_reporter->Report(config.output_message[gesture_index]);
            if (gesture_index == 0) {
                if (which_modeORsong == 4)
                    which_modeORsong = 0;
                else
                    which_modeORsong++;
            }
            else {
                if (which_modeORsong == 0)
                    which_modeORsong = 4;
                else
                    which_modeORsong--;
            }
        }
    }
}

void playNote(int freq) 
{
    for(int i = 0; i < kAudioTxBufferSize; i++) {
        waveform[i] = (int16_t) (sin((double)i * 2. * M_PI/(double) (kAudioSampleFrequency / freq)) * ((1<<16) - 1));
    }
    audio.spk.play(waveform, kAudioTxBufferSize);
}

void pause(void) 
{
    play_on = false;
    taiko_on = false;
    selection_queue.call(&mode_selection);
}

void taiko_hit_judge(char taiko_note)
{
    // if tilt and not moving than we hit a
    if (taiko_note == 'a' && x * x + y * y > z * z && 
        (x * x + y * y) <= 650) {
            taiko_hit = true;
    }
    // if not tilt but moving than we hit b
    if (taiko_note == 'b' && x * x + y * y <= z * z && 
        (x * x + y * y) > 650) {
            taiko_hit = true;
    }
}

void mode_selection(void) 
{
    uLCD.reset();
    while (1) {
        if (which_modeORsong == 0)
            uLCD.printf("^forward\n backward\n change songs\n load song\n taiko mode\n");
        else if (which_modeORsong == 1)
            uLCD.printf(" forward\n^backward\n change songs\n load song\n taiko mode\n");
        else if (which_modeORsong == 2)
            uLCD.printf(" forward\n backward\n^change songs\n load song\n taiko mode\n");
        else if (which_modeORsong == 3)
            uLCD.printf(" forward\n backward\n change songs\n^load song\n taiko mode\n");
        else 
            uLCD.printf(" forward\n backward\n change songs\n^load song\n^taiko mode\n");

        if (sw3 == 0) {// if the button been pressed
            if (which_modeORsong == 0) { // forward
                if (which_song == 3)
                    which_song = 0;
                else
                    which_song++;
                play_on = true;
                return;
            }
            else if (which_modeORsong == 1) { // backward
                if (which_song == 0)
                    which_song = 3;
                else
                    which_song--;
                play_on = true;
                return;
            }
            else if (which_modeORsong == 2)  {// change songs
                selection_queue.call(song_selection);
                return;
            }
            else if (which_modeORsong == 3) { // load songs
                int i = 0;
                char serialInBuffer[4];
                int serialCount = 0;

                while(i < 30) { // 30 is number of total notes
                    if(pc.readable()) {
                        serialInBuffer[serialCount] = pc.getc();
                        serialCount++;
                        if(serialCount == 3) {
                            serialInBuffer[serialCount] = '\0';
                            *(tone_array[1] + i) = (int) atoi(serialInBuffer);
                            serialCount = 0;
                            i++;
                        }
                    }
                }
            }
            else {// taiko mode
                which_song = 0;
                taiko_on = true;
                play_on = true;
                return;
            }
        }
        wait(0.1);
        uLCD.reset();
    }
}

void song_selection(void)
{
    uLCD.reset();
    while (1) {
        if (which_modeORsong == 0)
            uLCD.printf("^song1\n song2\n song3\n song4\n");
        else if (which_modeORsong == 1)
            uLCD.printf(" song1\n^song2\n song3\n song4\n");
        else if (which_modeORsong == 2)
            uLCD.printf(" song1\n song2\n^song3\n song4\n");
        else 
            uLCD.printf(" song1\n song2\n song3\n^song4\n");

        if (sw3 == 0) {// if the button been pressed
            switch (which_modeORsong) {
                case 0: which_song = 0; break;
                case 1; which_song = 1; break;
                case 2: which_song = 2; break;
                default: which_song = 3; 
            }
            play_on = true;
            return ;
        }
        wait(0.1);
        uLCD.reset();
}