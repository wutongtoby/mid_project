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
#include <cstring>

#define C4 262
#define CS4 277
#define D4 294
#define DS4 311
#define E4 330
#define F4 350
#define FS4 370
#define G4 392
#define GS4 415
#define A4 440
#define AS4 466
#define B4 494
#define C5 523
#define CS5 554
#define D5 587
#define DS5 622
#define E5 660
#define F5 698
#define FS5 740
#define G5 784
#define GS5 830
#define A5 880
#define AS5 932
#define B5 988

Serial pc(USBTX, USBRX);
DA7212 audio;
uLCD_4DGL uLCD(D1, D0, D2); // serial tx, serial rx, reset pin;
InterruptIn sw2(SW2); // use to pause the song
DigitalIn sw3(SW3);

Thread DNN_thread(osPriorityNormal, 120 * 1024 /*120K stack size*/);
Thread sound_thread; // thread for playing single sound
Thread song_mode_thread(osPriorityNormal); // thead for playing song and selection mode
Thread judge_thread(osPriorityNormal); // thread for taiko judgement

EventQueue song_mode_queue(32 * EVENTS_EVENT_SIZE);
EventQueue sound_queue(32 * EVENTS_EVENT_SIZE);
EventQueue judge_queue(32 * EVENTS_EVENT_SIZE);



int DNN_mode; // control by DNN, will be 0, 1, 2, 3, 4
int DNN_song; // control by DNN, will be 0, 1, 2, 3
int tone_array[4][10] = 
{{E4, D4, C4, D4, E4, F4, G4, A4, B4, C5},
{E5, D5, C5, B4, A4, G4, F4, E4, D4, C4},
{C4, D4, E4, F4, G4, A4, B4, C5, D5, E5},
{A4 , B4, CS5, B4, A4, GS4, FS4, GS4, A4, B4}
};

char taiko_array[15] = {'t', 's', 's', 's', 't', 't', 't', 't', 't', 's', ' ', ' ', ' ', ' ', ' '};
// the taiko_array note array, and will be 'a' or 'b'

bool play_on = true;
bool taiko_on = true;
int which_song = 0; // will be 0, 1, 2, 3
// when we select a song we will give this variable a value
// maybe depend on DNN_song


void playNote(int freq);
void pause(void);  // to be triggered after interrupt
void music(void); // to play song
void load_song(void);
int datatype_transform(char*);
void mode_selection(void);
void song_selection(void);
void DNN(void);
int PredictGesture(float* output);

int16_t waveform[kAudioTxBufferSize];
extern float x, y, z; // to get data from the acceremometer

int main(void) 
{
    sound_thread.start(callback(&sound_queue, &EventQueue::dispatch_forever));
    judge_thread.start(callback(&judge_queue, &EventQueue::dispatch_forever));
    song_mode_thread.start(callback(&song_mode_queue, &EventQueue::dispatch_forever));
    DNN_thread.start(&DNN);
    
    sw2.rise(&pause);
    song_mode_queue.call(&music);
    
    while(true);
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
                if (DNN_mode == 4)
                    DNN_mode = 0;
                else
                    DNN_mode++;

                if (DNN_song == 3)
                    DNN_song = 0;
                else
                    DNN_song++;
            }
            else {
                if (DNN_mode == 0)
                    DNN_mode = 4;
                else
                    DNN_mode--;
                
                if (DNN_song == 0)
                    DNN_song = 3;
                else
                    DNN_song--;
            }
        }
    }
    
    //while (true);
}

void music(void)
{  
    int i, j;  
    uLCD.cls();
    uLCD.printf("\nNow playing song%d \n", which_song); 
    bool taiko_hit;
    // to record that we hit the taiko note or not
    int taiko_score;

    // if the play_on is false, we will jumpt out from this song immediately
    for (i = 0, taiko_score = 0; i < 10 && play_on; i++) {
        if (taiko_on) {
            uLCD.locate(0, 5);
            // the loop below will play the note for the duration of 1s
            uLCD.printf("%c %c %c %c %c %c\n", taiko_array[i], taiko_array[i+1], taiko_array[i+2], taiko_array[i+3], taiko_array[i+4], taiko_array[i+5]);
        }
        for(j = 0; j < kAudioSampleFrequency / kAudioTxBufferSize; ++j) 
            sound_queue.call(playNote, tone_array[which_song][i]);
        
        if (taiko_on) {
            for (j = 0, taiko_hit = false; j < 5; j++) {
                if (taiko_array[i] == 't' && x * x + y * y > z * z && (x * x + y * y) <= 650 || // t than tilt
                    taiko_array[i] == 's' && x * x + y * y < z * z && (x * x + y * y) > 650) { // s than shake
                    taiko_hit = true;
                }
                wait(0.2);
            }
            if (taiko_hit) taiko_score++;
        }
        else 
            wait(1.0);
    }

    if (play_on) {
        sound_queue.call(playNote, 0);
        uLCD.cls();
        uLCD.printf("\nSong %d is over\n", which_song);
        if (taiko_on) {
            uLCD.printf("Final score is %d\n", taiko_score);
        }
    }
}

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


void playNote(int freq) 
{
    for(int i = 0; i < kAudioTxBufferSize; i++) {
        waveform[i] = (int16_t) (sin((double)i * 2. * M_PI/(double) (kAudioSampleFrequency / freq)) * ((1<<16) - 1));
    }
    audio.spk.play(waveform, kAudioTxBufferSize);
}

void pause(void) 
{
    sound_queue.call(playNote, 0);
    taiko_on = false;
    play_on = false;
    song_mode_queue.call(&mode_selection);
}

void load_song(void)
{
    int i = 0;
    char serialInBuffer[4];
    int serialCount = 0;
    
    printf("start\r\n");
    // we will not get data until we receive the starting code, 'z'
    uLCD.cls();
    uLCD.printf("Now Loading...\n");
    while(i < 30) { // 30 is number of total notes
        if(pc.readable()) {
            serialInBuffer[serialCount] = pc.getc();
            serialCount++;
            if(serialCount == 3) { 
            // which means we have got three char 0, 1, 2
                serialInBuffer[serialCount] = '\0';
                tone_array[1][i] = datatype_transform(serialInBuffer);
                serialCount = 0;
                i++;
                printf(serialInBuffer);
                uLCD.locate(0, 3);
                uLCD.printf("%d", i);
            }
        }
    }
    uLCD.cls();
    uLCD.printf("Loading complete1\n");
}

int datatype_transform(char * tone)
{
    if (strcmp(tone, "_C4") == 0)
        return C4;
    else if (strcmp(tone, "CS4") == 0)
        return CS4;
    else if (strcmp(tone, "_D4") == 0)
        return D4;
    else if (strcmp(tone, "DS4") == 0)
        return DS4;
    else if (strcmp(tone, "_E4") == 0)
        return E4;
    else if (strcmp(tone, "_F4") == 0)
        return F4;
    else if (strcmp(tone, "FS4") == 0)
        return FS4;
    else if (strcmp(tone, "_G4") == 0)
        return G4;
    else if (strcmp(tone, "GS4") == 0)
        return GS4;
    else if (strcmp(tone, "_A4") == 0)
        return A4;
    else if (strcmp(tone, "AS4") == 0)
        return AS4;
    else if (strcmp(tone, "_B4") == 0)
        return B4;
    else if (strcmp(tone, "_C5") == 0)
        return C5;
    else if (strcmp(tone, "CS5") == 0)
        return CS5;
    else if (strcmp(tone, "_D5") == 0)
        return D5;
    else if (strcmp(tone, "DS5") == 0)
        return DS5;
    else if (strcmp(tone, "_E5") == 0)
        return E5;
    else if (strcmp(tone, "_F5") == 0)
        return F5;
    else if (strcmp(tone, "FS5") == 0)
        return FS5;
    else if (strcmp(tone, "_G5") == 0)
        return G5;
    else if (strcmp(tone, "GS5") == 0)
        return GS5;
    else if (strcmp(tone, "_A5") == 0)
        return A5;
    else if (strcmp(tone, "AS5") == 0)
        return AS5;
    else if (strcmp(tone, "_B5") == 0)
        return B5;
    else
        return C4;
}

void mode_selection(void) 
{
    uLCD.cls();
    uLCD.printf(" forward\n backward\n change songs\n load song\n taiko mode\n");
    int now_DNN_mode = 0;
    while (1) {
        now_DNN_mode =DNN_mode; 
        // get new DNN_mode value, to get new state input
        
        uLCD.locate(0, now_DNN_mode);
        uLCD.printf("^");

        if (sw3 == 0) {// if the button been pressed
            if (now_DNN_mode == 0) { // forward
                if (which_song == 3)
                    which_song = 0;
                else
                    which_song++;
                taiko_on = false;
                play_on = true;
                song_mode_queue.call(music);
                return;
            }
            else if (now_DNN_mode == 1) { // backward
                if (which_song == 0)
                    which_song = 3;
                else
                    which_song--;
                taiko_on = false;
                play_on = true;
                song_mode_queue.call(music);
                return;
            }
            else if (now_DNN_mode == 2)  {// change songs
                wait(0.5);
                song_mode_queue.call(song_selection);
                return;
            }
            else if (now_DNN_mode == 3) { // load songs
                song_mode_queue.call(load_song);
                return;
            }
            else {// taiko mode
                which_song = 0;
                taiko_on = true;
                play_on = true;
                song_mode_queue.call(music);
                return;
            }
        }
        wait(0.05);
        uLCD.locate(0, now_DNN_mode);
        uLCD.printf(" ");
    }
}

void song_selection(void)
{
    uLCD.cls();
    int now_DNN_song = 0;
    uLCD.printf(" song0\n song1\n song2\n song3\n");

    while (1) {
        now_DNN_song = DNN_song;
        uLCD.locate(0, now_DNN_song);
        uLCD.printf("^");

        if (sw3 == 0) {// if the button been pressed
            wait(0.5); // add a little delay
            switch (DNN_song) {
                case 0: which_song = 0; break;
                case 1: which_song = 1; break;
                case 2: which_song = 2; break;
                default: which_song = 3; 
            }
            taiko_on = false;
            play_on = true;
            song_mode_queue.call(music);
            return;
        }
        wait(0.05);
        uLCD.locate(0, now_DNN_song);
        uLCD.printf(" ");
    }
}