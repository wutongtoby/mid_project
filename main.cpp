// will eventqueue pop up the task after the former task is completed?
// if so, we can actually put sound_and_ulcd, mode_selection, sound_selection 
// in the same thread, since they all won't funciton at the same time?

// maybe length_array, onte_array, can just be 2-dimensional array
// so that we just need to indicate which 1-dimensional array we want to play?

int *length_array;
int *tone_array;
int number_of_note;
int *table;
bool play_on, taiko_on;
void sound_and_ulcd(void); 
// the function is used to 
//	1. play the sound from table + length_array + tone_array
//	2. if play_on then play sound and vice versa
//	3. show song info on the uLCD
//	4. display taiko pattern if taiko_on
// and the function will be in a single thread with lower priority
// once the play_on is false, the function will clear the screen and return()

void pause(void);
// when the switch is push, we will use this function set taiko_on, 
// play_on to false so that the soound_and_ulcd fuction will 
// no longer play sound and show information 
// then, it will add the mode_selection function into the eventqueue

int which_modeORsong; // control by DNN
void mode_selection(void);
// there is a infinite loop detecting whether another swith is pressed.
// If the switch is pressed then if 
//	1. which_modeORsong shows that the mode is backward or forward, 
//	then we will just modify the length_array, tone_array,
//	number_of_note, and set play_on to true, and add the sound_and_ulcd
//	function to it's thread's eventqueue, then return()
//	
//	2. which_modeORsong shows that the mode is song_selection,
//	we will just add the song_selection into the eventqueue then return()
//
//	3. which_modeORsong shows that the mode is taiko,
//	we will just do the same thing as back_ward, forward, but this time we 
//	will set the taiko_on to true and ..... then return()
//	
//	4. which_modeORsong shows that the mode is load, then we will send a signal to python
//	to get the new songs. Everytime we call python, it will just send new three songs
//	and the three songs will replace the old three songs, so there is no need to indicate 
//	which songs we want! After loading the new songs, we will just go back to the infinite loop.
// Everytime excute the loop, we will refresh the ulcd.

void song_selection(void);
// there is a infinite loop detecting another switch is pressed too,
// if the switch is pressed, then we will see that what which_modeORsong
// is to modify tone_array, length_array, number_of_note and set play_on
// and add the sound_and_ulcd into its thread, then return()

void set_which_modeORsong;
// a funciton running in another thread and will use DNN 
