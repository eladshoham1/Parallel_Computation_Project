#ifndef __CONSTANTS_H__
#define __CONSTANTS_H__

enum size { ROOT = 0, SCORES_SIZE = 3, NUM_OF_WEIGHTS = 4, FIRST_GROUP_SIZE = 9, SECOND_GROUP_SIZE = 11, ENGLISH_LETTERS = 26, 
    NUM_OF_THREADS = 32, BUFFER_SIZE = 200, MAX_SIZE_SEQ = 2001, MAX_SIZE_FIRST_SEQ = 3001, PACK_BUFFER_SIZE = 1 << 20 };
    
enum symbols { NULL_TERMINATED_STRING = '\0', LETTER_A = 'A', LETTER_Z = 'Z' };

typedef struct
{
    int score;
    int offset;
    int mutant;
} Score;

#endif