#ifndef __C_FUNCTIONS_H__
#define __C_FUNCTIONS_H__

/**
 * Create new MPI data type for the struct Score.
 * @param MPI_Datatype* pointer to MPI_Datatype argument.
 */
void createScoreType(MPI_Datatype* dataType);

/**
 * Read the input data from standard input.
 * @param int* pointer to weights array.
 * @param char** pointer to first sequence.
 * @param int* pointer to number of sequences.
 * @return pointer to array of sequences.
 */
char** readData(int* weights, char** firstSeq, int* numOfSequences);

/**
 * Converts sequence to uppercase letters
 * @param char* pointer to sequence.
 */
void sequenceToUppercase(char* sequence);

/**
 * Check if 2 letters appear in one of the words in the group.
 * @param char** pointer to group of words.
 * @param char first letter.
 * @param char second letter.
 * @param int the size of the group.
 * @return 1 if both the letters appear in the same word, 0 else.
 */
int isLettersInGroup(const char** group, char letter1, char letter2, int size);

/**
 * Fill array of all the letters in English by weights array.
 * @param int* pointer array of all the letters in English.
 * @param int* pointer to weights array.
 */
void fillSymbolsWeights(int* symbolsWeights, int* weights);

/**
 * Calculate the alignment score of 2 sequences.
 * @param int* pointer to array of symbols weights.
 * @param char* pointer to first sequence.
 * @param char* pointer to second sequence.
 * @param int the size of the second sequence.
 * @param int the index of the offset.
 * @param int the index of the mutant.
 * @return alignment score.
 */
int alignmentScore(int* symbolsWeights, const char* firstSeq, const char* secondSeq, int secondSeqSize, int offset, int mutant);

/**
 * Find the max alignment score of 2 sequences.
 * @param int* pointer to array of symbols weights.
 * @param char* pointer to first sequence.
 * @param char* pointer to second sequence.
 * @param Score* pointer to array of max scores.
 */
void maxAlignmentScore(int* symbolsWeights, const char* firstSeq, const char* secondSeq, Score* maxScore);

/**
 * Scan array of scores to find the max score.
 * @param Score* pointer to array of scores.
 * @param int number of scores.
 * @return max score.
 */
Score findMaxScore(Score* allScores, int numAllScores);

/**
 * Using OpenMP and CUDA to calculate the max score for each sequences.
 * @param int* pointer to array of symbols weights.
 * @param char* pointer to first sequence.
 * @param char** pointer to array of sequences.
 * @param int number of sequences.
 * @param Score* pointer to array of max scores.
 * @return EXIT_SUCCESS if everything worked properly, EXIT_FAILURE else.
 */
int alignmentsScores(int* symbolsWeights, const char* firstSeq, char** sequences, int numOfSequences, Score* maxScores);

/**
 * Print offset and mutant that achieve max score for each sequence by format m = ..., k = ...
 * @param Score* pointer to array of max scores.
 * @param int number of sequences.
 */
void printMaxOffsetMutant(Score* maxScores, int numOfSequences);

/**
 * Call malloc and check if it was success.
 * @param int size in bytes.
 * @return pointer from malloc.
 */
void* doMalloc(unsigned int nbytes);

/**
 * Free the memory of all the data on the scores.
 * @param Score** pointer to matrix of scores.
 * @param int* pointer to array of scores sizes.
 * @param int number of sequences.
 * @return pointer to array of sequences.
 */
void freeAllScores(Score** scoresPerSequence, int* numAllScores, int numOfSequences);

/**
 * Free the memory of all the data on the sequences and their max scores.
 * @param char* pointer to first sequence.
 * @param char** pointer to array of sequences.
 * @param Score* pointer to array of scores.
 * @param int number of sequences.
 * @return pointer to array of sequences.
 */
void freeAllSequences(char* firstSeq, char** sequences, Score* maxScores, int numOfSequences);

#endif