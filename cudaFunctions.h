#ifndef __CUDA_FUNCTIONS_H__
#define __CUDA_FUNCTIONS_H__

/**
 * Calculate all the scores for each one of the sequences.
 * @param int* pointer to array of symbols weights.
 * @param char* pointer to the first sequence.
 * @param char** pointer to array of sequences.
 * @param Score** pointer to array of scores.
 * @param int* pointer to array of scores sizes.
 * @param int number of sequences.
 * @return EXIT_SUCCESS if everything worked properly, EXIT_FAILURE else.
 */
int allAlignmentsScores(int* symbolsWeights, const char* firstSeq, char** sequences, Score** scoresPerSequence, int* numAllScores, int numOfSequences);

#endif