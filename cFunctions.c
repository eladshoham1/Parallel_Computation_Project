#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h> 

#include "constants.h"
#include "cFunctions.h"
#include "cudaFunctions.h"

void createScoreType(MPI_Datatype* dataType)
{
   MPI_Type_contiguous(SCORES_SIZE, MPI_INT, dataType);
   MPI_Type_commit(dataType);
}

char** readData(int* weights, char** firstSeq, int* numOfSequences)
{
   char **sequences;
   char buffer[BUFFER_SIZE], firstSeqBuffer[MAX_SIZE_FIRST_SEQ], seq[MAX_SIZE_SEQ];
   int i;

   // Read weights
   fgets(buffer, BUFFER_SIZE, stdin);
   sscanf(buffer, "%d %d %d %d", &weights[0], &weights[1], &weights[2], &weights[3]);

   // Read first sequence
   fgets(firstSeqBuffer, MAX_SIZE_FIRST_SEQ, stdin);
   firstSeqBuffer[strcspn(firstSeqBuffer, "\n")] = NULL_TERMINATED_STRING;
   *firstSeq = strdup(firstSeqBuffer);
   if (*firstSeq == NULL)
      return NULL;
   sequenceToUppercase(*firstSeq);

   // Read number of sequences
   fgets(buffer, BUFFER_SIZE, stdin);
   sscanf(buffer, "%d", numOfSequences);
   sequences = (char**)malloc(*numOfSequences * sizeof(char*));
   if (sequences == NULL)
      return NULL;

   // Read all sequences
   for (i = 0; i < *numOfSequences; i++)
   {
      fgets(seq, MAX_SIZE_SEQ, stdin);
      seq[strcspn(seq, "\n")] = NULL_TERMINATED_STRING;
      sequences[i] = strdup(seq);
      if (sequences[i] == NULL)
         return NULL;
      sequenceToUppercase(sequences[i]);
   }

   return sequences;
}

void sequenceToUppercase(char* sequence)
{
   while (*sequence != NULL_TERMINATED_STRING)
   {
      *sequence = toupper(*sequence);
      *sequence++;
   }
}

int isLettersInGroup(const char** group, char letter1, char letter2, int size)
{
   int i;

   for (i = 0; i < size; i++)
   {
      if (strchr(group[i], letter1) != NULL && strchr(group[i], letter2) != NULL)
         return 1;
   }

   return 0;
}

void fillSymbolsWeights(int* symbolsWeights, int* weights)
{
   const char *firstGroup[FIRST_GROUP_SIZE] = { "NDEQ", "NEQK", "STA", "MILV", "QHRK", "NHQK", "FYW", "HY", "MILF" };
	const char *secondGroup[SECOND_GROUP_SIZE] = { "SAG", "ATV", "CSA", "SGND", "STPA", "STNK", "NEQHRK", "NDEQHK", "SNDEQK", "HFY", "FVLIM" };
   int i, j;
   char letter1, letter2;

#pragma omp parallel for private(letter1) private(letter2) private(j)
   for (i = 0; i < ENGLISH_LETTERS; i++)
   {
      letter1 = i + LETTER_A;

      for (j = 0; j < ENGLISH_LETTERS; j++)
      {
         letter2 = j + LETTER_A;
         if (letter1 == letter2)
            symbolsWeights[i * ENGLISH_LETTERS + j] = weights[0];
         else if (isLettersInGroup(firstGroup, letter1, letter2, FIRST_GROUP_SIZE))
            symbolsWeights[i * ENGLISH_LETTERS + j] = -weights[1];
         else if (isLettersInGroup(secondGroup, letter1, letter2, SECOND_GROUP_SIZE))
            symbolsWeights[i * ENGLISH_LETTERS + j] = -weights[2];
         else
            symbolsWeights[i * ENGLISH_LETTERS + j] = -weights[3];
      }
   }
}

int alignmentScore(int* symbolsWeights, const char* firstSeq, const char* secondSeq, int secondSeqSize, int offset, int mutant)
{
   int i, row, col, afterMutant = 0, score = 0;

   for (i = 0; i <= secondSeqSize; i++)
   {
      if (i != mutant)
      {
         row = firstSeq[i + offset] - LETTER_A;
         col = secondSeq[i - afterMutant] - LETTER_A;
         score += symbolsWeights[row * ENGLISH_LETTERS + col];
      }
      else
         afterMutant = 1; // Reduce 1 from the index after the mutant
   }

   return score;
}

void maxAlignmentScore(int* symbolsWeights, const char* firstSeq, const char* secondSeq, Score* maxScore)
{
   int offset, mutant, maxOffset, score, firstSeqSize = strlen(firstSeq), secondSeqSize = strlen(secondSeq);

   maxOffset = firstSeqSize - secondSeqSize;
   score = alignmentScore(symbolsWeights, firstSeq, secondSeq, secondSeqSize, maxOffset, secondSeqSize); // Calculate the score for the last option
   *maxScore = { score, maxOffset, secondSeqSize };
   for (offset = 0; offset < maxOffset; offset++)
   {
      for (mutant = 1; mutant <= secondSeqSize; mutant++)
      {
         score = alignmentScore(symbolsWeights, firstSeq, secondSeq, secondSeqSize, offset, mutant);
         if (score > maxScore->score)
            *maxScore = { score, offset, mutant };
      }
   }
}

Score findMaxScore(Score* allScores, int numAllScores)
{
   Score maxScore;
   int i;

   maxScore = allScores[0];
   for (i = 1; i < numAllScores; i++)
   {
      if (allScores[i].score > maxScore.score)
         maxScore = allScores[i];
   }

   return maxScore;
}

int alignmentsScores(int* symbolsWeights, const char* firstSeq, char** sequences, int numOfSequences, Score* maxScores)
{
   Score** scoresPerSequence;
   int *numAllScores, i, firstThreadSequences, isExitSuccess = EXIT_SUCCESS;

#pragma omp parallel private(i)
   {
      int tid, numOfThreads, sequencesPerThread;
      tid = omp_get_thread_num();
      numOfThreads = omp_get_num_threads();
      sequencesPerThread = numOfSequences / numOfThreads;
      firstThreadSequences = sequencesPerThread + numOfSequences % numOfThreads;
      
      if (tid == ROOT) // One thread calls CUDA
      {
         scoresPerSequence = (Score**)doMalloc(firstThreadSequences * sizeof(Score*));
         numAllScores = (int*)doMalloc(firstThreadSequences * sizeof(int));
         isExitSuccess = allAlignmentsScores(symbolsWeights, firstSeq, sequences, scoresPerSequence, numAllScores, firstThreadSequences);
      }
      else
      {
         for (i = firstThreadSequences + tid - 1; i < numOfSequences; i += numOfThreads - 1)
            maxAlignmentScore(symbolsWeights, firstSeq, sequences[i], &maxScores[i]);
      }
   }

   if (isExitSuccess != EXIT_SUCCESS)
   {
      freeAllScores(scoresPerSequence, numAllScores, firstThreadSequences);
      return EXIT_FAILURE;
   }

// Using OpenMP to finds the max score for each sequence that calculated in CUDA 
#pragma omp parallel for private(i)
   for (i = 0; i < firstThreadSequences; i++)
      maxScores[i] = findMaxScore(scoresPerSequence[i], numAllScores[i]);

   freeAllScores(scoresPerSequence, numAllScores, firstThreadSequences);
   return EXIT_SUCCESS;
}

void printMaxOffsetMutant(Score* maxScores, int numOfSequences)
{
   int i;

   for (i = 0; i < numOfSequences; i++)
      printf("n = %d\tk = %d\n", maxScores[i].offset, maxScores[i].mutant);
}

void* doMalloc(unsigned int nbytes) 
{
   void *p = malloc(nbytes);

   if (p == NULL)
   { 
      fprintf(stderr, "malloc failed\n"); 
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
   }

   return p;
}

void freeAllScores(Score** scoresPerSequence, int* numAllScores, int numOfSequences)
{
   int i;

   for (i = 0; i < numOfSequences; i++)
      free(scoresPerSequence[i]);
   free(scoresPerSequence);
   free(numAllScores);
}

void freeAllSequences(char* firstSeq, char** sequences, Score* maxScores, int numOfSequences)
{
   int i;

   free(firstSeq);
   for (i = 0; i < numOfSequences; i++)
      free(sequences[i]);
   free(sequences);
   free(maxScores);
}