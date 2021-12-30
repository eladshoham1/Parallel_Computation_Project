#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "constants.h"
#include "cFunctions.h"

int main(int argc, char *argv[])
{
   MPI_Datatype scoreType;
   Score *maxScores;
   char *firstSeq, **sequences, packBuffer[PACK_BUFFER_SIZE];
   int symbolsWeights[ENGLISH_LETTERS * ENGLISH_LETTERS], weights[NUM_OF_WEIGHTS];
   int numOfProcs, rank, numOfSequences, workerNumOfSequences, workerId, firstSeqSize, remainder, position = 0, tempPosition, offset, seqSize, i;
   double parallelTime, sequentialTime;

   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &numOfProcs);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   createScoreType(&scoreType);

   if (rank == ROOT)
   {
      sequences = readData(weights, &firstSeq, &numOfSequences);
      if (!sequences)
         MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);

      // Calculate all the data we will need to send to the other processes.
      fillSymbolsWeights(symbolsWeights, weights);
      workerNumOfSequences = numOfSequences / numOfProcs;
      remainder = numOfSequences % numOfProcs;
      firstSeqSize = strlen(firstSeq) + 1; // Plus 1 for the epsilon at the end of the sequence.

      // Using MPI_Pack to pack the first sequence, it size, the array of symbols weights and the number of sequences for each one of the processes.
      MPI_Pack(&firstSeqSize, 1 ,MPI_INT, packBuffer, PACK_BUFFER_SIZE, &position, MPI_COMM_WORLD);
      MPI_Pack(firstSeq, firstSeqSize ,MPI_CHAR, packBuffer, PACK_BUFFER_SIZE, &position, MPI_COMM_WORLD);
      MPI_Pack(symbolsWeights, ENGLISH_LETTERS * ENGLISH_LETTERS, MPI_INT, packBuffer, PACK_BUFFER_SIZE, &position, MPI_COMM_WORLD);
      MPI_Pack(&workerNumOfSequences, 1, MPI_INT, packBuffer, PACK_BUFFER_SIZE, &position, MPI_COMM_WORLD);

      tempPosition = position; // Temporary variable to maintain the position in pack.
      for (workerId = 1; workerId < numOfProcs; workerId++)
      {
         offset = workerNumOfSequences * workerId + remainder; // Calculate the offset from which to start the sequences for this process.

         for (i = 0; i < workerNumOfSequences; i++)
         {
            seqSize = strlen(sequences[offset + i]) + 1; // Plus 1 for the epsilon at the end of the sequence.
            MPI_Pack(&seqSize, 1, MPI_INT, packBuffer, PACK_BUFFER_SIZE, &position, MPI_COMM_WORLD);
            MPI_Pack(sequences[offset + i], seqSize, MPI_CHAR, packBuffer, PACK_BUFFER_SIZE, &position, MPI_COMM_WORLD);
         }

         MPI_Send(packBuffer, position, MPI_PACKED, workerId ,0 ,MPI_COMM_WORLD);
         position = tempPosition; // Initialize the position in the pack for insert the next sequences for the next process.
      }

      parallelTime = MPI_Wtime(); // Start time for the parallel code
      maxScores = (Score*)doMalloc(numOfSequences * sizeof(Score));
      if (alignmentsScores(symbolsWeights, firstSeq, sequences, workerNumOfSequences + remainder, maxScores) != EXIT_SUCCESS)
         MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);

      for (workerId = 1; workerId < numOfProcs; workerId++)
         MPI_Recv(maxScores + workerNumOfSequences * workerId + remainder, numOfSequences, scoreType, workerId, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      parallelTime = MPI_Wtime() - parallelTime;
      printMaxOffsetMutant(maxScores, numOfSequences);
      printf("Parallel run time: %lf\n", parallelTime);

      sequentialTime = MPI_Wtime(); // Start time for the sequential code
      for (i = 0; i < numOfSequences; i++)
         maxAlignmentScore(symbolsWeights, firstSeq, sequences[i], &maxScores[i]);
      sequentialTime = MPI_Wtime() - sequentialTime;
      printMaxOffsetMutant(maxScores, numOfSequences);
      printf("Sequential run time: %lf\n", sequentialTime);

      if (sequentialTime > parallelTime)
         printf("The parallel is faster than the sequential in %lf seconds\n", sequentialTime - parallelTime);
      else
         printf("The sequential is faster than the parallel in %lf seconds\n", parallelTime - sequentialTime);
   }
   else
   {
      // Using MPI_Unpack to unpack the first sequence, it size, the array of symbols weights and the number of sequences that arrived from the ROOT process.
      MPI_Recv(packBuffer, PACK_BUFFER_SIZE, MPI_PACKED, ROOT, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Unpack(packBuffer, PACK_BUFFER_SIZE, &position, &firstSeqSize, 1, MPI_INT, MPI_COMM_WORLD);
      firstSeq = (char*)doMalloc(firstSeqSize * sizeof(char));
      MPI_Unpack(packBuffer, PACK_BUFFER_SIZE, &position, firstSeq, firstSeqSize, MPI_CHAR, MPI_COMM_WORLD); 
      MPI_Unpack(packBuffer, PACK_BUFFER_SIZE, &position, symbolsWeights, ENGLISH_LETTERS * ENGLISH_LETTERS, MPI_INT, MPI_COMM_WORLD);
      MPI_Unpack(packBuffer, PACK_BUFFER_SIZE, &position, &numOfSequences, 1, MPI_INT, MPI_COMM_WORLD);
      sequences = (char**)doMalloc(numOfSequences * sizeof(char*));
      
      for (i = 0; i < numOfSequences; i++)
      {
         MPI_Unpack(packBuffer, PACK_BUFFER_SIZE, &position, &seqSize, 1, MPI_INT, MPI_COMM_WORLD);
         sequences[i] = (char*)doMalloc(seqSize * sizeof(char));
         MPI_Unpack(packBuffer, PACK_BUFFER_SIZE, &position, sequences[i], seqSize, MPI_CHAR, MPI_COMM_WORLD);
      }

      maxScores = (Score*)doMalloc(numOfSequences * sizeof(Score));
      if (alignmentsScores(symbolsWeights, firstSeq, sequences, numOfSequences, maxScores) != EXIT_SUCCESS)
         MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);

      MPI_Send(maxScores, numOfSequences, scoreType, ROOT ,0 ,MPI_COMM_WORLD);
   }

   freeAllSequences(firstSeq, sequences, maxScores, numOfSequences);
   MPI_Finalize();
   return EXIT_SUCCESS;
}