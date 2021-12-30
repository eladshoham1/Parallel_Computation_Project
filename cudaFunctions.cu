#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <iostream>
#include <string.h>
using namespace std;

#include "constants.h"
#include "cudaFunctions.h"

/**
 * Calculate the alignment score of 2 sequences.
 * @param int* pointer to array of symbols weights.
 * @param char* pointer to the first sequence.
 * @param char* pointer to the scecond sequence.
 * @param int the size of the second sequence.
 * @param int offset index.
 * @param int mutant index.
 * @return alignment score.
 */
__device__ int alignmentScoreGPU(int* symbolsWeights, const char* firstSeq, const char* secondSeq, int firstSeqSize, int secondSeqSize, int offset, int mutant)
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

/**
 * Calculate all scores of 2 sequences.
 * @param int* pointer to array of symbols weights.
 * @param char* pointer to the first sequence.
 * @param char* pointer to the scecond sequence.
 * @param int the size of the first sequence.
 * @param int the size of the second sequence.
 * @param Score* pointer to array of scores.
 */
__global__ void calcAlignmentsScores(int* symbolsWeights, const char* firstSeq, const char* secondSeq, int firstSeqSize, int secondSeqSize, Score* allScores)
{
    int mutant = blockDim.x * blockIdx.x + threadIdx.x;
    int offset = blockDim.y * blockIdx.y + threadIdx.y;
    int score, diffOffset = firstSeqSize - secondSeqSize;

    if (mutant < secondSeqSize && offset <= diffOffset)
    {
        score = offset == firstSeqSize - secondSeqSize && mutant + 1 != secondSeqSize ? INT_MIN : 
                alignmentScoreGPU(symbolsWeights, firstSeq, secondSeq, firstSeqSize, secondSeqSize, offset, mutant + 1);
        allScores[offset * secondSeqSize + mutant] = { score, offset, mutant + 1 };
    }
}

/**
 * Check if cuda status success.
 * @param cudaError_t* pointer to array of sequences.
 * @param int* pointer to array of symbols weights.
 * @param char* pointer to the first sequence.
 * @param char* pointer to sequence.
 * @param Score* pointer to array of scores.
 * @param string error message.
 * @return EXIT_SUCCESS if everything worked properly, EXIT_FAILURE else.
 */
int checkStatus(cudaError_t cudaStatus, int* symbolsWeights, char* firstSeq, char* sequence, Score* allScores, string err)
{
    if (cudaStatus != cudaSuccess)
    {
        cout << err << endl;

        cudaFree(symbolsWeights);
        cudaFree(firstSeq);
        cudaFree(sequence);
        cudaFree(allScores);

        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

int allAlignmentsScores(int* symbolsWeights, const char* firstSeq, char** sequences, Score** scoresPerSequence, int* numAllScores, int numOfSequences)
{
    int *devSymbolsWeights = 0, i;
    char *devFirstSeq = 0, *devSequence = 0;
    Score *devAllScores = 0;
    size_t firstSeqSize = strlen(firstSeq), seqSize, diffOffset, numOfScores;
    cudaError_t cudaStatus;
    dim3 threadsPerBlock(NUM_OF_THREADS, NUM_OF_THREADS), blocksPerGrid;

    cudaStatus = cudaMalloc((void**)&devSymbolsWeights, ENGLISH_LETTERS * ENGLISH_LETTERS * sizeof(int));
    if (checkStatus(cudaStatus, devSymbolsWeights, devFirstSeq, devSequence, devAllScores, "Cuda malloc failed on devSymbolsWeights!") == EXIT_FAILURE)
        return EXIT_FAILURE;

    cudaStatus = cudaMemcpy(devSymbolsWeights, symbolsWeights, ENGLISH_LETTERS * ENGLISH_LETTERS * sizeof(int), cudaMemcpyHostToDevice);
    if (checkStatus(cudaStatus, devSymbolsWeights, devFirstSeq, devSequence, devAllScores, "Cuda memcpy failed on devSymbolsWeights!") == EXIT_FAILURE)
        return EXIT_FAILURE;

    cudaStatus = cudaMalloc((void**)&devFirstSeq, firstSeqSize * sizeof(char));
    if (checkStatus(cudaStatus, devSymbolsWeights, devFirstSeq, devSequence, devAllScores, "Cuda malloc failed on devFirstSeq!") == EXIT_FAILURE)
        return EXIT_FAILURE;

    cudaStatus = cudaMemcpy(devFirstSeq, firstSeq, firstSeqSize * sizeof(char), cudaMemcpyHostToDevice);
    if (checkStatus(cudaStatus, devSymbolsWeights, devFirstSeq, devSequence, devAllScores, "Cuda memcpy failed on devFirstSeq!") == EXIT_FAILURE)
        return EXIT_FAILURE;

    for (i = 0; i < numOfSequences; i++)
    {
        seqSize = strlen(sequences[i]);
        diffOffset = firstSeqSize - seqSize + 1;
        numOfScores = diffOffset * seqSize;
        blocksPerGrid.x = (numOfScores + threadsPerBlock.x - 1) / threadsPerBlock.x;
        blocksPerGrid.y = (numOfScores + threadsPerBlock.y - 1) / threadsPerBlock.y;

        cudaStatus = cudaMalloc((void**)&devSequence, seqSize * sizeof(char));
        if (checkStatus(cudaStatus, devSymbolsWeights, devFirstSeq, devSequence, devAllScores, "Cuda malloc failed on devSequence!") == EXIT_FAILURE)
            return EXIT_FAILURE;

        cudaStatus = cudaMemcpy(devSequence, sequences[i], seqSize * sizeof(char), cudaMemcpyHostToDevice);
        if (checkStatus(cudaStatus, devSymbolsWeights, devFirstSeq, devSequence, devAllScores, "Cuda memcpy failed on devSequence!") == EXIT_FAILURE)
            return EXIT_FAILURE;

        cudaStatus = cudaMalloc((void**)&devAllScores, numOfScores * sizeof(Score));
        if (checkStatus(cudaStatus, devSymbolsWeights, devFirstSeq, devSequence, devAllScores, "Cuda malloc failed on devAllScores!") == EXIT_FAILURE)
            return EXIT_FAILURE;

        calcAlignmentsScores<<<blocksPerGrid, threadsPerBlock>>>(devSymbolsWeights, devFirstSeq, devSequence, firstSeqSize, seqSize, devAllScores);
        cudaStatus = cudaDeviceSynchronize();
        if (checkStatus(cudaStatus, devSymbolsWeights, devFirstSeq, devSequence, devAllScores, "Cuda kernel failed on calcAlignmentsScores!") == EXIT_FAILURE)
            return EXIT_FAILURE;

        numAllScores[i] = numOfScores; // Write the number of scores for each sequence
        scoresPerSequence[i] = (Score*)malloc(numAllScores[i] * sizeof(Score));
        if (!scoresPerSequence[i])
            return EXIT_FAILURE;
        
        cudaStatus = cudaMemcpy(scoresPerSequence[i], devAllScores, numAllScores[i] * sizeof(Score), cudaMemcpyDeviceToHost);
        if (checkStatus(cudaStatus, devSymbolsWeights, devFirstSeq, devSequence, devAllScores, "Cuda memcpy failed on scoresPerSequence!") == EXIT_FAILURE)
            return EXIT_FAILURE;

        cudaStatus = cudaFree(devSequence);
        if (checkStatus(cudaStatus, devSymbolsWeights, devFirstSeq, devSequence, devAllScores, "Cuda free failed on devSequence!") == EXIT_FAILURE)
        return EXIT_FAILURE;

        cudaStatus = cudaFree(devAllScores);
        if (checkStatus(cudaStatus, devSymbolsWeights, devFirstSeq, devSequence, devAllScores, "Cuda free failed on devAllScores!") == EXIT_FAILURE)
        return EXIT_FAILURE;
    }

    cudaStatus = cudaFree(devSymbolsWeights);
    if (checkStatus(cudaStatus, devSymbolsWeights, devFirstSeq, devSequence, devAllScores, "Cuda free failed on devSymbolsWeights!") == EXIT_FAILURE)
        return EXIT_FAILURE;

    cudaStatus = cudaFree(devFirstSeq);
    if (checkStatus(cudaStatus, devSymbolsWeights, devFirstSeq, devSequence, devAllScores, "Cuda free failed on devFirstSeq!") == EXIT_FAILURE)
        return EXIT_FAILURE;

    return EXIT_SUCCESS;
}