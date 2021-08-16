// Testclassify.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <thread>
#include <iostream>

#define USE_THREAD 1

#define DLL_EXPORT	extern "C" __declspec(dllimport)

//export function
DLL_EXPORT int initClaasifyEngine(char* modelpath);
DLL_EXPORT void exitClaasifyEngine();
DLL_EXPORT int prediction(char* path, float* pval); //return id 


char testlist[10][256] = { "../Bin/ball01 (1).png","../Bin/ball01 (2).png","../Bin/ball01 (3).png","../Bin/ball01 (4).png","../Bin/ball01 (5).png",
							"../Bin/ball01 (6).png","../Bin/ball01 (7).png","../Bin/ball01 (8).png","../Bin/ball01 (9).png","../Bin/ball01 (10).png", };

void threadproc(int k, int tid)
{
	
	char testlist[10][256] = { "../Bin/ball01 (1).png","../Bin/ball01 (2).png","../Bin/ball01 (3).png","../Bin/ball01 (4).png","../Bin/ball01 (5).png",
		"../Bin/ball01 (6).png","../Bin/ball01 (7).png","../Bin/ball01 (8).png","../Bin/ball01 (9).png","../Bin/ball01 (10).png", };	
	
	float pfval[2] = { 0.0f, 0.0f };
	prediction(testlist[k], pfval);
	printf("threadid %d %s bad: %.2f football: %.2f\n", tid, testlist[k], pfval[0], pfval[1]);
}
int main(int argc, char const* argv[])
{

	char modelpath[256] = { "../Bin/tasmodel.pb" };
	initClaasifyEngine(modelpath);
	float pfval[2] = { 0.0f, 0.0f };
#if USE_THREAD
	// thread
	for (int i = 0; i < 100000; i++)
	{
		int nthread = atoi(argv[1]);
		std::vector<std::thread> threads(nthread);

		for (int j = 0; j < nthread; j++) {
			int k = rand() % 10;
			threads[j] = std::thread(threadproc, k , j);
		}

		for (auto& th : threads) {
			th.join();
		}
	}
#else	
	for (int i = 0; i < 1000; i++)
	{
		int k = i % 10;

		prediction(testlist[k], pfval);
		printf("%s bad: %.2f football: %.2f\n", testlist[k], pfval[0], pfval[1]);
	}
#endif

	exitClaasifyEngine();

	return 0;
}
