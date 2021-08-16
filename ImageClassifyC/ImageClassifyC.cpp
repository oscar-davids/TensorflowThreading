// ImageClassifyC.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
//#include <stdio.h>
//#include <stdlib.h>
//#include <string.h>


#if 0
#ifdef __cplusplus
extern "C"
{
#endif
#endif

#include "opencv2/core/core_c.h"
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/core/version.hpp"

#include <tensorflow/c/c_api.h>
#include <mutex>
//using namespace cv;
#if 0
#ifdef __cplusplus
}
#endif
#endif

#define IMG_W 224
#define IMG_H 224
#define IMG_C 3
#define IMG_L IMG_W*IMG_H*IMG_C

std::mutex mtxinput;
std::mutex mtxout;

//[<tf.Tensor 'input_1:0' shape = (? , 224, 224, 3) dtype = float32>]
//[<tf.Tensor 'reshape_3/Reshape:0' shape = (? , 2) dtype = float32>]

//global variable
#define DLL_EXPORT	extern "C" __declspec(dllexport)

//export function
DLL_EXPORT int initClaasifyEngine(char* modelpath);
DLL_EXPORT void exitClaasifyEngine();
DLL_EXPORT int prediction(char* path, float* pval); //return id 


//test
char testlist[10][256] = { "../Bin/ball01 (1).jpg","../Bin/ball01 (2).jpg","../Bin/ball01 (3).jpg","../Bin/ball01 (4).jpg","../Bin/ball01 (5).jpg", 
							"../Bin/bad01 (1).jpg","../Bin/bad01 (2).jpg","../Bin/bad01 (3).jpg","../Bin/bad01 (4).jpg","../Bin/bad01 (5).jpg", };

//function(internal)
TF_Buffer* read_file(const char* file);
void free_buffer(void* data, size_t length) { free(data); }
void deallocator(void* ptr, size_t len, void* arg) { free((void*)ptr); }


int readimage(char* path, float* buffer)
{
	if (strlen(path) == 0) return -1;

	IplImage* srcimg = cvLoadImage(path, CV_LOAD_IMAGE_COLOR);

	if(srcimg == 0) return -1;
	
	IplImage *resizedimg = cvCreateImage(cvSize(IMG_W, IMG_H), srcimg->depth, srcimg->nChannels);

	cvResize(srcimg, resizedimg);
	cvCvtColor(resizedimg, resizedimg, CV_BGR2RGB);

	unsigned char *data = (unsigned char*)resizedimg->imageData;
	for (int k = 0; k < IMG_L; k++)
	{
		buffer[k] = 1.0f / 255.0f * (*data);
		data++;
	}

	cvReleaseImage(&srcimg);
	cvReleaseImage(&resizedimg);

	return 0;
}

TF_Buffer* g_defgraph = 0;
TF_Graph* g_graph = 0;
TF_Status* g_status = 0;
TF_Session* g_sess = 0;

float* raw_input_data = 0;
TF_Operation* input_op = 0;
int64_t* raw_input_dims = 0;
TF_Tensor* input_tensor = 0;
TF_Output* run_inputs = 0;
TF_Tensor** run_inputs_tensors = 0;

float* raw_output_data = 0;
TF_Operation* output_op = 0;
int64_t* raw_output_dims = 0;
TF_Tensor* output_tensor = 0;
TF_Output* run_outputs = 0;
TF_Tensor** run_output_tensors = 0;

#ifndef _WINDLL
int main(int argc, char const* argv[]) 
{

	char modelpath[256] = { "../Bin/my_model.pb" };
	initClaasifyEngine(modelpath);
	//copy data 
	float proval[2] = { 0.0f,0.0f };
	for (int i = 0; i < 1000; i++)
	{
		int k = i % 10;
		prediction(testlist[k], proval);
		
	}
	
	exitClaasifyEngine();

	return 0;
}
#endif

int initClaasifyEngine(char* modelpath)
{
	// load graph
	// ================================================================================	
	g_defgraph = read_file(modelpath);
	
	g_graph = TF_NewGraph();
	g_status = TF_NewStatus();
	TF_ImportGraphDefOptions* opts = TF_NewImportGraphDefOptions();
	TF_GraphImportGraphDef(g_graph, g_defgraph, opts, g_status);
	TF_DeleteImportGraphDefOptions(opts);
	if (TF_GetCode(g_status) != TF_OK) {
		fprintf(stderr, "ERROR: Unable to import graph %s\n", TF_Message(g_status));
		return 1;
	}

	fprintf(stdout, "Successfully imported graph\n");

	// create session
	// ================================================================================
	TF_SessionOptions* opt = TF_NewSessionOptions();

	uint8_t config[7] = { 0x32, 0x5, 0x20, 0x1, 0x2a, 0x01, 0x30 }; // protobuf data for auto memory gpu_options.allow_growth=True and gpu_options.visible_device_list="0" 
	TF_SetConfig(opt, (void*)config, 7, g_status);

	g_sess = TF_NewSession(g_graph, opt, g_status);
	TF_DeleteSessionOptions(opt);
	if (TF_GetCode(g_status) != TF_OK) {
		fprintf(stderr, "ERROR: Unable to create session %s\n", TF_Message(g_status));
		return -1;
	}
	fprintf(stdout, "Successfully created session\n");

	//convert image data
	raw_input_data = (float*)malloc(IMG_L * sizeof(float));

	// gerenate input
	// ================================================================================
	input_op = TF_GraphOperationByName(g_graph, "input_1");
	//printf("input_op has %i inputs\n", TF_OperationNumOutputs(input_op));

	// prepare inputs		
	raw_input_dims = (int64_t*)malloc(4 * sizeof(int64_t));
	raw_input_dims[0] = 1;		//count
	raw_input_dims[1] = IMG_H;	//height
	raw_input_dims[2] = IMG_W;	//width
	raw_input_dims[3] = IMG_C;	//channel	

	input_tensor = TF_AllocateTensor(TF_FLOAT, raw_input_dims, 4, IMG_L * sizeof(float));
	//void* tensor_data = TF_TensorData(input_tensor);
	//memcpy(TF_TensorData(input_tensor), raw_input_data, min(len, TF_TensorByteSize(tensor)));
	//memcpy(TF_TensorData(input_tensor), raw_input_data, IMG_L * sizeof(float));

	run_inputs = (TF_Output*)malloc(1 * sizeof(TF_Output));
	run_inputs[0].oper = input_op;
	run_inputs[0].index = 0;

	run_inputs_tensors = (TF_Tensor**)malloc(1 * sizeof(TF_Tensor*));
	run_inputs_tensors[0] = input_tensor;

	// prepare outputs
	// ================================================================================

	raw_output_data = (float*)malloc(2 * sizeof(float));
	raw_output_data[0] = 0.f;
	raw_output_data[1] = 0.f;

	output_op = TF_GraphOperationByName(g_graph, "reshape_3/Reshape");
	// printf("output_op has %i outputs\n", TF_OperationNumOutputs(output_op));

	raw_output_dims = (int64_t*)malloc(2 * sizeof(int64_t));
	raw_output_dims[0] = 1;
	raw_output_dims[1] = 2;

	//TF_Tensor* output_tensor = TF_NewTensor(TF_FLOAT, raw_output_dims, 2, raw_output_data,
	//										2 * sizeof(float), &deallocator, NULL);
	output_tensor = TF_AllocateTensor(TF_FLOAT, raw_output_dims, 2, 2 * sizeof(float));

	run_outputs = (TF_Output*)malloc(1 * sizeof(TF_Output));
	run_outputs[0].oper = output_op;
	run_outputs[0].index = 0;

	run_output_tensors = (TF_Tensor**)malloc(1 * sizeof(TF_Tensor*));
	run_output_tensors[0] = output_tensor;

	//load cv dll
	IplImage *testimg = cvCreateImage(cvSize(IMG_W, IMG_H), 8, 3);
	unsigned char *data = (unsigned char*)testimg->imageData;
	for (int k = 0; k < IMG_L; k++)
	{
		raw_input_data[k] = 1.0f / 225.0f * (*data);
		data++;
	}
	cvReleaseImage(&testimg);
	memcpy(TF_TensorData(input_tensor), raw_input_data, IMG_L * sizeof(float));
	//

	//
	// run network
	// ================================================================================
	TF_SessionRun(g_sess,
		/* RunOptions */ NULL,
		/* Input tensors */ run_inputs, run_inputs_tensors, 1,
		/* Output tensors */ run_outputs, run_output_tensors, 1,
		/* Target operations */ NULL, 0,
		/* RunMetadata */ NULL,
		/* Output status */ g_status);
	if (TF_GetCode(g_status) != TF_OK) {
		fprintf(stderr, "ERROR: Unable to run output_op: %s\n", TF_Message(g_status));
		return -1;
	}

	return 0;
}
void exitClaasifyEngine()
{
	// free up stuff
	// ================================================================================	
	TF_CloseSession(g_sess, g_status);
	TF_DeleteSession(g_sess, g_status);

	TF_DeleteStatus(g_status);
	TF_DeleteBuffer(g_defgraph);

	TF_DeleteGraph(g_graph);

	TF_DeleteTensor(input_tensor);
	TF_DeleteTensor(output_tensor);


	free((void*)raw_input_data);
	free((void*)raw_input_dims);
	free((void*)run_inputs);
	free((void*)run_inputs_tensors);

	free((void*)raw_output_data);
	free((void*)raw_output_dims);
	free((void*)run_outputs);
	free((void*)run_output_tensors);
}
int prediction(char* path, float* pval)
{	
	//fprintf(stdout, "%s\n", path);	

	float* local_input_data = (float*)malloc(IMG_L * sizeof(float));

	if (readimage(path, local_input_data) != 0)
	{
		fprintf(stdout, "error prediction\n");
		return -1;
	}
	mtxinput.lock();
	memcpy(TF_TensorData(input_tensor), local_input_data, IMG_L * sizeof(float));
	mtxinput.unlock();
	// run network
	// ================================================================================
	TF_SessionRun(g_sess,
		/* RunOptions */ NULL,
		/* Input tensors */ run_inputs, run_inputs_tensors, 1,
		/* Output tensors */ run_outputs, run_output_tensors, 1,
		/* Target operations */ NULL, 0,
		/* RunMetadata */ NULL,
		/* Output status */ g_status);
	if (TF_GetCode(g_status) != TF_OK) {
		fprintf(stderr, "ERROR: Unable to run output_op: %s\n", TF_Message(g_status));
		return 1;
	}

	// printf("output-tensor has %i dims\n", TF_NumDims(run_output_tensors[0]));
	mtxout.lock();
	float* output_data = (float*)TF_TensorData(run_output_tensors[0]);
	
	memcpy(pval, output_data, 2 * sizeof(float));
	mtxout.unlock();
	if (local_input_data)
		free(local_input_data);

	return 0;
}

TF_Buffer* read_file(const char* file) 
{	
	fprintf(stdout, "%s\n", file);
	FILE* f = fopen(file, "rb");

	if (f == 0) return 0;

	fseek(f, 0, SEEK_END);
	long fsize = ftell(f);
	fseek(f, 0, SEEK_SET);  // same as rewind(f);

	void* data = malloc(fsize);
	fread(data, fsize, 1, f);
	fclose(f);

	TF_Buffer* buf = TF_NewBuffer();
	buf->data = data;
	buf->length = fsize;
	buf->data_deallocator = free_buffer;
	return buf;
}

#if 0
int back(int argc, char const* argv[])
{
	// load graph
	// ================================================================================
	TF_Buffer* graph_def = read_file("../Bin/my_model.pb");
	TF_Graph* graph = TF_NewGraph();
	TF_Status* status = TF_NewStatus();
	TF_ImportGraphDefOptions* opts = TF_NewImportGraphDefOptions();
	TF_GraphImportGraphDef(graph, graph_def, opts, status);
	TF_DeleteImportGraphDefOptions(opts);
	if (TF_GetCode(status) != TF_OK) {
		fprintf(stderr, "ERROR: Unable to import graph %s\n", TF_Message(status));
		return 1;
	}

	fprintf(stdout, "Successfully imported graph\n");

	// create session
	// ================================================================================
	TF_SessionOptions* opt = TF_NewSessionOptions();
	TF_Session* sess = TF_NewSession(graph, opt, status);
	TF_DeleteSessionOptions(opt);
	if (TF_GetCode(status) != TF_OK) {
		fprintf(stderr, "ERROR: Unable to create session %s\n", TF_Message(status));
		return 1;
	}
	fprintf(stdout, "Successfully created session\n");


	//convert image data
	float* raw_input_data = (float*)malloc(IMG_L * sizeof(float));

	IplImage* srcimg = cvLoadImage("../Bin/ball01 (6).jpg", CV_LOAD_IMAGE_COLOR);
	if (srcimg == 0 || srcimg->nChannels != IMG_C)
		return -1;

	IplImage *resizedimg = cvCreateImage(cvSize(IMG_W, IMG_H), srcimg->depth, srcimg->nChannels);

	//cvShowImage("a-1", img);
	//cvWaitKey(0);

	cvResize(srcimg, resizedimg);
	cvCvtColor(resizedimg, resizedimg, CV_BGR2RGB);

	//cvShowImage("a-2", destination);
	//cvWaitKey(0);	

	// copying the data into the corresponding tensor
	unsigned char *data = (unsigned char*)resizedimg->imageData;
	for (int k = 0; k < IMG_L; k++)
	{
		raw_input_data[k] = 1.0f / 225.0f * (*data);
		data++;
	}


	// gerenate input
	// ================================================================================
	TF_Operation* input_op = TF_GraphOperationByName(graph, "input_1");
	printf("input_op has %i inputs\n", TF_OperationNumOutputs(input_op));

	// prepare inputs		
	int64_t* raw_input_dims = (int64_t*)malloc(4 * sizeof(int64_t));
	raw_input_dims[0] = 1;	 //count
	raw_input_dims[1] = 224; //height
	raw_input_dims[2] = 224; //width
	raw_input_dims[3] = 3;	 //channel	

	TF_Tensor* input_tensor = TF_AllocateTensor(TF_FLOAT, raw_input_dims, 4, IMG_L * sizeof(float));
	//void* tensor_data = TF_TensorData(input_tensor);
	//memcpy(TF_TensorData(input_tensor), raw_input_data, min(len, TF_TensorByteSize(tensor)));
	//memcpy(TF_TensorData(input_tensor), raw_input_data, IMG_L * sizeof(float));

	TF_Output* run_inputs = (TF_Output*)malloc(1 * sizeof(TF_Output));
	run_inputs[0].oper = input_op;
	run_inputs[0].index = 0;

	TF_Tensor** run_inputs_tensors = (TF_Tensor**)malloc(1 * sizeof(TF_Tensor*));
	run_inputs_tensors[0] = input_tensor;

	// prepare outputs
	// ================================================================================
	TF_Operation* output_op = TF_GraphOperationByName(graph, "reshape_3/Reshape");
	// printf("output_op has %i outputs\n", TF_OperationNumOutputs(output_op));


	float* raw_output_data = (float*)malloc(2 * sizeof(float));
	raw_output_data[0] = 0.f;
	raw_output_data[1] = 0.f;

	int64_t* raw_output_dims = (int64_t*)malloc(2 * sizeof(int64_t));
	raw_output_dims[0] = 1;
	raw_output_dims[1] = 2;

	//TF_Tensor* output_tensor = TF_NewTensor(TF_FLOAT, raw_output_dims, 2, raw_output_data,
	//										2 * sizeof(float), &deallocator, NULL);
	TF_Tensor* output_tensor = TF_AllocateTensor(TF_FLOAT, raw_output_dims, 2, 2 * sizeof(float));

	TF_Output* run_outputs = (TF_Output*)malloc(1 * sizeof(TF_Output));
	run_outputs[0].oper = output_op;
	run_outputs[0].index = 0;

	TF_Tensor** run_output_tensors = (TF_Tensor**)malloc(1 * sizeof(TF_Tensor*));
	run_output_tensors[0] = output_tensor;

	//copy data
	readimage(testlist[0], raw_input_data);
	memcpy(TF_TensorData(input_tensor), raw_input_data, IMG_L * sizeof(float));

	// run network
	// ================================================================================
	TF_SessionRun(sess,
		/* RunOptions */ NULL,
		/* Input tensors */ run_inputs, run_inputs_tensors, 1,
		/* Output tensors */ run_outputs, run_output_tensors, 1,
		/* Target operations */ NULL, 0,
		/* RunMetadata */ NULL,
		/* Output status */ status);
	if (TF_GetCode(status) != TF_OK) {
		fprintf(stderr, "ERROR: Unable to run output_op: %s\n", TF_Message(status));
		return 1;
	}

	// printf("output-tensor has %i dims\n", TF_NumDims(run_output_tensors[0]));

	void* output_data = TF_TensorData(run_output_tensors[0]);
	printf("output %f ,  %f\n", ((float*)output_data)[0], ((float*)output_data)[1]);
	printf("output %f ,  %f\n", raw_output_data[0], raw_output_data[1]);

	// you do not want see me creating all the other tensors; Enough lines for
	// this simple example!

	// free up stuff
	// ================================================================================	
	TF_CloseSession(sess, status);
	TF_DeleteSession(sess, status);

	TF_DeleteStatus(status);
	TF_DeleteBuffer(graph_def);

	TF_DeleteGraph(graph);

	TF_DeleteTensor(input_tensor);
	TF_DeleteTensor(output_tensor);


	free((void*)raw_input_data);
	free((void*)raw_input_dims);
	free((void*)run_inputs);
	free((void*)run_inputs_tensors);

	free((void*)raw_output_data);
	free((void*)raw_output_dims);
	free((void*)run_outputs);
	free((void*)run_output_tensors);

	cvReleaseImage(&srcimg);
	cvReleaseImage(&resizedimg);

	return 0;
}
#endif
/*
TF_CAPI_EXPORT extern void TF_SessionRun(
	TF_Session* session,
	// RunOptions
	const TF_Buffer* run_options,
	// Input tensors
	const TF_Output* inputs, TF_Tensor* const* input_values, int ninputs,
	// Output tensors
	const TF_Output* outputs, TF_Tensor** output_values, int noutputs,
	// Target operations
	const TF_Operation* const* target_opers, int ntargets,
	// RunMetadata
	TF_Buffer* run_metadata,
	// Output status
	TF_Status*);
*/