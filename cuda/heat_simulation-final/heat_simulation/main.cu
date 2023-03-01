
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "book.h"
#include "cpu_bitmap.h"
#include "cpu_Anim.h"

#include <stdio.h>
#include<stdlib.h>
#include<time.h>

#define DIM 1024
#define SPEED 0.25f
#define PI 3.141592f
#define MAX_TEMP 1.0f
#define MIN_TEMP 0.0000001f

//�� ���� �����Ѵٸ� ����
__global__ void copy_const_kernel(float *iptr, const float *cptr) {
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int idx = x + y * blockDim.x * gridDim.x;

	if (cptr[idx] != 0) iptr[idx] = cptr[idx];
}

//������ ������ ���
__global__ void blend_kernel(float *outData, const float *inData) {
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int idx = x + y * blockDim.x * gridDim.x;

	int top = idx - DIM * sizeof(char);
	int bottom = idx + DIM * sizeof(char);
	int right = idx + 1;
	int left = idx - 1;

	//���鿡 ���� �� ���ܻ���
	if (y == 0) { top += DIM * sizeof(char); }
	if (y == DIM - 1) { bottom -= DIM * sizeof(char); }
	if (x == DIM - 1) { right--; }
	if (x == 0) { left++; }

	//���� thread�� ��ġ�� ���� ������ ������ ����
	outData[idx] = inData[idx] + SPEED * (inData[top] + inData[bottom] + inData[left] + inData[right] \
		- 4 * inData[idx]);
}

//openGL ������ ����
struct DataBlock {
	unsigned char* outData_bitmap;
	float* dev_inData;
	float* dev_outData;
	float* dev_ConstData;
	CPUAnimBitmap* bitmap;
	cudaEvent_t start, end;
	float totalTime;
	float frames;
};


void anim_gpu(DataBlock* d, int tic) {
	cudaEventRecord(d->start, 0);

	dim3 BlockDim(16, 16);
	dim3 GridDim(DIM / BlockDim.x, DIM / BlockDim.y);

	CPUAnimBitmap *bitmap = d->bitmap;

	//ó���� 90�� �� �� animation ���
	//�� ���� �� �� animation�� ������ �ʴ�
	for (int i = 0; i < 90; i++) {
		copy_const_kernel << <GridDim, BlockDim >> > (d->dev_inData, d->dev_ConstData);
		blend_kernel << <GridDim, BlockDim >> > (d->dev_outData, d->dev_inData);
		swap(d->dev_inData, d->dev_outData);
	}

	//�迭�� ���� ����� ��ȯ�ϱ� ���� kernel
	float_to_color << <GridDim, BlockDim >> > (d->outData_bitmap, d->dev_inData);

	cudaMemcpy(bitmap->get_ptr(), d->outData_bitmap, bitmap->image_size(), cudaMemcpyDeviceToHost);

	//�� frame ó�� �ҿ� �ð��� Ȯ���ϱ� ���� event
	cudaEventRecord(d->end, 0);
	cudaEventSynchronize(d->end);

	//�� ������ ó�� �ϴµ� �ҿ� �ð�
	float elapseTime;
	cudaEventElapsedTime(&elapseTime, d->start, d->end);
	d->totalTime += elapseTime;
	++d->frames;
	printf("Normal Average Time per frame : %.2f ms \n", d->totalTime / d->frames);
}

//animation ����
void anim_exit(DataBlock* d) {
	cudaFree(d->dev_ConstData);
	cudaFree(d->dev_inData);
	cudaFree(d->dev_outData);

	cudaEventDestroy(d->start);
	cudaEventDestroy(d->end);
}

int main(void) {
	int num;
	printf("input tmp heat num :");
	scanf_s("%d", &num);
	srand((unsigned)time(NULL));

	DataBlock data;
	CPUAnimBitmap bitmap(DIM, DIM, &data);

	data.bitmap = &bitmap;
	data.totalTime = 0;
	data.frames = 0;
	cudaEventCreate(&data.start);
	cudaEventCreate(&data.end);

	//device �Ҵ� �۾�
	cudaMalloc(&data.outData_bitmap, bitmap.image_size());
	cudaMalloc(&data.dev_inData, bitmap.image_size());
	cudaMalloc(&data.dev_outData, bitmap.image_size());
	cudaMalloc(&data.dev_ConstData, bitmap.image_size());

	float* tmp = (float*)malloc(bitmap.image_size());

	//�߾ӿ� main �߿���ġ�� �α�
	//�ð��� �ڳ��� �µ��� ��ȭ���� �ʴ´�.
	for (int i = 0; i < DIM*DIM; i++) {
		tmp[i] = 0;
		int x = i % DIM;
		int y = i / DIM;

		if ((x > 450) && (x < 600) && (y > 410) && (y < 551)) { tmp[i] = MAX_TEMP; }
	}

	cudaMemcpy(data.dev_ConstData, tmp, bitmap.image_size(), cudaMemcpyHostToDevice);

	//random �Լ��� ��� �迭�� ���� ������ random�ϰ� �Ͻ��� �߿���ġ�� ��ġ�ϵ��� �Ѵ�
	//�Ͻ��� �߿� ��ġ�̹Ƿ� �ð��� ������ ���� �µ��� �����Ѵ�.
	for (int i = 0; i < num; i++) {
		int x_r = rand() % 800;
		int y_r = rand() % 800;

		for (int y = 0; y < DIM; y++) {
			for (int x = 0; x < DIM; x++) {
				if ((y > y_r) && (y < (y_r + 100)) && (x > x_r) && (x < x_r + 100)) {
					tmp[x + y*DIM] = MAX_TEMP;
				}
			}
		}
	}
	cudaMemcpy(data.dev_inData, tmp, bitmap.image_size(), cudaMemcpyHostToDevice);

	free(tmp);
	//��Ʈ������ �ִϸ��̼� ���
	bitmap.anim_and_exit((void(*)(void*, int))anim_gpu, (void(*)(void*))anim_exit);
}

