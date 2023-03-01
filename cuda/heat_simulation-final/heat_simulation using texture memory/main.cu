
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
#define MAX_tmp 1.0f
#define MIN_tmp 0.0000001f

//texture memory 선언
texture<float, 2>  texConstSrc;
texture<float, 2>  texIn;
texture<float, 2>  texOut;

__global__ void copy_const_kernel(float *iptr) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int idx = x + y * blockDim.x * gridDim.x;

	float c = tex2D(texConstSrc, x, y);
	if (c != 0)
		iptr[idx] = c;
}

__global__ void blend_kernel(float *dst, bool dstOut) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int idx = x + y * blockDim.x * gridDim.x;

	//texture memory에 바인딩 하여 다음과 같이 사용 가능
	float   t, l, c, r, b;
	if (dstOut) {
		t = tex2D(texIn, x, y - 1);
		l = tex2D(texIn, x - 1, y);
		c = tex2D(texIn, x, y);
		r = tex2D(texIn, x + 1, y);
		b = tex2D(texIn, x, y + 1);
	}
	else {
		t = tex2D(texOut, x, y - 1);
		l = tex2D(texOut, x - 1, y);
		c = tex2D(texOut, x, y);
		r = tex2D(texOut, x + 1, y);
		b = tex2D(texOut, x, y + 1);
	}
	dst[idx] = c + SPEED * (t + b + r + l - 4 * c);
}

struct DataBlock {
	unsigned char   *output_bitmap;
	float           *dev_inSrc;
	float           *dev_outSrc;
	float           *dev_constSrc;
	CPUAnimBitmap  *bitmap;
	cudaEvent_t     start, stop;
	float           totalTime;
	float           frames;
};

void anim_gpu(DataBlock *d, int ticks) {
	cudaEventRecord(d->start, 0);
	dim3    GridDim(DIM / 16, DIM / 16);
	dim3    BlockDim(16, 16);
	CPUAnimBitmap  *bitmap = d->bitmap;

	volatile bool dstOut = true;
	for (int i = 0; i<90; i++) {
		float   *in, *out;
		if (dstOut) {
			in = d->dev_inSrc;
			out = d->dev_outSrc;
		}
		else {
			out = d->dev_inSrc;
			in = d->dev_outSrc;
		}
		copy_const_kernel << <GridDim, BlockDim >> >(in);
		blend_kernel << <GridDim, BlockDim >> >(out, dstOut);
		dstOut = !dstOut;
	}
	float_to_color << <GridDim, BlockDim >> >(d->output_bitmap, d->dev_inSrc);
	cudaMemcpy(bitmap->get_ptr(), d->output_bitmap, bitmap->image_size(), cudaMemcpyDeviceToHost);

	cudaEventRecord(d->stop, 0);
	cudaEventSynchronize(d->stop);

	float   elapsedTime;
	cudaEventElapsedTime(&elapsedTime, d->start, d->stop);
	d->totalTime += elapsedTime;
	++d->frames;
	printf("Using texture memory Average Time per frame:  %.2f ms\n", d->totalTime / d->frames);
}

void anim_exit(DataBlock *d) {
	cudaUnbindTexture(texIn);
	cudaUnbindTexture(texOut);
	cudaUnbindTexture(texConstSrc);
	cudaFree(d->dev_inSrc);
	cudaFree(d->dev_outSrc);
	cudaFree(d->dev_constSrc);

	cudaEventDestroy(d->start);
	cudaEventDestroy(d->stop);
}


int main(void) {
	int num;
	printf("input tmp heat num :");
	scanf_s("%d", &num);
	srand((unsigned)time(NULL));
	DataBlock   data;
	CPUAnimBitmap bitmap(DIM, DIM, &data);
	data.bitmap = &bitmap;
	data.totalTime = 0;
	data.frames = 0;
	cudaEventCreate(&data.start);
	cudaEventCreate(&data.stop);

	cudaMalloc((void**)&data.output_bitmap, bitmap.image_size());


	cudaMalloc((void**)&data.dev_inSrc, bitmap.image_size());
	cudaMalloc((void**)&data.dev_outSrc, bitmap.image_size());
	cudaMalloc((void**)&data.dev_constSrc, bitmap.image_size());

	//texture memory에 할당
	//parameter는   size_t* offset, const textureReference* texref, const void* devPtr, const cudaChannelFormatDesc* desc, size_t width, size_t height, size_t pitch
	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
	cudaBindTexture2D(NULL, texConstSrc, data.dev_constSrc, desc, DIM, DIM, sizeof(float) * DIM);
	cudaBindTexture2D(NULL, texIn, data.dev_inSrc, desc, DIM, DIM, sizeof(float) * DIM);
	cudaBindTexture2D(NULL, texOut, data.dev_outSrc, desc, DIM, DIM, sizeof(float) * DIM);

	float *tmp = (float*)malloc(bitmap.image_size());
	for (int i = 0; i<DIM*DIM; i++) {
		tmp[i] = 0;
		int x = i % DIM;
		int y = i / DIM;
		if ((x>450) && (x<600) && (y>410) && (y<551))
			tmp[i] = MAX_tmp;
	}

	cudaMemcpy(data.dev_constSrc, tmp, bitmap.image_size(), cudaMemcpyHostToDevice);

	for (int i = 0; i < num; i++) {
		int x_r = rand() % 800;
		int y_r = rand() % 800;

		for (int y = 0; y < DIM; y++) {
			for (int x = 0; x < DIM; x++) {
				if ((y > y_r) && (y < (y_r + 100)) && (x > x_r) && (x < x_r + 100)) {
					tmp[x + y*DIM] = MAX_tmp;
				}
			}
		}
	}
	cudaMemcpy(data.dev_inSrc, tmp, bitmap.image_size(), cudaMemcpyHostToDevice);
	free(tmp);

	bitmap.anim_and_exit((void(*)(void*, int))anim_gpu, (void(*)(void*))anim_exit);
}