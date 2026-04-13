#include<cuda.h>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include<iostream>
using namespace std;

__global__ void matMul(float *A_d, float *B_d, float *C_d, int N, int M, int K)
{
  int row= blockIdx.y*blockDim.y+ threadIdx.y;
  int col= blockIdx.x*blockDim.x+ threadIdx.x;

  if((row<M)&&(col<N))
  {
    float sum= 0.0; 
    for(int k=0; k<K; ++k)
    {
      sum += A_d[row*K+k]*B_d[k*N+col]; 
    }
  C_d[row*N + col]= sum; 
  }
}

int main()
{
  //details about the matrix 
  int N= 512, M= 512, K= 512;
  size_t sizeA= M*K*sizeof(float);
  size_t sizeB= N*K*sizeof(float);
  size_t sizeC= M*N*sizeof(float);

  //allocate host memory
  float *A_h= (float*)malloc(sizeA);
  float *B_h= (float*)malloc(sizeB);
  float *C_h= (float*)malloc(sizeC);

  //allocate device memory
  float *A_d, *B_d, *C_d;
  cudaMalloc(&A_d, sizeA);
  cudaMalloc(&B_d, sizeB);
  cudaMalloc(&C_d, sizeC);

  //initiate A_h and B_h
  for(int z=0;z<M*K;++z)
  {
    A_h[z]=z;
  }
  for(int x=0;x<N*K;++x)
  {
    B_h[x]=x;
  }
  
  //copy from host to device
  cudaMemcpy(A_d,A_h,sizeA,cudaMemcpyHostToDevice);
  cudaMemcpy(B_d,B_h,sizeB,cudaMemcpyHostToDevice);

  //invoke kernels
  dim3 threadsperBlock(16,16);
  dim3 blocksperGrid(((N+15)/16),((M+15)/16));
  
  matMul<<<blocksperGrid,threadsperBlock>>>(A_d,B_d,C_d,N,M,K);

  //copy back to host
  cudaMemcpy(C_h,C_d,sizeC, cudaMemcpyDeviceToHost);

  for(int m=0;m<15;m++)
  {
    cout<<C_h[m]<<endl;
  }

  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);
  free(A_h);
  free(B_h);
  free(C_h);
  
  return 0;
}
