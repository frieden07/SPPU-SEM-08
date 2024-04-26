#include <iostream>
using namespace std;

__global__ void CudaMultiplication(int* x, int* y, int* z, int N1, int N2, int M1, int M2){
      int row = blockIdx.x * blockDim.x + threadIdx.x;
      int col = blockIdx.y * blockDim.y + threadIdx.y;
      int val = 0;


      for(int k=0; k<N2; k++){
          val += x[row*N2+k]*y[k*M2+col] ;
      }

      z[row*M2+col]=val;
}

int main(){

    int N1 = 3;
    int N2 = 4;
    int M1 = 4;
    int M2 = 1;

    int *x = (int *)malloc(sizeof(int)*N1*N2);
    int *y = (int *)malloc(sizeof(int)*M1*M2);
    int *z = (int *)malloc(sizeof(int)*N1*M2);

    for(int i=0;i<N1;i++){
        for(int j=0;j<N2;j++) {
            x[i*N2+j]=1;
        }
    }

    for(int i=0;i<M1;i++){
      for(int j=0;j<M2;j++) {
          y[i*M2+j]=1;
      }
    }

    int *a , *b , *c;
    cudaMalloc(&a , sizeof(int)*N1*N2);
    cudaMalloc(&b , sizeof(int)*M1*M2);
    cudaMalloc(&c , sizeof(int)*N1*M2);

    cudaMemcpy(a, x, sizeof(int)*N1*N2, cudaMemcpyHostToDevice);
    cudaMemcpy(b, y, sizeof(int)*M1*M2, cudaMemcpyHostToDevice);

    dim3 th(N1,M2);
    CudaMultiplication<<<1, th>>>(a, b, c, N1, N2, M1, M2);

    cudaMemcpy(z, c, sizeof(int)*N1*M2, cudaMemcpyDeviceToHost);

    for(int i=0;i<N1;i++){
        for(int j=0;j<M2;j++){
            cout<<z[i*M2+j]<<" ";
        }
        cout<<endl;
    }
    return 0;
}