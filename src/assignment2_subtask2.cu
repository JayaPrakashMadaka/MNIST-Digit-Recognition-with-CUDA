#include <bits/stdc++.h>
using namespace std;
#define T 256
#include <cuda_runtime.h>

__global__ void convolutionKernel(float* input, float* output, float* kernel, int input_size, int output_size, int kernel_size){
    int outputIndex = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(outputIndex < output_size * output_size){
        int row = outputIndex / output_size;
        int col = outputIndex % output_size;
        float sum = 0.0f;
        for (int k = 0; k < kernel_size; k++) {
            for (int l = 0; l < kernel_size; l++) {
                int input_i = row + k;
                int input_j = col + l;
                if (input_i >= 0 && input_i < input_size && input_j >= 0 && input_j < input_size) {
                    sum += input[input_i * input_size + input_j] * kernel[k * kernel_size + l];
                }
            }
        }
        output[outputIndex] = sum;
    }
}

void convolution(float* input, float* output, float* kernel, int input_size, int output_size, int kernel_size){
    int threadsPerBlock = T;
    int numElements = output_size * output_size;
    int numBlocks = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    
    float* finput;
    float* foutput;
    float* fkernel;
    
    cudaMalloc(&finput, input_size * input_size * sizeof(float));
    cudaMalloc(&fkernel, kernel_size * kernel_size * sizeof(float));
    cudaMalloc(&foutput, output_size * output_size * sizeof(float));
    
    cudaMemcpy(fkernel, kernel, kernel_size * kernel_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(finput, input, input_size * input_size * sizeof(float), cudaMemcpyHostToDevice);
    
    convolutionKernel<<<numBlocks, threadsPerBlock>>>(finput, foutput, fkernel, input_size, output_size, kernel_size);
    cudaDeviceSynchronize();
    
    cudaMemcpy(output, foutput, output_size * output_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(fkernel);
    cudaFree(foutput);
    cudaFree(finput);
}

__global__ void convolutionPaddingKernel(float* padded_input, float* output, float* kernel, int padded_input_size, int output_size, int kernel_size,int padding){
    int outputIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if(outputIndex < output_size * output_size){
        int i = outputIndex / output_size;
        int j = outputIndex % output_size;
        float sum = 0.0f;
        for (int k = 0; k < kernel_size; k++) {
            for (int l = 0; l < kernel_size; l++) {
                sum += padded_input[(i + k) * padded_input_size + (j + l)] * kernel[k * kernel_size + l];
            }
        }
        output[outputIndex] = sum;
    }
}

void convolution_with_padding(float* input, float* output, float* kernel, int input_size, int output_size, int kernel_size,int padding){
    int threadsPerBlock = T;
    int numElements = output_size * output_size;
    int numBlocks = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    int padded_input_size = input_size + 2 * padding;

    float* padded_input = (float*)malloc(padded_input_size*padded_input_size*sizeof(float));
    for (int i = 0; i < padded_input_size; ++i) {
        for (int j = 0; j < padded_input_size; ++j) {
            if (i < padding || i >= padded_input_size - padding || j < padding || j >= padded_input_size - padding) {
                padded_input[i * padded_input_size + j] = 0.0f;
            } else {
                padded_input[i * padded_input_size + j] = input[(i - padding) * input_size + (j - padding)];
            }
        }
    }

    float* foutput;
    float* fkernel;
    float* fpadded_input;
    
    cudaMalloc(&fkernel, kernel_size * kernel_size * sizeof(float));
    cudaMalloc(&foutput, output_size * output_size * sizeof(float));
    cudaMalloc(&fpadded_input, padded_input_size * padded_input_size * sizeof(float));
    
    cudaMemcpy(fkernel, kernel, kernel_size * kernel_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(fpadded_input, padded_input, padded_input_size * padded_input_size * sizeof(float), cudaMemcpyHostToDevice);
    
    convolutionPaddingKernel<<<numBlocks, threadsPerBlock>>>(fpadded_input, foutput, fkernel, padded_input_size, output_size, kernel_size,padding);
    cudaDeviceSynchronize();
    
    cudaMemcpy(output, foutput, output_size * output_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(fpadded_input);
    cudaFree(fkernel);
    cudaFree(foutput);
}



__global__ void maxpoolingKernel(float* input, float* output, int input_size, int output_size) {
    int output_index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = input_size / output_size;
    
    if (output_index < output_size * output_size) {

        int i = output_index / output_size;
        int j = output_index % output_size;

        float max_val = input[i * stride * input_size + j * stride];
        for (int m = 0; m < stride; ++m) {
            for (int n = 0; n < stride; ++n) {
                float val = input[(i * stride + m) * input_size + j * stride + n];
                max_val = fmaxf(max_val, val);
            }
        }
        output[output_index] = max_val;
    }
}

void max_pooling(float* input, float* output, int input_size, int output_size){
    int threadsPerBlock = T;
    int numElements = output_size * output_size;
    int numBlocks = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    
    float* finput;
    float* foutput;
    
    cudaMalloc(&finput, input_size * input_size * sizeof(float));
    cudaMalloc(&foutput, output_size * output_size * sizeof(float));
    
    cudaMemcpy(finput, input, input_size * input_size * sizeof(float), cudaMemcpyHostToDevice);
    
    maxpoolingKernel<<<numBlocks, threadsPerBlock>>>(finput, foutput, input_size, output_size);
    cudaDeviceSynchronize();
    
    cudaMemcpy(output, foutput, output_size * output_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(foutput);
    cudaFree(finput);
}

__global__ void avgpoolingKernel(float* input, float* output, int input_size, int output_size) {
    int output_index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = input_size / output_size;
    
    if (output_index < output_size * output_size) {

        int i = output_index / output_size;
        int j = output_index % output_size;

        float sum = 0.0f;
        for (int m = 0; m < stride; ++m) {
            for (int n = 0; n < stride; ++n) {
                float val = input[(i * stride + m) * input_size + j * stride + n];
                sum += val;
            }
        }
        output[output_index] = sum / (stride * stride);
    }
}

void avg_pooling(float* input, float* output, int input_size, int output_size){
    int threadsPerBlock = T;
    int numElements = output_size * output_size;
    int numBlocks = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    
    float* finput;
    float* foutput;
    
    cudaMalloc(&finput, input_size * input_size * sizeof(float));
    cudaMalloc(&foutput, output_size * output_size * sizeof(float));
    
    cudaMemcpy(finput, input, input_size * input_size * sizeof(float), cudaMemcpyHostToDevice);
    
    avgpoolingKernel<<<numBlocks, threadsPerBlock>>>(finput, foutput, input_size, output_size);
    cudaDeviceSynchronize();
    
    cudaMemcpy(output, foutput, output_size * output_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(foutput);
    cudaFree(finput);
}


__global__ void softmaxKernel(float* input, int size,float sum) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < size) {
        input[tid] = expf(input[tid]) / sum;
    }
}



void softmax(float* input, int size) {
    float sum = 0.0f;
    for(int i=0;i<size;i++){
        sum += exp(input[i]);
    }
    int threadsPerBlock = T;
    int numElements = size;
    int numBlocks = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    float* finput;
    cudaMalloc(&finput, size * sizeof(float));
    cudaMemcpy(finput, input, size * sizeof(float), cudaMemcpyHostToDevice);
    softmaxKernel<<<numBlocks, threadsPerBlock>>>(finput,size,sum);
    cudaDeviceSynchronize();
    cudaMemcpy(input, finput, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(finput);
}

__global__ void sigmoidKernel(float* input, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        input[tid] = 1 / (1 + expf(-input[tid]));
    }
}

void sigmoid(float* input,int size){
    int threadsPerBlock = T;
    int numElements = size;
    int numBlocks = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    float* finput;
    cudaMalloc(&finput, size * sizeof(float));
    cudaMemcpy(finput, input, size * sizeof(float), cudaMemcpyHostToDevice);
    sigmoidKernel<<<numBlocks, threadsPerBlock>>>(finput,size);
    cudaDeviceSynchronize();
    cudaMemcpy(input, finput, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(finput);
}

__global__ void reluKernel(float* input, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        input[tid] = fmaxf(0.0f, input[tid]);
    }
}

void relu(float* input, int size) {
    int threadsPerBlock = T;
    int numElements = size;
    int numBlocks = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    float* finput;
    cudaMalloc(&finput, size * sizeof(float));
    cudaMemcpy(finput, input, size * sizeof(float), cudaMemcpyHostToDevice);
    reluKernel<<<numBlocks, threadsPerBlock>>>(finput, size);
    cudaDeviceSynchronize();
    cudaMemcpy(input, finput, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(finput);
}

__global__ void tanhKernel(float* input, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        input[tid] = tanhf(input[tid]);
    }
}

void tanhfunc(float* input, int size) {
    int threadsPerBlock = T;
    int numElements = size;
    int numBlocks = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    float* finput;
    cudaMalloc(&finput, size * sizeof(float));
    cudaMemcpy(finput, input, size * sizeof(float), cudaMemcpyHostToDevice);
    tanhKernel<<<numBlocks, threadsPerBlock>>>(finput, size);
    cudaDeviceSynchronize();
    cudaMemcpy(input, finput, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(finput);
}

void printMatrix(float* matrix, int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            std::cout << matrix[i * size + j] << " ";
        }
        std::cout << std::endl;
    }
}

void printMatrix2(float* matrix,int n,int m){
    for(int i=0;i<n;i++){
        for(int j=0;j<m;j++){
            cout<<matrix[i*n+j]<<" ";
        }
        cout<<endl;
    }
}

void printVector(float* arr,int n){
    for(int i=0;i<n;i++){
        cout<<arr[i]<<" ";
    }
    cout<<endl;
}

int main (int argc, char *argv[]) {
    auto start = std::chrono::high_resolution_clock::now();
    int ch;
    ch = stoi(argv[1]);
    if(ch==1){
        int n = stoi(argv[2]);
        int k = stoi(argv[3]);
        int p = stoi(argv[4]);
        float* input = (float*)malloc(n*n*sizeof(float));
        float* kernel = (float*)malloc(k*k*sizeof(float));
        int ctr = 5;
        for(int i=0;i<n*n;i++){
            input[i] = stof(argv[ctr]);
            ctr++;
        }
        for(int i=0;i<k*k;i++){
            kernel[i] = stof(argv[ctr]);
            ctr++;
        }
        if(p==0){
            int osz = n - k + 1;
            float* output = (float*)malloc(osz*osz*sizeof(float));
            convolution(input,output,kernel,n,osz,k);
            printMatrix(output,osz);
        }
        else{
            int osz = n + 2*p - k + 1;
            float* output = (float*)malloc(osz*osz*sizeof(float));
            convolution_with_padding(input,output,kernel,n,osz,k,p);
            printMatrix(output,osz);
        }
    }
    else if(ch==2){
        int a = stoi(argv[2]);
        int n = atoi(argv[3]);
        int m = stoi(argv[4]);
        int ctr = 5;
        float* input = (float*)malloc(n*m*sizeof(float));
        for(int i=0;i<n*m;i++){
            input[i] = stof(argv[ctr]);
            ctr++;
        }
        if(a==0){
            relu(input,n*m);
        }
        else{
            tanhfunc(input,n*m);
        }
        printMatrix2(input,n,m);
    }
    else if(ch==3){
        int pool = stoi(argv[2]);
        int sz = stoi(argv[3]);
        int n = sz;
        if(sz%2 != 0){
            n = sz+1;
        }
        int output_size = n/2;
        float* input = (float*)malloc(n*n*sizeof(float));
        float* output = (float*)malloc(output_size*output_size*sizeof(float));
        int ctr = 4;
        for(int i=0;i<n*n;i++){
            input[i] = 0.0f;
        }
        for(int i=0;i<sz;i++){
            for(int j=0;j<sz;j++){
                input[i*n+j] = stof(argv[ctr]);
                ctr++;
            }
        }
        if(pool==0){
            max_pooling(input,output,n,output_size);
        }
        else{
            avg_pooling(input,output,n,output_size);
        }
        printMatrix(output,output_size);
    }
    else if(ch==4){
        int choice = stoi(argv[2]);
        int n = argc -  3;
        float* input = (float*)malloc(n*sizeof(float));
        int ctr = 3;
        for(int i=0;i<n;i++){
            input[i] = stof(argv[ctr]);
            ctr++;
        }
        if(choice==0){
            sigmoid(input,n);
            printVector(input,n);
        }
        else{
            softmax(input,n);
            printVector(input,n);
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Execution time: " << duration.count() << " milliseconds" << std::endl;
    
    return 0;
}
