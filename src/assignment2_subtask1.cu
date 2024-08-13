#include <bits/stdc++.h>
#include <chrono>
using namespace std;

void convolution(float* input, float* output, float* kernel, int input_size, int output_size, int kernel_size) {
    for (int i = 0; i < output_size; i++) {
        for (int j = 0; j < output_size; j++) {
            output[i * output_size + j]=0.0f;
            for (int k = 0; k < kernel_size; k++) {
                for (int l = 0; l < kernel_size; l++) {
                    output[i * output_size + j] += input[(i + k) * input_size + (j + l)] * kernel[k * kernel_size + l];
                }
            }
        }
    }
}

void convolution_with_padding(float* input, float* output, float* kernel, int input_size, int output_size, int kernel_size,int padding) {
    int padded_input_size = input_size + 2 * padding;
    float* padded_input = (float*)malloc(padded_input_size * padded_input_size * sizeof(float));
    
    for (int i = 0; i < padded_input_size; ++i) {
        for (int j = 0; j < padded_input_size; ++j) {
            if (i < padding || i >= padded_input_size - padding || j < padding || j >= padded_input_size - padding) {
                padded_input[i * padded_input_size + j] = 0.0f;
            } else {
                padded_input[i * padded_input_size + j] = input[(i - padding) * input_size + (j - padding)];
            }
        }
    }
    for (int i = 0; i < output_size; i++) {
        for (int j = 0; j < output_size; j++) {
            output[i * output_size + j] = 0.0f;
            for (int k = 0; k < kernel_size; k++) {
                for (int l = 0; l < kernel_size; l++) {
                    output[i * output_size + j] += padded_input[(i + k) * padded_input_size + (j + l)] * kernel[k * kernel_size + l];
                }
            }
        }
    }
    
    free(padded_input);
}

float relu(float x) {
    return std::max(0.0f, x);
}


void max_pooling(float* input, float* output, int input_size, int output_size) {
    int stride = input_size / output_size;
    for (int i = 0; i < output_size; ++i) {
        for (int j = 0; j < output_size; ++j) {
            float max_val = input[i * stride * input_size + j * stride];
            for (int m = 0; m < stride; ++m) {
                for (int n = 0; n < stride; ++n) {
                    max_val = std::max(max_val, input[(i * stride + m) * input_size + j * stride + n]);
                }
            }
            output[i * output_size + j] = max_val;
        }
    }
}

void avg_pooling(float* input, float* output, int input_size, int output_size) {
    int stride = input_size / output_size;
    for (int i = 0; i < output_size; ++i) {
        for (int j = 0; j < output_size; ++j) {
            float sum = 0.0f;
            for (int m = 0; m < stride; ++m) {
                for (int n = 0; n < stride; ++n) {
                    sum += input[(i * stride + m) * input_size + j * stride + n];
                }
            }
            output[i * output_size + j] = sum / (stride * stride);
        }
    }
}

void softmax(float* input, int size) {
    float sum = 0.0f;
    for(int i=0;i<size;i++){
        sum += exp(input[i]);
    }
    for(int i=0;i<size;i++){
        input[i]=exp(input[i])/sum;
    }
}

void sigmoid(float* input, int size) {
    for (int i = 0; i < size; ++i) {
        input[i] = 1 / (1 + std::exp(-input[i]));
    }
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
        for(int i=0;i<n*m;i++){
            input[i] = a==0 ? relu(input[i]) : tanh(input[i]);
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