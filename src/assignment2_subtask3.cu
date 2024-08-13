#include <bits/stdc++.h>
using namespace std;
#include <dirent.h> 
#include <sys/stat.h>


#define T 256

bool directoryExists(const std::string& path) {
    struct stat info;
    return stat(path.c_str(), &info) == 0 && (info.st_mode & S_IFDIR);
}

bool createDirectory(const std::string& path) {
    if (mkdir(path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) != 0) {
        std::cerr << "Error creating directory " << path << std::endl;
        return false;
    }
    return true;
}

__global__ void convLayerKernel(float* input, float* output, float* kernels, float* biases, int input_size, int output_size, int kernel_size, int inchannels, int outchannels) {
    int outputIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int output_element_index = outputIndex % (output_size * output_size);

    if (outputIndex < outchannels * output_size * output_size) {
        float* current_output = output + (outputIndex / (output_size * output_size)) * (output_size * output_size);
        float temp = 0.0f;
        for(int f=0;f<inchannels;f++){
            float* current_input = input + f*(input_size * input_size);
            float* current_kernel = kernels + (outputIndex / (output_size * output_size))*(kernel_size * kernel_size)*inchannels + f*kernel_size*kernel_size;
            
            int row = output_element_index / output_size;
            int col = output_element_index % output_size;

            float sum = 0.0f;
            for (int k = 0; k < kernel_size; k++) {
                for (int l = 0; l < kernel_size; l++) {
                    int input_i = row + k;
                    int input_j = col + l;
                    if (input_i >= 0 && input_i < input_size && input_j >= 0 && input_j < input_size) {
                        sum += current_input[input_i * input_size + input_j] * current_kernel[k * kernel_size + l];
                    }
                }
            }
            temp+=sum;
        }
        current_output[output_element_index] = temp + biases[outputIndex / (output_size * output_size)];
    }
}

void convlayer(float* input, float* output,float* kernels,float* biases, int input_size, int output_size,int kernel_size,int inchannels,int outchannels) {
    int threads_per_block = T;
    int num_elements = outchannels * output_size * output_size;
    int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;
    convLayerKernel<<<num_blocks, threads_per_block>>>(input, output, kernels, biases, input_size, output_size, kernel_size, inchannels, outchannels);
    cudaDeviceSynchronize();
}


__global__ void maxpoolLayerKernel(float* input, float* output, int input_size, int output_size, int num_filters) {
    int output_index = blockIdx.x * blockDim.x + threadIdx.x;
    int num_elements = output_size * output_size;
    int filter_index = output_index / num_elements;
    int output_element_index = output_index % num_elements;
    int stride = input_size / output_size;
    
    if (output_index < num_elements * num_filters) {
        float* current_input = input + filter_index * input_size * input_size;
        float* current_output = output + filter_index * num_elements;

        int i = output_element_index / output_size;
        int j = output_element_index % output_size;

        float max_val = current_input[i * stride * input_size + j * stride];
        for (int m = 0; m < stride; ++m) {
            for (int n = 0; n < stride; ++n) {
                float val = current_input[(i * stride + m) * input_size + j * stride + n];
                max_val = fmaxf(max_val, val);
            }
        }
        current_output[output_element_index] = max_val;
    }
}

void maxpoollayer(float* input, float* output, int input_size, int output_size,int inchannels) {
    int threads_per_block = T;
    int num_elements = output_size * output_size * inchannels;
    int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;
    maxpoolLayerKernel<<<num_blocks, threads_per_block>>>(input, output, input_size, output_size, inchannels);
    cudaDeviceSynchronize();
}


__global__ void reluLayerKernel(float* input,int input_size,int inchannels){
    int output_index = blockIdx.x * blockDim.x + threadIdx.x;
    if(output_index < input_size * input_size * inchannels){
        float val = input[output_index];
        if(val < 0.0f) val = 0.0f;
        input[output_index] = val;
    }
}

void relulayer(float* input, int input_size,int inchannels) {
    int threads_per_block = T;
    int num_elements = input_size * input_size * inchannels;
    int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;
    reluLayerKernel<<<num_blocks, threads_per_block>>>(input,input_size,inchannels);
    cudaDeviceSynchronize();
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

int main(int argc, char* argv[]) {

    auto start = std::chrono::high_resolution_clock::now();

    int kernel_size1 = 5;
    int inchannels1 = 1;
    int outchannels1 = 20;
    string filename1 = "./trained_weights/conv1.txt";

    int kernel_size2 = 5;
    int inchannels2 = 20;
    int outchannels2 = 50;
    string filename2 = "./trained_weights/conv2.txt";

    int kernel_size3 = 4;
    int inchannels3 = 50;
    int outchannels3 = 500;
    string filename3 = "./trained_weights/fc1.txt";

    int kernel_size4 = 1;
    int inchannels4 = 500;
    int outchannels4 = 10;
    string filename4 = "./trained_weights/fc2.txt";

    // Kernel and bias arrays
    float* kernel_input1 = (float*)malloc(inchannels1 * kernel_size1 * kernel_size1 * outchannels1 * sizeof(float));
    float* bias_input1 = (float*)malloc(outchannels1 * sizeof(float));
    float* kernel_input2 = (float*)malloc(inchannels2 * kernel_size2 * kernel_size2 * outchannels2 * sizeof(float));
    float* bias_input2 = (float*)malloc(outchannels2 * sizeof(float));
    float* kernel_input3 = (float*)malloc(inchannels3 * kernel_size3 * kernel_size3 * outchannels3 * sizeof(float));
    float* bias_input3 = (float*)malloc(outchannels3 * sizeof(float));
    float* kernel_input4 = (float*)malloc(inchannels4 * kernel_size4 * kernel_size4 * outchannels4 * sizeof(float));
    float* bias_input4 = (float*)malloc(outchannels4 * sizeof(float));

    // Reading kernel and bias data from files
    ifstream finput1(filename1);
    for (int i = 0; i < inchannels1 * kernel_size1 * kernel_size1 * outchannels1; i++) {
        finput1 >> kernel_input1[i];
    }
    for (int i = 0; i < outchannels1; i++) {
        finput1 >> bias_input1[i];
    }
    finput1.close();

    ifstream finput2(filename2);
    for (int i = 0; i < inchannels2 * kernel_size2 * kernel_size2 * outchannels2; i++) {
        finput2 >> kernel_input2[i];
    }
    for (int i = 0; i < outchannels2; i++) {
        finput2 >> bias_input2[i];
    }
    finput2.close();

    ifstream finput3(filename3);
    for (int i = 0; i < inchannels3 * kernel_size3 * kernel_size3 * outchannels3; i++) {
        finput3 >> kernel_input3[i];
    }
    for (int i = 0; i < outchannels3; i++) {
        finput3 >> bias_input3[i];
    }
    finput3.close();

    ifstream finput4(filename4);
    for (int i = 0; i < inchannels4 * kernel_size4 * kernel_size4 * outchannels4; i++) {
        finput4 >> kernel_input4[i];
    }
    for (int i = 0; i < outchannels4; i++) {
        finput4 >> bias_input4[i];
    }
    finput4.close();

    // CUDA kernel and bias arrays
    float* kernels1;
    float* biases1;
    float* kernels2;
    float* biases2;
    float* kernels3;
    float* biases3;
    float* kernels4;
    float* biases4;

    // CUDA memory allocation and copy
    cudaMalloc(&kernels1, inchannels1 * kernel_size1 * kernel_size1 * outchannels1 * sizeof(float));
    cudaMalloc(&biases1, outchannels1 * sizeof(float));
    cudaMemcpy(kernels1, kernel_input1, inchannels1 * kernel_size1 * kernel_size1 * outchannels1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(biases1, bias_input1, outchannels1 * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&kernels2, inchannels2 * kernel_size2 * kernel_size2 * outchannels2 * sizeof(float));
    cudaMalloc(&biases2, outchannels2 * sizeof(float));
    cudaMemcpy(kernels2, kernel_input2, inchannels2 * kernel_size2 * kernel_size2 * outchannels2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(biases2, bias_input2, outchannels2 * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&kernels3, inchannels3 * kernel_size3 * kernel_size3 * outchannels3 * sizeof(float));
    cudaMalloc(&biases3, outchannels3 * sizeof(float));
    cudaMemcpy(kernels3, kernel_input3, inchannels3 * kernel_size3 * kernel_size3 * outchannels3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(biases3, bias_input3, outchannels3 * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&kernels4, inchannels4 * kernel_size4 * kernel_size4 * outchannels4 * sizeof(float));
    cudaMalloc(&biases4, outchannels4 * sizeof(float));
    cudaMemcpy(kernels4, kernel_input4, inchannels4 * kernel_size4 * kernel_size4 * outchannels4 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(biases4, bias_input4, outchannels4 * sizeof(float), cudaMemcpyHostToDevice);

    // Free host memory
    free(kernel_input1);
    free(bias_input1);
    free(kernel_input2);
    free(bias_input2);
    free(kernel_input3);
    free(bias_input3);
    free(kernel_input4);
    free(bias_input4);

    int ctr = 0;

    std::string folder_path = "./pre-proc-img";

    DIR *dir = opendir(folder_path.c_str());

    int val = 0;

    if(dir){
        
        struct dirent *entry;
        
        while ((entry = readdir(dir)) != NULL) {
            
            ctr++;

            float* image = (float*)malloc(28*28 * sizeof(float));
            
            string filename = entry->d_name;

            std::string file_path = folder_path + "/" + filename;

            ifstream finput(file_path);

            for (int i = 0; i < 28 * 28; i++) {
                finput >> image[i];
            }
            finput.close();


            float* output1;
            cudaMalloc(&output1, 20 * 24 * 24 * sizeof(float));
            float* output2;
            cudaMalloc(&output2, 20 * 12 * 12 * sizeof(float));
            float* output3;
            cudaMalloc(&output3, 50 * 8 * 8 * sizeof(float));
            float* output4;
            cudaMalloc(&output4, 50 * 4 * 4 * sizeof(float));
            float* output5;
            cudaMalloc(&output5, 500 * sizeof(float));
            float* output6;
            cudaMalloc(&output6, 10 * sizeof(float));

            float* input1;
            cudaMalloc(&input1, 28 * 28 * sizeof(float));
            cudaMemcpy(input1, image, 28 * 28 * sizeof(float), cudaMemcpyHostToDevice);

            float* output = (float*)malloc(10 * sizeof(float));
        
            convlayer(input1, output1, kernels1, biases1, 28, 24, 5, 1, 20);
            maxpoollayer(output1, output2, 24, 12, 20);
            convlayer(output2, output3, kernels2, biases2, 12, 8, 5, 20, 50);
            maxpoollayer(output3, output4, 8, 4, 50);
            convlayer(output4, output5, kernels3, biases3, 4, 1, 4, 50, 500);
            relulayer(output5, 1, 500);
            convlayer(output5, output6, kernels4, biases4, 1, 1, 1, 500, 10);

            cudaMemcpy(output, output6, 10 * sizeof(float), cudaMemcpyDeviceToHost);

            softmax(output, 10);

            string outfolder = "./output/";
            if (!directoryExists(outfolder)) {
                if (!createDirectory(outfolder)) {
                    std::cerr << "Failed to create directory. Exiting...\n";
                    return 1;
                }
            }

            string outfile = "./output/"+filename;
            int ans = (int)(filename[10]-'0');

            ofstream out(outfile);
            priority_queue<pair<double, int>> pq;
            for (int i = 0; i < 10; i++) {
                pq.push({output[i] * 100.0, i});
            }
            int count = 0;
            while (count < 5 && !pq.empty()) {
                auto top_pair = pq.top();
                out <<top_pair.first <<" class "<< top_pair.second << endl;
                pq.pop();
                if(count==0 && top_pair.second == ans){
                    val++;
                }
                count++;
            }
            out.close();

            cudaFree(input1);
            cudaFree(output1);
            cudaFree(output2);
            cudaFree(output3);
            cudaFree(output4);
            cudaFree(output5);
            cudaFree(output6);

        }
    }

    cout<<"Accuracy : "<<ctr<<" images "<<((float)val/(float)ctr) * 100<<endl;

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Execution time: " << duration.count() << " milliseconds" << std::endl;

    cudaFree(kernels1);
    cudaFree(biases1);
    cudaFree(kernels2);
    cudaFree(biases2);
    cudaFree(kernels3);
    cudaFree(biases3);
    cudaFree(kernels4);
    cudaFree(biases4);

    

    return 0;
}