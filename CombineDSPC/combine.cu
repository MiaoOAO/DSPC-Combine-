#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <cuda_runtime.h>
#include <omp.h>
#include <limits>
#include <chrono>
#include <mpi.h>

using namespace cv;
using namespace std;

const string filePath = ".jpg";


//Laplacian
void mpi_Laplacian_Filter(int argc, char** argv, Mat& image);
Mat single_Laplacian_Filter(Mat& image);

//Unsharp masking
Mat single_thread_unsharp_masking(Mat& image);
Mat omp_unsharp_masking(Mat& image);
void mpi_unsharp_masking(int argc, char** argv);

//Laplacian function call
void applyLaplacianToColorImage(Mat& inputImage, Mat& outputImage, double sharpeningFactor);
void applyLaplacianSharpen(const Mat& inputChannel, Mat& outputChannel, double sharpeningFactor);

//Function to load/input an image
Mat loadImage(const string& filename) {
    //change to your own image path here
    Mat image = imread("C:/Users/josan/image/" + filename + ".jpg", IMREAD_COLOR);
    return image;
}

// Function to save an image
void saveImage(const string& filename, const Mat& image) {
    imwrite(filename, image);
}

// Measure time helper function
double measureTime(int64 start) {
    int64 end = getTickCount();
    return (end - start) / getTickFrequency();  // Convert to seconds
}


void apply_morphology(Mat& image, Mat& output) {
    Mat element = getStructuringElement(MORPH_RECT, Size(1, 1));
    morphologyEx(image, output, MORPH_OPEN, element);
    morphologyEx(output, output, MORPH_CLOSE, element);
}

void image_preprocessing(Mat& image, Mat& preprocessed_image) {
    int rows = image.rows;
    int cols = image.cols;
    int block_size = 64;

    int num_threads = omp_get_max_threads();
    //cout << "Number of threads available: " << num_threads << endl;

    omp_set_num_threads(num_threads);

    preprocessed_image = image.clone();

#pragma omp parallel
    {
#pragma omp for collapse(2) schedule(dynamic)
        for (int i = 0; i < rows; i += block_size) {
            for (int j = 0; j < cols; j += block_size) {
                int block_width = min(block_size, cols - j);
                int block_height = min(block_size, rows - i);

                Mat block = image(Rect(j, i, block_width, block_height));
                Mat output_block = block.clone();

                apply_morphology(block, output_block);

#pragma omp critical
                {
                    output_block.copyTo(preprocessed_image(Rect(j, i, block_width, block_height)));
                }
            }
        }
    }
}

__global__ void apply_laplacian_cuda(unsigned char* d_input, unsigned char* d_output, int width, int height, int channels, double alpha) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = (y * width + x) * channels;

    const double laplacianKernel[3][3] = {
        { 0, -1,  0 },
        {-1,  4, -1 },
        { 0, -1,  0 }
    };

    for (int c = 0; c < channels; ++c) {
        double result = 0.0;
        for (int ky = -1; ky <= 1; ++ky) {
            for (int kx = -1; kx <= 1; ++kx) {
                int ix = min(max(x + kx, 0), width - 1);
                int iy = min(max(y + ky, 0), height - 1);
                int input_idx = (iy * width + ix) * channels + c;
                result += d_input[input_idx] * laplacianKernel[ky + 1][kx + 1];
            }
        }

        double newVal = d_input[idx + c] + alpha * result;
        d_output[idx + c] = min(max(newVal, 0.0), 255.0);
    }
}



Mat cuda_Laplacian_Filter(Mat& image) {
    if (image.empty()) {
        cout << "Error: Image not found!" << endl;
        return Mat();
    }

    if (image.channels() != 3) {
        cout << "Error: Input image does not have 3 channels!" << endl;
        return Mat();
    }

    int rows = image.rows;
    int cols = image.cols;
    int channels = 1;  // Since we'll process one channel at a time
    size_t imageSize = rows * cols * channels * sizeof(unsigned char);
    double sharpeningFactor = 1.0;  // Adjust sharpening intensity

    // Split the image into 3 channels
    vector<Mat> inputChannels(3);
    split(image, inputChannels);

    vector<Mat> outputChannels(3);

    // Allocate device memory for input and output
    unsigned char* d_input;
    unsigned char* d_output;
    cudaMalloc((void**)&d_input, imageSize);
    cudaMalloc((void**)&d_output, imageSize);

    // Preprocessing buffer
    Mat preProcessed;

    // Loop through each channel, apply preprocessing and Laplacian sharpening on each
    for (int i = 0; i < 3; ++i) {
        // Apply preprocessing (e.g., Gaussian blur)
        image_preprocessing(inputChannels[i], preProcessed);

        // Allocate output channel and copy preprocessed channel to device
        cudaMemcpy(d_input, preProcessed.data, imageSize, cudaMemcpyHostToDevice);

        // Set block and grid sizes for CUDA kernel launch
        dim3 blockSize(16, 16);
        dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y);

        // Launch the Laplacian CUDA kernel
        apply_laplacian_cuda << <gridSize, blockSize >> > (d_input, d_output, cols, rows, channels, sharpeningFactor);

        // Copy output back to host memory
        outputChannels[i] = Mat(rows, cols, CV_8UC1);
        cudaMemcpy(outputChannels[i].data, d_output, imageSize, cudaMemcpyDeviceToHost);
    }

    // Free GPU memory
    cudaFree(d_input);
    cudaFree(d_output);

    // Merge the sharpened channels back into a color image
    Mat sharpenedColor;
    merge(outputChannels, sharpenedColor);

    return sharpenedColor;
}



__global__ void apply_unsharp_masking_cuda(unsigned char* d_input, unsigned char* d_blurred, unsigned char* d_output, int width, int height, int channels, double alpha) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = (y * width + x) * channels;

    for (int c = 0; c < channels; ++c) {
        // Calculate the mask: original - blurred
        double mask = (double)d_input[idx + c] - (double)d_blurred[idx + c];

        // Unsharp masking: original + alpha * mask
        double newVal = d_input[idx + c] + alpha * mask;

        // Clamp the result between 0 and 255
        d_output[idx + c] = min(max(newVal, 0.0), 255.0);
    }
}

Mat cuda_unsharp_masking(Mat& image) {
    // Image properties
    int width = image.cols;
    int height = image.rows;
    int channels = image.channels();
    int size = width * height * channels;

    // Allocate memory on the GPU
    unsigned char* d_input, * d_blurred, * d_output;
    cudaMalloc(&d_input, size * sizeof(unsigned char));
    cudaMalloc(&d_blurred, size * sizeof(unsigned char));
    cudaMalloc(&d_output, size * sizeof(unsigned char));

    // Copy the image to the GPU
    cudaMemcpy(d_input, image.data, size * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Perform image preprocessing (on CPU)
    Mat preProcessed;
    image_preprocessing(image, preProcessed); // Assuming this is CPU-based

    // Upload preProcessed image to GPU for blurring (assuming preprocessing outputs a Mat)
    cudaMemcpy(d_blurred, preProcessed.data, size * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Set parameters for CUDA
    double alpha = 1; // Set sharpening factor

    // Define CUDA grid and block dimensions
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    // Apply the unsharp masking kernel
    apply_unsharp_masking_cuda << <grid, block >> > (d_input, d_blurred, d_output, width, height, channels, alpha);

    // Copy the sharpened image back to the host
    Mat sharpenedImage(height, width, image.type()); // Create an empty Mat for the result
    cudaMemcpy(sharpenedImage.data, d_output, size * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Save the result
    //saveImage("C:/Users/wongc/source/repos/DSCP/DSCP/Image/Sharpened_CUDA.png", sharpenedImage);

    // Free GPU memory
    cudaFree(d_input);
    cudaFree(d_blurred);
    cudaFree(d_output);

    return sharpenedImage;
}




int main(int argc, char** argv) {
    Mat processed_image_single_laplacian, processed_image_omp_laplacian, processed_image_cuda_laplacian, processed_image_mpi_laplacian,
        processed_image_single_unsharp, processed_image_omp_unsharp, processed_image_cuda_unsharp, processed_image_mpi_unsharp;

    int64 startTime, endTime;
    double timeTaken;

    string imagePath = "";

    cout << "Image Path: ";
    getline(cin, imagePath);
    
    //  Load the image
    startTime = getTickCount();
    Mat image = loadImage(imagePath);
    timeTaken = measureTime(startTime);
    cout << "Time taken to load image: " << timeTaken << " seconds" << endl;


    //  Display the original image
    namedWindow("Original Image", WINDOW_AUTOSIZE);
    imshow("Original Image", image);
    moveWindow("Original Image", 0, 0);


    //Laplacian Filter
        //  Processed image by laplacian with single thread 
        startTime = getTickCount();
        processed_image_single_laplacian = single_Laplacian_Filter(image);
        timeTaken = measureTime(startTime);
        cout << "\n\nLAPLACIAN FILTER\nTime taken to apply single thread Laplacian filter: " << timeTaken << " seconds" << endl;
        // Display the sharpened image
        namedWindow("single thread Laplacian filter processed Image", WINDOW_AUTOSIZE);
        imshow("single thread Laplacian filter processed Image", processed_image_single_laplacian);
        moveWindow("single thread Laplacian filter processed Image", 500, 0);
        // Save the sharpened image
        imwrite("single_thread_Laplacian filter_processed_" + imagePath + filePath, processed_image_single_laplacian);


        //  Processed image by laplacian with CUDA
        startTime = getTickCount();
        processed_image_cuda_laplacian = cuda_Laplacian_Filter(image);
        timeTaken = measureTime(startTime);
        cout << "Time taken to apply cuda Laplacian filter: " << timeTaken << " seconds" << endl;
        // Display the sharpened image
        namedWindow("Cuda Laplacian filter processed Image", WINDOW_AUTOSIZE);
        imshow("Cuda Laplacian filter processed Image", processed_image_cuda_laplacian);
        moveWindow("Cuda Laplacian filter processed Image", 1000, 0);
        // Save the sharpened image
        imwrite("Cuda Laplacian filter_processed_" + imagePath + filePath, processed_image_cuda_laplacian);


        //  Processed image by laplacian with MPI
        //mpi_Laplacian_Filter(argc, argv);


    //Unsharp Masking
         //  Processed image by unsharp with single thread 
        startTime = getTickCount();
        processed_image_single_unsharp = single_thread_unsharp_masking(image);
        timeTaken = measureTime(startTime);
        cout << "\n\nUNSHARP MASKING\nTime taken to apply single thread Unsharp Masking: " << timeTaken << " seconds" << endl;
        // Display the sharpened image
        namedWindow("single thread unsharp masking processed Image", WINDOW_AUTOSIZE);
        imshow("single thread unsharp masking processed Image", processed_image_single_unsharp);
        moveWindow("single thread unsharp masking processed Image", 1000, 500);
        // Save the sharpened image
        imwrite("single thread unsharp masking_processed_" + imagePath + filePath, processed_image_single_unsharp);


        //  Processed image by unsharp with CUDA
        startTime = getTickCount();
        processed_image_omp_unsharp = omp_unsharp_masking(image);
        timeTaken = measureTime(startTime);
        cout << "Time taken to apply omp Unsharp Masking: " << timeTaken << " seconds" << endl;
        // Display the sharpened image
        namedWindow("omp unsharp masking processed Image", WINDOW_AUTOSIZE);
        imshow("omp unsharp masking processed Image", processed_image_omp_unsharp);
        moveWindow("omp unsharp masking processed Image", 500, 500);
        // Save the sharpened image
        imwrite("omp unsharp masking_processed_" + imagePath + filePath, processed_image_omp_unsharp);


        //  Processed image by unsharp with single thread 
        startTime = getTickCount();
        processed_image_cuda_unsharp = cuda_unsharp_masking(image);
        timeTaken = measureTime(startTime);
        cout << "Time taken to apply cuda Unsharp Masking: " << timeTaken << " seconds" << endl;
        // Display the sharpened image
        namedWindow("cuda unsharp masking processed Image", WINDOW_AUTOSIZE);
        imshow("cuda unsharp masking processed Image", processed_image_cuda_unsharp);
        moveWindow("cuda unsharp masking processed Image", 0, 500);
        // Save the sharpened image
        imwrite("cuda unsharp masking_processed_" + imagePath + filePath, processed_image_cuda_unsharp);

        //  Processed image by unsharp masking with MPI
        //mpi_unsharp_masking(argc, argv);


    waitKey(0); // Wait for a key press
    return 0;
}

// mpiexec -n 4 combineDSPC.exe




/*
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <cuda_runtime.h>
#include <omp.h>

using namespace cv;
using namespace std;

const string filePath = ".jpg";


Mat loadImage(const string& filename) {
    //change to your own image path here
    Mat image = imread("C:/Users/josan/image/" + filename + ".jpg", IMREAD_COLOR);
    return image;
}

// Function to save an image
void saveImage(const string& filename, const Mat& image) {
    imwrite(filename, image);
}

// Measure time helper function
double measureTime(int64 start) {
    int64 end = getTickCount();
    return (end - start) / getTickFrequency();  // Convert to seconds
}


void apply_morphology(Mat& image, Mat& output) {
    Mat element = getStructuringElement(MORPH_RECT, Size(1, 1));
    morphologyEx(image, output, MORPH_OPEN, element);
    morphologyEx(output, output, MORPH_CLOSE, element);
}

void image_preprocessing(Mat& image, Mat& preprocessed_image) {
    int rows = image.rows;
    int cols = image.cols;
    int block_size = 64;

    int num_threads = omp_get_max_threads();
    //cout << "Number of threads available: " << num_threads << endl;

    omp_set_num_threads(num_threads);

    preprocessed_image = image.clone();

#pragma omp parallel
    {
#pragma omp for collapse(2) schedule(dynamic)
        for (int i = 0; i < rows; i += block_size) {
            for (int j = 0; j < cols; j += block_size) {
                int block_width = min(block_size, cols - j);
                int block_height = min(block_size, rows - i);

                Mat block = image(Rect(j, i, block_width, block_height));
                Mat output_block = block.clone();

                apply_morphology(block, output_block);

#pragma omp critical
                {
                    output_block.copyTo(preprocessed_image(Rect(j, i, block_width, block_height)));
                }
            }
        }
    }
}


// CUDA kernel for applying Laplacian filter
__global__ void apply_laplacian_cuda(unsigned char* d_input, unsigned char* d_output, int width, int height, int channels, double alpha) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = (y * width + x) * channels;

    const double laplacianKernel[3][3] = {
        { 0, -1,  0 },
        {-1,  4, -1 },
        { 0, -1,  0 }
    };

    for (int c = 0; c < channels; ++c) {
        double result = 0.0;
        for (int ky = -1; ky <= 1; ++ky) {
            for (int kx = -1; kx <= 1; ++kx) {
                int ix = min(max(x + kx, 0), width - 1);
                int iy = min(max(y + ky, 0), height - 1);
                int input_idx = (iy * width + ix) * channels + c;
                result += d_input[input_idx] * laplacianKernel[ky + 1][kx + 1];
            }
        }

        double newVal = d_input[idx + c] + alpha * result;
        d_output[idx + c] = min(max(newVal, 0.0), 255.0);
    }
}

// Function to apply Laplacian filter on a single channel using CUDA

void applyLaplacianSharpen(const Mat& inputChannel, Mat& outputChannel, double sharpeningFactor) {
    int rows = inputChannel.rows;
    int cols = inputChannel.cols;
    int channels = 1;  // Single-channel image (grayscale)

    size_t imageSize = rows * cols * channels * sizeof(unsigned char);

    // Allocate device memory
    unsigned char* d_input;
    unsigned char* d_output;
    cudaMalloc((void**)&d_input, imageSize);
    cudaMalloc((void**)&d_output, imageSize);

    // Copy input channel data to device
    cudaMemcpy(d_input, inputChannel.data, imageSize, cudaMemcpyHostToDevice);

    // Set block and grid sizes
    dim3 blockSize(16, 16);
    dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y);

    // Launch the CUDA kernel for the Laplacian filter
    apply_laplacian_cuda << <gridSize, blockSize >> > (d_input, d_output, cols, rows, channels, sharpeningFactor);

    // Allocate output channel and copy data back from device
    outputChannel = Mat(rows, cols, CV_8UC1);
    cudaMemcpy(outputChannel.data, d_output, imageSize, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}



// Function to apply Laplacian filter on an image channel
void apply_laplacian_single(const Mat& inputChannel, Mat& outputChannel, int width, int height, double alpha) {
    const double laplacianKernel[3][3] = {
        { 0, -1,  0 },
        {-1,  4, -1 },
        { 0, -1,  0 }
    };

    // Iterate through each pixel
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            double result = 0.0;

            // Apply Laplacian kernel
            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    int ix = min(max(x + kx, 0), width - 1);
                    int iy = min(max(y + ky, 0), height - 1);
                    result += inputChannel.at<unsigned char>(iy, ix) * laplacianKernel[ky + 1][kx + 1];
                }
            }

            // Compute new pixel value after applying the Laplacian filter and sharpening factor
            double newVal = inputChannel.at<unsigned char>(y, x) + alpha * result;
            outputChannel.at<unsigned char>(y, x) = saturate_cast<uchar>(newVal);  // Clamp to [0, 255]
        }
    }
}

void applyLaplacianSharpen_single(const Mat& inputChannel, Mat& outputChannel, double sharpeningFactor) {
    int rows = inputChannel.rows;
    int cols = inputChannel.cols;

    // Initialize the output channel
    outputChannel = Mat(rows, cols, CV_8UC1);

    // Apply the Laplacian filter to the input image
    apply_laplacian_single(inputChannel, outputChannel, cols, rows, sharpeningFactor);
}


void applyLaplacianToColorImage(Mat& inputImage, Mat& outputImage, double sharpeningFactor) {
    Mat preProcessed;
    vector<Mat> channels(3);

    split(inputImage, channels);

    vector<Mat> outputChannels(3);
    for (int i = 0; i < 3; ++i) {
        image_preprocessing(channels[i], preProcessed);
        applyLaplacianSharpen(channels[i], outputChannels[i], sharpeningFactor);
    }

    // Merge the processed channels back into a color image
    merge(outputChannels, outputImage);
}

void applyLaplacianToColorImage_single(Mat& inputImage, Mat& outputImage, double sharpeningFactor) {
    Mat preProcessed;
    vector<Mat> channels(3);

    split(inputImage, channels);

    vector<Mat> outputChannels(3);
    for (int i = 0; i < 3; ++i) {
        image_preprocessing(channels[i], preProcessed);
        applyLaplacianSharpen_single(channels[i], outputChannels[i], sharpeningFactor);
    }

    // Merge the processed channels back into a color image
    merge(outputChannels, outputImage);
}

Mat single_Laplacian_Filter(Mat& image) {

    //image = loadImage("C:/Users/wongc/source/repos/DSCP/DSCP/Image/IC.png");

    Mat sharpenedColor;
    // Apply the Laplacian filter
    applyLaplacianToColorImage_single(image, sharpenedColor, 1);

    //saveImage("C:/Users/wongc/source/repos/DSCP/DSCP/Image/Sharpened_Single.png", sharpenedColor);

    return sharpenedColor;

}


// Function to apply Laplacian filter on a color image using CUDA
Mat cuda_Laplacian_Filter(Mat& image) {
    if (image.empty()) {
        cout << "Error: Image not found!" << endl;
        return Mat();
    }

    if (image.channels() != 3) {
        cout << "Error: Input image does not have 3 channels!" << endl;
        return Mat();
    }

    // Prepare sharpened color image
    Mat sharpenedColor;
    double sharpeningFactor = 1.0;  // Adjust sharpening intensity

    // Apply Laplacian filter on the entire color image using CUDA
    applyLaplacianToColorImage(image, sharpenedColor, sharpeningFactor);

    return sharpenedColor;  // Return the sharpened image
}

int main() {
    Mat processed_image, processed_image2;

    string imagePath = "";
    cout << "Image Path: ";
    getline(cin, imagePath);

    int64 startTime, endTime;
    double timeTaken;

    startTime = getTickCount();
    Mat image = loadImage(imagePath);
    timeTaken = measureTime(startTime);
    cout << "Time taken to load image: " << timeTaken << " seconds" << endl;

    if (image.empty()) {
        cout << "Could not load the image!" << endl;
        return -1;
    }


    // Display the original image
    namedWindow("Original Image", WINDOW_AUTOSIZE);
    imshow("Original Image", image);
    moveWindow("Original Image", 0, 0);


    //processed_image
    startTime = getTickCount();
    processed_image = single_Laplacian_Filter(image);
    timeTaken = measureTime(startTime);
    cout << "Time taken to apply single Laplacian filter: " << timeTaken << " seconds" << endl;


    // Display the sharpened image
    namedWindow("single processed Image", WINDOW_AUTOSIZE);
    imshow("single processed Image", processed_image);
    moveWindow("single processed Image", 500, 0);

    // Save the sharpened image
    imwrite("single_processed_" + imagePath + filePath, processed_image);


    startTime = getTickCount();
    processed_image2 = cuda_Laplacian_Filter(image);
    timeTaken = measureTime(startTime);
    cout << "Time taken to apply CUDA Laplacian filter: " << timeTaken << " seconds" << endl;


    // Display the sharpened image
    namedWindow("Cuda processed Image2", WINDOW_AUTOSIZE);
    imshow("Cuda processed Image2", processed_image2);
    moveWindow("Cuda processed Image2", 1000, 0);

    // Save the sharpened image
    imwrite("Cuda processed2_" + imagePath + filePath, processed_image2);


    waitKey(0); // Wait for a key press
    return 0;
}
*/

