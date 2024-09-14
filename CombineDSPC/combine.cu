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
    Mat processed_image;

    int choice;
    int method;
    bool flag = false;

    int64 startTime, endTime;
    double timeTaken;

    do {
        cout << "1:Laplacian Filter 2:Unsharp Masking 3:Display Chart\nWhat algorithm is used:";
        while (!(cin >> choice) && choice <= 3) {  // Check if the input is of the correct type (int in this case)
            cin.clear();  // Clear the error flag on cin
            cin.ignore(numeric_limits<streamsize>::max(), '\n');  // Discard invalid input
            cout << "Invalid input. Please enter an integer: ";
        }



        if (choice != 3) {
            cout << "1:OMP 2:CUDA 3:MPI 4:Single Thread\nWhat parallel platform you want to use: ";
            while (!(cin >> method)) {  // Check if the input is of the correct type (int in this case)
                cin.clear();  // Clear the error flag on cin
                cin.ignore(numeric_limits<streamsize>::max(), '\n');  // Discard invalid input
                cout << "Invalid input. Please enter an integer: ";
            }
        }

        string imagePath;


        Mat image;

        if (choice != 3) {

            cin.ignore();  // Ignore leftover newline character
            cout << "Image Path: ";
            getline(cin, imagePath);

            image = loadImage(imagePath);
            if (image.empty()) {
                cout << "Could not load the image!" << endl;
                return -1;
            }
        }





        //auto start = high_resolution_clock::now();

        switch (choice) {
        case 1: //1:Laplacian Filter
            if (method == 1) {          //1:OMP
            }

            else if (method == 2) {     //2:CUDA
                startTime = getTickCount();
                processed_image = cuda_Laplacian_Filter(image);
                timeTaken = measureTime(startTime);
                cout << "Time taken to apply cuda Laplacian filter: " << timeTaken << " seconds" << endl;
            }

            else if (method == 3) {     //3:MPI
                //processed_image = mpi_unsharp_masking(argc, argv);
                cout << "Please Run in PowerShell";
            }

            else if (method == 4) {     //4:Single Thread
                startTime = getTickCount();
                processed_image = single_Laplacian_Filter(image);
                timeTaken = measureTime(startTime);
                cout << "Time taken to apply Single thread Laplacian filter: " << timeTaken << " seconds" << endl;
                 }


            break;
        case 2:
            if (method == 1) {
                startTime = getTickCount();
                processed_image = omp_unsharp_masking(image);
                timeTaken = measureTime(startTime);
                cout << "Time taken to apply omp Unsharp Masking: " << timeTaken << " seconds" << endl;
            }
            else if (method == 2) {
                startTime = getTickCount();
                processed_image = cuda_unsharp_masking(image);
                timeTaken = measureTime(startTime);
                cout << "Time taken to apply cuda Unsharp Masking: " << timeTaken << " seconds" << endl;
            }
            else if (method == 3) {
                //processed_image = mpi_Laplacian_Filter(argc, argv);
                cout << "Please Run in PowerShell";
            }
            else if (method == 4) {
                startTime = getTickCount();
                processed_image = single_thread_unsharp_masking(image);
                timeTaken = measureTime(startTime);
                cout << "Time taken to apply single thread Unsharp Masking: " << timeTaken << " seconds" << endl;
                // Display the sharpened image

            }

            break;
        case 3:

            double mpi_e, omp_e, cuda_e, single_e;
            vector<double> executionTimes;
            cout << "Enter Execution time for OMP :";
            while (!(cin >> omp_e)) {
                cin.clear();
                cin.ignore(numeric_limits<streamsize>::max(), '\n');
                cout << "Invalid input. Please enter an double: ";
            }



            cout << "Enter Execution time for CUDa: ";
            while (!(cin >> cuda_e)) {
                cin.clear();
                cin.ignore(numeric_limits<streamsize>::max(), '\n');
                cout << "Invalid input. Please enter an double: ";
            }

            cout << "Enter Execution time for MPI: ";
            while (!(cin >> mpi_e)) {
                cin.clear();
                cin.ignore(numeric_limits<streamsize>::max(), '\n');
                cout << "Invalid input. Please enter an double: ";
            }

            cout << "Enter Execution time for Single Thread: ";
            while (!(cin >> single_e)) {
                cin.clear();
                cin.ignore(numeric_limits<streamsize>::max(), '\n');
                cout << "Invalid input. Please enter an double: ";
            }
            executionTimes.push_back(omp_e);
            executionTimes.push_back(cuda_e);
            executionTimes.push_back(mpi_e);
            executionTimes.push_back(single_e);



            //BarChart(executionTimes);
            break;
        }
        //auto stop = high_resolution_clock::now();

        //auto duration = duration_cast<milliseconds>(stop - start);

        if (choice != 3) {
            // Display the original image
            namedWindow("Original Image", WINDOW_AUTOSIZE);
            imshow("Original Image", image);
            moveWindow("Original Image", 0, 0);
        }



        // Load and display the processed image

        if (!processed_image.empty() && choice != 3) {
            namedWindow("Processed Image", WINDOW_AUTOSIZE);
            imshow("Processed Image", processed_image);
            moveWindow("Processed Image", 500, 0);
            waitKey(0); // Wait for a key press
            destroyAllWindows();
        }
        else {
            cout << "Could not load the processed image!" << endl;
            waitKey(0); // Wait for a key press
            destroyAllWindows();
        }

        /*
        if (choice != 3) {
            cout << "Time taken for to precess: " << duration.count() / 1000 << "seconds" << endl;

        }
        */

        char ans;
        cout << "\nWant to continue? (y/n)" << endl;
        cin >> ans;
        ans = tolower(ans);

        // Check if the input is valid (either 'y' or 'n')
        if (ans == 'y') {
            flag = true;
            cout << "Continue...\n" << endl;
        }
        else {
            flag = false;
            cout << "Finished...\n" << endl;
        }
        //cout << flag << endl;


    } while (flag);

    return 0;
}

// mpiexec -n 4 combineDSPC.exe
