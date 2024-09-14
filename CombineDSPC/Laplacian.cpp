#include <mpi.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <omp.h>

using namespace cv;
using namespace std;



void image_preprocessing(Mat& image, Mat& preprocessed_image);
void saveImage(const string& filename, const Mat& image);
Mat loadImage(const string& filename);



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


void applyLaplacianSharpen(const Mat& inputChannel, Mat& outputChannel, double sharpeningFactor) {
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









//Single Thread for Laplacian Algorithrm
Mat single_Laplacian_Filter(Mat& image) {

    //image = loadImage("C:/Users/wongc/source/repos/DSCP/DSCP/Image/IC.png");




    Mat sharpenedColor;
    // Apply the Laplacian filter
    applyLaplacianToColorImage(image, sharpenedColor, 1.0);

    saveImage("C:/Users/wongc/source/repos/DSCP/DSCP/Image/Sharpened_Single.png", sharpenedColor);

    return sharpenedColor;

}




//MPI Parallel Computing for Laplacian Algorithrm
void mpi_Laplacian_Filter(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_size, world_rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int rows, cols, channels;
    int blockSize;
    vector<uchar> imageData;
    Mat image;
    double start_time = 0, end_time = 0, elapsed_time = 0;

    if (world_rank == 0) {
        // Load the image as a color image
        string imagePath;
        ///cin.ignore();
        cout << "Image Path: ";
        getline(cin, imagePath);

        image = loadImage(imagePath);
        if (image.empty()) {
            cerr << "Error: Image not found!" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Ensure the image is in 3 channels (BGR)
        if (image.channels() != 3) {
            cout << "Error: Input image does not have 3 channels!" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        start_time = MPI_Wtime();

        rows = image.rows;
        cols = image.cols;
        channels = image.channels();
        blockSize = rows / world_size;

        // Flatten image data for distribution
        imageData.resize(rows * cols * channels);
        memcpy(imageData.data(), image.data, imageData.size());
    }

    // Broadcast image dimensions and block size
    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&channels, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&blockSize, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Scatter image data to all processes
    vector<uchar> localData(blockSize * cols * channels);
    MPI_Scatter(imageData.data(), blockSize * cols * channels, MPI_UNSIGNED_CHAR,
        localData.data(), blockSize * cols * channels, MPI_UNSIGNED_CHAR,
        0, MPI_COMM_WORLD);


    Mat localBlock(blockSize, cols, CV_8UC3, localData.data());


    Mat localOutputBlock;
    double sharpeningFactor = 1;  // Adjust sharpening intensity (0.3 to 0.7 for subtle or stronger effects)
    applyLaplacianToColorImage(localBlock, localOutputBlock, sharpeningFactor);

    // Gather processed blocks
    vector<uchar> resultData(rows * cols * channels);
    MPI_Gather(localOutputBlock.data, blockSize * cols * channels, MPI_UNSIGNED_CHAR,
        resultData.data(), blockSize * cols * channels, MPI_UNSIGNED_CHAR,
        0, MPI_COMM_WORLD);
    Mat resultImage;

    if (world_rank == 0) {


        end_time = MPI_Wtime(); // End the timer
        elapsed_time = end_time - start_time;




        // Reconstruct the image from result data
        resultImage = Mat(rows, cols, CV_8UC3, resultData.data());

        // Save the final sharpened image
        saveImage("C:/Users/wongc/source/repos/DSCP/DSCP/Image/Sharpened_Laplacian.png", resultImage);

        namedWindow("Original Image", WINDOW_AUTOSIZE);
        imshow("Original Image", image);
        moveWindow("Original Image", 0, 0);

        if (!resultImage.empty()) {
            namedWindow("Processed Image", WINDOW_AUTOSIZE);
            imshow("Processed Image", resultImage);
            moveWindow("Processed Image", 500, 500);
        }
        else {
            cout << "Could not load the processed image!" << endl;
        }
        waitKey(0); // Wait for a key press
        destroyAllWindows();

        cout << "Elapsed time: " << elapsed_time << " seconds" << endl;

    }

    MPI_Finalize();
    //return resultImage;
}