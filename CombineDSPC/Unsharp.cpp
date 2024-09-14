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

void unsharpMasking(const Mat& originalImageBlock, const Mat& preprocessedBlock, Mat& outputBlock) {
    Mat blurred, mask, sharpened;

    GaussianBlur(preprocessedBlock, blurred, Size(5, 5), 1.0);

    //addWeighted(originalImageBlock, 1.5, blurred, -0.5, 0, sharpened);
    subtract(originalImageBlock, blurred, mask);

    add(originalImageBlock, mask * 3.0, sharpened);
    outputBlock = sharpened;
}

Mat single_thread_unsharp_masking(Mat& image) {




    Mat preProcessed;

    image_preprocessing(image, preProcessed);


    Mat sharpenedImage;
    unsharpMasking(image, preProcessed, sharpenedImage);


    // Save the result
    saveImage("C:/Users/wongc/source/repos/DSCP/DSCP/Image/Sharpened_Single.png", sharpenedImage);

    return sharpenedImage;
}


void mpi_unsharp_masking(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    Mat preprocessedImage;
    int rows, cols, channels;
    int blockSize;
    vector<uchar> imageData, preprocessedData;
    Mat image;
    double start_time = 0, end_time = 0, elapsed_time = 0;

    if (world_rank == 0) {

        string imagePath;
        //cin.ignore();
        cout << "Image Path: ";
        getline(cin, imagePath);

        image = loadImage(imagePath);
        if (image.empty()) {
            cerr << "Error: Image not found!" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        // Load the image
        //image = loadImage("C:/Users/wongc/source/repos/DSCP/DSCP/Image/IC.png");
        if (image.empty()) {
            cerr << "Error: Image not found!" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        start_time = MPI_Wtime();

        // Preprocess the image
        image_preprocessing(image, preprocessedImage);

        rows = image.rows;
        cols = image.cols;
        channels = image.channels();
        blockSize = rows / world_size;


        // Flatten image data for distribution
        imageData.resize(rows * cols * channels);
        memcpy(imageData.data(), image.data, imageData.size());

        // Flatten preprocessed image data
        preprocessedData.resize(rows * cols * channels);
        memcpy(preprocessedData.data(), preprocessedImage.data, preprocessedData.size());
    }

    // Broadcast image dimensions and block size
    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&channels, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&blockSize, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Scatter image data and preprocessed image data to all processes
    vector<uchar> localData(blockSize * cols * channels);
    vector<uchar> localPreprocessedData(blockSize * cols * channels);
    MPI_Scatter(imageData.data(), blockSize * cols * channels, MPI_UNSIGNED_CHAR,
        localData.data(), blockSize * cols * channels, MPI_UNSIGNED_CHAR,
        0, MPI_COMM_WORLD);
    MPI_Scatter(preprocessedData.data(), blockSize * cols * channels, MPI_UNSIGNED_CHAR,
        localPreprocessedData.data(), blockSize * cols * channels, MPI_UNSIGNED_CHAR,
        0, MPI_COMM_WORLD);

    // Create local image block and preprocessed block
    Mat localBlock(blockSize, cols, CV_8UC3, localData.data());
    Mat localPreprocessedBlock(blockSize, cols, CV_8UC3, localPreprocessedData.data());

    // Allocate space for output block
    Mat localOutputBlock(blockSize, cols, CV_8UC3);

    // Apply unsharp masking on the original and preprocessed image blocks
    unsharpMasking(localBlock, localPreprocessedBlock, localOutputBlock);

    // Gather processed blocks
    vector<uchar> resultData(rows * cols * channels);
    MPI_Gather(localOutputBlock.data, blockSize * cols * channels, MPI_UNSIGNED_CHAR,
        resultData.data(), blockSize * cols * channels, MPI_UNSIGNED_CHAR,
        0, MPI_COMM_WORLD);

    Mat resultImage;

    if (world_rank == 0) {
        end_time = MPI_Wtime(); // End the timer
        elapsed_time = end_time - start_time;


        resultImage = Mat(rows, cols, CV_8UC3, resultData.data());
        saveImage("C:/Users/wongc/source/repos/DSCP/DSCP/Image/Sharpened_Unsharp_Masking.png", resultImage);

        namedWindow("Original Image", WINDOW_AUTOSIZE);
        imshow("Original Image", image);
        moveWindow("Original Image", 0, 0);

        // Load and display the processed image
        if (!resultImage.empty()) {
            namedWindow("Processed Image", WINDOW_AUTOSIZE);
            imshow("Processed Image", resultImage);
            moveWindow("Processed Image", 500, 0);
        }
        else {
            cout << "Could not load the processed image!" << endl;
        }

        waitKey(0); // Wait for a key press
        destroyAllWindows();

        cout << "Elapsed time: " << elapsed_time << " seconds" << endl;


    }

    MPI_Finalize();
    // resultImage;


}

Mat omp_unsharp_masking(Mat& image) {

    Mat preProcessed;

    // Preprocess the image using the existing function
    image_preprocessing(image, preProcessed);

    Mat sharpenedImage = image.clone();
    int rows = image.rows;
    int cols = image.cols;
    int block_size = 64; // Block size can be tuned for performance

    // Set the number of threads for parallel processing
    int num_threads = omp_get_max_threads();
    omp_set_num_threads(num_threads);

#pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i = 0; i < rows; i += block_size) {
        for (int j = 0; j < cols; j += block_size) {
            // Define the size of the block
            int block_width = min(block_size, cols - j);
            int block_height = min(block_size, rows - i);

            // Extract the image blocks
            Mat originalBlock = image(Rect(j, i, block_width, block_height));
            Mat preProcessedBlock = preProcessed(Rect(j, i, block_width, block_height));
            Mat outputBlock;

            // Call the existing unsharpMasking function on each block
            unsharpMasking(originalBlock, preProcessedBlock, outputBlock);

#pragma omp critical
            {
                // Copy the sharpened block back to the sharpened image
                outputBlock.copyTo(sharpenedImage(Rect(j, i, block_width, block_height)));
            }
        }
    }

    // Save the sharpened image result
    saveImage("C:/Users/wongc/source/repos/DSCP/DSCP/Image/Sharpened_Unsharp_Masking.png", sharpenedImage);

    return sharpenedImage;
}




