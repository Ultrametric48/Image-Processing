//@Author Dominic Gastaldo
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <cstdint>
#include "../Eigen/Dense"
#include "../Eigen/StdVector"




using Eigen::Matrix; 
using namespace Eigen;
typedef Matrix<int, Dynamic, Dynamic> intMatrix; //define 8-bit matrix
typedef Matrix<int8_t, Dynamic, Dynamic> Matrix8b;
typedef Matrix<int, Dynamic, 1> Vector8b; //define 8-bit vector of length N*P


//function prototypes
intMatrix conv2d(intMatrix input, intMatrix kernel);
intMatrix conv2d_toeplitz(intMatrix input, intMatrix kernel);
intMatrix multiply8b(intMatrix x, intMatrix y);
intMatrix conv_filter(std::vector<Matrix<int,Dynamic,Dynamic> > inputs, std::vector<Matrix<int,Dynamic,Dynamic> > filters);


intMatrix multiply8b(Matrix8b x, Matrix8b y){
    //model of an 8b multiplier
    return x*y;
    
}




intMatrix conv2d(intMatrix input, intMatrix kernel){
    /*
     * 2D image convolution with the Eigen matrix template
     */
    
    //Start convolution algorithm
    
    /*
     * Check for kernel matrix seprability
     * The kernel is seprable if it can be written as an outer product of two vectors, and
     * a seprable kernel can be computed faster then its non-seprable counterpart.
     * 
     * 2D Convolution is seprable
     */
    
    //Define pivot elements that map convolution element to image element
    int const p = 0;
    int const e = 0;
    
    int resultant_sum = 0;
    
    
    intMatrix conv2d = intMatrix::Zero(input.rows(), input.cols()); //resultant convolution matrix
    
    //resultant matrix loops
    for(int conv2d_row=0; conv2d_row < input.rows(); conv2d_row++){ 
        for(int conv2d_col=0; conv2d_col < input.cols(); conv2d_col++){ 
            
            //Image/kernel loops
            resultant_sum = 0;
            for(int i=0; i < kernel.rows(); i++){      
                for(int j=0; j < kernel.cols(); j++){
                       
                    
                    /*
                     * Boundary conditions:
                     * Kernel operations must be handled at the boundary, where the kernel can be disregarded.
                     */
                    
                    if((conv2d_row + i - p) < 0 ||  (conv2d_row + i - p) >= input.rows() || (conv2d_col + j - e) < 0 ||  (conv2d_col + j - e) >= input.cols()){
                        //Don't add values that are out of bounds to the sum
                        printf("Skipping (%d,%d)\n", (conv2d_row + i - p), (conv2d_col + j - e));
                        continue;
                    }
                    
                    
                    //compute each entry of resultant matrix
                        
                    std::cout << "(i,j)=" << "(" << conv2d_row + i - p << ", " << conv2d_col + j - e << ")" << std::endl;
                    std::cout << "Input matrix element " << "(" << conv2d_row + i - p << ", " << conv2d_col + j - e << ") "  << input(conv2d_row + i - p, conv2d_col + j - e) << std::endl;
                    std::cout << "Kernel matrix element" << "(" << i << "," << j << ") " << kernel(i,j) << std::endl;
                        
                    resultant_sum += input(conv2d_row + i - p, conv2d_col + j - e) * // Input matrix is evaluated by aligning with the pivot element Image(i,j) -> kernel(p,e)
                    kernel(std::abs(i - kernel.rows() + 1), std::abs(j - kernel.cols() + 1)); //Transform the evaluation of the kernel so that the kernel matrix is flipped along the vertical and horizontal 
                
  
                    
                }
            }
            // sum i,j elements, then assign to resultant matrix
            conv2d(conv2d_row, conv2d_col) = resultant_sum;

        }
    }
    
    
  return conv2d;
    
}




intMatrix conv2d_toeplitz(intMatrix input, intMatrix kernel){
    
    
    //Construct input vector
    //Construct Toeplitz matrix
    //Multiply for convolution
    
    //define column and row to save ops
    int zero_pad_row_size = (input.rows() + kernel.rows() - 1);
    int zero_pad_col_size = (input.cols() + kernel.cols() - 1);
    intMatrix zero_pad_imageMatrix;
    intMatrix mat_mult_row;
    intMatrix resultant_convoluton_vector;
    
    
    
    
    std::cout << "kernel:" << std::endl << kernel << std::endl;
    //flip kernel horizontally and vertically
    
    intMatrix tmp1; //Need tmp to avoid Eigen compile time reference errors
    intMatrix tmp2; //Need tmp to avoid Eigen compile time reference errors
    tmp1 = kernel.colwise().reverse();
    tmp2 = tmp1.rowwise().reverse();
    kernel = tmp2.transpose();
     

    
    
    Map<RowVectorXi> kernel_unroll(kernel.data(), kernel.size()); //unroll kernel matrix 
    std::cout << "kernel_unroll:" << std::endl << kernel_unroll << std::endl;
 
    
    
    //zero pad image matrix
    zero_pad_imageMatrix = intMatrix::Zero(zero_pad_row_size, zero_pad_col_size);
    zero_pad_imageMatrix.block(0, 0, input.rows(), input.cols()) = input; //zero pad image matrix
    
    std::cout << "zero pad image:" << std::endl << zero_pad_imageMatrix << std::endl;
    
    intMatrix mult_matrix = intMatrix::Zero(input.size(), kernel.size()); //check for non-square kernel matricies
    
    
    std::cout << "mult matrix:" << std::endl << mult_matrix << std::endl;
    
    

    //construct image matrix
    mat_mult_row = intMatrix::Zero(1, kernel.size());
    std::cout << "mat_mult_row:" << std::endl << mat_mult_row << std::endl;
    int i=0;
    int j=0;
    
    for(int c=0; c < input.rows(); c++){
        for(int k=0; k < input.cols(); k++){
            mat_mult_row = intMatrix::Zero(1, kernel.size());
            j=0;
            for(int r = 0; r < kernel.rows(); r++){
                for(int w=0; w < kernel.cols(); w++){
                
                    mat_mult_row(0, j++) = zero_pad_imageMatrix(c+r, k+w);
                
                }
            }
            mult_matrix.block(i++,0,1, kernel.size()) = mat_mult_row;
        }
    }
    
    std::cout << "mat_mult_matrix_final:" << std::endl << mult_matrix << std::endl;
    std::cout << "kernel final:" << std::endl << kernel_unroll << std::endl;
    
    
    //***************Insert 8b matrix vector multiplier here***************
    resultant_convoluton_vector = multiply8b(mult_matrix, kernel_unroll.transpose()); //8b convolution by matrix multiplication
    //***************Insert 8b matrix vector multiplier here***************
    
    std::cout << "resultant_convoluton_vector" << std::endl << resultant_convoluton_vector << std::endl;
    
    
    //resize resultant vector to convolution matrix
    
    Map<intMatrix> twoDconv(resultant_convoluton_vector.data(), input.rows(), input.cols());
    

    return twoDconv.transpose();
    
}


intMatrix conv_filter(std::vector<Matrix<int,Dynamic,Dynamic>, Eigen::aligned_allocator<Matrix<int,Dynamic,Dynamic> > > inputs, std::vector<Matrix<int,Dynamic,Dynamic>, Eigen::aligned_allocator<Matrix<int,Dynamic,Dynamic> > > filters){
    
    //This function handles a vector of images (e.g. 3-channel RGB images). The three channels would be stored in the "inputs" vector).
    //Each channel should have a kernel associated with it. Each kernel should be in the "kernels" vector
    //convolve each input with each kernel
    //add each convolved image to synthesize the resultant image
    
    std::vector<Matrix<int,Dynamic,Dynamic>, Eigen::aligned_allocator<Matrix<int,Dynamic,Dynamic> > > convolved_inputs;
    intMatrix resultant_image = intMatrix::Zero(inputs.at(0).rows(), inputs.at(0).cols());
  
    
    //sum the results for the final output
    for(int i=0; i < inputs.size(); i++) {
        
        resultant_image += conv2d_toeplitz(inputs.at(i), filters.at(i));
        
    }
    
    
    return resultant_image;
} 


    

int main(int argc, char **argv){
    
    /*
     * user must define matrices
     * 
     * This program will perform 2d image convolution on a set of user defined matrices (e.g. RGB 3-channel images with three associated kernels).
     * 
     * 
     * */
    
    //*****************DEFINE MATRICES*****************
    
    //This part of the code may be verbose and messy, as the matrices must be defined element-wise
    //This will certainly need to be replaced with proper imageI/O, but it will have to do for a two week project. 
    //TODO develop imageI/O that can read images, separate them into 8b pixel matrices by channel.
    
    
    
    /*
     *Expect an
     * 
     *Assertion failed: (row >= 0 && row < rows() && col >= 0 && col < cols()), function operator(), file ./host/../Eigen/src/Core/DenseCoeffsBase.h, line 367.
    make: *** [conv] Abort trap: 6
     * 
     * error if the elements are not indexed right manually e.g.
     * intMatrix image = intMatrix::Identity(3, 3);
     * image(4,0) = 1;
     * 
     * Double check indices
     */
    
    
    
    int8_t eight_bit = -125;
    
    
   
    std::cout << "unsigned 8bit int\n" << (int)eight_bit  << std::endl;
    
    
    
}
