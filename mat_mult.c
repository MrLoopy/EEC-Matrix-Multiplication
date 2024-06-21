
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <pthread.h>
#include <time.h>

#include <arm_neon.h>

#include "armpmu_lib_pmu.h"
#define N 1024 // dont change, or most functions wont work anymore

void kernel_8x32_save(float32_t *A, float32_t *B, float32_t *C){
    // matrix A needs to be transposed
    asm volatile(
        // load old value of C to registers
        "ldp	q16, q24, [x2]\n"
        "ldp	q17, q25, [x2, #32]\n"
        "ldp	q18, q26, [x2, #64]\n"
        "ldp	q19, q27, [x2, #96]\n"
        "ldp	q20, q28, [x2, #128]\n"
        "ldp	q21, q29, [x2, #160]\n"
        "ldp	q22, q30, [x2, #192]\n"
        "ldp	q23, q31, [x2, #224]\n"

        // load the first registers
        "ldp	q0, q1, [x0], #32\n"
        "ldp	q4, q5, [x1], #32\n"

        // start the loop
        "mov	x4, #15\n"
    "loop_kernel_8x32_save_%=:"
        // load the second registers
        "ldp	q2, q3, [x0], #32\n"
        "ldp	q6, q7, [x1], #32\n"
        // while second registers are loading, compute C with the first registers
        "fmla	v16.4s, v4.4s, v0.s[0]\n"
        "fmla	v17.4s, v4.4s, v0.s[1]\n"
        "fmla	v18.4s, v4.4s, v0.s[2]\n"
        "fmla	v19.4s, v4.4s, v0.s[3]\n"
        "fmla	v20.4s, v4.4s, v1.s[0]\n"
        "fmla	v21.4s, v4.4s, v1.s[1]\n"
        "fmla	v22.4s, v4.4s, v1.s[2]\n"
        "fmla	v23.4s, v4.4s, v1.s[3]\n"
        "fmla	v24.4s, v5.4s, v0.s[0]\n"
        "fmla	v25.4s, v5.4s, v0.s[1]\n"
        "fmla	v26.4s, v5.4s, v0.s[2]\n"
        "fmla	v27.4s, v5.4s, v0.s[3]\n"
        "fmla	v28.4s, v5.4s, v1.s[0]\n"
        "fmla	v29.4s, v5.4s, v1.s[1]\n"
        "fmla	v30.4s, v5.4s, v1.s[2]\n"
        "fmla	v31.4s, v5.4s, v1.s[3]\n"
        // reload the first registers
        "ldp	q0, q1, [x0], #32\n"
        "ldp	q4, q5, [x1], #32\n"
        // while first registers are loading, compute C with the second registers
        "fmla	v16.4s, v6.4s, v2.s[0]\n"
        "fmla	v17.4s, v6.4s, v2.s[1]\n"
        "fmla	v18.4s, v6.4s, v2.s[2]\n"
        "fmla	v19.4s, v6.4s, v2.s[3]\n"
        "fmla	v20.4s, v6.4s, v3.s[0]\n"
        "fmla	v21.4s, v6.4s, v3.s[1]\n"
        "fmla	v22.4s, v6.4s, v3.s[2]\n"
        "fmla	v23.4s, v6.4s, v3.s[3]\n"
        "fmla	v24.4s, v7.4s, v2.s[0]\n"
        "fmla	v25.4s, v7.4s, v2.s[1]\n"
        "fmla	v26.4s, v7.4s, v2.s[2]\n"
        "fmla	v27.4s, v7.4s, v2.s[3]\n"
        "fmla	v28.4s, v7.4s, v3.s[0]\n"
        "fmla	v29.4s, v7.4s, v3.s[1]\n"
        "fmla	v30.4s, v7.4s, v3.s[2]\n"
        "fmla	v31.4s, v7.4s, v3.s[3]\n"
        // restart loop if x4 has not reached 0 yet
        "sub	x4, x4, #1\n"
        "cbnz	x4, loop_kernel_8x32_save_%=\n"

        // last iteration after the loop to avoid access of registers out of bounds
        // load the second registers
        "ldp	q2, q3, [x0], #32\n"
        "ldp	q6, q7, [x1], #32\n"
        // while second registers are loading, compute C with the first registers
        "fmla	v16.4s, v4.4s, v0.s[0]\n"
        "fmla	v17.4s, v4.4s, v0.s[1]\n"
        "fmla	v18.4s, v4.4s, v0.s[2]\n"
        "fmla	v19.4s, v4.4s, v0.s[3]\n"
        "fmla	v20.4s, v4.4s, v1.s[0]\n"
        "fmla	v21.4s, v4.4s, v1.s[1]\n"
        "fmla	v22.4s, v4.4s, v1.s[2]\n"
        "fmla	v23.4s, v4.4s, v1.s[3]\n"
        "fmla	v24.4s, v5.4s, v0.s[0]\n"
        "fmla	v25.4s, v5.4s, v0.s[1]\n"
        "fmla	v26.4s, v5.4s, v0.s[2]\n"
        "fmla	v27.4s, v5.4s, v0.s[3]\n"
        "fmla	v28.4s, v5.4s, v1.s[0]\n"
        "fmla	v29.4s, v5.4s, v1.s[1]\n"
        "fmla	v30.4s, v5.4s, v1.s[2]\n"
        "fmla	v31.4s, v5.4s, v1.s[3]\n"
        // no reload of the first registers
        "fmla	v16.4s, v6.4s, v2.s[0]\n"
        "fmla	v17.4s, v6.4s, v2.s[1]\n"
        "fmla	v18.4s, v6.4s, v2.s[2]\n"
        "fmla	v19.4s, v6.4s, v2.s[3]\n"
        "fmla	v20.4s, v6.4s, v3.s[0]\n"
        "fmla	v21.4s, v6.4s, v3.s[1]\n"
        "fmla	v22.4s, v6.4s, v3.s[2]\n"
        "fmla	v23.4s, v6.4s, v3.s[3]\n"
        "fmla	v24.4s, v7.4s, v2.s[0]\n"
        "fmla	v25.4s, v7.4s, v2.s[1]\n"
        "fmla	v26.4s, v7.4s, v2.s[2]\n"
        "fmla	v27.4s, v7.4s, v2.s[3]\n"
        "fmla	v28.4s, v7.4s, v3.s[0]\n"
        "fmla	v29.4s, v7.4s, v3.s[1]\n"
        "fmla	v30.4s, v7.4s, v3.s[2]\n"
        "fmla	v31.4s, v7.4s, v3.s[3]\n"

        // store the results back to C
        "stp	q16, q24, [x2]\n"
        "stp	q17, q25, [x2, #32]\n"
        "stp	q18, q26, [x2, #64]\n"
        "stp	q19, q27, [x2, #96]\n"
        "stp	q20, q28, [x2, #128]\n"
        "stp	q21, q29, [x2, #160]\n"
        "stp	q22, q30, [x2, #192]\n"
        "stp	q23, q31, [x2, #224]\n"
        // end the kernel, return
        "ret\n"
        "nop\n"
		:
		: [x0] "r" (A), [x1] "r" (B), [x2] "r" (C)
		: "x4", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q16", "q17", "q18", "q19", "q20", "q21", "q22", "q23", "q24", "q25", "q26", "q27", "q28", "q29", "q30", "q31"
    );
}
void scale_kernel_8x32_block(float32_t *A, float32_t *B, float32_t *C){
    // 1024 / 8 = 128
    // 128x128 kernel-tiles in C with 4 kernel execution each
    for (int i = 0; i < 128; i++){
        for (int j = 0; j < 128; j++){
            for (int k = 0; k < 32; k++) {
		        kernel_8x32_save(&A[8192 * i + 256 * k], &B[32768 * k + 256 * j], &C[8192 * i + 64 * j]);
                if( i==0 && j==0 && k==0){
                    printf("%d\n",k);
		        }
            }
        }
    }
}

void kernel_8x8_save(float32_t *A, float32_t *B, float32_t *C){
    // matrix A needs to be transposed
    asm volatile(
        // load old value of C to registers
        "ldp	q16, q24, [x2]\n"
        "ldp	q17, q25, [x2, #32]\n"
        "ldp	q18, q26, [x2, #64]\n"
        "ldp	q19, q27, [x2, #96]\n"
        "ldp	q20, q28, [x2, #128]\n"
        "ldp	q21, q29, [x2, #160]\n"
        "ldp	q22, q30, [x2, #192]\n"
        "ldp	q23, q31, [x2, #224]\n"

        // load the first registers
        "ldp	q0, q1, [x0], #32\n"
        "ldp	q4, q5, [x1], #32\n"

        // start the loop
        "mov	x4, #3\n"
    "loop_kernel_8x8_save_%=:"
        // load the second registers
        "ldp	q2, q3, [x0], #32\n"
        "ldp	q6, q7, [x1], #32\n"
        // while second registers are loading, compute C with the first registers
        "fmla	v16.4s, v4.4s, v0.s[0]\n"
        "fmla	v17.4s, v4.4s, v0.s[1]\n"
        "fmla	v18.4s, v4.4s, v0.s[2]\n"
        "fmla	v19.4s, v4.4s, v0.s[3]\n"
        "fmla	v20.4s, v4.4s, v1.s[0]\n"
        "fmla	v21.4s, v4.4s, v1.s[1]\n"
        "fmla	v22.4s, v4.4s, v1.s[2]\n"
        "fmla	v23.4s, v4.4s, v1.s[3]\n"
        "fmla	v24.4s, v5.4s, v0.s[0]\n"
        "fmla	v25.4s, v5.4s, v0.s[1]\n"
        "fmla	v26.4s, v5.4s, v0.s[2]\n"
        "fmla	v27.4s, v5.4s, v0.s[3]\n"
        "fmla	v28.4s, v5.4s, v1.s[0]\n"
        "fmla	v29.4s, v5.4s, v1.s[1]\n"
        "fmla	v30.4s, v5.4s, v1.s[2]\n"
        "fmla	v31.4s, v5.4s, v1.s[3]\n"
        // reload the first registers
        "ldp	q0, q1, [x0], #32\n"
        "ldp	q4, q5, [x1], #32\n"
        // while first registers are loading, compute C with the second registers
        "fmla	v16.4s, v6.4s, v2.s[0]\n"
        "fmla	v17.4s, v6.4s, v2.s[1]\n"
        "fmla	v18.4s, v6.4s, v2.s[2]\n"
        "fmla	v19.4s, v6.4s, v2.s[3]\n"
        "fmla	v20.4s, v6.4s, v3.s[0]\n"
        "fmla	v21.4s, v6.4s, v3.s[1]\n"
        "fmla	v22.4s, v6.4s, v3.s[2]\n"
        "fmla	v23.4s, v6.4s, v3.s[3]\n"
        "fmla	v24.4s, v7.4s, v2.s[0]\n"
        "fmla	v25.4s, v7.4s, v2.s[1]\n"
        "fmla	v26.4s, v7.4s, v2.s[2]\n"
        "fmla	v27.4s, v7.4s, v2.s[3]\n"
        "fmla	v28.4s, v7.4s, v3.s[0]\n"
        "fmla	v29.4s, v7.4s, v3.s[1]\n"
        "fmla	v30.4s, v7.4s, v3.s[2]\n"
        "fmla	v31.4s, v7.4s, v3.s[3]\n"
        // restart loop if x4 has not reached 0 yet
        "sub	x4, x4, #1\n"
        "cbnz	x4, loop_kernel_8x8_save_%=\n"

        // last iteration, after the loop to avoid loading registers out of bounds
        // load the second registers
        "ldp	q2, q3, [x0], #32\n"
        "ldp	q6, q7, [x1], #32\n"
        // while second registers are loading, compute C with the first registers
        "fmla	v16.4s, v4.4s, v0.s[0]\n"
        "fmla	v17.4s, v4.4s, v0.s[1]\n"
        "fmla	v18.4s, v4.4s, v0.s[2]\n"
        "fmla	v19.4s, v4.4s, v0.s[3]\n"
        "fmla	v20.4s, v4.4s, v1.s[0]\n"
        "fmla	v21.4s, v4.4s, v1.s[1]\n"
        "fmla	v22.4s, v4.4s, v1.s[2]\n"
        "fmla	v23.4s, v4.4s, v1.s[3]\n"
        "fmla	v24.4s, v5.4s, v0.s[0]\n"
        "fmla	v25.4s, v5.4s, v0.s[1]\n"
        "fmla	v26.4s, v5.4s, v0.s[2]\n"
        "fmla	v27.4s, v5.4s, v0.s[3]\n"
        "fmla	v28.4s, v5.4s, v1.s[0]\n"
        "fmla	v29.4s, v5.4s, v1.s[1]\n"
        "fmla	v30.4s, v5.4s, v1.s[2]\n"
        "fmla	v31.4s, v5.4s, v1.s[3]\n"
        // reload of the first registers is not necessary any more
        // while first registers are loading, compute C with the second registers
        "fmla	v16.4s, v6.4s, v2.s[0]\n"
        "fmla	v17.4s, v6.4s, v2.s[1]\n"
        "fmla	v18.4s, v6.4s, v2.s[2]\n"
        "fmla	v19.4s, v6.4s, v2.s[3]\n"
        "fmla	v20.4s, v6.4s, v3.s[0]\n"
        "fmla	v21.4s, v6.4s, v3.s[1]\n"
        "fmla	v22.4s, v6.4s, v3.s[2]\n"
        "fmla	v23.4s, v6.4s, v3.s[3]\n"
        "fmla	v24.4s, v7.4s, v2.s[0]\n"
        "fmla	v25.4s, v7.4s, v2.s[1]\n"
        "fmla	v26.4s, v7.4s, v2.s[2]\n"
        "fmla	v27.4s, v7.4s, v2.s[3]\n"
        "fmla	v28.4s, v7.4s, v3.s[0]\n"
        "fmla	v29.4s, v7.4s, v3.s[1]\n"
        "fmla	v30.4s, v7.4s, v3.s[2]\n"
        "fmla	v31.4s, v7.4s, v3.s[3]\n"

        // store the results back to C
        "stp	q16, q24, [x2]\n"
        "stp	q17, q25, [x2, #32]\n"
        "stp	q18, q26, [x2, #64]\n"
        "stp	q19, q27, [x2, #96]\n"
        "stp	q20, q28, [x2, #128]\n"
        "stp	q21, q29, [x2, #160]\n"
        "stp	q22, q30, [x2, #192]\n"
        "stp	q23, q31, [x2, #224]\n"
        // end the kernel, return
        "ret\n"
        "nop\n"
		:
		: [x0] "r" (A), [x1] "r" (B), [x2] "r" (C)
		: "x4", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q16", "q17", "q18", "q19", "q20", "q21", "q22", "q23", "q24", "q25", "q26", "q27", "q28", "q29", "q30", "q31"
    );
}
void scale_kernel_8x8_block(float32_t *A, float32_t *B, float32_t *C){
    // 1024 / 8 = 128
    // 128x128 kernel-tiles in C with 128 kernel execution each
    for (int i = 0; i < 128; i++){
        for (int j = 0; j < 128; j++){
            for (int k = 0; k < 128; k++) {
//                kernel_8x8_block(A + (8192 * i + 64 * k), B + (8192 * k + 64 * j), C + (8192 * i + 64 * j));
                kernel_8x8_save(&A[8192 * i + 64 * k], &B[8192 * k + 64 * j], &C[8192 * i + 64 * j]);
                if( i==0 && j==0 && k==0)
                    printf("%d", k);
            }
        }
    }
}

void multiply_c(float32_t *C, float32_t *A, float32_t *B){
    int i, j, k;

    for (i = 0; i < N; i++){
        for (j = 0; j < N; j++){
            for (k = 0; k < N; k++)
		        C[i * N + j] = C[i * N + j] + A[i * N + k] * B[k * N + j];
        }	
    }
}

void matrix_init_rand(float32_t *mat, uint32_t length) {
    for (int i=0; i<length; i++) {
        mat[i] = (float)rand()/(float)(RAND_MAX);
    }
}
void matrix_init_const(float32_t *mat, uint32_t length, float32_t val) {
    for (int i=0; i<length; i++) {
        mat[i] = val;
    }
}
void print_mat(float32_t *mat, uint32_t mat_size, uint32_t print_window){
    for(int i = 0 ; i < print_window ; i++){
        for(int j = 0 ; j < print_window ; j++){
            printf(" %.1f ", mat[mat_size * i + j]);
        }
        printf("\n");
    }
    printf("\n");
}
bool mat_comp(float32_t *A, float32_t *B, uint32_t length){
    for(int i = 0 ; i < length ; i++){
        for(int j = 0 ; j < length ; j++){
            if (A[length * i + j] != B[length * i + j]){
                printf("i:%i, j:%i, mat_1:%f, mat_2:%f\n", i, j, A[length * i + j], B[length * i + j]);
                return false;
            }
        }
    }
    return true;
}

void block_mat_A_8x8(float32_t *mat_in, float32_t *mat_out){
    for(int i = 0 ; i < 128 ; i++){
        for(int j = 0 ; j < 128 ; j++){
            int start = 8192 * i + 64 * j;
            for(int ii = 0 ; ii < 8 ; ii++){
                for(int jj = 0 ; jj < 8 ; jj++){
                    mat_out[start + jj * 8 + ii] = mat_in[jj + ii * 1024 + j * 8 + i * 8192];
                }
            }
        }
    }
}
void block_mat_B_8x8(float32_t *mat_in, float32_t *mat_out){
    for(int i = 0 ; i < 128 ; i++){
        for(int j = 0 ; j < 128 ; j++){
            int start = 8192 * i + 64 * j;
            for(int ii = 0 ; ii < 8 ; ii++){
                for(int jj = 0 ; jj < 8 ; jj++){
                    mat_out[start + ii * 8 + jj] = mat_in[jj + ii * 1024 + j * 8 + i * 8192];
                }
            }
        }
    }
}
void unblock_mat(float32_t *mat_in, float32_t *mat_out){
    for(int i = 0 ; i < 128 ; i++){
        for(int j = 0 ; j < 128 ; j++){
            int start = 8192 * i + 64 * j;
            for(int ii = 0 ; ii < 8 ; ii++){
                for(int jj = 0 ; jj < 8 ; jj++){
                    mat_out[jj + ii * 1024 + j * 8 + i * 8192] = mat_in[start + ii * 8 + jj];                    
                }
            }
        }
    }
}

void block_mat_B_8x32(float32_t *mat_in, float32_t *mat_out){
    // 1024 / 32 = 32
    for(int i = 0 ; i < 32 ; i++){
        // 1024 / 8 = 128
        for(int j = 0 ; j < 128 ; j++){
            // 8 * 32 = 256
            // 32 * 1024 = 32768
            int start = 32768 * i + 256 * j;
            for(int ii = 0 ; ii < 32 ; ii++){
                for(int jj = 0 ; jj < 8 ; jj++){
                    mat_out[start + ii * 8 + jj] = mat_in[jj + ii * 1024 + j * 8 + i * 32768];
                }
            }
        }
    }
}
void block_mat_A_8x32(float32_t *mat_in, float32_t *mat_out){
    // 1024 / 8 = 128
    for(int i = 0 ; i < 128 ; i++){
        // 1024 / 32 = 32
        for(int j = 0 ; j < 32 ; j++){
            // 8 * 32 = 256
            // 8 * 1024 = 8192
            int start = 8192 * i + 256 * j;
            for(int ii = 0 ; ii < 8 ; ii++){
                for(int jj = 0 ; jj < 32 ; jj++){
                    mat_out[start + jj * 8 + ii] = mat_in[jj + ii * 1024 + j * 32 + i * 8192];
                }
            }
        }
    }
}

void L1_reorder_A(float32_t *mat_in, float32_t *mat_out){
    // complete matrix
    // 32 lines of 1024 x 32 values
    for(int line = 0 ; line < 32 ; line++){
        // each line skips 1024 x 32 = 32768 values

        // 32 L1-tiles with 32 x 32 values
        for(int L1T = 0 ; L1T < 32 ; L1T++){
            // each L1T skips 32 x 32 = 1024 values

            // 4 rows with 32 x 8 values
            for(int row = 0 ; row < 4 ; row++){
                // each row skips 32 x 8 = 256 values

                // 8 lines with 32 vaules 
                for(int ii = 0 ; ii < 8 ; ii++){
                    // each ii skips 32 values

                    // 32 values
                    for(int jj = 0 ; jj < 32 ; jj++){
                        // each jj skips 1 value
                        
                        mat_out[line * 32768 + L1T * 1024 + row * 256 + ii + jj * 8] = mat_in[line * 32768 + L1T * 32 + row * 8192 + ii * 1024 + jj];
                    }
                }
            }
        }
    }
}
void L1_reorder_B(float32_t *mat_in, float32_t *mat_out){
    // complete matrix
    // 32 column of 1024 x 32 values
    for(int column = 0 ; column < 32 ; column++){
        // each column skips 1024 x 32 = 32768 values

        // 32 L1-tiles with 32 x 32 values
        for(int L1T = 0 ; L1T < 32 ; L1T++){
            // each L1T skips 32 x 32 = 1024 values

            // 4 columns with 8 x 32 values
            for(int col = 0 ; col < 4 ; col++){
                // each col skips 32 x 8 = 256 values

                // 32 lines with 8 vaules 
                for(int ii = 0 ; ii < 32 ; ii++){
                    // each ii skips 8 values

                    // 8 values
                    for(int jj = 0 ; jj < 8 ; jj++){
                        // each jj skips 1 value
                        
                        mat_out[column * 32768 + L1T * 1024 + col * 256 + ii * 8 + jj] = mat_in[column * 32 + L1T * 32768 + col * 8 + ii * 1024 + jj];
                    }
                }
            }
        }
    }
}
void L1_reorder_C(float32_t *mat_in, float32_t *mat_out){
    // complete matrix
    // 32 lines of 1024 x 32 values
    for(int line = 0 ; line < 32 ; line++){
        // each line skips 1024 x 32 = 32768 values

        // 32 L1-tiles with 32 x 32 values
        for(int L1T = 0 ; L1T < 32 ; L1T++){
            // each L1T skips 32 x 32 = 1024 values

            // 4 columns with 8 x 32 values
            for(int row = 0 ; row < 4 ; row++){
                // each col skips 32 x 8 = 256 values

                // 4 rows with 8 x 8 values
                for(int col = 0 ; col < 4 ; col++){
                    // each row skips 8 x 8 = 64 values

                    // 8 lines with 8 vaules 
                    for(int ii = 0 ; ii < 8 ; ii++){
                        // each ii skips 8 values

                        // 8 values
                        for(int jj = 0 ; jj < 8 ; jj++){
                            // each jj skips 1 value
                            
                            mat_out[line * 32768 + L1T * 32 + row * 8192 + col * 8 + ii * 1024 + jj] = mat_in[line * 32768 + L1T * 1024 + row * 256 + col * 64 + ii * 8 + jj];
                        }
                    }
                }
            }
        }
    }
}    

struct thread_arguments{
    uint32_t id;
    float32_t *mat_A;
    float32_t *mat_B;
    float32_t *mat_C;
};
void *parallel_8x8(void *args){
    struct thread_arguments *param = (struct thread_arguments *)args;
//    printf("T%u:\tstart execution\n", param->id);
    for (uint32_t i = param->id; i < 128; i += 4){
        for (int j = 0; j < 128; j++){
            for (int k = 0; k < 128; k++) {
                kernel_8x8_save(&param->mat_A[8192 * i + 64 * k], &param->mat_B[8192 * k + 64 * j], &param->mat_C[8192 * i + 64 * j]);
                if( i==0 && j==0 && k==0)
                    printf("%d", k);
            }
        }
    }
//    printf("T%u:\tstop execution\n", param->id);
    pthread_exit(NULL);
}
void *parallel_8x32(void *args){
    struct thread_arguments *param = (struct thread_arguments *)args;
//    printf("T%u:\tstart execution\n", param->id);
    for (uint32_t i = param->id; i < 128; i += 4){
        for (int j = 0; j < 128; j++){
            for (int k = 0; k < 32; k++) {
        		kernel_8x32_save(&param->mat_A[8192 * i + 256 * k], &param->mat_B[32768 * k + 256 * j], &param->mat_C[8192 * i + 64 * j]);
                if( i==0 && j==0 && k==0){
                    printf("%d\n",k);
		        }
            }
        }
    }
//    printf("T%u:\tstop execution\n", param->id);
    pthread_exit(NULL);
}
void *parallel_L1(void *args){
    struct thread_arguments *param = (struct thread_arguments *)args;
//    printf("T%u:\tstart execution\n", param->id);

    int start_line = -1;
    int start_column = -1;
    switch (param->id){
        case 0:
            start_line = 0;
            start_column = 0;
            break;
        case 1:
            start_line = 0;
            start_column = 1;
            break;
        case 2:
            start_line = 1;
            start_column = 0;
            break;
        case 3:
            start_line = 1;
            start_column = 1;
            break;
        default:
            break;
    }
    // complete matrix
    // 32 lines of 1024 x 32 values
    for(int line = start_line ; line < 32 ; line += 2){
        // each line skips 1024 x 32 = 32768 values

        // 32 columns (L1-tiles) with 32 x 32 values
        for(int column = start_column ; column < 32 ; column += 2){
            // each column (L1-tile) skips 32 x 32 = 1024 values

            // load 32 x 32 C-tile to L1

            // for this 32 x 32 C-tile, iterate through all 32 Tiles of A and B
            for(int step = 0 ; step < 32 ; step++){
                // each step skips 32 x 32 = 1024 AB-values

                // load this 32 x 32 AB-tile to L1 (and the two ahead)

                // Per Tile:
                int start_ABT = 0;
                int k = 0;
                for(int i = 0 ; i < 4 ; i++){
                    for(int j = 0 ; j < 4 ; j++){
                        kernel_8x32_save(&param->mat_A[line * 32768 + step * 1024 + (i * 256)], &param->mat_B[column * 32768 + step * 1024 + (j * 256)], &param->mat_C[line * 32768 + column * 1024 + (k * 64)]);
                        k++;
                    }
                }
            }
        }
    }
//    printf("T%u:\tstop execution\n", param->id);
    pthread_exit(NULL);
}

int run_threads_8x8(float32_t *A, float32_t *B, float32_t *C){
    const uint32_t NUM_THREADS = 4;
    pthread_t threads[NUM_THREADS];
    struct thread_arguments thread_args[NUM_THREADS];

//    printf("Main:\tstart threads\n");
    for(int i = 0 ; i < NUM_THREADS ; i++){
        thread_args[i].id = i;
        thread_args[i].mat_A = A;
        thread_args[i].mat_B = B;
        thread_args[i].mat_C = C;
//        printf("Main:\tcreating thread %i\n", i);
        if(pthread_create(&threads[i], NULL, parallel_8x8, (void *)&thread_args[i]) != 0){
            printf("ERROR creating thread\n");
            return 1;
        }
    }
    for(int i = 0 ; i < NUM_THREADS ; i++){
//        printf("Main:\tjoining thread %i\n", i);
        if(pthread_join(threads[i], NULL) != 0){
            printf("ERROR joining thread\n");
            return 1;
        }
    }
//    printf("Main:\tthreads stopped\n");
    return 0;
}
int run_threads_8x32(float32_t *A, float32_t *B, float32_t *C){
    const uint32_t NUM_THREADS = 4;
    pthread_t threads[NUM_THREADS];
    struct thread_arguments thread_args[NUM_THREADS];

//    printf("Main:\tstart threads\n");
    for(int i = 0 ; i < NUM_THREADS ; i++){
        thread_args[i].id = i;
        thread_args[i].mat_A = A;
        thread_args[i].mat_B = B;
        thread_args[i].mat_C = C;
//        printf("Main:\tcreating thread %i\n", i);
        if(pthread_create(&threads[i], NULL, parallel_8x32, (void *)&thread_args[i]) != 0){
            printf("ERROR creating thread\n");
            return 1;
        }
    }
    for(int i = 0 ; i < NUM_THREADS ; i++){
//        printf("Main:\tjoining thread %i\n", i);
        if(pthread_join(threads[i], NULL) != 0){
            printf("ERROR joining thread\n");
            return 1;
        }
    }
//    printf("Main:\tthreads stopped\n");
    return 0;
}
int run_threads_L1(float32_t *A, float32_t *B, float32_t *C){
    const uint32_t NUM_THREADS = 4;
    pthread_t threads[NUM_THREADS];
    struct thread_arguments thread_args[NUM_THREADS];

//    printf("Main:\tstart threads\n");
    for(int i = 0 ; i < NUM_THREADS ; i++){
        thread_args[i].id = i;
        thread_args[i].mat_A = A;
        thread_args[i].mat_B = B;
        thread_args[i].mat_C = C;
//        printf("Main:\tcreating thread %i\n", i);
        if(pthread_create(&threads[i], NULL, parallel_L1, (void *)&thread_args[i]) != 0){
            printf("ERROR creating thread\n");
            return 1;
        }
    }
    for(int i = 0 ; i < NUM_THREADS ; i++){
//        printf("Main:\tjoining thread %i\n", i);
        if(pthread_join(threads[i], NULL) != 0){
            printf("ERROR joining thread\n");
            return 1;
        }
    }
//    printf("Main:\tthreads stopped\n");
    return 0;
}

float32_t mat_A[N * N];                     // input A
float32_t mat_B[N * N];                     // input B
float32_t mat_reference_result[N * N];      // result of matrix_multiply, keep as a reference to compare the other results
float32_t mat_compared_result[N * N];       // result of each tested program

float32_t mat_A_tr[N * N];                  // input A needs to be transformed to work with the programs
float32_t mat_B_tr[N * N];                  // input B needs to be transformed to work with the programs
float32_t mat_compared_result_tr[N * N];    // result of some programs, needs to be transformed into the real result

int main(){

    struct timespec start, end;
    uint64_t time_reference_ns = 0;
    uint64_t time_mem_a_ns = 0;
    uint64_t time_mem_b_ns = 0;
    uint64_t time_kernel_ns = 0;
    uint64_t time_mem_c_ns = 0;
    uint64_t time_total_ns = 0;

    bool mat_is_eq;
    const int window_size_of_matrix_print = 8;

    // you can use random input-matricies or constant. Real test of the same results only works with random
    matrix_init_rand(mat_A, N*N);
    matrix_init_rand(mat_B, N*N);
//    matrix_init_const(mat_A, N*N, 10);
//    matrix_init_const(mat_B, N*N, 20);

    matrix_init_const(mat_reference_result, N*N, 0);
    matrix_init_const(mat_compared_result, N*N, 0);
    matrix_init_const(mat_A_tr, N*N, 0);
    matrix_init_const(mat_B_tr, N*N, 0);
    matrix_init_const(mat_compared_result_tr, N*N, 0);
    
    printf("mat_A\n"); print_mat(mat_A, N, window_size_of_matrix_print);
    printf("mat_B\n"); print_mat(mat_B, N, window_size_of_matrix_print);


    // reference matrix multiply
    printf("\nreference matrix multiply\n");
    clock_gettime(CLOCK_MONOTONIC, &start);
    multiply_c(mat_reference_result, mat_A, mat_B);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_reference_ns = (end.tv_sec - start.tv_sec) * 1e9 + (end.tv_nsec - start.tv_nsec);
    printf("execution took [ns]: %lu\nreference matrix:\n", time_reference_ns);
    print_mat(mat_reference_result, N, window_size_of_matrix_print);


    // kernel 8x8, one core - timing each memory-operation and the execution itself
    printf("\nkernel 8x8, one core\n");
    clock_gettime(CLOCK_MONOTONIC, &start);
    block_mat_A_8x8(mat_A, mat_A_tr);                                       // prepare matrix A
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_mem_a_ns = (end.tv_sec - start.tv_sec) * 1e9 + (end.tv_nsec - start.tv_nsec);

    clock_gettime(CLOCK_MONOTONIC, &start);
    block_mat_B_8x8(mat_B, mat_B_tr);                                       // prepare matrix B
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_mem_b_ns = (end.tv_sec - start.tv_sec) * 1e9 + (end.tv_nsec - start.tv_nsec);
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    scale_kernel_8x8_block(mat_A_tr, mat_B_tr, mat_compared_result_tr); // execute kernel
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_kernel_ns = (end.tv_sec - start.tv_sec) * 1e9 + (end.tv_nsec - start.tv_nsec);
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    unblock_mat(mat_compared_result_tr, mat_compared_result);           // prepare result
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_mem_c_ns = (end.tv_sec - start.tv_sec) * 1e9 + (end.tv_nsec - start.tv_nsec);
    time_total_ns = time_mem_a_ns + time_mem_b_ns + time_kernel_ns + time_mem_c_ns;
    
    mat_is_eq = mat_comp(mat_reference_result, mat_compared_result, N); // compare results
    printf("results are equal? %d (1=yes, 0=no)\n", mat_is_eq);
    printf("timing\nmem_A: %lu, mem_B: %lu, kernel: %lu, mem_C: %lu\n--- total time: %lu >>> speedup: %.2lu ---\n", time_mem_a_ns, time_mem_b_ns, time_kernel_ns, time_mem_c_ns, time_total_ns, time_reference_ns / time_total_ns);
    printf("found matrix:\n"); print_mat(mat_compared_result, N, window_size_of_matrix_print);


    // reset results for next program
    matrix_init_const(mat_compared_result, N*N, 0);
    matrix_init_const(mat_compared_result_tr, N*N, 0);


    // nparallel kernel 8x8, all cores - timing each memory-operation and the execution itself
    printf("\nparallel kernel 8x8, all cores\n");
    clock_gettime(CLOCK_MONOTONIC, &start);
    block_mat_A_8x8(mat_A, mat_A_tr);                                       // prepare matrix A
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_mem_a_ns = (end.tv_sec - start.tv_sec) * 1e9 + (end.tv_nsec - start.tv_nsec);

    clock_gettime(CLOCK_MONOTONIC, &start);
    block_mat_B_8x8(mat_B, mat_B_tr);                                       // prepare matrix B
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_mem_b_ns = (end.tv_sec - start.tv_sec) * 1e9 + (end.tv_nsec - start.tv_nsec);
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    run_threads_8x8(mat_A_tr, mat_B_tr, mat_compared_result_tr);        // execute kernel
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_kernel_ns = (end.tv_sec - start.tv_sec) * 1e9 + (end.tv_nsec - start.tv_nsec);
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    unblock_mat(mat_compared_result_tr, mat_compared_result);           // prepare result
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_mem_c_ns = (end.tv_sec - start.tv_sec) * 1e9 + (end.tv_nsec - start.tv_nsec);
    time_total_ns = time_mem_a_ns + time_mem_b_ns + time_kernel_ns + time_mem_c_ns;
    
    mat_is_eq = mat_comp(mat_reference_result, mat_compared_result, N); // compare results
    printf("results are equal? %d (1=yes, 0=no)\n", mat_is_eq);
    printf("timing\nmem_A: %lu, mem_B: %lu, kernel: %lu, mem_C: %lu\n--- total time: %lu >>> speedup: %.2lu ---\n", time_mem_a_ns, time_mem_b_ns, time_kernel_ns, time_mem_c_ns, time_total_ns, time_reference_ns / time_total_ns);
    printf("found matrix:\n"); print_mat(mat_compared_result, N, window_size_of_matrix_print);


    // reset results for next program
    matrix_init_const(mat_compared_result, N*N, 0);
    matrix_init_const(mat_compared_result_tr, N*N, 0);


    // kernel 8x32, one core - timing each memory-operation and the execution itself
    printf("\nkernel 8x32, one core\n");
    clock_gettime(CLOCK_MONOTONIC, &start);
    block_mat_A_8x32(mat_A, mat_A_tr);                                      // prepare matrix A
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_mem_a_ns = (end.tv_sec - start.tv_sec) * 1e9 + (end.tv_nsec - start.tv_nsec);

    clock_gettime(CLOCK_MONOTONIC, &start);
    block_mat_B_8x32(mat_B, mat_B_tr);                                      // prepare matrix B
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_mem_b_ns = (end.tv_sec - start.tv_sec) * 1e9 + (end.tv_nsec - start.tv_nsec);
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    scale_kernel_8x32_block(mat_A_tr, mat_B_tr, mat_compared_result_tr);    // execute kernel
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_kernel_ns = (end.tv_sec - start.tv_sec) * 1e9 + (end.tv_nsec - start.tv_nsec);
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    unblock_mat(mat_compared_result_tr, mat_compared_result);               // prepare result
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_mem_c_ns = (end.tv_sec - start.tv_sec) * 1e9 + (end.tv_nsec - start.tv_nsec);
    time_total_ns = time_mem_a_ns + time_mem_b_ns + time_kernel_ns + time_mem_c_ns;
    
    mat_is_eq = mat_comp(mat_reference_result, mat_compared_result, N);     // compare results
    printf("results are equal? %d (1=yes, 0=no)\n", mat_is_eq);
    printf("timing\nmem_A: %lu, mem_B: %lu, kernel: %lu, mem_C: %lu\n--- total time: %lu >>> speedup: %.2lu ---\n", time_mem_a_ns, time_mem_b_ns, time_kernel_ns, time_mem_c_ns, time_total_ns, time_reference_ns / time_total_ns);
    printf("found matrix:\n"); print_mat(mat_compared_result, N, window_size_of_matrix_print);


    // reset results for next program
    matrix_init_const(mat_compared_result, N*N, 0);
    matrix_init_const(mat_compared_result_tr, N*N, 0);


    // parallel kernel 8x32, all cores - timing each memory-operation and the execution itself
    printf("\nparallel kernel 8x32, all cores\n");
    clock_gettime(CLOCK_MONOTONIC, &start);
    block_mat_A_8x32(mat_A, mat_A_tr);                                      // prepare matrix A
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_mem_a_ns = (end.tv_sec - start.tv_sec) * 1e9 + (end.tv_nsec - start.tv_nsec);

    clock_gettime(CLOCK_MONOTONIC, &start);
    block_mat_B_8x32(mat_B, mat_B_tr);                                      // prepare matrix B
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_mem_b_ns = (end.tv_sec - start.tv_sec) * 1e9 + (end.tv_nsec - start.tv_nsec);
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    run_threads_8x32(mat_A_tr, mat_B_tr, mat_compared_result_tr);           // execute kernel
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_kernel_ns = (end.tv_sec - start.tv_sec) * 1e9 + (end.tv_nsec - start.tv_nsec);
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    unblock_mat(mat_compared_result_tr, mat_compared_result);               // prepare result
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_mem_c_ns = (end.tv_sec - start.tv_sec) * 1e9 + (end.tv_nsec - start.tv_nsec);
    time_total_ns = time_mem_a_ns + time_mem_b_ns + time_kernel_ns + time_mem_c_ns;
    
    mat_is_eq = mat_comp(mat_reference_result, mat_compared_result, N);     // compare results
    printf("results are equal? %d (1=yes, 0=no)\n", mat_is_eq);
    printf("timing\nmem_A: %lu, mem_B: %lu, kernel: %lu, mem_C: %lu\n--- total time: %lu >>> speedup: %.2lu ---\n", time_mem_a_ns, time_mem_b_ns, time_kernel_ns, time_mem_c_ns, time_total_ns, time_reference_ns / time_total_ns);
    printf("found matrix:\n"); print_mat(mat_compared_result, N, window_size_of_matrix_print);
   

    // reset results for next program
    matrix_init_const(mat_compared_result, N*N, 0);
    matrix_init_const(mat_compared_result_tr, N*N, 0);


    // L1 tiling, all cores - timing each memory-operation and the execution itself
    printf("\nL1 tiling, all cores\n");
    clock_gettime(CLOCK_MONOTONIC, &start);
    L1_reorder_A(mat_A, mat_A_tr);                                      // prepare matrix A
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_mem_a_ns = (end.tv_sec - start.tv_sec) * 1e9 + (end.tv_nsec - start.tv_nsec);

    clock_gettime(CLOCK_MONOTONIC, &start);
    L1_reorder_B(mat_B, mat_B_tr);                                      // prepare matrix B
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_mem_b_ns = (end.tv_sec - start.tv_sec) * 1e9 + (end.tv_nsec - start.tv_nsec);
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    run_threads_L1(mat_A_tr, mat_B_tr, mat_compared_result_tr);         // execute kernel
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_kernel_ns = (end.tv_sec - start.tv_sec) * 1e9 + (end.tv_nsec - start.tv_nsec);
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    L1_reorder_C(mat_compared_result_tr, mat_compared_result);          // prepare result
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_mem_c_ns = (end.tv_sec - start.tv_sec) * 1e9 + (end.tv_nsec - start.tv_nsec);
    time_total_ns = time_mem_a_ns + time_mem_b_ns + time_kernel_ns + time_mem_c_ns;
    
    mat_is_eq = mat_comp(mat_reference_result, mat_compared_result, N); // compare results
    printf("results are equal? %d (1=yes, 0=no)\n", mat_is_eq);
    printf("timing\nmem_A: %lu, mem_B: %lu, kernel: %lu, mem_C: %lu\n--- total time: %lu >>> speedup: %.2lu ---\n", time_mem_a_ns, time_mem_b_ns, time_kernel_ns, time_mem_c_ns, time_total_ns, time_reference_ns / time_total_ns);
    printf("found matrix:\n"); print_mat(mat_compared_result, N, window_size_of_matrix_print);
   
    return 0;
}


