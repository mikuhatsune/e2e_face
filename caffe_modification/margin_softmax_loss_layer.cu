#include <algorithm>
#include <cfloat>
#define _USE_MATH_DEFINES
#include <cmath>
#include <vector>

//#include <cuda.h>
//#include "caffe/layer.hpp"
#include "caffe/layers/margin_softmax_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

//#define M_PI       3.14159265358979323846

namespace caffe {

// no need to take 'sqrt'!!!
template <typename Dtype>
void caffe_gpu_norm2(const int n, const Dtype* x, Dtype* out);
template <>
void caffe_gpu_norm2<float>(const int n, const float* x, float* out) {
    CUBLAS_CHECK(cublasSnrm2(Caffe::cublas_handle(), n, x, 1, out));
}
template <>
void caffe_gpu_norm2<double>(const int n, const double* x, double* out) {
    CUBLAS_CHECK(cublasDnrm2(Caffe::cublas_handle(), n, x, 1, out));
}

/*template <typename Dtype>
__global__ void sqrt_kernel(const int n, Dtype* y) {
    CUDA_KERNEL_LOOP(index, n) {
        y[index] = sqrt(y[index]);
    }
}*/
template <typename Dtype>
static __global__ void compute_exp_kernel(const int n, const int P, const Dtype *max_f, Dtype* y) {
    CUDA_KERNEL_LOOP(index, n) {
        y[index] = exp(y[index] - max_f[index / P]);
    }
}

__device__ inline void atomic_add(float * address, float val) {
    atomicAdd(address, val);
}
__device__ inline void atomic_add(double * address, double val) {
    unsigned long long int* address_as_ull =
            (unsigned long long int*) address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                __double_as_longlong(val +
                __longlong_as_double(assumed)));
        // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);
}

//__constant__ const double cos_k_table[5] = { 1, M_SQRT1_2, 0, -M_SQRT1_2, -1 };

template <typename Dtype>
static __global__ void compute_fyi(const int nthreads, const int P, const int F, const Dtype lambda,
    const Dtype* x, Dtype* x_norm, const Dtype* w, Dtype* w_norm, const Dtype *label,
    Dtype *ip_data, Dtype *phi_data, Dtype *prob_data, Dtype *max_f) {

    const Dtype cos_k_table[5] = { 1, M_SQRT1_2, 0, -M_SQRT1_2, -1 };

    CUDA_KERNEL_LOOP(i, nthreads) {
        const int yi = label[i];

        Dtype fyi = prob_data[i*P + yi];
        ip_data[i] = fyi;

        if (sizeof(Dtype) == sizeof(double)) {
            x_norm[i] = norm(F, (const double*)(x + i*F));
            w_norm[i] = norm(F, (const double*)(w + yi*F));
        }
        else {
            x_norm[i] = normf(F, (const float*)(x + i*F));
            w_norm[i] = normf(F, (const float*)(w + yi*F));
        }

        Dtype xw = x_norm[i] * w_norm[i];
        Dtype cos_th = xw == 0 ? 1 : fyi / xw;

        for (int k = 0; k < 4; ++k) {
            //if (cos(k*M_PI / 4) >= cos_th && cos_th >= cos((k + 1)*M_PI / 4)) {
            if (cos_k_table[k] >= cos_th && cos_th >= cos_k_table[k + 1]) {
                Dtype c2 = cos_th * cos_th;
                Dtype phi = (k & 1 ? (-1) : 1) * (8 * c2 * (c2 - 1) + 1) - 2 * k;
                phi_data[i] = phi;
                fyi = (fyi * lambda + xw * phi) / (1 + lambda);
                prob_data[i*P + yi] = fyi;
                break;
            }
        }

        for (int j = 0; j < P; ++j)
            if (prob_data[i*P + j] > fyi)
                fyi = prob_data[i*P + j];
        max_f[i] = fyi;
    }
}

template <typename Dtype>
static __global__ void compute_loss(const int nthreads, const int P,
    const Dtype *label, Dtype *prob_data, const Dtype *sum_exp, Dtype *loss) {

    CUDA_KERNEL_LOOP(index, nthreads) {
        const int i = index / P;
        const int j = index % P;
        const int yi = label[i];

        prob_data[index] /= sum_exp[i];
        if (j == yi) {
            atomic_add(loss, -log(prob_data[index]));
            //atomic_add(loss, log(sum_exp[i]) - log);
        }
    }
}

template <typename Dtype>
void MarginSoftmaxLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

    const Dtype* const x = bottom[0]->gpu_data();
    const Dtype* const label = bottom[1]->gpu_data();
    Dtype* const loss = top[0]->mutable_gpu_data();

    const Dtype* const w = this->blobs_[0]->gpu_data();

    Dtype* const ip_data = ip_.mutable_gpu_data();
    Dtype* const phi_data = ip_.mutable_gpu_diff();
    Dtype* const prob_data = prob_.mutable_gpu_data();

    Dtype* const x_norm = prob_.mutable_gpu_diff(); // N
    Dtype* const w_norm = x_norm + N;               // N!!! not P
    Dtype* max_f = w_norm + N;                      // N
    const Dtype* const ones_P = max_f + N;          // P
    Dtype* const sum_exp = (Dtype*)ones_P + P;      // N

    const Dtype lambda = this->lambda_.get_iter("lambda");
    
    // It turnes out to be TOO SLOW...to switch between modes
    /*//cublasSetPointerMode(Caffe::cublas_handle(), CUBLAS_POINTER_MODE_DEVICE);
    // compute |x|
    for (int i = 0; i < N; ++i) {
        caffe_gpu_norm2(F, x + i*F, x_norm + i);
    }
    // compute |w|
    //compute_norms<Dtype> << <CAFFE_GET_BLOCKS(P), CAFFE_CUDA_NUM_THREADS >> >(P, F, w, w_norm);
    for (int j = 0; j < P; j++) {
        caffe_gpu_norm2(F, w + j*F, w_norm + j);
    }
    cublasSetPointerMode(Caffe::cublas_handle(), CUBLAS_POINTER_MODE_HOST);*/

    //sqrt_kernel<Dtype> << <CAFFE_GET_BLOCKS(P + N), CAFFE_CUDA_NUM_THREADS >> >(P + N, x_norm);

    // compute inner product: w^T * x
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
        N, P, F, (Dtype)1.,
        x, w, (Dtype)0., prob_data);

    // f_j(j!=yi) = w_j^T * x
    // f_yi = ...
    compute_fyi<Dtype> << <CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS >> >(
        N, P, F, lambda,
        x, x_norm, w, w_norm, label,
        ip_data, phi_data, prob_data, max_f);

    // exp(f_j), max_f is employed to ensure numerical stability
    compute_exp_kernel<Dtype> << <CAFFE_GET_BLOCKS(N*P), CAFFE_CUDA_NUM_THREADS >> >(
        N*P, P, max_f, prob_data);

    // compute the denominator: sum(exp(...))
    caffe_gpu_gemv<Dtype>(CblasNoTrans,
        N, P, (Dtype)1,
        prob_data, ones_P, (Dtype)0, sum_exp);

    // compute and accumalate loss
    caffe_gpu_set(1, Dtype(0), loss);
    compute_loss<Dtype> << <CAFFE_GET_BLOCKS(N*P), CAFFE_CUDA_NUM_THREADS >> >(
        N*P, P, label, prob_data, sum_exp, loss);
    caffe_gpu_scal(1, Dtype(1) / N, loss);
    //top[0]->mutable_cpu_data()[0] = loss / N;*/

}

template <typename Dtype>
static __device__ void atomic_axpy(const int n, const Dtype a, const Dtype* x, Dtype* y) {
    for (int i = 0; i < n; ++i, ++x, ++y) {
        atomic_add(y, a * (*x));
    }
}
template <typename Dtype>
static __device__ void axy(const int n, const Dtype a, const Dtype* x, Dtype* y) {
    for (int i = 0; i < n; ++i, ++x, ++y)
        *y = a * (*x);
}
template <typename Dtype>
static __device__ void axpy(const int n, const Dtype a, const Dtype* x, Dtype* y) {
    for (int i = 0; i < n; ++i, ++x, ++y)
        *y += a * (*x);
}

template <typename Dtype>
static __global__ void backward_yi(const int nthreads, const int P, const int F, const Dtype lambda,
    const Dtype* x, const Dtype* x_norm, const Dtype* w, const Dtype* w_norm, const Dtype* label,
    const Dtype* ip_data, const Dtype* phi_data, Dtype* prob_data,
    Dtype* x_diff, Dtype* w_diff
    ) {

    CUDA_KERNEL_LOOP(i, nthreads) {
        const Dtype * const xi = x + i*F;
        const int yi = label[i];
        Dtype *x_diff_i = x_diff + i*F;
        const Dtype x_norm_i = x_norm[i];
        const Dtype ip = ip_data[i];
        const Dtype phi = phi_data[i];

        const Dtype * const wj = w + yi*F;
        Dtype *w_diff_j = w_diff + yi*F;
        //const Dtype w_norm_j = w_norm[yi];
        const Dtype w_norm_j = w_norm[i];
        const Dtype prob_ij = prob_data[i*P + yi] - 1;

        // avoid mis-computation during the later 'gemm'
        prob_data[i*P + yi] = 0;

        if (w_norm_j == 0) {
            atomic_axpy(F, prob_ij * 4, xi, w_diff_j);
            axy(F, Dtype(0), wj, x_diff_i);    // set to 0
        }
        else if (x_norm_i == 0) {
            axy(F, prob_ij * 4, wj, x_diff_i);
        }
        else {
            Dtype m = ((-1 > phi && phi >= -3) || (-5 > phi && phi >= -7)) * (-8);
            m = m * (2 * ip*ip / (w_norm_j*w_norm_j*x_norm_i*x_norm_i) - 1);

            // d(Li)/d(xi) = -d(f_yi)/d(xi) * ( 1-p(yi|xi,w) ) + \sum_{j!=y_i} w_j * p(j|wi,w)
            //   d( f_yi = |w||x|phi(th) ) / d(x) = 
            axy(F,
                prob_ij * (
                    w_norm_j * phi / x_norm_i
                    - m * 2 * ip*ip / (x_norm_i*x_norm_i*x_norm_i*w_norm_j)
                ) / (1 + lambda),
                xi, x_diff_i);

            // d(Li)/d(w_yi) = -d(f_yi)/d(w_yi) * ( 1-p(yi|xi,w) )
            //   d( f_yi = |w||x|phi(th) ) / d(w) = 
            atomic_axpy(F,
                prob_ij * (
                    x_norm_i * phi / w_norm_j
                    - m * 2 * ip*ip / (w_norm_j*w_norm_j*w_norm_j*x_norm_i)
                ) / (1 + lambda),
                wj, w_diff_j);


            m = prob_ij * (m * 2 * ip / (x_norm_i * w_norm_j) + lambda) / (1 + lambda);

            //atomic_axpy(F, m, wj, x_diff_i);
            axpy(F, m, wj, x_diff_i);

            atomic_axpy(F, m, xi, w_diff_j);
        }
    }
}

template <typename Dtype>
void MarginSoftmaxLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

    const Dtype* const x = bottom[0]->gpu_data();
    Dtype* const x_diff = bottom[0]->mutable_gpu_diff();

    const Dtype* const label = bottom[1]->gpu_data();
    const Dtype loss_weight = top[0]->cpu_diff()[0] / N;

    const Dtype* const w = this->blobs_[0]->gpu_data();
    //printf("w_diff[0]=%f\n", this->blobs_[0]->cpu_diff()[0]);
    Dtype* const w_diff = this->blobs_[0]->mutable_gpu_diff();

    const Dtype* const ip_data = ip_.gpu_data();
    const Dtype* const phi_data = ip_.gpu_diff();
    Dtype* const prob_data = prob_.mutable_gpu_data();

    const Dtype* const x_norm = prob_.gpu_diff(); // N
    const Dtype* const w_norm = x_norm + N;       // N!!! not P
    
    const Dtype lambda = this->lambda_.get();

    // compute special cases of j==yi
    backward_yi<Dtype> << <CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS >> >(
        N, P, F, lambda,
        x, x_norm, w, w_norm, label, ip_data, phi_data, prob_data,
        x_diff, w_diff);

    // then, collect gradients from output j!=y_i
    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
        P, F, N,
        (Dtype)1., prob_data, x,
        (Dtype)1., w_diff);

    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
        N, F, P,
        (Dtype)1., prob_data, w,
        (Dtype)1., x_diff);

    // scale the gradients according to top_loss
    caffe_gpu_scal<Dtype>(P*F, loss_weight, w_diff);
    caffe_gpu_scal<Dtype>(N*F, loss_weight, x_diff);
}

INSTANTIATE_LAYER_GPU_FUNCS(MarginSoftmaxLossLayer);

}  // namespace caffe
