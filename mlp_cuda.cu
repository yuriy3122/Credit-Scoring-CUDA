#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <cmath>
#include <numeric>

#define CHECK_CUDA(call) do { cudaError_t err = (call); if (err != cudaSuccess) { \
    std::cerr << "CUDA error " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
    std::exit(1);} } while(0)

struct Dataset {
    std::vector<float> X; // [N,D], row-major
    std::vector<int> y;   // [N]
    int N=0, D=0;
};

static inline std::vector<std::string> split_csv(const std::string& line) {
    std::vector<std::string> out; out.reserve(64);
    std::string cur; bool inq=false;
    for (char c: line) {
        if (c=='"') inq=!inq;
        else if (c==',' && !inq) { out.push_back(cur); cur.clear(); }
        else cur.push_back(c);
    }
    out.push_back(cur);
    return out;
}

static inline bool parse_float(const std::string& s, float& v) {
    try { size_t idx=0; v = std::stof(s, &idx); return idx>0; }
    catch(...) { return false; }
}

Dataset load_csv(const std::string& path, const std::string& target_col="action_taken", int max_features=14) {
    std::ifstream f(path);
    if (!f) { std::cerr<<"Cannot open "<<path<<"\n"; std::exit(1); }
    std::string line;
    if (!std::getline(f, line)) { std::cerr<<"Empty CSV\n"; std::exit(1); }
    auto headers = split_csv(line);
    int cols = (int)headers.size();
    int t_idx=-1;
    for (int i=0;i<cols;++i) if (headers[i]==target_col) { t_idx=i; break; }
    if (t_idx<0) { std::cerr<<"Target column "<<target_col<<" not found\n"; std::exit(1); }

    std::vector<std::vector<std::string>> rows;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        auto cells = split_csv(line);
        if ((int)cells.size()!=cols) continue;
        rows.emplace_back(std::move(cells));
    }
    if (rows.empty()) { std::cerr<<"No data rows\n"; std::exit(1); }

    // Detect numeric feature columns (exclude target)
    std::vector<int> feat_idx;
    int sample = std::min<int>(200, rows.size());
    for (int c=0;c<cols;++c) {
        if (c==t_idx) continue;
        int ok=0,tr=0; float tmp;
        for (int r=0;r<sample;++r) {
            const auto& s = rows[r][c];
            if (s.empty() || s=="Exempt") continue;
            tr++;
            if (parse_float(s, tmp)) ok++;
        }
        if (tr>0 && ok >= (int)(0.8f*tr)) feat_idx.push_back(c);
        if ((int)feat_idx.size()>=max_features) break;
    }
    if (feat_idx.empty()) { std::cerr<<"No numeric feature columns detected\n"; std::exit(1); }

    Dataset ds; ds.D = (int)feat_idx.size();
    ds.X.reserve(rows.size()*ds.D); ds.y.reserve(rows.size());
    for (auto& row: rows) {
        float tvalf; if (!parse_float(row[t_idx], tvalf)) continue;
        int ybin = ((int)std::round(tvalf) == 1) ? 1 : 0;
        std::vector<float> feats; feats.reserve(ds.D);
        bool ok=true;
        for (int c: feat_idx) {
            float v; if (row[c].empty() || row[c]=="Exempt" || !parse_float(row[c], v)) { ok=false; break; }
            feats.push_back(v);
        }
        if (!ok) continue;
        ds.y.push_back(ybin);
        ds.X.insert(ds.X.end(), feats.begin(), feats.end());
    }
    ds.N = (int)ds.y.size();
    if (ds.N==0) { std::cerr<<"No valid rows after cleaning\n"; std::exit(1); }
    std::cerr<<"Loaded N="<<ds.N<<" D="<<ds.D<<" from "<<path<<"\n";
    return ds;
}

void minmax_scale(std::vector<float>& X, int N, int D) {
    for (int d=0; d<D; ++d) {
        float mn=1e30f, mx=-1e30f;
        for (int i=0;i<N;++i) { float v=X[i*D+d]; mn=std::min(mn,v); mx=std::max(mx,v); }
        float den = (mx>mn)? (mx-mn) : 1.f;
        for (int i=0;i<N;++i) X[i*D+d] = (X[i*D+d]-mn)/den;
    }
}

// Kernels
__global__ void matmul(const float* A, const float* B, float* C, int M, int K, int N) {
    // C[M,N] = A[M,K] * B[K,N]
    int r = blockIdx.y*blockDim.y + threadIdx.y;
    int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (r<M && c<N) {
        float s=0.f;
        for (int k=0;k<K;++k) s += A[r*K+k] * B[k*N+c];
        C[r*N+c]=s;
    }
}
__global__ void add_bias_relu(float* Z, const float* b, int M, int N) {
    int r = blockIdx.y*blockDim.y + threadIdx.y;
    int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (r<M && c<N) {
        float v = Z[r*N+c] + b[c];
        Z[r*N+c] = v>0.f? v:0.f;
    }
}
__global__ void add_bias(float* Z, const float* b, int M, int N) {
    int r = blockIdx.y*blockDim.y + threadIdx.y;
    int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (r<M && c<N) Z[r*N+c] += b[c];
}
__global__ void sigmoid_inplace(float* A, int n) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i<n) { float x=A[i]; A[i] = 1.f/(1.f+expf(-x)); }
}
__global__ void outer_product(const float* a, const float* b, float* C, int M, int N) {
    // C[M,N] = a[M]*b[N]
    int r = blockIdx.y*blockDim.y + threadIdx.y;
    int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (r<M && c<N) C[r*N+c] = a[r]*b[c];
}
__global__ void bce_dz2(const float* A2, const int* y, float* dZ2, int N) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i<N) dZ2[i] = A2[i] - (float)y[i];
}
__global__ void relu_backward(float* dZ, const float* Z, int n) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i<n) dZ[i] = (Z[i]>0.f) ? dZ[i] : 0.f;
}
__global__ void reduce_cols_mean(const float* dZ, float* db, int M, int N) {
    int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c<N) {
        float s=0.f; for (int r=0;r<M;++r) s += dZ[r*N+c];
        db[c] = s / (float)M;
    }
}
__global__ void matmul_AT_B(const float* A, const float* B, float* C, int M, int K, int N) {
    // C[K,N] = A^T[K,M] * B[M,N]
    int r = blockIdx.y*blockDim.y + threadIdx.y; // K
    int c = blockIdx.x*blockDim.x + threadIdx.x; // N
    if (r<K && c<N) {
        float s=0.f;
        for (int i=0;i<M;++i) s += A[i*K+r] * B[i*N+c];
        C[r*N+c] = s;
    }
}
__global__ void scale_inplace(float* A, float alpha, int n) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i<n) A[i] *= alpha;
}
__global__ void sgd_update(float* W, const float* dW, float lr, int n) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i<n) W[i] -= lr * dW[i];
}

// Args
struct Args { std::string csv, target="action_taken"; int features=14, epochs=200, hidden=32; float lr=0.05f; unsigned seed=5; };
Args parse_args(int argc, char** argv) {
    Args a;
    for (int i=1;i<argc;++i) {
        std::string k=argv[i];
        auto need=[&](const char* name){ if (i+1>=argc){std::cerr<<"Missing "<<name<<"\n"; std::exit(1);} return std::string(argv[++i]); };
        if (k=="--csv") a.csv=need("--csv");
        else if (k=="--target") a.target=need("--target");
        else if (k=="--features") a.features=std::stoi(need("--features"));
        else if (k=="--epochs") a.epochs=std::stoi(need("--epochs"));
        else if (k=="--hidden") a.hidden=std::stoi(need("--hidden"));
        else if (k=="--lr") a.lr=std::stof(need("--lr"));
        else if (k=="--seed") a.seed=(unsigned)std::stoul(need("--seed"));
        else { std::cerr<<"Unknown arg "<<k<<"\n"; std::exit(1); }
    }
    if (a.csv.empty()) { std::cerr<<"Use --csv path\n"; std::exit(1); }
    return a;
}

float host_accuracy(const std::vector<float>& probs, const std::vector<int>& y) {
    int N=y.size(), ok=0;
    for (int i=0;i<N;++i) ok += ((probs[i]>=0.5f)==(y[i]==1));
    return ok/(float)N;
}

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);
    Dataset ds = load_csv(args.csv, args.target, args.features);
    minmax_scale(ds.X, ds.N, ds.D);

    // Split 70/30
    std::vector<int> perm(ds.N); std::iota(perm.begin(), perm.end(), 0);
    std::mt19937 rng(args.seed); std::shuffle(perm.begin(), perm.end(), rng);
    int Ntr = int(0.7*ds.N), Nte = ds.N - Ntr;
    std::vector<float> Xtr(Ntr*ds.D), Xte(Nte*ds.D);
    std::vector<int> ytr(Ntr), yte(Nte);
    for (int i=0;i<Ntr;++i) {
        int r=perm[i]; ytr[i]=ds.y[r];
        std::copy(ds.X.begin()+r*ds.D, ds.X.begin()+(r+1)*ds.D, Xtr.begin()+i*ds.D);
    }
    for (int i=0;i<Nte;++i) {
        int r=perm[Ntr+i]; yte[i]=ds.y[r];
        std::copy(ds.X.begin()+r*ds.D, ds.X.begin()+(r+1)*ds.D, Xte.begin()+i*ds.D);
    }

    int D=ds.D, H=args.hidden, O=1;

    // Allocate device buffers
    float *d_Xtr,*d_Xte,*d_W1,*d_b1,*d_W2,*d_b2,*d_Z1,*d_A1,*d_Z2,*d_A2,*d_dZ2,*d_dA1,*d_dZ1,*d_dW2,*d_db2,*d_dW1,*d_db1;
    int *d_ytr;
    CHECK_CUDA(cudaMalloc(&d_Xtr, Ntr*D*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Xte,  Nte*D*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_W1,   H*D*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b1,   H*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_W2,   H*sizeof(float))); // vector
    CHECK_CUDA(cudaMalloc(&d_b2,   sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Z1,   Ntr*H*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_A1,   Ntr*H*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Z2,   Ntr*O*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_A2,   Ntr*O*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_dZ2,  Ntr*O*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_dA1,  Ntr*H*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_dZ1,  Ntr*H*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_dW2,  H*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_db2,  O*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_dW1,  H*D*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_db1,  H*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_ytr,  Ntr*sizeof(int)));

    CHECK_CUDA(cudaMemcpy(d_Xtr, Xtr.data(), Ntr*D*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Xte, Xte.data(), Nte*D*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_ytr, ytr.data(), Ntr*sizeof(int), cudaMemcpyHostToDevice));

    // Init weights
    std::mt19937 gen(args.seed);
    std::normal_distribution<float> nd(0.f,1.f);
    auto init = [&](int n, float s){ std::vector<float> w(n); for (int i=0;i<n;++i) w[i]=nd(gen)*s; return w; };
    auto W1 = init(H*D, std::sqrt(2.f/(D+H))); std::vector<float> b1(H,0.f);
    auto W2 = init(H,   std::sqrt(2.f/(H+O))); std::vector<float> b2(1,0.f);
    CHECK_CUDA(cudaMemcpy(d_W1, W1.data(), H*D*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b1, b1.data(), H*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_W2, W2.data(), H*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b2, b2.data(), sizeof(float), cudaMemcpyHostToDevice));

    dim3 blk(16,16);
    dim3 grd1((H+blk.x-1)/blk.x, (Ntr+blk.y-1)/blk.y);
    dim3 grd2((O+blk.x-1)/blk.x, (Ntr+blk.y-1)/blk.y);
    int nTr = Ntr*O, nA1 = Ntr*H, nZ1 = Ntr*H;

    for (int ep=1; ep<=args.epochs; ++ep) {
        // Forward
        matmul<<<grd1, blk>>>(d_Xtr, d_W1, d_Z1, Ntr, D, H);
        add_bias_relu<<<grd1, blk>>>(d_Z1, d_b1, Ntr, H);
        // A1 stored in Z1; copy to A1 to keep Z1 for backprop
        CHECK_CUDA(cudaMemcpy(d_A1, d_Z1, Ntr*H*sizeof(float), cudaMemcpyDeviceToDevice));
        // Z2 = A1 * W2 + b2
        matmul<<<dim3((1+blk.x-1)/blk.x, (Ntr+blk.y-1)/blk.y), blk>>>(d_A1, d_W2, d_Z2, Ntr, H, 1);
        add_bias<<<grd2, blk>>>(d_Z2, d_b2, Ntr, 1);
        CHECK_CUDA(cudaMemcpy(d_A2, d_Z2, Ntr*sizeof(float), cudaMemcpyDeviceToDevice));
        sigmoid_inplace<<<(nTr+255)/256,256>>>(d_A2, nTr);

        // Backward
        bce_dz2<<<(Ntr+255)/256,256>>>(d_A2, d_ytr, d_dZ2, Ntr);
        // dW2 = (A1^T * dZ2)/Ntr
        matmul_AT_B<<<dim3((1+blk.x-1)/blk.x,(H+blk.y-1)/blk.y), blk>>>(d_A1, d_dZ2, d_dW2, Ntr, H, 1);
        scale_inplace<<<(H+255)/256,256>>>(d_dW2, 1.0f/float(Ntr), H);
        // db2 = mean(dZ2)
        reduce_cols_mean<<<1,1>>>(d_dZ2, d_db2, Ntr, 1);
        // dA1 = dZ2 * W2^T
        outer_product<<<dim3((H+blk.x-1)/blk.x,(Ntr+blk.y-1)/blk.y), blk>>>(d_dZ2, d_W2, d_dA1, Ntr, H);
        // dZ1 = dA1 * relu'(Z1)
        CHECK_CUDA(cudaMemcpy(d_dZ1, d_dA1, Ntr*H*sizeof(float), cudaMemcpyDeviceToDevice));
        relu_backward<<<(nZ1+255)/256,256>>>(d_dZ1, d_Z1, nZ1);
        // dW1 = (Xtr^T * dZ1)/Ntr
        matmul_AT_B<<<dim3((H+blk.x-1)/blk.x,(D+blk.y-1)/blk.y), blk>>>(d_Xtr, d_dZ1, d_dW1, Ntr, D, H);
        scale_inplace<<<(D*H+255)/256,256>>>(d_dW1, 1.0f/float(Ntr), D*H);
        // db1 = mean over rows
        reduce_cols_mean<<<(H+255)/256,256>>>(d_dZ1, d_db1, Ntr, H);

        // SGD step
        sgd_update<<<(D*H+255)/256,256>>>(d_W1, d_dW1, args.lr, D*H);
        sgd_update<<<(H+255)/256,256>>>(d_b1, d_db1, args.lr, H);
        sgd_update<<<(H+255)/256,256>>>(d_W2, d_dW2, args.lr, H);
        sgd_update<<<1,256>>>(d_b2, d_db2, args.lr, 1);

        if (ep%25==0 || ep==1) {
            std::cout << "Epoch " << ep << "/"<<args.epochs << " done\n";
        }
    }

    // Inference on test (CPU for simplicity)
    std::vector<float> h_W1(H*D), h_b1(H), h_W2(H), h_b2(1);
    CHECK_CUDA(cudaMemcpy(h_W1.data(), d_W1, H*D*sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_b1.data(), d_b1, H*sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_W2.data(), d_W2, H*sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_b2.data(), d_b2, sizeof(float), cudaMemcpyDeviceToHost));

    std::vector<float> probs(Nte);
    for (int i=0;i<Nte;++i) {
        std::vector<float> a1(H);
        for (int h=0; h<H; ++h) {
            float s=0.f;
            for (int d=0; d<D; ++d) s += Xte[i*D+d]*h_W1[h*D+d];
            s += h_b1[h];
            a1[h] = s>0.f? s:0.f;
        }
        float z2 = h_b2[0];
        for (int h=0; h<H; ++h) z2 += a1[h]*h_W2[h];
        float p = 1.f/(1.f+std::exp(-z2));
        probs[i]=p;
    }
    float acc = host_accuracy(probs, yte);
    std::cout << "Test accuracy: " << acc << std::endl;
    return 0;
}
