extern "C" __global__ void collapsed(const int* vertices, const int* indices,
    int* mask, int* polygons, int* bounds, int count, int size) {
    int id=blockIdx.x*blockDim.x+threadIdx.x;
    if(id>=count) return;
    int lx=size, ly=size, hx=-1, hy=-1;
    for(int i=0;i<4;i++) {
        int v=indices[id*4+i];
        int x=vertices[v*2], y=vertices[v*2+1];
        polygons[id*8+i*2]=x; polygons[id*8+i*2+1]=y;
        lx=min(lx,x); ly=min(ly,y); hx=max(hx,x); hy=max(hy,y);
    }
    bounds[id*4]=lx; bounds[id*4+1]=ly; bounds[id*4+2]=hx; bounds[id*4+3]=hy;
    if(lx==hx || ly==hy)
        for(int y=ly;y<=hy;y++) for(int x=lx;x<=hx;x++) atomicOr(mask+y*size+x,1);
}

extern "C" __global__ void residual(const int* bounds, const int* mask,
    unsigned char* needed, int count, int size) {
    int id=blockIdx.x*blockDim.x+threadIdx.x;
    if(id>=count) return;
    int lx=bounds[id*4], ly=bounds[id*4+1], hx=bounds[id*4+2], hy=bounds[id*4+3];
    needed[id]=0;
    if(lx==hx || ly==hy) return;
    for(int y=ly;y<=hy;y++) for(int x=lx;x<=hx;x++) {
        if(!mask[y*size+x]) { needed[id]=1; return; }
    }
}

extern "C" __global__ void intersections(const unsigned long long* masks,
    const unsigned long long* target, const int* eligible, int chunks, int* output) {
    __shared__ int counts[256];
    int t=threadIdx.x, row=eligible[blockIdx.x], total=0;
    for(int i=t;i<chunks;i+=256) total+=__popcll(masks[row*chunks+i]&target[i]);
    counts[t]=total;
    __syncthreads();
    for(int stride=128;stride;stride>>=1) {
        if(t<stride) counts[t]+=counts[t+stride];
        __syncthreads();
    }
    if(!t) output[blockIdx.x]=counts[0];
}
