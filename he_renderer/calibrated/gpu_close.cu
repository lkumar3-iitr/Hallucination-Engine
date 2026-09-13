// Match the installed OpenCV float-image sampler and its zero border.
extern "C" __global__ void source_sample(const float* points, const unsigned char* hits,
    const double* matrix, const float* source, int sw, int sh,
    double fx, double fy, double cx, double cy, double cropx, double cropy,
    float weight, float* layer, int n) {
    int id=blockIdx.x*blockDim.x+threadIdx.x;
    if(id>=n || !hits[id]) return;
    double p[3];
    for(int a=0;a<3;a++) p[a]=matrix[a*4+3]+matrix[a*4]*points[id*3]
        +matrix[a*4+1]*points[id*3+1]+matrix[a*4+2]*points[id*3+2];
    if(p[0]<=1.e-6) return;
    float u=(float)(cx+fx*p[1]/p[0]-cropx), v=(float)(cy-fy*p[2]/p[0]-cropy);
    if(!isfinite(u) || !isfinite(v) || u < -2 || v < -2 || u > sw+1 || v > sh+1) return;
    #ifdef QUANTIZED_LINEAR
        int iu=__float2int_rn(u*32), iv=__float2int_rn(v*32);
        int x=iu>>5, y=iv>>5;
        float a=(iu&31)/32.f, b=(iv&31)/32.f;
    #else
        int x=(int)floorf(u), y=(int)floorf(v);
        float a=u-x, b=v-y;
    #endif
    for(int c=0;c<4;c++) {
        float value=0;
        for(int j=0;j<2;j++) for(int i=0;i<2;i++) {
            int xx=x+i, yy=y+j;
            if(xx>=0 && yy>=0 && xx<sw && yy<sh)
                value+=source[(yy*sw+xx)*4+c]*(i?a:1-a)*(j?b:1-b);
        }
        layer[id*4+c]+=weight*value;
    }
}

extern "C" __global__ void finish(const float* points, const unsigned char* hits,
    const double* transform, const float* layer, const unsigned char* background,
    const double* scene, int use_depth, unsigned char* rgb, float* alpha,
    unsigned char* status, int n) {
    int id=blockIdx.x*blockDim.x+threadIdx.x;
    if(id>=n) return;
    double depth=0;
    for(int a=0;a<3;a++) depth+=(points[id*3+a]-transform[a*4+3])*transform[a*4];
    bool hidden=use_depth && hits[id] && depth>0 && isfinite(scene[id])
        && scene[id]>0 && scene[id]+.05<depth;
    status[id]=(hidden?1:0) | ((!hits[id] || !isfinite(depth) || depth<=0)?2:0);
    alpha[id]=hidden?0:layer[id*4+3];
    float opacity=fminf(1,fmaxf(0,layer[id*4+3]));
    for(int c=0;c<3;c++) {
        float value=layer[id*4+c]+background[id*3+c]*(1-opacity);
        rgb[id*3+c]=hidden?background[id*3+c]:(unsigned char)__float2int_rn(fminf(255,fmaxf(0,value)));
    }
}
