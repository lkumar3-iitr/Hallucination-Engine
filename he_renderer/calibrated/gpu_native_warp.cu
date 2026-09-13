extern "C" __global__ void warp(const float* source, int sw, int sh,
    const double* m, double x1, double y1, double x2, double y2,
    const unsigned char* background, unsigned char* image, float* alpha, int w, int h) {
    int id=blockIdx.x*blockDim.x+threadIdx.x;
    if(id>=w*h) return;
    int x=id%w, y=id/w;
    double d=m[6]*x+m[7]*y+m[8];
    double safe=d>1.e-6?d:1;
    double u=(m[0]*x+m[1]*y+m[2])/safe;
    double v=(m[3]*x+m[4]*y+m[5])/safe;
    bool valid=d>1.e-6 && u>=x1 && u<x2 && v>=y1 && v<y2;
    float sx=valid?(float)((u-x1+.5)*sw/(x2-x1)-.5):-1;
    float sy=valid?(float)((v-y1+.5)*sh/(y2-y1)-.5):-1;
    #ifdef QUANTIZED_LINEAR
        int ix=__float2int_rn(sx*32), iy=__float2int_rn(sy*32);
        int xx=ix>>5, yy=iy>>5;
        float a=(ix&31)/32.f, b=(iy&31)/32.f;
    #else
        int xx=(int)floorf(sx), yy=(int)floorf(sy);
        float a=sx-xx,b=sy-yy;
    #endif
    float sampled[4]={0,0,0,0};
    for(int j=0;j<2;j++) for(int i=0;i<2;i++) {
        int px=xx+i,py=yy+j;
        if(px>=0 && py>=0 && px<sw && py<sh)
            for(int c=0;c<4;c++) sampled[c]+=source[(py*sw+px)*4+c]*(i?a:1-a)*(j?b:1-b);
    }
    alpha[id]=sampled[3];
    for(int c=0;c<3;c++) {
        float value=sampled[c]+background[id*3+c]*(1-sampled[3]);
        image[id*3+c]=(unsigned char)__float2int_rn(fminf(255,fmaxf(0,value)));
    }
}
