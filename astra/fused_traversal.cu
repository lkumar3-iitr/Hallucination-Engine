// Isolated prototype: fixed reference lattice with per-ray first-hit exit.
__device__ float voxel(const float* v, int x, int y, int z, int nx, int ny, int nz) {
    return x<0 || y<0 || z<0 || x>=nx || y>=ny || z>=nz ? 0.f : v[(z*ny+y)*nx+x];
}
__device__ float sample(const float* v, float x, float y, float z, int nx, int ny, int nz) {
    int ix=(int)floorf(x), iy=(int)floorf(y), iz=(int)floorf(z);
    float a=x-ix, b=y-iy, c=z-iz, value=0;
    for(int k=0;k<2;k++) for(int j=0;j<2;j++) for(int i=0;i<2;i++)
        value += voxel(v,ix+i,iy+j,iz+k,nx,ny,nz)*(i?a:1-a)*(j?b:1-b)*(k?c:1-c);
    return value;
}
extern "C" __global__ void march(const float* v, const float* rays, const float* origin,
    const float* lo, const float* hi, float spacing, int nx, int ny, int nz, int n,
    float* points, unsigned char* hits) {
    int id=blockDim.x*blockIdx.x+threadIdx.x;
    if(id>=n) return;
    float near=0.f, far=1.e30f, d[3];
    for(int a=0;a<3;a++) {
        d[a]=rays[id*3+a];
        float safe=fabsf(d[a])>1.e-9f?d[a]:1.e-9f;
        float t0=(lo[a]-origin[a])/safe, t1=(hi[a]-origin[a])/safe;
        near=fmaxf(near,fminf(t0,t1)); far=fminf(far,fmaxf(t0,t1));
        points[id*3+a]=0.f;
    }
    hits[id]=0;
    if(far<=near) return;
    float step=spacing*.5f/sqrtf(d[0]*d[0]+d[1]*d[1]+d[2]*d[2]);
    int count=(int)ceilf((far-near)/step)+1;
    float before=0.f;
    for(int s=0;s<count;s++) {
        float t=near+s*step, p[3];
        int dims[3]={nx,ny,nz};
        for(int a=0;a<3;a++) {
            float q=origin[a]+d[a]*t;
            float grid=2.f*(q-lo[a])/(hi[a]-lo[a])-1.f;
            p[a]=((grid+1.f)/2.f)*(dims[a]-1);
        }
        float value=sample(v,p[0],p[1],p[2],nx,ny,nz);
        if(s==0) for(int a=0;a<3;a++) points[id*3+a]=origin[a]+d[a]*(near+step);
        if(value>=.5f && t<=far) {
            float previous=s?before:value;
            float frac=fminf(1.f,fmaxf(0.f,(.5f-previous)/fmaxf(value-previous,1.e-6f)));
            float depth=near+(s?s-1:0)*step+frac*step;
            for(int a=0;a<3;a++) points[id*3+a]=origin[a]+d[a]*depth;
            hits[id]=1; return;
        }
        before=value;
    }
}
