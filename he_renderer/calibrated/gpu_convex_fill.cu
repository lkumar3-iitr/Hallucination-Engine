// Restricted adaptation of OpenCV drawing.cpp FillConvexPoly/LineIterator:
// https://github.com/opencv/opencv/blob/4.x/modules/imgproc/src/drawing.cpp
// Integer coordinates, four corners, LINE_8, shift=0, all vertices in bounds.
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * The name of Intel Corporation may not be used to endorse or promote
//   products derived from this software without specific prior written permission.
// This software is provided by the copyright holders and contributors "as is"
// and any express or implied warranties, including, but not limited to, the
// implied warranties of merchantability and fitness for a particular purpose
// are disclaimed. In no event shall the Intel Corporation or contributors be
// liable for any direct, indirect, incidental, special, exemplary, or
// consequential damages (including, but not limited to, procurement of
// substitute goods or services; loss of use, data, or profits; or business
// interruption) however caused and on any theory of liability, whether in
// contract, strict liability, or tort (including negligence or otherwise)
// arising in any way out of the use of this software, even if advised of the
// possibility of such damage.

__device__ void edge_line(int* mask, int size, int x, int y, int bx, int by) {
    if (bx < x) { int t=x; x=bx; bx=t; t=y; y=by; by=t; }
    int dx=bx-x, dy=by-y, sy=dy<0 ? -1:1;
    dy=abs(dy);
    bool vertical=dy>dx;
    int major=vertical?dy:dx, minor=vertical?dx:dy;
    int error=major-2*minor;
    for(int i=0;i<=major;i++) {
        atomicOr(mask+y*size+x,1);
        bool diagonal=error<0;
        error+=-2*minor+(diagonal?2*major:0);
        if(vertical) { y+=sy; if(diagonal) x++; }
        else { x++; if(diagonal) y+=sy; }
    }
}

extern "C" __global__ void convex_fill(const int* polygons,
    const unsigned char* needed, int* mask, int count, int size, int separate) {
    int id=blockIdx.x*blockDim.x+threadIdx.x;
    if(id>=count || !needed[id]) return;
    const int* p=polygons+id*8;
    if(separate) mask+=id*size*size;
    int first=0, bottom=p[1];
    for(int i=0;i<4;i++) {
        // Defensive bounds guard: runtime normalization guarantees this domain.
        if(p[2*i]<0 || p[2*i]>=size || p[2*i+1]<0 || p[2*i+1]>=size) return;
        if(p[2*i+1]<p[2*first+1]) first=i;
        bottom=max(bottom,p[2*i+1]);
    }
    for(int i=0;i<4;i++) {
        int j=(i+3)%4;
        edge_line(mask,size,p[2*j],p[2*j+1],p[2*i],p[2*i+1]);
    }
    int index[2]={first,first}, end[2]={p[2*first+1],p[2*first+1]}, remaining=4;
    long long x[2]={-65536,-65536}, step[2]={0,0};
    for(int y=p[2*first+1];y<=bottom;y++) {
        for(int side=0;side<2;side++) {
            if(y>=end[side]) {
                int previous=index[side], direction=side?3:1;
                int next=(previous+direction)%4;
                while(remaining-->0) {
                    int ty=p[2*next+1];
                    if(ty>y) {
                        long long start=(long long)p[2*previous]*65536;
                        long long finish=(long long)p[2*next]*65536;
                        end[side]=ty;
                        step[side]=((finish-start)*2+(ty-y))/(2*(ty-y));
                        x[side]=start;
                        index[side]=next;
                        break;
                    }
                    previous=next;
                    next=(next+direction)%4;
                }
            }
        }
        if(remaining<0) break;
        int left=(int)((min(x[0],x[1])+32768)>>16);
        int right=(int)((max(x[0],x[1])+32768)>>16);
        for(int col=max(left,0);col<=min(right,size-1);col++) atomicOr(mask+y*size+col,1);
        x[0]+=step[0]; x[1]+=step[1];
    }
}
