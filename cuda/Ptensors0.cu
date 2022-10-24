#ifndef _Ptensors0_cu
#define _Ptensors0_cu

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/complex.h>
#include <thrust/tuple.h>

#include "Ptens_base.hpp"
#include "RtensorPack.hpp"
#include "AindexPack.hpp"
#include "CSRmatrix.hpp"
//#include "Ptensors0.hpp"
//#include "Rtensor2_view.hpp"
//#include "Rtensor3_view.hpp"
//#include "Itensor1_view.hpp"
//#include "Itensor2_view.hpp"
//#include "CUDAhelpers.hpp"


__global__ void Ptensors0_reduce0_kernel(float* rarr, const int* rdir, const float* xarr, const int* xdir){
  const int i=blockIdx.x;
  const int c=threadIdx.x;
  rarr[rdir[2*i]+c]+=xarr[xdir[2*i]+c];
}


__global__ void Ptensors0_reduce0_kernel(float* rarr, const int* rdir, const float* xarr, const int* xdir, const int* xiarr, const int* xidir){
  const int i=blockIdx.x;
  const int c=threadIdx.x;
  const int tix=xiarr[xidir[2*i]];
  rarr[rdir[2*i]+c]+=xarr[xdir[2*tix]+c];
}


__global__ void Ptensors0_broadcast0_kernel(float* xarr, const int* xdir, const float* rarr, const int* rdir){
  const int i=blockIdx.x;
  const int c=threadIdx.x;
  xarr[xdir[2*i]+c]+=rarr[rdir[2*i]+c];
}


__global__ void Ptensors0_broadcast0_kernel(float* xarr, const int* xdir, const int* xiarr, const int* xidir, const float* rarr, const int* rdir, const int* bmap){
  const int b=blockIdx.x;
  const int c=threadIdx.x;

  const int boffs=bmap[3*b];
  const int N=bmap[3*b+1];
  const int target=bmap[3*b+2];

  float t=0;
  for(int j=0; j<N; j++){
    const int src=bmap[boffs+2*j];
    const float w=*reinterpret_cast<const float*>(bmap+boffs+2*j+1);
    t+=w*rarr[rdir[2*src]+c];
  }
  xarr[xdir[2*target]+c]+=t;
}


__global__ void Ptensors0_gather_kernel(float* rarr, const int* rdir, const float* xarr, const int* xdir, const float* marr, const int* mdir){
  const int i=blockIdx.x;
  const int c=threadIdx.x;

  const int moffs=mdir[2*i];
  const int N=mdir[2*i+1]/2;
  float t=0;
  for(int j=0; j<N; j++){
    const int jix=*reinterpret_cast<const int*>(marr+moffs+2*j);
    const int jweight=marr[moffs+2*j+1];
    t+=jweight*xarr[xdir[2*jix]+c];
  }
  rarr[rdir[2*i]+c]+=t;
}


// -----------------------------------------------------------------------------------------------------------


namespace ptens{

  void Ptensors0_reduce0_cu(cnine::RtensorPack& R, const cnine::RtensorPack& x, int offs, int n, const cudaStream_t& stream){
    int dev=R.dev;
    PTENS_ASSRT(R.dev==1);
    PTENS_ASSRT(x.dev==1);
    int N=R.size();
    Ptensors0_reduce0_kernel<<<N,n,0,stream>>>(R.arrg,R.dir.garr(dev),x.arrg+offs,x.dir.garr(dev));
  }

  void Ptensors0_reduce0_cu(cnine::RtensorPack& R, const cnine::RtensorPack& x, const AindexPack& list, int offs, int n, const cudaStream_t& stream){
    int dev=R.dev;
    PTENS_ASSRT(R.dev==1);
    PTENS_ASSRT(x.dev==1);
    PTENS_ASSRT(list.dev==1);
    int N=R.size();
    Ptensors0_reduce0_kernel<<<N,n,0,stream>>>(R.arrg,R.dir.garr(dev),x.arrg+offs,x.dir.garr(dev),list.arrg,list.dir.garr(dev));
  }

  void Ptensors0_broadcast0_cu(cnine::RtensorPack& x, const cnine::RtensorPack& R, const int offs, const cudaStream_t& stream){
    int dev=R.dev;
    PTENS_ASSRT(R.dev==1);
    PTENS_ASSRT(x.dev==1);
    Ptensors0_broadcast0_kernel<<<R.size(),x.dim_of(0,0),0,stream>>>
      (x.arrg+offs,x.dir.garr(dev),R.arrg,R.dir.garr(dev));
  }

  void Ptensors0_broadcast0_cu(cnine::RtensorPack& x, const cnine::RtensorPack& R, const AindexPack& list, const int offs, const cudaStream_t& stream){
    int dev=R.dev;
    PTENS_ASSRT(R.dev==1);
    PTENS_ASSRT(x.dev==1);
    PTENS_ASSRT(list.dev==1);
    Ptensors0_broadcast0_kernel<<<R.size(),x.dim_of(0,0),0,stream>>>
      (x.arrg+offs,x.dir.garr(dev),list.arrg,list.dir.garr(dev),R.arrg,R.dir.garr(dev),list.get_barr(1));
  }

  void Ptensors0_gather_cu(cnine::RtensorPack& r, const cnine::RtensorPack& x, const cnine::CSRmatrix<float>& gmap, const cudaStream_t& stream){
    int dev=r.dev;
    PTENS_ASSRT(r.dev==1);
    PTENS_ASSRT(x.dev==1);
    const_cast<cnine::CSRmatrix<float>&>(gmap).to_device(dev);
    Ptensors0_gather_kernel<<<r.size(),x.dim_of(0,0),0,stream>>>
      (r.arrg,r.dir.garr(dev),x.arrg,x.dir.garr(dev),gmap.arrg,gmap.dir.garr(dev));
  }

}

#endif 
