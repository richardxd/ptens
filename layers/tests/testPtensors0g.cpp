#include "Cnine_base.cpp"
#include "CnineSession.hpp"

#include "LinmapLayers.hpp"
#include "EMPlayers.hpp"
#include "GatherLayers.hpp"


using namespace ptens;
using namespace cnine;


int main(int argc, char** argv){

  cnine_session session;
  #ifdef _WITH_CUDA

  Ptensors0 A=Ptensors0::sequential(2,3);
  Ptensors0 Ag(A,1);
  cout<<A<<endl;

  cout<<linmaps0(A)<<endl;
  cout<<linmaps0(Ag)<<endl;

  //Ptensors0 R=Ptensors0::zeros_like(A);
  //add_linmaps_back(R,A);
  //cout<<R<<endl;

  //Ptensors0 Rg=Ptensors0::zeros_like(Ag);
  //add_linmaps_back(Rg,Ag);
  //cout<<Ag<<Rg<<endl;

  exit(0);

  cout<<linmaps1(A)<<endl;
  cout<<linmaps1(Ag)<<endl;

  cout<<linmaps2(A)<<endl;
  cout<<linmaps2(Ag)<<endl;
  cout<<"-----"<<endl;

  Hgraph G=Hgraph::random(4,0.5);
  cout<<G<<endl;
  Ptensors0 B=Ptensors0::sequential(4,2);
  Ptensors0 Bg(B,1);

  cout<<gather(B,G)<<endl;
  cout<<gather(Bg,G)<<endl;


  #endif

  //Ptensors1 B=Ptensors1::sequential(5,3,3);
  //cout<<B<<endl;
  //cout<<linmaps0(B)<<endl;



}
