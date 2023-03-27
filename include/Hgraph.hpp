#ifndef _Hgraph
#define _Hgraph

#include <set>
#include "Ptens_base.hpp"
#include "SparseRmatrix.hpp"
#include "AtomsPack.hpp"
#include "AindexPack.hpp"
#include "GatherMap.hpp"
#include "labeled_tree.hpp"

#include <chrono>


namespace ptens{

  //class HgraphSubgraphListCache;


  class Hgraph: public cnine::SparseRmatrix{
  public:

    typedef cnine::labeled_tree<int> labeled_tree;
    typedef cnine::RtensorA RtensorA;

    using cnine::SparseRmatrix::SparseRmatrix;

    RtensorA labels;
    bool is_labeled=false;

    mutable Hgraph* _reverse=nullptr;
    mutable cnine::CSRmatrix<float>* gmap=nullptr; 
    mutable shared_ptr<cnine::GatherMap> bmap;
    mutable vector<AtomsPack*> _nhoods; 
    mutable AtomsPack* _edges=nullptr;
    mutable unordered_map<SparseRmatrix,cnine::array_pool<int>*> subgraphlist_cache;
    //mutable HgraphSubgraphListCache* subgraphlist_cache=nullptr;

    ~Hgraph(){
      // if(_reverse) delete _reverse; // hack!
      for(auto p:_nhoods)
	delete p;
      if(!_edges) delete _edges;
      if(gmap) delete gmap;
      //if(bmap) delete bmap;
      for(auto p:subgraphlist_cache) delete p.second;
      //delete subgraphlist_cache;
    }


  public: // ---- Constructors -------------------------------------------------------------------------------


    Hgraph(const int _n):
      Hgraph(_n,_n){}

    Hgraph(const int _n, const RtensorA& _labels):
      Hgraph(_n,_n){
      PTENS_ASSRT(_labels.dims.size()==1);
      PTENS_ASSRT(_labels.dims[0]==n);
      labels=_labels;
      is_labeled=true;
    }

    Hgraph(const int _n, const initializer_list<pair<int,int> >& list): 
      Hgraph(_n){
      for(auto p:list){
	set(p.first,p.second,1.0);
	set(p.second,p.first,1.0);
      }
    }

    Hgraph(const int _n, const initializer_list<pair<int,int> >& list, const RtensorA& _labels): 
      Hgraph(_n){
      PTENS_ASSRT(_labels.dims.size()==1);
      PTENS_ASSRT(_labels.dims[0]==n);
      labels=_labels;
      is_labeled=true;
      for(auto p:list){
	set(p.first,p.second,1.0);
	set(p.second,p.first,1.0);
      }
    }


  public: // ---- Named Constructors -------------------------------------------------------------------------


    static Hgraph edge_index(const cnine::RtensorA& M, int n=-1){
      PTENS_ASSRT(M.ndims()==2);
      PTENS_ASSRT(M.get_dim(0)==2);
      if(n==-1) n=M.max()+1;
      else PTENS_ASSRT(M.max()<n);
      int nedges=M.get_dim(1);
      Hgraph R(n);
      for(int i=0; i<nedges; i++)
	R.set(M(0,i),M(1,i),1.0);
      return R;
    }

    static Hgraph edge_index(const cnine::RtensorA& M, const cnine::RtensorA& L, int n=-1){
      PTENS_ASSRT(M.ndims()==2);
      PTENS_ASSRT(M.get_dim(0)==2);
      if(n==-1) n=M.max()+1;
      else PTENS_ASSRT(M.max()<n);
      int nedges=M.get_dim(1);
      Hgraph R(n,L);
      for(int i=0; i<nedges; i++)
	R.set(M(0,i),M(1,i),1.0);
      return R;
    }

    static Hgraph random(const int _n, const float p=0.5){
      return cnine::SparseRmatrix::random_symmetric(_n,p);
    }

    static Hgraph randomd(const int _n, const float p=0.5){
      auto R=cnine::SparseRmatrix::random_symmetric(_n,p);
      for(int i=0; i<_n; i++)
	R.set(i,i,1.0);
      return R;
    }

    static Hgraph overlaps(const cnine::array_pool<int>& x, const cnine::array_pool<int>& y){
      Hgraph R(x.size(),y.size());
      auto t0 = std::chrono::system_clock::now();
      if(y.size()<10){
	for(int i=0; i<x.size(); i++){
	  auto v=x(i);
	  for(int j=0; j<y.size(); j++){
	    auto w=y(j);
	    if([&](){for(auto p:v) if(std::find(w.begin(),w.end(),p)!=w.end()) return true; return false;}())
	      R.set(i,j,1.0);
	  }
	}
      }else{
	cout<<"New overlaps"<<endl;
	unordered_map<int,vector<int> > map;
	for(int j=0; j<y.size(); j++){
	  auto w=y(j);
	  for(auto p:w){
	    auto it=map.find(p);
	    if(it==map.end()) map[p]=vector<int>({j});
	    else it->second.push_back(j);
	  }
	  for(int i=0; i<x.size(); i++){
	    auto v=x(i);
	    for(auto p:v){
	      auto it=map.find(p);
	      if(it!=map.end())
		for(auto q:it->second)
		  R.set(i,q,1.0);
	    }
	  }
	}
      }
      auto elapsed=chrono::duration<double,std::milli>(chrono::system_clock::now()-t0).count();
      cout<<"Overlaps between "<<x.size()<<" domains and "<<y.size()<<" domains in "<<to_string(elapsed)<<"ms."<<endl; 
      return R;
    }


  public: // ---- Copying ------------------------------------------------------------------------------------


    Hgraph(const Hgraph& x):
      SparseRmatrix(x), 
      labels(x.labels),
      is_labeled(x.is_labeled){}

    Hgraph(Hgraph&& x):
      SparseRmatrix(std::move(x)),
      labels(std::move(x.labels)),
      is_labeled(x.is_labeled){}

    Hgraph& operator=(const Hgraph& x)=delete;


  public: // ---- Conversions --------------------------------------------------------------------------------


    Hgraph(const cnine::SparseRmatrix& x):
      cnine::SparseRmatrix(x){}

    Hgraph(const cnine::SparseRmatrix& x, const cnine::RtensorA& L):
      cnine::SparseRmatrix(x),
      labels(L),
      is_labeled(true){
      PTENS_ASSRT(labels.dims.size()==1);
      PTENS_ASSRT(labels.dims[0]==n);
    }


  public: // ---- Access -------------------------------------------------------------------------------------


    int getn() const{
      return n;
    }

    void insert(const Hgraph& H, vector<int> v){
      for(auto p:v)
	PTENS_ASSRT(p<n);
      H.for_each_edge([&](const int i, const int j){
	  set(v[i],v[j],1.0);});
    }

    vector<int> neighbors(const int i) const{
      vector<int> r;
      const auto _r=row(i);
      for(auto& p: _r)
	r.push_back(p.first);
      return r;
    }

    const Hgraph& reverse() const{
      if(!_reverse) _reverse=new Hgraph(transp());
      return *_reverse;
    }

    const cnine::CSRmatrix<float>& get_gmap() const{
      if(!gmap) gmap=new cnine::CSRmatrix<float>(csrmatrix());
      return *gmap;
    }

    void for_each_neighbor_of(const int i, std::function<void(const int, const float)> lambda) const{
      const auto& r=row(i);
      for(auto& p: r)
	lambda(p.first,p.second);
    }

    void for_each_edge(std::function<void(const int, const int)> lambda, const bool self=0) const{
      for(auto& p: lists){
	int i=p.first;
	if(self) lambda(i,i);
	p.second->forall_nonzero([&](const int j, const float v){
	    lambda(i,j);});
      }
    }

    void for_each_edge(std::function<void(const int, const int, const float)> lambda, const bool self=0) const{
      for(auto& p: lists){
	int i=p.first;
	if(self) lambda(i,i,1.0);
	p.second->forall_nonzero([&](const int j, const float v){
	    lambda(i,j,v);});
      }
    }

    void forall_edges(std::function<void(const int, const int, const float)> lambda, const bool self=0) const{
      for(auto& p: lists){
	int i=p.first;
	if(self) lambda(i,i,1.0);
	p.second->forall_nonzero([&](const int j, const float v){
	    lambda(i,j,v);});
      }
    }

    AtomsPack nhoods(const int i) const{
      if(_nhoods.size()==0) _nhoods.push_back(new AtomsPack(n));
      for(int j=_nhoods.size(); j<=i; j++){
	const AtomsPack& prev=*_nhoods.back();
	assert(prev.size()==n);
	AtomsPack* newlevel=new AtomsPack();
	for(int i=0; i<prev.size(); i++){
	  vector<int> v=prev(i);
	  std::set<int> w;
	  for(auto p:v){
	    w.insert(p);
	    for(auto q: const_cast<Hgraph&>(*this).row(p))
	      w.insert(q.first);
	  }
	  newlevel->push_back(w);
	}
	_nhoods.push_back(newlevel);
      }
      
      return AtomsPack(*_nhoods[i]);
    }

    const AtomsPack& edges(){
      if(!_edges) make_edges();
      return *_edges;
    }

    void make_edges(){
      delete _edges;
      _edges=new AtomsPack();
      for_each_edge([&](const int i, const int j, const float v){
	  _edges->push_back({i,j});});
    }


  public: // ---- Operations ---------------------------------------------------------------------------------


    AtomsPack merge(const AtomsPack& x) const{
      PTENS_ASSRT(m==x.size());
      AtomsPack R;
      for(int i=0; i<n; i++){
	std::set<int> w;
	for(auto q: const_cast<Hgraph&>(*this).row(i)){
	  auto a=x[q.first];
	  for(auto p:a)
	    w.insert(p);
	}
	R.push_back(w);
      }
      return R;
    }


    pair<AindexPack,AindexPack> intersects(const AtomsPack& inputs, const AtomsPack& outputs, const bool self=0) const{
      PTENS_ASSRT(outputs.size()==n);
      PTENS_ASSRT(inputs.size()==m);
      AindexPack in_indices;
      AindexPack out_indices;
      forall_edges([&](const int i, const int j, const float v){
	  Atoms in=inputs[j];
	  Atoms out=outputs[i];
	  Atoms common=out.intersect(in);
	  auto p=out(common);
	  in_indices.push_back(j,in(common));
	  out_indices.push_back(i,out(common));
	}, self);
      //out_indices.bmap=new cnine::GatherMap(get_bmap());
      if(!bmap) bmap=std::shared_ptr<cnine::GatherMap>(new cnine::GatherMap(broadcast_map())); 
      out_indices.bmap=bmap; //new cnine::GatherMap(get_bmap());
      return make_pair(in_indices, out_indices);
    }


    cnine::GatherMap broadcast_map() const{
      int nlists=0;
      int nedges=0;
      for(auto q:lists)
	if(q.second->size()>0){
	  nlists++;
	  nedges+=q.second->size();
	}

      cnine::GatherMap R(nlists,nedges);
      int i=0;
      int m=0;
      int tail=3*nlists;
      for(auto q:lists){
	int len=q.second->size();
	if(len==0) continue;
	R.arr[3*i]=tail;
	R.arr[3*i+1]=len;
	R.arr[3*i+2]=q.first;
	int j=0;
	for(auto p:*q.second){
	  R.arr[tail+2*j]=m++;
	  *reinterpret_cast<float*>(R.arr+tail+2*j+1)=p.second;
	  j++;
	}
	tail+=2*len;
	i++;
      }
      return R;
    }


  public: // ---- Subgraphs ----------------------------------------------------------------------------------


    labeled_tree greedy_spanning_tree(const int root=0) const{
      PTENS_ASSRT(getn()>0);
      vector<bool> matched(n,false);
      matched[root]=true;
      labeled_tree* T=greedy_spanning_tree(root,matched);
      return std::move(*T);
    }

    labeled_tree* greedy_spanning_tree(const int v, vector<bool>& matched) const{
      PTENS_ASSRT(v<n);
      labeled_tree* r=new labeled_tree(v);
      for(auto& p: row(v)){
	if(p.second==0) continue;
	if(matched[p.first]) continue;
	matched[p.first]=true;
	r->children.push_back(greedy_spanning_tree(p.first,matched));
      }
      return r;
    }
 

  public: // ---- I/O -----------------------------------------------------------------------------------------


    string classname() const{
      return "Hgraph";
    }

    string str(const string indent="") const{
      ostringstream oss;
      oss<<indent<<"Hgraph on "<<n<<" vertices:"<<endl;
      oss<<dense().str(indent+"  ")<<endl;
      if(is_labeled) oss<<labels.str(indent+"  ")<<endl;
      return oss.str();
    }

  };


}
  

namespace std{

  template<>
  struct hash<ptens::Hgraph>{
  public:
    size_t operator()(const ptens::Hgraph& x) const{
      if(x.is_labeled) return (hash<cnine::SparseRmatrix>()(x)<<1)^hash<cnine::RtensorA>()(x.labels);
      return hash<cnine::SparseRmatrix>()(x);
    }
  };
}

//namespace ptens{
//class HgraphSubgraphListCache: public unordered_map<Hgraph,AindexPack>{
//public:
//};
//}


#endif


   /*
    AindexPack all_subgraphs_isomorphic_to(const Hgraph& H      
      tforest matches;
      tnode S=H.greedy_spanning_tree();
      auto traversal=S.indexed_depth_first_traversal();
      vector<int> assignment(traversal.size(),-1);

      for(int i=0; i<n; i++){
	tnode* T=new tnode(i);
	matches.push_back(T);
	if(!complete_match(*T,matches,traversal,assignment)){
	  delete T;
	  matches.pop_back();
	}
      }

      AindexPack R;
      for(auto p:matches)
	p->forall_maximal_paths([&](const vector<int>& x){
	    R.push_back(x);});
      return R;
    }

    
    // try to match next vertex in traversal to w
    bool complete_match(tnode& node, const tforest& matches, 
      const vector<int>& traversal, vector<int> assignment, int m){

      PTENS_ASSRT(m<traversal.size());
      const int v=traversal[m].first;

      for(auto& p:H.row(v)){
	if(assignment[p.first]==-1) continue;
	if(p.second!=(*this)(w,assignment[p.first])) return;
      }
      for(auto& p:row(w)){
	auto it=std::find(assignment.begin(),assignment.end(),p.first);
	if(it==assignment.end()) return;
	if(p.second!=H(v,traversal[it-assignment.begin()])) return;
      }

      assignment[v]=w;
      if(m==traversal.size()-1)
	return !T.contains_some_permutation_of(assignment));

      //const int newv=traversal[m+1].first;
      const int parentv=traversal[traversal[m+1].second];
      const int parentw=assignment[parentv];

      // try to match next vertex in traversal to each neighbor of parentw  
      for(auto& p:row(parentw)){
	if(std::find(assignment.begin(),assignment.end(),p.first)!=assignment.end()) continue;
	auto subtree=match(traversal,assignment,p.first,m+1);
	if(subtree!=nullptr) R.children[p.first]=subtree;
      }
      if(R.children.size()>0) return R;

      delete R;
      return nullptr;
    }
    */

    /*
    AindexPack subgraphs(const Hgraph& H){
      
      PrefixTree T;
      PrefixTree S=H.greedy_spanning_tree(0);
      auto traversal=S.indexed_depth_first_traversal(0);
      //vector<int> traversal({0});
      //S.depth_first_traversal([&](const int i){traversal.push_back(i);});

      for(int i=0; i<n; i++){
	vector<int> assignment(traversal.size(),-1);
	PrefixTree* sub=match(traversal,assignment,i,0);
	if(sub!=nullptr) T.children[i]=sub;
      }

      AindexPack R;
      T.forall_maximal_paths([&](const vector<int>& x){
	  int n=x.size()
	  assert(df.size()==n);
	  vector<int> r(n);
	  for(int i=0; i<n; i++)
	    v[df[i]]=x[i];
	  R.push_back(r);
	});
      return R;
    }


    // try to match next vertex in traversal to w
    PrefixTree* match(const PrefixTree* root, const vector<int>& traversal, vector<int> assignment, 
      PrefixTree* parent, const int w, const int m){
      PTENS_ASSRT(m<traversal.size());
      const int v=traversal[m].first;

      for(auto& p:H.row(v)){
	if(assignment[p.first]==-1) continue;
	if(p.second!=(*this)(w,assignment[p.first])) return;
      }
      for(auto& p:row(w)){
	auto it=std::find(assignment.begin(),assignment.end(),p.first);
	if(it==assignment.end()) return;
	if(p.second!=H(v,traversal[it-assignment.begin()])) return;
      }

      assignment[v]=w;
      if(m==traversal.size()-1){
	if(!T.contains_some_permutation_of(vertices)) parent[w]=new PrefixTree(); 
	return;
      }

      //const int newv=traversal[m+1].first;
      const int parentv=traversal[traversal[m+1].second];
      const int parentw=assignment[parentv];

      // try to match next vertex in traversal to each neighbor of parentw  
      PrefixTree* R=new PrefixTree();
      
      for(auto& p:row(parentw)){
	if(std::find(assignment.begin(),assignment.end(),p.first)!=assignment.end()) continue;
	auto subtree=match(traversal,assignment,p.first,m+1);
	if(subtree!=nullptr) R.children[p.first]=subtree;
      }
      if(R.children.size()>0) return R;

      delete R;
      return nullptr;
    }
    */
    //const cnine::GatherMap& get_bmap() const{
    //if(!bmap) bmap=new cnine::GatherMap(broadcast_map());
    //return *bmap;
    //}


