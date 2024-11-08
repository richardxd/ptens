pybind11::class_<BatchedAtomsPackBase>(m,"batched_atomspack")

  .def(py::init([](const vector<AtomsPack>& x){return BatchedAtomsPackBase(x);}))
  .def(py::init([](const vector<vector<vector<int> > >& x){return BatchedAtomsPackBase(x);}))

  .def_static("cat",[](const vector<BatchedAtomsPackBase> v){return BatchedAtomsPackBase::cat(v);})

  .def("__len__",&BatchedAtomsPackBase::size)
  .def("__getitem__",[](const BatchedAtomsPackBase& x, const int i){return x[i];})
  .def("torch",[](const BatchedAtomsPackBase& x){return x.as_vecs();})

  .def("nrows0",[](const BatchedAtomsPackBase& x){return x.nrows0();})
  .def("nrows1",[](const BatchedAtomsPackBase& x){return x.nrows1();})
  .def("nrows2",[](const BatchedAtomsPackBase& x){return x.nrows2();})

  .def("nrows0",[](const BatchedAtomsPackBase& x, const int i){return x.nrows0(i);})
  .def("nrows1",[](const BatchedAtomsPackBase& x, const int i){return x.nrows1(i);})
  .def("nrows2",[](const BatchedAtomsPackBase& x, const int i){return x.nrows2(i);})

  .def("offset0",[](const BatchedAtomsPackBase& x, const int i){return x.offset0(i);})
  .def("offset1",[](const BatchedAtomsPackBase& x, const int i){return x.offset1(i);})
  .def("offset2",[](const BatchedAtomsPackBase& x, const int i){return x.offset2(i);})

  .def("str",&BatchedAtomsPackBase::str,py::arg("indent")="")
  .def("__str__",&BatchedAtomsPackBase::str,py::arg("indent")="");





