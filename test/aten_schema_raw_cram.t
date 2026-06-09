Parse native_functions.yaml into raw records and verify summary statistics.

  $ ./aten_schema_raw_test.exe native_functions.yaml
  total entries: 2650
  with dispatch:  1628
  structured:     272
  with tags:      826
  conv2d               dispatch=1 structured=false tags=[]
  relu_                dispatch=8 structured=false tags=[pointwise]
  abs.out              dispatch=3 structured=false tags=[pointwise]
  add.out              dispatch=6 structured=true  tags=[pointwise]
  batch_norm           dispatch=0 structured=false tags=[maybe_aliasing_or_mutating]
  chunk                dispatch=2 structured=false tags=[]
  split.Tensor         dispatch=1 structured=false tags=[]
  abs                  dispatch=4 structured=false tags=[core,pointwise]
