Parse native_functions.yaml into raw records and verify summary statistics.

  $ ./aten_schema_raw_test.exe native_functions.yaml
  total entries: 2674
  with dispatch:  1649
  structured:     273
  with tags:      930
  conv2d               dispatch=1 structured=false tags=[]
  relu_                dispatch=9 structured=false tags=[pointwise]
  abs.out              dispatch=3 structured=false tags=[pointwise]
  add.out              dispatch=8 structured=true  tags=[pointwise]
  batch_norm           dispatch=0 structured=false tags=[maybe_aliasing_or_mutating]
  chunk                dispatch=2 structured=false tags=[]
  split.Tensor         dispatch=1 structured=false tags=[]
  abs                  dispatch=4 structured=false tags=[core,pointwise]
