Explicit Const-SSA materialization is the boundary before direct evaluation.
The frozen six-dimensional MobileNetV2 constants lower to a four-dimensional
Native4D graph; evaluating both forms from the archive must agree exactly.

  $ ../bin/native_graph.exe to4d --pt2 "$PT2_DATA/mobilenetv2_050/mobilenetv2_050.pt2" --fold --input "$PT2_DATA/mobilenetv2_050/inputs.pt" | tail -1
  materialized Native4D: applies=469, native agreement: count=1000 max_abs=0 bias=0 mean_abs=0 rmse=0 stddev=0 relative_l2=0 cosine_similarity=1 cosine_distance=0
