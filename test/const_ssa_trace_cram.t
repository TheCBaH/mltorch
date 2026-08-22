The materialized-fold baseline for the current Native4D corpus.

`const-ssa-trace` constructs the pre-Const-SSA materialized state, so these are
the operations the former direct-fold path actually cached. Its exit status
also checks every observed operation against the closed Const-SSA registry; a
new unsupported family names its first full configuration and fails before the
manifest can be updated accidentally.

The frozen corpus entry itself is a successful Native4D conversion; keeping
this check beside the trace prevents a model that merely lowers from silently
remaining in the Native4D baseline.

  $ ../bin/native_graph.exe to4d --pt2 "$PT2_DATA/mobilenetv2_050/mobilenetv2_050.pt2" --fold | head -2
  canonical native: nodes=100
  native4d: nodes=100 inputs=107

  $ ../bin/native_graph.exe const-ssa-trace --pt2 "$PT2_DATA/mobilenetv2_050/mobilenetv2_050.pt2" > first.txt
  $ ../bin/native_graph.exe const-ssa-trace --pt2 "$PT2_DATA/mobilenetv2_050/mobilenetv2_050.pt2" > second.txt
  $ cmp first.txt second.txt
  $ diff -u const_ssa_native4d_fold_manifest.txt first.txt
