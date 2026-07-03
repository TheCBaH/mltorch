(* The native-operation random walk: each op's config space lives WITH the op
   (its [Walk] submodule in lib/native/ops); this lib assembles those into full
   walk subjects and runs them through the shared [Walk_core.Walk] loop with the
   native [verify] (Direct vs Symbolic). Hand-written per op — no code generation.
*)

type op = (module Walk_core.Walk.Op with type subject = Native_subject.t)

let all_walks : op list =
  [
    (module Relu_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
    (module Add_nwalk.M : Walk_core.Walk.Op with type subject = Native_subject.t);
    (module Mul_nwalk.M : Walk_core.Walk.Op with type subject = Native_subject.t);
    (module Mean_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
    (module Max_pool2d_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
    (module Avg_pool2d_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
    (module Conv2d_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
    (module Convolution_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
    (module Conv2d_padding_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
  ]

let run (m : op) ~ppf ~pcg ~steps =
  Walk_core.Walk.run m ~verify:Native_verify.run ~ppf ~pcg ~steps
