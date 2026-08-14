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
    (module Linear_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
    (module Batch_norm_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
    (module Rms_norm_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
    (module Max_pool2d_with_indices_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
    (module Reshape_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
    (module Clamp_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
    (module Clone_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
    (module Hardtanh_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
    (module Add_scalar_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
    (* Registered, unlike [Div]: [Div]'s hazard is a random TENSOR divisor
       hitting zero, whereas this op's divisor comes from a fixed candidate list
       that omits zero, so the walk stays informative rather than flaky. *)
    (module Div_scalar_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
    (* The only walk whose OUTPUT COUNT varies with its config, so it is the one
       that fuzzes the variable-arity builder and the per-ordinal eval loop
       rather than only an op's pixel math. *)
    (module Unbind_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
  ]

(* [native_op_walk.ml] shares the library's name, so it IS the library's
   interface: nothing else in this directory is reachable from outside it. A
   focused test that wants ONE walk therefore cannot name its module, and
   picking by index into [all_walks] would silently follow the list around.
   Selecting by the walk's own [target] is stable under reordering. *)
let find target =
  List.find_opt
    (fun (m : op) ->
      let module M =
        (val m : Walk_core.Walk.Op with type subject = Native_subject.t)
      in
      M.target = target)
    all_walks

let run (m : op) ~ppf ~pcg ~steps =
  Walk_core.Walk.run m ~verify:Native_verify.run ~ppf ~pcg ~steps
