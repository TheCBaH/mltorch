(* The native-operation random walk: each op's config space lives WITH the op
   (its [Walk] submodule in lib/native/ops); this lib assembles those into full
   walk subjects and runs them through the shared [Walk_core.Walk] loop with the
   native [verify] (Direct vs Symbolic). Hand-written per op — no code generation.
*)

type op = (module Walk_core.Walk.Op with type subject = Native_subject.t)

(* Alphabetical by walker target. [find] resolves by target rather than index,
   so this registry has no positional contract. *)
let all_walks : op list =
  [
    (module Adaptive_avg_pool2d_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
    (module Add_nwalk.M : Walk_core.Walk.Op with type subject = Native_subject.t);
    (module Add_scalar_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
    (module Amax_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
    (module Avg_pool2d_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
    (module Batch_norm_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
    (module Batched_matmul_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
    (module Bmm_nwalk.M : Walk_core.Walk.Op with type subject = Native_subject.t);
    (module Clamp_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
    (module Clone_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
    (module Conv2d_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
    (module Conv2d_padding_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
    (module Convolution_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
    (* Unlike [Div_scalar], this random tensor divisor is synthesized nonzero,
       so the walk remains informative rather than flaky. *)
    (module Div_nwalk.M : Walk_core.Walk.Op with type subject = Native_subject.t);
    (* Unlike [Div], this scalar divisor comes from a fixed candidate list that
       omits zero. *)
    (module Div_scalar_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
    (* The drawn shape is the TARGET; the source forces [H] to 1, so every
       step exercises a real broadcast read, not the degenerate identity
       case. *)
    (module Expand_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
    (* [Direct.erf] and grounded [Symbolic]'s [erf] share one implementation
       (see expr.ml), so this walk proves staging/scheduling agreement, not
       the erf approximation's accuracy against ATen -- that proof is
       native_bridge_test.ml's verify_print. *)
    (module Gelu_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
    (module Hardsigmoid_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
    (module Hardswish_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
    (module Hardtanh_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
    (* The affine operands are options, so this walk builds all four graph
       shapes rather than merely checking arithmetic. *)
    (module Layer_norm_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
    (module Linear_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
    (module Max_pool2d_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
    (module Max_pool2d_with_indices_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
    (module Mean_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
    (module Mul_nwalk.M : Walk_core.Walk.Op with type subject = Native_subject.t);
    (module Mul_scalar_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
    (* This pattern/mode walk derives valid configurations from its shape, so it
       exercises structural padding without admitting invalid candidates. *)
    (module Pad_nwalk.M : Walk_core.Walk.Op with type subject = Native_subject.t);
    (module Permute_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
    (module Pow_nwalk.M : Walk_core.Walk.Op with type subject = Native_subject.t);
    (module Relu_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
    (module Reshape_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
    (module Rms_norm_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
    (* Mask presence is a graph-shape distinction; its correlated tuple makes
       cross-operand mismatches unrepresentable. *)
    (module Sdpa_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
    (module Sigmoid_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
    (module Silu_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
    (* Its bounds are derived from the current extent, making empty and
       out-of-range configurations unreachable. *)
    (module Slice_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
    (module Softmax_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
    (module Sqrt_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
    (module Sub_nwalk.M : Walk_core.Walk.Op with type subject = Native_subject.t);
    (module Sum_nwalk.M : Walk_core.Walk.Op with type subject = Native_subject.t);
    (* The only walk with a config-dependent output count: it exercises the
       variable-arity builder and per-ordinal evaluator. *)
    (module Unbind_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
    (module Upsample_bilinear2d_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
    (module Upsample_nearest2d_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
    (module Vector_norm_nwalk.M : Walk_core.Walk.Op
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
