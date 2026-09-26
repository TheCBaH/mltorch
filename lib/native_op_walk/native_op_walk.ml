(* The native-operation random walk: each op's config space lives WITH the op
   (its [Walk] submodule in lib/native/ops); this lib assembles those into full
   walk subjects and runs them through the shared [Walk_core.Walk] loop with the
   native [verify] (Direct vs Symbolic). Hand-written per op — no code generation.
*)

(* Re-exported: this file is the library's interface, so a consumer that runs a
   walk with a verifier of its own can otherwise never name the subject it is
   handed. *)
module Subject = Native_subject

type op = (module Walk_core.Walk.Op with type subject = Native_subject.t)

(* Alphabetical by walker target. [find] resolves by target rather than index,
   so this registry has no positional contract. *)
let all_walks : op list =
  [
    (module Adaptive_avg_pool2d_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
    (module Add_nwalk.M : Walk_core.Walk.Op with type subject = Native_subject.t);
    (* The I64 exact-carrier arm ([Compute_i64]), not [Add_nwalk]'s own
       default float-pixel one -- both operands declared I64. *)
    (module Add_i64_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
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
    (* Its own first walk (design §2's own unwalked-op list): the Bool-storage
       output lands through [Nonzero_bool] on an ordinary float stage,
       already-working machinery, not gated on T6.0. *)
    (module Bitwise_not_nwalk.M : Walk_core.Walk.Op
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
    (module Cumsum_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
    (* Unlike [Div_scalar], this random tensor divisor is synthesized nonzero,
       so the walk remains informative rather than flaky. *)
    (module Div_nwalk.M : Walk_core.Walk.Op with type subject = Native_subject.t);
    (* Unlike [Div], this scalar divisor comes from a fixed candidate list that
       omits zero. *)
    (module Div_scalar_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
    (* Its own first walk: Bool-storage output via [Nonzero_bool]. An exact
       scalar match is measure zero against a random draw, so this checks
       Direct/Symbolic agreement on "false" as much as "true" -- exactly as
       informative either way. *)
    (module Eq_scalar_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
    (module Eq_tensor_nwalk.M : Walk_core.Walk.Op
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
    (* Its own first walk: Bool-storage output via [Nonzero_bool]. *)
    (module Gt_scalar_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
    (module Hardsigmoid_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
    (module Hardswish_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
    (module Hardtanh_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
    (* [index]'s values are drawn as random non-negative in-range int64s, not
       arbitrary floats -- this walk proves Direct/Symbolic agreement on
       [Compute.pixel], not ATen's negative-index normalization (already
       pinned by hand-derived fixtures elsewhere). *)
    (module Index_tensor_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
    (* The affine operands are options, so this walk builds all four graph
       shapes rather than merely checking arithmetic. *)
    (module Layer_norm_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
    (module Linear_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
    (module Lstm_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
    (module Max_dim_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
    (module Max_pool2d_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
    (module Max_pool2d_with_indices_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
    (module Mean_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
    (module Mul_nwalk.M : Walk_core.Walk.Op with type subject = Native_subject.t);
    (* The I64 exact-carrier arm ([Compute_i64]), not [Mul_nwalk]'s own
       default float-pixel one -- both operands declared I64. *)
    (module Mul_i64_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
    (module Mul_scalar_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
    (* The explicit [i64_to_float]/[i64_load] promotion arm, not
       [Mul_scalar_nwalk]'s own default float-pixel one -- the operand
       declared I64, output stays F32. *)
    (module Mul_scalar_i64_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
    (* Its own first walk: Bool-storage output via [Nonzero_bool]. *)
    (module Ne_scalar_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
    (module Ne_tensor_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
    (* This pattern/mode walk derives valid configurations from its shape, so it
       exercises structural padding without admitting invalid candidates. *)
    (module Pad_nwalk.M : Walk_core.Walk.Op with type subject = Native_subject.t);
    (module Permute_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
    (* The I64 exact-carrier arm ([Compute_i64]), not [Permute_nwalk]'s own
       default float-pixel one -- the input declared I64. *)
    (module Permute_i64_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
    (module Pow_nwalk.M : Walk_core.Walk.Op with type subject = Native_subject.t);
    (module Relu_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
    (module Reshape_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
    (* The I64 exact-carrier arm ([Compute_i64]/[Tensor.i64_load]), not
       [Reshape_nwalk]'s own default float-pixel one -- the input declared
       I64. *)
    (module Reshape_i64_nwalk.M : Walk_core.Walk.Op
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
    (* The I64 exact-carrier arm ([Compute_i64]), not [Sub_nwalk]'s own
       default float-pixel one -- both operands declared I64. *)
    (module Sub_i64_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
    (module Sum_nwalk.M : Walk_core.Walk.Op with type subject = Native_subject.t);
    (* The [Bool] target's real ATen cast, Bool-storage output via
       [Nonzero_bool]. *)
    (module To_copy_bool_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
    (* The [Float] target's I64-promotion arm. *)
    (module To_copy_float_i64_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
    (* The only walk with a config-dependent output count: it exercises the
       variable-arity builder and per-ordinal evaluator. *)
    (module Unbind_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
    (module Upsample_bicubic2d_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
    (module Upsample_bilinear2d_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
    (module Upsample_nearest2d_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
    (module Vector_norm_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
    (* Appended to preserve the historical per-index walk seeds above.  [find]
       resolves target names, so this has no public ordering contract. *)
    (module Batch_norm_no_stats_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
    (* Appended for the same reason (plan S6/T6.6): a zero-input factory,
       F32-fixed, its own first walk. *)
    (module Eye_nwalk.M : Walk_core.Walk.Op with type subject = Native_subject.t);
    (module Zeros_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
    (* Appended for the same reason (plan S6/T6.3): [To_copy]'s [Long] target,
       its own first walk. *)
    (module To_copy_long_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
    (* Appended for the same reason (plan S6/T6.6): [Arange]'s own first
       walks, default float form and exact-I64 form. *)
    (module Arange_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
    (module Arange_i64_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
    (* Appended for the same reason (plan S7/T7.1): [Unbind]'s I64-operand
       arm, exercising [Compute_i64] rather than the default float-pixel
       one. *)
    (module Unbind_i64_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
    (* Appended for the same reason (plan S7/T7.1; design's own unwalked-op
       list): [Split_with_sizes]'s own first walks, default float form and
       I64-operand form. *)
    (module Split_with_sizes_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
    (module Split_with_sizes_i64_nwalk.M : Walk_core.Walk.Op
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
