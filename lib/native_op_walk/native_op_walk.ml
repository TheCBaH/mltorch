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
    (module Sub_nwalk.M : Walk_core.Walk.Op with type subject = Native_subject.t);
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
    (* Beside RmsNorm, and for the same reason its entry gives: the affine
       operands are options, so the walk's value is that it builds all four
       graph SHAPES, not that it checks the arithmetic. *)
    (module Layer_norm_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
    (* Mask presence, like RmsNorm/LayerNorm's affine operands, is a genuinely
       different graph, not a ones-filled tensor; the shapes it draws are all
       DERIVED from one correlated (batch, heads, wq, wk, e) tuple, so no
       cross-operand mismatch is representable in the first place. *)
    (module Sdpa_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
    (module Max_pool2d_with_indices_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
    (module Reshape_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
    (module Permute_nwalk.M : Walk_core.Walk.Op
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
    (module Silu_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
    (module Sigmoid_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
    (* [Direct.erf] and grounded [Symbolic]'s [erf] share one implementation
       (see expr.ml), so this walk proves staging/scheduling agreement, not
       the erf approximation's accuracy against ATen -- that proof is
       native_bridge_test.ml's verify_print. *)
    (module Gelu_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
    (module Mul_scalar_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
    (module Hardsigmoid_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
    (module Hardswish_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
    (* The only walk that varies a STRUCTURAL parameter alongside the shape:
       its pattern axis steps between padding, cropping and both at once, and
       its mode axis between the constant fill and the reflect mirror. Both are
       correlated with the drawn shape inside the op's [Walk], so an invalid
       config is unreachable rather than filtered. *)
    (module Pad_nwalk.M : Walk_core.Walk.Op with type subject = Native_subject.t);
    (* The other structural walk. Its pattern axis steps between the identity
       selection, both ends, an interior range and two strides, each resolved
       against the drawn extent inside the op's [Walk] -- so the empty and
       out-of-range configurations [output_shape] refuses are unreachable here
       rather than filtered out after the fact. *)
    (module Slice_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
    (* Unlike [Div_scalar], the divisor here is a random TENSOR, drawn via
       [Native_tensor.synth_nonzero] to stay away from the flakiness that kept
       this one out of the list until now (see its own doc comment). *)
    (module Div_nwalk.M : Walk_core.Walk.Op with type subject = Native_subject.t);
    (module Sqrt_nwalk.M : Walk_core.Walk.Op
      with type subject = Native_subject.t);
    (module Bmm_nwalk.M : Walk_core.Walk.Op with type subject = Native_subject.t);
    (module Adaptive_avg_pool2d_nwalk.M : Walk_core.Walk.Op
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
