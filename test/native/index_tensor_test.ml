(* `index.Tensor`'s runtime gather: [Index_tensor.Index_tensor.Compute]
   exercised directly against hand-built tensors, per round 11's requirement
   -- the gathered axis at a NON-C position ([W]), with NONZERO values on the
   frame axes that pass straight through ([H]), and a known negative-index
   case. Direct/Symbolic bitwise agreement alone cannot catch a wrong-but-
   shared index-coordinate mapping in [Compute.pixel], since both backends
   run the identical code; only an independent, hand-derived oracle can. *)

open Compute_fixtures

(* self[h,w,c] = 100*h + 10*w + c -- every axis's contribution is a distinct
   decimal digit, so a wrong axis mapping shows up as a wrong digit. *)
let self_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:3 ~c:2

let self_tensor =
  Tensor.materialize self_shape (fun c ->
      float_of_int ((100 * row c) + (10 * col c) + chan c))

(* index = [2, -3]: -3 normalizes to 0 under extent 3 (self's own W extent).
   Its own shape is [C=2] -- rank 1, right-aligned, per the op's scoping. *)
let index_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:2

let index_tensor =
  Tensor.materialize_i64 index_shape (fun c -> if chan c = 0 then 2L else -3L)

let params : Index_tensor.Index_tensor.params = { axis = Axis.W }

module C = Index_tensor.Index_tensor.Compute (Direct)

let%expect_test
    "Direct: Index_tensor gathers along W, H passes through, negative index \
     normalizes" =
  let out_shape =
    Index_tensor.Index_tensor.output_shape ~self_shape ~index_shape params
  in
  Format.printf "%a@." (pp_result Tensor.pp)
    (eval_tensor out_shape
       (C.pixel params ~self_shape ~self:self_tensor ~index:index_tensor));
  [%expect {| tensor f32 [H=2 W=2 C=2] {20, 21, 0, 1, 120, 121, 100, 101} |}]
