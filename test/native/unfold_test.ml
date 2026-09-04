(* aten.unfold.default: a sliding-window view along one named axis. See
   unfold.ml's own header for the "every carried axis shifts one step toward
   N" rule this pins against hand-computed values. *)

open Compute_fixtures

let params ~axis ~size ~step =
  {
    Unfold.Unfold.axis;
    size = Dim.extent size;
    step = Op_config.Pos.of_int step;
  }

(* The simplest case: only [C] holds real data (a plain 8-element vector),
   so the appended window axis (always [C] in the OUTPUT) pushes the real
   data one step out to [W] -- [Unfold]'s own [source_of W = C] rule. *)
let%expect_test "Direct: unfold a plain vector onto W, overlapping windows" =
  let module U = Unfold.Unfold.Compute (Direct) in
  let x_shape = s1c 8 in
  let x = Tensor.materialize x_shape (fun c -> float_of_int (chan c)) in
  let p = params ~axis:Axis.W ~size:3 ~step:2 in
  Format.printf "%a@." (pp_result Tensor.pp)
    (eval_tensor (Unfold.Unfold.output_shape ~x_shape p) (U.pixel p ~x));
  (* windows start at 0, 2, 4: [0,1,2], [2,3,4], [4,5,6] *)
  [%expect {| tensor f32 [W=3 C=3] {0, 1, 2, 2, 3, 4, 4, 5, ...} |}]

(* Mirrors the corpus's own HaloAttn shape (eca_halonext26ts): the axis
   being unfolded is [H], so the window COUNT lands on [D] -- exercising a
   shift through axes other than the all-C-to-W case above. *)
let%expect_test "Direct: unfold H onto D, non-overlapping windows" =
  let module U = Unfold.Unfold.Compute (Direct) in
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:6 ~w:1 ~c:1 in
  let x = Tensor.materialize x_shape (fun c -> float_of_int (row c)) in
  let p = params ~axis:Axis.D ~size:2 ~step:2 in
  Format.printf "%a@." (pp_result Tensor.pp)
    (eval_tensor (Unfold.Unfold.output_shape ~x_shape p) (U.pixel p ~x));
  [%expect {| tensor f32 [D=3 H=1 W=1 C=2] {0, 1, 2, 3, 4, 5} |}]

let%expect_test "Direct: unfold rejects axis=C (reserved for the window offset)"
    =
  let x_shape = s1c 8 in
  let p = params ~axis:Axis.C ~size:3 ~step:1 in
  Format.printf "%a@." (pp_result Tensor.pp)
    (eval_tensor (Unfold.Unfold.output_shape ~x_shape p) (fun _ -> assert false));
  [%expect
    {| unfold axis must not be C (got C): C is reserved for the window offset |}]

let%expect_test "Direct: unfold rejects an input that already uses all six axes"
    =
  let x_shape = Vec6.shape ~n:2 ~t:1 ~d:1 ~h:1 ~w:1 ~c:8 in
  let p = params ~axis:Axis.W ~size:3 ~step:1 in
  Format.printf "%a@." (pp_result Tensor.pp)
    (eval_tensor (Unfold.Unfold.output_shape ~x_shape p) (fun _ -> assert false));
  [%expect
    {| unfold axis W: input already uses all six axes (N has extent > 1), no room for the new window axis |}]
