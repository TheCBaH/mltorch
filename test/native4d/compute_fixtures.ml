(* Shared rendering/build helpers for compute_test.ml and compute_shape_test.ml
   (split from a single compute_test.ml under the tracked file-size ceiling,
   scripts/check-file-size.sh). Not a test module itself. *)

open Native4d

let s4 ~n ~h ~w ~c = Shape4.of_ints ~n ~h ~w ~c

let build ~outputs m =
  Builder.build ~outputs m |> Err.or_raise ~pp_error:Builder.pp_error

(* Row-major fill, so every element is distinguishable in the output. *)
let seq shape =
  let i = ref (-1.) in
  Tensor.materialize (Shape4.to_vec6 shape) (fun _ ->
      i := !i +. 1.;
      !i)

let values t =
  let (Tensor.Tensor tt) = t in
  let shape = tt.Tensor.shape in
  let acc = ref [] in
  Vec6.iter shape (fun c -> acc := Tensor.read_at t (Vec6.get c) :: !acc);
  List.rev !acc

let pp_values fmt vs =
  Fmt.pf fmt "[%a]"
    (Fmt.list ~sep:(Fmt.any " ") (fun fmt v -> Fmt.pf fmt "%g" v))
    vs

let run_direct ?(constants = []) g ~inputs =
  Eval_direct4.run g ~constants ~inputs
  |> Err.or_raise ~pp_error:Eval_direct4.pp_error

let single g ~inputs ?(constants = []) () =
  let env = run_direct g ~constants ~inputs in
  let out = List.hd g.Graph.Graph.outputs in
  Tensor_id.Map.find out env
