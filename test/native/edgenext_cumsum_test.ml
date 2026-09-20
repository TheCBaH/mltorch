(* A minimal fixture reproducing EdgeNeXt's
   `PositionalEncodingFourier` Bool/cumsum subgraph, traced from the real
   exported graph in
   `modules/devcontainer.pytorch-image-models/models/edgenext_xx_small/
   models/model.json` (nodes 46-51, not downloaded weights -- this
   submodule ships the model.json directly) rather than guessed: a Float
   `zeros` tensor is cast to Bool (`_to_copy.default(dtype=BOOL)`, scalar
   type 12 in the PT2 schema.yaml `ScalarType` enum, confirmed against
   `modules/pytorch/torch/_export/serde/schema.yaml` rather than assumed
   from ATen's own different stable enum), negated
   (`bitwise_not.default`), and the SAME `bitwise_not` output feeds TWO
   `cumsum.default` nodes -- one `dim=1`, one `dim=2` -- each with an
   explicit `dtype=7` (FLOAT, i.e. Float32) argument. This fixture starts
   from a real (non-degenerate) Float input rather than the real graph's
   literal `zeros` (which casts to an all-false mask and would exercise no
   interesting cumsum values), matching [bool_acceptance_test.ml]'s own
   choice for the mask-pattern half of this same subgraph.

   Direct route only; Symbolic/Kernel, Native4D, JSOO and real-ATen
   differential coverage are explicitly NOT part of this fixture. *)

open Graph_ir
open Graph_direct_fixtures

(* [H=2 W=4], 8 elements: kept at [Tensor.pp]'s own truncation threshold so
   every printed value is checked, not just the first 8 of a larger grid. *)
let shape = s 1 1 1 2 4 1

let%expect_test
    "edgenext cumsum acceptance: input -> To_copy(Bool) -> Bitwise_not -> two \
     Float32 cumsums sharing one source" =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"edgenext_cumsum_pattern" ~outputs:(fun (h, w) ->
              [ h; w ])
          @@
          let* x = input ~shape ~name:"x" () in
          let* mask = to_copy ~name:"mask" Pointwise.To_copy.Bool x in
          let* inv = bitwise_not ~name:"inv" mask in
          let* cum_h =
            cumsum ~name:"cum_h" { Reduce.Cumsum.axis = Axis.H } inv
          in
          let* cum_w =
            cumsum ~name:"cum_w" { Reduce.Cumsum.axis = Axis.W } inv
          in
          return (cum_h, cum_w))
    in
    let row = [| [| 0.; 1.; 0.; 2. |]; [| 3.; 0.; 0.; 5. |] |] in
    let x =
      Tensor.materialize shape (fun c ->
          row.(Dim.to_int (Vec6.get c Axis.H)).(Dim.to_int (Vec6.get c Axis.W)))
    in
    let* env =
      lift_eval (Eval_direct.run g ~inputs:(List.combine g.Graph.inputs [ x ]))
    in
    (* [mask]/[inv] are nodes 0/1 in push order (to_copy then bitwise_not),
       read directly rather than through [id_of_name] (which does not know
       these names) -- confirms genuine [Payload.Bool] storage feeds the
       cumsums, not merely the F32-encoded output boundary. *)
    let node_output i =
      match List.nth g.Graph.nodes i with
      | { Node.outputs = id :: _; _ } -> id
      | _ -> assert false
    in
    let mask = Tensor_id.Map.find (node_output 0) env in
    let inv = Tensor_id.Map.find (node_output 1) env in
    let outs =
      List.map (fun oid -> Tensor_id.Map.find oid env) g.Graph.outputs
    in
    Err.return (mask, inv, outs)
  in
  let pp_all fmt (mask, inv, outs) =
    Format.fprintf fmt "mask = %a@.inv  = %a@." Tensor.pp mask Tensor.pp inv;
    List.iter (fun t -> Format.fprintf fmt "%a@." Tensor.pp t) outs
  in
  Format.printf "%a@." (pp_result pp_all) result;
  (* [mask]/[inv] are genuine Bool storage, not F32-encoded 0./1. --
     [mask] is nonzero-test True at every input position EXCEPT the two
     literal zeros, [inv] negates it. Independently hand-computed inclusive
     prefix sums of [inv] read as 0/1, {{1,0,1,0},{0,1,1,0}} -- cum_h sums
     down H (columns), cum_w sums across W (rows), both explicit Float32
     per the real traced graph's own `dtype=7` cumsum argument. *)
  [%expect
    {|
    mask = tensor bool [H=2 W=4 C=1] {0, 1, 0, 1, 1, 0, 0, 1}
    inv  = tensor bool [H=2 W=4 C=1] {1, 0, 1, 0, 0, 1, 1, 0}
    tensor f32 [H=2 W=4 C=1] {1, 0, 1, 0, 1, 1, 2, 0}
    tensor f32 [H=2 W=4 C=1] {1, 1, 2, 2, 0, 1, 2, 2} |}]
