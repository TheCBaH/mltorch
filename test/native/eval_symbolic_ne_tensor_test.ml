(* [Ne_tensor]'s Symbolic/Kernel route (P6.4/P6.5): [Eval_symbolic.run] builds an
   ordinary float [Stage.t] from the [SEMANTICS]-generic 0./1. formula, and
   [Graph_builder.ne_tensor] declares the output edge [Bool]. Bool storage is
   admitted at the Kernel boundary (a [Nonzero_bool] result conversion), so
   the kernel builds and its stored output is a genuine Bool payload. inputs a=[NaN, 0, 1] against b=[NaN, -0, 2]
   pins NaN (unequal to everything) and the signed-zero case on this route. *)

open Graph_ir

let shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:3

let a =
  Tensor.materialize shape (fun c ->
      [| Float.nan; 0.; 1. |].(Dim.to_int (Vec6.get c Axis.C)))

let b =
  Tensor.materialize shape (fun c ->
      [| Float.nan; -0.; 2. |].(Dim.to_int (Vec6.get c Axis.C)))

let build =
  Graph_builder.(
    build ~name:"ne_tensor_symbolic" ~outputs:(fun r -> [ r ])
    @@
    let* a = input ~shape ~name:"a" () in
    let* b = input ~shape ~name:"b" () in
    ne_tensor ~name:"out" a b)
  |> Err.or_raise ~pp_error:Graph_builder.pp_error

let%expect_test
    "Symbolic -> Kernel: Ne_tensor's Bool-declared output is stored as a Bool \
     payload" =
  let kernel =
    Kernel_adapt.of_stage_program (Eval_symbolic.run build)
    |> Err.or_raise ~pp_error:Kernel_adapt.pp_error
  in
  let result =
    Kernel_eval.run kernel ~bind:(fun id ->
        match List.combine build.Graph.inputs [ a; b ] with
        | pairs -> List.assoc_opt id pairs)
    |> Err.or_raise ~pp_error:Kernel_eval.pp_error
  in
  (match Tensor_id.Map.find_opt (List.hd build.Graph.outputs) result with
  | Some t -> Format.printf "%a@." Tensor.pp t
  | None -> print_endline "MISSING");
  [%expect {| tensor bool [C=3] {1, 0, 1} |}]
