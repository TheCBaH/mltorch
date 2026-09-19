(* [Ne_scalar]'s Symbolic/Kernel route (P6.4/P6.5): [Eval_symbolic.run] builds an
   ordinary float [Stage.t] from the [SEMANTICS]-generic 0./1. formula, and
   [Graph_builder.ne_scalar] declares the output edge [Bool]. Bool storage is
   admitted at the Kernel boundary (a [Nonzero_bool] result conversion), so
   the kernel builds and its stored output is a genuine Bool payload. input [2, NaN, +inf] against 2.
   pins NaN (unequal to everything) and the signed-zero case on this route. *)

open Graph_ir

let shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:3

let x =
  Tensor.materialize shape (fun c ->
      [| 2.; Float.nan; Float.infinity |].(Dim.to_int (Vec6.get c Axis.C)))

let build =
  Graph_builder.(
    build ~name:"ne_scalar_symbolic" ~outputs:(fun r -> [ r ])
    @@
    let* x = input ~shape ~name:"x" () in
    ne_scalar ~name:"out" 2. x)
  |> Err.or_raise ~pp_error:Graph_builder.pp_error

let%expect_test
    "Symbolic -> Kernel: Ne_scalar's Bool-declared output is stored as a Bool \
     payload" =
  let kernel =
    Kernel_adapt.of_stage_program (Eval_symbolic.run build)
    |> Err.or_raise ~pp_error:Kernel_adapt.pp_error
  in
  let result =
    Kernel_eval.run kernel ~bind:(fun id ->
        if Tensor_id.equal id (List.hd build.Graph.inputs) then Some x else None)
    |> Err.or_raise ~pp_error:Kernel_eval.pp_error
  in
  (match Tensor_id.Map.find_opt (List.hd build.Graph.outputs) result with
  | Some t -> Format.printf "%a@." Tensor.pp t
  | None -> print_endline "MISSING");
  [%expect {| tensor bool [C=3] {0, 1, 1} |}]
