(* Gate 7 item 4 (Symbolic/Kernel half): the same EdgeNeXt Bool/cumsum
   subgraph [edgenext_cumsum_test.ml] proves on Direct, run through
   Symbolic/Kernel instead. [inv] (the [Bitwise_not] output) is shared by
   BOTH cumsum consumers, so [Eval_symbolic] cannot fuse it into either
   consumer's own expression -- it must become a genuine stored
   [Kernel.Value.t], declared [Bool] by [Graph_builder.bitwise_not]. Bool
   storage is admitted at the Kernel boundary (a [Nonzero_bool] result
   conversion, canonical Bool bytes), so the kernel builds, runs, stores real
   Bool [mask]/[inv], and both Float32 cumsums must equal the Direct route's
   independently hand-computed prefix sums. *)

open Graph_ir

let shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:4 ~c:1

let build =
  Graph_builder.(
    build ~name:"edgenext_cumsum_symbolic" ~outputs:(fun (h, w) -> [ h; w ])
    @@
    let* x = input ~shape ~name:"x" () in
    let* mask = to_copy ~name:"mask" Pointwise.To_copy.Bool x in
    let* inv = bitwise_not ~name:"inv" mask in
    let* cum_h = cumsum ~name:"cum_h" { Reduce.Cumsum.axis = Axis.H } inv in
    let* cum_w = cumsum ~name:"cum_w" { Reduce.Cumsum.axis = Axis.W } inv in
    return (cum_h, cum_w))
  |> Err.or_raise ~pp_error:Graph_builder.pp_error

let row = [| [| 0.; 1.; 0.; 2. |]; [| 3.; 0.; 0.; 5. |] |]

let x =
  Tensor.materialize shape (fun c ->
      row.(Dim.to_int (Vec6.get c Axis.H)).(Dim.to_int (Vec6.get c Axis.W)))

let node_output i =
  match List.nth build.Graph.nodes i with
  | { Node.outputs = id :: _; _ } -> id
  | _ -> assert false

let%expect_test
    "Symbolic -> Kernel: EdgeNeXt's shared Bool [inv] intermediate is stored \
     as Bool and both Float32 cumsums match the Direct route" =
  let stage_program = Eval_symbolic.run build in
  let kernel =
    Kernel_adapt.of_stage_program stage_program
    |> Err.or_raise ~pp_error:Kernel_adapt.pp_error
  in
  let result =
    Kernel_eval.run kernel ~bind:(fun id ->
        if Tensor_id.equal id (List.hd build.Graph.inputs) then Some x else None)
    |> Err.or_raise ~pp_error:Kernel_eval.pp_error
  in
  let show name id =
    match Tensor_id.Map.find_opt id result with
    | Some t -> Format.printf "%s = %a@." name Tensor.pp t
    | None -> Format.printf "%s MISSING@." name
  in
  show "mask" (node_output 0);
  show "inv " (node_output 1);
  List.iter (show "out ") build.Graph.outputs;
  [%expect
    {|
    mask = tensor bool [H=2 W=4 C=1] {0, 1, 0, 1, 1, 0, 0, 1}
    inv  = tensor bool [H=2 W=4 C=1] {1, 0, 1, 0, 0, 1, 1, 0}
    out  = tensor f32 [H=2 W=4 C=1] {1, 0, 1, 0, 1, 1, 2, 0}
    out  = tensor f32 [H=2 W=4 C=1] {1, 1, 2, 2, 0, 1, 2, 2} |}]
