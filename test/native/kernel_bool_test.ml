(* Bool storage at the Kernel boundary. A Bool-declared value pairs with the
   [Nonzero_bool] conversion: its float body is mapped to exactly 0./1. wherever
   a consumer sees it (stored or virtual), and stored as canonical Bool bytes.
   The bodies here are deliberately NOT already 0/1 -- the graph ops' own
   formulas are, which would hide a missing conversion -- so 5., the subnormal
   1e-40 and NaN must all read as 1 through a virtual edge, and the two
   conversion/format pairings that disagree are rejected at construction. *)

let s1c n = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:n
let shape = s1c 5

let sg id fmt =
  Tensor_sig.create ~id:(Tensor_id.of_int id) ~name:"" ~shape ~fmt ()

let f32 = Payload.Fmt Payload.F32
let bool_fmt = Payload.Fmt Payload.Bool

let source id =
  Expr.Value.load
    (Expr_bridge.source_of_id (Tensor_id.of_int id))
    (Expr_bridge.coord_of_vec6 Symbolic.out_vec)

let value id fmt result body =
  {
    Kernel.Value.id = Tensor_id.of_int id;
    sg = sg id fmt;
    computation = Region_group.Ref.Solo (Region_program.pixel body);
    result;
  }

let input id binding fmt =
  { Kernel.Input.id = Tensor_id.of_int id; sg = sg id fmt; binding }

let create ~inputs ~values out =
  Kernel.create ~inputs ~values ~outputs:[ Tensor_id.of_int out ] ()

let cells = [| 0.; 5.; 1e-40; Float.nan; -0. |]

let x =
  Tensor.materialize shape (fun c -> cells.(Dim.to_int (Vec6.get c Axis.C)))

let bind id = if Tensor_id.equal id (Tensor_id.of_int 0) then Some x else None

(* t0 (f32 input) -> t1 (Bool, raw body) -> t2 (f32, reads t1) *)
let chain =
  create
    ~inputs:[ input 0 Kernel.Binding.Caller f32 ]
    ~values:
      [
        value 1 bool_fmt Kernel.Result_conversion.Nonzero_bool (source 0);
        value 2 f32 Kernel.Result_conversion.Round_f32 (source 1);
      ]
    2
  |> Err.or_raise ~pp_error:Kernel.pp_error

let show name = function
  | Some t -> Format.printf "%s = %a@." name Tensor.pp t
  | None -> Format.printf "%s not stored@." name

let%expect_test "stored, fused and on-demand readings agree" =
  let id = Tensor_id.of_int in
  let run k =
    Err.or_raise ~pp_error:Kernel_eval.pp_error (Kernel_eval.run k ~bind)
  in
  let stored = run chain in
  show "stored t1" (Tensor_id.Map.find_opt (id 1) stored);
  show "stored t2" (Tensor_id.Map.find_opt (id 2) stored);
  let plan, _ = Fusion_plan.plan chain in
  let fused =
    Err.or_raise ~pp_error:Kernel_eval.pp_error
      (Kernel_eval.run_plan plan ~bind)
  in
  show "fused t1 " (Tensor_id.Map.find_opt (id 1) fused);
  show "fused t2 " (Tensor_id.Map.find_opt (id 2) fused);
  let on_demand =
    List.init 5 (fun c ->
        Err.or_raise ~pp_error:Kernel_eval.pp_error
          (Kernel_eval.value_at chain ~bind (id 2)
             (Expr.Coord.make ~n:0 ~t:0 ~d:0 ~h:0 ~w:0 ~c)))
  in
  Format.printf "value_at t2 = %a@." Fmt.(list ~sep:(any ", ") float) on_demand;
  [%expect
    {|
    stored t1 = tensor bool [C=5] {0, 1, 1, 1, 0}
    stored t2 = tensor f32 [C=5] {0, 1, 1, 1, 0}
    fused t1  not stored
    fused t2  = tensor f32 [C=5] {0, 1, 1, 1, 0}
    value_at t2 = 0, 1, 1, 1, 0 |}]

let%expect_test
    "a conversion that disagrees with the declared format is rejected" =
  let report r =
    Format.printf "%a@."
      (Core.Pretty.err_result
         ~ok:(fun fmt _ -> Format.pp_print_string fmt "ok")
         ~error:Kernel.pp_error)
      r
  in
  let inputs = [ input 0 Kernel.Binding.Caller f32 ] in
  report
    (create ~inputs
       ~values:
         [ value 1 bool_fmt Kernel.Result_conversion.Round_f32 (source 0) ]
       1);
  report
    (create ~inputs
       ~values:[ value 1 f32 Kernel.Result_conversion.Nonzero_bool (source 0) ]
       1);
  [%expect
    {|
    t1: result conversion round_f32 does not produce bool storage
    t1: result conversion nonzero_bool does not produce f32 storage |}]

let%expect_test "a filled Bool input is canonical Bool storage" =
  let kernel =
    create
      ~inputs:[ input 0 (Kernel.Binding.Filled 2.) bool_fmt ]
      ~values:[ value 1 f32 Kernel.Result_conversion.Round_f32 (source 0) ]
      1
    |> Err.or_raise ~pp_error:Kernel.pp_error
  in
  let result =
    Err.or_raise ~pp_error:Kernel_eval.pp_error
      (Kernel_eval.run kernel ~bind:(fun _ -> None))
  in
  show "t1" (Tensor_id.Map.find_opt (Tensor_id.of_int 1) result);
  [%expect {| t1 = tensor f32 [C=5] {1, 1, 1, 1, 1} |}]

let%expect_test "a second invocation cannot mutate a returned Bool tensor" =
  let id = Tensor_id.of_int in
  let run input =
    Err.or_raise ~pp_error:Kernel_eval.pp_error
      (Kernel_eval.run chain ~bind:(fun i ->
           if Tensor_id.equal i (id 0) then Some input else None))
  in
  let first = run x in
  let inverted =
    Tensor.materialize shape (fun c ->
        if cells.(Dim.to_int (Vec6.get c Axis.C)) = 0. then 7. else 0.)
  in
  let second = run inverted in
  show "first t1 " (Tensor_id.Map.find_opt (id 1) first);
  show "second t1" (Tensor_id.Map.find_opt (id 1) second);
  [%expect
    {|
    first t1  = tensor bool [C=5] {0, 1, 1, 1, 0}
    second t1 = tensor bool [C=5] {1, 0, 0, 0, 1} |}]
