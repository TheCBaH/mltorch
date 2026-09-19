(* The shared storage boundary for a float-path value: which signatures are
   storable and how a computed tensor lands. Both consumers ([Kernel_eval] and
   [Stage_program.ground]) go through [Output_spec.store], and [Kernel.create]
   admits exactly [Output_spec.storable]. *)

let shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:4
let sg fmt = Tensor_sig.create ~id:(Tensor_id.of_int 0) ~name:"" ~shape ~fmt ()

let%expect_test "only unquantized f32 and bool are storable" =
  List.iter
    (fun (name, fmt) -> Fmt.pr "%-4s %b@." name (Output_spec.storable (sg fmt)))
    [
      ("f32", Payload.Fmt Payload.F32);
      ("bool", Payload.Fmt Payload.Bool);
      ("i64", Payload.Fmt Payload.I64);
      ("f16", Payload.Fmt Payload.F16);
      ("f64", Payload.Fmt Payload.F64);
    ];
  [%expect
    {|
    f32  true
    bool true
    i64  false
    f16  false
    f64  false |}]

let%expect_test "a Bool signature stores canonical bytes, an F32 one is as is" =
  let cells = [| 0.; 5.; Float.nan; -0. |] in
  let computed =
    Tensor.materialize shape (fun c -> cells.(Dim.to_int (Vec6.get c Axis.C)))
  in
  Fmt.pr "%a@." Tensor.pp
    (Output_spec.store (sg (Payload.Fmt Payload.Bool)) computed);
  Fmt.pr "same tensor for f32: %b@."
    (Output_spec.store (sg (Payload.Fmt Payload.F32)) computed == computed);
  [%expect
    {|
    tensor bool [C=4] {0, 1, 1, 0}
    same tensor for f32: true |}]
