(* The Const-SSA core is deliberately archive-free.  These tests exercise its
   typed plan contract before rewrite state starts producing plan values. *)

open Graph_ir

let t_ = Tensor_id.of_int
let value id = Const_ssa.Value_id.of_tensor_id (t_ id)
let f32 = Payload.Fmt Payload.F32
let sig_ id shape = Tensor_sig.create ~id:(t_ id) ~name:"" ~shape ~fmt:f32 ()

let ramp shape =
  Tensor.materialize shape (fun c ->
      float_of_int
        ((Dim.to_int (Vec6.get c Axis.H) * 10) + Dim.to_int (Vec6.get c Axis.W)))

let swap_hw =
  [
    (Axis.N, Axis.N);
    (Axis.T, Axis.T);
    (Axis.D, Axis.D);
    (Axis.H, Axis.W);
    (Axis.W, Axis.H);
    (Axis.C, Axis.C);
  ]

let pp_result pp = function
  | Ok () -> Format.printf "ok@."
  | Error e -> Format.printf "%a@." pp (Err.Error.kind e)

let%expect_test "Const-SSA: captured 6D input and permute export" =
  let input = sig_ 1 (Vec6.shape ~n:2 ~t:3 ~d:4 ~h:5 ~w:6 ~c:7) in
  let output = sig_ 2 (Vec6.shape ~n:2 ~t:3 ~d:4 ~h:6 ~w:5 ~c:7) in
  let plan =
    match
      Const_ssa.add Const_ssa.empty ~id:(value 1)
        (Const_ssa.Leaf
           {
             leaf = Const_ssa.Captured (Const_ssa.Capture.of_string "weight");
             output = input;
           })
    with
    | Ok plan ->
        Const_ssa.add plan ~id:(value 2)
          (Const_ssa.Apply
             {
               op =
                 Graph_ir.Permute { Permute.Permute.perm = swap_hw; x = t_ 1 };
               output;
             })
    | Error e -> Error e
  in
  (match plan with
  | Error e -> Format.printf "%a@." Const_ssa.pp_error (Err.Error.kind e)
  | Ok plan ->
      pp_result Const_ssa.pp_error (Const_ssa.validate plan);
      Format.printf "%a@." Const_ssa.pp plan);
  [%expect
    {|
    ok
    t1 = captured "weight"
    t2 = permute x=t1 perm=[H<-W, W<-H] |}]

let%expect_test
    "Const-SSA: typed literals and malformed definitions are rejected" =
  let scalar = sig_ 3 (Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:1) in
  let two = sig_ 4 (Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:2) in
  let literal = ramp scalar.shape in
  let bad_literal = ramp two.shape in
  let good =
    Const_ssa.add Const_ssa.empty ~id:(value 3)
      (Const_ssa.Leaf { leaf = Const_ssa.Literal literal; output = scalar })
  in
  (match good with
  | Ok _ -> Format.printf "literal: ok@."
  | Error e ->
      Format.printf "literal: %a@." Const_ssa.pp_error (Err.Error.kind e));
  pp_result Const_ssa.pp_error
    (Const_ssa.add Const_ssa.empty ~id:(value 3)
       (Const_ssa.Leaf { leaf = Const_ssa.Literal bad_literal; output = scalar })
    |> Result.map (fun _ -> ()));
  let missing =
    Const_ssa.add Const_ssa.empty ~id:(value 4)
      (Const_ssa.Apply
         {
           op = Graph_ir.Permute { Permute.Permute.perm = swap_hw; x = t_ 99 };
           output = two;
         })
  in
  pp_result Const_ssa.pp_error (Result.map (fun _ -> ()) missing);
  let unsupported =
    Const_ssa.add Const_ssa.empty ~id:(value 3)
      (Const_ssa.Apply
         { op = Graph_ir.Relu { Pointwise.Relu.x = t_ 3 }; output = scalar })
  in
  pp_result Const_ssa.pp_error (Result.map (fun _ -> ()) unsupported);
  let vector = sig_ 4 (Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:2) in
  let forged = sig_ 5 scalar.shape in
  let bad_broadcast =
    let open Err.Syntax in
    let* plan =
      Const_ssa.add Const_ssa.empty ~id:(value 3)
        (Const_ssa.Leaf { leaf = Const_ssa.Literal literal; output = scalar })
    in
    let* plan =
      Const_ssa.add plan ~id:(value 4)
        (Const_ssa.Leaf
           { leaf = Const_ssa.Literal bad_literal; output = vector })
    in
    Const_ssa.add plan ~id:(value 5)
      (Const_ssa.Apply
         {
           op = Graph_ir.Add { Pointwise.Bin.a = t_ 3; b = t_ 4 };
           output = forged;
         })
  in
  pp_result Const_ssa.pp_error (Result.map (fun _ -> ()) bad_broadcast);
  [%expect
    {|
    literal: ok
    literal for t3 does not match its declared signature
    Const-SSA value t4 refers to undefined operand t99
    Relu is not a Const-SSA operation
    Const-SSA output signature for t5 disagrees with its operation |}]

let%expect_test "Const-SSA: deterministic plan order and export remapping" =
  let shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:1 in
  let a = sig_ 9 shape and b = sig_ 2 shape in
  let payload = ramp shape in
  let store =
    match
      Constant_store.bind_literal Constant_store.empty ~tensor:a payload
    with
    | Error e -> Error e
    | Ok store ->
        Constant_store.bind_captured store ~tensor:b
          (Const_ssa.Capture.of_string "b")
  in
  (match store with
  | Error e -> Format.printf "%a@." Constant_store.pp_error (Err.Error.kind e)
  | Ok store -> (
      Format.printf "%a@." Constant_store.pp store;
      match
        Constant_store.restrict_and_rename_exports store (fun id ->
            if Tensor_id.equal id (t_ 9) then Some (t_ 4)
            else if Tensor_id.equal id (t_ 2) then None
            else Some id)
      with
      | Error e ->
          Format.printf "%a@." Constant_store.pp_error (Err.Error.kind e)
      | Ok packed -> Format.printf "packed:@,%a@." Constant_store.pp packed));
  [%expect
    {|
    exports:
    t2 -> t2
    t9 -> t9
    plan:
    t2 = captured "b"
    t9 = literal
    packed:
    exports:
    t4 -> t9
    plan:
    t2 = captured "b"
    t9 = literal |}]

let%expect_test "Const-SSA: materializes a captured permute once" =
  let source = sig_ 1 (Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:3 ~c:1) in
  let output = sig_ 2 (Vec6.shape ~n:1 ~t:1 ~d:1 ~h:3 ~w:2 ~c:1) in
  let store =
    match
      Constant_store.bind_captured Constant_store.empty ~tensor:source
        (Const_ssa.Capture.of_string "w")
    with
    | Error e ->
        failwith
          (Format.asprintf "%a" Constant_store.pp_error (Err.Error.kind e))
    | Ok store ->
        Constant_store.bind_apply store ~tensor:output
          (Graph_ir.Permute { Permute.Permute.perm = swap_hw; x = t_ 1 })
        |> Err.or_raise ~pp_error:Constant_store.pp_error
  in
  let resolver = function
    | capture
      when Const_ssa.Capture.equal capture (Const_ssa.Capture.of_string "w") ->
        Err.return (ramp source.shape)
    | capture -> Err.fail (`Missing_capture capture)
  in
  match Const_ssa_materialize.materialize resolver store with
  | Error e ->
      Format.printf "%a@." Const_ssa_materialize.pp_error (Err.Error.kind e)
  | Ok (store, report) ->
      Format.printf "captures=%d applies=%d cache_hits=%d@." report.captures
        report.applies report.cache_hits;
      Format.printf "%a@." Tensor.pp
        (Tensor_id.Map.find (t_ 2) (Constant_store.materialized store));
      [%expect
        {|
    captures=1 applies=1 cache_hits=1
    tensor f32 [H=3 W=2 C=1] {0, 10, 1, 11, 2, 12} |}]

let%expect_test "Const-SSA: materializing a preloaded capture is idempotent" =
  let source = sig_ 1 (Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:2) in
  let payload = ramp source.shape in
  let store =
    Constant_store.bind_captured Constant_store.empty ~tensor:source
      (Const_ssa.Capture.of_string "w")
    |> Err.or_raise ~pp_error:Constant_store.pp_error
    |> fun store ->
    Constant_store.bind_materialized store ~tensor:source payload
    |> Err.or_raise ~pp_error:Constant_store.pp_error
  in
  match
    Const_ssa_materialize.materialize
      (fun capture -> Err.fail (`Missing_capture capture))
      store
  with
  | Error e ->
      Format.printf "%a@." Const_ssa_materialize.pp_error (Err.Error.kind e)
  | Ok (store, report) ->
      Format.printf "captures=%d applies=%d cache_hits=%d present=%b@."
        report.captures report.applies report.cache_hits
        (Tensor_id.Map.mem (t_ 1) (Constant_store.materialized store));
      [%expect {| captures=0 applies=0 cache_hits=1 present=true |}]

let%expect_test "Const-SSA: materialization follows a packed export" =
  let source = sig_ 1 (Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:3 ~c:1) in
  let output = sig_ 2 (Vec6.shape ~n:1 ~t:1 ~d:1 ~h:3 ~w:2 ~c:1) in
  let store =
    Constant_store.bind_captured Constant_store.empty ~tensor:source
      (Const_ssa.Capture.of_string "w")
    |> Err.or_raise ~pp_error:Constant_store.pp_error
    |> fun store ->
    Constant_store.bind_apply store ~tensor:output
      (Graph_ir.Permute { Permute.Permute.perm = swap_hw; x = t_ 1 })
    |> Err.or_raise ~pp_error:Constant_store.pp_error
    |> fun store ->
    Constant_store.restrict_and_rename_exports store (fun id ->
        if Tensor_id.equal id (t_ 2) then Some (t_ 5) else None)
    |> Err.or_raise ~pp_error:Constant_store.pp_error
  in
  match
    Const_ssa_materialize.materialize
      (fun capture ->
        if Const_ssa.Capture.equal capture (Const_ssa.Capture.of_string "w")
        then Err.return (ramp source.shape)
        else Err.fail (`Missing_capture capture))
      store
  with
  | Error e ->
      Format.printf "%a@." Const_ssa_materialize.pp_error (Err.Error.kind e)
  | Ok (store, report) ->
      let cache = Constant_store.materialized store in
      Format.printf "applies=%d current=%b historical=%b@." report.applies
        (Tensor_id.Map.mem (t_ 5) cache)
        (Tensor_id.Map.mem (t_ 2) cache);
      [%expect {| applies=1 current=true historical=false |}]
