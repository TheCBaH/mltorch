(* The Const-SSA core is deliberately archive-free.  These tests exercise its
   typed plan contract before rewrite state starts producing plan values. *)

open Graph_ir

let arena = Ground_expr.Arena.create ()
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

let%expect_test "Const-SSA: captured input and reshape export" =
  let input = sig_ 1 (Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:3 ~c:1) in
  let output = sig_ 2 (Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:6) in
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
                 Graph_ir.Reshape
                   {
                     Reshape.Reshape.params = { shape = output.shape };
                     x = t_ 1;
                   };
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
    t2 = reshape x=t1 params={shape=[C=6]} |}]

let%expect_test "Const-SSA: reshape grounds to the row-major source coordinate"
    =
  let input = sig_ 1 (Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:3 ~c:1) in
  let output = sig_ 2 (Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:6) in
  let store =
    Constant_store.bind_captured Constant_store.empty ~tensor:input
      (Const_ssa.Capture.of_string "w")
    |> Err.or_raise ~pp_error:Constant_store.pp_error
    |> fun store ->
    Constant_store.bind_apply store ~tensor:output
      (Graph_ir.Reshape
         { Reshape.Reshape.params = { shape = output.shape }; x = t_ 1 })
    |> Err.or_raise ~pp_error:Constant_store.pp_error
  in
  Vec6.iter output.shape (fun coord ->
      match Const_ssa_symbolic.ground arena store (t_ 2) coord with
      | None -> Format.printf "%a -> none@." Vec6.pp_coord coord
      | Some expr ->
          Format.printf "%a -> %a@." Vec6.pp_coord coord Ground_expr.pp expr);
  [%expect
    {|
    (0) -> capture."w"(0)
    (1) -> capture."w"(1,0)
    (2) -> capture."w"(2,0)
    (3) -> capture."w"(1,0,0)
    (4) -> capture."w"(1,1,0)
    (5) -> capture."w"(1,2,0) |}]

let%expect_test "Const-SSA: materializes a captured reshape once" =
  let source = sig_ 1 (Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:3 ~c:1) in
  let output = sig_ 2 (Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:6) in
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
          (Graph_ir.Reshape
             { Reshape.Reshape.params = { shape = output.shape }; x = t_ 1 })
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
    tensor f32 [C=6] {0, 1, 2, 10, 11, 12} |}]

let%expect_test "Const-SSA: captured input and expand export" =
  let input = sig_ 1 (Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:3 ~c:1) in
  let output = sig_ 2 (Vec6.shape ~n:2 ~t:1 ~d:1 ~h:2 ~w:3 ~c:1) in
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
                 Graph_ir.Expand
                   {
                     Pointwise.Expand.params = { size = output.shape };
                     x = t_ 1;
                   };
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
    t2 = expand x=t1 params={size=[N=2 T=1 D=1 H=2 W=3 C=1]} |}]

let%expect_test "Const-SSA: expand grounds by collapsing the broadcast axis" =
  let input = sig_ 1 (Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:3 ~c:1) in
  let output = sig_ 2 (Vec6.shape ~n:2 ~t:1 ~d:1 ~h:2 ~w:3 ~c:1) in
  let store =
    Constant_store.bind_captured Constant_store.empty ~tensor:input
      (Const_ssa.Capture.of_string "w")
    |> Err.or_raise ~pp_error:Constant_store.pp_error
    |> fun store ->
    Constant_store.bind_apply store ~tensor:output
      (Graph_ir.Expand
         { Pointwise.Expand.params = { size = output.shape }; x = t_ 1 })
    |> Err.or_raise ~pp_error:Constant_store.pp_error
  in
  let show coord =
    match Const_ssa_symbolic.ground arena store (t_ 2) coord with
    | None -> Format.printf "%a -> none@." Vec6.pp_coord coord
    | Some expr ->
        Format.printf "%a -> %a@." Vec6.pp_coord coord Ground_expr.pp expr
  in
  show (Vec6.coord ~n:0 ~t:0 ~d:0 ~h:0 ~w:0 ~c:0);
  show (Vec6.coord ~n:0 ~t:0 ~d:0 ~h:1 ~w:2 ~c:0);
  show (Vec6.coord ~n:1 ~t:0 ~d:0 ~h:1 ~w:2 ~c:0);
  [%expect
    {|
    (0) -> capture."w"(0)
    (1,2,0) -> capture."w"(1,2,0)
    (1,0,0,1,2,0) -> capture."w"(1,2,0) |}]

let%expect_test "Const-SSA: materializes a captured expand once" =
  let source = sig_ 1 (Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:3 ~c:1) in
  let output = sig_ 2 (Vec6.shape ~n:2 ~t:1 ~d:1 ~h:2 ~w:3 ~c:1) in
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
          (Graph_ir.Expand
             { Pointwise.Expand.params = { size = output.shape }; x = t_ 1 })
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
    tensor f32 [N=2 T=1 D=1 H=2 W=3 C=1] {0, 1, 2, 10, 11, 12, 0, 1, ...} |}]

let%expect_test "Const-SSA: captured input and add_scalar export" =
  let input = sig_ 1 (Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:1 ~c:1) in
  let output = sig_ 2 (Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:1 ~c:1) in
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
                 Graph_ir.Add_scalar
                   { Pointwise_binary.Scalar_bin.x = t_ 1; scalar = 3.0 };
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
    t2 = add_scalar x=t1 scalar=3 |}]

let%expect_test "Const-SSA: add_scalar grounds to a rounded sum" =
  let input = sig_ 1 (Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:1 ~c:1) in
  let output = sig_ 2 (Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:1 ~c:1) in
  let store =
    Constant_store.bind_captured Constant_store.empty ~tensor:input
      (Const_ssa.Capture.of_string "w")
    |> Err.or_raise ~pp_error:Constant_store.pp_error
    |> fun store ->
    Constant_store.bind_apply store ~tensor:output
      (Graph_ir.Add_scalar
         { Pointwise_binary.Scalar_bin.x = t_ 1; scalar = 3.0 })
    |> Err.or_raise ~pp_error:Constant_store.pp_error
  in
  (match
     Const_ssa_symbolic.ground arena store (t_ 2)
       (Vec6.coord ~n:0 ~t:0 ~d:0 ~h:1 ~w:0 ~c:0)
   with
  | None -> Format.printf "none@."
  | Some expr -> Format.printf "%a@." Ground_expr.pp expr);
  [%expect {| f32((capture."w"(1,0,0) + 0x1.8p+1)) |}]

let%expect_test "Const-SSA: materializes a captured add_scalar once" =
  let source = sig_ 1 (Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:1 ~c:1) in
  let output = sig_ 2 (Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:1 ~c:1) in
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
          (Graph_ir.Add_scalar
             { Pointwise_binary.Scalar_bin.x = t_ 1; scalar = 3.0 })
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
    tensor f32 [H=2 W=1 C=1] {3, 13} |}]

let%expect_test "Const-SSA: captured input and mul_scalar export" =
  let input = sig_ 1 (Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:1 ~c:1) in
  let output = sig_ 2 (Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:1 ~c:1) in
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
                 Graph_ir.Mul_scalar
                   { Pointwise_binary.Scalar_bin.x = t_ 1; scalar = 3.0 };
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
    t2 = mul_scalar x=t1 scalar=3 |}]

let%expect_test "Const-SSA: mul_scalar grounds to a rounded product" =
  let input = sig_ 1 (Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:1 ~c:1) in
  let output = sig_ 2 (Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:1 ~c:1) in
  let store =
    Constant_store.bind_captured Constant_store.empty ~tensor:input
      (Const_ssa.Capture.of_string "w")
    |> Err.or_raise ~pp_error:Constant_store.pp_error
    |> fun store ->
    Constant_store.bind_apply store ~tensor:output
      (Graph_ir.Mul_scalar
         { Pointwise_binary.Scalar_bin.x = t_ 1; scalar = 3.0 })
    |> Err.or_raise ~pp_error:Constant_store.pp_error
  in
  (match
     Const_ssa_symbolic.ground arena store (t_ 2)
       (Vec6.coord ~n:0 ~t:0 ~d:0 ~h:1 ~w:0 ~c:0)
   with
  | None -> Format.printf "none@."
  | Some expr -> Format.printf "%a@." Ground_expr.pp expr);
  [%expect {| f32((capture."w"(1,0,0) * 0x1.8p+1)) |}]

let%expect_test "Const-SSA: materializes a captured mul_scalar once" =
  let source = sig_ 1 (Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:1 ~c:1) in
  let output = sig_ 2 (Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:1 ~c:1) in
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
          (Graph_ir.Mul_scalar
             { Pointwise_binary.Scalar_bin.x = t_ 1; scalar = 3.0 })
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
    tensor f32 [H=2 W=1 C=1] {0, 30} |}]

let%expect_test "Const-SSA: captured input and pow export" =
  let input = sig_ 1 (Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:1) in
  let output = sig_ 2 (Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:1) in
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
                 Graph_ir.Pow
                   { Pointwise_binary.Scalar_bin.x = t_ 1; scalar = 1.5 };
               output;
             })
    | Error e -> Error e
  in
  (match plan with
  | Error e -> Format.printf "%a@." Const_ssa.pp_error (Err.Error.kind e)
  | Ok plan ->
      pp_result Const_ssa.pp_error (Const_ssa.validate plan);
      Format.printf "%a@." Const_ssa.pp plan);
  [%expect {|
    ok
    t1 = captured "weight"
    t2 = pow x=t1 scalar=1.5 |}]

let%expect_test
    "Const-SSA: pow grounds each of ATen's special-cased exponents, and its \
     fallback, to the matching expression" =
  let scalar_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:1 in
  let source = sig_ 1 scalar_shape in
  let output = sig_ 2 scalar_shape in
  let literal = Tensor.materialize scalar_shape (fun _ -> 2.0) in
  List.iter
    (fun scalar ->
      let store =
        Constant_store.bind_literal Constant_store.empty ~tensor:source literal
        |> Err.or_raise ~pp_error:Constant_store.pp_error
        |> fun store ->
        Constant_store.bind_apply store ~tensor:output
          (Graph_ir.Pow { Pointwise_binary.Scalar_bin.x = t_ 1; scalar })
        |> Err.or_raise ~pp_error:Constant_store.pp_error
      in
      match Const_ssa_symbolic.ground arena store (t_ 2) Vec6.origin with
      | None -> Format.printf "%g: none@." scalar
      | Some expr ->
          Format.printf "%g: %.6f  (%a)@." scalar
            (Ground_expr.eval expr Ground_expr.Valuation.empty)
            Ground_expr.pp expr)
    [ 2.0; 3.0; -2.0; 0.5; -0.5; -1.0; 1.5 ];
  [%expect
    {|
    2: 4.000000  (f32((0x1p+1 * 0x1p+1)))
    3: 8.000000  (f32(((0x1p+1 * 0x1p+1) * 0x1p+1)))
    -2: 0.250000  (f32((0x1p+0 / (0x1p+1 * 0x1p+1))))
    0.5: 1.414214  (f32(sqrt(0x1p+1)))
    -0.5: 0.707107  (f32((0x1p+0 / sqrt(0x1p+1))))
    -1: 0.500000  (f32((0x1p+0 / 0x1p+1)))
    1.5: 2.828427  (f32(exp((0x1.8p+0 * log(0x1p+1))))) |}]

let%expect_test
    "Const-SSA: materializes a captured pow (fallback exponent) once" =
  let source = sig_ 1 (Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:1) in
  let output = sig_ 2 (Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:1) in
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
          (Graph_ir.Pow { Pointwise_binary.Scalar_bin.x = t_ 1; scalar = 1.5 })
        |> Err.or_raise ~pp_error:Constant_store.pp_error
  in
  let resolver = function
    | capture
      when Const_ssa.Capture.equal capture (Const_ssa.Capture.of_string "w") ->
        Err.return (Tensor.materialize source.shape (fun _ -> 2.0))
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
    tensor f32 [C=1] {2.82843} |}]

let%expect_test "Const-SSA: captured input and rsub_scalar export" =
  let input = sig_ 1 (Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:1 ~c:1) in
  let output = sig_ 2 (Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:1 ~c:1) in
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
                 Graph_ir.Rsub_scalar
                   {
                     Pointwise.Rsub_scalar.params = { other = 1.0; alpha = 2.0 };
                     x = t_ 1;
                   };
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
    t2 = rsub_scalar x=t1 params={other=1; alpha=2} |}]

let%expect_test "Const-SSA: rsub_scalar grounds to [other - alpha * x], rounded"
    =
  let input = sig_ 1 (Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:1 ~c:1) in
  let output = sig_ 2 (Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:1 ~c:1) in
  let store =
    Constant_store.bind_captured Constant_store.empty ~tensor:input
      (Const_ssa.Capture.of_string "w")
    |> Err.or_raise ~pp_error:Constant_store.pp_error
    |> fun store ->
    Constant_store.bind_apply store ~tensor:output
      (Graph_ir.Rsub_scalar
         {
           Pointwise.Rsub_scalar.params = { other = 1.0; alpha = 2.0 };
           x = t_ 1;
         })
    |> Err.or_raise ~pp_error:Constant_store.pp_error
  in
  (match
     Const_ssa_symbolic.ground arena store (t_ 2)
       (Vec6.coord ~n:0 ~t:0 ~d:0 ~h:1 ~w:0 ~c:0)
   with
  | None -> Format.printf "none@."
  | Some expr -> Format.printf "%a@." Ground_expr.pp expr);
  [%expect {| f32((0x1p+0 - (0x1p+1 * capture."w"(1,0,0)))) |}]

let%expect_test "Const-SSA: materializes a captured rsub_scalar once" =
  let source = sig_ 1 (Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:1) in
  let output = sig_ 2 (Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:1) in
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
          (Graph_ir.Rsub_scalar
             {
               Pointwise.Rsub_scalar.params = { other = 1.0; alpha = 1.0 };
               x = t_ 1;
             })
        |> Err.or_raise ~pp_error:Constant_store.pp_error
  in
  let resolver = function
    | capture
      when Const_ssa.Capture.equal capture (Const_ssa.Capture.of_string "w") ->
        Err.return (Tensor.materialize source.shape (fun _ -> 0.25))
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
    tensor f32 [C=1] {0.75} |}]

let%expect_test "Const-SSA: captured input and sigmoid export" =
  let input = sig_ 1 (Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:1) in
  let output = sig_ 2 (Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:1) in
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
             { op = Graph_ir.Sigmoid { Pointwise.Sigmoid.x = t_ 1 }; output })
    | Error e -> Error e
  in
  (match plan with
  | Error e -> Format.printf "%a@." Const_ssa.pp_error (Err.Error.kind e)
  | Ok plan ->
      pp_result Const_ssa.pp_error (Const_ssa.validate plan);
      Format.printf "%a@." Const_ssa.pp plan);
  [%expect {|
    ok
    t1 = captured "weight"
    t2 = sigmoid x=t1 |}]

let%expect_test
    "Const-SSA: sigmoid grounds to the same 1 / (1 + exp(-x)) expression \
     Compute.pixel uses" =
  let scalar_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:1 in
  let source = sig_ 1 scalar_shape in
  let output = sig_ 2 scalar_shape in
  let literal = Tensor.materialize scalar_shape (fun _ -> 0.0) in
  let store =
    Constant_store.bind_literal Constant_store.empty ~tensor:source literal
    |> Err.or_raise ~pp_error:Constant_store.pp_error
    |> fun store ->
    Constant_store.bind_apply store ~tensor:output
      (Graph_ir.Sigmoid { Pointwise.Sigmoid.x = t_ 1 })
    |> Err.or_raise ~pp_error:Constant_store.pp_error
  in
  (match Const_ssa_symbolic.ground arena store (t_ 2) Vec6.origin with
  | None -> Format.printf "none@."
  | Some expr ->
      Format.printf "%.6f  (%a)@."
        (Ground_expr.eval expr Ground_expr.Valuation.empty)
        Ground_expr.pp expr);
  [%expect {| 0.500000  (f32((0x1p+0 / (0x1p+0 + exp((0x0p+0 - 0x0p+0)))))) |}]

let%expect_test "Const-SSA: materializes a captured sigmoid once" =
  let source = sig_ 1 (Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:1) in
  let output = sig_ 2 (Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:1) in
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
          (Graph_ir.Sigmoid { Pointwise.Sigmoid.x = t_ 1 })
        |> Err.or_raise ~pp_error:Constant_store.pp_error
  in
  let resolver = function
    | capture
      when Const_ssa.Capture.equal capture (Const_ssa.Capture.of_string "w") ->
        Err.return (Tensor.materialize source.shape (fun _ -> 0.0))
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
    tensor f32 [C=1] {0.5} |}]

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
