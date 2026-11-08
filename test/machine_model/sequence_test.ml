module F = Native_test.Graph_fixtures
module M = Machine_model.Mir_model
module T = Model_test
module S = Machine_lower.Mir_lower.Sequence

(* The fast binary32 sequences ([Sequence.Fast]) against the binary32 oracle
   on every Machine IR route, scalar and vector: the same bits as the
   reference sequences, which every other route keeps. The operands sweep
   [exp]'s whole range, its clamps and beyond, with the non-finite values. *)

let wide ~salt i (sg : Tensor_sig.t) =
  let specials = [| Float.nan; Float.infinity; Float.neg_infinity; -0.; 0. |] in
  Tensor.materialize sg.Tensor_sig.shape (fun c ->
      let k = (Vec6.offset sg.Tensor_sig.shape c :> int) + salt + (101 * i) in
      if k mod 37 = 0 then specials.(k / 37 mod Array.length specials)
      else
        (* about -130 to 110, in steps that miss the integers of [t] *)
        (float_of_int (k * 7919 mod 2400) /. 10.) -. 130.03)

let unary name op =
  ( name,
    fun () ->
      F.build name
        Graph_builder.(
          let* x = input ~shape:(F.s 1 1 1 1 2 160) () in
          op x) )

let fixtures =
  [
    unary "exp" (fun x -> Graph_builder.exp x);
    unary "erf" (fun x -> Graph_builder.erf x);
    unary "exact gelu" (fun x -> Graph_builder.gelu Pointwise.Gelu.Exact x);
  ]
  @ List.filter
      (fun (name, _) ->
        String.starts_with ~prefix:"tanh gelu" name
        || String.starts_with ~prefix:"softmax" name)
      (T.vector_fixtures @ T.transformer_fixtures @ T.transformer_bf16_fixtures
     @ Transformer_fixtures.bf16_out)

let routes =
  [
    ("generic", None);
    ("aarch64 scanned", Some (M.Route.Aarch64 M.Stage.Scanned));
    ("x86_64 scanned", Some (M.Route.X86_64 M.Stage.Scanned));
  ]

let%expect_test "fast sequences: Machine IR is the binary32 oracle" =
  List.iter
    (fun (fname, pipeline) ->
      List.iter
        (fun (name, g) ->
          List.iter
            (fun (rname, route) ->
              T.check ?route ~sequence:S.Fast ~pipeline ~oracle:T.Fp32
                ~input:wide ~calls:1
                (Fmt.str "%s, %s, %s" name fname rname)
                g)
            routes)
        fixtures)
    [ ("scalar", T.fp32_ordered); ("neon128", T.fp32_vector) ];
  [%expect
    {|
    exp, scalar, generic (1 invocations): bitwise
    exp, scalar, aarch64 scanned (1 invocations): bitwise
    exp, scalar, x86_64 scanned (1 invocations): bitwise
    erf, scalar, generic (1 invocations): bitwise
    erf, scalar, aarch64 scanned (1 invocations): bitwise
    erf, scalar, x86_64 scanned (1 invocations): bitwise
    exact gelu, scalar, generic (1 invocations): bitwise
    exact gelu, scalar, aarch64 scanned (1 invocations): bitwise
    exact gelu, scalar, x86_64 scanned (1 invocations): bitwise
    softmax over H, W=32, scalar, generic (1 invocations): bitwise
    softmax over H, W=32, scalar, aarch64 scanned (1 invocations): bitwise
    softmax over H, W=32, scalar, x86_64 scanned (1 invocations): bitwise
    tanh gelu mul 2x160, scalar, generic (2 invocations): bitwise
    tanh gelu mul 2x160, scalar, aarch64 scanned (2 invocations): bitwise
    tanh gelu mul 2x160, scalar, x86_64 scanned (2 invocations): bitwise
    tanh gelu mul 1x67, scalar, generic (2 invocations): bitwise
    tanh gelu mul 1x67, scalar, aarch64 scanned (2 invocations): bitwise
    tanh gelu mul 1x67, scalar, x86_64 scanned (2 invocations): bitwise
    tanh gelu mul 1x33, scalar, generic (2 invocations): bitwise
    tanh gelu mul 1x33, scalar, aarch64 scanned (2 invocations): bitwise
    tanh gelu mul 1x33, scalar, x86_64 scanned (2 invocations): bitwise
    tanh gelu mul bf16 2x160, scalar, generic (2 invocations): bitwise
    tanh gelu mul bf16 2x160, scalar, aarch64 scanned (2 invocations): bitwise
    tanh gelu mul bf16 2x160, scalar, x86_64 scanned (2 invocations): bitwise
    tanh gelu mul bf16 1x67, scalar, generic (2 invocations): bitwise
    tanh gelu mul bf16 1x67, scalar, aarch64 scanned (2 invocations): bitwise
    tanh gelu mul bf16 1x67, scalar, x86_64 scanned (2 invocations): bitwise
    tanh gelu mul bf16 1x33, scalar, generic (2 invocations): bitwise
    tanh gelu mul bf16 1x33, scalar, aarch64 scanned (2 invocations): bitwise
    tanh gelu mul bf16 1x33, scalar, x86_64 scanned (2 invocations): bitwise
    tanh gelu mul bf16->bf16 2x160, scalar, generic (3 invocations): bitwise
    tanh gelu mul bf16->bf16 2x160, scalar, aarch64 scanned (3 invocations): bitwise
    tanh gelu mul bf16->bf16 2x160, scalar, x86_64 scanned (3 invocations): bitwise
    tanh gelu mul bf16->bf16 1x67, scalar, generic (3 invocations): bitwise
    tanh gelu mul bf16->bf16 1x67, scalar, aarch64 scanned (3 invocations): bitwise
    tanh gelu mul bf16->bf16 1x67, scalar, x86_64 scanned (3 invocations): bitwise
    tanh gelu mul bf16->bf16 1x33, scalar, generic (3 invocations): bitwise
    tanh gelu mul bf16->bf16 1x33, scalar, aarch64 scanned (3 invocations): bitwise
    tanh gelu mul bf16->bf16 1x33, scalar, x86_64 scanned (3 invocations): bitwise
    exp, neon128, generic (1 invocations): bitwise
    exp, neon128, aarch64 scanned (1 invocations): bitwise
    exp, neon128, x86_64 scanned (1 invocations): bitwise
    erf, neon128, generic (1 invocations): bitwise
    erf, neon128, aarch64 scanned (1 invocations): bitwise
    erf, neon128, x86_64 scanned (1 invocations): bitwise
    exact gelu, neon128, generic (1 invocations): bitwise
    exact gelu, neon128, aarch64 scanned (1 invocations): bitwise
    exact gelu, neon128, x86_64 scanned (1 invocations): bitwise
    softmax over H, W=32, neon128, generic (1 invocations): bitwise
    softmax over H, W=32, neon128, aarch64 scanned (1 invocations): bitwise
    softmax over H, W=32, neon128, x86_64 scanned (1 invocations): bitwise
    tanh gelu mul 2x160, neon128, generic (2 invocations): bitwise
    tanh gelu mul 2x160, neon128, aarch64 scanned (2 invocations): bitwise
    tanh gelu mul 2x160, neon128, x86_64 scanned (2 invocations): bitwise
    tanh gelu mul 1x67, neon128, generic (2 invocations): bitwise
    tanh gelu mul 1x67, neon128, aarch64 scanned (2 invocations): bitwise
    tanh gelu mul 1x67, neon128, x86_64 scanned (2 invocations): bitwise
    tanh gelu mul 1x33, neon128, generic (2 invocations): bitwise
    tanh gelu mul 1x33, neon128, aarch64 scanned (2 invocations): bitwise
    tanh gelu mul 1x33, neon128, x86_64 scanned (2 invocations): bitwise
    tanh gelu mul bf16 2x160, neon128, generic (2 invocations): bitwise
    tanh gelu mul bf16 2x160, neon128, aarch64 scanned (2 invocations): bitwise
    tanh gelu mul bf16 2x160, neon128, x86_64 scanned (2 invocations): bitwise
    tanh gelu mul bf16 1x67, neon128, generic (2 invocations): bitwise
    tanh gelu mul bf16 1x67, neon128, aarch64 scanned (2 invocations): bitwise
    tanh gelu mul bf16 1x67, neon128, x86_64 scanned (2 invocations): bitwise
    tanh gelu mul bf16 1x33, neon128, generic (2 invocations): bitwise
    tanh gelu mul bf16 1x33, neon128, aarch64 scanned (2 invocations): bitwise
    tanh gelu mul bf16 1x33, neon128, x86_64 scanned (2 invocations): bitwise
    tanh gelu mul bf16->bf16 2x160, neon128, generic (3 invocations): bitwise
    tanh gelu mul bf16->bf16 2x160, neon128, aarch64 scanned (3 invocations): bitwise
    tanh gelu mul bf16->bf16 2x160, neon128, x86_64 scanned (3 invocations): bitwise
    tanh gelu mul bf16->bf16 1x67, neon128, generic (3 invocations): bitwise
    tanh gelu mul bf16->bf16 1x67, neon128, aarch64 scanned (3 invocations): bitwise
    tanh gelu mul bf16->bf16 1x67, neon128, x86_64 scanned (3 invocations): bitwise
    tanh gelu mul bf16->bf16 1x33, neon128, generic (3 invocations): bitwise
    tanh gelu mul bf16->bf16 1x33, neon128, aarch64 scanned (3 invocations): bitwise
    tanh gelu mul bf16->bf16 1x33, neon128, x86_64 scanned (3 invocations): bitwise |}]

(* The route suites' share: [exp], [erf] and both GELUs, a tail shape and a
   bfloat16 one. *)
let route_fixtures =
  List.filter
    (fun (name, _) ->
      List.mem name
        [
          "exp";
          "erf";
          "exact gelu";
          "softmax over H, W=32";
          "tanh gelu mul 1x67";
          "tanh gelu mul bf16 1x33";
        ])
    fixtures

(* Every route check of [route_fixtures] under [Fast], on [wide] inputs. *)
let each ?route ~pipeline label =
  List.iter
    (fun (name, g) ->
      T.check ?route ~sequence:S.Fast ~pipeline ~oracle:T.Fp32 ~input:wide
        ~calls:1
        (Fmt.str "%s, %s" name label)
        g)
    route_fixtures

let%expect_test "fast sequences on RV32, interpreted" =
  List.iter
    (fun (rname, route) ->
      each ~route ~pipeline:T.fp32_ordered ("scalar " ^ rname);
      each ~route
        ~pipeline:
          (Ssa_backends.Pipeline.Planned
             {
               numerics = Ssa_ir.Ssa_numerics.Fp32_ordered;
               target = Ssa_ir.Ssa_target.zve32f128;
             })
        ("zve32f128 " ^ rname))
    [
      ("selected", M.Route.Riscv32 M.Stage.Selected);
      ("scanned", M.Route.Riscv32 M.Stage.Scanned);
    ];
  [%expect
    {|
    exp, scalar selected (1 invocations): bitwise
    erf, scalar selected (1 invocations): bitwise
    exact gelu, scalar selected (1 invocations): bitwise
    softmax over H, W=32, scalar selected (1 invocations): bitwise
    tanh gelu mul 1x67, scalar selected (2 invocations): bitwise
    tanh gelu mul bf16 1x33, scalar selected (2 invocations): bitwise
    exp, zve32f128 selected (1 invocations): bitwise
    erf, zve32f128 selected (1 invocations): bitwise
    exact gelu, zve32f128 selected (1 invocations): bitwise
    softmax over H, W=32, zve32f128 selected (1 invocations): bitwise
    tanh gelu mul 1x67, zve32f128 selected (2 invocations): bitwise
    tanh gelu mul bf16 1x33, zve32f128 selected (2 invocations): bitwise
    exp, scalar scanned (1 invocations): bitwise
    erf, scalar scanned (1 invocations): bitwise
    exact gelu, scalar scanned (1 invocations): bitwise
    softmax over H, W=32, scalar scanned (1 invocations): bitwise
    tanh gelu mul 1x67, scalar scanned (2 invocations): bitwise
    tanh gelu mul bf16 1x33, scalar scanned (2 invocations): bitwise
    exp, zve32f128 scanned (1 invocations): bitwise
    erf, zve32f128 scanned (1 invocations): bitwise
    exact gelu, zve32f128 scanned (1 invocations): bitwise
    softmax over H, W=32, zve32f128 scanned (1 invocations): bitwise
    tanh gelu mul 1x67, zve32f128 scanned (2 invocations): bitwise
    tanh gelu mul bf16 1x33, zve32f128 scanned (2 invocations): bitwise |}]
