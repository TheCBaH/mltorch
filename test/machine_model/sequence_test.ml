module F = Native_test.Graph_fixtures
module M = Machine_model.Mir_model
module T = Model_test

(* The binary32 sequences ({!F32_math.Make_fast}, the only form Machine IR
   lowers) against the binary32 oracle on every Machine IR route, scalar and
   vector: the reference sequence's bits, which the oracle runs. The operands
   sweep [exp]'s whole range, its clamps and beyond, with the non-finite
   values; [exp, around each k step] takes the patterns where the fast pieces
   change their result. *)

let wide ~salt i (sg : Tensor_sig.t) =
  let specials = [| Float.nan; Float.infinity; Float.neg_infinity; -0.; 0. |] in
  Tensor.materialize sg.Tensor_sig.shape (fun c ->
      let k = (Vec6.offset sg.Tensor_sig.shape c :> int) + salt + (101 * i) in
      if k mod 37 = 0 then specials.(k / 37 mod Array.length specials)
      else
        (* about -130 to 110, in steps that miss the integers of [t] *)
        (float_of_int (k * 7919 mod 2400) /. 10.) -. 130.03)

(* The binary32 patterns within two of each point where [exp]'s [k] steps
   ([t +- 0.5] crossing an integer, [k] in [-151, 129]: every power of two
   [pow2_int] builds), within four of the clamps and of zero (the [-0] region
   and the subnormals), and the non-finite values. *)
let k_steps =
  let around x d =
    let b = Int32.bits_of_float x in
    List.init
      ((2 * d) + 1)
      (fun j -> Int32.float_of_bits (Int32.add b (Int32.of_int (j - d))))
  in
  List.concat_map
    (fun k ->
      let k = float_of_int k in
      around ((k +. 0.5) /. 1.44269504088896341) 2
      @ around ((k -. 0.5) /. 1.44269504088896341) 2)
    (List.init 281 (fun j -> j - 151))
  @ List.concat_map (fun x -> around x 4) [ -104.; 89.; 0. ]
  @ [ Float.nan; Float.infinity; Float.neg_infinity; -0.; -110.; 100.; 3e38 ]
  |> Array.of_list

let steps ~salt:_ _ (sg : Tensor_sig.t) =
  Tensor.materialize sg.Tensor_sig.shape (fun c ->
      let k = (Vec6.offset sg.Tensor_sig.shape c :> int) in
      k_steps.(k mod Array.length k_steps))

let unary ?(shape = F.s 1 1 1 1 2 160) name op =
  ( name,
    fun () ->
      F.build name
        Graph_builder.(
          let* x = input ~shape () in
          op x) )

let k_steps_name = "exp, around each k step"

(* [steps] for [exp, around each k step] (one row of 23 x 128 holds every
   pattern), [wide] for the rest. *)
let input name = if String.equal name k_steps_name then steps else wide

let fixtures =
  [
    unary "exp" (fun x -> Graph_builder.exp x);
    unary ~shape:(F.s 1 1 1 1 23 128) k_steps_name (fun x ->
        Graph_builder.exp x);
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

let%expect_test "binary32 sequences: Machine IR is the binary32 oracle" =
  List.iter
    (fun (fname, pipeline) ->
      List.iter
        (fun (name, g) ->
          List.iter
            (fun (rname, route) ->
              T.check ?route ~pipeline ~oracle:T.Fp32 ~input:(input name)
                ~calls:1
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
    exp, around each k step, scalar, generic (1 invocations): bitwise
    exp, around each k step, scalar, aarch64 scanned (1 invocations): bitwise
    exp, around each k step, scalar, x86_64 scanned (1 invocations): bitwise
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
    exp, around each k step, neon128, generic (1 invocations): bitwise
    exp, around each k step, neon128, aarch64 scanned (1 invocations): bitwise
    exp, around each k step, neon128, x86_64 scanned (1 invocations): bitwise
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

(* The route suites' share: [exp] (and around each [k] step), [erf] and both
   GELUs, a tail shape and a bfloat16 one. *)
let route_fixtures =
  List.filter
    (fun (name, _) ->
      List.mem name
        [
          "exp";
          k_steps_name;
          "erf";
          "exact gelu";
          "softmax over H, W=32";
          "tanh gelu mul 1x67";
          "tanh gelu mul bf16 1x33";
        ])
    fixtures

(* Every route check of [route_fixtures], on each one's [input]. *)
let each ?route ~pipeline label =
  List.iter
    (fun (name, g) ->
      T.check ?route ~pipeline ~oracle:T.Fp32 ~input:(input name) ~calls:1
        (Fmt.str "%s, %s" name label)
        g)
    route_fixtures

let%expect_test "binary32 sequences on RV32, interpreted" =
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
    exp, around each k step, scalar selected (1 invocations): bitwise
    erf, scalar selected (1 invocations): bitwise
    exact gelu, scalar selected (1 invocations): bitwise
    softmax over H, W=32, scalar selected (1 invocations): bitwise
    tanh gelu mul 1x67, scalar selected (2 invocations): bitwise
    tanh gelu mul bf16 1x33, scalar selected (2 invocations): bitwise
    exp, zve32f128 selected (1 invocations): bitwise
    exp, around each k step, zve32f128 selected (1 invocations): bitwise
    erf, zve32f128 selected (1 invocations): bitwise
    exact gelu, zve32f128 selected (1 invocations): bitwise
    softmax over H, W=32, zve32f128 selected (1 invocations): bitwise
    tanh gelu mul 1x67, zve32f128 selected (2 invocations): bitwise
    tanh gelu mul bf16 1x33, zve32f128 selected (2 invocations): bitwise
    exp, scalar scanned (1 invocations): bitwise
    exp, around each k step, scalar scanned (1 invocations): bitwise
    erf, scalar scanned (1 invocations): bitwise
    exact gelu, scalar scanned (1 invocations): bitwise
    softmax over H, W=32, scalar scanned (1 invocations): bitwise
    tanh gelu mul 1x67, scalar scanned (2 invocations): bitwise
    tanh gelu mul bf16 1x33, scalar scanned (2 invocations): bitwise
    exp, zve32f128 scanned (1 invocations): bitwise
    exp, around each k step, zve32f128 scanned (1 invocations): bitwise
    erf, zve32f128 scanned (1 invocations): bitwise
    exact gelu, zve32f128 scanned (1 invocations): bitwise
    softmax over H, W=32, zve32f128 scanned (1 invocations): bitwise
    tanh gelu mul 1x67, zve32f128 scanned (2 invocations): bitwise
    tanh gelu mul bf16 1x33, zve32f128 scanned (2 invocations): bitwise |}]
