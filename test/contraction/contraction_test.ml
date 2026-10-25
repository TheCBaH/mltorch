(* The descriptor's refusals and its reference evaluator on hand-checked
   products: a 2x3 by 3x2, the same with A transposed by its strides, two
   batches, a seeded output, an empty K, and the two numerical requirements
   on a sum where they differ. *)

module C = Contraction

let get arr k = arr.(Int64.to_int k)

let run ?(seed = C.Seed.Zero) ?(numerics = C.Numerics.Ordered) ?(batch = 1L)
    ?(k = 3L) ~a_strides ~b_strides ~c_strides ~m ~n a b c =
  match
    Err.payload
      (C.make ~batch ~m ~n ~k ~a:a_strides ~b:b_strides ~c:c_strides
         ~format:C.Element.F32 ~seed ~numerics)
  with
  | Error e -> Fmt.pr "%a@." C.pp_error e
  | Ok t ->
      let out = Array.copy c in
      C.eval t ~a:(get a) ~b:(get b) ~c:(get c) ~write:(fun k x ->
          out.(Int64.to_int k) <- x);
      Fmt.pr "@[<h>%a: %a@]@." C.pp t Fmt.(array ~sep:(any " ") float) out

let a = [| 1.; 2.; 3.; 4.; 5.; 6. |] (* 2x3, row major *)
let b = [| 1.; 0.; 0.; 1.; 1.; 1. |] (* 3x2, row major *)

let%expect_test "products, strides and seeds" =
  let row_a = { C.Operand.batch = 6L; outer = 3L; k = 1L } in
  let row_b = { C.Operand.batch = 6L; outer = 1L; k = 2L } in
  let out = { C.Output.batch = 4L; row = 2L; col = 1L } in
  run ~a_strides:row_a ~b_strides:row_b ~c_strides:out ~m:2L ~n:2L a b
    (Array.make 4 0.);
  (* A stored transposed (3x2), the same product *)
  run
    ~a_strides:{ C.Operand.batch = 6L; outer = 1L; k = 2L }
    ~b_strides:row_b ~c_strides:out ~m:2L ~n:2L
    [| 1.; 4.; 2.; 5.; 3.; 6. |]
    b (Array.make 4 0.);
  (* two batches, B shared (batch stride 0) *)
  run ~batch:2L
    ~a_strides:{ C.Operand.batch = 0L; outer = 3L; k = 1L }
    ~b_strides:{ row_b with C.Operand.batch = 0L }
    ~c_strides:out ~m:2L ~n:2L a b (Array.make 8 0.);
  run ~seed:C.Seed.Tensor ~a_strides:row_a ~b_strides:row_b ~c_strides:out ~m:2L
    ~n:2L a b [| 10.; 20.; 30.; 40. |];
  run ~k:0L ~seed:C.Seed.Tensor ~a_strides:row_a ~b_strides:row_b ~c_strides:out
    ~m:2L ~n:2L a b [| 10.; 20.; 30.; 40. |];
  [%expect
    {|
    contract f32 batch=1 m=2 n=2 k=3 a=(6,3,1) b=(6,1,2) c=(4,2,1) seed=zero ordered: 4 5 10 11
    contract f32 batch=1 m=2 n=2 k=3 a=(6,1,2) b=(6,1,2) c=(4,2,1) seed=zero ordered: 4 5 10 11
    contract f32 batch=2 m=2 n=2 k=3 a=(0,3,1) b=(0,1,2) c=(4,2,1) seed=zero ordered: 4 5 10 11 4 5 10 11
    contract f32 batch=1 m=2 n=2 k=3 a=(6,3,1) b=(6,1,2) c=(4,2,1) seed=tensor ordered: 14 25 40 51
    contract f32 batch=1 m=2 n=2 k=0 a=(6,3,1) b=(6,1,2) c=(4,2,1) seed=tensor ordered: 10 20 30 40 |}]

(* [1 + (2^-24 + 2^-57)] as a two-term sum: ordered, the product rounds to
   2^-24 and the sum ties to 1; fused, one rounding gives [1 + 2^-23]. *)
let%expect_test "ordered and fused sums differ" =
  let u = ldexp 1. (-11) in
  let a = [| 1.; 1. +. u |]
  and b = [| 1.; ldexp 1. (-24) *. (1. -. u +. (u *. u)) |] in
  let s = { C.Operand.batch = 0L; outer = 0L; k = 1L } in
  let o = { C.Output.batch = 0L; row = 0L; col = 0L } in
  List.iter
    (fun numerics ->
      match
        Err.payload
          (C.make ~batch:1L ~m:1L ~n:1L ~k:2L ~a:s ~b:s ~c:o
             ~format:C.Element.F32 ~seed:C.Seed.Zero ~numerics)
      with
      | Error e -> Fmt.pr "%a@." C.pp_error e
      | Ok t ->
          let r = ref 0. in
          C.eval t ~a:(get a) ~b:(get b)
            ~c:(fun _ -> 0.)
            ~write:(fun _ x -> r := x);
          Fmt.pr "%s: %h@." (C.Numerics.name numerics) !r)
    [ C.Numerics.Ordered; C.Numerics.Fused_sequential ];
  [%expect {|
    ordered: 0x1p+0
    fused_sequential: 0x1.000002p+0 |}]

let%expect_test "refusals" =
  let s = { C.Operand.batch = 0L; outer = 1L; k = 1L } in
  let o = { C.Output.batch = 0L; row = 1L; col = 1L } in
  let show r =
    match Err.payload r with
    | Error e -> Fmt.pr "%a@." C.pp_error e
    | Ok _ -> Fmt.pr "accepted@."
  in
  let make ?(m = 1L) ?(k = 1L) ?(a = s) () =
    C.make ~batch:1L ~m ~n:1L ~k ~a ~b:s ~c:o ~format:C.Element.F32
      ~seed:C.Seed.Zero ~numerics:C.Numerics.Ordered
  in
  show (make ~m:0L ());
  show (make ~k:(-1L) ());
  show (make ~a:{ s with C.Operand.k = -4L } ());
  show (make ~k:0L ());
  (* an integer k at which 128 x 128 products could pass 2^31 - 1 *)
  let i8 k =
    C.make ~batch:1L ~m:1L ~n:1L ~k ~a:s ~b:s ~c:o
      ~format:(C.Element.I8 { a = C.Sign.Signed; b = C.Sign.Signed })
      ~seed:C.Seed.Zero ~numerics:C.Numerics.Ordered
  in
  show (i8 131071L);
  show (i8 131072L);
  [%expect
    {|
    contraction extent m = 0
    contraction extent k = -1
    contraction stride a.k = -4
    accepted
    accepted
    contraction k = 131072, past an exact 32-bit integer sum |}]

let%expect_test "a descriptor's name round-trips" =
  match
    Err.payload
      (C.make ~batch:3L ~m:7L ~n:29L ~k:63L
         ~a:{ C.Operand.batch = 441L; outer = 63L; k = 1L }
         ~b:{ C.Operand.batch = 1827L; outer = 1L; k = 29L }
         ~c:{ C.Output.batch = 203L; row = 29L; col = 1L }
         ~format:C.Element.F32 ~seed:C.Seed.Zero
         ~numerics:C.Numerics.Fused_sequential)
  with
  | Error e -> Fmt.pr "%a@." C.pp_error e
  | Ok t ->
      Fmt.pr "%s@.%s@." (C.name t) (C.symbol t);
      let bf16 =
        Result.get_ok
          (Err.payload
             (C.make ~batch:t.C.batch ~m:t.C.m ~n:t.C.n ~k:t.C.k ~a:t.C.a
                ~b:t.C.b ~c:t.C.c ~format:C.Element.Bf16 ~seed:t.C.seed
                ~numerics:t.C.numerics))
      in
      let i8 =
        Result.get_ok
          (Err.payload
             (C.make ~batch:t.C.batch ~m:t.C.m ~n:t.C.n ~k:t.C.k ~a:t.C.a
                ~b:t.C.b ~c:t.C.c
                ~format:
                  (C.Element.I8 { a = C.Sign.Unsigned; b = C.Sign.Signed })
                ~seed:t.C.seed ~numerics:t.C.numerics))
      in
      let trips t =
        Option.map C.name (C.of_name (C.name t)) = Some (C.name t)
      in
      Fmt.pr "%s: %b@." (C.name i8) (trips i8);
      Fmt.pr "%s@.round trip: %b, %b; garbage: %b@." (C.name bf16) (trips t)
        (trips bf16)
        (C.of_name "contract.f32.b1" = None);
      [%expect
        {|
        contract.f32.b3.m7.n29.k63.a441_63_1.b1827_1_29.c203_29_1.zero.fused_sequential
        mlt_contract_2ec0b43f4257b217
        contract.i8us.b3.m7.n29.k63.a441_63_1.b1827_1_29.c203_29_1.zero.fused_sequential: true
        contract.bf16.b3.m7.n29.k63.a441_63_1.b1827_1_29.c203_29_1.zero.fused_sequential
        round trip: true, true; garbage: true |}]

let%expect_test "which unit realizes a contraction" =
  let d ?(format = C.Element.F32) numerics =
    Result.get_ok
      (Err.payload
         (C.make ~batch:1L ~m:2L ~n:2L ~k:3L
            ~a:{ C.Operand.batch = 6L; outer = 3L; k = 1L }
            ~b:{ C.Operand.batch = 6L; outer = 1L; k = 2L }
            ~c:{ C.Output.batch = 4L; row = 2L; col = 1L }
            ~format ~seed:C.Seed.Zero ~numerics))
  in
  List.iter
    (fun format ->
      List.iter
        (fun (matrix, vector) ->
          List.iter
            (fun numerics ->
              let e, r =
                Contraction.Select.choose
                  { Contraction.Select.Capabilities.matrix; vector }
                  (d ~format numerics)
              in
              Fmt.pr "%s, matrix %b, vector %b, %s: %s (%s)@."
                (C.Element.name format) matrix vector (C.Numerics.name numerics)
                (Contraction.Select.Engine.name e)
                (Contraction.Select.Reason.name r))
            [ C.Numerics.Ordered; C.Numerics.Fused_sequential ])
        [ (true, true); (true, false); (false, true); (false, false) ])
    [
      C.Element.Bf16;
      C.Element.F32;
      C.Element.I8 { a = C.Sign.Signed; b = C.Sign.Unsigned };
    ];
  [%expect
    {|
    bf16, matrix true, vector true, ordered: rvv (format_off_matrix)
    bf16, matrix true, vector true, fused_sequential: rvv (format_off_matrix)
    bf16, matrix true, vector false, ordered: loops (no_vector_unit)
    bf16, matrix true, vector false, fused_sequential: loops (no_vector_unit)
    bf16, matrix false, vector true, ordered: rvv (vector_unit)
    bf16, matrix false, vector true, fused_sequential: rvv (vector_unit)
    bf16, matrix false, vector false, ordered: loops (no_vector_unit)
    bf16, matrix false, vector false, fused_sequential: loops (no_vector_unit)
    f32, matrix true, vector true, ordered: zvt (ordered_on_matrix)
    f32, matrix true, vector true, fused_sequential: rvv (fused_off_matrix)
    f32, matrix true, vector false, ordered: zvt (ordered_on_matrix)
    f32, matrix true, vector false, fused_sequential: loops (no_vector_unit)
    f32, matrix false, vector true, ordered: rvv (vector_unit)
    f32, matrix false, vector true, fused_sequential: rvv (vector_unit)
    f32, matrix false, vector false, ordered: loops (no_vector_unit)
    f32, matrix false, vector false, fused_sequential: loops (no_vector_unit)
    i8su, matrix true, vector true, ordered: zvt (integer_on_matrix)
    i8su, matrix true, vector true, fused_sequential: zvt (integer_on_matrix)
    i8su, matrix true, vector false, ordered: zvt (integer_on_matrix)
    i8su, matrix true, vector false, fused_sequential: zvt (integer_on_matrix)
    i8su, matrix false, vector true, ordered: rvv (vector_unit)
    i8su, matrix false, vector true, fused_sequential: rvv (vector_unit)
    i8su, matrix false, vector false, ordered: loops (no_vector_unit)
    i8su, matrix false, vector false, fused_sequential: loops (no_vector_unit) |}]

(* The paired requirement: each two k steps' products are summed and rounded to
   odd in binary32, then added to the accumulator to nearest. A bit far below
   the pair's last sets it (ordered leaves 2^-24), and a last odd step adds its
   one product alone. *)
let%expect_test "the paired requirement" =
  let s = { C.Operand.batch = 0L; outer = 0L; k = 1L } in
  let o = { C.Output.batch = 0L; row = 0L; col = 0L } in
  let eval numerics ~k a b =
    let t =
      Result.get_ok
        (Err.payload
           (C.make ~batch:1L ~m:1L ~n:1L ~k ~a:s ~b:s ~c:o
              ~format:C.Element.Bf16 ~seed:C.Seed.Zero ~numerics))
    in
    let r = ref 0. in
    C.eval t ~a:(get a) ~b:(get b) ~c:(fun _ -> 0.) ~write:(fun _ x -> r := x);
    !r
  in
  let p2 = ldexp 1. in
  List.iter
    (fun (what, k, a, b) ->
      Fmt.pr "%s: paired %h, ordered %h@." what
        (eval C.Numerics.Paired ~k a b)
        (eval C.Numerics.Ordered ~k a b))
    [
      ("2^-24 + 2^-48", 2L, [| p2 (-12); p2 (-12) |], [| p2 (-12); p2 (-36) |]);
      ("2^-24 + 2^-24", 2L, [| p2 (-12); p2 (-12) |], [| p2 (-12); p2 (-12) |]);
      ( "1, 2^-24, 2^-48",
        3L,
        [| 1.; p2 (-12); p2 (-12) |],
        [| 1.; p2 (-12); p2 (-36) |] );
    ];
  let make format numerics =
    C.make ~batch:1L ~m:1L ~n:1L ~k:2L ~a:s ~b:s ~c:o ~format ~seed:C.Seed.Zero
      ~numerics
  in
  List.iter
    (fun format ->
      match Err.payload (make format C.Numerics.Paired) with
      | Error e -> Fmt.pr "%a@." C.pp_error e
      | Ok t -> Fmt.pr "accepted: %s@." (C.name t))
    [
      C.Element.Bf16;
      C.Element.F32;
      C.Element.I8 { a = C.Sign.Signed; b = C.Sign.Signed };
    ];
  (match Err.payload (make C.Element.Bf16 C.Numerics.Paired) with
  | Ok t ->
      Fmt.pr "round trip %b@."
        (Option.map C.name (C.of_name (C.name t)) = Some (C.name t))
  | Error _ -> ());
  List.iter
    (fun (matrix, vector) ->
      let d =
        Result.get_ok (Err.payload (make C.Element.Bf16 C.Numerics.Paired))
      in
      let e, r = C.Select.choose { C.Select.Capabilities.matrix; vector } d in
      Fmt.pr "matrix %b, vector %b: %s (%s)@." matrix vector
        (C.Select.Engine.name e) (C.Select.Reason.name r))
    [ (true, true); (true, false); (false, true); (false, false) ];
  [%expect
    {|
    2^-24 + 2^-48: paired 0x1.000002p-24, ordered 0x1p-24
    2^-24 + 2^-24: paired 0x1p-23, ordered 0x1p-23
    1, 2^-24, 2^-48: paired 0x1.000002p+0, ordered 0x1p+0
    accepted: contract.bf16.b1.m1.n1.k2.a0_0_1.b0_0_1.c0_0_0.zero.paired
    contraction summing two k steps first over f32 operands
    contraction summing two k steps first over i8ss operands
    round trip true
    matrix true, vector true: zvt (paired_on_matrix)
    matrix true, vector false: zvt (paired_on_matrix)
    matrix false, vector true: loops (paired_needs_matrix)
    matrix false, vector false: loops (paired_needs_matrix) |}]

(* The transpose exchanges the operands and the output's strides and nothing
   else: every numerics gives the same bits at the same offsets, for products
   whose rounding depends on the order of the k steps. *)
let%expect_test "the transpose writes the same bits" =
  let m = 3L and n = 4L and k = 5L in
  let x s i =
    ldexp
      (float_of_int ((((s * 7919) + (i * 104729)) mod 255) - 127))
      (-(((i * 3) + s) mod 9))
  in
  let a = Array.init 15 (x 1) and b = Array.init 20 (x 2) in
  let show format numerics =
    let d =
      Result.get_ok
        (Err.payload
           (C.make ~batch:1L ~m ~n ~k
              ~a:{ C.Operand.batch = 0L; outer = k; k = 1L }
              ~b:{ C.Operand.batch = 0L; outer = 1L; k = n }
              ~c:{ C.Output.batch = 0L; row = n; col = 1L }
              ~format ~seed:C.Seed.Zero ~numerics))
    in
    let out d ~a ~b =
      let o = Array.make 12 nan in
      C.eval d ~a:(get a) ~b:(get b)
        ~c:(fun _ -> 0.)
        ~write:(fun at x -> o.(Int64.to_int at) <- x);
      o
    in
    let t = C.transpose d in
    let same =
      Array.for_all2
        (fun x y -> Int64.bits_of_float x = Int64.bits_of_float y)
        (out d ~a ~b) (out t ~a:b ~b:a)
    in
    Fmt.pr "%s %s: %s, twice is the same descriptor %b@."
      (C.Element.name format) (C.Numerics.name numerics)
      (if same then "same bits" else "DIFFERS")
      (C.name (C.transpose t) = C.name d)
  in
  show C.Element.F32 C.Numerics.Ordered;
  show C.Element.F32 C.Numerics.Fused_sequential;
  show C.Element.Bf16 C.Numerics.Paired;
  [%expect
    {|
    f32 ordered: same bits, twice is the same descriptor true
    f32 fused_sequential: same bits, twice is the same descriptor true
    bf16 paired: same bits, twice is the same descriptor true |}]
