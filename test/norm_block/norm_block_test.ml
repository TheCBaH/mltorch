(* The row normalization's reference on answers known by hand, its refusals and
   its name. *)

module N = Norm_block
module P = Attention_block.Precision

let dense width = { N.Strides.row = width; col = 1L }

let desc ?(kind = N.Kind.Layer) ?(eps = 0.) ?(weight = false) ?(bias = false)
    ?(precision = P.F32_ordered) ~rows ~width () =
  N.make ~kind ~rows ~width ~eps ~weight ~bias ~x:(dense width) ~y:(dense width)
    ~precision

let get d = Result.get_ok (Err.payload d)
let eps5 = F32_math.round32 1e-5

let run ?(w = [||]) ?(b = [||]) d xs =
  let out = Array.make (Array.length xs) Float.nan in
  N.eval d
    ~x:(fun o -> xs.(Int64.to_int o))
    ~weight:(fun c -> w.(Int64.to_int c))
    ~bias:(fun c -> b.(Int64.to_int c))
    ~write:(fun o v -> out.(Int64.to_int o) <- v);
  out

let show a =
  Fmt.pr "%a@." Fmt.(array ~sep:(any " ") (fun ppf x -> Fmt.pf ppf "%h" x)) a

let%expect_test "known answers" =
  (* a constant row centres to zero: the bias alone survives *)
  show
    (run ~w:[| 2.; 2. |] ~b:[| 0.5; -1. |]
       (get (desc ~eps:eps5 ~weight:true ~bias:true ~rows:1L ~width:2L ()))
       [| 3.; 3. |]);
  (* [1, -1]: mean 0, variance 1, so the output is the input at eps 0 *)
  show (run (get (desc ~rows:1L ~width:2L ())) [| 1.; -1. |]);
  (* rms of [3, 4] over two: sqrt 12.5 *)
  show (run (get (desc ~kind:N.Kind.Rms ~rows:1L ~width:2L ())) [| 3.; 4. |]);
  (* a dominant eps: the output is the centred input scaled by 1/sqrt eps *)
  show (run (get (desc ~eps:0x1p20 ~rows:1L ~width:2L ())) [| 1.; -1. |]);
  (* a NaN poisons its row as the canonical NaN, not the next *)
  show
    (Array.map
       (fun x -> Int32.float_of_bits (Int32.bits_of_float x))
       (run (get (desc ~rows:2L ~width:2L ())) [| Float.nan; 1.; 1.; -1. |]));
  Fmt.pr "%lx@."
    (Int32.bits_of_float
       (run (get (desc ~rows:1L ~width:2L ())) [| Float.nan; 1. |]).(0));
  (* a row of -0: +0 + -0 sums to a +0 mean and -0 - +0 centres to -0, kept
     through the scale; at eps 0 an RmsNorm scales -0 by an infinite inverse *)
  show (run (get (desc ~eps:eps5 ~rows:1L ~width:2L ())) [| -0.; -0. |]);
  show (run (get (desc ~kind:N.Kind.Rms ~rows:1L ~width:1L ())) [| -0. |]);
  (* bfloat16 rounds the output *)
  show
    (run
       (get (desc ~precision:P.Bf16_paired ~rows:1L ~width:3L ()))
       [| 1.; 2.; 4. |]);
  show (run (get (desc ~rows:1L ~width:3L ())) [| 1.; 2.; 4. |]);
  [%expect
    {|
    0x1p-1 -0x1p+0
    0x1p+0 -0x1p+0
    0x1.b27248p-1 0x1.21a186p+0
    0x1.fffffp-11 -0x1.fffffp-11
    nan nan 0x1p+0 -0x1p+0
    7fc00000
    -0x0p+0 -0x0p+0
    nan
    -0x1.12p+0 -0x1.12p-2 0x1.56p+0
    -0x1.11acecp+0 -0x1.11aceap-2 0x1.56182ap+0 |}]

let%expect_test "the fused sum of squares rounds once a step" =
  (* x * x + acc over rows of inexact squares: the fused and the rounded
     products give different sums on some rows, and only the sum differs *)
  let rows = 64 and width = 7 in
  let lcg = ref 7 in
  let xs =
    Array.init (rows * width) (fun _ ->
        lcg := ((!lcg * 1103515245) + 12345) land 0x3FFFFFFF;
        F32_math.round32 (float_of_int ((!lcg / 7 mod 4001) - 2000) /. 977.))
  in
  let differing kind =
    let d precision =
      get (desc ~kind ~precision ~rows:(Int64.of_int rows) ~width:7L ())
    in
    let ord = run (d P.F32_ordered) xs and fused = run (d P.F32_fused) xs in
    List.length
      (List.filter
         (fun r ->
           Array.sub ord (r * width) width <> Array.sub fused (r * width) width)
         (List.init rows Fun.id))
  in
  Fmt.pr "layer: %d of %d rows differ; rms: %d@." (differing N.Kind.Layer) rows
    (differing N.Kind.Rms);
  [%expect {| layer: 6 of 64 rows differ; rms: 8 |}]

let%expect_test "refusals" =
  let pr d =
    match Err.payload d with
    | Ok d -> Fmt.pr "%a@." N.pp d
    | Error e -> Fmt.pr "%a@." N.pp_error e
  in
  pr (desc ~rows:0L ~width:4L ());
  pr (desc ~rows:1L ~width:0x1000001L ());
  pr (desc ~eps:0.1 ~rows:1L ~width:4L ());
  pr (desc ~eps:(-1e-5) ~rows:1L ~width:4L ());
  pr (desc ~eps:Float.infinity ~rows:1L ~width:4L ());
  pr (desc ~kind:N.Kind.Rms ~bias:true ~rows:1L ~width:4L ());
  pr
    (N.make ~kind:N.Kind.Layer ~rows:1L ~width:4L ~eps:0. ~weight:false
       ~bias:false
       ~x:{ N.Strides.row = 4L; col = -1L }
       ~y:(dense 4L) ~precision:P.F32_ordered);
  pr (desc ~eps:eps5 ~weight:true ~bias:true ~rows:3L ~width:4L ());
  [%expect
    {|
    norm extent rows = 0
    norm extent width = 16777217
    norm eps 0x1.999999999999ap-4, not a finite non-negative binary32
    norm eps -0x1.4f8b588e368f1p-17, not a finite non-negative binary32
    norm eps infinity, not a finite non-negative binary32
    an RmsNorm has no bias
    norm stride x.col = -1
    norm layer rows=3 width=4 eps=0x1.4f8b58p-17 weight=true bias=true x=(4,1) y=(4,1) precision=f32_ordered |}]

let%expect_test "a name rebuilds its descriptor" =
  List.iter
    (fun d ->
      let d = get d in
      let back = N.of_name (N.name d) in
      Fmt.pr "%s %s %b@." (N.name d) (N.symbol d) (back = Some d))
    [
      desc ~eps:eps5 ~weight:true ~bias:true ~rows:3L ~width:4L ();
      desc ~kind:N.Kind.Rms ~weight:true ~precision:P.F32_fused ~rows:2L
        ~width:8L ();
      desc ~bias:true ~precision:P.Bf16_paired ~rows:1L ~width:2L ();
      desc ~rows:1L ~width:2L ();
    ];
  Fmt.pr "%b %b@."
    (N.of_name "norm.layer.r1.w2.e0.a.x2_1.y2_1.pnone" = None)
    (N.of_name "norm.rms.r1.w2.e0.ab.x2_1.y2_1.pf32_ordered" = None);
  [%expect
    {|
    norm.layer.r3.w4.e3727c5ac.awb.x4_1.y4_1.pf32_ordered mlt_norm_afb6809b77a070a9 true
    norm.rms.r2.w8.e0.aw.x8_1.y8_1.pf32_fused mlt_norm_ad7af538dd30655e true
    norm.layer.r1.w2.e0.ab.x2_1.y2_1.pbf16_paired mlt_norm_35d34437d67e5b63 true
    norm.layer.r1.w2.e0.a.x2_1.y2_1.pf32_ordered mlt_norm_4eae6e43ddb5d804 true
    true true |}]
