(* The generic realization (block scratch, then a tail loop) against the
   unattached reference, bit for bit, and each of its deliberate defects
   refused by the same comparison. A third oracle is written here without the
   library's tail code: the whole product, then each step over an array. *)

module C = Contraction
module T = Contraction_tail

let ok = function Ok x -> x | Error _ -> failwith "refused"

(* deterministic binary32 values over a wide exponent range, so a missing
   intermediate rounding shows *)
let values seed n =
  let s = ref seed in
  Array.init n (fun _ ->
      s := ((!s * 1103515245) + 12345) land 0x3FFFFFFF;
      let mant = float_of_int (!s land 0xFFFF) /. 65536. in
      let e = ((!s lsr 16) mod 24) - 20 in
      let sign = if !s land 0x2000000 = 0 then 1. else -1. in
      F32_math.round32 (sign *. (1. +. mant) *. ldexp 1. e))

let batch = 2L
let m = 5L
let n = 7L
let k = 4L
let a_data = values 1 40
let b_data = values 2 56
let c_data = values 3 70
let bias_data = values 4 7
let resid_data = values 5 70
let scale_data = values 6 10
let get arr off = arr.(Int64.to_int off)

let core ~format ~numerics ~seed =
  ok
    (Err.payload
       (C.make ~batch ~m ~n ~k
          ~a:{ C.Operand.batch = 20L; outer = 4L; k = 1L }
          ~b:{ C.Operand.batch = 28L; outer = 1L; k = 7L }
          ~c:{ C.Output.batch = 35L; row = 7L; col = 1L }
          ~format ~seed ~numerics))

let bias = { T.Side.name = "bias"; batch = 0L; row = 0L; col = 1L }
let resid = { T.Side.name = "resid"; batch = 35L; row = 7L; col = 1L }
let scale = { T.Side.name = "scale"; batch = 5L; row = 1L; col = 0L }

let tap_out =
  { T.Tap.name = "mid"; strides = { C.Output.batch = 35L; row = 7L; col = 1L } }

(* bias, residual add (also stored), scale, clamp, relu *)
let steps =
  [
    { T.Step.op = T.Op.Add bias; tap = None };
    { op = T.Op.Add resid; tap = Some tap_out };
    { op = T.Op.Mul scale; tap = None };
    { op = T.Op.Clamp { lo = -4.; hi = 0.5 }; tap = None };
    { op = T.Op.Relu; tap = None };
  ]

let side name off =
  match name with
  | "bias" -> get bias_data off
  | "resid" -> get resid_data off
  | "scale" -> get scale_data off
  | _ -> assert false

let bits arr = Array.map Int64.bits_of_float arr

(* the result of a realization: C's cells, and the tap's *)
let run f =
  let out = Array.make 70 nan and mid = Array.make 70 nan in
  f ~a:(get a_data) ~b:(get b_data) ~c:(get c_data) ~side
    ~tap:(fun _ off v -> mid.(Int64.to_int off) <- v)
    ~write:(fun off v -> out.(Int64.to_int off) <- v);
  (bits out, bits mid)

(* written without the library: product into an array, then the steps *)
let oracle core =
  let whole = Array.make 70 nan in
  C.eval core ~a:(get a_data) ~b:(get b_data) ~c:(get c_data)
    ~write:(fun off v -> whole.(Int64.to_int off) <- v);
  let r = F32_math.round32 in
  let mid = Array.make 70 nan in
  Array.iteri
    (fun off v ->
      let b = off / 35 and j = off mod 7 and i = off mod 35 / 7 in
      ignore b;
      let v = r (v +. bias_data.(j)) in
      let v = r (v +. resid_data.(off)) in
      mid.(off) <- v;
      let v = r (v *. scale_data.((b * 5) + i)) in
      let v = if v < -4. then -4. else if v > 0.5 then 0.5 else v in
      whole.(off) <- (if v < 0. then 0. else v))
    whole;
  (bits whole, bits mid)

let blocks = [ (1L, 1L); (2L, 3L); (5L, 7L); (4L, 4L); (3L, 100L); (100L, 2L) ]

let formats =
  [
    ("f32 ordered", C.Element.F32, C.Numerics.Ordered, C.Seed.Tensor);
    ("f32 fused", C.Element.F32, C.Numerics.Fused_sequential, C.Seed.Zero);
    ("bf16 paired", C.Element.Bf16, C.Numerics.Paired, C.Seed.Tensor);
  ]

let block (rows, cols) = ok (Err.payload (T.Block.make ~rows ~cols))

let%expect_test "every block gives the reference's bits, which are the oracle's"
    =
  List.iter
    (fun (label, format, numerics, seed) ->
      let core = core ~format ~numerics ~seed in
      let d = ok (Err.payload (T.make core steps)) in
      let reference = run (T.eval_reference d) in
      Fmt.pr "%s: reference = oracle %b@." label (reference = oracle core);
      List.iter
        (fun bk ->
          let r = run (T.eval_scratch d (block bk)) in
          Fmt.pr "  block %Ldx%Ld: %b@." (fst bk) (snd bk) (r = reference))
        blocks)
    formats;
  [%expect
    {|
    f32 ordered: reference = oracle true
      block 1x1: true
      block 2x3: true
      block 5x7: true
      block 4x4: true
      block 3x100: true
      block 100x2: true
    f32 fused: reference = oracle true
      block 1x1: true
      block 2x3: true
      block 5x7: true
      block 4x4: true
      block 3x100: true
      block 100x2: true
    bf16 paired: reference = oracle true
      block 1x1: true
      block 2x3: true
      block 5x7: true
      block 4x4: true
      block 3x100: true
      block 100x2: true |}]

let%expect_test "each defect of the realization differs from the reference" =
  let core =
    core ~format:C.Element.F32 ~numerics:C.Numerics.Ordered ~seed:C.Seed.Tensor
  in
  let d = ok (Err.payload (T.make core steps)) in
  let reference = run (T.eval_reference d) in
  List.iter
    (fun (name, fault) ->
      let r = run (T.eval_scratch ~fault d (block (2L, 3L))) in
      Fmt.pr "%s differs: %b@." name (r <> reference))
    [
      ("seed from scratch", T.Fault.Seed_from_scratch);
      ("block-local side", T.Fault.Side_block_local);
      ("unrounded intermediate", T.Fault.Unrounded_intermediate);
    ];
  [%expect
    {|
    seed from scratch differs: true
    block-local side differs: true
    unrounded intermediate differs: true |}]

let%expect_test "refusals" =
  let f32 =
    core ~format:C.Element.F32 ~numerics:C.Numerics.Ordered ~seed:C.Seed.Zero
  in
  let pp_err ppf e = Fmt.pf ppf "%a" T.pp_error e in
  let show = function
    | Ok _ -> Fmt.pr "accepted@."
    | Error e -> Fmt.pr "%a@." pp_err e
  in
  let step op = { T.Step.op; tap = None } in
  show (Err.payload (T.make f32 [ step (T.Op.Clamp { lo = 1.; hi = 0. }) ]));
  show (Err.payload (T.make f32 [ step (T.Op.Clamp { lo = nan; hi = 0. }) ]));
  show
    (Err.payload
       (T.make f32 [ step (T.Op.Add { bias with T.Side.row = -1L }) ]));
  show
    (Err.payload
       (T.make f32
          [
            { T.Step.op = T.Op.Relu; tap = Some tap_out };
            { T.Step.op = T.Op.Relu; tap = Some tap_out };
          ]));
  let i8 =
    core
      ~format:(C.Element.I8 { a = C.Sign.Signed; b = C.Sign.Signed })
      ~numerics:C.Numerics.Ordered ~seed:C.Seed.Zero
  in
  show (Err.payload (T.make i8 []));
  (match Err.payload (T.Block.make ~rows:0L ~cols:3L) with
  | Error e -> Fmt.pr "%a@." T.Block.pp_error e
  | Ok _ -> Fmt.pr "accepted@.");
  (match Err.payload (T.Block.make ~rows:1025L ~cols:1025L) with
  | Error e -> Fmt.pr "%a@." T.Block.pp_error e
  | Ok _ -> Fmt.pr "accepted@.");
  [%expect
    {|
    contraction tail clamp bounds 0x1p+0, 0x0p+0
    contraction tail clamp bounds nan, 0x0p+0
    contraction tail stride bias.row = -1
    contraction tail tap mid twice
    contraction tail over i8ss, an integer product
    block extent rows = 0
    block 1025 x 1025 past 1048576 cells |}]
