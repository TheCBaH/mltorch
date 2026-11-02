(* What {!Coral_transfer.Transfer_model.choose} picks for products of the
   shapes the subsystem suites run, at the DTCM budget the route offers a
   function. A product it leaves in place never needs the DMA engine, so the
   route reads it the same under [Forbidden] and [Allowed]; one it sends to
   the DMA is where the two differ. A change to the rates or the candidates
   that flips a choice shows here first. *)

module P = Coral_transfer.Transfer_plan
module Tm = Coral_transfer.Transfer_model

let i64 = Int64.of_int
let operand ~batch ~outer ~k = { Contraction.Operand.batch; outer; k }

(* [linear]: the weight is [n, k] (B's k stride 1); [bmm]: B is [k, n] *)
let desc ~linear ~numerics ~m ~n ~k =
  Result.get_ok
    (Err.payload
       (Contraction.make ~batch:1L ~m:(i64 m) ~n:(i64 n) ~k:(i64 k)
          ~a:(operand ~batch:(i64 (m * k)) ~outer:(i64 k) ~k:1L)
          ~b:
            (if linear then operand ~batch:(i64 (k * n)) ~outer:(i64 k) ~k:1L
             else operand ~batch:(i64 (k * n)) ~outer:1L ~k:(i64 n))
          ~c:{ Contraction.Output.batch = i64 (m * n); row = i64 n; col = 1L }
          ~format:Contraction.Element.F32 ~seed:Contraction.Seed.Zero ~numerics))

let fused = Contraction.Numerics.Fused_sequential
let ordered = Contraction.Numerics.Ordered

let%expect_test "the plan chosen for each shape" =
  List.iter
    (fun (name, d) ->
      let p = Tm.choose ~budget:32768 d in
      Fmt.pr "%-28s %s%a@." name
        (if p = P.in_place then "in place" else "DMA: ")
        (fun ppf p -> if p <> P.in_place then P.pp ppf p)
        p)
    [
      ("linear 1x8 k8", desc ~linear:true ~numerics:fused ~m:1 ~n:8 ~k:8);
      ("linear 4x16 k16", desc ~linear:true ~numerics:fused ~m:4 ~n:16 ~k:16);
      ("linear 9x32 k64", desc ~linear:true ~numerics:fused ~m:9 ~n:32 ~k:64);
      ("linear 48x32 k64", desc ~linear:true ~numerics:fused ~m:48 ~n:32 ~k:64);
      ( "linear 60x32 k64, ordered",
        desc ~linear:true ~numerics:ordered ~m:60 ~n:32 ~k:64 );
      ( "linear 192x32 k128",
        desc ~linear:true ~numerics:fused ~m:192 ~n:32 ~k:128 );
      ( "linear 120x24 k64, ordered",
        desc ~linear:true ~numerics:ordered ~m:120 ~n:24 ~k:64 );
      ("bmm 1x9x32 k64", desc ~linear:false ~numerics:fused ~m:9 ~n:32 ~k:64);
      ("bmm 1x12x64 k32", desc ~linear:false ~numerics:fused ~m:12 ~n:64 ~k:32);
      ("bmm 1x18x40 k48", desc ~linear:false ~numerics:fused ~m:18 ~n:40 ~k:48);
    ];
  [%expect
    {|
    linear 1x8 k8                in place
    linear 4x16 k16              DMA: A in place, B shared
    linear 9x32 k64              DMA: A copied, B shared
    linear 48x32 k64             DMA: A streamed by 30 rows, B shared, C by the DMA
    linear 60x32 k64, ordered    DMA: A streamed by 32 rows, B shared, C by the DMA
    linear 192x32 k128           DMA: A streamed by 12 rows, B shared, C by the DMA
    linear 120x24 k64, ordered   DMA: A streamed by 36 rows, B shared, C by the DMA
    bmm 1x9x32 k64               DMA: A in place, B shared
    bmm 1x12x64 k32              DMA: A copied, B shared
    bmm 1x18x40 k48              DMA: A copied, B shared |}]
