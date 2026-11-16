(* A contraction called on views of its buffers, a block of rows at a time,
   computes the bits the whole-tensor [Contract] computes; so does its
   expansion into loops. A base that lets the footprint leave its buffer is
   refused by the verifier, and the operation is refused where a target has no
   form for it. *)

open Ssa_ir
module F = Ssa_fixtures
module B = Ssa_builder

let i64 = Int64.of_int

let desc ~m ~k ~n =
  Result.get_ok
    (Err.payload
       (Contraction.make ~batch:1L ~m:(i64 m) ~n:(i64 n) ~k:(i64 k)
          ~a:{ Contraction.Operand.batch = 0L; outer = i64 k; k = 1L }
          ~b:{ Contraction.Operand.batch = 0L; outer = 1L; k = i64 n }
          ~c:{ Contraction.Output.batch = 0L; row = i64 n; col = 1L }
          ~format:Contraction.Element.F32 ~seed:Contraction.Seed.Zero
          ~numerics:Contraction.Numerics.Ordered))

let buffers ~m ~k ~n =
  [
    F.buffer 0 ~h:(i64 m) ~w:(i64 k) Ssa_format.F32 Ssa_buffer.Input;
    F.buffer 1 ~h:(i64 k) ~w:(i64 n) Ssa_format.F32 Ssa_buffer.Input;
    F.buffer 2 ~h:(i64 m) ~w:(i64 n) Ssa_format.F32 Ssa_buffer.Output;
  ]

let whole ~m ~k ~n =
  F.build ~buffers:(buffers ~m ~k ~n) (fun bld ->
      B.contract bld (desc ~m ~k ~n) ~a:(F.buf 0) ~b:(F.buf 1) ~c:(F.buf 2))

(* Blocks of [r] rows from a loop over [blocks] full blocks, then the tail, each
   call over views whose bases move by the block. *)
let banded_program ~m ~k ~n ~r ~blocks =
  B.program ~buffers:(buffers ~m ~k ~n) (fun bld ->
      let call bld ~rows ~at =
        let base stride = B.index_scale bld (i64 stride) at in
        B.view_call bld (desc ~m:rows ~k ~n)
          ~a:(F.buf 0, base (r * k))
          ~b:(F.buf 1, B.index bld 0L)
          ~c:(F.buf 2, base (r * n))
      in
      ignore
        (B.for_ bld ~lo:(B.index bld 0L)
           ~hi:(B.index bld (i64 blocks))
           ~init:B.Nil
           (fun bld blk B.Nil ->
             call bld ~rows:r ~at:blk;
             B.Nil));
      let tail = m - (blocks * r) in
      if tail > 0 then call bld ~rows:tail ~at:(B.index bld (i64 blocks)))

let banded ~m ~k ~n ~r =
  Err.or_raise ~pp_error:Ssa_verify.pp_error
    (banded_program ~m ~k ~n ~r ~blocks:(m / r))

let lcg = ref 11

let value () =
  lcg := ((!lcg * 1103515245) + 12345) land 0x3FFFFFFF;
  F32_math.round32 (float_of_int ((!lcg / 7 mod 4001) - 2000) /. 977.)

let run p ~a ~b ~nc =
  let memory =
    F.memory
      [ (0, F.floats a); (1, F.floats b); (2, F.floats (Array.make nc 0.)) ]
  in
  F.run p ~memory;
  match Ssa_id.Buffer.Map.find (F.buf 2) memory with
  | Ssa_memory.Floats cs -> cs
  | _ -> assert false

let same x y =
  Array.for_all2 (fun p q -> Int32.bits_of_float p = Int32.bits_of_float q) x y

let count_view_calls p =
  let n = ref 0 in
  let rec region (r : Ssa_region.t) = List.iter stmt r.Ssa_region.body
  and stmt = function
    | Ssa_stmt.Instr { op = Ssa_op.View_call _; _ } -> incr n
    | Ssa_stmt.Instr _ -> ()
    | Ssa_stmt.For f -> region f.body
    | Ssa_stmt.If f ->
        region f.then_;
        region f.else_
    | Ssa_stmt.Ordered_sum f -> region f.body
  in
  region p.Ssa_program.entry;
  !n

let case ~m ~k ~n ~r =
  let a = Array.init (m * k) (fun _ -> value ())
  and b = Array.init (k * n) (fun _ -> value ()) in
  let nc = m * n in
  let want = run (whole ~m ~k ~n) ~a ~b ~nc in
  let p = banded ~m ~k ~n ~r in
  let got = run p ~a ~b ~nc in
  let expanded = Ssa_expand_view_call.program p in
  let through = run (Ssa_precision.to_f32 ~exact:true expanded) ~a ~b ~nc in
  Fmt.pr "m=%d r=%d: view calls %d; banded %s; expanded (%d left) %s@." m r
    (count_view_calls p)
    (if same want got then "equals Contract" else "DIFFERS")
    (count_view_calls expanded)
    (if same want through then "equals Contract" else "DIFFERS")

let%expect_test "banded view calls equal the whole Contract" =
  case ~m:6 ~k:4 ~n:3 ~r:2;
  case ~m:7 ~k:5 ~n:4 ~r:3;
  case ~m:5 ~k:3 ~n:2 ~r:5;
  [%expect
    {|
    m=6 r=2: view calls 1; banded equals Contract; expanded (0 left) equals Contract
    m=7 r=3: view calls 2; banded equals Contract; expanded (0 left) equals Contract
    m=5 r=5: view calls 1; banded equals Contract; expanded (0 left) equals Contract |}]

let%expect_test "a base past the last block is refused, with the unproven bound"
    =
  (match Err.payload (banded_program ~m:6 ~k:4 ~n:3 ~r:2 ~blocks:4) with
  | Ok _ -> Fmt.pr "accepted@."
  | Error e -> Fmt.pr "%a@." Ssa_verify.pp_error e);
  [%expect
    {| r0, stmt3: a proof of an access staying in its buffer cannot be re-derived |}]

let%expect_test "the CFG executor runs the banded program to the same bits" =
  let m = 7 and k = 5 and n = 4 and r = 3 in
  let a = Array.init (m * k) (fun _ -> value ())
  and b = Array.init (k * n) (fun _ -> value ()) in
  let want = run (whole ~m ~k ~n) ~a ~b ~nc:(m * n) in
  let cfg =
    Err.or_raise ~pp_error:Ssa_cfg_lower.pp_error
      (Ssa_cfg_lower.program (banded ~m ~k ~n ~r))
  in
  let memory =
    F.memory
      [
        (0, F.floats a); (1, F.floats b); (2, F.floats (Array.make (m * n) 0.));
      ]
  in
  Err.or_raise ~pp_error:Ssa_cfg_interp.pp_error
    (Ssa_cfg_interp.run cfg ~memory);
  (match Ssa_id.Buffer.Map.find (F.buf 2) memory with
  | Ssa_memory.Floats got ->
      Fmt.pr "%s@." (if same want got then "equals Contract" else "DIFFERS")
  | _ -> assert false);
  [%expect {| equals Contract |}]

let%expect_test "the printer names each operand's base" =
  Fmt.pr "%a@." Ssa_pp.pp (banded ~m:4 ~k:2 ~n:2 ~r:2);
  [%expect
    {|
    buffer b0 in f32 [1, 1, 1, 4, 2, 1]
    buffer b1 in f32 [1, 1, 1, 2, 2, 1]
    buffer b2 out f32 [1, 1, 1, 4, 2, 1]
    entry(%0:effect) {
      (%1:index) = const 2:index
      (%2:index) = const 0:index
      (%3:effect) = for %2 to %1 step 1 iter(%4:index, %5:effect := %0) {
        (%6:index, %7:effect) = index.scale 4, %4 effect %5
        (%8:index) = const 0:index
        (%9:index, %10:effect) = index.scale 4, %4 effect %7
        (%11:effect) = view contract f32 batch=1 m=2 n=2 k=2 a=(0,2,1) b=(0,1,2) c=(0,2,1) seed=zero ordered a=b0+%9 b=b1+%8 c=b2+%6 effect %10
        yield %11
      }
      yield %3
    } |}]
