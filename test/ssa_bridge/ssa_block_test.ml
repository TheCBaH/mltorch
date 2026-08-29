open Ssa_bridge
open Ssa_fixtures
open Ssa_ir

(* Independent-output blocking, on matmul: whether the pass fires, that every
   output keeps its value and the logical work, and what it saves in reads. *)

let alias = Ssa_effects.Distinct_buffers

let run_pipeline ~group p =
  let passes =
    [
      Ssa_opt.simplify;
      Ssa_opt.guards;
      Ssa_opt.simplify;
      Ssa_opt.hoist ~alias;
      Ssa_opt.block ~alias ~group;
      Ssa_opt.share ~alias;
      Ssa_opt.simplify;
    ]
  in
  fst (Ssa_opt.run ~alias ~passes p)

let case ~group (m, k, n) =
  let plan = Fusion_plan.default (matmul_kernel ~m ~k ~n) in
  let a = operand 3 (m * k) and b = operand 5 (k * n) in
  let bind = matmul_bind ~m ~k ~n ~a ~b in
  let program =
    match Err.payload (Ssa_lower.Ssa_lower_plan.lower plan) with
    | Ok p -> p
    | Error _ -> failwith "refused"
  in
  let blocked = run_pipeline ~group program in
  let plain = run_pipeline ~group:(Ssa_opt_block.Fixed 1_000_000) program in
  let counters p =
    let c = Ssa_interp.Counters.create () in
    ignore (Ssa_lower.Ssa_exec.run ~counters:c plan p ~bind);
    c
  in
  let cb = counters blocked and cp = counters plain in
  let marks c = List.map (Ssa_interp.Counters.mark c) Ssa_mark.all in
  Fmt.pr "%dx%dx%d: %a; same work: %b; loads %d -> %d; loops %d -> %d@." m k n
    Ssa_check.pp_verdict
    (Ssa_check.run ~prepare:(run_pipeline ~group) plan ~bind)
    (marks cb = marks cp)
    (Ssa_interp.Counters.loads cp)
    (Ssa_interp.Counters.loads cb)
    (Ssa_stats.of_program plain).Ssa_stats.loops
    (Ssa_stats.of_program blocked).Ssa_stats.loops

let%expect_test "matmul blocked: values, work and reads" =
  List.iter
    (case ~group:Ssa_opt_block.Auto)
    [ (1, 1, 1); (3, 1, 4); (5, 7, 3); (4, 4, 4); (2, 9, 1); (3, 5, 9) ];
  [%expect
    {|
    1x1x1: agree; same work: true; loads 2 -> 2; loops 1 -> 1
    3x1x4: agree; same work: true; loads 24 -> 7; loops 3 -> 1
    5x7x3: agree; same work: true; loads 210 -> 175; loops 3 -> 3
    4x4x4: agree; same work: true; loads 128 -> 80; loops 3 -> 2
    2x9x1: agree; same work: true; loads 36 -> 27; loops 2 -> 1
    3x5x9: agree; same work: true; loads 270 -> 180; loops 3 -> 4 |}]

let%expect_test "every group size agrees, remainders included" =
  List.iter
    (fun g -> case ~group:(Ssa_opt_block.Fixed g) (3, 5, 11))
    [ 2; 3; 4; 8; 11 ];
  [%expect
    {|
    3x5x11: agree; same work: true; loads 330 -> 255; loops 3 -> 4
    3x5x11: agree; same work: true; loads 330 -> 240; loops 3 -> 5
    3x5x11: agree; same work: true; loads 330 -> 240; loops 3 -> 5
    3x5x11: agree; same work: true; loads 330 -> 225; loops 3 -> 4
    3x5x11: agree; same work: true; loads 330 -> 180; loops 3 -> 2 |}]

(* ---- what the pass refuses, by the condition that fails ---------------------- *)

module B = Ssa_builder
open Ssa_ir_test.Ssa_fixtures

let bufs =
  [
    buffer 0 ~h:1L ~w:8L Ssa_format.F32 Ssa_buffer.Input;
    buffer 1 ~h:1L ~w:8L Ssa_format.F32 Ssa_buffer.Output;
    buffer 2 ~h:1L ~w:8L Ssa_format.F32 Ssa_buffer.Scratch;
  ]

let idx bld n = B.index bld (Int64.of_int n)

let load_at bld id i =
  B.load_f64 bld (buf id) ~decode:Ssa_op.Decode.F32_to_f64
    (at bld ~h:(idx bld 0) ~w:i)

let store_at bld id i x =
  B.store_f64 bld (buf id) ~encode:Ssa_op.Encode.F32_round
    (at bld ~h:(idx bld 0) ~w:i)
    x

(* for w in [0, 4): out[w] = sum over k in [0, 3) of in[k] * in[w + k], with the
   term, the buffer written, and the stored address varied. *)
let outputs ?(source = 0)
    ?(term =
      fun bld ~k ~w ->
        B.f64_binary bld Expr.Value.Mul (load_at bld source k)
          (load_at bld source (B.index_add bld w k))) ?(target = 1)
    ?(store_at_w = fun _ w -> w) ?(hi = 4) () =
  match
    Err.payload
      (B.program ~buffers:bufs (fun bld ->
           let _ =
             B.for_ bld ~lo:(idx bld 0) ~hi:(idx bld hi) ~init:B.Nil
               (fun bld w B.Nil ->
                 let s =
                   B.ordered_sum bld ~lo:(idx bld 0) ~hi:(idx bld 3)
                     ~seed:(B.f64 bld 0.) (fun bld k -> term bld ~k ~w)
                 in
                 store_at bld target (store_at_w bld w) s;
                 B.Nil)
           in
           ()))
  with
  | Ok p -> p
  | Error e -> Fmt.failwith "%a" Ssa_verify.pp_error e

let decisions ?(group = Ssa_opt_block.Auto) p =
  let p =
    fst
      (Ssa_opt.run ~alias
         ~passes:[ Ssa_opt.simplify; Ssa_opt.guards; Ssa_opt.simplify ]
         p)
  in
  match Ssa_opt_block.analyze ~policy:alias ~group p with
  | [] -> Fmt.pr "not a candidate@."
  | l ->
      List.iter
        (fun (c : Ssa_opt_block.candidate) ->
          match c.decision with
          | Ok g -> Fmt.pr "blocks by %d (%Ld trips)@." g c.trips
          | Error r -> Fmt.pr "refused: %s@." (Ssa_opt_block.refusal_name r))
        l

let%expect_test "blocking is refused by the condition that fails" =
  decisions (outputs ());
  (* a term that can still fail *)
  decisions
    (outputs
       ~term:(fun bld ~k ~w:_ ->
         B.i64_to_f64 bld (B.float_to_i64 bld (load_at bld 0 k)))
       ());
  (* the output buffer is also read *)
  decisions (outputs ~source:2 ~target:2 ());
  (* every iteration stores to the same cell *)
  decisions (outputs ~store_at_w:(fun bld _ -> idx bld 0) ());
  (* nothing the reduction reads is shared across outputs *)
  decisions (outputs ~term:(fun bld ~k:_ ~w -> load_at bld 0 w) ());
  (* too few outputs for the group asked for *)
  decisions ~group:(Ssa_opt_block.Fixed 8) (outputs ());
  (* a scratch object in the reduction *)
  decisions
    (outputs
       ~term:(fun bld ~k ~w:_ ->
         let l = B.local_alloc bld ~slots:1L in
         B.local_write bld l (idx bld 0) (load_at bld 0 k);
         load_at bld 0 k)
       ());
  [%expect
    {|
    blocks by 4 (4 trips)
    refused: the body still contains an operation that can fail
    refused: the body reads a buffer it may write
    refused: the stores are not at addresses that differ per iteration
    refused: no operand is shared across the group
    refused: fewer iterations than one group
    refused: the body touches the scan meter or a scratch object |}]
