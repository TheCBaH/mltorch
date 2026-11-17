open Ssa_ir
module B = Ssa_builder
module F = Ssa_ir_test.Ssa_fixtures

(* A check ahead of a load of the same buffer and coordinate is the check the
   load makes again, so the guards pass drops it. The coordinate here is read
   from data, which no range can prove in bounds, so the load's own check is
   the only one left: a pass that kept the pair would show two. *)

let bufs =
  [
    F.buffer 0 ~h:1L ~w:1L Ssa_format.I64 Ssa_buffer.Input;
    F.buffer 1 ~h:1L ~w:8L Ssa_format.F32 Ssa_buffer.Input;
    F.buffer 2 ~h:1L ~w:1L Ssa_format.F32 Ssa_buffer.Output;
  ]

let program ~checked_twice =
  let zero bld = B.index bld 0L in
  match
    Err.payload
      (B.program ~buffers:bufs (fun bld ->
           let raw =
             B.load_i64 bld (F.buf 0) (F.at bld ~h:(zero bld) ~w:(zero bld))
           in
           let j = B.index_of_i64 bld raw in
           let at = F.at bld ~h:(zero bld) ~w:j in
           if checked_twice then B.check_access bld (F.buf 1) at;
           let x =
             B.load_f64 bld (F.buf 1) ~decode:Ssa_op.Decode.F32_to_f64 at
           in
           B.store_f64 bld (F.buf 2) ~encode:Ssa_op.Encode.F32_round
             (F.at bld ~h:(zero bld) ~w:(zero bld))
             x))
  with
  | Ok p -> p
  | Error e -> Fmt.failwith "%a" Ssa_verify.pp_error e

let checked p = (Ssa_stats.of_program p).Ssa_stats.checked

let%expect_test "the check a load repeats is dropped, the load's own stays" =
  let run p =
    fst
      (Ssa_opt.run ~alias:Ssa_effects.Distinct_buffers
         ~passes:[ Ssa_opt.guards ] p)
  in
  let once = program ~checked_twice:false in
  let twice = program ~checked_twice:true in
  Fmt.pr "once: %d -> %d@.twice: %d -> %d@." (checked once)
    (checked (run once))
    (checked twice)
    (checked (run twice));
  [%expect {|
    once: 3 -> 2
    twice: 4 -> 2 |}]
