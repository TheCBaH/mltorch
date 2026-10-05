open Ssa_ir
open Ssa_fixtures

(* Rename every value id with [f]: the same structure allocated in a different
   order, which is what two builder histories of one program differ by. *)
let rename f (p : Ssa_program.t) =
  let value (v : Ssa_value.t) =
    { v with Ssa_value.id = Ssa_id.Value.of_int (f (v.Ssa_value.id :> int)) }
  in
  let rec region (r : Ssa_region.t) =
    {
      r with
      Ssa_region.params = List.map value r.Ssa_region.params;
      body = List.map stmt r.Ssa_region.body;
      yields = List.map value r.Ssa_region.yields;
    }
  and stmt : Ssa_region.t Ssa_stmt.t -> Ssa_region.t Ssa_stmt.t = function
    | Ssa_stmt.For { lo; hi; step; inits; results; body } ->
        Ssa_stmt.For
          {
            lo = value lo;
            hi = value hi;
            step;
            inits = List.map value inits;
            results = List.map value results;
            body = region body;
          }
    | Ssa_stmt.If { cond; results; then_; else_ } ->
        Ssa_stmt.If
          {
            cond = value cond;
            results = List.map value results;
            then_ = region then_;
            else_ = region else_;
          }
    | Ssa_stmt.Instr i ->
        Ssa_stmt.Instr
          {
            i with
            Ssa_instr.results = List.map value i.Ssa_instr.results;
            op = Ssa_op.map_operands value i.Ssa_instr.op;
            token = Option.map value i.Ssa_instr.token;
          }
    | Ssa_stmt.Ordered_sum { lo; hi; seed; token; results; body } ->
        Ssa_stmt.Ordered_sum
          {
            lo = value lo;
            hi = value hi;
            seed = value seed;
            token = value token;
            results = List.map value results;
            body = region body;
          }
  in
  { p with Ssa_program.entry = region p.Ssa_program.entry }

let%expect_test "the text is independent of how ids were allocated" =
  let p = matmul ~m:2 ~k:2 ~n:2 in
  let renamed = rename (fun i -> 1000 - (3 * i)) p in
  Fmt.pr "same=%b valid=%b@."
    (String.equal (Ssa_pp.to_string p) (Ssa_pp.to_string renamed))
    (Result.is_ok (Err.payload (Ssa_verify.check renamed)));
  [%expect {| same=true valid=true |}]

let%expect_test "types, effects, buffers and nesting are printed" =
  let b = buffer 0 ~h:1L ~w:2L Ssa_format.F32 Ssa_buffer.Output in
  let p =
    build ~buffers:[ b ] (fun bld ->
        Ssa_builder.set_origin bld (Ssa_origin.Output (buf 0));
        let zero = Ssa_builder.index bld 0L in
        let two = Ssa_builder.index bld 2L in
        let seed = Ssa_builder.f64 bld 0. in
        let sum =
          Ssa_builder.ordered_sum bld ~lo:zero ~hi:two ~seed (fun bld i ->
              Ssa_builder.mark bld Ssa_mark.Reduction;
              Ssa_builder.index_to_f64 bld i)
        in
        Ssa_builder.store_f64 bld (buf 0) ~encode:Ssa_op.Encode.F32_round
          (Ssa_builder.Flat zero) sum)
  in
  print_string (Ssa_pp.to_string p);
  [%expect
    {|
    buffer b0 out f32 [1, 1, 1, 1, 2, 1]
    entry(%0:effect) {
      (%1:index) = const 0:index  ; out b0
      (%2:index) = const 2:index  ; out b0
      (%3:f64) = const 0x0p+0:f64  ; out b0
      (%4:f64, %5:effect) = ordered_sum %1 to %2 iter(%6:index, %7:effect := %3, %0) {
        (%8:effect) = mark reduction effect %7  ; out b0
        (%9:f64) = convert.index_to_f64 %6  ; out b0
        yield %9, %8
      }
      (%10:effect) = store.f32_round b0@%1, %4 effect %5  ; out b0
      yield %10
    }
    |}]
