(* Deliberately wrong semantic entries and mapping mutations, each of which must
   make some conformance instance disagree. Every one reruns the whole instance
   set, so they are spread over several test partitions to keep each one short. *)

open Machine_target_riscv32
module O = Rv32_op
module F = Machine_rivet_riscv32_mir.Rv32_mir_form
module E = Machine_interp.Mir_sel_env

let summary = Conformance_test.summary

(* A semantic entry made deliberately wrong: the semantics with one form's
   meaning replaced. *)
let with_form f env op =
  match f env op with Some r -> r | None -> Rv32_sem.exec env op

let bits env x = E.bits env x
let fb x = Machine_interp.Mir_numeric.f32 x
let b x = Machine_interp.Mir_datum.Bits x

let fmax_ieee =
  ( "fmax.s as IEEE maximum",
    fun env -> function
      | O.Fbin (O.Fop.Max, x, y) ->
          let x = fb (bits env x) and y = fb (bits env y) in
          Some [ b (Rv32_sem.fres (Machine_interp.Mir_numeric.fmax x y)) ]
      | _ -> None )

let fcvt_nan_zero =
  ( "fcvt.w.s of a NaN as 0",
    fun env -> function
      | O.Fcvt_w_s x when Float.is_nan (fb (bits env x)) -> Some [ b 0L ]
      | _ -> None )

let div_zero =
  ( "div by zero as 0",
    fun env -> function
      | O.Alu (O.Alu.Div, _, y) when Int64.equal (bits env y) 0L ->
          Some [ b 0L ]
      | _ -> None )

let sltiu_zero_extended =
  ( "sltiu immediate zero-extended",
    fun env -> function
      | O.Alui (O.Alui.Sltiu, x, k) ->
          let k = Int64.logand k 0xFFFL in
          Some
            [ b (if Int64.unsigned_compare (bits env x) k < 0 then 1L else 0L) ]
      | _ -> None )

let nan_not_canonical =
  ( "a NaN result not canonical",
    fun env -> function
      | O.Fbin (O.Fop.Add, x, y) ->
          let r = Rv32_sem.exec env (O.Fbin (O.Fop.Add, x, y)) in
          Some
            (List.map
               (function
                 | Machine_interp.Mir_datum.Bits 0x7FC0_0000L -> b 0x7FC0_0001L
                 | d -> d)
               r)
      | _ -> None )

let commuted_sub = ("commuted sub", F.Mutation.Commuted_sub)
let select_arms = ("select arms", F.Mutation.Select_arms)
let branch_sense = ("branch sense", F.Mutation.Branch_sense)

let report_semantic (name, f) =
  let _, _, failures = summary ~sem:(with_form f) () in
  Fmt.pr "%s: %d instances disagree@." name (List.length failures)

let report_mapping (name, mutation) =
  let _, _, failures = summary ~mutation ~sem:Rv32_sem.exec () in
  Fmt.pr "map %s: %d instances disagree@." name (List.length failures)
