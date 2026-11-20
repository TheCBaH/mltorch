(* The Coral family: what the Coral NPU route makes of one invocation. The
   estimate counts the invocation's final SSA program (the one [Mir_model]
   lowers, planned for contractions as the Coral route plans them) by cost class
   and prices it with the VME core's committed rates; the other views are the
   kernels the route generates for the program's matrix products. *)

module Census = Coral_estimate.Coral_census
module Model = Coral_estimate.Coral_model
module Refusal = Lowering_artifact.Refusal
module Note = Lowering_artifact.Note
module Language = Lowering_artifact.Language
module M = Machine_model.Mir_model
module K = Lowering_key
module Tp = Coral_transfer.Transfer_plan
module Tm = Coral_transfer.Transfer_model

let ( let* ) = Result.bind

(* The invocation's position in the schedule is not known to a per-node view; a
   refusal reports 0, as the Machine IR family does. *)
let pipeline =
  Ssa_backends.Pipeline.Planned
    {
      numerics = Ssa_ir.Ssa_numerics.Fp32_ordered;
      target = Ssa_ir.Ssa_target.scalar;
    }

(* The final program, with each product of a binary32 plan one helper call. *)
let census inv =
  let program = ref None in
  match
    M.lower_one ~contractions:true
      ~on_program:(fun _ p _ -> program := Some p)
      ~pipeline ~invocation:0l inv
  with
  | Error r -> Error (Refusal.Mir_lowering r)
  | Ok _ -> (
      match !program with
      | Some p -> Ok (Census.program p)
      | None -> invalid_arg "Lowering_coral: lowered without a final program")

let region = Census.Region.Dtcm

(* The route's rule: the packed function for a paired bfloat16 product whose
   panels fit the five eighths of the DTCM. [fast] raises on what it refuses,
   so [panel_bytes] is asked first. *)
let core =
  let module P = Coralnpu_kernels_zvt.Zvt_packed_contraction in
  Model.Core.vme ~packed:(fun d ->
      d.Contraction.numerics = Contraction.Numerics.Paired
      &&
      match P.panel_bytes d with
      | Ok b -> (b :> int) <= 32768 * 5 / 8 && P.fast d
      | Error _ -> false)

let estimate c =
  Model.Rates.predict Coral_estimate.Coral_rates.vme
    (Model.features ~rates:Coral_estimate.Coral_rates.vme core c)

(* A DTCM of 32 KiB, the budget the plan is chosen within. *)
let budget = Coral_transfer.Transfer_units.Bytes.of_int 32768

(* Each contraction the program calls, once per distinct descriptor. *)
let contractions (c : Census.t) =
  List.fold_left
    (fun acc ((call : Census.Contract_call.t), _) ->
      let d = call.Census.Contract_call.desc in
      if List.exists (fun d' -> d = d') acc then acc else acc @ [ d ])
    [] c.Census.contracts

let print m =
  Lowering_asm.print ~type_char:'@' ~instruction:Riscv32_encode.Instruction.pp m

let sections f ds =
  let multi = List.length ds > 1 in
  let* parts =
    List.fold_left
      (fun acc d ->
        let* acc = acc in
        let* text = f d in
        Ok
          (acc
          @ [
              (if multi then Fmt.str "# %s@.%s" (Contraction.symbol d) text
               else text);
            ]))
      (Ok []) ds
  in
  Ok (String.concat "\n" parts)

let kernel_text make d =
  match make d with
  | Ok m -> Ok (print m)
  | Error r -> Error (Refusal.Coral_kernel r)

let rvv d =
  let* w =
    Result.map_error
      (fun r -> Refusal.Coral_transfer r)
      (Tp.check ~budget d (Tm.choose ~budget d))
  in
  kernel_text (Coralnpu_kernels.Rvv_contraction.module_ ~plan:w) d

let zvt d =
  kernel_text (fun d -> Coralnpu_kernels_zvt.Zvt_contraction.module_ d) d

let plan d =
  let p = Tm.choose ~budget d in
  let* _ =
    Result.map_error (fun r -> Refusal.Coral_transfer r) (Tp.check ~budget d p)
  in
  Ok (Fmt.str "%a@.predicted cycles: %.0f" Tp.pp p (Tm.predict d p))

let of_invocation (view : K.Coral_view.t) inv =
  let* c = census inv in
  let kernel language f =
    match contractions c with
    | [] -> Error Refusal.Not_a_contraction
    | ds ->
        let* text = sections f ds in
        Ok (language, text, [ Note.Coral_transfer_budget budget ])
  in
  match view with
  | K.Coral_view.Estimate ->
      Ok
        ( Language.Coral_estimate,
          Fmt.str "%a@.estimated cycles: %.0f" Census.pp c (estimate c),
          [ Note.Coral_region_assumed region ] )
  | K.Coral_view.Rvv_kernel -> kernel (Language.Gnu_asm K.Target.Riscv32) rvv
  | K.Coral_view.Transfer_plan -> kernel Language.Coral_plan plan
  | K.Coral_view.Zvt_kernel -> kernel (Language.Gnu_asm K.Target.Riscv32) zvt
