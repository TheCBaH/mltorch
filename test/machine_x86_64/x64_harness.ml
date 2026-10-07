(* The x86-64 routes of a lowered case: selected, allocated (reference) and
   realized, each against the generic route and the SSA oracle. *)
open Machine_ir
open Machine_interp
open Machine_target_x86_64
module Src = Machine_source_test.Mir_source
module A = Machine_alloc.Mir_ref_alloc.Make (X64) (X64_regs)
module V = Mir_phys_verify.Make (X64)
module C = Machine_check.Mir_checker.Make (X64)
module P = Mir_phys_interp.Make (X64)
module Fr = Machine_alloc.Mir_frame.Make (X64) (X64_frame)

let regions_program = Machine_alloc_test.Alloc_harness.regions_program

let record_region (res : X64_select.result) =
  let sel = X64_stage.Sel.Verified.selected res.X64_select.selected in
  (Option.get
     (Mir_program.find_view sel.X64_stage.Sel.program res.X64_select.record))
    .Mir_view.region

let selected ?mutation ?features (case : Src.Case.t) =
  match
    Err.payload
      (X64_select.program ?mutation ?features
         case.Src.Case.lowered.Machine_lower.Mir_lower.program)
  with
  | Error r -> Error (Fmt.str "%a" X64_select.Refusal.pp r)
  | Ok res ->
      let sel = X64_stage.Sel.Verified.selected res.X64_select.selected in
      let memory = Mir_memory.create () in
      let binding =
        Result.get_ok
          (Mir_interp.instantiate sel.X64_stage.Sel.program memory
             ~bound:case.Src.Case.bound)
      in
      let r =
        X64_stage.Interp.run res.X64_select.selected memory binding ~args:[]
      in
      Ok
        ( res,
          Src.selected_observation
            ~layout:case.Src.Case.lowered.Machine_lower.Mir_lower.layout
            ~sites:[||] ~record:(record_region res) memory binding
            r.X64_stage.Interp.outcome r.X64_stage.Interp.events )

(* Callee-saved registers as a caller might leave them. *)
let caller_state regs =
  List.iter
    (fun k -> P.seed regs (X64_reg.q k) ~lo:(Int64.of_int (0x5A00 + k)) ~hi:0L)
    [ 3; 5; 12; 13; 14; 15 ]

type stage = Allocated | Realized | Selected

let route ?mutation ?features ?alloc_mutation ?frame_mutation ?pad stage
    (case : Src.Case.t) =
  match selected ?mutation ?features case with
  | Error e -> Error ("refused: " ^ e)
  | Ok (res, obs) -> (
      match stage with
      | Selected -> Ok obs
      | Allocated | Realized -> (
          let phys =
            A.allocate ?mutation:alloc_mutation res.X64_select.selected
          in
          let realized = stage = Realized in
          let phys =
            if realized then
              Result.map_error
                (Fmt.str "frame: %a" Machine_alloc.Mir_frame.Refusal.pp)
                (Fr.realize ?mutation:frame_mutation ?pad phys)
            else Ok phys
          in
          match phys with
          | Error e -> Error ("rejected: " ^ e)
          | Ok phys -> (
              match Err.payload (V.verify phys) with
              | Error d ->
                  Error
                    (Fmt.str "rejected: physical verifier: %a" Mir_diagnostic.pp
                       d)
              | Ok phys -> (
                  match Err.payload (C.check res.X64_select.selected phys) with
                  | Error e ->
                      Error
                        (Fmt.str "rejected: checker: %a"
                           Machine_check.Mir_checker.pp_error e)
                  | Ok () ->
                      let memory = Mir_memory.create () in
                      let binding =
                        Result.get_ok
                          (Mir_interp.instantiate (regions_program phys) memory
                             ~bound:case.Src.Case.bound)
                      in
                      let r =
                        P.run ~realized
                          ~seed:(if realized then caller_state else fun _ -> ())
                          phys memory binding ~args:[]
                      in
                      Ok
                        (Src.selected_observation
                           ~layout:
                             case.Src.Case.lowered
                               .Machine_lower.Mir_lower.layout ~sites:[||]
                           ~record:(record_region res) memory binding
                           r.P.outcome r.P.events)))))

let report ?mutation ?features ?alloc_mutation ?frame_mutation ?pad
    ?(stage = Selected) case =
  match
    route ?mutation ?features ?alloc_mutation ?frame_mutation ?pad stage case
  with
  | Error e -> e
  | Ok obs -> (
      let x = { Src.Route.name = "x86_64"; observation = obs } in
      let ds =
        List.filter_map Fun.id
          [
            Src.verdict ~expected:case.Src.Case.generic ~actual:x;
            Src.verdict ~expected:case.Src.Case.oracle ~actual:x;
          ]
      in
      Src.status_name obs
      ^ match ds with [] -> "" | d -> " DISAGREE " ^ String.concat "; " d)

let plan ?mutation ?features ?alloc_mutation ?frame_mutation ?pad ?stage kernel
    ~bind =
  match Src.case_of_plan (Fusion_plan.default kernel) ~bind with
  | Error e -> e
  | Ok c ->
      report ?mutation ?features ?alloc_mutation ?frame_mutation ?pad ?stage c

let program ?mutation ?features ?alloc_mutation ?frame_mutation ?pad ?stage ?fma
    p ~inputs =
  match Src.case_of_program p ~inputs ?fma () with
  | Error e -> e
  | Ok c ->
      report ?mutation ?features ?alloc_mutation ?frame_mutation ?pad ?stage c
