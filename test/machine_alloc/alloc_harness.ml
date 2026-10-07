(* The allocated route: select, allocate (reference), verify physically,
   check symbolically, then run on the physical interpreter and compare with
   the selected route. A checker rejection is a compiler error and the
   artifact is never run. *)
open Machine_ir
open Machine_interp
open Machine_target_aarch64
module Src = Machine_source_test.Mir_source
module A = Machine_alloc.Mir_ref_alloc.Make (A64) (A64_regs)
module V = Mir_phys_verify.Make (A64)
module C = Machine_check.Mir_checker.Make (A64)
module P = Mir_phys_interp.Make (A64)

let regions_program (p : (_, _) Mir_phys.Program.t) =
  {
    Mir_program.data_model = p.Mir_phys.Program.data_model;
    regions = p.Mir_phys.Program.regions;
    views = p.Mir_phys.Program.views;
    helpers = [];
    funcs = [];
    main = p.Mir_phys.Program.main;
    planning = None;
    revision = Mir_id.Revision.of_int 0;
  }

type verdict = Checked of Mir_observation.t | Rejected of string

(* Allocation, verification and checking of a selected program; [edit]
   rewrites the allocation first (a hand-made defect). *)
let allocate ?mutation ?(edit = Fun.id) (res : A64_select.result) =
  let phys = edit (A.allocate ?mutation res.A64_select.selected) in
  match Err.payload (V.verify phys) with
  | Error d -> Error (Fmt.str "physical verifier: %a" Mir_diagnostic.pp d)
  | Ok phys -> (
      match Err.payload (C.check res.A64_select.selected phys) with
      | Error e ->
          Error (Fmt.str "checker: %a" Machine_check.Mir_checker.pp_error e)
      | Ok () -> Ok phys)

let physical ?mutation ?edit ?(sites = [||]) (case : Src.Case.t) =
  match Machine_aarch64_test.A64_harness.selected ~sites case with
  | Error e -> Rejected ("selection: " ^ e)
  | Ok (res, _) -> (
      match allocate ?mutation ?edit res with
      | Error e -> Rejected e
      | Ok phys ->
          let memory = Mir_memory.create () in
          let binding =
            Result.get_ok
              (Mir_interp.instantiate (regions_program phys) memory
                 ~bound:case.Src.Case.bound)
          in
          let r =
            P.run ~models:Machine_interp.Mir_math_model.all phys memory binding
              ~args:[]
          in
          let sel = A64_stage.Sel.Verified.selected res.A64_select.selected in
          let record =
            (Option.get
               (Mir_program.find_view sel.A64_stage.Sel.program
                  res.A64_select.record))
              .Mir_view.region
          in
          Checked
            (Src.selected_observation
               ~layout:case.Src.Case.lowered.Machine_lower.Mir_lower.layout
               ~sites ~record memory binding r.P.outcome r.P.events))

let report ?mutation ?edit ?sites case =
  match physical ?mutation ?edit ?sites case with
  | Rejected e -> "rejected: " ^ e
  | Checked obs -> (
      let alloc = { Src.Route.name = "allocated"; observation = obs } in
      let sel =
        match Machine_aarch64_test.A64_harness.selected ?sites case with
        | Ok (_, o) -> Some { Src.Route.name = "aarch64"; observation = o }
        | Error _ -> None
      in
      let ds =
        List.filter_map Fun.id
          [
            Option.bind sel (fun s -> Src.verdict ~expected:s ~actual:alloc);
            Src.verdict ~expected:case.Src.Case.oracle ~actual:alloc;
          ]
      in
      Src.status_name obs
      ^ match ds with [] -> "" | d -> " DISAGREE " ^ String.concat "; " d)

let plan ?mutation ?edit kernel ~bind =
  match Src.case_of_plan (Fusion_plan.default kernel) ~bind with
  | Error e -> e
  | Ok c -> report ?mutation ?edit c

let program ?mutation ?edit ?sites ?fma ?precision p ~inputs =
  match Src.case_of_program p ~inputs ?fma ?precision () with
  | Error e -> e
  | Ok c -> report ?mutation ?edit ?sites c

module Fr = Machine_alloc.Mir_frame.Make (A64) (A64_frame)

(* Callee-saved registers and FPCR as a caller might leave them. *)
let caller_state regs =
  List.iter
    (fun k -> P.seed regs (A64_reg.x k) ~lo:(Int64.of_int (0x1900 + k)) ~hi:0L)
    (A64_reg.range 19 29);
  List.iter
    (fun k ->
      P.seed regs (A64_reg.q k)
        ~lo:(Int64.of_int (0x800 + k))
        ~hi:(Int64.of_int (0x8000 + k)))
    (A64_reg.range 8 15);
  (* flush-to-zero set by the caller: the entry function must replace it *)
  P.seed regs A64_reg.fpcr ~lo:0x0100_0000L ~hi:0L

(* Realizes frames, then verifies and checks the realized program. *)
let realize ?mutation ?pad ?edit res =
  match allocate ?edit res with
  | Error e -> Error e
  | Ok phys -> (
      match Fr.realize ?mutation ?pad phys with
      | Error r ->
          Error (Fmt.str "frame: %a" Machine_alloc.Mir_frame.Refusal.pp r)
      | Ok real -> (
          match Err.payload (V.verify real) with
          | Error d ->
              Error (Fmt.str "physical verifier: %a" Mir_diagnostic.pp d)
          | Ok real -> (
              match Err.payload (C.check res.A64_select.selected real) with
              | Error e ->
                  Error
                    (Fmt.str "checker: %a" Machine_check.Mir_checker.pp_error e)
              | Ok () -> Ok real)))

(* The realized route of a lowered case. *)
let realized_report ?mutation ?pad (case : Src.Case.t) =
  match Machine_aarch64_test.A64_harness.selected case with
  | Error e -> "selection: " ^ e
  | Ok (res, sel_obs) -> (
      match realize ?mutation ?pad res with
      | Error e -> "rejected: " ^ e
      | Ok real -> (
          let memory = Mir_memory.create () in
          let binding =
            Result.get_ok
              (Mir_interp.instantiate (regions_program real) memory
                 ~bound:case.Src.Case.bound)
          in
          let r =
            P.run ~models:Machine_interp.Mir_math_model.all ~realized:true
              ~seed:caller_state real memory binding ~args:[]
          in
          let sel = A64_stage.Sel.Verified.selected res.A64_select.selected in
          let record =
            (Option.get
               (Mir_program.find_view sel.A64_stage.Sel.program
                  res.A64_select.record))
              .Mir_view.region
          in
          let obs =
            Src.selected_observation
              ~layout:case.Src.Case.lowered.Machine_lower.Mir_lower.layout
              ~sites:[||] ~record memory binding r.P.outcome r.P.events
          in
          let ds =
            List.filter_map Fun.id
              [
                Src.verdict
                  ~expected:
                    { Src.Route.name = "aarch64"; observation = sel_obs }
                  ~actual:{ Src.Route.name = "realized"; observation = obs };
                Src.verdict ~expected:case.Src.Case.oracle
                  ~actual:{ Src.Route.name = "realized"; observation = obs };
              ]
          in
          Src.status_name obs
          ^ match ds with [] -> "" | d -> " DISAGREE " ^ String.concat "; " d))
