(* The selected route of a lowered case, and its verdicts against the generic
   route and the SSA oracle. *)
open Machine_ir
open Machine_interp
open Machine_target_aarch64
module Src = Machine_source_test.Mir_source

let selected ?mutation ?(sites = [||]) (case : Src.Case.t) =
  match
    Err.payload
      (A64_select.program ?mutation ~sites
         case.Src.Case.lowered.Machine_lower.Mir_lower.program)
  with
  | Error r -> Error (Fmt.str "%a" A64_select.Refusal.pp r)
  | Ok res -> (
      let sel = A64_stage.Sel.Verified.selected res.A64_select.selected in
      let program = sel.A64_stage.Sel.program in
      let memory = Mir_memory.create () in
      match
        Mir_interp.instantiate program memory ~bound:case.Src.Case.bound
      with
      | Error e -> Error e
      | Ok binding ->
          let r =
            A64_stage.Interp.run res.A64_select.selected memory binding ~args:[]
          in
          let record =
            (Option.get (Mir_program.find_view program res.A64_select.record))
              .Mir_view.region
          in
          Ok
            ( res,
              Src.selected_observation
                ~layout:case.Src.Case.lowered.Machine_lower.Mir_lower.layout
                ~sites ~record memory binding r.A64_stage.Interp.outcome
                r.A64_stage.Interp.events ))

let route observation = { Src.Route.name = "aarch64"; observation }

let report ?mutation ?sites case =
  match selected ?mutation ?sites case with
  | Error e -> "refused: " ^ e
  | Ok (_, obs) -> (
      let sel = route obs in
      let ds =
        List.filter_map Fun.id
          [
            Src.verdict ~expected:case.Src.Case.generic ~actual:sel;
            Src.verdict ~expected:case.Src.Case.oracle ~actual:sel;
          ]
      in
      Src.status_name obs
      ^ match ds with [] -> "" | d -> " DISAGREE " ^ String.concat "; " d)

let plan ?mutation ?sites kernel ~bind =
  match Src.case_of_plan (Fusion_plan.default kernel) ~bind with
  | Error e -> e
  | Ok case -> report ?mutation ?sites case

let program ?mutation ?sites ?fma p ~inputs =
  match Src.case_of_program p ~inputs ?fma () with
  | Error e -> e
  | Ok case -> report ?mutation ?sites case

(* A hand-built generic program with explicit arguments: its generic run
   against its selected run, comparing results (the status removed) or the
   stored failure record. *)
let generic_program ?mutation (p : Mir_program.generic) ~args =
  let g =
    match Err.payload (Mir_verify.generic p) with
    | Ok g -> g
    | Error d -> Fmt.failwith "%a" Mir_diagnostic.pp d
  in
  let memory = Mir_memory.create () in
  let binding =
    Result.get_ok (Mir_interp.instantiate p memory ~bound:(fun _ -> None))
  in
  let generic = Mir_interp.run g memory binding ~args in
  match Err.payload (A64_select.program ?mutation g) with
  | Error r -> Fmt.str "refused: %a" A64_select.Refusal.pp r
  | Ok res ->
      let sel = A64_stage.Sel.Verified.selected res.A64_select.selected in
      let memory = Mir_memory.create () in
      let binding =
        Result.get_ok
          (Mir_interp.instantiate sel.A64_stage.Sel.program memory
             ~bound:(fun _ -> None))
      in
      let r =
        A64_stage.Interp.run res.A64_select.selected memory binding ~args
      in
      let strip = function
        | Mir_interp.Outcome.Success vs -> (
            match List.rev vs with
            | Mir_datum.Bits 0L :: rest -> `Values (List.rev rest)
            | _ -> `Failed)
        | o -> `Other (Fmt.str "%a" Mir_interp.Outcome.pp o)
      in
      let g =
        match generic.Mir_interp.outcome with
        | Mir_interp.Outcome.Success vs -> `Values vs
        | Mir_interp.Outcome.Failure _ -> `Failed
        | o -> `Other (Fmt.str "%a" Mir_interp.Outcome.pp o)
      in
      let s = strip r.A64_stage.Interp.outcome in
      let text = function
        | `Values vs -> Fmt.str "[%a]" Fmt.(list ~sep:(any " ") Mir_datum.pp) vs
        | `Failed -> "failed"
        | `Other o -> o
      in
      if text g = text s then text s
      else Fmt.str "DISAGREE generic %s vs aarch64 %s" (text g) (text s)
