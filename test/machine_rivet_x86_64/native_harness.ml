(* A lowered source case through reference allocation, frames and publication,
   then on the physical interpreter and as an x86-64 process over the
   same bound bytes. The two are compared as observations: the status, the
   failure record it decodes to, and the output tensors. *)

open Machine_ir
open Machine_interp
open Machine_target_x86_64
module Src = Machine_source_test.Mir_source
module A = Machine_alloc.Mir_ref_alloc.Make (X64) (X64_regs)
module Fr = Machine_alloc.Mir_frame.Make (X64) (X64_frame)
module Pub = Machine_model.Mir_artifact.Make (X64)
module P = Mir_phys_interp.Make (X64)
module Art = Machine_model.Mir_artifact
module Qemu = Machine_rivet_x86_64.Rivet_x64_process

let ( let* ) = Result.bind

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

type built = {
  artifact : (X64_op.t, X64_op.test) Art.t;
  record : Mir_id.Region.t;
}

let build ?features ?pad (case : Src.Case.t) ~sites =
  let g = case.Src.Case.lowered.Machine_lower.Mir_lower.program in
  let* planning =
    Option.to_result ~none:"no planning summary"
      (Mir_verify.Generic.program g).Mir_program.planning
  in
  let* res =
    Result.map_error
      (Fmt.str "selection: %a" X64_select.Refusal.pp)
      (Err.payload (X64_select.program ?features ~sites g))
  in
  let v = res.X64_select.selected in
  let* real =
    Result.map_error
      (Fmt.str "frame: %a" Machine_alloc.Mir_frame.Refusal.pp)
      (Fr.realize ?pad (A.allocate v))
  in
  let* artifact = Pub.publish ~planning v real in
  let sel = X64_stage.Sel.Verified.selected v in
  let record =
    (Option.get
       (Mir_program.find_view sel.X64_stage.Sel.program res.X64_select.record))
      .Mir_view.region
  in
  Ok { artifact; record }

let observe (case : Src.Case.t) ~sites { record; _ } memory binding outcome =
  Src.selected_observation
    ~layout:case.Src.Case.lowered.Machine_lower.Mir_lower.layout ~sites ~record
    memory binding outcome []

let interpreted case ~sites ({ artifact; _ } as b) =
  let phys = Art.program artifact in
  let memory = Mir_memory.create () in
  match
    Mir_interp.instantiate (regions_program phys) memory
      ~bound:case.Src.Case.bound
  with
  | Error e -> Error e
  | Ok binding ->
      let r =
        P.run ~models:Mir_math_model.all ~realized:true phys memory binding
          ~args:[]
      in
      Ok (observe case ~sites b memory binding r.P.outcome)

(* The artifact as a process: the mutable regions start at the case's
   bound bytes and come back as the process left them. *)
let process_full ?runtime ?timeout ?probe ?entry_mutation case ~sites
    ({ artifact; _ } as b) =
  let phys = Art.program artifact in
  let bound r =
    match
      List.find_opt
        (fun (x : Mir_region.t) -> Mir_id.Region.equal x.Mir_region.id r)
        phys.Mir_phys.Program.regions
    with
    | Some { Mir_region.init = Mir_region.Bound; size; _ } ->
        Some
          (match case.Src.Case.bound r with
          | Some s -> s
          | None -> String.make (Int64.to_int size) '\000')
    | _ -> None
  in
  let* { Qemu.status; regions; abi } =
    Qemu.run ?runtime ?timeout ?probe ?entry_mutation ~bound artifact
  in
  let after =
    List.map (fun (r, bytes) -> (Mir_id.Region.to_int r, bytes)) regions
  in
  let memory = Mir_memory.create () in
  let bytes_of id = List.assoc_opt (Mir_id.Region.to_int id) after in
  let* binding =
    Mir_interp.instantiate (regions_program phys) memory ~bound:bytes_of
  in
  List.iter
    (fun (r : Mir_region.t) ->
      match
        ( r.Mir_region.init,
          Mir_interp.Binding.instance binding r.Mir_region.id,
          bytes_of r.Mir_region.id )
      with
      | Mir_region.Uninitialized, Some key, Some bytes ->
          Mir_memory.write_string memory key ~offset:0L bytes
      | _ -> ())
    phys.Mir_phys.Program.regions;
  Ok
    ( observe case ~sites b memory binding
        (Mir_interp.Outcome.Success
           [ Mir_datum.Bits (Int64.logand status 0xFFFF_FFFFL) ]),
      abi )

let process ?runtime ?timeout case ~sites b =
  Result.map fst (process_full ?runtime ?timeout case ~sites b)

(* The process observation against the interpreter's: a one-line verdict. *)
let compare ?runtime ?features ?pad ?probe ?entry_mutation ?(sites = [||])
    (case : Src.Case.t) =
  match build ?features ?pad case ~sites with
  | Error e -> "not built: " ^ e
  | Ok b -> (
      match
        ( interpreted case ~sites b,
          process_full ?runtime ?probe ?entry_mutation case ~sites b )
      with
      | Error e, _ -> "interpreter: " ^ e
      | _, Error e -> "process: " ^ e
      | Ok i, Ok (n, abi) -> (
          let abi =
            match Option.map Qemu.Abi.violations abi with
            | None | Some [] -> ""
            | Some v -> " ABI broken: " ^ String.concat "; " v
          in
          match Mir_compare.observations ~expected:i ~actual:n () with
          | Ok () -> Src.status_name n ^ " [process: agree]" ^ abi
          | Error d ->
              Fmt.str "%s [process: DISAGREE %a]%s" (Src.status_name n)
                Mir_compare.Difference.pp d abi))

(* The System V ABI around a kernel and its helpers, with stub helpers that
   clobber what a callee may: only the ABI record is meaningful. *)
let abi ?pad ?entry_mutation ?(sites = [||]) (case : Src.Case.t) =
  match build ?pad case ~sites with
  | Error e -> "not built: " ^ e
  | Ok b -> (
      match
        process_full ~runtime:Machine_rivet_common.Rivet_runtime.System_libm
          ~probe:true ?entry_mutation case ~sites b
      with
      | Error e -> "process: " ^ e
      | Ok (_, None) -> "no probe record"
      | Ok (_, Some a) -> (
          match Qemu.Abi.violations a with
          | [] -> "ABI kept"
          | v -> "ABI broken: " ^ String.concat "; " v))

(* The artifact's typed module, assembled by GNU as and linked by GNU ld at
   Rivet's addresses, against Rivet's own image. *)
let gnu ?pad ?tamper_gnu ?tamper_text ?(sites = [||]) (case : Src.Case.t) =
  match build ?pad case ~sites with
  | Error e -> "not built: " ^ e
  | Ok { artifact; _ } -> (
      let phys = Art.program artifact in
      let entry =
        (List.find
           (fun (f : (_, _) Mir_phys.Func.t) ->
             Mir_id.Func.equal f.Mir_phys.Func.id phys.Mir_phys.Program.main)
           phys.Mir_phys.Program.funcs)
          .Mir_phys.Func.name
      in
      match
        Err.payload (Machine_rivet_x86_64.Rivet_x64_module.of_artifact artifact)
      with
      | Error r ->
          Fmt.str "module: %a" Machine_rivet_x86_64.Rivet_x64_refusal.pp r
      | Ok m ->
          Fmt.str "%a" Machine_rivet_x86_64_gnu.Gnu_coherence.Verdict.pp
            (Machine_rivet_x86_64_gnu.Gnu_coherence.check ?tamper_gnu
               ?tamper_text ~entry [ m ]))

(* A planned binary32 vector case: the interpreter and the process run
   the same selected program. *)
let planned ?runtime ?features ?pad ~target ~numerics kernel ~bind =
  match
    Src.case_of_planned ~target ~numerics (Fusion_plan.default kernel) ~bind
  with
  | Error e -> e
  | Ok case -> compare ?runtime ?features ?pad case

(* The call-frame information of the artifact's modules, resident and
   table-bound (with the entry wrapper), against the instructions. *)
let cfi ?pad ?(tamper = Fun.id) ?(sites = [||]) (case : Src.Case.t) =
  match build ?pad case ~sites with
  | Error e -> "not built: " ^ e
  | Ok { artifact; _ } ->
      let number s =
        let s = String.trim s in
        let s = if String.length s > 0 && s.[0] = '$' then String.sub s 1 (String.length s - 1) else s in
        int_of_string_opt (List.hd (String.split_on_char ',' s))
      in
      let arch =
        {
          Native_cfi.prefix = "x86_64-linux-gnu-";
          entry_offset = 8;
          effect =
            (fun ~mnemonic ~operands ->
              let on_rsp = String.length operands > 5 && String.sub operands (String.length operands - 5) 5 = ",%rsp" in
              match mnemonic with
              | "sub" when on_rsp -> Option.map Fun.id (number operands)
              | "add" when on_rsp -> Option.map (fun n -> -n) (number operands)
              | "push" -> Some 8
              | "pop" -> Some (-8)
              | _ -> None);
          barrier =
            (fun m ->
              m = "call" || m = "ret"
              || (String.length m > 0 && m.[0] = 'j'));
          is_return = (fun m -> m = "ret");
        }
      in
      let module M = Machine_rivet_x86_64.Rivet_x64_module in
      let one binding =
        match Err.payload (M.of_artifact ~binding artifact) with
        | Error r -> Fmt.str "module: %a" Machine_rivet_x86_64.Rivet_x64_refusal.pp r
        | Ok m -> (
            match
              Native_cfi.check arch
                ~assembly:
                  (tamper (Machine_rivet_x86_64_gnu.Gnu_coherence.assembly m))
            with
            | Ok s -> s
            | Error e -> e)
      in
      String.concat "; " [ one M.Image_resident; one M.Table ]
