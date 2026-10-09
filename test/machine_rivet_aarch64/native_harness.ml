(* A lowered source case through reference allocation, frames and publication,
   then on the physical interpreter and natively through Rivet over the same
   bound bytes. The two are compared as observations: the status, the failure
   record it decodes to, and the output tensors. *)

open Machine_ir
open Machine_interp
open Machine_target_aarch64
module Src = Machine_source_test.Mir_source
module A = Machine_alloc.Mir_ref_alloc.Make (A64) (A64_regs)
module Fr = Machine_alloc.Mir_frame.Make (A64) (A64_frame)
module Pub = Machine_model.Mir_artifact.Make (A64)
module P = Mir_phys_interp.Make (A64)
module Art = Machine_model.Mir_artifact
module Image = Machine_rivet_aarch64.Rivet_a64_image
module Module = Machine_rivet_aarch64.Rivet_a64_module
module Refusal = Machine_rivet_aarch64.Rivet_a64_refusal

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
  artifact : (A64_op.t, A64_op.test) Art.t;
  record : Mir_id.Region.t;
}

let build ?select_mutation ?frame_mutation ?pad (case : Src.Case.t) ~sites =
  let g = case.Src.Case.lowered.Machine_lower.Mir_lower.program in
  let* planning =
    Option.to_result ~none:"no planning summary"
      (Mir_verify.Generic.program g).Mir_program.planning
  in
  let* res =
    Result.map_error
      (Fmt.str "selection: %a" A64_select.Refusal.pp)
      (Err.payload (A64_select.program ?mutation:select_mutation ~sites g))
  in
  let v = res.A64_select.selected in
  let* real =
    Result.map_error
      (Fmt.str "frame: %a" Machine_alloc.Mir_frame.Refusal.pp)
      (Fr.realize ?mutation:frame_mutation ?pad (A.allocate v))
  in
  let* artifact = Pub.publish ~planning v real in
  let sel = A64_stage.Sel.Verified.selected v in
  let record =
    (Option.get
       (Mir_program.find_view sel.A64_stage.Sel.program res.A64_select.record))
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

let region_symbol r = Fmt.str "mir_region_%d" (Mir_id.Region.to_int r)

(* Runs [f] in a forked child and returns its marshalled result: generated code
   that faults (a defective mapping, or a mutation made on purpose) ends the
   child, not the test process. *)
let isolated (f : unit -> ('a, string) result) : ('a, string) result =
  flush_all ();
  let r, w = Unix.pipe () in
  match Unix.fork () with
  | 0 ->
      Unix.close r;
      let oc = Unix.out_channel_of_descr w in
      Marshal.to_channel oc (f ()) [];
      close_out oc;
      Unix._exit 0
  | pid -> (
      Unix.close w;
      let ic = Unix.in_channel_of_descr r in
      let result =
        try Some (Marshal.from_channel ic) with End_of_file -> None
      in
      close_in ic;
      match (snd (Unix.waitpid [] pid), result) with
      | Unix.WEXITED 0, Some v -> v
      | Unix.WSIGNALED n, _ ->
          Error
            (Fmt.str "generated code killed by %s"
               (if n = Sys.sigsegv then "SIGSEGV"
                else if n = Sys.sigbus then "SIGBUS"
                else if n = Sys.sigill then "SIGILL"
                else Fmt.str "signal %d" n))
      | _ -> Error "the isolated run reported nothing")

(* The artifact in this process: bound regions written, the kernel called, every
   bound region read back into a fresh memory the observation decodes. *)
let native_full ?mutation
    ?(runtime = Machine_rivet_aarch64.Rivet_a64_runtime.System_libm) ?probe case
    ~sites ({ artifact; _ } as b) =
  let phys = Art.program artifact in
  let module Rt = Machine_rivet_aarch64.Rivet_a64_route in
  let* kernel, modules = Rt.modules ?mutation ~runtime artifact in
  let* loaded =
    match probe with
    | None -> Rt.load ~entry:kernel modules
    | Some _ ->
        Rt.load ~entry:Abi_probe.entry (Abi_probe.module_ ~kernel :: modules)
  in
  let regions = phys.Mir_phys.Program.regions in
  let run () =
    Fun.protect
      ~finally:(fun () -> Image.close loaded)
      (fun () ->
        let io_ r =
          Result.map_error (Fmt.str "%a" Image.Error.pp) (Err.payload r)
        in
        let* () =
          List.fold_left
            (fun acc (r : Mir_region.t) ->
              let* () = acc in
              match r.Mir_region.init with
              | Mir_region.Bound ->
                  let bytes =
                    match case.Src.Case.bound r.Mir_region.id with
                    | Some s -> s
                    | None ->
                        String.make (Int64.to_int r.Mir_region.size) '\000'
                  in
                  io_
                    (Image.write_global loaded
                       (region_symbol r.Mir_region.id)
                       bytes)
              | Mir_region.Constant _ | Mir_region.Uninitialized -> Ok ())
            (Ok ()) regions
        in
        let abi_io = Option.map (fun fpcr -> Abi_probe.input ~fpcr) probe in
        let* status = io_ (Image.call ?io:abi_io loaded) in
        let status =
          match abi_io with Some io -> Abi_probe.word io 39 | None -> status
        in
        let violations =
          match (abi_io, probe) with
          | Some io, Some fpcr -> Abi_probe.violations io ~fpcr
          | _ -> []
        in
        let* after =
          List.fold_left
            (fun acc (r : Mir_region.t) ->
              let* acc = acc in
              match r.Mir_region.init with
              | Mir_region.Constant _ -> Ok acc
              | Mir_region.Bound | Mir_region.Uninitialized ->
                  let* bytes =
                    io_
                      (Image.read_global loaded (region_symbol r.Mir_region.id))
                  in
                  Ok ((Mir_id.Region.to_int r.Mir_region.id, bytes) :: acc))
            (Ok []) regions
        in
        Ok (status, after, violations))
  in
  let* status, after, violations = isolated run in
  let memory = Mir_memory.create () in
  let bytes_of id = List.assoc_opt (Mir_id.Region.to_int id) after in
  let* binding =
    Mir_interp.instantiate (regions_program phys) memory ~bound:bytes_of
  in
  (* a region the program owns, undefined to the interpreter until it writes
     it: the bytes the CPU left, so a failure record decodes *)
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
    regions;
  Ok
    ( observe case ~sites b memory binding
        (Mir_interp.Outcome.Success
           [ Mir_datum.Bits (Int64.logand status 0xFFFF_FFFFL) ]),
      violations )

let native ?mutation ?runtime case ~sites b =
  Result.map fst (native_full ?mutation ?runtime case ~sites b)

(* The native observation against the interpreter's: a one-line verdict. *)
let compare ?mutation ?runtime ?select_mutation ?frame_mutation ?pad ?probe
    ?(sites = [||]) (case : Src.Case.t) =
  match build ?select_mutation ?frame_mutation ?pad case ~sites with
  | Error e -> "not built: " ^ e
  | Ok b -> (
      match
        ( interpreted case ~sites b,
          native_full ?mutation ?runtime ?probe case ~sites b )
      with
      | Error e, _ -> "interpreter: " ^ e
      | _, Error e -> "native: " ^ e
      | Ok i, Ok (n, violations) -> (
          let abi =
            match violations with
            | [] -> ""
            | v -> " ABI broken: " ^ String.concat " " v
          in
          match Mir_compare.observations ~expected:i ~actual:n () with
          | Ok () -> Src.status_name n ^ " [native: agree]" ^ abi
          | Error d ->
              Fmt.str "%s [native: DISAGREE %a]%s" (Src.status_name n)
                Mir_compare.Difference.pp d abi))

(* The artifact's typed modules, assembled by GNU as and linked by GNU ld at
   Rivet's addresses, against Rivet's own image. *)
let gnu ?mutation ?pad ?tamper_gnu ?tamper_text ?(sites = [||])
    (case : Src.Case.t) =
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
      let helpers =
        List.filter_map
          (fun (s : Art.Symbol.t) ->
            match s.Art.Symbol.kind with
            | Art.Symbol.External_function f -> Some f
            | _ -> None)
          (Art.symbols artifact)
      in
      let host =
        if helpers = [] then []
        else
          match
            Err.payload (Module.helpers ~host_symbol:Image.host_symbol helpers)
          with
          | Ok m -> [ m ]
          | Error _ -> []
      in
      match Err.payload (Module.of_artifact ?mutation artifact) with
      | Error r -> Fmt.str "module: %a" Refusal.pp r
      | Ok m ->
          Fmt.str "%a" Machine_rivet_aarch64_gnu.Gnu_coherence.Verdict.pp
            (Machine_rivet_aarch64_gnu.Gnu_coherence.check ?tamper_gnu
               ?tamper_text ~entry (m :: host)))

(* How many instructions of each mnemonic the artifact's module holds: evidence
   that a case exercises the forms it claims to. *)
let mnemonics ?select_mutation ?pad ?(sites = [||]) (case : Src.Case.t) =
  match build ?select_mutation ?pad case ~sites with
  | Error e -> Error e
  | Ok { artifact; _ } -> (
      match Err.payload (Module.of_artifact artifact) with
      | Error r -> Error (Fmt.str "%a" Refusal.pp r)
      | Ok m ->
          let t = Hashtbl.create 16 in
          List.iter
            (function
              | Asm_core.Normalized_ast.Instruction { insn; _ } ->
                  let bump k =
                    Hashtbl.replace t k
                      (1 + Option.value ~default:0 (Hashtbl.find_opt t k))
                  in
                  bump (Aarch64.Opcode.name insn.Aarch64.Instruction.op);
                  if
                    List.exists
                      (function
                        | Aarch64.Operand.Vec _ | Aarch64.Operand.Vlane _
                        | Aarch64.Operand.Vlist _ | Aarch64.Operand.Vlist_lane _
                        | Aarch64.Operand.Qreg _ ->
                            true
                        | _ -> false)
                      insn.Aarch64.Instruction.ops
                  then bump "<vector>"
              | _ -> ())
            m.Asm_core.Normalized_ast.items;
          Ok t)
