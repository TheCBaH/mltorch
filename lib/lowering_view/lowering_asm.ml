(* The Asm family: one invocation lowered to Machine IR, its published artifact
   (realized: reference allocation and frames; register groups, which only the
   scanned allocator places, take that stage) as the typed Rivet module of the
   target, printed as GNU assembly. The text is the one the manifest's
   [code_digest] is over, and the one Rivet encodes. *)

module Art = Machine_model.Mir_artifact
module Refusal = Lowering_artifact.Refusal
module Note = Lowering_artifact.Note
module M = Machine_model.Mir_model
module K = Lowering_key

let ( let* ) = Result.bind

(* A mir invocation is its position in the schedule, which a per-node view does
   not know: a refusal reports 0, as the Machine IR family does. *)
let pipeline =
  Ssa_backends.Pipeline.Planned
    {
      numerics = Ssa_ir.Ssa_numerics.Fp32_ordered;
      target = Ssa_ir.Ssa_target.scalar;
    }

let print ~type_char ~instruction m =
  Fmt.str "%a"
    (Asm_core.Gnu_module.pp { Asm_core.Gnu_module.type_char } ~instruction)
    m

module X64_layout = Driver_direct.Pipeline_direct.Make (X86_64_encode)
module A64_layout = Driver_direct.Pipeline_direct.Make (Aarch64_encode)

(* The bytes the module's text lays out as, by Rivet's own lowering and layout:
   what the image would hold. A module Rivet will not lay out has no size to
   note. *)
let code_bytes = function
  | Error _ -> []
  | Ok laid ->
      List.filter_map
        (fun (s : Image.segment_plan) ->
          if s.Image.seg_name = ".text" then
            Some (Note.Code_bytes (Int64.of_int s.Image.init_size))
          else None)
        (Image.plan_of laid).Image.segments

let notes artifact =
  let symbols = Art.symbols artifact in
  let bindings =
    List.filter_map
      (fun (s : Art.Symbol.t) ->
        match s.Art.Symbol.kind with
        | Art.Symbol.External_function native -> Some (s.Art.Symbol.name, native)
        | Art.Symbol.Data _ | Art.Symbol.Function _
        | Art.Symbol.Generated_attention _ | Art.Symbol.Generated_function _ ->
            None)
      symbols
  in
  [ Note.Symbols symbols; Note.Relocations (Art.relocations artifact) ]
  @ if bindings = [] then [] else [ Note.Helper_bindings bindings ]

let x86_64 l =
  let* a =
    Result.map_error (fun r -> Refusal.Mir_stage r) (M.Artifact.x86_64 l)
  in
  match
    Err.payload (Machine_rivet_x86_64_module.Rivet_x64_module.of_artifact a)
  with
  | Error r -> Error (Refusal.Asm_x86_64 r)
  | Ok m ->
      Ok
        ( print ~type_char:'@' ~instruction:X86_64_encode.Instruction.pp_gnu m,
          notes a
          @ code_bytes
              (Result.bind
                 (X64_layout.lower ~state:X86_64_encode.default_state m)
                 (X64_layout.plan ~entry:"kernel")) )

let aarch64 l =
  let* a =
    Result.map_error (fun r -> Refusal.Mir_stage r) (M.Artifact.aarch64 l)
  in
  match
    Err.payload (Machine_rivet_aarch64_module.Rivet_a64_module.of_artifact a)
  with
  | Error r -> Error (Refusal.Asm_aarch64 r)
  | Ok m ->
      Ok
        ( print ~type_char:'%' ~instruction:Aarch64_encode.Instruction.pp_gnu m,
          notes a
          @ code_bytes
              (Result.bind
                 (A64_layout.lower ~state:Aarch64_encode.default_state m)
                 (A64_layout.plan ~entry:"kernel")) )

let riscv32 artifact l =
  let* a = Result.map_error (fun r -> Refusal.Mir_stage r) (artifact l) in
  match
    Err.payload (Machine_rivet_riscv32_mir_module.Rv32_mir_module.of_artifact a)
  with
  | Error r -> Error (Refusal.Asm_riscv32 r)
  | Ok m ->
      Ok
        ( print ~type_char:'@' ~instruction:Riscv32_encode.Instruction.pp m,
          notes a
          @ code_bytes
              (Result.map_error ignore
                 (Err.payload
                    (Machine_rivet_riscv32_module.Rv32_image.plan
                       ~entry:"kernel" [ m ]))) )

let of_invocation (target : K.Target.t) inv =
  match M.lower_one ~pipeline ~invocation:0l inv with
  | Error r -> Error (Refusal.Mir_lowering r)
  | Ok l -> (
      match target with
      | K.Target.Aarch64 -> aarch64 l
      | K.Target.Riscv32 -> riscv32 M.Artifact.riscv32 l
      | K.Target.Riscv32_grouped -> riscv32 M.Artifact.riscv32_grouped l
      | K.Target.X86_64 -> x86_64 l)
