(* The LOADER of Loop_c_embed over the embedded CompCert for riscv32. The unit is
   compiled to assembly and lowered once; a call maps the caller's region as the
   shared output file of an RV32 process, so the model reads its weights and
   inputs from the mapping and writes its outputs into it, and the file is read
   back. No libc is in the process: the functions of {!Rv32_libc} are linked,
   and a unit that references another is refused when the image is planned. *)

module R = Compcert_embed_riscv32
module H = Machine_rivet_riscv32.Rv32_harness
module I = Machine_rivet_riscv32.Rv32_image

type loaded = {
  lowered : Riscv32_encode.fixup_kind Asm_core.Lowered_ast.module_ list;
  mutable elf : (int * string * H.prepared) option;  (** by region size *)
}

type error = string

let dialect = Loop_ir.Loop_c_dialect.Compcert_ilp32
let pp_error = Fmt.string
let ( let* ) = Result.bind
let render pp e = Fmt.str "%a" pp e

let lower_typed m =
  match I.P.lower ~state:Riscv32_encode.default_state m with
  | Ok l -> Ok l
  | Error e -> Error (Foundation.Diag.render e)

let load ~host_symbols:_ source =
  let* asm =
    Result.map_error
      (render Compcert_embed.pp_error)
      (R.compile_to_asm ~name:"model.c" source)
  in
  (* a developer's hook: keep the last unit's assembly *)
  Option.iter
    (fun keep ->
      Out_channel.with_open_bin keep (fun oc -> output_string oc asm))
    (Sys.getenv_opt "MLTORCH_RV32_ASM");
  let* units =
    Result.map_error
      (render Compcert_embed.pp_error)
      (R.with_runtime [ ("model", asm) ])
  in
  let* lowered =
    List.fold_left
      (fun acc u ->
        let* acc = acc in
        let* l =
          Result.map_error (render Compcert_embed.pp_error) (R.lower_unit u)
        in
        Ok (l :: acc))
      (Ok []) units
  in
  let* libc =
    Result.bind
      (Result.map_error (Fmt.str "no form for %s")
         (Machine_rivet_riscv32.Rv32_libc.module_ ()))
      lower_typed
  in
  Ok { lowered = libc :: List.rev lowered; elf = None }

(* The harness is generated for the region's size, which the mapping length
   depends on. *)
let prepared l size =
  match l.elf with
  | Some (n, _, p) when n = size -> Ok p
  | _ ->
      let slots = [ { H.size; align = 4096; placement = H.Placement.Out } ] in
      let lay = H.layout slots in
      let* harness =
        Result.bind
          (Result.map_error (Fmt.str "no form for %s")
             (H.harness ~abi:H.Abi.Slot0 ~entry:"entry" lay))
          lower_typed
      in
      let render_i e = Fmt.str "%a" I.Error.pp e in
      let* laid =
        Result.map_error render_i
          (Err.payload (I.plan_lowered ~entry:"_start" (harness :: l.lowered)))
      in
      let* image =
        Result.map_error render_i (Err.payload (I.bind ~base:H.base laid))
      in
      let* elf = Machine_rivet_riscv32.Rv32_elf.write image in
      let p = { H.elf; lay } in
      l.elf <- Some (size, "", p);
      Ok p

let call l ~io =
  let size = Bigarray.Array1.dim io in
  let* p = prepared l size in
  let region = String.init size (fun i -> Bigarray.Array1.get io i) in
  let* { H.status; outputs } =
    H.run p ~bound:(fun k -> if k = 0 then Some region else None)
  in
  match List.assoc_opt 0 outputs with
  | None -> Error "the process returned no region"
  | Some bytes ->
      String.iteri (fun i c -> Bigarray.Array1.set io i c) bytes;
      Ok (Int64.of_int32 status)

let close (_ : loaded) = ()
