(* [coral_peak.exe NAME [VTYPE INSTRUCTION]]: builds a peak-throughput kernel
   ({!Peak_kernels}: [gemm_<type>] or [gemm_<type>_w<N>], or INSTRUCTION repeated at VTYPE such as
   [e8,m4]) into a Coral guest image, prints its timed loop as coralview decodes
   the image's bytes, writes NAME.elf, runs it on the RTL vector core with its
   waveform traced to NAME.fst, and checks the kernel's output slot. Exits 1 on
   a wrong output or a failed run. CORALNPU_DATA names the emulators archive. *)

module K = Peak_kernels

let configuration = "rvv-core-mini-axi"

let or_exit pp = function
  | Ok v -> v
  | Error e ->
      Fmt.epr "%a@." pp e;
      exit 1

let kernel name args =
  if String.length name > 5 && String.sub name 0 5 = "gemm_" then
    K.gemm (String.sub name 5 (String.length name - 5))
  else
    match args with
    | [ vtype; instr ] -> (
        match String.split_on_char ',' vtype with
        | [ e; m ] when e.[0] = 'e' && m.[0] = 'm' ->
            let op =
              match K.Vv_op.of_mnemonic instr with
              | Some op -> op
              | None ->
                  invalid_arg ("coral_peak: no vector-vector form " ^ instr)
            in
            or_exit Fmt.string
              (K.elementwise
                 ~sew:(int_of_string (String.sub e 1 (String.length e - 1)))
                 ~lmul:(int_of_string (String.sub m 1 (String.length m - 1)))
                 op)
        | _ -> invalid_arg ("coral_peak: VTYPE is eN,mN, not " ^ vtype))
    | _ ->
        invalid_arg "coral_peak: NAME VTYPE INSTRUCTION, or gemm_<type>[_w<N>]"

(* The words from [peak_body] to [peak_body_end], as coralview disassembles
   them, at their offsets from [peak_body]. *)
let print_body (image : Image.t) =
  let address name =
    match List.assoc_opt name image.Image.exports with
    | Some a -> a
    | None -> invalid_arg ("coral_peak: the image exports no " ^ name)
  in
  let first = address "peak_body" and last = address "peak_body_end" in
  let s =
    List.find
      (fun (s : Image.segment) ->
        first >= s.Image.address
        && last
           <= Int64.add s.Image.address
                (Int64.of_int (String.length s.Image.bytes)))
      image.Image.segments
  in
  let d = or_exit Fmt.string (Coralview.Rivet_decode.create ()) in
  let base = Int64.to_int (Int64.sub first s.Image.address) in
  for k = 0 to (Int64.to_int (Int64.sub last first) / 4) - 1 do
    let word =
      Int32.to_int (String.get_int32_le s.Image.bytes (base + (4 * k)))
      land 0xFFFF_FFFF
    in
    Fmt.pr "  +%02x  %s@." (4 * k) (Coralview.Rivet_decode.disasm d word)
  done

let () =
  let dir = Sys.getenv "CORALNPU_DATA" in
  let name, args =
    match Array.to_list Sys.argv with
    | _ :: name :: args -> (name, args)
    | _ -> invalid_arg "coral_peak: no kernel"
  in
  let k = kernel name args in
  let profile =
    or_exit Coral_profile.pp_error
      (Err.payload (Coral_profile.load ~dir configuration))
  in
  let m = or_exit Fmt.string (K.build k K.rv32_profile) in
  let image =
    or_exit Coral_image.pp_error
      (Err.payload
         (Coral_image.build ~rv32_profile:K.rv32_profile profile ~entry:K.entry
            ~kernel:[ m ] k.K.slots))
  in
  Fmt.pr "%s@." k.K.shape;
  print_body image.Coral_image.image;
  Out_channel.with_open_bin (name ^ ".elf") (fun oc ->
      output_string oc image.Coral_image.elf);
  let trace = Filename.concat (Sys.getcwd ()) (name ^ ".fst") in
  let inputs = Array.of_list k.K.inputs in
  let outcome =
    or_exit Coral_run.Run_error.pp
      (Err.payload
         (Coral_run.run
            ~flags:[ "--cycles=2000000"; "--trace_file=" ^ trace ]
            ~dir image
            ~bound:(fun i ->
              if i < Array.length inputs then Some inputs.(i) else None)))
  in
  let out = List.assoc (List.length k.K.slots - 1) outcome.Coral_run.outputs in
  if outcome.Coral_run.status <> 0l then (
    Fmt.pr "%s: the kernel returned %ld@." name outcome.Coral_run.status;
    exit 1);
  if not (String.equal out k.K.expected) then (
    let rec miss i = if out.[i] <> k.K.expected.[i] then i else miss (i + 1) in
    Fmt.pr "%s: the output differs from the expected one at byte %d@." name
      (miss 0);
    exit 1);
  Fmt.pr "%s: output as expected, %Ld cycles in the kernel@." configuration
    outcome.Coral_run.mcycle
