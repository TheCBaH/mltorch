(* [soc_peak.exe DIR]: peak kernels of {!Peak_kernels} and {!Peak_zvt} on the
   subsystem's normal-TCM cores, everything in DTCM as on the standalone
   cores, each output checked by its hash. Prints the cycles in the kernel
   beside the standalone core's (from the peak suite). *)

module K = Peak_kernels

let dir = Sys.argv.(1)

let or_exit pp = function
  | Ok v -> v
  | Error e ->
      Fmt.epr "%a@." pp e;
      exit 1

let profile name =
  or_exit Coral_profile.pp_error (Err.payload (Coral_profile.load ~dir name))

let run configuration name (k : K.t) ~standalone =
  let p = profile configuration in
  let m = or_exit Fmt.string (K.build k K.rv32_profile) in
  let image =
    or_exit Coral_image.pp_error
      (Err.payload
         (Coral_image.build ~rv32_profile:K.rv32_profile ~matrix:k.K.matrix p
            ~entry:K.entry ~kernel:[ m ] k.K.slots))
  in
  let inputs = Array.of_list k.K.inputs in
  match
    Err.payload
      (Coral_soc_run.run ~dir image ~bound:(fun i ->
           if i < Array.length inputs then Some inputs.(i) else None))
  with
  | Error e ->
      Fmt.pr "%s %s: %a@." configuration name Coral_soc_run.Run_error.pp e
  | Ok o ->
      let out = List.length k.K.slots - 1 in
      Fmt.pr "%s %s: %s, %Ld cycles (standalone %d)@." configuration name
        (if o.status <> 0l then Printf.sprintf "returned %ld" o.status
         else if Coral_soc_run.matches o out k.K.expected then "as expected"
         else "WRONG")
        o.mcycle standalone

let () =
  List.iter
    (fun configuration ->
      List.iter
        (fun (name, input, standalone) ->
          run configuration name (K.gemm input) ~standalone)
        K.Gemm.Input.
          [
            ("gemm_bf16", Bf16, 5953);
            ("gemm_f32", F32, 3857);
            ("gemm_i8", I8, 4283);
          ])
    [ "subsystem-vector-normal"; "subsystem-matrix-normal" ];
  List.iter
    (fun (name, op, standalone) ->
      run "subsystem-matrix-normal" name (Peak_zvt.tile op) ~standalone)
    K.Insn.Tile_op.
      [
        ("tile_bf16", Vtfmm_alt, 1470);
        ("tile_f32", Vtfmm, 1469);
        ("tile_i8s", Vtmms, 1473);
      ];
  List.iter
    (fun (name, input, standalone) ->
      run "subsystem-matrix-normal" name (Peak_zvt.tile_gemm input) ~standalone)
    Peak_zvt.Tile_gemm.Input.
      [
        ("tgemm_bf16", Bf16, 2440);
        ("tgemm_f32", F32, 2440);
        ("tgemm_i8", I8, 2440);
      ]
