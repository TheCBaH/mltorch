(* A paired experiment on the calibrated VME core, using the generic
   transformer fixtures. Every prediction uses the compiled image's actual
   placement. All predictions are written before any RTL execution. Output
   files are exclusive: rerunning cannot overwrite the frozen evidence.

   CORALNPU_DATA=... dune exec test/coral_estimate_validation/probe.exe --
     --output NEW_DIRECTORY [--run]

   Without --run this only builds images and freezes predictions. With it,
   the already-prepared models run on RTL, checked against the binary32
   oracle. No rates are fitted and no measurements join the training set. *)

module CE = Coral_estimate.Coral_census
module CM = Coral_estimate.Coral_model
module C = Machine_rivet_riscv32_coral.Rv32_mir_coral
module M = Machine_model.Mir_model
module T = Machine_model_test.Model_test

module Variant = struct
  type t = { fixture : string; label : string; unroll : int }
end

module Prepared = struct
  type t = { label : string; predicted : float; run : unit -> int64 }
end

let rates = Coral_estimate.Coral_rates.vme
let core = CM.Core.vme ~packed:(fun _ -> false)

let exclusive path body =
  Out_channel.with_open_gen [ Open_binary; Open_creat; Open_excl; Open_wronly ]
    0o644 path (fun oc -> output_string oc body)

let md5 s = Digest.to_hex (Digest.string s)

let feature_text fs =
  String.concat "\n" (List.map (fun (k, v) -> Fmt.str "%s\t%.9f" k v) fs)

let prepare ~out (config : C.Config.t) (variant : Variant.t) =
  let label = variant.label in
  let file suffix = Filename.concat out (label ^ suffix) in
  let graph =
    (List.assoc variant.fixture
       (T.transformer_fixtures @ T.transformer_bf16_fixtures
      @ Machine_model_test.Transformer_fixtures.bf16_out))
      ()
  in
  let bundle = T.bundle graph in
  let target =
    Ssa_ir.Ssa_target.with_unroll variant.unroll Ssa_ir.Ssa_target.zve32f128
  in
  let pipeline =
    Ssa_backends.Pipeline.Planned
      { numerics = Ssa_ir.Ssa_numerics.Fp32_ordered; target }
  in
  let image = ref [] and censuses = ref [] and runs = ref [] in
  let on_image placed = image := placed in
  let on_program _ program regions =
    let region buffer =
      let machine_region = List.assoc buffer regions in
      match List.assoc machine_region !image with
      | Coral_profile.Role.Dtcm -> CE.Region.Dtcm
      | Coral_profile.Role.Ddr | Coral_profile.Role.Ram -> CE.Region.Ram
      | _ -> Fmt.failwith "%s: unmodeled buffer placement" label
    in
    censuses := CE.program ~region program :: !censuses
  in
  let on_run (r : C.Run.t) = runs := r.mcycle :: !runs in
  let on_elf bytes =
    exclusive (file (Fmt.str ".%d.elf" (List.length !runs))) bytes
  in
  let model =
    match
      M.prepare
        ~route:(C.route ~on_image ~on_run ~on_elf config)
        ~pipeline ~on_program bundle
    with
    | Ok m -> m
    | Error rs -> Fmt.failwith "%s: %a" label Fmt.(list M.Refusal.pp) rs
  in
  let cs = List.rev !censuses in
  if List.length cs <> M.invocations model then
    Fmt.failwith "%s: missing invocation census" label;
  let features = List.map (CM.features ~rates core) cs in
  let predicted =
    List.fold_left ( +. ) 0. (List.map (CM.Rates.predict rates) features)
  in
  if not (Float.is_finite predicted && predicted > 0.) then
    Fmt.failwith "%s: invalid prediction" label;
  let constants =
    List.map
      (fun id ->
        let sg = T.sig_of graph id in
        (id, T.in_format sg (T.positive sg)))
      bundle.Loop_ir.Loop_bundle.constants
  in
  let inputs =
    List.mapi
      (fun i id ->
        let sg = T.sig_of graph id in
        (id, T.in_format sg (T.values ~salt:i sg)))
      bundle.Loop_ir.Loop_bundle.inputs
  in
  let input_bits =
    String.concat "\n"
      (List.mapi
         (fun i (_, t) ->
           Fmt.str "tensor%d\t%s" i
             (String.concat "," (List.map Int64.to_string (T.bits t))))
         (constants @ inputs))
  in
  exclusive (file ".inputs") input_bits;
  exclusive (file ".features")
    (String.concat "\n"
       (List.mapi
          (fun i fs ->
            Fmt.str "invocation %d\n%s\ncensus\n%a\n" i (feature_text fs) CE.pp
              (List.nth cs i))
          features));
  exclusive (file ".prediction")
    (Fmt.str
       "fixture\t%s\n\
        configuration\t%s\n\
        registers\tm1\n\
        numerics\tFp32_ordered\n\
        unroll\t%d\n\
        invocations\t%d\n\
        inputs_md5\t%s\n\
        predicted_cycles\t%.9f\n\
        unknown\t%d\n"
       variant.fixture config.configuration variant.unroll (M.invocations model)
       (md5 input_bits) predicted
       (List.fold_left (fun n (c : CE.t) -> n + c.unknown) 0 cs));
  let run () =
    let expected = T.expected T.Fp32 graph bundle ~constants ~inputs in
    let cx =
      match
        M.Context.create model ~constants:(fun id ->
            List.assoc_opt id constants)
      with
      | Ok cx -> cx
      | Error e -> Fmt.failwith "%s: %a" label M.Stop.pp e
    in
    let outputs =
      match M.Context.run cx ~inputs:(fun id -> List.assoc_opt id inputs) with
      | Ok outputs -> outputs
      | Error e -> Fmt.failwith "%s: %a" label M.Stop.pp e
    in
    let oracle =
      List.map
        (fun id -> Graph_ir.Tensor_id.Map.find id expected)
        graph.Graph_ir.Graph.outputs
    in
    if List.map T.bits outputs <> List.map T.bits oracle then
      Fmt.failwith "%s: differs from binary32 oracle" label;
    if List.length !runs <> List.length cs then
      Fmt.failwith "%s: incomplete measurements" label;
    let measured = List.fold_left Int64.add 0L !runs in
    exclusive (file ".measurement")
      (Fmt.str "cycles\t%Ld\nper_invocation\t%a\nbitwise\tyes\n" measured
         Fmt.(list ~sep:(any ",") int64)
         (List.rev !runs));
    measured
  in
  { Prepared.label; predicted; run }

(* A group of two is a pair (its predicted and measured speedup are
   compared); every variant's own error is reported. *)
let gemma =
  List.concat_map
    (fun shape ->
      let v op suffix unroll =
        {
          Variant.fixture = op ^ " " ^ shape;
          label =
            String.map (function ' ' -> '_' | c -> c) op
            ^ "_" ^ shape ^ suffix;
          unroll;
        }
      in
      [
        [ v "residual add" ".unroll1" 1; v "residual add" ".unroll2" 2 ];
        [ v "tanh gelu mul" "" 1 ];
      ])
    [ "2x160"; "1x67" ]

(* The bfloat16 storage pairs: bf16 inputs with binary32 results, then the
   same with bfloat16 results (one more invocation, the narrowing store). *)
let bf16 =
  List.concat_map
    (fun shape ->
      let v op suffix =
        {
          Variant.fixture = op ^ suffix ^ " " ^ shape;
          label =
            String.map
              (function ' ' -> '_' | '>' -> '_' | c -> c)
              (op ^ suffix ^ "_" ^ shape);
          unroll = 1;
        }
      in
      List.map
        (fun op -> [ v op " bf16"; v op " bf16->bf16" ])
        [ "residual add"; "rms norm" ])
    [ "2x160"; "1x67" ]

let () =
  let out = ref "" and run = ref false and revision = ref "unrecorded" in
  let experiment = ref "gemma" in
  Arg.parse
    [
      ("--output", Arg.Set_string out, "DIR new evidence directory");
      ("--run", Arg.Set run, "run RTL after freezing every prediction");
      ( "--experiment",
        Arg.Symbol ([ "gemma"; "bf16" ], fun e -> experiment := e),
        " the pairs: unroll and math sequence (gemma), or bfloat16 storage" );
      ( "--revision",
        Arg.Set_string revision,
        "LABEL source revision or experiment label" );
    ]
    (fun _ -> raise (Arg.Bad "unexpected positional argument"))
    "probe.exe --output NEW_DIRECTORY [--run]";
  if !out = "" then raise (Arg.Bad "--output is required");
  let dir = Sys.getenv "CORALNPU_DATA" in
  let configuration = "vme-core-mini-axi" in
  let profile =
    Err.or_raise ~pp_error:Coral_profile.pp_error
      (Coral_profile.load ~dir configuration)
  in
  Unix.mkdir !out 0o755;
  exclusive
    (Filename.concat !out "metadata")
    (Fmt.str
       "source_revision\t%s\n\
        configuration_id\t%s\n\
        simulator_sha256\t%s\n\
        probe_md5\t%s\n\
        rates_md5\t%s\n\
        rates\n\
        %s\n"
       !revision profile.configuration_id profile.artifact_sha256
       (Digest.to_hex (Digest.file Sys.executable_name))
       (md5 (Marshal.to_string rates []))
       (feature_text rates));
  let config =
    {
      C.Config.dir;
      configuration;
      deadline = 3600.;
      flags = [ "--cycles=500000000" ];
    }
  in
  let groups = if !experiment = "bf16" then bf16 else gemma in
  let prepared = List.map (List.map (prepare ~out:!out config)) groups in
  let count = List.length (List.concat prepared) in
  exclusive
    (Filename.concat !out "frozen")
    (Fmt.str "predictions\t%d\nfinished_at_epoch\t%.6f\n" count
       (Unix.gettimeofday ()));
  Fmt.pr "frozen %d predictions in %s before any RTL run@." count !out;
  if !run then
    List.iter
      (fun group ->
        let measured = List.map (fun (p : Prepared.t) -> (p, p.run ())) group in
        List.iter
          (fun ((p : Prepared.t), m) ->
            Fmt.pr "%s: predicted %.0f, measured %Ld (%+.1f%%)@." p.label
              p.predicted m
              (100. *. ((p.predicted /. Int64.to_float m) -. 1.)))
          measured;
        match measured with
        | [ (a, ma); (b, mb) ] ->
            let a : Prepared.t = a and b : Prepared.t = b in
            Fmt.pr
              "%s / %s: predicted %.3fx, measured %.3fx (%Ld -> %Ld cycles)@."
              a.label b.label
              (a.predicted /. b.predicted)
              (Int64.to_float ma /. Int64.to_float mb)
              ma mb
        | _ -> ())
      prepared
