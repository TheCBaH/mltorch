(* [fuse_model COHORT CACHE ARTIFACT_ID]: open the released fixture offline
   (or a [.pt2], or a producer model directory with synthetic weights),
   lower the model, fuse its attention layers, and
   evaluate both graphs on the same deterministic inputs; print how many
   layers were fused, how many nodes they replaced, and the largest absolute
   difference of each output. *)

let or_die pp = function
  | Ok v -> v
  | Error e ->
      Fmt.epr "%a@." pp (Err.Error.kind e);
      exit 1

(* Deterministic inputs by the exporter's name: a mask all true, token types
   alternating, other integers ids varying over 4 .. 23 (past esm2's special
   tokens, below every vocabulary here) so the attended rows differ and the
   output depends on the scores. *)
let input ~name (sg : Tensor_sig.t) =
  let shape = sg.Tensor_sig.shape in
  let contains sub =
    let n = String.length sub in
    let rec at i =
      i + n <= String.length name && (String.sub name i n = sub || at (i + 1))
    in
    at 0
  in
  match sg.Tensor_sig.fmt with
  | Payload.Fmt Payload.I64 ->
      Tensor.materialize_i64 shape (fun c ->
          let o = (Vec6.offset shape c :> int) in
          Int64.of_int
            (if contains "mask" then 1
             else if contains "token_type" then o mod 2
             else 4 + (o * 7 mod 20)))
  | Payload.Fmt Payload.Bool -> Tensor.materialize_bool shape (fun _ -> true)
  | _ ->
      Tensor.materialize shape (fun c ->
          let k = (Vec6.offset shape c :> int) * 7919 mod 2003 in
          float_of_int (k - 1001) /. 1001.)

let layers (g : Graph_ir.graph) =
  List.length
    (List.filter
       (fun (n : Graph_ir.node) ->
         match n.Graph_ir.Node.op with
         | Graph_ir.Attention_layer _ -> true
         | _ -> false)
       g.Graph_ir.Graph.nodes)

(* With [FUSE_MODEL_ROUTE] set, the fused graph again, each layer granted at
   binary32 ([F32_ordered]) and run through the generic Machine IR route under
   the planned binary32 pipeline: its outputs against the original graph's
   binary64 reference. *)
(* With [FUSE_MODEL_CORAL] naming a Coral configuration (its data under
   [CORALNPU_SUBSYSTEM_DATA], else [CORALNPU_DATA]), the same granted run on
   that configuration's simulator instead: every tensor homed in RAM, the
   contractions on the matrix engine, each invocation its own image. *)
let coral () =
  match Sys.getenv_opt "FUSE_MODEL_CORAL" with
  | None -> (None, Ssa_ir.Ssa_target.scalar, false)
  | Some configuration ->
      let dir =
        match Sys.getenv_opt "CORALNPU_SUBSYSTEM_DATA" with
        | Some d -> d
        | None -> Sys.getenv "CORALNPU_DATA"
      in
      let module C = Machine_rivet_riscv32_coral.Rv32_mir_coral in
      let cycles = ref 0L in
      at_exit (fun () -> Fmt.pr "  coral: %Ld cycles@." !cycles);
      ( Some
          (C.route ~budget:1_048_576L
             ~home:(fun _ -> Coral_layout.Home.Ram)
             ~on_run:(fun (o : C.Run.t) ->
               cycles := Int64.add !cycles o.C.Run.mcycle)
             { C.Config.dir; configuration; deadline = 86400.; flags = [] }),
        Ssa_ir.Ssa_target.zve32f128,
        true )

let route g ~inputs ~constants ~reference ~outputs =
  let module M = Machine_model.Mir_model in
  let module LS = Machine_model.Layer_select in
  let grant =
    Some
      {
        Machine_model.Sdpa_select.Grant.block_keys = 16;
        precision = Attention_block.Precision.F32_ordered;
      }
  in
  let coral_route, target, on_coral = coral () in
  let caps =
    { Contraction.Select.Capabilities.matrix = on_coral; vector = on_coral }
  in
  let sites =
    List.filter_map
      (fun (n : Graph_ir.node) ->
        match n.Graph_ir.Node.op with
        | Graph_ir.Attention_layer _ -> (
            match LS.choose ~grant caps g n.Graph_ir.Node.id with
            | Ok s -> Some (n.Graph_ir.Node.id, s)
            | Error r ->
                Fmt.pr "  %a not selected: %a@." Graph_ir.Node_id.pp
                  n.Graph_ir.Node.id LS.Reason.pp r;
                None)
        | _ -> None)
      g.Graph_ir.Graph.nodes
  in
  let layer id = List.assoc_opt id sites in
  let bundle =
    or_die Loop_ir.Loop_bundle.pp_error
      (Loop_ir.Loop_bundle.build ~granted:(fun id -> List.mem_assoc id sites) g)
  in
  let pipeline =
    Ssa_backends.Pipeline.Planned
      { numerics = Ssa_ir.Ssa_numerics.Fp32_ordered; target }
  in
  match
    M.prepare ?route:coral_route ~contractions:on_coral ~layer ~pipeline bundle
  with
  | Error rs ->
      Fmt.pr "  route refused: %a@." Fmt.(list ~sep:(any "; ") M.Refusal.pp) rs
  | Ok m -> (
      match
        M.Context.create m ~constants:(fun id ->
            Tensor_id.Map.find_opt id constants)
      with
      | Error s -> Fmt.pr "  route: %a@." M.Stop.pp s
      | Ok cx -> (
          match
            M.Context.run ~fuel:100_000_000_000L cx ~inputs:(fun id ->
                List.assoc_opt id inputs)
          with
          | Error s -> Fmt.pr "  route: %a@." M.Stop.pp s
          | Ok outs ->
              List.iter2
                (fun id t ->
                  let r = Tensor_id.Map.find id reference in
                  let worst = ref 0. in
                  Vec6.iter (Tensor.packed_shape t) (fun c ->
                      worst :=
                        Float.max !worst
                          (Float.abs (Tensor.read t c -. Tensor.read r c)));
                  Fmt.pr
                    "  route: %d layer(s) granted, output %a largest \
                     difference %.2e from the binary64 reference@."
                    (List.length sites) Tensor_id.pp id !worst)
                outputs outs))

(* A released fixture, opened offline from the cache. *)
let fixture cohort_path cache_dir id =
  let module Fixture = Pt2_fixture_unix.Fixture in
  or_die Fixture.pp_error
    (let open Err.Syntax in
     let* text = Pt2_fixture_unix.Fetch.read cohort_path in
     let* cohort = Pt2_fixture.Cohort.of_string text in
     let* cache = Pt2_fixture_unix.Cache.create cache_dir in
     let config = Pt2_fixture_unix.Bundle.config cache in
     match
       List.find_opt
         (fun (e : Pt2_fixture.Cohort.entry) -> e.artifact_id = id)
         cohort.entries
     with
     | None ->
         Fmt.epr "no artifact %s in the cohort@." id;
         exit 1
     | Some entry ->
         let+ f = Fixture.open_ config cohort entry in
         f.Fixture.archive)

(* A producer model directory with no weights of its own (a [tiny] artifact,
   whose weights are random at export): its graph and configs, every captured
   tensor filled deterministically. Floats lie within +-1, large enough that
   the rewrite's rounding differences reach the output; integers are zero, a valid index into any table. The rewrite is
   checked against the unrewritten graph on the same weights, so any values
   serve. *)
let synthetic dir =
  let read rel =
    In_channel.with_open_bin (Filename.concat dir rel) In_channel.input_all
  in
  let program =
    or_die Pt2_archive.pp_error
      (Pt2_archive.program_of_json (read "models/model.json"))
  and weights =
    or_die Pt2_archive.pp_error
      (Pt2_archive.weights_config_of_json
         (read "data/weights/model_weights_config.json"))
  and constants =
    let rel = "data/constants/model_constants_config.json" in
    if Sys.file_exists (Filename.concat dir rel) then
      or_die Pt2_archive.pp_error
        (Pt2_archive.constants_config_of_json (read rel))
    else Pt2_archive.no_constants
  in
  let meta name =
    let find (c : Pytorch_weights_config.ModelWeightsConfig.t) =
      Schema_runtime.String_map.find_opt name
        c.Pytorch_weights_config.ModelWeightsConfig.config
    in
    match find weights with Some e -> Some e | None -> find constants
  in
  let load name =
    match meta name with
    | None -> Error ("no config entry " ^ name)
    | Some e -> (
        match
          Err.payload
            (Pt2_tensor.of_meta e.Pytorch_weights_config.WeightEntry.tensor_meta
               ~data:Pt2_storage.empty)
        with
        | Error _ -> Error ("unreadable metadata " ^ name)
        | Ok t ->
            let size = Pt2_dtype.element_size t.Pt2_tensor.dtype in
            let cells =
              if List.exists (fun n -> n = 0) t.Pt2_tensor.sizes then 0
              else
                List.fold_left2
                  (fun a n st -> a + ((n - 1) * st))
                  (t.Pt2_tensor.storage_offset + 1)
                  t.Pt2_tensor.sizes t.Pt2_tensor.strides
            in
            let b = Bytes.make (cells * size) '\000' in
            (match t.Pt2_tensor.dtype with
            | Pt2_dtype.Float32 ->
                for i = 0 to cells - 1 do
                  let k = (i + Hashtbl.hash name) * 7919 mod 2003 in
                  Bytes.set_int32_le b (4 * i)
                    (Int32.bits_of_float (float_of_int (k - 1001) /. 1001.))
                done
            | Pt2_dtype.Bool -> Bytes.fill b 0 cells '\001'
            | _ -> ());
            Ok (Pt2_storage.of_string (Bytes.unsafe_to_string b)))
  in
  Pt2_archive.of_parts ~program ~weights ~constants ~load

let () =
  let id, archive =
    match Sys.argv with
    | [| _; dir |] when Sys.is_directory dir -> (dir, synthetic dir)
    | [| _; pt2 |] ->
        (pt2, or_die Pt2_archive.pp_error (Pt2_archive.open_pt2 pt2))
    | [| _; cohort; cache; id |] -> (id, fixture cohort cache id)
    | _ ->
        prerr_endline
          "usage: fuse_model (MODEL.pt2 | MODEL_DIR | COHORT CACHE ARTIFACT_ID)";
        exit 2
  in
  let dir = id in
  let lowered =
    or_die Native_interp.pp_error (Native_interp.lower_archive archive)
  in
  let constants =
    or_die Native_interp.pp_error (Native_interp.preload archive lowered)
  in
  let g = lowered.Pt2_native_graph.graph in
  (match Sys.getenv_opt "FUSE_MODEL_DUMP" with
  | Some path ->
      Out_channel.with_open_text path (fun oc ->
          Fmt.pf (Format.formatter_of_out_channel oc) "%a@." Graph_ir.pp g)
  | None -> ());
  let fused =
    match Rewrite.origin g with
    | Error e -> or_die Rewrite.pp_error (Error e)
    | Ok (Rewrite.Origin state) -> (
        match Pass.run_all state [ Dce.pass; Fuse_attention_layer.pass ] with
        | Error e -> or_die Pass.pp_error (Error e)
        | Ok (Rewrite.Step (final, _)) -> Rewrite.graph final)
  in
  Fmt.pr "%s: %d nodes -> %d, %d attention layer(s)@." (Filename.basename dir)
    (List.length g.Graph_ir.Graph.nodes)
    (List.length fused.Graph_ir.Graph.nodes)
    (layers fused);
  let inputs =
    List.filter_map
      (fun id ->
        if Graph_ir.input_kind g id = Graph_common.Input.Input then
          let name =
            match
              Tensor_id.Map.find_opt id lowered.Pt2_native_graph.tensor_origins
            with
            | Some (Pt2_native_graph.Source o) ->
                o.Pt2_native_graph.Tensor_origin.ssa_name
            | Some Pt2_native_graph.Derived | None -> ""
          in
          Some (id, input ~name (Tensor_id.Map.find id g.Graph_ir.Graph.tensors))
        else None)
      g.Graph_ir.Graph.inputs
  in
  let run g =
    let constants =
      Tensor_id.Map.bindings constants
      |> List.filter (fun (id, _) ->
          Tensor_id.Map.mem id g.Graph_ir.Graph.tensors)
    in
    or_die Eval_direct.pp_error (Eval_direct.run g ~constants ~inputs)
  in
  let before = run g and after = run fused in
  if
    Sys.getenv_opt "FUSE_MODEL_ROUTE" <> None
    || Sys.getenv_opt "FUSE_MODEL_CORAL" <> None
  then
    route fused ~inputs ~constants ~reference:before
      ~outputs:g.Graph_ir.Graph.outputs;
  List.iter2
    (fun a b ->
      let ta = Tensor_id.Map.find a before
      and tb = Tensor_id.Map.find b after in
      let worst = ref 0. and n = ref 0 in
      Vec6.iter (Tensor.packed_shape ta) (fun c ->
          incr n;
          worst :=
            Float.max !worst (Float.abs (Tensor.read ta c -. Tensor.read tb c)));
      Fmt.pr "  output %a: %d cells, largest difference %.2e@." Tensor_id.pp a
        !n !worst)
    g.Graph_ir.Graph.outputs fused.Graph_ir.Graph.outputs
