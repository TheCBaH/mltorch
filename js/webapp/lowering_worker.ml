(* The lowering inspector's worker: a second, lazily spawned jsoo worker that
   answers a Build_lowering request. It is [webapp_worker.ml] with the lowering
   view supplied to [Me_export.handle] and a ONE-ENTRY memo of the canonical
   bundle, because the canonical transform is ~80% of source-to-bundle and a
   click through the nodes of one model would otherwise pay it every time. The
   memo caches a pure function under its whole input, so it cannot be stale.
   The page-facing frame handling is the session worker's, repeated here. *)

open Js_of_ocaml
module Buffer = Webapp_jsoo_buffer
module Artifact = Lowering_view.Artifact

(* What the lowering view produced, as the protocol carries it: tags, not
   prose. Every constructor is matched, so a new refusal or note is a compile
   error here rather than a silent gap on the wire. A payload is carried as the
   view's own rendering of it. *)
let tag kind data = { Me_artifact.Tag.kind; data }
let node_text n = Core.Pretty.to_string Graph_ir.Node_id.pp n

let refusal (r : Artifact.Refusal.t) =
  let rendered () = [ Core.Pretty.to_string Artifact.Refusal.pp r ] in
  match r with
  | Granted n -> tag "granted" [ node_text n ]
  | Loop_c _ -> tag "loop_c" (rendered ())
  | Loop_lowering _ -> tag "loop_lowering" (rendered ())
  | Loop_wasm _ -> tag "loop_wasm" (rendered ())
  | Asm_aarch64 _ -> tag "asm_aarch64" (rendered ())
  | Asm_riscv32 _ -> tag "asm_riscv32" (rendered ())
  | Asm_x86_64 _ -> tag "asm_x86_64" (rendered ())
  | Mir_lowering _ -> tag "mir_lowering" (rendered ())
  | Mir_stage _ -> tag "mir_stage" (rendered ())
  | Ssa_backends s -> tag "ssa_backends" [ s ]
  | Ssa_lowering _ -> tag "ssa_lowering" (rendered ())
  | Unknown_node n -> tag "unknown_node" [ node_text n ]
  | Unsupported_family f ->
      tag "unsupported_family" [ Lowering_key.family_name f ]
  | Unsupported_policy _ -> tag "unsupported_policy" (rendered ())

let note (n : Artifact.Note.t) =
  let rendered () = [ Core.Pretty.to_string Artifact.Note.pp n ] in
  match n with
  | Code_bytes _ -> tag "code_bytes" (rendered ())
  | Fp32_refusal _ -> tag "fp32_refusal" (rendered ())
  | Fused_multiply_adds -> tag "fused_multiply_adds" []
  | Helper_bindings _ -> tag "helper_bindings" (rendered ())
  | Helpers _ -> tag "helpers" (rendered ())
  | Local_doubles _ -> tag "local_doubles" (rendered ())
  | Loop_target _ -> tag "loop_target" (rendered ())
  | Mir_summary s -> tag "mir_summary" [ s ]
  | Numerics _ -> tag "numerics" (rendered ())
  | Numerics_not_applicable _ -> tag "numerics_not_applicable" (rendered ())
  | Relocations _ -> tag "relocations" (rendered ())
  | Row_blocking -> tag "row_blocking" []
  | Ssa_fp32_refusal _ -> tag "ssa_fp32_refusal" (rendered ())
  | Ssa_target _ -> tag "ssa_target" (rendered ())
  | Symbols _ -> tag "symbols" (rendered ())
  | Vectorization _ -> tag "vectorization" (rendered ())
  | Working_precision _ -> tag "working_precision" (rendered ())

(* The canonical bundle is built granting nothing, so no node of it is granted
   (see bin/native_graph_lower.ml). *)
let nobody (_ : Graph_ir.Node_id.t) = false

let lower ~limits (key : Lowering_key.t) bundle : Me_artifact.t =
  let a =
    Lowering_view.of_node ~dialect:key.dialect ~granted:nobody bundle key.node
      key.variant
  in
  let body =
    match a.Artifact.body with
    | Artifact.Refused r -> Me_artifact.Body.Refused (refusal r)
    | Artifact.Text { language; text; truncated = _ } ->
        Me_artifact.text ~limits
          ~language:(Artifact.Language.name language)
          text
  in
  { Me_artifact.key; body; notes = List.map note a.Artifact.notes }

(* One entry: the source bytes' digest, everything else the bundle depends on,
   and the bundle. A failure is not remembered. *)
let memo = ref None

let bundle ~limits ~options ~bytes =
  let digest = Digest.string bytes in
  let stages = options.Me_export.Options.stages
  and fold = options.Me_export.Options.fold in
  match !memo with
  | Some (d, st, f, l, b)
    when d = digest && st = stages && f = fold && l = limits ->
      Ok b
  | _ -> (
      match Me_export.canonical_bundle ~limits ~options ~bytes with
      | Ok b ->
          memo := Some (digest, stages, fold, limits, b);
          Ok b
      | Error _ as e -> e)

let post wire =
  let message =
    match wire.Me_response.Wire.payload with
    | None ->
        Js.Unsafe.obj [| ("meta", Js.Unsafe.inject (Js.string wire.meta)) |]
    | Some payload ->
        let payload = Buffer.fresh_array_buffer payload in
        Js.Unsafe.obj
          [|
            ("meta", Js.Unsafe.inject (Js.string wire.meta));
            ("payload", Js.Unsafe.inject payload);
          |]
  in
  let args =
    match wire.Me_response.Wire.payload with
    | None -> [| Js.Unsafe.inject message |]
    | Some _ ->
        [|
          Js.Unsafe.inject message;
          Js.Unsafe.inject (Js.array [| Js.Unsafe.get message "payload" |]);
        |]
  in
  ignore
    (Js.Unsafe.fun_call (Js.Unsafe.get Js.Unsafe.global "postMessage") args)

let protocol_failure () = post Me_response.Wire.constant_protocol_failure

let emit progress =
  match Me_response.Wire.of_progress progress with
  | Ok wire -> post wire
  | Error _ -> protocol_failure ()

let handle event =
  try
    let data = Js.Unsafe.get event "data" in
    let request = Js.Unsafe.get data "request" in
    let bytes = Js.Unsafe.get data "bytes" in
    let request_length =
      match Buffer.array_buffer_length request with
      | Error _ -> raise Exit
      | Ok n ->
          Err.or_raise
            ~pp_error:(fun _ _ -> ())
            (Me_bytes.checked_length n
               ~max:(Int64.of_int Me_limits.Hard.max_request_json_bytes))
    in
    let request =
      match
        Jsont_bytesrw.decode_string Me_request.Request.jsont
          (Buffer.to_string (Js.Unsafe.coerce request) ~len:request_length)
      with
      | Ok request -> request
      | Error _ -> raise Exit
    in
    let bytes_length =
      match Buffer.array_buffer_length bytes with
      | Error _ -> raise Exit
      | Ok n ->
          Err.or_raise
            ~pp_error:(fun _ _ -> ())
            (Me_bytes.checked_length n ~max:Me_limits.Hard.jsoo_safe_bytes)
    in
    if
      Int64.compare
        (Int64.of_int bytes_length)
        (Me_request.Source.bytes (Me_request.Request.source request))
      <> 0
    then raise Exit;
    let result =
      Me_export.handle ~lower ~bundle ~emit request
        ~bytes:(Buffer.to_string (Js.Unsafe.coerce bytes) ~len:bytes_length)
    in
    match
      Me_response.Wire.of_final (Me_response.Final.of_handle_result result)
    with
    | Ok wire -> post wire
    | Error _ -> protocol_failure ()
  with e ->
    let detail =
      match e with
      | Err.Exn.E packed -> Format.asprintf "%a" Err.Exn.pp_kind packed
      | e -> Printexc.to_string e
    in
    ignore
      (Js.Unsafe.meth_call Console.console "error"
         [| Js.Unsafe.inject (Js.string detail) |]);
    protocol_failure ()

let () = Js.Unsafe.set Js.Unsafe.global "onmessage" (Js.wrap_callback handle)
