(* The embedded CompCert route as a graph engine: the body's Loop IR bundle is
   generated as C in the Compcert_scalar dialect, compiled in this process by
   the embedded CompCert, assembled and loaded by Rivet, and called on one
   caller-owned region. No host compiler is involved and a refusal at any step
   is the engine's own. *)

module E = Loop_c_embed.Make (Loop_c_embed_compcert.Compcert_loader)

module Make (X : sig
  val numerics : Loop_ir.Loop_numerics.t
end) =
struct
  type op = Graph_ir.op
  type compiled = E.prepared

  let text pp e = Format.asprintf "%a" pp (Err.Error.kind e)

  let compile (g : Graph_ir.graph) =
    match
      Loop_ir.Loop_bundle.build ~config:Loop_ir.Loop_bundle_c.default_config g
    with
    | Error e -> Error (text Loop_ir.Loop_bundle.pp_error e)
    | Ok bundle -> (
        match Err.payload (E.prepare ~numerics:X.numerics bundle) with
        | Ok p -> Ok p
        | Error e -> Error (Format.asprintf "%a" Loop_c_embed.pp_error e))

  let run ~poison p ~inputs =
    let bind id = List.assoc_opt id inputs in
    match Err.payload (E.context p ~constants:(fun _ -> None)) with
    | Error e -> Error (Format.asprintf "%a" Loop_c_embed.pp_error e)
    | Ok ctx ->
        let r = Err.payload (E.run ~poison ctx ~bind) in
        E.close_context ctx;
        Result.map_error (Format.asprintf "%a" Loop_c_embed.pp_error) r

  let execute p (_ : Graph_ir.graph) ~inputs = run ~poison:false p ~inputs

  (* The workspace is filled with a pattern before the call: a kernel that read
     scratch it had not written would differ from {!execute}. *)
  let execute_poisoned p (_ : Graph_ir.graph) ~inputs =
    run ~poison:true p ~inputs

  let release = E.close
end

include Make (struct
  let numerics = Loop_ir.Loop_numerics.Reference_f64
end)
