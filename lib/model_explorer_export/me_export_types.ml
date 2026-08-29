(* Small, independent helpers for me_export.ml: format detection, the error
   vocabulary, the encode-with-a-byte-ceiling writer, and the [staged]/[shape]
   types both unlowered_shape and lowered_shape build. Split out
   — see me_export.ml for the public entry points and me_export_shape.ml for
   the two shape-construction functions. *)

type source_format = Model_json | Pt2_archive

(* CONTENT, never a declared extension: a mislabelled file is the ordinary case
   rather than the adversarial one, and the two loaders fail very differently on
   each other's input. *)
let detect ~bytes =
  if String.length bytes >= 4 && String.sub bytes 0 4 = "PK\003\004" then
    Err.return Pt2_archive
  else if String.length bytes >= 1 && bytes.[0] = '{' then Err.return Model_json
  else Err.fail `Unrecognised_format

(* Which ceiling applies is a fact about the CONTENT, the same rule [detect]
   uses. An unrecognised format falls back to [max_pt2_bytes]: [load] rejects
   it with [`Unrecognised_format] right after, so this check only needs to not
   itself be the reason a legitimately-typed file is missed. *)
let max_bytes_for ~limits bytes =
  match detect ~bytes with
  | Ok Model_json -> limits.Me_limits.Limits.max_json_bytes
  | Ok Pt2_archive | Error _ -> limits.Me_limits.Limits.max_pt2_bytes

type error =
  [ `Archive of Pt2_archive.error
  | `Declared_format_disagrees
  | `Document of Me_session.Session.error
  | `Document_too_large
    (** the ENCODED document overran [max_session_bytes]/[max_detail_bytes]. No
        byte count: the writer aborts mid-stream once the limit is hit, so
        unlike [`Too_large] there is no final length to report. *)
  | `Flow_graph of Me_flow_graph.error
  | `Fusion of Me_fusion.error
  | `Identifier of Me_ids.error
  | `Kernel of Kernel_adapt.error
  | `Lowering of Native_interp.error
  | `Model_json_decode of string
  | `Native4d of Native4d.Error.t
  | `Navigation of Me_pt2.error
  | `Project of Me_build.error
  | `Source_view of Me_source.error
  | `Too_large of int64
  | `Unrecognised_format
  | `Unsupported_detail_key
  | `Value_graph of Me_kernel.error
  | `Verification of Me_verify.error
  | `View of Graph_view.error ]

let pp_error fmt : [< error ] -> unit = function
  | `Archive e -> Pt2_archive.pp_error fmt e
  | `Declared_format_disagrees ->
      Fmt.string fmt "the bytes are not the format the request declared"
  | `Document e -> Me_session.Session.pp_error fmt e
  | `Document_too_large ->
      Fmt.string fmt "the encoded document is over the ceiling"
  | `Flow_graph e -> Me_flow_graph.pp_error fmt e
  | `Fusion e -> Me_fusion.pp_error fmt e
  | `Identifier e -> Me_ids.pp_error fmt e
  | `Kernel e -> Kernel_adapt.pp_error fmt e
  | `Lowering e -> Native_interp.pp_error fmt e
  | `Model_json_decode m -> Fmt.pf fmt "model.json: %s" m
  | `Native4d e -> Native4d.Error.pp fmt e
  | `Navigation e -> Me_pt2.pp_error fmt e
  | `Project e -> Me_build.pp_error fmt e
  | `Source_view e -> Me_source.pp_error fmt e
  | `Too_large n -> Fmt.pf fmt "%Ld bytes is over the ceiling" n
  | `Unrecognised_format ->
      Fmt.string fmt "not a zip archive and not a JSON object"
  | `Unsupported_detail_key ->
      Fmt.string fmt "the key names no value this model produces"
  | `Value_graph e -> Me_kernel.pp_error fmt e
  | `Verification e -> Me_verify.pp_error fmt e
  | `View e -> Graph_view.pp_error fmt e

(* The same line [Me_classify] draws: a model that is not what it claimed is
   [Invalid_source], a ceiling is [Over_limit], and an invariant of ours is
   [Internal]. A ceiling reached inside the projection is not our defect -- a
   real model can be too big, and that is a bound doing its job. *)
let diagnostic_code : [< error ] -> Me_limits.Diagnostic.Code.t =
  let module Code = Me_limits.Diagnostic.Code in
  (* Grouped by the diagnostic outcome, not alphabetically: each branch names
     one code category that crosses the wire. *)
  function
  | `Archive _ | `Declared_format_disagrees | `Model_json_decode _
  | `Unrecognised_format ->
      Code.Invalid_source
  | `Document_too_large | `Too_large _ -> Code.Over_limit
  | `Unsupported_detail_key -> Code.Unsupported_detail_key
  | `Kernel _ | `Lowering _ | `Native4d _ -> Code.Internal
  | `Document (`Over_limit _)
  | `Flow_graph (`Over_limit _)
  | `Fusion (`Over_limit _)
  | `Navigation (`Over_limit _)
  | `Project (`Over_limit _)
  | `Source_view (`Over_limit _)
  | `Value_graph (`Over_limit _)
  | `Verification (`Over_limit _) ->
      Code.Over_limit
  | `Document _ | `Flow_graph _ | `Identifier _ | `Navigation _ | `Project _
  | `Source_view _ | `Value_graph _ | `Verification _ | `View _ ->
      Code.Internal

(* The one place a component's error is widened into this module's. Named,
   rather than a [Err.map_error ~pos:__POS__] at each of the twelve crossings, so the
   wrapping constructor is visible at the call site and the detection backtrace
   is preserved by [map_error] rather than rebuilt. *)
let wrap tag r = Err.map_error ~pos:__POS__ tag r
let ( let* ) = Err.Syntax.( let* )
let ( let+ ) = Err.Syntax.( let+ )

(* The PRODUCER side of the session/detail byte ceilings: a writer that
   refuses to grow past [max_bytes] rather than a check on the string after
   [encode_string] already built the whole thing. Same shape as
   [Me_response.Wire.encode_bounded], which bounds the META wrapper -- this
   bounds the document itself, which is the larger of the two and the one an
   oversized model can actually reach. No byte count on failure: the writer
   aborts mid-stream once the limit is hit, so there is no final length to
   report, same as [Wire]'s [`Meta_too_large]. *)
let encode_bounded ~max_bytes jsont v =
  let buf = Buffer.create 4096 in
  let w =
    Bytesrw.Bytes.Writer.limit max_bytes
      (Bytesrw.Bytes.Writer.of_buffer buf)
      ~eod:true
  in
  match Jsont_bytesrw.encode jsont v ~eod:true w with
  | Ok () -> Err.return (Buffer.contents buf)
  | Error _ -> Err.fail `Document_too_large
  | exception Bytesrw.Bytes.Stream.Error _ -> Err.fail `Document_too_large
  (* bytesrw 0.3.0's own over-limit signal, [Bytesrw.Bytes.Stream.Error], is
     what [Writer.limit] is DOCUMENTED to raise -- but its implementation has
     an off-by-one: when the remaining budget lands on EXACTLY 0 right at a
     slice boundary, it computes [Slice.take 0 s], which [Slice.take] treats
     as an empty/invalid request and returns [None] (its [n <= 0] guard
     doesn't distinguish "asked for nothing" from "asked for a valid
     zero-length remainder"), and the writer immediately [Option.get]s that
     -- raising [Invalid_argument] instead of the documented error. This is
     an upstream bug, still present on bytesrw's main branch; patched at
     vendored/bytesrw (a submodule pointing at
     github.com/TheCBaH/bytesrw, a fork of github.com/dbuenzli/bytesrw --
     see vendored/bytesrw's own git log). Not wired into this repo's own
     build: bytesrw is also a fixed transitive dependency of the
     opam-installed `jsont` package, and rebuilding that package against a
     patched bytesrw needs `opam pin`, which needs write access to the opam
     switch that this devcontainer doesn't grant this process. The OUTCOME
     here is identical to the documented case (the
     document didn't fit), so it gets the same diagnostic rather than
     crashing the whole export. Caught by mobilenetv2_050 + `--limits small`,
     whose session lands almost exactly on that boundary
     (test/me_visualize_json_cram.t). Drop this arm once the opam-installed
     bytesrw incorporates the upstream fix. *)
  | exception Invalid_argument _ -> Err.fail `Document_too_large

let passes ~fold = [ Pipeline.canonical ~fold ]
let capability key status = { Me_session.Capability.key; status }

(* The outcome of a stage past Canonical, one step richer than the
   [(_, reason * string) result] the classified stages already used: that
   type has no way to say "never attempted" distinctly from "attempted and
   refused", and conflating them would report [Not_requested] rows as
   [Unavailable] with a manufactured reason, or vice versa. [Skipped] and
   [Refused] both read as absent to a comparison or a flow state -- the same
   place the two-case version already collapsed [Error] to "leave it out" --
   only the capability row needs to tell them apart. *)
type ('a, 'reason) staged =
  | Ready of 'a
  | Refused of 'reason * string
  | Skipped

let staged_status ~wanted staged payload =
  match (wanted, staged) with
  | false, _ -> Me_session.Capability.Not_requested
  | true, Ready a -> Me_session.Capability.Available (payload a)
  | true, Refused (reason, detail) ->
      Me_session.Capability.Unavailable { reason; detail = Some detail }
  | true, Skipped -> Me_session.Capability.Not_requested

(* The exported program's own distinct op targets -- what the field NAMES, and
   the one figure both shapes below can produce, since a session whose lowering
   failed has no native graph whose nodes it could count instead. *)
let op_targets (g : Pytorch_types.Graph.t) =
  List.map
    (fun (n : Pytorch_types.Node.t) -> n.Pytorch_types.Node.target)
    g.Pytorch_types.Graph.nodes
  |> List.sort_uniq compare |> List.length

(* Everything that differs between "we lowered this" and "we could not", as ONE
   record rather than six threaded options. Those are the only two shapes a
   session takes here, and six independent options describe combinations that
   are not sessions -- a flow spine with no Native state, a comparison naming a
   graph the collection does not hold. *)
type shape = {
  graphs : Model_explorer.Graph.t list;
  views : Me_session.View.t list;
  comparisons : Me_session.Comparison.t list;
  node_data_sets : Me_session.Node_data_set.t list;
  diagnostics : Me_limits.Diagnostic.t list;
      (** what a STAGE had to say beyond its capability row. The vector says
          which rows are missing; a diagnostic carries the bounded detail, and a
          stage that succeeded can still have something to report -- fusion's
          rejections are the case. *)
  flow : Me_flow.t option;
  capabilities : Me_session.Capability.t list;
  default_view : string;
}

(* A model this repository cannot lower still has a source view: decoding
   succeeded, which is the whole of what that row claims. So it is a SUCCESSFUL
   session whose capabilities say what is missing and why -- not a usage error,
   because the browser shell cannot surface one and the same code path serves
   both. *)
