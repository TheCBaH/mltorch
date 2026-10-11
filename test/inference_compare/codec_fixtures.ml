(* Shared builders for the codec-layer tests. *)

open Inference_compare_schema
module Codecs = Inference_compare_codecs

let text r = Err.or_raise ~pp_error:Text_error.pp r
let codec_error_pp = Err.Error.pp_kind Codecs.Codec_error.pp
let ok r = Err.or_raise ~pp_error:Codecs.Codec_error.pp r

let context ?limit ?(hashers = Inference_compare_sha256.registry)
    ?(algorithm = Inference_compare_sha256.algorithm) () :
    Codecs.Logical_value.Context.t =
  {
    codecs = Codecs.Element_codec.Registry.builtin;
    hashers;
    algorithm;
    ledger = Codecs.Owned_blob.Ledger.create ?limit ();
  }

let spec codec shape : Tensor_spec.t =
  {
    dtype = { codec; parameters_digest = None };
    logical_shape = shape;
    layout_id = text (Id.Layout.of_string "row_major");
    value_semantics = true;
  }

let f32 = spec Codecs.Element_codec.f32
let f64 = spec Codecs.Element_codec.f64
let i32 = spec Codecs.Element_codec.i32
let i64 = spec Codecs.Element_codec.i64
let u8 = spec Codecs.Element_codec.u8
let canonical s = text (Id.Canonical_value.of_string s)
let node s = text (Id.Conversion_node.of_string s)
let recipe_id s = text (Id.Recipe.of_string s)

let graph : Graph_key.t =
  let d = text (Id.Digest_algorithm.of_string "sha256") in
  {
    dialect = text (Id.Dialect.of_string "native");
    version = text (Id.Graph_version.of_string "v1");
    graph_digest =
      Err.or_raise ~pp_error:Digest_error.pp
        (Digest.Graph.make d ~hex:(String.make 64 '0'));
  }

let boundary name : Tensor_ref.t =
  { graph; scope = []; local_id = text (Id.Tensor_local.of_string name) }

let step_node id operation arguments : Recipe.node =
  { id = node id; operation; arguments }

let recipe ?(lossless = true) id nodes result output_spec : Recipe.t =
  { id = recipe_id id; nodes; result; output_spec; lossless }

let env ?(canonical = []) ?(boundary = [])
    ?(operations = Codecs.Conversion_registry.empty) () : Codecs.Recipe_eval.env
    =
  {
    codecs = Codecs.Element_codec.Registry.builtin;
    operations;
    canonical =
      (fun c ->
        List.find_map
          (fun (k, s) -> if Id.Canonical_value.equal k c then Some s else None)
          canonical);
    boundary =
      (fun r ->
        List.find_map
          (fun (k, s) -> if Tensor_ref.equal k r then Some s else None)
          boundary);
  }

let inputs ?(canonical = []) ?(boundary = []) () : Codecs.Recipe_eval.inputs =
  {
    canonical =
      (fun c ->
        List.find_map
          (fun (k, v) -> if Id.Canonical_value.equal k c then Some v else None)
          canonical);
    boundary =
      (fun r ->
        List.find_map
          (fun (k, v) -> if Tensor_ref.equal k r then Some v else None)
          boundary);
  }

let integers ctx s values = ok (Codecs.Logical_value.of_integers ctx s values)
let floats ctx s values = ok (Codecs.Logical_value.of_floats ctx s values)

(* The integers of a value, for printing. *)
let to_integers ctx v =
  let c = Codecs.Logical_value.codec ctx v in
  List.init (Codecs.Logical_value.element_count v) (fun i ->
      Codecs.Element_codec.integer_value c (Codecs.Logical_value.contents v) i)

let to_floats ctx v =
  let c = Codecs.Logical_value.codec ctx v in
  List.init (Codecs.Logical_value.element_count v) (fun i ->
      Codecs.Element_codec.float_value c (Codecs.Logical_value.contents v) i)

let to_bits ctx v =
  let c = Codecs.Logical_value.codec ctx v in
  List.init (Codecs.Logical_value.element_count v) (fun i ->
      Codecs.Element_codec.bits c (Codecs.Logical_value.contents v) i)

let pp_ints = Fmt.(brackets (list ~sep:semi int64))
let pp_floats = Fmt.(brackets (list ~sep:semi float))

let pp_hex_bits =
  Fmt.(brackets (list ~sep:semi (fun ppf b -> Fmt.pf ppf "0x%Lx" b)))

(* Evaluate a recipe end to end and return its value, or print the error. *)
let run ctx env_ inputs_ recipe_ =
  match Codecs.Recipe_eval.check env_ recipe_ with
  | Error e -> Error (`Check e)
  | Ok checked -> (
      match Codecs.Recipe_eval.evaluate ctx env_ checked inputs_ with
      | Error e -> Error (`Eval e)
      | Ok (v, lineage) -> Ok (checked, v, lineage))

let pp_failure ppf = function
  | `Check e -> Fmt.pf ppf "check: %a" codec_error_pp e
  | `Eval e -> Fmt.pf ppf "eval: %a" codec_error_pp e
