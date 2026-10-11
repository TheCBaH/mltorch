open Err.Syntax

module Output_mismatch = struct
  type t = { actual : float; index : int; reference : float }
end

module Failure = struct
  type t =
    | Case_count_outside_supported_limits
    | Case_index_outside_publication
    | Declared_signature_arity_disagrees
    | Decoded_image_dimensions_exceed_limit
    | Decoded_image_is_not_rgb_chw
    | Decoded_image_is_not_uint8
    | Duplicate of string
    | Dynamic_shape_unsupported
    | Expected_array
    | Expected_finite_number
    | Expected_object
    | Expected_string
    | Field_disagreement of string
    | Final_output_arity_differs
    | Geometry_event_count_differs
    | Http_status of string
    | Image_preprocessing_requires_one_chw_image_binding
    | Integer_exceeds_supported_limit
    | Invalid_asset_filename
    | Invalid_normalization
    | Invalid_padding_pixel
    | Invalid_producer_commit
    | Invalid_sha256
    | Json_exceeds_byte_limit
    | Missing_bindings
    | Missing_decoded_image of string
    | Missing_field of string
    | Missing_graph_tensor_metadata
    | Missing_image_input_reference
    | Missing_tensor of string
    | Missing_weight_map
    | Negative_comparison_tolerance
    | Nonfinite_tensor_value
    | Nonpositive_dimension
    | Output_arity_differs
    | Output_mismatch of Output_mismatch.t
    | Output_select_requires_leading_extent_1
    | Output_transform_used_for_input
    | Resize_exceeds_dimension_limit
    | Resized_image_exceeds_byte_limit
    | Smoke_model_outside_release_population
    | Tensor_byte_count_disagrees
    | Too_many_assets
    | Unexpected_or_missing_object_field
    | Unknown_model of string
    | Unresolved_asset of string
    | Unsupported_asset_format
    | Unsupported_crop_mode
    | Unsupported_crop_percentage
    | Unsupported_input_axis
    | Unsupported_input_transform
    | Unsupported_interpolation
    | Unsupported_layout
    | Unsupported_manifest_scope
    | Unsupported_model_count
    | Unsupported_output_transform
    | Unsupported_rank
    | Unsupported_rgb_target_dimensions
    | Unsupported_sample_layout
    | Unsupported_tensor_count
    | Unsupported_tensor_dtype
    | Usage
    | V2_manifest_required_for_input_verification
end

module External = struct
  module Source = struct
    type t =
      | Archive
      | Cache
      | Checkpoint
      | Logical
      | Native
      | Zip
      | Zip_limits

    let name = function
      | Archive -> "pt2 archive"
      | Cache -> "fixture cache"
      | Checkpoint -> "safetensors checkpoint"
      | Logical -> "logical tensor"
      | Native -> "native interpreter"
      | Zip -> "zip"
      | Zip_limits -> "zip limits"
  end

  type t = { message : string; source : Source.t }
end

module Io = struct
  type t = { message : string; path : string }
end

type error =
  [ `Jsont of string
  | `Native_layout of string
  | `Timm_fixture of Failure.t
  | `Timm_fixture_external of External.t
  | `Timm_fixture_io of Io.t ]

let pp_failure ppf = function
  | Failure.Case_count_outside_supported_limits ->
      Fmt.string ppf "case count outside supported limits"
  | Failure.Case_index_outside_publication ->
      Fmt.string ppf "case index outside publication"
  | Failure.Declared_signature_arity_disagrees ->
      Fmt.string ppf "declared signature arity disagrees"
  | Failure.Decoded_image_dimensions_exceed_limit ->
      Fmt.string ppf "decoded image dimensions exceed limit"
  | Failure.Decoded_image_is_not_rgb_chw ->
      Fmt.string ppf "decoded image is not RGB CHW"
  | Failure.Decoded_image_is_not_uint8 ->
      Fmt.string ppf "decoded image is not uint8"
  | Failure.Duplicate value -> Fmt.pf ppf "duplicate %s" value
  | Failure.Dynamic_shape_unsupported ->
      Fmt.string ppf "dynamic shape unsupported"
  | Failure.Expected_array -> Fmt.string ppf "expected array"
  | Failure.Expected_finite_number -> Fmt.string ppf "expected finite number"
  | Failure.Expected_object -> Fmt.string ppf "expected object"
  | Failure.Expected_string -> Fmt.string ppf "expected string"
  | Failure.Field_disagreement value -> Fmt.pf ppf "%s disagrees" value
  | Failure.Final_output_arity_differs ->
      Fmt.string ppf "final output arity differs"
  | Failure.Geometry_event_count_differs ->
      Fmt.string ppf "geometry event count differs"
  | Failure.Http_status value -> Fmt.pf ppf "v2 discovery failed: %s" value
  | Failure.Image_preprocessing_requires_one_chw_image_binding ->
      Fmt.string ppf "image preprocessing requires one CHW image binding"
  | Failure.Integer_exceeds_supported_limit ->
      Fmt.string ppf "integer exceeds supported limit"
  | Failure.Invalid_asset_filename -> Fmt.string ppf "invalid asset filename"
  | Failure.Invalid_normalization -> Fmt.string ppf "invalid normalization"
  | Failure.Invalid_padding_pixel -> Fmt.string ppf "invalid padding pixel"
  | Failure.Invalid_producer_commit -> Fmt.string ppf "invalid producer commit"
  | Failure.Invalid_sha256 -> Fmt.string ppf "invalid sha256"
  | Failure.Json_exceeds_byte_limit -> Fmt.string ppf "JSON exceeds byte limit"
  | Failure.Missing_bindings -> Fmt.string ppf "missing bindings"
  | Failure.Missing_decoded_image value ->
      Fmt.pf ppf "missing decoded image %s" value
  | Failure.Missing_field value -> Fmt.pf ppf "missing field %s" value
  | Failure.Missing_graph_tensor_metadata ->
      Fmt.string ppf "missing graph tensor metadata"
  | Failure.Missing_image_input_reference ->
      Fmt.string ppf "missing image input reference"
  | Failure.Missing_tensor value -> Fmt.pf ppf "missing tensor key %s" value
  | Failure.Missing_weight_map -> Fmt.string ppf "missing weight map"
  | Failure.Negative_comparison_tolerance ->
      Fmt.string ppf "negative comparison tolerance"
  | Failure.Nonfinite_tensor_value -> Fmt.string ppf "nonfinite tensor value"
  | Failure.Nonpositive_dimension -> Fmt.string ppf "nonpositive dimension"
  | Failure.Output_arity_differs -> Fmt.string ppf "output arity differs"
  | Failure.Output_mismatch e ->
      Fmt.pf ppf "output element %d: %.9g vs %.9g" e.index e.actual e.reference
  | Failure.Output_select_requires_leading_extent_1 ->
      Fmt.string ppf "output select requires leading extent 1"
  | Failure.Output_transform_used_for_input ->
      Fmt.string ppf "output transform used for input"
  | Failure.Resize_exceeds_dimension_limit ->
      Fmt.string ppf "resize exceeds dimension limit"
  | Failure.Resized_image_exceeds_byte_limit ->
      Fmt.string ppf "resized image exceeds byte limit"
  | Failure.Smoke_model_outside_release_population ->
      Fmt.string ppf "smoke model outside release population"
  | Failure.Tensor_byte_count_disagrees ->
      Fmt.string ppf "tensor byte count disagrees"
  | Failure.Too_many_assets -> Fmt.string ppf "too many assets"
  | Failure.Unexpected_or_missing_object_field ->
      Fmt.string ppf "unexpected or missing object field"
  | Failure.Unknown_model value -> Fmt.pf ppf "unknown current model %s" value
  | Failure.Unresolved_asset value -> Fmt.pf ppf "unresolved asset %s" value
  | Failure.Unsupported_asset_format ->
      Fmt.string ppf "unsupported asset format"
  | Failure.Unsupported_crop_mode -> Fmt.string ppf "unsupported crop mode"
  | Failure.Unsupported_crop_percentage ->
      Fmt.string ppf "unsupported crop percentage"
  | Failure.Unsupported_input_axis -> Fmt.string ppf "unsupported input axis"
  | Failure.Unsupported_input_transform ->
      Fmt.string ppf "unsupported input transform"
  | Failure.Unsupported_interpolation ->
      Fmt.string ppf "unsupported interpolation"
  | Failure.Unsupported_layout -> Fmt.string ppf "unsupported layout"
  | Failure.Unsupported_manifest_scope ->
      Fmt.string ppf "unsupported manifest scope"
  | Failure.Unsupported_model_count -> Fmt.string ppf "unsupported model count"
  | Failure.Unsupported_output_transform ->
      Fmt.string ppf "unsupported output transform"
  | Failure.Unsupported_rank -> Fmt.string ppf "unsupported rank"
  | Failure.Unsupported_rgb_target_dimensions ->
      Fmt.string ppf "unsupported RGB target dimensions"
  | Failure.Unsupported_sample_layout ->
      Fmt.string ppf "unsupported sample layout"
  | Failure.Unsupported_tensor_count ->
      Fmt.string ppf "unsupported tensor count"
  | Failure.Unsupported_tensor_dtype ->
      Fmt.string ppf "unsupported tensor dtype"
  | Failure.Usage ->
      Fmt.string ppf
        "usage: timm_fixture_tool \
         (download|replay|replay-images|verify-inputs) MODEL; \
         verify-inputs-all; verify-patterns DIRECTORY"
  | Failure.V2_manifest_required_for_input_verification ->
      Fmt.string ppf "v2 manifest required for input verification"

let pp_error ppf = function
  | `Jsont message -> Fmt.pf ppf "JSON: %s" message
  | `Native_layout message -> Fmt.pf ppf "native layout: %s" message
  | `Timm_fixture failure -> pp_failure ppf failure
  | `Timm_fixture_external e ->
      Fmt.pf ppf "%s: %s" (External.Source.name e.External.source) e.message
  | `Timm_fixture_io e -> Fmt.pf ppf "%s: %s" e.Io.path e.message

let bad failure = Err.fail (`Timm_fixture failure)

let wrap source pp r =
  Err.map_error
    (fun e ->
      `Timm_fixture_external { External.message = Fmt.str "%a" pp e; source })
    r

let ( >>= ) = Err.bind

let members = function
  | Jsont.Object (xs, _) -> Ok (List.map (fun ((k, _), v) -> (k, v)) xs)
  | _ -> bad Failure.Expected_object

let member key value =
  let* xs = members value in
  Err.of_option (`Timm_fixture (Failure.Missing_field key))
    (List.assoc_opt key xs)

let path keys value = Err.List.fold_left (fun j k -> member k j) value keys

let string = function
  | Jsont.String (s, _) -> Ok s
  | _ -> bad Failure.Expected_string

let array = function
  | Jsont.Array (xs, _) -> Ok xs
  | _ -> bad Failure.Expected_array

let field key j = member key j >>= string

let fields names j =
  let* xs = members j in
  let actual = List.map fst xs in
  if List.sort String.compare names = List.sort String.compare actual then Ok ()
  else bad Failure.Unexpected_or_missing_object_field

let number = function
  | Jsont.Number (n, _) when Float.is_finite n -> Ok n
  | _ -> bad Failure.Expected_finite_number

let integer j =
  let* n = number j in
  if n < 0. || n > 536870912. || Float.floor n <> n then
    bad Failure.Integer_exceeds_supported_limit
  else Ok (int_of_float n)

let same what a b =
  if a = b then Ok () else bad (Failure.Field_disagreement what)

let unique names =
  let sorted = List.sort String.compare names in
  let rec go = function
    | a :: (b :: _ as rest) ->
        if a = b then bad (Failure.Duplicate a) else go rest
    | _ -> Ok ()
  in
  go sorted

let rec keys = function
  | Jsont.Object (xs, _) ->
      let* () = unique (List.map (fun ((k, _), _) -> k) xs) in
      Err.List.iter (fun (_, j) -> keys j) xs
  | Jsont.Array (xs, _) -> Err.List.iter keys xs
  | _ -> Ok ()

let parse text =
  if String.length text > 8 * 1024 * 1024 then
    bad Failure.Json_exceeds_byte_limit
  else
    let* j =
      Jsont_bytesrw.decode_string Jsont.json text
      |> Err.import ~pos:__POS__ (fun e -> `Jsont e)
    in
    let+ () = keys j in
    j

let version n j =
  member "schema_version" j >>= integer >>= same "schema_version" n

let shape j =
  let* xs = array j in
  if List.length xs > 8 then bad Failure.Unsupported_rank
  else
    let* dims = Err.List.map integer xs in
    if List.exists (( >= ) 0) dims then bad Failure.Nonpositive_dimension
    else Ok dims

let digest s =
  Err.of_option (`Timm_fixture Failure.Invalid_sha256)
    (Pt2_sha256.Digest.of_hex s)
