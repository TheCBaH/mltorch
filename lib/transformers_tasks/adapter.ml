open Err.Syntax
open Transformers_metadata.Json_util
module M = Transformers_metadata
module L = Pt2_fixture.Logical
module D = Pt2_checkpoint_map.Dtype

let number = function
  | Jsont.Number (v, _) when Float.is_finite v -> Ok v
  | _ -> invalid "finite processor number required"

let extent key value =
  let* n = member key value >>= integer in
  if n >= 2L && n <= 4096L then Ok (Int64.to_int n)
  else invalid ("bounded recipe extent: " ^ key)

let lines s =
  match List.rev (String.split_on_char '\n' s) with
  | "" :: rest -> List.rev rest
  | all -> List.rev all

let bert vocab_text ~length text =
  let vocab = Wordpiece.vocab_of_lines (lines vocab_text) in
  let* e =
    Err.import
      (fun e -> `Metadata_invalid (Fmt.str "%a" Wordpiece.pp_error e))
      (Wordpiece.encode vocab ~max_length:length text)
  in
  Err.List.map
    (fun (name, values) ->
      let+ v = Input.integers [ 1L; Int64.of_int length ] values in
      (name, v))
    [
      ("input_ids", e.input_ids);
      ("attention_mask", e.attention_mask);
      ("token_type_ids", e.token_type_ids);
    ]

let clip tokenizer ~length text =
  let* tok =
    Err.import (fun e -> `Metadata_invalid e) (Clip_input.Bpe.of_json tokenizer)
  in
  let* e =
    Err.import
      (fun e -> `Metadata_invalid e)
      (Clip_input.Bpe.encode tok ~max_length:length text)
  in
  Err.List.map
    (fun (name, values) ->
      let+ v = Input.integers [ 1L; Int64.of_int length ] values in
      (name, v))
    [ ("input_ids", e.input_ids); ("attention_mask", e.attention_mask) ]

let image params ppm =
  let* img =
    Err.import (fun e -> `Metadata_invalid e) (Clip_input.Ppm.of_string ppm)
  in
  let* resize = extent "shortest_edge" params in
  let* crop = extent "crop_size" params in
  let* () = Spec.require (resize >= crop) "resize must cover crop" in
  let* resample = field "resample" params in
  let* filter, flip, mean, std =
    match resample with
    | "bilinear" ->
        let+ flip = member "flip_channels" params >>= bool in
        (Clip_input.Resample.Bilinear, flip, [| 0.; 0.; 0. |], [| 1.; 1.; 1. |])
    | "bicubic" ->
        let floats key = member key params >>= array >>= Err.List.map number in
        let* mean = floats "image_mean" in
        let* std = floats "image_std" in
        let+ () =
          Spec.require
            (List.length mean = 3
            && List.length std = 3
            && List.for_all (( < ) 0.) std)
            "processor channel normalization"
        in
        ( Clip_input.Resample.Bicubic,
          false,
          Array.of_list mean,
          Array.of_list std )
    | _ -> invalid "unsupported processor resampling"
  in
  let pixels =
    Clip_input.Prep.tensor img ~filter ~resize ~crop ~flip ~mean ~std
  in
  Input.f32 [ 1L; 3L; Int64.of_int crop; Int64.of_int crop ] pixels

let recipe producer manifest =
  let* recipe_id = field "recipe_id" manifest in
  let* bytes =
    M.Producer.checked_text producer.M.Producer.root producer.commit
      "task-recipes.json"
  in
  let* tasks = parse bytes >>= member "tasks" >>= array in
  let* task = Spec.row_by "recipe_id" recipe_id tasks in
  Err.List.iter
    (fun key ->
      let* actual = member key manifest in
      let* expected = member key task in
      equal ~identity:recipe_id ~field:key actual expected)
    [ "recipe"; "kind"; "expected_cases" ]

let role bundle case name =
  let* id = field "id" case in
  let* files = Tensors.case_files bundle case in
  let file = "cases/" ^ id ^ "/" ^ name ^ ".pt" in
  let* descriptors =
    Err.of_option (`Metadata_missing file) (List.assoc_opt file files)
  in
  Tensors.load (Filename.concat bundle.Reference.Bundle.dir file) descriptors

let raw bundle case suffix =
  let* rows = member "raw" case >>= array in
  let* matches =
    filter
      (fun row ->
        let+ path = field "path" row in
        String.ends_with ~suffix path)
      rows
  in
  match matches with
  | [ row ] -> field "path" row >>= Reference.read_member bundle
  | _ -> invalid ("one raw input required: " ^ suffix)

let asset bundle name = Reference.read_member bundle ("assets/" ^ name)

let prepare config producer bundle reference case =
  let* kind = field "recipe_id" bundle.Reference.Bundle.manifest in
  let contract = reference.Reference.Reference.contract in
  let* text = text contract in
  let* contract = Pt2_fixture.Contract.of_string text in
  if kind = "bert-wordpiece-v1" then
    let* vocab = asset bundle "vocab.txt" in
    let* text = raw bundle case ".txt" in
    let* length = M.Assets.text_length contract in
    bert vocab ~length text
  else if kind = "tinyclip-pillow-v1" || kind = "mobilevit-xxs-pillow-v1" then
    let* model = field "model_id" reference.contract in
    let* task_model = M.Producer.task producer model in
    let* config_json = asset bundle "config.json" >>= parse in
    let adapter =
      if kind = "tinyclip-pillow-v1" then M.Assets.Tinyclip
      else M.Assets.Mobilevit
    in
    let* params =
      M.Assets.image config task_model contract config_json adapter
    in
    let* ppm = raw bundle case ".ppm" in
    let* pixels = image params ppm in
    let* tokens =
      if adapter = M.Assets.Mobilevit then Ok []
      else
        let* tokenizer = asset bundle "tokenizer.json" in
        let* text = raw bundle case ".txt" in
        let* length = M.Assets.text_length contract in
        clip tokenizer ~length text
    in
    Ok (tokens @ [ ("pixel_values", pixels) ])
  else invalid ("unsupported raw-input recipe: " ^ kind)
