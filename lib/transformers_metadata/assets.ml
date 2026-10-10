open Err.Syntax
open Json_util
module J = Jsont.Json
module U = Pt2_fixture_unix
module C = Pt2_fixture.Cohort
module D = Pt2_checkpoint_map.Document
module Spec = Pt2_fixture.Contract.Tensor_spec

type adapter = Bert | Mobilevit | Tinyclip

module Request = struct
  type t = { adapter : adapter; artifact_id : string; output : string }
end

let requests json =
  let* () = schema 1 json in
  let* rows = member "adapters" json >>= array in
  let* () =
    if rows <> [] && List.length rows <= 100 then Ok ()
    else invalid "adapter request count"
  in
  let* requests =
    Err.List.map
      (fun row ->
        let* kind = field "adapter" row in
        let* adapter =
          match kind with
          | "bert-wordpiece" -> Ok Bert
          | "mobilevit" -> Ok Mobilevit
          | "tinyclip" -> Ok Tinyclip
          | _ -> invalid ("unsupported adapter: " ^ kind)
        in
        let* artifact_id = field "artifact_id" row in
        let* output = field "output" row in
        let* parts = Source.safe output in
        let* () =
          if List.length parts = 1 then Ok ()
          else invalid "adapter output must be a filename"
        in
        Ok { Request.adapter; artifact_id; output })
      rows
  in
  let* () = unique (List.map (fun r -> r.Request.output) requests) in
  let* () = unique (List.map (fun r -> r.Request.artifact_id) requests) in
  Ok requests

let file_pin model name =
  let* raw = path [ "files"; name ] model in
  let* fields = members raw in
  pin_of (obj (("name", J.string name) :: fields))

let fetch (config : U.Bundle.config) model name =
  let* p = file_pin model name in
  let* file =
    U.Fetch.ensure ?transport:config.transport
      ~layer:(Pt2_fixture.Fault.Source name) config.cache p
  in
  let+ bytes = U.Fetch.read ~max_bytes:8_000_000 file in
  (p, bytes)

let spec contract name =
  Err.of_option
    (`Metadata_missing ("contract input " ^ name))
    (List.find_opt
       (fun (s : Spec.t) -> s.name = name)
       contract.Pt2_fixture.Contract.inputs)

let text_length contract =
  let* ids = spec contract "input_ids" in
  let* mask = spec contract "attention_mask" in
  let* () =
    if
      ids.dtype = Pt2_checkpoint_map.Dtype.I64
      && mask.dtype = ids.dtype && mask.shape = ids.shape
    then Ok ()
    else invalid "text ID/mask dtype or shape"
  in
  match ids.shape with
  | [ 1L; n ] when n >= 2L && n <= 512L -> Ok (Int64.to_int n)
  | _ -> invalid "bounded text contract requires batch one and 2..512 positions"

let source_text model =
  let* recipe = path [ "recipe"; "text" ] model in
  let* () =
    Err.List.iter
      (fun key ->
        let* actual = field key recipe in
        same ~identity:"text recipe" ~field:key actual "right")
      [ "padding_side"; "truncation_side" ]
  in
  Ok recipe

let bert config model contract config_json =
  let* length = text_length contract in
  let* segments = spec contract "token_type_ids" in
  let* ids = spec contract "input_ids" in
  let* () =
    if segments.dtype = ids.dtype && segments.shape = ids.shape then Ok ()
    else invalid "BERT token_type_ids contract"
  in
  let* recipe = source_text model in
  let* vocab_pin, vocab_bytes = fetch config model "vocab.txt" in
  let lines = String.split_on_char '\n' vocab_bytes in
  let lines =
    match List.rev lines with "" :: rest -> List.rev rest | _ -> lines
  in
  let* () = unique lines in
  let vocab = Wordpiece.vocab_of_lines lines in
  let* expected_size = member "vocab_size" config_json >>= integer in
  let* source_size = member "vocab_size" recipe >>= integer in
  let* () =
    if
      Int64.of_int (Wordpiece.size vocab) = expected_size
      && source_size = expected_size
    then Ok ()
    else invalid "BERT vocabulary size differs from config/recipe"
  in
  let* () =
    Err.List.iter
      (fun (key, token) ->
        let* id = path [ "special_tokens"; key; "id" ] recipe >>= integer in
        let* spelling =
          path [ "special_tokens"; key; "token" ] recipe >>= string
        in
        let* () = same ~identity:"BERT" ~field:key spelling token in
        match Wordpiece.id vocab token with
        | Some actual when Int64.of_int actual = id -> Ok ()
        | _ -> invalid ("BERT special token ID: " ^ token))
      [
        ("cls_token", "[CLS]");
        ("mask_token", "[MASK]");
        ("pad_token", "[PAD]");
        ("sep_token", "[SEP]");
        ("unk_token", "[UNK]");
      ]
  in
  let* max_positions =
    member "max_position_embeddings" config_json >>= integer
  in
  let* () =
    if Int64.of_int length <= max_positions then Ok ()
    else invalid "BERT positions exceed config"
  in
  Ok
    [
      ("do_lower_case", J.bool true);
      ("max_length", J.int length);
      ("vocab", pin vocab_pin);
    ]

let image config model contract config_json adapter =
  let* processor_pin, bytes = fetch config model "preprocessor_config.json" in
  let* processor = parse bytes in
  let* source_recipe =
    path [ "recipe"; "image"; "preprocessor" ] model >>= members
  in
  let* () =
    Err.List.iter
      (fun (key, expected) ->
        let* actual = member key processor in
        equal ~identity:"processor recipe" ~field:key actual expected)
      source_recipe
  in
  let* feature_type = field "feature_extractor_type" processor in
  let expected_type =
    match adapter with
    | Mobilevit -> "MobileViTFeatureExtractor"
    | Tinyclip -> "CLIPFeatureExtractor"
    | Bert -> ""
  in
  let* () =
    same ~identity:"processor" ~field:"feature_extractor_type" feature_type
      expected_type
  in
  let* () =
    Err.List.iter
      (fun key ->
        let* value = member key processor >>= bool in
        if value then Ok () else invalid ("processor requires " ^ key))
      [ "do_resize"; "do_center_crop" ]
  in
  let* resize = member "size" processor >>= integer in
  let* crop = member "crop_size" processor >>= integer in
  let* resample = member "resample" processor >>= integer in
  let* () =
    if resize >= crop && crop > 0L && resize <= 4096L then Ok ()
    else invalid "unsupported resize/crop extent"
  in
  let* input = spec contract "pixel_values" in
  let* () =
    if
      input.dtype = Pt2_checkpoint_map.Dtype.F32
      && input.shape = [ 1L; 3L; crop; crop ]
    then Ok ()
    else invalid "pixel recipe differs from release input contract"
  in
  (* These old configs omit do_rescale/rescale_factor. The fixed processor
     defaults are part of this supported recipe, checked against explicit
     overrides when present, rather than inferred from an arbitrary digest. *)
  let* ms = members processor in
  let* () =
    Err.List.iter
      (fun (key, expected) ->
        match List.assoc_opt key ms with
        | None -> Ok ()
        | Some value ->
            equal ~identity:"processor default" ~field:key value expected)
      [ ("do_rescale", J.bool true); ("rescale_factor", J.number (1. /. 255.)) ]
  in
  let base =
    [
      ("crop_size", J.number (Int64.to_float crop));
      ("shortest_edge", J.number (Int64.to_float resize));
      ("preprocessor_config", pin processor_pin);
    ]
  in
  match adapter with
  | Bert -> invalid "BERT image recipe"
  | Mobilevit ->
      let* flip = member "do_flip_channels" processor >>= bool in
      let* () =
        if resample = 2L && flip then Ok ()
        else invalid "MobileViT requires bilinear resize and BGR flip"
      in
      let* () =
        match List.assoc_opt "do_normalize" ms with
        | None | Some (Jsont.Bool (false, _)) -> Ok ()
        | _ -> invalid "MobileViT normalization unsupported"
      in
      let* labels = member "id2label" config_json >>= members in
      let* count =
        path [ "recipe"; "output_decoding"; "num_labels" ] model >>= integer
      in
      let* () =
        if Int64.of_int (List.length labels) = count && count = 1000L then Ok ()
        else invalid "MobileViT label count"
      in
      let* () =
        Err.List.iter
          (fun i ->
            let* value = member (string_of_int i) (obj labels) in
            let+ _ = string value in
            ())
          (List.init 1000 Fun.id)
      in
      Ok
        (obj
           (("flip_channels", J.bool true)
           :: ("resample", J.string "bilinear")
           :: base))
  | Tinyclip ->
      let* normalized = member "do_normalize" processor >>= bool in
      let* mean = member "image_mean" processor in
      let* std = member "image_std" processor in
      let* mean_values = array mean in
      let* std_values = array std in
      let* () =
        if
          resample = 3L && normalized
          && List.length mean_values = 3
          && List.length std_values = 3
          && List.for_all
               (function
                 | Jsont.Number (v, _) -> Float.is_finite v && v > 0.
                 | _ -> false)
               std_values
        then Ok ()
        else invalid "CLIP requires bicubic resize and channel normalization"
      in
      let* image_size =
        path [ "vision_config"; "image_size" ] config_json >>= integer
      in
      let* () =
        if image_size = crop then Ok ()
        else invalid "CLIP crop differs from vision config"
      in
      Ok
        (obj
           (("image_mean", mean) :: ("image_std", std)
           :: ("resample", J.string "bicubic")
           :: base))

let clip_text config model contract config_json =
  let* length = text_length contract in
  let* recipe = source_text model in
  let* tokenizer_pin, bytes = fetch config model "tokenizer.json" in
  let* tokenizer = parse bytes in
  let* () =
    Err.List.iter
      (fun (key, expected_bytes) ->
        let* expected = parse expected_bytes in
        let* actual = member key tokenizer in
        equal ~identity:"CLIP tokenizer pipeline" ~field:key actual expected)
      [
        ( "normalizer",
          {|{"type":"Sequence","normalizers":[{"type":"NFC"},{"type":"Replace","pattern":{"Regex":"\\s+"},"content":" "},{"type":"Lowercase"}]}|}
        );
        ( "pre_tokenizer",
          {|{"type":"Sequence","pretokenizers":[{"type":"Split","pattern":{"Regex":"<\\|startoftext\\|>|<\\|endoftext\\|>|'s|'t|'re|'ve|'m|'ll|'d|[\\p{L}]+|[\\p{N}]|[^\\s\\p{L}\\p{N}]+"},"behavior":"Removed","invert":true},{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true}]}|}
        );
        ( "post_processor",
          {|{"type":"RobertaProcessing","sep":["<|endoftext|>",49407],"cls":["<|startoftext|>",49406],"trim_offsets":false,"add_prefix_space":false}|}
        );
      ]
  in
  let* model_json = member "model" tokenizer in
  let* () =
    Err.List.iter
      (fun (key, expected) ->
        let* actual = field key model_json in
        same ~identity:"CLIP BPE" ~field:key actual expected)
      [ ("type", "BPE"); ("end_of_word_suffix", "</w>") ]
  in
  let* vocab = member "vocab" model_json >>= members in
  let* vocab_size = member "vocab_size" recipe >>= integer in
  let* ids = Err.List.map (fun (_, j) -> integer j) vocab in
  let* () =
    if
      vocab_size = 49408L
      && List.length ids = 49408
      && List.sort Int64.compare ids = List.init 49408 Int64.of_int
    then Ok ()
    else invalid "CLIP vocabulary IDs/size"
  in
  let* merges = member "merges" model_json >>= array in
  let* merge_strings = Err.List.map string merges in
  let* () = unique merge_strings in
  let* _ =
    Err.import ~pos:__POS__
      (fun e -> `Metadata_invalid e)
      (Clip_input.Bpe.of_json bytes)
  in
  let* () =
    Err.List.iter
      (fun (key, token, post_key) ->
        let* id = path [ "special_tokens"; key; "id" ] recipe >>= integer in
        let* spelling =
          path [ "special_tokens"; key; "token" ] recipe >>= string
        in
        let* () = same ~identity:"CLIP text recipe" ~field:key spelling token in
        let* actual_id =
          path [ "model"; "vocab"; token ] tokenizer >>= integer
        in
        let* post = path [ "post_processor"; post_key ] tokenizer >>= array in
        let* () =
          match post with
          | [ s; n ] ->
              let* s = string s in
              let* n = integer n in
              if s = token && n = id then Ok ()
              else invalid "CLIP special-token post processing"
          | _ -> invalid "CLIP tokenizer post processor"
        in
        if actual_id = id then Ok ()
        else invalid "CLIP special-token vocabulary")
      [
        ("bos_token", "<|startoftext|>", "cls");
        ("eos_token", "<|endoftext|>", "sep");
      ]
  in
  let* positions =
    path [ "text_config"; "max_position_embeddings" ] config_json >>= integer
  in
  let* () =
    if Int64.of_int length <= positions then Ok ()
    else invalid "CLIP positions exceed config"
  in
  Ok [ ("max_length", J.int length); ("tokenizer", pin tokenizer_pin) ]

let build config producer cohort cohort_json (request : Request.t) =
  let id = request.artifact_id in
  let* source, _ = Producer.source_artifact producer id in
  let* model = field "id" source in
  let expected_model =
    match request.adapter with
    | Bert -> "bert-tiny"
    | Mobilevit -> "mobilevit-xxs"
    | Tinyclip -> "tinyclip"
  in
  let* () = same ~identity:id ~field:"adapter model" model expected_model in
  let* entry =
    Err.of_option
      (`Metadata_missing ("cohort artifact " ^ id))
      (C.find cohort id)
  in
  let* rows = member "artifacts" cohort_json >>= array in
  let* cohort_row = Selection.row_by_id rows id in
  let* weights = member "weight_source" cohort_row in
  let* () =
    Err.List.iter
      (fun key ->
        let* actual = field key weights in
        let* expected = path [ "reference"; key ] source >>= string in
        same ~identity:id ~field:("adapter checkpoint " ^ key) actual expected)
      [ "repo"; "revision"; "config_sha256" ]
  in
  let* task_model = Producer.task producer model in
  let* () =
    Err.List.iter
      (fun key ->
        let* actual = field key task_model in
        let* expected = field key weights in
        same ~identity:id ~field:("task " ^ key) actual expected)
      [ "repo"; "revision" ]
  in
  let* config_pin, config_bytes = fetch config task_model "config.json" in
  let* config_digest = field "config_sha256" weights in
  let* () =
    same ~identity:id ~field:"task config"
      (Pt2_sha256.Digest.to_hex config_pin.sha256)
      config_digest
  in
  let* config_json = parse config_bytes in
  let* bundle = U.Bundle.ensure config cohort entry in
  let* contract_bytes = U.Bundle.read_member bundle "contract.json" in
  let* contract = Pt2_fixture.Contract.of_string contract_bytes in
  let expected_inputs =
    match request.adapter with
    | Bert -> [ "attention_mask"; "input_ids"; "token_type_ids" ]
    | Mobilevit -> [ "pixel_values" ]
    | Tinyclip -> [ "attention_mask"; "input_ids"; "pixel_values" ]
  in
  let* () =
    if
      List.sort String.compare
        (List.map (fun (s : Spec.t) -> s.name) contract.inputs)
      = expected_inputs
      && not contract.dynamic
    then Ok ()
    else invalid "adapter requires its complete static input contract"
  in
  let* () =
    same ~identity:id ~field:"contract artifact_id" contract.artifact_id id
  in
  let* parsed_contract = parse contract_bytes in
  let* contract_weights = member "weights" parsed_contract in
  let* () =
    equal ~identity:id ~field:"contract checkpoint identity" contract_weights
      weights
  in
  let* derived =
    match request.adapter with
    | Bert -> bert config task_model contract config_json
    | Mobilevit ->
        let+ image = image config task_model contract config_json Mobilevit in
        [ ("image", image) ]
    | Tinyclip ->
        let* image = image config task_model contract config_json Tinyclip in
        let+ text = clip_text config task_model contract config_json in
        ("image", image) :: text
  in
  let* repo = member "repo" task_model in
  let* revision = member "revision" task_model in
  Ok
    (obj
       ([
          ("schema_version", J.int 1);
          ("artifact_id", J.string id);
          ("repo", repo);
          ("revision", revision);
          ("config", pin config_pin);
        ]
       @ derived))

let generate config producer cohort_json requests =
  let* () = schema 1 cohort_json in
  let* bytes = text cohort_json in
  let* cohort = C.of_string bytes in
  let* rows = member "artifacts" cohort_json >>= array in
  let* ids = Err.List.map (field "artifact_id") rows in
  let* () = unique ids in
  Err.List.map
    (fun r ->
      let+ j = build config producer cohort cohort_json r in
      (r.Request.output, j))
    requests
