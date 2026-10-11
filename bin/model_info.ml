(* The Makefile's model.* rules ask this where a model comes from.

   model_info resolve MODELS_DIR COHORT CACHE_DIR NAME
     prints "KIND ID MODEL_JSON": "timm NAME MODELS_DIR/NAME/models/model.json" or
     "fixture ARTIFACT_ID BUNDLE/models/model.json" ("-" while the fixture is not
     cached). See [Model_source.resolve] for what NAME may be; an unknown or
     ambiguous one lists the candidates on stderr and exits 2.
   model_info checkpoint MODEL_DIR
     prints the path of a timm model's pinned Hub checkpoint, fetching it. *)

let die s =
  prerr_endline s;
  exit 2

let () =
  match Array.to_list Sys.argv with
  | [ _; "resolve"; models; cohort; cache; name ] -> (
      match Model_source.resolve { models; cohort; cache } name with
      | Ok (Timm { name; json; _ }) -> Printf.printf "timm %s %s\n" name json
      | Ok (Fixture { id; json }) ->
          Printf.printf "fixture %s %s\n" id (Option.value json ~default:"-")
      | Error m -> die m)
  | [ _; "checkpoint"; dir ] -> (
      match Model_source.checkpoint dir with
      | Ok path -> print_endline path
      | Error m -> die m)
  | _ ->
      die
        "usage: model_info resolve MODELS_DIR COHORT CACHE_DIR NAME\n\
        \       model_info checkpoint MODEL_DIR"
