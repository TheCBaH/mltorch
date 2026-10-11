(* The Makefile's model.* rules ask this where a model comes from.

   model_info (kind|id|json) MODELS_DIR COHORT CACHE_DIR NAME
     kind  "timm" (a model of the producer submodule) or "fixture" (a released
           Transformers artifact)
     id    the model's name, or the fixture's artifact id
     json  the graph JSON: MODELS_DIR/NAME/models/model.json, or the fixture's
           in the cache ("-" while it is not cached)
   See [Model_source.resolve] for what NAME may be; an unknown or ambiguous one
   lists the candidates on stderr and exits 2.
   model_info checkpoint MODEL_DIR
     prints the path of a timm model's pinned Hub checkpoint, fetching it. *)

let die s =
  prerr_endline s;
  exit 2

let () =
  match Array.to_list Sys.argv with
  | [ _; (("kind" | "id" | "json") as what); models; cohort; cache; name ] -> (
      match Model_source.resolve { models; cohort; cache } name with
      | Error m -> die m
      | Ok model ->
          print_endline
            (match (what, model) with
            | "kind", Timm _ -> "timm"
            | "kind", Fixture _ -> "fixture"
            | "id", Timm { name; _ } -> name
            | "id", Fixture { id; _ } -> id
            | _, Timm { json; _ } -> json
            | _, Fixture { json; _ } -> Option.value json ~default:"-"))
  | [ _; "checkpoint"; dir ] -> (
      match Model_source.checkpoint dir with
      | Ok path -> print_endline path
      | Error m -> die m)
  | _ ->
      die
        "usage: model_info (kind|id|json) MODELS_DIR COHORT CACHE_DIR NAME\n\
        \       model_info checkpoint MODEL_DIR"
