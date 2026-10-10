open Err.Syntax
open Transformers_metadata.Json_util
module F = Pt2_fixture
module J = Jsont.Json

let required ~models recipe =
  if not models then []
  else
    match recipe with
    | "mobilevit-xxs-pillow-v1" -> [ "host" ]
    | "tinyclip-pillow-v1" -> [ "host"; "image-tower"; "text-tower" ]
    | _ -> []

let row name report = obj [ ("boundary", J.string name); ("report", report) ]

let compare ~backend ~pins artifact id contract expected actual =
  let* checks =
    Diagnostic.compare ~atol:contract.F.Contract.atol ~rtol:contract.rtol
      expected actual
  in
  Task_report.make ~backend ~pins artifact id ~atol:contract.atol
    ~rtol:contract.rtol checks

let labels bundle ids =
  let* config = Reference.read_member bundle "assets/config.json" >>= parse in
  let* labels = Reference.read_member bundle "labels.json" >>= parse in
  let* original = member "id2label" config in
  let* () = equal ~identity:"classification" ~field:"labels" labels original in
  Err.List.map
    (fun i ->
      let index = F.Logical.get_int64 ids i in
      let+ label = member (Int64.to_string index) labels >>= string in
      obj [ ("id", J.int64 index); ("label", J.string label) ])
    (List.init 5 Fun.id)

let tower_reference bundle output =
  let* matches =
    Err.List.map
      (fun (r : Reference.Reference.t) ->
        let* outputs = member "outputs" r.contract >>= array in
        let+ names = Err.List.map (field "name") outputs in
        if names = [ output ] then Some r else None)
      bundle.Reference.Bundle.references
  in
  match List.filter_map Fun.id matches with
  | [ r ] -> Ok r
  | _ -> invalid ("task requires exactly one tower for " ^ output)

let tower config cohort bundle id forward inputs expected output boundary =
  let* reference = tower_reference bundle output in
  let* () =
    Transformers_metadata.Chaining.contracts forward reference.contract
  in
  let* fixture = Demo.execution_fixture config cohort reference in
  let* contract = text reference.contract >>= F.Contract.of_string in
  let* arguments =
    Err.List.map
      (fun (spec : F.Contract.Tensor_spec.t) ->
        let+ value = Lifecycle.named spec.name inputs in
        (spec.name, value))
      contract.inputs
  in
  let normalizations = ref [] in
  let* actual =
    Input.run
      ~on_empty_caches:(fun r -> normalizations := Input.normalizations r)
      fixture.archive contract arguments
  in
  let* expected = Lifecycle.named output expected in
  let* value = Lifecycle.named output actual in
  let* checks =
    Diagnostic.compare ~atol:contract.atol ~rtol:contract.rtol
      [ (output, expected) ]
      actual
  in
  let+ report =
    Task_report.make ~normalizations:!normalizations
      ~pins:(Demo.execution_pins fixture)
      contract.artifact_id id ~atol:contract.atol ~rtol:contract.rtol checks
  in
  (value, row boundary report)

let run config cohort bundle case reference fixture contract inputs outputs =
  let* recipe = field "recipe_id" bundle.Reference.Bundle.manifest in
  let* id = field "id" case in
  let artifact = contract.F.Contract.artifact_id in
  let pins = Demo.execution_pins fixture in
  match recipe with
  | "mobilevit-xxs-pillow-v1" ->
      let* logits = Lifecycle.named "logits" outputs in
      let* actual = Host.top5 logits in
      let* expected = Adapter.role bundle case "host" in
      let* report =
        compare ~backend:"consumer-host:top-five" ~pins artifact id contract
          expected actual
      in
      let* ids = Lifecycle.named "top5_ids" actual in
      let+ decoded = labels bundle ids in
      [
        obj
          [
            ("boundary", J.string "host");
            ("report", report);
            ("decoded_labels", J.list decoded);
          ];
      ]
  | "tinyclip-pillow-v1" ->
      let* () =
        Spec.require
          (List.length bundle.references = 3)
          "CLIP task requires forward and both towers"
      in
      let* expected = Adapter.role bundle case "host" in
      let* image_features, image_report =
        tower config cohort bundle id reference.Reference.Reference.contract
          inputs expected "image_features" "image-tower"
      in
      let* text_features, text_report =
        tower config cohort bundle id reference.contract inputs expected
          "text_features" "text-tower"
      in
      let* log_scale = Host.checkpoint_scale fixture in
      let* actual = Host.clip ~log_scale ~image_features ~text_features in
      let+ report =
        compare ~backend:"consumer-host:binary64-norm-dot, binary32-steps" ~pins
          artifact id contract expected actual
      in
      [ row "host" report; image_report; text_report ]
  | _ -> Ok []
