(* One module per semantic identifier role. Each is generative, so the types
   are pairwise distinct even though every encoding is a string. Order is
   alphabetical; no behavior depends on it. *)

module Canonical_value =
  Text_value.Make_identifier
    (struct
      let role = "canonical_value"
    end)
    ()

module Case =
  Text_value.Make_identifier
    (struct
      let role = "case"
    end)
    ()

module Codec =
  Text_value.Make_identifier
    (struct
      let role = "codec"
    end)
    ()

module Codec_operation =
  Text_value.Make_identifier
    (struct
      let role = "codec_operation"
    end)
    ()

module Comparator =
  Text_value.Make_identifier
    (struct
      let role = "comparator"
    end)
    ()

module Comparison =
  Text_value.Make_identifier
    (struct
      let role = "comparison"
    end)
    ()

module Conversion_node =
  Text_value.Make_identifier
    (struct
      let role = "conversion_node"
    end)
    ()

module Dialect =
  Text_value.Make_identifier
    (struct
      let role = "dialect"
    end)
    ()

module Digest_algorithm =
  Text_value.Make_identifier
    (struct
      let role = "digest_algorithm"
    end)
    ()

module Engine =
  Text_value.Make_identifier
    (struct
      let role = "engine"
    end)
    ()

module Engine_route =
  Text_value.Make_identifier
    (struct
      let role = "engine_route"
    end)
    ()

module Experiment =
  Text_value.Make_identifier
    (struct
      let role = "experiment"
    end)
    ()

module Graph_version =
  Text_value.Make_identifier
    (struct
      let role = "graph_version"
    end)
    ()

module Layout =
  Text_value.Make_identifier
    (struct
      let role = "layout"
    end)
    ()

module Member_pair =
  Text_value.Make_identifier
    (struct
      let role = "member_pair"
    end)
    ()

module Metric_rule =
  Text_value.Make_identifier
    (struct
      let role = "metric_rule"
    end)
    ()

module Metric_support =
  Text_value.Make_identifier
    (struct
      let role = "metric_support"
    end)
    ()

module Node_local =
  Text_value.Make_identifier
    (struct
      let role = "node_local"
    end)
    ()

module Operator =
  Text_value.Make_identifier
    (struct
      let role = "operator"
    end)
    ()

module Oracle =
  Text_value.Make_identifier
    (struct
      let role = "oracle"
    end)
    ()

module Output_occurrence =
  Text_value.Make_identifier
    (struct
      let role = "output_occurrence"
    end)
    ()

module Payload =
  Text_value.Make_identifier
    (struct
      let role = "payload"
    end)
    ()

module Precision =
  Text_value.Make_identifier
    (struct
      let role = "precision"
    end)
    ()

module Recipe =
  Text_value.Make_identifier
    (struct
      let role = "recipe"
    end)
    ()

module Run =
  Text_value.Make_identifier
    (struct
      let role = "run"
    end)
    ()

module Tensor_local =
  Text_value.Make_identifier
    (struct
      let role = "tensor_local"
    end)
    ()

module Tensor_metric =
  Text_value.Make_identifier
    (struct
      let role = "tensor_metric"
    end)
    ()

module Tensor_relation =
  Text_value.Make_identifier
    (struct
      let role = "tensor_relation"
    end)
    ()

module Unit =
  Text_value.Make_identifier
    (struct
      let role = "unit"
    end)
    ()

module Value_instance =
  Text_value.Make_identifier
    (struct
      let role = "value_instance"
    end)
    ()

module Value_route =
  Text_value.Make_identifier
    (struct
      let role = "value_route"
    end)
    ()
