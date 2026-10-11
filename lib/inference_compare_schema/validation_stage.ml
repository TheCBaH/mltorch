module W = Wire

type t = Full_graph_confirmation | Reference_trajectory | Tested_trajectory

let decode, encode =
  W.enum ~kind:"validation_stage"
    [
      (Full_graph_confirmation, "full_graph_confirmation");
      (Reference_trajectory, "reference_trajectory");
      (Tested_trajectory, "tested_trajectory");
    ]
