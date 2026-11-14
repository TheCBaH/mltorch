type t = {
  source : string;
  prelude : string;
  helpers : Loop_c_runtime.Name.t list;
  local_doubles : int64;
  precision : Loop_numerics.Precision.t;
  refusal : Loop_numerics.Refusal.t option;
  buffer_types : string list;
}
