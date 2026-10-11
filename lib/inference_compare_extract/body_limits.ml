(* Bounds for validating a single extracted body: the body is a subset of an
   already-bounded endpoint graph, so the limits only need to be finite. *)
let v : Inference_compare_schema.Resource_limits.t =
  {
    max_live_bytes = Int64.max_int;
    max_conversion_scratch_bytes = Int64.max_int;
    max_snapshot_bytes = Int64.max_int;
    max_retained_disk_bytes = 0L;
    max_mismatch_samples = 0L;
    retention = No_retention;
    retention_requirement = Best_effort;
    verified_spill = false;
    max_nodes = Int64.max_int;
    max_edges = Int64.max_int;
    max_units = Int64.max_int;
    max_recipe_nodes = Int64.max_int;
    max_selector_nodes = Int64.max_int;
    max_selector_depth = Int64.max_int;
    max_path_length = Int64.max_int;
    max_retained_witnesses = Int64.max_int;
  }
