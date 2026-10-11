module W = Wire

type requirement = Best_effort | Required
type retention = All | First_mismatch | No_retention | Selected_units

type t = {
  max_live_bytes : int64;
  max_conversion_scratch_bytes : int64;
  max_snapshot_bytes : int64;
  max_retained_disk_bytes : int64;
  max_mismatch_samples : int64;
  retention : retention;
  retention_requirement : requirement;
  verified_spill : bool;
  max_nodes : int64;
  max_edges : int64;
  max_units : int64;
  max_recipe_nodes : int64;
  max_selector_nodes : int64;
  max_selector_depth : int64;
  max_path_length : int64;
  max_retained_witnesses : int64;
}

let decode_requirement, encode_requirement =
  W.enum ~kind:"retention_requirement"
    [ (Best_effort, "best_effort"); (Required, "required") ]

let decode_retention, encode_retention =
  W.enum ~kind:"retention"
    [
      (All, "all");
      (First_mismatch, "first_mismatch");
      (No_retention, "none");
      (Selected_units, "selected_units");
    ]

let decode : t W.decoder =
  W.record (fun o ->
      let ( let* ) = W.( let* ) in
      let n name = W.Obj.required o name W.nonneg_int64 in
      let* max_live_bytes = n "max_live_bytes" in
      let* max_conversion_scratch_bytes = n "max_conversion_scratch_bytes" in
      let* max_snapshot_bytes = n "max_snapshot_bytes" in
      let* max_retained_disk_bytes = n "max_retained_disk_bytes" in
      let* max_mismatch_samples = n "max_mismatch_samples" in
      let* retention = W.Obj.required o "retention" decode_retention in
      let* retention_requirement =
        W.Obj.required o "retention_requirement" decode_requirement
      in
      let* verified_spill = W.Obj.required o "verified_spill" W.bool in
      let* max_nodes = n "max_nodes" in
      let* max_edges = n "max_edges" in
      let* max_units = n "max_units" in
      let* max_recipe_nodes = n "max_recipe_nodes" in
      let* max_selector_nodes = n "max_selector_nodes" in
      let* max_selector_depth = n "max_selector_depth" in
      let* max_path_length = n "max_path_length" in
      let* max_retained_witnesses = n "max_retained_witnesses" in
      Ok
        {
          max_live_bytes;
          max_conversion_scratch_bytes;
          max_snapshot_bytes;
          max_retained_disk_bytes;
          max_mismatch_samples;
          retention;
          retention_requirement;
          verified_spill;
          max_nodes;
          max_edges;
          max_units;
          max_recipe_nodes;
          max_selector_nodes;
          max_selector_depth;
          max_path_length;
          max_retained_witnesses;
        })

let encode l =
  W.obj
    [
      W.field "max_live_bytes" W.enc_int64 l.max_live_bytes;
      W.field "max_conversion_scratch_bytes" W.enc_int64
        l.max_conversion_scratch_bytes;
      W.field "max_snapshot_bytes" W.enc_int64 l.max_snapshot_bytes;
      W.field "max_retained_disk_bytes" W.enc_int64 l.max_retained_disk_bytes;
      W.field "max_mismatch_samples" W.enc_int64 l.max_mismatch_samples;
      W.field "retention" encode_retention l.retention;
      W.field "retention_requirement" encode_requirement l.retention_requirement;
      W.field "verified_spill" W.enc_bool l.verified_spill;
      W.field "max_nodes" W.enc_int64 l.max_nodes;
      W.field "max_edges" W.enc_int64 l.max_edges;
      W.field "max_units" W.enc_int64 l.max_units;
      W.field "max_recipe_nodes" W.enc_int64 l.max_recipe_nodes;
      W.field "max_selector_nodes" W.enc_int64 l.max_selector_nodes;
      W.field "max_selector_depth" W.enc_int64 l.max_selector_depth;
      W.field "max_path_length" W.enc_int64 l.max_path_length;
      W.field "max_retained_witnesses" W.enc_int64 l.max_retained_witnesses;
    ]
