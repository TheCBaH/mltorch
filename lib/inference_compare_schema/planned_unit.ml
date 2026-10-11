module W = Wire
module Unit_id = W.Text (Id.Unit)
module Canonical = W.Text (Id.Canonical_value)

type t = {
  unit : Id.Unit.t;
  left_graph : Extracted_graph.t;
  right_graph : Extracted_graph.t;
  predecessors : Id.Unit.t list;
  canonical_reads : Id.Canonical_value.t list;
  canonical_writes : Id.Canonical_value.t list;
  input_bindings : Input_binding.t list;
  input_checks : Input_check.t list;
  output_bindings : Output_binding.t list;
  output_pairs : Member_pair.t list;
}

let decode : t W.decoder =
  W.record (fun o ->
      let ( let* ) = W.( let* ) in
      let* unit = W.Obj.required o "unit" Unit_id.decode in
      let* left_graph = W.Obj.required o "left_graph" Extracted_graph.decode in
      let* right_graph =
        W.Obj.required o "right_graph" Extracted_graph.decode
      in
      let* predecessors =
        W.Obj.required o "predecessors" (W.list Unit_id.decode)
      in
      let* canonical_reads =
        W.Obj.required o "canonical_reads" (W.list Canonical.decode)
      in
      let* canonical_writes =
        W.Obj.required o "canonical_writes" (W.list Canonical.decode)
      in
      let* input_bindings =
        W.Obj.required o "input_bindings" (W.list Input_binding.decode)
      in
      let* input_checks =
        W.Obj.required o "input_checks" (W.list Input_check.decode)
      in
      let* output_bindings =
        W.Obj.required o "output_bindings" (W.list Output_binding.decode)
      in
      let* output_pairs =
        W.Obj.required o "output_pairs" (W.list Member_pair.decode)
      in
      Ok
        {
          unit;
          left_graph;
          right_graph;
          predecessors;
          canonical_reads;
          canonical_writes;
          input_bindings;
          input_checks;
          output_bindings;
          output_pairs;
        })

let encode u =
  W.obj
    [
      W.field "unit" Unit_id.encode u.unit;
      W.field "left_graph" Extracted_graph.encode u.left_graph;
      W.field "right_graph" Extracted_graph.encode u.right_graph;
      W.field "predecessors" (W.enc_list Unit_id.encode) u.predecessors;
      W.field "canonical_reads" (W.enc_list Canonical.encode) u.canonical_reads;
      W.field "canonical_writes"
        (W.enc_list Canonical.encode)
        u.canonical_writes;
      W.field "input_bindings"
        (W.enc_list Input_binding.encode)
        u.input_bindings;
      W.field "input_checks" (W.enc_list Input_check.encode) u.input_checks;
      W.field "output_bindings"
        (W.enc_list Output_binding.encode)
        u.output_bindings;
      W.field "output_pairs" (W.enc_list Member_pair.encode) u.output_pairs;
    ]
