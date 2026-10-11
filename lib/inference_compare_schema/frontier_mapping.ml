module W = Wire

type t = {
  canonical_values : Canonical_spec.t list;
  final_outputs : Final_output.t list;
  initial_values : Initial_binding.t list;
  input_bindings : Input_binding.t list;
  input_checks : Input_check.t list;
  output_bindings : Output_binding.t list;
}

let decode : t W.decoder =
  W.record (fun o ->
      let ( let* ) = W.( let* ) in
      let* canonical_values =
        W.Obj.required o "canonical_values"
          (W.unique
             ~key:(fun (c : Canonical_spec.t) ->
               Id.Canonical_value.to_string c.key)
             (W.list Canonical_spec.decode))
      in
      let* final_outputs =
        W.Obj.required o "final_outputs"
          (W.unique
             ~key:(fun (f : Final_output.t) ->
               Id.Output_occurrence.to_string f.occurrence)
             (W.list Final_output.decode))
      in
      let* initial_values =
        W.Obj.required o "initial_values"
          (W.unique
             ~key:(fun (b : Initial_binding.t) ->
               Id.Canonical_value.to_string b.key)
             (W.list Initial_binding.decode))
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
      Ok
        {
          canonical_values;
          final_outputs;
          initial_values;
          input_bindings;
          input_checks;
          output_bindings;
        })

let encode f =
  W.obj
    [
      W.field "canonical_values"
        (W.enc_list Canonical_spec.encode)
        f.canonical_values;
      W.field "final_outputs" (W.enc_list Final_output.encode) f.final_outputs;
      W.field "initial_values"
        (W.enc_list Initial_binding.encode)
        f.initial_values;
      W.field "input_bindings"
        (W.enc_list Input_binding.encode)
        f.input_bindings;
      W.field "input_checks" (W.enc_list Input_check.encode) f.input_checks;
      W.field "output_bindings"
        (W.enc_list Output_binding.encode)
        f.output_bindings;
    ]
