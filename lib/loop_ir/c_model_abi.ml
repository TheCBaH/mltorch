(* The declarations shared by [model_main.c] and [model_infer.c], emitted into
   both from this one text so they cannot drift. The requirement accessors are
   constants of the inference unit: payload sizes, workspace size and base
   alignment, and the three fixed payload headers (the identity is inside
   them). *)
let declarations =
  String.concat "\n"
    [
      "extern const uint64_t model_weights_size;";
      "extern const uint64_t model_inputs_size;";
      "extern const uint64_t model_outputs_size;";
      "extern const uint64_t model_workspace_size;";
      "extern const uint64_t model_workspace_alignment;";
      "extern const unsigned char model_weights_header[64];";
      "extern const unsigned char model_inputs_header[64];";
      "extern const unsigned char model_outputs_header[64];";
      "int model_run(const void *weights, const void *inputs, void *workspace,";
      "              void *outputs, struct model_error *error);";
      "";
    ]

(* Process exit categories of the standalone application. *)
let exit_ok = 0
let exit_usage = 2
let exit_payload = 3
let exit_allocation = 4
let exit_inference = 5

(* The last stderr line of an inference failure. *)
let error_line_prefix = "model_error:"
