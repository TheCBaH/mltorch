let register_all () =
  Inference_compare_generated.Generated.register ();
  Inference_compare_js.Js_engines.register ();
  Rivet_register.register ();
  Compcert_register.register ()
