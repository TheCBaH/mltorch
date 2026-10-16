(* The embedded CompCert as a variant word. It is available where the
   embedding for this host's ISA was built (MLTORCH_COMPCERT names the ISA
   when the library is built); the identity names that ISA and the pinned
   CompCert artifacts. *)

module P = Inference_compare_pt2

let available () = Ok ()

let provider : P.Provider.t =
  {
    name = "compcert";
    available;
    instantiate =
      (fun () ->
        let id = Compiled_runners.compcert_identity () in
        Ok { P.Provider.runner = Compiled_runners.compcert (); identity = id });
  }

let provider_fp32 : P.Provider.t =
  {
    name = "compcert-fp32";
    available;
    instantiate =
      (fun () ->
        let id = Compiled_runners.compcert_fp32_identity () in
        Ok
          {
            P.Provider.runner = Compiled_runners.compcert_fp32 ();
            identity = id;
          });
  }

let register () =
  P.Provider.register provider;
  P.Provider.register provider_fp32
