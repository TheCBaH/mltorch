(* What the two compiled routes are made of, and which parts they share. The
   embedded CompCert and the direct assembly route are different compilers
   that end in the same assembler, image planner and loader (Rivet), so an
   agreement between them does not vouch for those. *)

module Pins = Inference_compare_rivet.Pins

type component = {
  name : string;
  pin : string;  (** The pinned source revision or artifact digest. *)
  routes : [ `Both | `Compcert | `Assembly ];
}

let of_pin routes (p : Inference_compare_rivet.Compiled_identity.component) =
  { name = p.name; pin = p.pin; routes }

(* The submodule revisions and CompCert artifacts this build pins; the lock
   file of the CompCert embedding is the source of the artifact digests. *)
let components =
  [
    {
      name = "CompCert (embedded, aarch64: export and asm artifacts)";
      pin =
        "v3.17-2 \
         export-aarch64=7c1e6e487718684d52d0b7308e13ccc3265d652a043d3db835ee1b28a90df967 \
         asm-aarch64=1dd93f7528b49cf19bb5e38e97ba70dee541ceb403e60453d4e96811fcca2729 \
         (vendored/rivet-compcert@c150c478b28348b9fc1ac352271f8052b87a4dfa)";
      routes = `Compcert;
    };
    {
      name = "Loop IR bundle to C (Compcert_scalar dialect)";
      pin = "this repository";
      routes = `Compcert;
    };
    of_pin `Assembly Pins.selection;
    of_pin `Both Pins.rivet;
    of_pin `Both Pins.schedule;
  ]

let shared = List.filter (fun c -> c.routes = `Both) components
