(** The generated-code engines as variant words: [loop-c] and [ssa-c] (C built
    with the host compiler), [loop-wasm] and [ssa-wasm] (WebAssembly under
    node). [ssa-*] make the kernels through the structured SSA backend's exact
    pipeline, [loop-*] through the Loop emitter; all run the reference
    (binary64) numerics. *)

val register : unit -> unit
(** Makes the words known to {!Inference_compare_pt2.Variant.of_string}. A word
    whose compiler or runtime is missing is a refusal when a pair uses it,
    naming what is missing. *)

(** The engines themselves, for tests that drive a body without a pair. *)
module Engines : sig
  module Loop_c :
    Inference_compare_pt2.Graph_runner.ENGINE with type op = Graph_ir.op

  module Loop_wasm :
    Inference_compare_pt2.Graph_runner.ENGINE with type op = Graph_ir.op

  module Ssa_c :
    Inference_compare_pt2.Graph_runner.ENGINE with type op = Graph_ir.op

  module Ssa_wasm :
    Inference_compare_pt2.Graph_runner.ENGINE with type op = Graph_ir.op
end
