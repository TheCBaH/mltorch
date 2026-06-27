(* A decoded verification case: one ATen operation invocation.

   [target] selects the op (and, via Aten_op_config, its return arity). [args]
   are in the op's declared parameter order, each carrying its decoded value.

   Tensor payloads are synthesized from a single default PCG instance
   ([Pcg.default]) threaded through all [args] in order, so a case is
   reproducible from the spec alone. (A per-tensor seed in the Random payload may
   later make individual draws independently reproducible.)

   The codec is generated (aten_op_spec): it reads the target and dispatches the
   "args" object to the matching per-op codec. This module is only the type. *)

type t = { target : string; args : (string * Arg_value.t) list }
