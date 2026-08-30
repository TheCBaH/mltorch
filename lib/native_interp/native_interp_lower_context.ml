(* Immutable state shared by the PT2 operator dispatch families.  Provenance
   accumulation deliberately stays in [Native_interp_lower]: dispatch only needs
   graph metadata, the active error escape, and the lazily-computed read set. *)

open Pytorch_types
open Schema_runtime
open Native_interp_error
open Native_interp_decode
module Tensor_id = Graph_ir.Tensor_id

type env = Tensor_id.t String_map.t

type t = {
  esc : error Err.Escape.t;
  graph : Graph.t;
  reads : unit String_map.t Lazy.t;
  (* SSA names of Parameter/Buffer/Tensor_constant-kind graph inputs -- model
     data fixed across every run, as opposed to a genuine [User_input]. Used
     only by [index.Tensor]'s trace-past-Clone rule
     ([Native_interp_decode.resolve_index_source]): only a directly-bound
     captured constant can hold a non-F32 dtype in this engine, so that is
     the one condition under which tracing past a wrapping [clone.default]
     has anything to gain. *)
  constant_names : unit String_map.t;
}

let get ctx env node name = env_find ctx.esc env (tensor_name ctx.esc node name)

let tensor_or_scalar ctx env node name =
  match find_arg ctx.esc node name with
  | Argument.Tensor t -> `Tensor (env_find ctx.esc env t.TensorArgument.name)
  | Argument.Int i -> `Scalar (float_of_int i)
  | Argument.Float f -> `Scalar f
  | _ ->
      malformed ctx.esc
        (`Wrong_arg_kind
           { op = node.target; arg = name; expected = `Tensor_or_scalar })
