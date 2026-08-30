(* Shared builders/fixtures for the other modules in this directory (split
   from the former test/native_bridge_test.ml). Not a test
   module itself. *)

open Bigarray
module T = Aten_tensor
module D = Aten_dtype
module Stype = Aten_scalar_type
module PT = Pytorch_types
module Sm = Schema_runtime.String_map

let float_tensor shape vals =
  let t = T.create shape in
  let v = Option.get (T.data D.float32 t) in
  List.iteri (fun i x -> v.{i} <- x) vals;
  t

let i64_tensor shape vals =
  let t = T.create ~dtype:Stype.Long shape in
  let v = Option.get (T.data D.int64 t) in
  List.iteri (fun i x -> v.{i} <- x) vals;
  t

let native_f32 shape vals =
  let n = List.fold_left ( * ) 1 shape in
  let data = Array1.create float32 c_layout n in
  List.iteri (fun i x -> data.{i} <- x) vals;
  let shape6 =
    Aten_shape.of_aten (Array.of_list shape)
    |> Err.or_raise ~pp_error:Aten_shape.pp_error
  in
  Tensor.Tensor
    {
      shape = shape6;
      payload = { Payload.fmt = Payload.F32; quant = Payload.No_quant; data };
    }

let native_i64 shape vals =
  let n = List.fold_left ( * ) 1 shape in
  let data = Array1.create int64 c_layout n in
  List.iteri (fun i x -> data.{i} <- x) vals;
  let shape6 =
    Aten_shape.of_aten (Array.of_list shape)
    |> Err.or_raise ~pp_error:Aten_shape.pp_error
  in
  Tensor.Tensor
    {
      shape = shape6;
      payload = { Payload.fmt = Payload.I64; quant = Payload.No_quant; data };
    }

(* Eta-expanded (not [let pp_result = Core.Pretty.err_result ...]): as a bare
   partial application this is a value, not a function, so its ['a Err.t]
   result type cannot be generalized across call sites once it lives outside
   the file that also calls it (relaxed value restriction). *)
let pp_result ppf x =
  Core.Pretty.err_result ~ok:(Fmt.any "Ok")
    ~error:(fun ppf e -> Fmt.pf ppf "Error: %a" Verify.pp_error e)
    ppf x

(* [Tensor_bridge.of_aten] returns a plain (not [Err.t]) error, so this
   composes with [Core.Pretty.result] rather than [err_result]; [~ok] varies
   by test (the tensor itself, just its shape, or a surprise-success marker). *)
let pp_of_aten_result ~ok =
  Core.Pretty.err_result ~ok ~error:(fun ppf e ->
      Fmt.pf ppf "Error: %a" Tensor_bridge.pp_error e)

let targ name = PT.Argument.Tensor (PT.TensorArgument.make name)
let in_tensor name = PT.NamedArgument.make name (targ name) None
let in_int name i = PT.NamedArgument.make name (PT.Argument.Int i) None
let in_ints name xs = PT.NamedArgument.make name (PT.Argument.Ints xs) None
let in_floats name xs = PT.NamedArgument.make name (PT.Argument.Floats xs) None
let in_bool name b = PT.NamedArgument.make name (PT.Argument.Bool b) None
let in_float name f = PT.NamedArgument.make name (PT.Argument.Float f) None
let in_none name = PT.NamedArgument.make name (PT.Argument.None false) None
let in_string name s = PT.NamedArgument.make name (PT.Argument.String s) None

let in_memory_format name (mf : PT.MemoryFormat.t) =
  PT.NamedArgument.make name (PT.Argument.Memory_format mf) None

(* [index.Tensor]'s [indices : Tensor?[]] -- a mix of live tensor entries
   ([`T name]) and explicit [None]s. *)
let in_optional_tensors name entries =
  PT.NamedArgument.make name
    (PT.Argument.Optional_tensors
       (List.map
          (function
            | `T tname ->
                PT.OptionalTensorArgument.Tensor (PT.TensorArgument.make tname)
            | `None -> PT.OptionalTensorArgument.None false)
          entries))
    None

(* Bind each (name, ATen tensor) into an env and dispatch a one-node graph;
   print each native output as "shape {values}".  The env key and the input's
   TensorArgument name are the same [name], which is how the bridge resolves it. *)
let dispatch_print_with_graph ~print_graph ~target ~bindings ~inputs ~noutputs =
  let env = List.fold_left (fun m (k, t) -> Sm.add k t m) Sm.empty bindings in
  let outputs = List.init noutputs (fun i -> targ (Printf.sprintf "out%d" i)) in
  let node = PT.Node.make target inputs outputs Sm.empty None (Some "test") in
  match Op_bridge.dispatch ~aten_env:env node with
  | None -> print_string "no native impl\n"
  | Some (Error e) ->
      Format.printf "error: %a@." Op_bridge.pp_error (Err.Error.kind e)
  | Some (Ok (graph, bindings)) -> (
      if print_graph then Format.printf "%a@." Graph_ir.pp graph;
      match Eval_direct.run graph ~inputs:bindings with
      | Error e ->
          Format.printf "eval error: %a@." Eval_direct.pp_error
            (Err.Error.kind e)
      | Ok result_env ->
          let outs =
            List.map
              (fun oid -> Graph_ir.Tensor_id.Map.find oid result_env)
              graph.Graph_ir.Graph.outputs
          in
          List.iter (fun o -> Format.printf "%a@." Tensor.pp o) outs)

let dispatch_print ~target ~bindings ~inputs ~noutputs =
  dispatch_print_with_graph ~print_graph:false ~target ~bindings ~inputs
    ~noutputs

(* The bridge arms above pin the native compute against hand-derived values, and
   the generated walks (test/native_walk_test.ml) compare whole ops against real
   ATen — but neither reaches THIS case. A compile-time scalar in a Tensor slot
   (`aten.add.Tensor(x, 3)`, which is how MobileNet-v3's hardsigmoid is
   serialised) is skipped by [bin/pt2_spec_gen] when it writes node fixtures,
   because a Tensor-typed param holding an [Argument.Int] is not something an
   op-spec can express; and the walk generator synthesises tensor arguments, so
   it never produces one either.

   [Interp_verify.dispatch ~verify:true] is the dual path: it runs the node
   through [Interp_dispatch] (real ATen, which materialises the scalar with
   [full_like]) AND through [Op_bridge] + [Eval_direct] (native, which routes it
   into an op parameter), then compares element-wise with [Verify.verify_node].
   Silence means the two agree. *)
let verify_print ~target ~bindings ~inputs =
  let env = List.fold_left (fun m (k, t) -> Sm.add k t m) Sm.empty bindings in
  let node =
    PT.Node.make target inputs [ targ "out0" ] Sm.empty None (Some "test")
  in
  match
    Interp_verify.dispatch ~verify:true ~ppf:Format.std_formatter env node
  with
  | Error e ->
      Format.printf "dispatch error: %a@." Interp_verify.pp_interp_error
        (Err.Error.kind e)
  | Ok _ -> print_string "aten and native agree\n"

(* A repeatable non-constant fill: constant inputs make a transposed kernel or a
   swapped relayout indistinguishable from the correct one. *)
let ramp n = List.init n (fun i -> float_of_int ((i mod 7) - 3) /. 2.)
let jstr fmt = Printf.sprintf fmt

(* A local, deliberately minimal copy of test/native_interp/programs.ml's
   builder. Depending on that library would pull its inline tests into this
   runner, where they would run a second time under a different profile. *)
let meta_json sizes =
  jstr
    {|{"dtype":7,"sizes":[%s],"requires_grad":false,"device":{"type":"cpu"},"strides":[{"as_int":1}],"storage_offset":{"as_int":0},"layout":7}|}
    (String.concat "," (List.map (fun i -> jstr {|{"as_int":%d}|} i) sizes))

let in_sym_int name i =
  PT.NamedArgument.make name (PT.Argument.Sym_int (PT.SymIntArgument.Int i))
    None

let in_sym_name name s =
  PT.NamedArgument.make name (PT.Argument.Sym_int (PT.SymIntArgument.Name s))
    None

let in_sym_ints name xs =
  PT.NamedArgument.make name
    (PT.Argument.Sym_ints (List.map (fun i -> PT.SymIntArgument.Int i) xs))
    None

let in_sym_ints_named name xs =
  PT.NamedArgument.make name
    (PT.Argument.Sym_ints
       (List.map
          (function
            | `I i -> PT.SymIntArgument.Int i | `N s -> PT.SymIntArgument.Name s)
          xs))
    None
