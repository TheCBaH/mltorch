(* The Native dialect, as a [Dialect.S]. Its own unit rather than a submodule of
   anything: [Graph_view] and [Snapshot] are specialized AGAINST it, so putting
   it inside either would be a cycle, and it deliberately depends on no
   framework module itself — only on the IR and the op tables. *)

type op = Graph_ir.op
type shape_error = Graph_shape.error

(* Annotated rather than inferred: [Graph_shape.pp_error] takes an OPEN row
   ([< shape_error]) and the constructors produce one ([> ...]), neither of
   which matches [Dialect.S]'s closed [shape_error]. Without the annotations the
   inferred signature is more general and the functor application is rejected. *)
let pp_shape_error : Format.formatter -> shape_error -> unit =
  Graph_shape.pp_error

let missing_sig : Tensor_id.t -> shape_error = fun id -> `Missing_tensor_sig id
let operands = Graph_ir.operands
let map_operands = Graph_ir.map_operands
let output_shape = Graph_shape.output_shape
let classify = Output_transfer.classify
let pp_op pp_ref fmt op = Graph_ir.pp_op_with ~pp_ref fmt op

(* Native constrains nothing beyond what shape inference already checks: every
   six-axis shape is a legal Native shape. The hook exists for dialects whose
   invariant is not implied by their operations — Native4D's four-axis rule
   holds of graph inputs and captured constants, which no op produces. *)
let validate_sig : Tensor_sig.t -> (unit, shape_error) Core.result =
 fun _ -> Core.return ()
