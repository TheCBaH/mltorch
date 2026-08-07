(* The Native instantiation of the projection.

   A module rather than a [Make] applied at each call site, so there is one
   Native projection in the program and every caller gets the same one. The
   Native4D instantiation is the same functor over [Ops4], and lands with
   Stage 4. *)

include Me_build.Make (struct
  type op = Graph_ir.op

  let op_name = Graph_ir.op_name
  let operands = Graph_ir.operands

  (* [pp_op_with] rather than [pp_op]: the latter needs a graph to resolve
     operand references against, and the attribute wants the op's PARAMETERS.
     Operands are already edges, so printing them into the attribute too would
     duplicate the graph structure as text. *)
  let pp_op fmt op = Graph_ir.pp_op_with ~pp_ref:Graph_ir.Tensor_id.pp fmt op
end)
