(* Native4D as a [Dialect.S].

   Named [dialect4], not [dialect], for the reason .ai/native4d_plan.md gives:
   a wrapped library compiles each unit with its alias module opened, so a
   sibling named [Dialect] would shadow the generic [Dialect] this very file
   must satisfy — and [Graph_view.Make] would then have no signature to name. *)

type op = Op.t
type shape_error = Graph_shape4.error

let pp_shape_error = Graph_shape4.pp_error
let missing_sig id = `Missing_tensor_sig id
let operands = Op.operands
let map_operands = Op.map_operands
let classify = Output_transfer4.classify
let pp_op pp_ref fmt op = Op.pp_with ~pp_ref fmt op

(* The generic validator compares against the stored [Tensor_sig.shape], which
   is a [Vec6.shape] in every dialect, so [Shape4] is unwrapped on the way out
   and never leaks into [Dialect.S]. *)
let output_shape op ~sig_of =
  let open Core.Syntax in
  let+ shapes = Graph_shape4.output_shape op ~sig_of in
  List.map Shape4.to_vec6 shapes

(* THE hook, and the reason [Dialect.S] has one. Shape inference constrains what
   OPS produce; a graph input or a captured constant is produced by no op, so
   without this a Native4D graph whose input is directly its output would
   validate with a signature carrying extent on T or D. *)
let validate_sig (sg : Tensor_sig.t) =
  let open Core.Syntax in
  let+ _ = Shape4.of_vec6 sg.Tensor_sig.shape in
  ()
