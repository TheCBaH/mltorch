(* A reference to a produced edge. Edges are single-assignment, so a reference is
   just the producing edge's id — the same thing [Graph_ir] calls [tensor_ref].
   It lives in its own module so the per-op modules can name their operands (and
   serialise / pretty-print them) without depending on [Graph_ir]: an op carries
   its operand refs as payload, but never the graph that owns them. *)

type t = Tensor_id.t

let jsont : t Jsont.t = Tensor_id.jsont
let pp = Tensor_id.pp
