(* Native to Native4D: a partial, checked dialect conversion.

   PARTIAL BY DESIGN. .ai/native4d_design.md §1: "A Native graph that does not
   satisfy the shape precondition or contains an operation without a sound
   legalization is not a failed implementation of Native4D; it is outside the
   dialect's domain." [Domain.check] states that domain; this produces the graph
   and the evidence for the graphs inside it.

   Only meaningful on a CANONICAL graph — run [Pipeline.canonical] first. The
   importer emits conv weights right-aligned from ATen's [out, in, kh, kw],
   which lands on D/H/W/C, so an unpipelined ResNet-18 is outside the four-axis
   dialect on every conv weight. *)

(* The result carries the destination SNAPSHOT, not a raw graph. Every consumer
   — [Graph_map.create], [clusters_over], [check_claim_closure],
   [Map_verify.run] — is typed on a snapshot whose brand is ['dst], and that
   brand exists nowhere else: rebuilding a snapshot from the graph mints a fresh
   one that unifies with nothing, so a graph-only result could not be verified
   at all. Corrects design §9.1. *)
type ('src, 'dst) t = {
  dst : 'dst Framework.Snapshot4.t;
  map : ('src, 'dst) Graph_map.t;
  constants : Tensor.packed Tensor_id.Map.t;
}

(* Packed for the same reason [Snapshot.create] packs: handing back an
   existential removes the caller's choice of destination tag. *)
type 'src packed = Pack : ('src, 'dst) t -> 'src packed

val graph : ('src, 'dst) t -> Graph.graph

(* [constants] is validated in full BEFORE any legalization — a payload that
   disagrees with its signature would otherwise be copied through into the
   result and read much later, far from the mistake. Absence is different: it is
   an error only for a legalization that needs the payload AT CONVERSION TIME,
   which is batch norm alone, so a graph whose other constants are unbound
   converts fine. *)
val convert :
  ?constants:Tensor.packed Tensor_id.Map.t ->
  'src Snapshot.t ->
  ('src packed, Error.t) Core.result
