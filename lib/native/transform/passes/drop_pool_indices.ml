(* [Max_pool2d_with_indices] whose index output is dead becomes an ordinary
   [Max_pool2d]; [Adaptive_max_pool2d_with_indices] narrows to
   [Adaptive_max_pool2d] the same way -- ATen has no value-only adaptive
   overload to import directly (unlike [max_pool2d.default]/
   [max_pool2d_with_indices.default]'s genuine two-ATen-op split), so this
   pass is the ONLY route by which the narrower op is ever produced. See
   .ai/native_transform_design.md §12g.

   This is NOT dead-code elimination, and conflating the two is the mistake
   .ai/native4d_design.md §7.8 makes when it says DCE "removes the sink and
   unused edge and the value output becomes ordinary MaxPool2D". SSA node
   outputs are all-or-nothing: [Dce] deletes a node with NO live outputs, but it
   cannot drop one dead edge from a node whose other output is live. Narrowing
   the arity is a node REPLACEMENT, which is this pass, and it runs after [Dce]
   has removed the [Discard] sink that was keeping the index edge in use.

   The value claim is [Identical], and that is not a judgement call:
   [MaxPool2dWithIndices.Compute.value_pixel] and [MaxPool2d.Compute.pixel] are
   the same call to [S.max_pool2d] with the same parameters, so the two agree
   bit for bit and the symbolic verifier proves it structurally. The index edge
   is simply not mentioned, which [Rewrite.apply] turns into a deletion cluster.
   [AdaptiveMaxPool2dWithIndices.Compute.value_pixel] IS
   [AdaptiveMaxPool2d.Compute.pixel] (the former is defined as the latter's own
   [V.pixel]), so the same structural argument proves that pair too.

   Why it exists at all: Native4D has no argmax-pool operation, so an index edge
   that survives to the dialect boundary is a conversion failure. Removing the
   ones that are dead anyway is what makes the common inference graph — where
   the bridge routes indices straight into a [Discard] — convertible. *)

open Graph_ir

(* Dead by the time this runs: [Dce] has removed the [Discard] sink, so a dead
   index has no uses at all rather than one sink use. Checking [uses] alone
   would therefore be enough; [is_graph_output] is checked too because a graph
   output has no uses either and must not be mistaken for dead. *)
let dead view id =
  (not (Graph_view.is_graph_output view id)) && Graph_view.uses view id = []

let on_node : type v. Pass.env -> node -> (v, unit) Recipe.t option =
 fun { view; _ } n ->
  match (n.Node.op, n.Node.outputs) with
  | ( Max_pool2d_with_indices { Pool.MaxPool2dWithIndices.params; x },
      [ values; indices ] )
    when dead view indices ->
      Some
        (let open Recipe in
         let* values = existing values in
         replace ~remove:[ n.Node.id ]
           ~insert:
             [
               {
                 op = Max_pool2d { Pool.MaxPool2d.params; x };
                 outputs = [ Preserved values ];
                 from = [ n.Node.id ];
               };
             ]
             (* The value edge keeps its id while its producer changes, which
                [apply] requires a claim for. Stated outright rather than left to
                [replace]'s inference, which keys on [subst] — and there is no
                substitution here, the id is simply taken over. *)
           ~claims:[ (values, Preserved values, Correspondence.Identical) ]
           ())
  | ( Adaptive_max_pool2d_with_indices
        { Pool.AdaptiveMaxPool2dWithIndices.params; x },
      [ values; indices ] )
    when dead view indices ->
      Some
        (let open Recipe in
         let* values = existing values in
         replace ~remove:[ n.Node.id ]
           ~insert:
             [
               {
                 op = Adaptive_max_pool2d { Pool.AdaptiveMaxPool2d.params; x };
                 outputs = [ Preserved values ];
                 from = [ n.Node.id ];
               };
             ]
           ~claims:[ (values, Preserved values, Correspondence.Identical) ]
           ())
  | _ -> None

let pass = Pass.per_node ~name:"drop_pool_indices" { Pass.on_node }
