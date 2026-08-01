(* Drop a [Permute4] that permutes nothing, tying its output to its input.

   The pass exists to prove the abstraction, which is stage 7's whole point: it
   is written against [Recipe]/[Pattern]/[Pass] — the same modules Native's nine
   passes use — with no Native4D copy of recipes, planning, the id discipline or
   claim propagation. The only Native4D-specific thing in it is the projector
   that recognises the op, which is exactly the line [Pattern]'s design says
   should be dialect-specific: "the projector is a bare [op -> 'a option], so
   there is no per-op DSL".

   An identity permutation is worth removing on its own merits — the Bmm
   legalization emits a [Permute4], and a chain of two inverse ones composes to
   the identity — but a single pass is enough to establish that the framework
   admits a second dialect. Whether Native4D grows the rest of the permute
   family is a question for when it has passes that create the opportunities. *)

let is_identity (perm : Ops4.Permute4.perm) =
  List.for_all (fun (out, inp) -> Axis4.equal out inp) perm

let pass =
  Framework.Pass4.per_node ~name:"trim_permute4"
    {
      Framework.Pass4.on_node =
        (fun _env (n : Graph.node) ->
          match (n.Graph_common.Node.op, n.Graph_common.Node.outputs) with
          | Op.Permute4 { Ops4.Permute4.perm; x }, [ out ] when is_identity perm
            ->
              Some
                (let open Framework.Recipe4 in
                 let* out = existing out in
                 let* x = existing x in
                 (* [trim] states the claim itself: both ends are existing
                    source edges and the surviving one takes over. *)
                 trim ~remove:[ n.Graph_common.Node.id ] ~tie:[ (out, x) ])
          | _ -> None);
    }
